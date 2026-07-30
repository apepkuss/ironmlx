import Darwin
import Foundation

public struct ProviderModelFileDownloader: ModelFileDownloading {
    private let huggingFace: RustHuggingFaceFileDownloader
    private let standard: ResumableFileDownloader

    public init(
        httpClient: any ModelDownloadHTTPClient,
        executableURL: URL = BackendBinaryResolver.resolve()
    ) {
        huggingFace = RustHuggingFaceFileDownloader(executableURL: executableURL)
        standard = ResumableFileDownloader(httpClient: httpClient)
    }

    public func download(
        _ request: ResumableDownloadRequest,
        progress: @escaping @Sendable (Int64) async -> Void
    ) async throws -> ModelValidatedFile {
        switch request.identity.provider {
        case .huggingFace:
            try await huggingFace.download(request, progress: progress)
        case .modelScope:
            try await standard.download(request, progress: progress)
        }
    }
}

public struct RustHuggingFaceFileDownloader: ModelFileDownloading {
    private let executableURL: URL
    private let parallelism: Int
    private let chunkSize: Int

    public init(
        executableURL: URL = BackendBinaryResolver.resolve(),
        parallelism: Int = 4,
        chunkSize: Int = 10_000_000
    ) {
        self.executableURL = executableURL
        self.parallelism = parallelism
        self.chunkSize = chunkSize
    }

    public static func recoverableBytes(
        destination: URL,
        identity: ModelPartialIdentity
    ) -> Int64 {
        let cacheDirectory = destination.appendingPathExtension("hf-transfer")
        let identityURL = cacheDirectory.appendingPathComponent("identity.json")
        guard let data = try? Data(contentsOf: identityURL),
              let stored = try? JSONDecoder().decode(RustTransferIdentity.self, from: data),
              stored.matches(identity),
              Self.safeETag(stored.etag)
        else {
            return 0
        }
        let repositoryFolder = "models--" + identity.repoID.replacingOccurrences(of: "/", with: "--")
        let partial = cacheDirectory
            .appendingPathComponent(repositoryFolder, isDirectory: true)
            .appendingPathComponent("blobs", isDirectory: true)
            .appendingPathComponent(stored.etag)
            .appendingPathExtension("sync.part")
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: partial.path),
              let storedSize = (attributes[.size] as? NSNumber)?.int64Value,
              storedSize == identity.expectedSize + Int64(MemoryLayout<UInt64>.size),
              let handle = try? FileHandle(forReadingFrom: partial)
        else {
            return 0
        }
        defer { try? handle.close() }
        do {
            try handle.seek(toOffset: UInt64(identity.expectedSize))
            guard let marker = try handle.read(upToCount: MemoryLayout<UInt64>.size),
                  marker.count == MemoryLayout<UInt64>.size
            else {
                return 0
            }
            let committed = marker.withUnsafeBytes {
                UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
            }
            return min(identity.expectedSize, Int64(clamping: committed))
        } catch {
            return 0
        }
    }

    public func download(
        _ request: ResumableDownloadRequest,
        progress: @escaping @Sendable (Int64) async -> Void = { _ in }
    ) async throws -> ModelValidatedFile {
        guard request.identity.provider == .huggingFace else {
            throw RustHuggingFaceTransferError.unsupportedProvider(request.identity.provider.rawValue)
        }
        guard let requestURL = request.urlRequest.url,
              let endpoint = Self.endpoint(for: requestURL)
        else {
            throw URLError(.badURL)
        }
        try FileManager.default.createDirectory(
            at: request.destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: request.destination.path) {
            if let validation = try verifiedExistingFile(request.destination, identity: request.identity) {
                await progress(request.identity.expectedSize)
                return validation
            }
            try FileManager.default.removeItem(at: request.destination)
        }

        let token = Self.bearerToken(from: request.urlRequest)
        let transfer = RustHuggingFaceTransferCommand(
            executableURL: executableURL,
            repoID: request.identity.repoID,
            revision: request.identity.commitSHA,
            filename: request.identity.path,
            destination: request.destination,
            cacheDirectory: request.destination.appendingPathExtension("hf-transfer"),
            expectedSize: request.identity.expectedSize,
            expectedSHA256: request.identity.expectedSHA256,
            endpoint: endpoint,
            token: token,
            parallelism: parallelism,
            chunkSize: chunkSize
        )
        let result: RustHuggingFaceTransferResult
        do {
            result = try await transfer.run(progress: progress)
        } catch let error as RustHuggingFaceTransferError {
            if case let .processFailed(detail) = error,
               let actual = Self.checksumFromProcessFailure(detail) {
                throw ResumableDownloadError.downloadedChecksumMismatch(
                    expected: request.identity.expectedSHA256,
                    actual: actual
                )
            }
            throw error
        }
        guard result.size == request.identity.expectedSize else {
            throw ResumableDownloadError.downloadedSizeMismatch(
                expected: request.identity.expectedSize,
                actual: result.size
            )
        }
        guard result.sha256 == request.identity.expectedSHA256 else {
            throw ResumableDownloadError.downloadedChecksumMismatch(
                expected: request.identity.expectedSHA256,
                actual: result.sha256
            )
        }
        return ModelValidatedFile(
            path: request.identity.path,
            sha256: result.sha256,
            etag: result.etag,
            identity: try ModelSnapshotVerifier.fileIdentity(of: request.destination)
        )
    }

    private func verifiedExistingFile(
        _ url: URL,
        identity: ModelPartialIdentity
    ) throws -> ModelValidatedFile? {
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            return nil
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size == identity.expectedSize else {
            return nil
        }
        let sha256 = try ModelSnapshotVerifier.sha256(of: url)
        guard sha256 == identity.expectedSHA256 else {
            return nil
        }
        try? FileManager.default.removeItem(
            at: url.appendingPathExtension("hf-transfer.lock")
        )
        return ModelValidatedFile(
            path: identity.path,
            sha256: sha256,
            identity: try ModelSnapshotVerifier.fileIdentity(of: url)
        )
    }

    private static func bearerToken(from request: URLRequest) -> String? {
        guard let authorization = request.value(forHTTPHeaderField: "Authorization"),
              authorization.hasPrefix("Bearer ")
        else {
            return nil
        }
        let token = String(authorization.dropFirst("Bearer ".count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return token.isEmpty ? nil : token
    }

    private static func endpoint(for url: URL) -> String? {
        guard let scheme = url.scheme,
              let host = url.host
        else {
            return nil
        }
        if let port = url.port {
            return "\(scheme)://\(host):\(port)"
        }
        return "\(scheme)://\(host)"
    }

    private static func checksumFromProcessFailure(_ detail: String) -> String? {
        let marker = "downloaded file SHA-256 "
        let separator = " does not match expected "
        guard let markerRange = detail.range(of: marker),
              let separatorRange = detail.range(
                  of: separator,
                  range: markerRange.upperBound..<detail.endIndex
              )
        else {
            return nil
        }
        let checksum = String(detail[markerRange.upperBound..<separatorRange.lowerBound])
            .lowercased()
        guard checksum.count == 64,
              checksum.allSatisfy(\.isHexDigit)
        else {
            return nil
        }
        return checksum
    }

    private static func safeETag(_ etag: String) -> Bool {
        !etag.isEmpty
            && etag != "."
            && etag != ".."
            && !etag.contains("/")
            && !etag.contains("\\")
    }
}

public enum RustHuggingFaceTransferError: LocalizedError {
    case unsupportedProvider(String)
    case processFailed(String)
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedProvider(provider):
            "Rust Hugging Face transfer does not support provider \(provider)."
        case let .processFailed(detail):
            detail
        case let .invalidResponse(detail):
            "ironmlx hf-transfer returned an invalid response: \(detail)"
        }
    }
}

private struct RustHuggingFaceTransferResult: Sendable {
    var size: Int64
    var sha256: String
    var etag: String
}

private struct RustHuggingFaceTransferCommand: Sendable {
    var executableURL: URL
    var repoID: String
    var revision: String
    var filename: String
    var destination: URL
    var cacheDirectory: URL
    var expectedSize: Int64
    var expectedSHA256: String
    var endpoint: String
    var token: String?
    var parallelism: Int
    var chunkSize: Int

    func run(
        progress: @escaping @Sendable (Int64) async -> Void
    ) async throws -> RustHuggingFaceTransferResult {
        let processBox = RustTransferProcessBox()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withThrowingTaskGroup(
                of: RustHuggingFaceTransferResult.self
            ) { group in
                group.addTask {
                    try await withCheckedThrowingContinuation { continuation in
                        DispatchQueue.global(qos: .utility).async {
                            do {
                                continuation.resume(
                                    returning: try runSynchronously(
                                        processBox: processBox,
                                        progress: progress
                                    )
                                )
                            } catch {
                                continuation.resume(throwing: error)
                            }
                        }
                    }
                }
                group.addTask {
                    do {
                        try await Task.sleep(nanoseconds: .max)
                    } catch {
                        processBox.cancel()
                        throw CancellationError()
                    }
                    throw CancellationError()
                }
                defer { group.cancelAll() }
                guard let result = try await group.next() else {
                    throw RustHuggingFaceTransferError.invalidResponse(
                        "transfer process ended without a result"
                    )
                }
                return result
            }
        } onCancel: {
            processBox.cancel()
        }
    }

    private func runSynchronously(
        processBox: RustTransferProcessBox,
        progress: @escaping @Sendable (Int64) async -> Void
    ) throws -> RustHuggingFaceTransferResult {
        let process = Process()
        let standardOutput = Pipe()
        let standardError = Pipe()
        let commandArguments = [
            "hf-transfer",
            "--repo-id", repoID,
            "--revision", revision,
            "--filename", filename,
            "--destination", destination.path,
            "--cache-dir", cacheDirectory.path,
            "--expected-size", String(expectedSize),
            "--expected-sha256", expectedSHA256,
            "--endpoint", endpoint,
            "--parallelism", String(parallelism),
            "--chunk-size", String(chunkSize),
        ]
        if executableURL.path == "/usr/bin/env" {
            process.executableURL = executableURL
            process.arguments = ["ironmlx"] + commandArguments
        } else {
            process.executableURL = executableURL
            process.arguments = commandArguments
        }
        var environment = ProcessInfo.processInfo.environment
        if let token {
            environment["HF_TOKEN"] = token
        } else {
            environment.removeValue(forKey: "HF_TOKEN")
        }
        process.environment = environment
        process.standardOutput = standardOutput
        process.standardError = standardError
        guard processBox.install(process) else {
            throw CancellationError()
        }

        let output = RustTransferOutputCollector(progress: progress)
        let errorData = RustTransferDataBox()
        let readers = DispatchGroup()
        do {
            try process.run()
        } catch {
            processBox.clear(process)
            throw error
        }
        readers.enter()
        DispatchQueue.global(qos: .utility).async {
            Self.read(
                standardOutput.fileHandleForReading,
                consume: { output.consume($0) }
            )
            readers.leave()
        }
        readers.enter()
        DispatchQueue.global(qos: .utility).async {
            Self.read(
                standardError.fileHandleForReading,
                consume: { errorData.append($0) }
            )
            readers.leave()
        }
        process.waitUntilExit()
        readers.wait()
        processBox.clear(process)

        if processBox.wasCancelled {
            throw CancellationError()
        }
        guard process.terminationStatus == 0 else {
            let detail = errorData.string
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw RustHuggingFaceTransferError.processFailed(
                detail.isEmpty
                    ? "ironmlx hf-transfer exited with status \(process.terminationStatus)."
                    : detail
            )
        }
        try output.finish()
        guard let result = output.result else {
            throw RustHuggingFaceTransferError.invalidResponse("missing completion event")
        }
        return result
    }

    private static func read(
        _ handle: FileHandle,
        consume: @escaping @Sendable (Data) -> Void
    ) {
        while true {
            let data = handle.availableData
            guard !data.isEmpty else {
                return
            }
            consume(data)
        }
    }
}

private final class RustTransferProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false

    var wasCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func install(_ process: Process) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !cancelled else {
            return false
        }
        self.process = process
        return true
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let processIdentifier = process?.processIdentifier ?? 0
        lock.unlock()
        if processIdentifier > 0 {
            Darwin.kill(processIdentifier, SIGTERM)
        }
    }

    func clear(_ process: Process) {
        lock.lock()
        if self.process === process {
            self.process = nil
        }
        lock.unlock()
    }
}

private final class RustTransferDataBox: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    var string: String {
        lock.lock()
        defer { lock.unlock() }
        return String(data: data, encoding: .utf8) ?? ""
    }

    func append(_ chunk: Data) {
        lock.lock()
        data.append(chunk)
        lock.unlock()
    }
}

private final class RustTransferOutputCollector: @unchecked Sendable {
    private let lock = NSLock()
    private let progress: @Sendable (Int64) async -> Void
    private var buffer = Data()
    private var parseError: Error?
    private var completion: RustHuggingFaceTransferResult?

    init(progress: @escaping @Sendable (Int64) async -> Void) {
        self.progress = progress
    }

    var result: RustHuggingFaceTransferResult? {
        lock.lock()
        defer { lock.unlock() }
        return completion
    }

    func consume(_ data: Data) {
        lock.lock()
        buffer.append(data)
        var lines: [Data] = []
        while let newline = buffer.firstIndex(of: 0x0A) {
            lines.append(Data(buffer[..<newline]))
            buffer.removeSubrange(...newline)
        }
        lock.unlock()
        lines.forEach(process)
    }

    func finish() throws {
        lock.lock()
        let tail = buffer
        buffer.removeAll()
        lock.unlock()
        if !tail.isEmpty {
            process(tail)
        }
        lock.lock()
        let error = parseError
        lock.unlock()
        if let error {
            throw error
        }
    }

    private func process(_ line: Data) {
        guard !line.isEmpty else {
            return
        }
        do {
            let event = try JSONDecoder().decode(RustTransferEvent.self, from: line)
            switch event.type {
            case "progress":
                if let bytes = event.bytes {
                    let semaphore = DispatchSemaphore(value: 0)
                    Task {
                        await progress(bytes)
                        semaphore.signal()
                    }
                    semaphore.wait()
                }
            case "complete":
                guard let size = event.size,
                      let sha256 = event.sha256,
                      let etag = event.etag
                else {
                    throw RustHuggingFaceTransferError.invalidResponse("incomplete completion event")
                }
                lock.lock()
                completion = RustHuggingFaceTransferResult(
                    size: size,
                    sha256: sha256,
                    etag: etag
                )
                lock.unlock()
            default:
                throw RustHuggingFaceTransferError.invalidResponse("unknown event \(event.type)")
            }
        } catch {
            lock.lock()
            if parseError == nil {
                parseError = error
            }
            lock.unlock()
        }
    }
}

private struct RustTransferEvent: Decodable {
    var type: String
    var bytes: Int64?
    var size: Int64?
    var sha256: String?
    var etag: String?
}

private struct RustTransferIdentity: Decodable {
    var version: Int
    var provider: String
    var repoID: String
    var commitSHA: String
    var path: String
    var expectedSize: Int64
    var expectedSHA256: String
    var etag: String

    enum CodingKeys: String, CodingKey {
        case version
        case provider
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case path
        case expectedSize = "expected_size"
        case expectedSHA256 = "expected_sha256"
        case etag
    }

    func matches(_ identity: ModelPartialIdentity) -> Bool {
        version == 1
            && provider == ModelRepositoryProvider.huggingFace.rawValue
            && repoID == identity.repoID
            && commitSHA == identity.commitSHA.lowercased()
            && path == identity.path
            && expectedSize == identity.expectedSize
            && expectedSHA256 == identity.expectedSHA256.lowercased()
    }
}
