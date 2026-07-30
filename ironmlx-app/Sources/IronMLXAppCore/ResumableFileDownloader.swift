import CryptoKit
import Foundation

public struct ResumableDownloadRequest: Sendable {
    public var urlRequest: URLRequest
    public var identity: ModelPartialIdentity
    public var destination: URL

    public init(urlRequest: URLRequest, identity: ModelPartialIdentity, destination: URL) {
        self.urlRequest = urlRequest
        self.identity = identity
        self.destination = destination
    }
}

public protocol ModelFileDownloading: Sendable {
    func download(
        _ request: ResumableDownloadRequest,
        progress: @escaping @Sendable (Int64) async -> Void
    ) async throws -> ModelValidatedFile
}

public enum ResumableDownloadError: LocalizedError {
    case invalidContentRange(String?)
    case responseIdentityChanged(expected: String, actual: String?)
    case rangeNotSatisfiable
    case downloadedSizeMismatch(expected: Int64, actual: Int64)
    case downloadedChecksumMismatch(expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case let .invalidContentRange(value):
            "Resume response returned an invalid Content-Range: \(value ?? "<missing>")."
        case let .responseIdentityChanged(expected, actual):
            "Resume response identity changed from \(expected) to \(actual ?? "<missing>")."
        case .rangeNotSatisfiable:
            "Resume response rejected the requested byte range."
        case let .downloadedSizeMismatch(expected, actual):
            "Downloaded file has size \(actual), expected \(expected)."
        case let .downloadedChecksumMismatch(expected, actual):
            "Downloaded file has SHA-256 \(actual), expected \(expected)."
        }
    }
}

public struct ResumableFileDownloader: ModelFileDownloading {
    private let httpClient: any ModelDownloadHTTPClient

    public init(httpClient: any ModelDownloadHTTPClient) {
        self.httpClient = httpClient
    }

    public func download(
        _ request: ResumableDownloadRequest,
        progress: @escaping @Sendable (Int64) async -> Void = { _ in }
    ) async throws -> ModelValidatedFile {
        try FileManager.default.createDirectory(
            at: request.destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if let validation = try verifiedExistingFile(request.destination, identity: request.identity) {
            await progress(request.identity.expectedSize)
            return validation
        }

        let partial = request.destination.appendingPathExtension("partial")
        let metadata = partial.appendingPathExtension("meta.json")
        let prepared = try preparePartial(
            partial: partial,
            metadata: metadata,
            identity: request.identity
        )
        var offset = prepared.offset
        let partialIdentity = prepared.identity
        if offset == request.identity.expectedSize {
            let validation = try promoteVerifiedPartial(
                partial,
                metadata: metadata,
                request: request,
                actualHash: ModelSnapshotVerifier.sha256(of: partial)
            )
            await progress(request.identity.expectedSize)
            return validation
        }

        var urlRequest = request.urlRequest
        if offset > 0 {
            urlRequest.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
            if let etag = partialIdentity.etag {
                urlRequest.setValue(etag, forHTTPHeaderField: "If-Range")
            }
        }

        let sink = DownloadFileSink(
            partial: partial,
            metadata: metadata,
            identity: partialIdentity,
            requestedOffset: offset,
            expectedSize: request.identity.expectedSize,
            expectedSHA256: request.identity.expectedSHA256,
            progress: progress
        )
        let actualHash: String
        do {
            try await httpClient.stream(
                for: urlRequest,
                onResponse: { response in
                    try await sink.accept(response: response)
                },
                onData: { data in
                    try await sink.append(data)
                }
            )
            actualHash = try await sink.finish()
        } catch {
            try? await sink.suspend()
            if case ResumableDownloadError.downloadedChecksumMismatch = error {
                try? FileManager.default.removeItem(at: partial)
                try? FileManager.default.removeItem(at: metadata)
            }
            throw error
        }

        offset = try Self.fileSize(partial)
        guard offset == request.identity.expectedSize else {
            throw ResumableDownloadError.downloadedSizeMismatch(
                expected: request.identity.expectedSize,
                actual: offset
            )
        }
        return try promoteVerifiedPartial(
            partial,
            metadata: metadata,
            request: request,
            actualHash: actualHash
        )
    }

    private func preparePartial(
        partial: URL,
        metadata: URL,
        identity: ModelPartialIdentity
    ) throws -> (offset: Int64, identity: ModelPartialIdentity) {
        let fileExists = FileManager.default.fileExists(atPath: partial.path)
        let storedIdentity = try? JSONDecoder().decode(
            ModelPartialIdentity.self,
            from: Data(contentsOf: metadata)
        )
        var effectiveIdentity = identity
        if fileExists,
           let storedIdentity,
           Self.sameFileIdentity(storedIdentity, identity),
           identity.etag == nil || storedIdentity.etag == identity.etag {
            effectiveIdentity.etag = identity.etag ?? storedIdentity.etag
        } else if fileExists {
            try FileManager.default.removeItem(at: partial)
            try? FileManager.default.removeItem(at: metadata)
        } else if !fileExists {
            try? FileManager.default.removeItem(at: metadata)
        }
        if !FileManager.default.fileExists(atPath: partial.path) {
            guard FileManager.default.createFile(atPath: partial.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try ModelDownloadStore.atomicWrite(identity, to: metadata)
        }
        let size = try Self.fileSize(partial)
        if size > identity.expectedSize {
            try FileManager.default.removeItem(at: partial)
            try? FileManager.default.removeItem(at: metadata)
            guard FileManager.default.createFile(atPath: partial.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try ModelDownloadStore.atomicWrite(identity, to: metadata)
            return (0, identity)
        }
        return (size, effectiveIdentity)
    }

    private static func sameFileIdentity(
        _ lhs: ModelPartialIdentity,
        _ rhs: ModelPartialIdentity
    ) -> Bool {
        lhs.version == rhs.version
            && lhs.provider == rhs.provider
            && lhs.repoID == rhs.repoID
            && lhs.commitSHA == rhs.commitSHA
            && lhs.path == rhs.path
            && lhs.expectedSize == rhs.expectedSize
            && lhs.expectedSHA256 == rhs.expectedSHA256
    }

    private func promoteVerifiedPartial(
        _ partial: URL,
        metadata: URL,
        request: ResumableDownloadRequest,
        actualHash: String
    ) throws -> ModelValidatedFile {
        guard actualHash == request.identity.expectedSHA256 else {
            try? FileManager.default.removeItem(at: partial)
            try? FileManager.default.removeItem(at: metadata)
            throw ResumableDownloadError.downloadedChecksumMismatch(
                expected: request.identity.expectedSHA256,
                actual: actualHash
            )
        }
        if FileManager.default.fileExists(atPath: request.destination.path) {
            try FileManager.default.removeItem(at: request.destination)
        }
        try FileManager.default.moveItem(at: partial, to: request.destination)
        try? FileManager.default.removeItem(at: metadata)
        return ModelValidatedFile(
            path: request.identity.path,
            sha256: actualHash,
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
        guard try Self.fileSize(url) == identity.expectedSize else {
            try FileManager.default.removeItem(at: url)
            return nil
        }
        let actualHash = try ModelSnapshotVerifier.sha256(of: url)
        guard actualHash == identity.expectedSHA256 else {
            try FileManager.default.removeItem(at: url)
            return nil
        }
        return ModelValidatedFile(
            path: identity.path,
            sha256: actualHash,
            identity: try ModelSnapshotVerifier.fileIdentity(of: url)
        )
    }

    private static func fileSize(_ url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }
}

private actor DownloadFileSink {
    private let partial: URL
    private let metadata: URL
    private var identity: ModelPartialIdentity
    private let requestedOffset: Int64
    private let expectedSize: Int64
    private let expectedSHA256: String
    private let progress: @Sendable (Int64) async -> Void
    private var handle: FileHandle?
    private var bytesWritten: Int64
    private var digest = SHA256()

    init(
        partial: URL,
        metadata: URL,
        identity: ModelPartialIdentity,
        requestedOffset: Int64,
        expectedSize: Int64,
        expectedSHA256: String,
        progress: @escaping @Sendable (Int64) async -> Void
    ) {
        self.partial = partial
        self.metadata = metadata
        self.identity = identity
        self.requestedOffset = requestedOffset
        self.expectedSize = expectedSize
        self.expectedSHA256 = expectedSHA256
        self.progress = progress
        bytesWritten = requestedOffset
    }

    func accept(response: HTTPURLResponse) async throws {
        let responseETag = response.value(forHTTPHeaderField: "ETag")
        if let expectedETag = identity.etag {
            guard responseETag == expectedETag else {
                throw ResumableDownloadError.responseIdentityChanged(
                    expected: expectedETag,
                    actual: responseETag
                )
            }
        } else if let responseETag {
            identity.etag = responseETag
            try ModelDownloadStore.atomicWrite(identity, to: metadata)
        }
        switch response.statusCode {
        case 206:
            let contentRange = response.value(forHTTPHeaderField: "Content-Range")
            guard Self.validContentRange(contentRange, offset: requestedOffset, expectedSize: expectedSize) else {
                throw ResumableDownloadError.invalidContentRange(contentRange)
            }
            let handle = try FileHandle(forWritingTo: partial)
            try handle.seekToEnd()
            self.handle = handle
            try seedDigestFromExistingPartial()
        case 200:
            let handle = try FileHandle(forWritingTo: partial)
            try handle.truncate(atOffset: 0)
            try handle.seek(toOffset: 0)
            bytesWritten = 0
            self.handle = handle
            digest = SHA256()
        case 416:
            throw ResumableDownloadError.rangeNotSatisfiable
        default:
            throw ModelDownloadHTTPError(statusCode: response.statusCode)
        }
        await progress(bytesWritten)
    }

    func append(_ data: Data) async throws {
        try Task.checkCancellation()
        guard let handle else {
            throw URLError(.badServerResponse)
        }
        let next = bytesWritten + Int64(data.count)
        guard next <= expectedSize else {
            throw ResumableDownloadError.downloadedSizeMismatch(expected: expectedSize, actual: next)
        }
        try handle.write(contentsOf: data)
        digest.update(data: data)
        bytesWritten = next
        await progress(bytesWritten)
    }

    func finish() throws -> String {
        try handle?.synchronize()
        try handle?.close()
        handle = nil
        let actualHash = digest.finalize().map { String(format: "%02x", $0) }.joined()
        guard actualHash == expectedSHA256 else {
            throw ResumableDownloadError.downloadedChecksumMismatch(
                expected: expectedSHA256,
                actual: actualHash
            )
        }
        return actualHash
    }

    func suspend() throws {
        try handle?.synchronize()
        try handle?.close()
        handle = nil
    }

    private func seedDigestFromExistingPartial() throws {
        guard requestedOffset > 0 else {
            digest = SHA256()
            return
        }
        let reader = try FileHandle(forReadingFrom: partial)
        defer { try? reader.close() }
        var remaining = requestedOffset
        var seeded = SHA256()
        while remaining > 0 {
            try Task.checkCancellation()
            let count = Int(min(Int64(4 * 1_024 * 1_024), remaining))
            let data = try reader.read(upToCount: count) ?? Data()
            guard !data.isEmpty else {
                throw ResumableDownloadError.downloadedSizeMismatch(
                    expected: requestedOffset,
                    actual: requestedOffset - remaining
                )
            }
            seeded.update(data: data)
            remaining -= Int64(data.count)
        }
        digest = seeded
    }

    private static func validContentRange(_ value: String?, offset: Int64, expectedSize: Int64) -> Bool {
        guard let value else {
            return false
        }
        let prefix = "bytes \(offset)-"
        guard value.hasPrefix(prefix),
              let slash = value.lastIndex(of: "/"),
              Int64(value[value.index(after: slash)...]) == expectedSize
        else {
            return false
        }
        return true
    }
}
