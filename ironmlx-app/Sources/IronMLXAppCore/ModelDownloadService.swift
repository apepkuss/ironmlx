import CryptoKit
import Foundation

public protocol ModelDownloadHTTPClient: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
    func stream(
        for request: URLRequest,
        onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData: @escaping @Sendable (Data) async throws -> Void
    ) async throws
}

public struct URLSessionModelDownloadHTTPClient: ModelDownloadHTTPClient {
    private let sessionConfiguration: URLSessionConfiguration

    public init(sessionConfiguration: URLSessionConfiguration = .default) {
        self.sessionConfiguration = sessionConfiguration
    }

    public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        try validate(http)
        return (data, http)
    }

    public func stream(
        for request: URLRequest,
        onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        let delegate = ModelDownloadStreamingDelegate(
            onResponse: onResponse,
            onData: onData
        )
        let delegateQueue = OperationQueue()
        delegateQueue.name = "com.ironmlx.model-download-stream"
        delegateQueue.maxConcurrentOperationCount = 1
        let session = URLSession(
            configuration: sessionConfiguration,
            delegate: delegate,
            delegateQueue: delegateQueue
        )
        defer { session.invalidateAndCancel() }
        try await delegate.run(request: request, session: session)
    }

    private func validate(_ response: HTTPURLResponse) throws {
        guard (200..<300).contains(response.statusCode) else {
            throw ModelDownloadHTTPError(statusCode: response.statusCode)
        }
    }
}

private final class ModelDownloadStreamingDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let onResponse: @Sendable (HTTPURLResponse) async throws -> Void
    private let onData: @Sendable (Data) async throws -> Void
    private let stateLock = NSLock()
    private var continuation: CheckedContinuation<Void, any Error>?
    private var dataTask: URLSessionDataTask?
    private var processingError: (any Error)?
    private var cancellationRequested = false
    private var completed = false

    init(
        onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData: @escaping @Sendable (Data) async throws -> Void
    ) {
        self.onResponse = onResponse
        self.onData = onData
    }

    func run(request: URLRequest, session: URLSession) async throws {
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try await withCheckedThrowingContinuation { continuation in
                let task = session.dataTask(with: request)
                stateLock.lock()
                self.continuation = continuation
                dataTask = task
                let shouldCancel = cancellationRequested
                stateLock.unlock()
                if shouldCancel {
                    task.cancel()
                } else {
                    task.resume()
                }
            }
        } onCancel: {
            self.cancel()
        }
    }

    func urlSession(
        _: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let response = response as? HTTPURLResponse else {
            recordProcessingError(URLError(.badServerResponse))
            completionHandler(.cancel)
            return
        }
        switch waitForAsyncOperation({ try await self.onResponse(response) }) {
        case .success:
            completionHandler(.allow)
        case let .failure(error):
            recordProcessingError(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(
        _: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        guard !shouldStopProcessing else {
            recordProcessingError(CancellationError())
            dataTask.cancel()
            return
        }
        switch waitForAsyncOperation({ try await self.onData(data) }) {
        case .success:
            if isCancellationRequested {
                recordProcessingError(CancellationError())
                dataTask.cancel()
            }
        case let .failure(error):
            recordProcessingError(error)
            dataTask.cancel()
        }
    }

    func urlSession(
        _: URLSession,
        task _: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        stateLock.lock()
        guard !completed else {
            stateLock.unlock()
            return
        }
        completed = true
        let continuation = continuation
        self.continuation = nil
        let finalError = processingError
            ?? (cancellationRequested ? CancellationError() : error)
        stateLock.unlock()

        if let finalError {
            continuation?.resume(throwing: finalError)
        } else {
            continuation?.resume()
        }
    }

    private var isCancellationRequested: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return cancellationRequested
    }

    private var shouldStopProcessing: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return cancellationRequested || processingError != nil
    }

    private func cancel() {
        stateLock.lock()
        cancellationRequested = true
        let dataTask = dataTask
        stateLock.unlock()
        dataTask?.cancel()
    }

    private func recordProcessingError(_ error: any Error) {
        stateLock.lock()
        if processingError == nil {
            processingError = error
        }
        stateLock.unlock()
    }

    private func waitForAsyncOperation(
        _ operation: @escaping @Sendable () async throws -> Void
    ) -> Result<Void, any Error> {
        let semaphore = DispatchSemaphore(value: 0)
        let result = ModelDownloadStreamingResult()
        Task {
            do {
                try await operation()
                result.store(.success(()))
            } catch {
                result.store(.failure(error))
            }
            semaphore.signal()
        }
        semaphore.wait()
        return result.value
    }
}

private final class ModelDownloadStreamingResult: @unchecked Sendable {
    private let lock = NSLock()
    private var storedValue: Result<Void, any Error>?

    var value: Result<Void, any Error> {
        lock.lock()
        defer { lock.unlock() }
        return storedValue!
    }

    func store(_ value: Result<Void, any Error>) {
        lock.lock()
        storedValue = value
        lock.unlock()
    }
}

public struct ModelDownloadHTTPError: LocalizedError, Sendable {
    public var statusCode: Int

    public var errorDescription: String? {
        "Download endpoint returned HTTP \(statusCode)."
    }
}

public struct HuggingFaceSearchResult: Codable, Equatable, Sendable {
    public var id: String
    public var modelId: String?
    public var downloads: Int?
    public var likes: Int?
    public var pipelineTag: String?

    enum CodingKeys: String, CodingKey {
        case id
        case modelId
        case downloads
        case likes
        case pipelineTag = "pipeline_tag"
    }
}

public struct ModelDownloadProgress: Equatable, Sendable {
    public var percent: Double
    public var filename: String
}

public struct ModelDownloadCompletion: Codable, Equatable, Sendable {
    public var success: Bool
    public var message: String?
    public var error: String?
    public var code: String?
    public var repoID: String?

    enum CodingKeys: String, CodingKey {
        case success
        case message
        case error
        case code
        case repoID = "repo_id"
    }
}

public struct ModelDownloadStartResponse: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var repoID: String
    public var error: String?
    public var code: String?

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case repoID = "repo_id"
        case error
        case code
    }
}

public struct ModelDownloadStatus: Codable, Equatable, Sendable {
    public var repoID: String
    public var provider: String
    public var status: String
    public var progressPct: Double
    public var currentFile: String?
    public var commitSHA: String?
    public var error: String?
    public var errorCode: String?

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case provider
        case status
        case progressPct = "progress_pct"
        case currentFile = "current_file"
        case commitSHA = "commit_sha"
        case error
        case errorCode = "error_code"
    }
}

private struct ModelDownloadResumePlan {
    var remainingBytes: Int64 = 0
    var resumedBytesByPath: [String: Int64] = [:]
}

public actor ModelDownloadService {
    private let rootURL: URL
    private let httpClient: any ModelDownloadHTTPClient
    private let huggingFaceEndpoint: URL
    private let resolver: ModelRepositoryResolver
    private let store: ModelDownloadStore
    private let downloader: any ModelFileDownloading
    private let metadataPreflight: any ModelMetadataPreflighting
    private let telemetryLogger: @Sendable (String) -> Void
    private var statuses: [String: ModelDownloadStatus] = [:]
    private var tasks: [String: Task<ModelDownloadCompletion, Never>] = [:]

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true),
        httpClient: any ModelDownloadHTTPClient = URLSessionModelDownloadHTTPClient(),
        huggingFaceEndpoint: URL = URL(string: "https://huggingface.co")!,
        modelScopeEndpoint: URL = URL(string: "https://modelscope.cn/api/v1/models")!,
        modelScopeGitEndpoint: URL = URL(string: "https://www.modelscope.cn")!,
        metadataPreflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight(),
        fileDownloader: (any ModelFileDownloading)? = nil,
        telemetryLogger: @escaping @Sendable (String) -> Void = {
            IronMLXAppLogger.info($0)
        }
    ) {
        self.rootURL = rootURL
        self.httpClient = httpClient
        self.huggingFaceEndpoint = huggingFaceEndpoint
        resolver = ModelRepositoryResolver(
            httpClient: httpClient,
            huggingFaceEndpoint: huggingFaceEndpoint,
            modelScopeAPIEndpoint: modelScopeEndpoint,
            modelScopeGitEndpoint: modelScopeGitEndpoint
        )
        store = ModelDownloadStore(rootURL: rootURL)
        downloader = fileDownloader ?? ProviderModelFileDownloader(httpClient: httpClient)
        self.metadataPreflight = metadataPreflight
        self.telemetryLogger = telemetryLogger

        for journal in store.recoverInterruptedJournals() {
            let key = Self.taskKey(provider: journal.provider, repoID: journal.repoID)
            statuses[key] = Self.status(from: journal)
        }
    }

    public func searchHuggingFace(query: String, sort: String) async throws -> [HuggingFaceSearchResult] {
        var components = URLComponents(
            url: huggingFaceEndpoint
                .appendingPathComponent("api")
                .appendingPathComponent("models"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "search", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "20"),
            URLQueryItem(name: "filter", value: "mlx"),
        ]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        let (data, _) = try await httpClient.data(for: URLRequest(url: url))
        return try JSONDecoder().decode([HuggingFaceSearchResult].self, from: data)
    }

    public func downloadHuggingFace(
        repoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void = { _ in }
    ) async -> ModelDownloadCompletion {
        await download(
            provider: .huggingFace,
            repoID: repoID,
            token: token,
            progress: progress
        )
    }

    public func startModelScopeDownload(repoID: String) async -> ModelDownloadStartResponse {
        let repoID = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isValidRepoID(repoID) else {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "repo_id must be organization/model",
                code: "invalid_repo_id"
            )
        }
        let key = Self.taskKey(provider: .modelScope, repoID: repoID)
        guard tasks[key] == nil else {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "\(repoID) is already downloading",
                code: "download_in_progress"
            )
        }
        let task = Task {
            await self.executeDownload(
                provider: .modelScope,
                repoID: repoID,
                token: nil,
                progress: { _ in }
            )
        }
        tasks[key] = task
        Task {
            _ = await task.value
            self.clearTask(key)
        }
        return ModelDownloadStartResponse(
            success: true,
            status: "accepted",
            repoID: repoID,
            error: nil,
            code: nil
        )
    }

    public func cancelDownload(provider: ModelRepositoryProvider, repoID: String) -> Bool {
        let key = Self.taskKey(provider: provider, repoID: repoID)
        guard let task = tasks[key] else {
            return false
        }
        task.cancel()
        return true
    }

    public func cancelAllDownloads() {
        for task in tasks.values {
            task.cancel()
        }
    }

    public func downloadStatuses() -> [ModelDownloadStatus] {
        statuses.values.sorted {
            if $0.repoID == $1.repoID {
                return $0.provider < $1.provider
            }
            return $0.repoID.localizedStandardCompare($1.repoID) == .orderedAscending
        }
    }

    private func download(
        provider: ModelRepositoryProvider,
        repoID rawRepoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion {
        let repoID = rawRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isValidRepoID(repoID) else {
            return failure(message: "repo_id must be organization/model", code: "invalid_repo_id", repoID: repoID)
        }
        let key = Self.taskKey(provider: provider, repoID: repoID)
        guard tasks[key] == nil else {
            return failure(
                message: "\(repoID) is already downloading",
                code: "download_in_progress",
                repoID: repoID
            )
        }
        let task = Task {
            await self.executeDownload(
                provider: provider,
                repoID: repoID,
                token: token,
                progress: progress
            )
        }
        tasks[key] = task
        let result = await task.value
        tasks[key] = nil
        return result
    }

    private func executeDownload(
        provider: ModelRepositoryProvider,
        repoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion {
        let key = Self.taskKey(provider: provider, repoID: repoID)
        let telemetry = ModelDownloadTelemetryTracker(
            provider: provider,
            repoID: repoID,
            logger: telemetryLogger
        )
        await telemetry.start()
        setStatus(key: key, provider: provider, repoID: repoID, phase: .resolving)
        var journal: ModelDownloadJournal?

        do {
            let repositoryLock = try store.acquireRepositoryLock(provider: provider, repoID: repoID)
            defer { withExtendedLifetime(repositoryLock) {} }
            let repository = try await resolver.resolve(provider: provider, repoID: repoID, token: token)
            setStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                phase: .preflighting,
                commitSHA: repository.commitSHA
            )

            let finalSnapshot = try store.snapshotURL(
                provider: provider,
                repoID: repoID,
                commitSHA: repository.commitSHA
            )
            if FileManager.default.fileExists(atPath: finalSnapshot.path) {
                let manifest = try ModelSnapshotVerifier().verifyStructure(
                    snapshot: finalSnapshot,
                    expectedProvider: provider,
                    expectedRepoID: repoID
                )
                try store.updateRef(for: manifest)
                setStatus(
                    key: key,
                    provider: provider,
                    repoID: repoID,
                    phase: .completed,
                    progressPct: 100,
                    commitSHA: repository.commitSHA
                )
                await telemetry.finish(outcome: "already_present")
                return success(repoID: repoID)
            }

            let staging = try store.prepareStaging(
                provider: provider,
                repoID: repoID,
                commitSHA: repository.commitSHA
            )
            journal = ModelDownloadJournal(
                provider: provider,
                repoID: repoID,
                requestedRevision: repository.requestedRevision,
                commitSHA: repository.commitSHA,
                phase: .preflighting
            )
            try store.writeJournal(journal!)

            await telemetry.beginNetwork()
            let metadataResult = try await acquireMetadata(
                repository: repository,
                token: token,
                staging: staging,
                telemetry: telemetry
            )
            await telemetry.endNetwork()
            let metadata = metadataResult.files
            var validations = metadataResult.validations
            let weights = try selectedWeightFiles(repository: repository, staging: staging)
            guard !weights.isEmpty else {
                throw DownloadFailure(
                    repoID: repoID,
                    code: "missing_safetensors",
                    message: "Repository \(repoID) does not contain a complete safetensors weight set."
                )
            }
            for weight in weights {
                guard let sha256 = weight.sha256,
                      sha256.count == 64,
                      sha256.allSatisfy(\.isHexDigit)
                else {
                    throw DownloadFailure(
                        repoID: repoID,
                        code: "weight_identity_missing",
                        message: "Weight \(weight.path) has no provider SHA-256."
                    )
                }
            }

            let compatibility: ModelMetadataPreflightResult
            do {
                compatibility = try await metadataPreflight.validate(metadataDirectory: staging)
            } catch {
                throw DownloadFailure(
                    repoID: repoID,
                    code: "unsupported_model_metadata",
                    message: error.localizedDescription
                )
            }
            let weightBytes = try checkedTotalBytes(
                weights,
                repoID: repoID,
                size: \.size
            )
            let resumePlan = try downloadResumePlan(
                files: weights,
                staging: staging,
                repository: repository
            )
            let remainingBytes = resumePlan.remainingBytes
            let resources = ModelResourcePreflight(
                weightBytes: weightBytes,
                remainingDownloadBytes: remainingBytes,
                availableDiskBytes: availableCapacity(at: rootURL),
                physicalMemoryBytes: Int64(clamping: ProcessInfo.processInfo.physicalMemory)
            )
            try resources.validate()

            let metadataBytes = try checkedTotalBytes(
                metadata,
                repoID: repoID,
                size: \.size
            )
            let totalResult = metadataBytes.addingReportingOverflow(weightBytes)
            guard !totalResult.overflow else {
                throw DownloadFailure(
                    repoID: repoID,
                    code: "repo_size_overflow",
                    message: "Repository \(repoID) reports an invalid total file size."
                )
            }
            let totalBytes = totalResult.partialValue
            await telemetry.setExpectedBytes(totalBytes)
            var completedBase = metadataBytes
            journal?.phase = .downloading
            journal?.totalBytes = totalBytes
            journal?.progressBytes = completedBase
            journal?.updatedAt = Date()
            try store.writeJournal(journal!)

            await telemetry.beginNetwork()
            var downloadedFiles = metadata
            for weight in weights {
                try Task.checkCancellation()
                let request = try resolver.request(for: weight, repository: repository, token: token)
                let destination = try ModelSnapshotVerifier.safeFileURL(path: weight.path, beneath: staging)
                let identity = ModelPartialIdentity(
                    provider: provider,
                    repoID: repoID,
                    commitSHA: repository.commitSHA,
                    path: weight.path,
                    expectedSize: weight.size,
                    expectedSHA256: weight.sha256!,
                    etag: weight.etag
                )
                await telemetry.registerFile(
                    path: weight.path,
                    expectedBytes: weight.size,
                    resumedBytes: resumePlan.resumedBytesByPath[weight.path] ?? 0
                )
                journal?.currentFile = weight.path
                journal?.updatedAt = Date()
                try store.writeJournal(journal!)
                let fileBase = completedBase
                let validation = try await downloader.download(
                    ResumableDownloadRequest(
                        urlRequest: request,
                        identity: identity,
                        destination: destination
                    ),
                    progress: { fileBytes in
                        await telemetry.recordProgress(
                            path: weight.path,
                            availableBytes: fileBytes
                        )
                        let aggregate = fileBase + fileBytes
                        await self.reportProgress(
                            key: key,
                            filename: weight.path,
                            bytes: aggregate,
                            totalBytes: totalBytes,
                            callback: progress
                        )
                    }
                )
                validations.append(validation)
                completedBase += weight.size
                journal?.progressBytes = completedBase
                journal?.updatedAt = Date()
                try store.writeJournal(journal!)
                downloadedFiles.append(
                    ModelSnapshotFile(
                        path: weight.path,
                        size: weight.size,
                        sha256: weight.sha256!,
                        etag: validation.etag ?? weight.etag,
                        blobID: weight.blobID
                    )
                )
            }

            await telemetry.endNetwork()
            await telemetry.beginVerification()
            journal?.phase = .verifying
            journal?.progressBytes = totalBytes
            journal?.currentFile = nil
            journal?.updatedAt = Date()
            try store.writeJournal(journal!)
            setStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                phase: .verifying,
                progressPct: 100,
                commitSHA: repository.commitSHA
            )

            let manifest = ModelSnapshotManifest(
                provider: provider,
                repoID: repoID,
                requestedRevision: repository.requestedRevision,
                commitSHA: repository.commitSHA,
                files: downloadedFiles,
                compatibility: ModelSnapshotCompatibility(
                    modelType: compatibility.modelType,
                    artifactRole: compatibility.artifactRole,
                    quantizationMode: compatibility.quantization?.mode,
                    quantizationBits: compatibility.quantization?.bits,
                    quantizationGroupSize: compatibility.quantization?.groupSize
                ),
                resources: ModelSnapshotResources(
                    weightBytes: weightBytes,
                    estimatedPeakMemoryBytes: resources.estimatedPeakMemoryBytes
                )
            )
            try store.writeManifest(manifest, to: staging)
            try ModelSnapshotVerifier().verifyForPublish(
                snapshot: staging,
                manifest: manifest,
                validations: validations
            )
            try ModelDownloadStore.atomicWrite(
                ModelSnapshotIntegrityRecord(
                    provider: provider,
                    repoID: repoID,
                    commitSHA: repository.commitSHA,
                    state: .verified,
                    verifiedAt: Date()
                ),
                to: staging.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
            )
            await telemetry.endVerification()

            journal?.phase = .publishing
            journal?.updatedAt = Date()
            try store.writeJournal(journal!)
            setStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                phase: .publishing,
                progressPct: 100,
                commitSHA: repository.commitSHA
            )
            await telemetry.beginPublication()
            _ = try store.publish(manifest)
            await telemetry.endPublication()

            journal?.phase = .completed
            journal?.updatedAt = Date()
            try store.writeJournal(journal!)
            setStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                phase: .completed,
                progressPct: 100,
                commitSHA: repository.commitSHA
            )
            await telemetry.finish(outcome: "completed")
            return success(repoID: repoID)
        } catch is CancellationError {
            await telemetry.finish(outcome: "cancelled", errorCode: "cancelled")
            persistFailure(&journal, phase: .cancelled, code: "cancelled", message: "Download cancelled.")
            setFailureStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                journal: journal,
                phase: .cancelled,
                code: "cancelled",
                message: "Download cancelled."
            )
            return failure(message: "Download cancelled.", code: "cancelled", repoID: repoID)
        } catch let error as DownloadFailure {
            await telemetry.finish(outcome: "rejected", errorCode: error.code)
            persistFailure(&journal, phase: .rejected, code: error.code, message: error.message)
            setFailureStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                journal: journal,
                phase: .rejected,
                code: error.code,
                message: error.message
            )
            return failure(message: error.message, code: error.code, repoID: repoID)
        } catch let error as ModelResourcePreflightError {
            let code: String
            switch error {
            case .insufficientDisk:
                code = "insufficient_disk"
            case .insufficientMemory:
                code = "insufficient_memory"
            }
            await telemetry.finish(outcome: "rejected", errorCode: code)
            persistFailure(&journal, phase: .rejected, code: code, message: error.localizedDescription)
            setFailureStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                journal: journal,
                phase: .rejected,
                code: code,
                message: error.localizedDescription
            )
            return failure(message: error.localizedDescription, code: code, repoID: repoID)
        } catch let error as ResumableDownloadError {
            let phase: ModelDownloadPhase
            switch error {
            case .downloadedChecksumMismatch:
                phase = .corrupt
            default:
                phase = .interrupted
            }
            await telemetry.finish(outcome: phase.rawValue, errorCode: "download_interrupted")
            persistFailure(&journal, phase: phase, code: "download_interrupted", message: error.localizedDescription)
            setFailureStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                journal: journal,
                phase: phase,
                code: "download_interrupted",
                message: error.localizedDescription
            )
            return failure(message: error.localizedDescription, code: "download_interrupted", repoID: repoID)
        } catch {
            let message = error.localizedDescription
            let phase: ModelDownloadPhase = journal == nil ? .rejected : .interrupted
            await telemetry.finish(outcome: phase.rawValue, errorCode: "download_failed")
            persistFailure(&journal, phase: phase, code: "download_failed", message: message)
            setFailureStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                journal: journal,
                phase: phase,
                code: "download_failed",
                message: message
            )
            return failure(message: message, code: "download_failed", repoID: repoID)
        }
    }

    private func acquireMetadata(
        repository: ResolvedModelRepository,
        token: String?,
        staging: URL,
        telemetry: ModelDownloadTelemetryTracker
    ) async throws -> (files: [ModelSnapshotFile], validations: [ModelValidatedFile]) {
        let required = ["config.json", "tokenizer.json"]
        let byPath = Dictionary(uniqueKeysWithValues: repository.files.map { ($0.path, $0) })
        for path in required where byPath[path] == nil {
            throw DownloadFailure(
                repoID: repository.repoID,
                code: "repo_missing_metadata",
                message: "Repository \(repository.repoID) is missing \(path)."
            )
        }
        let metadataFiles = repository.files.filter(Self.isRuntimeMetadata)
        var resolved: [ModelSnapshotFile] = []
        var validations: [ModelValidatedFile] = []
        for file in metadataFiles {
            try Task.checkCancellation()
            let destination = try ModelSnapshotVerifier.safeFileURL(path: file.path, beneath: staging)
            let request = try resolver.request(for: file, repository: repository, token: token)
            if let sha256 = file.sha256 {
                let identity = ModelPartialIdentity(
                    provider: repository.provider,
                    repoID: repository.repoID,
                    commitSHA: repository.commitSHA,
                    path: file.path,
                    expectedSize: file.size,
                    expectedSHA256: sha256,
                    etag: file.etag
                )
                await telemetry.registerFile(
                    path: file.path,
                    expectedBytes: file.size,
                    resumedBytes: recoverableBytes(
                        destination: destination,
                        identity: identity
                    )
                )
                let validation = try await downloader.download(
                    ResumableDownloadRequest(
                        urlRequest: request,
                        identity: identity,
                        destination: destination
                    ),
                    progress: { bytes in
                        await telemetry.recordProgress(
                            path: file.path,
                            availableBytes: bytes
                        )
                    }
                )
                validations.append(validation)
                resolved.append(
                    ModelSnapshotFile(
                        path: file.path,
                        size: file.size,
                        sha256: sha256,
                        etag: validation.etag ?? file.etag,
                        blobID: file.blobID
                    )
                )
                continue
            }

            await telemetry.registerFile(
                path: file.path,
                expectedBytes: file.size,
                resumedBytes: 0
            )
            let (data, response) = try await httpClient.data(for: request)
            if let responseCommit = response.value(forHTTPHeaderField: "X-Repo-Commit"),
               responseCommit.lowercased() != repository.commitSHA {
                throw DownloadFailure(
                    repoID: repository.repoID,
                    code: "remote_identity_changed",
                    message: "Remote commit changed while fetching \(file.path)."
                )
            }
            guard Int64(data.count) == file.size else {
                throw DownloadFailure(
                    repoID: repository.repoID,
                    code: "metadata_size_mismatch",
                    message: "Metadata \(file.path) has size \(data.count), expected \(file.size)."
                )
            }
            if let blobID = file.blobID?.lowercased() {
                let actualBlobID = Self.gitBlobSHA1(data)
                guard blobID == actualBlobID else {
                    throw DownloadFailure(
                        repoID: repository.repoID,
                        code: "metadata_identity_mismatch",
                        message: "Metadata \(file.path) does not match remote blob identity."
                    )
                }
            }
            try ModelDownloadStore.atomicWrite(data, to: destination)
            await telemetry.recordProgress(
                path: file.path,
                availableBytes: file.size
            )
            let actualSHA256 = Self.sha256(data)
            validations.append(
                ModelValidatedFile(
                    path: file.path,
                    sha256: actualSHA256,
                    identity: try ModelSnapshotVerifier.fileIdentity(of: destination)
                )
            )
            resolved.append(
                ModelSnapshotFile(
                    path: file.path,
                    size: file.size,
                    sha256: actualSHA256,
                    etag: response.value(forHTTPHeaderField: "ETag"),
                    blobID: file.blobID
                )
            )
        }
        return (
            resolved.sorted { $0.path < $1.path },
            validations.sorted { $0.path < $1.path }
        )
    }

    private func selectedWeightFiles(
        repository: ResolvedModelRepository,
        staging: URL
    ) throws -> [RemoteModelFile] {
        let byPath = Dictionary(uniqueKeysWithValues: repository.files.map { ($0.path, $0) })
        var selected = Set<String>()
        let indexURL = staging.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.isReadableFile(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = object["weight_map"] as? [String: Any],
                  !weightMap.isEmpty
            else {
                throw DownloadFailure(
                    repoID: repository.repoID,
                    code: "repo_broken_shards",
                    message: "model.safetensors.index.json is invalid."
                )
            }
            selected.formUnion(weightMap.values.compactMap { $0 as? String })
        } else if byPath["model.safetensors"] != nil {
            selected.insert("model.safetensors")
        } else {
            selected.formUnion(repository.files.filter(\.isWeight).map(\.path))
        }

        let configURL = staging.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let optiqVision = config["optiq_vision"] as? [String: Any],
           let sidecar = optiqVision["sidecar"] as? String,
           !sidecar.isEmpty {
            selected.insert(sidecar)
        }

        return try selected.sorted().map { path in
            guard let file = byPath[path], file.isWeight else {
                throw DownloadFailure(
                    repoID: repository.repoID,
                    code: "repo_broken_shards",
                    message: "Repository is missing weight \(path)."
                )
            }
            return file
        }
    }

    private func downloadResumePlan(
        files: [RemoteModelFile],
        staging: URL,
        repository: ResolvedModelRepository
    ) throws -> ModelDownloadResumePlan {
        try files.reduce(into: ModelDownloadResumePlan()) { plan, file in
            let destination = try ModelSnapshotVerifier.safeFileURL(path: file.path, beneath: staging)
            let identity = ModelPartialIdentity(
                provider: repository.provider,
                repoID: repository.repoID,
                commitSHA: repository.commitSHA,
                path: file.path,
                expectedSize: file.size,
                expectedSHA256: file.sha256 ?? "",
                etag: file.etag
            )
            let existing = recoverableBytes(
                destination: destination,
                identity: identity
            )
            plan.resumedBytesByPath[file.path] = existing
            let remaining = max(0, file.size - existing)
            let result = plan.remainingBytes.addingReportingOverflow(remaining)
            guard !result.overflow else {
                throw DownloadFailure(
                    repoID: repository.repoID,
                    code: "repo_size_overflow",
                    message: "Repository \(repository.repoID) reports an invalid total file size."
                )
            }
            plan.remainingBytes = result.partialValue
        }
    }

    private func recoverableBytes(
        destination: URL,
        identity: ModelPartialIdentity
    ) -> Int64 {
        if let size = try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           Int64(size) == identity.expectedSize {
            return identity.expectedSize
        }
        switch identity.provider {
        case .huggingFace:
            guard identity.expectedSHA256.count == 64 else {
                return 0
            }
            return RustHuggingFaceFileDownloader.recoverableBytes(
                destination: destination,
                identity: identity
            )
        case .modelScope:
            let partial = destination.appendingPathExtension("partial")
            if let size = try? partial.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                return min(identity.expectedSize, Int64(size))
            }
            return 0
        }
    }

    private func checkedTotalBytes<Element>(
        _ values: [Element],
        repoID: String,
        size: KeyPath<Element, Int64>
    ) throws -> Int64 {
        try values.reduce(Int64(0)) { total, value in
            let result = total.addingReportingOverflow(value[keyPath: size])
            guard !result.overflow else {
                throw DownloadFailure(
                    repoID: repoID,
                    code: "repo_size_overflow",
                    message: "Repository \(repoID) reports an invalid total file size."
                )
            }
            return result.partialValue
        }
    }

    private func availableCapacity(at url: URL) -> Int64? {
        let existing = Self.existingAncestor(of: url)
        guard let values = try? existing.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey,
        ]) else {
            return nil
        }
        if let important = values.volumeAvailableCapacityForImportantUsage {
            return important
        }
        return values.volumeAvailableCapacity.map(Int64.init)
    }

    private func reportProgress(
        key: String,
        filename: String,
        bytes: Int64,
        totalBytes: Int64,
        callback: @Sendable (ModelDownloadProgress) async -> Void
    ) async {
        let percent = totalBytes > 0
            ? min(100, Double(bytes) / Double(totalBytes) * 100)
            : 0
        statuses[key]?.progressPct = percent
        statuses[key]?.currentFile = filename
        await callback(ModelDownloadProgress(percent: percent, filename: filename))
    }

    private func setStatus(
        key: String,
        provider: ModelRepositoryProvider,
        repoID: String,
        phase: ModelDownloadPhase,
        progressPct: Double = 0,
        currentFile: String? = nil,
        commitSHA: String? = nil
    ) {
        statuses[key] = ModelDownloadStatus(
            repoID: repoID,
            provider: provider.rawValue,
            status: phase.rawValue,
            progressPct: progressPct,
            currentFile: currentFile,
            commitSHA: commitSHA,
            error: nil,
            errorCode: nil
        )
    }

    private func setFailureStatus(
        key: String,
        provider: ModelRepositoryProvider,
        repoID: String,
        journal: ModelDownloadJournal?,
        phase: ModelDownloadPhase,
        code: String?,
        message: String
    ) {
        statuses[key] = ModelDownloadStatus(
            repoID: repoID,
            provider: provider.rawValue,
            status: phase.rawValue,
            progressPct: journal.map {
                $0.totalBytes > 0 ? Double($0.progressBytes) / Double($0.totalBytes) * 100 : 0
            } ?? 0,
            currentFile: journal?.currentFile,
            commitSHA: journal?.commitSHA,
            error: message,
            errorCode: code
        )
    }

    private func persistFailure(
        _ journal: inout ModelDownloadJournal?,
        phase: ModelDownloadPhase,
        code: String?,
        message: String
    ) {
        guard journal != nil else {
            return
        }
        journal?.phase = phase
        journal?.errorCode = code
        journal?.error = message
        journal?.updatedAt = Date()
        try? store.writeJournal(journal!)
    }

    private func clearTask(_ key: String) {
        tasks[key] = nil
    }

    private func isValidRepoID(_ repoID: String) -> Bool {
        (try? ModelRepositoryLayout.repositoryName(repoID: repoID)) != nil
    }

    private func success(repoID: String) -> ModelDownloadCompletion {
        ModelDownloadCompletion(
            success: true,
            message: "Model \(repoID) downloaded and verified successfully.",
            error: nil,
            code: nil,
            repoID: repoID
        )
    }

    private func failure(message: String, code: String?, repoID: String) -> ModelDownloadCompletion {
        ModelDownloadCompletion(
            success: false,
            message: nil,
            error: message,
            code: code,
            repoID: repoID
        )
    }

    private static func taskKey(provider: ModelRepositoryProvider, repoID: String) -> String {
        "\(provider.rawValue):\(repoID)"
    }

    private static func status(from journal: ModelDownloadJournal) -> ModelDownloadStatus {
        ModelDownloadStatus(
            repoID: journal.repoID,
            provider: journal.provider.rawValue,
            status: journal.phase.rawValue,
            progressPct: journal.totalBytes > 0
                ? Double(journal.progressBytes) / Double(journal.totalBytes) * 100
                : 0,
            currentFile: journal.currentFile,
            commitSHA: journal.commitSHA,
            error: journal.error,
            errorCode: journal.errorCode
        )
    }

    private static func isRuntimeMetadata(_ file: RemoteModelFile) -> Bool {
        guard !file.isWeight else {
            return false
        }
        let name = file.path.lowercased()
        if name == "readme.md" || name == ".gitattributes" || name.hasPrefix("license") {
            return false
        }
        return name.hasSuffix(".json")
            || name.hasSuffix(".jinja")
            || name.hasSuffix(".txt")
            || name.hasSuffix(".model")
    }

    private static func sha256(_ data: Data) -> String {
        importCryptoSHA256(data)
    }

    private static func gitBlobSHA1(_ data: Data) -> String {
        var digest = Insecure.SHA1()
        digest.update(data: Data("blob \(data.count)\0".utf8))
        digest.update(data: data)
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func existingAncestor(of url: URL) -> URL {
        var candidate = url
        while !FileManager.default.fileExists(atPath: candidate.path) {
            let parent = candidate.deletingLastPathComponent()
            if parent.path == candidate.path {
                return URL(fileURLWithPath: "/", isDirectory: true)
            }
            candidate = parent
        }
        return candidate
    }
}

private struct DownloadFailure: Error {
    var repoID: String
    var code: String?
    var message: String
}

private func importCryptoSHA256(_ data: Data) -> String {
    // Kept outside the actor to avoid retaining any mutable hashing state.
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
