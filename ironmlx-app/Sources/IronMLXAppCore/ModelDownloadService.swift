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
    public var sha: String?
    public var downloads: Int?
    public var likes: Int?
    public var pipelineTag: String?
    public var localState: ModelSearchLocalState?
    public var localCommitSHA: String?

    enum CodingKeys: String, CodingKey {
        case id
        case modelId
        case sha
        case downloads
        case likes
        case pipelineTag = "pipeline_tag"
        case localState = "local_state"
        case localCommitSHA = "local_commit_sha"
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
    public var queuePosition: Int? = nil

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case repoID = "repo_id"
        case error
        case code
        case queuePosition = "queue_position"
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
    public var queuePosition: Int? = nil
    public var totalBytes: Int64? = nil
    public var remainingBytes: Int64? = nil
    public var enqueuedAt: Date? = nil

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case provider
        case status
        case progressPct = "progress_pct"
        case currentFile = "current_file"
        case commitSHA = "commit_sha"
        case error
        case errorCode = "error_code"
        case queuePosition = "queue_position"
        case totalBytes = "total_bytes"
        case remainingBytes = "remaining_bytes"
        case enqueuedAt = "enqueued_at"
    }
}

public struct ModelDownloadQueueSnapshot: Codable, Equatable, Sendable {
    public var maxConcurrent: Int
    public var activeCount: Int
    public var queuedCount: Int
    public var tasks: [ModelDownloadStatus]
    public var recoveryReminders: [ModelDownloadRecoveryReminder]

    enum CodingKeys: String, CodingKey {
        case maxConcurrent = "max_concurrent"
        case activeCount = "active_count"
        case queuedCount = "queued_count"
        case tasks
        case recoveryReminders = "recovery_reminders"
    }
}

private struct ModelDownloadResumePlan {
    var remainingBytes: Int64 = 0
    var resumedBytesByPath: [String: Int64] = [:]
}

private struct QueuedModelDownload: Sendable {
    var provider: ModelRepositoryProvider
    var repoID: String
    var token: String?
    var progress: @Sendable (ModelDownloadProgress) async -> Void
    var order: Int
    var enqueuedAt: Date
}

private enum DownloadPreparationState: Equatable, Sendable {
    case awaiting
    case preflighting
    case ready
}

private struct DownloadQueuePreflightEstimate: Sendable {
    var commitSHA: String
    var totalBytes: Int64
    var remainingBytes: Int64
    var observedBytesByPath: [String: Int64]
}

public actor ModelDownloadService {
    public static let defaultMaxConcurrentDownloads = 3

    private let rootURL: URL
    private let httpClient: any ModelDownloadHTTPClient
    private let huggingFaceEndpoint: URL
    private let resolver: ModelRepositoryResolver
    private let store: ModelDownloadStore
    private let downloader: any ModelFileDownloading
    private let metadataPreflight: any ModelMetadataPreflighting
    private let telemetryLogger: @Sendable (String) -> Void
    private let maxConcurrentDownloads: Int
    private let availableCapacityProvider: @Sendable (URL) -> Int64?
    private let reminderStore: ModelDownloadQueueReminderStore
    private var statuses: [String: ModelDownloadStatus] = [:]
    private var activeTasks: [String: Task<ModelDownloadCompletion, Never>] = [:]
    private var pendingDownloads: [QueuedModelDownload] = []
    private var preparationStates: [String: DownloadPreparationState] = [:]
    private var activePreflightKey: String?
    private var activePreflightTask: Task<Void, Never>?
    private var completionResults: [String: ModelDownloadCompletion] = [:]
    private var completionWaiters: [String: [CheckedContinuation<ModelDownloadCompletion, Never>]] = [:]
    private var recoveryReminders: [String: ModelDownloadRecoveryReminder] = [:]
    private var currentReminders: [String: ModelDownloadRecoveryReminder] = [:]
    private var shutdownReminderKeys: Set<String> = []
    private var isShuttingDown = false
    private var diskReservations: [String: Int64] = [:]
    private var reservationObservedBytes: [String: [String: Int64]] = [:]
    private var nextQueueOrder = 0

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true),
        httpClient: any ModelDownloadHTTPClient = URLSessionModelDownloadHTTPClient(),
        huggingFaceEndpoint: URL = URL(string: "https://huggingface.co")!,
        modelScopeEndpoint: URL = URL(string: "https://modelscope.cn/api/v1/models")!,
        modelScopeGitEndpoint: URL = URL(string: "https://www.modelscope.cn")!,
        metadataPreflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight(),
        fileDownloader: (any ModelFileDownloading)? = nil,
        maxConcurrentDownloads: Int = ModelDownloadService.defaultMaxConcurrentDownloads,
        availableCapacityProvider: (@Sendable (URL) -> Int64?)? = nil,
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
        reminderStore = ModelDownloadQueueReminderStore(rootURL: rootURL)
        downloader = fileDownloader ?? ProviderModelFileDownloader(httpClient: httpClient)
        self.metadataPreflight = metadataPreflight
        self.maxConcurrentDownloads = max(1, maxConcurrentDownloads)
        self.availableCapacityProvider = availableCapacityProvider ?? { url in
            Self.systemAvailableCapacity(at: url)
        }
        self.telemetryLogger = telemetryLogger

        for reminder in reminderStore.load() {
            let key = Self.taskKey(provider: reminder.provider, repoID: reminder.repoID)
            recoveryReminders[key] = reminder
            nextQueueOrder = max(nextQueueOrder, reminder.queueOrder + 1)
        }

        // Journals are durable integrity/recovery records, not the current App
        // session's queue history. Recover interrupted writes on launch without
        // flooding the queue UI with every previously completed download.
        _ = store.recoverInterruptedJournals()
    }

    public func searchHuggingFace(
        query: String,
        sort: String,
        token: String?
    ) async throws -> [HuggingFaceSearchResult] {
        let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if Self.isCanonicalHuggingFaceRepoID(normalizedQuery) {
            let repository = try await resolver.resolve(
                provider: .huggingFace,
                repoID: normalizedQuery,
                token: token
            )
            return [
                HuggingFaceSearchResult(
                    id: repository.repoID,
                    modelId: repository.repoID,
                    sha: repository.commitSHA,
                    downloads: nil,
                    likes: nil,
                    pipelineTag: nil,
                    localState: nil,
                    localCommitSHA: nil
                ),
            ]
        }

        var components = URLComponents(
            url: huggingFaceEndpoint
                .appendingPathComponent("api")
                .appendingPathComponent("models"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "search", value: normalizedQuery),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "20"),
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "full", value: "true"),
        ]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        Self.applyHuggingFaceToken(token, to: &request)
        let (data, _) = try await httpClient.data(for: request)
        var results = try JSONDecoder().decode([HuggingFaceSearchResult].self, from: data)
        for index in results.indices
            where results[index].sha.map(ModelSnapshotVerifier.isCommitSHA) != true
        {
            try Task.checkCancellation()
            let repoID = results[index].modelId ?? results[index].id
            do {
                let repository = try await resolver.resolve(
                    provider: .huggingFace,
                    repoID: repoID,
                    token: token
                )
                results[index].sha = repository.commitSHA
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                continue
            }
        }
        try Task.checkCancellation()
        return results
    }

    public nonisolated static func isCanonicalHuggingFaceRepoID(_ value: String) -> Bool {
        let components = value.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2 else {
            return false
        }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return components.allSatisfy { component in
            !component.isEmpty && component.unicodeScalars.allSatisfy(allowed.contains)
        }
    }

    public func downloadHuggingFace(
        repoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void = { _ in }
    ) async -> ModelDownloadCompletion {
        await downloadAndWait(
            provider: .huggingFace,
            repoID: repoID,
            token: token,
            progress: progress
        )
    }

    public func startHuggingFaceDownload(
        repoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void = { _ in }
    ) -> ModelDownloadStartResponse {
        enqueueDownload(
            provider: .huggingFace,
            repoID: repoID,
            token: token,
            progress: progress
        )
    }

    public func startModelScopeDownload(repoID: String) async -> ModelDownloadStartResponse {
        enqueueDownload(
            provider: .modelScope,
            repoID: repoID,
            token: nil,
            progress: { _ in }
        )
    }

    public func cancelDownload(provider: ModelRepositoryProvider, repoID: String) -> Bool {
        let key = Self.taskKey(provider: provider, repoID: repoID)
        if activePreflightKey == key {
            activePreflightTask?.cancel()
            return true
        }
        if let index = pendingDownloads.firstIndex(where: { Self.taskKey(provider: $0.provider, repoID: $0.repoID) == key }) {
            let request = pendingDownloads.remove(at: index)
            preparationStates[key] = nil
            releaseDiskReservation(key: key)
            setFailureStatus(
                key: key,
                provider: request.provider,
                repoID: request.repoID,
                journal: nil,
                phase: .cancelled,
                code: "cancelled",
                message: "Download cancelled."
            )
            finishQueuedTask(
                key: key,
                result: failure(message: "Download cancelled.", code: "cancelled", repoID: request.repoID)
            )
            refreshQueuePositions()
            pumpPreflightQueue()
            pumpDownloadQueue()
            return true
        }
        guard let task = activeTasks[key] else {
            return false
        }
        task.cancel()
        return true
    }

    public func cancelAllDownloads() {
        isShuttingDown = true
        shutdownReminderKeys.formUnion(currentReminders.keys)
        persistReminderState()
        activePreflightTask?.cancel()
        for task in activeTasks.values {
            task.cancel()
        }
    }

    public func downloadStatuses() -> [ModelDownloadStatus] {
        sortedStatuses()
    }

    public func downloadQueueSnapshot() -> ModelDownloadQueueSnapshot {
        ModelDownloadQueueSnapshot(
            maxConcurrent: maxConcurrentDownloads,
            activeCount: activeTasks.count,
            queuedCount: pendingDownloads.count,
            tasks: sortedStatuses(),
            recoveryReminders: recoveryReminders.values.sorted { $0.queueOrder < $1.queueOrder }
        )
    }

    @discardableResult
    public func clearFinishedDownloads() -> Int {
        let finishedKeys = statuses.compactMap { key, status in
            Self.isActiveStatus(status.status) ? nil : key
        }
        for key in finishedKeys {
            statuses[key] = nil
            completionResults[key] = nil
        }
        return finishedKeys.count
    }

    public func dismissRecoveryReminders(provider: ModelRepositoryProvider?, repoID: String?) {
        if let provider, let repoID {
            recoveryReminders[Self.taskKey(provider: provider, repoID: repoID)] = nil
        } else if provider == nil, repoID == nil {
            recoveryReminders.removeAll()
        } else {
            return
        }
        persistReminderState()
    }

    private func downloadAndWait(
        provider: ModelRepositoryProvider,
        repoID rawRepoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion {
        let response = enqueueDownload(
            provider: provider,
            repoID: rawRepoID,
            token: token,
            progress: progress
        )
        guard response.success else {
            return failure(
                message: response.error ?? "Download could not be queued.",
                code: response.code,
                repoID: response.repoID
            )
        }
        let key = Self.taskKey(provider: provider, repoID: response.repoID)
        if let result = completionResults[key] {
            return result
        }
        return await withCheckedContinuation { continuation in
            completionWaiters[key, default: []].append(continuation)
        }
    }

    private func enqueueDownload(
        provider: ModelRepositoryProvider,
        repoID rawRepoID: String,
        token: String?,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) -> ModelDownloadStartResponse {
        let repoID = rawRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isShuttingDown else {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "The application is shutting down.",
                code: "app_shutting_down"
            )
        }
        guard isValidRepoID(repoID) else {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "repo_id must be organization/model",
                code: "invalid_repo_id"
            )
        }
        let key = Self.taskKey(provider: provider, repoID: repoID)
        if activeTasks[key] != nil || preparationStates[key] != nil {
            return ModelDownloadStartResponse(
                success: true,
                status: statuses[key]?.status ?? ModelDownloadPhase.queued.rawValue,
                repoID: repoID,
                error: nil,
                code: "download_already_queued",
                queuePosition: statuses[key]?.queuePosition
            )
        }

        completionResults[key] = nil
        recoveryReminders[key] = nil
        let credential = token?.trimmingCharacters(in: .whitespacesAndNewlines)
        let request = QueuedModelDownload(
            provider: provider,
            repoID: repoID,
            token: credential?.isEmpty == false ? credential : nil,
            progress: progress,
            order: nextQueueOrder,
            enqueuedAt: Date()
        )
        nextQueueOrder += 1
        pendingDownloads.append(request)
        preparationStates[key] = .awaiting
        setStatus(
            key: key,
            provider: provider,
            repoID: repoID,
            phase: .queued,
            enqueuedAt: request.enqueuedAt
        )
        currentReminders[key] = ModelDownloadRecoveryReminder(
            provider: provider,
            repoID: repoID,
            queueOrder: request.order,
            previousStatus: ModelDownloadPhase.queued.rawValue,
            usedCredential: request.token != nil,
            enqueuedAt: request.enqueuedAt
        )
        refreshQueuePositions()
        persistReminderState()
        pumpPreflightQueue()
        pumpDownloadQueue()
        return ModelDownloadStartResponse(
            success: true,
            status: statuses[key]?.status ?? ModelDownloadPhase.queued.rawValue,
            repoID: repoID,
            error: nil,
            code: nil,
            queuePosition: statuses[key]?.queuePosition
        )
    }

    private func pumpPreflightQueue() {
        guard !isShuttingDown,
              activePreflightTask == nil,
              let request = pendingDownloads.first(where: {
                  preparationStates[Self.taskKey(provider: $0.provider, repoID: $0.repoID)] == .awaiting
              })
        else {
            return
        }
        let key = Self.taskKey(provider: request.provider, repoID: request.repoID)
        preparationStates[key] = .preflighting
        activePreflightKey = key
        setStatus(
            key: key,
            provider: request.provider,
            repoID: request.repoID,
            phase: .preflighting,
            enqueuedAt: request.enqueuedAt
        )
        updateCurrentReminderStatus(key: key, status: ModelDownloadPhase.preflighting.rawValue)
        refreshQueuePositions()
        let task = Task {
            do {
                let estimate = try await self.preflightQueuedDownload(request)
                self.finishQueuePreflight(key: key, estimate: estimate, error: nil)
            } catch {
                self.finishQueuePreflight(key: key, estimate: nil, error: error)
            }
        }
        activePreflightTask = task
    }

    private func preflightQueuedDownload(
        _ request: QueuedModelDownload
    ) async throws -> DownloadQueuePreflightEstimate {
        try Task.checkCancellation()
        let repository = try await resolver.resolve(
            provider: request.provider,
            repoID: request.repoID,
            token: request.token
        )
        let staging = try store.stagingSnapshotURL(
            provider: request.provider,
            repoID: request.repoID,
            commitSHA: repository.commitSHA
        )
        var totalBytes: Int64 = 0
        var remainingBytes: Int64 = 0
        var observedBytesByPath: [String: Int64] = [:]
        let downloadableFiles = repository.files.filter { file in
            file.isWeight || Self.isRuntimeMetadata(file)
        }
        for file in downloadableFiles {
            try Task.checkCancellation()
            guard file.size >= 0 else {
                throw DownloadFailure(
                    repoID: request.repoID,
                    code: "repo_size_invalid",
                    message: "Repository \(request.repoID) reports an invalid file size."
                )
            }
            let totalResult = totalBytes.addingReportingOverflow(file.size)
            guard !totalResult.overflow else {
                throw DownloadFailure(
                    repoID: request.repoID,
                    code: "repo_size_overflow",
                    message: "Repository \(request.repoID) reports an invalid total file size."
                )
            }
            totalBytes = totalResult.partialValue

            var existing: Int64 = 0
            if let sha256 = file.sha256,
               sha256.count == 64,
               sha256.allSatisfy(\.isHexDigit)
            {
                let destination = try ModelSnapshotVerifier.safeFileURL(path: file.path, beneath: staging)
                existing = recoverableBytes(
                    destination: destination,
                    identity: ModelPartialIdentity(
                        provider: request.provider,
                        repoID: request.repoID,
                        commitSHA: repository.commitSHA,
                        path: file.path,
                        expectedSize: file.size,
                        expectedSHA256: sha256,
                        etag: file.etag
                    )
                )
            }
            observedBytesByPath[file.path] = existing
            let remainingResult = remainingBytes.addingReportingOverflow(max(0, file.size - existing))
            guard !remainingResult.overflow else {
                throw DownloadFailure(
                    repoID: request.repoID,
                    code: "repo_size_overflow",
                    message: "Repository \(request.repoID) reports an invalid total file size."
                )
            }
            remainingBytes = remainingResult.partialValue
        }
        return DownloadQueuePreflightEstimate(
            commitSHA: repository.commitSHA,
            totalBytes: totalBytes,
            remainingBytes: remainingBytes,
            observedBytesByPath: observedBytesByPath
        )
    }

    private func finishQueuePreflight(
        key: String,
        estimate: DownloadQueuePreflightEstimate?,
        error: (any Error)?
    ) {
        guard activePreflightKey == key else {
            return
        }
        activePreflightKey = nil
        activePreflightTask = nil
        guard let index = pendingDownloads.firstIndex(where: {
            Self.taskKey(provider: $0.provider, repoID: $0.repoID) == key
        }) else {
            pumpPreflightQueue()
            return
        }
        let request = pendingDownloads[index]

        do {
            if let error {
                throw error
            }
            guard let estimate else {
                throw CancellationError()
            }
            try reserveDisk(
                key: key,
                remainingBytes: estimate.remainingBytes,
                observedBytesByPath: estimate.observedBytesByPath
            )
            preparationStates[key] = .ready
            setStatus(
                key: key,
                provider: request.provider,
                repoID: request.repoID,
                phase: .queued,
                commitSHA: estimate.commitSHA,
                totalBytes: estimate.totalBytes,
                remainingBytes: estimate.remainingBytes,
                enqueuedAt: request.enqueuedAt
            )
            updateCurrentReminderStatus(key: key, status: ModelDownloadPhase.queued.rawValue)
        } catch {
            pendingDownloads.remove(at: index)
            preparationStates[key] = nil
            releaseDiskReservation(key: key)
            let failure = queuePreflightFailure(error: error, repoID: request.repoID)
            setFailureStatus(
                key: key,
                provider: request.provider,
                repoID: request.repoID,
                journal: nil,
                phase: error is CancellationError ? .cancelled : .rejected,
                code: failure.code,
                message: failure.error ?? "Download preflight failed."
            )
            finishQueuedTask(key: key, result: failure)
        }
        refreshQueuePositions()
        persistReminderState()
        pumpDownloadQueue()
        pumpPreflightQueue()
    }

    private func pumpDownloadQueue() {
        guard !isShuttingDown else {
            return
        }
        while activeTasks.count < maxConcurrentDownloads,
              let request = pendingDownloads.first
        {
            let key = Self.taskKey(provider: request.provider, repoID: request.repoID)
            guard preparationStates[key] == .ready else {
                return
            }
            do {
                try validateDiskReservations()
            } catch {
                pendingDownloads.removeFirst()
                preparationStates[key] = nil
                releaseDiskReservation(key: key)
                let result = queuePreflightFailure(error: error, repoID: request.repoID)
                setFailureStatus(
                    key: key,
                    provider: request.provider,
                    repoID: request.repoID,
                    journal: nil,
                    phase: .rejected,
                    code: result.code,
                    message: result.error ?? "Insufficient disk space."
                )
                finishQueuedTask(key: key, result: result)
                refreshQueuePositions()
                continue
            }

            pendingDownloads.removeFirst()
            preparationStates[key] = nil
            refreshQueuePositions()
            updateCurrentReminderStatus(key: key, status: ModelDownloadPhase.resolving.rawValue)
            persistReminderState()
            let task = Task {
                await self.executeDownload(
                    provider: request.provider,
                    repoID: request.repoID,
                    token: request.token,
                    progress: request.progress
                )
            }
            activeTasks[key] = task
            Task {
                let result = await task.value
                self.finishActiveDownload(key: key, result: result)
            }
        }
    }

    private func finishActiveDownload(key: String, result: ModelDownloadCompletion) {
        activeTasks[key] = nil
        releaseDiskReservation(key: key)
        finishQueuedTask(key: key, result: result)
        pumpDownloadQueue()
        pumpPreflightQueue()
    }

    private func finishQueuedTask(key: String, result: ModelDownloadCompletion) {
        completionResults[key] = result
        if !shutdownReminderKeys.contains(key) {
            currentReminders[key] = nil
        }
        persistReminderState()
        let waiters = completionWaiters.removeValue(forKey: key) ?? []
        for waiter in waiters {
            waiter.resume(returning: result)
        }
    }

    private func queuePreflightFailure(error: any Error, repoID: String) -> ModelDownloadCompletion {
        if error is CancellationError {
            return failure(message: "Download cancelled.", code: "cancelled", repoID: repoID)
        }
        if let failure = error as? DownloadFailure {
            return self.failure(message: failure.message, code: failure.code, repoID: repoID)
        }
        if let resourceError = error as? ModelResourcePreflightError {
            let code: String
            switch resourceError {
            case .insufficientDisk:
                code = "insufficient_disk"
            case .insufficientMemory:
                code = "insufficient_memory"
            }
            return failure(message: resourceError.localizedDescription, code: code, repoID: repoID)
        }
        return failure(message: error.localizedDescription, code: "download_failed", repoID: repoID)
    }

    private func reserveDisk(
        key: String,
        remainingBytes: Int64,
        observedBytesByPath: [String: Int64]
    ) throws {
        let previous = diskReservations.removeValue(forKey: key) ?? 0
        defer {
            if diskReservations[key] == nil, previous > 0 {
                diskReservations[key] = previous
            }
        }
        let reserved = diskReservations.values.reduce(Int64(0)) { total, value in
            let result = total.addingReportingOverflow(value)
            return result.overflow ? Int64.max : result.partialValue
        }
        let candidateResult = reserved.addingReportingOverflow(max(0, remainingBytes))
        let withSafety = candidateResult.partialValue.addingReportingOverflow(ModelResourcePreflight.diskSafetyBytes)
        let required = candidateResult.overflow || withSafety.overflow ? Int64.max : withSafety.partialValue
        if let available = availableCapacityProvider(rootURL), required > available {
            throw ModelResourcePreflightError.insufficientDisk(required: required, available: available)
        }
        diskReservations[key] = max(0, remainingBytes)
        reservationObservedBytes[key] = observedBytesByPath
    }

    private func validateDiskReservations() throws {
        let reserved = diskReservations.values.reduce(Int64(0)) { total, value in
            let result = total.addingReportingOverflow(value)
            return result.overflow ? Int64.max : result.partialValue
        }
        let withSafety = reserved.addingReportingOverflow(ModelResourcePreflight.diskSafetyBytes)
        let required = withSafety.overflow ? Int64.max : withSafety.partialValue
        if let available = availableCapacityProvider(rootURL), required > available {
            throw ModelResourcePreflightError.insufficientDisk(required: required, available: available)
        }
    }

    private func adjustDiskReservation(
        key: String,
        remainingBytes: Int64,
        observedBytesByPath: [String: Int64]
    ) throws {
        try reserveDisk(
            key: key,
            remainingBytes: remainingBytes,
            observedBytesByPath: observedBytesByPath
        )
    }

    private func consumeDiskReservation(key: String, path: String, availableBytes: Int64) {
        guard var reservation = diskReservations[key] else {
            return
        }
        var observed = reservationObservedBytes[key] ?? [:]
        let previous = observed[path] ?? 0
        let current = max(previous, availableBytes)
        reservation = max(0, reservation - max(0, current - previous))
        observed[path] = current
        diskReservations[key] = reservation
        reservationObservedBytes[key] = observed
        statuses[key]?.remainingBytes = reservation
    }

    private func releaseDiskReservation(key: String) {
        diskReservations[key] = nil
        reservationObservedBytes[key] = nil
    }

    private func refreshQueuePositions() {
        for (index, request) in pendingDownloads.enumerated() {
            let key = Self.taskKey(provider: request.provider, repoID: request.repoID)
            statuses[key]?.queuePosition = index + 1
        }
        for key in activeTasks.keys {
            statuses[key]?.queuePosition = nil
        }
    }

    private func sortedStatuses() -> [ModelDownloadStatus] {
        statuses.values.sorted {
            let lhsActive = Self.isActiveStatus($0.status)
            let rhsActive = Self.isActiveStatus($1.status)
            if lhsActive != rhsActive {
                return lhsActive
            }
            if let lhsPosition = $0.queuePosition, let rhsPosition = $1.queuePosition,
               lhsPosition != rhsPosition
            {
                return lhsPosition < rhsPosition
            }
            return ($0.enqueuedAt ?? .distantPast) > ($1.enqueuedAt ?? .distantPast)
        }
    }

    private static func isActiveStatus(_ status: String) -> Bool {
        guard let phase = ModelDownloadPhase(rawValue: status) else {
            return false
        }
        return phase.isActive
    }

    private func updateCurrentReminderStatus(key: String, status: String) {
        guard var reminder = currentReminders[key] else {
            return
        }
        reminder.previousStatus = status
        currentReminders[key] = reminder
        persistReminderState()
    }

    private func persistReminderState() {
        let reminders = Array(recoveryReminders.values) + Array(currentReminders.values)
        do {
            try reminderStore.save(reminders.sorted { $0.queueOrder < $1.queueOrder })
        } catch {
            telemetryLogger("model_download_reminder_persist_failed error=\(error.localizedDescription)")
        }
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
                let verifier = ModelSnapshotVerifier()
                if let manifest = try? verifier.verifyStructure(
                    snapshot: finalSnapshot,
                    expectedProvider: provider,
                    expectedRepoID: repoID
                ),
                   let record = try? verifier.loadIntegrityRecord(at: finalSnapshot),
                   record.provider == provider,
                   record.repoID == repoID,
                   record.commitSHA == repository.commitSHA,
                   record.state == .verified
                {
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
            try adjustDiskReservation(
                key: key,
                remainingBytes: remainingBytes,
                observedBytesByPath: resumePlan.resumedBytesByPath
            )
            let resources = ModelResourcePreflight(
                weightBytes: weightBytes,
                remainingDownloadBytes: remainingBytes,
                availableDiskBytes: nil,
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
            setStatus(
                key: key,
                provider: provider,
                repoID: repoID,
                phase: .downloading,
                commitSHA: repository.commitSHA,
                totalBytes: totalBytes,
                remainingBytes: remainingBytes
            )
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
                        await self.consumeDiskReservation(
                            key: key,
                            path: weight.path,
                            availableBytes: fileBytes
                        )
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

    private nonisolated static func systemAvailableCapacity(at url: URL) -> Int64? {
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
        commitSHA: String? = nil,
        queuePosition: Int? = nil,
        totalBytes: Int64? = nil,
        remainingBytes: Int64? = nil,
        enqueuedAt: Date? = nil
    ) {
        let existing = statuses[key]
        statuses[key] = ModelDownloadStatus(
            repoID: repoID,
            provider: provider.rawValue,
            status: phase.rawValue,
            progressPct: progressPct,
            currentFile: currentFile,
            commitSHA: commitSHA,
            error: nil,
            errorCode: nil,
            queuePosition: queuePosition ?? existing?.queuePosition,
            totalBytes: totalBytes ?? existing?.totalBytes,
            remainingBytes: remainingBytes ?? existing?.remainingBytes,
            enqueuedAt: enqueuedAt ?? existing?.enqueuedAt
        )
        if currentReminders[key] != nil {
            updateCurrentReminderStatus(key: key, status: phase.rawValue)
        }
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
        let existing = statuses[key]
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
            errorCode: code,
            queuePosition: nil,
            totalBytes: journal?.totalBytes ?? existing?.totalBytes,
            remainingBytes: existing?.remainingBytes,
            enqueuedAt: existing?.enqueuedAt
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

    private nonisolated static func applyHuggingFaceToken(
        _ token: String?,
        to request: inout URLRequest
    ) {
        guard let token = token?.trimmingCharacters(in: .whitespacesAndNewlines),
              !token.isEmpty
        else {
            return
        }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
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
            errorCode: journal.errorCode,
            queuePosition: nil,
            totalBytes: journal.totalBytes,
            remainingBytes: max(0, journal.totalBytes - journal.progressBytes),
            enqueuedAt: journal.updatedAt
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
