import Foundation

/// The download operations the beginner flow needs, so tests can replace the service.
protocol StarterModelDownloading: Sendable {
    /// Downloads one pinned revision from the given host and waits for the result.
    func downloadStarterModel(
        repoID: String,
        revision: String,
        pinnedFiles: [PinnedModelFile],
        useHfMirror: Bool,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion
    func cancelStarterDownload(repoID: String) async
    func installedModelPath(repoID: String) async throws -> String
}

extension ModelDownloadService: StarterModelDownloading {
    func downloadStarterModel(
        repoID: String,
        revision: String,
        pinnedFiles: [PinnedModelFile],
        useHfMirror: Bool,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion {
        // Starter repositories are public, so no token is ever sent to either host.
        await downloadHuggingFace(
            repoID: repoID,
            token: nil,
            useHfMirror: useHfMirror,
            revision: revision,
            pinnedFiles: pinnedFiles,
            progress: progress
        )
    }

    func cancelStarterDownload(repoID: String) {
        if cancelDownload(provider: .huggingFace, repoID: repoID) { return }
        // Nothing is running: drop the paused task (or the task the last App
        // session left), so the next one-click start picks a host afresh.
        _ = deleteDownload(provider: .huggingFace, repoID: repoID)
        dismissRecoveryReminders(provider: .huggingFace, repoID: repoID)
    }

    nonisolated func installedModelPath(repoID: String) async throws -> String {
        try await LocalModelScanner(rootURL: rootURL).verifiedModelPathAsync(for: repoID)
    }
}

enum StarterDownloadState: Equatable, Sendable {
    case idle
    case probing
    case downloading(source: DownloadSource, fraction: Double, bytes: Int64, total: Int64, bytesPerSecond: Double?)
    /// Keeps the bytes reached, or found on disk after an App restart.
    case paused(source: DownloadSource, bytes: Int64 = 0, total: Int64 = 0)
    case loading
    case ready(modelID: String)
    case failed(code: String)
}

/// What the progress bar and byte line show.
struct StarterDownloadProgress: Equatable, Sendable {
    var fraction: Double
    var bytes: Int64
    var total: Int64
    /// Nil until a meaningful sample exists, and always nil while paused.
    var bytesPerSecond: Double?
}

extension StarterDownloadState {
    var progress: StarterDownloadProgress? {
        switch self {
        case let .downloading(_, fraction, bytes, total, rate):
            StarterDownloadProgress(fraction: fraction, bytes: bytes, total: total, bytesPerSecond: rate)
        case let .paused(_, bytes, total):
            StarterDownloadProgress(fraction: Self.fraction(bytes: bytes, total: total), bytes: bytes,
                                    total: total, bytesPerSecond: nil)
        case .idle, .probing, .loading, .ready, .failed:
            nil
        }
    }

    static func fraction(bytes: Int64, total: Int64) -> Double {
        total > 0 ? min(1, Double(max(0, bytes)) / Double(total)) : 0
    }
}

/// Download speed for the progress line. A rate exists only for a meaningful
/// sample: at least `minimumSpan` and `minimumBytes` of transfer inside the
/// recent window. Waiting before the first new byte (a resume verifies its
/// files and probes the host first) is not part of the sample, so a resume
/// never shows "0.0 MB/s · about 5480 min".
struct StarterTransferRate {
    static let window: Duration = .seconds(5)
    static let minimumSpan: Duration = .seconds(2)
    static let minimumBytes: Int64 = 64_000

    private var samples: [(time: ContinuousClock.Instant, bytes: Int64)] = []

    mutating func record(bytes: Int64, at now: ContinuousClock.Instant) -> Double? {
        if let last = samples.last, bytes < last.bytes {
            // A new attempt started below the old count: measure afresh.
            samples = [(now, bytes)]
            return nil
        }
        if samples.count == 1, samples[0].bytes == bytes {
            // Nothing has arrived yet; idle time is not transfer time.
            samples[0] = (now, bytes)
            return nil
        }
        samples.append((now, bytes))
        samples.removeAll { now - $0.time > Self.window }
        guard let oldest = samples.first else { return nil }
        let span = now - oldest.time
        let transferred = bytes - oldest.bytes
        guard span >= Self.minimumSpan, transferred >= Self.minimumBytes else { return nil }
        let seconds = Double(span.components.seconds) + Double(span.components.attoseconds) / 1e18
        return Double(transferred) / seconds
    }
}

enum StarterDownloadStart: Equatable, Sendable {
    case started(source: DownloadSource)
    case alreadyInProgress
    case failed(code: String)
}

/// Runs the one-click download: picks a host, downloads the pinned revision,
/// then (for the first model) loads it as the default model.
///
/// The host is chosen once before the download. A failure before any weight
/// byte arrives retries once on the other reachable host; after that the host
/// never changes, and a paused download resumes on its original host.
actor StarterDownloadCoordinator {
    typealias ModelManagerProvider = @Sendable () async throws -> any BackendModelManaging

    static let networkUnreachableCode = "network_unreachable"
    static let modelLoadFailedCode = "model_load_failed"
    /// Failures that another host cannot fix.
    private static let hostIndependentCodes: Set<String> = [
        "insufficient_disk", "insufficient_memory", "app_shutting_down",
        "invalid_revision", "invalid_repo_id", "cancelled", "paused",
        "starter_download_busy",
    ]

    private let downloader: any StarterModelDownloading
    private let probe: any ModelSourceProbing
    private let modelManager: ModelManagerProvider
    private var onReady: @Sendable (String) async -> Void
    private var stateObserver: (@Sendable (StarterDownloadState) async -> Void)?
    private(set) var state: StarterDownloadState = .idle
    private var recommendation: StarterRecommendation?
    private var runTask: Task<Void, Never>?
    /// Bumped by every start and every cancel. Work started under an older
    /// generation must not change state, load a model or call `onReady`.
    private var generation = 0
    private var attemptID = 0
    private var attemptReceivedBytes = false
    /// Attempts whose late progress already triggered a repeat cancel.
    private var recancelledAttempts: Set<Int> = []
    private var transferRate = StarterTransferRate()
    /// Bytes of the current run, kept when it pauses.
    private var lastBytes: Int64 = 0
    private var lastTotal: Int64 = 0
    private let clock = ContinuousClock()

    init(
        downloader: any StarterModelDownloading,
        probe: any ModelSourceProbing = DownloadSourceProbe(),
        modelManager: @escaping ModelManagerProvider,
        onReady: @escaping @Sendable (String) async -> Void = { _ in }
    ) {
        self.downloader = downloader
        self.probe = probe
        self.modelManager = modelManager
        self.onReady = onReady
    }

    func setOnReady(_ handler: @escaping @Sendable (String) async -> Void) {
        onReady = handler
    }

    /// Called after every state change, in order.
    func setStateObserver(_ observer: (@Sendable (StarterDownloadState) async -> Void)?) {
        stateObserver = observer
    }

    func start(_ recommendation: StarterRecommendation) async -> StarterDownloadStart {
        switch state {
        case .probing, .downloading, .loading:
            return .alreadyInProgress
        case let .paused(source, bytes, _) where self.recommendation?.entry.id == recommendation.entry.id:
            generation += 1
            launch(recommendation, sources: [source], generation: generation, startBytes: bytes)
            return .started(source: source)
        default:
            break
        }
        generation += 1
        let current = generation
        self.recommendation = recommendation
        await setState(.probing)
        let sources = await chooseDownloadSource(
            probe: probe, repoID: recommendation.entry.hfRepo, revision: recommendation.entry.metadataRevision
        )
        guard current == generation else {
            return .failed(code: "cancelled")
        }
        guard let first = sources.first else {
            await setState(.failed(code: Self.networkUnreachableCode))
            return .failed(code: Self.networkUnreachableCode)
        }
        launch(recommendation, sources: sources, generation: current)
        return .started(source: first)
    }

    /// Shows a starter download that an earlier App session left unfinished as
    /// paused, so `start` resumes it on the host it used. Does nothing while
    /// another run is active.
    /// `bytes` is what is already on disk, so the paused bar is not empty.
    func restorePaused(_ recommendation: StarterRecommendation, source: DownloadSource, bytes: Int64 = 0) async {
        switch state {
        case .idle, .failed:
            generation += 1
            self.recommendation = recommendation
            let total = recommendation.entry.downloadBytes
            await setState(.paused(source: source, bytes: min(max(0, bytes), total), total: total))
        case .probing, .downloading, .paused, .loading, .ready:
            return
        }
    }

    /// Stops probing or downloading and returns to `.idle` at once. Loading
    /// cannot be interrupted.
    func cancel() async {
        let stopsDownload: Bool
        switch state {
        case .loading:
            return
        case .downloading, .paused:
            stopsDownload = true
        case .probing, .idle, .ready, .failed:
            stopsDownload = false
        }
        // Invalidate the run first: if the service has not received the
        // download yet, the run stops before (or ignores) it.
        generation += 1
        await setState(.idle)
        if stopsDownload, let repoID = recommendation?.entry.hfRepo {
            await downloader.cancelStarterDownload(repoID: repoID)
        }
    }

    /// Waits for the background download (and load) started by `start` to end.
    func waitUntilIdle() async {
        while let task = runTask {
            await task.value
            if runTask == task { runTask = nil }
        }
    }

    private func launch(
        _ recommendation: StarterRecommendation, sources: [DownloadSource], generation current: Int, startBytes: Int64 = 0
    ) {
        self.recommendation = recommendation
        let total = recommendation.entry.downloadBytes
        state = .downloading(source: sources[0], fraction: StarterDownloadState.fraction(bytes: startBytes, total: total),
                             bytes: startBytes, total: total, bytesPerSecond: nil)
        runTask = Task { await self.run(recommendation, sources: sources, generation: current, startBytes: startBytes) }
    }

    private func run(
        _ recommendation: StarterRecommendation, sources: [DownloadSource], generation current: Int, startBytes: Int64
    ) async {
        let entry = recommendation.entry
        for (index, source) in sources.enumerated() {
            attemptID += 1
            let attempt = attemptID
            attemptReceivedBytes = false
            transferRate = StarterTransferRate()
            // A resume starts from the bytes already on disk.
            lastBytes = startBytes
            lastTotal = entry.downloadBytes
            await setState(.downloading(source: source,
                                        fraction: StarterDownloadState.fraction(bytes: lastBytes, total: lastTotal),
                                        bytes: lastBytes, total: lastTotal, bytesPerSecond: nil), generation: current)
            // A cancel may have landed while the observer ran.
            guard current == generation else { return }
            let result = await downloader.downloadStarterModel(
                repoID: entry.hfRepo,
                revision: entry.metadataRevision,
                pinnedFiles: recommendation.pinnedFiles,
                useHfMirror: source.usesHfMirror
            ) { progress in
                await self.record(progress, source: source, attempt: attempt, generation: current)
            }
            if attemptID == attempt { attemptID += 1 }
            // Cancelled meanwhile: even a completed download is not loaded.
            guard current == generation else { return }
            if result.success {
                await finish(recommendation, generation: current)
                return
            }
            let code = result.code ?? "download_failed"
            if code == "cancelled" {
                await setState(.idle, generation: current)
                return
            }
            if code == "paused" {
                await setState(.paused(source: source, bytes: lastBytes, total: lastTotal), generation: current)
                return
            }
            let canFallBack = !attemptReceivedBytes && index + 1 < sources.count
                && !Self.hostIndependentCodes.contains(code)
            if !canFallBack {
                await setState(.failed(code: code), generation: current)
                return
            }
            IronMLXAppLogger.info("Starter download from \(source.rawValue) failed before data (\(code)); trying \(sources[index + 1].rawValue)")
        }
    }

    private func record(
        _ progress: ModelDownloadProgress, source: DownloadSource, attempt: Int, generation current: Int
    ) async {
        guard current == generation else {
            // The download reached the service after the user cancelled, so
            // the earlier cancel missed it; cancel it now that it is running.
            if recancelledAttempts.insert(attempt).inserted, let repoID = recommendation?.entry.hfRepo {
                await downloader.cancelStarterDownload(repoID: repoID)
            }
            return
        }
        guard attempt == attemptID else { return }
        attemptReceivedBytes = true
        let total = progress.totalBytes ?? recommendation?.entry.downloadBytes ?? 0
        let bytes = progress.bytes ?? Int64(Double(total) * progress.percent / 100)
        let rate = transferRate.record(bytes: bytes, at: clock.now)
        lastBytes = bytes
        lastTotal = total
        let fraction = StarterDownloadState.fraction(bytes: bytes, total: total)
        await setState(.downloading(source: source, fraction: fraction, bytes: bytes, total: total, bytesPerSecond: rate),
                       generation: current)
    }

    private func finish(_ recommendation: StarterRecommendation, generation current: Int) async {
        let modelID = recommendation.entry.hfRepo
        // An upgrade replaces the chat's default model; chat does that switch.
        if !recommendation.isUpgrade {
            // `.loading` is set synchronously after the caller's generation
            // check, and `cancel()` leaves loading alone.
            await setState(.loading, generation: current)
            do {
                let modelDir = try await downloader.installedModelPath(repoID: modelID)
                let manager = try await modelManager()
                // Loading registers the new model with the backend; only a
                // registered model can be made the default.
                let loaded = try await manager.loadModel(model: modelID, modelDir: modelDir, setDefault: true)
                guard loaded.success else { throw StarterLoadError(code: loaded.code) }
                let defaulted = try await manager.setDefaultModel(modelID)
                guard defaulted.success else { throw StarterLoadError(code: defaulted.code) }
            } catch {
                IronMLXAppLogger.error("Starter model \(modelID) could not be loaded: \(error)")
                await setState(.failed(code: Self.modelLoadFailedCode), generation: current)
                return
            }
        }
        guard current == generation else { return }
        await setState(.ready(modelID: modelID))
        await onReady(modelID)
    }

    /// Changes state only for the current generation.
    private func setState(_ newState: StarterDownloadState, generation current: Int) async {
        guard current == generation else { return }
        await setState(newState)
    }

    private func setState(_ newState: StarterDownloadState) async {
        if case let .failed(code) = newState {
            // The window shows plain words; the code is kept here.
            IronMLXAppLogger.error("Starter download failed: \(code)")
        }
        state = newState
        await stateObserver?(newState)
    }
}

private struct StarterLoadError: Error {
    var code: String?
}
