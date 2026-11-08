import Foundation
import Testing

@testable import IronMLXAppCore

/// Ordered record of every side effect the coordinator causes.
private actor CallLog {
    private(set) var entries: [String] = []
    func append(_ entry: String) { entries.append(entry) }
}

private struct ScriptedAttempt: Sendable {
    var code: String?
    var bytesBeforeResult: Int64 = 0

    static let success = ScriptedAttempt(code: nil, bytesBeforeResult: 100)
    static func failure(_ code: String, afterBytes bytes: Int64 = 0) -> Self {
        ScriptedAttempt(code: code, bytesBeforeResult: bytes)
    }
}

private actor FakeStarterDownloader: StarterModelDownloading {
    private var attempts: [ScriptedAttempt]
    private let log: CallLog
    private(set) var mirrorFlags: [Bool] = []
    private(set) var requests: [(repoID: String, revision: String, pins: [PinnedModelFile])] = []
    private var cancelled = false
    private var inFlight = false
    private(set) var cancelCalls = 0

    init(_ attempts: [ScriptedAttempt], log: CallLog) {
        self.attempts = attempts
        self.log = log
    }

    func downloadStarterModel(
        repoID: String,
        revision: String,
        pinnedFiles: [PinnedModelFile],
        useHfMirror: Bool,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion {
        mirrorFlags.append(useHfMirror)
        requests.append((repoID, revision, pinnedFiles))
        await log.append("download:\(useHfMirror ? "mirror" : "official")")
        let attempt = attempts.isEmpty ? .success : attempts.removeFirst()
        inFlight = true
        defer { inFlight = false; cancelled = false }
        if attempt.code == "finish_after_cancel" {
            // Models a cancel that never reached the service: it completes anyway.
            await progress(ModelDownloadProgress(percent: 1, filename: "model.safetensors", bytes: 1, totalBytes: 200))
            while cancelCalls == 0 { try? await Task.sleep(for: .milliseconds(5)) }
            return ModelDownloadCompletion(success: true, message: nil, error: nil, code: nil, repoID: repoID)
        }
        if attempt.code == "block" {
            await progress(ModelDownloadProgress(percent: 1, filename: "model.safetensors", bytes: 1, totalBytes: 200))
            while !cancelled { try? await Task.sleep(for: .milliseconds(5)) }
            return ModelDownloadCompletion(success: false, message: nil, error: "cancelled", code: "cancelled", repoID: repoID)
        }
        if attempt.bytesBeforeResult > 0 {
            await progress(ModelDownloadProgress(
                percent: 50, filename: "model.safetensors", bytes: attempt.bytesBeforeResult, totalBytes: 200
            ))
        }
        return ModelDownloadCompletion(
            success: attempt.code == nil, message: nil, error: attempt.code, code: attempt.code, repoID: repoID
        )
    }

    /// Like the service, cancelling only affects a download that is running.
    func cancelStarterDownload(repoID: String) async {
        cancelCalls += 1
        if inFlight { cancelled = true }
        await log.append("cancel:\(repoID)")
    }

    func installedModelPath(repoID: String) async throws -> String {
        "/models/\(repoID)"
    }
}

private struct FakeProbe: ModelSourceProbing {
    var reachable: Set<DownloadSource>
    func isReachable(_ source: DownloadSource, repoID _: String, revision _: String) async -> Bool {
        reachable.contains(source)
    }
}

private struct ManagerError: Error {}

private actor FakeModelManager: BackendModelManaging {
    private let log: CallLog
    init(log: CallLog) { self.log = log }

    private func response(_ model: String) -> BackendModelAdminResponse {
        BackendModelAdminResponse(success: true, status: "ok", code: nil, model: model, loadedModels: [],
                                  warningCode: nil, warning: nil, error: nil)
    }

    func fetchHealthz() async throws -> HealthzSnapshot { throw ManagerError() }
    func loadModel(
        model: String, modelDir: String, setDefault: Bool, maxCacheCap _: Int?, pinned _: Bool,
        promptLookup _: BackendPromptLookupConfig?, audio _: BackendAudioResources?, decision _: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        await log.append("load:\(model):\(modelDir):\(setDefault)")
        return response(model)
    }
    func unloadModel(model: String, modelDir _: String?) async throws -> BackendModelAdminResponse {
        await log.append("unload:\(model)")
        return response(model)
    }
    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse {
        await log.append("setDefault:\(model)")
        return response(model)
    }
    func pinModel(model: String) async throws -> BackendModelAdminResponse { response(model) }
    func unpinModel(model: String) async throws -> BackendModelAdminResponse { response(model) }
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] { [] }
    func clearSharedPromptLookup(model _: String?) async throws -> BackendPromptLookupClearResponse {
        throw ManagerError()
    }
}

private struct Harness {
    let log = CallLog()
    let downloader: FakeStarterDownloader
    let coordinator: StarterDownloadCoordinator

    init(attempts: [ScriptedAttempt], reachable: Set<DownloadSource>) {
        downloader = FakeStarterDownloader(attempts, log: log)
        let manager = FakeModelManager(log: log)
        let log = log
        coordinator = StarterDownloadCoordinator(
            downloader: downloader,
            probe: FakeProbe(reachable: reachable),
            modelManager: { manager },
            onReady: { await log.append("ready:\($0)") }
        )
    }

    func finalState() async throws -> StarterDownloadState {
        await coordinator.waitUntilIdle()
        return await coordinator.state
    }
}

@Suite("Starter download coordinator")
struct StarterDownloadCoordinatorTests {
    private func recommendation(isUpgrade: Bool = false) throws -> StarterRecommendation {
        let catalog = try SupportedModelCatalog.bundled()
        let selected = isUpgrade
            ? StarterModelSelector.upgrade(catalog: catalog, physicalMemoryBytes: 64 << 30, currentModelID: "qwen35-4b-affine4")
            : StarterModelSelector.initial(catalog: catalog, physicalMemoryBytes: 16 << 30)
        return try #require(selected)
    }

    @Test func usesFirstReachableSourceExplicitly() async throws {
        let harness = Harness(attempts: [.success], reachable: [.huggingFace, .hfMirror])
        let starter = try recommendation()
        #expect(await harness.coordinator.start(starter) == .started(source: .huggingFace))
        _ = try await harness.finalState()
        #expect(await harness.downloader.mirrorFlags == [false])
        let request = try #require(await harness.downloader.requests.first)
        #expect(request.repoID == starter.entry.hfRepo)
        #expect(request.revision == starter.entry.metadataRevision)
        #expect(request.pins == starter.pinnedFiles)
    }

    @Test func mirrorUsedWhenOfficialUnreachable() async throws {
        let harness = Harness(attempts: [.success], reachable: [.hfMirror])
        #expect(await harness.coordinator.start(try recommendation()) == .started(source: .hfMirror))
        _ = try await harness.finalState()
        #expect(await harness.downloader.mirrorFlags == [true])
    }

    @Test func fallsBackOnceBeforeFirstByte() async throws {
        let recovered = Harness(attempts: [.failure("download_network_failed"), .success],
                                reachable: [.huggingFace, .hfMirror])
        _ = await recovered.coordinator.start(try recommendation())
        #expect(try await recovered.finalState() == .ready(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit"))
        #expect(await recovered.downloader.mirrorFlags == [false, true])

        let failing = Harness(attempts: [.failure("download_network_failed"), .failure("download_network_timeout")],
                              reachable: [.huggingFace, .hfMirror])
        _ = await failing.coordinator.start(try recommendation())
        #expect(try await failing.finalState() == .failed(code: "download_network_timeout"))
        #expect(await failing.downloader.mirrorFlags == [false, true])
    }

    @Test func doesNotSwitchAfterBytesReceived() async throws {
        let harness = Harness(attempts: [.failure("download_interrupted", afterBytes: 10)],
                              reachable: [.huggingFace, .hfMirror])
        _ = await harness.coordinator.start(try recommendation())
        #expect(try await harness.finalState() == .failed(code: "download_interrupted"))
        #expect(await harness.downloader.mirrorFlags == [false])
    }

    @Test func noSourceReachableFails() async throws {
        let harness = Harness(attempts: [], reachable: [])
        #expect(await harness.coordinator.start(try recommendation()) == .failed(code: "network_unreachable"))
        #expect(try await harness.finalState() == .failed(code: "network_unreachable"))
        #expect(await harness.downloader.mirrorFlags.isEmpty)
    }

    @Test func pausedDownloadResumesOnItsOriginalSource() async throws {
        let harness = Harness(attempts: [.failure("paused", afterBytes: 10), .success],
                              reachable: [.hfMirror])
        let starter = try recommendation()
        _ = await harness.coordinator.start(starter)
        // Paused keeps the progress it reached, so the bar does not empty.
        #expect(try await harness.finalState() == .paused(source: .hfMirror, bytes: 10, total: 200))
        #expect(await harness.coordinator.start(starter) == .started(source: .hfMirror))
        #expect(try await harness.finalState() == .ready(modelID: starter.entry.hfRepo))
        #expect(await harness.downloader.mirrorFlags == [true, true])
    }

    @Test func readyCallsOnReadyAfterSetDefaultAndLoad() async throws {
        let harness = Harness(attempts: [.success], reachable: [.huggingFace])
        let starter = try recommendation()
        let model = starter.entry.hfRepo
        _ = await harness.coordinator.start(starter)
        #expect(try await harness.finalState() == .ready(modelID: model))
        #expect(await harness.log.entries == [
            "download:official",
            "load:\(model):/models/\(model):true",
            "setDefault:\(model)",
            "ready:\(model)",
        ])
    }

    @Test func upgradeLeavesLoadingToChat() async throws {
        let harness = Harness(attempts: [.success], reachable: [.huggingFace])
        let upgrade = try recommendation(isUpgrade: true)
        let model = upgrade.entry.hfRepo
        _ = await harness.coordinator.start(upgrade)
        #expect(try await harness.finalState() == .ready(modelID: model))
        #expect(await harness.log.entries == ["download:official", "ready:\(model)"])
    }

    @Test func cancelStopsTheDownloadAndReturnsToIdle() async throws {
        let harness = Harness(attempts: [ScriptedAttempt(code: "block")], reachable: [.huggingFace])
        let starter = try recommendation()
        _ = await harness.coordinator.start(starter)
        #expect(await harness.coordinator.start(starter) == .alreadyInProgress)
        // Cancel once the first progress is recorded: an earlier cancel stops the
        // run before the service, and progress after it triggers a repeat cancel.
        while true {
            if case let .downloading(_, _, bytes, _, _) = await harness.coordinator.state, bytes > 0 { break }
            try await Task.sleep(for: .milliseconds(2))
        }
        await harness.coordinator.cancel()
        #expect(try await harness.finalState() == .idle)
        #expect(await harness.log.entries == ["download:official", "cancel:\(starter.entry.hfRepo)"])
    }

    @Test func cancelWhilePausedReturnsToIdle() async throws {
        let harness = Harness(attempts: [.failure("paused", afterBytes: 10)], reachable: [.huggingFace])
        let starter = try recommendation()
        _ = await harness.coordinator.start(starter)
        #expect(try await harness.finalState() == .paused(source: .huggingFace, bytes: 10, total: 200))
        await harness.coordinator.cancel()
        #expect(await harness.coordinator.state == .idle)
    }

    @Test func restoredPausedDownloadResumesOnItsOriginalSource() async throws {
        // Official is reachable now, but the task left by the last session used the mirror.
        let harness = Harness(attempts: [.success], reachable: [.huggingFace, .hfMirror])
        let starter = try recommendation()
        await harness.coordinator.restorePaused(starter, source: .hfMirror)
        #expect(await harness.coordinator.state == .paused(source: .hfMirror, total: starter.entry.downloadBytes))
        #expect(await harness.coordinator.start(starter) == .started(source: .hfMirror))
        #expect(try await harness.finalState() == .ready(modelID: starter.entry.hfRepo))
        #expect(await harness.downloader.mirrorFlags == [true])
    }

    @Test func restorePausedNeverReplacesARunningDownload() async throws {
        let harness = Harness(attempts: [ScriptedAttempt(code: "block")], reachable: [.huggingFace])
        let starter = try recommendation()
        _ = await harness.coordinator.start(starter)
        await harness.coordinator.restorePaused(starter, source: .hfMirror)
        #expect(await harness.coordinator.state != .paused(source: .hfMirror))
        await harness.coordinator.cancel()
        _ = try await harness.finalState()
    }

    @Test func cancelBeforeTheDownloaderIsCalledStopsTheRun() async throws {
        let harness = Harness(attempts: [.success], reachable: [.huggingFace])
        let coordinator = harness.coordinator
        let trigger = CancelTrigger(onDownloadingState: 1)
        await coordinator.setStateObserver { state in
            if await trigger.shouldCancel(state) { await coordinator.cancel() }
        }
        _ = await coordinator.start(try recommendation())
        #expect(try await harness.finalState() == .idle)
        #expect(await harness.downloader.requests.isEmpty)
        #expect(await harness.log.entries.allSatisfy { !$0.hasPrefix("load") && !$0.hasPrefix("ready") })
    }

    @Test func cancelBetweenFallbackAttemptsStopsTheRun() async throws {
        let harness = Harness(attempts: [.failure("download_network_failed"), .success],
                              reachable: [.huggingFace, .hfMirror])
        let coordinator = harness.coordinator
        let trigger = CancelTrigger(onDownloadingState: 2)
        await coordinator.setStateObserver { state in
            if await trigger.shouldCancel(state) { await coordinator.cancel() }
        }
        _ = await coordinator.start(try recommendation())
        #expect(try await harness.finalState() == .idle)
        #expect(await harness.downloader.mirrorFlags == [false])
        #expect(await harness.log.entries.allSatisfy { !$0.hasPrefix("load") && !$0.hasPrefix("ready") })
    }

    @Test func downloadFinishingAfterCancelIsNotLoaded() async throws {
        let harness = Harness(attempts: [ScriptedAttempt(code: "finish_after_cancel")], reachable: [.huggingFace])
        _ = await harness.coordinator.start(try recommendation())
        while await harness.downloader.requests.isEmpty { try await Task.sleep(for: .milliseconds(5)) }
        await harness.coordinator.cancel()
        #expect(await harness.coordinator.state == .idle)
        #expect(try await harness.finalState() == .idle)
        #expect(await harness.log.entries.allSatisfy { !$0.hasPrefix("load") && !$0.hasPrefix("ready") })
    }

    @Test func busyServiceIsReportedWithoutSwitchingHosts() async throws {
        let harness = Harness(attempts: [.failure("starter_download_busy")], reachable: [.huggingFace, .hfMirror])
        _ = await harness.coordinator.start(try recommendation())
        #expect(try await harness.finalState() == .failed(code: "starter_download_busy"))
        #expect(await harness.downloader.mirrorFlags == [false])
    }

    @Test func downloadingStateReportsSourceAndBytes() async throws {
        let harness = Harness(attempts: [.failure("download_interrupted", afterBytes: 50)], reachable: [.hfMirror])
        let observed = StateRecorder()
        await harness.coordinator.setStateObserver { await observed.record($0) }
        _ = await harness.coordinator.start(try recommendation())
        _ = try await harness.finalState()
        let downloading = await observed.states.compactMap { state -> (DownloadSource, Double, Int64, Int64)? in
            if case let .downloading(source, fraction, bytes, total, _) = state { return (source, fraction, bytes, total) }
            return nil
        }
        let last = try #require(downloading.last)
        #expect(last.0 == .hfMirror)
        #expect(last.2 == 50 && last.3 == 200)
        #expect(last.1 == 0.25)
        #expect(await observed.states.first == .probing)
    }

    @Test func restoredPauseShowsTheBytesOnDiskAndResumesFromThem() async throws {
        let harness = Harness(attempts: [ScriptedAttempt(code: "block")], reachable: [.huggingFace])
        let starter = try recommendation()
        let total = starter.entry.downloadBytes
        let onDisk = total / 4
        let expectedFraction = Double(onDisk) / Double(total)
        await harness.coordinator.restorePaused(starter, source: .huggingFace, bytes: onDisk)
        #expect(await harness.coordinator.state == .paused(source: .huggingFace, bytes: onDisk, total: total))
        #expect(await harness.coordinator.state.progress?.fraction == expectedFraction)

        let observed = StateRecorder()
        await harness.coordinator.setStateObserver { await observed.record($0) }
        _ = await harness.coordinator.start(starter)
        while await observed.states.isEmpty { try await Task.sleep(for: .milliseconds(2)) }
        // Resuming starts from the bytes on disk, never from an empty bar.
        guard case let .downloading(_, fraction, bytes, _, rate) = await observed.states[0] else {
            Issue.record("expected a downloading state")
            return
        }
        #expect(bytes == onDisk && fraction == expectedFraction && rate == nil)
        await harness.coordinator.cancel()
        _ = try await harness.finalState()
    }
}

@Suite("Starter transfer rate")
struct StarterTransferRateTests {
    private let start = ContinuousClock.now

    private func at(_ seconds: Double) -> ContinuousClock.Instant {
        start.advanced(by: .milliseconds(Int(seconds * 1000)))
    }

    @Test func noRateBeforeTwoSecondsOfTransfer() {
        var rate = StarterTransferRate()
        #expect(rate.record(bytes: 0, at: at(0)) == nil)
        #expect(rate.record(bytes: 8_000_000, at: at(0.5)) == nil)
        #expect(rate.record(bytes: 20_000_000, at: at(1.5)) == nil)
        #expect(rate.record(bytes: 26_000_000, at: at(2)) == 13_000_000)
    }

    @Test func idleTimeAfterAResumeIsNotPartOfTheRate() {
        var rate = StarterTransferRate()
        // Resume reports the bytes on disk, then verifies for a while.
        #expect(rate.record(bytes: 500_000_000, at: at(0)) == nil)
        #expect(rate.record(bytes: 500_000_000, at: at(4)) == nil)
        #expect(rate.record(bytes: 512_000_000, at: at(5)) == nil)
        #expect(rate.record(bytes: 524_000_000, at: at(6)) == 12_000_000)
    }

    @Test func aFewKilobytesNeverShowAsARate() {
        var rate = StarterTransferRate()
        #expect(rate.record(bytes: 1_000, at: at(0)) == nil)
        #expect(rate.record(bytes: 9_000, at: at(1)) == nil)
        #expect(rate.record(bytes: 30_000, at: at(3)) == nil)
        #expect(rate.record(bytes: 60_000, at: at(4)) == nil)
    }

    @Test func slowLinksStillShowARateAfterTwoSeconds() {
        var rate = StarterTransferRate()
        #expect(rate.record(bytes: 0, at: at(0)) == nil)
        #expect(rate.record(bytes: 40_000, at: at(1)) == nil)
        #expect(rate.record(bytes: 100_000, at: at(2.5)) == 40_000)
    }

    @Test func goingBackwardsRestartsTheSample() {
        var rate = StarterTransferRate()
        _ = rate.record(bytes: 0, at: at(0))
        _ = rate.record(bytes: 30_000_000, at: at(3))
        #expect(rate.record(bytes: 1_000_000, at: at(3.5)) == nil)
        #expect(rate.record(bytes: 3_000_000, at: at(4.5)) == nil)
        #expect(rate.record(bytes: 9_000_000, at: at(5.5)) == 4_000_000)
    }
}

private actor StateRecorder {
    private(set) var states: [StarterDownloadState] = []
    func record(_ state: StarterDownloadState) { states.append(state) }
}

/// Cancels once, on the n-th `.downloading` notification.
private actor CancelTrigger {
    private let target: Int
    private var seen = 0
    init(onDownloadingState target: Int) { self.target = target }
    func shouldCancel(_ state: StarterDownloadState) -> Bool {
        guard case .downloading = state else { return false }
        seen += 1
        return seen == target
    }
}
