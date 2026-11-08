import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

/// A backend whose start blocks until released, like a cold launch.
@MainActor
private final class StartingBackend: BackendRuntimeManaging {
    var isRunning = false
    var state: BackendProcessState = .stopped
    var currentLaunchID: UUID?
    var lastError: String?
    var lastEvent: BackendRuntimeEvent?
    var lastIncident: BackendIncidentRecord?
    private(set) var ensureRunningCalls = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func ensureRunning() async throws {
        ensureRunningCalls += 1
        state = .starting
        await withCheckedContinuation { waiters.append($0) }
        state = .running
        isRunning = true
    }

    func release() {
        waiters.forEach { $0.resume() }
        waiters = []
    }

    func restart(intent _: BackendStopIntent) async -> BackendRestartResult {
        BackendRestartResult(success: true, status: "restarted", port: 9068)
    }
    func stop(intent _: BackendStopIntent) async {}
    func stopForAppQuit() async {}
    func confirmLoadedModels(_: [BackendLoadedModelInfo], parameterConfirmedModelIDs _: Set<String>) {}
    func refreshConfirmedSnapshot() {}
}

/// Finishes every starter download at once.
private struct InstantDownloader: StarterModelDownloading {
    func downloadStarterModel(
        repoID: String, revision _: String, pinnedFiles _: [PinnedModelFile], useHfMirror _: Bool,
        progress: @escaping @Sendable (ModelDownloadProgress) async -> Void
    ) async -> ModelDownloadCompletion {
        await progress(ModelDownloadProgress(percent: 50, filename: "model.safetensors", bytes: 50, totalBytes: 100))
        return ModelDownloadCompletion(success: true, message: nil, error: nil, code: nil, repoID: repoID)
    }
    func cancelStarterDownload(repoID _: String) async {}
    func installedModelPath(repoID: String) async throws -> String { "/models/\(repoID)" }
}

private struct OfficialHostReachable: ModelSourceProbing {
    func isReachable(_ source: DownloadSource, repoID _: String, revision _: String) async -> Bool {
        source == .huggingFace
    }
}

/// Loads and sets the default successfully, like a running backend.
private actor AcceptingManager: BackendModelManaging {
    private func ok(_ model: String) -> BackendModelAdminResponse {
        BackendModelAdminResponse(success: true, status: "loaded", code: nil, model: model, loadedModels: [],
                                  warningCode: nil, warning: nil, error: nil)
    }

    func fetchHealthz() async throws -> HealthzSnapshot { throw CancellationError() }
    func loadModel(
        model: String, modelDir _: String, setDefault _: Bool, maxCacheCap _: Int?, pinned _: Bool,
        promptLookup _: BackendPromptLookupConfig?, audio _: BackendAudioResources?, decision _: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse { ok(model) }
    func unloadModel(model: String, modelDir _: String?) async throws -> BackendModelAdminResponse { ok(model) }
    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse { ok(model) }
    func pinModel(model: String) async throws -> BackendModelAdminResponse { ok(model) }
    func unpinModel(model: String) async throws -> BackendModelAdminResponse { ok(model) }
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] { [] }
    func clearSharedPromptLookup(model _: String?) async throws -> BackendPromptLookupClearResponse {
        throw CancellationError()
    }
}

private final class RecordingPresenter: ChatWindowPresenting {
    private(set) var shown: [(modelID: String, needsLoad: Bool)] = []
    func showChat(modelID: String, needsLoad: Bool) { shown.append((modelID, needsLoad)) }
}

@MainActor
@Suite("Chat entry controller", .serialized)
struct ChatEntryControllerTests {
    private static let starter = "mlx-community/Qwen3.5-4B-MLX-4bit"

    @MainActor
    private struct Fixture {
        let root: URL
        let backend: StartingBackend
        let presenter: RecordingPresenter
        let controller: ChatEntryController
        let routes: RouteLog
        var dashboardRoutes: [DashboardInitialRoute] { routes.values }
    }

    @MainActor
    final class RouteLog {
        var values: [DashboardInitialRoute] = []
    }

    private func fixture(
        localModels: [LocalModel],
        presenterAvailable: Bool = true,
        downloader: (any StarterModelDownloading)? = nil,
        loads: Bool = false
    ) throws -> Fixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let backend = StartingBackend()
        let presenter = RecordingPresenter()
        let routes = RouteLog()
        let service = ModelDownloadService(rootURL: root, telemetryLogger: { _ in })
        let controller = ChatEntryController(
            configStore: AppConfigStore(url: root.appendingPathComponent("config.json")),
            backend: backend,
            scanner: LocalModelScanner(rootURL: root),
            downloadService: service,
            starterDownloads: StarterDownloadCoordinator(
                downloader: downloader ?? service, probe: OfficialHostReachable(), modelManager: {
                    guard loads else { throw CancellationError() }
                    return AcceptingManager()
                }
            ),
            notifier: ModelReadyNotifier(center: nil),
            physicalMemoryBytes: 16 << 30,
            presenter: { presenterAvailable ? presenter : nil },
            showDashboard: { routes.values.append($0) },
            localModelsProvider: { _ in localModels },
            loadedModelsProvider: { [] }
        )
        return Fixture(root: root, backend: backend, presenter: presenter, controller: controller, routes: routes)
    }

    private func waitUntil(_ condition: @MainActor () -> Bool) async {
        for _ in 0 ..< 200 where !condition() {
            try? await Task.sleep(for: .milliseconds(5))
        }
    }

    @Test func coldStartWithoutLocalModelShowsNoModelWindowAtOnce() async throws {
        let f = try fixture(localModels: [])
        defer { try? FileManager.default.removeItem(at: f.root) }
        f.controller.openChat()
        await waitUntil { f.controller.isNoModelWindowVisible }
        #expect(f.controller.isNoModelWindowVisible)
        #expect(f.backend.state == .starting)
        #expect(f.backend.ensureRunningCalls == 1)
        #expect(f.presenter.shown.isEmpty)

        // A second click while the backend still starts brings the window back.
        f.controller.closeNoModelWindow()
        f.controller.openChat()
        await waitUntil { f.controller.isNoModelWindowVisible }
        #expect(f.controller.isNoModelWindowVisible)
        f.controller.closeNoModelWindow()
        f.backend.release()
    }

    @Test func coldStartWithLocalModelOpensChatAtOnceWithLoad() async throws {
        let model = LocalModel(id: Self.starter, repoID: Self.starter, source: "hf", sizeMB: 3_000)
        let f = try fixture(localModels: [model])
        defer { try? FileManager.default.removeItem(at: f.root) }
        f.controller.openChat()
        await waitUntil { !f.presenter.shown.isEmpty }
        #expect(f.presenter.shown.map(\.modelID) == [Self.starter])
        #expect(f.presenter.shown.map(\.needsLoad) == [true])
        #expect(f.backend.state == .starting)
        #expect(f.backend.ensureRunningCalls == 1)
        f.backend.release()
    }

    @Test func coldStartWithoutPresenterKeepsTheDashboardFallback() async throws {
        let model = LocalModel(id: Self.starter, repoID: Self.starter, source: "hf", sizeMB: 3_000)
        let f = try fixture(localModels: [model], presenterAvailable: false)
        defer { try? FileManager.default.removeItem(at: f.root) }
        f.controller.openChat()
        await waitUntil { !f.dashboardRoutes.isEmpty }
        #expect(f.dashboardRoutes == [.status])
        f.backend.release()
    }

    /// The upgrade's download reports `.ready` without loading the model, so
    /// it must go to chat (which switches to it), not be announced as a
    /// loaded model that opens chat.
    @Test func upgradeReadyGoesToChatWithoutAnnouncingIt() async throws {
        let f = try fixture(localModels: [], downloader: InstantDownloader())
        defer { try? FileManager.default.removeItem(at: f.root) }
        await f.controller.start()
        let upgrade = try #require(StarterModelSelector.upgrade(
            catalog: try SupportedModelCatalog.bundled(), physicalMemoryBytes: 64 << 30, currentModelID: Self.starter
        ))
        // Were it announced, a visible no-model window would open chat as loaded.
        f.controller.showNoModelWindow()
        var states: [StarterDownloadState] = []
        f.controller.startUpgradeDownload(upgrade) { states.append($0) }
        #expect(f.controller.starterModelName == upgrade.entry.name)
        await waitUntil { states.last == .ready(modelID: upgrade.entry.hfRepo) }
        #expect(states.last == .ready(modelID: upgrade.entry.hfRepo))
        #expect(states.contains { if case .downloading = $0 { true } else { false } })
        #expect(f.presenter.shown.isEmpty)
        #expect(f.controller.menuStarterState == .idle)
        #expect(f.controller.starterModelName != upgrade.entry.name)
        f.controller.closeNoModelWindow()
    }

    /// A starter that downloaded but failed to load says so in the menu, and
    /// the row goes away once chat opens on another model.
    @Test func starterLoadFailureRowClearsWhenChatOpens() async throws {
        let model = LocalModel(id: "mlx-community/Qwen3.5-2B-4bit", repoID: "mlx-community/Qwen3.5-2B-4bit", source: "hf", sizeMB: 1_500)
        let f = try fixture(localModels: [model], downloader: InstantDownloader())
        defer { try? FileManager.default.removeItem(at: f.root) }
        await f.controller.start()
        f.controller.startStarterDownload()
        let failed = StarterDownloadState.failed(code: StarterDownloadCoordinator.modelLoadFailedCode)
        await waitUntil { f.controller.menuStarterState == failed }
        #expect(f.controller.menuStarterState == failed)
        f.controller.closeNoModelWindow()

        f.controller.openChat()
        await waitUntil { !f.presenter.shown.isEmpty }
        #expect(f.presenter.shown.map(\.modelID) == [model.id])
        #expect(f.controller.menuStarterState == .idle)
        f.backend.release()
    }

    /// After the starter opened chat, a later "no model" (models deleted)
    /// shows the offer again, not "ready, opening chat".
    @Test func noModelWindowAfterAReadyStarterShowsTheOffer() async throws {
        let f = try fixture(localModels: [], downloader: InstantDownloader(), loads: true)
        defer { try? FileManager.default.removeItem(at: f.root) }
        await f.controller.start()
        f.controller.startStarterDownload()
        await waitUntil { !f.presenter.shown.isEmpty }
        #expect(f.presenter.shown.map(\.modelID) == [Self.starter])
        #expect(!f.controller.isNoModelWindowVisible)

        f.controller.openChat()
        await waitUntil { f.controller.isNoModelWindowVisible }
        #expect(f.controller.isNoModelWindowVisible)
        #expect(f.controller.noModelWindowState == .idle)
        f.controller.closeNoModelWindow()
        f.backend.release()
    }
}
