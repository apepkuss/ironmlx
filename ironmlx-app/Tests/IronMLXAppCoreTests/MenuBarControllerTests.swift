import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor func menuBarRestartShowsStartingStateImmediately() async throws {
    let (root, _) = try menuRestartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let configURL = root
        .appendingPathComponent("config", isDirectory: true)
        .appendingPathComponent("app_config.json")
    let configStore = AppConfigStore(url: configURL)
    configStore.save(AppConfig(port: 19068))

    let backend = FakeMenuRestartBackend()
    let scanner = LocalModelScanner(rootURL: root)
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in FakeMenuRestartLoader() }
    )
    let notificationCenter = NotificationCenter()
    let dashboardBackend = BackendProcessManager(configStore: configStore, scanner: scanner)
    let dashboard = DashboardWindowController(configStore: configStore, backend: dashboardBackend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        scanner: scanner,
        restartCoordinator: coordinator,
        notificationCenter: notificationCenter
    )

    backend.isRunning = true
    controller.restartServer(NSMenuItem())

    let snapshot = controller.rebuildMenu()
    #expect(snapshot.state == .starting)
}

@Test @MainActor func menuBarRestartUsesDashboardRestartSemanticsWhenNoModelIsLoaded() async throws {
    let (root, snapshot) = try menuRestartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let configURL = root
        .appendingPathComponent("config", isDirectory: true)
        .appendingPathComponent("app_config.json")
    let configStore = AppConfigStore(url: configURL)
    configStore.save(AppConfig(port: 19068))

    let backend = FakeMenuRestartBackend()
    let loader = FakeMenuRestartLoader()
    let scanner = LocalModelScanner(rootURL: root)
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )
    let notificationCenter = NotificationCenter()
    let dashboardBackend = BackendProcessManager(configStore: configStore, scanner: scanner)
    let dashboard = DashboardWindowController(configStore: configStore, backend: dashboardBackend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        scanner: scanner,
        restartCoordinator: coordinator,
        notificationCenter: notificationCenter
    )

    backend.isRunning = true
    controller.restartServer(NSMenuItem())

    try await waitForMenuRestart {
        await loader.calls.contains("register:mlx-community/Tiny-4bit:\(snapshot.path):false:nil:false:nil:nil")
    }
    #expect(backend.calls == [
        "stop",
        "start",
    ])
    #expect(configStore.load().loadedModels == [])
}

@Test @MainActor func menuBarStopServerNotifiesDashboardToRefreshLoadedModelState() async throws {
    let (root, _) = try menuRestartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let configURL = root
        .appendingPathComponent("config", isDirectory: true)
        .appendingPathComponent("app_config.json")
    let configStore = AppConfigStore(url: configURL)
    configStore.save(AppConfig(port: 9068, loadedModels: ["mlx-community/Tiny-4bit"]))
    let backend = FakeMenuRestartBackend()
    backend.isRunning = true
    let scanner = LocalModelScanner(rootURL: root)
    let notificationCenter = NotificationCenter()
    let dashboardBackend = BackendProcessManager(configStore: configStore, scanner: scanner)
    let dashboard = DashboardWindowController(configStore: configStore, backend: dashboardBackend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        scanner: scanner,
        notificationCenter: notificationCenter
    )
    let probe = MenuLoadedModelsNotificationProbe()
    notificationCenter.addObserver(
        probe,
        selector: #selector(MenuLoadedModelsNotificationProbe.loadedModelsDidChange(_:)),
        name: .ironMLXLoadedModelsDidChange,
        object: controller
    )
    defer {
        notificationCenter.removeObserver(probe)
    }

    controller.stopServer(NSMenuItem())

    #expect(probe.notified)
    #expect(backend.calls == ["stop"])
}

@MainActor
private final class MenuLoadedModelsNotificationProbe: NSObject {
    private(set) var notified = false

    @objc func loadedModelsDidChange(_ notification: Notification) {
        notified = true
    }
}

private func menuRestartModelRoot(repoID: String) throws -> (root: URL, snapshot: URL) {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-menu-restart-tests-\(UUID().uuidString)", isDirectory: true)
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    return (root, snapshot)
}

@MainActor
private func waitForMenuRestart(
    timeoutSeconds: TimeInterval = 2.0,
    condition: @escaping () async -> Bool
) async throws {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    while Date() < deadline {
        if await condition() {
            return
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
    Issue.record("Timed out waiting for menu restart")
}

@MainActor
private final class FakeMenuRestartBackend: MenuBarBackendProcessManaging {
    var isRunning = false
    private(set) var state: BackendProcessState = .running
    private(set) var calls: [String] = []

    func start() throws {
        calls.append("start")
        isRunning = true
        state = .running
    }

    func stop() {
        calls.append("stop")
        isRunning = false
        state = .stopped
    }

    func stopForAppQuit() {
        calls.append("stopForAppQuit")
        isRunning = false
        state = .stopped
    }
}

private actor FakeMenuRestartLoader: BackendModelLoading {
    private(set) var calls: [String] = []

    func waitUntilReady(timeout: TimeInterval) async throws {
        calls.append("waitUntilReady")
    }

    func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        calls.append("register:\(model):\(modelDir):\(setDefault):\(maxCacheCap.map(String.init) ?? "nil"):\(pinned):\(mtpModelDir ?? "nil"):\(mtpDraftTokens.map(String.init) ?? "nil")")
        return BackendModelAdminResponse(
            success: true,
            status: "registered",
            code: nil,
            model: model,
            loadedModels: [],
            warningCode: nil,
            warning: nil,
            error: nil
        )
    }

    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        calls.append("load:\(model):\(modelDir):\(setDefault):\(maxCacheCap.map(String.init) ?? "nil"):\(pinned):\(mtpModelDir ?? "nil"):\(mtpDraftTokens.map(String.init) ?? "nil")")
        return BackendModelAdminResponse(
            success: true,
            status: "loaded",
            code: nil,
            model: model,
            loadedModels: [
                BackendLoadedModelInfo(
                    id: model,
                    model: model,
                    path: modelDir,
                    architecture: "llm",
                    isDefault: true,
                    maxPositionEmbeddings: 4096,
                    pinned: pinned
                ),
            ],
            warningCode: nil,
            warning: nil,
            error: nil
        )
    }
}
