import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func menuBarIconIncludesStandardAndRetinaRepresentations() throws {
    let image = try #require(MenuBarIconLoader.load())

    #expect(image.size == NSSize(width: 34, height: 22))
    #expect(image.isTemplate)
    #expect(image.representations.count == 2)
    #expect(
        image.representations.contains {
            $0.size == NSSize(width: 34, height: 22)
                && $0.pixelsWide == 34
                && $0.pixelsHigh == 22
        }
    )
    #expect(
        image.representations.contains {
            $0.size == NSSize(width: 34, height: 22)
                && $0.pixelsWide == 68
                && $0.pixelsHigh == 44
        }
    )
}

@Test @MainActor
func menuBarRefreshesImmediatelyForBackendRuntimeNotification() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .running, isRunning: true)
    let notificationCenter = NotificationCenter()
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter
    )

    backend.state = .recovering
    notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: backend)

    #expect(controller.rebuildMenu().state == .recovering)
}

@Test
func menuBarRefreshesLoadedModelsOnlyAfterBackendReadiness() {
    #expect(!MenuBarController.shouldRefreshLoadedModelNames(in: .stopped))
    #expect(!MenuBarController.shouldRefreshLoadedModelNames(in: .starting))
    #expect(!MenuBarController.shouldRefreshLoadedModelNames(in: .stopping))
    #expect(!MenuBarController.shouldRefreshLoadedModelNames(in: .recovering))
    #expect(!MenuBarController.shouldRefreshLoadedModelNames(in: .failed))
    #expect(MenuBarController.shouldRefreshLoadedModelNames(in: .running))
    #expect(MenuBarController.shouldRefreshLoadedModelNames(in: .degraded))
}

@Test @MainActor
func menuBarRestartUsesExplicitPlannedRestartIntent() async throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(
        state: .running,
        isRunning: true,
        restartResult: BackendRestartResult(
            success: true,
            status: "restarted",
            port: 9068,
            loadedModels: []
        )
    )
    let notificationCenter = NotificationCenter()
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter
    )

    controller.restartServer(NSMenuItem())

    try await waitForMenuCondition {
        backend.calls.contains("restart:plannedRestart")
    }
    #expect(backend.state == .running)
}

@Test @MainActor
func menuBarStopUsesUserStopAndNotifiesLoadedModelObservers() async throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .running, isRunning: true)
    let notificationCenter = NotificationCenter()
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
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

    try await waitForMenuCondition {
        backend.calls.contains("stop:userStop") && probe.notified
    }
    #expect(backend.state == .stopped)
}

@Test @MainActor
func menuBarReflectsUpdaterAvailabilityAndForwardsManualCheck() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .stopped, isRunning: false)
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let updateManager = TestAppUpdateManager(canCheckForUpdates: true)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        updateManager: updateManager
    )

    #expect(controller.rebuildMenu().updatesEnabled)
    controller.checkForUpdates(NSMenuItem())
    #expect(updateManager.checkCount == 1)

    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: controller.rebuildMenu(),
        target: controller
    )
    updateManager.canCheckForUpdates = false
    controller.menuWillOpen(menu)
    let updateItem = menu.items.first { $0.action == #selector(MenuBarController.checkForUpdates(_:)) }
    #expect(updateItem?.isEnabled == false)
}

@Test @MainActor
func menuBarReflectsConfigurationRecoveryAndForwardsRecoveryAction() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .stopped, isRunning: false)
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let recovery = TestConfigurationRecoveryManager(hasIssues: true)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        configurationRecovery: recovery
    )

    #expect(controller.rebuildMenu().configurationRecoveryAvailable)
    controller.showConfigurationRecovery(NSMenuItem())
    #expect(recovery.presentationCount == 1)
}

@Test
func dflash2TargetComesFromModelSettingsNotTheDefaultModel() throws {
    let root = try menuTemporaryDirectory()
    let parameters = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try parameters.recordDFlash2LoadPreference(modelID: "b", enabled: true, dflash2ModelID: "draft-b")
    try parameters.recordDFlash2LoadPreference(modelID: "c", enabled: true, dflash2ModelID: "draft-c")

    // The default model neither selects nor vetoes the target.
    #expect(AppConfig(defaultModel: "a", loadedModels: ["a", "b"])
        .configuredDFlash2Target(parameterStore: parameters) == "b")
    #expect(AppConfig(defaultModel: "c", loadedModels: ["a"])
        .configuredDFlash2Target(parameterStore: parameters) == nil)
    // Several loaded candidates: the earliest in the persisted load order wins.
    #expect(AppConfig(defaultModel: "b", loadedModels: ["c", "b"])
        .configuredDFlash2Target(parameterStore: parameters) == "c")
    #expect(AppConfig(defaultModel: "c", loadedModels: ["b", "c"])
        .configuredDFlash2Target(parameterStore: parameters) == "b")
}

@Test @MainActor
func toggleMenuBarSpeedFlipsPreference() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .stopped, isRunning: false)
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let suite = "ironmlx.glance.menu-tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let notificationCenter = NotificationCenter()
    let glance = RuntimeGlanceCoordinator(
        preferences: GlancePreferences(defaults: defaults),
        endpoint: { nil },
        backendState: { backend.state },
        notificationCenter: notificationCenter
    )
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter,
        glance: glance
    )
    #expect(controller.rebuildMenu().menuBarSpeedEnabled)

    controller.toggleMenuBarSpeed(NSMenuItem())

    #expect(!glance.preferences.menuBarSpeedEnabled)
    let snapshot = controller.rebuildMenu()
    #expect(!snapshot.menuBarSpeedEnabled)
    let menu = MenuBarMenuBuilder.makeMenu(snapshot: snapshot, target: controller)
    let speed = menu.items.first { $0.action == #selector(MenuBarController.toggleMenuBarSpeed(_:)) }
    #expect(speed?.state == .off)

    controller.toggleHUD(NSMenuItem())
    #expect(glance.preferences.hudVisible)
    #expect(controller.rebuildMenu().hudVisible)

    // The installed menu's check marks follow the preferences: the actions update them in place,
    // and opening the menu picks up changes made elsewhere (e.g. the HUD hotkey) without a rebuild.
    let installed = try #require(controller.installedMenu)
    func toggleStates() -> (hud: NSControl.StateValue?, speed: NSControl.StateValue?) {
        (
            installed.items.first { $0.action == #selector(MenuBarController.toggleHUD(_:)) }?.state,
            installed.items.first { $0.action == #selector(MenuBarController.toggleMenuBarSpeed(_:)) }?.state
        )
    }
    #expect(toggleStates() == (.on, .off))

    controller.toggleMenuBarSpeed(NSMenuItem())
    #expect(controller.installedMenu === installed)
    #expect(toggleStates() == (.on, .on))

    glance.preferences.hudVisible = false
    glance.preferences.menuBarSpeedEnabled = false
    controller.menuWillOpen(installed)
    #expect(controller.installedMenu === installed)
    #expect(toggleStates() == (.off, .off))
}

@Test @MainActor
func hudClickThroughItemFollowsPreferences() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .stopped, isRunning: false)
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let suite = "ironmlx.glance.menu-tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let notificationCenter = NotificationCenter()
    let glance = RuntimeGlanceCoordinator(
        preferences: GlancePreferences(defaults: defaults),
        endpoint: { nil },
        backendState: { backend.state },
        notificationCenter: notificationCenter
    )
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter,
        glance: glance
    )
    let installed = try #require(controller.installedMenu)
    func clickThroughItem() -> NSMenuItem? {
        installed.items.first { $0.action == #selector(MenuBarController.toggleHUDClickThrough(_:)) }
    }
    #expect(clickThroughItem()?.isHidden != false)

    // Turning the HUD on (here as the hotkey would, with the menu installed) reveals the item in place.
    glance.preferences.hudVisible = true
    controller.menuWillOpen(installed)
    #expect(controller.installedMenu === installed)
    #expect(clickThroughItem()?.isHidden == false)
    #expect(clickThroughItem()?.state == .off)

    controller.toggleHUDClickThrough(NSMenuItem())
    #expect(glance.preferences.hudClickThrough)
    #expect(clickThroughItem()?.state == .on)
    controller.toggleHUDClickThrough(NSMenuItem())
    #expect(!glance.preferences.hudClickThrough)

    glance.preferences.hudVisible = false
    controller.menuWillOpen(installed)
    #expect(clickThroughItem()?.isHidden != false)
}

@Test @MainActor
func statusCardRefitsWhenMenuOpens() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .stopped, isRunning: false)
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let suite = "ironmlx.glance.menu-tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let notificationCenter = NotificationCenter()
    let glance = RuntimeGlanceCoordinator(
        preferences: GlancePreferences(defaults: defaults),
        endpoint: { nil },
        backendState: { backend.state },
        notificationCenter: notificationCenter
    )
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter,
        glance: glance
    )
    // Built while the store is not connected: a one-line card.
    let menu = try #require(controller.installedMenu)
    let card = try #require(menu.items.first?.view)
    let oneLineHeight = card.frame.height
    #expect(card.frame.width == RuntimeStatusCardView.width)

    // The stream goes live with no menu rebuild in between.
    let store = glance.store
    store.setBackendState(.running)
    store.setConnection(.connected)
    store.apply(
        RuntimeMetricsSnapshot(
            activity: .init(state: .generating, activeRequests: 1),
            performance: .init(liveDecodeTokensPerSecond: 42.8, prefillTokensPerSecond: 1_840, ttftMs: 312),
            memory: .init(processBytes: 1 << 33, ceilingBytes: 1 << 36),
            models: [.init(id: "mlx-community/Qwen3-0.6B-4bit")]
        ),
        receivedAt: Date()
    )

    controller.menuWillOpen(menu)

    #expect(controller.installedMenu === menu)
    #expect(menu.items.first?.view === card)
    #expect(card.frame.width == RuntimeStatusCardView.width)
    #expect(card.frame.height == card.fittingSize.height)
    #expect(card.frame.height > oneLineHeight + 100)
}

@Test @MainActor
func menuBarBackendListCannotEraseDFlash2StartupIntent() throws {
    let root = try menuTemporaryDirectory()
    let parameters = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let model = "standalone/flash"
    try parameters.recordDFlash2LoadPreference(modelID: model, enabled: true, dflash2ModelID: "draft")
    let idle = AppConfig(defaultModel: "other", loadedModels: [])
    let dflash2 = AppConfig(defaultModel: model, loadedModels: [model])

    // The query started before the DFlash2 intent was saved.
    #expect(!MenuBarController.backendModelListCanReplacePersistedState(
        queriedConfig: idle,
        currentConfig: dflash2,
        parameterStore: parameters
    ))
    // The query started after the save, while the old daemon still answers.
    #expect(!MenuBarController.backendModelListCanReplacePersistedState(
        queriedConfig: dflash2,
        currentConfig: dflash2,
        parameterStore: parameters
    ))
    // Ordinary daemons stay authoritative for their loaded-model list.
    #expect(MenuBarController.backendModelListCanReplacePersistedState(
        queriedConfig: idle,
        currentConfig: idle,
        parameterStore: parameters
    ))
}

@MainActor
private final class MenuLoadedModelsNotificationProbe: NSObject {
    private(set) var notified = false

    @objc func loadedModelsDidChange(_ notification: Notification) {
        notified = true
    }
}

@MainActor
private final class TestAppUpdateManager: AppUpdateManaging {
    var canCheckForUpdates: Bool
    private(set) var checkCount = 0

    init(canCheckForUpdates: Bool) {
        self.canCheckForUpdates = canCheckForUpdates
    }

    func checkForUpdates(_ sender: Any?) {
        checkCount += 1
    }
}

@MainActor
private final class TestConfigurationRecoveryManager: ConfigurationRecoveryManaging {
    var hasIssues: Bool
    private(set) var presentationCount = 0

    init(hasIssues: Bool) {
        self.hasIssues = hasIssues
    }

    func inspect() {}

    func presentRecovery(_ sender: Any?) {
        presentationCount += 1
    }
}

private func menuTemporaryDirectory() throws -> URL {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-menu-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

@MainActor
private func waitForMenuCondition(
    // Hundreds of WebKit tests share MainActor in the full suite. A two
    // second scheduling deadline was flaky under build/UI load; keep the
    // condition unchanged and allow a bounded five second wait.
    timeoutSeconds: TimeInterval = 5.0,
    condition: @escaping () -> Bool
) async throws {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    while Date() < deadline {
        if condition() {
            return
        }
        try await Task.sleep(for: .milliseconds(20))
    }
    Issue.record("Timed out waiting for menu operation")
}
