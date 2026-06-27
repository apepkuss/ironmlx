import AppKit
import Foundation

@MainActor
public protocol MenuBarBackendProcessManaging: BackendProcessManaging {
    var state: BackendProcessState { get }

    func stopForAppQuit()
}

extension BackendProcessManager: MenuBarBackendProcessManaging {}

@MainActor
public final class MenuBarController: NSObject, NSMenuDelegate {
    private let statusItem: NSStatusItem
    private let configStore: AppConfigStore
    private let backend: any MenuBarBackendProcessManaging
    private let dashboard: DashboardWindowController
    private let restartCoordinator: BackendRestartCoordinator
    private let fileManager: FileManager
    private var loadedModelNames: [String]?
    private var isRefreshingLoadedModelNames = false
    private var menuStateOverride: BackendProcessState?
    private var refreshTimer: Timer?

    public init(
        configStore: AppConfigStore,
        backend: any MenuBarBackendProcessManaging,
        dashboard: DashboardWindowController,
        scanner: LocalModelScanner = LocalModelScanner(),
        restartCoordinator: BackendRestartCoordinator? = nil,
        fileManager: FileManager = .default
    ) {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        self.configStore = configStore
        self.backend = backend
        self.dashboard = dashboard
        let parameterStore = ModelParameterStore.shared
        self.restartCoordinator = restartCoordinator ?? BackendRestartCoordinator(
            scanner: scanner,
            parameterStore: parameterStore
        )
        self.fileManager = fileManager
        super.init()
        observeLanguageChanges()
        observeLoadedModelChanges()
        configureStatusItem()
        startLoadedModelRefreshTimer()
        rebuildMenu()
    }

    deinit {
        MainActor.assumeIsolated {
            refreshTimer?.invalidate()
        }
    }

    @discardableResult
    public func rebuildMenu() -> MenuBarMenuSnapshot {
        let snapshot = snapshot()
        let menu = MenuBarMenuBuilder.makeMenu(snapshot: snapshot, target: self)
        menu.delegate = self
        statusItem.menu = menu
        refreshLoadedModelNamesIfNeeded(state: snapshot.state)
        return snapshot
    }

    private func configureStatusItem() {
        guard let button = statusItem.button else {
            return
        }
        if let iconURL = Bundle.module.url(forResource: "menubar-icon", withExtension: "png"),
           let image = NSImage(contentsOf: iconURL) {
            image.isTemplate = true
            button.image = image
        } else {
            button.title = "MLX"
        }
    }

    @objc public func openDashboard(_ sender: NSMenuItem) {
        dashboard.show()
    }

    @objc public func openOpenClawChat(_ sender: NSMenuItem) {
        startOpenClawGatewayIfConfigured()
        openOpenClawURL(path: "openclaw")
    }

    @objc public func openIronHermes(_ sender: NSMenuItem) {
        startIronHermesIfInstalled()
        NSWorkspace.shared.open(URL(string: "http://127.0.0.1:9069")!)
    }

    @objc public func startServer(_ sender: NSMenuItem) {
        restartConfiguredBackendFromMenu()
    }

    @objc public func stopServer(_ sender: NSMenuItem) {
        menuStateOverride = nil
        backend.stop()
        loadedModelNames = []
        rebuildMenu()
    }

    @objc public func restartServer(_ sender: NSMenuItem) {
        restartConfiguredBackendFromMenu()
    }

    @objc public func checkForUpdates(_ sender: NSMenuItem) {
        dashboard.show()
    }

    @objc public func quit(_ sender: NSMenuItem) {
        backend.stopForAppQuit()
        NSApp.terminate(nil)
    }

    @objc private func menuLanguageDidChange(_ notification: Notification) {
        rebuildMenu()
    }

    @objc private func loadedModelsDidChange(_ notification: Notification) {
        if let object = notification.object as AnyObject?,
           object === self {
            return
        }
        loadedModelNames = nil
        rebuildMenu()
    }

    public func menuWillOpen(_ menu: NSMenu) {
        refreshLoadedModelNamesIfNeeded(state: snapshot().state)
    }

    private func snapshot() -> MenuBarMenuSnapshot {
        let config = configStore.load()
        let state = menuStateOverride ?? (backend.isRunning ? .running : backend.state)
        let modelNames = menuModelNames(config: config, state: state)
        return MenuBarMenuSnapshot(
            state: state,
            modelNames: modelNames,
            openClawInstalled: fileManager.fileExists(atPath: openClawCLIPath().path),
            openClawGatewayConfigured: fileManager.fileExists(atPath: openClawGatewayPlistPath().path),
            ironHermesInstalled: fileManager.fileExists(atPath: ironHermesBinaryPath().path),
            updatesEnabled: false,
            language: config.language
        )
    }

    private func observeLanguageChanges() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(menuLanguageDidChange(_:)),
            name: .ironMLXMenuLanguageDidChange,
            object: nil
        )
    }

    private func observeLoadedModelChanges() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(loadedModelsDidChange(_:)),
            name: .ironMLXLoadedModelsDidChange,
            object: nil
        )
    }

    private func startLoadedModelRefreshTimer() {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else {
                    return
                }
                let state = self.snapshot().state
                self.refreshLoadedModelNamesIfNeeded(state: state)
            }
        }
    }

    private func menuModelNames(config: AppConfig, state: BackendProcessState) -> [String] {
        MenuBarMenuBuilder.snapshotModelNames(
            cached: loadedModelNames,
            config: config,
            state: state
        )
    }

    private func refreshLoadedModelNamesIfNeeded(state: BackendProcessState) {
        guard menuStateOverride == nil,
              state == .running || state == .starting,
              !isRefreshingLoadedModelNames
        else {
            return
        }
        isRefreshingLoadedModelNames = true
        let config = configStore.load()

        Task {
            let models: [BackendLoadedModelInfo]?
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                models = try await client.fetchLoadedModels()
            } catch {
                IronMLXAppLogger.error("Failed to refresh menu loaded models: \(error)")
                models = nil
            }

            await MainActor.run {
                self.isRefreshingLoadedModelNames = false
                guard let models else {
                    return
                }
                self.persistLoadedModelsIfNeeded(models)
                let names = MenuBarMenuBuilder.modelNames(from: models)
                guard names != self.loadedModelNames else {
                    return
                }
                self.loadedModelNames = names
                self.rebuildMenu()
            }
        }
    }

    private func persistLoadedModelsIfNeeded(_ models: [BackendLoadedModelInfo]) {
        let loaded = AppConfig.normalizedModelReferences(models.map(\.id))
        let backendDefault = AppConfig.normalizedModelReference(models.first(where: \.isDefault)?.id)
        var config = configStore.load()
        let persistedLoaded = AppConfig.normalizedModelReferences(config.loadedModels ?? [])
        let defaultChanged = backendDefault != nil && backendDefault != config.defaultModelReference
        guard Set(loaded) != Set(persistedLoaded) || defaultChanged else {
            return
        }
        config.replaceLoadedModels(loaded, defaultModel: backendDefault)
        configStore.save(config)
        NotificationCenter.default.post(name: .ironMLXLoadedModelsDidChange, object: self)
    }

    private func restartConfiguredBackendFromMenu() {
        let config = configStore.load()
        menuStateOverride = .starting
        loadedModelNames = []
        rebuildMenu()

        Task {
            let result = await self.restartCoordinator.restartDefaultModel(config: config, backend: self.backend)
            await MainActor.run {
                self.menuStateOverride = nil
                if result.success {
                    var updatedConfig = self.configStore.load()
                    updatedConfig.replaceLoadedModels(result.loadedModels, defaultModel: result.model)
                    self.configStore.save(updatedConfig)
                    self.loadedModelNames = result.loadedModels
                    NotificationCenter.default.post(name: .ironMLXLoadedModelsDidChange, object: self)
                    IronMLXAppLogger.info(
                        "Restarted ironmlx backend from menu: status=\(result.status) loaded_models=\(result.loadedModels.count)"
                    )
                } else {
                    self.loadedModelNames = []
                    IronMLXAppLogger.error(
                        "Failed to restart ironmlx backend from menu: \(result.error ?? result.status)"
                    )
                }
                self.rebuildMenu()
            }
        }
    }

    private func openOpenClawURL(path: String) {
        let token = openClawGatewayToken()
        let suffix = token.isEmpty ? "" : "#token=\(token)"
        NSWorkspace.shared.open(URL(string: "http://127.0.0.1:18789/\(path)\(suffix)")!)
    }

    private func startOpenClawGatewayIfConfigured() {
        guard fileManager.fileExists(atPath: openClawGatewayPlistPath().path) else {
            return
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["start", "ai.openclaw.gateway"]
        try? process.run()
    }

    private func startIronHermesIfInstalled() {
        let binary = ironHermesBinaryPath()
        guard fileManager.fileExists(atPath: binary.path) else {
            return
        }
        let process = Process()
        process.executableURL = binary
        try? process.run()
    }

    private func openClawGatewayToken() -> String {
        let configURL = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".openclaw", isDirectory: true)
            .appendingPathComponent("openclaw.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ""
        }
        if let gateway = object["gateway"] as? [String: Any],
           let token = gateway["token"] as? String {
            return token
        }
        return object["gateway_token"] as? String ?? ""
    }

    private func openClawCLIPath() -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".openclaw", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("openclaw")
    }

    private func openClawGatewayPlistPath() -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("LaunchAgents", isDirectory: true)
            .appendingPathComponent("ai.openclaw.gateway.plist")
    }

    private func ironHermesBinaryPath() -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".iron-hermes", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("iron-hermes")
    }
}
