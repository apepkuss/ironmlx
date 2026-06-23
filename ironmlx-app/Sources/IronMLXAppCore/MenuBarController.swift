import AppKit
import Foundation

@MainActor
public final class MenuBarController: NSObject {
    private let statusItem: NSStatusItem
    private let configStore: AppConfigStore
    private let backend: BackendProcessManager
    private let dashboard: DashboardWindowController
    private let scanner: LocalModelScanner
    private let parameterStore: ModelParameterStore
    private let fileManager: FileManager
    private var loadedModelNames: [String] = []
    private var isRefreshingLoadedModelNames = false

    public init(
        configStore: AppConfigStore,
        backend: BackendProcessManager,
        dashboard: DashboardWindowController,
        scanner: LocalModelScanner = LocalModelScanner(),
        fileManager: FileManager = .default
    ) {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        self.configStore = configStore
        self.backend = backend
        self.dashboard = dashboard
        self.scanner = scanner
        self.parameterStore = .shared
        self.fileManager = fileManager
        super.init()
        observeLanguageChanges()
        configureStatusItem()
        rebuildMenu()
    }

    public func rebuildMenu() {
        let snapshot = snapshot()
        statusItem.menu = MenuBarMenuBuilder.makeMenu(snapshot: snapshot, target: self)
        refreshLoadedModelNamesIfNeeded(state: snapshot.state)
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
        startConfiguredBackend()
    }

    @objc public func stopServer(_ sender: NSMenuItem) {
        backend.stop()
        loadedModelNames = []
        rebuildMenu()
    }

    @objc public func restartServer(_ sender: NSMenuItem) {
        backend.stop()
        loadedModelNames = []
        startConfiguredBackend()
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

    private func snapshot() -> MenuBarMenuSnapshot {
        let config = configStore.load()
        let state = backend.isRunning ? .running : backend.state
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

    private func menuModelNames(config: AppConfig, state: BackendProcessState) -> [String] {
        guard state == .running || state == .starting else {
            return []
        }
        if !loadedModelNames.isEmpty {
            return loadedModelNames
        }
        if let model = config.lastModel?.trimmingCharacters(in: .whitespacesAndNewlines), !model.isEmpty {
            return [model]
        }
        return []
    }

    private func refreshLoadedModelNamesIfNeeded(state: BackendProcessState) {
        guard state == .running || state == .starting, !isRefreshingLoadedModelNames else {
            return
        }
        isRefreshingLoadedModelNames = true
        let config = configStore.load()
        let fallback = config.lastModel

        Task {
            let names: [String]
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                let models = try await client.fetchLoadedModels()
                names = MenuBarMenuBuilder.modelNames(from: models, fallback: fallback)
            } catch {
                IronMLXAppLogger.error("Failed to refresh menu loaded models: \(error)")
                names = MenuBarMenuBuilder.modelNames(from: [], fallback: fallback)
            }

            await MainActor.run {
                self.isRefreshingLoadedModelNames = false
                guard names != self.loadedModelNames else {
                    return
                }
                self.loadedModelNames = names
                self.rebuildMenu()
            }
        }
    }

    private func startConfiguredBackend() {
        let config = configStore.load()
        guard let model = config.lastModel?.trimmingCharacters(in: .whitespacesAndNewlines), !model.isEmpty else {
            dashboard.show(route: .modelsManage)
            return
        }

        do {
            try backend.start(modelReference: model)
            rebuildMenu()
        } catch {
            IronMLXAppLogger.error("Failed to start ironmlx backend from menu: \(error)")
            dashboard.show(route: .modelsManage)
            return
        }

        let resolvedModel = scanner.resolveModelPath(for: model) ?? model
        Task {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                let response = try await client.loadModel(
                    model: model,
                    modelDir: resolvedModel,
                    setDefault: true,
                    maxCacheCap: ModelLoadParameters.maxCacheCap(
                        for: model,
                        scanner: self.scanner,
                        parameterStore: self.parameterStore
                    ),
                    reloadWhenIdle: false,
                    samplingDefaults: self.parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
                )
                let names = MenuBarMenuBuilder.modelNames(from: response.loadedModels, fallback: model)
                await MainActor.run {
                    self.loadedModelNames = names
                }
            } catch {
                IronMLXAppLogger.error("Failed to load ironmlx model from menu: \(error)")
            }
            await MainActor.run {
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
