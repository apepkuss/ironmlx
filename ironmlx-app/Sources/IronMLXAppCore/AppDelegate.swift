import AppKit
import Foundation

@MainActor
public final class AppDelegate: NSObject, NSApplicationDelegate {
    private let configStore: AppConfigStore
    private let backend: BackendProcessManager
    private let scanner: LocalModelScanner
    private let parameterStore: ModelParameterStore
    private let launchPlanner: AppLaunchPlanner
    private var dashboard: DashboardWindowController?
    private var menu: MenuBarController?

    public override init() {
        let store = AppConfigStore.shared
        self.configStore = store
        self.backend = BackendProcessManager(configStore: store)
        self.scanner = LocalModelScanner()
        self.parameterStore = .shared
        self.launchPlanner = AppLaunchPlanner()
        super.init()
    }

    public func applicationDidFinishLaunching(_ notification: Notification) {
        IronMLXAppLogger.startSession()
        IronMLXAppLogger.info("Application did finish launching")
        NSApp.mainMenu = ApplicationMenuBuilder.makeMainMenu()
        NSApp.setActivationPolicy(.accessory)

        let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
        let menu = MenuBarController(configStore: configStore, backend: backend, dashboard: dashboard)
        self.dashboard = dashboard
        self.menu = menu

        let config = configStore.load()
        let localModels = scanner.scan(loadedModel: config.lastModel)
        let launchPlan = launchPlanner.plan(config: config, localModels: localModels)
        if let model = launchPlan.backendModelReference {
            do {
                try backend.start(modelReference: model)
                let resolvedModel = scanner.resolveModelPath(for: model) ?? model
                Task {
                    do {
                        let client = BackendAPIClient(host: config.host, port: config.port)
                        try await client.waitUntilReady()
                        _ = try await client.loadModel(
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
                    } catch {
                        IronMLXAppLogger.error("Failed to load ironmlx model on app launch: \(error)")
                    }
                }
            } catch {
                IronMLXAppLogger.error("Failed to start ironmlx backend on app launch: \(error)")
            }
            menu.rebuildMenu()
        }

        dashboard.show(route: launchPlan.dashboardRoute)
    }

    public func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        backend.stopForAppQuit()
        return .terminateNow
    }

    public func applicationWillTerminate(_ notification: Notification) {
        IronMLXAppLogger.info("Application will terminate")
        backend.stopForAppQuit()
    }
}
