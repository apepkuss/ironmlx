import AppKit
import Foundation

@MainActor
public final class AppDelegate: NSObject, NSApplicationDelegate {
    private let configStore: AppConfigStore
    private let backend: BackendRuntimeSupervisor
    private let scanner: LocalModelScanner
    private let launchPlanner: AppLaunchPlanner
    private var dashboard: DashboardWindowController?
    private var menu: MenuBarController?
    private var terminationReplyPending = false

    public override init() {
        let store = AppConfigStore.shared
        let scanner = LocalModelScanner()
        let parameterStore = ModelParameterStore.shared
        let processManager = BackendProcessManager(configStore: store)
        self.configStore = store
        self.backend = BackendRuntimeSupervisor(
            processManager: processManager,
            configStore: store,
            scanner: scanner,
            parameterStore: parameterStore
        )
        self.scanner = scanner
        self.launchPlanner = AppLaunchPlanner()
        super.init()
    }

    public func applicationDidFinishLaunching(_ notification: Notification) {
#if IRONMLX_APP_BUNDLE
        do {
            _ = try BundledRuntimeLayout.resolve()
        } catch {
            fatalError("Invalid IronMLX App Bundle: \(error.localizedDescription)")
        }
#endif
        IronMLXAppLogger.startSession()
        IronMLXAppLogger.info("Application did finish launching")
        Task.detached {
            ModelVersionManagementService().purgeTrash()
        }
        NSApp.mainMenu = ApplicationMenuBuilder.makeMainMenu()
        NSApp.setActivationPolicy(.accessory)

        let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
        let menu = MenuBarController(configStore: configStore, backend: backend, dashboard: dashboard)
        self.dashboard = dashboard
        self.menu = menu

        let config = configStore.load()
        let pinnedModels = Set(config.pinnedModelReferences)
        let localModels = scanner.scan(
            loadedModels: Set(config.restoredModelReferences),
            pinnedModels: pinnedModels,
            mtpEnabledModels: []
        )
        let launchPlan = launchPlanner.plan(config: config, localModels: localModels)
        if !localModels.isEmpty || !launchPlan.backendModelReferences.isEmpty {
            Task {
                do {
                    try await backend.ensureRunning()
                } catch {
                    IronMLXAppLogger.error(
                        "Failed to start ironmlx backend on app launch: \(error)"
                    )
                }
                menu.rebuildMenu()
            }
        }

        dashboard.show(route: launchPlan.dashboardRoute)
    }

    public func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        dashboard?.cancelAllDownloads()
        guard backend.state != .stopped else {
            return .terminateNow
        }
        guard !terminationReplyPending else {
            return .terminateLater
        }
        terminationReplyPending = true
        Task {
            await backend.stopForAppQuit()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    public func applicationWillTerminate(_ notification: Notification) {
        IronMLXAppLogger.info("Application will terminate")
        dashboard?.cancelAllDownloads()
    }
}
