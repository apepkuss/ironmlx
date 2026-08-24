import AppKit
import Foundation

@MainActor
public final class AppDelegate: NSObject, NSApplicationDelegate {
    private let configStore: AppConfigStore
    private let backend: BackendRuntimeSupervisor
    private let scanner: LocalModelScanner
    private let parameterStore: ModelParameterStore
    private let launchPlanner: AppLaunchPlanner
    private let configurationRecovery: ConfigurationRecoveryManager
    private var dashboard: DashboardWindowController?
    private var menu: MenuBarController?
    private var updateManager: (any AppUpdateManaging)?
    private var terminationReplyPending = false

    public override init() {
        let store = AppConfigStore.shared
        let scanner = LocalModelScanner()
        let parameterStore = ModelParameterStore.shared
        let configurationRecovery = ConfigurationRecoveryManager(
            appConfigStore: store,
            modelParameterStore: parameterStore
        )
        let processManager = BackendProcessManager(
            configStore: store,
            scanner: scanner,
            parameterStore: parameterStore
        )
        self.configStore = store
        self.backend = BackendRuntimeSupervisor(
            processManager: processManager,
            configStore: store,
            scanner: scanner,
            parameterStore: parameterStore
        )
        self.scanner = scanner
        self.parameterStore = parameterStore
        self.launchPlanner = AppLaunchPlanner()
        self.configurationRecovery = configurationRecovery
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
        configurationRecovery.inspect()

        let config = configStore.load()
        let pinnedModels = Set(config.pinnedModelReferences)
        let localModels = scanner.scan(
            loadedModels: Set(config.restoredModelReferences),
            pinnedModels: pinnedModels,
            mtpEnabledModels: [],
            dflash2EnabledModels: Set(config.restoredModelReferences.filter {
                parameterStore.parameters(for: $0)?.dflash2Enabled == true
            })
        )
        let launchPlan = launchPlanner.plan(config: config, localModels: localModels)
        Task {
            do {
                try await backend.ensureRunning { [weak self] in
                    self?.showInterface(route: launchPlan.dashboardRoute)
                }
                menu?.rebuildMenu()
            } catch {
                IronMLXAppLogger.error(
                    "Failed to start ironmlx backend on app launch: \(error)"
                )
                if (error as? BackendRuntimeSupervisorError)?.failureCode
                    == .instanceAlreadyRunning {
                    BackendInstanceConflictPresentation.presentAlert(
                        language: configStore.load().language
                    )
                    NSApp.terminate(nil)
                    return
                }
                // Preserve the existing diagnostics path for launch failures that
                // are unrelated to a competing backend instance.
                showInterface(route: launchPlan.dashboardRoute)
            }
        }
    }

    private func showInterface(route: DashboardInitialRoute) {
        if let dashboard {
            dashboard.show(route: route)
            return
        }

        let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
        let updateManager = SparkleAppUpdateManager.make()
        let menu = MenuBarController(
            configStore: configStore,
            backend: backend,
            dashboard: dashboard,
            updateManager: updateManager,
            configurationRecovery: configurationRecovery
        )
        self.dashboard = dashboard
        self.menu = menu
        self.updateManager = updateManager

        dashboard.show(route: route)
        if configurationRecovery.hasIssues {
            configurationRecovery.presentRecovery(nil)
            menu.rebuildMenu()
        }
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
