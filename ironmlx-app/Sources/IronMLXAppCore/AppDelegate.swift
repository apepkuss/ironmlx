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
    /// Every model load the App starts, and the backend's own, for the Dashboard.
    private let loadActivity: ModelLoadActivityCenter
    /// The single download queue shared by the Dashboard and the starter download.
    private let downloadService: ModelDownloadService
    let starterDownloads: StarterDownloadCoordinator
    /// Set by the chat window; until then "Chat" falls back to the Dashboard.
    weak var chatPresenter: (any ChatWindowPresenting)?
    private var chatEntry: ChatEntryController?
    private var chatWindow: ChatWindowController?
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
        let loadActivity = ModelLoadActivityCenter(configStore: store)
        let processManager = BackendProcessManager(
            configStore: store,
            scanner: scanner,
            parameterStore: parameterStore
        )
        let backend = BackendRuntimeSupervisor(
            processManager: processManager,
            configStore: store,
            scanner: scanner,
            parameterStore: parameterStore,
            loadActivity: loadActivity
        )
        let downloadService = ModelDownloadService()
        self.configStore = store
        self.backend = backend
        self.scanner = scanner
        self.parameterStore = parameterStore
        self.launchPlanner = AppLaunchPlanner()
        self.configurationRecovery = configurationRecovery
        self.loadActivity = loadActivity
        self.downloadService = downloadService
        self.starterDownloads = StarterDownloadCoordinator(
            downloader: downloadService,
            modelManager: {
                try await backend.ensureRunning()
                let config = store.load()
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                return LoadTrackingModelManager(base: client, loadActivity: loadActivity, origin: .starter)
            }
        )
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
        IronMLXAppLogger.setMinimumLevel(.saved(configStore.load().logLevel))
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
        let chatEntry = makeChatEntry()
        Task {
            // Restore an unfinished starter download before any window shows it.
            await chatEntry.start()
            do {
                try await backend.ensureRunning { [weak self] in
                    self?.showFirstSurface(launchPlan.firstSurface)
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

    private func makeChatEntry() -> ChatEntryController {
        let systemCenter = ModelReadyNotifier.systemCenter()
        let notifier = ModelReadyNotifier(center: systemCenter)
        systemCenter?.onResponse = { [weak notifier] userInfo in
            notifier?.handleNotificationResponse(userInfo: userInfo)
        }
        let chatEntry = ChatEntryController(
            configStore: configStore,
            backend: backend,
            scanner: scanner,
            downloadService: downloadService,
            starterDownloads: starterDownloads,
            notifier: notifier,
            presenter: { [weak self] in self?.chatPresenter },
            showDashboard: { [weak self] route in self?.showInterface(route: route) },
            activation: .shared
        )
        self.chatEntry = chatEntry
        let chatWindow = ChatWindowController(
            session: .live(
                configStore: configStore, parameterStore: parameterStore, scanner: scanner, backend: backend,
                loadActivity: loadActivity
            ),
            environment: .live(
                configStore: configStore,
                scanner: scanner,
                openDashboard: { [weak self] route, guide in self?.showInterface(route: route, agentGuide: guide) },
                startUpgrade: { [weak chatEntry] upgrade, onState in
                    chatEntry?.startUpgradeDownload(upgrade, onState: onState)
                }
            )
        )
        self.chatWindow = chatWindow
        chatPresenter = chatWindow
        return chatEntry
    }

    private func showFirstSurface(_ surface: LaunchSurface) {
        switch surface {
        case let .dashboard(route):
            showInterface(route: route)
        case .noModelWindow:
            let created = makeInterfaceIfNeeded()
            chatEntry?.showNoModelWindow()
            if created { presentConfigurationRecoveryIfNeeded() }
        }
    }

    private func showInterface(route: DashboardInitialRoute, agentGuide: AgentGuideID? = nil) {
        let created = makeInterfaceIfNeeded()
        dashboard?.show(route: route, agentGuide: agentGuide)
        if created { presentConfigurationRecoveryIfNeeded() }
    }

    /// Creates the Dashboard controller and the menu bar item once, without
    /// showing the Dashboard. Returns whether it created them.
    private func makeInterfaceIfNeeded() -> Bool {
        if dashboard != nil {
            return false
        }

        let dashboard = DashboardWindowController(
            configStore: configStore, backend: backend, downloadService: downloadService,
            loadActivity: loadActivity
        )
        dashboard.chatEntry = chatEntry
        let updateManager = SparkleAppUpdateManager.make()
        let menu = MenuBarController(
            configStore: configStore,
            backend: backend,
            dashboard: dashboard,
            updateManager: updateManager,
            configurationRecovery: configurationRecovery
        )
        menu.chatEntry = chatEntry
        chatEntry?.menu = menu
        self.dashboard = dashboard
        self.menu = menu
        self.updateManager = updateManager
        return true
    }

    private func presentConfigurationRecoveryIfNeeded() {
        if configurationRecovery.hasIssues {
            configurationRecovery.presentRecovery(nil)
            menu?.rebuildMenu()
        }
    }

    public func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Before the backend stops under it: a reply still streaming is
        // cancelled and what arrived is saved as a stopped turn.
        saveChatReplyInProgress()
        cancelAllDownloads()
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
        saveChatReplyInProgress()
        cancelAllDownloads()
    }

    /// Same as closing the chat window; does nothing when no reply streams.
    private func saveChatReplyInProgress() {
        chatWindow?.session.windowClosed()
    }

    /// Downloads may run without the Dashboard ever being opened.
    private func cancelAllDownloads() {
        dashboard?.cancelAllDownloads()
        let downloadService = downloadService
        Task {
            await downloadService.cancelAllDownloads()
        }
    }
}
