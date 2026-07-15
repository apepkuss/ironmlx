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
        let pinnedModels = Set(config.pinnedModelReferences)
        let localModels = scanner.scan(
            loadedModels: Set(config.restoredModelReferences),
            pinnedModels: pinnedModels,
            mtpEnabledModels: []
        )
        let launchPlan = launchPlanner.plan(config: config, localModels: localModels)
        if !localModels.isEmpty || !launchPlan.backendModelReferences.isEmpty {
            do {
                try backend.start()
                let models = launchPlan.backendModelReferences
                let defaultModel = config.defaultModelReference
                let localModelsToRegister = localModels
                Task {
                    do {
                        let client = BackendAPIClient(host: config.host, port: config.port)
                        try await client.waitUntilReady()
                        await LocalModelBackendRegistrar.register(
                            localModels: localModelsToRegister,
                            defaultModel: defaultModel,
                            scanner: self.scanner,
                            parameterStore: self.parameterStore,
                            activeKvOffloadEnabled: config.activeKvOffload == true,
                            client: client
                        )
                        var latestResponse: BackendModelAdminResponse?
                        for model in models {
                            do {
                                if let readiness = self.scanner.readiness(for: model), !readiness.isLoadable {
                                    let detail = readiness.message ?? "model snapshot is not ready to load"
                                    IronMLXAppLogger.error("Skipped restoring ironmlx model on app launch: \(model): \(detail)")
                                    continue
                                }
                                let resolvedModel = self.scanner.resolveModelPath(for: model) ?? model
                                let setDefault = model == defaultModel || defaultModel == nil && model == models.first
                                let mtpRuntime = try? ModelMtpRuntimeResolver.runtime(
                                    for: model,
                                    useMtp: nil,
                                    scanner: self.scanner,
                                    parameterStore: self.parameterStore
                                )
                                latestResponse = try await client.loadModel(
                                    model: model,
                                    modelDir: resolvedModel,
                                    setDefault: setDefault,
                                    maxCacheCap: ModelLoadParameters.maxCacheCap(
                                        for: model,
                                        scanner: self.scanner,
                                        parameterStore: self.parameterStore,
                                        activeKvOffloadEnabled: config.activeKvOffload == true
                                    ),
                                    pinned: pinnedModels.contains(model),
                                    mtpModelDir: mtpRuntime?.modelDir,
                                    mtpDraftTokens: mtpRuntime?.draftTokens,
                                    reloadWhenIdle: false,
                                    samplingDefaults: self.parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
                                )
                            } catch {
                                IronMLXAppLogger.error("Failed to restore ironmlx model on app launch: \(model): \(error)")
                            }
                        }
                        if let loadedModels = latestResponse?.loadedModels {
                            await MainActor.run {
                                var updatedConfig = self.configStore.load()
                                updatedConfig.replaceLoadedModels(
                                    loadedModels.map(\.id),
                                    defaultModel: Self.defaultModelForLaunchRestore(
                                        config: updatedConfig,
                                        backendLoadedModels: loadedModels
                                    )
                                )
                                updatedConfig.replacePinnedModels(loadedModels.filter(\.pinned).map(\.id))
                                self.configStore.save(updatedConfig)
                                NotificationCenter.default.post(name: .ironMLXLoadedModelsDidChange, object: self)
                            }
                        }
                    } catch {
                        IronMLXAppLogger.error("Failed to restore ironmlx models on app launch: \(error)")
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

    nonisolated static func defaultModelForLaunchRestore(
        config: AppConfig,
        backendLoadedModels: [BackendLoadedModelInfo]
    ) -> String? {
        config.defaultModelReference ?? backendLoadedModels.first(where: \.isDefault)?.id
    }
}
