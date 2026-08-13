import AppKit
import Foundation

@MainActor
public protocol MenuBarBackendProcessManaging: BackendRuntimeManaging {}

extension BackendRuntimeSupervisor: MenuBarBackendProcessManaging {}

@MainActor
public final class MenuBarController: NSObject, NSMenuDelegate {
    private let statusItem: NSStatusItem
    private let configStore: AppConfigStore
    private let backend: any MenuBarBackendProcessManaging
    private let dashboard: DashboardWindowController
    private let updateManager: any AppUpdateManaging
    private let configurationRecovery: any ConfigurationRecoveryManaging
    private let notificationCenter: NotificationCenter
    private var loadedModelNames: [String]?
    private var isRefreshingLoadedModelNames = false
    private var refreshTimer: Timer?

    public init(
        configStore: AppConfigStore,
        backend: any MenuBarBackendProcessManaging,
        dashboard: DashboardWindowController,
        updateManager: any AppUpdateManaging = DisabledAppUpdateManager(),
        configurationRecovery: any ConfigurationRecoveryManaging = DisabledConfigurationRecoveryManager(),
        notificationCenter: NotificationCenter = .default
    ) {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        self.configStore = configStore
        self.backend = backend
        self.dashboard = dashboard
        self.updateManager = updateManager
        self.configurationRecovery = configurationRecovery
        self.notificationCenter = notificationCenter
        super.init()
        observeLanguageChanges()
        observeLoadedModelChanges()
        observeBackendRuntimeChanges()
        configureStatusItem()
        startLoadedModelRefreshTimer()
        rebuildMenu()
    }

    deinit {
        MainActor.assumeIsolated {
            refreshTimer?.invalidate()
        }
        notificationCenter.removeObserver(self)
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
        guard let iconURL = IronMLXAppResourceResolver.url(forResource: "menubar-icon", withExtension: "png"),
              let image = NSImage(contentsOf: iconURL)
        else {
            preconditionFailure("IronMLX App Bundle is missing menubar-icon.png")
        }
        image.isTemplate = true
        button.image = image
    }

    @objc public func openDashboard(_ sender: NSMenuItem) {
        dashboard.show()
    }

    @objc public func startServer(_ sender: NSMenuItem) {
        restartConfiguredBackendFromMenu()
    }

    @objc public func stopServer(_ sender: NSMenuItem) {
        Task {
            await backend.stop(intent: .userStop)
            loadedModelNames = []
            notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: self)
            rebuildMenu()
        }
    }

    @objc public func restartServer(_ sender: NSMenuItem) {
        restartConfiguredBackendFromMenu()
    }

    @objc public func checkForUpdates(_ sender: NSMenuItem) {
        updateManager.checkForUpdates(sender)
    }

    @objc public func showConfigurationRecovery(_ sender: NSMenuItem) {
        configurationRecovery.presentRecovery(sender)
        rebuildMenu()
    }

    @objc public func quit(_ sender: NSMenuItem) {
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

    @objc private func backendRuntimeDidChange(_ notification: Notification) {
        if backend.state != .running, backend.state != .degraded {
            loadedModelNames = []
        }
        rebuildMenu()
    }

    public func menuWillOpen(_ menu: NSMenu) {
        menu.items.first(where: { $0.action == #selector(checkForUpdates(_:)) })?.isEnabled =
            updateManager.canCheckForUpdates
        refreshLoadedModelNamesIfNeeded(state: snapshot().state)
    }

    private func snapshot() -> MenuBarMenuSnapshot {
        let config = configStore.load()
        let state = backend.state
        let modelNames = menuModelNames(config: config, state: state)
        return MenuBarMenuSnapshot(
            state: state,
            modelNames: modelNames,
            updatesEnabled: updateManager.canCheckForUpdates,
            configurationRecoveryAvailable: configurationRecovery.hasIssues,
            language: config.language
        )
    }

    private func observeLanguageChanges() {
        notificationCenter.addObserver(
            self,
            selector: #selector(menuLanguageDidChange(_:)),
            name: .ironMLXMenuLanguageDidChange,
            object: nil
        )
    }

    private func observeLoadedModelChanges() {
        notificationCenter.addObserver(
            self,
            selector: #selector(loadedModelsDidChange(_:)),
            name: .ironMLXLoadedModelsDidChange,
            object: nil
        )
    }

    private func observeBackendRuntimeChanges() {
        notificationCenter.addObserver(
            self,
            selector: #selector(backendRuntimeDidChange(_:)),
            name: .ironMLXBackendRuntimeDidChange,
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
                let snapshot = self.snapshot()
                let menuHasRecoveryItem = self.statusItem.menu?.items.contains {
                    $0.action == #selector(MenuBarController.showConfigurationRecovery(_:))
                } ?? false
                if menuHasRecoveryItem != snapshot.configurationRecoveryAvailable {
                    self.rebuildMenu()
                    return
                }
                self.refreshLoadedModelNamesIfNeeded(state: snapshot.state)
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
        guard state == .running || state == .starting || state == .degraded,
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
        let pinned = AppConfig.normalizedModelReferences(models.filter(\.pinned).map(\.id))
        let backendDefault = AppConfig.normalizedModelReference(models.first(where: \.isDefault)?.id)
        let config = configStore.load()
        let persistedLoaded = AppConfig.normalizedModelReferences(config.loadedModels ?? [])
        let persistedPinned = config.pinnedModelReferences
        let defaultChanged = backendDefault != nil && backendDefault != config.defaultModelReference
        guard Set(loaded) != Set(persistedLoaded)
            || Set(pinned) != Set(persistedPinned)
            || defaultChanged
        else {
            return
        }
        configStore.update { config in
            config.replaceLoadedModels(loaded, defaultModel: backendDefault)
            config.replacePinnedModels(pinned)
        }
        backend.confirmLoadedModels(models, parameterConfirmedModelIDs: [])
        notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: self)
    }

    private func restartConfiguredBackendFromMenu() {
        loadedModelNames = []
        rebuildMenu()

        Task {
            let result = await self.backend.restart(intent: .plannedRestart)
            if result.success {
                self.configStore.update { updatedConfig in
                    updatedConfig.replaceLoadedModels(
                        result.loadedModels,
                        defaultModel: result.model
                    )
                }
                self.loadedModelNames = result.loadedModels
                self.notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: self)
                IronMLXAppLogger.info(
                    "Restarted ironmlx backend from menu: status=\(result.status) loaded_models=\(result.loadedModels.count)"
                )
            } else {
                self.loadedModelNames = result.loadedModels
                IronMLXAppLogger.error(
                    "Failed to restart ironmlx backend from menu: \(result.error ?? result.status)"
                )
                if result.errorCode == BackendRuntimeFailureCode.instanceAlreadyRunning.rawValue {
                    BackendInstanceConflictPresentation.presentAlert(
                        language: self.configStore.load().language
                    )
                }
            }
            self.rebuildMenu()
        }
    }

}
