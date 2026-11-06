import AppKit
import Foundation
import Observation

@MainActor
public protocol MenuBarBackendProcessManaging: BackendRuntimeManaging {}

extension BackendRuntimeSupervisor: MenuBarBackendProcessManaging {}

enum MenuBarIconLoader {
    static let logicalSize = NSSize(width: 34, height: 22)

    static func load() -> NSImage? {
        guard
            let iconURL = IronMLXAppResourceResolver.url(
                forResource: "menubar-icon",
                withExtension: "png"
            ),
            let retinaIconURL = IronMLXAppResourceResolver.url(
                forResource: "menubar-icon@2x",
                withExtension: "png"
            )
        else {
            return nil
        }
        return load(iconURL: iconURL, retinaIconURL: retinaIconURL)
    }

    static func load(iconURL: URL, retinaIconURL: URL) -> NSImage? {
        guard
            let iconData = try? Data(contentsOf: iconURL),
            let retinaIconData = try? Data(contentsOf: retinaIconURL),
            let iconRepresentation = NSBitmapImageRep(data: iconData),
            let retinaIconRepresentation = NSBitmapImageRep(data: retinaIconData),
            iconRepresentation.pixelsWide == 34,
            iconRepresentation.pixelsHigh == 22,
            retinaIconRepresentation.pixelsWide == 68,
            retinaIconRepresentation.pixelsHigh == 44
        else {
            return nil
        }

        iconRepresentation.size = logicalSize
        retinaIconRepresentation.size = logicalSize

        let image = NSImage(size: logicalSize)
        image.addRepresentation(iconRepresentation)
        image.addRepresentation(retinaIconRepresentation)
        image.isTemplate = true
        return image
    }
}

@MainActor
public final class MenuBarController: NSObject, NSMenuDelegate {
    private let statusItem: NSStatusItem
    private let configStore: AppConfigStore
    private let parameterStore: ModelParameterStore
    private let backend: any MenuBarBackendProcessManaging
    private let dashboard: DashboardWindowController
    private let updateManager: any AppUpdateManaging
    private let configurationRecovery: any ConfigurationRecoveryManaging
    private let notificationCenter: NotificationCenter
    private var loadedModelNames: [String]?
    private var isRefreshingLoadedModelNames = false
    private var loadedModelRefreshGeneration: UInt64 = 0
    private var loadedModelRefreshTask: Task<Void, Never>?
    private var refreshTimer: Timer?
    /// Live runtime status: the speed readout and dot on the status item, and the card at the top of the menu.
    private let glance: (any RuntimeGlanceControlling)?
    private let statusDot = MenuBarStatusDotView()
    private var defaultImagePosition: NSControl.ImagePosition = .imageOnly
    /// The menu language, cached so the readout never reads the config file.
    private var glanceLanguage = "en"
    /// Handles "Chat" and supplies the starter download row; set by the App.
    weak var chatEntry: (any ChatEntryHandling)? {
        didSet { rebuildMenu() }
    }

    public init(
        configStore: AppConfigStore,
        backend: any MenuBarBackendProcessManaging,
        dashboard: DashboardWindowController,
        parameterStore: ModelParameterStore = .shared,
        updateManager: any AppUpdateManaging = DisabledAppUpdateManager(),
        configurationRecovery: any ConfigurationRecoveryManaging = DisabledConfigurationRecoveryManager(),
        notificationCenter: NotificationCenter = .default,
        glance: (any RuntimeGlanceControlling)? = nil
    ) {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        self.configStore = configStore
        self.parameterStore = parameterStore
        self.backend = backend
        self.dashboard = dashboard
        self.updateManager = updateManager
        self.configurationRecovery = configurationRecovery
        self.notificationCenter = notificationCenter
        self.glance = glance
        super.init()
        observeLanguageChanges()
        observeLoadedModelChanges()
        observeBackendRuntimeChanges()
        configureStatusItem()
        startLoadedModelRefreshTimer()
        rebuildMenu()
        observeGlance()
    }

    deinit {
        MainActor.assumeIsolated {
            refreshTimer?.invalidate()
            loadedModelRefreshTask?.cancel()
        }
        notificationCenter.removeObserver(self)
    }

    @discardableResult
    public func rebuildMenu() -> MenuBarMenuSnapshot {
        let snapshot = snapshot()
        installMenu(snapshot)
        refreshLoadedModelNamesIfNeeded(state: snapshot.state)
        return snapshot
    }

    /// The menu currently attached to the status item.
    var installedMenu: NSMenu? {
        statusItem.menu
    }

    /// Builds and installs the menu. The status card is created here once per build; while the
    /// menu is open it updates through Observation, never by rebuilding the menu.
    private func installMenu(_ snapshot: MenuBarMenuSnapshot) {
        let statusCard = glance.map {
            RuntimeStatusCardView.makeMenuItemView(store: $0.store, language: snapshot.language)
        }
        let menu = MenuBarMenuBuilder.makeMenu(snapshot: snapshot, target: self, statusCard: statusCard)
        menu.delegate = self
        statusItem.menu = menu
        if glanceLanguage != snapshot.language {
            glanceLanguage = snapshot.language
            updateStatusItemGlance()
        }
    }

    private func configureStatusItem() {
        guard let button = statusItem.button else {
            return
        }
        guard let image = MenuBarIconLoader.load() else {
            preconditionFailure("IronMLX App Bundle has invalid menu bar icon resources")
        }
        button.image = image
        guard glance != nil else {
            return
        }
        defaultImagePosition = button.imagePosition
        button.font = NSFont.monospacedDigitSystemFont(ofSize: 0, weight: .regular)
        statusDot.install(in: button)
    }

    /// Keeps the speed readout and the status dot current. Observation re-runs it only when
    /// something it read changes (store or preferences), so an idle App does no work here.
    private func observeGlance() {
        guard glance != nil else {
            return
        }
        withObservationTracking {
            updateStatusItemGlance()
            // The HUD hotkey can flip the HUD toggle while the menu is open.
            updateGlanceToggleStates(in: statusItem.menu)
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                self?.observeGlance()
            }
        }
    }

    private func updateStatusItemGlance() {
        guard let glance, let button = statusItem.button else {
            return
        }
        let title = glance.store.menuBarReadout(
            language: glanceLanguage,
            enabled: glance.preferences.menuBarSpeedEnabled
        ) ?? ""
        statusDot.indicator = glance.store.indicator
        guard button.title != title else {
            return
        }
        button.title = title
        // With no text, the icon sits exactly where it did before the readout existed.
        button.imagePosition = title.isEmpty ? defaultImagePosition : .imageLeading
        statusDot.reposition()
    }

    @objc public func openDashboard(_ sender: NSMenuItem) {
        dashboard.show()
    }

    @objc public func openChat(_ sender: NSMenuItem) {
        guard let chatEntry else {
            IronMLXAppLogger.info("Chat entry is not ready; showing the Dashboard")
            dashboard.show()
            return
        }
        chatEntry.openChat()
    }

    /// Updates the starter download row in place, so an open menu shows live
    /// progress; rebuilds the menu only when the row appears, disappears or
    /// changes what clicking it does.
    func starterDownloadDidChange() {
        let snapshot = snapshot()
        let title = MenuBarMenuBuilder.starterRowTitle(
            snapshot.starterDownload, modelName: snapshot.starterModelName, language: snapshot.language
        )
        let row = statusItem.menu?.item(withTag: MenuBarMenuBuilder.starterRowTag)
        let opensChat = MenuBarMenuBuilder.starterRowOpensChat(snapshot.starterDownload)
        if let row, let title, (row.action != nil) == opensChat {
            row.title = title
            return
        }
        if row == nil, title == nil {
            return
        }
        installMenu(snapshot)
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

    @objc public func toggleHUD(_ sender: NSMenuItem) {
        handleHUDMenuAction(event: NSApp?.currentEvent)
    }

    /// ⌃⌥⌘H pressed while the menu is open arrives here as the item's key equivalent instead of
    /// as the Carbon hotkey, so it takes the hotkey's path (which also drops a second delivery of
    /// the same press). Clicks and Return on the item flip the checked preference.
    func handleHUDMenuAction(event: NSEvent?) {
        if MenuBarMenuBuilder.isHUDHotKeyKeystroke(event, hotKeyRegistered: glance?.hudHotKeyDisplay != nil) {
            glance?.hudHotKeyPressed()
        } else {
            glance?.toggleHUD()
        }
        updateGlanceToggleStates(in: statusItem.menu)
    }

    @objc public func toggleHUDClickThrough(_ sender: NSMenuItem) {
        glance?.preferences.hudClickThrough.toggle()
        updateGlanceToggleStates(in: statusItem.menu)
    }

    @objc public func toggleMenuBarSpeed(_ sender: NSMenuItem) {
        glance?.toggleMenuBarSpeed()
        updateGlanceToggleStates(in: statusItem.menu)
    }

    /// Sets the glance toggles' check marks (and whether "HUD Click-Through" shows) in place; the
    /// HUD can also be toggled by its hotkey or its context menu.
    private func updateGlanceToggleStates(in menu: NSMenu?) {
        guard let glance, let menu else {
            return
        }
        let hudVisible = glance.preferences.hudVisible
        menu.items.first { $0.action == #selector(toggleHUD(_:)) }?.state = hudVisible ? .on : .off
        if let clickThrough = menu.items.first(where: { $0.action == #selector(toggleHUDClickThrough(_:)) }) {
            clickThrough.isHidden = !hudVisible
            clickThrough.state = glance.preferences.hudClickThrough ? .on : .off
        }
        menu.items.first { $0.action == #selector(toggleMenuBarSpeed(_:)) }?.state =
            glance.preferences.menuBarSpeedEnabled ? .on : .off
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
            cancelLoadedModelRefresh()
        }
        rebuildMenu()
    }

    public func menuWillOpen(_ menu: NSMenu) {
        menu.items.first(where: { $0.action == #selector(checkForUpdates(_:)) })?.isEnabled =
            updateManager.canCheckForUpdates
        updateGlanceToggleStates(in: menu)
        if let card = menu.items.first?.view {
            // The card may have changed shape since the menu was built (for example the stream
            // went live after a rebuild while reconnecting) while it was not in a window.
            RuntimeStatusCardView.refit(card)
        }
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
            language: config.language,
            starterDownload: chatEntry?.menuStarterState ?? .idle,
            starterModelName: chatEntry?.starterModelName,
            hudVisible: glance?.preferences.hudVisible ?? false,
            hudClickThrough: glance?.preferences.hudClickThrough ?? false,
            menuBarSpeedEnabled: glance?.preferences.menuBarSpeedEnabled ?? true,
            hudHotKeyDisplay: glance?.hudHotKeyDisplay
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
        guard Self.shouldRefreshLoadedModelNames(in: state),
              !isRefreshingLoadedModelNames
        else {
            return
        }
        isRefreshingLoadedModelNames = true
        loadedModelRefreshGeneration &+= 1
        let refreshGeneration = loadedModelRefreshGeneration
        let launchID = backend.currentLaunchID
        let config = configStore.load()

        loadedModelRefreshTask = Task { [weak self] in
            guard let self else {
                return
            }
            defer {
                self.finishLoadedModelRefresh(generation: refreshGeneration)
            }
            let models: [BackendLoadedModelInfo]?
            let backendModelListIsAuthoritative: Bool
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                models = try await client.fetchLoadedModels()
                backendModelListIsAuthoritative = true
            } catch let adminError {
                do {
                    let client = BackendAPIClient(host: config.host, port: config.port)
                    let health = try await client.fetchHealthz()
                    guard health.dflash2?.enabled == true else {
                        throw adminError
                    }
                    models = LegacyHealthAdapter().legacyStatus(from: health).runtimeModels
                    backendModelListIsAuthoritative = false
                } catch {
                    guard !Task.isCancelled,
                          refreshGeneration == self.loadedModelRefreshGeneration,
                          Self.shouldRefreshLoadedModelNames(in: self.backend.state)
                    else {
                        return
                    }
                    IronMLXAppLogger.error("Failed to refresh menu loaded models: \(error)")
                    models = nil
                    backendModelListIsAuthoritative = false
                }
            }

            guard !Task.isCancelled,
                  refreshGeneration == loadedModelRefreshGeneration,
                  backend.currentLaunchID == launchID,
                  Self.shouldRefreshLoadedModelNames(in: backend.state),
                  let models
            else {
                return
            }
            if backendModelListIsAuthoritative {
                persistLoadedModelsIfNeeded(models, queriedConfig: config)
            }
            let names = MenuBarMenuBuilder.modelNames(from: models)
            guard names != loadedModelNames else {
                return
            }
            loadedModelNames = names
            rebuildMenu()
        }
    }

    nonisolated static func shouldRefreshLoadedModelNames(in state: BackendProcessState) -> Bool {
        state == .running || state == .degraded
    }

    private func cancelLoadedModelRefresh() {
        loadedModelRefreshGeneration &+= 1
        loadedModelRefreshTask?.cancel()
        loadedModelRefreshTask = nil
        isRefreshingLoadedModelNames = false
    }

    private func finishLoadedModelRefresh(generation: UInt64) {
        guard generation == loadedModelRefreshGeneration else {
            return
        }
        loadedModelRefreshTask = nil
        isRefreshingLoadedModelNames = false
    }

    // A pending DFlash2 restart is still answered by the old daemon, whose list
    // must not erase the startup intent that the restart is about to read.
    static func backendModelListCanReplacePersistedState(
        queriedConfig: AppConfig,
        currentConfig: AppConfig,
        parameterStore: ModelParameterStore
    ) -> Bool {
        queriedConfig.hasSameModelIntent(as: currentConfig)
            && currentConfig.configuredDFlash2Target(parameterStore: parameterStore) == nil
    }

    private func persistLoadedModelsIfNeeded(
        _ models: [BackendLoadedModelInfo],
        queriedConfig: AppConfig
    ) {
        let loaded = AppConfig.normalizedModelReferences(models.map(\.id))
        let pinned = AppConfig.normalizedModelReferences(models.filter(\.pinned).map(\.id))
        let backendDefault = AppConfig.normalizedModelReference(models.first(where: \.isDefault)?.id)
        let config = configStore.load()
        guard Self.backendModelListCanReplacePersistedState(
            queriedConfig: queriedConfig,
            currentConfig: config,
            parameterStore: parameterStore
        ) else {
            return
        }
        parameterStore.clearDFlash2Setting(forOrdinaryModels: models)
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
        cancelLoadedModelRefresh()

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
