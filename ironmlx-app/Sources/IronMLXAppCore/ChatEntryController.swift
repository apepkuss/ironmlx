import AppKit
import Foundation

/// The App's ways into chat, used by the menu bar and the Dashboard.
@MainActor
protocol ChatEntryHandling: AnyObject {
    /// The starter download as the menu row shows it; `.ready` lasts until
    /// the user opens chat.
    var menuStarterState: StarterDownloadState { get }
    var starterModelName: String? { get }
    func openChat()
    func startStarterDownload()
}

/// Routes "Chat" to the chat window or the no-model window, drives the
/// one-click starter download and announces when its model is ready.
@MainActor
final class ChatEntryController: ChatEntryHandling {
    /// Scans downloaded models; the argument is the loaded model IDs.
    typealias LocalModelsProvider = @Sendable (Set<String>) async -> [LocalModel]
    typealias LoadedModelsProvider = @MainActor () async throws -> [BackendLoadedModelInfo]

    private let configStore: AppConfigStore
    private let backend: any BackendRuntimeManaging
    private let downloadService: ModelDownloadService
    private let starterDownloads: StarterDownloadCoordinator
    private let catalog: SupportedModelCatalog?
    private let recommendation: StarterRecommendation?
    private let notifier: ModelReadyNotifier
    private let presenter: () -> (any ChatWindowPresenting)?
    private let showDashboard: (DashboardInitialRoute) -> Void
    private let window: NoModelWindowController
    weak var menu: MenuBarController?
    private(set) var starterState: StarterDownloadState = .idle
    /// The user opened chat after the starter ended: the menu's "ready" or
    /// "failed" row has served its purpose.
    private var outcomeAcknowledged = false
    /// The upgrade chat asked for. Its download reports `.ready` without
    /// loading the model; chat switches to it, so it is never announced here.
    private var upgrade: (recommendation: StarterRecommendation, onState: @MainActor (StarterDownloadState) -> Void)?
    private var openChatTask: Task<Void, Never>?
    private var backendStartTask: Task<Void, Never>?
    private let localModelsProvider: LocalModelsProvider
    private let loadedModelsProvider: LoadedModelsProvider

    init(
        configStore: AppConfigStore,
        backend: any BackendRuntimeManaging,
        scanner: LocalModelScanner,
        downloadService: ModelDownloadService,
        starterDownloads: StarterDownloadCoordinator,
        notifier: ModelReadyNotifier,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        presenter: @escaping () -> (any ChatWindowPresenting)?,
        showDashboard: @escaping (DashboardInitialRoute) -> Void,
        localModelsProvider: LocalModelsProvider? = nil,
        loadedModelsProvider: LoadedModelsProvider? = nil,
        activation: AppWindowActivation? = nil
    ) {
        self.localModelsProvider = localModelsProvider ?? { loadedIDs in
            await Task.detached { scanner.scan(loadedModels: loadedIDs) }.value
        }
        self.loadedModelsProvider = loadedModelsProvider ?? {
            let config = configStore.load()
            let client = BackendAPIClient(host: config.host, port: config.port)
            try await client.waitUntilReady()
            return try await client.fetchLoadedModels()
        }
        self.configStore = configStore
        self.backend = backend
        self.downloadService = downloadService
        self.starterDownloads = starterDownloads
        self.notifier = notifier
        self.presenter = presenter
        self.showDashboard = showDashboard
        let catalog = try? SupportedModelCatalog.bundled()
        self.catalog = catalog
        recommendation = catalog.flatMap {
            StarterModelSelector.initial(catalog: $0, physicalMemoryBytes: physicalMemoryBytes)
        }
        window = NoModelWindowController(model: NoModelWindowModel(
            recommendation: recommendation,
            physicalMemoryBytes: physicalMemoryBytes,
            language: configStore.load().language
        ), activation: activation)
        window.model.onDownload = { [weak self] in self?.startStarterDownload() }
        window.model.onChooseModel = { [weak self] in
            // The Dashboard comes up first, so the App never leaves the Dock in between.
            self?.showDashboard(.modelsDownload)
            self?.window.close()
        }
        window.model.onCancelDownload = { [weak self] in self?.cancelStarterDownload() }
        window.onClosedWhileDownloading = { [weak self] in
            guard let notifier = self?.notifier else { return }
            Task { await notifier.userLeftDownloadWindow() }
        }
        notifier.onOpenChat = { [weak self] _ in self?.openChat() }
    }

    /// Subscribes to the coordinator and shows a starter download an earlier
    /// session left unfinished. Call once at launch.
    func start() async {
        await starterDownloads.setStateObserver { [weak self] state in
            await MainActor.run { self?.starterStateChanged(state) }
        }
        await starterDownloads.setOnReady { [weak self] modelID in
            await self?.persistStarterModel(modelID)
        }
        await restorePausedStarterIfNeeded()
    }

    var menuStarterState: StarterDownloadState {
        switch starterState {
        case .ready, .failed: outcomeAcknowledged ? .idle : starterState
        case .idle, .probing, .downloading, .paused, .loading: starterState
        }
    }

    /// What the no-model window shows.
    var noModelWindowState: StarterDownloadState { window.model.state }

    var starterModelName: String? { upgrade?.recommendation.entry.name ?? recommendation?.entry.name }

    func showNoModelWindow() {
        window.show(language: configStore.load().language)
    }

    var isNoModelWindowVisible: Bool { window.isVisible }

    func closeNoModelWindow() {
        window.close()
    }

    func openChat() {
        if openChatTask != nil {
            // A decision is on its way; bring forward what is already on screen.
            if window.isVisible { showNoModelWindow() }
            return
        }
        openChatTask = Task {
            let decision = await currentDecision()
            openChatTask = nil
            route(decision)
        }
    }

    func startStarterDownload() {
        guard let recommendation else {
            showDashboard(.modelsDownload)
            return
        }
        showNoModelWindow()
        Task {
            await restorePausedStarterIfNeeded()
            let result = await starterDownloads.start(recommendation)
            IronMLXAppLogger.info("Starter download start: \(result)")
        }
    }

    /// Downloads the upgrade chat offered. Progress and the result go to
    /// `onState` only: no no-model window, no "model ready" notification.
    func startUpgradeDownload(
        _ recommendation: StarterRecommendation,
        onState: @escaping @MainActor (StarterDownloadState) -> Void
    ) {
        switch starterState {
        case .probing, .downloading, .paused, .loading:
            // The first starter model is still on its way; its states stay its own.
            onState(.failed(code: "starter_download_busy"))
            return
        case .idle, .ready, .failed:
            break
        }
        upgrade = (recommendation, onState)
        Task {
            let result = await starterDownloads.start(recommendation)
            IronMLXAppLogger.info("Upgrade download start: \(result)")
            switch result {
            case .started:
                break
            case .alreadyInProgress:
                finishUpgrade(with: .failed(code: "starter_download_busy"))
            case let .failed(code):
                finishUpgrade(with: .failed(code: code))
            }
        }
    }

    private func finishUpgrade(with state: StarterDownloadState) {
        guard let upgrade else { return }
        self.upgrade = nil
        upgrade.onState(state)
        menu?.starterDownloadDidChange()
    }

    private func cancelStarterDownload() {
        Task { await starterDownloads.cancel() }
    }

    /// With the backend not running, decides from downloaded models alone and
    /// starts the backend behind the window it shows: nothing can be loaded
    /// yet, so a downloaded model opens chat with `needsLoad`.
    private func currentDecision() async -> ChatEntryDecision {
        let config = configStore.load()
        var loaded: [BackendLoadedModelInfo] = []
        if backend.state == .running || backend.state == .degraded {
            do {
                loaded = try await loadedModelsProvider()
            } catch {
                IronMLXAppLogger.error("Chat entry could not read loaded models: \(error)")
            }
        } else {
            startBackendInBackground()
        }
        let localModels = await localModelsProvider(Set(loaded.map(\.id)))
        return ChatEntryCoordinator.decide(
            loaded: loaded,
            localModels: localModels,
            defaultModel: config.defaultModelReference,
            starterState: starterState,
            isConversationModel: ChatEntryCoordinator.conversationModelPredicate(
                catalog: catalog, localModels: localModels, loaded: loaded
            ),
            starterModelIDs: ChatEntryCoordinator.starterModelIDs(catalog: catalog)
        )
    }

    private func startBackendInBackground() {
        guard backendStartTask == nil else { return }
        let backend = backend
        backendStartTask = Task {
            do {
                try await backend.ensureRunning()
            } catch {
                IronMLXAppLogger.error("Chat entry could not start the backend: \(error)")
            }
            backendStartTask = nil
        }
    }

    private func route(_ decision: ChatEntryDecision) {
        switch decision {
        case let .openChat(modelID, needsLoad):
            acknowledgeOutcome()
            guard let presenter = presenter() else {
                IronMLXAppLogger.info("Chat window is not available yet; showing the Dashboard for \(modelID)")
                showDashboard(.status)
                return
            }
            presenter.showChat(modelID: modelID, needsLoad: needsLoad)
        case .showDownloadProgress:
            showNoModelWindow()
        case .showNoModel:
            // A starter that became ready earlier no longer opens chat (its
            // model is gone); the window offers a download again.
            if case .ready = window.model.state { window.update(.idle) }
            showNoModelWindow()
        }
    }

    private func acknowledgeOutcome() {
        switch starterState {
        case .ready, .failed: break
        case .idle, .probing, .downloading, .paused, .loading: return
        }
        guard !outcomeAcknowledged else { return }
        outcomeAcknowledged = true
        menu?.starterDownloadDidChange()
    }

    private func starterStateChanged(_ state: StarterDownloadState) {
        if let upgrade {
            switch state {
            case .ready, .failed, .idle:
                // Chat switches to the model or shows the failure; the menu
                // shows neither a "ready" nor a "failed" row for it.
                starterState = .idle
                finishUpgrade(with: state)
            case .probing, .downloading, .paused, .loading:
                starterState = state
                upgrade.onState(state)
                menu?.starterDownloadDidChange()
            }
            return
        }
        if state != starterState { outcomeAcknowledged = false }
        starterState = state
        window.update(state)
        menu?.starterDownloadDidChange()
        if case let .ready(modelID) = state {
            announceReady(modelID)
        }
    }

    private func announceReady(_ modelID: String) {
        let windowVisible = window.isVisible
        let name = recommendation?.entry.name ?? modelID
        let language = configStore.load().language
        let notifier = notifier
        Task {
            let action = await notifier.modelReady(
                modelID: modelID, modelName: name, windowVisible: windowVisible, language: language
            )
            if action == .openChat {
                // Chat comes up first, so the App never leaves the Dock in between.
                route(.openChat(modelID: modelID, needsLoad: false))
                window.close()
            }
        }
    }

    private func restorePausedStarterIfNeeded() async {
        guard let recommendation else { return }
        let snapshot = await downloadService.downloadQueueSnapshot()
        guard let source = ChatEntryCoordinator.pausedStarterSource(
            in: snapshot, repoID: recommendation.entry.hfRepo
        ) else { return }
        let bytes = await downloadService.recoverableDownloadBytes(
            provider: .huggingFace, repoID: recommendation.entry.hfRepo, commitSHA: recommendation.entry.metadataRevision
        )
        await starterDownloads.restorePaused(recommendation, source: source, bytes: bytes)
    }

    /// R9: records the loaded starter model as the default, the way a
    /// Dashboard load does, so the next launch restores it. An upgrade does
    /// not load here; chat switches to it.
    private func persistStarterModel(_ modelID: String) async {
        guard recommendation?.entry.hfRepo == modelID else { return }
        let config = configStore.load()
        let client = BackendAPIClient(host: config.host, port: config.port)
        let loaded: [BackendLoadedModelInfo]?
        do {
            loaded = try await client.fetchLoadedModels()
        } catch {
            IronMLXAppLogger.error("Starter model \(modelID) loaded, but the loaded list is unavailable: \(error)")
            loaded = nil
        }
        let saved = configStore.update { ChatEntryCoordinator.persistStarterModel(modelID, loaded: loaded, into: &$0) }
        if !saved {
            IronMLXAppLogger.error("Starter model \(modelID) could not be saved as the default model")
        }
        if let loaded {
            backend.confirmLoadedModels(loaded, parameterConfirmedModelIDs: [modelID])
        }
        NotificationCenter.default.post(name: .ironMLXLoadedModelsDidChange, object: nil)
    }
}
