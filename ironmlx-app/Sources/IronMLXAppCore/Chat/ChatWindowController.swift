import AppKit
import ImageIO
import UniformTypeIdentifiers
import WebKit

/// A downloaded conversation model the model menu offers.
struct ChatModelOption: Equatable, Sendable {
    var id: String
    var name: String
    var sizeBytes: Int64
    var loaded: Bool
    var pinned: Bool
}

/// What the chat window needs from the rest of the App.
@MainActor
struct ChatWindowEnvironment {
    var language: () -> String
    var theme: () -> String?
    /// Downloaded conversation models, with what the backend has loaded.
    var models: () async -> [ChatModelOption]
    var openDashboard: (DashboardInitialRoute) -> Void
    /// Downloads the offered upgrade; states arrive until `.ready`, `.failed` or `.idle`.
    var startUpgrade: (StarterRecommendation, @escaping @MainActor (StarterDownloadState) -> Void) -> Void
    /// Brings the window to the front.
    var present: (NSWindow) -> Void
    var pageURL: URL?
    /// The window closed (`AppWindowActivation`).
    var closed: () -> Void = {}
}

/// The chat window: one native window (transparent title bar, content under
/// it, frosted glass behind a transparent web view) showing `chat.html`, and
/// the glue between the page and `ChatSessionController`. Only one exists.
@MainActor
final class ChatWindowController: NSObject, ChatWindowPresenting, NSWindowDelegate {
    static let frameAutosaveName = "IronMLXChatWindow"
    static let defaultSize = NSSize(width: 760, height: 680)
    static let minimumSize = NSSize(width: 480, height: 440)

    let session: ChatSessionController
    private let environment: ChatWindowEnvironment
    private(set) var window: NSWindow?
    private(set) var webView: WKWebView?
    private(set) var bridge: ChatBridge?
    private var models: [ChatModelOption] = []
    private var requestedModelID: String?
    private var upgradeOffer: StarterRecommendation?
    private var upgradeOfferShown = false
    private var pendingUpgradeSwitch: String?
    private var thumbnails: [String: String] = [:]
    private var observers: [NSObjectProtocol] = []
    private var modelRefresh: Task<Void, Never>?
    /// Counts replies and errors reported to the page, to see whether a send got an answer.
    private var replyEvents = 0

    init(session: ChatSessionController, environment: ChatWindowEnvironment) {
        self.session = session
        self.environment = environment
        super.init()
        wireSession()
        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: .ironMLXMenuLanguageDidChange, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.postState() }
        })
        observers.append(center.addObserver(forName: .ironMLXLoadedModelsDidChange, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.refreshModels()
                let session = self.session
                Task { await session.loadedModelsChanged() }
            }
        })
    }

    isolated deinit {
        observers.forEach(NotificationCenter.default.removeObserver)
    }

    // MARK: ChatWindowPresenting

    func showChat(modelID: String, needsLoad: Bool) {
        let wasVisible = window?.isVisible == true
        let window = window ?? makeWindow()
        applyTheme()
        environment.present(window)
        if let webView { window.makeFirstResponder(webView) }
        refreshModels()
        // A load or switch in flight is never restarted, even when the window
        // was hidden: a second `open` would cancel the switch half way. An
        // open window already serving a model just comes forward.
        switch session.modelState {
        case .loading: return
        case .ready: if wasVisible { return }
        case .idle, .failed, .noModel: break
        }
        requestedModelID = modelID
        postState()
        let session = session
        // Opening also warms the model up again (ChatSessionController).
        Task { await session.open(modelID: modelID, needsLoad: needsLoad) }
    }

    func windowWillClose(_ notification: Notification) {
        session.windowClosed()
        environment.closed()
    }

    // MARK: Window

    private func makeWindow() -> NSWindow {
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: Self.defaultSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "IronMLX"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        // An empty compact toolbar centres the traffic lights in a taller bar.
        window.toolbar = NSToolbar(identifier: "IronMLXChatToolbar")
        window.toolbarStyle = .unifiedCompact
        window.titlebarSeparatorStyle = .none
        window.isReleasedWhenClosed = false
        window.minSize = Self.minimumSize
        window.collectionBehavior = [.fullScreenPrimary]
        window.delegate = self

        let barHeight = max(window.frame.height - window.contentLayoutRect.height, 28)
        let content = ChatContentView(barHeight: barHeight)
        content.material = .underWindowBackground
        content.blendingMode = .behindWindow
        content.state = .followsWindowActiveState
        window.contentView = content

        let pageURL = environment.pageURL
        let bridge = ChatBridge(pageURL: pageURL ?? URL(fileURLWithPath: "/"))
        bridge.onMessage = { [weak self] in self?.handle($0) }
        bridge.onPageReloaded = { [weak self] in self?.thumbnails.removeAll() }
        bridge.onRefusedSend = { [weak self] in self?.postRejectedSend() }

        let configuration = WKWebViewConfiguration()
        let userContent = WKUserContentController()
        userContent.add(bridge, name: ChatBridge.handlerName)
        userContent.addUserScript(WKUserScript(
            source: "document.documentElement.style.setProperty('--bar-h', '\(Int(barHeight.rounded()))px');",
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        ))
        configuration.userContentController = userContent
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        let webView = ChatWebView(frame: .zero, configuration: configuration)
        // Transparent, so the frosted glass behind shows through the page.
        webView.setValue(false, forKey: "drawsBackground")
        webView.underPageBackgroundColor = .clear
        webView.allowsMagnification = false
        webView.allowsBackForwardNavigationGestures = false
        webView.navigationDelegate = bridge
        webView.uiDelegate = bridge
        bridge.webView = webView
        content.webView = webView

        if !window.setFrameUsingName(Self.frameAutosaveName) {
            window.center()
        }
        window.setFrameAutosaveName(Self.frameAutosaveName)

        self.window = window
        self.webView = webView
        self.bridge = bridge
        if let pageURL {
            webView.loadFileURL(pageURL, allowingReadAccessTo: pageURL.deletingLastPathComponent())
        } else {
            IronMLXAppLogger.error("IronMLX App Bundle is missing chat/chat.html")
        }
        return window
    }

    private func applyTheme() {
        guard let view = window?.contentView else { return }
        DashboardThemeAppearance.apply(environment.theme(), to: view)
    }

    // MARK: Session → page

    private func wireSession() {
        // `onDefaultModelChanged` stays as the live factory wired it (persistence).
        session.onDelta = { [weak self] delta in
            self?.bridge?.post(["type": "delta", "reasoning": delta.reasoning, "text": delta.text])
        }
        session.onTurnFinished = { [weak self] turn in
            guard let self else { return }
            replyEvents += 1
            bridge?.post(Self.turnFinishedPayload(turn))
            postHistory()
            switchToUpgradeIfWaiting()
        }
        session.onError = { [weak self] error in
            guard let self else { return }
            replyEvents += 1
            bridge?.post(Self.errorPayload(error, name: displayName))
            switchToUpgradeIfWaiting()
        }
        session.onModelStateChanged = { [weak self] _ in
            self?.postState()
            self?.refreshModels()
        }
        session.onUpgradeOffer = { [weak self] recommendation in
            guard let self else { return }
            upgradeOffer = recommendation
            upgradeOfferShown = true
            postUpgradeOffer()
        }
        session.onHistoryTrimmed = { [weak self] count in
            self?.bridge?.post(["type": "trimmed", "count": count])
        }
    }

    private func postState() {
        bridge?.post(statePayload())
    }

    private func postHistory() {
        let current = session.conversation.id
        let items: [[String: Any]] = session.conversations().map {
            [
                "id": $0.id.uuidString,
                "title": $0.title,
                "updatedAt": ($0.updatedAt.timeIntervalSince1970 * 1000).rounded(),
                "current": $0.id == current,
            ]
        }
        bridge?.post(["type": "history", "items": items])
    }

    private func postConversation() {
        let conversation = session.conversation
        let turns: [[String: Any]] = conversation.turns.map { turn in
            var item = Self.turnPayload(turn)
            item["role"] = turn.role.rawValue
            item["images"] = turn.attachments.compactMap(thumbnail)
            return item
        }
        bridge?.post(["type": "conversation", "id": conversation.id.uuidString, "turns": turns])
    }

    private func postUpgradeOffer() {
        guard let offer = upgradeOffer, upgradeOfferShown else { return }
        bridge?.post([
            "type": "upgradeOffer",
            "name": offer.entry.name,
            "sizeText": Self.sizeText(offer.entry.downloadBytes),
        ])
    }

    func statePayload() -> [String: Any] {
        var model: [String: Any]
        var reasoning: Any = NSNull()
        var vision = false
        var currentID: String?
        switch session.modelState {
        case .idle:
            let id = requestedModelID ?? ""
            model = ["id": id, "name": id.isEmpty ? "" : displayName(id), "status": "idle"]
        case let .loading(id):
            currentID = id
            model = ["id": id, "name": displayName(id), "status": "loading"]
        case let .ready(info):
            currentID = info.modelID
            model = ["id": info.modelID, "name": displayName(info.modelID), "status": "ready"]
            if let capability = info.reasoning { reasoning = ["native_efforts": capability.nativeEfforts] }
            vision = info.supportsVision
        case let .failed(id, message):
            currentID = id
            model = ["id": id, "name": displayName(id), "status": "failed", "message": message]
            if message == ChatSessionController.modelUnloadedMessage { model["reason"] = "unloaded" }
        case let .noModel(failed, previous, message):
            model = [
                "id": failed, "name": displayName(failed), "status": "noModel", "message": message,
                "failedModelID": failed, "failedName": displayName(failed),
                "previousModelID": previous, "previousName": displayName(previous),
            ]
        }
        var options = models
        if let currentID, !options.contains(where: { $0.id == currentID }) {
            options.insert(ChatModelOption(id: currentID, name: displayName(currentID), sizeBytes: 0, loaded: true, pinned: false), at: 0)
        }
        return [
            "type": "state",
            "language": Self.normalizedLanguage(environment.language()),
            "models": options.map {
                [
                    "id": $0.id, "name": $0.name, "sizeBytes": $0.sizeBytes, "loaded": $0.loaded,
                    "pinned": $0.pinned, "current": $0.id == currentID,
                ] as [String: Any]
            },
            "model": model,
            "reasoning": reasoning,
            "reasoningChoice": session.reasoningChoice.rawValue,
            "supportsVision": vision,
            "generating": session.isGenerating,
        ]
    }

    private func refreshModels() {
        modelRefresh?.cancel()
        let load = environment.models
        modelRefresh = Task { [weak self] in
            let models = await load()
            guard !Task.isCancelled, let self else { return }
            if models != self.models {
                self.models = models
                postState()
            }
        }
    }

    private func displayName(_ id: String) -> String {
        if let option = models.first(where: { $0.id == id }) { return option.name }
        return URL(fileURLWithPath: id).lastPathComponent
    }

    // MARK: Page → session

    private func handle(_ message: ChatPageMessage) {
        switch message {
        case .ready:
            postState()
            postHistory()
            postConversation()
            postUpgradeOffer()
        case let .send(text, images, retry):
            Task { await send(text: text, images: images, retry: retry) }
        case .stop:
            session.stop()
        case .newConversation:
            session.newConversation()
            postConversation()
            postHistory()
        case let .openConversation(id):
            session.openConversation(id: id)
            postConversation()
            postHistory()
        case let .switchModel(id):
            guard acceptsModelID(id) else {
                IronMLXAppLogger.error("Chat page asked to switch to a model it was not offered")
                return
            }
            let session = session
            // Failures arrive through `onError`.
            Task { try? await session.switchModel(to: id) }
        case let .setReasoning(choice):
            do {
                try session.setReasoningChoice(choice)
            } catch {
                IronMLXAppLogger.error("Chat could not save the thinking setting: \(error)")
            }
        case .openAgents:
            // Task 7 adds an agent preselection; until then the Agent page opens.
            environment.openDashboard(.agents)
        case .downloadUpgrade:
            guard let offer = upgradeOffer else { return }
            environment.startUpgrade(offer) { [weak self] state in self?.upgradeStateChanged(state) }
        case .dismissUpgrade:
            session.dismissUpgradeOffer()
            upgradeOffer = nil
        case let .reloadModel(id):
            guard acceptsModelID(id) else {
                IronMLXAppLogger.error("Chat page asked to load a model it was not offered")
                return
            }
            requestedModelID = id
            let session = session
            Task { await session.open(modelID: id, needsLoad: true) }
        case .openDashboardModels:
            environment.openDashboard(.modelsManage)
        }
    }

    /// Every send ends in `turnFinished` or `error` for the page. The session
    /// returns without either when, for example, Stop came while the images
    /// were prepared; then the page hears that nothing was sent.
    private func send(text: String, images: [String], retry: Bool) async {
        let answered = replyEvents
        await sendToSession(text: text, images: images, retry: retry)
        if replyEvents == answered { postRejectedSend() }
    }

    private func postRejectedSend() {
        bridge?.post(["type": "error", "kind": "rejected"])
    }

    /// The models the page may ask for: those it was offered, and the ones
    /// the current state names (current, requested, failed, previous).
    func acceptsModelID(_ id: String) -> Bool {
        var ids = Set(models.map(\.id))
        if let requestedModelID { ids.insert(requestedModelID) }
        switch session.modelState {
        case .idle: break
        case let .loading(modelID), let .failed(modelID, _): ids.insert(modelID)
        case let .ready(info): ids.insert(info.modelID)
        case let .noModel(failed, previous, _):
            ids.insert(failed)
            ids.insert(previous)
        }
        return ids.contains(id)
    }

    private func sendToSession(text: String, images: [String], retry: Bool) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if retry, let last = session.conversation.turns.last, last.role == .user, last.text == trimmed {
            await session.retry()
            return
        }
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("IronMLXChat-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var paths: [String] = []
        if !images.isEmpty {
            do {
                paths = try Self.writeImages(images, to: folder)
            } catch {
                bridge?.post(Self.errorPayload(.attachmentUnreadable(path: ""), name: displayName))
                replyEvents += 1
                return
            }
        }
        await session.send(text: text, imagePaths: paths)
        postHistory()
    }

    // MARK: Upgrade

    private func upgradeStateChanged(_ state: StarterDownloadState) {
        switch state {
        case .idle:
            bridge?.post(["type": "upgradeStatus", "phase": "done"])
        case .probing:
            bridge?.post(["type": "upgradeStatus", "phase": "preparing"])
        case let .downloading(_, fraction, _, _, _):
            bridge?.post(["type": "upgradeStatus", "phase": "downloading", "fraction": fraction])
        case .paused:
            bridge?.post(["type": "upgradeStatus", "phase": "paused"])
        case .loading:
            bridge?.post(["type": "upgradeStatus", "phase": "switching"])
        case let .ready(modelID):
            // Downloaded, not loaded: chat replaces its model with it.
            bridge?.post(["type": "upgradeStatus", "phase": "switching"])
            pendingUpgradeSwitch = modelID
            switchToUpgradeIfWaiting()
        case .failed:
            bridge?.post(["type": "upgradeStatus", "phase": "failed"])
        }
    }

    /// Switches after the reply in progress, never in the middle of it.
    private func switchToUpgradeIfWaiting() {
        guard let modelID = pendingUpgradeSwitch, !session.isGenerating else { return }
        pendingUpgradeSwitch = nil
        let session = session
        Task { [weak self] in
            try? await session.switchModel(to: modelID)
            self?.upgradeOffer = nil
            self?.bridge?.post(["type": "upgradeStatus", "phase": "done"])
        }
    }

    // MARK: Payloads

    static func normalizedLanguage(_ value: String) -> String {
        switch value.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "zh", "zh-Hans": "zh-Hans"
        case "zh-Hant": "zh-Hant"
        case "ja": "ja"
        case "ko": "ko"
        default: "en"
        }
    }

    static func sizeText(_ bytes: Int64) -> String {
        String(format: "%.1f GB", Double(max(0, bytes)) / 1_000_000_000)
    }

    static func turnPayload(_ turn: ChatTurn) -> [String: Any] {
        var metrics: Any = NSNull()
        if let value = turn.metrics {
            var item: [String: Any] = [:]
            if let first = value.firstTokenSeconds { item["firstTokenSeconds"] = first }
            if let thinking = value.thinkingSeconds { item["thinkingSeconds"] = thinking }
            if let speed = value.tokensPerSecond, speed.isFinite { item["tokensPerSecond"] = speed }
            if let tokens = value.outputTokens { item["outputTokens"] = tokens }
            metrics = item
        }
        return [
            "text": turn.text,
            "reasoning": turn.reasoning.map { $0 as Any } ?? NSNull(),
            "stopped": turn.stopped,
            "truncated": turn.truncated == true,
            "metrics": metrics,
        ]
    }

    static func turnFinishedPayload(_ turn: ChatTurn) -> [String: Any] {
        var payload = turnPayload(turn)
        payload["type"] = "turnFinished"
        return payload
    }

    static func errorPayload(_ error: ChatError, name: (String) -> String) -> [String: Any] {
        var payload: [String: Any] = ["type": "error"]
        switch error {
        case .modelNotReady: payload["kind"] = "modelNotReady"
        case .busy: payload["kind"] = "busy"
        case .imagesNotSupported: payload["kind"] = "imagesNotSupported"
        case .attachmentUnreadable: payload["kind"] = "attachment"
        case .contextOverflow: payload["kind"] = "contextOverflow"
        case let .server(_, message):
            payload["kind"] = "server"
            payload["message"] = message
        case let .connection(message):
            payload["kind"] = "connection"
            payload["message"] = message
        case let .modelLoadFailed(modelID, message):
            payload["kind"] = "modelLoadFailed"
            payload["modelID"] = modelID
            payload["modelName"] = name(modelID)
            payload["message"] = message
        case let .switchLeftNoModel(modelID, previous, message):
            payload["kind"] = "noModel"
            payload["modelID"] = modelID
            payload["modelName"] = name(modelID)
            payload["previousModelID"] = previous
            payload["message"] = message
        case let .modelRequiresDashboard(modelID):
            payload["kind"] = "modelRequiresDashboard"
            payload["modelID"] = modelID
            payload["modelName"] = name(modelID)
        case let .switchBlockedByPinnedModel(modelID, pinnedModelID, message):
            payload["kind"] = "modelLoadFailed"
            payload["modelID"] = modelID
            payload["modelName"] = name(modelID)
            payload["message"] = message
            payload["unpinHint"] = true
            payload["pinnedModelName"] = name(pinnedModelID)
        }
        return payload
    }

    /// Writes the page's `data:image/...;base64,` URLs as files for `send`.
    static func writeImages(_ dataURLs: [String], to folder: URL) throws -> [String] {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return try dataURLs.enumerated().map { index, url in
            guard let comma = url.firstIndex(of: ","),
                  let data = Data(base64Encoded: String(url[url.index(after: comma)...]), options: .ignoreUnknownCharacters),
                  !data.isEmpty
            else { throw CocoaError(.fileReadCorruptFile) }
            let mime = url[url.index(url.startIndex, offsetBy: 5)..<comma].split(separator: ";").first.map(String.init) ?? ""
            let ext = UTType(mimeType: mime)?.preferredFilenameExtension ?? "img"
            let file = folder.appendingPathComponent("image-\(index).\(ext)")
            try data.write(to: file)
            return file.path
        }
    }

    /// A small JPEG data URL of a stored image, for the thread.
    private func thumbnail(_ name: String) -> String? {
        if let cached = thumbnails[name] { return cached }
        let url = session.attachmentURL(for: name)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: 480,
              ] as CFDictionary)
        else { return nil }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        let dataURL = "data:image/jpeg;base64," + (data as Data).base64EncodedString()
        thumbnails[name] = dataURL
        return dataURL
    }
}

extension ChatWindowEnvironment {
    /// The App's wiring: the model list from the local scan and the loaded
    /// list, names from the bundled catalogue, the window brought to front.
    static func live(
        configStore: AppConfigStore,
        scanner: LocalModelScanner,
        openDashboard: @escaping (DashboardInitialRoute) -> Void,
        startUpgrade: @escaping (StarterRecommendation, @escaping @MainActor (StarterDownloadState) -> Void) -> Void
    ) -> ChatWindowEnvironment {
        let catalog = try? SupportedModelCatalog.bundled()
        return ChatWindowEnvironment(
            language: { configStore.load().language },
            theme: { configStore.load().theme },
            models: { await ChatWindowEnvironment.loadModels(configStore: configStore, scanner: scanner, catalog: catalog) },
            openDashboard: openDashboard,
            startUpgrade: startUpgrade,
            present: { window in
                AppWindowActivation.shared.shown(.chat)
                NSApp.activate(ignoringOtherApps: true)
                window.makeKeyAndOrderFront(nil)
            },
            pageURL: IronMLXAppResourceResolver.url(forResource: "chat", withExtension: "html", subdirectory: "chat"),
            closed: { AppWindowActivation.shared.hidden(.chat) }
        )
    }

    nonisolated static func loadModels(
        configStore: AppConfigStore,
        scanner: LocalModelScanner,
        catalog: SupportedModelCatalog?
    ) async -> [ChatModelOption] {
        let config = configStore.load()
        let loaded = (try? await BackendAPIClient(host: config.host, port: config.port).fetchLoadedModels()) ?? []
        let local = await Task.detached { scanner.scan(loadedModels: Set(loaded.map(\.id))) }.value
        let isConversation = ChatEntryCoordinator.conversationModelPredicate(
            catalog: catalog, localModels: local, loaded: loaded
        )
        var names: [String: String] = [:]
        for entry in catalog?.entries ?? [] {
            for key in [entry.hfRepo, entry.msRepo, entry.id].compactMap({ $0 }) where names[key] == nil {
                names[key] = entry.name
            }
        }
        func name(_ id: String, display: String?) -> String {
            names[id] ?? display.flatMap { $0.isEmpty ? nil : $0 } ?? URL(fileURLWithPath: id).lastPathComponent
        }
        var options: [ChatModelOption] = local
            .filter { isConversation($0.id) && $0.readiness?.isLoadable != false }
            .map { model in
                let backend = loaded.first { $0.id == model.id }
                return ChatModelOption(
                    id: model.id,
                    name: name(model.id, display: model.displayName),
                    sizeBytes: Int64(model.sizeMB * 1_000_000),
                    loaded: backend?.runtimeState == "loaded",
                    pinned: backend?.pinned ?? model.pinned
                )
            }
        for model in loaded where isConversation(model.id) && !options.contains(where: { $0.id == model.id }) {
            options.append(ChatModelOption(
                id: model.id, name: name(model.id, display: nil), sizeBytes: 0,
                loaded: model.runtimeState == "loaded", pinned: model.pinned
            ))
        }
        return options
    }
}

/// The chat page acts on the click that also activates its window, so Stop,
/// Think and the banner buttons work on the first click. The Dashboard keeps
/// the default.
final class ChatWebView: WKWebView {
    override func acceptsFirstMouse(for _: NSEvent?) -> Bool { true }
}

/// The window's frosted-glass content: the page fills it, and the empty
/// parts of the top bar drag the window (the web view would swallow those
/// clicks). The model menu in the middle and the buttons on the right stay
/// with the page.
final class ChatContentView: NSVisualEffectView {
    private let barHeight: CGFloat
    private let leftStrip = ChatTitlebarDragView()
    private let rightStrip = ChatTitlebarDragView()
    /// Half the width kept clear around the centred model menu.
    private static let centerClearance: CGFloat = 150
    private static let leadingInset: CGFloat = 76
    private static let trailingInset: CGFloat = 86

    var webView: WKWebView? {
        didSet {
            oldValue?.removeFromSuperview()
            if let webView { addSubview(webView, positioned: .below, relativeTo: leftStrip) }
            needsLayout = true
        }
    }

    init(barHeight: CGFloat) {
        self.barHeight = barHeight
        super.init(frame: .zero)
        addSubview(leftStrip)
        addSubview(rightStrip)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    override var isFlipped: Bool { true }

    override func layout() {
        super.layout()
        webView?.frame = bounds
        let center = bounds.midX
        let left = Self.leadingInset
        let leftEnd = max(left, center - Self.centerClearance)
        let rightStart = center + Self.centerClearance
        let rightEnd = max(rightStart, bounds.width - Self.trailingInset)
        leftStrip.frame = NSRect(x: left, y: 0, width: leftEnd - left, height: barHeight)
        rightStrip.frame = NSRect(x: rightStart, y: 0, width: rightEnd - rightStart, height: barHeight)
    }
}

/// An empty stretch of the title bar: drags the window, and a double click
/// does what the user chose in System Settings (zoom by default).
final class ChatTitlebarDragView: NSView {
    override var mouseDownCanMoveWindow: Bool { true }

    override func mouseDown(with event: NSEvent) {
        guard let window else { return }
        if event.clickCount == 2 {
            switch UserDefaults.standard.string(forKey: "AppleActionOnDoubleClick") {
            case "Minimize": window.performMiniaturize(nil)
            case "None": break
            default: window.performZoom(nil)
            }
            return
        }
        window.performDrag(with: event)
    }
}
