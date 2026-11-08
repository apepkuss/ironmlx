import AppKit
import UniformTypeIdentifiers
import WebKit

/// A message chat.html posts with
/// `window.webkit.messageHandlers.chat.postMessage({type, ...})`. Anything
/// else, including unknown types and malformed fields, is rejected.
enum ChatPageMessage: Equatable {
    case ready
    /// `images` are `data:image/...;base64,` URLs read in the page. `retry`
    /// asks again for the reply to the same message after an error.
    case send(text: String, images: [String], retry: Bool)
    case stop
    case newConversation
    case openConversation(id: UUID)
    case switchModel(id: String)
    case setReasoning(ReasoningChoice)
    case openAgents
    case downloadUpgrade
    case dismissUpgrade
    /// Loads a model again after a failed load or switch (`open(modelID:needsLoad:)`).
    case reloadModel(id: String)
    /// The Dashboard's model page, for DFlash2 models and model management.
    case openDashboardModels

    static let maxImages = 4
    /// A data URL of a 20 MB image, the page's own limit, plus base64 overhead.
    static let maxImageURLLength = 28 * 1024 * 1024
    static let maxTextLength = 200_000

    init?(body: Any) {
        guard let body = body as? [String: Any], let type = body["type"] as? String else { return nil }
        switch type {
        case "ready": self = .ready
        case "stop": self = .stop
        case "newConversation": self = .newConversation
        case "openAgents": self = .openAgents
        case "downloadUpgrade": self = .downloadUpgrade
        case "dismissUpgrade": self = .dismissUpgrade
        case "openDashboardModels": self = .openDashboardModels
        case "send":
            guard let text = body["text"] as? String, text.count <= Self.maxTextLength else { return nil }
            let images = body["images"] as? [Any] ?? []
            guard images.count <= Self.maxImages else { return nil }
            var urls: [String] = []
            for image in images {
                guard let url = image as? String, url.utf8.count <= Self.maxImageURLLength,
                      url.prefixMatch(of: /^data:image\/[a-z0-9.+-]+;base64,/) != nil
                else { return nil }
                urls.append(url)
            }
            self = .send(text: text, images: urls, retry: body["retry"] as? Bool ?? false)
        case "openConversation":
            guard let id = (body["id"] as? String).flatMap(UUID.init(uuidString:)) else { return nil }
            self = .openConversation(id: id)
        case "switchModel", "reloadModel":
            guard let id = body["id"] as? String, !id.isEmpty, id.count < 1024 else { return nil }
            self = type == "switchModel" ? .switchModel(id: id) : .reloadModel(id: id)
        case "setReasoning":
            guard let choice = (body["choice"] as? String).flatMap(ReasoningChoice.init(rawValue:)) else { return nil }
            self = .setReasoning(choice)
        default:
            return nil
        }
    }
}

/// The page side of the chat window: receives page messages, sends
/// `window.ironChat.receive(...)` calls (queued until the page says it is
/// ready), keeps every navigation inside the bundled page and opens links in
/// the user's browser.
@MainActor
final class ChatBridge: NSObject, WKScriptMessageHandler, WKNavigationDelegate, WKUIDelegate {
    static let handlerName = "chat"

    weak var webView: WKWebView?
    var onMessage: (ChatPageMessage) -> Void = { _ in }
    /// Called after the web content process restarted and the page reloaded.
    var onPageReloaded: () -> Void = {}
    var openExternal: (URL) -> Void = { NSWorkspace.shared.open($0) }
    /// A `send` failed validation; the page is waiting for its reply and
    /// must be told, or it would show "generating" forever.
    var onRefusedSend: () -> Void = {}
    /// Every message posted to the page (for tests).
    var didPost: ([String: Any]) -> Void = { _ in }
    private(set) var isPageReady = false
    private var queue: [String] = []
    private let pageURL: URL

    init(pageURL: URL) {
        self.pageURL = pageURL
        super.init()
    }

    // MARK: Page → Swift

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == Self.handlerName, message.frameInfo.isMainFrame else { return }
        receive(message.body)
    }

    /// Returns whether the message was accepted.
    @discardableResult
    func receive(_ body: Any) -> Bool {
        guard let message = ChatPageMessage(body: body) else {
            let type = (body as? [String: Any])?["type"] as? String ?? "?"
            IronMLXAppLogger.error("Chat page sent an unknown or malformed message: \(type.prefix(64))")
            if type == "send" { onRefusedSend() }
            return false
        }
        if message == .ready { pageBecameReady() }
        onMessage(message)
        return true
    }

    // MARK: Swift → page

    func post(_ payload: [String: Any]) {
        didPost(payload)
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8)
        else {
            IronMLXAppLogger.error("Chat could not encode a page message: \(payload["type"] ?? "?")")
            return
        }
        let script = "window.ironChat && window.ironChat.receive(\(json));"
        if isPageReady {
            evaluate(script)
        } else {
            queue.append(script)
        }
    }

    private func pageBecameReady() {
        isPageReady = true
        let pending = queue
        queue.removeAll()
        pending.forEach(evaluate)
    }

    private func evaluate(_ script: String) {
        webView?.evaluateJavaScript(script) { _, error in
            if let error {
                IronMLXAppLogger.error("Chat page script error: \(error.localizedDescription)")
            }
        }
    }

    // MARK: Navigation

    enum NavigationDecision: Equatable {
        case allow
        case openExternally(URL)
        case cancel
    }

    /// Only the bundled page itself loads in the web view. Web and mail
    /// links open in the user's browser; everything else is refused.
    static func decision(for url: URL?, pageURL: URL) -> NavigationDecision {
        guard let url else { return .cancel }
        if url.isFileURL {
            return url.standardizedFileURL.path == pageURL.standardizedFileURL.path ? .allow : .cancel
        }
        if url.scheme == "about" && url.absoluteString == "about:blank" { return .allow }
        if ["http", "https", "mailto"].contains(url.scheme?.lowercased() ?? "") { return .openExternally(url) }
        return .cancel
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
    ) {
        switch Self.decision(for: navigationAction.request.url, pageURL: pageURL) {
        case .allow:
            decisionHandler(navigationAction.targetFrame?.isMainFrame == false ? .cancel : .allow)
        case let .openExternally(url):
            decisionHandler(.cancel)
            if navigationAction.navigationType == .linkActivated { openExternal(url) }
        case .cancel:
            decisionHandler(.cancel)
        }
    }

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        if case let .openExternally(url) = Self.decision(for: navigationAction.request.url, pageURL: pageURL) {
            openExternal(url)
        }
        return nil
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        IronMLXAppLogger.error("Chat page content process ended; reloading")
        isPageReady = false
        queue.removeAll()
        webView.loadFileURL(pageURL, allowingReadAccessTo: pageURL.deletingLastPathComponent())
        onPageReloaded()
    }

    // MARK: Image picker ("+")

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image]
        if let window = webView.window {
            panel.beginSheetModal(for: window) { response in
                completionHandler(response == .OK ? panel.urls : nil)
            }
        } else {
            completionHandler(panel.runModal() == .OK ? panel.urls : nil)
        }
    }
}
