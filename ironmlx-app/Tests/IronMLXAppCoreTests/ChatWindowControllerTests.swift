import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

private struct Unavailable: Error {}

/// A backend with one loaded chat model; nothing else is called.
private actor OneModelManager: BackendModelManaging {
    let model = BackendLoadedModelInfo(
        id: "mlx-community/Qwen3.5-4B-MLX-4bit", model: "mlx-community/Qwen3.5-4B-MLX-4bit", path: "/models/q",
        architecture: "qwen3_5", isDefault: true, maxPositionEmbeddings: 32_768, pinned: false, mtpEnabled: false,
        mtpModelDir: nil, mtpDraftTokens: nil,
        capabilities: BackendModelCapabilities(
            runtimeKind: "causal", supportsStreaming: true, supportsVision: true, supportsMtp: false,
            supportsPromptLookup: true, supportsSpeculativeDecoding: false, supportsKvCache: true,
            supportedSamplingParameters: []
        ),
        runtimeState: "loaded",
        reasoning: BackendModelReasoning(nativeEfforts: [])
    )

    func fetchHealthz() async throws -> HealthzSnapshot { throw Unavailable() }
    func loadModel(
        model _: String, modelDir _: String, setDefault _: Bool, maxCacheCap _: Int?, pinned _: Bool,
        promptLookup _: BackendPromptLookupConfig?, audio _: BackendAudioResources?, decision _: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func unloadModel(model _: String, modelDir _: String?) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func setDefaultModel(_: String) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func pinModel(model _: String) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func unpinModel(model _: String) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] { [model] }
    func clearSharedPromptLookup(model _: String?) async throws -> BackendPromptLookupClearResponse { throw Unavailable() }
}

/// A backend whose one chat model stays "loading" until `finish()`.
private actor LoadingModelManager: BackendModelManaging {
    private var state = "loading"
    private let base = OneModelManager()

    func finish() { state = "loaded" }
    /// Unloaded elsewhere, as by the Dashboard.
    func unload() { state = "gone" }

    func fetchHealthz() async throws -> HealthzSnapshot { throw Unavailable() }
    func loadModel(
        model _: String, modelDir _: String, setDefault _: Bool, maxCacheCap _: Int?, pinned _: Bool,
        promptLookup _: BackendPromptLookupConfig?, audio _: BackendAudioResources?, decision _: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func unloadModel(model _: String, modelDir _: String?) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func setDefaultModel(_: String) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func pinModel(model _: String) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func unpinModel(model _: String) async throws -> BackendModelAdminResponse { throw Unavailable() }
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        guard state != "gone" else { return [] }
        var model = base.model
        model.runtimeState = state
        return [model]
    }
    func clearSharedPromptLookup(model _: String?) async throws -> BackendPromptLookupClearResponse { throw Unavailable() }
}

private struct NoTransport: ChatResponsesTransport {
    func stream(body _: Data) async throws -> AsyncThrowingStream<Data, Error> { throw Unavailable() }
}

private struct NoLoader: ChatModelLoading {
    func configuredModel(_: String) async throws -> BackendRecoveryModel { throw Unavailable() }
    func load(_: BackendRecoveryModel, setDefault _: Bool) async throws -> BackendModelAdminResponse { throw Unavailable() }
}

@MainActor
private final class Fixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("chat-window-\(UUID().uuidString)")
    var presented: [NSWindow] = []
    var routes: [DashboardInitialRoute] = []
    var agentGuides: [AgentGuideID?] = []
    let controller: ChatWindowController

    init(models: [ChatModelOption] = [], manager: any BackendModelManaging = OneModelManager()) {
        let session = ChatSessionController(
            store: ChatStore(root: root),
            modelManager: manager,
            transport: NoTransport(),
            loader: NoLoader(),
            warmsUp: false,
            pollInterval: .milliseconds(1)
        )
        var presentedRef: ((NSWindow) -> Void)?
        var routeRef: ((DashboardInitialRoute, AgentGuideID?) -> Void)?
        controller = ChatWindowController(session: session, environment: ChatWindowEnvironment(
            language: { "zh-Hans" },
            theme: { nil },
            models: { models },
            openDashboard: { routeRef?($0, $1) },
            startUpgrade: { _, _ in },
            present: { presentedRef?($0) },
            pageURL: IronMLXAppResourceResolver.url(forResource: "chat", withExtension: "html", subdirectory: "chat")
        ))
        presentedRef = { [unowned self] in presented.append($0) }
        routeRef = { [unowned self] in
            routes.append($0)
            agentGuides.append($1)
        }
    }

    deinit {
        try? FileManager.default.removeItem(at: root)
    }

    func waitUntil(_ condition: () -> Bool, seconds: Double = 15) async throws {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !condition() {
            guard ContinuousClock.now < deadline else { throw CancellationError() }
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

@MainActor
@Suite("Chat window", .serialized)
struct ChatWindowControllerTests {
    @Test func windowIsTransparentTitledFullSize() throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        #expect(window.titlebarAppearsTransparent)
        #expect(window.titleVisibility == .hidden)
        #expect(window.styleMask.contains(.titled))
        #expect(window.styleMask.contains(.fullSizeContentView))
        #expect(window.styleMask.contains(.resizable))
        #expect(window.frameAutosaveName == ChatWindowController.frameAutosaveName)
        let webView = try #require(fixture.controller.webView)
        #expect(webView.value(forKey: "drawsBackground") as? Bool == false)
        let glass = try #require(window.contentView as? NSVisualEffectView)
        #expect(webView.superview === glass)
        #expect(glass.blendingMode == .behindWindow)
        #expect(webView.navigationDelegate === fixture.controller.bridge)
    }

    /// A click on the inactive chat window acts at once (Stop, Think, banner
    /// buttons) instead of only activating the window.
    @Test func chatPageAcceptsTheFirstClick() throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "a", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        let webView = try #require(fixture.controller.webView)
        #expect(webView is ChatWebView)
        #expect(webView.acceptsFirstMouse(for: nil))
    }

    @Test func singleWindowReused() throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "a", needsLoad: false)
        let first = try #require(fixture.controller.window)
        defer { first.close() }
        fixture.controller.showChat(modelID: "a", needsLoad: true)
        #expect(fixture.controller.window === first)
        #expect(fixture.presented.count == 2)
        #expect(fixture.presented.allSatisfy { $0 === first })
    }

    @Test func bridgeRejectsUnknownMessageTypes() {
        let bridge = ChatBridge(pageURL: URL(fileURLWithPath: "/tmp/chat/chat.html"))
        var accepted: [ChatPageMessage] = []
        bridge.onMessage = { accepted.append($0) }

        #expect(!bridge.receive(["type": "evaluate", "code": "x"]))
        #expect(!bridge.receive("send"))
        #expect(!bridge.receive(["text": "hi"]))
        #expect(!bridge.receive(["type": "send"]))
        #expect(!bridge.receive(["type": "send", "text": "hi", "images": ["https://example.com/a.png"]]))
        #expect(!bridge.receive(["type": "send", "text": "hi", "images": Array(repeating: "data:image/png;base64,AA", count: 5)]))
        #expect(!bridge.receive(["type": "openConversation", "id": "../../etc"]))
        #expect(!bridge.receive(["type": "setReasoning", "choice": "max"]))
        #expect(!bridge.receive(["type": "switchModel", "id": ""]))
        #expect(accepted.isEmpty)

        let id = UUID()
        #expect(bridge.receive(["type": "send", "text": "你好", "images": ["data:image/png;base64,AAAA"], "retry": true]))
        #expect(bridge.receive(["type": "openConversation", "id": id.uuidString]))
        #expect(bridge.receive(["type": "setReasoning", "choice": "medium"]))
        #expect(bridge.receive(["type": "openAgents"]))
        #expect(accepted == [
            .send(text: "你好", images: ["data:image/png;base64,AAAA"], retry: true),
            .openConversation(id: id),
            .setReasoning(.medium),
            .openAgents,
        ])
    }

    @Test func navigationStaysOnThePageAndLinksOpenOutside() {
        let page = URL(fileURLWithPath: "/Applications/IronMLX.app/Contents/Resources/chat/chat.html")
        #expect(ChatBridge.decision(for: page, pageURL: page) == .allow)
        #expect(ChatBridge.decision(for: URL(fileURLWithPath: "/etc/passwd"), pageURL: page) == .cancel)
        let link = URL(string: "https://example.com/docs")!
        #expect(ChatBridge.decision(for: link, pageURL: page) == .openExternally(link))
        #expect(ChatBridge.decision(for: URL(string: "javascript:alert(1)"), pageURL: page) == .cancel)
        #expect(ChatBridge.decision(for: URL(string: "ironmlx://x"), pageURL: page) == .cancel)
    }

    @Test func pageImagesBecomeFilesForSending() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("chat-images-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let png = Data([0x89, 0x50, 0x4E, 0x47])
        let paths = try ChatWindowController.writeImages(["data:image/png;base64," + png.base64EncodedString()], to: folder)
        #expect(paths.count == 1)
        #expect(paths[0].hasSuffix(".png"))
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths[0])) == png)
        #expect(throws: (any Error).self) { try ChatWindowController.writeImages(["data:image/png;base64,"], to: folder) }
    }

    @Test func errorsReachThePageAsKinds() {
        let name: (String) -> String = { "name:\($0)" }
        let dflash = ChatWindowController.errorPayload(.modelRequiresDashboard(modelID: "q"), name: name)
        #expect(dflash["kind"] as? String == "modelRequiresDashboard")
        #expect(dflash["modelName"] as? String == "name:q")
        let lost = ChatWindowController.errorPayload(
            .switchLeftNoModel(modelID: "b", previousModelID: "a", message: "oom"), name: name
        )
        #expect(lost["kind"] as? String == "noModel")
        #expect(lost["previousModelID"] as? String == "a")
        #expect(ChatWindowController.errorPayload(.connection(message: "x"), name: name)["kind"] as? String == "connection")
        let pinned = ChatWindowController.errorPayload(
            .switchBlockedByPinnedModel(modelID: "b", pinnedModelID: "a", message: "oom"), name: name
        )
        #expect(pinned["kind"] as? String == "modelLoadFailed")
        #expect(pinned["modelName"] as? String == "name:b")
        #expect(pinned["unpinHint"] as? Bool == true)
        #expect(pinned["pinnedModelName"] as? String == "name:a")
        let turn = ChatWindowController.turnFinishedPayload(ChatTurn(role: .assistant, text: "t", truncated: true))
        #expect(turn["truncated"] as? Bool == true)
        #expect(turn["type"] as? String == "turnFinished")
    }

    /// The real page in WebKit: DOMPurify runs, model HTML never becomes
    /// tags or handlers, and the page reports ready and receives state.
    @Test func pageSanitizesWithDOMPurifyInWebKit() async throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        let bridge = try #require(fixture.controller.bridge)
        let webView = try #require(fixture.controller.webView)
        try await fixture.waitUntil { bridge.isPageReady }
        try await fixture.waitUntil { fixture.controller.session.modelState != .idle }

        let check = """
        (() => {
          const box = document.createElement('div');
          box.innerHTML = window.ironChat.renderMarkdown('<img src=x onerror=alert(1)> <script>alert(2)<\\/script> [a](javascript:alert(3)) **ok**', true);
          const attrs = [...box.querySelectorAll('*')].flatMap(el => [...el.attributes].map(a => a.name + '=' + a.value));
          return JSON.stringify({
            supported: DOMPurify.isSupported,
            tags: [...box.querySelectorAll('*')].map(el => el.tagName.toLowerCase()),
            handlers: attrs.filter(a => /^on/i.test(a) || /javascript:/i.test(a)),
            direct: DOMPurify.sanitize('<img src=x onerror=alert(1)>')
          });
        })()
        """
        let raw = try #require(try await webView.evaluateJavaScript(check) as? String)
        let result = try #require(try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any])
        #expect(result["supported"] as? Bool == true)
        #expect((result["handlers"] as? [String])?.isEmpty == true)
        let tags = result["tags"] as? [String] ?? []
        #expect(!tags.contains("img") && !tags.contains("script"))
        #expect(tags.contains("strong"))
        #expect(result["direct"] as? String == "<img src=\"x\">")
    }

    /// The real page in WebKit: TeX becomes KaTeX markup that DOMPurify keeps,
    /// and the bundled fonts load under the page's CSP.
    @Test func pageRendersMathWithBundledFonts() async throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        let bridge = try #require(fixture.controller.bridge)
        let webView = try #require(fixture.controller.webView)
        try await fixture.waitUntil { bridge.isPageReady }

        let check = """
        const box = document.createElement('div');
        box.className = 'atext';
        box.innerHTML = window.ironChat.renderMarkdown('$\\\\frac{1}{6}$ and $$\\\\sqrt{x}$$ and $\\\\href{javascript:alert(1)}{x}$', true);
        document.body.appendChild(box);
        const loaded = await document.fonts.load('16px KaTeX_Main');
        const result = {
          inline: box.querySelectorAll('.katex').length,
          display: box.querySelectorAll('.katex-display').length,
          svg: box.querySelectorAll('svg path').length,
          links: box.querySelectorAll('a, [href]').length,
          fonts: loaded.map(face => face.family + ':' + face.status)
        };
        box.remove();
        return JSON.stringify(result);
        """
        let raw = try #require(try await webView.callAsyncJavaScript(check, contentWorld: .page) as? String)
        let result = try #require(try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any])
        #expect(result["inline"] as? Int == 3)
        #expect(result["display"] as? Int == 1)
        #expect((result["svg"] as? Int ?? 0) > 0, "the square root's SVG survives DOMPurify")
        #expect(result["links"] as? Int == 0)
        #expect(result["fonts"] as? [String] == ["KaTeX_Main:loaded"])
    }

    @Test func historyAgentsLinkOpensTheAgentPage() async throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "a", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        try await fixture.waitUntil { fixture.controller.bridge?.isPageReady == true }
        fixture.controller.bridge?.receive(["type": "openAgents"])
        fixture.controller.bridge?.receive(["type": "openDashboardModels"])
        #expect(fixture.routes == [.agents, .modelsManage])
        // The link names DeepSeek Harness, so its guide opens whatever is installed.
        #expect(fixture.agentGuides == [.dsh, nil])
    }

    /// A send the bridge refuses, or one the session drops without a reply
    /// or an error, still answers the page, so it never waits forever.
    @Test func refusedOrUnansweredSendResetsThePage() async throws {
        let fixture = Fixture()
        fixture.controller.showChat(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        let bridge = try #require(fixture.controller.bridge)
        try await fixture.waitUntil { bridge.isPageReady }
        try await fixture.waitUntil { if case .ready = fixture.controller.session.modelState { true } else { false } }
        var errors: [String] = []
        bridge.didPost = { payload in
            if payload["type"] as? String == "error", let kind = payload["kind"] as? String { errors.append(kind) }
        }

        // An image the page had not finished reading arrives as null.
        #expect(!bridge.receive(["type": "send", "text": "hi", "images": [NSNull()]]))
        #expect(errors == ["rejected"])
        // Blank text: `ChatSessionController.send` returns without a callback.
        #expect(bridge.receive(["type": "send", "text": "  ", "images": [String]()]))
        try await fixture.waitUntil { errors.count == 2 }
        #expect(errors == ["rejected", "rejected"])
        // Other refused messages need no answer.
        #expect(!bridge.receive(["type": "nope"]))
        #expect(errors.count == 2)
    }

    @Test func modelSwitchesOnlyToOfferedModels() async throws {
        let offered = ChatModelOption(id: "mlx-community/Qwen3.6-35B-A3B-4bit", name: "Qwen 3.6", sizeBytes: 1, loaded: false, pinned: false)
        let fixture = Fixture(models: [offered])
        fixture.controller.showChat(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        try await fixture.waitUntil { if case .ready = fixture.controller.session.modelState { true } else { false } }
        try await fixture.waitUntil { fixture.controller.acceptsModelID(offered.id) }
        #expect(fixture.controller.acceptsModelID("mlx-community/Qwen3.5-4B-MLX-4bit"))
        #expect(!fixture.controller.acceptsModelID("/Users/someone/elsewhere"))
        #expect(!fixture.controller.acceptsModelID("mlx-community/not-offered"))

        // A refused id never reaches the session: the current model keeps serving.
        let before = fixture.controller.session.modelState
        fixture.controller.bridge?.receive(["type": "switchModel", "id": "/Users/someone/elsewhere"])
        fixture.controller.bridge?.receive(["type": "reloadModel", "id": "mlx-community/not-offered"])
        try await Task.sleep(for: .milliseconds(50))
        #expect(fixture.controller.session.modelState == before)
    }

    @Test func reopeningAHiddenWindowDuringALoadOnlyShowsIt() async throws {
        let manager = LoadingModelManager()
        let fixture = Fixture(manager: manager)
        let first = "mlx-community/Qwen3.5-4B-MLX-4bit"
        fixture.controller.showChat(modelID: first, needsLoad: true)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        try await fixture.waitUntil { fixture.controller.session.modelState == .loading(modelID: first) }
        window.close()

        // A switch or load in flight is left alone: no second `open`.
        fixture.controller.showChat(modelID: "mlx-community/other", needsLoad: true)
        #expect(fixture.presented.count == 2)
        try await Task.sleep(for: .milliseconds(50))
        #expect(fixture.controller.session.modelState == .loading(modelID: first))

        await manager.finish()
        try await fixture.waitUntil { if case .ready = fixture.controller.session.modelState { true } else { false } }
    }

    @Test func modelUnloadedElsewhereShowsTheLoadAction() async throws {
        let manager = LoadingModelManager()
        await manager.finish()
        let fixture = Fixture(manager: manager)
        fixture.controller.showChat(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit", needsLoad: false)
        let window = try #require(fixture.controller.window)
        defer { window.close() }
        try await fixture.waitUntil { if case .ready = fixture.controller.session.modelState { true } else { false } }

        await manager.unload()
        NotificationCenter.default.post(name: .ironMLXLoadedModelsDidChange, object: nil)

        try await fixture.waitUntil { if case .failed = fixture.controller.session.modelState { true } else { false } }
        let model = try #require(fixture.controller.statePayload()["model"] as? [String: Any])
        #expect(model["status"] as? String == "failed")
        #expect(model["reason"] as? String == "unloaded")
        #expect(fixture.controller.acceptsModelID("mlx-community/Qwen3.5-4B-MLX-4bit"))
    }
}
