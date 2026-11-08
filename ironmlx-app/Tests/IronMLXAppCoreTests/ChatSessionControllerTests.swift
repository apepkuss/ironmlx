import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import IronMLXAppCore

// MARK: - Fakes

/// Monotonic seconds that the fake stream moves forward as it hands out chunks.
private final class FakeClock: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Double = 0
    var now: Double {
        get { lock.withLock { value } }
        set { lock.withLock { value = newValue } }
    }
}

private struct StreamStep: Sendable {
    var time: Double
    var data: Data
}

private enum Script: Sendable {
    case http(status: Int, body: String)
    case stream([StreamStep], hangAtEnd: Bool = false)
}

private func sse(_ kind: String, _ fields: String) -> Data {
    Data("event: \(kind)\ndata: {\"type\":\"\(kind)\",\"sequence_number\":0,\(fields)}\n\n".utf8)
}

private func text(_ delta: String, at time: Double) -> StreamStep {
    StreamStep(time: time, data: sse("response.output_text.delta", #""delta":"\#(delta)","item_id":"m","output_index":0,"content_index":0"#))
}

private func reasoning(_ delta: String, at time: Double) -> StreamStep {
    StreamStep(time: time, data: sse("response.reasoning_text.delta", #""delta":"\#(delta)","item_id":"r","output_index":0,"content_index":0"#))
}

private func completed(outputTokens: Int, reasoningTokens: Int = 0, at time: Double) -> StreamStep {
    StreamStep(time: time, data: sse("response.completed", #""response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":\#(outputTokens),"output_tokens_details":{"reasoning_tokens":\#(reasoningTokens)},"total_tokens":9}}"#))
}

/// The 413 body `api_error.rs` writes for `SchedulerError::RequestTooLarge`.
private func overflowBody(serverMax: Int) -> String {
    #"{"error":{"message":"request requires 9000 total tokens","type":"invalid_request_error","param":null,"code":"request_token_capacity_exceeded","details":{"required_total_tokens":9000,"input_tokens":4904,"requested_max_output_tokens":4096,"server_max_context_tokens":\#(serverMax),"max_allowed_output_tokens":0}}}"#
}

private final class FakeTransport: ChatResponsesTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var scripts: [Script]
    private var bodies: [[String: Any]] = []
    private var cancelled = false
    let clock: FakeClock

    init(clock: FakeClock, scripts: [Script] = []) {
        self.clock = clock
        self.scripts = scripts
    }

    func enqueue(_ script: Script) { lock.withLock { scripts.append(script) } }
    var requests: [[String: Any]] { lock.withLock { bodies } }
    var wasCancelled: Bool { lock.withLock { cancelled } }

    func stream(body: Data) async throws -> AsyncThrowingStream<Data, Error> {
        let object = (try JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]
        let script: Script = lock.withLock {
            bodies.append(object)
            return scripts.isEmpty ? .stream([completed(outputTokens: 1, at: clock.now)]) : scripts.removeFirst()
        }
        switch script {
        case let .http(status, body):
            throw ChatTransportError.http(status: status, body: Data(body.utf8))
        case let .stream(steps, hangAtEnd):
            let state = StepCursor(steps: steps)
            let clock = clock
            return AsyncThrowingStream(unfolding: { [weak self] in
                if let step = state.next() {
                    clock.now = step.time
                    return step.data
                }
                guard hangAtEnd else { return nil }
                do {
                    try await Task.sleep(for: .seconds(3600))
                } catch {
                    self?.lock.withLock { self?.cancelled = true }
                    throw error
                }
                return nil
            })
        }
    }
}

private final class StepCursor: @unchecked Sendable {
    private let lock = NSLock()
    private var steps: [StreamStep]
    init(steps: [StreamStep]) { self.steps = steps }
    func next() -> StreamStep? { lock.withLock { steps.isEmpty ? nil : steps.removeFirst() } }
}

private struct FakeManagerError: Error {}

private actor FakeModelManager: BackendModelManaging {
    private(set) var log: [String] = []
    private var loaded: [BackendLoadedModelInfo]
    /// Answers for the next `fetchLoadedModels` calls; `nil` means "backend not up yet".
    private var scriptedFetches: [[BackendLoadedModelInfo]?] = []
    /// Models whose load fails (as when memory runs out), with the admin error code.
    private var failingLoads: Set<String> = []
    private var failureCode = "memory_budget_exceeded"

    init(loaded: [BackendLoadedModelInfo]) { self.loaded = loaded }

    func failLoads(of models: Set<String>, code: String = "memory_budget_exceeded") {
        failingLoads = models
        failureCode = code
    }
    var loadedModels: [BackendLoadedModelInfo] { loaded }

    func script(_ fetches: [[BackendLoadedModelInfo]?]) { scriptedFetches = fetches }
    func restore(_ model: BackendLoadedModelInfo) { loaded.append(model) }

    private func response(_ model: String, status: String = "loaded") -> BackendModelAdminResponse {
        BackendModelAdminResponse(success: true, status: status, code: nil, model: model, loadedModels: loaded,
                                  warningCode: nil, warning: nil, error: nil)
    }

    func fetchHealthz() async throws -> HealthzSnapshot { throw FakeManagerError() }
    func loadModel(
        model: String, modelDir: String, setDefault: Bool, maxCacheCap _: Int?, pinned: Bool,
        promptLookup _: BackendPromptLookupConfig?, audio _: BackendAudioResources?, decision _: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        log.append("load:\(model):\(modelDir):\(setDefault)")
        if failingLoads.contains(model) {
            return BackendModelAdminResponse(success: false, status: "error", code: failureCode,
                                             model: model, loadedModels: loaded, warningCode: nil, warning: nil,
                                             error: "not enough memory")
        }
        if let index = loaded.firstIndex(where: { $0.id == model }), loaded[index].runtimeState == "loaded" {
            return response(model, status: "already_loaded")
        }
        if setDefault {
            for index in loaded.indices { loaded[index].isDefault = false }
        }
        loaded.removeAll { $0.id == model }
        loaded.append(chatModel(model, isDefault: setDefault, pinned: pinned))
        return response(model)
    }
    func unloadModel(model: String, modelDir _: String?) async throws -> BackendModelAdminResponse {
        log.append("unload:\(model)")
        loaded.removeAll { $0.id == model }
        return response(model, status: "unloaded")
    }
    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse {
        log.append("setDefault:\(model)")
        for index in loaded.indices { loaded[index].isDefault = loaded[index].id == model }
        return response(model)
    }
    func pinModel(model: String) async throws -> BackendModelAdminResponse { response(model) }
    func unpinModel(model: String) async throws -> BackendModelAdminResponse { response(model) }
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        if !scriptedFetches.isEmpty {
            guard let next = scriptedFetches.removeFirst() else { throw FakeManagerError() }
            return next
        }
        return loaded
    }
    func clearSharedPromptLookup(model _: String?) async throws -> BackendPromptLookupClearResponse {
        throw FakeManagerError()
    }
}

/// Stands in for the App's configured load path: every load carries the
/// model's configured parameters (here a fixed cache cap and the config's pins).
private final class FakeLoader: ChatModelLoading, @unchecked Sendable {
    static let configuredCacheCap = 8192
    private let lock = NSLock()
    private let manager: FakeModelManager
    private var requests: [(model: BackendRecoveryModel, setDefault: Bool)] = []
    let configuredPins: Set<String>

    init(manager: FakeModelManager, configuredPins: Set<String> = []) {
        self.manager = manager
        self.configuredPins = configuredPins
    }

    var loads: [(model: BackendRecoveryModel, setDefault: Bool)] { lock.withLock { requests } }

    func configuredModel(_ modelID: String) async throws -> BackendRecoveryModel {
        BackendRecoveryModel(
            id: modelID, modelDir: "/models/\(modelID)", isDefault: false, pinned: configuredPins.contains(modelID),
            maxCacheCap: Self.configuredCacheCap, mtpModelDir: nil, mtpDraftTokens: nil, promptLookup: nil,
            samplingDefaults: .empty
        )
    }

    func load(_ model: BackendRecoveryModel, setDefault: Bool) async throws -> BackendModelAdminResponse {
        lock.withLock { requests.append((model, setDefault)) }
        return try await manager.loadModel(
            model: model.id, modelDir: model.modelDir ?? model.id, setDefault: setDefault,
            maxCacheCap: model.maxCacheCap, pinned: model.pinned, promptLookup: model.promptLookup,
            audio: model.audio, decision: model.decision
        )
    }
}

/// The App's DFlash2 settings: which model is the running exclusive target
/// and which models are set to load with DFlash2.
private struct FakeDFlash2: ChatDFlash2Policy {
    var target: ChatDFlash2Target?
    var dflash2Models: Set<String> = []

    func configuredTarget() async -> String? { target?.modelID }
    func usesDFlash2(_ modelID: String) async -> Bool { dflash2Models.contains(modelID) || target?.modelID == modelID }
    func servedTarget() async throws -> ChatDFlash2Target? { target }
}

private let starterModel = "mlx-community/Qwen3.5-4B-MLX-4bit"
private let toggle = BackendModelReasoning(nativeEfforts: [])

private func chatModel(
    _ id: String,
    isDefault: Bool = true,
    state: String = "loaded",
    reasoning: BackendModelReasoning? = nil,
    vision: Bool = false,
    runtimeKind: String = "causal",
    pinned: Bool = false,
    mtpModelDir: String? = nil
) -> BackendLoadedModelInfo {
    BackendLoadedModelInfo(
        id: id, model: id, path: "/models/\(id)", architecture: "qwen3_5", isDefault: isDefault,
        maxPositionEmbeddings: 262_144, pinned: pinned, mtpEnabled: mtpModelDir != nil, mtpModelDir: mtpModelDir,
        mtpDraftTokens: mtpModelDir == nil ? nil : 3,
        capabilities: BackendModelCapabilities(
            runtimeKind: runtimeKind, supportsStreaming: true, supportsVision: vision, supportsMtp: false,
            supportsPromptLookup: true, supportsSpeculativeDecoding: false, supportsKvCache: true,
            supportedSamplingParameters: ["max_tokens"]
        ),
        runtimeState: state,
        reasoning: reasoning
    )
}

@MainActor
private final class Harness {
    let root: URL
    let store: ChatStore
    let loader: FakeLoader
    private let ownsRoot: Bool
    let clock = FakeClock()
    let transport: FakeTransport
    let manager: FakeModelManager
    let controller: ChatSessionController
    var deltas: [CoalescedDelta] = []
    var finished: [ChatTurn] = []
    var errors: [ChatError] = []
    var states: [ChatModelState] = []
    var offers: [StarterRecommendation] = []
    var trimmed: [Int] = []
    var defaultChanges: [String] = []

    init(
        loaded: [BackendLoadedModelInfo],
        memoryGiB: UInt64 = 16,
        catalog: SupportedModelCatalog? = nil,
        warmsUp: Bool = false,
        configuredPins: Set<String> = [],
        sharing root: URL? = nil,
        dflash2: FakeDFlash2 = FakeDFlash2(),
        waitForBackend: @escaping (FakeModelManager) async throws -> Void = { _ in },
        configure: (ChatStore) -> Void = { _ in }
    ) {
        ownsRoot = root == nil
        self.root = root ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-session-\(UUID().uuidString)", isDirectory: true)
        store = ChatStore(root: self.root)
        configure(store)
        transport = FakeTransport(clock: clock)
        manager = FakeModelManager(loaded: loaded)
        loader = FakeLoader(manager: manager, configuredPins: configuredPins)
        let clock = clock
        controller = ChatSessionController(
            store: store,
            modelManager: manager,
            transport: transport,
            loader: loader,
            dflash2: dflash2,
            waitForBackend: { [manager] in try await waitForBackend(manager) },
            catalog: catalog,
            physicalMemoryBytes: memoryGiB << 30,
            now: { clock.now },
            warmsUp: warmsUp,
            pollInterval: .milliseconds(1)
        )
        controller.onDefaultModelChanged = { [weak self] model, _ in self?.defaultChanges.append(model) }
        controller.onDelta = { [weak self] in self?.deltas.append($0) }
        controller.onTurnFinished = { [weak self] in self?.finished.append($0) }
        controller.onError = { [weak self] in self?.errors.append($0) }
        controller.onModelStateChanged = { [weak self] in self?.states.append($0) }
        controller.onUpgradeOffer = { [weak self] in self?.offers.append($0) }
        controller.onHistoryTrimmed = { [weak self] in self?.trimmed.append($0) }
    }

    deinit {
        if ownsRoot { try? FileManager.default.removeItem(at: root) }
    }

    func open(_ model: String = starterModel) async {
        await controller.open(modelID: model, needsLoad: false)
    }

    func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !condition() {
            guard ContinuousClock.now < deadline else { throw CancellationError() }
            try await Task.sleep(for: .milliseconds(2))
        }
    }
}

private func inputCount(_ body: [String: Any]) -> Int { (body["input"] as? [[String: Any]])?.count ?? 0 }

private func roles(_ body: [String: Any]) -> [String] {
    (body["input"] as? [[String: Any]])?.compactMap { $0["role"] as? String } ?? []
}

// MARK: - Tests

@MainActor
@Suite("Chat session controller", .serialized)
struct ChatSessionControllerTests {
    @Test func metricsWithThinkingOff() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        harness.transport.enqueue(.stream([
            text("Hel", at: 0.25), text("lo", at: 0.5), completed(outputTokens: 50, at: 2.25),
        ]))
        await harness.controller.send(text: "hi", imagePaths: [])

        let turn = try #require(harness.finished.last)
        #expect(turn.role == .assistant)
        #expect(turn.text == "Hello")
        #expect(turn.stopped == false)
        let metrics = try #require(turn.metrics)
        #expect(metrics.firstTokenSeconds == 0.25)
        #expect(metrics.thinkingSeconds == nil)
        #expect(metrics.outputTokens == 50)
        #expect(metrics.tokensPerSecond == 25)
        #expect(harness.deltas.map(\.text).joined() == "Hello")
        #expect(harness.errors.isEmpty)
        #expect(harness.controller.isGenerating == false)
        let body = try #require(harness.transport.requests.last)
        #expect(body["reasoning"] == nil)
        #expect(body["max_output_tokens"] as? Int == 4096)
        // Both turns are on disk.
        #expect(ChatStore(root: harness.root).list().first?.turns.map(\.text) == ["hi", "Hello"])
        #expect(ChatStore(root: harness.root).list().first?.title == "hi")
    }

    @Test func replyCutOffByTheOutputBudgetIsMarkedTruncated() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        let incomplete = StreamStep(time: 1.25, data: sse(
            "response.incomplete",
            #""response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"output_tokens":20}}"#
        ))
        harness.transport.enqueue(.stream([text("Long", at: 0.25), incomplete]))
        await harness.controller.send(text: "hi", imagePaths: [])

        let turn = try #require(harness.finished.last)
        #expect(turn.text == "Long")
        #expect(turn.truncated == true)
        #expect(turn.metrics?.tokensPerSecond == 20)
        #expect(harness.errors.isEmpty)
        #expect(ChatStore(root: harness.root).list().first?.turns.last?.truncated == true)
    }

    @Test func metricsWithThinkingOn() async throws {
        let harness = Harness(loaded: [chatModel(starterModel, reasoning: toggle)])
        await harness.open()
        try harness.controller.setReasoningChoice(.on)
        harness.transport.enqueue(.stream([
            reasoning("hmm", at: 0.5), reasoning(" ok", at: 1.0), text("A", at: 2.0),
            completed(outputTokens: 105, reasoningTokens: 20, at: 4.0),
        ]))
        await harness.controller.send(text: "why?", imagePaths: [])

        let turn = try #require(harness.finished.last)
        #expect(turn.reasoning == "hmm ok")
        #expect(turn.text == "A")
        let metrics = try #require(turn.metrics)
        #expect(metrics.firstTokenSeconds == 0.5)
        #expect(metrics.thinkingSeconds == 1.5)
        #expect(metrics.tokensPerSecond == 30)
        let body = try #require(harness.transport.requests.last)
        #expect((body["reasoning"] as? [String: Any])?["effort"] as? String == "medium")
        #expect(body["max_output_tokens"] as? Int == 8192)
        // The choice is remembered for this model.
        #expect(ChatStore(root: harness.root).reasoningChoice(for: starterModel) == .on)
    }

    @Test func stopCancelsRequestAndMarksTurnStopped() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        harness.transport.enqueue(.stream([text("part", at: 0.1)], hangAtEnd: true))
        let sending = Task { await harness.controller.send(text: "long story", imagePaths: []) }
        try await harness.waitUntil { harness.deltas.map(\.text).joined() == "part" }
        #expect(harness.controller.isGenerating)

        harness.controller.stop()
        await sending.value

        #expect(harness.transport.wasCancelled)
        #expect(harness.controller.isGenerating == false)
        let turn = try #require(harness.finished.last)
        #expect(turn.stopped)
        #expect(turn.text == "part")
        #expect(harness.controller.conversation.turns.count == 2)
        #expect(harness.errors.isEmpty)
        #expect(ChatStore(root: harness.root).list().first?.turns.last?.stopped == true)
    }

    @Test func closingWindowCancelsAndPersistsPartialReply() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        harness.transport.enqueue(.stream([reasoning("plan", at: 0.1), text("half", at: 0.2)], hangAtEnd: true))
        let sending = Task { await harness.controller.send(text: "write", imagePaths: []) }
        try await harness.waitUntil { harness.deltas.map(\.text).joined() == "half" }

        harness.controller.windowClosed()
        await sending.value

        #expect(harness.transport.wasCancelled)
        let saved = try #require(ChatStore(root: harness.root).list().first)
        #expect(saved.turns.count == 2)
        #expect(saved.turns[1].text == "half")
        #expect(saved.turns[1].reasoning == "plan")
        #expect(saved.turns[1].stopped)
    }

    @Test func retriesOnceAfterContextOverflowError() async throws {
        let history = ChatConversation(modelID: starterModel, turns: [
            ChatTurn(role: .user, text: String(repeating: "a", count: 4000)),
            ChatTurn(role: .assistant, text: String(repeating: "b", count: 4000)),
            ChatTurn(role: .user, text: "second"),
            ChatTurn(role: .assistant, text: "answer"),
        ])
        let harness = Harness(loaded: [chatModel(starterModel)], configure: { try? $0.save(history) })
        await harness.open()
        harness.controller.openConversation(id: history.id)
        harness.transport.enqueue(.http(status: 413, body: overflowBody(serverMax: 8192)))
        harness.transport.enqueue(.stream([text("fits", at: 0.1), completed(outputTokens: 1, at: 0.2)]))

        await harness.controller.send(text: "third", imagePaths: [])

        let requests = harness.transport.requests
        #expect(requests.count == 2)
        #expect(inputCount(requests[0]) == 5)
        #expect(inputCount(requests[1]) == 3)
        #expect(harness.trimmed == [2])
        #expect(harness.errors.isEmpty)
        #expect(harness.finished.last?.text == "fits")
        #expect(harness.controller.conversation.turns.count == 6)
    }

    @Test func secondContextOverflowIsReported() async throws {
        let history = ChatConversation(modelID: starterModel, turns: [
            ChatTurn(role: .user, text: "first"), ChatTurn(role: .assistant, text: "reply"),
        ])
        let harness = Harness(loaded: [chatModel(starterModel)], configure: { try? $0.save(history) })
        await harness.open()
        harness.controller.openConversation(id: history.id)
        harness.transport.enqueue(.http(status: 413, body: overflowBody(serverMax: 8192)))
        harness.transport.enqueue(.http(status: 413, body: overflowBody(serverMax: 8192)))
        harness.transport.enqueue(.stream([completed(outputTokens: 1, at: 0.1)]))

        await harness.controller.send(text: "again", imagePaths: [])

        #expect(harness.transport.requests.count == 2)
        #expect(harness.errors == [.contextOverflow])
        #expect(harness.finished.isEmpty)
        // The question stays so the user can retry or edit.
        #expect(harness.controller.conversation.turns.last?.role == .user)
        #expect(harness.controller.isGenerating == false)
    }

    @Test func switchModelReplacesDefault() async throws {
        let harness = Harness(loaded: [
            chatModel(starterModel, isDefault: true),
            chatModel("agent/Coder", isDefault: false),
        ])
        await harness.open()
        try await harness.controller.switchModel(to: "mlx-community/Qwen3.6-35B-A3B-4bit")

        #expect(await harness.manager.log == [
            "unload:\(starterModel)",
            "load:mlx-community/Qwen3.6-35B-A3B-4bit:/models/mlx-community/Qwen3.6-35B-A3B-4bit:true",
        ])
        guard case let .ready(info) = harness.controller.modelState else {
            Issue.record("not ready: \(harness.controller.modelState)")
            return
        }
        #expect(info.modelID == "mlx-community/Qwen3.6-35B-A3B-4bit")
        #expect(harness.states.contains(.loading(modelID: "mlx-community/Qwen3.6-35B-A3B-4bit")))
        #expect(harness.defaultChanges == ["mlx-community/Qwen3.6-35B-A3B-4bit"])
    }

    @Test func switchToAnAlreadyLoadedModelMakesItDefault() async throws {
        let harness = Harness(loaded: [
            chatModel(starterModel, isDefault: true),
            chatModel("agent/Coder", isDefault: false),
        ])
        await harness.open()
        try await harness.controller.switchModel(to: "agent/Coder")
        #expect(await harness.manager.log == [
            "unload:\(starterModel)", "load:agent/Coder:/models/agent/Coder:true", "setDefault:agent/Coder",
        ])
    }

    @Test func firstThreeRepliesTriggerUpgradeOffer() async throws {
        let catalog = try SupportedModelCatalog.bundled()
        let harness = Harness(loaded: [chatModel(starterModel)], memoryGiB: 48, catalog: catalog)
        await harness.open()
        for index in 1...4 {
            await harness.controller.send(text: "q\(index)", imagePaths: [])
            #expect(harness.offers.count == (index >= 3 ? 1 : 0))
        }
        #expect(harness.offers.first?.entry.id == "qwen36-35b-a3b-affine4")

        let dismissed = Harness(loaded: [chatModel(starterModel)], memoryGiB: 48, catalog: catalog, configure: {
            $0.upgradePromptDismissed = true
        })
        await dismissed.open()
        for index in 1...3 {
            await dismissed.controller.send(text: "q\(index)", imagePaths: [])
        }
        #expect(dismissed.offers.isEmpty)

        let small = Harness(loaded: [chatModel(starterModel)], memoryGiB: 16, catalog: catalog)
        await small.open()
        for index in 1...3 {
            await small.controller.send(text: "q\(index)", imagePaths: [])
        }
        #expect(small.offers.isEmpty)
    }

    @Test func dismissingTheOfferIsRemembered() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        harness.controller.dismissUpgradeOffer()
        #expect(ChatStore(root: harness.root).upgradePromptDismissed)
    }

    @Test func openWaitsForAModelThatIsAlreadyLoading() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.manager.script([
            nil, // backend still starting
            [chatModel(starterModel, state: "loading")],
            [chatModel(starterModel, state: "loading")],
        ])
        await harness.controller.open(modelID: starterModel, needsLoad: true)

        #expect(await harness.manager.log.isEmpty)
        #expect(harness.states.first == .loading(modelID: starterModel))
        guard case .ready = harness.states.last else {
            Issue.record("not ready: \(harness.states)")
            return
        }
    }

    @Test func openWaitsForBackendRestoreBeforeLoading() async throws {
        // The backend is starting and restores the model during the wait.
        let harness = Harness(loaded: [], waitForBackend: { manager in
            await manager.restore(chatModel(starterModel))
        })
        await harness.controller.open(modelID: starterModel, needsLoad: true)
        #expect(await harness.manager.log.isEmpty)
        guard case .ready = harness.controller.modelState else {
            Issue.record("not ready: \(harness.controller.modelState)")
            return
        }
    }

    @Test func openReportsABackendThatCannotStart() async throws {
        let harness = Harness(loaded: [], waitForBackend: { _ in throw FakeManagerError() })
        await harness.controller.open(modelID: starterModel, needsLoad: true)
        #expect(await harness.manager.log.isEmpty)
        guard case .failed(starterModel, _) = harness.controller.modelState else {
            Issue.record("not failed: \(harness.controller.modelState)")
            return
        }
        #expect(harness.errors.count == 1)
    }

    @Test func openLoadsAModelThatIsNotLoaded() async throws {
        let harness = Harness(loaded: [])
        await harness.controller.open(modelID: starterModel, needsLoad: true)
        #expect(await harness.manager.log == ["load:\(starterModel):/models/\(starterModel):true"])
        #expect(harness.defaultChanges == [starterModel])
        guard case let .ready(info) = harness.controller.modelState else {
            Issue.record("not ready: \(harness.controller.modelState)")
            return
        }
        #expect(info.contextWindow == 262_144)
    }

    @Test func openWarmsUpTheModel() async throws {
        let harness = Harness(loaded: [chatModel(starterModel, reasoning: toggle)], warmsUp: true)
        await harness.open()
        try await harness.waitUntil { !harness.transport.requests.isEmpty }
        let warmup = try #require(harness.transport.requests.first)
        #expect(warmup["max_output_tokens"] as? Int == ChatSessionController.warmupOutputTokens)
        #expect((warmup["reasoning"] as? [String: Any])?["effort"] as? String == "none")
        #expect(harness.controller.conversation.turns.isEmpty)
        #expect(ChatStore(root: harness.root).list().isEmpty)
    }

    @Test func sendStoresImagesAsJPEGAttachments() async throws {
        let harness = Harness(loaded: [chatModel(starterModel, vision: true)])
        await harness.open()
        let png = harness.root.appendingPathComponent("drop.png")
        try FileManager.default.createDirectory(at: harness.root, withIntermediateDirectories: true)
        try writePNG(to: png)

        await harness.controller.send(text: "what is this?", imagePaths: [png.path])

        let user = try #require(harness.controller.conversation.turns.first)
        #expect(user.attachments.count == 1)
        #expect(user.attachments.first?.hasSuffix(".jpg") == true)
        let body = try #require(harness.transport.requests.last)
        let parts = (body["input"] as? [[String: Any]])?.first?["content"] as? [[String: Any]]
        #expect((parts?.first?["image_url"] as? String)?.hasPrefix("data:image/jpeg;base64,") == true)
    }

    @Test func imagesAreRefusedWhenTheModelCannotSee() async throws {
        let harness = Harness(loaded: [chatModel(starterModel, vision: false)])
        await harness.open()
        await harness.controller.send(text: "look", imagePaths: ["/tmp/none.png"])
        #expect(harness.errors == [.imagesNotSupported])
        #expect(harness.transport.requests.isEmpty)
    }

    @Test func upgradeOfferIsShownOnceAcrossControllers() async throws {
        let catalog = try SupportedModelCatalog.bundled()
        let first = Harness(loaded: [chatModel(starterModel)], memoryGiB: 48, catalog: catalog)
        await first.open()
        for index in 1...3 {
            await first.controller.send(text: "q\(index)", imagePaths: [])
        }
        #expect(first.offers.count == 1)
        let conversationID = first.controller.conversation.id

        // The next App run continues the same first conversation.
        let next = Harness(loaded: [chatModel(starterModel)], memoryGiB: 48, catalog: catalog, sharing: first.root)
        await next.open()
        next.controller.openConversation(id: conversationID)
        await next.controller.send(text: "q4", imagePaths: [])
        #expect(next.offers.isEmpty)
    }

    @Test func secondConversationNeverOffersTheUpgrade() async throws {
        let catalog = try SupportedModelCatalog.bundled()
        let harness = Harness(loaded: [chatModel(starterModel)], memoryGiB: 48, catalog: catalog)
        await harness.open()
        await harness.controller.send(text: "only one", imagePaths: [])
        harness.controller.newConversation()
        for index in 1...4 {
            await harness.controller.send(text: "q\(index)", imagePaths: [])
        }
        #expect(harness.offers.isEmpty)
    }

    @Test func switchLoadsTheNewModelWithItsConfiguredParameters() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel)], configuredPins: [target])
        await harness.open()
        try await harness.controller.switchModel(to: target)
        let load = try #require(harness.loader.loads.last)
        #expect(load.model.id == target)
        #expect(load.setDefault)
        #expect(load.model.maxCacheCap == FakeLoader.configuredCacheCap)
        #expect(load.model.pinned)
    }

    @Test func switchKeepsAPinnedDefaultLoaded() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel, pinned: true)])
        await harness.open()
        try await harness.controller.switchModel(to: target)
        #expect(await harness.manager.log == ["load:\(target):/models/\(target):true"])
        let pinnedStarter = await harness.manager.loadedModels.first { $0.id == starterModel }
        #expect(pinnedStarter?.pinned == true)
        #expect(pinnedStarter?.isDefault == false)
    }

    @Test func failedSwitchReloadsThePreviousModelWithItsSettings() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel, mtpModelDir: "/mtp/starter")])
        await harness.open()
        await harness.manager.failLoads(of: [target])

        await #expect(throws: (any Error).self) { try await harness.controller.switchModel(to: target) }

        #expect(await harness.manager.log == [
            "unload:\(starterModel)",
            "load:\(target):/models/\(target):true",
            "load:\(starterModel):/models/\(starterModel):true",
        ])
        let restore = try #require(harness.loader.loads.last)
        #expect(restore.model.mtpModelDir == "/mtp/starter")
        #expect(restore.model.mtpDraftTokens == 3)
        #expect(restore.model.pinned == false)
        #expect(restore.model.maxCacheCap == FakeLoader.configuredCacheCap)
        guard case let .ready(info) = harness.controller.modelState else {
            Issue.record("not ready: \(harness.controller.modelState)")
            return
        }
        #expect(info.modelID == starterModel)
        guard case .modelLoadFailed(target, _) = harness.errors.last else {
            Issue.record("unexpected errors: \(harness.errors)")
            return
        }
    }

    @Test func failedRollbackReportsThatNoModelIsLeft() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        await harness.manager.failLoads(of: [target, starterModel])

        await #expect(throws: (any Error).self) { try await harness.controller.switchModel(to: target) }

        guard case .noModel(failedModelID: target, previousModelID: starterModel, _) = harness.controller.modelState else {
            Issue.record("unexpected state: \(harness.controller.modelState)")
            return
        }
        #expect(harness.states.last == harness.controller.modelState)
        guard case .switchLeftNoModel(modelID: target, previousModelID: starterModel, _) = harness.errors.last else {
            Issue.record("unexpected errors: \(harness.errors)")
            return
        }
    }

    @Test func failedReplyKeepsRolesAlternating() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        harness.transport.enqueue(.stream([
            text("par", at: 0.1),
            StreamStep(time: 0.2, data: sse("response.failed", #""response":{"status":"failed","error":{"code":"generation_error","message":"boom"}}"#)),
        ]))
        await harness.controller.send(text: "q1", imagePaths: [])
        #expect(harness.errors == [.server(code: nil, message: "boom")])

        await harness.controller.send(text: "q2", imagePaths: [])
        #expect(roles(try #require(harness.transport.requests.last)) == ["user"])
        #expect(harness.controller.conversation.turns.map(\.text) == ["q1", "q2", ""])
    }

    @Test func replyStoppedBeforeAnyTextKeepsRolesAlternating() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        harness.transport.enqueue(.stream([], hangAtEnd: true))
        let sending = Task { await harness.controller.send(text: "q1", imagePaths: []) }
        try await harness.waitUntil { harness.transport.requests.count == 1 }
        harness.controller.stop()
        await sending.value

        await harness.controller.send(text: "q2", imagePaths: [])
        #expect(roles(try #require(harness.transport.requests.last)) == ["user"])
    }

    @Test func persistingTheNewDefaultNeverDropsAPin() {
        var config = AppConfig(
            defaultModel: starterModel,
            loadedModels: [starterModel, "agent/Coder"],
            pinnedModels: ["agent/Coder"]
        )
        // The load response may not report the pin; the config keeps it.
        ChatSessionController.persistDefaultModel(
            "new/Model",
            loaded: [chatModel("agent/Coder", isDefault: false), chatModel("new/Model")],
            into: &config
        )
        #expect(config.defaultModelReference == "new/Model")
        #expect(config.restoredModelReferences.sorted() == ["agent/Coder", "new/Model"])
        #expect(config.pinnedModels == ["agent/Coder"])
    }

    @Test func openUsesTheRunningDFlash2TargetWithoutLoading() async throws {
        let target = ChatDFlash2Target(modelID: starterModel, contextWindow: 65_536, supportsVision: true, reasoning: toggle)
        let harness = Harness(loaded: [], dflash2: FakeDFlash2(target: target))
        // The exclusive DFlash2 server has no /admin/api/models/loaded.
        await harness.manager.script(Array(repeating: nil, count: 50))
        await harness.controller.open(modelID: starterModel, needsLoad: true)

        #expect(await harness.manager.log.isEmpty)
        #expect(harness.controller.modelState == .ready(ChatModelInfo(
            modelID: starterModel, reasoning: toggle, supportsVision: true, contextWindow: 65_536
        )))
        await harness.controller.send(text: "hi", imagePaths: [])
        #expect(harness.finished.count == 1)
        #expect(harness.errors.isEmpty)
    }

    @Test func switchingAwayFromARunningDFlash2TargetIsRefused() async throws {
        let target = ChatDFlash2Target(modelID: starterModel, contextWindow: 65_536, supportsVision: false, reasoning: nil)
        let harness = Harness(loaded: [], dflash2: FakeDFlash2(target: target))
        await harness.open()
        let before = harness.controller.modelState

        await #expect(throws: ChatError.modelRequiresDashboard(modelID: starterModel)) {
            try await harness.controller.switchModel(to: "other/Model")
        }
        #expect(harness.errors == [.modelRequiresDashboard(modelID: starterModel)])
        #expect(harness.controller.modelState == before)
        #expect(await harness.manager.log.isEmpty)
    }

    @Test func switchingToADFlash2ModelIsRefused() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel)], dflash2: FakeDFlash2(dflash2Models: [target]))
        await harness.open()
        let before = harness.controller.modelState

        await #expect(throws: ChatError.modelRequiresDashboard(modelID: target)) {
            try await harness.controller.switchModel(to: target)
        }
        #expect(harness.errors == [.modelRequiresDashboard(modelID: target)])
        #expect(harness.controller.modelState == before)
        #expect(await harness.manager.log.isEmpty)
    }

    @Test func openNeverLoadsADFlash2ModelItself() async throws {
        let harness = Harness(loaded: [], dflash2: FakeDFlash2(dflash2Models: [starterModel]))
        await harness.controller.open(modelID: starterModel, needsLoad: true)
        #expect(await harness.manager.log.isEmpty)
        #expect(harness.errors == [.modelRequiresDashboard(modelID: starterModel)])
        #expect(harness.controller.modelState == .idle)
    }

    @Test func failedSwitchWithAPinnedDefaultReturnsToIt() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel, pinned: true)])
        await harness.open()
        let before = harness.controller.modelState
        await harness.manager.failLoads(of: [target])

        await #expect(throws: (any Error).self) { try await harness.controller.switchModel(to: target) }

        #expect(await harness.manager.log == ["load:\(target):/models/\(target):true"])
        #expect(harness.controller.modelState == before)
        guard case .modelLoadFailed(target, _) = harness.errors.last else {
            Issue.record("unexpected errors: \(harness.errors)")
            return
        }
    }

    @Test func failedSwitchForMemoryWithAPinnedDefaultSuggestsUnpinning() async throws {
        let target = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let harness = Harness(loaded: [chatModel(starterModel, pinned: true)])
        await harness.open()
        let before = harness.controller.modelState
        await harness.manager.failLoads(of: [target], code: "gpu_memory_insufficient")

        await #expect(throws: (any Error).self) { try await harness.controller.switchModel(to: target) }

        #expect(harness.controller.modelState == before)
        guard case .switchBlockedByPinnedModel(modelID: target, pinnedModelID: starterModel, _) = harness.errors.last else {
            Issue.record("unexpected errors: \(harness.errors)")
            return
        }
    }

    @Test func memoryAdmissionCodesComeFromEitherAdminErrorShape() {
        // `model_manager.rs` answers a refused load with HTTP 503 and this body.
        let body = #"{"success":false,"status":"error","code":"gpu_memory_insufficient","error":"Not enough available GPU memory"}"#
        let thrown = BackendAPIError.serverResponse(statusCode: 503, body: body)
        #expect(ChatSessionController.adminErrorCode(thrown) == "gpu_memory_insufficient")
        #expect(ChatSessionController.isMemoryAdmissionRejection(thrown))
        let other = BackendAPIError.serverResponse(statusCode: 400, body: #"{"code":"mtp_model_dir_required"}"#)
        #expect(!ChatSessionController.isMemoryAdmissionRejection(other))
        #expect(ChatSessionController.adminErrorCode(BackendAPIError.serverResponse(statusCode: 500, body: "oops")) == nil)
    }

    @Test func sendToAModelUnloadedElsewhereOffersToLoadItAgain() async throws {
        for (status, code) in [(404, "model_not_found"), (404, "model_not_loaded")] {
            let harness = Harness(loaded: [chatModel(starterModel)])
            await harness.open()
            harness.transport.enqueue(.http(status: status, body: #"{"error":{"message":"Model not found","type":"invalid_request_error","param":null,"code":"\#(code)"}}"#))
            await harness.controller.send(text: "hi", imagePaths: [])

            #expect(harness.controller.modelState == .failed(
                modelID: starterModel, message: ChatSessionController.modelUnloadedMessage
            ))
            guard case .modelLoadFailed(starterModel, _) = harness.errors.last else {
                Issue.record("unexpected errors for \(code): \(harness.errors)")
                continue
            }
            // The user's message stays, so loading again and Retry can answer it.
            #expect(harness.controller.conversation.turns.last?.role == .user)
        }
    }

    @Test func modelUnloadedElsewhereLeavesChatFailed() async throws {
        let harness = Harness(loaded: [chatModel(starterModel), chatModel("other/Model", isDefault: false)])
        await harness.open()
        await harness.controller.loadedModelsChanged()
        #expect(harness.controller.modelState.isReady)

        _ = try await harness.manager.unloadModel(model: starterModel, modelDir: nil)
        await harness.controller.loadedModelsChanged()

        #expect(harness.controller.modelState == .failed(
            modelID: starterModel, message: ChatSessionController.modelUnloadedMessage
        ))
    }

    @Test func decodesTheDFlash2ModelListWithReasoning() throws {
        let data = Data(#"{"object":"list","data":[{"id":"m","context_window":65536,"max_output_tokens":4096,"object":"model","created":0,"owned_by":"ironmlx","reasoning":{"native_efforts":["low","medium","xhigh"]}}]}"#.utf8)
        let entry = try #require(BackendDFlash2ModelList.entry(for: "m", in: data))
        #expect(entry.contextWindow == 65_536)
        #expect(entry.reasoning == BackendModelReasoning(nativeEfforts: ["low", "medium", "xhigh"]))
        let plain = Data(#"{"object":"list","data":[{"id":"m","object":"model","created":0,"owned_by":"ironmlx"}]}"#.utf8)
        let bare = try #require(BackendDFlash2ModelList.entry(for: "m", in: plain))
        #expect(bare.reasoning == nil)
        #expect(bare.contextWindow == nil)
        #expect(BackendDFlash2ModelList.entry(for: "other", in: data) == nil)
    }

    @Test func newConversationStartsEmpty() async throws {
        let harness = Harness(loaded: [chatModel(starterModel)])
        await harness.open()
        await harness.controller.send(text: "one", imagePaths: [])
        let first = harness.controller.conversation.id
        harness.controller.newConversation()
        #expect(harness.controller.conversation.turns.isEmpty)
        #expect(harness.controller.conversation.id != first)
        #expect(harness.controller.conversations().map(\.id) == [first])
    }
}

private func writePNG(to url: URL) throws {
    let context = try #require(CGContext(
        data: nil, width: 8, height: 4, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ))
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 8, height: 4))
    let image = try #require(context.makeImage())
    let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    #expect(CGImageDestinationFinalize(destination))
}

private extension ChatModelState {
    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}
