import Foundation

/// What chat needs to know about the model it talks to.
struct ChatModelInfo: Equatable, Sendable {
    var modelID: String
    /// Non-nil only when the backend reports a native reasoning channel.
    var reasoning: BackendModelReasoning?
    var supportsVision: Bool
    /// The model's `max_position_embeddings`; the server may allow less.
    var contextWindow: Int
}

enum ChatModelState: Equatable, Sendable {
    case idle
    case loading(modelID: String)
    case ready(ChatModelInfo)
    case failed(modelID: String, message: String)
    /// A switch unloaded `previousModelID`, the new model failed to load and
    /// loading the previous one again failed too: no chat model is loaded.
    /// The page offers to load `previousModelID` (`open(modelID:needsLoad:)`).
    case noModel(failedModelID: String, previousModelID: String, message: String)
}

/// Errors the page shows in place of the reply, with "Retry" where it helps.
enum ChatError: Error, Equatable, Sendable {
    case modelNotReady
    case busy
    case imagesNotSupported
    case attachmentUnreadable(path: String)
    /// The history still did not fit after dropping older turns once.
    case contextOverflow
    case server(code: String?, message: String)
    case connection(message: String)
    case modelLoadFailed(modelID: String, message: String)
    /// See `ChatModelState.noModel`.
    case switchLeftNoModel(modelID: String, previousModelID: String, message: String)
    /// `modelID` runs with DFlash2, which is set up in the Dashboard; chat
    /// neither loads it nor switches to or away from it.
    case modelRequiresDashboard(modelID: String)
    /// Memory admission refused `modelID` while the pinned old default
    /// `pinnedModelID` stays loaded; unpinning it in the Dashboard frees room.
    case switchBlockedByPinnedModel(modelID: String, pinnedModelID: String, message: String)
}

/// The exclusive DFlash2 target as the running server reports it.
struct ChatDFlash2Target: Equatable, Sendable {
    var modelID: String
    var contextWindow: Int?
    var supportsVision: Bool
    var reasoning: BackendModelReasoning?
}

/// DFlash2 is configured in the Dashboard. An exclusive DFlash2 backend
/// serves one model and has no model admin API, so chat uses that model as
/// it runs and never loads a DFlash2 model itself.
protocol ChatDFlash2Policy: Sendable {
    /// The exclusive DFlash2 target the App runs, decided the way the
    /// Dashboard does (the restored default model with DFlash2 on); nil otherwise.
    func configuredTarget() async -> String?
    /// Whether `modelID` is set to load with DFlash2.
    func usesDFlash2(_ modelID: String) async -> Bool
    /// The served target from `/healthz` and `/v1/models`; nil when the
    /// backend is not running in exclusive DFlash2 mode.
    func servedTarget() async throws -> ChatDFlash2Target?
}

/// For a setup without DFlash2.
struct NoChatDFlash2: ChatDFlash2Policy {
    func configuredTarget() async -> String? { nil }
    func usesDFlash2(_: String) async -> Bool { false }
    func servedTarget() async throws -> ChatDFlash2Target? { nil }
}

/// Loads models the way the App does elsewhere: with the cache cap, pin,
/// MTP, prompt lookup, sampling defaults, audio and decision settings the
/// user configured.
protocol ChatModelLoading: Sendable {
    /// The App's load parameters for `modelID`, with its folder checked.
    /// Throws when the model is not available locally.
    func configuredModel(_ modelID: String) async throws -> BackendRecoveryModel
    func load(_ model: BackendRecoveryModel, setDefault: Bool) async throws -> BackendModelAdminResponse
}

private struct ChatModelLoadError: LocalizedError {
    var message: String
    /// The admin API's `code` when the backend answered with one.
    var code: String?
    var errorDescription: String? { message }

    init(message: String, code: String? = nil) {
        self.message = message
        self.code = code
    }

    /// A refused admin call: its `error` text and `code`.
    init(_ response: BackendModelAdminResponse, fallback: String) {
        self.init(message: response.error ?? response.code ?? fallback, code: response.code)
    }
}

/// The chat window's non-UI core: loads and switches the model, sends the
/// conversation to `/v1/responses`, streams the reply in ~16 ms batches,
/// measures speed and keeps history on disk. The page bridge subscribes to
/// the `on…` callbacks; all of them run on the main actor.
@MainActor
final class ChatSessionController {
    static let warmupOutputTokens = 1
    static let upgradeOfferReplyCount = 3
    /// Used when the backend reports no `max_position_embeddings`.
    static let fallbackContextWindow = 32_768
    /// `api_error.rs` code for `SchedulerError::RequestTooLarge` (HTTP 413).
    static let contextOverflowCode = "request_token_capacity_exceeded"
    /// The model a request names is not served. `/v1/responses` answers
    /// `model_not_found` (HTTP 404, `ApiError::engine_resolution` for
    /// `EngineRegistryError::UnknownModel`/`ModelDisabled`, api_error.rs);
    /// the admin API says `model_not_loaded` (HTTP 404, `MODEL_NOT_LOADED_CODE`
    /// in model_manager.rs).
    nonisolated static let modelUnavailableCodes: Set<String> = ["model_not_found", "model_not_loaded"]
    /// Admin load refusals (HTTP 503, model_manager.rs) that other loaded
    /// models cause: GPU memory, the model/total memory limits and the
    /// loaded-model count. `kv_memory_budget_exceeded` is a KV setting of the
    /// model itself, so it is not one of them.
    nonisolated static let memoryAdmissionCodes: Set<String> = [
        "gpu_memory_insufficient", "model_memory_limit_exceeded",
        "total_memory_limit_exceeded", "max_loaded_models_reached",
    ]
    /// `.failed` message for a ready model the backend no longer serves.
    nonisolated static let modelUnloadedMessage = "The model is no longer loaded."

    /// Batched reply text, at most once per coalescing interval.
    var onDelta: ((CoalescedDelta) -> Void)?
    /// A finished or stopped assistant turn, already saved.
    var onTurnFinished: ((ChatTurn) -> Void)?
    var onError: ((ChatError) -> Void)?
    var onModelStateChanged: ((ChatModelState) -> Void)?
    /// Once ever: after the third finished reply in the user's first
    /// conversation, on a starter model, when memory allows the upgrade and
    /// the user has not declined it.
    var onUpgradeOffer: ((StarterRecommendation) -> Void)?
    /// Number of oldest turns left out of the request that is now streaming.
    var onHistoryTrimmed: ((Int) -> Void)?
    /// The backend's default model changed (chat loaded or switched a model);
    /// the App records it so the next launch and the menu bar agree.
    var onDefaultModelChanged: ((String, [BackendLoadedModelInfo]) -> Void)?

    private(set) var conversation: ChatConversation
    private(set) var modelState: ChatModelState = .idle
    var isGenerating: Bool { activeReply != nil }

    private let store: ChatStore
    private let modelManager: any BackendModelManaging
    private let transport: any ChatResponsesTransport
    private let loader: any ChatModelLoading
    private let dflash2: any ChatDFlash2Policy
    private let waitForBackend: () async throws -> Void
    private let configuredMaxOutputTokens: (String) -> Int?
    private let catalog: SupportedModelCatalog?
    private let physicalMemoryBytes: UInt64
    private let now: () -> Double
    private let coalescingInterval: Duration
    private let warmsUp: Bool
    private let pollInterval: Duration
    private let loadTimeout: Duration

    private var activeReply: ActiveReply?
    private var modelGeneration = 0
    private var warmupTask: Task<Void, Never>?
    /// Context sizes the server reported when it rejected a request; the
    /// server cap can be smaller than the model's own window.
    private var serverContextLimits: [String: Int] = [:]

    init(
        store: ChatStore,
        modelManager: any BackendModelManaging,
        transport: any ChatResponsesTransport,
        loader: any ChatModelLoading,
        dflash2: any ChatDFlash2Policy = NoChatDFlash2(),
        waitForBackend: @escaping () async throws -> Void = {},
        configuredMaxOutputTokens: @escaping (String) -> Int? = { _ in nil },
        catalog: SupportedModelCatalog? = nil,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        now: @escaping () -> Double = { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 },
        coalescingInterval: Duration = .milliseconds(16),
        warmsUp: Bool = true,
        pollInterval: Duration = .milliseconds(500),
        loadTimeout: Duration = .seconds(600)
    ) {
        self.store = store
        self.modelManager = modelManager
        self.transport = transport
        self.loader = loader
        self.dflash2 = dflash2
        self.waitForBackend = waitForBackend
        self.configuredMaxOutputTokens = configuredMaxOutputTokens
        self.catalog = catalog
        self.physicalMemoryBytes = physicalMemoryBytes
        self.now = now
        self.coalescingInterval = coalescingInterval
        self.warmsUp = warmsUp
        self.pollInterval = pollInterval
        self.loadTimeout = loadTimeout
        conversation = ChatConversation(modelID: "")
    }

    // MARK: Model

    /// Opens chat on `modelID`. A model that is already loaded is used as is;
    /// one that is loading (for example restored by a backend that is still
    /// starting) is waited for; otherwise it is loaded and made the default.
    /// `needsLoad` only decides whether "loading" shows at once. A backend in
    /// exclusive DFlash2 mode serves only its target, which chat then uses.
    /// A DFlash2 model that is not running is never loaded here.
    func open(modelID: String, needsLoad: Bool) async {
        let prior = modelState
        modelGeneration += 1
        let generation = modelGeneration
        if needsLoad { setState(.loading(modelID: modelID)) }
        do {
            try await waitForBackend()
            if await dflash2.configuredTarget() != nil, let served = try await dflash2.servedTarget() {
                guard generation == modelGeneration else { return }
                becomeReady(ChatModelInfo(
                    modelID: served.modelID,
                    reasoning: served.reasoning,
                    supportsVision: served.supportsVision,
                    contextWindow: served.contextWindow.flatMap { $0 > 0 ? $0 : nil } ?? Self.fallbackContextWindow
                ))
                return
            }
            let entry = try await prepareModel(modelID, generation: generation)
            guard generation == modelGeneration else { return }
            becomeReady(entry)
        } catch let error as ChatError {
            guard generation == modelGeneration else { return }
            setState(prior)
            onError?(error)
        } catch {
            guard generation == modelGeneration else { return }
            reportLoadFailure(modelID, error)
        }
    }

    /// Replaces the default conversation model: unloads it (unless the user
    /// pinned it), loads `modelID` with its configured parameters as the new
    /// default, and leaves every other loaded model alone. The page asks the
    /// user first. When the new model fails to load, the old one is loaded
    /// again as it was; when that fails too, the state becomes `.noModel`.
    func switchModel(to modelID: String) async throws {
        // DFlash2 is switched in the Dashboard; the state stays as it is.
        if let running = await dflash2.configuredTarget() {
            guard running == modelID else { throw refuse(.modelRequiresDashboard(modelID: running)) }
            await open(modelID: modelID, needsLoad: false)
            return
        }
        if await dflash2.usesDFlash2(modelID) {
            throw refuse(.modelRequiresDashboard(modelID: modelID))
        }
        let prior = modelState
        // Until something is unloaded or loaded, the old model still serves.
        var backendChanged = false
        var pinnedPrevious: String?
        stop()
        warmupTask?.cancel()
        modelGeneration += 1
        let generation = modelGeneration
        setState(.loading(modelID: modelID))
        do {
            try await waitForBackend()
            let loaded = try await modelManager.fetchLoadedModels()
            let target = loaded.first { Self.matches($0, modelID) }
            if let target, target.isDefault, target.runtimeState == "loaded" {
                becomeReady(target)
                return
            }
            let previous = loaded.first {
                $0.isDefault && !Self.matches($0, modelID) && Self.isConversationRuntime($0)
            }
            // Read the parameters (and check the folder) before unloading, so
            // a missing model never leaves the user with none.
            let request = if let target, target.runtimeState == "loaded" {
                BackendRecoveryModel.reloading(target, configured: nil)
            } else {
                try await loader.configuredModel(modelID)
            }
            // A model the user pinned stays loaded; only the default moves.
            let unloaded = previous.flatMap { $0.pinned ? nil : $0 }
            pinnedPrevious = previous.flatMap { $0.pinned ? $0.id : nil }
            if let unloaded {
                let response = try await modelManager.unloadModel(model: unloaded.id, modelDir: unloaded.path)
                guard response.success else {
                    throw ChatModelLoadError(response, fallback: "unload failed")
                }
                backendChanged = true
            }
            let latest: [BackendLoadedModelInfo]
            do {
                let response = try await loader.load(request, setDefault: true)
                guard response.success else {
                    throw ChatModelLoadError(response, fallback: "load failed")
                }
                if response.status == "already_loaded" {
                    // An already-loaded model ignores `setDefault` on load.
                    latest = try await modelManager.setDefaultModel(modelID).loadedModels
                } else {
                    latest = response.loadedModels
                }
            } catch {
                guard let unloaded else { throw error }
                try await rollBack(to: unloaded, after: error, switchingTo: modelID, generation: generation)
                throw error
            }
            backendChanged = true
            onDefaultModelChanged?(modelID, latest)
            let entry = try await prepareModel(modelID, generation: generation, loadRequested: true)
            guard generation == modelGeneration else { return }
            becomeReady(entry)
        } catch {
            guard generation == modelGeneration, case .loading = modelState else { throw error }
            if !backendChanged, case .ready = prior {
                // Nothing was unloaded (or the old default is pinned): it still serves.
                setState(prior)
                if let pinnedPrevious, Self.isMemoryAdmissionRejection(error) {
                    onError?(.switchBlockedByPinnedModel(
                        modelID: modelID, pinnedModelID: pinnedPrevious, message: error.localizedDescription
                    ))
                } else {
                    onError?(.modelLoadFailed(modelID: modelID, message: error.localizedDescription))
                }
            } else {
                reportLoadFailure(modelID, error)
            }
            throw error
        }
    }

    private func refuse(_ error: ChatError) -> ChatError {
        onError?(error)
        return error
    }

    var reasoningChoice: ReasoningChoice {
        guard case let .ready(model) = modelState, model.reasoning != nil else { return .off }
        return store.reasoningChoice(for: model.modelID) ?? .off
    }

    /// Remembered per model in the chat preferences.
    func setReasoningChoice(_ choice: ReasoningChoice) throws {
        guard case let .ready(model) = modelState else { return }
        try store.setReasoningChoice(choice, for: model.modelID)
    }

    // MARK: Conversations

    func newConversation() {
        stop()
        conversation = ChatConversation(modelID: currentModelID ?? conversation.modelID)
    }

    func openConversation(id: UUID) {
        guard let stored = store.conversation(id: id) else { return }
        stop()
        conversation = stored
    }

    /// Newest first, at most 20.
    func conversations() -> [ChatConversation] {
        store.list()
    }

    func dismissUpgradeOffer() {
        store.upgradePromptDismissed = true
    }

    /// Where an image of this conversation is stored (`ChatTurn.attachments`).
    func attachmentURL(for name: String) -> URL {
        store.attachmentURL(for: name)
    }

    // MARK: Replies

    /// Adds the user's message and streams the reply. Returns when the reply
    /// is finished, stopped or failed.
    func send(text: String, imagePaths: [String]) async {
        guard activeReply == nil else {
            onError?(.busy)
            return
        }
        guard case let .ready(model) = modelState else {
            onError?(.modelNotReady)
            return
        }
        let message = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty || !imagePaths.isEmpty else { return }
        guard imagePaths.isEmpty || model.supportsVision else {
            onError?(.imagesNotSupported)
            return
        }
        let reply = ActiveReply()
        activeReply = reply
        var attachments: [String] = []
        for path in imagePaths {
            guard let data = await Self.normalizedImage(at: path),
                  let name = try? store.saveAttachment(data, ext: "jpg")
            else {
                discard(attachments)
                if activeReply === reply { activeReply = nil }
                onError?(.attachmentUnreadable(path: path))
                return
            }
            attachments.append(name)
        }
        guard activeReply === reply else {
            // Stopped while the images were prepared.
            discard(attachments)
            return
        }
        if conversation.turns.isEmpty {
            conversation.title = ChatConversation.title(from: message)
            conversation.createdAt = Date()
        }
        conversation.modelID = model.modelID
        conversation.turns.append(ChatTurn(role: .user, text: message, attachments: attachments))
        persist()
        await run(reply, model: model)
    }

    /// Asks again for the reply to the last user message, after an error.
    func retry() async {
        guard activeReply == nil else { return }
        guard case let .ready(model) = modelState else {
            onError?(.modelNotReady)
            return
        }
        guard conversation.turns.last?.role == .user else { return }
        let reply = ActiveReply()
        activeReply = reply
        await run(reply, model: model)
    }

    /// Cancels the request and keeps what has arrived as a stopped turn.
    func stop() {
        guard let reply = activeReply else { return }
        activeReply = nil
        reply.task?.cancel()
        guard reply.isStreaming else { return }
        let end = now()
        let metrics = reply.firstToken.map { _ in reply.metrics(outputTokens: nil, end: end) }
        finish(with: ChatTurn(
            role: .assistant,
            text: reply.text,
            reasoning: reply.reasoning.isEmpty ? nil : reply.reasoning,
            metrics: metrics,
            stopped: true
        ))
    }

    /// The chat window closed (or the App quits): the request is cancelled
    /// and the partial reply saved.
    func windowClosed() {
        stop()
        warmupTask?.cancel()
        warmupTask = nil
    }

    /// The backend's loaded models changed (the Dashboard unloaded a model,
    /// changed the default, or the backend restarted without it). When the
    /// ready model is gone, chat says so and offers to load it again instead
    /// of sending to a model the server no longer has.
    func loadedModelsChanged() async {
        guard case let .ready(model) = modelState, activeReply == nil else { return }
        let generation = modelGeneration
        guard await dflash2.configuredTarget() == nil,
              let loaded = try? await modelManager.fetchLoadedModels()
        else { return }
        guard generation == modelGeneration, activeReply == nil,
              case let .ready(current) = modelState, current.modelID == model.modelID
        else { return }
        let entry = loaded.first { Self.matches($0, model.modelID) }
        if let state = entry?.runtimeState, ["loaded", "loading", "draining"].contains(state) { return }
        modelUnavailable(model.modelID)
    }

    // MARK: Private: replies

    private enum Outcome {
        case finished
        case contextOverflow(serverMaxContext: Int?, maxOutputTokens: Int?)
        /// The server no longer serves the model (see `modelUnavailableCodes`).
        case modelUnavailable
        case failed(ChatError)
    }

    private func run(_ reply: ActiveReply, model: ChatModelInfo) async {
        warmupTask?.cancel()
        warmupTask = nil
        reply.isStreaming = true
        reply.start = now()
        let task = Task { [weak self] in
            guard let self else { return }
            await stream(reply, model: model)
        }
        reply.task = task
        await task.value
    }

    /// One retry after a context overflow: trim history when possible, or
    /// use the server's remaining output budget for the last user message.
    private func stream(_ reply: ActiveReply, model: ChatModelInfo) async {
        var contextWindow = min(model.contextWindow, serverContextLimits[model.modelID] ?? .max)
        var minimumDropped = 0
        var outputBudget = configuredMaxOutputTokens(model.modelID)
        for attempt in 0..<2 {
            let request = ChatRequestBuilder.build(
                conversation: conversation,
                modelID: model.modelID,
                reasoning: reasoningChoice,
                capability: model.reasoning,
                contextWindow: contextWindow,
                configuredMaxOutputTokens: outputBudget,
                minimumDroppedTurns: minimumDropped,
                loadAttachment: { [store] in store.attachmentData(named: $0) }
            )
            let outcome = await perform(request, reply: reply)
            guard activeReply === reply else { return }
            switch outcome {
            case .finished:
                return
            case let .contextOverflow(serverMax, maxOutput):
                if let serverMax { serverContextLimits[model.modelID] = serverMax }
                let smallerWindow = min(contextWindow, serverMax ?? contextWindow)
                let inputCount = (request.body["input"] as? [[String: Any]])?.count ?? 0
                let historyLeft = inputCount > 1
                let requestedOutput = request.body["max_output_tokens"] as? Int ?? 1
                let smallerOutput = maxOutput.flatMap { $0 > 0 && $0 < requestedOutput ? $0 : nil }
                guard attempt == 0, historyLeft || smallerWindow < contextWindow || smallerOutput != nil else {
                    fail(reply, .contextOverflow)
                    return
                }
                contextWindow = smallerWindow
                if !historyLeft, let smallerOutput {
                    // Text estimates omit template overhead. The server's
                    // tokenized input gives an exact limit without discarding
                    // the last question or changing the model's saved settings.
                    outputBudget = smallerOutput
                    minimumDropped = request.droppedTurnCount
                } else {
                    minimumDropped = request.droppedTurnCount + 1
                }
            case .modelUnavailable:
                guard activeReply === reply else { return }
                activeReply = nil
                modelUnavailable(model.modelID)
                return
            case let .failed(error):
                fail(reply, error)
                return
            }
        }
        fail(reply, .contextOverflow)
    }

    private func perform(_ request: ChatRequest, reply: ActiveReply) async -> Outcome {
        let body: Data
        do {
            body = try JSONSerialization.data(withJSONObject: request.body)
        } catch {
            return .failed(.server(code: nil, message: error.localizedDescription))
        }
        let stream: AsyncThrowingStream<Data, Error>
        do {
            stream = try await transport.stream(body: body)
        } catch let ChatTransportError.http(status, data) {
            return Self.outcome(status: status, body: data)
        } catch {
            return activeReply === reply ? .failed(.connection(message: error.localizedDescription)) : .finished
        }
        guard activeReply === reply else { return .finished }
        if request.droppedTurnCount > 0 { onHistoryTrimmed?(request.droppedTurnCount) }

        let coalescer = StreamCoalescer(interval: coalescingInterval) { [weak self] delta in
            await self?.deliver(delta, for: reply)
        }
        var parser = ResponsesStreamParser()
        do {
            for try await chunk in stream {
                guard activeReply === reply else { return .finished }
                for event in parser.feed(chunk) {
                    let time = now()
                    switch event {
                    case let .reasoningDelta(delta):
                        reply.received(reasoning: delta, at: time)
                        await coalescer.append(event)
                    case let .textDelta(delta):
                        reply.received(text: delta, at: time)
                        await coalescer.append(event)
                    case let .completed(outputTokens, _):
                        await coalescer.append(event)
                        complete(reply, outputTokens: outputTokens, end: time)
                        return .finished
                    case let .incomplete(_, outputTokens):
                        // Hit the output budget: keep what arrived, marked as cut off.
                        await coalescer.append(event)
                        complete(reply, outputTokens: outputTokens, end: time, truncated: true)
                        return .finished
                    case let .failed(message):
                        await coalescer.finish()
                        if reply.firstToken == nil, let serverMax = Self.contextOverflowLimit(in: message) {
                            return .contextOverflow(
                                serverMaxContext: serverMax,
                                maxOutputTokens: Self.contextOverflowOutputLimit(in: message)
                            )
                        }
                        return .failed(.server(code: nil, message: message))
                    }
                    guard activeReply === reply else { return .finished }
                }
            }
        } catch {
            guard activeReply === reply else { return .finished }
            await coalescer.finish()
            return .failed(.connection(message: error.localizedDescription))
        }
        guard activeReply === reply else { return .finished }
        await coalescer.finish()
        return .failed(.connection(message: "The reply ended before it was complete."))
    }

    private func deliver(_ delta: CoalescedDelta, for reply: ActiveReply) {
        guard activeReply === reply else { return }
        onDelta?(delta)
    }

    private func complete(_ reply: ActiveReply, outputTokens: Int?, end: Double, truncated: Bool = false) {
        guard activeReply === reply else { return }
        activeReply = nil
        finish(with: ChatTurn(
            role: .assistant,
            text: reply.text,
            reasoning: reply.reasoning.isEmpty ? nil : reply.reasoning,
            metrics: reply.metrics(outputTokens: outputTokens, end: end),
            truncated: truncated ? true : nil
        ))
        offerUpgradeIfDue()
    }

    private func finish(with turn: ChatTurn) {
        conversation.turns.append(turn)
        persist()
        onTurnFinished?(turn)
    }

    /// The user's message stays, so the page can offer "Retry".
    private func fail(_ reply: ActiveReply, _ error: ChatError) {
        guard activeReply === reply else { return }
        activeReply = nil
        onError?(error)
    }

    private func persist() {
        conversation.updatedAt = Date()
        do {
            try store.save(conversation)
        } catch {
            IronMLXAppLogger.error("Chat conversation could not be saved: \(error)")
        }
    }

    private func discard(_ attachments: [String]) {
        for name in attachments {
            try? FileManager.default.removeItem(at: store.attachmentURL(for: name))
        }
    }

    private func offerUpgradeIfDue() {
        // Shown once ever, and only in the user's first conversation.
        guard !store.upgradePromptOffered, !store.upgradePromptDismissed, let catalog,
              conversation.id == store.firstConversationID,
              case let .ready(model) = modelState
        else { return }
        let replies = conversation.turns.filter { $0.role == .assistant && !$0.stopped }.count
        guard replies >= Self.upgradeOfferReplyCount,
              let upgrade = StarterModelSelector.upgrade(
                  catalog: catalog, physicalMemoryBytes: physicalMemoryBytes, currentModelID: model.modelID
              )
        else { return }
        store.upgradePromptOffered = true
        onUpgradeOffer?(upgrade)
    }

    /// `api_error.rs` answers a request that does not fit with HTTP 413 and
    /// `error.code == "request_token_capacity_exceeded"`, plus the server's
    /// context size and remaining output budget in `error.details`.
    private static func outcome(status: Int, body: Data) -> Outcome {
        let envelope = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
        let error = envelope?["error"] as? [String: Any]
        let code = error?["code"] as? String
        if code == contextOverflowCode {
            let details = error?["details"] as? [String: Any]
            return .contextOverflow(
                serverMaxContext: (details?["server_max_context_tokens"] as? NSNumber)?.intValue,
                maxOutputTokens: (details?["max_allowed_output_tokens"] as? NSNumber)?.intValue
            )
        }
        if let code, modelUnavailableCodes.contains(code) {
            return .modelUnavailable
        }
        let message = error?["message"] as? String
            ?? String(data: body, encoding: .utf8).flatMap { $0.isEmpty ? nil : $0 }
            ?? "HTTP \(status)"
        return .failed(.server(code: code, message: message))
    }

    /// The same overflow can arrive as `response.failed` when it is found
    /// after admission; its message is `SchedulerError::RequestTooLarge`'s text.
    static func contextOverflowLimit(in message: String) -> Int? {
        guard let match = message.firstMatch(of: /the server maximum context capacity is (\d+) tokens/) else {
            return nil
        }
        return Int(match.1)
    }

    private static func contextOverflowOutputLimit(in message: String) -> Int? {
        guard let match = message.firstMatch(of: /Reduce request max_tokens to (\d+) or less/) else {
            return nil
        }
        return Int(match.1)
    }

    nonisolated private static func normalizedImage(at path: String) async -> Data? {
        await Task.detached(priority: .userInitiated) {
            ChatImageNormalizer.jpegData(contentsOf: URL(fileURLWithPath: path))
        }.value
    }

    // MARK: Private: model

    private var currentModelID: String? {
        if case let .ready(model) = modelState { return model.modelID }
        return nil
    }

    private func setState(_ state: ChatModelState) {
        guard state != modelState else { return }
        modelState = state
        onModelStateChanged?(state)
    }

    private func becomeReady(_ entry: BackendLoadedModelInfo) {
        becomeReady(ChatModelInfo(
            modelID: entry.id,
            reasoning: entry.reasoning,
            supportsVision: entry.capabilities.supportsVision,
            contextWindow: entry.maxPositionEmbeddings > 0 ? entry.maxPositionEmbeddings : Self.fallbackContextWindow
        ))
    }

    private func becomeReady(_ model: ChatModelInfo) {
        setState(.ready(model))
        if conversation.turns.isEmpty { conversation.modelID = model.modelID }
        startWarmup(model)
    }

    /// The ready model is gone: `.failed` makes the page offer to load it
    /// again, and a reply waiting for an answer hears that the model is not ready.
    private func modelUnavailable(_ modelID: String) {
        warmupTask?.cancel()
        warmupTask = nil
        modelGeneration += 1
        setState(.failed(modelID: modelID, message: Self.modelUnloadedMessage))
        onError?(.modelLoadFailed(modelID: modelID, message: Self.modelUnloadedMessage))
    }

    /// The admin API's error `code`: from a refused response, from a load the
    /// backend registry reported failed, or from the JSON body of a non-2xx
    /// answer (`BackendAPIError.serverResponse`).
    nonisolated static func adminErrorCode(_ error: Error) -> String? {
        if let error = error as? ChatModelLoadError { return error.code }
        if let error = error as? ModelLoadFailedError { return error.code }
        guard case let BackendAPIError.serverResponse(_, body) = error,
              let data = body?.data(using: .utf8),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        return object["code"] as? String ?? (object["error"] as? [String: Any])?["code"] as? String
    }

    nonisolated static func isMemoryAdmissionRejection(_ error: Error) -> Bool {
        adminErrorCode(error).map(memoryAdmissionCodes.contains) ?? false
    }

    private func reportLoadFailure(_ modelID: String, _ error: Error) {
        let message = error.localizedDescription
        setState(.failed(modelID: modelID, message: message))
        onError?(.modelLoadFailed(modelID: modelID, message: message))
    }

    /// Polls the loaded list until `modelID` is loaded. First waits for a
    /// starting backend to finish restoring its models, so a model it is
    /// restoring is never loaded a second time; a model still loading is
    /// waited for, and only a model that is not loaded is loaded here.
    private func prepareModel(
        _ modelID: String,
        generation: Int,
        loadRequested alreadyRequested: Bool = false
    ) async throws -> BackendLoadedModelInfo {
        let deadline = ContinuousClock.now + loadTimeout
        var loadRequested = alreadyRequested
        while true {
            guard generation == modelGeneration else { throw CancellationError() }
            var loaded: [BackendLoadedModelInfo]?
            do {
                loaded = try await modelManager.fetchLoadedModels()
            } catch {
                guard ContinuousClock.now < deadline else { throw error }
            }
            guard generation == modelGeneration else { throw CancellationError() }
            if let loaded {
                let entry = loaded.first { Self.matches($0, modelID) }
                switch entry?.runtimeState {
                case "loaded":
                    return entry!
                case "loading", "draining":
                    setState(.loading(modelID: modelID))
                default:
                    guard !loadRequested else {
                        throw ChatModelLoadError(message: "\(modelID) is not loaded (\(entry?.runtimeState ?? "missing"))")
                    }
                    if await dflash2.usesDFlash2(modelID) {
                        throw ChatError.modelRequiresDashboard(modelID: modelID)
                    }
                    loadRequested = true
                    setState(.loading(modelID: modelID))
                    let configured = try await loader.configuredModel(modelID)
                    let response = try await loader.load(configured, setDefault: true)
                    guard response.success else {
                        throw ChatModelLoadError(response, fallback: "load failed")
                    }
                    onDefaultModelChanged?(modelID, response.loadedModels)
                    continue
                }
            }
            guard ContinuousClock.now < deadline else {
                throw ChatModelLoadError(message: "\(modelID) did not finish loading in time")
            }
            try await Task.sleep(for: pollInterval)
        }
    }

    /// Loads the model a failed switch unloaded, exactly as it ran before
    /// (folder, pin, MTP, prompt lookup, decision). When that fails too, no
    /// chat model is left and the page is told so.
    private func rollBack(
        to previous: BackendLoadedModelInfo,
        after failure: Error,
        switchingTo modelID: String,
        generation: Int
    ) async throws {
        do {
            let configured = try? await loader.configuredModel(previous.id)
            let response = try await loader.load(
                BackendRecoveryModel.reloading(previous, configured: configured), setDefault: true
            )
            guard response.success else {
                throw ChatModelLoadError(response, fallback: "load failed")
            }
            onDefaultModelChanged?(previous.id, response.loadedModels)
            let entry = try await prepareModel(previous.id, generation: generation, loadRequested: true)
            guard generation == modelGeneration else { return }
            becomeReady(entry)
            onError?(.modelLoadFailed(modelID: modelID, message: failure.localizedDescription))
        } catch {
            guard generation == modelGeneration else { throw error }
            let message = failure.localizedDescription
            setState(.noModel(failedModelID: modelID, previousModelID: previous.id, message: message))
            let lost = ChatError.switchLeftNoModel(modelID: modelID, previousModelID: previous.id, message: message)
            onError?(lost)
            throw lost
        }
    }

    /// A tiny request right after the model is ready, so the first real
    /// message does not pay for cold kernels. Sending cancels it.
    private func startWarmup(_ model: ChatModelInfo) {
        guard warmsUp, activeReply == nil else { return }
        warmupTask?.cancel()
        var body: [String: Any] = [
            "model": model.modelID,
            "input": [["type": "message", "role": "user", "content": [["type": "input_text", "text": "Hi"]]]],
            "stream": true,
            "store": false,
            "max_output_tokens": Self.warmupOutputTokens,
        ]
        if model.reasoning != nil { body["reasoning"] = ["effort": "none"] }
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return }
        let transport = transport
        warmupTask = Task.detached(priority: .utility) {
            do {
                for try await _ in try await transport.stream(body: data) {}
            } catch {
                // Warm-up is best effort.
            }
        }
    }

    private static func matches(_ entry: BackendLoadedModelInfo, _ modelID: String) -> Bool {
        entry.id == modelID || entry.model == modelID
    }

    private static func isConversationRuntime(_ entry: BackendLoadedModelInfo) -> Bool {
        ["causal", "block_diffusion"].contains(entry.capabilities.runtimeKind)
    }
}

/// The reply being streamed; times are `now()` seconds.
@MainActor
private final class ActiveReply {
    var task: Task<Void, Never>?
    /// False while images are still being prepared, before the request.
    var isStreaming = false
    var start: Double = 0
    var text = ""
    var reasoning = ""
    var firstToken: Double?
    var firstReasoning: Double?
    var firstText: Double?

    func received(reasoning delta: String, at time: Double) {
        firstToken = firstToken ?? time
        firstReasoning = firstReasoning ?? time
        reasoning += delta
    }

    func received(text delta: String, at time: Double) {
        firstToken = firstToken ?? time
        firstText = firstText ?? time
        text += delta
    }

    /// Thinking runs from the first reasoning token to the first answer token
    /// (or to the end when no answer came); speed is output tokens over the
    /// time from the first token to the end.
    func metrics(outputTokens: Int?, end: Double) -> ChatMetrics {
        let thinking = firstReasoning.map { (firstText ?? end) - $0 }
        var speed: Double?
        if let outputTokens, let firstToken, end > firstToken {
            speed = Double(outputTokens) / (end - firstToken)
        }
        return ChatMetrics(
            firstTokenSeconds: firstToken.map { $0 - start },
            thinkingSeconds: thinking,
            tokensPerSecond: speed,
            outputTokens: outputTokens
        )
    }
}
