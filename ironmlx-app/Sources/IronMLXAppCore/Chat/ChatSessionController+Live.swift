import Foundation

/// Builds a `BackendAPIClient` from the current config for every call, so a
/// changed port is followed while the chat window stays open.
struct ConfiguredBackendModelManager: BackendModelManaging {
    let configStore: AppConfigStore
    /// Loads go through the App's load tracking when set.
    var loadActivity: ModelLoadActivityCenter?

    private var client: BackendAPIClient {
        let config = configStore.load()
        return BackendAPIClient(host: config.host, port: config.port)
    }

    func fetchHealthz() async throws -> HealthzSnapshot {
        try await client.fetchHealthz()
    }

    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        promptLookup: BackendPromptLookupConfig?,
        audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        let manager: any BackendModelManaging = if let loadActivity {
            LoadTrackingModelManager(base: client, loadActivity: loadActivity, origin: .chat)
        } else {
            client
        }
        return try await manager.loadModel(
            model: model, modelDir: modelDir, setDefault: setDefault, maxCacheCap: maxCacheCap,
            pinned: pinned, promptLookup: promptLookup, audio: audio, decision: decision
        )
    }

    func unloadModel(model: String, modelDir: String?) async throws -> BackendModelAdminResponse {
        let response = try await client.unloadModel(model: model, modelDir: modelDir)
        loadActivity?.forgetRegistration(of: model)
        return response
    }

    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse {
        try await client.setDefaultModel(model)
    }

    func pinModel(model: String) async throws -> BackendModelAdminResponse {
        try await client.pinModel(model: model)
    }

    func unpinModel(model: String) async throws -> BackendModelAdminResponse {
        try await client.unpinModel(model: model)
    }

    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        try await client.fetchLoadedModels()
    }

    func clearSharedPromptLookup(model: String?) async throws -> BackendPromptLookupClearResponse {
        try await client.clearSharedPromptLookup(model: model)
    }
}

/// Chat's loads go through the same parameters the App restores models
/// with (`BackendRecoveryModel.configured` and `.load`).
struct ConfiguredChatModelLoader: ChatModelLoading {
    let configStore: AppConfigStore
    let scanner: LocalModelScanner
    let parameterStore: ModelParameterStore
    var loadActivity: ModelLoadActivityCenter?

    func configuredModel(_ modelID: String) async throws -> BackendRecoveryModel {
        let config = configStore.load()
        var model = await BackendRecoveryModel.configured(
            modelID, config: config, scanner: scanner, parameterStore: parameterStore
        )
        // A quick check now; `load` verifies again as configured.
        model.modelDir = try await scanner.verifiedModelPathAsync(for: model.modelDir ?? modelID)
        return model
    }

    func load(_ model: BackendRecoveryModel, setDefault: Bool) async throws -> BackendModelAdminResponse {
        let config = configStore.load()
        return try await model.load(
            using: BackendAPIClient(host: config.host, port: config.port),
            setDefault: setDefault,
            scanner: scanner,
            parameterStore: parameterStore,
            fullChecksum: config.verifyModelOnLoad == true,
            loadActivity: loadActivity,
            origin: .chat
        )
    }
}

/// The exclusive DFlash2 server's `/v1/models` (`dflash2_model_list` in
/// `ironmlx/src/server/mod.rs`).
struct BackendDFlash2ModelList: Decodable {
    struct Entry: Decodable, Equatable {
        var id: String
        var contextWindow: Int?
        var maxOutputTokens: Int?
        /// Same contract as `BackendLoadedModelInfo.reasoning`.
        var reasoning: BackendModelReasoning?

        enum CodingKeys: String, CodingKey {
            case id
            case contextWindow = "context_window"
            case maxOutputTokens = "max_output_tokens"
            case reasoning
        }
    }

    var data: [Entry]

    static func entry(for modelID: String, in data: Data) -> Entry? {
        (try? JSONDecoder().decode(Self.self, from: data))?.data.first { $0.id == modelID }
    }
}

/// Decides DFlash2 mode the way the Dashboard does (`configuredDFlash2Target`)
/// and reads the served target from the running server.
struct ConfiguredChatDFlash2Policy: ChatDFlash2Policy {
    let configStore: AppConfigStore
    let parameterStore: ModelParameterStore

    func configuredTarget() async -> String? {
        let config = configStore.load()
        guard let target = config.defaultModelReference,
              parameterStore.parameters(for: target)?.dflash2Enabled == true,
              config.restoredModelReferences.contains(target)
        else { return nil }
        return target
    }

    func usesDFlash2(_ modelID: String) async -> Bool {
        parameterStore.parameters(for: modelID)?.dflash2Enabled == true
    }

    func servedTarget() async throws -> ChatDFlash2Target? {
        let config = configStore.load()
        let client = BackendAPIClient(host: config.host, port: config.port)
        let health = try await client.fetchHealthz()
        guard health.dflash2?.enabled == true,
              let modelID = AppConfig.normalizedModelReference(health.model.name)
        else { return nil }
        let entry = BackendDFlash2ModelList.entry(for: modelID, in: try await client.fetchData(path: "/v1/models"))
        let runtime = LegacyHealthAdapter().legacyStatus(from: health).runtimeModels.first { $0.id == modelID }
        return ChatDFlash2Target(
            modelID: modelID,
            contextWindow: entry?.contextWindow,
            supportsVision: runtime?.capabilities.supportsVision ?? false,
            reasoning: entry?.reasoning
        )
    }
}

extension ChatSessionController {
    /// The App's wiring: history in `~/.ironmlx/chats/`, requests to the
    /// loopback listener, model folders from the local scan, the per-model
    /// output budget from the model parameters, and a changed default model
    /// recorded in `app_config.json` the way a Dashboard load records it.
    static func live(
        configStore: AppConfigStore,
        parameterStore: ModelParameterStore,
        scanner: LocalModelScanner,
        backend: any BackendRuntimeManaging,
        loadActivity: ModelLoadActivityCenter? = nil
    ) -> ChatSessionController {
        let controller = ChatSessionController(
            store: ChatStore(),
            modelManager: ConfiguredBackendModelManager(configStore: configStore, loadActivity: loadActivity),
            transport: URLSessionResponsesTransport(configStore: configStore),
            loader: ConfiguredChatModelLoader(
                configStore: configStore, scanner: scanner, parameterStore: parameterStore,
                loadActivity: loadActivity
            ),
            dflash2: ConfiguredChatDFlash2Policy(configStore: configStore, parameterStore: parameterStore),
            // Joins a launch in progress, which returns after model restore.
            waitForBackend: { [weak backend] in try await backend?.ensureRunning() },
            configuredMaxOutputTokens: { parameterStore.parameters(for: $0)?.defaultMaxOutputTokens },
            catalog: try? SupportedModelCatalog.bundled()
        )
        controller.onDefaultModelChanged = { [weak backend] modelID, loaded in
            let saved = configStore.update { persistDefaultModel(modelID, loaded: loaded, into: &$0) }
            if !saved {
                IronMLXAppLogger.error("Chat could not save \(modelID) as the default model")
            }
            if !loaded.isEmpty {
                backend?.confirmLoadedModels(loaded, parameterConfirmedModelIDs: [modelID])
            }
            NotificationCenter.default.post(name: .ironMLXLoadedModelsDidChange, object: nil)
        }
        return controller
    }

    /// Records the new default and the loaded models the way a Dashboard
    /// load does, but never drops a pin of a model that is still loaded, even
    /// when the backend's list does not report it. An empty list would drop
    /// every other model from the config, so then only the default is recorded.
    nonisolated static func persistDefaultModel(
        _ modelID: String,
        loaded: [BackendLoadedModelInfo],
        into config: inout AppConfig
    ) {
        ChatEntryCoordinator.persistStarterModel(modelID, loaded: loaded.isEmpty ? nil : loaded, into: &config)
    }
}
