import Foundation

public struct BackendRecoveryModel: Codable, Equatable, Sendable {
    public var decision: BackendDecisionSettings?
    public var audio: BackendAudioResources?
    public var id: String
    public var modelDir: String?
    public var isDefault: Bool
    public var pinned: Bool
    public var maxCacheCap: Int?
    public var mtpModelDir: String?
    public var mtpDraftTokens: Int?
    public var dflash2ModelDir: String?
    public var dflash2BlockSize: Int?
    public var dflash2DraftBits: Int?
    public var dflash2TensorBatchMaxWidth: Int?
    public var promptLookup: BackendPromptLookupConfig?
    public var samplingDefaults: BackendSamplingDefaults
    public var capabilities: BackendModelCapabilities?

    public init(
        id: String,
        modelDir: String?,
        isDefault: Bool,
        pinned: Bool,
        maxCacheCap: Int?,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        dflash2ModelDir: String? = nil,
        dflash2BlockSize: Int? = nil,
        dflash2DraftBits: Int? = nil,
        dflash2TensorBatchMaxWidth: Int? = nil,
        promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults,
        capabilities: BackendModelCapabilities? = nil,
        audio: BackendAudioResources? = nil,
        decision: BackendDecisionSettings? = nil
    ) {
        self.id = id
        self.modelDir = modelDir
        self.isDefault = isDefault
        self.pinned = pinned
        self.maxCacheCap = maxCacheCap
        self.mtpModelDir = mtpModelDir
        self.mtpDraftTokens = mtpDraftTokens
        self.dflash2ModelDir = dflash2ModelDir
        self.dflash2BlockSize = dflash2BlockSize
        self.dflash2DraftBits = dflash2DraftBits
        self.dflash2TensorBatchMaxWidth = dflash2TensorBatchMaxWidth
        self.promptLookup = promptLookup
        self.samplingDefaults = samplingDefaults
        self.capabilities = capabilities
        self.audio = audio
        self.decision = decision
    }
}

public struct BackendRecoverySnapshot: Codable, Equatable, Sendable {
    public var capturedAt: Date
    public var config: AppConfig
    public var models: [BackendRecoveryModel]

    public init(capturedAt: Date = Date(), config: AppConfig, models: [BackendRecoveryModel]) {
        self.capturedAt = capturedAt
        self.config = config
        self.models = models
    }

    @MainActor
    public static func capture(
        config: AppConfig,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore
    ) -> BackendRecoverySnapshot {
        let models = config.restoredModelReferences.map { model in
            BackendRecoveryModel.configured(model, config: config, scanner: scanner, parameterStore: parameterStore)
        }
        return BackendRecoverySnapshot(config: config, models: models)
    }
}

extension BackendRecoveryModel {
    /// The App's load parameters for `model` from the config and the model
    /// parameters: folder, default and pin, cache cap, MTP, DFlash2, prompt
    /// lookup, sampling defaults, audio resources and decision settings.
    @MainActor
    public static func configured(
        _ model: String,
        config: AppConfig,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore
    ) -> BackendRecoveryModel {
        let defaultModel = config.defaultModelReference
        let pinned = Set(config.pinnedModelReferences)
        let parameters = parameterStore.parameters(for: model)
        let capabilities = scanner.model(for: model)?.capabilities
        let supportsKvCache = capabilities?.supportsKvCache != false
        let supportsMtp = capabilities?.supportsMtp != false
        let supportsPromptLookup = capabilities?.supportsPromptLookup != false
        let mtpRuntime = supportsMtp
            ? try? ModelMtpRuntimeResolver.runtime(
                for: model,
                useMtp: parameters?.mtpEnabled,
                explicitMtpModelID: parameters?.mtpModelID,
                scanner: scanner,
                parameterStore: parameterStore
            )
            : nil
        let dflash2Runtime = try? ModelDFlash2RuntimeResolver.runtime(
            for: model,
            useDFlash2: parameters?.dflash2Enabled,
            scanner: scanner,
            parameterStore: parameterStore
        )
        return BackendRecoveryModel(
            id: model,
            modelDir: scanner.resolveModelPath(for: model),
            isDefault: model == defaultModel,
            pinned: pinned.contains(model),
            maxCacheCap: supportsKvCache
                ? ModelLoadParameters.maxCacheCap(
                    for: model,
                    scanner: scanner,
                    parameterStore: parameterStore,
                    activeKvOffloadEnabled: config.activeKvOffload == true
                )
                : nil,
            mtpModelDir: mtpRuntime?.modelDir,
            mtpDraftTokens: mtpRuntime?.draftTokens,
            dflash2ModelDir: dflash2Runtime?.draftModelDir,
            dflash2BlockSize: dflash2Runtime?.blockSize,
            dflash2DraftBits: dflash2Runtime?.draftBits,
            dflash2TensorBatchMaxWidth: dflash2Runtime?.tensorBatchMaxWidth,
            promptLookup: supportsPromptLookup ? parameters?.promptLookupConfig : nil,
            samplingDefaults: (parameters?.samplingDefaults ?? .empty)
                .filtered(for: capabilities),
            capabilities: capabilities,
            audio: try? scanner.audioResources(
                for: model,
                execution: parameters?.audioExecution
            ),
            decision: parameters?.decision
        )
    }

    /// `loaded` as the backend runs it now, keeping its folder, pin, MTP,
    /// prompt lookup and decision settings; cache cap, sampling defaults and
    /// audio resources come from `configured` when known.
    public static func reloading(
        _ loaded: BackendLoadedModelInfo,
        configured: BackendRecoveryModel?
    ) -> BackendRecoveryModel {
        BackendRecoveryModel(
            id: loaded.id,
            modelDir: loaded.path,
            isDefault: loaded.isDefault,
            pinned: loaded.pinned,
            maxCacheCap: configured?.maxCacheCap,
            mtpModelDir: loaded.mtpModelDir,
            mtpDraftTokens: loaded.mtpDraftTokens,
            promptLookup: loaded.promptLookup,
            samplingDefaults: configured?.samplingDefaults ?? .empty,
            capabilities: loaded.capabilities,
            audio: configured?.audio,
            decision: loaded.decision
        )
    }

    /// Loads this model the way the App restores it after a restart: the
    /// folder is verified first, and every configured parameter is sent.
    /// With `loadActivity` the load is tracked (verification included) and
    /// runs asynchronously in the backend; `client` then only serves as the
    /// fallback when no center is given.
    public func load(
        using client: any BackendModelLoading,
        setDefault: Bool,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool,
        loadActivity: ModelLoadActivityCenter? = nil,
        origin: ModelLoadOrigin = .restore
    ) async throws -> BackendModelAdminResponse {
        guard let loadActivity else {
            let resolvedModel = try await scanner.verifiedModelPathAsync(
                for: modelDir ?? id,
                fullChecksum: fullChecksum
            )
            return try await client.loadModel(
                model: id,
                modelDir: resolvedModel,
                setDefault: setDefault,
                maxCacheCap: maxCacheCap,
                pinned: pinned,
                mtpModelDir: mtpModelDir,
                mtpDraftTokens: mtpDraftTokens,
                promptLookup: promptLookup,
                reloadWhenIdle: false,
                samplingDefaults: samplingDefaults,
                audio: try scanner.audioResources(
                    for: id,
                    execution: parameterStore.parameters(for: id)?.audioExecution
                ) ?? audio,
                decision: decision
            )
        }
        let model = self
        return try await loadActivity.track(
            model: id,
            variant: mtpModelDir == nil ? .plain : .mtp,
            origin: origin
        ) { ticket in
            loadActivity.enterAppStage(ticket, .verifyingFiles)
            let resolvedModel = try await scanner.verifiedModelPathAsync(
                for: model.modelDir ?? model.id,
                fullChecksum: fullChecksum
            )
            let request = BackendLoadModelRequest(
                model: model.id,
                modelDir: resolvedModel,
                setDefault: setDefault,
                maxCacheCap: model.maxCacheCap,
                pinned: model.pinned,
                mtpModelDir: model.mtpModelDir,
                mtpDraftTokens: model.mtpDraftTokens,
                promptLookup: model.promptLookup,
                reloadWhenIdle: false,
                samplingDefaults: model.samplingDefaults,
                audio: try scanner.audioResources(
                    for: model.id,
                    execution: parameterStore.parameters(for: model.id)?.audioExecution
                ) ?? model.audio,
                decision: model.decision
            )
            return try await loadActivity.loadThroughBackend(ticket, request: request)
        }
    }
}
