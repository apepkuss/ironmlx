import Foundation

public struct BackendRecoveryModel: Codable, Equatable, Sendable {
    public var id: String
    public var modelDir: String?
    public var isDefault: Bool
    public var pinned: Bool
    public var maxCacheCap: Int?
    public var mtpModelDir: String?
    public var mtpDraftTokens: Int?
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
        promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults,
        capabilities: BackendModelCapabilities? = nil
    ) {
        self.id = id
        self.modelDir = modelDir
        self.isDefault = isDefault
        self.pinned = pinned
        self.maxCacheCap = maxCacheCap
        self.mtpModelDir = mtpModelDir
        self.mtpDraftTokens = mtpDraftTokens
        self.promptLookup = promptLookup
        self.samplingDefaults = samplingDefaults
        self.capabilities = capabilities
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
        let defaultModel = config.defaultModelReference
        let pinned = Set(config.pinnedModelReferences)
        let models = config.restoredModelReferences.map { model in
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
                promptLookup: supportsPromptLookup ? parameters?.promptLookupConfig : nil,
                samplingDefaults: (parameters?.samplingDefaults ?? .empty)
                    .filtered(for: capabilities),
                capabilities: capabilities
            )
        }
        return BackendRecoverySnapshot(config: config, models: models)
    }
}
