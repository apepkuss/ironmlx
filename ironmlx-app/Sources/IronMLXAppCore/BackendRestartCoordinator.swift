import Foundation

public protocol BackendModelLoading: Sendable {
    func waitUntilReady(timeout: TimeInterval) async throws
    func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse
    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        promptLookup: BackendPromptLookupConfig?,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse
}

extension BackendAPIClient: BackendModelLoading {}

public protocol BackendModelRestoring: Sendable {
    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult
}

public struct BackendRestartResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var port: UInt16
    public var model: String?
    public var modelLoaded: Bool
    public var loadedModels: [String]
    public var failedModels: [String]
    public var failures: [BackendModelRecoveryFailure]
    public var errorCode: String?
    public var error: String?

    public init(
        success: Bool,
        status: String,
        port: UInt16,
        model: String? = nil,
        modelLoaded: Bool = false,
        loadedModels: [String] = [],
        failedModels: [String] = [],
        failures: [BackendModelRecoveryFailure] = [],
        errorCode: String? = nil,
        error: String? = nil
    ) {
        self.success = success
        self.status = status
        self.port = port
        self.model = model
        self.modelLoaded = modelLoaded
        self.loadedModels = loadedModels
        self.failedModels = failedModels
        self.failures = failures
        self.errorCode = errorCode
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case port
        case model
        case modelLoaded = "model_loaded"
        case loadedModels = "loaded_models"
        case failedModels = "failed_models"
        case failures
        case errorCode = "code"
        case error
    }
}

public struct BackendRestartCoordinator: Sendable {
    public typealias ClientFactory = @Sendable (String, UInt16) -> any BackendModelLoading

    private let scanner: LocalModelScanner
    private let parameterStore: ModelParameterStore
    private let clientFactory: ClientFactory

    public init(
        scanner: LocalModelScanner = LocalModelScanner(),
        parameterStore: ModelParameterStore = .shared,
        clientFactory: @escaping ClientFactory = { host, port in
            BackendAPIClient(host: host, port: port)
        }
    ) {
        self.scanner = scanner
        self.parameterStore = parameterStore
        self.clientFactory = clientFactory
    }

    public func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        let config = snapshot.config
        let client = clientFactory(config.host, config.port)
        let pinnedModels = Set(config.pinnedModelReferences)
        let localModels = scanner.scan(
            loadedModels: Set(snapshot.models.map(\.id)),
            pinnedModels: pinnedModels,
            mtpEnabledModels: []
        )
        let registrationFailures = await LocalModelBackendRegistrar.register(
            localModels: localModels,
            defaultModel: config.defaultModelReference,
            scanner: scanner,
            parameterStore: parameterStore,
            activeKvOffloadEnabled: config.activeKvOffload == true,
            client: client
        )

        let recoveryModelIDs = Set(snapshot.models.map(\.id))
        let relevantRegistrationFailures = Dictionary(
            uniqueKeysWithValues: registrationFailures
                .filter { recoveryModelIDs.contains($0.model) }
                .map { ($0.model, $0) }
        )
        var loadedModels: [String] = []
        var failures: [BackendModelRecoveryFailure] = []
        let recoveryModels = snapshot.models.sorted(by: Self.recoveryOrder)
        let hasConfirmedDefault = recoveryModels.contains(where: \.isDefault)

        for (index, model) in recoveryModels.enumerated() {
            do {
                let resolvedModel = try await scanner.verifiedModelPathAsync(
                    for: model.modelDir ?? model.id,
                    fullChecksum: config.verifyModelOnLoad == true
                )
                _ = try await client.loadModel(
                    model: model.id,
                    modelDir: resolvedModel,
                    setDefault: model.isDefault || !hasConfirmedDefault && index == 0,
                    maxCacheCap: model.maxCacheCap,
                    pinned: model.pinned,
                    mtpModelDir: model.mtpModelDir,
                    mtpDraftTokens: model.mtpDraftTokens,
                    promptLookup: model.promptLookup,
                    reloadWhenIdle: false,
                    samplingDefaults: model.samplingDefaults
                )
                loadedModels.append(model.id)
            } catch {
                let failure = BackendRecoveryFailureClassifier.failure(
                    model: model.id,
                    stage: .loading,
                    error: error
                )
                failures.append(failure)
                IronMLXAppLogger.error(
                    "Failed to restore model after backend restart: \(model.id): \(failure.message)"
                )
            }
        }

        let loadedModelIDs = Set(loadedModels)
        for failure in relevantRegistrationFailures.values
        where !loadedModelIDs.contains(failure.model)
            && !failures.contains(where: { $0.model == failure.model }) {
            failures.append(failure)
        }
        failures.sort { lhs, rhs in
            if lhs.model != rhs.model {
                return lhs.model < rhs.model
            }
            return lhs.stage.rawValue < rhs.stage.rawValue
        }
        let failedModels = AppConfig.normalizedModelReferences(failures.map(\.model))
        if !failedModels.isEmpty {
            let partial = !loadedModels.isEmpty
            let primaryFailure = failures.first
            return BackendRestartResult(
                success: false,
                status: partial ? "models_partially_loaded" : "model_load_failed",
                port: config.port,
                model: config.defaultModelReference
                    ?? recoveryModels.first(where: \.isDefault)?.id
                    ?? loadedModels.first,
                modelLoaded: !loadedModels.isEmpty,
                loadedModels: loadedModels,
                failedModels: failedModels,
                failures: failures,
                errorCode: primaryFailure?.code,
                error: primaryFailure?.message
            )
        }

        return BackendRestartResult(
            success: true,
            status: loadedModels.isEmpty ? "restarted" : "models_loaded",
            port: config.port,
            model: config.defaultModelReference
                ?? recoveryModels.first(where: \.isDefault)?.id
                ?? loadedModels.first,
            modelLoaded: !loadedModels.isEmpty,
            loadedModels: loadedModels
        )
    }

    private static func recoveryOrder(
        _ lhs: BackendRecoveryModel,
        _ rhs: BackendRecoveryModel
    ) -> Bool {
        if lhs.isDefault != rhs.isDefault {
            return lhs.isDefault
        }
        return lhs.id < rhs.id
    }

}

extension BackendRestartCoordinator: BackendModelRestoring {}
