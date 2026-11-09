import Foundation

public protocol BackendModelLoading: Sendable {
    func updateASRSettings(model: String, settings: BackendASRSettings) async throws
    func waitUntilReady(timeout: TimeInterval) async throws
    func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool?,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults,
        audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
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
        samplingDefaults: BackendSamplingDefaults,
        audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse
}

extension BackendAPIClient: BackendModelLoading {}

public protocol BackendModelRestoring: Sendable {
    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult
    /// Restores `snapshot` into the backend launch `launchID`.
    func restore(_ snapshot: BackendRecoverySnapshot, launchID: UUID?) async -> BackendRestartResult
}

extension BackendModelRestoring {
    public func restore(_ snapshot: BackendRecoverySnapshot, launchID: UUID?) async -> BackendRestartResult {
        await restore(snapshot)
    }
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
    /// Models whose restore the user cancelled; they are neither loaded nor
    /// failed, and the App stops listing them as loaded. Not part of the
    /// JSON payload.
    public var cancelledModels: [String] = []

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
    /// Tracks each model load so the Dashboard shows restore progress.
    private let loadActivity: ModelLoadActivityCenter?
    /// Records what this restore registers, so the next rescan skips it.
    private let registrationLedger: LocalModelRegistrationLedger

    public init(
        scanner: LocalModelScanner = LocalModelScanner(),
        parameterStore: ModelParameterStore = .shared,
        clientFactory: @escaping ClientFactory = { host, port in
            BackendAPIClient(host: host, port: port)
        },
        loadActivity: ModelLoadActivityCenter? = nil,
        registrationLedger: LocalModelRegistrationLedger = .shared
    ) {
        self.scanner = scanner
        self.parameterStore = parameterStore
        self.clientFactory = clientFactory
        self.loadActivity = loadActivity
        self.registrationLedger = registrationLedger
    }

    public func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        await restore(snapshot, launchID: nil)
    }

    /// Registers every local model with the launch `launchID`, then loads the
    /// snapshot's models.
    public func restore(_ snapshot: BackendRecoverySnapshot, launchID: UUID?) async -> BackendRestartResult {
        let config = snapshot.config
        if let dflash2Model = snapshot.models.first(where: { $0.dflash2ModelDir != nil }) {
            let invalidShape = snapshot.models.count != 1 || !dflash2Model.isDefault
            if invalidShape {
                return BackendRestartResult(
                    success: false,
                    status: "dflash2_restore_invalid",
                    port: config.port,
                    model: dflash2Model.id,
                    failedModels: snapshot.models.map(\.id),
                    errorCode: "dflash2_requires_exclusive_model",
                    error: "DFlash2 App mode requires exactly one default target model."
                )
            }
            return BackendRestartResult(
                success: true,
                status: "dflash2_model_loaded",
                port: config.port,
                model: dflash2Model.id,
                modelLoaded: true,
                loadedModels: [dflash2Model.id]
            )
        }
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
            client: client,
            ledger: registrationLedger,
            launchID: launchID,
            pass: .restore
        )

        let recoveryModelIDs = Set(snapshot.models.map(\.id))
        let relevantRegistrationFailures = Dictionary(
            uniqueKeysWithValues: registrationFailures
                .filter { recoveryModelIDs.contains($0.model) }
                .map { ($0.model, $0) }
        )
        var loadedModels: [String] = []
        var cancelledModels: [String] = []
        var failures: [BackendModelRecoveryFailure] = []
        let recoveryModels = snapshot.models.sorted(by: Self.recoveryOrder)
        let defaultCandidates = recoveryModels.filter { model in
            let capabilities = model.capabilities ?? localModels.first(where: { $0.id == model.id })?.capabilities
            return !["embedding", "asr", "alignment", "diarization"].contains(capabilities?.runtimeKind ?? "")
        }
        let defaultModelID = defaultCandidates.first(where: \.isDefault)?.id ?? defaultCandidates.first?.id

        for model in recoveryModels {
            do {
                _ = try await model.load(
                    using: client,
                    setDefault: model.id == defaultModelID,
                    scanner: scanner,
                    parameterStore: parameterStore,
                    fullChecksum: config.verifyModelOnLoad == true,
                    loadActivity: loadActivity,
                    origin: .restore
                )
                loadedModels.append(model.id)
            } catch where ModelLoadActivityCenter.isUserCancellation(error) {
                // The user chose not to load it now; that is not a failure.
                cancelledModels.append(model.id)
                IronMLXAppLogger.info("Model restore cancelled by the user: \(model.id)")
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
            var result = BackendRestartResult(
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
            result.cancelledModels = cancelledModels
            return result
        }

        var result = BackendRestartResult(
            success: true,
            status: loadedModels.isEmpty ? "restarted" : "models_loaded",
            port: config.port,
            model: config.defaultModelReference
                ?? recoveryModels.first(where: \.isDefault)?.id
                ?? loadedModels.first,
            modelLoaded: !loadedModels.isEmpty,
            loadedModels: loadedModels
        )
        result.cancelledModels = cancelledModels
        return result
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
