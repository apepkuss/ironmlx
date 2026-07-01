import Foundation

@MainActor
public protocol BackendProcessManaging: AnyObject {
    var isRunning: Bool { get }

    func start() throws
    func stop()
}

extension BackendProcessManager: BackendProcessManaging {}

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
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse
}

extension BackendAPIClient: BackendModelLoading {}

public struct BackendRestartResult: Codable, Equatable {
    public var success: Bool
    public var status: String
    public var port: UInt16
    public var model: String?
    public var modelLoaded: Bool
    public var loadedModels: [String]
    public var failedModels: [String]
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
        case errorCode = "code"
        case error
    }
}

public struct BackendRestartCoordinator {
    public typealias ClientFactory = (String, UInt16) -> any BackendModelLoading

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

    @MainActor
    public func restartDefaultModel(
        config: AppConfig,
        backend: BackendProcessManaging
    ) async -> BackendRestartResult {
        backend.stop()

        let models = config.restoredModelReferences
        let pinnedModels = Set(config.pinnedModelReferences)
        let localModels = scanner.scan(loadedModels: Set(models), pinnedModels: pinnedModels, mtpEnabledModels: [])
        guard !models.isEmpty || !localModels.isEmpty else {
            return BackendRestartResult(
                success: true,
                status: "restarted",
                port: config.port
            )
        }

        do {
            try backend.start()
            let client = clientFactory(config.host, config.port)
            try await client.waitUntilReady(timeout: 5.0)
            await LocalModelBackendRegistrar.register(
                localModels: localModels,
                defaultModel: config.defaultModelReference,
                scanner: scanner,
                parameterStore: parameterStore,
                client: client
            )
            var loadedModels: [String] = []
            var failedModels: [String] = []
            var lastError: String?
            var lastErrorCode: String?
            for model in models {
                do {
                    let resolvedModel = scanner.resolveModelPath(for: model) ?? model
                    let maxCacheCap = ModelLoadParameters.maxCacheCap(
                        for: model,
                        scanner: scanner,
                        parameterStore: parameterStore
                    )
                    let mtpRuntime = try? ModelMtpRuntimeResolver.runtime(
                        for: model,
                        useMtp: nil,
                        scanner: scanner,
                        parameterStore: parameterStore
                    )
                    _ = try await client.loadModel(
                        model: model,
                        modelDir: resolvedModel,
                        setDefault: model == config.defaultModelReference
                            || config.defaultModelReference == nil && loadedModels.isEmpty,
                        maxCacheCap: maxCacheCap,
                        pinned: pinnedModels.contains(model),
                        mtpModelDir: mtpRuntime?.modelDir,
                        mtpDraftTokens: mtpRuntime?.draftTokens,
                        reloadWhenIdle: false,
                        samplingDefaults: parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
                    )
                    loadedModels.append(model)
                } catch {
                    let details = Self.errorDetails(from: error)
                    failedModels.append(model)
                    lastError = details.message
                    lastErrorCode = details.code
                    IronMLXAppLogger.error("Failed to restore model after backend restart: \(model): \(details.message)")
                }
            }
            if loadedModels.isEmpty, let lastError {
                return BackendRestartResult(
                    success: false,
                    status: "model_load_failed",
                    port: config.port,
                    model: models.first,
                    modelLoaded: false,
                    loadedModels: loadedModels,
                    failedModels: failedModels,
                    errorCode: lastErrorCode,
                    error: lastError
                )
            }
            return BackendRestartResult(
                success: true,
                status: loadedModels.isEmpty ? "restarted" : "models_loaded",
                port: config.port,
                model: config.defaultModelReference ?? loadedModels.first,
                modelLoaded: !loadedModels.isEmpty,
                loadedModels: loadedModels,
                failedModels: failedModels,
                errorCode: failedModels.isEmpty ? nil : lastErrorCode,
                error: failedModels.isEmpty ? nil : lastError
            )
        } catch {
            let details = Self.errorDetails(from: error)
            IronMLXAppLogger.error("Failed to restart backend and restore models: \(details.message)")
            return BackendRestartResult(
                success: false,
                status: "model_load_failed",
                port: config.port,
                model: models.first,
                modelLoaded: false,
                loadedModels: [],
                failedModels: models,
                errorCode: details.code,
                error: details.message
            )
        }
    }

    private static func errorDetails(from error: Error) -> BackendErrorDetails {
        if case BackendAPIError.serverResponse(statusCode: _, body: let body) = error,
           let body,
           let data = body.data(using: .utf8),
           let payload = try? JSONDecoder().decode(BackendErrorPayload.self, from: data),
           let message = payload.error?.trimmingCharacters(in: .whitespacesAndNewlines),
           !message.isEmpty {
            return BackendErrorDetails(message: message, code: payload.code)
        }
        return BackendErrorDetails(message: error.localizedDescription, code: nil)
    }

    private struct BackendErrorDetails {
        var message: String
        var code: String?
    }

    private struct BackendErrorPayload: Decodable {
        var error: String?
        var code: String?
    }
}
