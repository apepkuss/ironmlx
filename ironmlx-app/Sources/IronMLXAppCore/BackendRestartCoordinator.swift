import Foundation

@MainActor
public protocol BackendProcessManaging: AnyObject {
    var isRunning: Bool { get }

    func start(modelReference: String) throws
    func stop()
}

extension BackendProcessManager: BackendProcessManaging {}

public protocol BackendModelLoading: Sendable {
    func waitUntilReady(timeout: TimeInterval) async throws
    func loadModel(model: String, modelDir: String, setDefault: Bool, maxCacheCap: Int?) async throws -> BackendModelAdminResponse
    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse
}

public extension BackendModelLoading {
    func loadModel(model: String, modelDir: String, setDefault: Bool) async throws -> BackendModelAdminResponse {
        try await loadModel(model: model, modelDir: modelDir, setDefault: setDefault, maxCacheCap: nil)
    }

    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        try await loadModel(model: model, modelDir: modelDir, setDefault: setDefault, maxCacheCap: maxCacheCap)
    }
}

extension BackendAPIClient: BackendModelLoading {}

public struct BackendRestartResult: Codable, Equatable {
    public var success: Bool
    public var status: String
    public var port: UInt16
    public var model: String?
    public var modelLoaded: Bool
    public var error: String?

    public init(
        success: Bool,
        status: String,
        port: UInt16,
        model: String? = nil,
        modelLoaded: Bool = false,
        error: String? = nil
    ) {
        self.success = success
        self.status = status
        self.port = port
        self.model = model
        self.modelLoaded = modelLoaded
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case port
        case model
        case modelLoaded = "model_loaded"
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

        guard let model = config.lastModel?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty
        else {
            return BackendRestartResult(
                success: true,
                status: "restarted",
                port: config.port
            )
        }

        do {
            try backend.start(modelReference: model)
            let client = clientFactory(config.host, config.port)
            try await client.waitUntilReady(timeout: 5.0)
            let resolvedModel = scanner.resolveModelPath(for: model) ?? model
            let maxCacheCap = ModelLoadParameters.maxCacheCap(
                for: model,
                scanner: scanner,
                parameterStore: parameterStore
            )
            _ = try await client.loadModel(
                model: model,
                modelDir: resolvedModel,
                setDefault: true,
                maxCacheCap: maxCacheCap,
                reloadWhenIdle: false,
                samplingDefaults: parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
            )
            return BackendRestartResult(
                success: true,
                status: "model_loaded",
                port: config.port,
                model: model,
                modelLoaded: true
            )
        } catch {
            let message = Self.errorMessage(from: error)
            IronMLXAppLogger.error("Failed to restart backend and load default model: \(message)")
            return BackendRestartResult(
                success: false,
                status: "model_load_failed",
                port: config.port,
                model: model,
                modelLoaded: false,
                error: message
            )
        }
    }

    private static func errorMessage(from error: Error) -> String {
        if case BackendAPIError.serverResponse(statusCode: _, body: let body) = error,
           let body,
           let data = body.data(using: .utf8),
           let payload = try? JSONDecoder().decode(BackendErrorPayload.self, from: data),
           let message = payload.error?.trimmingCharacters(in: .whitespacesAndNewlines),
           !message.isEmpty {
            return message
        }
        return error.localizedDescription
    }

    private struct BackendErrorPayload: Decodable {
        var error: String?
    }
}
