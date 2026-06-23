import Foundation

public protocol BackendModelManaging: Sendable {
    func fetchHealthz() async throws -> HealthzSnapshot
    func loadModel(model: String, modelDir: String, setDefault: Bool, maxCacheCap: Int?) async throws -> BackendModelAdminResponse
    func unloadModel(model: String, modelDir: String?) async throws -> BackendModelAdminResponse
    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo]
}

public extension BackendModelManaging {
    func loadModel(model: String, modelDir: String, setDefault: Bool) async throws -> BackendModelAdminResponse {
        try await loadModel(model: model, modelDir: modelDir, setDefault: setDefault, maxCacheCap: nil)
    }
}

public struct BackendAPIClient: Sendable {
    public var host: String
    public var port: UInt16

    public init(host: String, port: UInt16) {
        self.host = host
        self.port = port
    }

    public func fetchData(path: String) async throws -> Data {
        guard let url = URL(string: "http://\(host):\(port)\(path)") else {
            throw URLError(.badURL)
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        return data
    }

    public func postJSON<T: Encodable>(path: String, body: T) async throws -> Data {
        guard let url = URL(string: "http://\(host):\(port)\(path)") else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let body = String(data: data, encoding: .utf8)
            throw BackendAPIError.serverResponse(statusCode: http.statusCode, body: body)
        }
        return data
    }

    public func fetchHealthz() async throws -> HealthzSnapshot {
        let data = try await fetchData(path: "/healthz")
        return try JSONDecoder().decode(HealthzSnapshot.self, from: data)
    }

    public func waitUntilReady(timeout: TimeInterval = 5.0) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        var lastError: Error?
        repeat {
            do {
                _ = try await fetchData(path: "/health")
                return
            } catch {
                lastError = error
                try await Task.sleep(nanoseconds: 200_000_000)
            }
        } while Date() < deadline
        throw lastError ?? URLError(.timedOut)
    }

    public func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool = true,
        maxCacheCap: Int? = nil
    ) async throws -> BackendModelAdminResponse {
        try await loadModel(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: maxCacheCap,
            reloadWhenIdle: false,
            samplingDefaults: .empty
        )
    }

    public func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool = true,
        maxCacheCap: Int? = nil,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        let request = BackendLoadModelRequest(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: maxCacheCap,
            reloadWhenIdle: reloadWhenIdle,
            samplingDefaults: samplingDefaults
        )
        let data = try await postJSON(path: "/admin/api/models/load", body: request)
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func unloadModel(model: String, modelDir: String? = nil) async throws -> BackendModelAdminResponse {
        let request = BackendUnloadModelRequest(model: model, modelDir: modelDir)
        let data = try await postJSON(path: "/admin/api/models/unload", body: request)
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse {
        let data = try await postJSON(
            path: "/admin/api/models/default",
            body: BackendSetDefaultModelRequest(model: model)
        )
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        let data = try await fetchData(path: "/admin/api/models/loaded")
        return try JSONDecoder().decode([BackendLoadedModelInfo].self, from: data)
    }
}

extension BackendAPIClient: BackendModelManaging {}

public enum BackendAPIError: LocalizedError {
    case serverResponse(statusCode: Int, body: String?)

    public var errorDescription: String? {
        switch self {
        case .serverResponse(let statusCode, let body):
            if let body, !body.isEmpty {
                return "Backend returned HTTP \(statusCode): \(body)"
            }
            return "Backend returned HTTP \(statusCode)."
        }
    }
}

public struct BackendLoadModelRequest: Codable, Equatable, Sendable {
    public var model: String
    public var modelDir: String
    public var setDefault: Bool
    public var maxCacheCap: Int?
    public var reloadWhenIdle: Bool?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var repetitionPenalty: Double?

    enum CodingKeys: String, CodingKey {
        case model
        case modelDir = "model_dir"
        case setDefault = "set_default"
        case maxCacheCap = "max_cache_cap"
        case reloadWhenIdle = "reload_when_idle"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
    }

    public init(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int? = nil,
        reloadWhenIdle: Bool? = nil,
        samplingDefaults: BackendSamplingDefaults = .empty
    ) {
        self.model = model
        self.modelDir = modelDir
        self.setDefault = setDefault
        self.maxCacheCap = maxCacheCap
        self.reloadWhenIdle = reloadWhenIdle
        self.temperature = samplingDefaults.temperature
        self.topP = samplingDefaults.topP
        self.topK = samplingDefaults.topK
        self.repetitionPenalty = samplingDefaults.repetitionPenalty
    }
}

public struct BackendSamplingDefaults: Codable, Equatable, Sendable {
    public static let empty = BackendSamplingDefaults()

    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var repetitionPenalty: Double?

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
    }

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        repetitionPenalty: Double? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
    }
}

public struct BackendUnloadModelRequest: Codable, Equatable, Sendable {
    public var model: String
    public var modelDir: String?

    enum CodingKeys: String, CodingKey {
        case model
        case modelDir = "model_dir"
    }
}

public struct BackendSetDefaultModelRequest: Codable, Equatable, Sendable {
    public var model: String
}

public struct BackendModelAdminResponse: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var model: String?
    public var loadedModels: [BackendLoadedModelInfo]
    public var warning: String?
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case model
        case loadedModels = "loaded_models"
        case warning
        case error
    }
}

public struct BackendLoadedModelInfo: Codable, Equatable, Sendable {
    public var id: String
    public var model: String
    public var path: String
    public var architecture: String
    public var isDefault: Bool
    public var maxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case id
        case model
        case path
        case architecture
        case isDefault = "default"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}
