import Foundation

public protocol BackendModelManaging: Sendable {
    func fetchHealthz() async throws -> HealthzSnapshot
    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        promptLookup: BackendPromptLookupConfig?
    ) async throws -> BackendModelAdminResponse
    func unloadModel(model: String, modelDir: String?) async throws -> BackendModelAdminResponse
    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse
    func pinModel(model: String) async throws -> BackendModelAdminResponse
    func unpinModel(model: String) async throws -> BackendModelAdminResponse
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo]
    func clearSharedPromptLookup(model: String?) async throws -> BackendPromptLookupClearResponse
}

public extension BackendModelManaging {
    func loadModel(model: String, modelDir: String, setDefault: Bool) async throws -> BackendModelAdminResponse {
        try await loadModel(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: nil,
            pinned: false,
            promptLookup: nil
        )
    }
}

public struct BackendAPIClient: Sendable {
    public var host: String
    public var port: UInt16

    public init(host: String, port: UInt16) {
        self.host = Self.connectableHost(for: host)
        self.port = port
    }

    public static func connectableHost(for host: String) -> String {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized == "0.0.0.0" || normalized == "::" || normalized == "[::]" {
            return "127.0.0.1"
        }
        return host
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
        maxCacheCap: Int? = nil,
        pinned: Bool = false,
        promptLookup: BackendPromptLookupConfig? = nil
    ) async throws -> BackendModelAdminResponse {
        try await loadModel(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: maxCacheCap,
            pinned: pinned,
            promptLookup: promptLookup,
            reloadWhenIdle: false,
            samplingDefaults: .empty
        )
    }

    public func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool = true,
        maxCacheCap: Int? = nil,
        pinned: Bool = false,
        mtpModelDir: String? = nil,
        mtpDraftTokens: Int? = nil,
        promptLookup: BackendPromptLookupConfig? = nil,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        try await loadModel(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: maxCacheCap,
            pinned: pinned,
            mtpModelDir: mtpModelDir,
            mtpDraftTokens: mtpDraftTokens,
            promptLookup: promptLookup,
            reloadWhenIdle: reloadWhenIdle,
            deferWhenBusy: nil,
            samplingDefaults: samplingDefaults
        )
    }

    public func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool = true,
        maxCacheCap: Int? = nil,
        pinned: Bool = false,
        mtpModelDir: String? = nil,
        mtpDraftTokens: Int? = nil,
        promptLookup: BackendPromptLookupConfig? = nil,
        reloadWhenIdle: Bool,
        deferWhenBusy: Bool? = nil,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        let request = BackendLoadModelRequest(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: maxCacheCap,
            pinned: pinned,
            mtpModelDir: mtpModelDir,
            mtpDraftTokens: mtpDraftTokens,
            promptLookup: promptLookup,
            reloadWhenIdle: reloadWhenIdle,
            deferWhenBusy: deferWhenBusy,
            samplingDefaults: samplingDefaults
        )
        let data = try await postJSON(path: "/admin/api/models/load", body: request)
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool = false,
        maxCacheCap: Int? = nil,
        pinned: Bool = false,
        mtpModelDir: String? = nil,
        mtpDraftTokens: Int? = nil,
        promptLookup: BackendPromptLookupConfig? = nil,
        samplingDefaults: BackendSamplingDefaults = .empty
    ) async throws -> BackendModelAdminResponse {
        let request = BackendLoadModelRequest(
            model: model,
            modelDir: modelDir,
            setDefault: setDefault,
            maxCacheCap: maxCacheCap,
            pinned: pinned,
            mtpModelDir: mtpModelDir,
            mtpDraftTokens: mtpDraftTokens,
            promptLookup: promptLookup,
            samplingDefaults: samplingDefaults
        )
        let data = try await postJSON(path: "/admin/api/models/register", body: request)
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func unloadModel(model: String, modelDir: String? = nil) async throws -> BackendModelAdminResponse {
        let request = BackendUnloadModelRequest(model: model, modelDir: modelDir)
        let data = try await postJSON(path: "/admin/api/models/unload", body: request)
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func pinModel(model: String) async throws -> BackendModelAdminResponse {
        let request = BackendPinModelRequest(model: model)
        let data = try await postJSON(path: "/admin/api/models/pin", body: request)
        return try JSONDecoder().decode(BackendModelAdminResponse.self, from: data)
    }

    public func unpinModel(model: String) async throws -> BackendModelAdminResponse {
        let request = BackendPinModelRequest(model: model)
        let data = try await postJSON(path: "/admin/api/models/unpin", body: request)
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

    public func clearSharedPromptLookup(
        model: String? = nil
    ) async throws -> BackendPromptLookupClearResponse {
        let data = try await postJSON(
            path: "/admin/api/prompt-lookup/clear",
            body: BackendPromptLookupClearRequest(model: model)
        )
        return try JSONDecoder().decode(BackendPromptLookupClearResponse.self, from: data)
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
    public var pinned: Bool?
    public var mtpModelDir: String?
    public var mtpDraftTokens: Int?
    public var promptLookup: BackendPromptLookupConfig?
    public var reloadWhenIdle: Bool?
    public var deferWhenBusy: Bool?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var repetitionPenalty: Double?

    enum CodingKeys: String, CodingKey {
        case model
        case modelDir = "model_dir"
        case setDefault = "set_default"
        case maxCacheCap = "max_cache_cap"
        case pinned
        case mtpModelDir = "mtp_model_dir"
        case mtpDraftTokens = "mtp_draft_tokens"
        case promptLookup = "prompt_lookup"
        case reloadWhenIdle = "reload_when_idle"
        case deferWhenBusy = "defer_when_busy"
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
        pinned: Bool? = nil,
        mtpModelDir: String? = nil,
        mtpDraftTokens: Int? = nil,
        promptLookup: BackendPromptLookupConfig? = nil,
        reloadWhenIdle: Bool? = nil,
        deferWhenBusy: Bool? = nil,
        samplingDefaults: BackendSamplingDefaults = .empty
    ) {
        self.model = model
        self.modelDir = modelDir
        self.setDefault = setDefault
        self.maxCacheCap = maxCacheCap
        self.pinned = pinned
        self.mtpModelDir = mtpModelDir
        self.mtpDraftTokens = mtpDraftTokens
        self.promptLookup = promptLookup
        self.reloadWhenIdle = reloadWhenIdle
        self.deferWhenBusy = deferWhenBusy
        self.temperature = samplingDefaults.temperature
        self.topP = samplingDefaults.topP
        self.topK = samplingDefaults.topK
        self.repetitionPenalty = samplingDefaults.repetitionPenalty
    }
}

public struct BackendPromptLookupConfig: Codable, Equatable, Sendable {
    public static let requestLocal = BackendPromptLookupConfig(crossRequest: false)
    public static let crossRequest = BackendPromptLookupConfig(crossRequest: true)

    public var minNgram: Int
    public var maxNgram: Int
    public var maxDraftTokens: Int
    public var historyWindowTokens: Int
    public var maxIndexEntries: Int
    public var crossRequest: Bool

    enum CodingKeys: String, CodingKey {
        case minNgram = "min_ngram"
        case maxNgram = "max_ngram"
        case maxDraftTokens = "max_draft_tokens"
        case historyWindowTokens = "history_window_tokens"
        case maxIndexEntries = "max_index_entries"
        case crossRequest = "cross_request"
    }

    public init(
        minNgram: Int = 2,
        maxNgram: Int = 4,
        maxDraftTokens: Int = 4,
        historyWindowTokens: Int = 32 * 1_024,
        maxIndexEntries: Int = 64 * 1_024,
        crossRequest: Bool
    ) {
        self.minNgram = minNgram
        self.maxNgram = maxNgram
        self.maxDraftTokens = maxDraftTokens
        self.historyWindowTokens = historyWindowTokens
        self.maxIndexEntries = maxIndexEntries
        self.crossRequest = crossRequest
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

public struct BackendPinModelRequest: Codable, Equatable, Sendable {
    public var model: String
}

public struct BackendPromptLookupClearRequest: Codable, Equatable, Sendable {
    public var model: String?
}

public struct BackendPromptLookupClearResponse: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var model: String?
    public var clearedModels: Int
    public var clearedEntries: Int

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case model
        case clearedModels = "cleared_models"
        case clearedEntries = "cleared_entries"
    }
}

public struct BackendModelAdminResponse: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var code: String?
    public var model: String?
    public var loadedModels: [BackendLoadedModelInfo]
    public var warningCode: String?
    public var warning: String?
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case code
        case model
        case loadedModels = "loaded_models"
        case warningCode = "warning_code"
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
    public var pinned: Bool
    public var mtpEnabled: Bool
    public var mtpModelDir: String?
    public var mtpDraftTokens: Int?
    public var promptLookup: BackendPromptLookupConfig?

    public init(
        id: String,
        model: String,
        path: String,
        architecture: String,
        isDefault: Bool,
        maxPositionEmbeddings: Int,
        pinned: Bool = false,
        mtpEnabled: Bool = false,
        mtpModelDir: String? = nil,
        mtpDraftTokens: Int? = nil,
        promptLookup: BackendPromptLookupConfig? = nil
    ) {
        self.id = id
        self.model = model
        self.path = path
        self.architecture = architecture
        self.isDefault = isDefault
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.pinned = pinned
        self.mtpEnabled = mtpEnabled
        self.mtpModelDir = mtpModelDir
        self.mtpDraftTokens = mtpDraftTokens
        self.promptLookup = promptLookup
    }

    enum CodingKeys: String, CodingKey {
        case id
        case model
        case path
        case architecture
        case isDefault = "default"
        case maxPositionEmbeddings = "max_position_embeddings"
        case pinned
        case mtpEnabled = "mtp_enabled"
        case mtpModelDir = "mtp_model_dir"
        case mtpDraftTokens = "mtp_draft_tokens"
        case promptLookup = "prompt_lookup"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.model = try container.decode(String.self, forKey: .model)
        self.path = try container.decode(String.self, forKey: .path)
        self.architecture = try container.decode(String.self, forKey: .architecture)
        self.isDefault = try container.decode(Bool.self, forKey: .isDefault)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.pinned = try container.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
        self.mtpEnabled = try container.decodeIfPresent(Bool.self, forKey: .mtpEnabled) ?? false
        self.mtpModelDir = try container.decodeIfPresent(String.self, forKey: .mtpModelDir)
        self.mtpDraftTokens = try container.decodeIfPresent(Int.self, forKey: .mtpDraftTokens)
        self.promptLookup = try container.decodeIfPresent(
            BackendPromptLookupConfig.self,
            forKey: .promptLookup
        )
    }
}
