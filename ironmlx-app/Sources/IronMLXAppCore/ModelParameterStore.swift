import Foundation

public struct ModelParameters: Codable, Equatable, Sendable {
    public var modelID: String
    public var alias: String?
    public var modelType: String?
    public var contextSize: String?
    public var maxTokens: String?
    public var temperature: String?
    public var topP: String?
    public var topK: String?
    public var repeatPenalty: String?
    public var mtpEnabled: Bool?
    public var mtpModelID: String?
    public var mtpDraftTokens: String?

    public init(
        modelID: String,
        alias: String? = nil,
        modelType: String? = nil,
        contextSize: String? = nil,
        maxTokens: String? = nil,
        temperature: String? = nil,
        topP: String? = nil,
        topK: String? = nil,
        repeatPenalty: String? = nil,
        mtpEnabled: Bool? = nil,
        mtpModelID: String? = nil,
        mtpDraftTokens: String? = nil
    ) {
        self.modelID = modelID
        self.alias = alias
        self.modelType = modelType
        self.contextSize = contextSize
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.mtpEnabled = mtpEnabled
        self.mtpModelID = mtpModelID
        self.mtpDraftTokens = mtpDraftTokens
    }

    public var maxCacheCap: Int? {
        guard let value = positiveInt(maxTokens) else {
            return nil
        }
        return value
    }

    public var samplingDefaults: BackendSamplingDefaults {
        BackendSamplingDefaults(
            temperature: positiveDouble(temperature),
            topP: probability(topP),
            topK: positiveInt(topK),
            repetitionPenalty: positiveDouble(repeatPenalty)
        )
    }

    public var mtpDraftTokensValue: Int? {
        positiveInt(mtpDraftTokens)
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case alias
        case modelType = "model_type"
        case contextSize = "context_size"
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repeatPenalty = "repeat_penalty"
        case mtpEnabled = "mtp_enabled"
        case mtpModelID = "mtp_model_id"
        case mtpDraftTokens = "mtp_draft_tokens"
    }

    private func positiveInt(_ value: String?) -> Int? {
        guard let value else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsed = Int(trimmed), parsed > 0 else {
            return nil
        }
        return parsed
    }

    private func positiveDouble(_ value: String?) -> Double? {
        guard let value else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsed = Double(trimmed), parsed > 0 else {
            return nil
        }
        return parsed
    }

    private func probability(_ value: String?) -> Double? {
        guard let parsed = positiveDouble(value), parsed <= 1 else {
            return nil
        }
        return parsed
    }
}

public final class ModelParameterStore: @unchecked Sendable {
    public static let shared = ModelParameterStore()

    public let url: URL
    private let fileManager: FileManager

    public init(
        url: URL = ModelParameterStore.defaultURL(),
        fileManager: FileManager = .default
    ) {
        self.url = url
        self.fileManager = fileManager
    }

    public func loadAll() throws -> [String: ModelParameters] {
        guard fileManager.fileExists(atPath: url.path) else {
            return [:]
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: ModelParameters].self, from: data)
    }

    public func parameters(for modelID: String) -> ModelParameters? {
        let key = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            return nil
        }
        return try? loadAll()[key]
    }

    public func save(_ parameters: ModelParameters) throws {
        let key = parameters.modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            return
        }
        var all = try loadAll()
        all[key] = parameters
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder.prettyIronMLX.encode(all)
        try data.write(to: url, options: .atomic)
    }

    public func recordMtpLoadPreference(
        modelID: String,
        enabled: Bool,
        mtpModelID: String?
    ) throws {
        let key = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            return
        }
        var all = try loadAll()
        var parameters = all[key] ?? ModelParameters(modelID: key)
        parameters.mtpEnabled = enabled
        let selectedMtp = mtpModelID?.trimmingCharacters(in: .whitespacesAndNewlines)
        if enabled, let selectedMtp, !selectedMtp.isEmpty {
            parameters.mtpModelID = selectedMtp
        }
        all[key] = parameters
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder.prettyIronMLX.encode(all)
        try data.write(to: url, options: .atomic)
    }

    public func jsonString() -> String {
        let all = (try? loadAll()) ?? [:]
        guard let data = try? JSONEncoder().encode(all),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }

    public static func defaultURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("model_params.json")
    }
}

public enum ModelLoadParameters {
    public static let conservativeLongContextCap = 32_768

    public static func maxCacheCap(
        for modelReference: String,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        activeKvOffloadEnabled: Bool
    ) -> Int? {
        effectiveMaxCacheCap(
            savedMaxCacheCap: parameterStore.parameters(for: modelReference)?.maxCacheCap,
            contextWindow: scanner.maxPositionEmbeddings(for: modelReference),
            activeKvOffloadEnabled: activeKvOffloadEnabled
        )
    }

    public static func effectiveMaxCacheCap(
        savedMaxCacheCap: Int?,
        contextWindow: Int?,
        activeKvOffloadEnabled: Bool
    ) -> Int? {
        if let savedMaxCacheCap {
            return savedMaxCacheCap
        }
        guard let contextWindow else {
            return nil
        }
        return defaultMaxCacheCap(
            contextWindow: contextWindow,
            activeKvOffloadEnabled: activeKvOffloadEnabled
        )
    }

    public static func defaultMaxCacheCap(
        contextWindow: Int,
        activeKvOffloadEnabled: Bool
    ) -> Int {
        guard contextWindow > 0 else {
            return conservativeLongContextCap
        }
        guard !activeKvOffloadEnabled else {
            return contextWindow
        }
        return min(contextWindow, conservativeLongContextCap)
    }
}

private extension JSONEncoder {
    static var prettyIronMLX: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}
