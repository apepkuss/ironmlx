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
    public var promptLookupEnabled: Bool?
    public var promptLookupCrossRequest: Bool?

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
        mtpDraftTokens: String? = nil,
        promptLookupEnabled: Bool? = nil,
        promptLookupCrossRequest: Bool? = nil
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
        self.promptLookupEnabled = promptLookupEnabled
        self.promptLookupCrossRequest = promptLookupCrossRequest
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

    public var promptLookupConfig: BackendPromptLookupConfig? {
        guard promptLookupEnabled == true else {
            return nil
        }
        return BackendPromptLookupConfig(
            crossRequest: promptLookupCrossRequest == true
        )
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
        case promptLookupEnabled = "prompt_lookup_enabled"
        case promptLookupCrossRequest = "prompt_lookup_cross_request"
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

private struct ModelParameterEnvelope: Codable, Equatable {
    let schemaVersion: Int
    let models: [String: ModelParameters]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case models
    }
}

public final class ModelParameterStore: @unchecked Sendable {
    public static let shared = ModelParameterStore()
    public static let currentSchemaVersion = 1

    public let url: URL
    private let fileManager: FileManager
    private let recoveryState = ConfigurationRecoveryState()
    private let coordinator: ConfigurationFileCoordinator

    public var recoveryIssue: ConfigurationRecoveryIssue? {
        recoveryState.issue
    }

    public init(
        url: URL = ModelParameterStore.defaultURL(),
        fileManager: FileManager = .default
    ) {
        self.url = url
        self.fileManager = fileManager
        self.coordinator = ConfigurationFileCoordinator(activeURL: url, fileManager: fileManager)
    }

    public func loadAll() throws -> [String: ModelParameters] {
        try coordinator.withLock {
            do {
                try coordinator.recoverInterruptedTransactionIfNeeded()
                guard fileManager.fileExists(atPath: url.path) else {
                    let data = try encodedV1([:])
                    try coordinator.commitActiveAndLKG(data)
                    recoveryState.clear()
                    return [:]
                }
                let data = try Data(contentsOf: url)
                let object = try ConfigurationJSON.object(from: data)
                if let version = try ConfigurationJSON.schemaVersion(in: object) {
                    let models = try decodeVersioned(data, object: object, version: version)
                    try refreshLKGIfNeeded(data)
                    recoveryState.clear()
                    return models
                }
                return try migrateV0(data, object: object)
            } catch let error as ConfigurationPersistenceError {
                switch error {
                case let .unsupportedSchemaVersion(found, supported):
                    recordIssue(
                        reason: .unsupportedVersion(found: found, supported: supported),
                        error: error,
                        preservedURL: nil,
                        preservationError: nil
                    )
                default:
                    recordCorruption(data: try? Data(contentsOf: url), error: error)
                }
                throw error
            } catch {
                recordCorruption(data: try? Data(contentsOf: url), error: error)
                throw error
            }
        }
    }

    public func parameters(for modelID: String) -> ModelParameters? {
        let key = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            return nil
        }
        return try? loadAll()[key]
    }

    public func save(_ parameters: ModelParameters) throws {
        try coordinator.withLock {
            try assertWritable()
            let key = parameters.modelID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else {
                return
            }
            try validate(parameters, key: key)
            var all = try loadAll()
            try assertWritable()
            all[key] = parameters
            try commit(all)
        }
    }

    public func recordMtpLoadPreference(
        modelID: String,
        enabled: Bool,
        mtpModelID: String?
    ) throws {
        try coordinator.withLock {
            try assertWritable()
            let key = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else {
                return
            }
            var all = try loadAll()
            try assertWritable()
            var parameters = all[key] ?? ModelParameters(modelID: key)
            parameters.mtpEnabled = enabled
            let selectedMtp = mtpModelID?.trimmingCharacters(in: .whitespacesAndNewlines)
            if enabled, let selectedMtp, !selectedMtp.isEmpty {
                parameters.mtpModelID = selectedMtp
            }
            try validate(parameters, key: key)
            all[key] = parameters
            try commit(all)
        }
    }

    public func jsonString() -> String {
        let all = (try? loadAll()) ?? [:]
        guard let data = try? JSONEncoder().encode(all),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }

    public func resetAfterCorruption() throws {
        try coordinator.withLock {
            guard let recoveryIssue else {
                return
            }
            guard case .unsupportedVersion = recoveryIssue.reason else {
                guard let preservedURL = recoveryIssue.preservedURL,
                      fileManager.fileExists(atPath: preservedURL.path) else {
                    throw ConfigurationRecoveryResetError.preservedCopyMissing(url)
                }
                try commit([:])
                recoveryState.clear()
                return
            }
            throw ConfigurationRecoveryResetError.unsupportedVersion(url)
        }
    }

    public func restoreFromLKG() throws {
        try coordinator.withLock {
            guard let recoveryIssue else {
                return
            }
            guard case .unsupportedVersion = recoveryIssue.reason else {
                let data = try Data(contentsOf: coordinator.layout.lkgURL)
                _ = try decodeV1(data)
                try coordinator.commitActiveAndLKG(data)
                _ = try decodeV1(Data(contentsOf: url))
                recoveryState.clear()
                return
            }
            throw ConfigurationRecoveryResetError.unsupportedVersion(url)
        }
    }

    private func assertWritable() throws {
        if let issue = recoveryIssue {
            throw ConfigurationRecoveryWriteError.unresolvedIssue(url, issue.dashboardErrorCode)
        }
    }

    private func commit(_ models: [String: ModelParameters]) throws {
        try validate(models)
        let data = try encodedV1(models)
        _ = try decodeV1(data)
        try coordinator.commitActiveAndLKG(data)
        _ = try decodeV1(Data(contentsOf: url))
        recoveryState.clear()
    }

    private func encodedV1(_ models: [String: ModelParameters]) throws -> Data {
        try validate(models)
        return try JSONEncoder.prettyIronMLX.encode(
            ModelParameterEnvelope(schemaVersion: Self.currentSchemaVersion, models: models)
        )
    }

    private func decodeVersioned(
        _ data: Data,
        object: [String: Any],
        version: Int
    ) throws -> [String: ModelParameters] {
        guard version <= Self.currentSchemaVersion else {
            throw ConfigurationPersistenceError.unsupportedSchemaVersion(
                found: version,
                supported: Self.currentSchemaVersion
            )
        }
        guard version == Self.currentSchemaVersion else {
            throw ConfigurationPersistenceError.invalidSchemaVersion
        }
        return try decodeV1(data)
    }

    private func decodeV1(_ data: Data) throws -> [String: ModelParameters] {
        let object = try ConfigurationJSON.object(from: data)
        try ConfigurationJSON.requireKeys(Set(object.keys), allowed: ["schema_version", "models"])
        guard try ConfigurationJSON.schemaVersion(in: object) == Self.currentSchemaVersion else {
            throw ConfigurationPersistenceError.invalidSchemaVersion
        }
        guard let modelsObject = object["models"] as? [String: Any] else {
            throw ConfigurationPersistenceError.invalidValue("models")
        }
        try validateRawModelObjects(modelsObject)
        let envelope = try JSONDecoder().decode(ModelParameterEnvelope.self, from: data)
        try validate(envelope.models)
        return envelope.models
    }

    private func migrateV0(_ data: Data, object: [String: Any]) throws -> [String: ModelParameters] {
        var preservedURL: URL?
        do {
            preservedURL = try coordinator.preservePreMigration(data, schemaVersion: 0)
            try validateRawModelObjects(object)
            let models = try JSONDecoder().decode([String: ModelParameters].self, from: data)
            try validate(models)
            let candidate = try encodedV1(models)
            _ = try decodeV1(candidate)
            try coordinator.commitActiveAndLKG(candidate)
            let verified = try decodeV1(Data(contentsOf: url))
            recoveryState.clear()
            IronMLXAppLogger.info(
                "event=configuration_migrated kind=model_parameters from_schema=0 to_schema=1 removed_keys="
            )
            return verified
        } catch {
            recordIssue(
                reason: .migrationFailed(from: 0, to: 1),
                error: error,
                preservedURL: preservedURL,
                preservationError: preservedURL == nil ? error.localizedDescription : nil
            )
            throw ModelParameterMigrationRecordedError.underlying(error.localizedDescription)
        }
    }

    private func validateRawModelObjects(_ object: [String: Any]) throws {
        for (key, value) in object {
            guard let model = value as? [String: Any] else {
                throw ConfigurationPersistenceError.invalidValue("models.\(key)")
            }
            try ConfigurationJSON.requireKeys(Set(model.keys), allowed: Self.modelKeys)
        }
    }

    private func validate(_ models: [String: ModelParameters]) throws {
        for (key, parameters) in models {
            try validate(parameters, key: key)
        }
    }

    private func validate(_ parameters: ModelParameters, key: String) throws {
        let normalizedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedID = parameters.modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedKey.isEmpty, normalizedKey == key, normalizedID == normalizedKey else {
            throw ConfigurationPersistenceError.invalidValue("model_id")
        }
        try validatePositiveInteger(parameters.contextSize, field: "context_size")
        try validatePositiveInteger(parameters.maxTokens, field: "max_tokens")
        try validatePositiveInteger(parameters.topK, field: "top_k")
        try validatePositiveInteger(parameters.mtpDraftTokens, field: "mtp_draft_tokens")
        try validatePositiveDouble(parameters.temperature, field: "temperature")
        try validatePositiveDouble(parameters.repeatPenalty, field: "repeat_penalty")
        if let value = nonEmpty(parameters.topP),
           !(Double(value).map { $0.isFinite && $0 > 0 && $0 <= 1 } ?? false) {
            throw ConfigurationPersistenceError.invalidValue("top_p")
        }
        if let mtpModelID = parameters.mtpModelID,
           mtpModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw ConfigurationPersistenceError.invalidValue("mtp_model_id")
        }
    }

    private func validatePositiveInteger(_ value: String?, field: String) throws {
        guard let value = nonEmpty(value) else {
            return
        }
        guard let parsed = Int(value), parsed > 0 else {
            throw ConfigurationPersistenceError.invalidValue(field)
        }
    }

    private func validatePositiveDouble(_ value: String?, field: String) throws {
        guard let value = nonEmpty(value) else {
            return
        }
        guard let parsed = Double(value), parsed.isFinite, parsed > 0 else {
            throw ConfigurationPersistenceError.invalidValue(field)
        }
    }

    private func nonEmpty(_ value: String?) -> String? {
        guard let value else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func refreshLKGIfNeeded(_ data: Data) throws {
        do {
            if fileManager.fileExists(atPath: coordinator.layout.lkgURL.path) {
                _ = try decodeV1(Data(contentsOf: coordinator.layout.lkgURL))
            }
            try coordinator.refreshLKG(data)
        } catch {
            IronMLXAppLogger.warning("Failed to refresh model parameter LKG: \(error)")
        }
    }

    private func recordCorruption(data: Data?, error: Error) {
        if recoveryState.recordIfNeeded({
            ConfigurationCorruptionPreserver.makeIssue(
                kind: .modelParameters,
                sourceURL: url,
                data: data,
                error: error,
                fileManager: fileManager,
                lkgURL: coordinator.layout.lkgURL,
                lkgErrorDescription: lkgErrorDescription()
            )
        }) {
            IronMLXAppLogger.error(
                "IronMLX model parameters are unreadable and require explicit recovery: \(url.path); \(error)"
            )
        }
    }

    private func recordIssue(
        reason: ConfigurationRecoveryIssue.Reason,
        error: Error,
        preservedURL: URL?,
        preservationError: String?
    ) {
        if recoveryState.recordIfNeeded({
            ConfigurationRecoveryIssue(
                kind: .modelParameters,
                sourceURL: url,
                preservedURL: preservedURL,
                lkgURL: coordinator.layout.lkgURL,
                lkgErrorDescription: lkgErrorDescription(),
                reason: reason,
                errorDescription: error.localizedDescription,
                preservationErrorDescription: preservationError
            )
        }) {
            IronMLXAppLogger.error(
                "IronMLX model parameters require explicit recovery: \(url.path); \(error)"
            )
        }
    }

    private func lkgErrorDescription() -> String? {
        guard fileManager.fileExists(atPath: coordinator.layout.lkgURL.path) else {
            return ConfigurationPersistenceError.lkgUnavailable(coordinator.layout.lkgURL).localizedDescription
        }
        do {
            _ = try decodeV1(Data(contentsOf: coordinator.layout.lkgURL))
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    private static let modelKeys: Set<String> = [
        "model_id", "alias", "model_type", "context_size", "max_tokens", "temperature",
        "top_p", "top_k", "repeat_penalty", "mtp_enabled", "mtp_model_id",
        "mtp_draft_tokens", "prompt_lookup_enabled", "prompt_lookup_cross_request",
    ]

    public static func defaultURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("model_params.json")
    }
}

public enum ConfigurationRecoveryWriteError: LocalizedError, Equatable {
    case unresolvedIssue(URL, String)

    public var errorDescription: String? {
        switch self {
        case let .unresolvedIssue(url, code):
            "Refusing to overwrite unresolved configuration before explicit recovery (\(code)): \(url.path)"
        }
    }
}

private enum ModelParameterMigrationRecordedError: LocalizedError {
    case underlying(String)

    var errorDescription: String? {
        switch self {
        case let .underlying(description):
            description
        }
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
