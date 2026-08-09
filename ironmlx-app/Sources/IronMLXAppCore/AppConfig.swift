import Foundation

public struct AppConfig: Codable, Equatable, Sendable {
    public var host: String
    public var port: UInt16
    public var networkMode: String?
    public var lanHost: String?
    public var lanCredentialID: String?
    public var lanCertificateFingerprint: String?
    public var defaultModel: String?
    public var loadedModels: [String]?
    public var pinnedModels: [String]?
    public var language: String
    public var theme: String?
    public var logLevel: String?
    public var memLimitTotal: Int?
    public var memLimitModel: Int?
    public var memTotalAuto: Bool?
    public var memTotal: Int?
    public var memModelAuto: Bool?
    public var memModel: Int?
    public var hotCache: Int?
    public var coldCache: Int?
    public var cacheEnable: Bool?
    public var cacheDir: String?
    public var kvQuant: String?
    public var activeKvOffload: Bool?
    public var maxSequences: Int?
    public var maxModels: Int?
    public var modelTtlMinutes: Int?
    public var verifyModelOnLoad: Bool?
    public var distributedBackend: String?
    public var parallelMode: String?
    public var prefillChunkSize: Int?
    public var bMax: Int?
    public var admissionDeadlineMs: Int?
    public var admissionQueueMax: Int?
    public var maxCacheCap: Int?
    public var decodeCadenceMidChunkCap: Int?
    public var schedulerProfile: String?
    public var schedulerAutotuneReport: Bool?

    public init(
        host: String = "127.0.0.1",
        port: UInt16 = 9068,
        networkMode: String? = "local",
        lanHost: String? = nil,
        lanCredentialID: String? = nil,
        lanCertificateFingerprint: String? = nil,
        defaultModel: String? = nil,
        loadedModels: [String]? = nil,
        pinnedModels: [String]? = nil,
        language: String = "en",
        theme: String? = nil,
        logLevel: String? = nil,
        memLimitTotal: Int? = nil,
        memLimitModel: Int? = nil,
        memTotalAuto: Bool? = nil,
        memTotal: Int? = nil,
        memModelAuto: Bool? = nil,
        memModel: Int? = nil,
        hotCache: Int? = nil,
        coldCache: Int? = nil,
        cacheEnable: Bool? = nil,
        cacheDir: String? = nil,
        kvQuant: String? = nil,
        activeKvOffload: Bool? = nil,
        maxSequences: Int? = nil,
        maxModels: Int? = nil,
        modelTtlMinutes: Int? = nil,
        verifyModelOnLoad: Bool? = nil,
        distributedBackend: String? = nil,
        parallelMode: String? = nil,
        prefillChunkSize: Int? = nil,
        bMax: Int? = nil,
        admissionDeadlineMs: Int? = nil,
        admissionQueueMax: Int? = nil,
        maxCacheCap: Int? = nil,
        decodeCadenceMidChunkCap: Int? = nil,
        schedulerProfile: String? = nil,
        schedulerAutotuneReport: Bool? = nil
    ) {
        self.host = host
        self.port = port
        self.networkMode = networkMode
        self.lanHost = lanHost
        self.lanCredentialID = lanCredentialID
        self.lanCertificateFingerprint = lanCertificateFingerprint
        self.defaultModel = defaultModel
        self.loadedModels = loadedModels
        self.pinnedModels = pinnedModels
        self.language = language
        self.theme = theme
        self.logLevel = logLevel
        self.memLimitTotal = memLimitTotal
        self.memLimitModel = memLimitModel
        self.memTotalAuto = memTotalAuto
        self.memTotal = memTotal
        self.memModelAuto = memModelAuto
        self.memModel = memModel
        self.hotCache = hotCache
        self.coldCache = coldCache
        self.cacheEnable = cacheEnable
        self.cacheDir = cacheDir
        self.kvQuant = kvQuant
        self.activeKvOffload = activeKvOffload
        self.maxSequences = maxSequences
        self.maxModels = maxModels
        self.modelTtlMinutes = modelTtlMinutes
        self.verifyModelOnLoad = verifyModelOnLoad
        self.distributedBackend = distributedBackend
        self.parallelMode = parallelMode
        self.prefillChunkSize = prefillChunkSize
        self.bMax = bMax
        self.admissionDeadlineMs = admissionDeadlineMs
        self.admissionQueueMax = admissionQueueMax
        self.maxCacheCap = maxCacheCap
        self.decodeCadenceMidChunkCap = decodeCadenceMidChunkCap
        self.schedulerProfile = schedulerProfile
        self.schedulerAutotuneReport = schedulerAutotuneReport
    }

    enum CodingKeys: String, CodingKey {
        case host
        case port
        case networkMode = "network_mode"
        case lanHost = "lan_host"
        case lanCredentialID = "lan_credential_id"
        case lanCertificateFingerprint = "lan_certificate_fingerprint"
        case defaultModel = "default_model"
        case loadedModels = "loaded_models"
        case pinnedModels = "pinned_models"
        case language
        case theme
        case logLevel = "log_level"
        case memLimitTotal = "mem_limit_total"
        case memLimitModel = "mem_limit_model"
        case memTotalAuto = "mem_total_auto"
        case memTotal = "mem_total"
        case memModelAuto = "mem_model_auto"
        case memModel = "mem_model"
        case hotCache = "hot_cache"
        case coldCache = "cold_cache"
        case cacheEnable = "cache_enable"
        case cacheDir = "cache_dir"
        case kvQuant = "kv_quant"
        case activeKvOffload = "active_kv_offload"
        case maxSequences = "max_sequences"
        case maxModels = "max_models"
        case modelTtlMinutes = "model_ttl_minutes"
        case verifyModelOnLoad = "verify_model_on_load"
        case distributedBackend = "distributed_backend"
        case parallelMode = "parallel_mode"
        case prefillChunkSize = "prefill_chunk_size"
        case bMax = "b_max"
        case admissionDeadlineMs = "admission_deadline_ms"
        case admissionQueueMax = "admission_queue_max"
        case maxCacheCap = "max_cache_cap"
        case decodeCadenceMidChunkCap = "decode_cadence_mid_chunk_cap"
        case schedulerProfile = "scheduler_profile"
        case schedulerAutotuneReport = "scheduler_autotune_report"
    }
}

public enum AppLanguageResolver {
    public static func resolve(preferredLanguages: [String]) -> String {
        guard let identifier = preferredLanguages.first else {
            return "en"
        }
        return resolve(identifier: identifier) ?? "en"
    }

    private static func resolve(identifier: String) -> String? {
        let components = identifier
            .replacingOccurrences(of: "_", with: "-")
            .lowercased()
            .split(separator: "-")
            .map(String.init)
        guard let language = components.first else {
            return nil
        }

        switch language {
        case "en":
            return "en"
        case "ja":
            return "ja"
        case "ko":
            return "ko"
        case "zh":
            if components.contains("hant")
                || components.contains("tw")
                || components.contains("hk")
                || components.contains("mo")
            {
                return "zh-Hant"
            }
            return "zh-Hans"
        default:
            return nil
        }
    }
}

public extension AppConfig {
    var isLANMode: Bool {
        networkMode?.lowercased() == "lan"
    }

    var defaultModelReference: String? {
        Self.normalizedModelReference(defaultModel)
    }

    var restoredModelReferences: [String] {
        var references = Self.normalizedModelReferences(loadedModels ?? [])
        if let defaultModelReference,
           let index = references.firstIndex(of: defaultModelReference) {
            if index != references.startIndex {
                references.remove(at: index)
                references.insert(defaultModelReference, at: 0)
            }
        }
        return references
    }

    var pinnedModelReferences: [String] {
        let loaded = Set(restoredModelReferences)
        return Self.normalizedModelReferences(pinnedModels ?? [])
            .filter { loaded.contains($0) }
    }

    mutating func recordLoadedModel(_ model: String, setDefault: Bool) {
        guard let model = Self.normalizedModelReference(model) else {
            return
        }
        var references = Self.normalizedModelReferences(loadedModels ?? [])
        if !references.contains(model) {
            references.append(model)
        }
        loadedModels = references
        if setDefault {
            defaultModel = model
        }
    }

    mutating func recordUnloadedModel(_ model: String) {
        guard let model = Self.normalizedModelReference(model) else {
            return
        }
        let references = Self.normalizedModelReferences(loadedModels ?? [])
            .filter { $0 != model }
        loadedModels = references
        pinnedModels = Self.normalizedModelReferences(pinnedModels ?? [])
            .filter { $0 != model && references.contains($0) }
    }

    mutating func replaceLoadedModels(_ models: [String], defaultModel: String?) {
        let references = Self.normalizedModelReferences(models)
        loadedModels = references
        pinnedModels = Self.normalizedModelReferences(pinnedModels ?? [])
            .filter { references.contains($0) }
        if let defaultModel = Self.normalizedModelReference(defaultModel) {
            self.defaultModel = defaultModel
        } else if let existingDefault = defaultModelReference {
            self.defaultModel = existingDefault
        } else {
            self.defaultModel = references.first
        }
    }

    mutating func replacePinnedModels(_ models: [String]) {
        let loaded = Set(restoredModelReferences)
        pinnedModels = Self.normalizedModelReferences(models)
            .filter { loaded.contains($0) }
    }

    mutating func recordPinnedModel(_ model: String, pinned: Bool) {
        guard let model = Self.normalizedModelReference(model) else {
            return
        }
        var references = Self.normalizedModelReferences(pinnedModels ?? [])
        if pinned {
            guard restoredModelReferences.contains(model) else {
                return
            }
            if !references.contains(model) {
                references.append(model)
            }
        } else {
            references.removeAll { $0 == model }
        }
        pinnedModels = references
    }

    static func normalizedModelReference(_ value: String?) -> String? {
        guard let value else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func normalizedModelReferences(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var references: [String] = []
        for value in values {
            guard let reference = normalizedModelReference(value),
                  !seen.contains(reference)
            else {
                continue
            }
            seen.insert(reference)
            references.append(reference)
        }
        return references
    }
}

public final class AppConfigStore: @unchecked Sendable {
    public static let shared = AppConfigStore()

    public let url: URL
    private let fileManager: FileManager
    private let preferredLanguages: @Sendable () -> [String]
    private let recoveryState = ConfigurationRecoveryState()

    public var recoveryIssue: ConfigurationRecoveryIssue? {
        recoveryState.issue
    }

    public init(
        url: URL = AppConfigStore.defaultConfigURL(),
        fileManager: FileManager = .default,
        preferredLanguages: @escaping @Sendable () -> [String] = { Locale.preferredLanguages }
    ) {
        self.url = url
        self.fileManager = fileManager
        self.preferredLanguages = preferredLanguages
    }

    public func load() -> AppConfig {
        guard fileManager.fileExists(atPath: url.path) else {
            recoveryState.clear()
            let config = AppConfig(
                language: AppLanguageResolver.resolve(preferredLanguages: preferredLanguages())
            )
            save(config)
            return config
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            recordCorruption(data: nil, error: error)
            return fallbackConfig()
        }
        do {
            let config = try JSONDecoder().decode(AppConfig.self, from: data)
            recoveryState.clear()
            return config
        } catch {
            recordCorruption(data: data, error: error)
            return fallbackConfig()
        }
    }

    @discardableResult
    public func save(_ config: AppConfig) -> Bool {
        guard recoveryIssue == nil else {
            IronMLXAppLogger.error(
                "Refusing to overwrite unreadable ironmlx app config before explicit recovery: \(url.path)"
            )
            return false
        }
        do {
            try write(config)
            return true
        } catch {
            IronMLXAppLogger.error("Failed to save ironmlx app config: \(error)")
            return false
        }
    }

    public func resetAfterCorruption() throws {
        guard let recoveryIssue else {
            return
        }
        guard let preservedURL = recoveryIssue.preservedURL,
              fileManager.fileExists(atPath: preservedURL.path)
        else {
            throw ConfigurationRecoveryResetError.preservedCopyMissing(url)
        }
        try write(fallbackConfig())
        recoveryState.clear()
    }

    private func fallbackConfig() -> AppConfig {
        AppConfig(
            language: AppLanguageResolver.resolve(preferredLanguages: preferredLanguages())
        )
    }

    private func write(_ config: AppConfig) throws {
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try data.write(to: url, options: .atomic)
    }

    private func recordCorruption(data: Data?, error: Error) {
        if recoveryState.recordIfNeeded({
            ConfigurationCorruptionPreserver.makeIssue(
                kind: .appConfig,
                sourceURL: url,
                data: data,
                error: error,
                fileManager: fileManager
            )
        }) {
            IronMLXAppLogger.error(
                "IronMLX app config is unreadable and requires explicit recovery: \(url.path); \(error)"
            )
        }
    }

    public static func defaultConfigURL() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("config", isDirectory: true)
            .appendingPathComponent("app_config.json")
    }
}
