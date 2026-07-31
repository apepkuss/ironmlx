import Foundation

public enum BackendLaunchValidationError: Error, Equatable, LocalizedError {
    public var errorDescription: String? {
        message
    }

    public var message: String {
        switch self {
        case .localBindAddressRequired:
            return "Local mode requires a loopback host."
        case .lanBindAddressRequired:
            return "LAN mode requires a selected active LAN IP address."
        case .lanSecurityMaterialMissing:
            return "LAN mode requires API key and TLS material in Keychain."
        }
    }

    case localBindAddressRequired
    case lanBindAddressRequired
    case lanSecurityMaterialMissing
}

public struct BackendLaunchOptions: Equatable {
    public static let bytesPerGigabyte = 1_073_741_824
    public static let defaultPagedPrefixCacheDirectory = "~/.ironmlx/cache/paged_prefix_cache"
    public static let automaticHotCacheLimitBytes = 8 * bytesPerGigabyte
    public static let defaultColdCacheLimitGB = 10
    public static let defaultModelTtlMinutes = 30

    public var prefillChunkSize: Int?
    public var bMax: Int?
    public var admissionDeadlineMs: Int?
    public var admissionQueueMax: Int?
    public var maxCacheCap: Int?
    public var decodeCadenceMidChunkCap: Int?
    public var schedulerProfile: String?
    public var schedulerAutotuneReport: Bool
    public var kvQuant: String?
    public var pagedPrefixCacheDir: String?
    public var prefixLruCacheMaxBytes: Int?
    public var ssdPrefixCacheMaxGB: Int?
    public var activeKvOffload: Bool
    public var maxLoadedModels: Int?
    public var modelTtlMinutes: Int?
    public var memoryLimitTotalGB: Int?
    public var memoryLimitModelGB: Int?

    public init(
        prefillChunkSize: Int? = nil,
        bMax: Int? = nil,
        admissionDeadlineMs: Int? = nil,
        admissionQueueMax: Int? = nil,
        maxCacheCap: Int? = nil,
        decodeCadenceMidChunkCap: Int? = nil,
        schedulerProfile: String? = nil,
        schedulerAutotuneReport: Bool = false,
        kvQuant: String? = nil,
        pagedPrefixCacheDir: String? = nil,
        prefixLruCacheMaxBytes: Int? = nil,
        ssdPrefixCacheMaxGB: Int? = nil,
        activeKvOffload: Bool = false,
        maxLoadedModels: Int? = nil,
        modelTtlMinutes: Int? = nil,
        memoryLimitTotalGB: Int? = nil,
        memoryLimitModelGB: Int? = nil
    ) {
        self.prefillChunkSize = prefillChunkSize
        self.bMax = bMax
        self.admissionDeadlineMs = admissionDeadlineMs
        self.admissionQueueMax = admissionQueueMax
        self.maxCacheCap = maxCacheCap
        self.decodeCadenceMidChunkCap = decodeCadenceMidChunkCap
        self.schedulerProfile = schedulerProfile
        self.schedulerAutotuneReport = schedulerAutotuneReport
        self.kvQuant = Self.normalizedKVQuant(kvQuant)
        self.pagedPrefixCacheDir = Self.normalizedPath(pagedPrefixCacheDir)
        self.prefixLruCacheMaxBytes = prefixLruCacheMaxBytes
        self.ssdPrefixCacheMaxGB = ssdPrefixCacheMaxGB
        self.activeKvOffload = activeKvOffload
        self.maxLoadedModels = maxLoadedModels
        self.modelTtlMinutes = modelTtlMinutes
        self.memoryLimitTotalGB = Self.positiveGigabytes(memoryLimitTotalGB)
        self.memoryLimitModelGB = Self.positiveGigabytes(memoryLimitModelGB)
    }

    public init(
        config: AppConfig,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) {
        let cacheEnabled = config.cacheEnable ?? true
        let prefixCacheDir = cacheEnabled
            ? (Self.normalizedPath(config.cacheDir) ?? Self.defaultPagedPrefixCacheDirectory)
            : nil
        let prefixLruCacheMaxBytes = cacheEnabled
            ? Self.hotCacheLimitBytes(hotCacheGigabytes: config.hotCache, physicalMemoryBytes: physicalMemoryBytes)
            : nil
        let ssdPrefixCacheMaxGB = cacheEnabled
            ? Self.coldCacheLimitGigabytes(config.coldCache)
            : nil
        self.init(
            prefillChunkSize: config.prefillChunkSize,
            bMax: config.maxSequences,
            admissionDeadlineMs: config.admissionDeadlineMs,
            admissionQueueMax: config.admissionQueueMax,
            maxCacheCap: config.maxCacheCap,
            decodeCadenceMidChunkCap: config.decodeCadenceMidChunkCap,
            schedulerProfile: config.schedulerProfile,
            schedulerAutotuneReport: config.schedulerAutotuneReport == true,
            kvQuant: config.kvQuant,
            pagedPrefixCacheDir: prefixCacheDir,
            prefixLruCacheMaxBytes: prefixLruCacheMaxBytes,
            ssdPrefixCacheMaxGB: ssdPrefixCacheMaxGB,
            activeKvOffload: config.activeKvOffload == true,
            maxLoadedModels: config.maxModels,
            modelTtlMinutes: config.modelTtlMinutes ?? Self.defaultModelTtlMinutes,
            memoryLimitTotalGB: config.memLimitTotal,
            memoryLimitModelGB: config.memLimitModel
        )
    }

    public static func normalizedKVQuant(_ value: String?) -> String? {
        guard let value else {
            return nil
        }
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "":
            return nil
        case "none", "turbo3", "turbo4", "k3v4":
            return normalized
        case "adaptive":
            return "k3v4"
        default:
            return nil
        }
    }

    public static func hotCacheLimitBytes(
        hotCacheGigabytes: Int?,
        physicalMemoryBytes: UInt64
    ) -> Int {
        if let hotCacheGigabytes, hotCacheGigabytes > 0 {
            return hotCacheGigabytes * bytesPerGigabyte
        }
        let automaticBytes = min(
            physicalMemoryBytes / 8,
            UInt64(automaticHotCacheLimitBytes)
        )
        return Int(automaticBytes)
    }

    public static func coldCacheLimitGigabytes(_ value: Int?) -> Int {
        max(value ?? defaultColdCacheLimitGB, 1)
    }

    public static func positiveGigabytes(_ value: Int?) -> Int? {
        guard let value, value > 0 else {
            return nil
        }
        return value
    }

    public var validationError: BackendLaunchValidationError? {
        return nil
    }

    public var isValid: Bool {
        validationError == nil
    }

    private static func normalizedPath(_ value: String?) -> String? {
        guard let value else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

public struct BackendLaunchConfiguration: Equatable {
    public var executableURL: URL
    public var host: String
    public var port: UInt16
    public var options: BackendLaunchOptions
    public var networkMode: String
    public var lanHost: String?
    public var securityBootstrapStdin: Bool

    public init(
        executableURL: URL,
        host: String,
        port: UInt16,
        options: BackendLaunchOptions = BackendLaunchOptions(),
        networkMode: String = "local",
        lanHost: String? = nil,
        securityBootstrapStdin: Bool = false
    ) {
        self.executableURL = executableURL
        self.host = host
        self.port = port
        self.options = options
        self.networkMode = networkMode
        self.lanHost = lanHost
        self.securityBootstrapStdin = securityBootstrapStdin
    }

    public var validationError: BackendLaunchValidationError? {
        let normalizedHost = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard normalizedHost == "127.0.0.1" || normalizedHost == "::1" || normalizedHost == "[::1]" else {
            return .localBindAddressRequired
        }
        if networkMode == "lan" {
            guard let lanHost, EndpointPayload.isSafeLANAddress(lanHost) else {
                return .lanBindAddressRequired
            }
            guard securityBootstrapStdin else {
                return .lanSecurityMaterialMissing
            }
        }
        return nil
    }

    public var arguments: [String] {
        var arguments = [
            "serve",
            "--host", host,
            "--port", String(port),
            "--network-mode", networkMode,
        ]
        if networkMode == "lan", let lanHost {
            arguments += ["--lan-host", lanHost]
            if securityBootstrapStdin {
                arguments.append("--security-bootstrap-stdin")
            }
        }
        appendIntegerFlag("--prefill-chunk-size", options.prefillChunkSize, to: &arguments, allowsZero: true)
        appendIntegerFlag("--max-sequences", options.bMax, to: &arguments)
        appendIntegerFlag("--admission-deadline-ms", options.admissionDeadlineMs, to: &arguments, allowsZero: true)
        appendIntegerFlag("--admission-queue-max", options.admissionQueueMax, to: &arguments, allowsZero: true)
        appendIntegerFlag("--max-cache-cap", options.maxCacheCap, to: &arguments)
        appendIntegerFlag("--decode-cadence-mid-chunk-cap", options.decodeCadenceMidChunkCap, to: &arguments)
        if let schedulerProfile = options.schedulerProfile?.trimmingCharacters(in: .whitespacesAndNewlines),
           !schedulerProfile.isEmpty {
            arguments += ["--scheduler-profile", schedulerProfile]
        }
        if options.schedulerAutotuneReport {
            arguments.append("--scheduler-autotune-report")
        }
        if let pagedPrefixCacheDir = options.pagedPrefixCacheDir {
            arguments += ["--paged-prefix-cache-dir", pagedPrefixCacheDir]
        }
        appendIntegerFlag("--prefix-lru-cache-max-bytes", options.prefixLruCacheMaxBytes, to: &arguments)
        appendIntegerFlag("--ssd-prefix-cache-max-gb", options.ssdPrefixCacheMaxGB, to: &arguments)
        appendIntegerFlag("--max-loaded-models", options.maxLoadedModels, to: &arguments)
        appendIntegerFlag("--model-ttl-minutes", options.modelTtlMinutes, to: &arguments)
        appendIntegerFlag("--memory-limit-total-gb", options.memoryLimitTotalGB, to: &arguments)
        appendIntegerFlag("--memory-limit-model-gb", options.memoryLimitModelGB, to: &arguments)
        if let kvQuant = options.kvQuant {
            arguments += ["--kv-quant", kvQuant]
        }
        if options.activeKvOffload {
            arguments.append("--active-kv-offload")
        }
        return arguments
    }

    private func appendIntegerFlag(
        _ flag: String,
        _ value: Int?,
        to arguments: inout [String],
        allowsZero: Bool = false
    ) {
        guard let value, allowsZero ? value >= 0 : value > 0 else {
            return
        }
        arguments += [flag, String(value)]
    }
}

public enum BackendBinaryResolver {
    public static func resolve() -> URL {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment

        if let explicit = environment["IRONMLX_BIN"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit)
        }

        let cwd = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        let candidates = [
            cwd.appendingPathComponent("../target/debug/ironmlx").standardizedFileURL,
            cwd.appendingPathComponent("../target/release/ironmlx").standardizedFileURL,
            cwd.appendingPathComponent("target/debug/ironmlx").standardizedFileURL,
            cwd.appendingPathComponent("target/release/ironmlx").standardizedFileURL,
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("ironmlx"),
        ]

        for candidate in candidates where fileManager.isExecutableFile(atPath: candidate.path) {
            return candidate
        }

        return URL(fileURLWithPath: "/usr/bin/env")
    }

    public static func resolvedExecutableAndArguments(
        for configuration: BackendLaunchConfiguration
    ) -> (URL, [String]) {
        if configuration.executableURL.path == "/usr/bin/env" {
            return (configuration.executableURL, ["ironmlx"] + configuration.arguments)
        }
        return (configuration.executableURL, configuration.arguments)
    }
}
