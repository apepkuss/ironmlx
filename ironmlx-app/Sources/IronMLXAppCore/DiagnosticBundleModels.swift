import Foundation

public struct DiagnosticBundleLimits: Sendable, Equatable {
    public var manifestBytes = 65_536
    public var systemBytes = 32_768
    public var runtimeHealthBytes = 196_608
    public var modelsBytes = 524_288
    public var maximumModelVersions = 256
    public var incidentsBytes = 524_288
    public var appLogBytes = 524_288
    public var appLogLines = 4_000
    public var backendLogBytes = 1_572_864
    public var backendLogLines = 10_000
    public var maximumUncompressedBytes = 3_670_016
    public var maximumArchiveBytes = 4_194_304
    public var healthTimeout: TimeInterval = 2

    public init() {}
}

public struct DiagnosticFileManifest: Codable, Sendable, Equatable {
    public var status: String
    public var bytes: Int
    public var truncated: Bool
}

public struct DiagnosticBundleManifest: Codable, Sendable, Equatable {
    public static let schemaVersion = 1

    public var schemaVersion: Int
    public var generatedAt: Date
    public var appVersion: String
    public var appBuild: String
    public var backendVersion: String?
    public var ironMLXSourceCommit: String
    public var mlxCommit: String
    public var distributionChannel: String
    public var sourceTreeState: String
    public var developerIDStatus: String
    public var notarizationStatus: String
    public var backendOnline: Bool
    public var entries: [String: DiagnosticFileManifest]
    public var totalUncompressedBytes: Int
    public var contentTruncated: Bool

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case generatedAt = "generated_at"
        case appVersion = "app_version"
        case appBuild = "app_build"
        case backendVersion = "backend_version"
        case ironMLXSourceCommit = "ironmlx_source_commit"
        case mlxCommit = "mlx_commit"
        case distributionChannel = "distribution_channel"
        case sourceTreeState = "source_tree_state"
        case developerIDStatus = "developer_id_status"
        case notarizationStatus = "notarization_status"
        case backendOnline = "backend_online"
        case entries
        case totalUncompressedBytes = "total_uncompressed_bytes"
        case contentTruncated = "content_truncated"
    }
}

public struct DiagnosticSystemSnapshot: Codable, Sendable, Equatable {
    public var macOSVersion: String
    public var macOSBuild: String
    public var chip: String
    public var physicalMemoryBytes: UInt64
    public var appArchitecture: String
    public var signatureValidity: String
    public var signatureKind: String
    public var developerIDStatus: String
    public var notarizationStatus: String
    public var stapledTicketStatus: String

    enum CodingKeys: String, CodingKey {
        case macOSVersion = "macos_version"
        case macOSBuild = "macos_build"
        case chip
        case physicalMemoryBytes = "physical_memory_bytes"
        case appArchitecture = "app_architecture"
        case signatureValidity = "signature_validity"
        case signatureKind = "signature_kind"
        case developerIDStatus = "developer_id_status"
        case notarizationStatus = "notarization_status"
        case stapledTicketStatus = "stapled_ticket_status"
    }
}

public struct DiagnosticRuntimeHealth: Codable, Sendable, Equatable {
    public var status: String
    public var errorCode: String?
    public var backendVersion: String?
    public var mode: String?
    public var deviceName: String?
    public var uptimeSeconds: UInt64?
    public var scheduler: HealthzSnapshot.SchedulerInfo?
    public var memory: HealthzSnapshot.MemoryInfo?
    public var activeKV: DiagnosticActiveKV?
    public var loadedModels: [DiagnosticLoadedModel]

    enum CodingKeys: String, CodingKey {
        case status
        case errorCode = "error_code"
        case backendVersion = "backend_version"
        case mode
        case deviceName = "device_name"
        case uptimeSeconds = "uptime_seconds"
        case scheduler
        case memory
        case activeKV = "active_kv"
        case loadedModels = "loaded_models"
    }
}

public struct DiagnosticActiveKV: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var status: String
    public var active: Bool
    public var degraded: Bool
    public var mode: String
    public var residentPages: UInt64
    public var offloadedPages: UInt64
    public var parkedRequests: UInt64
    public var offloadedBytes: UInt64
    public var swapOutCount: UInt64
    public var swapInCount: UInt64
    public var swapErrorCount: UInt64

    init(_ source: HealthzSnapshot.ActiveKvOffloadInfo) {
        enabled = source.enabled
        status = source.status
        active = source.active
        degraded = source.degraded
        mode = source.mode
        residentPages = source.residentPages
        offloadedPages = source.offloadedPages
        parkedRequests = source.parkedRequests
        offloadedBytes = source.offloadedBytes
        swapOutCount = source.swapOutCount
        swapInCount = source.swapInCount
        swapErrorCount = source.swapErrorCount
    }

    enum CodingKeys: String, CodingKey {
        case enabled, status, active, degraded, mode
        case residentPages = "resident_pages"
        case offloadedPages = "offloaded_pages"
        case parkedRequests = "parked_requests"
        case offloadedBytes = "offloaded_bytes"
        case swapOutCount = "swap_out_count"
        case swapInCount = "swap_in_count"
        case swapErrorCount = "swap_error_count"
    }
}

public struct DiagnosticLoadedModel: Codable, Sendable, Equatable {
    public var id: String
    public var repoID: String
    public var architecture: String
    public var runtimeKind: String
    public var runtimeState: String
    public var scheduler: String?
    public var isDefault: Bool
    public var pinned: Bool
    public var mtpEnabled: Bool
    public var activeRequests: Int
    public var queuedRequests: Int
    public var queueCapacity: Int

    enum CodingKeys: String, CodingKey {
        case id
        case repoID = "repo_id"
        case architecture
        case runtimeKind = "runtime_kind"
        case runtimeState = "runtime_state"
        case scheduler
        case isDefault = "is_default"
        case pinned
        case mtpEnabled = "mtp_enabled"
        case activeRequests = "active_requests"
        case queuedRequests = "queued_requests"
        case queueCapacity = "queue_capacity"
    }
}

public struct DiagnosticModelInventory: Codable, Sendable, Equatable {
    public var status: String
    public var truncated: Bool
    public var models: [DiagnosticModelVersion]
}

public struct DiagnosticModelVersion: Codable, Sendable, Equatable {
    public var provider: String
    public var repoID: String
    public var commitSHA: String?
    public var requestedRevision: String?
    public var quantization: LocalModelQuantization?
    public var modelType: String
    public var architecture: String?
    public var runtimeKind: String?
    public var supportsVision: Bool?
    public var supportsMTP: Bool?
    public var isLoaded: Bool
    public var isActiveRevision: Bool
    public var integrityStatus: String
    public var verifiedAt: Date?
    public var metadataStatus: String

    enum CodingKeys: String, CodingKey {
        case provider
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case requestedRevision = "requested_revision"
        case quantization
        case modelType = "model_type"
        case architecture
        case runtimeKind = "runtime_kind"
        case supportsVision = "supports_vision"
        case supportsMTP = "supports_mtp"
        case isLoaded = "is_loaded"
        case isActiveRevision = "is_active_revision"
        case integrityStatus = "integrity_status"
        case verifiedAt = "verified_at"
        case metadataStatus = "metadata_status"
    }
}

extension JSONEncoder {
    static var ironMLXDiagnostic: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}
