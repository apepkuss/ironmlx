import Foundation

public enum BackendIncidentType: String, Codable, Equatable, Sendable {
    case unexpectedBackendExit = "unexpected_backend_exit"
}

public enum BackendIncidentRecoveryStatus: String, Codable, Equatable, Sendable {
    case recoveryPending = "recovery_pending"
    case recovering
    case recovered
    case partiallyRecovered = "partially_recovered"
    case recoveryFailed = "recovery_failed"
    case automaticRecoveryStopped = "automatic_recovery_stopped"
}

public enum BackendIncidentRecoveryAction: String, Codable, Equatable, Sendable {
    case automaticRestartStarted = "automatic_restart_started"
    case readinessCheckPassed = "readiness_check_passed"
    case modelRestoreStarted = "model_restore_started"
    case stableWindowStarted = "stable_window_started"
    case automaticRecoveryStopped = "automatic_recovery_stopped"
}

public struct BackendIncidentRecoveryStep: Codable, Equatable, Sendable {
    public var action: BackendIncidentRecoveryAction
    public var occurredAt: Date

    public init(action: BackendIncidentRecoveryAction, occurredAt: Date = Date()) {
        self.action = action
        self.occurredAt = occurredAt
    }

    enum CodingKeys: String, CodingKey {
        case action
        case occurredAt = "occurred_at"
    }
}

public struct BackendIncidentRecord: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var incidentType: BackendIncidentType
    public var id: UUID
    public var occurredAt: Date
    public var updatedAt: Date
    public var launchID: UUID
    public var generation: UInt64
    public var pid: Int32
    public var terminationStatus: Int32
    public var terminationReason: String
    public var stopIntent: BackendStopIntent
    public var recoveryAttempt: Int
    public var modelsBeforeCrash: [String]
    public var defaultModel: String?
    public var pinnedModels: [String]
    public var recoveredModels: [String]
    public var failedModels: [String]
    public var failures: [BackendModelRecoveryFailure]
    public var recoverySteps: [BackendIncidentRecoveryStep]
    public var recoveryResult: String?
    public var error: String?
    public var logTail: String

    public var recoveryStatus: BackendIncidentRecoveryStatus {
        switch recoveryResult {
        case "recovered":
            .recovered
        case "degraded":
            .partiallyRecovered
        case "failed":
            .recoveryFailed
        case "breaker":
            .automaticRecoveryStopped
        case nil where recoveryAttempt == 0:
            .recoveryPending
        case nil:
            .recovering
        default:
            .recoveryFailed
        }
    }

    public var primaryFailureReason: BackendRecoveryFailureReason? {
        if recoveryStatus == .automaticRecoveryStopped {
            return .crashLoopBreaker
        }
        if let reason = failures.first?.reason {
            return reason
        }
        if recoveryStatus == .recoveryFailed {
            return .unknownModelLoadFailure
        }
        return nil
    }

    public init(
        id: UUID = UUID(),
        termination: BackendProcessTermination,
        snapshot: BackendRecoverySnapshot
    ) {
        schemaVersion = Self.currentSchemaVersion
        incidentType = .unexpectedBackendExit
        self.id = id
        occurredAt = termination.occurredAt
        updatedAt = termination.occurredAt
        launchID = termination.launchID
        generation = termination.generation
        pid = termination.pid
        terminationStatus = termination.terminationStatus
        terminationReason = termination.terminationReason
        stopIntent = termination.stopIntent
        recoveryAttempt = 0
        modelsBeforeCrash = snapshot.models.map(\.id)
        defaultModel = snapshot.config.defaultModelReference
        pinnedModels = snapshot.config.pinnedModelReferences
        recoveredModels = []
        failedModels = []
        failures = []
        recoverySteps = []
        recoveryResult = nil
        error = nil
        logTail = Self.sanitizedLogTail(termination.logTail)
    }

    public mutating func appendRecoveryStep(
        _ action: BackendIncidentRecoveryAction,
        at date: Date = Date()
    ) {
        recoverySteps.append(BackendIncidentRecoveryStep(action: action, occurredAt: date))
        updatedAt = date
    }

    static func sanitizedLogTail(_ text: String) -> String {
        BackendIncidentPrivacy.sanitizedLogTail(text)
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case incidentType
        case id
        case occurredAt
        case updatedAt
        case launchID
        case generation
        case pid
        case terminationStatus
        case terminationReason
        case stopIntent
        case recoveryAttempt
        case modelsBeforeCrash
        case defaultModel
        case pinnedModels
        case recoveredModels
        case failedModels
        case failures
        case recoverySteps
        case recoveryResult
        case error
        case logTail
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
            ?? Self.currentSchemaVersion
        incidentType = try container.decodeIfPresent(
            BackendIncidentType.self,
            forKey: .incidentType
        ) ?? .unexpectedBackendExit
        id = try container.decode(UUID.self, forKey: .id)
        occurredAt = try container.decode(Date.self, forKey: .occurredAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? occurredAt
        launchID = try container.decode(UUID.self, forKey: .launchID)
        generation = try container.decode(UInt64.self, forKey: .generation)
        pid = try container.decode(Int32.self, forKey: .pid)
        terminationStatus = try container.decode(Int32.self, forKey: .terminationStatus)
        terminationReason = try container.decode(String.self, forKey: .terminationReason)
        stopIntent = try container.decode(BackendStopIntent.self, forKey: .stopIntent)
        recoveryAttempt = try container.decode(Int.self, forKey: .recoveryAttempt)
        modelsBeforeCrash = try container.decode([String].self, forKey: .modelsBeforeCrash)
        defaultModel = try container.decodeIfPresent(String.self, forKey: .defaultModel)
        pinnedModels = try container.decode([String].self, forKey: .pinnedModels)
        recoveredModels = try container.decode([String].self, forKey: .recoveredModels)
        failedModels = try container.decode([String].self, forKey: .failedModels)
        failures = try container.decode([BackendModelRecoveryFailure].self, forKey: .failures)
        recoverySteps = try container.decodeIfPresent(
            [BackendIncidentRecoveryStep].self,
            forKey: .recoverySteps
        ) ?? []
        recoveryResult = try container.decodeIfPresent(String.self, forKey: .recoveryResult)
        error = try container.decodeIfPresent(String.self, forKey: .error)
        logTail = try container.decode(String.self, forKey: .logTail)
    }
}

public struct BackendIncidentQuery: Equatable, Sendable {
    public var status: BackendIncidentRecoveryStatus?
    public var model: String?
    public var reason: BackendRecoveryFailureReason?
    public var from: Date?
    public var to: Date?
    public var limit: Int?

    public init(
        status: BackendIncidentRecoveryStatus? = nil,
        model: String? = nil,
        reason: BackendRecoveryFailureReason? = nil,
        from: Date? = nil,
        to: Date? = nil,
        limit: Int? = nil
    ) {
        self.status = status
        self.model = model?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.reason = reason
        self.from = from
        self.to = to
        self.limit = limit
    }

    func matches(_ record: BackendIncidentRecord) -> Bool {
        if let status, record.recoveryStatus != status {
            return false
        }
        if let model, !model.isEmpty {
            let needle = model.localizedLowercase
            let models = record.modelsBeforeCrash + record.recoveredModels + record.failedModels
            guard models.contains(where: { $0.localizedLowercase.contains(needle) }) else {
                return false
            }
        }
        if let reason, record.primaryFailureReason != reason,
           !record.failures.contains(where: { $0.reason == reason }) {
            return false
        }
        if let from, record.occurredAt < from {
            return false
        }
        if let to, record.occurredAt > to {
            return false
        }
        return true
    }
}

public struct BackendIncidentSummary: Codable, Equatable, Sendable {
    public var id: UUID
    public var schemaVersion: Int
    public var incidentType: BackendIncidentType
    public var occurredAt: Date
    public var updatedAt: Date
    public var affectedModels: [String]
    public var recoveryStatus: BackendIncidentRecoveryStatus
    public var recoveryAttempt: Int
    public var primaryFailureReason: BackendRecoveryFailureReason?

    init(record: BackendIncidentRecord) {
        id = record.id
        schemaVersion = record.schemaVersion
        incidentType = record.incidentType
        occurredAt = record.occurredAt
        updatedAt = record.updatedAt
        affectedModels = record.modelsBeforeCrash
        recoveryStatus = record.recoveryStatus
        recoveryAttempt = record.recoveryAttempt
        primaryFailureReason = record.primaryFailureReason
    }

    enum CodingKeys: String, CodingKey {
        case id
        case schemaVersion = "schema_version"
        case incidentType = "incident_type"
        case occurredAt = "occurred_at"
        case updatedAt = "updated_at"
        case affectedModels = "affected_models"
        case recoveryStatus = "recovery_status"
        case recoveryAttempt = "recovery_attempt"
        case primaryFailureReason = "primary_failure_reason"
    }
}

public struct BackendIncidentDetail: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var incidentType: BackendIncidentType
    public var id: UUID
    public var occurredAt: Date
    public var updatedAt: Date
    public var launchID: UUID
    public var generation: UInt64
    public var pid: Int32
    public var terminationStatus: Int32
    public var terminationReason: String
    public var modelsBeforeCrash: [String]
    public var defaultModel: String?
    public var pinnedModels: [String]
    public var recoveredModels: [String]
    public var failedModels: [String]
    public var failures: [BackendModelRecoveryFailure]
    public var recoverySteps: [BackendIncidentRecoveryStep]
    public var recoveryAttempt: Int
    public var recoveryStatus: BackendIncidentRecoveryStatus
    public var primaryFailureReason: BackendRecoveryFailureReason?
    public var error: String?
    public var logTail: String

    init(record: BackendIncidentRecord) {
        schemaVersion = record.schemaVersion
        incidentType = record.incidentType
        id = record.id
        occurredAt = record.occurredAt
        updatedAt = record.updatedAt
        launchID = record.launchID
        generation = record.generation
        pid = record.pid
        terminationStatus = record.terminationStatus
        terminationReason = record.terminationReason
        modelsBeforeCrash = record.modelsBeforeCrash
        defaultModel = record.defaultModel
        pinnedModels = record.pinnedModels
        recoveredModels = record.recoveredModels
        failedModels = record.failedModels
        failures = record.failures
        recoverySteps = record.recoverySteps
        recoveryAttempt = record.recoveryAttempt
        recoveryStatus = record.recoveryStatus
        primaryFailureReason = record.primaryFailureReason
        error = record.error
        logTail = record.logTail
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case incidentType = "incident_type"
        case id
        case occurredAt = "occurred_at"
        case updatedAt = "updated_at"
        case launchID = "launch_id"
        case generation
        case pid
        case terminationStatus = "termination_status"
        case terminationReason = "termination_reason"
        case modelsBeforeCrash = "models_before_crash"
        case defaultModel = "default_model"
        case pinnedModels = "pinned_models"
        case recoveredModels = "recovered_models"
        case failedModels = "failed_models"
        case failures
        case recoverySteps = "recovery_steps"
        case recoveryAttempt = "recovery_attempt"
        case recoveryStatus = "recovery_status"
        case primaryFailureReason = "primary_failure_reason"
        case error
        case logTail = "log_tail"
    }
}

public struct BackendIncidentListPayload: Codable, Equatable, Sendable {
    public var success = true
    public var schemaVersion = BackendIncidentRecord.currentSchemaVersion
    public var total: Int
    public var returned: Int
    public var retentionLimit: Int
    public var oldestRetainedOccurredAt: Date?
    public var incidents: [BackendIncidentSummary]

    enum CodingKeys: String, CodingKey {
        case success
        case schemaVersion = "schema_version"
        case total
        case returned
        case retentionLimit = "retention_limit"
        case oldestRetainedOccurredAt = "oldest_retained_occurred_at"
        case incidents
    }
}

public struct BackendIncidentDetailPayload: Codable, Equatable, Sendable {
    public var success = true
    public var incident: BackendIncidentDetail
}

public struct BackendIncidentExportPayload: Codable, Equatable, Sendable {
    public static let currentFormatVersion = 1

    public var exportFormatVersion = Self.currentFormatVersion
    public var generatedAt: Date
    public var truncated: Bool
    public var retentionLimit: Int
    public var maximumBytes: Int
    public var incidents: [BackendIncidentDetail]

    enum CodingKeys: String, CodingKey {
        case exportFormatVersion = "export_format_version"
        case generatedAt = "generated_at"
        case truncated
        case retentionLimit = "retention_limit"
        case maximumBytes = "maximum_bytes"
        case incidents
    }
}

enum BackendIncidentStoreError: LocalizedError {
    case recordExceedsCapacity
    case exportExceedsCapacity

    var errorDescription: String? {
        switch self {
        case .recordExceedsCapacity:
            "The incident record exceeds the local history capacity."
        case .exportExceedsCapacity:
            "The incident export exceeds the diagnostic export capacity."
        }
    }
}

@MainActor
public final class BackendIncidentStore {
    public static let defaultRetainedIncidents = 20
    public static let defaultMaximumStoreBytes = 1_048_576
    public static let defaultMaximumExportBytes = 524_288

    public let url: URL
    public let retainedIncidents: Int
    public let maximumStoreBytes: Int
    public let maximumExportBytes: Int
    private let fileManager: FileManager

    public init(
        url: URL = BackendIncidentStore.defaultURL(),
        retainedIncidents: Int = BackendIncidentStore.defaultRetainedIncidents,
        maximumStoreBytes: Int = BackendIncidentStore.defaultMaximumStoreBytes,
        maximumExportBytes: Int = BackendIncidentStore.defaultMaximumExportBytes,
        fileManager: FileManager = .default
    ) {
        self.url = url
        self.retainedIncidents = max(1, retainedIncidents)
        self.maximumStoreBytes = max(1_024, maximumStoreBytes)
        self.maximumExportBytes = max(1_024, maximumExportBytes)
        self.fileManager = fileManager
    }

    public func records() -> [BackendIncidentRecord] {
        guard fileSize() <= maximumStoreBytes,
              let data = try? Data(contentsOf: url),
              let records = try? JSONDecoder.ironMLXIncident.decode(
                  [BackendIncidentRecord].self,
                  from: data
              )
        else {
            return []
        }
        return records
            .map(Self.normalized)
            .sorted { $0.occurredAt < $1.occurredAt }
            .suffix(retainedIncidents)
    }

    public func records(matching query: BackendIncidentQuery) -> [BackendIncidentRecord] {
        let limit = min(max(1, query.limit ?? retainedIncidents), retainedIncidents)
        return Array(
            records()
                .reversed()
                .filter(query.matches)
                .prefix(limit)
        )
    }

    public func listPayload(matching query: BackendIncidentQuery) -> BackendIncidentListPayload {
        let retainedRecords = records()
        let allMatches = retainedRecords.reversed().filter(query.matches)
        let limit = min(max(1, query.limit ?? retainedIncidents), retainedIncidents)
        let selected = Array(allMatches.prefix(limit))
        return BackendIncidentListPayload(
            total: allMatches.count,
            returned: selected.count,
            retentionLimit: retainedIncidents,
            oldestRetainedOccurredAt: retainedRecords.first?.occurredAt,
            incidents: selected.map(BackendIncidentSummary.init)
        )
    }

    public func detail(id: UUID) -> BackendIncidentDetail? {
        records().first(where: { $0.id == id }).map(BackendIncidentDetail.init)
    }

    public func upsert(_ record: BackendIncidentRecord) throws {
        let record = Self.normalized(record)
        var current = records()
        if let index = current.firstIndex(where: { $0.id == record.id }) {
            current[index] = record
        } else {
            current.append(record)
        }
        current.sort { $0.occurredAt < $1.occurredAt }
        current = Array(current.suffix(retainedIncidents))
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var data = try JSONEncoder.ironMLXIncident.encode(current)
        while data.count > maximumStoreBytes, current.count > 1 {
            current.removeFirst()
            data = try JSONEncoder.ironMLXIncident.encode(current)
        }
        guard data.count <= maximumStoreBytes else {
            throw BackendIncidentStoreError.recordExceedsCapacity
        }
        try data.write(to: url, options: .atomic)
    }

    public func clear() throws {
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("[]".utf8).write(to: url, options: .atomic)
    }

    public func exportData(matching query: BackendIncidentQuery) throws -> Data {
        var incidents = records(matching: query).map(BackendIncidentDetail.init)
        let originalCount = incidents.count
        var payload = BackendIncidentExportPayload(
            generatedAt: Date(),
            truncated: false,
            retentionLimit: retainedIncidents,
            maximumBytes: maximumExportBytes,
            incidents: incidents
        )
        var data = try JSONEncoder.ironMLXIncident.encode(payload)
        while data.count > maximumExportBytes, !incidents.isEmpty {
            incidents.removeLast()
            payload.incidents = incidents
            payload.truncated = incidents.count < originalCount
            data = try JSONEncoder.ironMLXIncident.encode(payload)
        }
        guard data.count <= maximumExportBytes else {
            throw BackendIncidentStoreError.exportExceedsCapacity
        }
        return data
    }

    public static func defaultURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("incidents", isDirectory: true)
            .appendingPathComponent("backend-incidents.json")
    }

    private func fileSize() -> Int {
        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber
        else {
            return 0
        }
        return size.intValue
    }

    private static func normalized(_ source: BackendIncidentRecord) -> BackendIncidentRecord {
        var record = source
        record.schemaVersion = BackendIncidentRecord.currentSchemaVersion
        record.modelsBeforeCrash = sanitizedModelReferences(record.modelsBeforeCrash)
        record.defaultModel = record.defaultModel.map(sanitizedModelReference)
        record.pinnedModels = sanitizedModelReferences(record.pinnedModels)
        record.recoveredModels = sanitizedModelReferences(record.recoveredModels)
        record.failedModels = sanitizedModelReferences(record.failedModels)
        record.failures = Array(record.failures.prefix(100)).map { failure in
            var failure = failure
            failure.model = sanitizedModelReference(failure.model)
            failure.message = BackendIncidentPrivacy.sanitizedText(
                failure.message,
                maximumBytes: 2_048
            )
            return failure
        }
        record.recoverySteps = Array(record.recoverySteps.prefix(50))
        record.error = record.error.map {
            BackendIncidentPrivacy.sanitizedText($0, maximumBytes: 4_096)
        }
        record.logTail = BackendIncidentPrivacy.sanitizedLogTail(record.logTail)
        return record
    }

    private static func sanitizedModelReferences(_ values: [String]) -> [String] {
        Array(values.prefix(100)).map(sanitizedModelReference)
    }

    private static func sanitizedModelReference(_ value: String) -> String {
        BackendIncidentPrivacy.sanitizedText(value, maximumBytes: 1_024)
    }
}

enum BackendIncidentPrivacy {
    static let maximumLogTailBytes = 32_768

    static func sanitizedLogTail(_ text: String) -> String {
        DiagnosticPrivacy.sanitizedLog(text, maximumBytes: maximumLogTailBytes)
    }

    static func sanitizedText(_ text: String, maximumBytes: Int) -> String {
        DiagnosticPrivacy.sanitizedText(text, maximumBytes: maximumBytes)
    }
}

extension JSONEncoder {
    static var ironMLXIncident: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

extension JSONDecoder {
    static var ironMLXIncident: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
