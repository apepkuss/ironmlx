import Foundation

public struct BackendIncidentRecord: Codable, Equatable, Sendable {
    public var id: UUID
    public var occurredAt: Date
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
    public var recoveryResult: String?
    public var error: String?
    public var logTail: String

    public init(
        id: UUID = UUID(),
        termination: BackendProcessTermination,
        snapshot: BackendRecoverySnapshot
    ) {
        self.id = id
        self.occurredAt = termination.occurredAt
        self.launchID = termination.launchID
        self.generation = termination.generation
        self.pid = termination.pid
        self.terminationStatus = termination.terminationStatus
        self.terminationReason = termination.terminationReason
        self.stopIntent = termination.stopIntent
        self.recoveryAttempt = 0
        self.modelsBeforeCrash = snapshot.models.map(\.id)
        self.defaultModel = snapshot.config.defaultModelReference
        self.pinnedModels = snapshot.config.pinnedModelReferences
        self.recoveredModels = []
        self.failedModels = []
        self.failures = []
        self.recoveryResult = nil
        self.error = nil
        self.logTail = Self.sanitizedLogTail(termination.logTail)
    }

    static func sanitizedLogTail(_ text: String) -> String {
        let patterns = [
            #"(?i)("(?:authorization|api[_-]?key|token)"\s*:\s*")([^"]+)"#,
            #"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"#,
            #"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]+)"#,
            #"(?i)((?:api[_-]?key|token)\s*[:=]\s*)([^\s,;]+)"#,
        ]
        return patterns.reduce(text) { value, pattern in
            guard let expression = try? NSRegularExpression(pattern: pattern) else {
                return value
            }
            let range = NSRange(value.startIndex..., in: value)
            return expression.stringByReplacingMatches(
                in: value,
                range: range,
                withTemplate: "$1<redacted>"
            )
        }
    }
}

@MainActor
public final class BackendIncidentStore {
    public static let defaultRetainedIncidents = 20

    public let url: URL
    public let retainedIncidents: Int
    private let fileManager: FileManager

    public init(
        url: URL = BackendIncidentStore.defaultURL(),
        retainedIncidents: Int = BackendIncidentStore.defaultRetainedIncidents,
        fileManager: FileManager = .default
    ) {
        self.url = url
        self.retainedIncidents = max(1, retainedIncidents)
        self.fileManager = fileManager
    }

    public func records() -> [BackendIncidentRecord] {
        guard let data = try? Data(contentsOf: url),
              let records = try? JSONDecoder.ironMLXIncident.decode(
                  [BackendIncidentRecord].self,
                  from: data
              )
        else {
            return []
        }
        return records
    }

    public func upsert(_ record: BackendIncidentRecord) throws {
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
        let data = try JSONEncoder.ironMLXIncident.encode(current)
        try data.write(to: url, options: .atomic)
    }

    public static func defaultURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("incidents", isDirectory: true)
            .appendingPathComponent("backend-incidents.json")
    }
}

private extension JSONEncoder {
    static var ironMLXIncident: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static var ironMLXIncident: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
