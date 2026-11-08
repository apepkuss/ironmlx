import Foundation

enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
}

/// Speed figures shown under a finished reply. All durations are in seconds.
struct ChatMetrics: Codable, Equatable, Sendable {
    /// From sending to the first streamed token of any kind.
    var firstTokenSeconds: Double?
    /// From the first reasoning token to the first answer token; only with thinking on.
    var thinkingSeconds: Double?
    /// Output tokens over the time from the first token to the end of the reply.
    var tokensPerSecond: Double?
    var outputTokens: Int?
}

/// One message. `attachments` holds file names inside the store's
/// `attachments/` directory.
struct ChatTurn: Codable, Equatable, Sendable {
    var role: ChatRole
    var text: String
    var reasoning: String?
    var attachments: [String]
    var metrics: ChatMetrics?
    var stopped: Bool
    /// The reply hit its output budget (`response.incomplete`) and was cut off.
    var truncated: Bool?

    init(
        role: ChatRole,
        text: String,
        reasoning: String? = nil,
        attachments: [String] = [],
        metrics: ChatMetrics? = nil,
        stopped: Bool = false,
        truncated: Bool? = nil
    ) {
        self.role = role
        self.text = text
        self.reasoning = reasoning
        self.attachments = attachments
        self.metrics = metrics
        self.stopped = stopped
        self.truncated = truncated
    }
}

struct ChatConversation: Codable, Identifiable, Equatable, Sendable {
    let id: UUID
    var title: String
    var createdAt: Date
    var updatedAt: Date
    var modelID: String
    var turns: [ChatTurn]

    init(
        id: UUID = UUID(),
        title: String = "",
        createdAt: Date = Date(),
        updatedAt: Date? = nil,
        modelID: String,
        turns: [ChatTurn] = []
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt ?? createdAt
        self.modelID = modelID
        self.turns = turns
    }

    /// The first user message, on one line, cut to `limit` characters.
    static func title(from text: String, limit: Int = 30) -> String {
        let oneLine = text
            .components(separatedBy: .newlines)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)
        guard oneLine.count > limit else { return oneLine }
        return String(oneLine.prefix(limit)) + "…"
    }
}

/// The user's thinking setting for one model. `on` is the plain switch;
/// `low`/`medium`/`high` are native levels.
enum ReasoningChoice: String, Codable, Sendable {
    case off
    case on
    case low
    case medium
    case high
}
