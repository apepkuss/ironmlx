import Foundation

public struct ModelDownloadRecoveryReminder: Codable, Equatable, Sendable {
    public var provider: ModelRepositoryProvider
    public var repoID: String
    public var queueOrder: Int
    public var previousStatus: String
    public var usedCredential: Bool
    public var enqueuedAt: Date

    public init(
        provider: ModelRepositoryProvider,
        repoID: String,
        queueOrder: Int,
        previousStatus: String,
        usedCredential: Bool,
        enqueuedAt: Date = Date()
    ) {
        self.provider = provider
        self.repoID = repoID
        self.queueOrder = queueOrder
        self.previousStatus = previousStatus
        self.usedCredential = usedCredential
        self.enqueuedAt = enqueuedAt
    }

    enum CodingKeys: String, CodingKey {
        case provider
        case repoID = "repo_id"
        case queueOrder = "queue_order"
        case previousStatus = "previous_status"
        case usedCredential = "used_credential"
        case enqueuedAt = "enqueued_at"
    }
}

struct ModelDownloadQueueReminderStore: Sendable {
    private struct Document: Codable, Sendable {
        var version: Int
        var reminders: [ModelDownloadRecoveryReminder]
    }

    private let url: URL

    init(rootURL: URL) {
        url = rootURL
            .appendingPathComponent("state", isDirectory: true)
            .appendingPathComponent("model-download-reminders.json")
    }

    func load() -> [ModelDownloadRecoveryReminder] {
        guard let data = try? Data(contentsOf: url),
              let document = try? JSONDecoder().decode(Document.self, from: data),
              document.version == 1
        else {
            return []
        }
        return document.reminders.sorted {
            if $0.queueOrder == $1.queueOrder {
                return $0.enqueuedAt < $1.enqueuedAt
            }
            return $0.queueOrder < $1.queueOrder
        }
    }

    func save(_ reminders: [ModelDownloadRecoveryReminder]) throws {
        if reminders.isEmpty {
            try? FileManager.default.removeItem(at: url)
            return
        }
        try ModelDownloadStore.atomicWrite(
            Document(version: 1, reminders: reminders),
            to: url
        )
    }
}
