import Foundation

/// Chat history and chat preferences under `~/.ironmlx/chats/`: one JSON file
/// per conversation, images in `attachments/`, preferences in
/// `preferences.json`. Nothing here touches `app_config.json`, whose keys are a
/// strict allow-list. Only the newest `maxConversations` are kept.
final class ChatStore {
    static let maxConversations = 20
    static let preferencesFileName = "preferences.json"

    static var defaultRoot: URL {
        IronMLXDataRoot.url
            .appendingPathComponent("chats", isDirectory: true)
    }

    let root: URL
    private let fileManager: FileManager
    private var preferences: Preferences

    init(root: URL = ChatStore.defaultRoot, fileManager: FileManager = .default) {
        self.root = root
        self.fileManager = fileManager
        preferences = Self.readPreferences(at: root.appendingPathComponent(Self.preferencesFileName))
    }

    var attachmentsDirectory: URL {
        root.appendingPathComponent("attachments", isDirectory: true)
    }

    // MARK: Conversations

    /// Newest first. Unreadable files are skipped.
    func list() -> [ChatConversation] {
        guard let names = try? fileManager.contentsOfDirectory(atPath: root.path) else { return [] }
        let decoder = Self.decoder()
        return names
            .filter { $0.hasSuffix(".json") && $0 != Self.preferencesFileName }
            .compactMap { name -> ChatConversation? in
                let url = root.appendingPathComponent(name)
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(ChatConversation.self, from: data)
            }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    func conversation(id: UUID) -> ChatConversation? {
        guard let data = try? Data(contentsOf: fileURL(for: id)) else { return nil }
        return try? Self.decoder().decode(ChatConversation.self, from: data)
    }

    /// Writes the conversation, then drops the oldest ones beyond the limit
    /// together with their images. The conversation just saved is never
    /// dropped. The first conversation ever saved is remembered; history from
    /// before that record existed has none.
    func save(_ conversation: ChatConversation) throws {
        try ensureRoot()
        let data = try Self.encoder().encode(conversation)
        try data.write(to: fileURL(for: conversation.id), options: [.atomic])
        let all = list()
        if preferences.firstConversationID == nil, all.map(\.id) == [conversation.id] {
            var updated = preferences
            updated.firstConversationID = conversation.id
            try writePreferences(updated)
        }
        try trim(all, keeping: conversation.id)
    }

    func delete(id: UUID) throws {
        let url = fileURL(for: id)
        if let conversation = conversation(id: id) {
            removeAttachments(of: conversation)
        }
        if fileManager.fileExists(atPath: url.path) {
            try fileManager.removeItem(at: url)
        }
    }

    // MARK: Attachments

    /// Stores image bytes and returns the file name to keep in `ChatTurn.attachments`.
    func saveAttachment(_ data: Data, ext: String) throws -> String {
        try ensureRoot()
        try fileManager.createDirectory(at: attachmentsDirectory, withIntermediateDirectories: true)
        let cleanExt = ext.lowercased().filter { $0.isLetter || $0.isNumber }
        let name = UUID().uuidString + (cleanExt.isEmpty ? "" : ".\(cleanExt)")
        try data.write(to: attachmentURL(for: name), options: [.atomic])
        return name
    }

    /// Only the last path component is used, so a stored name can never point
    /// outside `attachments/`.
    func attachmentURL(for name: String) -> URL {
        attachmentsDirectory.appendingPathComponent((name as NSString).lastPathComponent)
    }

    func attachmentData(named name: String) -> Data? {
        try? Data(contentsOf: attachmentURL(for: name))
    }

    // MARK: Preferences

    func reasoningChoice(for modelID: String) -> ReasoningChoice? {
        preferences.reasoningChoices[modelID]
    }

    func setReasoningChoice(_ choice: ReasoningChoice, for modelID: String) throws {
        var updated = preferences
        updated.reasoningChoices[modelID] = choice
        try writePreferences(updated)
    }

    var upgradePromptDismissed: Bool {
        get { preferences.upgradePromptDismissed }
        set { updatePreferences { $0.upgradePromptDismissed = newValue } }
    }

    /// The upgrade offer has been shown once; it is never shown again.
    var upgradePromptOffered: Bool {
        get { preferences.upgradePromptOffered }
        set { updatePreferences { $0.upgradePromptOffered = newValue } }
    }

    /// The user's first conversation, recorded when it is first saved.
    var firstConversationID: UUID? {
        preferences.firstConversationID
    }

    // MARK: Private

    private struct Preferences: Codable, Equatable {
        var reasoningChoices: [String: ReasoningChoice] = [:]
        var upgradePromptDismissed = false
        var upgradePromptOffered = false
        var firstConversationID: UUID?

        init() {}

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            // Unknown choices from a newer App version are dropped, not fatal.
            let raw = try container.decodeIfPresent([String: String].self, forKey: .reasoningChoices) ?? [:]
            reasoningChoices = raw.compactMapValues(ReasoningChoice.init(rawValue:))
            upgradePromptDismissed = try container.decodeIfPresent(Bool.self, forKey: .upgradePromptDismissed) ?? false
            upgradePromptOffered = try container.decodeIfPresent(Bool.self, forKey: .upgradePromptOffered) ?? false
            firstConversationID = try container.decodeIfPresent(UUID.self, forKey: .firstConversationID)
        }
    }

    private static func readPreferences(at url: URL) -> Preferences {
        guard let data = try? Data(contentsOf: url) else { return Preferences() }
        return (try? JSONDecoder().decode(Preferences.self, from: data)) ?? Preferences()
    }

    private func updatePreferences(_ change: (inout Preferences) -> Void) {
        var updated = preferences
        change(&updated)
        do {
            try writePreferences(updated)
        } catch {
            IronMLXAppLogger.error("Chat preferences could not be saved: \(error)")
        }
    }

    private func writePreferences(_ updated: Preferences) throws {
        try ensureRoot()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(updated).write(
            to: root.appendingPathComponent(Self.preferencesFileName), options: [.atomic]
        )
        preferences = updated
    }

    private func fileURL(for id: UUID) -> URL {
        root.appendingPathComponent("\(id.uuidString).json")
    }

    /// Conversations are private; the directory is readable by the owner only.
    private func ensureRoot() throws {
        guard !fileManager.fileExists(atPath: root.path) else { return }
        try fileManager.createDirectory(
            at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700]
        )
    }

    private func trim(_ all: [ChatConversation], keeping keptID: UUID) throws {
        guard all.count > Self.maxConversations else { return }
        let newestFirst = all.filter { $0.id == keptID } + all.filter { $0.id != keptID }
        let keptIDs = Set(newestFirst.prefix(Self.maxConversations).map(\.id))
        for conversation in all where !keptIDs.contains(conversation.id) {
            try delete(id: conversation.id)
        }
    }

    private func removeAttachments(of conversation: ChatConversation) {
        for name in conversation.turns.flatMap(\.attachments) {
            try? fileManager.removeItem(at: attachmentURL(for: name))
        }
    }

    private static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return encoder
    }

    private static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        return decoder
    }
}
