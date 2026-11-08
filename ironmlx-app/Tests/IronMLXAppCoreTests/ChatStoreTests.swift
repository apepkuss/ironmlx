import Foundation
import Testing

@testable import IronMLXAppCore

@Suite("Chat store")
struct ChatStoreTests {
    private func makeRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-store-\(UUID().uuidString)", isDirectory: true)
    }

    private func conversation(_ index: Int, attachments: [String] = []) -> ChatConversation {
        let date = Date(timeIntervalSince1970: 1_800_000_000 + Double(index) * 60)
        return ChatConversation(
            title: "c\(index)",
            createdAt: date,
            updatedAt: date,
            modelID: "m",
            turns: [ChatTurn(role: .user, text: "q\(index)", attachments: attachments),
                    ChatTurn(role: .assistant, text: "a\(index)")]
        )
    }

    private func jsonFiles(in root: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: root.path)
            .filter { $0.hasSuffix(".json") && $0 != "preferences.json" }
    }

    @Test func savesOneJSONFilePerConversation() throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatStore(root: root)
        let first = conversation(1)
        let second = conversation(2)
        try store.save(first)
        try store.save(second)
        var updated = first
        updated.title = "renamed"
        try store.save(updated)

        #expect(try Set(jsonFiles(in: root)) == ["\(first.id.uuidString).json", "\(second.id.uuidString).json"])
        let listed = store.list()
        #expect(listed.map(\.id) == [second.id, first.id])
        #expect(listed.last?.title == "renamed")
        #expect(listed.last?.turns == first.turns)
    }

    @Test func keepsOnlyNewest20AndDeletesTheirAttachments() throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatStore(root: root)
        let oldestAttachment = try store.saveAttachment(Data([1, 2, 3]), ext: "jpg")
        let keptAttachment = try store.saveAttachment(Data([4, 5, 6]), ext: "png")
        let oldest = conversation(0, attachments: [oldestAttachment])
        try store.save(oldest)
        for index in 1...19 {
            try store.save(conversation(index, attachments: index == 1 ? [keptAttachment] : []))
        }
        #expect(store.list().count == 20)
        #expect(FileManager.default.fileExists(atPath: store.attachmentURL(for: oldestAttachment).path))

        try store.save(conversation(20))

        let listed = store.list()
        #expect(listed.count == 20)
        #expect(!listed.contains { $0.id == oldest.id })
        #expect(try jsonFiles(in: root).count == 20)
        #expect(!FileManager.default.fileExists(atPath: store.attachmentURL(for: oldestAttachment).path))
        #expect(FileManager.default.fileExists(atPath: store.attachmentURL(for: keptAttachment).path))
    }

    @Test func deleteRemovesConversationAndItsAttachments() throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatStore(root: root)
        let name = try store.saveAttachment(Data([9]), ext: "jpg")
        let item = conversation(1, attachments: [name])
        try store.save(item)
        try store.delete(id: item.id)
        #expect(store.list().isEmpty)
        #expect(!FileManager.default.fileExists(atPath: store.attachmentURL(for: name).path))
    }

    @Test func preferencesLiveInPreferencesJSONNotAppConfig() throws {
        let ironmlx = makeRoot()
        defer { try? FileManager.default.removeItem(at: ironmlx) }
        try FileManager.default.createDirectory(at: ironmlx, withIntermediateDirectories: true)
        let appConfig = ironmlx.appendingPathComponent("app_config.json")
        let appConfigBytes = Data(#"{"schema_version":1}"#.utf8)
        try appConfigBytes.write(to: appConfig)
        let root = ironmlx.appendingPathComponent("chats", isDirectory: true)
        let store = ChatStore(root: root)

        #expect(store.reasoningChoice(for: "qwen") == nil)
        #expect(store.upgradePromptDismissed == false)
        try store.setReasoningChoice(.medium, for: "qwen")
        store.upgradePromptDismissed = true

        #expect(try Data(contentsOf: appConfig) == appConfigBytes)
        #expect(try FileManager.default.contentsOfDirectory(atPath: ironmlx.path).sorted() == ["app_config.json", "chats"])
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["preferences.json"])
        let reopened = ChatStore(root: root)
        #expect(reopened.reasoningChoice(for: "qwen") == .medium)
        #expect(reopened.upgradePromptDismissed)
    }

    @Test func recordsTheFirstConversationEverAndTheOfferInPreferences() throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatStore(root: root)
        #expect(store.firstConversationID == nil)
        #expect(store.upgradePromptOffered == false)
        let first = conversation(1)
        try store.save(first)
        try store.save(conversation(2))
        try store.save(first)
        #expect(store.firstConversationID == first.id)
        store.upgradePromptOffered = true

        let reopened = ChatStore(root: root)
        #expect(reopened.firstConversationID == first.id)
        #expect(reopened.upgradePromptOffered)
    }

    @Test func historyFromBeforeTheRecordHasNoFirstConversation() throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let older = ChatStore(root: root)
        try older.save(conversation(1))
        // Simulate history written before the first conversation was recorded.
        try FileManager.default.removeItem(at: root.appendingPathComponent("preferences.json"))
        let store = ChatStore(root: root)
        try store.save(conversation(2))
        #expect(store.firstConversationID == nil)
    }

    @Test func corruptConversationFileIsSkipped() throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ChatStore(root: root)
        let good = conversation(1)
        try store.save(good)
        try Data("{not json".utf8).write(to: root.appendingPathComponent("\(UUID().uuidString).json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("\(UUID().uuidString).json"))

        #expect(store.list().map(\.id) == [good.id])
    }

    @Test func attachmentNamesCannotEscapeTheAttachmentsDirectory() throws {
        let root = makeRoot()
        let store = ChatStore(root: root)
        let url = store.attachmentURL(for: "../../secret.json")
        #expect(url.deletingLastPathComponent().standardizedFileURL
            == root.appendingPathComponent("attachments", isDirectory: true).standardizedFileURL)
    }
}
