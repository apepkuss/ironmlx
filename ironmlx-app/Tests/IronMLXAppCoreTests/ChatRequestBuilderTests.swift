import Foundation
import Testing

@testable import IronMLXAppCore

@Suite("Chat request builder")
struct ChatRequestBuilderTests {
    private let toggle = BackendModelReasoning(nativeEfforts: [])

    private func conversation(_ turns: [ChatTurn]) -> ChatConversation {
        ChatConversation(modelID: "m", turns: turns)
    }

    private func input(_ request: ChatRequest) -> [[String: Any]] {
        request.body["input"] as? [[String: Any]] ?? []
    }

    private func parts(_ item: [String: Any]) -> [[String: Any]] {
        item["content"] as? [[String: Any]] ?? []
    }

    private func build(
        _ turns: [ChatTurn],
        reasoning: ReasoningChoice = .off,
        capability: BackendModelReasoning? = nil,
        contextWindow: Int = 262_144,
        configured: Int? = nil,
        images: [String: Data] = [:]
    ) -> ChatRequest {
        ChatRequestBuilder.build(
            conversation: conversation(turns),
            modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
            reasoning: reasoning,
            capability: capability,
            contextWindow: contextWindow,
            configuredMaxOutputTokens: configured,
            loadAttachment: { images[$0] }
        )
    }

    @Test func buildsStreamingStatelessRequest() throws {
        let image = Data([0xFF, 0xD8, 0xFF, 0x01])
        let request = build([
            ChatTurn(role: .user, text: "hello"),
            ChatTurn(role: .assistant, text: "hi there", reasoning: "thought"),
            ChatTurn(role: .user, text: "what is this?", attachments: ["a.jpg"]),
        ], images: ["a.jpg": image])

        #expect(request.body["stream"] as? Bool == true)
        #expect(request.body["store"] as? Bool == false)
        #expect(request.body["model"] as? String == "mlx-community/Qwen3.5-4B-MLX-4bit")
        #expect(request.droppedTurnCount == 0)
        let items = input(request)
        #expect(items.map { $0["role"] as? String } == ["user", "assistant", "user"])
        #expect(items.allSatisfy { $0["type"] as? String == "message" })
        #expect(parts(items[0]).first?["type"] as? String == "input_text")
        #expect(parts(items[0]).first?["text"] as? String == "hello")
        #expect(parts(items[1]).first?["type"] as? String == "output_text")
        #expect(parts(items[1]).first?["text"] as? String == "hi there")
        let last = parts(items[2])
        #expect(last.count == 2)
        #expect(last[0]["type"] as? String == "input_image")
        #expect(last[0]["image_url"] as? String == "data:image/jpeg;base64,\(image.base64EncodedString())")
        #expect(last[0]["file_id"] == nil)
        #expect(last[1]["type"] as? String == "input_text")
        #expect(last[1]["text"] as? String == "what is this?")
        // The body must be valid JSON for URLSession.
        #expect(JSONSerialization.isValidJSONObject(request.body))
    }

    @Test func reasoningMapping() {
        let turns = [ChatTurn(role: .user, text: "q")]
        let cases: [(ReasoningChoice, String)] = [
            (.off, "none"), (.on, "medium"), (.low, "low"), (.medium, "medium"), (.high, "high"),
        ]
        for (choice, effort) in cases {
            let body = build(turns, reasoning: choice, capability: toggle).body
            #expect((body["reasoning"] as? [String: Any])?["effort"] as? String == effort)
        }
        #expect(build(turns, reasoning: .on, capability: nil).body["reasoning"] == nil)
        #expect(build(turns, reasoning: .off, capability: nil).body["reasoning"] == nil)
    }

    @Test func outputBudget() {
        let turns = [ChatTurn(role: .user, text: String(repeating: "a", count: 200))]
        #expect(build(turns, configured: 2048).body["max_output_tokens"] as? Int == 2048)
        #expect(build(turns, reasoning: .on, capability: toggle, configured: 2048).body["max_output_tokens"] as? Int == 2048)
        #expect(build(turns).body["max_output_tokens"] as? Int == 4096)
        #expect(build(turns, reasoning: .on, capability: nil).body["max_output_tokens"] as? Int == 4096)
        #expect(build(turns, reasoning: .on, capability: toggle).body["max_output_tokens"] as? Int == 8192)
        // Output takes at most half of a 3000-token window.
        #expect(build(turns, contextWindow: 3000).body["max_output_tokens"] as? Int == 1500)
        #expect(build(turns, reasoning: .on, capability: toggle, contextWindow: 3000, configured: 16_384)
            .body["max_output_tokens"] as? Int == 1500)
        // A message larger than half the window gets what the estimate leaves:
        // 4000 bytes ≈ 2000 input tokens.
        let long = [ChatTurn(role: .user, text: String(repeating: "a", count: 4000))]
        #expect(build(long, contextWindow: 3000).body["max_output_tokens"] as? Int == 1000)
    }

    @Test func outputReachingTheContextKeepsHistory() {
        let turns = [
            ChatTurn(role: .user, text: "first"), ChatTurn(role: .assistant, text: "reply"),
            ChatTurn(role: .user, text: "second"),
        ]
        let request = build(turns, contextWindow: 4096, configured: 4096)
        #expect(request.droppedTurnCount == 0)
        #expect(input(request).count == 3)
        #expect(request.body["max_output_tokens"] as? Int == 2048)
    }

    @Test func trimsOldestTurnsToFitContext() {
        // Each text turn costs 500 tokens (1000 UTF-8 bytes / 2); an image costs 1024.
        let text = String(repeating: "x", count: 1000)
        let turns = [
            ChatTurn(role: .user, text: text, attachments: ["old.jpg"]),
            ChatTurn(role: .assistant, text: text),
            ChatTurn(role: .user, text: text),
            ChatTurn(role: .assistant, text: text),
            ChatTurn(role: .user, text: "newest " + text),
        ]
        let images = ["old.jpg": Data([1])]
        // All five turns are 3527 input tokens; with 1024 output they fit in 10000 but not in 3600.
        let full = build(turns, contextWindow: 10_000, configured: 1024, images: images)
        #expect(full.droppedTurnCount == 0)
        #expect(input(full).count == 5)

        let trimmed = build(turns, contextWindow: 3600, configured: 1024, images: images)
        #expect(trimmed.droppedTurnCount == 2)
        #expect(input(trimmed).count == 3)
        #expect((parts(input(trimmed)[0]).first?["text"] as? String) == text)
        #expect(trimmed.body["max_output_tokens"] as? Int == 1024)

        let onlyLast = build(turns, contextWindow: 1200, configured: 1024, images: images)
        #expect(onlyLast.droppedTurnCount == 4)
        #expect(input(onlyLast).count == 1)
        #expect((parts(input(onlyLast)[0]).first?["text"] as? String)?.hasPrefix("newest") == true)

        // The last user message is never dropped, even when it alone is too long.
        let tooLong = build(turns, contextWindow: 100, configured: 1024, images: images)
        #expect(tooLong.droppedTurnCount == 4)
        #expect(input(tooLong).count == 1)
        #expect(tooLong.body["max_output_tokens"] as? Int == 1)
    }

    @Test func extraDropsRemoveOneMoreExchange() {
        let turns = [
            ChatTurn(role: .user, text: "1"), ChatTurn(role: .assistant, text: "1"),
            ChatTurn(role: .user, text: "2"), ChatTurn(role: .assistant, text: "2"),
            ChatTurn(role: .user, text: "3"),
        ]
        let request = ChatRequestBuilder.build(
            conversation: conversation(turns), modelID: "m", reasoning: .off, capability: nil,
            contextWindow: 100_000, configuredMaxOutputTokens: nil, minimumDroppedTurns: 1
        )
        #expect(request.droppedTurnCount == 2)
        #expect(input(request).count == 3)
    }

    @Test func orphanUserTurnsAreLeftOutSoRolesAlternate() {
        // q1 was stopped before any text, q2 failed without a reply; strict
        // templates reject two user messages in a row.
        let request = build([
            ChatTurn(role: .user, text: "q1"),
            ChatTurn(role: .assistant, text: "", stopped: true),
            ChatTurn(role: .user, text: "q2"),
            ChatTurn(role: .user, text: "q3"),
            ChatTurn(role: .assistant, text: "a3"),
            ChatTurn(role: .user, text: "q4"),
        ])
        let items = input(request)
        #expect(items.map { $0["role"] as? String } == ["user", "assistant", "user"])
        #expect(items.map { parts($0).first?["text"] as? String } == ["q3", "a3", "q4"])
        // Leaving out unanswered questions is not trimming.
        #expect(request.droppedTurnCount == 0)
    }

    @Test func pngAttachmentKeepsItsMediaType() {
        let request = build([ChatTurn(role: .user, text: "q", attachments: ["b.png"])], images: ["b.png": Data([2])])
        #expect((parts(input(request)[0]).first?["image_url"] as? String)?.hasPrefix("data:image/png;base64,") == true)
    }
}
