import Foundation

/// A `/v1/responses` body plus how many history turns were left out to fit
/// the context window.
struct ChatRequest {
    let body: [String: Any]
    /// Number of `ChatTurn`s (user and assistant messages) left out, oldest first.
    let droppedTurnCount: Int
}

/// Builds the stateless streaming request for one reply. The whole history is
/// sent each time (`store: false`); when history plus the output budget does
/// not fit, whole exchanges are dropped from the oldest, never the last user
/// message.
enum ChatRequestBuilder {
    /// Output budget when the model has no configured `max_output_tokens`.
    /// Thinking and the answer share it, so thinking gets more.
    static let defaultOutputTokens = 4096
    static let defaultThinkingOutputTokens = 8192
    /// Rough input cost without a tokenizer: UTF-8 bytes / 2, and a flat
    /// cost per image.
    static let bytesPerToken = 2
    static let tokensPerImage = 1024

    static func build(
        conversation: ChatConversation,
        modelID: String,
        reasoning: ReasoningChoice,
        capability: BackendModelReasoning?,
        contextWindow: Int,
        configuredMaxOutputTokens: Int?,
        minimumDroppedTurns: Int = 0,
        loadAttachment: (String) -> Data? = { _ in nil }
    ) -> ChatRequest {
        let thinking = capability != nil && reasoning != .off
        let desiredOutput = configuredMaxOutputTokens.flatMap { $0 > 0 ? $0 : nil }
            ?? (thinking ? defaultThinkingOutputTokens : defaultOutputTokens)

        let turns = conversation.turns
        var messages = turns.indices.compactMap { index in
            isAnswered(index, in: turns) ? message(for: turns[index], at: index, loadAttachment: loadAttachment) : nil
        }
        // Once trimming starts, everything before the first kept message is
        // gone, including unanswered questions that were never sent.
        var trimmed = false
        func droppedTurns() -> Int { trimmed ? messages.first?.turnIndex ?? turns.count : 0 }
        func inputTokens() -> Int { messages.reduce(0) { $0 + $1.tokens } }
        while let exchange = oldestExchangeLength(in: messages),
              droppedTurns() < minimumDroppedTurns || inputTokens() + desiredOutput > contextWindow {
            messages.removeFirst(exchange)
            trimmed = true
        }
        let dropped = droppedTurns()

        let maxOutput = max(1, min(desiredOutput, contextWindow - inputTokens()))
        var body: [String: Any] = [
            "model": modelID,
            "input": messages.map(\.item),
            "stream": true,
            "store": false,
            "max_output_tokens": maxOutput,
        ]
        // Only models with a native reasoning channel accept `reasoning`; the
        // server treats a missing value as thinking off.
        if capability != nil {
            body["reasoning"] = ["effort": effort(for: reasoning)]
        }
        return ChatRequest(body: body, droppedTurnCount: dropped)
    }

    static func effort(for choice: ReasoningChoice) -> String {
        switch choice {
        case .off: "none"
        case .on: "medium"
        case .low: "low"
        case .medium: "medium"
        case .high: "high"
        }
    }

    static func estimatedTokens(text: String, imageCount: Int) -> Int {
        text.utf8.count / bytesPerToken + imageCount * tokensPerImage
    }

    static func mediaType(forAttachment name: String) -> String {
        switch (name as NSString).pathExtension.lowercased() {
        case "png": "image/png"
        case "gif": "image/gif"
        case "webp": "image/webp"
        case "heic": "image/heic"
        default: "image/jpeg"
        }
    }

    private struct Message {
        var role: ChatRole
        var item: [String: Any]
        var tokens: Int
        /// Position of the source turn in the conversation.
        var turnIndex: Int
    }

    /// Keeps user and assistant strictly alternating, which some chat
    /// templates require: a user turn is sent only when a non-empty reply
    /// follows it, or when it is the message being answered. Unanswered
    /// questions (failed, or stopped before any text) and empty replies are
    /// left out.
    private static func isAnswered(_ index: Int, in turns: [ChatTurn]) -> Bool {
        let turn = turns[index]
        switch turn.role {
        case .user:
            guard index + 1 < turns.count else { return true }
            let next = turns[index + 1]
            return next.role == .assistant && !next.text.isEmpty
        case .assistant:
            return !turn.text.isEmpty && index > 0 && turns[index - 1].role == .user
        }
    }

    private static func message(for turn: ChatTurn, at index: Int, loadAttachment: (String) -> Data?) -> Message? {
        switch turn.role {
        case .user:
            var parts: [[String: Any]] = []
            var images = 0
            for name in turn.attachments {
                guard let data = loadAttachment(name) else { continue }
                parts.append([
                    "type": "input_image",
                    "image_url": "data:\(mediaType(forAttachment: name));base64,\(data.base64EncodedString())",
                ])
                images += 1
            }
            parts.append(["type": "input_text", "text": turn.text])
            return Message(
                role: .user,
                item: ["type": "message", "role": "user", "content": parts],
                tokens: estimatedTokens(text: turn.text, imageCount: images),
                turnIndex: index
            )
        case .assistant:
            return Message(
                role: .assistant,
                item: ["type": "message", "role": "assistant", "content": [["type": "output_text", "text": turn.text]]],
                tokens: estimatedTokens(text: turn.text, imageCount: 0),
                turnIndex: index
            )
        }
    }

    /// Length of the oldest exchange (a user message and the replies before
    /// the next user message), or nil when only the last user message is left.
    private static func oldestExchangeLength(in messages: [Message]) -> Int? {
        guard let lastUser = messages.lastIndex(where: { $0.role == .user }), lastUser > 0 else { return nil }
        let nextUser = messages.indices.dropFirst().first { messages[$0].role == .user } ?? lastUser
        return nextUser
    }
}
