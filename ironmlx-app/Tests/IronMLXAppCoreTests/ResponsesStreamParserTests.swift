import Foundation
import Testing

@testable import IronMLXAppCore

@Suite("Responses stream parser")
struct ResponsesStreamParserTests {
    /// Frames as `responses.rs` writes them: `event: <kind>\ndata: <json>\n\n`.
    private func frame(_ kind: String, _ payload: String) -> String {
        let body = payload.dropFirst() // drop "{"
        return "event: \(kind)\ndata: {\"type\":\"\(kind)\",\"sequence_number\":0,\(body)\n\n"
    }

    private var wire: String {
        frame("response.created", #"{"response":{"id":"resp_1","status":"in_progress"}}"#)
            + frame("response.output_item.added", #"{"output_index":0,"item":{"type":"reasoning"}}"#)
            + frame("response.reasoning_text.delta", #"{"content_index":0,"delta":"Let me think","item_id":"rs_1","output_index":0}"#)
            + frame("response.reasoning_text.done", #"{"content_index":0,"text":"Let me think","item_id":"rs_1","output_index":0}"#)
            + frame("response.output_text.delta", #"{"content_index":0,"delta":"你好","item_id":"msg_1","logprobs":[],"output_index":1}"#)
            + frame("response.output_text.delta", #"{"content_index":0,"delta":", world","item_id":"msg_1","logprobs":[],"output_index":1}"#)
            + frame("response.completed", #"{"response":{"id":"resp_1","status":"completed","usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0},"output_tokens":42,"output_tokens_details":{"reasoning_tokens":7},"total_tokens":54}}}"#)
    }

    @Test func parsesEventsAcrossChunkBoundaries() {
        let bytes = Data(wire.utf8)
        // Cut inside the multi-byte "你" and at several other points.
        let reference = Data(frame("response.created", #"{"response":{}}"#).utf8).count
        let chinese = bytes.firstRange(of: Data("你".utf8))!.lowerBound + 1
        let cuts = [5, reference + 3, chinese, chinese + 40, bytes.count - 10]
        var parser = ResponsesStreamParser()
        var events: [ResponsesStreamEvent] = []
        var start = 0
        for cut in cuts + [bytes.count] {
            events += parser.feed(bytes.subdata(in: start..<cut))
            start = cut
        }
        #expect(events == [
            .reasoningDelta("Let me think"),
            .textDelta("你好"),
            .textDelta(", world"),
            .completed(outputTokens: 42, reasoningTokens: 7),
        ])
    }

    @Test func byteByByteFeedingGivesTheSameEvents() {
        var parser = ResponsesStreamParser()
        var events: [ResponsesStreamEvent] = []
        for byte in Data(wire.utf8) {
            events += parser.feed(Data([byte]))
        }
        #expect(events.count == 4)
        #expect(events.last == .completed(outputTokens: 42, reasoningTokens: 7))
    }

    @Test func parsesFailedAndIncomplete() {
        var parser = ResponsesStreamParser()
        let failed = frame("response.failed", #"{"response":{"id":"r","object":"response","created_at":1,"status":"failed","error":{"code":"generation_error","message":"native allocation failure"}}}"#)
        let incomplete = frame("response.incomplete", #"{"response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"output_tokens":5}}}"#)
        #expect(parser.feed(Data(failed.utf8)) == [.failed(message: "native allocation failure")])
        #expect(parser.feed(Data(incomplete.utf8)) == [.incomplete(reason: "max_output_tokens", outputTokens: 5)])
    }

    @Test func handlesCRLFAndIgnoresUnknownEventsAndComments() {
        var parser = ResponsesStreamParser()
        let text = ": keep-alive\r\n\r\n"
            + "event: response.content_part.added\r\ndata: {\"type\":\"response.content_part.added\"}\r\n\r\n"
            + "event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\r\n\r\n"
            + "data: [DONE]\r\n\r\n"
        #expect(parser.feed(Data(text.utf8)) == [.textDelta("ok")])
    }

    @Test func completedWithoutUsageHasNilCounts() {
        var parser = ResponsesStreamParser()
        let text = frame("response.completed", #"{"response":{"id":"r","status":"completed"}}"#)
        #expect(parser.feed(Data(text.utf8)) == [.completed(outputTokens: nil, reasoningTokens: nil)])
    }
}
