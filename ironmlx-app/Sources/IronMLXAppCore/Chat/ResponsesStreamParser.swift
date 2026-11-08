import Foundation

/// The `/v1/responses` stream events chat uses; all others are ignored.
enum ResponsesStreamEvent: Equatable, Sendable {
    case reasoningDelta(String)
    case textDelta(String)
    case completed(outputTokens: Int?, reasoningTokens: Int?)
    case failed(message: String)
    case incomplete(reason: String?, outputTokens: Int?)

    var isTerminal: Bool {
        switch self {
        case .reasoningDelta, .textDelta: false
        case .completed, .failed, .incomplete: true
        }
    }
}

/// Incremental SSE parser for the frames `responses.rs` writes
/// (`event: <kind>\ndata: <json>\n\n`). Bytes are buffered until a full line,
/// so chunks may split lines and multi-byte characters anywhere.
struct ResponsesStreamParser {
    private var buffer = Data()
    private var eventName: String?
    private var dataLines: [Data] = []

    mutating func feed(_ chunk: Data) -> [ResponsesStreamEvent] {
        buffer.append(chunk)
        var events: [ResponsesStreamEvent] = []
        var lineStart = buffer.startIndex
        while let newline = buffer[lineStart...].firstIndex(of: 0x0A) {
            var line = buffer[lineStart..<newline]
            if line.last == 0x0D { line = line.dropLast() }
            lineStart = buffer.index(after: newline)
            if let event = consume(line: line) {
                events.append(event)
            }
        }
        buffer = Data(buffer[lineStart...])
        return events
    }

    private mutating func consume(line: Data) -> ResponsesStreamEvent? {
        if line.isEmpty {
            defer {
                eventName = nil
                dataLines.removeAll(keepingCapacity: true)
            }
            return dispatch()
        }
        guard line.first != UInt8(ascii: ":") else { return nil }
        let colon = line.firstIndex(of: UInt8(ascii: ":")) ?? line.endIndex
        let field = String(decoding: line[..<colon], as: UTF8.self)
        var value = colon < line.endIndex ? line[line.index(after: colon)...] : Data()[...]
        if value.first == UInt8(ascii: " ") { value = value.dropFirst() }
        switch field {
        case "event": eventName = String(decoding: value, as: UTF8.self)
        case "data": dataLines.append(Data(value))
        default: break
        }
        return nil
    }

    private func dispatch() -> ResponsesStreamEvent? {
        guard !dataLines.isEmpty else { return nil }
        let data = dataLines.count == 1 ? dataLines[0] : Data(dataLines.joined(separator: Data([0x0A])))
        guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        let kind = (object["type"] as? String) ?? eventName ?? ""
        let response = object["response"] as? [String: Any]
        switch kind {
        case "response.output_text.delta":
            return (object["delta"] as? String).map(ResponsesStreamEvent.textDelta)
        case "response.reasoning_text.delta":
            return (object["delta"] as? String).map(ResponsesStreamEvent.reasoningDelta)
        case "response.completed":
            let usage = response?["usage"] as? [String: Any]
            let details = usage?["output_tokens_details"] as? [String: Any]
            return .completed(
                outputTokens: (usage?["output_tokens"] as? NSNumber)?.intValue,
                reasoningTokens: (details?["reasoning_tokens"] as? NSNumber)?.intValue
            )
        case "response.failed":
            let error = response?["error"] as? [String: Any]
            return .failed(message: error?["message"] as? String ?? "")
        case "response.incomplete":
            let details = response?["incomplete_details"] as? [String: Any]
            let usage = response?["usage"] as? [String: Any]
            return .incomplete(
                reason: details?["reason"] as? String,
                outputTokens: (usage?["output_tokens"] as? NSNumber)?.intValue
            )
        default:
            return nil
        }
    }
}
