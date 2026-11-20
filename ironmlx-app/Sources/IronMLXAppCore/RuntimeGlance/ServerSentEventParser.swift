import Foundation

struct ServerSentEvent: Equatable, Sendable {
    var event: String
    var data: String
}

/// Line-oriented `text/event-stream` parser.
///
/// A blank line dispatches the pending event; lines starting with `:` are comments (keepalives);
/// several `data:` lines are joined with `\n`; one leading space is stripped from a field value;
/// the event name defaults to `message`; an event without any `data:` line is not dispatched.
/// `id:` and `retry:` are ignored because the client owns its reconnect policy.
struct ServerSentEventParser {
    private var eventName: String?
    private var dataLines: [String] = []

    mutating func consume(line: String) -> ServerSentEvent? {
        if line.isEmpty {
            return dispatch()
        }
        if line.hasPrefix(":") {
            return nil
        }
        let field: Substring
        var value: Substring
        if let colon = line.firstIndex(of: ":") {
            field = line[..<colon]
            value = line[line.index(after: colon)...]
            if value.hasPrefix(" ") {
                value = value.dropFirst()
            }
        } else {
            field = Substring(line)
            value = ""
        }
        switch field {
        case "event":
            eventName = String(value)
        case "data":
            dataLines.append(String(value))
        default:
            break
        }
        return nil
    }

    private mutating func dispatch() -> ServerSentEvent? {
        defer {
            eventName = nil
            dataLines.removeAll(keepingCapacity: true)
        }
        guard !dataLines.isEmpty else {
            return nil
        }
        let name = eventName.flatMap { $0.isEmpty ? nil : $0 } ?? "message"
        return ServerSentEvent(event: name, data: dataLines.joined(separator: "\n"))
    }
}

/// Splits a byte stream into SSE lines, accepting `\n`, `\r\n` and `\r` terminators.
///
/// `AsyncBytes.lines` cannot be used for SSE: it drops empty lines, and empty lines are what
/// dispatch events.
struct ServerSentEventLineBuffer {
    private var bytes: [UInt8] = []
    private var lastWasCarriageReturn = false

    /// Appends one byte and returns a completed line, if this byte ended one.
    mutating func append(_ byte: UInt8) -> String? {
        switch byte {
        case UInt8(ascii: "\n"):
            if lastWasCarriageReturn {
                lastWasCarriageReturn = false
                return nil
            }
            return takeLine()
        case UInt8(ascii: "\r"):
            lastWasCarriageReturn = true
            return takeLine()
        default:
            lastWasCarriageReturn = false
            bytes.append(byte)
            return nil
        }
    }

    /// Bytes of the line still being assembled.
    var pendingByteCount: Int { bytes.count }

    private mutating func takeLine() -> String {
        defer { bytes.removeAll(keepingCapacity: true) }
        return String(decoding: bytes, as: UTF8.self)
    }
}
