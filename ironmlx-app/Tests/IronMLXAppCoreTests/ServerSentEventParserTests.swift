import Foundation
import Testing

@testable import IronMLXAppCore

private func feed(_ lines: [String]) -> [ServerSentEvent] {
    var parser = ServerSentEventParser()
    return lines.compactMap { parser.consume(line: $0) }
}

@Test
func dispatchesEventOnBlankLine() {
    var parser = ServerSentEventParser()

    #expect(parser.consume(line: "event: metrics") == nil)
    #expect(parser.consume(line: #"data: {"seq":1}"#) == nil)
    #expect(parser.consume(line: "") == ServerSentEvent(event: "metrics", data: #"{"seq":1}"#))
    // The parser resets after dispatch.
    #expect(parser.consume(line: "") == nil)
}

@Test
func joinsMultilineData() {
    #expect(feed(["data: a", "data: b", ""]) == [ServerSentEvent(event: "message", data: "a\nb")])
}

@Test
func ignoresKeepAliveComments() {
    #expect(feed([": keepalive", "", ":", ""]).isEmpty)
    #expect(
        feed(["event: metrics", ": comment between fields", "data: x", ""])
            == [ServerSentEvent(event: "metrics", data: "x")]
    )
}

@Test
func defaultsEventNameToMessage() {
    #expect(feed(["data: hello", ""]) == [ServerSentEvent(event: "message", data: "hello")])
    // The event name does not leak into the next event.
    #expect(
        feed(["event: metrics", "data: 1", "", "data: 2", ""])
            == [
                ServerSentEvent(event: "metrics", data: "1"),
                ServerSentEvent(event: "message", data: "2"),
            ]
    )
}

@Test
func stripsSingleLeadingSpaceOnly() {
    #expect(feed(["data:  x", ""]) == [ServerSentEvent(event: "message", data: " x")])
    #expect(feed(["data:x", ""]) == [ServerSentEvent(event: "message", data: "x")])
}

@Test
func doesNotDispatchEventWithoutData() {
    #expect(feed(["event: metrics", ""]).isEmpty)
    #expect(feed(["data:", ""]) == [ServerSentEvent(event: "message", data: "")])
}

@Test
func lineBufferKeepsBlankLinesAndHandlesCRLF() {
    var buffer = ServerSentEventLineBuffer()
    let bytes = Array("event: metrics\r\ndata: a\r\n\r\n: ping\n\ndata: b\rdata: c\n\n".utf8)
    let lines = bytes.compactMap { buffer.append($0) }

    #expect(lines == ["event: metrics", "data: a", "", ": ping", "", "data: b", "data: c", ""])
}
