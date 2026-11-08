import Foundation
import Testing

@testable import IronMLXAppCore

/// A sleep that returns only when the test opens it, so flush timing does not
/// depend on the machine's speed.
private actor SleepGate {
    private(set) var requested: [Duration] = []
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var isOpen = false

    func sleep(_ duration: Duration) async {
        requested.append(duration)
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        waiters.forEach { $0.resume() }
        waiters.removeAll()
    }
}

private actor FlushLog {
    private(set) var deltas: [CoalescedDelta] = []
    func append(_ delta: CoalescedDelta) { deltas.append(delta) }
}

private func waitUntil(_ condition: @Sendable () async -> Bool) async throws {
    let deadline = ContinuousClock.now + .seconds(5)
    while !(await condition()) {
        guard ContinuousClock.now < deadline else { throw CancellationError() }
        try await Task.sleep(for: .milliseconds(2))
    }
}

@Suite("Stream coalescer")
struct StreamCoalescerTests {
    @Test func mergesDeltasWithinInterval() async throws {
        let gate = SleepGate()
        let log = FlushLog()
        let coalescer = StreamCoalescer(
            flush: { await log.append($0) },
            sleep: { await gate.sleep($0) }
        )
        var expected = ""
        for index in 0..<50 {
            let piece = "t\(index) "
            expected += piece
            await coalescer.append(index == 0 ? .reasoningDelta("r") : .textDelta(piece))
        }
        // The first delta starts one 16 ms timer; the rest wait for it.
        try await waitUntil { await !gate.requested.isEmpty }
        #expect(await gate.requested == [.milliseconds(16)])
        #expect(await log.deltas.isEmpty)

        await gate.open()
        try await waitUntil { await !log.deltas.isEmpty }
        let deltas = await log.deltas
        #expect(deltas.count == 1)
        #expect(deltas.first == CoalescedDelta(reasoning: "r", text: String(expected.dropFirst(3))))
    }

    @Test func flushesRemainderOnCompletion() async throws {
        let gate = SleepGate()
        let log = FlushLog()
        let coalescer = StreamCoalescer(
            interval: .milliseconds(16),
            flush: { await log.append($0) },
            sleep: { await gate.sleep($0) }
        )
        await coalescer.append(.reasoningDelta("think"))
        await coalescer.append(.textDelta("a"))
        await coalescer.append(.textDelta("b"))
        // A terminal event flushes at once, without waiting for the timer.
        await coalescer.append(.completed(outputTokens: 3, reasoningTokens: 1))
        #expect(await log.deltas == [CoalescedDelta(reasoning: "think", text: "ab")])

        // The pending timer later finds nothing left to send.
        await gate.open()
        try await Task.sleep(for: .milliseconds(20))
        #expect(await log.deltas.count == 1)
    }

    @Test func laterDeltasStartANewInterval() async throws {
        let gate = SleepGate()
        let log = FlushLog()
        let coalescer = StreamCoalescer(flush: { await log.append($0) }, sleep: { await gate.sleep($0) })
        await gate.open()
        await coalescer.append(.textDelta("one"))
        try await waitUntil { await log.deltas.count == 1 }
        await coalescer.append(.textDelta("two"))
        try await waitUntil { await log.deltas.count == 2 }
        #expect(await log.deltas.map(\.text) == ["one", "two"])
        #expect(await gate.requested.count == 2)
    }

    @Test func finishFlushesWithoutATerminalEvent() async {
        let gate = SleepGate()
        let log = FlushLog()
        let coalescer = StreamCoalescer(flush: { await log.append($0) }, sleep: { await gate.sleep($0) })
        await coalescer.append(.textDelta("partial"))
        await coalescer.finish()
        #expect(await log.deltas == [CoalescedDelta(reasoning: "", text: "partial")])
        await coalescer.finish()
        #expect(await log.deltas.count == 1)
    }
}
