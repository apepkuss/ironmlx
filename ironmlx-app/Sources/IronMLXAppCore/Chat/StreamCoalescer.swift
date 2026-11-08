import Foundation

/// Text gathered between two page updates.
struct CoalescedDelta: Equatable, Sendable {
    var reasoning: String
    var text: String

    var isEmpty: Bool { reasoning.isEmpty && text.isEmpty }
}

/// Gathers streamed deltas and hands them on at most once per `interval`
/// (about one screen refresh), so the page is updated per frame rather than
/// per token. A terminal event flushes what is left at once and returns only
/// after it is delivered. Flushes are delivered in order.
actor StreamCoalescer {
    typealias Flush = @Sendable (CoalescedDelta) async -> Void
    typealias Sleep = @Sendable (Duration) async throws -> Void

    private let interval: Duration
    private let flush: Flush
    private let sleep: Sleep
    private var pending = CoalescedDelta(reasoning: "", text: "")
    private var timer: Task<Void, Never>?
    private var timerGeneration = 0
    private var lastDelivery: Task<Void, Never>?

    init(
        interval: Duration = .milliseconds(16),
        flush: @escaping Flush,
        sleep: @escaping Sleep = { try await Task.sleep(for: $0) }
    ) {
        self.interval = interval
        self.flush = flush
        self.sleep = sleep
    }

    func append(_ event: ResponsesStreamEvent) async {
        switch event {
        case .reasoningDelta(let text):
            pending.reasoning += text
            scheduleTimer()
        case .textDelta(let text):
            pending.text += text
            scheduleTimer()
        case .completed, .failed, .incomplete:
            await finish()
        }
    }

    /// Delivers whatever is pending now and waits until every earlier
    /// delivery is done.
    func finish() async {
        cancelTimer()
        await deliverPending()
        await lastDelivery?.value
    }

    private func scheduleTimer() {
        guard timer == nil else { return }
        timerGeneration += 1
        let generation = timerGeneration
        let sleep = sleep
        let interval = interval
        timer = Task { [weak self] in
            do {
                try await sleep(interval)
            } catch {
                return
            }
            await self?.timerFired(generation)
        }
    }

    private func cancelTimer() {
        timer?.cancel()
        timer = nil
        timerGeneration += 1
    }

    private func timerFired(_ generation: Int) async {
        guard generation == timerGeneration else { return }
        timer = nil
        await deliverPending()
    }

    private func deliverPending() async {
        guard !pending.isEmpty else { return }
        let delta = pending
        pending = CoalescedDelta(reasoning: "", text: "")
        let previous = lastDelivery
        let flush = flush
        let delivery = Task {
            await previous?.value
            await flush(delta)
        }
        lastDelivery = delivery
        await delivery.value
    }
}
