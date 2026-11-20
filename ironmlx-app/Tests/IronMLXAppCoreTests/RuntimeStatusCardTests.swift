import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func statusCardShowsNumbersOnlyForLiveData() {
    let store = RuntimeMetricsStore()
    #expect(RuntimeStatusCardStatus(store: store) == .notRunning)

    store.setBackendState(.starting)
    #expect(RuntimeStatusCardStatus(store: store) == .starting)

    store.setBackendState(.running)
    store.setConnection(.reconnecting)
    #expect(RuntimeStatusCardStatus(store: store) == .reconnecting)

    let snapshot = RuntimeMetricsSnapshot(activity: .init(state: .generating))
    store.setConnection(.connected)
    store.apply(snapshot, receivedAt: Date())
    #expect(RuntimeStatusCardStatus(store: store) == .live(snapshot))

    store.setConnection(.reconnecting)
    #expect(RuntimeStatusCardStatus(store: store) == .reconnecting)
    store.setBackendState(.stopped)
    #expect(RuntimeStatusCardStatus(store: store) == .notRunning)
}

@Test
func statusCardFormatsMemoryPrefixAndActive() {
    let gb: Int64 = 1_073_741_824
    let memory = RuntimeMetricsSnapshot.Memory(
        processBytes: 38 * gb, ceilingBytes: 96 * gb, kvCacheBytes: 10 * gb, mlxCacheBytes: 5 * gb
    )
    #expect(RuntimeStatusCardFormat.memoryUsage(memory) == "38.0 / 96.0 GB")
    #expect(
        RuntimeStatusCardFormat.memorySegments(memory)
            == .init(modelsAndOther: 23 * gb, kv: 10 * gb, cache: 5 * gb, total: 96 * gb)
    )
    // KV and cache larger than the process footprint never make a negative segment.
    let skewed = RuntimeMetricsSnapshot.Memory(processBytes: 4 * gb, kvCacheBytes: 3 * gb, mlxCacheBytes: 2 * gb)
    #expect(RuntimeStatusCardFormat.memorySegments(skewed).modelsAndOther == 0)
    #expect(RuntimeStatusCardFormat.memorySegments(skewed).total == 5 * gb)
    #expect(RuntimeStatusCardFormat.memoryUsage(skewed) == "4.0 GB")

    #expect(RuntimeStatusCardFormat.prefixHit(.init(hitTokens: 0, eligibleTokens: 0)) == "—")
    #expect(RuntimeStatusCardFormat.prefixHit(.init(hitTokens: 87, eligibleTokens: 100)) == "87%")

    // Only the active count: queue capacity is the admission limit, not concurrency.
    let withCapacity = RuntimeMetricsSnapshot(
        activity: .init(activeRequests: 2),
        models: [
            .init(id: "a", activeRequests: 1, queueCapacity: 16),
            .init(id: "b", activeRequests: 1, queueCapacity: 16),
        ]
    )
    #expect(RuntimeStatusCardFormat.active(withCapacity) == "2")

    let models: [RuntimeMetricsSnapshot.Model] = [
        .init(id: "mlx-community/Other-4bit", isDefault: true),
        .init(id: "incoai/Qwen3.6-35B-A3B-4bit", mode: "dflash2", activeRequests: 1),
    ]
    #expect(RuntimeStatusCardFormat.modelLine(models) == "Qwen3.6-35B-A3B-4bit +1 · DFlash2")
    #expect(RuntimeStatusCardFormat.modelLine([.init(id: "a/B", mode: "normal")]) == "B")
}

@Test @MainActor
func statusCardMenuViewIsCardWideAndTallerWithLiveData() {
    let store = RuntimeMetricsStore()
    let stopped = RuntimeStatusCardView.makeMenuItemView(store: store, language: "en")
    #expect(stopped.frame.width == RuntimeStatusCardView.width)
    #expect(stopped.frame.height > 0)

    store.setBackendState(.running)
    store.setConnection(.connected)
    store.apply(
        RuntimeMetricsSnapshot(
            activity: .init(state: .generating),
            performance: .init(liveDecodeTokensPerSecond: 42.8),
            models: [.init(id: "mlx-community/Qwen3-0.6B-4bit")]
        ),
        receivedAt: Date()
    )
    let live = RuntimeStatusCardView.makeMenuItemView(store: store, language: "zh-Hans")
    #expect(live.frame.width == RuntimeStatusCardView.width)
    #expect(live.frame.height > stopped.frame.height + 100)
}

@Test
func alertReasonMatchesTheIndicatorColor() {
    func pick(_ reasons: [String], _ indicator: GlanceIndicator?) -> String? {
        RuntimeStatusCardFormat.alertReasonCode(reasons, indicator: indicator)
    }
    // Red: emergency, then model failure, then hard memory pressure, whatever the received order.
    #expect(pick(["kv_cache_near_soft_limit", "process_memory_emergency"], .red) == "process_memory_emergency")
    #expect(pick(["scheduler_queue_high", "model_failed"], .red) == "model_failed")
    #expect(
        pick(["process_memory_hard", "model_failed", "process_memory_emergency"], .red)
            == "process_memory_emergency"
    )
    #expect(pick(["scheduler_queue_high", "process_memory_hard", "model_failed"], .red) == "model_failed")
    #expect(pick(["scheduler_queue_high", "process_memory_hard"], .red) == "process_memory_hard")
    // Amber: memory pressure first, then the rest in received order.
    #expect(pick(["scheduler_queue_high", "process_memory_soft"], .amber) == "process_memory_soft")
    #expect(pick(["kv_cache_near_soft_limit", "process_memory_hard"], .amber) == "process_memory_hard")
    #expect(pick(["kv_cache_near_soft_limit", "scheduler_queue_high"], .amber) == "kv_cache_near_soft_limit")
    // Unknown reasons come last, but still show when they are all there is.
    #expect(pick(["future_reason", "scheduler_queue_high"], .amber) == "scheduler_queue_high")
    #expect(pick(["future_reason", "scheduler_queue_high"], .red) == "scheduler_queue_high")
    #expect(pick(["future_reason"], .amber) == "future_reason")
    #expect(pick([], .red) == nil)

    let down = RuntimeMetricsSnapshot(
        health: .init(status: .down, degradedReasons: ["kv_cache_near_soft_limit", "process_memory_emergency"])
    )
    #expect(RuntimeStatusCardFormat.alertReason(down, indicator: .red, language: "en") == "Memory pressure")
    let healthy = RuntimeMetricsSnapshot(health: .init(status: .healthy, degradedReasons: ["scheduler_queue_high"]))
    #expect(RuntimeStatusCardFormat.alertReason(healthy, indicator: nil, language: "en") == nil)
}
