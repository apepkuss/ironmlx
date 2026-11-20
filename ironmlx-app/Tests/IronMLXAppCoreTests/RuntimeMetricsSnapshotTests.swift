import Foundation
import Testing

@testable import IronMLXAppCore

@Test
func decodesContractFixture() throws {
    let url = try #require(
        Bundle.module.url(
            forResource: "runtime-metrics-v1",
            withExtension: "json",
            subdirectory: "Fixtures"
        )
            ?? Bundle.module.url(forResource: "runtime-metrics-v1", withExtension: "json")
    )
    let snapshot = try RuntimeMetricsSnapshot.decode(from: Data(contentsOf: url))

    #expect(snapshot.schemaVersion == 1)
    #expect(snapshot.seq == 1842)
    #expect(snapshot.backend.startedUnixMs == 1_763_268_480_000)
    #expect(snapshot.backend.deviceName == "Apple M5 Max")
    #expect(snapshot.health.status == .degraded)
    #expect(snapshot.health.degradedReasons == ["scheduler_queue_high"])
    #expect(snapshot.activity.state == .generating)
    #expect(snapshot.activity.activeRequests == 2)
    #expect(snapshot.activity.queuedRequests == 1)
    #expect(snapshot.performance.liveDecodeTokensPerSecond == 42.8)
    #expect(snapshot.performance.ttftMs == 312.0)
    #expect(snapshot.memory.pressureLevel == .normal)
    #expect(snapshot.memory.processBytes == 41_000_000_000)
    #expect(snapshot.memory.kvCacheBytes == 10_300_000_000)
    #expect(snapshot.prefixCache.hitTokens == 120_345)
    #expect(snapshot.models.count == 2)
    #expect(snapshot.models[0].mode == "dflash2")
    #expect(snapshot.models[0].isDefault)
    #expect(snapshot.models[0].queueCapacity == 2)
    #expect(snapshot.models[1].kind == "asr")
    #expect(snapshot.models[1].state == .failed)
    #expect(snapshot.models[1].lastError == "model weights not found")
    #expect(snapshot.models[1].mode == nil)
    #expect(snapshot.models[1].queueCapacity == nil)
}

@Test
func decodesUnknownEnumValuesAndMissingOptionalFields() throws {
    let json = """
        {"schema_version":1,"activity":{"state":"thinking"},\
        "health":{"status":"strained","degraded_reasons":["future_reason"]}}
        """
    let snapshot = try RuntimeMetricsSnapshot.decode(from: Data(json.utf8))

    #expect(snapshot.schemaVersion == 1)
    #expect(snapshot.activity.state == .unknown)
    #expect(snapshot.activity.activeRequests == 0)
    #expect(snapshot.health.status == .unknown)
    #expect(snapshot.health.degradedReasons == ["future_reason"])
    #expect(snapshot.memory.pressureLevel == .unknown)
    #expect(snapshot.performance.liveDecodeTokensPerSecond == nil)
    #expect(snapshot.models.isEmpty)
}

@Test
func decodesRealBackendSnapshotWithNullPerformanceFields() throws {
    let json = """
        {"schema_version":1,"seq":2,"unix_ms":1794829227991,\
        "backend":{"version":"0.5.0","started_unix_ms":1794829132584,"device_name":"Apple M5 Max"},\
        "health":{"status":"healthy","degraded_reasons":[]},\
        "activity":{"state":"prefilling","active_requests":1,"prefilling_requests":1,"queued_requests":0},\
        "performance":{"live_decode_tokens_per_second":null,"prefill_tokens_per_second":2207.19,\
        "decode_tokens_per_second":124.38,"ttft_ms":5673.5,"window_seconds":60},\
        "memory":{"pressure_level":"normal","process_bytes":21606670784,"ceiling_bytes":110437853184,\
        "kv_cache_bytes":0,"mlx_cache_bytes":139058690,"telemetry_degraded":false},\
        "prefix_cache":{"hit_tokens":0,"eligible_tokens":0},\
        "models":[{"id":"mlx-community/Qwen3.6-35B-A3B-4bit","kind":"causal","state":"loaded",\
        "mode":"normal","is_default":true,"active_requests":0,"queued_requests":0,"queue_capacity":32}]}
        """
    let snapshot = try RuntimeMetricsSnapshot.decode(from: Data(json.utf8))

    #expect(snapshot.activity.state == .prefilling)
    #expect(snapshot.activity.prefillingRequests == 1)
    #expect(snapshot.performance.liveDecodeTokensPerSecond == nil)
    #expect(snapshot.performance.prefillTokensPerSecond == 2207.19)
    #expect(snapshot.performance.windowSeconds == 60)
    #expect(snapshot.memory.ceilingBytes == 110_437_853_184)
    #expect(snapshot.models.first?.state == .loaded)
    #expect(snapshot.models.first?.liveDecodeTokensPerSecond == nil)
}

@Test
func rejectsSnapshotWithoutSchemaVersion() {
    #expect(throws: (any Error).self) {
        try RuntimeMetricsSnapshot.decode(from: Data(#"{"seq":1}"#.utf8))
    }
}
