import Foundation
import Testing

@testable import IronMLXAppCore

@Test func healthzMapsToDashboardHealthShape() throws {
    let healthz = """
    {
      "status": "healthy",
      "uptime_secs": 42,
      "model": {
        "name": "Qwen3",
        "max_position_embeddings": 32768
      },
      "scheduler": {
        "b_max": 1,
        "b_active": 0,
        "b_queued": 0,
        "queue_max": 32,
        "admission_queue_full_count": 0,
        "memory_budget_exceeded_count": 0
      },
      "memory": {
        "total_ram_bytes": 68719476736,
        "free_ram_bytes": 34359738368,
        "kv_cache_active_bytes": 1048576,
        "kv_cache_soft_limit_bytes": 2097152,
        "mlx_total_bytes": 51539607552,
        "mlx_max_recommended_bytes": 38654705664,
        "mlx_active_bytes": 1073741824,
        "mlx_cache_bytes": 536870912,
        "mlx_peak_bytes": 2147483648,
        "mlx_memory_limit_bytes": 8589934592
      },
      "device_name": "Apple M3 Max",
      "version": "0.0.1"
    }
    """.data(using: .utf8)!

    let snapshot = try JSONDecoder().decode(HealthzSnapshot.self, from: healthz)
    let legacy = LegacyHealthAdapter(statusNow: Date(timeIntervalSince1970: 1_700_000_000))
        .legacyStatus(from: snapshot)

    #expect(legacy.startedAt == 1_699_999_958)
    #expect(legacy.model == "Qwen3")
    #expect(legacy.memory.activeMB == 1024)
    #expect(legacy.memory.cacheMB == 512)
    #expect(legacy.memory.peakMB == 2048)
    #expect(legacy.memory.totalMB == 49152)
    #expect(legacy.memory.maxMB == 36864)
    #expect(legacy.deviceName == "Apple M3 Max")
}

@Test func dashboardStatusPageUsesAdaptiveGpuPollingAndOneHourHistory() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains("const GPU_ACTIVE_POLL_MS = 500;"))
    #expect(html.contains("const GPU_IDLE_POLL_MS = 1000;"))
    #expect(html.contains("const GPU_HISTORY_LIMIT_MS = 60 * 60 * 1000;"))
    #expect(html.contains("data-gpu-window=\"5\""))
    #expect(html.contains("data-gpu-window=\"15\""))
    #expect(html.contains("data-gpu-window=\"30\""))
    #expect(html.contains("data-gpu-window=\"60\""))
    #expect(html.contains("window.__HEALTH_PENDING__"))
}
