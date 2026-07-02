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
        "kv_cache_logical_cap_tokens": 262144,
        "kv_cache_resident_cap_tokens": 1024,
        "kv_cache_budget_policy": "active_kv_offload",
        "mlx_total_bytes": 51539607552,
        "mlx_max_recommended_bytes": 38654705664,
        "mlx_active_bytes": 1073741824,
        "mlx_cache_bytes": 536870912,
        "mlx_peak_bytes": 2147483648,
        "mlx_memory_limit_bytes": 8589934592
      },
      "active_kv_offload": {
        "enabled": true,
        "mode": "request_residency_v1",
        "storage_dir": "/tmp/ironmlx-active-kv",
        "resident_pages": 2,
        "offloaded_pages": 3,
        "loading_pages": 0,
        "dirty_pages": 1,
        "parked_requests": 1,
        "offloaded_bytes": 1048576,
        "swap_out_count": 4,
        "swap_in_count": 2,
        "swap_error_count": 0,
        "last_swap_out_us": 123,
        "last_swap_in_us": 456,
        "supported_cache_kinds": ["full_attention_dense"],
        "not_applicable_cache_kinds": ["gated_delta_linear"]
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
    #expect(snapshot.memory.kvCacheLogicalCapTokens == 262144)
    #expect(snapshot.memory.kvCacheResidentCapTokens == 1024)
    #expect(snapshot.memory.kvCacheBudgetPolicy == "active_kv_offload")
    #expect(legacy.activeKvOffload.enabled == true)
    #expect(legacy.activeKvOffload.parkedRequests == 1)
    #expect(legacy.activeKvOffload.swapOutCount == 4)
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

@Test func dashboardStatusPageRendersActiveKVOffloadState() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains("id=\"stat-active-kv\""))
    #expect(html.contains("data-i18n=\"active_kv_offload\""))
    #expect(html.contains("cfg-active-kv-offload"))
    #expect(html.contains("updateActiveKvStatus(data.active_kv_offload)"))
}

@Test func dashboardEndpointCardShowsBindAddressAsSecondaryHint() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains("endpoints_listen"))
    #expect(html.contains("endpoints_reachable"))
    #expect(html.contains("监听"))
    #expect(html.contains("可访问端点"))
    #expect(html.contains("const listenAddress = data.host + ':' + data.port"))
    #expect(html.contains("const listenText = (dict.endpoints_listen || 'Listening') + ': ' + listenAddress"))
    #expect(html.contains("hint.innerHTML ="))
    #expect(html.contains("rows.push(buildEndpointSectionHeader"))
    #expect(!html.contains("buildEndpointListenRow"))
}

@Test func dashboardActiveKVHelpWarningUsesGrayDefaultAndOrangeHover() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains(#"class="profile-help-trigger warning" id="active-kv-help-trigger""#))
    #expect(html.contains("""
  .profile-help-trigger.warning {
    border-color: var(--border);
    background: var(--bg);
    color: var(--text-secondary);
  }
"""))
    #expect(html.contains("""
  .profile-help-trigger.warning:hover,
  .profile-help-trigger.warning:focus {
    border-color: #ff9500;
    background: rgba(255, 149, 0, 0.08);
    color: #ff9500;
  }
"""))
}

@Test func dashboardActiveKVHelpTooltipExplainsAutomaticTriggerConditions() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains(#"data-i18n="active_kv_offload_help_trigger""#))
    #expect(html.contains("开启后不代表立即发生卸载。系统会在请求调度、KV 驻留压力或请求暂停/恢复需要时自动触发；空闲或内存压力较低时可能保持 idle。"))
}

@Test func dashboardActiveKVHelpTooltipSeparatesApplicableAndNotApplicableCacheKinds() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains(#"data-i18n="active_kv_offload_help_applicable""#))
    #expect(html.contains(#"data-i18n="active_kv_offload_help_not_applicable""#))
    #expect(html.contains("适用：Full Attention dense/paged、TurboQuant packed Full Attention、MLA。"))
    #expect(html.contains("不适用：GatedDelta/Linear、MTP speculative side cache。"))
    #expect(!html.contains("支持：Full Attention dense/paged、TurboQuant packed Full Attention、MLA。不适用"))
}

@Test func dashboardActiveKVHelpTooltipUsesFixedPanelForLongCopy() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let htmlURL = packageRoot
        .appendingPathComponent("Sources/IronMLXAppCore/Resources/dashboard2.html")
    let html = try String(contentsOf: htmlURL, encoding: .utf8)

    #expect(html.contains(#"class="profile-help profile-help-fixed""#))
    #expect(html.contains(#"class="profile-help-tooltip profile-help-tooltip-fixed" id="active-kv-help-tooltip""#))
    #expect(html.contains("""
  .profile-help-tooltip-fixed {
    position: fixed;
    top: var(--profile-tooltip-top, 16px);
    left: var(--profile-tooltip-left, 50vw);
    width: min(340px, calc(100vw - 32px));
    max-height: calc(100vh - 32px);
    overflow-y: auto;
    transform: translate(-50%, -4px);
  }
"""))
    #expect(html.contains("function setupFixedTooltips()"))
    #expect(html.contains("setupFixedTooltips();"))
}
