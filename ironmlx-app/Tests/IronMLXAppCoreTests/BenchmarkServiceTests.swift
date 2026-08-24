import Foundation
import Testing

@testable import IronMLXAppCore

private func dashboardHTML(_ html: String, contains needle: String) -> Bool {
    html.contains(needle)
}

@Test func benchmarkPlanBuildsSequentialIronBenchCommand() {
    let request = BenchmarkRequest(
        model: "mlx-community/Tiny-4bit",
        modelPath: "/tmp/Tiny-4bit/snapshot",
        promptTokens: 1024,
        maxTokens: 128,
        batchSize: 1
    )
    let plan = BenchmarkPlan(
        ironBenchURL: URL(fileURLWithPath: "/tmp/iron-bench"),
        request: request,
        host: "127.0.0.1",
        port: 9068
    )

    #expect(plan.arguments == [
        "--target", "ironmlx=http://127.0.0.1:9068",
        "--model-dir", "/tmp/Tiny-4bit/snapshot",
        "--model", "mlx-community/Tiny-4bit",
        "--prompt-len", "1024",
        "--max-tokens", "128",
        "--format", "json",
        "--timeout", "300",
        "--runs", "1",
        "--warmup", "0",
    ])
}

@Test func benchmarkPlanBuildsConcurrentIronBenchCommand() {
    let request = BenchmarkRequest(
        model: "mlx-community/Tiny-4bit",
        modelPath: "/tmp/Tiny-4bit/snapshot",
        promptTokens: 4096,
        maxTokens: 128,
        batchSize: 4
    )
    let plan = BenchmarkPlan(
        ironBenchURL: URL(fileURLWithPath: "/tmp/iron-bench"),
        request: request,
        host: "127.0.0.1",
        port: 9068
    )

    #expect(plan.arguments.suffix(6) == [
        "--concurrent", "4",
        "--duration", "10",
        "--warmup-duration", "0",
    ])
}

@Test func benchmarkPlanUsesLoopbackTargetForWildcardBindHost() {
    let request = BenchmarkRequest(
        model: "mlx-community/Tiny-4bit",
        modelPath: "/tmp/Tiny-4bit/snapshot",
        promptTokens: 1024,
        maxTokens: 128,
        batchSize: 1
    )
    let plan = BenchmarkPlan(
        ironBenchURL: URL(fileURLWithPath: "/tmp/iron-bench"),
        request: request,
        host: "0.0.0.0",
        port: 9068
    )

    #expect(plan.arguments.prefix(2) == [
        "--target", "ironmlx=http://127.0.0.1:9068",
    ])
}

@Test func benchmarkResultParsesSequentialIronBenchJson() throws {
    let json = """
    {
      "stats": [{
        "ttft_ms_median": 125.5,
        "tg_tps_median": 42.25,
        "tpot_ms_median": 23.7,
        "pp_tps_median": 8159.4,
        "e2e_s_median": 3.25
      }]
    }
    """
    let result = try BenchmarkResult.parse(
        ironBenchJSON: Data(json.utf8),
        request: BenchmarkRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            promptTokens: 1024,
            maxTokens: 128,
            batchSize: 1
        ),
        memoryPeakMB: 2048.5
    )

    #expect(result.batchSize == 1)
    #expect(result.ttftMs == 125.5)
    #expect(result.tpotMs == 23.7)
    #expect(result.tgTps == 42.25)
    #expect(result.ppTps == 8159.4)
    #expect(result.totalMs == 3250)
    #expect(result.memoryPeakMB == 2048.5)
}

@Test func benchmarkResultParsesConcurrentIronBenchJson() throws {
    let json = """
    {
      "mode": "concurrent",
      "cells": [{
        "wall_duration_s": 10.0,
        "n_requests": 6,
        "ttft_ms": { "p50": 211.0, "p95": 390.0 },
        "itl_ms": { "p50": 31.5, "p95": 52.0 },
        "aggregate": { "tokens_per_sec": 155.25, "req_per_sec": 0.6 }
      }]
    }
    """
    let result = try BenchmarkResult.parse(
        ironBenchJSON: Data(json.utf8),
        request: BenchmarkRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            promptTokens: 4096,
            maxTokens: 128,
            batchSize: 4
        ),
        memoryPeakMB: nil
    )

    #expect(result.batchSize == 4)
    #expect(result.ttftMs == 211.0)
    #expect(result.tpotMs == 31.5)
    #expect(result.tgTps == 155.25)
    #expect(result.totalThroughput == 155.25)
    #expect(result.totalMs == 10_000)
    #expect(result.requestCount == 6)
}

@Test func dashboardBenchmarkTableIncludesMemoryPeakColumn() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains("Peak MB"))
    #expect(html.contains("memory_peak_mb"))
}

@Test func dashboardBenchmarkUsesExclusiveSessionFlow() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains("/admin/api/benchmark/preflight"))
    #expect(html.contains("/admin/api/benchmark/prepare"))
    #expect(html.contains("/admin/api/benchmark/restore"))
    #expect(html.contains("bench_confirm_point_1"))
    #expect(html.contains("bench_confirm_point_6"))
    #expect(html.contains("benchmarkUiLocked"))
    #expect(html.contains("onBenchmarkRestoreResult"))
}

@Test func dashboardBenchmarkControlsAreLocalizedInSimplifiedChinese() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    let expectedStrings = [
        "bench_start: \"Start\"",
        "bench_results: \"Results\"",
        "clear: \"Clear\"",
        "bench_start: \"开始\"",
        "bench_results: \"结果\"",
        "clear: \"清空\"",
        "bench_start: \"開始\"",
        "bench_results: \"結果\"",
        "bench_start: \"시작\"",
        "bench_results: \"결과\"",
        "clear: \"비우기\"",
    ]
    for expectedString in expectedStrings {
        #expect(dashboardHTML(html, contains: expectedString), "\(expectedString) should be localized")
    }
    #expect(!dashboardHTML(html, contains: "bench_start: \"开始基准测试\""))
}

@Test func dashboardBenchmarkBatchRequiresSelectedPromptSize() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    let hasNoPp1024Fallback = !dashboardHTML(html, contains: "singlePpSizes.length > 0 ? singlePpSizes : [1024]")
    let hasNoPromptSizeMessage = dashboardHTML(html, contains: "bench_no_prompt_size")
    let showsNoPromptSizeToast = dashboardHTML(html, contains: "showToast(t('bench_no_prompt_size'")

    #expect(hasNoPp1024Fallback)
    #expect(hasNoPromptSizeMessage)
    #expect(showsNoPromptSizeToast)
}

@Test func dashboardKVQuantOptionsUseBackendValues() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    let expectedOptions = [
        "<option value=\"none\" data-i18n=\"kv_quant_off\">Off</option>",
        "<option value=\"turbo3\">TurboQuant K3V3</option>",
        "<option value=\"turbo4\">TurboQuant K4V4</option>",
        "<option value=\"k3v4\">TurboQuant K3V4</option>",
    ]
    for expectedOption in expectedOptions {
        #expect(dashboardHTML(html, contains: expectedOption), "\(expectedOption) should be present")
    }
    #expect(!dashboardHTML(html, contains: "value=\"adaptive\""))
}

@Test func dashboardKVQuantInputUsesPersistedSetting() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "window.__IRONMLX_KV_QUANT__"))
    #expect(dashboardHTML(html, contains: "cfgKvQuant.value = normalizeKVQuantValue(window.__IRONMLX_KV_QUANT__)"))
}

@Test func dashboardKVQuantDescriptionExplainsTurboQuantTradeoffsWithoutRestartText() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "使用 TurboQuant 降低长上下文运行时 KV Cache 的内存占用。K3V3 更省内存，K4V4 更保守，K3V4 为推荐均衡选项。"))
    #expect(!dashboardHTML(html, contains: "kv_quant_desc: \"压缩运行时 KV 缓存以降低内存占用。可与 Prefix/SSD 缓存同时使用。保存设置并重启服务后生效。\""))
}

@Test func dashboardRuntimeSettingsInputsUsePersistedConfig() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "window.__IRONMLX_APP_CONFIG__"))
    #expect(dashboardHTML(html, contains: "applyPersistedSettingsToInputs"))
    #expect(dashboardHTML(html, contains: #"value="1" style="min-width:80px;" id="cfg-max-sequences""#))
    #expect(dashboardHTML(html, contains: "setNumberInput('cfg-max-sequences', appConfig.max_sequences, 1)"))
    #expect(dashboardHTML(html, contains: "numberInputValue('cfg-max-sequences', 1)"))
    #expect(dashboardHTML(html, contains: "setNumberInput('cfg-max-models', appConfig.max_models, 3)"))
    #expect(!dashboardHTML(html, contains: "cfg-init-cache-blocks"))
    #expect(!dashboardHTML(html, contains: "init_cache_blocks"))
    #expect(dashboardHTML(html, contains: "setNumberInput('cfg-model-ttl', appConfig.model_ttl_minutes, 30)"))
    #expect(dashboardHTML(html, contains: #"id="cfg-verify-model-on-load""#))
    #expect(dashboardHTML(html, contains: "setCheckboxInput('cfg-verify-model-on-load', appConfig.verify_model_on_load, false)"))
    #expect(dashboardHTML(html, contains: "verify_model_on_load: checkedInputValue('cfg-verify-model-on-load', false)"))
}

@Test func dashboardMaxModelsCopyDescribesExplicitLoadedModelLimit() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "最大同时加载模型数"))
    #expect(dashboardHTML(html, contains: "达到上限后，新模型不会自动替换旧模型"))
    #expect(!dashboardHTML(html, contains: "按 LRU 淘汰"))
    #expect(!dashboardHTML(html, contains: "LRU eviction"))
}

@Test func dashboardHotCacheDescriptionUsesUnifiedMemoryAutoPolicy() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "system unified memory"))
    #expect(dashboardHTML(html, contains: "上限为 8GB"))
    #expect(!dashboardHTML(html, contains: "GPU memory / 4"))
    #expect(!dashboardHTML(html, contains: "GPU 内存 / 4"))
}

@Test func dashboardMemoryLimitAutoLabelDoesNotShowSliderValue() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "label.textContent = dict.auto || 'Auto'"))
    #expect(dashboardHTML(html, contains: "sliderId === 'cfg-hot-cache'"))
    #expect(!dashboardHTML(html, contains: "Auto (24GB)"))
    #expect(!dashboardHTML(html, contains: "Auto (22GB)"))
}

@Test func dashboardMemoryLimitAutoSlidersMoveToZeroAndRestoreManualValue() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "const usesVisualAutoZero = true"))
    #expect(dashboardHTML(html, contains: "slider.dataset.manualValue = slider.value"))
    #expect(dashboardHTML(html, contains: "slider.value = '0'"))
    #expect(dashboardHTML(html, contains: "slider.value = slider.dataset.manualValue"))
}

@Test func dashboardHotCacheAutoModeShowsAutoLabelAndSeparateEstimate() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "hot_cache_auto_hint"))
    #expect(dashboardHTML(html, contains: "settings-control-main"))
    #expect(dashboardHTML(html, contains: "当前自动估算值：{value} GB"))
    #expect(dashboardHTML(html, contains: "label.textContent = dict.auto || 'Auto'"))
    #expect(!dashboardHTML(html, contains: "hot_cache_auto_label"))
    #expect(!dashboardHTML(html, contains: "Auto ({value} GB)"))
    #expect(!dashboardHTML(html, contains: "自动（{value} GB）"))
}

@Test func dashboardCacheLimitDescriptionsUseHelpTooltips() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"aria-label="Hot cache help""#))
    #expect(dashboardHTML(html, contains: #"aria-label="Cold cache help""#))
    #expect(dashboardHTML(html, contains: #"data-i18n="hot_cache_help_title""#))
    #expect(dashboardHTML(html, contains: #"data-i18n="hot_cache_help_body""#))
    #expect(dashboardHTML(html, contains: #"data-i18n="cold_cache_help_title""#))
    #expect(dashboardHTML(html, contains: #"data-i18n="cold_cache_help_body""#))
    #expect(dashboardHTML(html, contains: #"hot_cache_desc: "用于频繁访问的前缀缓存。""#))
    #expect(dashboardHTML(html, contains: #"cold_cache_desc: "限制 SSD Prefix Cache 的最大磁盘占用。""#))
}

@Test func dashboardModelManagementShowsMtpAvailabilityAndLoadAction() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "mtp_badge_available"))
    #expect(dashboardHTML(html, contains: "mtp_badge_enabled"))
    #expect(!dashboardHTML(html, contains: "mtp_badge_incompatible"))
    #expect(!dashboardHTML(html, contains: "badge-mtp-warning"))
    #expect(!dashboardHTML(html, contains: "MTP 不兼容"))
    #expect(dashboardHTML(html, contains: #"mtp_badge_available: "MTP""#))
    #expect(dashboardHTML(html, contains: #"mtp_badge_enabled: "MTP""#))
    #expect(dashboardHTML(html, contains: "action_load_mtp"))
    #expect(dashboardHTML(html, contains: "action_load_model_only"))
    #expect(dashboardHTML(html, contains: #"action_load_model_only: "仅模型""#))
    #expect(dashboardHTML(html, contains: #"action_load_mtp: "模型+MTP""#))
    #expect(dashboardHTML(html, contains: "handleModelLoadChoice"))
    #expect(dashboardHTML(html, contains: "action-load-select"))
    #expect(dashboardHTML(html, contains: "-webkit-appearance: none"))
    #expect(dashboardHTML(html, contains: "text-align-last: center"))
    #expect(dashboardHTML(html, contains: #"<option value="" selected disabled hidden>"#))
    #expect(dashboardHTML(html, contains: "loadModelWithMtp"))
    #expect(dashboardHTML(html, contains: "renderMtpBadge"))
    #expect(dashboardHTML(html, contains: "renderModelMoreActions"))
    #expect(dashboardHTML(html, contains: "verifyModelIntegrity"))
    #expect(dashboardHTML(html, contains: "onModelIntegrityStatus"))
    #expect(dashboardHTML(html, contains: "status_verifying"))
    #expect(dashboardHTML(html, contains: "status_verified"))
    #expect(dashboardHTML(html, contains: "status_corrupt"))
    #expect(!dashboardHTML(html, contains: "model-action-group"))
}

@Test func dashboardPinTooltipExplainsFixedLoadingBehavior() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "pin_model_tooltip"))
    #expect(dashboardHTML(html, contains: "unpin_model_tooltip"))
    #expect(dashboardHTML(html, contains: "固定加载：不会被 TTL 自动卸载，也不会参与自动释放策略。仍可手动卸载。"))
    #expect(dashboardHTML(html, contains: "已固定加载：点击取消固定，恢复 TTL 和自动释放策略。"))
    #expect(dashboardHTML(html, contains: "pinTooltip(m.pinned)"))
    #expect(dashboardHTML(html, contains: "pinTooltip(nowPinned)"))
    #expect(!dashboardHTML(html, contains: #"title="' + (m.pinned ? 'Unpin' : 'Pin') + '""#))
    #expect(!dashboardHTML(html, contains: "pp.btn.title = nowPinned ? 'Unpin' : 'Pin'"))
}

@Test func dashboardModelStatusColumnUsesLifecycleColorsAndPendingStates() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #".status-dot.model-ready { background: var(--accent); }"#))
    #expect(dashboardHTML(html, contains: #".status-dot.model-loaded { background: var(--green); }"#))
    #expect(dashboardHTML(html, contains: #".status-dot.model-busy { background: var(--warning); }"#))
    #expect(dashboardHTML(html, contains: #"status_ready: "未加载""#))
    #expect(dashboardHTML(html, contains: "status_loading"))
    #expect(dashboardHTML(html, contains: "status_unloading"))
    #expect(dashboardHTML(html, contains: "updateModelStatusForAction(btn, 'model-busy', dict.status_loading"))
    #expect(dashboardHTML(html, contains: "updateModelStatusForAction(btn, 'model-busy', dict.status_unloading"))
}

@Test func dashboardModelTablePrioritizesModelColumnWidth() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="col-model" style="width:32%;""#))
    #expect(dashboardHTML(html, contains: #"class="col-type" style="width:8%;""#))
    #expect(dashboardHTML(html, contains: #"class="col-quant" style="width:8%;""#))
    #expect(dashboardHTML(html, contains: #"class="col-default" style="width:6%;""#))
    #expect(dashboardHTML(html, contains: #"class="col-params" style="width:6%;""#))
    #expect(dashboardHTML(html, contains: #"class="col-select" style="width:44px; min-width:44px; max-width:44px;""#))
    #expect(dashboardHTML(html, contains: #"""
  .data-table th {
    text-align: center;
"""#))
    #expect(dashboardHTML(html, contains: #".data-table th:first-child,"#))
    #expect(dashboardHTML(html, contains: #"text-overflow: clip;"#))
    #expect(dashboardHTML(html, contains: #".data-table th:nth-child(3),"#))
    #expect(dashboardHTML(html, contains: #".data-table td:nth-child(3),"#))
    #expect(dashboardHTML(html, contains: #".data-table th:nth-child(4),"#))
    #expect(dashboardHTML(html, contains: #".data-table td:nth-child(4),"#))
    #expect(dashboardHTML(html, contains: #".data-table th:nth-child(5),"#))
    #expect(dashboardHTML(html, contains: #".data-table td:nth-child(5),"#))
    #expect(dashboardHTML(html, contains: #".data-table th:nth-child(6),"#))
    #expect(dashboardHTML(html, contains: #".data-table td:nth-child(6),"#))
    #expect(dashboardHTML(html, contains: #".data-table th:nth-child(7),"#))
    #expect(dashboardHTML(html, contains: #".data-table td:nth-child(7),"#))
    #expect(dashboardHTML(html, contains: #".data-table th:nth-child(8),"#))
    #expect(dashboardHTML(html, contains: #".data-table td:nth-child(8)"#))
    #expect(dashboardHTML(html, contains: #"text-align: center;"#))
    #expect(dashboardHTML(html, contains: #".data-table .model-name-text"#))
    #expect(dashboardHTML(html, contains: #"min-width: 0;"#))
    #expect(dashboardHTML(html, contains: #"text-overflow: ellipsis;"#))
}

@Test func dashboardModelsPageKeepsHeaderAndTabsFixedAboveAnIndependentScroller() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="models-page-header""#))
    #expect(dashboardHTML(
        html,
        contains: #"class="models-page-scroll page-body-scroll model-manager-active""#
    ))
    #expect(dashboardHTML(html, contains: #".content.page-scroll-managed {"#))
    #expect(dashboardHTML(html, contains: #"#page-models.active,"#))
    #expect(dashboardHTML(html, contains: #"#page-models > .tab-bar {"#))
    #expect(dashboardHTML(html, contains: #"overflow-y: hidden;"#))
    #expect(dashboardHTML(html, contains: #"overflow-y: auto;"#))
    #expect(dashboardHTML(
        html,
        contains: #"page === 'status' || page === 'models' || page === 'benchmark'"#
    ))
    #expect(dashboardHTML(html, contains: #"scrollContainer.scrollTop = 0"#))
}

@Test func dashboardBenchmarkPageKeepsHeaderFixedAboveAnIndependentScroller() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="benchmark-page-header""#))
    #expect(dashboardHTML(
        html,
        contains: #"class="benchmark-page-scroll page-body-scroll""#
    ))
    #expect(dashboardHTML(html, contains: #"#page-benchmark.active,"#))
    #expect(dashboardHTML(html, contains: #".benchmark-page-header,"#))
    #expect(dashboardHTML(
        html,
        contains: #"page === 'models' || page === 'benchmark' ||"#
    ))
    #expect(dashboardHTML(
        html,
        contains: #"const pageScroll = pageEl && pageEl.querySelector('.page-body-scroll')"#
    ))
}

@Test func dashboardLogsPageKeepsHeaderFixedAboveAnIndependentScroller() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="logs-page-header""#))
    #expect(dashboardHTML(
        html,
        contains: #"class="logs-page-scroll page-body-scroll""#
    ))
    #expect(dashboardHTML(html, contains: #"#page-logs.active,"#))
    #expect(dashboardHTML(html, contains: #".logs-page-header,"#))
    #expect(dashboardHTML(
        html,
        contains: #"page === 'logs' || page === 'settings'"#
    ))
    #expect(dashboardHTML(html, contains: #"logOutput.scrollTop = logOutput.scrollHeight"#))

    let tabBarStart = try #require(html.range(of: ".log-tab-bar {"))
    let tabBarEnd = try #require(html[tabBarStart.upperBound...].firstIndex(of: "}"))
    let tabBarRule = String(html[tabBarStart.lowerBound...tabBarEnd])
    #expect(tabBarRule.contains("position: sticky;"))
    #expect(tabBarRule.contains("top: 0;"))
    #expect(tabBarRule.contains("z-index: 10;"))
    #expect(tabBarRule.contains("background: var(--bg);"))
}

@Test func dashboardLogsPageDescriptionCoversRuntimeLogsAndIncidentHistory() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    let expectedDescriptions = [
        "View runtime logs and incident history to understand service status and troubleshoot issues.",
        "查看运行日志与故障历史，帮助了解服务状态并排查问题。",
        "查看執行日誌與故障歷史，協助了解服務狀態並排查問題。",
        "実行ログと障害履歴を確認し、サービスの状態把握と問題の調査に役立てます。",
        "실행 로그와 장애 기록을 확인하여 서비스 상태를 파악하고 문제를 해결합니다.",
    ]

    for description in expectedDescriptions {
        #expect(dashboardHTML(html, contains: description))
    }
    #expect(!dashboardHTML(html, contains: "Server inference logs."))
    #expect(!dashboardHTML(html, contains: "服务器推理日志。"))
}

@Test func dashboardSettingsPageKeepsHeaderFixedAboveAnIndependentScroller() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="settings-page-header""#))
    #expect(dashboardHTML(
        html,
        contains: #"class="settings-page-scroll page-body-scroll""#
    ))
    #expect(dashboardHTML(html, contains: #"#page-settings.active {"#))
    #expect(dashboardHTML(html, contains: #".settings-page-header,"#))
    #expect(dashboardHTML(html, contains: #".page-body-scroll {"#))
    #expect(dashboardHTML(
        html,
        contains: #"page === 'logs' || page === 'settings'"#
    ))
    #expect(dashboardHTML(
        html,
        contains: #"const pageScroll = pageEl && pageEl.querySelector('.page-body-scroll')"#
    ))
    #expect(dashboardHTML(html, contains: #"if (pageScroll) pageScroll.scrollTop = 0"#))
}

@Test func dashboardStatusPageKeepsHeaderFixedAboveAnIndependentScroller() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="status-page-header""#))
    #expect(
        dashboardHTML(
            html,
            contains: #"class="status-page-scroll page-body-scroll""#
        )
    )
    #expect(dashboardHTML(html, contains: #"#page-status.active,"#))
    #expect(dashboardHTML(html, contains: #".status-page-header,"#))
    #expect(dashboardHTML(html, contains: #"class="content page-scroll-managed""#))
    #expect(dashboardHTML(html, contains: #"page === 'status' || page === 'models'"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"document.querySelector('#page-status .status-page-scroll')"#
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: #"statusPage.querySelector('.status-page-scroll')"#
        )
    )
}

@Test func dashboardSchedulerProfileUsesItsOwnModelsTab() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(
        html,
        contains: #"data-tab="models-scheduler" data-i18n="scheduler_profile_tab""#
    ))
    #expect(dashboardHTML(html, contains: #"id="tab-models-scheduler""#))
    #expect(dashboardHTML(html, contains: #"scheduler_profile_tab: "调度配置""#))

    let managerPanel = try #require(html.range(of: #"id="tab-models-manage""#))
    let schedulerPanel = try #require(html.range(of: #"id="tab-models-scheduler""#))
    let profileCard = try #require(html.range(of: #"id="profile-generation-card""#))
    let downloadPanel = try #require(html.range(of: #"id="tab-models-download""#))

    #expect(managerPanel.lowerBound < schedulerPanel.lowerBound)
    #expect(schedulerPanel.lowerBound < profileCard.lowerBound)
    #expect(profileCard.lowerBound < downloadPanel.lowerBound)
}

@Test func dashboardModelManagerKeepsTableHeaderAboveIndependentRowScroller() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(
        html,
        contains: #"class="models-page-scroll page-body-scroll model-manager-active""#
    ))
    #expect(dashboardHTML(html, contains: #".models-page-scroll.model-manager-active {"#))
    #expect(dashboardHTML(html, contains: #"#tab-models-manage.active {"#))
    #expect(dashboardHTML(html, contains: #"#tab-models-manage > .card {"#))
    #expect(dashboardHTML(html, contains: #"#tab-models-manage .data-table-wrapper {"#))
    #expect(dashboardHTML(html, contains: #"#tab-models-manage .data-table thead th {"#))
    #expect(dashboardHTML(html, contains: #"position: sticky;"#))
    #expect(dashboardHTML(html, contains: #"scrollContainer.classList.toggle('model-manager-active', tabName === 'models-manage')"#))
    #expect(dashboardHTML(
        html,
        contains: #"btn.addEventListener('click', () => switchToTab(btn.dataset.tab))"#
    ))
}

@Test func dashboardHuggingFaceSearchResultStartsTheExistingDownloadFlow() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"onclick="handleSearchResultAction(\'"#))
    #expect(dashboardHTML(html, contains: #"function handleSearchResultAction(repoId, localState, commitSHA, button) {"#))
    #expect(dashboardHTML(html, contains: #"function downloadSearchResult(repoId, button) {"#))
    #expect(dashboardHTML(html, contains: #"startDownload(repoId);"#))
    #expect(!dashboardHTML(html, contains: #"function fillRepoId(repoId) {"#))
    #expect(dashboardHTML(
        html,
        contains: #"download_in_progress: "已有 HuggingFace 模型正在下载。""#
    ))

    let oneClickFlow = try #require(
        html.range(of: #"function downloadSearchResult(repoId, button) {"#)
    )
    let start = try #require(
        html.range(of: #"startDownload(repoId);"#, range: oneClickFlow.lowerBound ..< html.endIndex)
    )
    #expect(oneClickFlow.lowerBound < start.lowerBound)
}

@Test func dashboardHuggingFaceSearchDownloadButtonTracksTaskLifecycle() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"\', this)">"#))
    #expect(dashboardHTML(html, contains: #"setSearchResultDownloadState(button, true);"#))
    #expect(dashboardHTML(html, contains: #"dict.download_btn_downloading || 'Downloading...'"#))
    #expect(dashboardHTML(html, contains: #".search-result-dl:disabled {"#))
    #expect(dashboardHTML(
        html,
        contains: #"const task = downloadTask('huggingface', repoId);"#
    ))
    #expect(dashboardHTML(
        html,
        contains: #"const isDownloading = isActiveDownload(task);"#
    ))
    #expect(dashboardHTML(html, contains: #"refreshSearchResultDownloadStates();"#))
}

@Test func dashboardHuggingFaceDownloadActionsShareButtonAndIconStyling() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(
        html,
        contains: #"class="btn-accent hf-download-action search-result-dl""#
    ))
    #expect(dashboardHTML(html, contains: #".hf-download-action .download-button-icon {"#))
    #expect(dashboardHTML(html, contains: #".search-result-dl {"#))
    #expect(dashboardHTML(html, contains: #"min-width: 82px;"#))
    #expect(dashboardHTML(html, contains: #"justify-content: center;"#))
    #expect(
        html.components(separatedBy: #"class="download-button-icon""#).count - 1 == 1
    )
}

@Test func dashboardHuggingFaceSearchOwnsTokenResultsAndDownloadProgress() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(!dashboardHTML(html, contains: #"id="dl-repo-id""#))
    #expect(!dashboardHTML(html, contains: #"id="dl-hf-token""#))
    #expect(!dashboardHTML(html, contains: #"id="dl-download-btn""#))
    #expect(dashboardHTML(html, contains: #"data-i18n="hf_model_query""#))
    #expect(dashboardHTML(html, contains: #"id="search-hf-token""#))
    #expect(dashboardHTML(html, contains: #"class="field-label-with-help""#))
    #expect(dashboardHTML(html, contains: #"class="profile-help profile-help-fixed""#))
    #expect(dashboardHTML(html, contains: #"aria-describedby="hf-token-help-tooltip""#))
    #expect(
        dashboardHTML(
            html,
            contains: #"class="profile-help-tooltip profile-help-tooltip-fixed" id="hf-token-help-tooltip" role="tooltip""#
        )
    )
    #expect(dashboardHTML(html, contains: #"data-i18n="hf_token_help_title""#))
    #expect(dashboardHTML(html, contains: #"data-i18n="hf_token_help_body""#))
    #expect(
        dashboardHTML(
            html,
            contains: #"hf_token_help_body: "通常无需填写。下载私有模型、已在 Hugging Face 接受许可的受限模型，或 Hugging Face 要求身份验证时，请输入访问令牌。令牌仅用于 Hugging Face 请求。""#
        )
    )
    #expect(dashboardHTML(html, contains: #"data-i18n="sort_by""#))
    #expect(dashboardHTML(html, contains: #".hf-search-controls {"#))
    #expect(dashboardHTML(html, contains: #"token: token || null"#))
    #expect(dashboardHTML(html, contains: #"request_id: requestId"#))
    #expect(dashboardHTML(html, contains: #"function onSearchError(requestId, jsonStr) {"#))
    #expect(dashboardHTML(html, contains: #"resultsDiv.replaceChildren(error);"#))

    let query = try #require(html.range(of: #"id="search-hf-query""#))
    let token = try #require(
        html.range(of: #"id="search-hf-token""#, range: query.upperBound ..< html.endIndex)
    )
    let sort = try #require(
        html.range(of: #"id="search-hf-sort""#, range: token.upperBound ..< html.endIndex)
    )
    let results = try #require(
        html.range(of: #"id="search-results""#, range: sort.upperBound ..< html.endIndex)
    )
    #expect(query.lowerBound < token.lowerBound)
    #expect(token.lowerBound < sort.lowerBound)
    #expect(sort.lowerBound < results.lowerBound)
    #expect(dashboardHTML(html, contains: #"id="download-task-list""#))
    #expect(dashboardHTML(html, contains: #"function onDownloadProgress(provider, repoId, pct, filename) {"#))
}

@Test func dashboardHuggingFaceSearchPlaceholderFollowsDashboardLanguage() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(
        dashboardHTML(
            html,
            contains: #"data-i18n-placeholder="search_hf_placeholder""#
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: #"document.querySelectorAll('[data-i18n-placeholder]')"#
        )
    )
    #expect(dashboardHTML(html, contains: #"const key = el.dataset.i18nPlaceholder;"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"el.setAttribute('placeholder', dict[key]);"#
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: #"search_hf_placeholder: "输入模型名称或组织/模型""#
        )
    )
}

@Test func dashboardClearsHuggingFaceResultsAndRejectsStaleResponsesForEmptyQuery() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"function onHuggingFaceQueryInput() {"#))
    #expect(dashboardHTML(html, contains: #"activeHuggingFaceSearchRequestId = null;"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"document.getElementById('search-results').replaceChildren();"#
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: #"huggingFaceSearchQueryInput.addEventListener('input', onHuggingFaceQueryInput);"#
        )
    )
    #expect(dashboardHTML(html, contains: #"function onSearchResults(requestId, jsonStr) {"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"if (requestId !== activeHuggingFaceSearchRequestId) return;"#
        )
    )
    #expect(dashboardHTML(html, contains: #"request_id: requestId"#))
}

@Test func dashboardHuggingFaceSearchDebouncesTypingAndSupportsImmediateActions() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"const HUGGING_FACE_SEARCH_DEBOUNCE_MS = 600;"#))
    #expect(dashboardHTML(html, contains: #"const HUGGING_FACE_SEARCH_MIN_CHARACTERS = 2;"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"huggingFaceSearchDebounceTimer = setTimeout(function() {"#
        )
    )
    #expect(dashboardHTML(html, contains: #"searchHuggingFace(true);"#))
    #expect(dashboardHTML(html, contains: #"function searchHuggingFace(automatic = false) {"#))
    #expect(dashboardHTML(html, contains: #"activeHuggingFaceSearchShowsErrorToast = !automatic;"#))
    #expect(dashboardHTML(html, contains: #"if (showErrorToast) showToast(message, 'warn');"#))
    #expect(dashboardHTML(html, contains: #"window.webkit.messageHandlers.cancelHFSearch"#))
    #expect(dashboardHTML(html, contains: #"'compositionstart'"#))
    #expect(dashboardHTML(html, contains: #"'compositionend'"#))
    #expect(dashboardHTML(html, contains: #"event.key !== 'Enter'"#))
    #expect(dashboardHTML(html, contains: #"searchHuggingFace(false);"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"document.getElementById('search-hf-sort').addEventListener('change'"#
        )
    )
}

@Test func dashboardUsesUnifiedDownloadQueueAndRestartRecoveryReminder() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"class="search-result-entry" data-repo-id=""#))
    #expect(dashboardHTML(html, contains: #"id="download-queue-card""#))
    #expect(dashboardHTML(html, contains: #"id="download-recovery""#))
    #expect(dashboardHTML(html, contains: #"const ACTIVE_DOWNLOAD_PHASES = new Set(["#))
    #expect(dashboardHTML(html, contains: #"function renderDownloadQueue(snapshot) {"#))
    #expect(dashboardHTML(html, contains: #"id="download-clear-finished""#))
    #expect(dashboardHTML(html, contains: #"download_clear_finished: "清除记录""#))
    #expect(dashboardHTML(html, contains: #"data-i18n-aria-label="download_clear_finished_accessible_label""#))
    #expect(dashboardHTML(html, contains: #"function clearFinishedDownloadTasks() {"#))
    #expect(dashboardHTML(html, contains: #"apiPost('/admin/api/models/downloads/clear-finished', {});"#))
    #expect(dashboardHTML(html, contains: #"function readdPublicDownloadReminders() {"#))
    #expect(dashboardHTML(html, contains: #"window.__DOWNLOAD_QUEUE_POLL__ = setInterval(refreshDownloadQueue, 1500);"#))
    #expect(!dashboardHTML(html, contains: #"id="hf-progress-home""#))
    #expect(!dashboardHTML(html, contains: #"id="dl-ms-progress""#))
}

@Test func dashboardModelMoreActionsUseBodyPortalOutsideClippedTableCells() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #".model-more-menu {"#))
    #expect(dashboardHTML(html, contains: #"position: fixed;"#))
    #expect(dashboardHTML(html, contains: #"document.body.appendChild(menu)"#))
    #expect(dashboardHTML(html, contains: #"positionModelMoreActions(menu, trigger)"#))
    #expect(dashboardHTML(html, contains: #"width: max-content;"#))
    #expect(dashboardHTML(html, contains: #"gap: 0.45em;"#))
    #expect(dashboardHTML(html, contains: #"width: 1em;"#))
    #expect(dashboardHTML(html, contains: #"height: 1em;"#))
    #expect(dashboardHTML(html, contains: #"font-size: 13px;"#))
    #expect(dashboardHTML(html, contains: #"stroke-width', '1.4'"#))
    #expect(dashboardHTML(html, contains: #"createModelMoreActionIcon('verify')"#))
    #expect(dashboardHTML(html, contains: #"createModelMoreActionIcon('versions')"#))
    #expect(dashboardHTML(html, contains: #"createModelMoreActionIcon('delete')"#))
    #expect(dashboardHTML(html, contains: #"'M6 3h8l4 4v14H6V3Z'"#))
    #expect(dashboardHTML(html, contains: #"'m9 14 2 2 4-4'"#))
    #expect(dashboardHTML(html, contains: #"stroke', 'currentColor'"#))
    #expect(dashboardHTML(html, contains: #"verifyButton.className = 'model-more-verify'"#))
    #expect(dashboardHTML(html, contains: #"versionsButton.className = 'model-more-versions'"#))
    #expect(dashboardHTML(html, contains: #"deleteButton.className = 'model-more-delete'"#))
    #expect(dashboardHTML(html, contains: #"button.model-more-verify svg { color: var(--accent); }"#))
    #expect(dashboardHTML(html, contains: #"button.model-more-versions svg { color: #7c6ee6; }"#))
    #expect(dashboardHTML(html, contains: #"button.model-more-delete svg { color: var(--destructive); }"#))
    #expect(dashboardHTML(html, contains: #"aria-haspopup="menu""#))
    #expect(dashboardHTML(html, contains: #"closeModelMoreActions({ restoreFocus: true })"#))
    #expect(dashboardHTML(html, contains: #"verify_integrity: "验证模型完整性""#))
    #expect(dashboardHTML(html, contains: #"manage_versions: "管理版本""#))
    #expect(dashboardHTML(html, contains: #"delete_model: "删除模型""#))
    #expect(!dashboardHTML(html, contains: #".model-more-menu button.danger"#))
    #expect(!dashboardHTML(html, contains: #"deleteButton.className = 'danger'"#))
    #expect(!dashboardHTML(html, contains: #"<details class="model-more">"#))
}

@Test func dashboardModelVersionManagementSupportsActivationRollbackAndExplicitCleanup() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"id="model-versions-modal""#))
    #expect(dashboardHTML(html, contains: #"function openModelVersions(modelId) {"#))
    #expect(dashboardHTML(html, contains: #"window.webkit.messageHandlers.listModelVersions"#))
    #expect(dashboardHTML(html, contains: #"window.webkit.messageHandlers.activateModelVersion"#))
    #expect(dashboardHTML(html, contains: #"window.webkit.messageHandlers.deleteModelVersions"#))
    #expect(dashboardHTML(html, contains: #"t('rollback_reload', 'Roll back and reload')"#))
    #expect(dashboardHTML(html, contains: #"class="model-version-checkbox""#))
    #expect(dashboardHTML(html, contains: #"confirm_delete_versions: "确定删除所选 {n} 个版本并释放 {size}？此操作不可恢复。""#))
}

@Test func dashboardHuggingFaceSearchDistinguishesExactLocalAndUpdateStates() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"const localState = m.local_state || 'available';"#))
    #expect(dashboardHTML(html, contains: #"exists: dict.local_model_exists || '✓ Exists'"#))
    #expect(dashboardHTML(html, contains: #"update_available: dict.local_model_update || 'Download update'"#))
    #expect(dashboardHTML(html, contains: #"local_inactive: dict.local_model_inactive || 'Use local version'"#))
    #expect(dashboardHTML(html, contains: #"repair: dict.local_model_repair || 'Repair download'"#))
    #expect(dashboardHTML(html, contains: #"identity_unavailable: dict.local_model_identity_unavailable || 'Version unavailable'"#))
    #expect(dashboardHTML(html, contains: #"localState === 'identity_unavailable'"#))
    #expect(dashboardHTML(html, contains: #"const actionIcon = localState === 'exists' || isCompleted"#))
    #expect(dashboardHTML(html, contains: #"+ actionIcon"#))
    #expect(dashboardHTML(html, contains: #"local_model_exists: "✓ 已存在""#))
    #expect(dashboardHTML(html, contains: #"local_model_completed: "✓ 已完成""#))
    #expect(dashboardHTML(html, contains: #"local_model_update: "下载新版本""#))
    #expect(dashboardHTML(html, contains: #"local_model_identity_unavailable: "版本身份无法确认""#))
}

@Test func dashboardModelTypeColumnUsesCapabilityLabels() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "function renderModelType(model)"))
    #expect(dashboardHTML(html, contains: "vlm: 'LLM/VLM'"))
    #expect(dashboardHTML(html, contains: "block_diffusion_vlm: 'Block Diffusion VLM'"))
    #expect(dashboardHTML(html, contains: "embedding: 'Embedding'"))
    #expect(dashboardHTML(html, contains: "reranker: 'Reranker'"))
    #expect(dashboardHTML(html, contains: "asr: 'ASR'"))
    #expect(dashboardHTML(html, contains: "tts: 'TTS'"))
    #expect(dashboardHTML(html, contains: "renderModelType(m)"))
    #expect(dashboardHTML(html, contains: #"option value="vlm">LLM/VLM</option>"#))
    #expect(
        dashboardHTML(
            html,
            contains: #"option value="block_diffusion_vlm">Block Diffusion VLM</option>"#
        )
    )
    #expect(dashboardHTML(html, contains: #"option value="asr">ASR</option>"#))
    #expect(dashboardHTML(html, contains: #"option value="tts">TTS</option>"#))
}

@Test func dashboardQuantColumnHidesAffinePrefixInDisplayOnly() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "function quantDisplayLabel(quant)"))
    #expect(dashboardHTML(html, contains: "quant.kind === 'affine' && quant.bits"))
    #expect(dashboardHTML(html, contains: "return quant.bits + '-bit';"))
    #expect(dashboardHTML(html, contains: "const label = quantDisplayLabel(quant);"))
    #expect(dashboardHTML(html, contains: "titleParts.push('kind=' + quant.kind)"))
}

@Test func dashboardModelParamsModalIncludesMtpRuntimeControls() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"id="modal-mtp-enabled""#))
    #expect(dashboardHTML(html, contains: #"id="modal-mtp-model""#))
    #expect(dashboardHTML(html, contains: #"id="modal-mtp-draft-tokens""#))
    #expect(dashboardHTML(html, contains: "mtp_enabled"))
    #expect(dashboardHTML(html, contains: "mtp_model_id"))
    #expect(dashboardHTML(html, contains: "mtp_draft_tokens"))
    #expect(dashboardHTML(html, contains: #"id="modal-causal-sampling-row""#))
    #expect(dashboardHTML(html, contains: #"id="modal-mtp-section""#))
    #expect(dashboardHTML(html, contains: #"class="modal-row mtp-header-row""#))
    #expect(
        dashboardHTML(
            html,
            contains: ".mtp-header-row {\n    align-items: center;\n    margin-bottom: 12px;"
        )
    )
    #expect(dashboardHTML(html, contains: #"id="modal-prompt-lookup-section""#))
    #expect(dashboardHTML(html, contains: "capabilities.runtime_kind === 'block_diffusion'"))
    #expect(dashboardHTML(html, contains: "capabilities.supports_mtp === false"))
    #expect(dashboardHTML(html, contains: "capabilities.supports_prompt_lookup === false"))
    #expect(
        dashboardHTML(
            html,
            contains: "models.filter(m => m.capabilities?.runtime_kind !== 'block_diffusion')"
        )
    )
}

@Test func dashboardModelParamsAndLoadActionsExposeDFlash2AsAnExclusiveMode() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"id="modal-dflash2-enabled""#))
    #expect(dashboardHTML(html, contains: #"id="modal-dflash2-model""#))
    #expect(dashboardHTML(html, contains: #"id="modal-dflash2-block-size""#))
    #expect(dashboardHTML(html, contains: #"id="modal-dflash2-draft-bits""#))
    #expect(dashboardHTML(html, contains: #"id="modal-dflash2-tensor-batch-max-width""#))
    #expect(dashboardHTML(html, contains: #"class="modal-row dflash2-tensor-row""#))
    #expect(
        dashboardHTML(
            html,
            contains: ".dflash2-tensor-row .modal-field {\n    flex: 0 0 calc((100% - 10px) / 2);"
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: #"id="dflash2-tensor-batch-help-tooltip" role="tooltip""#
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: #"data-i18n-aria-label="dflash2_tensor_batch_help_label""#
        )
    )
    #expect(dashboardHTML(html, contains: "saved.dflash2_tensor_batch_max_width || ''"))
    #expect(dashboardHTML(html, contains: "Max Sequences、该上限和当前就绪兼容请求数的最小值"))
    #expect(dashboardHTML(html, contains: "【设置 → 高级 → 最大序列数】"))
    #expect(dashboardHTML(html, contains: "function updateAccelerationExclusivity"))
    #expect(dashboardHTML(html, contains: "loadModelWithDFlash2"))
    #expect(dashboardHTML(html, contains: "action_load_dflash2"))
    #expect(dashboardHTML(html, contains: "renderDFlash2Badge"))
    #expect(dashboardHTML(html, contains: "dflash2_help_exclusive"))
    #expect(dashboardHTML(html, contains: "runtime_dflash2_acceptance"))
    #expect(dashboardHTML(html, contains: "exact_residual_corrections"))
    #expect(
        html.components(separatedBy: "err_backend_readiness_failed:").count - 1 == 5,
        "DFlash2 backend readiness failure must be localized in all five languages"
    )
    #expect(
        html.components(separatedBy: "params_invalid_dflash2_tensor_batch_max_width:").count - 1
            == 5,
        "DFlash2 tensor batch width validation must be localized in all five languages"
    )
    for key in [
        "dflash2_tensor_batch_help_label",
        "dflash2_tensor_batch_help_title",
        "dflash2_tensor_batch_help_body",
        "dflash2_tensor_batch_help_constraint",
    ] {
        #expect(
            html.components(separatedBy: "\(key):").count - 1 == 5,
            "\(key) must be localized in all five languages"
        )
    }
}

@Test func dashboardRendersUnifiedRuntimeHealth() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"id="runtime-model-health""#))
    #expect(dashboardHTML(html, contains: "function renderRuntimeModels(models)"))
    #expect(dashboardHTML(html, contains: "model.runtime_kind === 'block_diffusion'"))
    #expect(!dashboardHTML(html, contains: "const scheduler ="))
    #expect(dashboardHTML(html, contains: "runtime-model-card"))
    #expect(dashboardHTML(html, contains: "runtime-status-rail"))
    #expect(dashboardHTML(html, contains: "runtime-performance-rail"))
    #expect(dashboardHTML(html, contains: "runtime-model-footer"))
    #expect(dashboardHTML(html, contains: "runtime-kv-disclosure"))
    #expect(dashboardHTML(html, contains: "model.mtp_enabled === true"))
    #expect(dashboardHTML(html, contains: "badge-mtp-enabled"))
    #expect(dashboardHTML(html, contains: "runtime_cumulative_tokens"))
    #expect(dashboardHTML(html, contains: "runtime_cache_hit_tokens"))
    #expect(dashboardHTML(html, contains: "runtime_cache_hit_rate"))
    #expect(dashboardHTML(html, contains: "runtime_prefill_rate"))
    #expect(dashboardHTML(html, contains: "runtime_live_decode_rate"))
    #expect(dashboardHTML(html, contains: "runtime_recent_decode_rate"))
    #expect(dashboardHTML(html, contains: "runtime_session_decode_rate"))
    #expect(dashboardHTML(html, contains: "runtime_ttft"))
    #expect(dashboardHTML(html, contains: "performance.completed_requests"))
    #expect(dashboardHTML(html, contains: "performance.live_decode_tokens_per_second"))
    #expect(dashboardHTML(html, contains: "performance.prefill_tokens_per_second"))
    #expect(dashboardHTML(html, contains: "performance.decode_tokens_per_second"))
    #expect(dashboardHTML(html, contains: "performance.session_decode_tokens_per_second"))
    #expect(dashboardHTML(html, contains: "performance.ttft_ms"))
    #expect(dashboardHTML(html, contains: "if (!isDiffusion)"))
    #expect(dashboardHTML(html, contains: "if (usage.prefix_cache)"))
    #expect(dashboardHTML(html, contains: "eligibleTokens > 0"))
    #expect(dashboardHTML(html, contains: "renderRuntimeModels(window.__RUNTIME_MODELS__ || [])"))
    #expect(dashboardHTML(html, contains: "const activeKv = model.active_kv_offload"))
    #expect(dashboardHTML(html, contains: "if (activeKv)"))
    #expect(dashboardHTML(html, contains: #"runtime_not_applicable: "不适用""#))
    #expect(dashboardHTML(html, contains: "renderRuntimeModels(runtimeModels)"))
}

@Test func dashboardKeepsRuntimeCardVisibleWithoutLoadedModels() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(
        dashboardHTML(
            html,
            contains: #"id="runtime-model-health-card" style="margin-bottom:16px;""#
        )
    )
    #expect(dashboardHTML(html, contains: #"runtime_models: "Model Runtime Status""#))
    #expect(dashboardHTML(html, contains: #"runtime_models: "模型运行状态""#))
    #expect(dashboardHTML(html, contains: #"runtime_models: "模型執行狀態""#))
    #expect(dashboardHTML(html, contains: #"runtime_models: "モデル実行状況""#))
    #expect(dashboardHTML(html, contains: #"runtime_models: "모델 실행 상태""#))
    #expect(dashboardHTML(html, contains: #"runtime_empty_title: "暂无运行中的模型""#))
    #expect(
        dashboardHTML(
            html,
            contains: #"runtime_empty_description: "加载模型后，此处将显示各模型的请求队列与累计统计。""#
        )
    )
    #expect(dashboardHTML(html, contains: "card.style.display = '';"))
    #expect(dashboardHTML(html, contains: "dict.runtime_empty_title"))
    #expect(dashboardHTML(html, contains: "dict.runtime_empty_description"))
    #expect(
        !dashboardHTML(
            html,
            contains: "window.__RUNTIME_MODELS__ = [];\n      card.style.display = 'none';"
        )
    )
}

@Test func dashboardPreservesActiveKVDetailsAcrossRuntimePolling() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "const runtimeActiveKvOpenModelIds = new Set();"))
    #expect(dashboardHTML(html, contains: "function captureRuntimeActiveKvOpenState(container)"))
    #expect(
        dashboardHTML(
            html,
            contains: "container.querySelectorAll('details[data-active-kv-model-id]')"
        )
    )
    #expect(dashboardHTML(html, contains: "captureRuntimeActiveKvOpenState(container);"))
    #expect(dashboardHTML(html, contains: "runtimeActiveKvOpenModelIds.clear();"))
    #expect(
        dashboardHTML(
            html,
            contains: #"data-active-kv-model-id=""#
        )
    )
    #expect(
        dashboardHTML(
            html,
            contains: "runtimeActiveKvOpenModelIds.has(modelId) ? ' open' : ''"
        )
    )
}

@Test func dashboardGroupsServiceStatusAndUptimeIntoServiceOverview() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(!dashboardHTML(html, contains: #"id="stat-model""#))
    #expect(!dashboardHTML(html, contains: #"id="stat-total-tokens""#))
    #expect(!dashboardHTML(html, contains: #"id="stat-cached-tokens""#))
    #expect(!dashboardHTML(html, contains: #"id="stat-cache-efficiency""#))
    #expect(dashboardHTML(html, contains: #"id="stat-server-status""#))
    #expect(dashboardHTML(html, contains: #"id="stat-uptime""#))
    #expect(!dashboardHTML(html, contains: #"id="stat-active-kv""#))
    #expect(dashboardHTML(html, contains: #"id="server-endpoints-card""#))
    #expect(dashboardHTML(html, contains: #"class="service-overview-summary""#))
    #expect(dashboardHTML(html, contains: #"id="server-status-dot""#))
    #expect(dashboardHTML(html, contains: #"justify-content: flex-start;"#))
    #expect(dashboardHTML(html, contains: #"flex: 0 0 auto;"#))
    #expect(dashboardHTML(html, contains: #"white-space: nowrap;"#))
    #expect(dashboardHTML(html, contains: #"id="endpoints-list""#))
    #expect(dashboardHTML(html, contains: #"id="endpoints-hint""#))
    #expect(dashboardHTML(html, contains: #"service_overview: "Service Overview""#))
    #expect(dashboardHTML(html, contains: #"service_overview: "服务概览""#))
    #expect(dashboardHTML(html, contains: #"service_overview: "服務概覽""#))
    #expect(dashboardHTML(html, contains: #"service_overview: "サービス概要""#))
    #expect(dashboardHTML(html, contains: #"service_overview: "서비스 개요""#))
    #expect(dashboardHTML(html, contains: "function updateServerStatusVisual(text, color)"))
    #expect(!dashboardHTML(html, contains: #"class="stats-grid""#))
    #expect(!dashboardHTML(html, contains: #"class="stat-card""#))
    #expect(dashboardHTML(html, contains: "statusPage.querySelector('#runtime-model-health-card')"))
    #expect(!dashboardHTML(html, contains: #"total_tokens: "Total Tokens""#))
    #expect(!dashboardHTML(html, contains: #"cached_tokens: "Cached Tokens""#))
    #expect(!dashboardHTML(html, contains: #"cache_efficiency: "Cache Efficiency""#))
}

@Test func dashboardColdCacheUsesPositiveSsdLimit() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"min="1" max="10" value="10" id="cfg-cold-cache""#))
    #expect(dashboardHTML(html, contains: "numberInputValue('cfg-cold-cache', 10)"))
    #expect(dashboardHTML(html, contains: "const coldCache = clampColdCacheLimit(rawColdCache)"))
    #expect(!dashboardHTML(html, contains: "Set 0 to disable"))
    #expect(!dashboardHTML(html, contains: "设为 0 则禁用"))
    #expect(!dashboardHTML(html, contains: "Off (0GB)"))
}

@Test func dashboardColdCacheUsesDynamicDiskCapacityLimit() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "window.__IRONMLX_COLD_CACHE_CAPACITY__"))
    #expect(dashboardHTML(html, contains: "applyColdCacheCapacity("))
    #expect(dashboardHTML(html, contains: "clampColdCacheLimit("))
    #expect(dashboardHTML(html, contains: "/admin/api/cache/capacity"))
    #expect(dashboardHTML(html, contains: "cold_cache_too_large"))
    #expect(!dashboardHTML(html, contains: #"min="1" max="100" value="10" id="cfg-cold-cache""#))
}

@Test func dashboardRestartModelLoadFailureLocalizesMemoryBudgetErrors() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "function localizeBackendErrorMessage(message)"))
    #expect(dashboardHTML(html, contains: "err_memory_budget_exceeded"))
    #expect(dashboardHTML(html, contains: "err_kv_memory_budget_exceeded"))
    #expect(dashboardHTML(html, contains: "localizeErrorResult(result)"))
    #expect(dashboardHTML(html, contains: "内存预算不足"))
    #expect(dashboardHTML(html, contains: "活跃 KV Cache 分层卸载"))
    #expect(dashboardHTML(html, contains: "logical cap"))
    #expect(dashboardHTML(html, contains: "MAX CONTEXT TOKENS"))
    #expect(dashboardHTML(html, contains: "最大序列数（{max_sequences}）"))
    #expect(!dashboardHTML(html, contains: "b_max={b_max}"))
    #expect(!dashboardHTML(html, contains: "b_max="))
    #expect(!dashboardHTML(html, contains: "请调低「最大序列数」或 Max Cache Cap"))
}

@Test func dashboardSuccessfulManualRecoveryClearsCrashBanner() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "function onServerRestarted(payload)"))
    #expect(
        dashboardHTML(
            html,
            contains: """
            } else {
                  clearServerBanner();
                  if (result.model_loaded) {
            """
        )
    )
}

@Test func dashboardCrashBannersDoNotExposeRawBackendDetails() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: #"server_crash_restarting: "服务异常退出，正在重启...""#))
    #expect(dashboardHTML(html, contains: #"server_crash_recovering: "正在恢复服务...""#))
    #expect(
        dashboardHTML(
            html,
            contains: #"server_crash_breaker: "服务连续崩溃，已停止自动恢复。请检查日志后手动重试。""#
        )
    )
    #expect(!dashboardHTML(html, contains: #"server_crash_restarting: "服务异常退出（{detail}）"#))
    #expect(
        dashboardHTML(
            html,
            contains: "dict.recovery_reason_unknown || 'The service could not be restored."
        )
    )
}

@Test func dashboardLocalizesMaxLoadedModelsErrorCode() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "err_max_loaded_models_reached"))
    #expect(dashboardHTML(html, contains: "err_gpu_memory_insufficient"))
    #expect(dashboardHTML(html, contains: "err_model_memory_limit_exceeded"))
    #expect(dashboardHTML(html, contains: "err_total_memory_limit_exceeded"))
    #expect(dashboardHTML(html, contains: "Maximum concurrent loaded models reached"))
    #expect(dashboardHTML(html, contains: "已达到最大同时加载模型数"))
    #expect(dashboardHTML(html, contains: "已达到模型内存限制"))
    #expect(dashboardHTML(html, contains: "已达到总内存限制"))
    #expect(dashboardHTML(html, contains: "已達到最大同時載入模型數"))
    #expect(dashboardHTML(html, contains: "最大同時ロードモデル数に達しました"))
    #expect(dashboardHTML(html, contains: "최대 동시 로드 모델 수에 도달했습니다"))
    #expect(dashboardHTML(html, contains: "localizeErrorResult(result)"))
}

@Test func dashboardLocalizesBackendWarningCodes() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(dashboardHTML(html, contains: "function localizeWarningResult(result)"))
    #expect(dashboardHTML(html, contains: "warn_default_scheduler_profile_used"))
    #expect(dashboardHTML(html, contains: "warn_model_reload_deferred"))
    #expect(dashboardHTML(html, contains: "未找到该模型匹配的 scheduler profile"))
    #expect(dashboardHTML(html, contains: "showToast(localizeWarningResult(result), 'info')"))
}

@Test func dashboardLocalizesBackendAdminErrorCodes() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    let expectedKeys = [
        "err_model_required",
        "err_model_directory_not_found",
        "err_invalid_max_cache_cap",
        "err_model_not_loaded",
        "err_model_not_registered",
        "err_backend_unload_error",
        "err_kv_memory_budget_exceeded",
        "err_mtp_model_dir_required",
        "err_mtp_base_model_not_found",
        "err_mtp_model_not_found",
        "err_mtp_unsupported_architecture",
        "err_mtp_invalid_model_type",
        "err_mtp_invalid_config",
        "err_mtp_invalid_draft_tokens",
        "err_mtp_incompatible",
    ]
    for expectedKey in expectedKeys {
        #expect(dashboardHTML(html, contains: expectedKey), "\(expectedKey) should be localized")
    }

    let expectedMTPArchitectureMessages = [
        "MTP acceleration currently supports Qwen dense/MoE and Gemma4 models only.",
        "当前 MTP 加速仅支持 Qwen dense/MoE 和 Gemma4 模型。",
        "目前 MTP 加速僅支援 Qwen dense/MoE 和 Gemma4 模型。",
        "MTP アクセラレーションは現在 Qwen dense/MoE および Gemma4 モデルのみ対応しています。",
        "MTP 가속은 현재 Qwen dense/MoE 및 Gemma4 모델만 지원합니다.",
    ]
    for expectedMessage in expectedMTPArchitectureMessages {
        #expect(
            dashboardHTML(html, contains: expectedMessage),
            "MTP architecture error should include Gemma4 in every supported language"
        )
    }
}
