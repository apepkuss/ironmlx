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
    #expect(dashboardHTML(html, contains: "setNumberInput('cfg-init-cache-blocks', appConfig.init_cache_blocks, 0)"))
    #expect(dashboardHTML(html, contains: "setNumberInput('cfg-model-ttl', appConfig.model_ttl_minutes, 30)"))
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
    #expect(dashboardHTML(html, contains: "localizeBackendErrorMessage(detail)"))
    #expect(dashboardHTML(html, contains: "内存预算不足"))
    #expect(dashboardHTML(html, contains: "模型参数设置"))
    #expect(dashboardHTML(html, contains: "MAX TOKENS"))
    #expect(dashboardHTML(html, contains: "最大序列数={max_sequences}"))
    #expect(!dashboardHTML(html, contains: "b_max={b_max}"))
    #expect(!dashboardHTML(html, contains: "b_max="))
    #expect(!dashboardHTML(html, contains: "请调低「最大序列数」或 Max Cache Cap"))
}
