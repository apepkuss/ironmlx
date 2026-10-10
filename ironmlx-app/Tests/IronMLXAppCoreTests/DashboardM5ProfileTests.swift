import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
private final class CapturedSettingsMessages: NSObject, WKScriptMessageHandler {
    var bodies: [String] = []

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if let body = message.body as? String {
            bodies.append(body)
        }
    }
}

@MainActor
private func loadedDashboard(config: AppConfig, messages: CapturedSettingsMessages) async throws -> WKWebView {
    let configuration = WKWebViewConfiguration()
    configuration.userContentController.add(messages, name: "saveSettings")
    configuration.userContentController.add(messages, name: "setLogLevel")
    let bootstrap = try DashboardWindowController.bootstrapScript(config: config, route: .status)
    configuration.userContentController.addUserScript(
        WKUserScript(source: bootstrap, injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1000, height: 650), configuration: configuration)
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    let deadline = Date().addingTimeInterval(15)
    while Date() < deadline {
        if (try? await webView.evaluateJavaScript(
            "typeof settingsBaseline !== 'undefined' && settingsBaseline !== null")) as? Bool == true
        {
            break
        }
        try await Task.sleep(for: .milliseconds(100))
    }
    return webView
}

@MainActor
private func snapshot(_ webView: WKWebView, named name: String) async throws {
    guard let directory = ProcessInfo.processInfo.environment["IRONMLX_M5_PROFILE_SCREENSHOTS"] else { return }
    // Let CSS transitions (toggle slider) settle before capturing.
    try await Task.sleep(for: .milliseconds(600))
    let image = try await webView.takeSnapshot(configuration: nil)
    if let data = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: data),
       let png = bitmap.representation(using: .png, properties: [:])
    {
        try png.write(to: URL(fileURLWithPath: directory).appendingPathComponent("\(name).png"))
    }
}

@MainActor
@Test func dashboardM5ProfileToggleDefaultsOnAndSavesOffWithRestart() async throws {
    let messages = CapturedSettingsMessages()
    let webView = try await loadedDashboard(config: AppConfig(), messages: messages)
    defer { webView.configuration.userContentController.removeAllScriptMessageHandlers() }
    let result = try await webView.evaluateJavaScript("""
    (() => {
      navigateTo('settings'); setLanguage('en');
      // A window-less WKWebView does not advance CSS transitions; disable
      // them so the snapshot shows the settled toggle state.
      const noTransitions = document.createElement('style');
      noTransitions.textContent = '*, *::before, *::after { transition: none !important; }';
      document.head.appendChild(noTransitions);
      const toggle = document.getElementById('cfg-m5-dflash2-profile');
      const button = document.getElementById('settings-save-button');
      const status = document.getElementById('settings-save-status');
      const out = {defaultChecked: toggle.checked, cleanDisabled: button.disabled,
                   label: document.querySelector('[data-i18n="m5_dflash2_profile"]').textContent};
      settingsBackendRunning = true;
      toggle.click();
      out.afterToggleChecked = toggle.checked;
      out.buttonText = button.textContent;
      out.statusText = status.textContent;
      out.restartField = RESTART_FIELDS.includes('cfg-m5-dflash2-profile');
      out.sliderColor = getComputedStyle(toggle.nextElementSibling).backgroundColor;
      out.activeKvSliderColor = getComputedStyle(
        document.getElementById('cfg-active-kv-offload').nextElementSibling).backgroundColor;
      toggle.scrollIntoView({block: 'center'});
      return JSON.stringify(out);
    })()
    """) as? String
    let state = try #require(result.flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] })
    #expect(state["defaultChecked"] as? Bool == true)
    #expect(state["cleanDisabled"] as? Bool == true)
    #expect(state["label"] as? String == "M5 DFlash2 Profile")
    #expect(state["afterToggleChecked"] as? Bool == false)
    #expect(state["restartField"] as? Bool == true)
    #expect(state["buttonText"] as? String == "Save and restart service")
    // Off renders like the other unchecked toggle (Active KV offload).
    #expect(state["sliderColor"] as? String == state["activeKvSliderColor"] as? String)
    try await snapshot(webView, named: "settings-m5-toggle-off-en")

    _ = try await webView.evaluateJavaScript("saveSettings(); void 0")
    let deadline = Date().addingTimeInterval(5)
    while messages.bodies.isEmpty && Date() < deadline {
        try await Task.sleep(for: .milliseconds(50))
    }
    let payload = try #require(messages.bodies.last.flatMap {
        try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
    })
    #expect(payload["m5_dflash2_profile"] as? Bool == false)
}

@MainActor
@Test func dashboardM5ProfileRowSitsInAdvancedWithHelpTooltip() async throws {
    let messages = CapturedSettingsMessages()
    let webView = try await loadedDashboard(config: AppConfig(), messages: messages)
    defer { webView.configuration.userContentController.removeAllScriptMessageHandlers() }
    func layout(language: String) async throws -> [String: Any] {
        let result = try await webView.evaluateJavaScript("""
        (() => {
          navigateTo('settings'); setLanguage('\(language)');
          if (!document.getElementById('no-transitions')) {
            const style = document.createElement('style');
            style.id = 'no-transitions';
            style.textContent = '*, *::before, *::after { transition: none !important; }';
            document.head.appendChild(style);
          }
          const toggle = document.getElementById('cfg-m5-dflash2-profile');
          const row = toggle.closest('.settings-row');
          const card = row.closest('.card-settings');
          const rows = [...card.querySelectorAll(':scope > .settings-row')];
          const wrapper = row.querySelector('.profile-help');
          const trigger = row.querySelector('.profile-help-trigger');
          const tooltip = document.getElementById(trigger.getAttribute('aria-describedby'));
          const before = getComputedStyle(tooltip).opacity;
          trigger.scrollIntoView({block: 'center'});
          trigger.focus();
          const rect = tooltip.getBoundingClientRect();
          const triggerRect = trigger.getBoundingClientRect();
          const out = {
            cardKey: card.querySelector('.card-header [data-i18n]').dataset.i18n,
            lastRowInCard: rows[rows.length - 1] === row,
            inCacheCard: !!toggle.closest('.card-settings').querySelector('#cfg-cache-enable'),
            label: row.querySelector('[data-i18n="m5_dflash2_profile"]').textContent,
            desc: row.querySelector('.settings-row-desc').textContent,
            triggerText: trigger.textContent,
            triggerWarning: trigger.classList.contains('warning'),
            fixed: wrapper.classList.contains('profile-help-fixed'),
            ariaLabel: trigger.getAttribute('aria-label'),
            role: tooltip.getAttribute('role'),
            title: tooltip.querySelector('.profile-help-tooltip-title').textContent,
            body: tooltip.querySelector('.profile-help-tooltip-body').textContent,
            hiddenOpacity: before,
            focusOpen: wrapper.classList.contains('is-open'),
            shownOpacity: getComputedStyle(tooltip).opacity,
            insideViewport: rect.top >= 0 && rect.left >= 0 && rect.bottom <= window.innerHeight
              && rect.right <= window.innerWidth && rect.height > 0,
            besideTrigger: rect.top >= triggerRect.bottom || rect.bottom <= triggerRect.top,
            sameStyleAsHotCache: getComputedStyle(trigger).width === getComputedStyle(
              document.querySelector('[aria-describedby="hot-cache-help-tooltip"]')).width
          };
          return JSON.stringify(out);
        })()
        """) as? String
        return try #require(result.flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] })
    }
    let en = try await layout(language: "en")
    #expect(en["cardKey"] as? String == "advanced")
    #expect(en["lastRowInCard"] as? Bool == true)
    #expect(en["inCacheCard"] as? Bool == false)
    #expect(en["label"] as? String == "M5 DFlash2 Profile")
    #expect(en["desc"] as? String == "Automatically optimizes DFlash2 inference on supported devices.")
    #expect(en["triggerText"] as? String == "?")
    #expect(en["triggerWarning"] as? Bool == false)
    #expect(en["fixed"] as? Bool == true)
    #expect(en["ariaLabel"] as? String == "M5 DFlash2 Profile help")
    #expect(en["role"] as? String == "tooltip")
    #expect(en["title"] as? String == "About the M5 DFlash2 Profile")
    #expect(en["body"] as? String == "On Apple M5 (GPU family 17) and newer, DFlash2 serving uses the tuned M5 kernels, tree drafting and batching under load. Each feature falls back automatically when it does not apply. Turn off to use the generic settings. Qwen 3.6 35B A3B currently uses the generic path.")
    #expect(en["hiddenOpacity"] as? String == "0")
    #expect(en["focusOpen"] as? Bool == true)
    #expect(en["shownOpacity"] as? String == "1")
    #expect(en["insideViewport"] as? Bool == true)
    #expect(en["besideTrigger"] as? Bool == true)
    #expect(en["sameStyleAsHotCache"] as? Bool == true)
    try await snapshot(webView, named: "settings-m5-advanced-help-en")

    let zh = try await layout(language: "zh")
    #expect(zh["label"] as? String == "M5 DFlash2 优化配置")
    #expect(zh["desc"] as? String == "在支持的设备上自动优化 DFlash2 推理。")
    #expect(zh["ariaLabel"] as? String == "M5 DFlash2 优化配置说明")
    #expect(zh["title"] as? String == "关于 M5 DFlash2 优化配置")
    #expect(zh["body"] as? String == "在 Apple M5（GPU 第 17 代）及更新机型上，DFlash2 服务启用针对 M5 调优的内核、树式草稿和负载下的合批；各项不适用时自动回退。关闭后使用通用设置。Qwen 3.6 35B A3B 目前使用通用路径。")
    try await snapshot(webView, named: "settings-m5-advanced-help-zh")

    // Hover opens the same tooltip; Escape closes it.
    let hover = try await webView.evaluateJavaScript("""
    (() => {
      const trigger = document.getElementById('m5-dflash2-profile-help-trigger');
      const wrapper = trigger.closest('.profile-help');
      trigger.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
      const closed = !wrapper.classList.contains('is-open');
      trigger.dispatchEvent(new MouseEvent('mouseenter'));
      const opened = wrapper.classList.contains('is-open');
      trigger.dispatchEvent(new MouseEvent('mouseleave'));
      return JSON.stringify({closed, opened, closedAfterLeave: !wrapper.classList.contains('is-open')});
    })()
    """) as? String
    let state = try #require(hover.flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] })
    #expect(state["closed"] as? Bool == true)
    #expect(state["opened"] as? Bool == true)
    #expect(state["closedAfterLeave"] as? Bool == true)

    // Every dashboard language has the new keys.
    let missing = try await webView.evaluateJavaScript("""
    JSON.stringify(Object.entries(I18N).flatMap(([lang, dict]) =>
      ['m5_dflash2_profile', 'm5_dflash2_profile_desc', 'm5_dflash2_profile_help_title',
       'm5_dflash2_profile_help_body', 'm5_dflash2_profile_help_label']
        .filter(key => !dict[key]).map(key => lang + ':' + key)))
    """) as? String
    #expect(missing == "[]")
}

@MainActor
@Test func dashboardM5ProfileToggleReflectsSavedOffSetting() async throws {
    let messages = CapturedSettingsMessages()
    let webView = try await loadedDashboard(config: AppConfig(m5Dflash2Profile: false), messages: messages)
    defer { webView.configuration.userContentController.removeAllScriptMessageHandlers() }
    let checked = try await webView.evaluateJavaScript(
        "navigateTo('settings'); document.getElementById('cfg-m5-dflash2-profile').checked") as? Bool
    #expect(checked == false)
}

@MainActor
@Test func dashboardRuntimeStatusShowsM5ProfileStatus() async throws {
    let messages = CapturedSettingsMessages()
    let webView = try await loadedDashboard(config: AppConfig(), messages: messages)
    defer { webView.configuration.userContentController.removeAllScriptMessageHandlers() }
    func meta(status: String, language: String) async throws -> String? {
        try await webView.evaluateJavaScript("""
        (() => {
          setLanguage('\(language)'); navigateTo('status');
          renderRuntimeModels([{id: 'benchmark', model: 'benchmark', loaded: true,
            dflash2_draft_model: 'z-lab/Qwen3.8-27B-DFlash2',
            dflash2: {enabled: true, block_size: 8, draft_quantization_bits: 4, windows: 97,
                      rollback_count: 0, exact_residual_corrections: 0, latest_acceptance_rate: 0.6,
                      tree_max_nodes: 15,
                      m5_profile: {installed: true, mode: 'auto', status: '\(status)',
                                   architecture: 'applegpu_g17s'}}}]);
          const metas = [...document.querySelectorAll('#runtime-model-health .runtime-section-meta')];
          const meta = metas.find(m => m.textContent.includes('Block 8'));
          return meta ? meta.textContent : null;
        })()
        """) as? String
    }
    let active = try await meta(status: "active", language: "en")
    #expect(active?.contains("M5 profile active") == true)
    try await snapshot(webView, named: "runtime-m5-active-en")
    let disabled = try await meta(status: "disabled", language: "en")
    #expect(disabled?.contains("M5 profile off") == true)
    let activeZh = try await meta(status: "active", language: "zh")
    #expect(activeZh?.contains("M5 配置：已启用") == true)
    try await snapshot(webView, named: "runtime-m5-active-zh")
    let unsupported = try await meta(status: "unsupported_gpu", language: "zh")
    #expect(unsupported?.contains("GPU 不支持") == true)
}

@MainActor
@Test func dashboardM5ProfileSettingRestartsBackendAndRelaunchesDFlash2WithFlag() throws {
    let existing = AppConfig()
    let updated = try DashboardBridge.config(applyingSettingsJSON: #"{"m5_dflash2_profile":false}"#, to: existing)
    #expect(updated.m5Dflash2Profile == false)
    #expect(DashboardBridge.backendRestartRequired(from: existing, to: updated))
    let unchanged = try DashboardBridge.config(applyingSettingsJSON: #"{"verify_model_on_load":true}"#, to: updated)
    #expect(unchanged.m5Dflash2Profile == false)
    #expect(!DashboardBridge.backendRestartRequired(from: updated, to: unchanged))

    // Switching to a DFlash2 model relaunches the backend from the saved
    // settings; the relaunch carries the profile choice.
    let runtime = ModelDFlash2Runtime(
        targetModelID: "mlx-community/Qwen3.8-27B-4bit",
        targetModelDir: "/models/target",
        draftModelID: "z-lab/Qwen3.8-27B-DFlash2",
        draftModelDir: "/models/draft",
        checkpointBlockSize: 8,
        blockSize: 8,
        draftBits: 4,
        tensorBatchMaxWidth: nil,
        maxCacheCap: 32_768
    )
    func arguments(_ config: AppConfig) -> [String] {
        BackendLaunchConfiguration(
            executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
            host: "127.0.0.1",
            port: config.port,
            options: BackendLaunchOptions(config: config),
            dflash2Runtime: runtime
        ).arguments
    }
    let off = arguments(updated)
    let offIndex = try #require(off.firstIndex(of: "--m5-dflash2-profile"))
    #expect(off[offIndex + 1] == "off")
    #expect(!arguments(existing).contains("--m5-dflash2-profile"))
}

@MainActor
@Test func dashboardRuntimeStatusShowsEmbeddingPerformance() async throws {
    let messages = CapturedSettingsMessages()
    let webView = try await loadedDashboard(config: AppConfig(), messages: messages)
    defer { webView.configuration.userContentController.removeAllScriptMessageHandlers() }
    for language in ["en", "zh"] {
        let result = try await webView.evaluateJavaScript("""
        (() => {
          setLanguage('\(language)'); navigateTo('status');
          renderRuntimeModels([{id: 'mlx-community/embeddinggemma-2-4bit', runtime_kind: 'embedding', supports_audio: true,
            active_requests: 0, queued_requests: 0, queue_capacity: 8,
            usage: {cumulative_tokens: 1200},
            embedding_metrics: {window_seconds: 60, completed_requests: 7, failed_requests: 1,
              recent_completed_requests: 4, latency_ms_p50: 18.5,
              input_tokens_per_second: 1200, vectors_per_second: 41.2}}]);
          const card = document.getElementById('runtime-model-health');
          const dict = I18N[currentLang];
          return JSON.stringify({
            visible: document.getElementById('runtime-model-health-card').style.display !== 'none',
            audio: card.textContent.includes(dict.catalog_audio_embedding),
            labels: ['performance', 'completed', 'errors', 'latency', 'input_rate', 'vector_rate']
              .every(key => card.textContent.includes(dict['runtime_embedding_' + key])),
            generation: ['runtime_live_decode_rate', 'runtime_prefill_rate', 'runtime_ttft']
              .some(key => card.textContent.includes(dict[key])),
            values: card.textContent.includes('18.5') && card.textContent.includes('41.2')
          });
        })()
        """) as? String
        let data = try #require(result?.data(using: .utf8))
        let flags = try #require(JSONSerialization.jsonObject(with: data) as? [String: Bool])
        #expect(flags["visible"] == true)
        #expect(flags["audio"] == true)
        #expect(flags["labels"] == true)
        #expect(flags["generation"] == false)
        #expect(flags["values"] == true)
        try await snapshot(webView, named: "runtime-embedding-\(language)")
    }
}
