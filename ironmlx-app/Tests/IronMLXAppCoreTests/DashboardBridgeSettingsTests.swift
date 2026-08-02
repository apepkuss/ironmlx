import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
@Test func dashboardSettingsPayloadParsesOneAsMaxSequencesInteger() throws {
    let existing = AppConfig(
        host: "127.0.0.1",
        port: 9068,
        language: "zh-Hans",
        maxSequences: 16
    )
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "language": "zh-Hans",
      "max_sequences": 1
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.maxSequences == 1)
}

@MainActor
@Test func dashboardSettingsPayloadAllowsCacheAndKVQuantTogether() throws {
    let existing = AppConfig(cacheEnable: true, kvQuant: "none")
    let json = """
    {
      "cache_enable": true,
      "cache_dir": "/tmp/cache",
      "kv_quant": "k3v4"
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.cacheEnable == true)
    #expect(config.cacheDir == "/tmp/cache")
    #expect(config.kvQuant == "k3v4")
}

@MainActor
@Test func dashboardSettingsPayloadParsesActiveKVOffloadToggle() throws {
    let existing = AppConfig(activeKvOffload: false)
    let json = """
    {
      "active_kv_offload": true
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.activeKvOffload == true)
}

@MainActor
@Test func dashboardSettingsPayloadParsesLoadIntegrityToggleWithoutRequiringRestart() throws {
    let existing = AppConfig(verifyModelOnLoad: false)
    let config = try DashboardBridge.config(
        applyingSettingsJSON: #"{"verify_model_on_load":true}"#,
        to: existing
    )

    #expect(config.verifyModelOnLoad == true)
    #expect(!DashboardBridge.backendRestartRequired(from: existing, to: config))
}

@MainActor
@Test func backendRuntimeSettingStillRequiresRestart() {
    let existing = AppConfig(maxSequences: 1, verifyModelOnLoad: false)
    let updated = AppConfig(maxSequences: 2, verifyModelOnLoad: true)

    #expect(DashboardBridge.backendRestartRequired(from: existing, to: updated))
}

@MainActor
@Test func dashboardSettingsPayloadParsesMemoryLimits() throws {
    let existing = AppConfig(
        memLimitTotal: nil,
        memLimitModel: nil,
        memTotalAuto: true,
        memTotal: 24,
        memModelAuto: true,
        memModel: 22
    )
    let json = """
    {
      "mem_limit_total": 64,
      "mem_limit_model": 40,
      "mem_total_auto": false,
      "mem_total": 64,
      "mem_model_auto": false,
      "mem_model": 40
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.memLimitTotal == 64)
    #expect(config.memLimitModel == 40)
    #expect(config.memTotalAuto == false)
    #expect(config.memTotal == 64)
    #expect(config.memModelAuto == false)
    #expect(config.memModel == 40)
}

@Test func dashboardLoadDoesNotMakeSecondModelDefaultUnlessUserSelectedIt() {
    let config = AppConfig(
        defaultModel: "mlx-community/Existing-4bit",
        loadedModels: ["mlx-community/Existing-4bit"]
    )

    let shouldSetDefault = DashboardBridge.shouldSetDefaultWhenLoadingModel(
        "mlx-community/New-4bit",
        config: config,
        currentLoadedModelCount: 1
    )

    #expect(shouldSetDefault == false)
}

@Test func dashboardLoadMakesModelDefaultWhenItIsTheOnlyLoadedModelOrExplicitDefault() {
    let firstModelConfig = AppConfig()
    let explicitDefaultConfig = AppConfig(
        defaultModel: "mlx-community/New-4bit",
        loadedModels: ["mlx-community/Existing-4bit"]
    )

    #expect(DashboardBridge.shouldSetDefaultWhenLoadingModel(
        "mlx-community/New-4bit",
        config: firstModelConfig,
        currentLoadedModelCount: 0
    ))
    #expect(DashboardBridge.shouldSetDefaultWhenLoadingModel(
        "mlx-community/New-4bit",
        config: explicitDefaultConfig,
        currentLoadedModelCount: 1
    ))
}

@MainActor
@Test func dashboardBridgeRefreshesModelListWhenLoadedModelsNotificationArrives() async throws {
    let root = try dashboardBridgeNotificationModelRoot(repoID: "mlx-community/Tiny-4bit")
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig(loadedModels: ["mlx-community/Tiny-4bit"]))
    let webView = CapturingDashboardWebView()
    let notificationCenter = NotificationCenter()
    let backend = TestRuntimeBackend()
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: backend,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        notificationCenter: notificationCenter
    )

    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: bridge)
    #expect(await webView.waitForScript(containing: "onLocalModelsScanned") == false)

    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: nil)

    #expect(await webView.waitForScript(containing: "onLocalModelsScanned"))
}

@MainActor
@Test func dashboardBridgePublishesStructuredBackendRuntimeEventImmediately() async throws {
    let root = try dashboardBridgeNotificationModelRoot(repoID: "mlx-community/Tiny-4bit")
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let webView = CapturingDashboardWebView()
    let notificationCenter = NotificationCenter()
    let backend = TestRuntimeBackend(state: .failed)
    backend.lastEvent = BackendRuntimeEvent(
        phase: .breaker,
        runtimeState: .failed,
        incidentID: UUID(),
        launchID: UUID(),
        pid: 1234,
        terminationStatus: 9,
        terminationReason: "uncaught_signal",
        recoveryAttempt: 1,
        detail: "Crash-loop breaker stopped automatic recovery.",
        logTail: "backend log tail",
        canRetry: true,
        processHealthy: false
    )
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: backend,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        notificationCenter: notificationCenter
    )

    notificationCenter.post(
        name: .ironMLXBackendRuntimeDidChange,
        object: NSObject()
    )
    #expect(await webView.waitForScript(containing: "onServerCrash") == false)

    notificationCenter.post(
        name: .ironMLXBackendRuntimeDidChange,
        object: backend
    )
    let script = try #require(await webView.script(containing: "onServerCrash"))
    #expect(script.contains("breaker"))
    #expect(script.contains(#"\"termination_status\":9"#))
    #expect(script.contains(#"\"can_retry\":true"#))
    #expect(script.contains("backend log tail"))
    withExtendedLifetime(bridge) {}
}

@MainActor
@Test func dashboardScannedModelsExposeEffectiveMaxTokensUsedForLoad() async throws {
    let root = try dashboardBridgeNotificationModelRoot(
        repoID: "mlx-community/LongContext-4bit",
        configJSON: #"{"max_position_embeddings":262144}"#
    )
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig(activeKvOffload: false))
    let webView = CapturingDashboardWebView()
    let notificationCenter = NotificationCenter()
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: TestRuntimeBackend(),
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        notificationCenter: notificationCenter
    )

    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: nil)

    let script = try #require(await webView.script(containing: "onLocalModelsScanned"))
    let payload = try decodedJavaScriptStringArgument(from: script, functionName: "onLocalModelsScanned")
    let data = try #require(payload.data(using: .utf8))
    let models = try #require(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    let model = try #require(models.first)
    #expect(model["effective_max_tokens"] as? Int == 32768)
    withExtendedLifetime(bridge) {}
}

@MainActor
@Test func dashboardScannedModelsExposeQuantizationAndReadiness() async throws {
    let root = try dashboardBridgeNotificationModelRoot(
        repoID: "mlx-community/Affine-6bit",
        configJSON: #"{"quantization":{"group_size":64,"bits":6,"mode":"affine"}}"#
    )
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    let webView = CapturingDashboardWebView()
    let notificationCenter = NotificationCenter()
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: TestRuntimeBackend(),
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        notificationCenter: notificationCenter
    )

    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: nil)

    let script = try #require(await webView.script(containing: "onLocalModelsScanned"))
    let payload = try decodedJavaScriptStringArgument(from: script, functionName: "onLocalModelsScanned")
    let data = try #require(payload.data(using: .utf8))
    let models = try #require(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    let model = try #require(models.first)
    let quantization = try #require(model["quantization"] as? [String: Any])
    let readiness = try #require(model["readiness"] as? [String: Any])

    #expect(quantization["kind"] as? String == "affine")
    #expect(quantization["bits"] as? Int == 6)
    #expect(quantization["label"] as? String == "affine 6-bit")
    #expect(readiness["status"] as? String == "ready")
    withExtendedLifetime(bridge) {}
}

@Test func dashboardModelParamsUseLanguageInvariantCapacityLabels() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"<label data-i18n="context_size">CONTEXT SIZE</label>"#))
    #expect(html.contains(#"context_size: "CONTEXT SIZE""#))
    #expect(!html.contains("上下文大小"))
    #expect(!html.contains("コンテキストサイズ"))
    #expect(!html.contains("컨텍스트 크기"))
    #expect(html.contains(#"<label data-i18n="max_context_tokens">MAX CONTEXT TOKENS</label>"#))
    #expect(html.contains(#"max_context_tokens: "MAX CONTEXT TOKENS""#))
    #expect(!html.contains("single_request_max_tokens"))
    #expect(!html.contains("单请求最大 Token 数"))
    #expect(!html.contains("單請求最大 Token 數"))
    #expect(!html.contains("リクエスト最大 Token 数"))
    #expect(!html.contains("단일 요청 최대 Token 수"))
    #expect(!html.contains(">MAX TOKENS</label>"))
}

@Test func dashboardModelParamsKeepCapacityInputsAligned() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"<div class="modal model-params-modal">"#))
    #expect(
        html.contains(
            "<div class=\"modal-row model-params-capacity-row\">\n"
                + "        <div class=\"modal-field\"><label data-i18n=\"context_size\">"
        )
    )
    #expect(html.contains(".model-params-modal {\n    width: 480px;\n    max-width: calc(100vw - 32px);"))
    #expect(html.contains(".model-params-capacity-row .modal-field > label {\n    white-space: nowrap;"))
}

@Test func dashboardExposesCrossRequestPromptLookupControlsAndClearAction() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"id="modal-prompt-lookup-enabled""#))
    #expect(html.contains(#"id="modal-prompt-lookup-cross-request""#))
    #expect(html.contains(#"onclick="clearSharedPromptLookup()""#))
    #expect(html.contains(#"path === '/admin/api/prompt-lookup/clear'"#))
    #expect(html.contains("cross_request_prompt_lookup"))
    #expect(html.contains("prompt_lookup_cleared"))
}

@Test func dashboardPromptLookupControlsUseLocalizedAccessibleHelp() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"class="prompt-lookup-label""#))
    #expect(html.contains("font-weight: 400;"))
    #expect(html.contains(#"data-i18n="prompt_lookup_acceleration_help""#))
    #expect(html.contains(#"data-i18n="prompt_lookup_cross_request_help""#))
    #expect(html.contains(#"data-i18n-aria-label="prompt_lookup_acceleration_help_label""#))
    #expect(html.contains(#"data-i18n-aria-label="prompt_lookup_cross_request_help_label""#))
    #expect(html.contains("document.querySelectorAll('[data-i18n-aria-label]')"))
    #expect(html.contains("trigger.addEventListener('mouseenter', showTooltip)"))
    #expect(html.contains("trigger.addEventListener('focus', showTooltip)"))
    #expect(html.contains("trigger.addEventListener('click', showTooltip)"))

    let localizedKeys = [
        "prompt_lookup_acceleration_help_label",
        "prompt_lookup_acceleration_help",
        "prompt_lookup_cross_request_help_label",
        "prompt_lookup_cross_request_help",
        "prompt_lookup_clear_failed",
    ]
    for key in localizedKeys {
        #expect(
            html.components(separatedBy: "\(key):").count - 1 == 5,
            "\(key) must be present in all five locale dictionaries"
        )
    }

    for localizedLabel in [
        "Repeated-Text Acceleration",
        "重复文本加速",
        "重複文字加速",
        "繰り返しテキストの高速化",
        "반복 텍스트 가속",
        "Reuse Across Requests",
        "跨请求复用",
        "跨請求重用",
        "リクエスト間で再利用",
        "요청 간 재사용",
    ] {
        #expect(html.contains(localizedLabel), "Missing localized label: \(localizedLabel)")
    }

    for localizedDependency in [
        "Requires Repeated-Text Acceleration to be enabled.",
        "需先启用“重复文本加速”。",
        "需先啟用「重複文字加速」。",
        "先に「繰り返しテキストの高速化」を有効にする必要があります。",
        "먼저 ‘반복 텍스트 가속’을 활성화해야 합니다.",
    ] {
        #expect(
            html.contains(localizedDependency),
            "Missing localized PromptLookup dependency: \(localizedDependency)"
        )
    }

    #expect(!html.contains("PromptLookup Acceleration"))
    #expect(!html.contains("在 App 请求之间复用已完成的历史"))
}

@Test func dashboardMtpControlUsesLocalizedAccessibleHelpAndKeepsFailureStatusVisible() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"data-i18n-aria-label="mtp_help_label""#))
    #expect(html.contains(#"data-i18n="mtp_help_title""#))
    #expect(html.contains(#"data-i18n="mtp_help_body""#))
    #expect(html.contains(#"data-i18n="mtp_help_requirement""#))
    #expect(html.contains(#"data-i18n="mtp_help_draft_tokens""#))
    #expect(html.contains(#"id="modal-mtp-help-status""#))
    #expect(html.contains("status.hidden = hasCandidates;"))
    #expect(html.contains("status.dataset.i18n = statusKey;"))
    #expect(html.contains("helpStatus.dataset.i18n = helpStatusKey;"))

    let localizedKeys = [
        "mtp_help_label",
        "mtp_help_title",
        "mtp_help_body",
        "mtp_help_requirement",
        "mtp_help_draft_tokens",
        "mtp_help_status_available",
        "mtp_help_status_incompatible",
        "mtp_help_status_unavailable",
    ]
    for key in localizedKeys {
        #expect(
            html.components(separatedBy: "\(key):").count - 1 == 5,
            "\(key) must be present in all five locale dictionaries"
        )
    }
}

@Test func dashboardOnboardingOffersAVisiblePersistentLanguageSelector() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"class="onboarding-language""#))
    #expect(
        html.contains(
            #"class="visually-hidden" for="onboarding-lang-select" data-i18n="interface_language""#
        )
    )
    #expect(html.contains(#"id="onboarding-lang-select""#))
    #expect(html.contains(#"<option value="en">English</option>"#))
    #expect(html.contains(#"<option value="zh-Hans">简体中文</option>"#))
    #expect(html.contains(#"<option value="zh-Hant">繁體中文</option>"#))
    #expect(html.contains(#"<option value="ja">日本語</option>"#))
    #expect(html.contains(#"<option value="ko">한국어</option>"#))
    #expect(html.contains("['lang-select', 'onboarding-lang-select'].forEach"))
    #expect(html.contains("selectInterfaceLanguage(this.value);"))
    #expect(html.contains("window.webkit.messageHandlers.setLanguage.postMessage(value);"))
    #expect(html.contains(".visually-hidden {"))
    #expect(html.contains(".onboarding-language {\n    position: absolute;"))
    #expect(html.contains("gap: 6px;"))
    #expect(html.contains(".onboarding-language select:focus"))
}

private func dashboardBridgeNotificationModelRoot(repoID: String) throws -> URL {
    try dashboardBridgeNotificationModelRoot(repoID: repoID, configJSON: "{}")
}

private func dashboardBridgeNotificationModelRoot(repoID: String, configJSON: String) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-dashboard-bridge-notification-\(UUID().uuidString)", isDirectory: true)
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: repoID,
        files: [
            "config.json": Data(configJSON.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    return root
}

private func decodedJavaScriptStringArgument(from script: String, functionName: String) throws -> String {
    let prefix = "\(functionName)("
    guard script.hasPrefix(prefix), script.hasSuffix(")") else {
        throw CocoaError(.coderInvalidValue)
    }
    let literal = String(script.dropFirst(prefix.count).dropLast())
    return try JSONDecoder().decode(String.self, from: Data(literal.utf8))
}

@MainActor
private final class CapturingDashboardWebView: WKWebView {
    private var scripts: [String] = []

    override func evaluateJavaScript(
        _ javaScriptString: String,
        completionHandler: (@MainActor @Sendable (Any?, (any Error)?) -> Void)? = nil
    ) {
        scripts.append(javaScriptString)
        completionHandler?(nil, nil)
    }

    func waitForScript(containing needle: String, timeoutSeconds: TimeInterval = 0.4) async -> Bool {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if scripts.contains(where: { $0.contains(needle) }) {
                return true
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return false
    }

    func script(containing needle: String, timeoutSeconds: TimeInterval = 0.4) async -> String? {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if let script = scripts.first(where: { $0.contains(needle) }) {
                return script
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return nil
    }
}
