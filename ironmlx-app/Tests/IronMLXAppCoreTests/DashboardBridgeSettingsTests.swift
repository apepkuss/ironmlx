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
@Test func dashboardSettingsPayloadNormalizesThemePreference() throws {
    let existing = AppConfig(theme: "dark")

    let light = try DashboardBridge.config(
        applyingSettingsJSON: #"{"theme":"light"}"#,
        to: existing
    )
    let system = try DashboardBridge.config(
        applyingSettingsJSON: #"{"theme":"system"}"#,
        to: existing
    )
    let unknown = try DashboardBridge.config(
        applyingSettingsJSON: #"{"theme":"unknown"}"#,
        to: existing
    )

    #expect(light.theme == "light")
    #expect(system.theme == nil)
    #expect(unknown.theme == nil)
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
@Test func dashboardModelOperationErrorPreservesStructuredFailureCode() throws {
    let json = DashboardBridge.modelOperationErrorJSON(
        error: "Backend did not become healthy.",
        code: BackendRuntimeFailureCode.backendReadinessFailed.rawValue
    )
    let data = try #require(json.data(using: .utf8))
    let payload = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(payload["success"] as? Bool == false)
    #expect(payload["status"] as? String == "error")
    #expect(payload["code"] as? String == "backend_readiness_failed")
    #expect(payload["error"] as? String == "Backend did not become healthy.")
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
    #expect(
        await webView.waitForScript(
            containing: "onLocalModelsScanned",
            timeoutSeconds: 0.4
        ) == false
    )

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
    #expect(
        await webView.waitForScript(
            containing: "onServerCrash",
            timeoutSeconds: 0.4
        ) == false
    )

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

@MainActor
@Test func dashboardKeepsUnloadedDFlash2PreferenceAvailableForLoad() async throws {
    let targetID = "mlx-community/Qwen3.8-27B-4bit"
    let draftID = "z-lab/Qwen3.8-27B-DFlash2"
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-dashboard-dflash2-\(UUID().uuidString)", isDirectory: true)
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: targetID,
        files: [
            "config.json": Data(dashboardDFlash2TargetConfig.utf8),
            "model.safetensors": Data("target-weights".utf8),
        ]
    )
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: draftID,
        files: [
            "config.json": Data(dashboardDFlash2DraftConfig.utf8),
            "model.safetensors": Data("draft-weights".utf8),
        ]
    )
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig(defaultModel: targetID, loadedModels: []))
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try parameterStore.save(ModelParameters(
        modelID: targetID,
        dflash2Enabled: true,
        dflash2ModelID: draftID
    ))
    let webView = CapturingDashboardWebView()
    let notificationCenter = NotificationCenter()
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: TestRuntimeBackend(),
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: parameterStore,
        notificationCenter: notificationCenter
    )

    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: nil)

    let script = try #require(await webView.script(containing: "onLocalModelsScanned"))
    let payload = try decodedJavaScriptStringArgument(from: script, functionName: "onLocalModelsScanned")
    let data = try #require(payload.data(using: .utf8))
    let models = try #require(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    let target = try #require(models.first(where: { $0["id"] as? String == targetID }))
    let dflash2 = try #require(target["dflash2"] as? [String: Any])
    let candidates = try #require(dflash2["candidates"] as? [[String: Any]])

    #expect(target["loaded"] as? Bool == false)
    #expect(dflash2["status"] as? String == "available")
    #expect(dflash2["enabled"] as? Bool == false)
    #expect(candidates.map { $0["id"] as? String } == [draftID])
    withExtendedLifetime(bridge) {}
}

@MainActor
@Test func dashboardTreatsConfiguredDFlash2TargetAsLoadedDuringActorTransition() async throws {
    let targetID = "mlx-community/Qwen3.8-27B-4bit"
    let draftID = "z-lab/Qwen3.8-27B-DFlash2"
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-dashboard-dflash2-running-\(UUID().uuidString)", isDirectory: true)
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: targetID,
        files: [
            "config.json": Data(dashboardDFlash2TargetConfig.utf8),
            "model.safetensors": Data("target-weights".utf8),
        ]
    )
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: draftID,
        files: [
            "config.json": Data(dashboardDFlash2DraftConfig.utf8),
            "model.safetensors": Data("draft-weights".utf8),
        ]
    )
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig(defaultModel: targetID, loadedModels: [targetID]))
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try parameterStore.save(ModelParameters(
        modelID: targetID,
        dflash2Enabled: true,
        dflash2ModelID: draftID
    ))
    let webView = CapturingDashboardWebView()
    let notificationCenter = NotificationCenter()
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: TestRuntimeBackend(state: .running, isRunning: true),
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: parameterStore,
        notificationCenter: notificationCenter
    )

    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: nil)

    let script = try #require(await webView.script(containing: "onLocalModelsScanned"))
    let payload = try decodedJavaScriptStringArgument(from: script, functionName: "onLocalModelsScanned")
    let data = try #require(payload.data(using: .utf8))
    let models = try #require(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    let target = try #require(models.first(where: { $0["id"] as? String == targetID }))
    let dflash2 = try #require(target["dflash2"] as? [String: Any])

    #expect(target["loaded"] as? Bool == true)
    #expect(dflash2["enabled"] as? Bool == true)
    #expect(configStore.load().restoredModelReferences == [targetID])
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

@Test func dashboardConfigurationRecoveryCodesAreLocalizedWithoutNewVisibleUI() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    for key in [
        "err_configuration_recovery_required",
        "err_configuration_migration_failed",
        "err_configuration_version_unsupported",
        "err_configuration_lkg_unavailable",
    ] {
        #expect(
            html.components(separatedBy: "\(key):").count - 1 == 5,
            "\(key) must be present in all five locale dictionaries"
        )
    }
    #expect(html.contains("function onModelParamsSaved(jsonStr)"))
    #expect(html.contains("showToast(localizeErrorResult(result), 'warn')"))
    #expect(!html.contains("id=\"configuration-schema"))
    #expect(!html.contains("id=\"configuration-lkg"))
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

private let dashboardDFlash2TargetConfig = """
{
  "model_type": "qwen3_5",
  "text_config": {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 0.000001,
    "rope_parameters": {
      "rope_type": "default",
      "rope_theta": 10000000
    }
  }
}
"""

private let dashboardDFlash2DraftConfig = """
{
  "architectures": ["DFlash2DraftModel"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_act": "silu",
  "attention_bias": false,
  "is_causal": false,
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "vocab_size": 248320,
  "max_position_embeddings": 262144,
  "head_dim": 128,
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "rms_norm_eps": 0.000001,
  "rope_parameters": {
    "rope_type": "default",
    "rope_theta": 10000000
  },
  "sliding_window": 2048,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61]
  }
}
"""

@MainActor
private final class CapturingDashboardWebView: WKWebView {
    private struct ScriptWaiter {
        let needle: String
        let continuation: CheckedContinuation<String?, Never>
        let timeoutTask: Task<Void, Never>
    }

    private var scripts: [String] = []
    private var scriptWaiters: [UUID: ScriptWaiter] = [:]

    override func evaluateJavaScript(
        _ javaScriptString: String,
        completionHandler: (@MainActor @Sendable (Any?, (any Error)?) -> Void)? = nil
    ) {
        scripts.append(javaScriptString)
        let matchingWaiterIDs = scriptWaiters.compactMap { id, waiter in
            javaScriptString.contains(waiter.needle) ? id : nil
        }
        for id in matchingWaiterIDs {
            resolveScriptWaiter(id: id, script: javaScriptString)
        }
        completionHandler?(nil, nil)
    }

    func waitForScript(containing needle: String, timeoutSeconds: TimeInterval = 5) async -> Bool {
        await script(containing: needle, timeoutSeconds: timeoutSeconds) != nil
    }

    func script(containing needle: String, timeoutSeconds: TimeInterval = 5) async -> String? {
        if let script = scripts.first(where: { $0.contains(needle) }) {
            return script
        }

        let waiterID = UUID()
        return await withCheckedContinuation { continuation in
            let timeoutNanoseconds = UInt64(max(0, timeoutSeconds) * 1_000_000_000)
            let timeoutTask = Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: timeoutNanoseconds)
                guard !Task.isCancelled else { return }
                self?.resolveScriptWaiter(id: waiterID, script: nil)
            }
            scriptWaiters[waiterID] = ScriptWaiter(
                needle: needle,
                continuation: continuation,
                timeoutTask: timeoutTask
            )
        }
    }

    private func resolveScriptWaiter(id: UUID, script: String?) {
        guard let waiter = scriptWaiters.removeValue(forKey: id) else { return }
        waiter.timeoutTask.cancel()
        waiter.continuation.resume(returning: script)
    }
}
