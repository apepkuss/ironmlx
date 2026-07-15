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
    let backend = BackendProcessManager(
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root)
    )
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
        backend: BackendProcessManager(
            configStore: configStore,
            scanner: LocalModelScanner(rootURL: root)
        ),
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
        backend: BackendProcessManager(
            configStore: configStore,
            scanner: LocalModelScanner(rootURL: root)
        ),
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
    #expect(html.contains(#"<label data-i18n="single_request_max_tokens">MAX TOKENS</label>"#))
    #expect(html.contains(#"single_request_max_tokens: "MAX TOKENS""#))
    #expect(!html.contains("单请求最大 Token 数"))
    #expect(!html.contains("單請求最大 Token 數"))
    #expect(!html.contains("リクエスト最大 Token 数"))
    #expect(!html.contains("단일 요청 최대 Token 수"))
}

private func dashboardBridgeNotificationModelRoot(repoID: String) throws -> URL {
    try dashboardBridgeNotificationModelRoot(repoID: repoID, configJSON: "{}")
}

private func dashboardBridgeNotificationModelRoot(repoID: String, configJSON: String) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-dashboard-bridge-notification-\(UUID().uuidString)", isDirectory: true)
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(configJSON.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
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
