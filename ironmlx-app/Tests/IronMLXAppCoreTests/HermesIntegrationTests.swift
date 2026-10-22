import AppKit
import Foundation
import Testing
import WebKit
import Yams
@testable import IronMLXAppCore

@MainActor
private struct HermesFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("ironmlx-hermes-test-\(UUID().uuidString)")
    var home: URL { root.appendingPathComponent(".hermes") }
    var config: URL { home.appendingPathComponent("profiles/ironmlx/config.yaml") }
    func service(_ version: String = "Hermes Agent v0.20.0 (2026-05-01)\nInstall directory: /test/hermes") -> HermesIntegrationService {
        HermesIntegrationService(hermesHome: home, findExecutable: { URL(fileURLWithPath: "/test/hermes") },
                                 versionOutput: { _ in version }, desktopInstallations: { [] }, environment: [:])
    }
    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }
    func clean() { try? FileManager.default.removeItem(at: root) }
}

@MainActor
@Test func hermesInstallationAndVersionGatesNeverWriteFiles() async throws {
    let fixture = HermesFixture()
    defer { fixture.clean() }
    var service = fixture.service()
    service.findExecutable = { nil }
    #expect(await service.configure(modelID: "MODEL_ID", contextWindow: 0, port: 0).code == "not_installed")
    for (version, code) in [("Hermes Agent v0.19.9", "version_too_old"),
                            ("Hermes Agent v0.20.0-beta.1", "version_unknown"), ("other tool 1.0.0", "version_unknown")] {
        #expect(await fixture.service(version).configure(modelID: "test/model", contextWindow: 64000, port: 9068).code == code)
    }
    for version in ["Hermes Agent v0.20.0", "Hermes Agent v0.21.0+12.gabcdef · main", "Hermes Agent v1.0.0 · stable"] {
        #expect(await fixture.service(version).inspect().success)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.root.path))
}

@MainActor
@Test func hermesLocalDesktopWorksWithoutCLIAndRemoteClientDoesNotCount() async throws {
    let fixture = HermesFixture()
    defer { fixture.clean() }
    var service = fixture.service()
    service.findExecutable = { nil }
    service.desktopInstallations = { [.init(bundleIdentifier: "com.nousresearch.hermes-light", version: "1.0.0")] }
    #expect(await service.inspect().code == "not_installed")
    service.desktopInstallations = { [.init(bundleIdentifier: "com.nousresearch.hermes-bundled", version: "0.19.0")] }
    #expect(await service.inspect().code == "version_too_old")
    service.desktopInstallations = { [.init(bundleIdentifier: "com.nousresearch.hermes", version: "unknown")] }
    #expect(await service.inspect().code == "version_unknown")
    service.desktopInstallations = { [.init(bundleIdentifier: "com.nousresearch.hermes-bundled", version: "0.27.1")] }
    #expect(await service.configure(modelID: "test/model", contextWindow: 128000, port: 9068).success)
    service.findExecutable = { URL(fileURLWithPath: "/old/hermes") }
    service.versionOutput = { _ in "Hermes Agent v0.19.0" }
    #expect(await service.inspect().installedVersion == "0.27.1")
}

@MainActor
@Test func hermesDesktopDetectionRequiresOfficialIdentityAndBundledLocalRuntime() throws {
    let fixture = HermesFixture()
    defer { fixture.clean() }
    let app = fixture.root.appendingPathComponent("Hermes Agent.app")
    let contents = app.appendingPathComponent("Contents")
    let resources = contents.appendingPathComponent("Resources")
    try fixture.write("#!/bin/sh\nexit 0\n", to: contents.appendingPathComponent("MacOS/Hermes"))
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: contents.appendingPathComponent("MacOS/Hermes").path)
    let info: [String: String] = ["CFBundleIdentifier": "com.nousresearch.hermes-bundled", "CFBundleExecutable": "Hermes",
                                "CFBundleShortVersionString": "0.27.1", "CFBundlePackageType": "APPL"]
    let data = try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
    try data.write(to: contents.appendingPathComponent("Info.plist"))
    try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
    #expect(HermesIntegrationService.readDesktopInstallation(at: app)?.hasLocalRuntime == false)
    try fixture.write("{\"payload\":\"bundled\",\"baseVersion\":\"0.27.1\",\"runtime\":{\"repoDir\":\"repo\",\"commands\":{\"hermes\":\"bin/hermes\"}}}",
                      to: resources.appendingPathComponent("install-stamp.json"))
    try fixture.write("# bundled agent\n", to: resources.appendingPathComponent("agent-payload/repo/hermes_cli/main.py"))
    try fixture.write("#!/bin/sh\nexit 0\n", to: resources.appendingPathComponent("agent-payload/bin/hermes"))
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: resources.appendingPathComponent("agent-payload/bin/hermes").path)
    let installation = try #require(HermesIntegrationService.readDesktopInstallation(at: app))
    #expect(installation.hasLocalRuntime)
    #expect(installation.version == "0.27.1")
    #expect(HermesIntegrationService.readDesktopInstallation(at: fixture.root) == nil)
}

@MainActor
@Test func hermesConfigurationPreservesDefaultCredentialsAndOtherSettings() async throws {
    let fixture = HermesFixture()
    defer { fixture.clean() }
    let defaults = fixture.home.appendingPathComponent("config.yaml")
    let credentials = fixture.config.deletingLastPathComponent().appendingPathComponent(".env")
    try fixture.write("model: default-provider\n", to: defaults)
    try fixture.write("CUSTOM_API_KEY=existing-local-fixture\n", to: credentials)
    let original = """
    # Existing profile
    model:
      default: old-model
      base_url: https://old.example.com
      temperature: 0.2
    providers:
      other:
        api: https://other.example.com
      ironmlx-responses:
        base_url: https://old.example.com
        api_mode: chat_completions
        enabled: false
        extra_headers:
          X-Custom: keep
    terminal:
      cwd: /existing/workspace
    memory:
      provider: local
    """
    try fixture.write(original, to: fixture.config)
    let result = await fixture.service().configure(modelID: "test/model", contextWindow: 131072, port: 9077)
    #expect(result.success)
    #expect(result.backupPaths.count == 1)
    #expect(try String(contentsOfFile: #require(result.backupPaths.first), encoding: .utf8) == original)
    let data = try Data(contentsOf: fixture.config)
    let document = try #require(Yams.load(yaml: String(decoding: data, as: UTF8.self)) as? [String: Any])
    let model = try #require(document["model"] as? [String: Any])
    #expect(model["default"] as? String == "test/model")
    #expect(model["provider"] as? String == "custom:ironmlx-responses")
    #expect(model["context_length"] as? Int == 131072)
    #expect(model["base_url"] as? String == "http://127.0.0.1:9077/v1")
    #expect(model["api_mode"] as? String == "codex_responses")
    #expect(model["temperature"] as? Double == 0.2)
    let providers = try #require(document["providers"] as? [String: [String: Any]])
    #expect(providers["other"]?["api"] as? String == "https://other.example.com")
    #expect(providers["ironmlx-responses"]?["transport"] as? String == "codex_responses")
    #expect(providers["ironmlx-responses"]?["api_mode"] as? String == "codex_responses")
    #expect(providers["ironmlx-responses"]?["discover_models"] as? Bool == false)
    #expect(providers["ironmlx-responses"]?["enabled"] as? Bool == true)
    #expect((providers["ironmlx-responses"]?["extra_headers"] as? [String: String])?["X-Custom"] == "keep")
    #expect((document["terminal"] as? [String: String])?["cwd"] == "/existing/workspace")
    #expect((document["memory"] as? [String: String])?["provider"] == "local")
    #expect(try String(contentsOf: defaults, encoding: .utf8) == "model: default-provider\n")
    #expect(try String(contentsOf: credentials, encoding: .utf8) == "CUSTOM_API_KEY=existing-local-fixture\n")
    for path in [fixture.config.path] + result.backupPaths {
        #expect(try FileManager.default.attributesOfItem(atPath: path)[.posixPermissions] as? Int == 0o600)
    }
    let again = await fixture.service().configure(modelID: "test/model", contextWindow: 131072, port: 9077)
    #expect(again.success && again.backupPaths.isEmpty)
    #expect(try Data(contentsOf: fixture.config) == data)
}

@MainActor
@Test func hermesInvalidConfigurationCapacityAndCustomHomeStopWithoutWriting() async throws {
    let fixture = HermesFixture()
    defer { fixture.clean() }
    #expect(await fixture.service().configure(modelID: "test/model", contextWindow: 32768, port: 9068).code == "context_too_small")
    var service = fixture.service()
    service.environment = ["HERMES_HOME": "/custom/home"]
    #expect(await service.configure(modelID: "test/model", contextWindow: 64000, port: 9068).code == "custom_home")
    #expect(!FileManager.default.fileExists(atPath: fixture.root.path))
    for text in ["[]", "model: [", "model: old-model", "providers: invalid", "providers:\n  ironmlx-responses: invalid"] {
        try fixture.write(text, to: fixture.config)
        #expect(await fixture.service().configure(modelID: "test/model", contextWindow: 64000, port: 9068).code == "configuration_failed")
        #expect(try String(contentsOf: fixture.config, encoding: .utf8) == text)
    }
    try FileManager.default.removeItem(at: fixture.config)
    let marker = fixture.home.appendingPathComponent("profiles/.deleted/ironmlx")
    try fixture.write("deleted\n", to: marker)
    #expect(await fixture.service().configure(modelID: "test/model", contextWindow: 64000, port: 9068).code == "profile_conflict")
    #expect(!FileManager.default.fileExists(atPath: fixture.config.path))
    try FileManager.default.removeItem(at: marker)
    let external = fixture.root.appendingPathComponent("external-config.yaml")
    try fixture.write("model: external\n", to: external)
    try FileManager.default.createSymbolicLink(at: fixture.config, withDestinationURL: external)
    #expect(await fixture.service().configure(modelID: "test/model", contextWindow: 64000, port: 9068).code == "profile_conflict")
    #expect(try String(contentsOf: external, encoding: .utf8) == "model: external\n")
}

@MainActor
@Test func hermesDashboardInstallationGateAndManualConfiguration() async throws {
    let fixture = HermesFixture()
    defer { fixture.clean() }
    let store = AppConfigStore(url: fixture.root.appendingPathComponent("app-config.json"))
    let config = AppConfig(language: "zh-Hans")
    #expect(store.save(config))
    let content = WKUserContentController()
    content.addUserScript(WKUserScript(source: try DashboardWindowController.bootstrapScript(config: config, route: .status),
                                      injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = content
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1180, height: 1000), configuration: configuration)
    DashboardThemeAppearance.apply("light", to: webView)
    let bridge = DashboardBridge(webView: webView, configStore: store, backend: TestRuntimeBackend(state: .stopped))
    var missing = fixture.service()
    missing.findExecutable = { nil }
    bridge.hermesService = missing
    content.add(bridge, name: "fetchAPIPost")
    defer { content.removeScriptMessageHandler(forName: "fetchAPIPost") }
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    try await waitForHermesUI(webView, "document.readyState === 'complete' && typeof configureHermes === 'function' && typeof currentLang !== 'undefined'")
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); navigateTo('chat'); selectAgentGuide('hermes'); document.getElementById('hermes-configure').click()")
    try await waitForHermesUI(webView, "document.getElementById('hermes-feedback').hidden === false && document.getElementById('hermes-configure').disabled === false")
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-feedback').dataset.tone") as? String == "warning")
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-setup-status').textContent.includes('未检测到 Hermes CLI 或本地 Desktop')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-install').href") as? String == HermesIntegrationService.installationURL)
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-install').hidden") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-manual').open") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-verify-code').closest('details') === null") as? Bool == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.home.path))
    for language in ["en", "zh-Hans", "zh-Hant", "ja", "ko"] {
        _ = try await webView.evaluateJavaScript("setLanguage('\(language)')")
        #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-setup-status').textContent.includes('0.20.0')") as? Bool == true)
    }
    _ = try await webView.evaluateJavaScript("document.querySelector('#hermes-manual summary').click()")
    #expect(try await webView.evaluateJavaScript("document.getElementById('hermes-manual').open") as? Bool == true)
    bridge.hermesService = fixture.service("Hermes Agent v0.19.0")
    _ = try await webView.evaluateJavaScript("document.getElementById('hermes-configure').click()")
    try await waitForHermesUI(webView, "document.getElementById('hermes-setup-status').textContent.includes('0.19.0')")
    #expect(!FileManager.default.fileExists(atPath: fixture.home.path))
    if ProcessInfo.processInfo.environment["IRONMLX_TEST_HERMES_BACKEND"] == "1" {
        // Exercise the native success path against IronMLX with a simulated installed client.
        bridge.hermesService = fixture.service()
        _ = try await webView.evaluateJavaScript("window.__LAST_LOADED_IDS__ = ['mlx-community/Qwen3.5-4B-MLX-4bit']; renderAgentGuideConfiguration(); document.getElementById('hermes-configure').click()")
        try await waitForHermesUI(webView, "document.getElementById('hermes-feedback').dataset.tone === 'success'")
        #expect(FileManager.default.fileExists(atPath: fixture.config.path))
    }
}

@MainActor
private func waitForHermesUI(_ webView: WKWebView, _ expression: String) async throws {
    for _ in 0..<100 {
        if (try? await webView.evaluateJavaScript(expression)) as? Bool == true { return }
        try await Task.sleep(for: .milliseconds(100))
    }
    throw HermesUIError.timeout
}
private enum HermesUIError: Error { case timeout }

@MainActor
@Test func hermesUninstalledMachineDetectionDoesNotCreateConfiguration() async throws {
    guard ProcessInfo.processInfo.environment["IRONMLX_TEST_HERMES_MISSING"] == "1" else { return }
    let fixture = HermesFixture()
    defer { fixture.clean() }
    var service = HermesIntegrationService()
    service.hermesHome = fixture.home
    let result = await service.configure(modelID: "test/model", contextWindow: 64000, port: 9068)
    #expect(result.code == "not_installed")
    #expect(!FileManager.default.fileExists(atPath: fixture.root.path))
}

@MainActor
@Test func hermesRepositoryAcceptsGeneratedProfileAndProvider() async throws {
    guard let repo = ProcessInfo.processInfo.environment["IRONMLX_TEST_HERMES_REPO"],
          let python = ProcessInfo.processInfo.environment["IRONMLX_TEST_HERMES_PYTHON"] else { return }
    let fixture = HermesFixture()
    defer { fixture.clean() }
    let result = await fixture.service().configure(modelID: "test/model", contextWindow: 131072, port: 9068)
    #expect(result.success)
    let yaml = try String(contentsOf: fixture.config, encoding: .utf8)
    let document = try #require(Yams.load(yaml: yaml) as? [String: Any])
    let json = fixture.root.appendingPathComponent("config.json")
    try JSONSerialization.data(withJSONObject: document).write(to: json)
    var environment = ProcessInfo.processInfo.environment
    environment["HERMES_HOME"] = fixture.home.path
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    let script = """
        import json, sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from hermes_cli.config_providers import providers_dict_to_custom_providers
        from hermes_constants import named_profile_is_live, get_hermes_home
        config = json.loads(Path(sys.argv[2]).read_text())
        providers = providers_dict_to_custom_providers(config['providers'])
        provider = next(p for p in providers if p['provider_key'] == 'ironmlx-responses')
        assert provider['api_mode'] == 'codex_responses'
        assert provider['base_url'] == 'http://127.0.0.1:9068/v1'
        assert config['model']['provider'] == 'custom:ironmlx-responses'
        assert config['model']['default'] == 'test/model'
        assert config['model']['context_length'] == 131072
        assert named_profile_is_live(get_hermes_home() / 'profiles' / 'ironmlx')
        print('HERMES_CONFIGURATION_OK')
        """
    let output = try await AgentCLI.commandOutput(URL(fileURLWithPath: python),
        arguments: ["-c", script, repo, json.path], environment: environment)
    #expect(output.trimmingCharacters(in: .whitespacesAndNewlines) == "HERMES_CONFIGURATION_OK")
}
