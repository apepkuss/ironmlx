import AppKit
import Foundation
import Testing
import WebKit
import Yams
@testable import IronMLXAppCore

@MainActor
private struct OMPFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("ironmlx-omp-test-\(UUID().uuidString)")
    var home: URL { root.appendingPathComponent("agent") }
    var primary: URL { home.appendingPathComponent("models.yml") }
    var alternative: URL { home.appendingPathComponent("models.yaml") }
    var model: OMPIntegrationService.Model { .init(id: "test/model", contextWindow: 64000, maxTokens: 8192) }
    func service(_ version: String = "omp/18.1.16\n") -> OMPIntegrationService {
        OMPIntegrationService(agentHome: home, findExecutable: { URL(fileURLWithPath: "/test/omp") },
                              versionOutput: { _ in version })
    }
    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }
    func clean() { try? FileManager.default.removeItem(at: root) }
}

@MainActor
@Test func ompInstallationAndVersionGatesNeverWriteFiles() async throws {
    let fixture = OMPFixture()
    defer { fixture.clean() }
    var missing = fixture.service()
    missing.findExecutable = { nil }
    #expect(await missing.configure(model: fixture.model, port: 9068).code == "not_installed")
    for (version, code) in [("omp/18.1.15", "version_too_old"), ("omp/17.2.12", "version_too_old"),
                            ("invalid", "version_unknown"), ("omp/18.1.16-beta.1", "version_unknown")] {
        let result = await fixture.service(version).configure(model: fixture.model, port: 9068)
        #expect(!result.success)
        #expect(result.code == code)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.root.path))
    for version in ["omp/18.1.16", "omp/18.1.20", "omp/19.0.0"] {
        #expect(await fixture.service(version).inspect().result.success)
    }
}

@MainActor
@Test func ompConfigurationPreservesProvidersBacksUpAndIsIdempotent() async throws {
    let fixture = OMPFixture()
    defer { fixture.clean() }
    let original = """
    # User comment
    providers:
      other:
        baseUrl: https://example.com/v1
        apiKey: EXISTING_KEY
      ironmlx:
        modelOverrides:
          other-model:
            maxTokens: 1024
          test/model:
            name: My model
    """
    try fixture.write(original, to: fixture.primary)
    let result = await fixture.service().configure(model: fixture.model, port: 9077)
    #expect(result.success)
    #expect(result.backupPaths.count == 1)
    #expect(try String(contentsOfFile: #require(result.backupPaths.first), encoding: .utf8) == original)
    let contents = try Data(contentsOf: fixture.primary)
    let document = try #require(Yams.load(yaml: String(decoding: contents, as: UTF8.self)) as? [String: Any])
    let providers = try #require(document["providers"] as? [String: [String: Any]])
    #expect(providers["other"]?["apiKey"] as? String == "EXISTING_KEY")
    let provider = try #require(providers["ironmlx"])
    #expect(provider["baseUrl"] as? String == "http://127.0.0.1:9077/v1")
    #expect(provider["auth"] as? String == "none")
    #expect(provider["api"] as? String == "openai-responses")
    #expect((provider["discovery"] as? [String: String])?["type"] == "openai-models-list")
    let overrides = try #require(provider["modelOverrides"] as? [String: [String: Any]])
    #expect(overrides["other-model"]?["maxTokens"] as? Int == 1024)
    #expect(overrides["test/model"]?["name"] as? String == "My model")
    #expect(overrides["test/model"]?["contextWindow"] as? Int == 64000)
    #expect(overrides["test/model"]?["maxTokens"] as? Int == 4096)
    #expect(overrides["test/model"]?["reasoning"] as? Bool == false)
    #expect(try FileManager.default.attributesOfItem(atPath: fixture.primary.path)[.posixPermissions] as? Int == 0o600)
    let second = await fixture.service().configure(model: fixture.model, port: 9077)
    #expect(second.success)
    #expect(second.backupPaths.isEmpty)
    #expect(try Data(contentsOf: fixture.primary) == contents)
}

@MainActor
@Test func ompConfigurationHonorsYAMLPrecedenceAndLegacyMigration() async throws {
    let fixture = OMPFixture()
    defer { fixture.clean() }
    try fixture.write("providers: {}", to: fixture.alternative)
    let result = await fixture.service().configure(model: fixture.model, port: 9068)
    #expect(result.success)
    #expect(result.configurationPath == fixture.alternative.path)
    #expect(!FileManager.default.fileExists(atPath: fixture.primary.path))
    let alternative = try Data(contentsOf: fixture.alternative)
    try fixture.write("providers: {}", to: fixture.primary)
    #expect(await fixture.service().configure(model: fixture.model, port: 9068).configurationPath == fixture.primary.path)
    #expect(try Data(contentsOf: fixture.alternative) == alternative)
    try FileManager.default.removeItem(at: fixture.primary)
    try FileManager.default.removeItem(at: fixture.alternative)
    let legacy = fixture.home.appendingPathComponent("models.json")
    try fixture.write("{}", to: legacy)
    #expect(await fixture.service().configure(model: fixture.model, port: 9068).code == "initialize_cli")
    #expect(!FileManager.default.fileExists(atPath: fixture.primary.path))
    #expect(try String(contentsOf: legacy, encoding: .utf8) == "{}")
}

@MainActor
@Test func ompInvalidYAMLAndModelAreNeverOverwritten() async throws {
    let fixture = OMPFixture()
    defer { fixture.clean() }
    for text in ["providers: [", "[]", "providers: invalid", "unknown: true",
                 "providers:\n  ironmlx: invalid", "providers:\n  ironmlx:\n    modelOverrides: invalid"] {
        try fixture.write(text, to: fixture.primary)
        let result = await fixture.service().configure(model: fixture.model, port: 9068)
        #expect(result.code == "configuration_failed")
        #expect(try String(contentsOf: fixture.primary, encoding: .utf8) == text)
    }
    let invalid = OMPIntegrationService.Model(id: "MODEL_ID", contextWindow: 1, maxTokens: 1)
    #expect(await fixture.service().configure(model: invalid, port: 9068).code == "model_unavailable")
}

@Test func ompVersionProbeIsBounded() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("ironmlx-omp-probe-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let script = root.appendingPathComponent("omp")
    try "#!/bin/sh\nexec /bin/sleep 30\n".write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
    let start = Date()
    await #expect(throws: (any Error).self) {
        try await OMPIntegrationService.commandOutput(script, arguments: ["--version"], timeout: 0.1)
    }
    #expect(Date().timeIntervalSince(start) < 3)
}

@MainActor
@Test func ompDashboardShowsInstallationGateAndPreservesManualInstructions() async throws {
    let fixture = OMPFixture()
    defer { fixture.clean() }
    let store = AppConfigStore(url: fixture.root.appendingPathComponent("app-config.json"))
    let config = AppConfig(language: "zh-Hans")
    #expect(store.save(config))
    let content = WKUserContentController()
    content.addUserScript(WKUserScript(
        source: try DashboardWindowController.bootstrapScript(config: config, route: .status),
        injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = content
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1180, height: 900), configuration: configuration)
    DashboardThemeAppearance.apply("light", to: webView)
    let bridge = DashboardBridge(webView: webView, configStore: store,
                                 backend: TestRuntimeBackend(state: .stopped))
    var missing = fixture.service()
    missing.findExecutable = { nil }
    bridge.ompService = missing
    content.add(bridge, name: "fetchAPIPost")
    defer { content.removeScriptMessageHandler(forName: "fetchAPIPost") }
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    try await waitForOMPUI(webView, "document.readyState === 'complete' && typeof configureOMP === 'function' && typeof currentLang !== 'undefined'")
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); navigateTo('chat'); selectAgentGuide('omp'); document.getElementById('omp-configure').click()")
    try await waitForOMPUI(webView, "document.getElementById('omp-feedback').hidden === false && document.getElementById('omp-configure').disabled === false")
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-feedback').dataset.tone") as? String == "warning")
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-setup-status').textContent.includes('18.1.16')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-install').hidden") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-install').href") as? String == OMPIntegrationService.installationURL)
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-manual').open") as? Bool == false)
    _ = try await webView.evaluateJavaScript("document.querySelector('#omp-manual summary').click()")
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-manual').open") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('omp-verify-code').closest('details') === null") as? Bool == true)
    for language in ["zh-Hans", "zh-Hant", "en", "ja", "ko"] {
        _ = try await webView.evaluateJavaScript("setLanguage('\(language)')")
        #expect(try await webView.evaluateJavaScript("document.getElementById('omp-setup-status').textContent.includes('18.1.16')") as? Bool == true)
        #expect(try await webView.evaluateJavaScript("document.getElementById('omp-configure').textContent !== 'omp_auto_button'") as? Bool == true)
    }
    bridge.ompService = fixture.service("omp/17.2.12")
    _ = try await webView.evaluateJavaScript("document.getElementById('omp-configure').click()")
    try await waitForOMPUI(webView, "document.getElementById('omp-setup-status').textContent.includes('17.2.12')")
    #expect(!FileManager.default.fileExists(atPath: fixture.home.path))
    if ProcessInfo.processInfo.environment["IRONMLX_TEST_OMP_LIVE"] == "1" {
        var live = OMPIntegrationService()
        live.agentHome = fixture.home
        bridge.ompService = live
        _ = try await webView.evaluateJavaScript("window.__LAST_LOADED_IDS__ = ['mlx-community/Qwen3.5-4B-MLX-4bit']; renderAgentGuideConfiguration(); document.getElementById('omp-configure').click()")
        try await waitForOMPUI(webView, "document.getElementById('omp-feedback').dataset.tone === 'success'")
        #expect(FileManager.default.fileExists(atPath: fixture.primary.path))
        if let directory = ProcessInfo.processInfo.environment["IRONMLX_OMP_SNAPSHOT_DIR"] {
            let folder = URL(fileURLWithPath: directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); document.getElementById('omp-manual').open = false; document.getElementById('page-chat').scrollTop = 0")
            // Offscreen WKWebView transitions can remain at the first animation
            // frame; disable them for deterministic appearance snapshots only.
            _ = try await webView.evaluateJavaScript("document.querySelectorAll('.agent-guide-nav-item').forEach(node => node.style.transition = 'none')")
            for (name, appearance) in [("light", "light"), ("dark", "dark")] {
                DashboardThemeAppearance.apply(appearance, to: webView)
                try await waitForOMPUI(webView, "window.matchMedia('(prefers-color-scheme: dark)').matches === \(name == "dark" ? "true" : "false")")
                try await Task.sleep(for: .milliseconds(300))
                print("OMP snapshot \(name): \(try await webView.evaluateJavaScript("JSON.stringify({background:getComputedStyle(document.querySelector('.agent-guide-nav-item.active')).backgroundColor,color:getComputedStyle(document.querySelector('.agent-guide-nav-name')).color,dark:matchMedia('(prefers-color-scheme: dark)').matches})"))")
                let image = try await webView.takeSnapshot(configuration: nil)
                let bitmap = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
                try bitmap.representation(using: .png, properties: [:])?.write(to: folder.appendingPathComponent("omp-one-click-\(name).png"))
            }
        }
    }
}

@MainActor
private func waitForOMPUI(_ webView: WKWebView, _ expression: String) async throws {
    for _ in 0..<100 {
        if (try? await webView.evaluateJavaScript(expression)) as? Bool == true { return }
        try await Task.sleep(for: .milliseconds(100))
    }
    throw OMPUIError.timeout
}
private enum OMPUIError: Error { case timeout }

@MainActor
@Test func ompInstalledCLIAcceptsGeneratedConfigurationAndResponds() async throws {
    guard ProcessInfo.processInfo.environment["IRONMLX_TEST_OMP_LIVE"] == "1" else { return }
    let fixture = OMPFixture()
    defer { fixture.clean() }
    var service = OMPIntegrationService()
    service.agentHome = fixture.home
    let installation = await service.inspect()
    #expect(installation.result.success)
    let executable = try #require(installation.executableURL)
    let model = OMPIntegrationService.Model(id: "mlx-community/Qwen3.5-4B-MLX-4bit", contextWindow: 64000, maxTokens: 4096)
    let result = await service.configure(model: model, port: 9068)
    #expect(result.success)
    var environment = ProcessInfo.processInfo.environment
    environment["PI_CODING_AGENT_DIR"] = fixture.home.path
    environment["PI_CONFIG_DIR"] = fixture.root.appendingPathComponent(".omp").path
    environment.removeValue(forKey: "OMP_PROFILE")
    environment.removeValue(forKey: "PI_PROFILE")
    let listing = try await OMPIntegrationService.commandOutput(executable,
        arguments: ["models", "refresh", "ironmlx", "--json", "--no-extensions"], timeout: 60, environment: environment)
    #expect(listing.contains(model.id))
    let output = try await OMPIntegrationService.commandOutput(executable, arguments: [
        "--cwd", fixture.root.path, "--model", "ironmlx/\(model.id)", "-p", "--no-session", "--no-tools",
        "--no-lsp", "--no-extensions", "--no-skills", "--no-rules", "Reply with exactly IRONMLX_OK"
    ], timeout: 120, environment: environment)
    #expect(output.contains("IRONMLX_OK"))
    print("OMP live verification: generated configuration accepted; reply=\(output.trimmingCharacters(in: .whitespacesAndNewlines))")
    let toolOutput = try await OMPIntegrationService.commandOutput(executable, arguments: [
        "--cwd", fixture.root.path, "--model", "ironmlx/\(model.id)", "--mode", "json", "-p", "--no-session",
        "--tools", "bash", "--approval-mode", "yolo", "--no-lsp", "--no-extensions", "--no-skills", "--no-rules",
        "Use the bash tool exactly once to run pwd, then report its output."
    ], timeout: 120, environment: environment, maximumOutputBytes: 2 * 1024 * 1024)
    let events = toolOutput.split(separator: "\n").compactMap {
        try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
    }
    let starts = events.filter { $0["type"] as? String == "tool_execution_start" }
    let ends = events.filter { $0["type"] as? String == "tool_execution_end" }
    #expect(starts.count == 1)
    #expect(starts.first?["toolName"] as? String == "bash")
    #expect(ends.count == 1)
    #expect(ends.first?["isError"] as? Bool == false)
    #expect(toolOutput.contains(fixture.root.path))
    print("OMP live tool verification: bash executions=\(starts.count), errors=\(ends.first?["isError"] ?? "unknown")")
}
