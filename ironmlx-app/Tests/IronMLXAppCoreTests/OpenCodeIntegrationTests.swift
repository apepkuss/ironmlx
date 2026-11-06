import AppKit
import Foundation
import Testing
import WebKit
@testable import IronMLXAppCore

@MainActor
private struct OpenCodeFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("ironmlx-opencode-test-\(UUID().uuidString)")
    var directory: URL { root.appendingPathComponent(".config/opencode") }
    var json: URL { directory.appendingPathComponent("opencode.json") }
    var jsonc: URL { directory.appendingPathComponent("opencode.jsonc") }
    var model: OpenCodeIntegrationService.Model { .init(id: "test/model", contextWindow: 32768, maxTokens: 8192) }

    func service(_ version: String = "opencode v2.2.5\n") -> OpenCodeIntegrationService {
        OpenCodeIntegrationService(home: root, environment: [:], findExecutable: { URL(fileURLWithPath: "/test/opencode") },
                                   findDesktopExecutable: { nil },
                                   versionOutput: { _ in version })
    }

    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    func document(_ url: URL) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url), options: .json5Allowed) as? [String: Any])
    }

    func clean() { try? FileManager.default.removeItem(at: root) }
}

@MainActor
@Test func openCodeDesktopRuntimeWorksWithoutAStandaloneCLI() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    let application = fixture.root.appendingPathComponent("Applications/OpenCode.app")
    let executable = application.appendingPathComponent("Contents/Resources/opencode-cli")
    try fixture.write("bundled runtime fixture", to: executable)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    let info = try PropertyListSerialization.data(fromPropertyList: [
        "CFBundleIdentifier": "ai.opencode.desktop", "CFBundlePackageType": "APPL",
    ], format: .xml, options: 0)
    try info.write(to: application.appendingPathComponent("Contents/Info.plist"))
    #expect(OpenCodeIntegrationService.desktopExecutable(in: application) == executable)
    var service = fixture.service()
    service.findExecutable = { nil }
    service.findDesktopExecutable = { OpenCodeIntegrationService.desktopExecutable(in: application) }
    #expect(await service.configure(model: fixture.model, port: 9068).success)
    // A stale shell installation must not hide a supported Desktop runtime.
    let stale = URL(fileURLWithPath: "/test/old-cli")
    service.findExecutable = { stale }
    service.versionOutput = { $0 == stale ? "2.2.4" : "opencode v2.2.5" }
    #expect(await service.inspect().installedVersion == "2.2.5")
    service.findExecutable = { nil }
    service.versionOutput = { _ in "2.2.4" }
    #expect(await service.inspect().code == "version_too_old")
    try FileManager.default.removeItem(at: executable)
    #expect(OpenCodeIntegrationService.desktopExecutable(in: application) == nil)
}

@MainActor
@Test func openCodeInstallationAndModelGatesDoNotWrite() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    var missing = fixture.service()
    missing.findExecutable = { nil }
    #expect(await missing.configure(model: fixture.model, port: 9068).code == "not_installed")
    for (version, code) in [("2.2.4", "version_too_old"), ("1.9.0", "version_too_old"),
                            ("2.2.5-beta.1", "version_unknown"), ("unexpected output", "version_unknown")] {
        #expect(await fixture.service(version).configure(model: fixture.model, port: 9068).code == code)
    }
    for model in [OpenCodeIntegrationService.Model(id: "MODEL_ID", contextWindow: 1, maxTokens: 1),
                  .init(id: "test#variant", contextWindow: 32768, maxTokens: 4096),
                  .init(id: "test/model", contextWindow: 32, maxTokens: 33)] {
        #expect(await fixture.service().configure(model: model, port: 9068).code == "model_unavailable")
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.root.path))
    #expect(await fixture.service("2.3.0").inspect().success)
}

@MainActor
@Test func openCodeJSONCPreservesUnrelatedBytesAndDefaultModel() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    let unrelated = #"""
      // Keep the default and its spelling.
      "model": "other/default",
      "permissions": [{"action":"shell", "effect":"ask", "resource":"*"}],
      "providers": {
        "other": {"settings": {"apiKey": "existing-key", "body": "// a string }, not a comment"}},
        "ironmlx": {
          // Keep this model's comment and configuration.
          "models": {"other-model": {"limit": {"output": 1024}},},
          "body": {"store": true},
          "settings": {"baseURL":"https://old.example", "apiKey":"old-key"},
        },
      },
    """#
    let original = "{\n" + unrelated + "\n}\n"
    try fixture.write(original, to: fixture.jsonc)
    let result = await fixture.service().configure(model: fixture.model, port: 9077)
    #expect(result.success)
    #expect(result.configurationPath == fixture.jsonc.path)
    #expect(result.backupPaths.count == 1)
    #expect(try String(contentsOfFile: #require(result.backupPaths.first), encoding: .utf8) == original)
    let updated = try String(contentsOf: fixture.jsonc, encoding: .utf8)
    for unchanged in [#""model": "other/default""#, #""other": {"settings": {"apiKey": "existing-key", "body": "// a string }, not a comment"}}"#,
                      "// Keep this model's comment and configuration.", #""other-model": {"limit": {"output": 1024}}"#] {
        #expect(updated.contains(unchanged))
    }
    let document = try fixture.document(fixture.jsonc)
    let providers = try #require(document["providers"] as? [String: [String: Any]])
    let provider = try #require(providers["ironmlx"])
    #expect(provider["package"] as? String == OpenCodeIntegrationService.providerPackage)
    #expect((provider["body"] as? [String: [String: Bool]])?["chat_template_kwargs"] == ["enable_thinking": false])
    let settings = try #require(provider["settings"] as? [String: String])
    #expect(settings == ["baseURL": "http://127.0.0.1:9077/v1", "apiKey": "", "provider": "ironmlx"])
    let models = try #require(provider["models"] as? [String: [String: Any]])
    #expect(models["test/model"]?["limit"] as? [String: Int] == ["context": 32768, "output": 4096])
    #expect(document["model"] as? String == "other/default")
    #expect(try FileManager.default.attributesOfItem(atPath: fixture.jsonc.path)[.posixPermissions] as? Int == 0o600)
    let second = await fixture.service().configure(model: fixture.model, port: 9077)
    #expect(second.success)
    #expect(second.backupPaths.isEmpty)
    #expect(try String(contentsOf: fixture.jsonc, encoding: .utf8) == updated)
}

@MainActor
@Test func openCodeConfigurationHandlesBothGlobalFilesAndEscapedModelIDs() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    try fixture.write(#"{"providers":{"ironmlx":{"canonical":"openai","headers":{"Authorization":"old"},"models":{"other":{"name":"keep"}}}}}"#, to: fixture.json)
    try fixture.write("{ /* keep this comment */ \"model\": \"other/default\" // keep this too\n}", to: fixture.jsonc)
    let model = OpenCodeIntegrationService.Model(id: "org/模型-\"quoted\"", contextWindow: 8192, maxTokens: 1024, supportsVision: true)
    let result = await fixture.service().configure(model: model, port: 9068)
    #expect(result.success)
    #expect(result.configurationPath == fixture.jsonc.path)
    #expect(result.backupPaths.count == 2)
    for url in [fixture.json, fixture.jsonc] {
        let providers = try #require(fixture.document(url)["providers"] as? [String: [String: Any]])
        let provider = try #require(providers["ironmlx"])
        #expect(provider["canonical"] == nil)
        #expect(provider["headers"] == nil)
        let models = try #require(provider["models"] as? [String: [String: Any]])
        let selected = try #require(models[model.id])
        #expect(selected["limit"] as? [String: Int] == ["context": 8192, "output": 1024])
        #expect((selected["capabilities"] as? [String: Any])?["input"] as? [String] == ["text", "image"])
    }
    let jsonProvider = try #require(fixture.document(fixture.json)["providers"] as? [String: [String: Any]])
    #expect((jsonProvider["ironmlx"]?["models"] as? [String: Any])?["other"] != nil)
    #expect(try String(contentsOf: fixture.jsonc, encoding: .utf8).contains("/* keep this comment */"))
    #expect(await fixture.service().configure(model: model, port: 9068).backupPaths.isEmpty)
}

@MainActor
@Test func openCodeInvalidOrLegacyConfigurationIsNeverOverwritten() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    for original in ["[", "[]", #"{"providers":[]}"#, #"{"providers":{"ironmlx":false}}"#,
                     #"{"providers":{"ironmlx":{"models":[]}}}"#,
                     #"{"providers":{"ironmlx":{"models":{"test/model":false}}}}"#,
                     #"{"providers":{},"providers":{}}"#] {
        try fixture.write(original, to: fixture.jsonc)
        #expect(await fixture.service().configure(model: fixture.model, port: 9068).code == "configuration_failed")
        #expect(try String(contentsOf: fixture.jsonc, encoding: .utf8) == original)
        #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path).count == 1)
    }
    let legacy = #"{"provider":{"ironmlx":{"npm":"@ai-sdk/openai"}}}"#
    try fixture.write(legacy, to: fixture.jsonc)
    #expect(await fixture.service().configure(model: fixture.model, port: 9068).code == "legacy_configuration")
    #expect(try String(contentsOf: fixture.jsonc, encoding: .utf8) == legacy)
}

@MainActor
@Test func openCodeConfigurationUsesXDGAndExplicitDirectories() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    var service = fixture.service()
    let xdg = fixture.root.appendingPathComponent("xdg")
    service.environment = ["XDG_CONFIG_HOME": xdg.path]
    #expect(await service.configure(model: fixture.model, port: 9068).configurationPath == xdg.appendingPathComponent("opencode/opencode.jsonc").path)
    let explicit = fixture.root.appendingPathComponent("custom")
    service.environment["OPENCODE_CONFIG_DIR"] = explicit.path
    #expect(await service.configure(model: fixture.model, port: 9068).configurationPath == explicit.appendingPathComponent("opencode.jsonc").path)
    #expect(!FileManager.default.fileExists(atPath: fixture.directory.path))
    for key in ["OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT"] {
        service.environment = [key: "custom"]
        #expect(await service.configure(model: fixture.model, port: 9068).code == "custom_configuration")
    }
}

@MainActor
@Test func openCodeNativeReasoningVariantsEnableThinkingAndReplay() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    var model = fixture.model
    model.reasoning = .init(nativeEfforts: ["low", "medium", "xhigh"], defaultEffort: "xhigh",
                            supportsDisable: true, source: "chat_template")
    #expect(await fixture.service().configure(model: model, port: 9068).success)
    let first = try Data(contentsOf: fixture.jsonc)
    #expect(await fixture.service().configure(model: model, port: 9068).backupPaths.isEmpty)
    #expect(try Data(contentsOf: fixture.jsonc) == first)
    let document = try fixture.document(fixture.jsonc)
    let provider = try #require((document["providers"] as? [String: [String: Any]])?["ironmlx"])
    let selected = try #require((provider["models"] as? [String: [String: Any]])?[model.id])
    let variants = try #require(selected["variants"] as? [[String: Any]])
    #expect(variants.compactMap { $0["id"] as? String } == ["none", "low", "medium", "xhigh"])
    for variant in variants {
        let id = try #require(variant["id"] as? String)
        let kwargs = try #require((variant["body"] as? [String: [String: Bool]])?["chat_template_kwargs"])
        #expect(kwargs["enable_thinking"] == (id != "none"))
        if id == "none" { #expect(variant["settings"] == nil) }
        else {
            #expect((variant["settings"] as? [String: String])?["reasoningEffort"] == id)
            #expect(kwargs["preserve_thinking"] == true)
        }
    }
    #expect((selected["compatibility"] as? [String: Any])?["reasoningField"] as? String == "reasoning_content")
    // Reconfiguring a model whose template is unknown must remove obsolete native controls.
    #expect(await fixture.service().configure(model: fixture.model, port: 9068).success)
    let reset = try fixture.document(fixture.jsonc)
    let resetProvider = try #require((reset["providers"] as? [String: [String: Any]])?["ironmlx"])
    let resetModel = try #require((resetProvider["models"] as? [String: [String: Any]])?[model.id])
    #expect(resetModel["variants"] == nil)
    #expect(resetModel["compatibility"] == nil)
}

@MainActor
@Test func openCodeNewConfigurationIsNativeChatAndLeavesSelectionToTheUser() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    #expect(await fixture.service().configure(model: fixture.model, port: 9068).success)
    let document = try fixture.document(fixture.jsonc)
    #expect(document["model"] == nil)
    #expect(!FileManager.default.fileExists(atPath: fixture.json.path))
    let provider = try #require((document["providers"] as? [String: [String: Any]])?["ironmlx"])
    #expect(provider["package"] as? String == "@opencode/ai/providers/openai-compatible")
    #expect(!(try String(contentsOf: fixture.jsonc, encoding: .utf8)).contains("responses"))
    if let output = ProcessInfo.processInfo.environment["IRONMLX_OPENCODE_GENERATED_CONFIG"] {
        try Data(contentsOf: fixture.jsonc).write(to: URL(fileURLWithPath: output), options: .atomic)
    }
}

@MainActor
private func waitForOpenCodeUI(_ webView: WKWebView, _ condition: String) async throws {
    let deadline = Date().addingTimeInterval(10)
    while Date() < deadline {
        if (try? await webView.evaluateJavaScript(condition)) as? Bool == true { return }
        try await Task.sleep(for: .milliseconds(25))
    }
    Issue.record("OpenCode UI did not reach: \(condition)")
    throw URLError(.timedOut)
}

@MainActor
@Test func openCodeDashboardUsesNativeBridgeAndDisplaysInstallationAndModelGates() async throws {
    let fixture = OpenCodeFixture()
    defer { fixture.clean() }
    let store = AppConfigStore(url: fixture.root.appendingPathComponent("app-config.json"))
    #expect(store.save(AppConfig(language: "zh-Hans")))
    let content = WKUserContentController()
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = content
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1180, height: 900), configuration: configuration)
    let parameters = ModelParameterStore(url: fixture.root.appendingPathComponent("model-params.json"))
    let bridge = DashboardBridge(webView: webView, configStore: store, backend: TestRuntimeBackend(state: .stopped),
                                 parameterStore: parameters)
    var missing = fixture.service()
    missing.findExecutable = { nil }
    bridge.openCodeService = missing
    content.add(bridge, name: "fetchAPIPost")
    defer { content.removeScriptMessageHandler(forName: "fetchAPIPost") }
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    try await waitForOpenCodeUI(webView, "document.readyState === 'complete' && typeof configureOpenCode === 'function'")
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); navigateTo('chat'); selectAgentGuide('opencode'); document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "document.getElementById('opencode-feedback').dataset.tone === 'warning'")
    #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-install').hidden") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-install').href") as? String == OpenCodeIntegrationService.installationURL)
    #expect(try await webView.evaluateJavaScript("document.getElementById('copy-opencode-config').disabled") as? Bool == true)
    for language in ["zh-Hans", "zh-Hant", "en", "ja", "ko"] {
        _ = try await webView.evaluateJavaScript("setLanguage('\(language)')")
        #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-setup-status').textContent.includes('2.2.5')") as? Bool == true)
        #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-configure').textContent !== 'opencode_auto_button'") as? Bool == true)
    }
    bridge.openCodeService = fixture.service()
    _ = try await webView.evaluateJavaScript("document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.code === 'model_unavailable'")
    #expect(!FileManager.default.fileExists(atPath: fixture.directory.path))
    _ = try await webView.evaluateJavaScript("document.querySelector('#opencode-manual summary').click()")
    #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-manual').open") as? Bool == true)
    _ = try await webView.evaluateJavaScript("window.__LAST_LOADED_IDS__=['test/model']; dshModelCapacities.set('test/model',{contextWindow:32768,maxOutputTokens:8192}); renderAgentGuideConfiguration()")
    #expect(try await webView.evaluateJavaScript("JSON.parse(document.getElementById('opencode-config-code').textContent).providers.ironmlx.limit === undefined") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("JSON.parse(document.getElementById('opencode-config-code').textContent).providers.ironmlx.models['test/model'].limit.output") as? Int == 4096)
    #expect(try await webView.evaluateJavaScript("document.getElementById('copy-opencode-config').disabled") as? Bool == false)

    // Exercise the production bridge through real loopback reads and a real configuration write.
    let payload = fixture.root.appendingPathComponent("backend.json")
    let ready = fixture.root.appendingPathComponent("backend.port")
    let script = fixture.root.appendingPathComponent("backend.py")
    var model = BackendLoadedModelInfo(id: fixture.model.id, model: fixture.model.id, path: "/test/model",
                                      architecture: "qwen3_5", isDefault: true, maxPositionEmbeddings: 32768)
    func writeBackend() throws {
        let loaded = try JSONSerialization.jsonObject(with: JSONEncoder().encode([model]))
        let data = try JSONSerialization.data(withJSONObject: [
            "/admin/api/models/loaded": loaded,
            "/v1/models": ["data": [["id": fixture.model.id, "context_window": 32768, "max_output_tokens": 8192]]],
        ])
        try data.write(to: payload, options: .atomic)
    }
    try writeBackend()
    try """
    import json, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            with open(sys.argv[1]) as f: payload = json.load(f)
            body = json.dumps(payload.get(self.path, {})).encode()
            self.send_response(200 if self.path in payload else 404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    with open(sys.argv[2], 'w') as f: f.write(str(server.server_port))
    server.serve_forever()
    """.write(to: script, atomically: true, encoding: .utf8)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
    process.arguments = [script.path, payload.path, ready.path]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try process.run()
    defer { process.terminate(); process.waitUntilExit() }
    let deadline = Date().addingTimeInterval(5)
    while !FileManager.default.fileExists(atPath: ready.path) && Date() < deadline {
        try await Task.sleep(for: .milliseconds(25))
    }
    let port = try #require(UInt16(String(contentsOf: ready, encoding: .utf8)))
    var config = store.load()
    config.port = port
    #expect(store.save(config))
    if let cli = ProcessInfo.processInfo.environment["IRONMLX_OPENCODE_CLI"] {
        var service = fixture.service()
        service.findExecutable = { URL(fileURLWithPath: cli) }
        service.versionOutput = { try await AgentCLI.commandOutput($0, arguments: ["--version"]) }
        bridge.openCodeService = service
    }
    _ = try await webView.evaluateJavaScript("window.__IRONMLX_APP_CONFIG__={port:\(port)}; renderAgentGuideConfiguration(); document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.success === true")
    let written = try Data(contentsOf: fixture.jsonc)
    let provider = try #require((fixture.document(fixture.jsonc)["providers"] as? [String: [String: Any]])?["ironmlx"])
    #expect((provider["settings"] as? [String: String])?["baseURL"] == "http://127.0.0.1:\(port)/v1")
    #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-feedback').dataset.tone") as? String == "success")
    #expect(try await webView.evaluateJavaScript("document.getElementById('opencode-install').hidden") as? Bool == true)

    if let output = ProcessInfo.processInfo.environment["IRONMLX_OPENCODE_SNAPSHOT_DIR"] {
        try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)
        _ = try await webView.evaluateJavaScript("document.querySelectorAll('.nav-item, .agent-guide-nav-item').forEach(node=>node.style.transition='none'); setLanguage('zh-Hans'); navigateTo('chat'); selectAgentGuide('opencode'); document.getElementById('opencode-manual').open=false")
        for theme in ["light", "dark"] {
            DashboardThemeAppearance.apply(theme, to: webView)
            try await waitForOpenCodeUI(webView, "matchMedia('(prefers-color-scheme: dark)').matches === \(theme == "dark" ? "true" : "false")")
            let image = try await webView.takeSnapshot(configuration: nil)
            let bitmap = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
            try #require(bitmap.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: output).appendingPathComponent("opencode-\(theme).png"))
        }
    }
    config.networkMode = "lan"
    #expect(store.save(config))
    _ = try await webView.evaluateJavaScript("document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.code === 'local_only'")
    config.networkMode = "local"
    #expect(store.save(config))
    model.capabilities.runtimeKind = "embedding"
    try writeBackend()
    _ = try await webView.evaluateJavaScript("document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.code === 'model_unavailable'")
    #expect(try Data(contentsOf: fixture.jsonc) == written)

    // Exclusive DFlash2 has no /admin/api/models/loaded route. Bind its live
    // health identity to the configured target, then use the public capacity.
    config.defaultModel = fixture.model.id
    config.loadedModels = [fixture.model.id]
    #expect(store.save(config))
    try parameters.save(ModelParameters(modelID: fixture.model.id, dflash2Enabled: true))
    var health = dashboardDFlash2HealthSnapshot(modelID: fixture.model.id)
    func writeExclusiveBackend() throws {
        let healthJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(health))
        let data = try JSONSerialization.data(withJSONObject: [
            "/healthz": healthJSON,
            "/v1/models": ["data": [["id": fixture.model.id, "context_window": 32768, "max_output_tokens": 8192]]],
        ])
        try data.write(to: payload, options: .atomic)
    }
    try writeExclusiveBackend()
    _ = try await webView.evaluateJavaScript("document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.success === true")
    #expect(try Data(contentsOf: fixture.jsonc) == written)
    health.model.name = "other/model"
    try writeExclusiveBackend()
    _ = try await webView.evaluateJavaScript("document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.code === 'model_unavailable'")
    health.model.name = fixture.model.id
    health.dflash2?.enabled = false
    try writeExclusiveBackend()
    _ = try await webView.evaluateJavaScript("document.getElementById('opencode-configure').click()")
    try await waitForOpenCodeUI(webView, "openCodeSetupResult?.code === 'model_unavailable'")
    #expect(try Data(contentsOf: fixture.jsonc) == written)
}
