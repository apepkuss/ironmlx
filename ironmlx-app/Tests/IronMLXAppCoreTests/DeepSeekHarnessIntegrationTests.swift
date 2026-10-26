import AppKit
import WebKit
import Foundation
import Testing
import Yams

@testable import IronMLXAppCore

@MainActor
private struct HarnessFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    var home: URL { root.appendingPathComponent("official-home") }
    var app: URL { root.appendingPathComponent("DeepSeek Harness.app") }
    var cli: URL { root.appendingPathComponent("cli/lib/bin.js") }
    var model: DeepSeekHarnessIntegrationService.Model { .init(id: "test/model", contextWindow: 64000, maxTokens: 8192) }
    func patch(_ profile: String) -> URL { home.appendingPathComponent("profiles/\(profile)/cordis.patch.yml") }
    func write(_ text: String, _ url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }
    func installApp(version: String = "0.2.1-rc.2", id: String = "com.deepseek.dsh") throws {
        let data = try PropertyListSerialization.data(fromPropertyList: [
            "CFBundleIdentifier": id, "CFBundleShortVersionString": version,
        ], format: .xml, options: 0)
        try FileManager.default.createDirectory(at: app.appendingPathComponent("Contents"), withIntermediateDirectories: true)
        try data.write(to: app.appendingPathComponent("Contents/Info.plist"))
    }
    func installCLI() throws {
        try write(#"{"name":"@deepseek-ai/dsh","bin":{"dsh":"lib/bin.js"}}"#, root.appendingPathComponent("cli/package.json"))
        try write("#!/usr/bin/env node\n", cli)
    }
    func service(cliVersion: String = "0.2.1-rc.2", hasCLI: Bool = false) -> DeepSeekHarnessIntegrationService {
        DeepSeekHarnessIntegrationService(harnessHome: home, environment: [:],
            findExecutable: { hasCLI ? cli : nil }, versionOutput: { _ in cliVersion },
            loginShellHome: { "" }, applicationURLs: { [app] }, isDesktopRunning: { false })
    }
    func clean() { try? FileManager.default.removeItem(at: root) }
}

@MainActor
@Test func officialHarnessDoesNotMistakeCommunityDesktopForOfficialInstallation() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installApp(version: "0.12.1", id: "io.dsh.desktop")
    try f.write("community sentinel", f.root.appendingPathComponent("community-home/cordis.patch.yml"))
    for mode in [DeepSeekHarnessIntegrationService.Mode.cli, .desktop] {
        let result = await f.service().configure(mode: mode, model: f.model, port: 9068)
        #expect(result.code == "not_installed")
        #expect(!FileManager.default.fileExists(atPath: f.home.path))
    }
    // Even the official npm package inside a community .app must not count as an installed official CLI.
    let bundled = f.app.appendingPathComponent("Contents/Resources/dsh/lib/bin.js")
    try f.write(#"{"name":"@deepseek-ai/dsh","bin":{"dsh":"lib/bin.js"}}"#, bundled.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("package.json"))
    try f.write("stub", bundled)
    #expect(!DeepSeekHarnessIntegrationService.isOfficialCLI(bundled))
    #expect(try String(contentsOf: f.root.appendingPathComponent("community-home/cordis.patch.yml"), encoding: .utf8) == "community sentinel")
    try f.installApp()
    #expect(await f.service().inspect(mode: .desktop).success)
    #expect(await f.service().inspect(mode: .cli).code == "not_installed")
}

@MainActor
@Test func officialHarnessRecognizesReleaseBundleAndItsCLIWhileRejectingDevelopmentBuilds() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installApp()
    // Match the actual signed macOS 0.2.1-rc.2 distribution, not the unsigned source-build default.
    #expect(DeepSeekHarnessIntegrationService.bundleIdentifier == "com.deepseek.dsh")
    #expect(await f.service().inspect(mode: .desktop).installedVersion == "0.2.1-rc.2")
    let launcher = f.app.appendingPathComponent("Contents/Resources/runtime/cli/bin/dsh")
    try f.write("stub", launcher)
    #expect(DeepSeekHarnessIntegrationService.isOfficialCLI(launcher))
    for identifier in ["com.deepseek.harness", "com.deepseek.harness.dev.example", "io.dsh.desktop", "io.dsh.desktop.dev"] {
        try f.installApp(id: identifier)
        #expect(await f.service().inspect(mode: .desktop).code == "not_installed")
        #expect(!DeepSeekHarnessIntegrationService.isOfficialCLI(launcher))
    }
    if ProcessInfo.processInfo.environment["IRONMLX_TEST_OFFICIAL_HARNESS_INSTALLED"] == "1" {
        let result = await DeepSeekHarnessIntegrationService().inspect(mode: .desktop)
        #expect(result.success)
        #expect(result.code == "ready")
        #expect(result.installedVersion == "0.2.1-rc.2")
        #expect(DSHDesktopIntegrationService().inspect().success)
    }
}

@MainActor
@Test func officialHarnessVersionAndIdentityGatesDoNotWriteConfiguration() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installCLI()
    for version in ["0.2.1-rc.2", "0.2.1-rc.10", "0.2.1", "0.2.2", "1.0.0"] {
        #expect(DeepSeekHarnessIntegrationService.supports(version))
    }
    for version in ["0.1.99", "0.2.0", "0.2.1-rc.1", "0.2.1-alpha.1", "0.2.1-test.1", "unknown", "00.2.1"] {
        #expect(!DeepSeekHarnessIntegrationService.supports(version))
    }
    for (version, code) in [("0.2.1-rc.1", "version_too_old"), ("0.2.1-test.1", "version_unknown")] {
        try f.installApp(version: version)
        for mode in [DeepSeekHarnessIntegrationService.Mode.cli, .desktop] {
            #expect(await f.service(cliVersion: version, hasCLI: true).configure(mode: mode, model: f.model, port: 9068).code == code)
        }
    }
    try f.write(#"{"name":"another-dsh","bin":{"dsh":"lib/bin.js"}}"#, f.root.appendingPathComponent("cli/package.json"))
    #expect(await f.service(hasCLI: true).inspect(mode: .cli).code == "cli_unrecognized")
    #expect(!FileManager.default.fileExists(atPath: f.home.path))
    let spaced = f.root.appendingPathComponent("CLI with spaces/dsh")
    try f.write("#!/bin/sh\nprintf '0.2.1-rc.2\\n'\n", spaced)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: spaced.path)
    let probe = DeepSeekHarnessIntegrationService().versionOutput
    #expect(try await probe(spaced).trimmingCharacters(in: .whitespacesAndNewlines) == "0.2.1-rc.2")
}

@MainActor
@Test func officialHarnessMergesProfilesAndCredentialsBacksUpAndIsIdempotent() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installCLI()
    let original = """
    # Keep this original in a backup
    - id: unrelated
      config:
        enabled: true
    - id: llm-pi-ai
      config:
        timeout: 123
        providers:
          existing:
            baseURL: https://example.com/v1
          ironmlx-local:
            models:
              - id: previous/model
    - id: agent-default-model
      config:
        provider: existing
        model: old
        reasoningEffort: high
        customSetting: keep
    """
    try f.write(original, f.patch("web"))
    let credentialURL = f.home.appendingPathComponent(".credentials.yaml")
    let oldCredentials = "version: 1\nrefs:\n  EXISTING_KEY: preserved\nrecords:\n  custom:\n    value: keep\n"
    try f.write(oldCredentials, credentialURL)
    let result = await f.service(hasCLI: true).configure(mode: .cli, model: f.model, port: 9077)
    #expect(result.success)
    #expect(result.backupPaths.count == 2)
    #expect(!FileManager.default.fileExists(atPath: f.patch("desktop").path))
    for profile in ["web", "headless"] {
        let rows = try #require(Yams.load(yaml: String(contentsOf: f.patch(profile), encoding: .utf8)) as? [[String: Any]])
        let config = try #require(rows.first { $0["id"] as? String == "llm-pi-ai" }?["config"] as? [String: Any])
        let providers = try #require(config["providers"] as? [String: [String: Any]])
        let provider = try #require(providers["ironmlx-local"])
        #expect(provider["api"] as? String == "openai-responses")
        #expect(provider["transport"] as? String == "sse")
        #expect(provider["baseURL"] as? String == "http://127.0.0.1:9077/v1")
        let models = try #require(provider["models"] as? [[String: Any]])
        let selected = try #require(models.first { $0["id"] as? String == f.model.id })
        #expect(selected["contextWindow"] as? Int == 64000)
        #expect(selected["maxTokens"] as? Int == 4096)
        #expect(selected["reasoningEfforts"] as? Bool == false)
        let defaults = try #require(rows.first { $0["id"] as? String == "agent-default-model" }?["config"] as? [String: Any])
        #expect(defaults["provider"] as? String == "ironmlx-local")
        #expect(defaults["model"] as? String == f.model.id)
        #expect(defaults["reasoningEffort"] == nil)
        if profile == "web" {
            #expect(providers["existing"]?["baseURL"] as? String == "https://example.com/v1")
            #expect(models.contains { $0["id"] as? String == "previous/model" })
            #expect(config["timeout"] as? Int == 123)
            #expect(defaults["customSetting"] as? String == "keep")
        }
    }
    let document = try #require(Yams.load(yaml: String(contentsOf: credentialURL, encoding: .utf8)) as? [String: Any])
    #expect((document["refs"] as? [String: String])?["EXISTING_KEY"] == "preserved")
    #expect((document["refs"] as? [String: String])?[DeepSeekHarnessIntegrationService.credentialReference] == "local")
    #expect(document["records"] != nil)
    for path in result.configurationPaths + result.backupPaths {
        #expect(try FileManager.default.attributesOfItem(atPath: path)[.posixPermissions] as? Int == 0o600)
    }
    let backups = try result.backupPaths.map { try String(contentsOfFile: $0, encoding: .utf8) }
    #expect(backups.contains(original))
    #expect(backups.contains(oldCredentials))
    let second = await f.service(hasCLI: true).configure(mode: .cli, model: f.model, port: 9077)
    #expect(second.success)
    #expect(second.backupPaths.isEmpty)
    try f.installApp()
    let desktop = await f.service().configure(mode: .desktop, model: f.model, port: 9077)
    #expect(desktop.success)
    #expect(desktop.configurationPaths.contains(f.patch("desktop").path))
}

@MainActor
@Test func officialHarnessStopsForCustomHomeConflictsMigrationAndLocks() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installCLI()
    try f.installApp()
    var service = f.service(hasCLI: true)
    service.environment = ["DSH_HOME": f.root.appendingPathComponent("community-home").path]
    #expect(await service.configure(mode: .desktop, model: f.model, port: 9068).code == "custom_home")
    service.environment = [:]
    service.loginShellHome = { "/custom/home" }
    #expect(await service.configure(mode: .cli, model: f.model, port: 9068).code == "custom_home")
    #expect(!FileManager.default.fileExists(atPath: f.home.path))
    service = f.service(hasCLI: true)
    service.isDesktopRunning = { true }
    #expect(await service.configure(mode: .cli, model: f.model, port: 9068).code == "close_app")
    for (path, content, code) in [
        ("settings.yaml", "legacy: true", "initialize_app"),
        ("cordis.patch.yml", "- id: llm-pi-ai\n  config: {}", "home_conflict"),
        ("profiles/desktop/lock", "12345", "close_app"),
    ] {
        let url = f.home.appendingPathComponent(path)
        try f.write(content, url)
        #expect(await f.service().configure(mode: .desktop, model: f.model, port: 9068).code == code)
        #expect(!FileManager.default.fileExists(atPath: f.patch("desktop").path))
        #expect(try String(contentsOf: url, encoding: .utf8) == content)
        try FileManager.default.removeItem(at: url)
    }
}

@MainActor
@Test func officialHarnessPreflightsAllDocumentsAndRejectsSymlinkHomes() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installCLI()
    let web = "- id: untouched\n  config: {}\n"
    try f.write(web, f.patch("web"))
    for invalid in ["invalid: [", "- id: llm-pi-ai\n  config: nope", "- id: credentials\n  config: {path: /custom}", "- id: llm-pi-ai\n  disabled: true"] {
        try f.write(invalid, f.patch("headless"))
        #expect(await f.service(hasCLI: true).configure(mode: .cli, model: f.model, port: 9068).code == "configuration_failed")
        #expect(try String(contentsOf: f.patch("web"), encoding: .utf8) == web)
        #expect(!FileManager.default.fileExists(atPath: f.home.appendingPathComponent(".credentials.yaml").path))
    }
    try FileManager.default.removeItem(at: f.home)
    let community = f.root.appendingPathComponent("community-home")
    try FileManager.default.createDirectory(at: community, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(at: f.home, withDestinationURL: community)
    #expect(await f.service(hasCLI: true).configure(mode: .cli, model: f.model, port: 9068).code == "configuration_failed")
    #expect(try FileManager.default.contentsOfDirectory(atPath: community.path).isEmpty)
}

@Test func officialHarnessDownloadLinksAreSeparateFromCommunitySite() throws {
    for value in ["https://www.deepseek.com/zh/download/", "https://www.deepseek.com/en/download/", "https://github.com/deepseek-ai/deepseek-harness"] {
        #expect(DashboardBridge.isOfficialDSHInstallationURL(try #require(URL(string: value))))
    }
    for value in ["https://dshdesktop.com/en/", "http://www.deepseek.com/zh/download/", "https://www.deepseek.com.evil/zh/download/", "https://www.deepseek.com/en/download/?next=evil"] {
        #expect(!DashboardBridge.isOfficialDSHInstallationURL(try #require(URL(string: value))))
    }
}

@MainActor
private final class HarnessLinkRecorder: NSObject, WKScriptMessageHandler {
    var urls: [String] = []
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if let url = message.body as? String { urls.append(url) }
    }
}

@MainActor
@Test func officialHarnessDashboardSeparatesMissingCLIAndDesktopAndLocalizesLinks() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installApp(version: "0.12.1", id: "io.dsh.desktop")
    let store = AppConfigStore(url: f.root.appendingPathComponent("config.json"))
    let config = AppConfig(language: "zh-Hans")
    #expect(store.save(config))
    let content = WKUserContentController()
    content.addUserScript(WKUserScript(source: try DashboardWindowController.bootstrapScript(config: config, route: .status),
        injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = content
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1180, height: 850), configuration: configuration)
    DashboardThemeAppearance.apply("light", to: webView)
    let bridge = DashboardBridge(webView: webView, configStore: store, backend: TestRuntimeBackend(state: .stopped))
    if ProcessInfo.processInfo.environment["IRONMLX_TEST_OFFICIAL_HARNESS_NOT_INSTALLED"] == "1" {
        #expect(await DeepSeekHarnessIntegrationService().inspect(mode: .cli).code == "not_installed")
        #expect(await DeepSeekHarnessIntegrationService().inspect(mode: .desktop).code == "not_installed")
        #expect(DSHDesktopIntegrationService().inspect().success)
    } else { bridge.deepSeekHarnessService = f.service() }
    content.add(bridge, name: "fetchAPIPost")
    let links = HarnessLinkRecorder()
    content.add(links, name: "openAgentGuide")
    defer {
        content.removeScriptMessageHandler(forName: "fetchAPIPost")
        content.removeScriptMessageHandler(forName: "openAgentGuide")
    }
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    func wait(_ condition: String) async throws {
        for _ in 0..<120 {
            if try await webView.evaluateJavaScript(condition) as? Bool == true { return }
            try await Task.sleep(for: .milliseconds(50))
        }
        Issue.record("Dashboard condition timed out: \(condition)")
    }
    try await wait("document.readyState === 'complete' && typeof configureDeepSeekHarness === 'function' && typeof currentLang !== 'undefined'")
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); navigateTo('chat'); selectAgentGuide('dsh')")
    #expect(try await webView.evaluateJavaScript("document.querySelector('[data-dsh-setup-mode=desktop]').checked") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.querySelector('[data-dsh-setup-panel=desktop]').hidden") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.querySelector('[data-dsh-setup-panel=cli]').hidden") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.querySelector('#agent-guide-dsh [data-i18n=dsh_setup_method]').textContent") as? String == "选择应用")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-desktop-manual').open") as? Bool == false)
    _ = try await webView.evaluateJavaScript("document.getElementById('dsh-harness-configure').click()")
    try await wait("document.getElementById('dsh-harness-setup-status').textContent.includes('未检测到官方桌面版')")
    _ = try await webView.evaluateJavaScript("selectDshSetupMode('cli')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-feedback').hidden") as? Bool == true)
    _ = try await webView.evaluateJavaScript("document.getElementById('dsh-harness-configure').click()")
    try await wait("document.getElementById('dsh-harness-setup-status').textContent.includes('未检测到DSH CLI')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-install').href") as? String == "https://github.com/deepseek-ai/deepseek-harness")
    _ = try await webView.evaluateJavaScript("selectDshSetupMode('desktop')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-setup-status').textContent.includes('未检测到官方桌面版')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-feedback').dataset.tone") as? String == "warning")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-setup-status').textContent.includes('未写入任何配置')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-harness-configure').disabled") as? Bool == false)
    for language in ["zh-Hans", "zh-Hant", "en", "ja", "ko"] {
        _ = try await webView.evaluateJavaScript("setLanguage('\(language)'); document.getElementById('dsh-harness-install').click()")
        try await Task.sleep(for: .milliseconds(50))
        let url = "https://www.deepseek.com/" + (language.hasPrefix("zh") ? "zh" : "en") + "/download/"
        #expect(links.urls.last == url)
    }
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); document.getElementById('dsh-harness-feedback').scrollIntoView({block:'center'})")
    if let path = ProcessInfo.processInfo.environment["IRONMLX_TEST_HARNESS_SCREENSHOT"] {
        try await Task.sleep(for: .milliseconds(150))
        let image = try await webView.takeSnapshot(configuration: nil)
        let bitmap = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
        try bitmap.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
    }
    #expect(!FileManager.default.fileExists(atPath: f.home.path))
}

@MainActor
@Test func officialHarnessRestoresEarlierWritesWhenALaterFileFails() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installCLI()
    let original = "- id: unrelated\n  config: {}\n"
    try f.write(original, f.patch("web"))
    // A directory-shaped path occupied by a file fails after the earlier Web write.
    try f.write("blocked", f.home.appendingPathComponent("profiles/headless"))
    let result = await f.service(hasCLI: true).configure(mode: .cli, model: f.model, port: 9068)
    #expect(!result.success)
    #expect(result.code == "configuration_failed")
    #expect(try String(contentsOf: f.patch("web"), encoding: .utf8) == original)
    #expect(!FileManager.default.fileExists(atPath: f.home.appendingPathComponent(".credentials.yaml").path))
}

@MainActor
@Test func harnessChatPreflightNeverWritesAndOpensTheResolvedApplication() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    var service = HarnessChatService(integration: f.service(), runningApplications: { [] },
        loadedModelIDs: { _ in ["test/model"] }, openApplication: { _ in Issue.record("unexpected launch") })
    var config = AppConfig()
    #expect(await service.open(config: config, backendState: .running) == .notInstalled)
    #expect(!FileManager.default.fileExists(atPath: f.home.path))
    try f.installApp()
    #expect(await service.open(config: config, backendState: .stopped) == .backendUnavailable)
    #expect(await service.open(config: config, backendState: .running) == .configurationRequired)
    #expect(!FileManager.default.fileExists(atPath: f.home.path))
    let result = await f.service().configure(mode: .desktop, model: f.model, port: config.port)
    #expect(result.success)
    let patchBefore = try Data(contentsOf: f.patch("desktop"))
    var opened: [URL] = []
    service.openApplication = { opened.append($0) }
    #expect(await service.open(config: config, backendState: .running) == nil)
    #expect(opened == [f.app])
    #expect(try Data(contentsOf: f.patch("desktop")) == patchBefore)
    config.port += 1
    #expect(await service.open(config: config, backendState: .running) == .configurationRequired)
    config.port -= 1
    service.loadedModelIDs = { _ in [] }
    #expect(await service.open(config: config, backendState: .running) == .modelUnavailable)
    service.loadedModelIDs = { _ in ["other/model"] }
    #expect(await service.open(config: config, backendState: .running) == .configurationRequired)
    service.loadedModelIDs = { _ in throw URLError(.cannotConnectToHost) }
    #expect(await service.open(config: config, backendState: .running) == .backendUnavailable)
    #expect(opened.count == 1)
}

@MainActor
@Test func harnessChatPreservesRunningCopyOwnershipAndReportsLaunchFailure() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installApp()
    let config = AppConfig()
    #expect(await f.service().configure(mode: .desktop, model: f.model, port: config.port).success)
    var service = HarnessChatService(integration: f.service(), runningApplications: { [] },
        loadedModelIDs: { _ in ["test/model"] }, openApplication: { _ in throw URLError(.fileDoesNotExist) })
    #expect(await service.open(config: config, backendState: .running) == .launchFailed)
    let other = f.root.appendingPathComponent("old/DeepSeek Harness.app")
    try f.write(#"<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>com.deepseek.dsh</string><key>CFBundleShortVersionString</key><string>0.1.0</string></dict></plist>"#,
        other.appendingPathComponent("Contents/Info.plist"))
    service.runningApplications = { [other] }
    #expect(await service.open(config: config, backendState: .running) == .incompatibleVersion)
    service.runningApplications = { [f.app] }
    var opened: URL?
    service.openApplication = { opened = $0 }
    #expect(await service.open(config: config, backendState: .running) == nil)
    #expect(opened == f.app)
}

@MainActor
@Test func harnessChatRejectsCredentialAndHomeOverridesWithoutModifyingThem() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installApp()
    let config = AppConfig()
    var integration = f.service()
    #expect(await integration.configure(mode: .desktop, model: f.model, port: config.port).success)
    #expect(integration.desktopConfigurationIsReady(modelIDs: [f.model.id], port: config.port))
    integration.environment = ["DSH_HOME": "/tmp/custom-harness"]
    #expect(!integration.desktopConfigurationIsReady(modelIDs: [f.model.id], port: config.port))
    integration.environment = [:]
    try f.write("version: 1\nrefs: {}\n", f.home.appendingPathComponent(".credentials.yaml"))
    #expect(!integration.desktopConfigurationIsReady(modelIDs: [f.model.id], port: config.port))
}

@MainActor
@Test(.enabled(if: ProcessInfo.processInfo.environment["IRONMLX_TEST_HARNESS_CHAT_LIVE"] == "1"))
func harnessChatOpensExistingConfiguredDesktopAgainstLiveLocalService() async throws {
    let service = HarnessChatService()
    let issue = await service.open(config: AppConfig(), backendState: .running)
    #expect(issue == nil)
    if issue == nil {
        #expect(!NSRunningApplication.runningApplications(
            withBundleIdentifier: DeepSeekHarnessIntegrationService.bundleIdentifier).isEmpty)
    }
}

@MainActor
@Test func harnessMenuLaunchForwardsToServiceAndIgnoresDuplicateClicks() async throws {
    let f = HarnessFixture()
    defer { f.clean() }
    try f.installApp()
    let store = AppConfigStore(url: f.root.appendingPathComponent("app_config.json"))
    let config = store.load()
    #expect(await f.service().configure(mode: .desktop, model: f.model, port: config.port).success)
    let backend = TestRuntimeBackend(state: .running, isRunning: true)
    let dashboard = DashboardWindowController(configStore: store, backend: backend)
    let controller = MenuBarController(configStore: store, backend: backend, dashboard: dashboard)
    var opened: [URL] = []
    controller.harnessChatService = HarnessChatService(integration: f.service(), runningApplications: { [] },
        loadedModelIDs: { _ in ["test/model"] }, openApplication: { opened.append($0) })
    controller.chatWithDSH(NSMenuItem())
    controller.chatWithDSH(NSMenuItem())
    for _ in 0..<50 {
        if !opened.isEmpty { break }
        try await Task.sleep(for: .milliseconds(10))
    }
    #expect(opened == [f.app])
}
