import AppKit
import Foundation
import Testing
import WebKit
import Yams

@testable import IronMLXAppCore

@MainActor
private final class DSHWebsiteLinkRecorder: NSObject, WKScriptMessageHandler {
    var urls: [String] = []
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if let url = message.body as? String { urls.append(url) }
    }
}

@Test func dshDesktopWebsiteBridgeAllowsChineseAndEnglishDestinations() throws {
    for path in ["zh", "en"] {
        #expect(DashboardBridge.isDSHDesktopDownloadURL(try #require(URL(string: "https://dshdesktop.com/\(path)/"))))
    }
    for value in ["http://dshdesktop.com/zh/", "https://dshdesktop.com/other/",
                  "https://dshdesktop.com.example.org/zh/", "https://user@dshdesktop.com/zh/"] {
        #expect(!DashboardBridge.isDSHDesktopDownloadURL(try #require(URL(string: value))))
    }
}

@MainActor
private struct DSHFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    var home: URL { root.appendingPathComponent("harness") }
    var app: URL { root.appendingPathComponent("DSH Desktop.app") }
    var patch: URL { home.appendingPathComponent("profiles/web/cordis.patch.yml") }
    var credentials: URL { home.appendingPathComponent(".credentials.yaml") }
    var model: DSHDesktopIntegrationService.Model {
        .init(id: "test/model", contextWindow: 64000, maxTokens: 8192)
    }
    func install(version: String = "0.12.1", identifier: String = "io.dsh.desktop") throws {
        let contents = app.appendingPathComponent("Contents")
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        let data = try PropertyListSerialization.data(fromPropertyList: [
            "CFBundleIdentifier": identifier, "CFBundleShortVersionString": version,
        ], format: .xml, options: 0)
        try data.write(to: contents.appendingPathComponent("Info.plist"))
    }
    func service() -> DSHDesktopIntegrationService {
        DSHDesktopIntegrationService(applicationURLs: { [app] }, isRunning: { false }, harnessHome: home)
    }
    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }
    func clean() { try? FileManager.default.removeItem(at: root) }
}

@MainActor
@Test func dshDesktopInstallationGatesNeverCreateConfiguration() throws {
    let fixture = DSHFixture()
    defer { fixture.clean() }
    let service = fixture.service()
    #expect(service.configure(model: fixture.model, port: 9068).code == "not_installed")
    for identifier in ["com.deepseek.desktop", "io.dsh.desktop.dev"] {
        try fixture.install(identifier: identifier)
        #expect(service.inspect().code == "not_installed")
    }
    for (version, code) in [("0.12.0", "version_too_old"), ("invalid", "version_unknown"),
                            ("0.12.1-beta.1", "version_unknown")] {
        try fixture.install(version: version)
        let result = service.configure(model: fixture.model, port: 9068)
        #expect(!result.success)
        #expect(result.code == code)
        #expect(result.installedVersion == version)
        #expect(result.downloadURL == "https://dshdesktop.com/en/")
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.home.path))
    for version in ["0.12.1", "0.12.10", "0.13.0", "1.0.0"] {
        #expect(DSHDesktopIntegrationService.supports(version))
    }
    #expect(!DSHDesktopIntegrationService.supports("0.9.99"))
}

@MainActor
@Test func dshDesktopConfigurationMergesBacksUpAndIsIdempotent() throws {
    let fixture = DSHFixture()
    defer { fixture.clean() }
    try fixture.install()
    let oldPatch = """
    # User settings
    - id: unrelated
      config:
        enabled: true
    - id: llm-pi-ai
      config:
        timeout: 123
        providers:
          existing:
            baseURL: https://example.com/v1
            api: openai-responses
    """
    let oldCredentials = "version: 1\nrefs:\n  EXISTING_KEY: preserved\n"
    try fixture.write(oldPatch, to: fixture.patch)
    try fixture.write(oldCredentials, to: fixture.credentials)
    let result = fixture.service().configure(model: fixture.model, port: 9077)
    #expect(result.success)
    #expect(result.code == "configured")
    #expect(result.backupPaths.count == 2)
    let backups = try result.backupPaths.map { try String(contentsOfFile: $0, encoding: .utf8) }
    #expect(backups.contains(oldPatch))
    #expect(backups.contains(oldCredentials))
    let rows = try #require(Yams.load(yaml: String(contentsOf: fixture.patch, encoding: .utf8)) as? [[String: Any]])
    #expect(rows[0]["id"] as? String == "unrelated")
    let config = try #require(rows[1]["config"] as? [String: Any])
    #expect(config["timeout"] as? Int == 123)
    let providers = try #require(config["providers"] as? [String: [String: Any]])
    #expect(providers["existing"]?["baseURL"] as? String == "https://example.com/v1")
    let ironmlx = try #require(providers["ironmlx-local"])
    #expect(ironmlx["baseURL"] as? String == "http://127.0.0.1:9077/v1")
    #expect(ironmlx["api"] as? String == "openai-responses")
    let models = try #require(ironmlx["models"] as? [[String: Any]])
    #expect(models[0]["contextWindow"] as? Int == 64000)
    #expect(models[0]["maxTokens"] as? Int == 4096)
    let credentials = try #require(Yams.load(yaml: String(contentsOf: fixture.credentials, encoding: .utf8)) as? [String: Any])
    let refs = try #require(credentials["refs"] as? [String: String])
    #expect(refs["EXISTING_KEY"] == "preserved")
    #expect(refs[DSHDesktopIntegrationService.credentialReference] == "local")
    #expect(try FileManager.default.attributesOfItem(atPath: fixture.credentials.path)[.posixPermissions] as? Int == 0o600)
    let before = try Data(contentsOf: fixture.patch)
    let second = fixture.service().configure(model: fixture.model, port: 9077)
    #expect(second.success)
    #expect(second.backupPaths.isEmpty)
    #expect(try Data(contentsOf: fixture.patch) == before)
}

@MainActor
@Test func dshDesktopFreshConfigurationAndLegacyMigrationGate() throws {
    let fixture = DSHFixture()
    defer { fixture.clean() }
    try fixture.install()
    let result = fixture.service().configure(model: fixture.model, port: 9068)
    #expect(result.success)
    #expect(result.backupPaths.isEmpty)
    #expect(FileManager.default.fileExists(atPath: fixture.patch.path))
    #expect(FileManager.default.fileExists(atPath: fixture.credentials.path))
    let patch = try Data(contentsOf: fixture.patch)
    try fixture.write("legacy: true", to: fixture.home.appendingPathComponent("settings.yaml"))
    #expect(fixture.service().configure(model: fixture.model, port: 9068).code == "initialize_app")
    #expect(try Data(contentsOf: fixture.patch) == patch)
}

@MainActor
@Test func dshDesktopInvalidConfigurationIsNotOverwritten() throws {
    let fixture = DSHFixture()
    defer { fixture.clean() }
    try fixture.install()
    for invalid in ["not: a-sequence", "- id: llm-pi-ai\n  config: invalid", "- id: ["] {
        try fixture.write(invalid, to: fixture.patch)
        let result = fixture.service().configure(model: fixture.model, port: 9068)
        #expect(!result.success)
        #expect(result.code == "configuration_failed")
        #expect(try String(contentsOf: fixture.patch, encoding: .utf8) == invalid)
        #expect(!FileManager.default.fileExists(atPath: fixture.credentials.path))
    }
    try fixture.write("[]", to: fixture.patch)
    try fixture.write("version: 2\nrefs: {}", to: fixture.credentials)
    #expect(fixture.service().configure(model: fixture.model, port: 9068).code == "configuration_failed")
    #expect(try String(contentsOf: fixture.patch, encoding: .utf8) == "[]")
    var running = fixture.service()
    running.isRunning = { true }
    #expect(running.configure(model: fixture.model, port: 9068).code == "close_app")
}

@MainActor
@Test func dshDesktopDashboardShowsInstallPromptWithoutBackendOrModel() async throws {
    let fixture = DSHFixture()
    defer { fixture.clean() }
    let store = AppConfigStore(url: fixture.root.appendingPathComponent("config.json"))
    let config = AppConfig(language: "zh-Hans")
    #expect(store.save(config))
    let content = WKUserContentController()
    content.addUserScript(WKUserScript(
        source: try DashboardWindowController.bootstrapScript(config: config, route: .status),
        injectionTime: .atDocumentStart, forMainFrameOnly: true
    ))
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = content
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1180, height: 780), configuration: configuration)
    DashboardThemeAppearance.apply("light", to: webView)
    let bridge = DashboardBridge(webView: webView, configStore: store,
                                 backend: TestRuntimeBackend(state: .stopped))
    let live = ProcessInfo.processInfo.environment["IRONMLX_TEST_DSH_DESKTOP_NOT_INSTALLED"] == "1"
    if live {
        #expect(DSHDesktopIntegrationService().inspect().code == "not_installed")
    } else {
        bridge.dshDesktopService = fixture.service()
    }
    content.add(bridge, name: "fetchAPIPost")
    let links = DSHWebsiteLinkRecorder()
    content.add(links, name: "openAgentGuide")
    defer {
        content.removeScriptMessageHandler(forName: "fetchAPIPost")
        content.removeScriptMessageHandler(forName: "openAgentGuide")
    }
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    try await waitForDSHUI(webView, "document.readyState === 'complete' && typeof configureDSHDesktop === 'function' && typeof currentLang !== 'undefined'")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-feedback').hidden") as? Bool == true)
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); navigateTo('chat'); selectAgentGuide('dsh-desktop'); document.getElementById('dsh-desktop-configure').click()")
    try await waitForDSHUI(webView, "document.getElementById('dsh-desktop-setup-status').textContent.includes('未检测到社区版')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-download').hidden") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-download').href") as? String == "https://dshdesktop.com/zh/")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-setup-status').textContent.includes('未写入任何配置')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-configure').disabled") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.querySelector('.nav-item[data-page=chat]').classList.contains('active')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.querySelector('[data-agent-guide=dsh-desktop]').classList.contains('active')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-feedback').dataset.tone") as? String == "warning")
    #expect(try await webView.evaluateJavaScript("document.querySelector('#dsh-desktop-download [data-i18n]').textContent") as? String == "安装 DSH Desktop")
    _ = try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-download').click()")
    try await Task.sleep(for: .milliseconds(100))
    #expect(links.urls.last == "https://dshdesktop.com/zh/")
    let lightBackground = try #require(try await webView.evaluateJavaScript("getComputedStyle(document.getElementById('dsh-desktop-feedback')).backgroundColor") as? String)
    #expect(try await webView.evaluateJavaScript("getComputedStyle(document.getElementById('dsh-desktop-setup-status')).color !== getComputedStyle(document.querySelector('#agent-guide-dsh-desktop .agent-guide-section-desc')).color") as? Bool == true)
    if live { try await captureDSHFeedback(webView, name: "light") }
    DashboardThemeAppearance.apply("dark", to: webView)
    try await waitForDSHUI(webView, "window.matchMedia('(prefers-color-scheme: dark)').matches")
    let darkBackground = try #require(try await webView.evaluateJavaScript("getComputedStyle(document.getElementById('dsh-desktop-feedback')).backgroundColor") as? String)
    #expect(darkBackground != lightBackground)
    if live { try await captureDSHFeedback(webView, name: "dark") }
    DashboardThemeAppearance.apply("light", to: webView)
    for (language, destination) in [("zh-Hans", "zh"), ("zh-Hant", "zh"), ("en", "en"), ("ja", "en"), ("ko", "en")] {
        _ = try await webView.evaluateJavaScript("setLanguage('\(language)'); document.getElementById('dsh-desktop-download').click()")
        try await Task.sleep(for: .milliseconds(50))
        #expect(links.urls.last == "https://dshdesktop.com/\(destination)/")
        #expect(try await webView.evaluateJavaScript("Array.from(document.querySelectorAll('[data-dsh-desktop-website]')).every(link => link.href === 'https://dshdesktop.com/\(destination)/')") as? Bool == true)
    }
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans')")
    #expect(!FileManager.default.fileExists(atPath: fixture.home.path))
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-manual').open") as? Bool == false)
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-verify').closest('details') === null") as? Bool == true)
    _ = try await webView.evaluateJavaScript("document.querySelector('#dsh-desktop-manual summary').click()")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-manual').open") as? Bool == true)
    _ = try await webView.evaluateJavaScript("document.querySelector('#dsh-desktop-manual summary').click()")
    if live {
        _ = try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-configure').scrollIntoView({block:'center'})")
        try await waitForDSHUI(webView, "document.getElementById('page-chat').classList.contains('active') && document.getElementById('dsh-desktop-setup-status').getBoundingClientRect().height > 0")
        try await Task.sleep(for: .milliseconds(150))
        let image = try await webView.takeSnapshot(configuration: nil)
        let bitmap = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
        try bitmap.representation(using: .png, properties: [:])?.write(
            to: URL(fileURLWithPath: "/tmp/ironmlx-dsh-uninstalled.png")
        )
    }
    // Official Desktop has a status placeholder; community configuration stays in its own panel.
    _ = try await webView.evaluateJavaScript("selectAgentGuide('dsh'); selectDshSetupMode('desktop')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('agent-guide-dsh-desktop').hidden") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.querySelector('#agent-guide-dsh [data-dsh-setup-panel=desktop]').textContent.includes('尚未适配')") as? Bool == true)
    #expect(try await webView.evaluateJavaScript("document.querySelector('#agent-guide-dsh #dsh-desktop-configure') === null") as? Bool == true)
    _ = try await webView.evaluateJavaScript("selectDshSetupMode('cli')")
    #expect(try await webView.evaluateJavaScript("document.querySelector('#agent-guide-dsh [data-dsh-setup-panel=cli]').hidden") as? Bool == false)
    _ = try await webView.evaluateJavaScript("selectAgentGuide('dsh-desktop')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('agent-guide-dsh-desktop').hidden") as? Bool == false)
    _ = try await webView.evaluateJavaScript("setLanguage('en')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-setup-status').textContent.includes('not installed')") as? Bool == true)
    // The retry repeats native discovery and blocks an outdated installation too.
    try fixture.install(version: "0.12.0")
    bridge.dshDesktopService = fixture.service()
    _ = try await webView.evaluateJavaScript("setLanguage('zh-Hans'); document.getElementById('dsh-desktop-configure').click()")
    try await waitForDSHUI(webView, "document.getElementById('dsh-desktop-setup-status').textContent.includes('低于要求的 0.12.1')")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-manual').open") as? Bool == false)
    #expect(!FileManager.default.fileExists(atPath: fixture.home.path))
    _ = try await webView.evaluateJavaScript("dshDesktopSetupResult = {success:true,code:'configured'}; renderDSHDesktopSetupResult()")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-feedback').dataset.tone") as? String == "success")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-download').hidden") as? Bool == true)
    _ = try await webView.evaluateJavaScript("dshDesktopSetupResult = {success:false,code:'configuration_failed'}; renderDSHDesktopSetupResult()")
    #expect(try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-feedback').dataset.tone") as? String == "error")
}

@MainActor
private func captureDSHFeedback(_ webView: WKWebView, name: String) async throws {
    _ = try await webView.evaluateJavaScript("document.getElementById('dsh-desktop-feedback').scrollIntoView({block:'center'})")
    try await Task.sleep(for: .milliseconds(150))
    let bounds = try #require(try await webView.evaluateJavaScript("(() => {const r=document.getElementById('dsh-desktop-feedback').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};})()") as? [String: Double])
    let snapshot = WKSnapshotConfiguration()
    snapshot.rect = NSRect(x: bounds["x"] ?? 0, y: bounds["y"] ?? 0,
                           width: bounds["width"] ?? 1, height: bounds["height"] ?? 1)
    let image = try await webView.takeSnapshot(configuration: snapshot)
    let bitmap = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
    try bitmap.representation(using: .png, properties: [:])?.write(
        to: URL(fileURLWithPath: "/tmp/ironmlx-dsh-feedback-\(name).png")
    )
}

@MainActor
private func waitForDSHUI(_ webView: WKWebView, _ expression: String) async throws {
    let deadline = Date().addingTimeInterval(15)
    while Date() < deadline {
        if (try? await webView.evaluateJavaScript(expression)) as? Bool == true { return }
        try await Task.sleep(for: .milliseconds(100))
    }
    Issue.record("DSH Desktop UI did not converge: \(expression)")
    throw DSHUIWaitError.timeout
}
private enum DSHUIWaitError: Error { case timeout }
