import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
@Test func dashboardRefreshesLANInterfacesWithoutReopeningOrRunningBackend() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let configStore = AppConfigStore(url: root.appendingPathComponent("config.json"))
    let config = AppConfig(networkMode: "lan", lanHost: "192.168.0.106")
    configStore.save(config)
    var interfaces = [EndpointPayload.NetworkInterface(name: "en0", ip: "192.168.0.106")]
    let controller = WKUserContentController()
    let bootstrap = try DashboardWindowController.bootstrapScript(config: config, route: .status)
    controller.addUserScript(WKUserScript(
        source: bootstrap + "\nwindow.__IRONMLX_NETWORK_INTERFACES__ = [{name:'en0',ip:'192.168.0.106',is_thunderbolt:false}];",
        injectionTime: .atDocumentStart,
        forMainFrameOnly: true
    ))
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = controller
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1100, height: 800), configuration: configuration)
    let bridge = DashboardBridge(
        webView: webView,
        configStore: configStore,
        backend: TestRuntimeBackend(state: .stopped),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("params.json")),
        notificationCenter: NotificationCenter(),
        networkInterfacesProvider: { interfaces }
    )
    controller.add(bridge, name: "fetchAPI")
    defer { controller.removeScriptMessageHandler(forName: "fetchAPI") }
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    try await waitForLANUI(webView, "document.getElementById('cfg-lan-host')?.value === '192.168.0.106'")

    _ = try await webView.evaluateJavaScript("navigateTo('settings')")

    // The existing five-second endpoint poll must refresh an already-open page.
    interfaces = [EndpointPayload.NetworkInterface(name: "en0", ip: "192.168.0.104")]
    try await waitForLANUI(webView, "Array.from(document.getElementById('cfg-lan-host').options).some(o => o.value === '192.168.0.104')")
    let staleRemoved = try await webView.evaluateJavaScript("!Array.from(document.getElementById('cfg-lan-host').options).some(o => o.value === '192.168.0.106')") as? Bool
    #expect(staleRemoved == true)
    #expect(try await webView.evaluateJavaScript("document.getElementById('cfg-lan-host').value") as? String == "")

    // Keep unsaved edits, including another valid interface selected by the user.
    _ = try await webView.evaluateJavaScript("document.getElementById('cfg-lan-host').value = '192.168.0.104'; document.getElementById('cfg-port').value = '9077';")
    interfaces.append(EndpointPayload.NetworkInterface(name: "en1", ip: "192.168.0.105"))
    _ = try await webView.evaluateJavaScript("navigateTo('settings')")
    try await waitForLANUI(webView, "document.getElementById('cfg-lan-host').options.length === 3")
    #expect(try await webView.evaluateJavaScript("document.getElementById('cfg-lan-host').value") as? String == "192.168.0.104")
    #expect(try await webView.evaluateJavaScript("document.getElementById('cfg-port').value") as? String == "9077")
    #expect(configStore.load().lanHost == "192.168.0.106")

    interfaces = []
    _ = try await webView.evaluateJavaScript("refreshNetworkInterfaces()")
    try await waitForLANUI(webView, "document.getElementById('cfg-lan-host').options.length === 1 && document.getElementById('cfg-lan-host').value === ''")
    interfaces = [EndpointPayload.NetworkInterface(name: "en0", ip: "192.168.0.104")]
    _ = try await webView.evaluateJavaScript("refreshNetworkInterfaces()")
    try await waitForLANUI(webView, "document.getElementById('cfg-lan-host').options.length === 2")
    #expect(try await webView.evaluateJavaScript("document.getElementById('cfg-lan-host').value") as? String == "")
}

@MainActor
private func waitForLANUI(_ webView: WKWebView, _ expression: String) async throws {
    let deadline = Date().addingTimeInterval(12)
    while Date() < deadline {
        if (try? await webView.evaluateJavaScript(expression)) as? Bool == true { return }
        try await Task.sleep(for: .milliseconds(100))
    }
    Issue.record("LAN interface UI did not converge: \(expression)")
    throw LANUIWaitError.timeout
}

private enum LANUIWaitError: Error { case timeout }
