import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
private final class LogLevelMessages: NSObject, WKScriptMessageHandler {
    var messages: [(String, String)] = []
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        messages.append((message.name, message.body as? String ?? ""))
    }
}

@MainActor
@Test func dashboardLogLevelAppliesOnlyItsOwnSettingAndRestoresFailedSelection() async throws {
    let controller = WKUserContentController()
    let messages = LogLevelMessages()
    for name in ["setLogLevel", "saveSettings"] { controller.add(messages, name: name) }
    defer { controller.removeAllScriptMessageHandlers() }
    controller.addUserScript(WKUserScript(
        source: try DashboardWindowController.bootstrapScript(config: AppConfig(logLevel: "INFO"), route: .status),
        injectionTime: .atDocumentStart, forMainFrameOnly: true
    ))
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = controller
    let view = WKWebView(frame: NSRect(x: 0, y: 0, width: 1000, height: 700), configuration: configuration)
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    view.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    let deadline = Date().addingTimeInterval(12)
    while Date() < deadline {
        if (try? await view.evaluateJavaScript("typeof logLevelPending !== 'undefined' && typeof settingsBaseline !== 'undefined' && settingsBaseline !== null")) as? Bool == true { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    _ = try await view.evaluateJavaScript("""
      navigateTo('settings');
      document.getElementById('cfg-port').value = '9077';
      const control = document.getElementById('cfg-log-level');
      control.value = 'DEBUG'; control.dispatchEvent(new Event('change'));
    """)
    try await Task.sleep(for: .milliseconds(100))
    #expect(messages.messages.count == 1)
    #expect(messages.messages.first?.0 == "setLogLevel")
    #expect(messages.messages.first?.1 == "DEBUG")
    #expect(try await view.evaluateJavaScript("document.getElementById('cfg-log-level').disabled") as? Bool == true)
    _ = try await view.evaluateJavaScript("onLogLevelChanged(JSON.stringify({success:true,level:'DEBUG'}))")
    #expect(try await view.evaluateJavaScript("window.__IRONMLX_APP_CONFIG__.port") as? Int == 9068)
    #expect(try await view.evaluateJavaScript("document.getElementById('cfg-port').value") as? String == "9077")
    _ = try await view.evaluateJavaScript("onLogLevelChanged(JSON.stringify({success:false,level:'DEBUG',code:'log_level_apply_failed'}))")
    #expect(try await view.evaluateJavaScript("document.getElementById('cfg-log-level').value") as? String == "DEBUG")
    _ = try await view.evaluateJavaScript("saveSettings(); void 0")
    try await Task.sleep(for: .milliseconds(100))
    let saved = try #require(messages.messages.last)
    #expect(saved.0 == "saveSettings")
    let body = try #require(JSONSerialization.jsonObject(with: Data(saved.1.utf8)) as? [String: Any])
    #expect(body["log_level"] == nil)
    #expect(body["port"] as? Int == 9077)
}
