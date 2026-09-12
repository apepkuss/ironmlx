import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
private final class SettingsSaveMessages: NSObject, WKScriptMessageHandler {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {}
}

@MainActor
@Test func dashboardSettingsSaveBarTracksChanges() async throws {
    let configuration = WKWebViewConfiguration()
    let messages = SettingsSaveMessages()
    configuration.userContentController.add(messages, name: "saveSettings")
    configuration.userContentController.add(messages, name: "setLogLevel")
    defer { configuration.userContentController.removeAllScriptMessageHandlers() }
    let bootstrap = try DashboardWindowController.bootstrapScript(config: AppConfig(), route: .status)
    configuration.userContentController.addUserScript(WKUserScript(source: bootstrap, injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1000, height: 650), configuration: configuration)
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    webView.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    let deadline = Date().addingTimeInterval(15)
    while Date() < deadline {
        if (try? await webView.evaluateJavaScript("typeof settingsBaseline !== 'undefined' && settingsBaseline !== null")) as? Bool == true { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    let result = try await webView.evaluateJavaScript("""
    (() => {
      navigateTo('settings'); setLanguage('zh');
      const button = document.getElementById('settings-save-button');
      const status = document.getElementById('settings-save-status');
      const scroll = document.querySelector('.settings-page-scroll');
      const port = document.getElementById('cfg-port');
      const initial = port.value;
      let count = 0;
      const check = (ok, name) => { if (!ok) throw new Error(name); count++; };
      check(button.disabled, 'clean');
      onLogLevelChanged(JSON.stringify({success:true,level:'DEBUG'}));
      check(button.disabled && !status.textContent.includes('重启'), 'live level stays clean');
      settingsBackendRunning = true;
      port.value = '9077'; port.dispatchEvent(new Event('input'));
      check(button.textContent === '保存并重启服务' && !button.disabled && status.textContent.includes('有未保存的更改'), 'restart');
      check(!document.getElementById('toast-msg').classList.contains('show'), 'no dirty toast');
      scroll.scrollTop = scroll.scrollHeight;
      const bottom = button.getBoundingClientRect();
      scroll.scrollTop = 0;
      const top = button.getBoundingClientRect();
      check(Math.abs(top.y-bottom.y)<1 && top.bottom<=innerHeight && top.top>=0, 'visible');
      port.value = initial; checkSettingsDirty();
      check(button.disabled, 'reverted');
      document.getElementById('cfg-verify-model-on-load').click();
      check(!button.disabled && button.textContent === '保存设置', 'non restart');
      snapshotSettingsBaseline(); port.value = '9077'; checkSettingsDirty();
      saveSettings();
      check(button.disabled && status.textContent === '正在保存并重启服务…', 'saving');
      port.value = '9088'; onSettingsSaved(JSON.stringify({status:'ok',restarted:true}));
      check(!button.disabled && settingsBaseline['cfg-port'] === '9077', 'inflight edits');
      saveSettings();
      onSettingsSaved(JSON.stringify({status:'error',code:'settings_invalid'}));
      check(!button.disabled && status.textContent.includes('未保存'), 'retry');
      const failure = status.textContent;
      checkSettingsDirty();
      check(status.textContent === failure, 'failure stays until resolved');
      port.value = settingsBaseline['cfg-port']; checkSettingsDirty();
      check(button.disabled && !status.textContent.includes('重启'), 'failure reverted');
      port.value = '9088'; checkSettingsDirty();
      settingsBackendRunning = false; checkSettingsDirty();
      check(button.textContent === '保存设置' && status.textContent.includes('下次启动'), 'stopped');
      setLanguage('en');
      check(status.textContent.includes('next time'), 'language');
      const cards = [...document.querySelectorAll('#page-settings .card-settings')].filter(c => c.querySelector('[data-i18n="resource_mgmt"], [data-i18n="cache"], [data-i18n="advanced"]'));
      check(cards.length===3 && !document.querySelector('#page-settings .badge-restart'), 'no restart badges');
      return count;
    })()
    """)
    #expect(result as? Int == 15)
    _ = try await webView.evaluateJavaScript("settingsBackendRunning = true; checkSettingsDirty(); window.__persistentReminder = document.getElementById('settings-save-status').textContent; void 0")
    try await Task.sleep(for: .milliseconds(4500))
    #expect(try await webView.evaluateJavaScript("document.getElementById('settings-save-status').textContent === window.__persistentReminder") as? Bool == true)
    _ = try await webView.evaluateJavaScript("setLanguage('zh'); void 0")
    if let output = ProcessInfo.processInfo.environment["IRONMLX_SETTINGS_SCREENSHOT"] {
        let image = try await webView.takeSnapshot(configuration: nil)
        if let data = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: data),
           let png = bitmap.representation(using: .png, properties: [:]) {
            try png.write(to: URL(fileURLWithPath: output))
        }
    }
    _ = try await webView.evaluateJavaScript("saveSettings(); onSettingsSaved(JSON.stringify({status:'ok',restarted:true})); void 0")
    #expect(try await webView.evaluateJavaScript("document.getElementById('settings-save-button').disabled && !document.getElementById('settings-save-status').textContent.includes('重启')") as? Bool == true)

}
