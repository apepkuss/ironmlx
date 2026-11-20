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
      const bar = document.getElementById('settings-save-bar');
      check(button.disabled && getComputedStyle(bar).display === 'none', 'clean bar hidden');
      onLogLevelChanged(JSON.stringify({success:true,level:'DEBUG'}));
      check(button.disabled && !status.textContent.includes('重启'), 'live level stays clean');
      settingsBackendRunning = true;
      port.value = '9077'; port.dispatchEvent(new Event('input'));
      check(button.textContent === '保存并重启服务' && !button.disabled && status.textContent.startsWith('1 项改动') && status.textContent.includes('保存后会重启后端') && getComputedStyle(bar).display !== 'none' && bar.dataset.restart === 'true', 'restart');
      check(!document.getElementById('toast-msg').classList.contains('show'), 'no dirty toast');
      scroll.scrollTop = scroll.scrollHeight;
      const bottom = button.getBoundingClientRect();
      scroll.scrollTop = 0;
      const top = button.getBoundingClientRect();
      check(Math.abs(top.y-bottom.y)<1 && top.bottom<=innerHeight && top.top>=0, 'visible');
      port.value = initial; checkSettingsDirty();
      check(button.disabled, 'reverted');
      document.getElementById('cfg-verify-model-on-load').click();
      check(!button.disabled && button.textContent === '保存设置' && status.textContent === '1 项改动' && bar.dataset.restart === 'false', 'non restart');
      snapshotSettingsBaseline();
      const memAuto = document.getElementById('cfg-mem-total-auto');
      memAuto.click();
      check(status.textContent.startsWith('1 项改动') && !status.textContent.includes('2 项'), 'auto and slider count once');
      memAuto.click();
      check(button.disabled, 'auto restored');
      const backend = document.getElementById('cfg-distributed-backend');
      backend.value = 'ring'; backend.dispatchEvent(new Event('change'));
      check(status.textContent.startsWith('1 项改动') && status.textContent.includes('其中 1 项在集群页'), 'cluster note');
      cancelSettingsChanges();
      check(backend.value === 'auto' && button.disabled && getComputedStyle(bar).display === 'none', 'cancel restores cluster change');
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
      const groups = [...document.querySelectorAll('#page-settings [data-settings-group]')];
      check(groups.map(g => g.dataset.settingsGroup).join() === 'general,server,models,cache,acceleration'
        && groups.every(g => g.classList.contains('card-settings'))
        && !document.querySelector('#page-settings .badge-restart'), 'five groups, no restart badges');
      return count;
    })()
    """)
    #expect(result as? Int == 19)
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

@MainActor
private final class SettingsThemeMessages: NSObject, WKScriptMessageHandler {
    var themes: [String] = []

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if let body = message.body as? String { themes.append(body) }
    }
}

// The regrouped page still runs its load-time wiring: the memory slider follows
// its Auto switch, the appearance control applies at once without touching the
// save bar, and the LAN rows appear right after Network Mode in the Server group.
@MainActor
@Test func dashboardSettingsGroupsKeepControlsWorking() async throws {
    let configuration = WKWebViewConfiguration()
    let messages = SettingsSaveMessages()
    let themes = SettingsThemeMessages()
    configuration.userContentController.add(messages, name: "saveSettings")
    configuration.userContentController.add(messages, name: "setLogLevel")
    configuration.userContentController.add(themes, name: "setTheme")
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
      navigateTo('settings'); setLanguage('en');
      const bar = document.getElementById('settings-save-bar');
      const saveButton = document.getElementById('settings-save-button');
      const visible = el => getComputedStyle(el).display !== 'none' && el.getBoundingClientRect().height > 0;
      const group = el => el.closest('[data-settings-group]').dataset.settingsGroup;
      let count = 0;
      const check = (ok, name) => { if (!ok) throw new Error(name); count++; };

      // Memory limit: turning Auto off enables the slider (setupMemSlider ran on load).
      const memAuto = document.getElementById('cfg-mem-total-auto');
      const memSlider = document.getElementById('cfg-mem-total');
      check(group(memSlider) === 'models' && memAuto.checked && memSlider.disabled, 'auto disables slider');
      memAuto.click();
      check(!memAuto.checked && !memSlider.disabled
        && /GB|No limit/.test(memSlider.parentElement.querySelector('.slider-value').textContent), 'manual enables slider');
      memAuto.click();
      check(memSlider.disabled && bar.dataset.state === 'clean', 'auto restored');

      // Appearance applies at once and stays out of the save bar.
      const theme = document.getElementById('theme-select');
      const dark = theme.querySelector('.seg-btn[data-value="dark"]');
      check(group(theme) === 'general' && theme.classList.contains('ds-seg'), 'appearance in general');
      dark.click();
      const thumb = theme.querySelector('.ds-seg-thumb');
      check(dark.classList.contains('active') && dark.getAttribute('aria-pressed') === 'true'
        && theme.querySelector('[data-value="system"]').getAttribute('aria-pressed') === 'false', 'dark selected');
      check(thumb.style.width === dark.offsetWidth + 'px' && thumb.style.transform === 'translateX(' + dark.offsetLeft + 'px)', 'thumb follows');
      check(bar.dataset.state === 'clean' && saveButton.disabled, 'theme stays clean');

      // LAN rows: shown right after Network Mode inside Server, hidden again for local.
      const mode = document.getElementById('cfg-network-mode');
      const modeRow = mode.closest('.settings-row');
      const hostRow = document.getElementById('cfg-lan-host-row');
      const securityRow = document.getElementById('cfg-lan-security-row');
      check(!visible(hostRow) && !visible(securityRow), 'lan rows hidden for local');
      mode.value = 'lan'; mode.dispatchEvent(new Event('change', { bubbles: true }));
      check(visible(hostRow) && visible(securityRow) && group(hostRow) === 'server' && group(securityRow) === 'server'
        && modeRow.nextElementSibling === hostRow && hostRow.nextElementSibling === securityRow
        && hostRow.getBoundingClientRect().top >= modeRow.getBoundingClientRect().bottom - 1, 'lan rows follow network mode');
      mode.value = 'local'; mode.dispatchEvent(new Event('change', { bubbles: true }));
      check(!visible(hostRow) && !visible(securityRow), 'lan rows hidden again');
      return count;
    })()
    """)
    #expect(result as? Int == 10)
    let themeDeadline = Date().addingTimeInterval(5)
    while themes.themes.isEmpty && Date() < themeDeadline {
        try await Task.sleep(for: .milliseconds(50))
    }
    #expect(themes.themes == ["dark"])
}

// Switching language re-renders the hot-cache auto estimate with its value, and the
// save bar hands focus to the page heading only when focus was inside it.
@MainActor
@Test func dashboardSettingsHintAndFocusSurviveChanges() async throws {
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
      navigateTo('settings'); setLanguage('en');
      let count = 0;
      const check = (ok, name) => { if (!ok) throw new Error(name); count++; };
      const hint = document.getElementById('cfg-hot-cache-auto-hint');
      check(hint.textContent.includes('Current auto estimate') && !hint.textContent.includes('{value}'), 'en hint');
      setLanguage('zh-Hans');
      check(hint.textContent.includes('当前自动估算值') && /\\d GB/.test(hint.textContent) && !hint.textContent.includes('{value}'), 'zh hint ' + hint.textContent);

      const bar = document.getElementById('settings-save-bar');
      const button = document.getElementById('settings-save-button');
      const port = document.getElementById('cfg-port');
      const initial = port.value;
      port.value = '9077'; port.dispatchEvent(new Event('input'));
      check(bar.dataset.state === 'pending', 'dirty');
      button.focus();
      check(document.activeElement === button, 'button focused');
      port.value = initial; checkSettingsDirty();
      check(bar.dataset.state === 'clean' && document.activeElement.id === 'settings-page-title', 'focus moved to heading');

      port.value = '9078'; port.dispatchEvent(new Event('input'));
      port.focus();
      port.value = initial; checkSettingsDirty();
      check(document.activeElement === port, 'focus outside the bar stays');
      port.value = initial; checkSettingsDirty();

      // Save success: the button is disabled while saving and focus may already be on body.
      settingsBackendRunning = true;
      port.value = '9079'; port.dispatchEvent(new Event('input'));
      button.focus();
      saveSettings();
      check(bar.dataset.state === 'saving', 'saving');
      button.blur();
      onSettingsSaved(JSON.stringify({ status: 'ok' }));
      check(bar.dataset.state === 'clean' && document.activeElement.id === 'settings-page-title', 'focus after save success');

      // Editing another field while saving must not lose focus to the heading later.
      const seq = document.getElementById('cfg-max-sequences');
      const seqInitial = seq.value;
      port.value = '9080'; port.dispatchEvent(new Event('input'));
      button.focus();
      saveSettings();
      button.blur();
      seq.value = String(parseInt(seqInitial) + 1); seq.dispatchEvent(new Event('input'));
      seq.focus();
      onSettingsSaved(JSON.stringify({ status: 'ok' }));
      check(bar.dataset.state === 'pending' && document.activeElement === seq, 'focus stays in edited field');
      seq.value = seqInitial; checkSettingsDirty();
      check(bar.dataset.state === 'clean' && document.activeElement === seq, 'focus stays after revert');

      // Language switch must not re-enable the hot-cache slider while the prefix cache is off.
      const cacheEnable = document.getElementById('cfg-cache-enable');
      const hotAuto = document.getElementById('cfg-hot-cache-auto');
      const hot = document.getElementById('cfg-hot-cache');
      if (hotAuto.checked) { hotAuto.checked = false; hotAuto.dispatchEvent(new Event('change')); }
      if (cacheEnable.checked) { cacheEnable.checked = false; cacheEnable.dispatchEvent(new Event('change')); }
      check(hot.disabled === true, 'hot cache disabled with cache off');
      checkSettingsDirty();
      const stateBefore = bar.dataset.state;
      setLanguage('ja'); setLanguage('en');
      check(hot.disabled === true, 'hot cache still disabled after language switch');
      check(bar.dataset.state === stateBefore, 'save bar unchanged by language switch');
      return count;
    })()
    """)
    #expect(result as? Int == 13)
}
