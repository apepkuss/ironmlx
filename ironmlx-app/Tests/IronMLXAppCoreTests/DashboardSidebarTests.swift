import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
@Test func dashboardSidebarTooltipsAndIconAlignment() async throws {
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.userContentController.addUserScript(WKUserScript(
        source: try DashboardWindowController.bootstrapScript(config: AppConfig(), route: .status),
        injectionTime: .atDocumentStart, forMainFrameOnly: true
    ))
    let view = WKWebView(frame: NSRect(x: 0, y: 0, width: 1000, height: 650), configuration: configuration)
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    view.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    let deadline = Date().addingTimeInterval(15)
    while Date() < deadline {
        if (try? await view.evaluateJavaScript("typeof settingsBaseline !== 'undefined' && settingsBaseline !== null")) as? Bool == true { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    for (theme, appearance) in [("light", NSAppearance.Name.aqua), ("dark", .darkAqua)] {
        view.appearance = NSAppearance(named: appearance)
        try await Task.sleep(for: .milliseconds(100))
        #expect(try await view.evaluateJavaScript("matchMedia('(prefers-color-scheme: dark)').matches") as? Bool == (theme == "dark"))
        let geometry = try await view.evaluateJavaScript("""
        (() => {
          const check = (condition, message) => { if (!condition) throw new Error(message); };
          setLanguage('zh');
          document.documentElement.classList.remove('sidebar-collapsed'); updateSidebarControls();
          const toggle = document.getElementById('sidebar-toggle');
          const tooltip = document.getElementById('sidebar-tooltip');
          const enter = element => element.dispatchEvent(new MouseEvent('mouseenter'));
          const leave = element => element.dispatchEvent(new MouseEvent('mouseleave'));
          const isVisible = () => !tooltip.hidden && getComputedStyle(tooltip).display !== 'none' && tooltip.getBoundingClientRect().height > 0;
          const inViewport = () => {
            const rect = tooltip.getBoundingClientRect();
            return rect.left >= 0 && rect.top >= 0 && rect.right <= innerWidth && rect.bottom <= innerHeight;
          };
          check(toggle.textContent.trim() === '', 'toggle has no visible text label');
          enter(toggle);
          check(isVisible() && tooltip.textContent === '收起侧栏' && inViewport(), 'expanded hover tooltip');
          check(toggle.getAttribute('aria-describedby') === tooltip.id, 'accessible tooltip association');
          leave(toggle); check(!isVisible(), 'tooltip leaves with pointer');
          toggle.click();
          check(document.documentElement.classList.contains('sidebar-collapsed'), 'collapse click');
          check(localStorage.getItem('ironmlx.sidebar.collapsed') === 'true', 'remember collapsed state');
          enter(toggle);
          check(isVisible() && tooltip.textContent === '展开侧栏' && inViewport(), 'collapsed hover tooltip');
          setLanguage('en');
          check(tooltip.textContent === 'Expand sidebar', 'visible tooltip follows language');
          setLanguage('zh');
          leave(toggle);
          toggle.focus();
          check(isVisible(), 'keyboard focus tooltip');
          toggle.dispatchEvent(new KeyboardEvent('keydown', {key:'Escape'}));
          check(!isVisible(), 'Escape dismisses tooltip');
          toggle.blur();
          const items = [...document.querySelectorAll('.sidebar .nav-item')].filter(item => !item.hidden);
          check(items.length === 5, 'only released pages are visible');
          const geometry = items.map(item => {
            item.click();
            const tile = item.getBoundingClientRect();
            const icon = item.querySelector('svg').getBoundingClientRect();
            const dx = (icon.left + icon.right - tile.left - tile.right) / 2;
            const dy = (icon.top + icon.bottom - tile.top - tile.bottom) / 2;
            check(item.classList.contains('active'), 'selected icon highlight');
            check(Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5, item.dataset.page + ' icon is centered');
            enter(item);
            check(isVisible() && tooltip.textContent === item.getAttribute('aria-label') && inViewport(), 'page icon tooltip');
            leave(item); check(!isVisible(), 'page tooltip dismissed');
            return {page:item.dataset.page, dx, dy};
          });
          toggle.click();
          check(!document.documentElement.classList.contains('sidebar-collapsed'), 'expand click');
          enter(items[0]); check(!isVisible(), 'expanded pages already have labels');
          toggle.click();
          document.querySelector('.nav-item[data-page="chat"]').click();
          enter(toggle);
          return JSON.stringify(geometry);
        })()
        """)
        if let directory = ProcessInfo.processInfo.environment["IRONMLX_SIDEBAR_EVIDENCE"] {
            // Let the navigation highlight's CSS transition finish before capture.
            try await Task.sleep(for: .milliseconds(250))
            let root = URL(fileURLWithPath: directory, isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let measurements = try #require(geometry as? String)
            try measurements.write(to: root.appendingPathComponent("\(theme)-alignment.json"), atomically: true, encoding: .utf8)
            let snapshotConfiguration = WKSnapshotConfiguration()
            snapshotConfiguration.rect = NSRect(x: 0, y: 0, width: 280, height: 650)
            let image = try await view.takeSnapshot(configuration: snapshotConfiguration)
            let tiff = try #require(image.tiffRepresentation)
            let bitmap = try #require(NSBitmapImageRep(data: tiff))
            let png = try #require(bitmap.representation(using: .png, properties: [:]))
            try png.write(to: root.appendingPathComponent("\(theme)-sidebar.png"))
        }
    }
}
