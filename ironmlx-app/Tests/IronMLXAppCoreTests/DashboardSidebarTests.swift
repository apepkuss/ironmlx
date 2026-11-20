import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
@Test func dashboardNarrowSidebarIconsTooltipsAndHiddenPages() async throws {
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
        let expectedDark = theme == "dark"
        let appearanceDeadline = Date().addingTimeInterval(5)
        while Date() < appearanceDeadline {
            if (try? await view.evaluateJavaScript("matchMedia('(prefers-color-scheme: dark)').matches")) as? Bool == expectedDark { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        try #require(try await view.evaluateJavaScript("matchMedia('(prefers-color-scheme: dark)').matches") as? Bool == expectedDark)
        let geometry = try await view.evaluateJavaScript("""
        (() => {
          const check = (condition, message) => { if (!condition) throw new Error(message); };
          setLanguage('zh');
          // WebKit can retain stale computed styles in a previously visited,
          // display:none panel after appearance changes. Inspect the tooltip
          // in its rendered page/tab, as a user would encounter it.
          navigateTo('models');
          switchToTab('models-scheduler');
          const sidebar = document.getElementById('dashboard-sidebar');
          const tooltip = document.getElementById('sidebar-tooltip');
          const helpTooltip = document.getElementById('profile-help-tooltip');
          const enter = element => element.dispatchEvent(new MouseEvent('mouseenter'));
          const leave = element => element.dispatchEvent(new MouseEvent('mouseleave'));
          const isVisible = () => !tooltip.hidden && getComputedStyle(tooltip).display !== 'none' && tooltip.getBoundingClientRect().height > 0;
          const inViewport = () => {
            const rect = tooltip.getBoundingClientRect();
            return rect.left >= 0 && rect.top >= 0 && rect.right <= innerWidth && rect.bottom <= innerHeight;
          };
          const transparent = color => color === 'transparent' || color === 'rgba(0, 0, 0, 0)';
          const dark = matchMedia('(prefers-color-scheme: dark)').matches;

          // Fixed narrow icon rail; the collapse toggle and its preference are gone.
          check(document.getElementById('sidebar-toggle') === null, 'no sidebar toggle');
          check(!document.documentElement.classList.contains('sidebar-collapsed'), 'no collapsed class');
          const railRect = sidebar.getBoundingClientRect();
          check(railRect.width >= 56 && railRect.width <= 64, 'narrow sidebar width: ' + railRect.width);
          const contentRect = document.querySelector('.content').getBoundingClientRect();
          check(Math.abs(contentRect.left - railRect.right) < 0.5, 'content starts at the rail edge');

          // Design-system tokens resolve in both appearances.
          const root = getComputedStyle(document.documentElement);
          const token = name => root.getPropertyValue(name).trim().toLowerCase();
          check(token('--ds-green') === (dark ? '#30d158' : '#34c759'), 'ds green follows appearance');
          check(token('--ds-red') === (dark ? '#ff453a' : '#ff3b30'), 'ds red follows appearance');
          check(token('--ds-radius-card') === '12px', 'ds card radius');

          const compactStyle = getComputedStyle(tooltip);
          const helpRect = helpTooltip.getBoundingClientRect();
          check(helpRect.width > 0 && helpRect.height > 0, 'explanatory tooltip has rendered layout');
          const helpStyle = getComputedStyle(helpTooltip);
          check(compactStyle.backgroundColor === (dark ? 'rgba(58, 58, 60, 0.98)' : 'rgba(44, 44, 46, 0.96)'), 'adaptive compact tooltip background');
          check(compactStyle.color === (dark ? 'rgb(245, 245, 247)' : 'rgb(255, 255, 255)'), 'adaptive compact tooltip text');
          const expectedHelpBackground = dark ? 'rgba(44, 44, 46, 0.96)' : 'rgba(255, 255, 255, 0.96)';
          check(helpStyle.backgroundColor === expectedHelpBackground, 'adaptive explanatory tooltip background: ' + JSON.stringify({dark, actual:helpStyle.backgroundColor, expected:expectedHelpBackground}));
          check(helpStyle.color === (dark ? 'rgb(209, 209, 214)' : 'rgb(81, 81, 84)'), 'adaptive explanatory tooltip text');

          // Navigation items, order and hidden state are unchanged.
          const all = [...document.querySelectorAll('.sidebar .nav-item')];
          check(all.map(item => item.dataset.page).join(',') === 'status,models,voices,chat,benchmark,logs,settings,cluster', 'navigation order');
          for (const page of ['benchmark', 'cluster']) {
            const item = document.querySelector('.nav-item[data-page="' + page + '"]');
            check(item.hidden && item.getBoundingClientRect().height === 0, page + ' stays hidden');
            enter(item); check(!isVisible(), page + ' has no tooltip'); leave(item);
          }
          const items = all.filter(item => !item.hidden);
          check(items.map(item => item.dataset.page).join(',') === 'status,models,voices,chat,logs,settings', 'only released pages are visible');

          const names = {
            zh: ['状态', '模型', '声音', 'Agent', '日志', '设置'],
            en: ['Status', 'Models', 'Voices', 'Agent', 'Logs', 'Settings'],
          };
          for (const lang of ['zh', 'en']) {
            setLanguage(lang);
            items.forEach((item, index) => {
              const label = item.querySelector('[data-i18n]');
              check(label.getBoundingClientRect().width === 0, item.dataset.page + ' shows only its icon');
              const tile = item.getBoundingClientRect();
              check(tile.left >= railRect.left && tile.right <= railRect.right, item.dataset.page + ' fits the rail');
              enter(item);
              check(isVisible() && tooltip.textContent === names[lang][index] && inViewport(),
                lang + ' ' + item.dataset.page + ' tooltip: ' + JSON.stringify(tooltip.textContent));
              check(item.getAttribute('aria-label') === names[lang][index], lang + ' ' + item.dataset.page + ' accessible name');
              // The tooltip repeats the accessible name, so it is not linked as a description.
              check(item.getAttribute('aria-describedby') === null, 'name is not announced twice');
              leave(item); check(!isVisible(), 'tooltip leaves with pointer');
            });
          }
          // A tooltip already on screen follows a language change.
          enter(items[0]);
          setLanguage('zh');
          check(tooltip.textContent === '状态', 'visible tooltip follows language');
          leave(items[0]);
          items[1].focus();
          check(isVisible() && tooltip.textContent === '模型', 'keyboard focus tooltip');
          items[1].dispatchEvent(new KeyboardEvent('keydown', {key:'Escape'}));
          check(!isVisible(), 'Escape dismisses tooltip');
          items[1].blur();

          // The load indicator's tooltip follows its updates and leaves with it.
          const indicator = document.getElementById('model-load-indicator');
          const GB = 1024 * 1024 * 1024;
          const startedAt = Date.now() - 1000;
          const load = done => ({model:'org/Demo-4bit', variant:'plain', kind:'load', origin:'dashboard', state:'loading',
            stage:'reading_weights', completedStages:[], bytesDone:done, bytesTotal:100 * GB,
            startedAt, cancellable:true, autoOpenPanel:false});
          renderModelLoadIndicator([load(40 * GB)]);
          check(!indicator.hidden, 'load indicator shows');
          enter(indicator);
          const firstLabel = tooltip.textContent;
          check(isVisible() && firstLabel === indicator.getAttribute('aria-label') && inViewport(), 'load indicator tooltip');
          check(indicator.getAttribute('aria-describedby') === null, 'load indicator name is not announced twice');
          renderModelLoadIndicator([load(70 * GB)]);
          check(isVisible() && tooltip.textContent === indicator.getAttribute('aria-label') && tooltip.textContent !== firstLabel,
            'load indicator tooltip follows updates: ' + JSON.stringify([firstLabel, tooltip.textContent]));
          renderModelLoadIndicator([]);
          check(indicator.hidden && !isVisible(), 'load indicator tooltip leaves with the indicator');

          // Read the settled highlight, not a frame of its fade-in.
          const settle = document.createElement('style');
          settle.textContent = '.nav-item, .nav-item::before { transition: none !important; }';
          document.head.appendChild(settle);
          const geometry = items.map(item => {
            item.click();
            const tile = item.getBoundingClientRect();
            const icon = item.querySelector('svg').getBoundingClientRect();
            const dx = (icon.left + icon.right - tile.left - tile.right) / 2;
            const dy = (icon.top + icon.bottom - tile.top - tile.bottom) / 2;
            check(item.classList.contains('active'), 'selected page is marked');
            const highlight = getComputedStyle(item, '::before').backgroundColor;
            check(!transparent(highlight) || !transparent(getComputedStyle(item).backgroundColor), item.dataset.page + ' selected background');
            check(Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5, item.dataset.page + ' icon is centered');
            return {page:item.dataset.page, width:railRect.width, dx, dy};
          });
          document.querySelector('.nav-item[data-page="chat"]').click();
          enter(document.querySelector('.nav-item[data-page="models"]'));
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
            for (name, width) in [("sidebar", 280.0), ("window", 1000.0)] {
                let snapshotConfiguration = WKSnapshotConfiguration()
                snapshotConfiguration.rect = NSRect(x: 0, y: 0, width: width, height: 650)
                let image = try await view.takeSnapshot(configuration: snapshotConfiguration)
                let tiff = try #require(image.tiffRepresentation)
                let bitmap = try #require(NSBitmapImageRep(data: tiff))
                let png = try #require(bitmap.representation(using: .png, properties: [:]))
                try png.write(to: root.appendingPathComponent("\(theme)-\(name).png"))
            }
        }
    }
}
