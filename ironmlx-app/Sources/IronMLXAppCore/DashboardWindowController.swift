import AppKit
import Foundation
import WebKit

@MainActor
public final class DashboardWindowController {
    public static let dashboardWindowStyleMask: NSWindow.StyleMask = [
        .titled,
        .closable,
        .miniaturizable,
        .resizable,
    ]
    public static let dashboardWindowCollectionBehavior: NSWindow.CollectionBehavior = [.fullScreenPrimary]

    private let configStore: AppConfigStore
    private let backend: BackendProcessManager
    private var window: NSWindow?
    private var webView: WKWebView?
    private var bridge: DashboardBridge?
    private var windowDelegate: DashboardWindowDelegate?

    public init(configStore: AppConfigStore, backend: BackendProcessManager) {
        self.configStore = configStore
        self.backend = backend
    }

    public func show(route: DashboardInitialRoute = .status) {
        if let window {
            windowDelegate?.cancelPendingHideAfterFullScreenExit()
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            apply(route: route)
            return
        }

        let configuration = WKWebViewConfiguration()
        let userContentController = WKUserContentController()
        let config = configStore.load()
        userContentController.addUserScript(
            WKUserScript(source: Self.dashboardLoggingScript, injectionTime: .atDocumentStart, forMainFrameOnly: true)
        )
        let bootstrap = (try? Self.bootstrapScript(config: config, route: route)) ?? ""
        userContentController.addUserScript(
            WKUserScript(source: bootstrap, injectionTime: .atDocumentStart, forMainFrameOnly: true)
        )
        userContentController.addUserScript(
            WKUserScript(source: Self.routeScript(for: route), injectionTime: .atDocumentEnd, forMainFrameOnly: true)
        )
        configuration.userContentController = userContentController

        let webView = WKWebView(frame: .zero, configuration: configuration)
        let bridge = DashboardBridge(
            webView: webView,
            configStore: configStore,
            backend: backend
        )
        DashboardBridge.handlerNames.forEach {
            userContentController.add(bridge, name: $0)
        }

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1180, height: 780),
            styleMask: Self.dashboardWindowStyleMask,
            backing: .buffered,
            defer: false
        )
        window.title = "ironmlx"
        window.collectionBehavior = Self.dashboardWindowCollectionBehavior
        window.center()
        window.contentView = webView
        window.isReleasedWhenClosed = false
        let windowDelegate = DashboardWindowDelegate(
            isFullScreen: { [weak window] in
                window?.styleMask.contains(.fullScreen) == true
            },
            exitFullScreen: { [weak window] in
                window?.toggleFullScreen(nil)
            },
            hideWindow: { [weak window] in
                window?.orderOut(nil)
            }
        )
        window.delegate = windowDelegate

        self.window = window
        self.webView = webView
        self.bridge = bridge
        self.windowDelegate = windowDelegate

        if let htmlURL = Bundle.module.url(forResource: "dashboard2", withExtension: "html") {
            webView.loadFileURL(htmlURL, allowingReadAccessTo: htmlURL.deletingLastPathComponent())
        }

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public static func bootstrapScript(config: AppConfig, route: DashboardInitialRoute) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let configData = try encoder.encode(config)
        let configJSON = String(data: configData, encoding: .utf8) ?? "{}"
        let automaticHotCacheBytes = BackendLaunchOptions.hotCacheLimitBytes(
            hotCacheGigabytes: nil,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
        )
        let coldCacheCapacity = ColdCacheCapacityPolicy.capacity(
            forDirectoryPath: config.cacheDir ?? BackendLaunchOptions.defaultPagedPrefixCacheDirectory
        )
        let coldCacheCapacityData = try encoder.encode(coldCacheCapacity)
        let coldCacheCapacityJSON = String(data: coldCacheCapacityData, encoding: .utf8) ?? "{}"
        return """
        window.__IRONMLX_APP_CONFIG__ = \(configJSON);
        window.__IRONMLX_PORT__ = \(config.port);
        window.__DEFAULT_MODEL__ = \(DashboardBridge.jsStringLiteral(config.lastModel ?? ""));
        window.__APP_LANGUAGE__ = \(DashboardBridge.jsStringLiteral(config.language));
        window.__IRONMLX_KV_QUANT__ = \(DashboardBridge.jsStringLiteral(BackendLaunchOptions.normalizedKVQuant(config.kvQuant) ?? "none"));
        window.__IRONMLX_AUTO_HOT_CACHE_BYTES__ = \(automaticHotCacheBytes);
        window.__IRONMLX_COLD_CACHE_CAPACITY__ = \(coldCacheCapacityJSON);
        window.__IRONMLX_INITIAL_ROUTE__ = \(DashboardBridge.jsStringLiteral(route.rawValue));
        """
    }

    private func apply(route: DashboardInitialRoute) {
        webView?.evaluateJavaScript(Self.routeScript(for: route)) { _, error in
            if let error {
                IronMLXAppLogger.error("Dashboard route script error: \(error)")
            }
        }
    }

    private static func routeScript(for route: DashboardInitialRoute) -> String {
        let routeValue = DashboardBridge.jsStringLiteral(route.rawValue)
        return """
        (function() {
          var route = \(routeValue);
          function applyInitialRoute() {
            if (route === 'status') {
              return;
            }
            if (route === 'onboarding' && typeof showOnboarding === 'function') {
              showOnboarding();
              return;
            }
            if (typeof navigateTo === 'function') {
              navigateTo('models');
            }
            if (typeof switchToTab === 'function') {
              if (route === 'modelsManage') {
                switchToTab('models-manage');
              } else if (route === 'modelsDownload') {
                switchToTab('models-download');
              }
            }
          }
          if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', applyInitialRoute, { once: true });
          } else {
            setTimeout(applyInitialRoute, 0);
          }
        })();
        """
    }

    private static let dashboardLoggingScript = """
    (function() {
      if (window.__IRONMLX_DASHBOARD_LOGGER__) return;
      window.__IRONMLX_DASHBOARD_LOGGER__ = true;

      function normalizeLogValue(value) {
        try {
          if (value instanceof Error) return value.stack || value.message || String(value);
          if (typeof value === 'object') return JSON.stringify(value);
          return String(value);
        } catch (e) {
          return String(value);
        }
      }

      function postDashboardLog(level, args) {
        try {
          if (!window.webkit || !window.webkit.messageHandlers || !window.webkit.messageHandlers.dashboardLog) return;
          var message = Array.prototype.slice.call(args).map(normalizeLogValue).join(' ');
          window.webkit.messageHandlers.dashboardLog.postMessage(JSON.stringify({ level: level, message: message }));
        } catch (e) {}
      }

      var originalWarn = console.warn;
      console.warn = function() {
        postDashboardLog('WARN', arguments);
        if (originalWarn) originalWarn.apply(console, arguments);
      };

      var originalError = console.error;
      console.error = function() {
        postDashboardLog('ERROR', arguments);
        if (originalError) originalError.apply(console, arguments);
      };

      window.addEventListener('error', function(event) {
        postDashboardLog('ERROR', [
          (event.message || 'Script error') + ' at ' + (event.filename || '-') + ':' + (event.lineno || 0) + ':' + (event.colno || 0)
        ]);
      });

      window.addEventListener('unhandledrejection', function(event) {
        postDashboardLog('ERROR', ['Unhandled promise rejection', event.reason]);
      });
    })();
    """
}

final class DashboardWindowDelegate: NSObject, NSWindowDelegate {
    private let isFullScreen: () -> Bool
    private let exitFullScreen: () -> Void
    private let hideWindow: () -> Void
    private var shouldHideAfterFullScreenExit = false

    init(
        isFullScreen: @escaping () -> Bool,
        exitFullScreen: @escaping () -> Void,
        hideWindow: @escaping () -> Void
    ) {
        self.isFullScreen = isFullScreen
        self.exitFullScreen = exitFullScreen
        self.hideWindow = hideWindow
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if isFullScreen() {
            shouldHideAfterFullScreenExit = true
            exitFullScreen()
        } else {
            hideWindow()
        }
        return false
    }

    func windowDidExitFullScreen(_ notification: Notification) {
        guard shouldHideAfterFullScreenExit else {
            return
        }
        shouldHideAfterFullScreenExit = false
        hideWindow()
    }

    func cancelPendingHideAfterFullScreenExit() {
        shouldHideAfterFullScreenExit = false
    }
}
