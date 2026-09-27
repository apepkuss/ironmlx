import AppKit
import Foundation
import UniformTypeIdentifiers
import WebKit

@MainActor
final class DashboardUIDelegate: NSObject, WKUIDelegate {
    struct ConfirmationButtonTitles: Equatable {
        var accept: String
        var cancel: String

        static func resolved(for language: String) -> Self {
            switch language {
            case "zh-Hans":
                Self(accept: "继续", cancel: "取消")
            case "zh-Hant":
                Self(accept: "繼續", cancel: "取消")
            case "ja":
                Self(accept: "続ける", cancel: "キャンセル")
            case "ko":
                Self(accept: "계속", cancel: "취소")
            default:
                Self(accept: "Continue", cancel: "Cancel")
            }
        }
    }

    typealias ConfirmationPresenter = @MainActor (
        _ message: String,
        _ window: NSWindow?,
        _ completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
    ) -> Void

    static let supportedAudioFileExtensions = ["wav", "flac", "mp3"]
    private let confirmationPresenter: ConfirmationPresenter

    init(
        language: String = "en",
        confirmationPresenter: ConfirmationPresenter? = nil
    ) {
        let buttonTitles = ConfirmationButtonTitles.resolved(for: language)
        self.confirmationPresenter = confirmationPresenter ?? { message, window, completionHandler in
            DashboardUIDelegate.presentConfirmation(
                message: message,
                window: window,
                buttonTitles: buttonTitles,
                completionHandler: completionHandler
            )
        }
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.allowedContentTypes = Self.supportedAudioFileExtensions.compactMap {
            UTType(filenameExtension: $0)
        }
        let finish: (NSApplication.ModalResponse) -> Void = { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
        if let window = webView.window {
            panel.beginSheetModal(for: window, completionHandler: finish)
        } else {
            panel.begin(completionHandler: finish)
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        presentJavaScriptConfirmation(
            message: message,
            in: webView,
            completionHandler: completionHandler
        )
    }

    func presentJavaScriptConfirmation(
        message: String,
        in webView: WKWebView,
        completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        confirmationPresenter(message, webView.window, completionHandler)
    }

    private static func presentConfirmation(
        message: String,
        window: NSWindow?,
        buttonTitles: ConfirmationButtonTitles,
        completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "IronMLX"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: buttonTitles.accept)
        alert.addButton(withTitle: buttonTitles.cancel)
        let finish: (NSApplication.ModalResponse) -> Void = { response in
            completionHandler(response == .alertFirstButtonReturn)
        }
        if let window {
            alert.beginSheetModal(for: window, completionHandler: finish)
        } else {
            finish(alert.runModal())
        }
    }
}

@MainActor
enum DashboardThemeAppearance {
    static func normalizedPreference(_ value: String?) -> String? {
        switch value {
        case "light", "dark":
            return value
        default:
            return nil
        }
    }

    static func appearance(for preference: String?) -> NSAppearance? {
        switch normalizedPreference(preference) {
        case "light":
            return NSAppearance(named: .aqua)
        case "dark":
            return NSAppearance(named: .darkAqua)
        default:
            return nil
        }
    }

    static func apply(_ preference: String?, to view: NSView) {
        let appearance = appearance(for: preference)
        view.appearance = appearance
        view.window?.appearance = appearance
    }
}

@MainActor
public final class DashboardWindowController {
    public static let dashboardWindowStyleMask: NSWindow.StyleMask = [
        .titled,
        .closable,
        .miniaturizable,
        .resizable,
    ]
    public static let dashboardWindowCollectionBehavior: NSWindow.CollectionBehavior = [.fullScreenPrimary]
    public static let dashboardWindowTitleVisibility: NSWindow.TitleVisibility = .hidden
    public static let dashboardVisibleActivationPolicy: NSApplication.ActivationPolicy = .regular
    public static let dashboardHiddenActivationPolicy: NSApplication.ActivationPolicy = .accessory

    private let configStore: AppConfigStore
    private let backend: any BackendRuntimeManaging
    private var window: NSWindow?
    private var webView: WKWebView?
    private var bridge: DashboardBridge?
    private var windowDelegate: DashboardWindowDelegate?
    private var dashboardUIDelegate: DashboardUIDelegate?

    public init(configStore: AppConfigStore, backend: any BackendRuntimeManaging) {
        self.configStore = configStore
        self.backend = backend
    }

    public func show(route: DashboardInitialRoute = .status) {
        if let window {
            windowDelegate?.cancelPendingHideAfterFullScreenExit()
            Self.applyVisibleActivationPolicy()
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
        let dashboardUIDelegate = DashboardUIDelegate(language: config.language)
        webView.uiDelegate = dashboardUIDelegate
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
        window.titleVisibility = Self.dashboardWindowTitleVisibility
        window.collectionBehavior = Self.dashboardWindowCollectionBehavior
        window.center()
        window.contentView = webView
        window.isReleasedWhenClosed = false
        DashboardThemeAppearance.apply(config.theme, to: webView)
        let windowDelegate = DashboardWindowDelegate(
            isFullScreen: { [weak window] in
                window?.styleMask.contains(.fullScreen) == true
            },
            exitFullScreen: { [weak window] in
                window?.toggleFullScreen(nil)
            },
            hideWindow: { [weak window, weak bridge] in
                bridge?.cancelDiagnosticExport()
                window?.orderOut(nil)
            },
            restoreAccessoryActivation: {
                Self.applyHiddenActivationPolicy()
            }
        )
        window.delegate = windowDelegate

        self.window = window
        self.webView = webView
        self.bridge = bridge
        self.windowDelegate = windowDelegate
        self.dashboardUIDelegate = dashboardUIDelegate

        guard let htmlURL = IronMLXAppResourceResolver.url(forResource: "dashboard2", withExtension: "html") else {
            preconditionFailure("IronMLX App Bundle is missing dashboard2.html")
        }
        webView.loadFileURL(htmlURL, allowingReadAccessTo: htmlURL.deletingLastPathComponent())

        Self.applyVisibleActivationPolicy()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public func cancelAllDownloads() {
        bridge?.cancelAllDownloads()
    }

    private static func applyVisibleActivationPolicy() {
        NSApp.setActivationPolicy(dashboardVisibleActivationPolicy)
    }

    private static func applyHiddenActivationPolicy() {
        NSApp.setActivationPolicy(dashboardHiddenActivationPolicy)
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
        let networkInterfacesData = try encoder.encode(EndpointPayload.localNetworkInterfaces())
        let networkInterfacesJSON = String(data: networkInterfacesData, encoding: .utf8) ?? "[]"
        return """
        window.__IRONMLX_APP_CONFIG__ = \(configJSON);
        window.__IRONMLX_PORT__ = \(config.port);
        window.__DEFAULT_MODEL__ = \(DashboardBridge.jsStringLiteral(config.defaultModel ?? ""));
        window.__APP_LANGUAGE__ = \(DashboardBridge.jsStringLiteral(config.language));
        window.__IRONMLX_KV_QUANT__ = \(DashboardBridge.jsStringLiteral(BackendLaunchOptions.normalizedKVQuant(config.kvQuant) ?? "none"));
        window.__IRONMLX_AUTO_HOT_CACHE_BYTES__ = \(automaticHotCacheBytes);
        window.__IRONMLX_COLD_CACHE_CAPACITY__ = \(coldCacheCapacityJSON);
        window.__IRONMLX_NETWORK_INTERFACES__ = \(networkInterfacesJSON);
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
    private let restoreAccessoryActivation: () -> Void
    private var shouldHideAfterFullScreenExit = false

    init(
        isFullScreen: @escaping () -> Bool,
        exitFullScreen: @escaping () -> Void,
        hideWindow: @escaping () -> Void,
        restoreAccessoryActivation: @escaping () -> Void = {}
    ) {
        self.isFullScreen = isFullScreen
        self.exitFullScreen = exitFullScreen
        self.hideWindow = hideWindow
        self.restoreAccessoryActivation = restoreAccessoryActivation
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if isFullScreen() {
            shouldHideAfterFullScreenExit = true
            exitFullScreen()
        } else {
            hideDashboardWindow()
        }
        return false
    }

    func windowDidExitFullScreen(_ notification: Notification) {
        guard shouldHideAfterFullScreenExit else {
            return
        }
        shouldHideAfterFullScreenExit = false
        hideDashboardWindow()
    }

    func cancelPendingHideAfterFullScreenExit() {
        shouldHideAfterFullScreenExit = false
    }

    private func hideDashboardWindow() {
        hideWindow()
        restoreAccessoryActivation()
    }
}
