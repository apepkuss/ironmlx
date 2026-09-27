import AppKit
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
@Test func dashboardUIDelegateImplementsJavaScriptConfirmation() {
    let delegate = DashboardUIDelegate()
    let selector = NSSelectorFromString(
        "webView:runJavaScriptConfirmPanelWithMessage:initiatedByFrame:completionHandler:"
    )

    #expect(delegate.responds(to: selector))
}

@MainActor
@Test func dashboardUIDelegateLocalizesConfirmationButtons() {
    #expect(
        DashboardUIDelegate.ConfirmationButtonTitles.resolved(for: "en")
            == .init(accept: "Continue", cancel: "Cancel")
    )
    #expect(
        DashboardUIDelegate.ConfirmationButtonTitles.resolved(for: "zh-Hans")
            == .init(accept: "继续", cancel: "取消")
    )
    #expect(
        DashboardUIDelegate.ConfirmationButtonTitles.resolved(for: "zh-Hant")
            == .init(accept: "繼續", cancel: "取消")
    )
    #expect(
        DashboardUIDelegate.ConfirmationButtonTitles.resolved(for: "ja")
            == .init(accept: "続ける", cancel: "キャンセル")
    )
    #expect(
        DashboardUIDelegate.ConfirmationButtonTitles.resolved(for: "ko")
            == .init(accept: "계속", cancel: "취소")
    )
    #expect(
        DashboardUIDelegate.ConfirmationButtonTitles.resolved(for: "unsupported")
            == .init(accept: "Continue", cancel: "Cancel")
    )
}

@MainActor
@Test func dashboardUIDelegateForwardsAcceptedJavaScriptConfirmation() throws {
    var presentedMessage: String?
    var presentedWindow: NSWindow?
    var pendingCompletion: (@MainActor @Sendable (Bool) -> Void)?
    var accepted: Bool?
    let delegate = DashboardUIDelegate(confirmationPresenter: { message, window, completion in
        presentedMessage = message
        presentedWindow = window
        pendingCompletion = completion
    })
    let webView = WKWebView(frame: .zero)

    delegate.presentJavaScriptConfirmation(message: "Continue download?", in: webView) {
        accepted = $0
    }

    #expect(presentedMessage == "Continue download?")
    #expect(presentedWindow == nil)
    let completion = try #require(pendingCompletion)
    completion(true)
    #expect(accepted == true)
}

@MainActor
@Test func dashboardUIDelegateForwardsCancelledJavaScriptConfirmation() throws {
    var pendingCompletion: (@MainActor @Sendable (Bool) -> Void)?
    var accepted: Bool?
    let delegate = DashboardUIDelegate(confirmationPresenter: { _, _, completion in
        pendingCompletion = completion
    })
    let webView = WKWebView(frame: .zero)

    delegate.presentJavaScriptConfirmation(message: "Continue download?", in: webView) {
        accepted = $0
    }

    let completion = try #require(pendingCompletion)
    completion(false)
    #expect(accepted == false)
}

@MainActor
@Test func dashboardWindowSupportsNativeFullscreenWithoutInitialFullscreenStyle() {
    #expect(DashboardWindowController.dashboardWindowStyleMask.contains(.resizable))
    #expect(!DashboardWindowController.dashboardWindowStyleMask.contains(.fullScreen))
    #expect(DashboardWindowController.dashboardWindowCollectionBehavior.contains(.fullScreenPrimary))
    #expect(DashboardWindowController.dashboardWindowTitleVisibility == .hidden)
}

@MainActor
@Test func dashboardWindowUsesRegularActivationPolicyWhileVisibleAndAccessoryWhenHidden() {
    #expect(DashboardWindowController.dashboardVisibleActivationPolicy == .regular)
    #expect(DashboardWindowController.dashboardHiddenActivationPolicy == .accessory)
}

@MainActor
@Test func dashboardThemeAppearanceForcesLightDarkAndRestoresSystemPreference() {
    let view = NSView(frame: NSRect(x: 0, y: 0, width: 320, height: 240))
    let window = NSWindow(
        contentRect: view.frame,
        styleMask: DashboardWindowController.dashboardWindowStyleMask,
        backing: .buffered,
        defer: true
    )
    window.contentView = view

    DashboardThemeAppearance.apply("dark", to: view)
    #expect(view.appearance?.name == .darkAqua)
    #expect(window.appearance?.name == .darkAqua)

    DashboardThemeAppearance.apply("light", to: view)
    #expect(view.appearance?.name == .aqua)
    #expect(window.appearance?.name == .aqua)

    DashboardThemeAppearance.apply("system", to: view)
    #expect(view.appearance == nil)
    #expect(window.appearance == nil)
}

@MainActor
@Test func dashboardThemeAppearanceRejectsUnknownPreferences() {
    #expect(DashboardThemeAppearance.normalizedPreference("light") == "light")
    #expect(DashboardThemeAppearance.normalizedPreference("dark") == "dark")
    #expect(DashboardThemeAppearance.normalizedPreference("system") == nil)
    #expect(DashboardThemeAppearance.normalizedPreference("unknown") == nil)
    #expect(DashboardThemeAppearance.normalizedPreference(nil) == nil)
}

@MainActor
@Test func dashboardWindowDelegateHidesRegularWindowInsteadOfClosing() {
    var hideWindowCalls = 0
    var exitFullScreenCalls = 0
    let delegate = DashboardWindowDelegate(
        isFullScreen: { false },
        exitFullScreen: { exitFullScreenCalls += 1 },
        hideWindow: { hideWindowCalls += 1 }
    )
    let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 320, height: 240),
        styleMask: DashboardWindowController.dashboardWindowStyleMask,
        backing: .buffered,
        defer: true
    )

    #expect(!delegate.windowShouldClose(window))
    #expect(hideWindowCalls == 1)
    #expect(exitFullScreenCalls == 0)
}

@MainActor
@Test func dashboardWindowDelegateRestoresAccessoryActivationWhenHidingRegularWindow() {
    var hideWindowCalls = 0
    var restoreAccessoryActivationCalls = 0
    let delegate = DashboardWindowDelegate(
        isFullScreen: { false },
        exitFullScreen: {},
        hideWindow: { hideWindowCalls += 1 },
        restoreAccessoryActivation: { restoreAccessoryActivationCalls += 1 }
    )
    let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 320, height: 240),
        styleMask: DashboardWindowController.dashboardWindowStyleMask,
        backing: .buffered,
        defer: true
    )

    #expect(!delegate.windowShouldClose(window))
    #expect(hideWindowCalls == 1)
    #expect(restoreAccessoryActivationCalls == 1)
}

@MainActor
@Test func dashboardWindowDelegateExitsFullScreenBeforeHiding() {
    var isFullScreen = true
    var hideWindowCalls = 0
    var exitFullScreenCalls = 0
    var restoreAccessoryActivationCalls = 0
    let delegate = DashboardWindowDelegate(
        isFullScreen: { isFullScreen },
        exitFullScreen: { exitFullScreenCalls += 1 },
        hideWindow: { hideWindowCalls += 1 },
        restoreAccessoryActivation: { restoreAccessoryActivationCalls += 1 }
    )
    let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 320, height: 240),
        styleMask: DashboardWindowController.dashboardWindowStyleMask,
        backing: .buffered,
        defer: true
    )

    #expect(!delegate.windowShouldClose(window))
    #expect(exitFullScreenCalls == 1)
    #expect(hideWindowCalls == 0)
    #expect(restoreAccessoryActivationCalls == 0)

    isFullScreen = false
    delegate.windowDidExitFullScreen(
        Notification(name: NSWindow.didExitFullScreenNotification, object: window)
    )
    #expect(hideWindowCalls == 1)
    #expect(restoreAccessoryActivationCalls == 1)
}

@MainActor
@Test func dashboardWindowDelegateCanCancelPendingFullScreenHide() {
    var hideWindowCalls = 0
    let delegate = DashboardWindowDelegate(
        isFullScreen: { true },
        exitFullScreen: {},
        hideWindow: { hideWindowCalls += 1 }
    )
    let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 320, height: 240),
        styleMask: DashboardWindowController.dashboardWindowStyleMask,
        backing: .buffered,
        defer: true
    )

    #expect(!delegate.windowShouldClose(window))
    delegate.cancelPendingHideAfterFullScreenExit()
    delegate.windowDidExitFullScreen(
        Notification(name: NSWindow.didExitFullScreenNotification, object: window)
    )
    #expect(hideWindowCalls == 0)
}

@MainActor
@Test func dashboardBootstrapIncludesPersistedRuntimeSettings() throws {
    let script = try DashboardWindowController.bootstrapScript(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Qwen3.5-4B-MLX-4bit",
            language: "zh-Hans",
            kvQuant: "k3v4",
            maxSequences: 1,
            maxModels: 2,
            modelTtlMinutes: 15
        ),
        route: .status
    )

    #expect(script.contains("window.__IRONMLX_APP_CONFIG__"))
    #expect(script.contains(#""max_sequences":1"#))
    #expect(script.contains(#""max_models":2"#))
    #expect(!script.contains("init_cache_blocks"))
    #expect(script.contains(#""model_ttl_minutes":15"#))
    #expect(script.contains("window.__IRONMLX_AUTO_HOT_CACHE_BYTES__"))
    #expect(script.contains("window.__IRONMLX_COLD_CACHE_CAPACITY__"))
    #expect(script.contains(#""default_gb":10"#))
    #expect(script.contains(#""min_gb":1"#))
}
