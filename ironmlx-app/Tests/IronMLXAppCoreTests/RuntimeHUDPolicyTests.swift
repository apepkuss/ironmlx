import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

/// Records what the coordinator asks of the HUD, so no real panel is created.
@MainActor
private final class RecordingHUD: RuntimeHUDPresenting {
    let preferences: GlancePreferences
    var visibleCalls: [Bool] = []
    var toggles = 0
    var languages: [String] = []

    init(preferences: GlancePreferences) {
        self.preferences = preferences
    }

    func setVisible(_ visible: Bool) {
        visibleCalls.append(visible)
        preferences.hudVisible = visible
    }

    func toggle() {
        toggles += 1
        setVisible(!preferences.hudVisible)
    }

    func resetPosition() {}

    func setLanguage(_ language: String) {
        languages.append(language)
    }
}

/// Stands in for the Carbon hotkey; the coordinator only keeps it alive.
private final class FakeHotKey {}

@MainActor
private final class LanguageSetting {
    var value = "en"
}

@MainActor
private final class TestClock {
    var now = Date(timeIntervalSince1970: 2_000_000)
}

private func keyDownEvent(_ characters: String, _ modifiers: NSEvent.ModifierFlags) throws -> NSEvent {
    try #require(
        NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0,
            context: nil, characters: characters, charactersIgnoringModifiers: characters,
            isARepeat: false, keyCode: 4
        )
    )
}

@Suite("RuntimeHUD policy")
struct RuntimeHUDPolicyTests {
    /// The HUD's context menu is plain AppKit (SwiftUI's never opened in the non-activating
    /// panel): same items and strings as before, states from the preferences, and each item
    /// changes the right preference or calls the right action.
    @Test @MainActor
    func contextMenuItemsReflectAndChangePreferences() throws {
        let suite = "ironmlx.glance.hud-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let preferences = GlancePreferences(defaults: defaults)
        preferences.hudDensity = .standard
        preferences.hudClickThrough = false
        var resets = 0
        var closes = 0
        let builder = RuntimeHUDContextMenu(
            preferences: preferences,
            language: "en",
            onResetPosition: { resets += 1 },
            onClose: { closes += 1 }
        )
        let menu = builder.makeMenu()
        #expect(
            menu.items.map { $0.isSeparatorItem ? "-" : $0.title }
                == ["Density", "Click Through", "-", "Reset Position", "Close"]
        )
        let density = try #require(menu.items[0].submenu)
        #expect(density.items.map(\.title) == ["Mini", "Standard"])
        #expect(density.items.map(\.state) == [.off, .on])
        #expect(menu.items[1].state == .off)
        #expect(menu.items.allSatisfy { $0.isSeparatorItem || $0.submenu != nil || $0.target === builder })

        density.performActionForItem(at: 0)
        #expect(preferences.hudDensity == .mini)
        menu.performActionForItem(at: 1)
        #expect(preferences.hudClickThrough)
        menu.performActionForItem(at: 3)
        #expect(resets == 1 && closes == 0)
        menu.performActionForItem(at: 4)
        #expect(closes == 1)

        // A fresh menu shows the new states; titles follow the language.
        let rebuilt = builder.makeMenu()
        #expect(rebuilt.items[0].submenu?.items.map(\.state) == [.on, .off])
        #expect(rebuilt.items[1].state == .on)
        rebuilt.items[0].submenu?.performActionForItem(at: 1)
        #expect(preferences.hudDensity == .standard)
        let chinese = RuntimeHUDContextMenu(preferences: preferences, language: "zh-Hans").makeMenu()
        #expect(chinese.items.map(\.title).contains("点击穿透"))
    }

    /// Right-click and control-click open the menu; a plain click (drag) and other events do not.
    @Test @MainActor
    func contextMenuOpensOnRightClickAndControlClick() throws {
        func mouse(_ type: NSEvent.EventType, _ modifiers: NSEvent.ModifierFlags = []) throws -> NSEvent {
            try #require(
                NSEvent.mouseEvent(
                    with: type, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0,
                    context: nil, eventNumber: 0, clickCount: 1, pressure: 0
                )
            )
        }
        #expect(RuntimeHUDPanel.isContextMenuClick(try mouse(.rightMouseDown)))
        #expect(RuntimeHUDPanel.isContextMenuClick(try mouse(.leftMouseDown, [.control])))
        #expect(!RuntimeHUDPanel.isContextMenuClick(try mouse(.leftMouseDown)))
        #expect(!RuntimeHUDPanel.isContextMenuClick(try mouse(.rightMouseUp)))
        #expect(!RuntimeHUDPanel.isContextMenuClick(try mouse(.leftMouseUp, [.control])))
        #expect(!RuntimeHUDPanel.isContextMenuClick(try keyDownEvent("h", [.control])))
    }

    @Test func placementUsesSavedOriginOnMainScreen() {
        let screens: [(id: String, visibleFrame: CGRect)] = [
            (id: "main", visibleFrame: CGRect(x: 0, y: 25, width: 1512, height: 920)),
            (id: "side", visibleFrame: CGRect(x: 1512, y: 0, width: 1920, height: 1055)),
        ]
        let saved = [
            "main": NSStringFromPoint(NSPoint(x: 300, y: 400)),
            "side": NSStringFromPoint(NSPoint(x: 2000, y: 500)),
        ]
        #expect(
            HUDPlacement.origin(
                saved: saved, screens: screens, mainScreenID: "main", panelSize: CGSize(width: 220, height: 120)
            ) == CGPoint(x: 300, y: 400)
        )
        #expect(
            HUDPlacement.origin(
                saved: saved, screens: screens, mainScreenID: "side", panelSize: CGSize(width: 220, height: 120)
            ) == CGPoint(x: 2000, y: 500)
        )
    }

    @Test func placementUsesSavedOriginOnSecondaryScreen() {
        let screens: [(id: String, visibleFrame: CGRect)] = [
            (id: "main", visibleFrame: CGRect(x: 0, y: 25, width: 1512, height: 920)),
            (id: "side", visibleFrame: CGRect(x: 1512, y: 0, width: 1920, height: 1055)),
        ]
        let size = CGSize(width: 220, height: 120)
        let sidePoint = CGPoint(x: 2000, y: 500)
        let onlySide = ["side": NSStringFromPoint(sidePoint)]
        // The HUD was last on the secondary screen: it comes back there, not to the main screen.
        #expect(
            HUDPlacement.origin(
                saved: onlySide, screens: screens, mainScreenID: "main", lastScreenID: "side", panelSize: size
            ) == sidePoint
        )
        // Any still-connected screen's saved position beats the default.
        #expect(
            HUDPlacement.origin(saved: onlySide, screens: screens, mainScreenID: "main", panelSize: size) == sidePoint
        )
        // With positions on both screens, the last screen wins over the main screen.
        let both = ["main": NSStringFromPoint(NSPoint(x: 300, y: 400)), "side": NSStringFromPoint(sidePoint)]
        #expect(
            HUDPlacement.origin(
                saved: both, screens: screens, mainScreenID: "main", lastScreenID: "side", panelSize: size
            ) == sidePoint
        )
        // The last screen was unplugged: the main screen's saved position is used.
        #expect(
            HUDPlacement.origin(
                saved: both, screens: screens, mainScreenID: "main", lastScreenID: "gone", panelSize: size
            ) == CGPoint(x: 300, y: 400)
        )
    }

    /// Health alerts (and a crashed backend) give the HUD its alert style; a backend the user
    /// stopped does not.
    @Test func stoppedBackendIsNotAnAlert() {
        #expect(!HUDVisibilityPolicy.isAlert(indicator: .red, backendState: .stopped))
        #expect(HUDVisibilityPolicy.isAlert(indicator: .red, backendState: .failed))
        #expect(HUDVisibilityPolicy.isAlert(indicator: .red, backendState: .running))
        #expect(HUDVisibilityPolicy.isAlert(indicator: .amber, backendState: .running))
        #expect(HUDVisibilityPolicy.isAlert(indicator: .amber, backendState: .degraded))
        #expect(!HUDVisibilityPolicy.isAlert(indicator: .blue, backendState: .starting))
        #expect(!HUDVisibilityPolicy.isAlert(indicator: .green, backendState: .running))
        #expect(!HUDVisibilityPolicy.isAlert(indicator: nil, backendState: .running))
    }

    /// ⌃⌥⌘H reaches the App either as the Carbon hotkey or, while the status menu is tracking, as a
    /// key-down that triggers the item's key equivalent. Whichever path (or both) it takes, the
    /// HUD toggles once; ordinary menu selections toggle once too.
    @Test @MainActor
    func oneKeypressTogglesOnceWhicheverPathItTakes() throws {
        let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
            .appendingPathComponent("ironmlx-hud-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "ironmlx.glance.hud-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let preferences = GlancePreferences(defaults: defaults)
        let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
        let backend = TestRuntimeBackend(state: .stopped, isRunning: false)
        let notificationCenter = NotificationCenter()
        let clock = TestClock()
        var hud: RecordingHUD?
        var hotKeyHandler: (@MainActor () -> Void)?
        let glance = RuntimeGlanceCoordinator(
            preferences: preferences,
            streaming: URLSessionRuntimeMetricsStreaming(),
            endpoint: { nil },
            backendState: { .stopped },
            notificationCenter: notificationCenter,
            hotKeyFactory: { handler in
                hotKeyHandler = handler
                return FakeHotKey()
            },
            hudFactory: { _, preferences, _ in
                let made = RecordingHUD(preferences: preferences)
                hud = made
                return made
            },
            now: { clock.now }
        )
        glance.start()
        let controller = MenuBarController(
            configStore: configStore,
            backend: backend,
            dashboard: DashboardWindowController(configStore: configStore, backend: backend),
            notificationCenter: notificationCenter,
            glance: glance
        )
        let recorded = try #require(hud)
        let pressHotKey = try #require(hotKeyHandler)
        let hotKeyEvent = try keyDownEvent("h", [.control, .option, .command])

        // Both paths fire for one press (menu key equivalent, then Carbon within 300 ms): one toggle.
        controller.handleHUDMenuAction(event: hotKeyEvent)
        clock.now += 0.1
        pressHotKey()
        #expect(recorded.visibleCalls.count == 1)
        #expect(recorded.toggles == 1)

        // Carbon first, then the menu echo: still one.
        clock.now += 1
        pressHotKey()
        clock.now += 0.2
        controller.handleHUDMenuAction(event: hotKeyEvent)
        #expect(recorded.visibleCalls.count == 2)

        // The keystroke reaches only the tracking menu: it toggles, through the hotkey path.
        clock.now += 1
        controller.handleHUDMenuAction(event: hotKeyEvent)
        #expect(recorded.visibleCalls.count == 3)
        #expect(recorded.toggles == 3)

        // Two separate presses more than 300 ms apart toggle twice.
        clock.now += 1
        pressHotKey()
        clock.now += 0.4
        pressHotKey()
        #expect(recorded.visibleCalls.count == 5)

        // Mouse click and Return on the highlighted item: ordinary selections, one toggle each,
        // even right after a hotkey press.
        let click = try #require(
            NSEvent.mouseEvent(
                with: .leftMouseUp, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
                context: nil, eventNumber: 0, clickCount: 1, pressure: 0
            )
        )
        clock.now += 0.05
        controller.handleHUDMenuAction(event: click)
        #expect(recorded.visibleCalls.count == 6)
        controller.handleHUDMenuAction(event: try keyDownEvent("\r", []))
        #expect(recorded.visibleCalls.count == 7)
        #expect(recorded.toggles == 5)
        glance.stop()
    }

    @Test func placementFallsBackToMainScreenWhenSavedScreenIsGone() {
        let main = CGRect(x: 0, y: 25, width: 1512, height: 920)
        let screens: [(id: String, visibleFrame: CGRect)] = [(id: "main", visibleFrame: main)]
        let size = CGSize(width: 220, height: 120)
        let expected = CGPoint(x: main.maxX - 24 - size.width, y: main.minY + 24)

        // The saved point lies only on a display that has been unplugged.
        let unplugged = ["main": NSStringFromPoint(NSPoint(x: 2000, y: 500))]
        #expect(HUDPlacement.origin(saved: unplugged, screens: screens, mainScreenID: "main", panelSize: size) == expected)

        // Only the unplugged display had a saved position; nothing saved, or unreadable.
        let otherScreen = ["side": NSStringFromPoint(NSPoint(x: 300, y: 400))]
        #expect(HUDPlacement.origin(saved: otherScreen, screens: screens, mainScreenID: "main", panelSize: size) == expected)
        #expect(HUDPlacement.origin(saved: [:], screens: screens, mainScreenID: "main", panelSize: size) == expected)
        #expect(
            HUDPlacement.origin(saved: ["main": "garbage"], screens: screens, mainScreenID: "main", panelSize: size)
                == expected
        )
    }

    /// Only the exact ⌃⌥⌘H keystroke, while the hotkey is registered, counts as the hotkey when it
    /// reaches the menu; Return on the item, other shortcuts and clicks are ordinary menu selections.
    @Test @MainActor
    func menuRecognizesOnlyTheHotKeyKeystrokeWhileRegistered() throws {
        func keyDown(_ characters: String, _ modifiers: NSEvent.ModifierFlags) throws -> NSEvent {
            try #require(
                NSEvent.keyEvent(
                    with: .keyDown, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0,
                    context: nil, characters: characters, charactersIgnoringModifiers: characters,
                    isARepeat: false, keyCode: 4
                )
            )
        }
        let hotKey = try keyDown("h", [.control, .option, .command])
        let returnKey = try keyDown("\r", [])
        let plainH = try keyDown("h", [.command])

        #expect(MenuBarMenuBuilder.isHUDHotKeyKeystroke(hotKey, hotKeyRegistered: true))
        #expect(!MenuBarMenuBuilder.isHUDHotKeyKeystroke(hotKey, hotKeyRegistered: false))
        #expect(!MenuBarMenuBuilder.isHUDHotKeyKeystroke(returnKey, hotKeyRegistered: true))
        #expect(!MenuBarMenuBuilder.isHUDHotKeyKeystroke(plainH, hotKeyRegistered: true))
        #expect(!MenuBarMenuBuilder.isHUDHotKeyKeystroke(nil, hotKeyRegistered: true))
    }

    @Test @MainActor
    func coordinatorWiresHUDAndHotKey() throws {
        let suite = "ironmlx.glance.hud-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let preferences = GlancePreferences(defaults: defaults)
        preferences.hudVisible = true
        let notificationCenter = NotificationCenter()
        var hud: RecordingHUD?
        var hotKeyHandler: (@MainActor () -> Void)?
        let language = LanguageSetting()
        let coordinator = RuntimeGlanceCoordinator(
            preferences: preferences,
            streaming: URLSessionRuntimeMetricsStreaming(),
            endpoint: { nil },
            backendState: { .stopped },
            notificationCenter: notificationCenter,
            language: { language.value },
            hotKeyFactory: { handler in
                hotKeyHandler = handler
                return FakeHotKey()
            },
            hudFactory: { _, preferences, _ in
                let made = RecordingHUD(preferences: preferences)
                hud = made
                return made
            }
        )
        #expect(coordinator.hudHotKeyDisplay == nil)

        coordinator.start()
        #expect(coordinator.hudHotKeyDisplay == "⌃⌥⌘H")
        let recorded = try #require(hud)
        // Shown at launch because the user left it on.
        #expect(recorded.visibleCalls == [true])

        // One hotkey press toggles exactly once, through the HUD controller's on-screen toggle.
        let pressHotKey = try #require(hotKeyHandler)
        pressHotKey()
        #expect(recorded.toggles == 1)
        #expect(recorded.visibleCalls == [true, false])
        #expect(!preferences.hudVisible)
        // The menu item flips the preference its check mark shows.
        coordinator.toggleHUD()
        #expect(recorded.toggles == 1)
        #expect(recorded.visibleCalls == [true, false, true])
        #expect(preferences.hudVisible)

        language.value = "zh-Hans"
        notificationCenter.post(name: .ironMLXMenuLanguageDidChange, object: nil)
        #expect(recorded.languages.last == "zh-Hans")

        coordinator.stop()
        #expect(coordinator.hudHotKeyDisplay == nil)
    }

    @Test @MainActor
    func coordinatorHidesHotKeyWhenRegistrationFails() throws {
        let suite = "ironmlx.glance.hud-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let preferences = GlancePreferences(defaults: defaults)
        var hud: RecordingHUD?
        let coordinator = RuntimeGlanceCoordinator(
            preferences: preferences,
            streaming: URLSessionRuntimeMetricsStreaming(),
            endpoint: { nil },
            backendState: { .stopped },
            notificationCenter: NotificationCenter(),
            hotKeyFactory: { _ in nil },
            hudFactory: { _, preferences, _ in
                let made = RecordingHUD(preferences: preferences)
                hud = made
                return made
            }
        )
        coordinator.start()
        #expect(coordinator.hudHotKeyDisplay == nil)
        // The HUD stays off at launch when the user has not turned it on.
        #expect(try #require(hud).visibleCalls.isEmpty)
        coordinator.stop()
    }
}
