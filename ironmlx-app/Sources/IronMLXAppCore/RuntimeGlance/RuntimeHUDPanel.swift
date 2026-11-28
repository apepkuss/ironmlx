import AppKit
import SwiftUI

/// The floating HUD window (spec 5.6). It never becomes key or main, so showing, dragging or
/// right-clicking it leaves keyboard focus with the user's frontmost App; it joins every Space
/// and shows over full-screen Apps.
final class RuntimeHUDPanel: NSPanel {
    init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 1, height: 1),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: true
        )
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        isMovableByWindowBackground = true
        hidesOnDeactivate = false
        isFloatingPanel = true
        becomesKeyOnlyIfNeeded = true
        isReleasedWhenClosed = false
        isOpaque = false
        backgroundColor = .clear
        // The glass background draws its own edge; a window shadow would not follow its shape changes.
        hasShadow = false
        animationBehavior = .none
    }

    override var canBecomeKey: Bool {
        false
    }

    override var canBecomeMain: Bool {
        false
    }

    /// Opens the HUD's context menu. Set by the controller.
    var contextMenuHandler: ((NSEvent) -> Void)?

    /// Right-click and control-click open the context menu here, before the window-background
    /// drag or SwiftUI see the event: SwiftUI's `.contextMenu` never opened in this panel.
    override func sendEvent(_ event: NSEvent) {
        if Self.isContextMenuClick(event), let contextMenuHandler {
            contextMenuHandler(event)
            return
        }
        super.sendEvent(event)
    }

    static func isContextMenuClick(_ event: NSEvent) -> Bool {
        switch event.type {
        case .rightMouseDown:
            true
        case .leftMouseDown:
            event.modifierFlags.contains(.control)
        default:
            false
        }
    }
}

/// The HUD's context menu (spec 5.6): density, click-through, reset position,
/// close. Built fresh for each right-click so its check marks match the preferences; the
/// builder is each item's target, so the caller keeps it alive while the menu is open.
@MainActor
final class RuntimeHUDContextMenu: NSObject {
    let preferences: GlancePreferences
    let language: String
    private let onResetPosition: () -> Void
    private let onClose: () -> Void

    init(
        preferences: GlancePreferences,
        language: String,
        onResetPosition: @escaping () -> Void = {},
        onClose: @escaping () -> Void = {}
    ) {
        self.preferences = preferences
        self.language = language
        self.onResetPosition = onResetPosition
        self.onClose = onClose
    }

    func makeMenu() -> NSMenu {
        let menu = NSMenu()
        menu.autoenablesItems = false

        let density = NSMenu()
        density.autoenablesItems = false
        for (value, key) in [(HUDDensity.mini, GlanceText.hudMini), (.standard, .hudStandard)] {
            let item = item(key, #selector(selectDensity(_:)))
            item.representedObject = value.rawValue
            item.state = preferences.hudDensity == value ? .on : .off
            density.addItem(item)
        }
        let densityItem = NSMenuItem(title: text(.hudDensity), action: nil, keyEquivalent: "")
        densityItem.submenu = density
        menu.addItem(densityItem)

        let clickThrough = item(.hudClickThrough, #selector(toggleClickThrough(_:)))
        clickThrough.state = preferences.hudClickThrough ? .on : .off
        menu.addItem(clickThrough)
        menu.addItem(.separator())
        menu.addItem(item(.hudResetPosition, #selector(resetPosition(_:))))
        menu.addItem(item(.hudClose, #selector(close(_:))))
        return menu
    }

    @objc private func selectDensity(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let density = HUDDensity(rawValue: raw) else {
            return
        }
        preferences.hudDensity = density
    }

    @objc private func toggleClickThrough(_ sender: NSMenuItem) {
        preferences.hudClickThrough.toggle()
    }

    @objc private func resetPosition(_ sender: NSMenuItem) {
        onResetPosition()
    }

    @objc private func close(_ sender: NSMenuItem) {
        onClose()
    }

    private func item(_ key: GlanceText, _ action: Selector) -> NSMenuItem {
        let item = NSMenuItem(title: text(key), action: action, keyEquivalent: "")
        item.target = self
        return item
    }

    private func text(_ key: GlanceText) -> String {
        GlanceText.localized(key, language: language)
    }
}

/// Hosts the SwiftUI HUD. A drag anywhere on the HUD moves the window, since SwiftUI content
/// otherwise claims the mouse-down and `isMovableByWindowBackground` would never apply.
final class RuntimeHUDHostingView: NSHostingView<RuntimeHUDView> {
    override var mouseDownCanMoveWindow: Bool {
        true
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool {
        true
    }
}
