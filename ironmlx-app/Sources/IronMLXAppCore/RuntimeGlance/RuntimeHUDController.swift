import AppKit
import CoreGraphics
import Observation
import SwiftUI

/// The HUD's alert rule. Visibility has no policy: the HUD is on screen whenever the user has
/// it on.
enum HUDVisibilityPolicy {
    /// Whether the HUD shows its alert style: a health alert (red or amber, the menu bar dot's
    /// rule) or a crashed backend. A backend the user stopped keeps its red menu bar dot but is
    /// not an alert here.
    static func isAlert(indicator: GlanceIndicator?, backendState: BackendProcessState) -> Bool {
        guard indicator == .red || indicator == .amber else {
            return false
        }
        return backendState != .stopped
    }
}

/// Where the HUD goes on screen. Positions are remembered per screen as `NSStringFromPoint`.
enum HUDPlacement {
    static let inset: CGFloat = 24

    /// A saved origin that is still on some screen's visible area, preferring the screen the HUD
    /// was last on, then the main screen, then any other connected screen; otherwise the main
    /// screen's bottom-right corner, 24 pt in. Saved origins of disconnected screens are ignored.
    static func origin(
        saved: [String: String],
        screens: [(id: String, visibleFrame: CGRect)],
        mainScreenID: String,
        lastScreenID: String? = nil,
        panelSize: CGSize
    ) -> CGPoint {
        let connected = screens.map(\.id)
        var candidates: [String] = []
        for id in [lastScreenID, mainScreenID].compactMap({ $0 }) + connected
            where connected.contains(id) && !candidates.contains(id)
        {
            candidates.append(id)
        }
        for id in candidates {
            if let point = saved[id].flatMap(parse),
               screens.contains(where: { $0.visibleFrame.contains(point) })
            {
                return point
            }
        }
        guard let main = screens.first(where: { $0.id == mainScreenID })?.visibleFrame ?? screens.first?.visibleFrame
        else {
            return .zero
        }
        return CGPoint(x: main.maxX - inset - panelSize.width, y: main.minY + inset)
    }

    /// Reads `NSStringFromPoint` output ("{x, y}"); nil for anything else.
    static func parse(_ text: String) -> CGPoint? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}") else {
            return nil
        }
        let parts = trimmed.dropFirst().dropLast().split(separator: ",")
        guard parts.count == 2,
              let x = Double(parts[0].trimmingCharacters(in: .whitespaces)),
              let y = Double(parts[1].trimmingCharacters(in: .whitespaces)),
              x.isFinite, y.isFinite
        else {
            return nil
        }
        return CGPoint(x: x, y: y)
    }

    static func format(_ point: CGPoint) -> String {
        NSStringFromPoint(point)
    }

    /// A stable ID for `screen`: the display's UUID, which survives reboots and reconnects.
    @MainActor
    static func id(of screen: NSScreen) -> String {
        guard let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber else {
            return screen.localizedName
        }
        let displayID = CGDirectDisplayID(number.uint32Value)
        if let uuid = CGDisplayCreateUUIDFromDisplayID(displayID)?.takeRetainedValue(),
           let text = CFUUIDCreateString(nil, uuid) as String?
        {
            return text
        }
        return "display-\(displayID)"
    }
}

/// What the coordinator needs from the HUD; a fake stands in for it in tests.
@MainActor
protocol RuntimeHUDPresenting: AnyObject {
    func setVisible(_ visible: Bool)
    func toggle()
    func resetPosition()
    func setLanguage(_ language: String)
}

/// Shows the HUD panel while the user has it on, follows the click-through
/// preference and remembers the position per screen. The panel is created on first show.
@MainActor
final class RuntimeHUDController: NSObject, NSWindowDelegate, RuntimeHUDPresenting {
    static let fadeDuration: TimeInterval = 0.2

    let store: RuntimeMetricsStore
    let preferences: GlancePreferences
    private(set) var language: String
    private var panel: RuntimeHUDPanel?
    private var hostingView: RuntimeHUDHostingView?
    /// The state being shown or faded to; the panel may still be fading out while false.
    private var isPresented = false
    /// Set while the controller itself moves the panel, so only user drags are remembered.
    private var isPlacing = false
    /// The panel is placed on first show, when screens change and on "Reset Position" only, so
    /// showing it again never moves a HUD the user left on another screen or spot.
    private var needsPlacement = true
    private var screenObserver: (any NSObjectProtocol)?
    private var contextMenu: RuntimeHUDContextMenu?

    init(store: RuntimeMetricsStore, preferences: GlancePreferences, language: String = "en") {
        self.store = store
        self.preferences = preferences
        self.language = language
        super.init()
        observe()
        screenObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: nil
        ) { [weak self] _ in
            if Thread.isMainThread {
                MainActor.assumeIsolated { self?.screensDidChange() }
            } else {
                Task { @MainActor in self?.screensDidChange() }
            }
        }
    }

    isolated deinit {
        panel?.orderOut(nil)
        if let screenObserver {
            NotificationCenter.default.removeObserver(screenObserver)
        }
    }

    private func screensDidChange() {
        guard panel != nil else {
            return
        }
        place()
    }

    /// Turns the HUD on or off; it stays on screen for as long as it is on.
    func setVisible(_ visible: Bool) {
        preferences.hudVisible = visible
        evaluate()
    }

    /// The hotkey: a plain on/off switch.
    func toggle() {
        setVisible(!preferences.hudVisible)
    }

    func resetPosition() {
        var origins = preferences.hudOrigins
        for screen in [panel?.screen, NSScreen.main].compactMap({ $0 }) {
            origins.removeValue(forKey: HUDPlacement.id(of: screen))
        }
        preferences.hudOrigins = origins
        preferences.hudLastScreenID = nil
        place()
    }

    func setLanguage(_ language: String) {
        guard self.language != language else {
            return
        }
        self.language = language
        hostingView?.rootView = makeView()
    }

    // MARK: Visibility

    /// Re-evaluates whenever a preference read here changes. The store is not read: what the
    /// backend is doing never shows or hides the HUD.
    private func observe() {
        let inputs = withObservationTracking {
            readInputs()
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                self?.observe()
            }
        }
        apply(inputs)
    }

    private struct Inputs {
        var enabled: Bool
        var clickThrough: Bool
    }

    private func readInputs() -> Inputs {
        Inputs(enabled: preferences.hudVisible, clickThrough: preferences.hudClickThrough)
    }

    private func evaluate() {
        apply(readInputs())
    }

    private func apply(_ inputs: Inputs) {
        if inputs.enabled {
            present(clickThrough: inputs.clickThrough)
        } else {
            panel?.ignoresMouseEvents = inputs.clickThrough
            dismiss()
        }
    }

    private func present(clickThrough: Bool) {
        let panel = panel ?? makePanel()
        panel.ignoresMouseEvents = clickThrough
        guard !isPresented else {
            return
        }
        isPresented = true
        if !panel.isVisible {
            panel.alphaValue = 0
            fitToContent()
            if needsPlacement {
                place()
            }
            panel.orderFrontRegardless()
        }
        NSAnimationContext.runAnimationGroup { context in
            context.duration = Self.fadeDuration
            panel.animator().alphaValue = 1
        }
    }

    private func dismiss() {
        guard isPresented, let panel else {
            return
        }
        isPresented = false
        NSAnimationContext.runAnimationGroup { context in
            context.duration = Self.fadeDuration
            panel.animator().alphaValue = 0
        } completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                // Shown again while fading out: leave it on screen.
                guard let self, !self.isPresented else {
                    return
                }
                self.panel?.orderOut(nil)
            }
        }
    }

    // MARK: Panel

    private func makePanel() -> RuntimeHUDPanel {
        let panel = RuntimeHUDPanel()
        let container = NSView(frame: panel.contentRect(forFrameRect: panel.frame))
        let hosting = RuntimeHUDHostingView(rootView: makeView())
        // Intrinsic size only: no min/max constraints that would make AppKit resize the window
        // around a different anchor; the controller sizes it from the content.
        hosting.sizingOptions = [.intrinsicContentSize]
        container.addSubview(hosting)
        panel.contentView = container
        panel.delegate = self
        panel.contextMenuHandler = { [weak self] event in
            self?.showContextMenu(for: event)
        }
        self.panel = panel
        hostingView = hosting
        return panel
    }

    /// Pops up the context menu at the click. Works while the App is inactive: the panel is
    /// non-activating and the menu is tracked without activating the App.
    private func showContextMenu(for event: NSEvent) {
        guard let hostingView else {
            return
        }
        let builder = RuntimeHUDContextMenu(
            preferences: preferences,
            language: language,
            onResetPosition: { [weak self] in self?.resetPosition() },
            onClose: { [weak self] in self?.setVisible(false) }
        )
        // The menu does not retain its items' target; keep the builder until the next menu.
        contextMenu = builder
        NSMenu.popUpContextMenu(builder.makeMenu(), with: event, for: hostingView)
    }

    private func makeView() -> RuntimeHUDView {
        RuntimeHUDView(
            store: store,
            preferences: preferences,
            language: language,
            onSizeChange: { [weak self] size in self?.resize(to: size) }
        )
    }

    private func fitToContent() {
        guard let hostingView else {
            return
        }
        hostingView.layoutSubtreeIfNeeded()
        resize(to: hostingView.fittingSize)
    }

    /// Follows the content's size, keeping the bottom-left corner where it is so a HUD near the
    /// bottom of the screen grows upward.
    private func resize(to size: CGSize) {
        guard let panel, let hostingView, size.width > 0, size.height > 0 else {
            return
        }
        let size = CGSize(width: size.width.rounded(.up), height: size.height.rounded(.up))
        hostingView.frame = CGRect(origin: .zero, size: size)
        guard panel.frame.size != size else {
            return
        }
        isPlacing = true
        panel.setFrame(CGRect(origin: panel.frame.origin, size: size), display: true)
        isPlacing = false
    }

    private func place() {
        guard let panel else {
            return
        }
        let screens = NSScreen.screens
        guard let main = NSScreen.main ?? screens.first else {
            return
        }
        let origin = HUDPlacement.origin(
            saved: preferences.hudOrigins,
            screens: screens.map { (id: HUDPlacement.id(of: $0), visibleFrame: $0.visibleFrame) },
            mainScreenID: HUDPlacement.id(of: main),
            lastScreenID: preferences.hudLastScreenID,
            panelSize: panel.frame.size
        )
        isPlacing = true
        panel.setFrameOrigin(origin)
        isPlacing = false
        needsPlacement = false
    }

    func windowDidMove(_ notification: Notification) {
        guard !isPlacing, let panel, let screen = panel.screen else {
            return
        }
        let id = HUDPlacement.id(of: screen)
        var origins = preferences.hudOrigins
        origins[id] = HUDPlacement.format(panel.frame.origin)
        preferences.hudOrigins = origins
        if preferences.hudLastScreenID != id {
            preferences.hudLastScreenID = id
        }
    }
}
