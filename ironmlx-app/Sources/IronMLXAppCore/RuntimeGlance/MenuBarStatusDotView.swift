import AppKit

/// The 6 pt status dot drawn over the bottom-left corner of the menu bar icon. The icon stays
/// a template image; the dot is a separate view so it keeps its color in every appearance.
@MainActor
final class MenuBarStatusDotView: NSView {
    static let diameter: CGFloat = 6
    /// Where the dot's center sits inside the 34 × 22 pt icon, measured from the top-left:
    /// on the bottom-left corner of the "Fe" frame, nudged slightly inward.
    static let centerInIcon = NSPoint(x: 10, y: 18)

    var indicator: GlanceIndicator? {
        didSet {
            guard indicator != oldValue else {
                return
            }
            isHidden = indicator == nil
            needsDisplay = true
        }
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: NSRect(x: 0, y: 0, width: Self.diameter, height: Self.diameter))
        isHidden = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    /// Clicks go to the status item button underneath.
    override func hitTest(_ point: NSPoint) -> NSView? {
        nil
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let indicator else {
            return
        }
        Self.color(for: indicator).setFill()
        NSBezierPath(ovalIn: bounds).fill()
    }

    static func color(for indicator: GlanceIndicator) -> NSColor {
        switch indicator {
        case .red: .systemRed
        case .amber: .systemOrange
        case .blue: .systemBlue
        case .green: .systemGreen
        }
    }

    /// Adds the dot to `button` and keeps it on the icon as the button resizes (the icon moves
    /// when the speed text appears or disappears).
    func install(in button: NSButton) {
        button.addSubview(self)
        button.postsFrameChangedNotifications = true
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(buttonFrameDidChange(_:)),
            name: NSView.frameDidChangeNotification,
            object: button
        )
        reposition()
    }

    @objc private func buttonFrameDidChange(_ notification: Notification) {
        reposition()
    }

    /// Places the dot from the rect the button's cell draws the icon in.
    func reposition() {
        guard let button = superview as? NSButton, let cell = button.cell, let image = button.image else {
            return
        }
        let imageRect = cell.imageRect(forBounds: button.bounds)
        guard imageRect.width > 0, imageRect.height > 0, image.size.width > 0, image.size.height > 0 else {
            return
        }
        let x = imageRect.minX + Self.centerInIcon.x * imageRect.width / image.size.width
        let fromTop = Self.centerInIcon.y * imageRect.height / image.size.height
        let y = button.isFlipped ? imageRect.minY + fromTop : imageRect.maxY - fromTop
        let radius = Self.diameter / 2
        let frame = NSRect(x: x - radius, y: y - radius, width: Self.diameter, height: Self.diameter)
        if self.frame != frame {
            self.frame = frame
        }
    }
}
