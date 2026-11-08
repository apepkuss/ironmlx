import AppKit

/// IronMLX is a menu bar app. While any of its windows is open (Dashboard,
/// chat, no-model window) it is a regular app with a Dock icon and a place
/// in Cmd-Tab, so a covered window can be found again; when the last one
/// closes it goes back to the menu bar only. Every window reports here
/// instead of switching the activation policy itself.
@MainActor
final class AppWindowActivation {
    enum Surface: Hashable, Sendable {
        case dashboard
        case chat
        case noModel
    }

    static let shared = AppWindowActivation { NSApp.setActivationPolicy($0) }

    private let apply: (NSApplication.ActivationPolicy) -> Void
    private var visible: Set<Surface> = []
    /// Matches what `AppDelegate` sets at launch.
    private(set) var policy: NSApplication.ActivationPolicy = .accessory

    init(apply: @escaping (NSApplication.ActivationPolicy) -> Void) {
        self.apply = apply
    }

    /// Call before the window is brought to the front, so it can activate.
    func shown(_ surface: Surface) {
        visible.insert(surface)
        update()
    }

    func hidden(_ surface: Surface) {
        visible.remove(surface)
        update()
    }

    private func update() {
        let next: NSApplication.ActivationPolicy = visible.isEmpty ? .accessory : .regular
        guard next != policy else { return }
        policy = next
        apply(next)
    }
}
