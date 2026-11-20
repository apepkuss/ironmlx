import Foundation
import Observation

public enum HUDDensity: String, CaseIterable, Sendable {
    case mini
    case standard
}

/// Interface preferences for the menu bar readout and the HUD. Kept in `UserDefaults`, never in `app_config`.
///
/// Observable, so the menu bar readout and the HUD react to a change without polling. Only
/// changes made through this object are observed, not writes to the defaults from elsewhere.
public final class GlancePreferences: Observable {
    enum Key {
        static let menuBarSpeed = "ironmlx.glance.menuBarSpeed"
        static let hudVisible = "ironmlx.glance.hudVisible"
        static let hudDensity = "ironmlx.glance.hudDensity"
        static let hudAutoHide = "ironmlx.glance.hudAutoHide"
        static let hudClickThrough = "ironmlx.glance.hudClickThrough"
        static let hudOrigins = "ironmlx.glance.hudOrigins"
        static let hudLastScreen = "ironmlx.glance.hudLastScreen"
    }

    private let defaults: UserDefaults
    private let registrar = ObservationRegistrar()

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public var menuBarSpeedEnabled: Bool {
        get {
            registrar.access(self, keyPath: \.menuBarSpeedEnabled)
            return bool(Key.menuBarSpeed, default: true)
        }
        set {
            registrar.withMutation(of: self, keyPath: \.menuBarSpeedEnabled) {
                defaults.set(newValue, forKey: Key.menuBarSpeed)
            }
        }
    }

    public var hudVisible: Bool {
        get {
            registrar.access(self, keyPath: \.hudVisible)
            return bool(Key.hudVisible, default: false)
        }
        set {
            registrar.withMutation(of: self, keyPath: \.hudVisible) {
                defaults.set(newValue, forKey: Key.hudVisible)
            }
        }
    }

    public var hudDensity: HUDDensity {
        get {
            registrar.access(self, keyPath: \.hudDensity)
            return defaults.string(forKey: Key.hudDensity).flatMap(HUDDensity.init(rawValue:)) ?? .standard
        }
        set {
            registrar.withMutation(of: self, keyPath: \.hudDensity) {
                defaults.set(newValue.rawValue, forKey: Key.hudDensity)
            }
        }
    }

    public var hudAutoHide: Bool {
        get {
            registrar.access(self, keyPath: \.hudAutoHide)
            return bool(Key.hudAutoHide, default: true)
        }
        set {
            registrar.withMutation(of: self, keyPath: \.hudAutoHide) {
                defaults.set(newValue, forKey: Key.hudAutoHide)
            }
        }
    }

    public var hudClickThrough: Bool {
        get {
            registrar.access(self, keyPath: \.hudClickThrough)
            return bool(Key.hudClickThrough, default: false)
        }
        set {
            registrar.withMutation(of: self, keyPath: \.hudClickThrough) {
                defaults.set(newValue, forKey: Key.hudClickThrough)
            }
        }
    }

    /// HUD origin per screen: key is the screen ID, value is `NSStringFromPoint`.
    public var hudOrigins: [String: String] {
        get {
            registrar.access(self, keyPath: \.hudOrigins)
            return defaults.dictionary(forKey: Key.hudOrigins) as? [String: String] ?? [:]
        }
        set {
            registrar.withMutation(of: self, keyPath: \.hudOrigins) {
                defaults.set(newValue, forKey: Key.hudOrigins)
            }
        }
    }

    /// The screen the HUD was last moved on, preferred when placing it again.
    public var hudLastScreenID: String? {
        get {
            registrar.access(self, keyPath: \.hudLastScreenID)
            return defaults.string(forKey: Key.hudLastScreen)
        }
        set {
            registrar.withMutation(of: self, keyPath: \.hudLastScreenID) {
                defaults.set(newValue, forKey: Key.hudLastScreen)
            }
        }
    }

    private func bool(_ key: String, default value: Bool) -> Bool {
        defaults.object(forKey: key) as? Bool ?? value
    }
}
