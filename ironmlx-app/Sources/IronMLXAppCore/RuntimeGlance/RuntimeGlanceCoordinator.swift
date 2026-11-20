import Foundation
import Observation

/// What the menu bar controller needs from the runtime glance surfaces.
@MainActor
public protocol RuntimeGlanceControlling: AnyObject {
    var store: RuntimeMetricsStore { get }
    var preferences: GlancePreferences { get }
    /// The HUD hotkey for display, or nil when it is not registered.
    var hudHotKeyDisplay: String? { get }
    /// The menu item: flips the "Show Floating HUD" preference.
    func toggleHUD()
    /// One ⌃⌥⌘H press, from the Carbon hotkey or from the menu's key equivalent while it is open.
    func hudHotKeyPressed()
    func toggleMenuBarSpeed()
}

/// Owns the metrics store, the one metrics client the App keeps for its lifetime, the glance
/// preferences, the floating HUD and its global hotkey; drives the per-second sparkline sample.
@MainActor
public final class RuntimeGlanceCoordinator: RuntimeGlanceControlling {
    typealias HotKeyFactory = @MainActor (_ handler: @escaping @MainActor () -> Void) -> AnyObject?
    typealias HUDFactory = @MainActor (RuntimeMetricsStore, GlancePreferences, String) -> any RuntimeHUDPresenting

    public let store: RuntimeMetricsStore
    public let preferences: GlancePreferences
    /// The runtime events for the Dashboard page, fed by the one metrics client.
    public let relay: RuntimeEventRelay
    private let client: RuntimeMetricsClient
    private let notificationCenter: NotificationCenter
    private let language: @MainActor () -> String
    private let hotKeyFactory: HotKeyFactory
    private let hudFactory: HUDFactory
    private var isStarted = false
    private var sparklineTimer: Timer?
    private var hud: (any RuntimeHUDPresenting)?
    /// Keeps the registered hotkey alive; nil when registration failed or before `start()`.
    private var hotKey: AnyObject?
    private var languageObserver: (any NSObjectProtocol)?
    private let now: @MainActor () -> Date
    private var lastHotKeyPress: Date?

    /// Deliveries of ⌃⌥⌘H closer together than this count as one press, in case one press ever
    /// reaches both the Carbon hotkey and the open menu's key equivalent.
    static let hotKeyRepeatWindow: TimeInterval = 0.3

    /// The backend state the glance surfaces and the Dashboard show. The supervisor marks the
    /// backend failed when restoring every model failed, although its process keeps serving;
    /// that reads as degraded, as a partial restore does, so live metrics keep flowing.
    public nonisolated static func observedBackendState(
        _ state: BackendProcessState,
        processRunning: Bool
    ) -> BackendProcessState {
        state == .failed && processRunning ? .degraded : state
    }

    /// `language` is read at start and whenever the menu language changes, never per update.
    public convenience init(
        store: RuntimeMetricsStore = RuntimeMetricsStore(),
        preferences: GlancePreferences = GlancePreferences(),
        streaming: any RuntimeMetricsStreaming = URLSessionRuntimeMetricsStreaming(),
        endpoint: @escaping @MainActor () -> URL?,
        backendState: @escaping @MainActor () -> BackendProcessState,
        notificationCenter: NotificationCenter = .default,
        language: @escaping @MainActor () -> String = { "en" }
    ) {
        self.init(
            store: store,
            preferences: preferences,
            streaming: streaming,
            endpoint: endpoint,
            backendState: backendState,
            notificationCenter: notificationCenter,
            language: language,
            hotKeyFactory: { GlobalHotKey.hudToggle(handler: $0) },
            hudFactory: { RuntimeHUDController(store: $0, preferences: $1, language: $2) }
        )
    }

    /// Tests pass factories that neither register a system-wide hotkey nor create a window.
    init(
        store: RuntimeMetricsStore = RuntimeMetricsStore(),
        preferences: GlancePreferences = GlancePreferences(),
        streaming: any RuntimeMetricsStreaming = URLSessionRuntimeMetricsStreaming(),
        endpoint: @escaping @MainActor () -> URL?,
        backendState: @escaping @MainActor () -> BackendProcessState,
        notificationCenter: NotificationCenter = .default,
        language: @escaping @MainActor () -> String = { "en" },
        hotKeyFactory: @escaping HotKeyFactory,
        hudFactory: @escaping HUDFactory,
        now: @escaping @MainActor () -> Date = { Date() }
    ) {
        self.now = now
        self.store = store
        self.preferences = preferences
        self.notificationCenter = notificationCenter
        self.language = language
        self.hotKeyFactory = hotKeyFactory
        self.hudFactory = hudFactory
        let relay = RuntimeEventRelay()
        self.relay = relay
        self.client = RuntimeMetricsClient(
            store: store,
            streaming: streaming,
            endpoint: endpoint,
            backendState: backendState,
            notificationCenter: notificationCenter,
            relay: relay
        )
    }

    /// The client's connection task keeps the client alive, so it must be stopped here;
    /// nothing else holds the coordinator's resources.
    isolated deinit {
        sparklineTimer?.invalidate()
        client.stop()
        if let languageObserver {
            notificationCenter.removeObserver(languageObserver)
        }
    }

    /// The hotkey shown on "Show Floating HUD", only while it is registered.
    public var hudHotKeyDisplay: String? {
        hotKey == nil ? nil : GlobalHotKey.hudDisplay
    }

    /// Starts the metrics stream, creates the HUD (shown now if the user left it on) and
    /// registers the HUD hotkey.
    public func start() {
        guard !isStarted else {
            return
        }
        isStarted = true
        client.start()
        updateSparklineSampling()
        let hud = hud ?? hudFactory(store, preferences, language())
        self.hud = hud
        observeLanguage()
        hotKey = hotKeyFactory { [weak self] in
            self?.hudHotKeyPressed()
        }
        if preferences.hudVisible {
            hud.setVisible(true)
        }
    }

    /// Stops the stream and releases the hotkey. The HUD keeps the user's choice; it hides or
    /// shows its last state as the store goes stale.
    public func stop() {
        guard isStarted else {
            return
        }
        isStarted = false
        hotKey = nil
        client.stop()
        updateSparklineSampling()
    }

    /// The menu item: flips the preference its check mark shows. The hotkey instead toggles what
    /// is on screen (see `RuntimeHUDController.toggle()`).
    public func toggleHUD() {
        guard let hud else {
            preferences.hudVisible.toggle()
            return
        }
        hud.setVisible(!preferences.hudVisible)
    }

    /// Toggles what is on screen (see `RuntimeHUDController.toggle()`), once per press however
    /// many paths the press arrives by.
    public func hudHotKeyPressed() {
        let now = now()
        if let last = lastHotKeyPress, now >= last, now.timeIntervalSince(last) < Self.hotKeyRepeatWindow {
            return
        }
        lastHotKeyPress = now
        guard let hud else {
            preferences.hudVisible.toggle()
            return
        }
        hud.toggle()
    }

    public func toggleMenuBarSpeed() {
        preferences.menuBarSpeedEnabled.toggle()
    }

    private func observeLanguage() {
        guard languageObserver == nil else {
            return
        }
        // No queue: delivered synchronously on the posting thread, which is the main thread for
        // the Dashboard's language setting; hop there otherwise.
        languageObserver = notificationCenter.addObserver(
            forName: .ironMLXMenuLanguageDidChange, object: nil, queue: nil
        ) { [weak self] _ in
            if Thread.isMainThread {
                MainActor.assumeIsolated { self?.languageDidChange() }
            } else {
                Task { @MainActor in self?.languageDidChange() }
            }
        }
    }

    private func languageDidChange() {
        hud?.setLanguage(language())
    }

    var isSparklineTimerRunning: Bool {
        sparklineTimer != nil
    }

    /// Runs the 1 s sparkline timer only while the store has live, non-idle data, so an idle
    /// App has no periodic wakeups. Re-evaluates whenever the store's connection or snapshot changes.
    private func updateSparklineSampling() {
        guard isStarted else {
            stopSparklineTimer()
            return
        }
        let sampling = withObservationTracking {
            store.isSamplingSparkline
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                self?.updateSparklineSampling()
            }
        }
        if sampling {
            startSparklineTimer()
        } else {
            stopSparklineTimer()
        }
    }

    private func startSparklineTimer() {
        guard sparklineTimer == nil else {
            return
        }
        let timer = Timer(timeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.store.advanceSparkline(now: Date())
            }
        }
        timer.tolerance = 0.1
        // Common modes, so the sparkline keeps moving while the status menu is open.
        RunLoop.main.add(timer, forMode: .common)
        sparklineTimer = timer
    }

    private func stopSparklineTimer() {
        sparklineTimer?.invalidate()
        sparklineTimer = nil
    }
}
