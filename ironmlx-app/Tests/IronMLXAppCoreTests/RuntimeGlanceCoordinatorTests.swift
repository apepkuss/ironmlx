import Foundation
import Synchronization
import Testing

@testable import IronMLXAppCore

/// Every connection stays open until the consumer cancels; counts opens and closes.
private final class OpenStreamCounter: RuntimeMetricsStreaming {
    private let counts = Mutex((opened: 0, closed: 0))

    var opened: Int { counts.withLock { $0.opened } }
    var closed: Int { counts.withLock { $0.closed } }

    func snapshots(from url: URL) -> AsyncThrowingStream<RuntimeMetricsSnapshot, Error> {
        counts.withLock { $0.opened += 1 }
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [self] _ in
                self.counts.withLock { $0.closed += 1 }
            }
        }
    }
}

@MainActor
private func waitFor(timeout: Duration = .seconds(5), _ condition: @MainActor () -> Bool) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now + timeout
    while !condition() {
        if clock.now >= deadline {
            Issue.record("condition not met before timeout")
            return
        }
        try await Task.sleep(for: .milliseconds(5))
    }
}

/// A HUD that never creates a window, for tests that start the coordinator.
@MainActor
private final class HeadlessHUD: RuntimeHUDPresenting {
    func setVisible(_ visible: Bool) {}
    func toggle() {}
    func resetPosition() {}
    func setLanguage(_ language: String) {}
}

private func inMemoryPreferences() throws -> (GlancePreferences, () -> Void) {
    let suite = "ironmlx.glance.coordinator-tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    return (GlancePreferences(defaults: defaults), { defaults.removePersistentDomain(forName: suite) })
}

@Test @MainActor
func sparklineTimerRunsOnlyWhileConnectedAndActive() async throws {
    let (preferences, cleanup) = try inMemoryPreferences()
    defer { cleanup() }
    // The backend is stopped, so the client never connects; the test drives the store directly.
    let coordinator = RuntimeGlanceCoordinator(
        preferences: preferences,
        streaming: OpenStreamCounter(),
        endpoint: { nil },
        backendState: { .stopped },
        notificationCenter: NotificationCenter(),
        hotKeyFactory: { _ in nil },
        hudFactory: { _, _, _ in HeadlessHUD() }
    )
    coordinator.start()
    #expect(!coordinator.isSparklineTimerRunning)

    let store = coordinator.store
    store.setBackendState(.running)
    store.setConnection(.connected)
    store.apply(
        RuntimeMetricsSnapshot(activity: .init(state: .generating), performance: .init(liveDecodeTokensPerSecond: 30)),
        receivedAt: Date()
    )
    try await waitFor { coordinator.isSparklineTimerRunning }

    store.apply(RuntimeMetricsSnapshot(activity: .init(state: .idle)), receivedAt: Date())
    try await waitFor { !coordinator.isSparklineTimerRunning }

    store.apply(RuntimeMetricsSnapshot(activity: .init(state: .prefilling)), receivedAt: Date())
    try await waitFor { coordinator.isSparklineTimerRunning }
    store.setConnection(.reconnecting)
    try await waitFor { !coordinator.isSparklineTimerRunning }

    store.setConnection(.connected)
    try await waitFor { coordinator.isSparklineTimerRunning }
    coordinator.stop()
    #expect(!coordinator.isSparklineTimerRunning)
}

@Test @MainActor
func coordinatorTogglesPreferences() throws {
    let (preferences, cleanup) = try inMemoryPreferences()
    defer { cleanup() }
    let coordinator = RuntimeGlanceCoordinator(
        preferences: preferences,
        endpoint: { nil },
        backendState: { .stopped },
        notificationCenter: NotificationCenter()
    )
    #expect(coordinator.hudHotKeyDisplay == nil)

    coordinator.toggleHUD()
    coordinator.toggleMenuBarSpeed()
    #expect(preferences.hudVisible)
    #expect(!preferences.menuBarSpeedEnabled)

    coordinator.toggleHUD()
    #expect(!preferences.hudVisible)
}

@Test @MainActor
func releasingCoordinatorClosesConnection() async throws {
    let (preferences, cleanup) = try inMemoryPreferences()
    defer { cleanup() }
    let streaming = OpenStreamCounter()
    var coordinator: RuntimeGlanceCoordinator? = RuntimeGlanceCoordinator(
        preferences: preferences,
        streaming: streaming,
        endpoint: { URL(string: "http://127.0.0.1:65530/admin/api/metrics/stream") },
        backendState: { .running },
        notificationCenter: NotificationCenter(),
        hotKeyFactory: { _ in nil },
        hudFactory: { _, _, _ in HeadlessHUD() }
    )
    weak var released = coordinator
    coordinator?.start()
    try await waitFor { streaming.opened == 1 }

    coordinator = nil

    #expect(released == nil)
    try await waitFor { streaming.closed == 1 }
}
