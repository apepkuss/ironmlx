import Foundation
import Synchronization
import Testing

@testable import IronMLXAppCore

private struct FakeStreamFailure: Error {}

private enum FakeStreamScript: Sendable {
    /// Fail right away, as a refused connection would.
    case fail
    /// Deliver these snapshots, then drop the connection.
    case deliverThenFail([RuntimeMetricsSnapshot])
    /// Stay open until the consumer cancels.
    case hang
    /// Deliver these snapshots, then stay open until the consumer cancels.
    case deliverThenHang([RuntimeMetricsSnapshot])
}

private final class FakeRuntimeMetricsStreaming: RuntimeMetricsStreaming {
    private struct State {
        var script: [FakeStreamScript]
        var requestedURLs: [URL] = []
        var terminations = 0
    }

    private let state: Mutex<State>

    init(script: [FakeStreamScript]) {
        state = Mutex(State(script: script))
    }

    var callCount: Int { state.withLock { $0.requestedURLs.count } }
    var requestedURLs: [URL] { state.withLock { $0.requestedURLs } }
    var terminations: Int { state.withLock { $0.terminations } }

    func snapshots(from url: URL) -> AsyncThrowingStream<RuntimeMetricsSnapshot, Error> {
        let step = state.withLock { state -> FakeStreamScript in
            state.requestedURLs.append(url)
            return state.script.isEmpty ? .fail : state.script.removeFirst()
        }
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [self] _ in
                self.state.withLock { $0.terminations += 1 }
            }
            switch step {
            case .fail:
                continuation.finish(throwing: FakeStreamFailure())
            case let .deliverThenFail(snapshots):
                for snapshot in snapshots {
                    continuation.yield(snapshot)
                }
                continuation.finish(throwing: FakeStreamFailure())
            case .hang:
                break
            case let .deliverThenHang(snapshots):
                for snapshot in snapshots {
                    continuation.yield(snapshot)
                }
            }
        }
    }
}

/// Records every backoff wait and ends the client's loop after `limit` waits.
private final class SleepRecorder: Sendable {
    private let delays = Mutex<[Duration]>([])
    private let limit: Int

    init(limit: Int) {
        self.limit = limit
    }

    var recorded: [Duration] { delays.withLock { $0 } }

    var sleep: @Sendable (Duration) async throws -> Void {
        { [self] duration in
            let count = delays.withLock { delays -> Int in
                delays.append(duration)
                return delays.count
            }
            if count >= limit {
                throw CancellationError()
            }
        }
    }
}

@MainActor
private final class BackendStateBox {
    var state: BackendProcessState
    init(_ state: BackendProcessState) { self.state = state }
}

@MainActor
private func waitUntil(
    timeout: Duration = .seconds(5),
    _ condition: @MainActor () -> Bool
) async throws {
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

private let testEndpoint = URL(string: "http://127.0.0.1:65530/admin/api/metrics/stream")!

private func liveSnapshot() -> RuntimeMetricsSnapshot {
    RuntimeMetricsSnapshot(
        health: .init(status: .healthy),
        activity: .init(state: .generating, activeRequests: 1),
        performance: .init(liveDecodeTokensPerSecond: 40)
    )
}

@MainActor
private func makeClient(
    streaming: FakeRuntimeMetricsStreaming,
    sleeper: SleepRecorder,
    backend: BackendStateBox,
    notificationCenter: NotificationCenter = NotificationCenter()
) -> (RuntimeMetricsClient, RuntimeMetricsStore) {
    let store = RuntimeMetricsStore()
    let client = RuntimeMetricsClient(
        store: store,
        streaming: streaming,
        endpoint: { testEndpoint },
        backendState: { backend.state },
        notificationCenter: notificationCenter,
        sleep: sleeper.sleep
    )
    return (client, store)
}

@Test
func reconnectPolicyDelays() {
    #expect(RuntimeMetricsReconnectPolicy.delay(afterFailures: 1) == .milliseconds(500))
    #expect(RuntimeMetricsReconnectPolicy.delay(afterFailures: 2) == .seconds(1))
    #expect(RuntimeMetricsReconnectPolicy.delay(afterFailures: 3) == .seconds(2))
    #expect(RuntimeMetricsReconnectPolicy.delay(afterFailures: 4) == .seconds(5))
    #expect(RuntimeMetricsReconnectPolicy.delay(afterFailures: 40) == .seconds(5))
}

@Test @MainActor
func reconnectDelaysFollowPolicy() async throws {
    let streaming = FakeRuntimeMetricsStreaming(script: [])
    let sleeper = SleepRecorder(limit: 5)
    let (client, store) = makeClient(streaming: streaming, sleeper: sleeper, backend: BackendStateBox(.running))

    client.start()
    try await waitUntil { sleeper.recorded.count == 5 }
    client.stop()

    #expect(sleeper.recorded == [.milliseconds(500), .seconds(1), .seconds(2), .seconds(5), .seconds(5)])
    #expect(streaming.callCount == 5)
    #expect(streaming.requestedURLs.allSatisfy { $0 == testEndpoint })
    #expect(store.connection == .reconnecting)
    #expect(store.backendState == .running)
}

@Test @MainActor
func successfulSnapshotResetsBackoff() async throws {
    let streaming = FakeRuntimeMetricsStreaming(
        script: [.fail, .fail, .deliverThenFail([liveSnapshot()]), .fail]
    )
    let sleeper = SleepRecorder(limit: 4)
    let (client, store) = makeClient(streaming: streaming, sleeper: sleeper, backend: BackendStateBox(.running))

    client.start()
    try await waitUntil { sleeper.recorded.count == 4 }
    client.stop()

    #expect(sleeper.recorded == [.milliseconds(500), .seconds(1), .milliseconds(500), .seconds(1)])
    #expect(store.snapshot == liveSnapshot())
    #expect(store.isStale)
}

@Test @MainActor
func doesNotConnectWhileBackendStopped() async throws {
    let streaming = FakeRuntimeMetricsStreaming(script: [.hang])
    let sleeper = SleepRecorder(limit: 1)
    let backend = BackendStateBox(.stopped)
    let notificationCenter = NotificationCenter()
    let (client, store) = makeClient(
        streaming: streaming, sleeper: sleeper, backend: backend, notificationCenter: notificationCenter
    )

    client.start()
    try await Task.sleep(for: .milliseconds(50))

    #expect(streaming.callCount == 0)
    #expect(store.connection == .backendNotRunning)
    #expect(store.backendState == .stopped)
    #expect(store.indicator == .red)

    backend.state = .starting
    notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: nil)
    try await Task.sleep(for: .milliseconds(50))
    #expect(streaming.callCount == 0)
    #expect(store.backendState == .starting)
    #expect(store.indicator == .blue)

    client.stop()
}

@Test @MainActor
func runtimeChangeNotificationReconnectsImmediately() async throws {
    let streaming = FakeRuntimeMetricsStreaming(
        script: [.hang, .deliverThenFail([]), .hang]
    )
    // Any backoff wait would be a failure here, so a single wait ends the loop.
    let sleeper = SleepRecorder(limit: 100)
    let backend = BackendStateBox(.running)
    let notificationCenter = NotificationCenter()
    let (client, store) = makeClient(
        streaming: streaming, sleeper: sleeper, backend: backend, notificationCenter: notificationCenter
    )

    client.start()
    try await waitUntil { streaming.callCount == 1 }

    // A restart cancels the open connection and reconnects without any backoff.
    backend.state = .degraded
    notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: nil)
    try await waitUntil { streaming.callCount >= 2 && streaming.terminations >= 1 }
    #expect(store.backendState == .degraded)

    // The second connection drops; the client backs off once, then connects again.
    try await waitUntil { streaming.callCount == 3 }
    #expect(sleeper.recorded == [.milliseconds(500)])

    // Stopping the backend tears the connection down and stays disconnected.
    backend.state = .stopped
    notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: nil)
    try await waitUntil { streaming.terminations == 3 }
    #expect(store.connection == .backendNotRunning)
    #expect(store.backendState == .stopped)
    try await Task.sleep(for: .milliseconds(50))
    #expect(streaming.callCount == 3)

    client.stop()
}

@Test @MainActor
func stopCancelsOpenConnection() async throws {
    let streaming = FakeRuntimeMetricsStreaming(script: [.hang])
    let sleeper = SleepRecorder(limit: 1)
    let (client, _) = makeClient(streaming: streaming, sleeper: sleeper, backend: BackendStateBox(.running))

    client.start()
    try await waitUntil { streaming.callCount == 1 }
    client.stop()
    try await waitUntil { streaming.terminations == 1 }
    #expect(sleeper.recorded.isEmpty)
}

@Test @MainActor
func stopLeavesStoreStaleInsteadOfConnected() async throws {
    let streaming = FakeRuntimeMetricsStreaming(script: [.deliverThenHang([liveSnapshot()])])
    let sleeper = SleepRecorder(limit: 1)
    let backend = BackendStateBox(.running)
    let (client, store) = makeClient(streaming: streaming, sleeper: sleeper, backend: backend)

    client.start()
    try await waitUntil { store.connection == .connected }
    #expect(store.menuBarReadout(language: "en", enabled: true) == "40 t/s")

    client.stop()

    // The backend is still running as far as the App knows; only the numbers are no longer current.
    #expect(store.connection == .reconnecting)
    #expect(store.isStale)
    #expect(store.menuBarReadout(language: "en", enabled: true) == nil)
    #expect(store.sparkline.values.isEmpty)
    try await waitUntil { streaming.terminations == 1 }

    // With the backend stopped, stop() reports it as not running.
    let stoppedBackend = BackendStateBox(.stopped)
    let (stoppedClient, stoppedStore) = makeClient(
        streaming: FakeRuntimeMetricsStreaming(script: []), sleeper: SleepRecorder(limit: 1), backend: stoppedBackend
    )
    stoppedClient.start()
    stoppedStore.setConnection(.connected)
    stoppedClient.stop()
    #expect(stoppedStore.connection == .backendNotRunning)
}

@Test
func streamEndpointMatchesBackendClientHost() throws {
    #expect(
        RuntimeMetricsClient.streamURL(host: "0.0.0.0", port: 8080)?.absoluteString
            == "http://127.0.0.1:8080/admin/api/metrics/stream"
    )
    #expect(
        RuntimeMetricsClient.streamURL(host: "127.0.0.1", port: 11435)?.absoluteString
            == "http://127.0.0.1:11435/admin/api/metrics/stream"
    )
}
