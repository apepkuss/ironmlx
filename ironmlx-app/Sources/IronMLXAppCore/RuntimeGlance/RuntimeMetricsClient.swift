import Foundation

public enum RuntimeMetricsConnection: Equatable, Sendable {
    case connected
    case reconnecting
    case backendNotRunning
}

enum RuntimeMetricsReconnectPolicy {
    /// Backoff before the next attempt after `count` consecutive failures: 0.5 s, 1 s, 2 s, then 5 s.
    static func delay(afterFailures count: Int) -> Duration {
        switch count {
        case ...1: .milliseconds(500)
        case 2: .seconds(1)
        case 3: .seconds(2)
        default: .seconds(5)
        }
    }
}

/// One event from the backend's metrics stream. `raw` is the event's JSON data line as the
/// backend sent it, for the Dashboard page.
public enum RuntimeMetricsStreamEvent: Sendable {
    case metrics(RuntimeMetricsSnapshot, raw: String)
    case history(raw: String)
    case request(raw: String)
}

public protocol RuntimeMetricsStreaming: Sendable {
    /// Events from one SSE connection. The stream throws or finishes when the connection ends,
    /// and the connection closes when the consumer stops iterating.
    func events(from url: URL) -> AsyncThrowingStream<RuntimeMetricsStreamEvent, Error>
}

public enum RuntimeMetricsStreamError: Error, Equatable {
    case unexpectedStatus(Int)
    case lineTooLong
}

public struct URLSessionRuntimeMetricsStreaming: RuntimeMetricsStreaming {
    /// The backend sends a keepalive comment every 15 s, so a connection silent for this long is dead.
    static let idleTimeout: TimeInterval = 45
    /// A full snapshot is about 1 KB; anything far larger is not our backend.
    static let maxLineBytes = 1 << 20

    private let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func events(from url: URL) -> AsyncThrowingStream<RuntimeMetricsStreamEvent, Error> {
        let session = session
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: url)
                    request.timeoutInterval = Self.idleTimeout
                    request.cachePolicy = .reloadIgnoringLocalCacheData
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    let (bytes, response) = try await session.bytes(for: request)
                    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard status == 200 else {
                        throw RuntimeMetricsStreamError.unexpectedStatus(status)
                    }
                    let decoder = RuntimeMetricsSnapshot.makeDecoder()
                    var lines = ServerSentEventLineBuffer()
                    var parser = ServerSentEventParser()
                    for try await byte in bytes {
                        guard let line = lines.append(byte) else {
                            if lines.pendingByteCount > Self.maxLineBytes {
                                throw RuntimeMetricsStreamError.lineTooLong
                            }
                            continue
                        }
                        guard let event = parser.consume(line: line) else {
                            continue
                        }
                        switch event.event {
                        case "metrics":
                            let snapshot = try decoder.decode(RuntimeMetricsSnapshot.self, from: Data(event.data.utf8))
                            continuation.yield(.metrics(snapshot, raw: event.data))
                        case "history" where Self.isJSONObject(event.data):
                            continuation.yield(.history(raw: event.data))
                        case "request" where Self.isJSONObject(event.data):
                            continuation.yield(.request(raw: event.data))
                        default:
                            continue
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The Dashboard evaluates the raw text as a JavaScript expression, so only a JSON object
    /// may pass; anything else is dropped without ending the stream.
    static func isJSONObject(_ text: String) -> Bool {
        (try? JSONSerialization.jsonObject(with: Data(text.utf8))) is [String: Any]
    }
}

/// Keeps one SSE connection to the backend's metrics stream while the backend runs, feeds
/// every snapshot into a `RuntimeMetricsStore`, and passes every event, plus backend state
/// changes, to a `RuntimeEventRelay` for the Dashboard.
@MainActor
public final class RuntimeMetricsClient {
    public nonisolated static let streamPath = "/admin/api/metrics/stream"

    private let store: RuntimeMetricsStore
    private let relay: RuntimeEventRelay
    private let streaming: any RuntimeMetricsStreaming
    private let endpoint: @MainActor () -> URL?
    private let backendState: @MainActor () -> BackendProcessState
    private let notificationCenter: NotificationCenter
    private let sleep: @Sendable (Duration) async throws -> Void

    private var connectionTask: Task<Void, Never>?
    private var runtimeObserver: (any NSObjectProtocol)?

    public init(
        store: RuntimeMetricsStore,
        streaming: any RuntimeMetricsStreaming,
        endpoint: @escaping @MainActor () -> URL?,
        backendState: @escaping @MainActor () -> BackendProcessState,
        notificationCenter: NotificationCenter,
        relay: RuntimeEventRelay? = nil,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.store = store
        self.relay = relay ?? RuntimeEventRelay()
        self.streaming = streaming
        self.endpoint = endpoint
        self.backendState = backendState
        self.notificationCenter = notificationCenter
        self.sleep = sleep
    }

    /// The stream URL for a configured backend address, built the same way as `BackendAPIClient`.
    /// The App talks to the loopback listener, which needs no API key (LAN keys only guard the LAN listener).
    public nonisolated static func streamURL(host: String, port: UInt16) -> URL? {
        URL(string: "http://\(BackendAPIClient.connectableHost(for: host)):\(port)\(streamPath)")
    }

    public func start() {
        if runtimeObserver == nil {
            runtimeObserver = notificationCenter.addObserver(
                forName: .ironMLXBackendRuntimeDidChange,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                if Thread.isMainThread {
                    MainActor.assumeIsolated { self?.reconnect() }
                } else {
                    Task { @MainActor in self?.reconnect() }
                }
            }
        }
        reconnect()
    }

    public func stop() {
        if let runtimeObserver {
            notificationCenter.removeObserver(runtimeObserver)
            self.runtimeObserver = nil
        }
        connectionTask?.cancel()
        connectionTask = nil
        // Nothing feeds the store any more, so its last snapshot must not read as live.
        // Same invariant as `reconnect()`: "not running" only when the backend is not
        // running; otherwise the data is merely out of date.
        let state = backendState()
        publish(state: state, connection: Self.keepsConnection(in: state) ? .reconnecting : .backendNotRunning)
    }

    /// Drops the current connection and starts over from the backend's current state.
    private func reconnect() {
        connectionTask?.cancel()
        connectionTask = nil

        let state = backendState()
        guard Self.keepsConnection(in: state) else {
            publish(state: state, connection: .backendNotRunning)
            return
        }
        publish(state: state, connection: .reconnecting)
        connectionTask = Task { [weak self] in
            await self?.run()
        }
    }

    private static func keepsConnection(in state: BackendProcessState) -> Bool {
        state == .running || state == .degraded
    }

    /// The single place backend state and connection change: updates the store and tells the
    /// relay what changed. Without a connection there is no live snapshot to show.
    private func publish(state: BackendProcessState, connection: RuntimeMetricsConnection) {
        store.setBackendState(state)
        store.setConnection(connection)
        let backend = RuntimeRelayEvent.backend(
            state: Self.relayStateName(state),
            connected: connection == .connected
        )
        if relay.lastBackend != backend {
            relay.send(backend)
        }
        if connection != .connected, relay.lastMetrics != nil {
            relay.send(.metrics(nil))
        }
    }

    /// The backend state's name on the Dashboard page.
    nonisolated static func relayStateName(_ state: BackendProcessState) -> String {
        switch state {
        case .starting: "starting"
        case .running: "running"
        case .degraded: "degraded"
        case .recovering: "recovering"
        case .stopping: "stopping"
        case .stopped: "stopped"
        case .failed: "failed"
        }
    }

    private func run() async {
        var failures = 0
        while !Task.isCancelled {
            if let url = endpoint() {
                do {
                    for try await event in streaming.events(from: url) {
                        if Task.isCancelled {
                            return
                        }
                        failures = 0
                        switch event {
                        case let .metrics(snapshot, raw):
                            store.apply(snapshot, receivedAt: Date())
                            publish(state: store.backendState, connection: .connected)
                            relay.send(.metrics(raw))
                        case let .history(raw):
                            relay.send(.history(raw))
                        case let .request(raw):
                            relay.send(.request(raw))
                        }
                    }
                } catch {
                    // Any failure, including a decode error, falls through to the backoff below.
                }
            }
            if Task.isCancelled {
                return
            }
            // A stream that ends, cleanly or not, is a lost connection.
            failures += 1
            publish(state: store.backendState, connection: .reconnecting)
            do {
                try await sleep(RuntimeMetricsReconnectPolicy.delay(afterFailures: failures))
            } catch {
                return
            }
        }
    }
}
