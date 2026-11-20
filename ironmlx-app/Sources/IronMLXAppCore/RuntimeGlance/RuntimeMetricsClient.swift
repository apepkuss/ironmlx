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

public protocol RuntimeMetricsStreaming: Sendable {
    /// Snapshots from one SSE connection. The stream throws or finishes when the connection ends,
    /// and the connection closes when the consumer stops iterating.
    func snapshots(from url: URL) -> AsyncThrowingStream<RuntimeMetricsSnapshot, Error>
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

    public func snapshots(from url: URL) -> AsyncThrowingStream<RuntimeMetricsSnapshot, Error> {
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
                        guard let event = parser.consume(line: line), event.event == "metrics" else {
                            continue
                        }
                        let snapshot = try decoder.decode(RuntimeMetricsSnapshot.self, from: Data(event.data.utf8))
                        continuation.yield(snapshot)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Keeps one SSE connection to the backend's metrics stream while the backend runs, and feeds
/// every snapshot into a `RuntimeMetricsStore`.
@MainActor
public final class RuntimeMetricsClient {
    public nonisolated static let streamPath = "/admin/api/metrics/stream"

    private let store: RuntimeMetricsStore
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
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.store = store
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
        store.setBackendState(state)
        store.setConnection(Self.keepsConnection(in: state) ? .reconnecting : .backendNotRunning)
    }

    /// Drops the current connection and starts over from the backend's current state.
    private func reconnect() {
        connectionTask?.cancel()
        connectionTask = nil

        let state = backendState()
        store.setBackendState(state)
        guard Self.keepsConnection(in: state) else {
            store.setConnection(.backendNotRunning)
            return
        }
        store.setConnection(.reconnecting)
        connectionTask = Task { [weak self] in
            await self?.run()
        }
    }

    private static func keepsConnection(in state: BackendProcessState) -> Bool {
        state == .running || state == .degraded
    }

    private func run() async {
        var failures = 0
        while !Task.isCancelled {
            if let url = endpoint() {
                do {
                    for try await snapshot in streaming.snapshots(from: url) {
                        if Task.isCancelled {
                            return
                        }
                        failures = 0
                        store.apply(snapshot, receivedAt: Date())
                        store.setConnection(.connected)
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
            store.setConnection(.reconnecting)
            do {
                try await sleep(RuntimeMetricsReconnectPolicy.delay(afterFailures: failures))
            } catch {
                return
            }
        }
    }
}
