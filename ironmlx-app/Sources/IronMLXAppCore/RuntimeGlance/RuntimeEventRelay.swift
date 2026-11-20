import Foundation

/// One event from the metrics stream, or a change in what the App knows about the backend, as
/// the Dashboard page receives it. Payloads are the backend's JSON text, passed on unparsed.
public enum RuntimeRelayEvent: Equatable, Sendable {
    /// The latest snapshot; nil once there is no live snapshot (connection lost, backend not running).
    case metrics(String?)
    case history(String)
    case request(String)
    /// `state` is the supervisor state's lowercase name: starting, running, degraded, recovering,
    /// stopping, stopped or failed. `connected` is whether the metrics stream is up.
    case backend(state: String, connected: Bool)
}

public struct RuntimeRelayToken: Hashable, Sendable {
    fileprivate let id: UInt64
}

/// Fans the runtime events out to whoever is listening (the open Dashboard page) and remembers
/// the latest snapshot and backend state for listeners that join later.
@MainActor
public final class RuntimeEventRelay {
    private var listeners: [(token: RuntimeRelayToken, deliver: @MainActor (RuntimeRelayEvent) -> Void)] = []
    private var nextToken: UInt64 = 0

    /// The latest live snapshot's JSON, or nil when there is none.
    public private(set) var lastMetrics: String?
    /// The latest `.backend` event.
    public private(set) var lastBackend: RuntimeRelayEvent?

    public init() {}

    /// Adds a listener and replays the current state to it: `lastBackend` (if any), then
    /// `.metrics(lastMetrics)`. History points and requests are not replayed.
    @discardableResult
    public func addListener(_ listener: @escaping @MainActor (RuntimeRelayEvent) -> Void) -> RuntimeRelayToken {
        nextToken += 1
        let token = RuntimeRelayToken(id: nextToken)
        listeners.append((token, listener))
        if let lastBackend {
            listener(lastBackend)
        }
        listener(.metrics(lastMetrics))
        return token
    }

    public func removeListener(_ token: RuntimeRelayToken) {
        listeners.removeAll { $0.token == token }
    }

    var listenerCount: Int {
        listeners.count
    }

    func send(_ event: RuntimeRelayEvent) {
        switch event {
        case let .metrics(raw):
            lastMetrics = raw
        case .backend:
            lastBackend = event
        case .history, .request:
            break
        }
        // A copy, so a listener may add or remove listeners while being called.
        for listener in listeners {
            listener.deliver(event)
        }
    }
}
