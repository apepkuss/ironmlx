import Foundation
import Observation

public enum GlanceIndicator: Equatable, Sendable {
    case red, amber, blue, green
}

/// Per-second history of the live decode speed, newest last.
public struct RuntimeSparkline: Equatable, Sendable {
    public static let capacity = 60

    public private(set) var values: [Double?] = []
    private var lastSecond: Int64?

    public init() {}

    /// Records `value` in the bucket of the whole second containing `date`. A later value in the
    /// same second replaces the earlier one; skipped seconds repeat the previous value.
    public mutating func record(_ value: Double?, at date: Date) {
        let second = Int64(date.timeIntervalSince1970.rounded(.down))
        guard let lastSecond, !values.isEmpty else {
            values = [value]
            self.lastSecond = second
            return
        }
        guard second > lastSecond else {
            // Same second, or the clock went backwards: overwrite the newest bucket.
            values[values.count - 1] = value
            return
        }
        let skipped = Int(min(second - lastSecond - 1, Int64(Self.capacity)))
        if skipped > 0, let previous = values.last {
            values.append(contentsOf: repeatElement(previous, count: skipped))
        }
        values.append(value)
        if values.count > Self.capacity {
            values.removeFirst(values.count - Self.capacity)
        }
        self.lastSecond = second
    }

    public mutating func clear() {
        values.removeAll()
        lastSecond = nil
    }
}

/// The runtime metrics shared by the menu bar readout, the menu status card and the HUD.
@MainActor
@Observable
public final class RuntimeMetricsStore {
    public private(set) var snapshot: RuntimeMetricsSnapshot?
    public private(set) var connection: RuntimeMetricsConnection
    public private(set) var backendState: BackendProcessState
    public private(set) var sparkline = RuntimeSparkline()

    public init(
        connection: RuntimeMetricsConnection = .backendNotRunning,
        backendState: BackendProcessState = .stopped
    ) {
        self.connection = connection
        self.backendState = backendState
    }

    /// True whenever the last snapshot may be out of date; views must not show its numbers then.
    public var isStale: Bool {
        connection != .connected
    }

    private var liveSnapshot: RuntimeMetricsSnapshot? {
        isStale ? nil : snapshot
    }

    /// The status dot next to the menu bar icon; nil means no dot.
    public var indicator: GlanceIndicator? {
        switch backendState {
        case .stopped, .failed:
            return .red
        case .starting, .recovering:
            return .blue
        case .stopping:
            return nil
        case .degraded:
            return .amber
        case .running:
            break
        }
        guard let snapshot = liveSnapshot else {
            return nil
        }
        let reasons = snapshot.health.degradedReasons
        if snapshot.health.status == .down
            || reasons.contains("process_memory_emergency")
            || reasons.contains("model_failed")
        {
            return .red
        }
        if snapshot.health.status == .degraded {
            return .amber
        }
        switch snapshot.activity.state {
        case .loading:
            return .blue
        case .generating, .prefilling:
            return .green
        case .idle, .unknown:
            return nil
        }
    }

    /// The text shown right of the menu bar icon, or nil to show the icon alone.
    public func menuBarReadout(language: String, enabled: Bool) -> String? {
        guard enabled, let snapshot = liveSnapshot else {
            return nil
        }
        switch snapshot.activity.state {
        case .generating:
            guard let speed = snapshot.performance.liveDecodeTokensPerSecond, speed.isFinite else {
                return nil
            }
            return "\(Int(speed.rounded())) t/s"
        case .prefilling:
            return GlanceText.localized(.prefillShort, language: language)
        case .idle, .loading, .unknown:
            return nil
        }
    }

    /// Seconds since the backend started, from the latest live snapshot.
    public func uptime(now: Date) -> TimeInterval? {
        guard let startedUnixMs = liveSnapshot?.backend.startedUnixMs, startedUnixMs > 0 else {
            return nil
        }
        return max(0, now.timeIntervalSince1970 - TimeInterval(startedUnixMs) / 1_000)
    }

    public func apply(_ snapshot: RuntimeMetricsSnapshot, receivedAt: Date) {
        self.snapshot = snapshot
        if snapshot.activity.state == .idle {
            sparkline.clear()
        } else {
            sparkline.record(snapshot.performance.liveDecodeTokensPerSecond, at: receivedAt)
        }
    }

    /// True while the sparkline should be sampled every second: the data is live and the
    /// backend is not idle. The per-second timer runs only while this holds.
    public var isSamplingSparkline: Bool {
        guard let snapshot = liveSnapshot else {
            return false
        }
        return snapshot.activity.state != .idle
    }

    /// Records the most recent live decode speed in the bucket for `now`. The backend pushes
    /// only on meaningful change, so a steady speed can mean seconds without events; a
    /// once-per-second call keeps the history moving. Does nothing for stale or idle data.
    public func advanceSparkline(now: Date) {
        guard isSamplingSparkline, let snapshot else {
            return
        }
        sparkline.record(snapshot.performance.liveDecodeTokensPerSecond, at: now)
    }

    public func setConnection(_ connection: RuntimeMetricsConnection) {
        guard self.connection != connection else {
            return
        }
        self.connection = connection
        if connection != .connected {
            // History across a gap would be filled with stale values; start over on reconnect.
            sparkline.clear()
        }
    }

    public func setBackendState(_ state: BackendProcessState) {
        guard backendState != state else {
            return
        }
        backendState = state
    }
}
