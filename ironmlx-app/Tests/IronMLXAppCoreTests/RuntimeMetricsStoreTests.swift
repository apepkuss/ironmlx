import Foundation
import Testing

@testable import IronMLXAppCore

private func glanceSnapshot(
    health: RuntimeHealthStatus = .healthy,
    reasons: [String] = [],
    activity: RuntimeActivityState = .idle,
    liveSpeed: Double? = nil,
    startedUnixMs: Int64 = 0
) -> RuntimeMetricsSnapshot {
    RuntimeMetricsSnapshot(
        backend: .init(version: "0.5.0", startedUnixMs: startedUnixMs, deviceName: "Apple M5 Max"),
        health: .init(status: health, degradedReasons: reasons),
        activity: .init(state: activity),
        performance: .init(liveDecodeTokensPerSecond: liveSpeed)
    )
}

@MainActor
private func connectedStore(
    _ snapshot: RuntimeMetricsSnapshot? = nil,
    backendState: BackendProcessState = .running
) -> RuntimeMetricsStore {
    let store = RuntimeMetricsStore()
    store.setBackendState(backendState)
    store.setConnection(.connected)
    if let snapshot {
        store.apply(snapshot, receivedAt: Date(timeIntervalSince1970: 1_000))
    }
    return store
}

@Test @MainActor
func indicatorPriority() {
    // App-side process state wins over any snapshot.
    #expect(connectedStore(glanceSnapshot(activity: .generating), backendState: .stopped).indicator == .red)
    #expect(connectedStore(glanceSnapshot(activity: .generating), backendState: .failed).indicator == .red)
    #expect(connectedStore(glanceSnapshot(), backendState: .starting).indicator == .blue)
    #expect(connectedStore(glanceSnapshot(), backendState: .recovering).indicator == .blue)
    #expect(connectedStore(glanceSnapshot(activity: .generating), backendState: .stopping).indicator == nil)
    #expect(connectedStore(glanceSnapshot(), backendState: .degraded).indicator == .amber)

    // Running: the snapshot decides.
    #expect(connectedStore(glanceSnapshot(health: .down)).indicator == .red)
    #expect(
        connectedStore(glanceSnapshot(health: .degraded, reasons: ["model_failed"], activity: .generating))
            .indicator == .red
    )
    #expect(
        connectedStore(glanceSnapshot(health: .degraded, reasons: ["process_memory_emergency"])).indicator
            == .red
    )
    #expect(
        connectedStore(glanceSnapshot(health: .degraded, reasons: ["scheduler_queue_high"], activity: .generating))
            .indicator == .amber
    )
    #expect(connectedStore(glanceSnapshot(activity: .loading)).indicator == .blue)
    #expect(connectedStore(glanceSnapshot(activity: .prefilling)).indicator == .green)
    #expect(connectedStore(glanceSnapshot(activity: .generating, liveSpeed: 40)).indicator == .green)
    #expect(connectedStore(glanceSnapshot(activity: .idle)).indicator == nil)
    #expect(connectedStore(nil).indicator == nil)
}

@Test @MainActor
func menuBarReadoutRoundsSpeedAndHidesWhenDisabled() {
    let generating = connectedStore(glanceSnapshot(activity: .generating, liveSpeed: 42.6))
    #expect(generating.menuBarReadout(language: "en", enabled: true) == "43 t/s")
    #expect(generating.menuBarReadout(language: "en", enabled: false) == nil)

    // A non-language model (e.g. TTS) generates without a token speed.
    let tts = connectedStore(glanceSnapshot(activity: .generating, liveSpeed: nil))
    #expect(tts.menuBarReadout(language: "en", enabled: true) == nil)

    let prefilling = connectedStore(glanceSnapshot(activity: .prefilling))
    #expect(prefilling.menuBarReadout(language: "en", enabled: true) == "Prefill")
    #expect(prefilling.menuBarReadout(language: "zh-Hans", enabled: true) == "预填充")

    #expect(connectedStore(glanceSnapshot(activity: .idle)).menuBarReadout(language: "en", enabled: true) == nil)
}

@Test
func sparklineCarriesForwardGapsAndClearsOnIdle() async {
    var sparkline = RuntimeSparkline()
    sparkline.record(10, at: Date(timeIntervalSince1970: 100.2))
    sparkline.record(20, at: Date(timeIntervalSince1970: 103.7))
    #expect(sparkline.values == [10, 10, 10, 20])

    // Same second overwrites.
    sparkline.record(25, at: Date(timeIntervalSince1970: 103.9))
    #expect(sparkline.values == [10, 10, 10, 25])

    await MainActor.run {
        let store = connectedStore()
        store.apply(glanceSnapshot(activity: .generating, liveSpeed: 10), receivedAt: Date(timeIntervalSince1970: 0))
        store.apply(glanceSnapshot(activity: .generating, liveSpeed: 20), receivedAt: Date(timeIntervalSince1970: 3))
        #expect(store.sparkline.values == [10, 10, 10, 20])

        store.apply(glanceSnapshot(activity: .idle), receivedAt: Date(timeIntervalSince1970: 4))
        #expect(store.sparkline.values.isEmpty)
    }
}

@Test
func sparklineKeepsSixtySeconds() {
    var sparkline = RuntimeSparkline()
    for second in 0..<70 {
        sparkline.record(Double(second), at: Date(timeIntervalSince1970: TimeInterval(second)))
    }
    #expect(sparkline.values.count == RuntimeSparkline.capacity)
    #expect(sparkline.values.first == 10)
    #expect(sparkline.values.last == 69)

    // A gap longer than the window keeps only the most recent seconds.
    sparkline.record(1, at: Date(timeIntervalSince1970: 10_000))
    #expect(sparkline.values.count == 60)
    #expect(sparkline.values.last == 1)
    #expect(sparkline.values.dropLast().allSatisfy { $0 == 69 })

    sparkline.clear()
    #expect(sparkline.values.isEmpty)
}

@Test @MainActor
func storeMarksSnapshotStaleWhenConnectionDrops() {
    let store = connectedStore(glanceSnapshot(activity: .generating, liveSpeed: 42, startedUnixMs: 1_000))
    #expect(!store.isStale)
    #expect(store.indicator == .green)

    store.setConnection(.reconnecting)

    #expect(store.isStale)
    #expect(store.indicator == nil)
    #expect(store.menuBarReadout(language: "en", enabled: true) == nil)
    #expect(store.uptime(now: Date(timeIntervalSince1970: 100)) == nil)
    #expect(store.sparkline.values.isEmpty)
}

@Test @MainActor
func uptimeFromBackendStart() {
    let now = Date(timeIntervalSince1970: 1_763_280_000)
    let startedUnixMs = Int64((now.timeIntervalSince1970 - 11_520) * 1_000)
    let store = connectedStore(glanceSnapshot(startedUnixMs: startedUnixMs))

    #expect(store.uptime(now: now) == 11_520)
    #expect(connectedStore(glanceSnapshot(startedUnixMs: 0)).uptime(now: now) == nil)
}

@Test
func glanceTextLocalizesAndFallsBack() {
    #expect(GlanceText.localized(.backendNotRunning, language: "en") == "Server not running")
    #expect(GlanceText.localized(.backendNotRunning, language: "zh") == "后端未运行")
    #expect(GlanceText.localized(.uptime, language: "en") == "Up %@")
    #expect(GlanceText.localized(.stateIdle, language: "fr") == "Idle")
    #expect(!GlanceText.localized(.hudClose, language: "ja").isEmpty)
    #expect(GlanceText.localized(.hudClose, language: "zh-Hant") != GlanceText.localized(.hudClose, language: "en"))

    #expect(GlanceText.reasonTitle("scheduler_queue_high", language: "zh") == "排队过多")
    #expect(GlanceText.reasonTitle("process_memory_hard", language: "en") == "Memory pressure")
    #expect(GlanceText.reasonTitle("future_reason", language: "zh") == "future_reason")
}

@Test
func glancePreferencesDefaultsAndPersistence() throws {
    let suite = "ironmlx.glance.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }

    let preferences = GlancePreferences(defaults: defaults)
    #expect(preferences.menuBarSpeedEnabled)
    #expect(!preferences.hudVisible)
    #expect(preferences.hudDensity == .standard)
    #expect(!preferences.hudClickThrough)
    #expect(preferences.hudOrigins.isEmpty)

    preferences.menuBarSpeedEnabled = false
    preferences.hudVisible = true
    preferences.hudDensity = .mini
    preferences.hudClickThrough = true
    preferences.hudOrigins = ["1": "{10, 20}"]

    let reloaded = GlancePreferences(defaults: defaults)
    #expect(!reloaded.menuBarSpeedEnabled)
    #expect(reloaded.hudVisible)
    #expect(reloaded.hudDensity == .mini)
    #expect(reloaded.hudClickThrough)
    #expect(reloaded.hudOrigins == ["1": "{10, 20}"])
    #expect(defaults.object(forKey: "ironmlx.glance.hudDensity") as? String == "mini")
}

@Test @MainActor
func advanceSparklineSamplesLastLiveValueEachSecond() {
    let store = connectedStore()
    store.apply(glanceSnapshot(activity: .generating, liveSpeed: 40), receivedAt: Date(timeIntervalSince1970: 10.2))
    #expect(store.isSamplingSparkline)

    // No new event for two seconds: the timer repeats the last value received.
    store.advanceSparkline(now: Date(timeIntervalSince1970: 11.2))
    store.advanceSparkline(now: Date(timeIntervalSince1970: 12.2))
    #expect(store.sparkline.values == [40, 40, 40])

    store.apply(glanceSnapshot(activity: .generating, liveSpeed: 44), receivedAt: Date(timeIntervalSince1970: 12.6))
    store.advanceSparkline(now: Date(timeIntervalSince1970: 13.1))
    #expect(store.sparkline.values == [40, 40, 44, 44])

    // Idle clears the history and the timer no longer records.
    store.apply(glanceSnapshot(activity: .idle), receivedAt: Date(timeIntervalSince1970: 14))
    #expect(!store.isSamplingSparkline)
    store.advanceSparkline(now: Date(timeIntervalSince1970: 15))
    #expect(store.sparkline.values.isEmpty)

    // Stale data is never sampled.
    store.apply(glanceSnapshot(activity: .generating, liveSpeed: 50), receivedAt: Date(timeIntervalSince1970: 16))
    store.setConnection(.reconnecting)
    #expect(!store.isSamplingSparkline)
    store.advanceSparkline(now: Date(timeIntervalSince1970: 17))
    #expect(store.sparkline.values.isEmpty)

    // Nothing to sample before the first snapshot.
    let empty = connectedStore()
    #expect(!empty.isSamplingSparkline)
    empty.advanceSparkline(now: Date(timeIntervalSince1970: 1))
    #expect(empty.sparkline.values.isEmpty)
}
