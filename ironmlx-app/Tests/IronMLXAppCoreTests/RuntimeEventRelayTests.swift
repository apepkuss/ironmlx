import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
private final class RelayEventLog {
    var events: [RuntimeRelayEvent] = []
}

@MainActor
private func waitForRelay(timeout: Duration = .seconds(5), _ condition: @MainActor () -> Bool) async throws {
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

@Test @MainActor
func relayReplaysLastMetricsAndBackendToNewListener() {
    let relay = RuntimeEventRelay()
    relay.send(.backend(state: "starting", connected: false))
    relay.send(.backend(state: "running", connected: true))
    relay.send(.metrics(#"{"schema_version":1}"#))
    relay.send(.history(#"{"t":5000}"#))
    relay.send(.request(#"{"id":1}"#))

    let log = RelayEventLog()
    relay.addListener { log.events.append($0) }

    // Only the latest backend state and snapshot are replayed, backend first; history and
    // requests are not, the page fetches those itself.
    #expect(log.events == [.backend(state: "running", connected: true), .metrics(#"{"schema_version":1}"#)])
    #expect(relay.lastMetrics == #"{"schema_version":1}"#)
    #expect(relay.lastBackend == .backend(state: "running", connected: true))

    relay.send(.history(#"{"t":10000}"#))
    #expect(log.events.last == .history(#"{"t":10000}"#))

    // Once the snapshot is no longer live, a new listener is told so.
    relay.send(.metrics(nil))
    #expect(relay.lastMetrics == nil)
    let late = RelayEventLog()
    relay.addListener { late.events.append($0) }
    #expect(late.events == [.backend(state: "running", connected: true), .metrics(nil)])
}

@Test @MainActor
func relayStopsDeliveringAfterRemove() {
    let relay = RuntimeEventRelay()
    let removed = RelayEventLog()
    let kept = RelayEventLog()
    let token = relay.addListener { removed.events.append($0) }
    relay.addListener { kept.events.append($0) }
    #expect(relay.listenerCount == 2)

    relay.send(.request(#"{"id":1}"#))
    relay.removeListener(token)
    relay.removeListener(token)
    relay.send(.request(#"{"id":2}"#))

    #expect(relay.listenerCount == 1)
    #expect(removed.events == [.metrics(nil), .request(#"{"id":1}"#)])
    #expect(kept.events == [.metrics(nil), .request(#"{"id":1}"#), .request(#"{"id":2}"#)])
}

@Test
func bridgeScriptEmbedsRawJsonAndNull() {
    #expect(
        DashboardBridge.script(for: .metrics(nil))
            == "typeof window.onRuntimeMetrics === 'function' && window.onRuntimeMetrics(null); void 0;"
    )

    // The backend's JSON goes in verbatim, as an object literal, not as a string to parse.
    let raw = #"{"schema_version":1,"models":[{"id":"a\"b","weight_bytes":1}]}"#
    #expect(
        DashboardBridge.script(for: .metrics(raw))
            == "typeof window.onRuntimeMetrics === 'function' && window.onRuntimeMetrics(\(raw)); void 0;"
    )
    #expect(
        DashboardBridge.script(for: .history(#"{"t":5000}"#))
            == #"typeof window.onRuntimeHistoryPoint === 'function' && window.onRuntimeHistoryPoint({"t":5000}); void 0;"#
    )
    #expect(
        DashboardBridge.script(for: .request(#"{"id":7}"#))
            == #"typeof window.onRuntimeRequest === 'function' && window.onRuntimeRequest({"id":7}); void 0;"#
    )
    #expect(
        DashboardBridge.script(for: .backend(state: "running", connected: true))
            == #"typeof window.onRuntimeBackendState === 'function' && window.onRuntimeBackendState({"connected":true,"state":"running"}); void 0;"#
    )
}

@Test @MainActor
func bridgeDeliversRelayEventsOnlyWhileWindowOpenAndPageLoaded() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 400, height: 300))
    let bridge = DashboardBridge(
        webView: webView,
        configStore: AppConfigStore(url: root.appendingPathComponent("config.json")),
        backend: TestRuntimeBackend(state: .running),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("params.json")),
        notificationCenter: NotificationCenter()
    )
    webView.navigationDelegate = bridge
    let relay = RuntimeEventRelay()
    relay.send(.backend(state: "running", connected: true))
    relay.send(.metrics(#"{"n":1}"#))

    bridge.attachRuntimeRelay(relay)
    bridge.setDashboardWindowVisible(true)
    // Nothing to deliver to before the page has loaded.
    #expect(relay.listenerCount == 0)

    // The page defines no onRuntimeRequest; that event must be skipped without an error.
    webView.loadHTMLString(
        """
        <script>
        window.received = [];
        window.onRuntimeMetrics = function (m) { received.push(['metrics', m]); };
        window.onRuntimeHistoryPoint = function (h) { received.push(['history', h]); };
        window.onRuntimeBackendState = function (b) { received.push(['backend', b]); };
        window.statusWindowReopened = function () { received.push(['reopened']); };
        </script>
        """,
        baseURL: nil
    )
    try await waitForRelay { relay.listenerCount == 1 }

    func received() async throws -> String? {
        try await webView.evaluateJavaScript("JSON.stringify(window.received)") as? String
    }
    #expect(
        try await received()
            == #"[["backend",{"connected":true,"state":"running"}],["metrics",{"n":1}]]"#
    )

    relay.send(.history(#"{"t":5000}"#))
    relay.send(.request(#"{"id":1}"#))
    relay.send(.metrics(nil))
    #expect(
        try await received()
            == #"[["backend",{"connected":true,"state":"running"}],["metrics",{"n":1}],["history",{"t":5000}],["metrics",null]]"#
    )

    // Closing the window stops delivery; reopening replays the current state and asks the page,
    // once, to fetch history and recent requests again. The first load above did not ask.
    bridge.setDashboardWindowVisible(false)
    #expect(relay.listenerCount == 0)
    relay.send(.metrics(#"{"n":2}"#))
    relay.send(.history(#"{"t":10000}"#))
    #expect(try await received()?.contains(#""n":2"#) == false)

    bridge.setDashboardWindowVisible(true)
    bridge.setDashboardWindowVisible(true)
    #expect(relay.listenerCount == 1)
    #expect(
        try await received()?.hasSuffix(
            #"["metrics",null],["backend",{"connected":true,"state":"running"}],["metrics",{"n":2}],["reopened"]]"#
        ) == true
    )
    #expect(try await received()?.components(separatedBy: #"["reopened"]"#).count == 2)
}

@Test @MainActor
func bridgePausesRelayDeliveryWhilePageReloads() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 400, height: 300))
    let bridge = DashboardBridge(
        webView: webView,
        configStore: AppConfigStore(url: root.appendingPathComponent("config.json")),
        backend: TestRuntimeBackend(state: .running),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("params.json")),
        notificationCenter: NotificationCenter()
    )
    webView.navigationDelegate = bridge
    let relay = RuntimeEventRelay()
    relay.send(.backend(state: "running", connected: true))
    relay.send(.metrics(#"{"n":1}"#))
    bridge.attachRuntimeRelay(relay)
    bridge.setDashboardWindowVisible(true)

    func page(_ generation: Int) -> String {
        """
        <script>
        window.generation = \(generation);
        window.received = [];
        window.onRuntimeMetrics = function (m) { received.push(['metrics', m]); };
        window.onRuntimeHistoryPoint = function (h) { received.push(['history', h]); };
        window.onRuntimeBackendState = function (b) { received.push(['backend', b]); };
        window.statusWindowReopened = function () { received.push(['reopened']); };
        </script>
        """
    }
    func state() async throws -> String? {
        try await webView.evaluateJavaScript("window.generation + ':' + JSON.stringify(window.received)") as? String
    }
    func waitForPage(_ expected: String) async throws {
        let deadline = ContinuousClock().now + .seconds(5)
        while try await state() != expected {
            if ContinuousClock().now >= deadline {
                Issue.record("page never reached \(expected)")
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
    let replay = #"[["backend",{"connected":true,"state":"running"}],["metrics",{"n":1}]]"#

    webView.loadHTMLString(page(1), baseURL: nil)
    try await waitForPage("1:" + replay)
    #expect(relay.listenerCount == 1)

    // Between the start of a navigation and its finish nothing is delivered.
    bridge.webView(webView, didStartProvisionalNavigation: nil)
    #expect(relay.listenerCount == 0)
    relay.send(.history(#"{"t":5000}"#))
    #expect(try await state() == "1:" + replay)

    // A finished load listens again once and replays the current state once; it is not a
    // window reopen, so the page is not asked to refetch.
    bridge.webView(webView, didFinish: nil)
    bridge.webView(webView, didFinish: nil)
    #expect(relay.listenerCount == 1)
    relay.send(.history(#"{"t":10000}"#))
    let afterRefinish = try await state()
    #expect(
        afterRefinish
            == #"1:[["backend",{"connected":true,"state":"running"}],["metrics",{"n":1}],"#
            + #"["backend",{"connected":true,"state":"running"}],["metrics",{"n":1}],["history",{"t":10000}]]"#
    )

    // A real reload: the new page gets the replay, then live events, from one listener.
    webView.loadHTMLString(page(2), baseURL: nil)
    try await waitForPage("2:" + replay)
    #expect(relay.listenerCount == 1)
    relay.send(.history(#"{"t":15000}"#))
    let afterReload = try await state()
    #expect(
        afterReload
            == #"2:[["backend",{"connected":true,"state":"running"}],["metrics",{"n":1}],["history",{"t":15000}]]"#
    )
}
