import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func menuBarRefreshesImmediatelyForBackendRuntimeNotification() throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .running, isRunning: true)
    let notificationCenter = NotificationCenter()
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter
    )

    backend.state = .recovering
    notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: backend)

    #expect(controller.rebuildMenu().state == .recovering)
}

@Test @MainActor
func menuBarRestartUsesExplicitPlannedRestartIntent() async throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(
        state: .running,
        isRunning: true,
        restartResult: BackendRestartResult(
            success: true,
            status: "restarted",
            port: 9068,
            loadedModels: []
        )
    )
    let notificationCenter = NotificationCenter()
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter
    )

    controller.restartServer(NSMenuItem())

    try await waitForMenuCondition {
        backend.calls.contains("restart:plannedRestart")
    }
    #expect(backend.state == .running)
}

@Test @MainActor
func menuBarStopUsesUserStopAndNotifiesLoadedModelObservers() async throws {
    let root = try menuTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let backend = TestRuntimeBackend(state: .running, isRunning: true)
    let notificationCenter = NotificationCenter()
    let dashboard = DashboardWindowController(configStore: configStore, backend: backend)
    let controller = MenuBarController(
        configStore: configStore,
        backend: backend,
        dashboard: dashboard,
        notificationCenter: notificationCenter
    )
    let probe = MenuLoadedModelsNotificationProbe()
    notificationCenter.addObserver(
        probe,
        selector: #selector(MenuLoadedModelsNotificationProbe.loadedModelsDidChange(_:)),
        name: .ironMLXLoadedModelsDidChange,
        object: controller
    )
    defer {
        notificationCenter.removeObserver(probe)
    }

    controller.stopServer(NSMenuItem())

    try await waitForMenuCondition {
        backend.calls.contains("stop:userStop") && probe.notified
    }
    #expect(backend.state == .stopped)
}

@MainActor
private final class MenuLoadedModelsNotificationProbe: NSObject {
    private(set) var notified = false

    @objc func loadedModelsDidChange(_ notification: Notification) {
        notified = true
    }
}

private func menuTemporaryDirectory() throws -> URL {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-menu-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

@MainActor
private func waitForMenuCondition(
    timeoutSeconds: TimeInterval = 2.0,
    condition: @escaping () -> Bool
) async throws {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    while Date() < deadline {
        if condition() {
            return
        }
        try await Task.sleep(for: .milliseconds(20))
    }
    Issue.record("Timed out waiting for menu operation")
}
