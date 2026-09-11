import Foundation
import Testing

@testable import IronMLXAppCore

@Test func appLogLevelFiltersActualFileOutput() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = IronMLXLogStore(rootURL: root)
    let writer = AppDiagnosticLogWriter(store: store)
    for threshold in AppLogLevel.allCases {
        writer.setMinimumLevel(threshold)
        for level in AppLogLevel.allCases {
            let marker = "\(threshold.rawValue)-\(level.rawValue)"
            writer.write(line: marker, level: level)
        }
    }
    let text = try String(contentsOf: store.url(for: .app), encoding: .utf8)
    for threshold in AppLogLevel.allCases {
        for level in AppLogLevel.allCases {
            #expect(text.components(separatedBy: .newlines).contains("\(threshold.rawValue)-\(level.rawValue)") == threshold.includes(level))
        }
    }
    #expect(AppLogLevel.saved("TRACE") == .all)
    #expect(AppLogLevel.saved("WARN") == .warning)
    #expect(AppLogLevel.saved(nil) == .info)
}

private actor LogClient: BackendLogLevelControlling {
    var snapshot = BackendLogLevelSnapshot(level: .info, revision: 0, processID: 42)
    var sets = 0
    let loseFirstReply: Bool
    let rejectRollback: Bool
    let onApply: @Sendable () async -> Void
    init(loseFirstReply: Bool = false, rejectRollback: Bool = false, onApply: @escaping @Sendable () async -> Void = {}) {
        self.loseFirstReply = loseFirstReply
        self.rejectRollback = rejectRollback
        self.onApply = onApply
    }
    func logLevel() async throws -> BackendLogLevelSnapshot { snapshot }
    func setLogLevel(_ level: AppLogLevel, expected: BackendLogLevelSnapshot) async throws -> BackendLogLevelSnapshot {
        guard expected == snapshot else { throw RuntimeLogLevelError.backendChanged }
        sets += 1
        if sets > 1 && rejectRollback { throw RuntimeLogLevelError.backendUnavailable }
        snapshot.level = level
        snapshot.revision += 1
        if sets == 1 {
            await onApply()
            if loseFirstReply { throw URLError(.timedOut) }
        }
        return snapshot
    }
}

@MainActor
@Test func runtimeLogLevelAppliesWithoutRestartAndPreservesOtherSettings() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = AppConfigStore(url: root.appendingPathComponent("config.json"))
    #expect(store.save(AppConfig(port: 9068, logLevel: "INFO")))
    let backend = TestRuntimeBackend(state: .running, isRunning: true)
    let client = LogClient(onApply: { _ = store.update { $0.theme = "dark" } })
    var applied: AppLogLevel = .info
    let controller = RuntimeLogLevelController(configStore: store, backend: backend, clientFactory: { _ in client }, applyAppLevel: { applied = $0 })
    try await controller.apply(.debug)
    #expect(applied == .debug)
    #expect(store.load().logLevel == "DEBUG")
    #expect(store.load().theme == "dark")
    #expect(store.load().port == 9068)
    #expect(try await client.logLevel().level == .debug)
    #expect(backend.calls.isEmpty)
    #expect(AppConfigStore(url: root.appendingPathComponent("config.json")).load().logLevel == "DEBUG")
}

@MainActor
@Test func runtimeLogLevelReconcilesLostReplyAndReportsRollbackFailure() async throws {
    for rollbackFails in [false, true] {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AppConfigStore(url: root.appendingPathComponent("config.json"))
        #expect(store.save(AppConfig(logLevel: "INFO")))
        let client = LogClient(loseFirstReply: true, rejectRollback: rollbackFails)
        var applied = false
        let controller = RuntimeLogLevelController(configStore: store, backend: TestRuntimeBackend(state: .running, isRunning: true), clientFactory: { _ in client }, applyAppLevel: { _ in applied = true })
        do {
            try await controller.apply(.all)
            Issue.record("Expected lost reply failure")
        } catch RuntimeLogLevelError.rollbackFailed {
            #expect(rollbackFails)
        } catch {
            #expect(!rollbackFails)
        }
        #expect(!applied)
        #expect(store.load().logLevel == "INFO")
        #expect(try await client.logLevel().level == (rollbackFails ? .all : .info))
    }
}

@MainActor
@Test func runtimeLogLevelPersistenceFailureRollsBackendBack() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = AppConfigStore(url: root.appendingPathComponent("config.json"))
    #expect(store.save(AppConfig(logLevel: "INFO")))
    let client = LogClient()
    var applied = false
    let controller = RuntimeLogLevelController(configStore: store, backend: TestRuntimeBackend(state: .running, isRunning: true), clientFactory: { _ in client }, applyAppLevel: { _ in applied = true }, persistLevel: { _ in false })
    await #expect(throws: RuntimeLogLevelError.self) { try await controller.apply(.debug) }
    #expect(!applied)
    #expect(store.load().logLevel == "INFO")
    #expect(try await client.logLevel().level == .info)
}

@MainActor
@Test func stoppedBackendLogLevelSavesWithoutCallingBackend() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = AppConfigStore(url: root.appendingPathComponent("config.json"))
    #expect(store.save(AppConfig(logLevel: "INFO")))
    let client = LogClient()
    let controller = RuntimeLogLevelController(configStore: store, backend: TestRuntimeBackend(), clientFactory: { _ in client }, applyAppLevel: { _ in })
    try await controller.apply(.warning)
    #expect(store.load().logLevel == "WARNING")
    #expect(await client.sets == 0)
}
