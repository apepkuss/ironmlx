import Foundation

@testable import IronMLXAppCore

@MainActor
final class TestRuntimeBackend: MenuBarBackendProcessManaging {
    var isRunning: Bool
    var state: BackendProcessState
    var lastError: String?
    var lastEvent: BackendRuntimeEvent?
    var lastIncident: BackendIncidentRecord?
    var restartResult: BackendRestartResult
    private(set) var calls: [String] = []

    init(
        state: BackendProcessState = .stopped,
        isRunning: Bool = false,
        restartResult: BackendRestartResult = BackendRestartResult(
            success: true,
            status: "restarted",
            port: 9068
        )
    ) {
        self.state = state
        self.isRunning = isRunning
        self.restartResult = restartResult
    }

    func ensureRunning() async throws {
        calls.append("ensureRunning")
        isRunning = true
        state = .running
    }

    func restart(intent: BackendStopIntent) async -> BackendRestartResult {
        calls.append("restart:\(intent.rawValue)")
        isRunning = true
        state = restartResult.success ? .running : .failed
        lastError = restartResult.error
        return restartResult
    }

    func stop(intent: BackendStopIntent) async {
        calls.append("stop:\(intent.rawValue)")
        isRunning = false
        state = .stopped
    }

    func stopForAppQuit() async {
        await stop(intent: .appQuit)
    }

    func confirmLoadedModels(
        _ models: [BackendLoadedModelInfo],
        parameterConfirmedModelIDs: Set<String>
    ) {
        calls.append("confirm:\(models.count):\(parameterConfirmedModelIDs.count)")
    }
}
