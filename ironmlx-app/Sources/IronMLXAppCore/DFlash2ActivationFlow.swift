import Foundation

enum DFlash2AppActivationError: LocalizedError {
    case restartFailed(message: String, code: String?)

    var errorDescription: String? {
        switch self {
        case .restartFailed(let message, _):
            return message
        }
    }

    var failureCode: String? {
        switch self {
        case .restartFailed(_, let code):
            return code
        }
    }
}

/// A Dashboard switch to DFlash2: the App config already names the target as
/// the only loaded model; this verifies the files, stops the backend and
/// starts it in DFlash2 mode, tracked as one load. A failure or a cancel puts
/// the previous config and parameters back; once the backend was stopped it
/// restarts with them, so the models loaded before the switch load again.
@MainActor
struct DFlash2ActivationFlow {
    let backend: any BackendRuntimeManaging
    let configStore: AppConfigStore
    let parameterStore: ModelParameterStore
    let loadActivity: ModelLoadActivityCenter
    /// Resolves (and, when configured, fully checks) the target and draft.
    let verify: @Sendable () async throws -> Void

    func run(
        model: String,
        kind: ModelLoadKind,
        previousConfig: AppConfig,
        previousParameters: ModelParameters?
    ) async throws -> BackendRestartResult {
        let backend = backend
        let configStore = configStore
        let parameterStore = parameterStore
        let loadActivity = loadActivity
        let verify = verify
        return try await loadActivity.track(
            model: model,
            variant: .dflash2,
            origin: .dashboard,
            kind: kind
        ) { ticket in
            var backendStopped = false
            do {
                loadActivity.enterAppStage(ticket, .verifyingFiles)
                try await verify()
                try Task.checkCancellation()
                loadActivity.enterAppStage(ticket, .stoppingBackend)
                backend.refreshConfirmedSnapshot()
                backendStopped = true
                await backend.stop(intent: .plannedRestart)
                try Task.checkCancellation()
                loadActivity.enterAppStage(ticket, .restartingBackend)
                let result = await backend.restart(intent: .plannedRestart)
                try Task.checkCancellation()
                guard result.success else {
                    throw DFlash2AppActivationError.restartFailed(
                        message: result.error ?? "DFlash2 backend restart failed.",
                        code: result.errorCode
                    )
                }
                return result
            } catch {
                _ = configStore.save(previousConfig)
                try? parameterStore.replaceParameters(for: model, with: previousParameters)
                backend.refreshConfirmedSnapshot()
                let cancelled = Task.isCancelled || error is CancellationError
                if backendStopped {
                    // A separate task: this one is cancelled when the user cancels.
                    _ = await Task { @MainActor in
                        await backend.restart(intent: .plannedRestart)
                    }.value
                }
                if cancelled {
                    throw ModelLoadCancelledError(code: ModelLoadCancelledError.dflash2RestoredCode)
                }
                throw error
            }
        }
    }
}
