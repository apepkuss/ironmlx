import Foundation

public struct BackendRecoveryPolicy: Equatable, Sendable {
    public var maximumAutomaticRecoveryAttempts: Int
    public var automaticRecoveryDelay: TimeInterval
    public var stableWindow: TimeInterval
    public var readinessTimeout: TimeInterval

    public init(
        maximumAutomaticRecoveryAttempts: Int = 1,
        automaticRecoveryDelay: TimeInterval = 1.0,
        stableWindow: TimeInterval = 60.0,
        readinessTimeout: TimeInterval = 5.0
    ) {
        self.maximumAutomaticRecoveryAttempts = maximumAutomaticRecoveryAttempts
        self.automaticRecoveryDelay = automaticRecoveryDelay
        self.stableWindow = stableWindow
        self.readinessTimeout = readinessTimeout
    }
}

public enum BackendRuntimeEventPhase: String, Codable, Equatable, Sendable {
    case exited
    case recovering
    case recovered
    case failed
    case breaker
}

public struct BackendRuntimeEvent: Codable, Equatable, Sendable {
    public var phase: BackendRuntimeEventPhase
    public var runtimeState: BackendProcessState
    public var incidentID: UUID?
    public var launchID: UUID?
    public var pid: Int32?
    public var terminationStatus: Int32?
    public var terminationReason: String?
    public var recoveryAttempt: Int
    public var recoveredModels: [String]
    public var failedModels: [String]
    public var failures: [BackendModelRecoveryFailure]
    public var detail: String
    public var logTail: String?
    public var canRetry: Bool
    public var processHealthy: Bool

    public init(
        phase: BackendRuntimeEventPhase,
        runtimeState: BackendProcessState,
        incidentID: UUID? = nil,
        launchID: UUID? = nil,
        pid: Int32? = nil,
        terminationStatus: Int32? = nil,
        terminationReason: String? = nil,
        recoveryAttempt: Int = 0,
        recoveredModels: [String] = [],
        failedModels: [String] = [],
        failures: [BackendModelRecoveryFailure] = [],
        detail: String,
        logTail: String? = nil,
        canRetry: Bool,
        processHealthy: Bool
    ) {
        self.phase = phase
        self.runtimeState = runtimeState
        self.incidentID = incidentID
        self.launchID = launchID
        self.pid = pid
        self.terminationStatus = terminationStatus
        self.terminationReason = terminationReason
        self.recoveryAttempt = recoveryAttempt
        self.recoveredModels = recoveredModels
        self.failedModels = failedModels
        self.failures = failures
        self.detail = detail
        self.logTail = logTail
        self.canRetry = canRetry
        self.processHealthy = processHealthy
    }

    enum CodingKeys: String, CodingKey {
        case phase
        case runtimeState = "runtime_state"
        case incidentID = "incident_id"
        case launchID = "launch_id"
        case pid
        case terminationStatus = "termination_status"
        case terminationReason = "termination_reason"
        case recoveryAttempt = "recovery_attempt"
        case recoveredModels = "recovered_models"
        case failedModels = "failed_models"
        case failures
        case detail
        case logTail = "log_tail"
        case canRetry = "can_retry"
        case processHealthy = "process_healthy"
    }
}

@MainActor
public protocol BackendRuntimeManaging: AnyObject {
    var isRunning: Bool { get }
    var state: BackendProcessState { get }
    var lastError: String? { get }
    var lastEvent: BackendRuntimeEvent? { get }
    var lastIncident: BackendIncidentRecord? { get }

    func ensureRunning() async throws
    func restart(intent: BackendStopIntent) async -> BackendRestartResult
    func stop(intent: BackendStopIntent) async
    func stopForAppQuit() async
    func confirmLoadedModels(
        _ models: [BackendLoadedModelInfo],
        parameterConfirmedModelIDs: Set<String>
    )
}

@MainActor
public final class BackendRuntimeSupervisor: BackendRuntimeManaging {
    public typealias Sleep = @Sendable (TimeInterval) async -> Void
    public typealias ReadinessWaiter =
        @Sendable (_ config: AppConfig, _ timeout: TimeInterval) async throws -> Void

    public private(set) var lastEvent: BackendRuntimeEvent?
    public private(set) var lastIncident: BackendIncidentRecord?

    public var isRunning: Bool {
        processManager.isRunning
    }

    public var state: BackendProcessState {
        processManager.state
    }

    public var lastError: String? {
        processManager.lastError
    }

    private let processManager: BackendProcessManager
    private let configStore: AppConfigStore
    private let scanner: LocalModelScanner
    private let parameterStore: ModelParameterStore
    private let restartCoordinator: any BackendModelRestoring
    private let incidentStore: BackendIncidentStore
    private let notificationCenter: NotificationCenter
    private let policy: BackendRecoveryPolicy
    private let sleep: Sleep
    private let readinessWaiter: ReadinessWaiter
    private var activeIncident: ActiveBackendIncident?
    private var breakerActive = false
    private var recoveryTask: Task<Void, Never>?
    private var stableWindowTask: Task<Void, Never>?
    private var launchCompletionWaiters: [UUID: [CheckedContinuation<Void, Never>]] = [:]
    private var confirmedSnapshot: BackendRecoverySnapshot

    public init(
        processManager: BackendProcessManager,
        configStore: AppConfigStore = .shared,
        scanner: LocalModelScanner = LocalModelScanner(),
        parameterStore: ModelParameterStore = .shared,
        restartCoordinator: (any BackendModelRestoring)? = nil,
        incidentStore: BackendIncidentStore = BackendIncidentStore(),
        notificationCenter: NotificationCenter = .default,
        policy: BackendRecoveryPolicy = BackendRecoveryPolicy(),
        sleep: @escaping Sleep = { interval in
            guard interval > 0 else {
                return
            }
            try? await Task.sleep(for: .seconds(interval))
        },
        readinessWaiter: @escaping ReadinessWaiter = { config, timeout in
            try await BackendAPIClient(host: config.host, port: config.port)
                .waitUntilReady(timeout: timeout)
        }
    ) {
        self.processManager = processManager
        self.configStore = configStore
        self.scanner = scanner
        self.parameterStore = parameterStore
        self.restartCoordinator = restartCoordinator ?? BackendRestartCoordinator(
            scanner: scanner,
            parameterStore: parameterStore
        )
        self.incidentStore = incidentStore
        self.notificationCenter = notificationCenter
        self.policy = policy
        self.sleep = sleep
        self.readinessWaiter = readinessWaiter
        self.confirmedSnapshot = BackendRecoverySnapshot.capture(
            config: configStore.load(),
            scanner: scanner,
            parameterStore: parameterStore
        )
        self.lastIncident = incidentStore.records().last
        processManager.terminationObserver = { [weak self] termination in
            self?.processDidTerminate(termination)
        }
    }

    public func ensureRunning() async throws {
        if processManager.isRunning {
            if let launchID = processManager.currentLaunchID,
               state == .starting || state == .recovering {
                await waitForLaunchCompletion(launchID)
                guard processManager.isRunning else {
                    throw BackendRuntimeSupervisorError.launchFailed(
                        processManager.lastError ?? "Backend launch was interrupted."
                    )
                }
            }
            return
        }
        clearBreakerForExplicitStart()
        let snapshot = currentSnapshot()
        _ = try await launchAndRestore(
            snapshot: snapshot,
            launchState: .starting,
            incidentID: nil,
            isAutomaticRecovery: false
        )
    }

    public func restart(intent: BackendStopIntent = .plannedRestart) async -> BackendRestartResult {
        clearBreakerForExplicitStart()
        recoveryTask?.cancel()
        stableWindowTask?.cancel()
        let snapshot = currentSnapshot()
        if processManager.isRunning {
            await processManager.stop(intent: intent)
            processManager.markStoppedAfterPlannedTermination()
        }
        do {
            return try await launchAndRestore(
                snapshot: snapshot,
                launchState: .starting,
                incidentID: nil,
                isAutomaticRecovery: false
            )
        } catch {
            return BackendRestartResult(
                success: false,
                status: "restart_failed",
                port: snapshot.config.port,
                model: snapshot.config.defaultModelReference,
                failedModels: snapshot.models.map(\.id),
                error: error.localizedDescription
            )
        }
    }

    public func stop(intent: BackendStopIntent = .userStop) async {
        recoveryTask?.cancel()
        stableWindowTask?.cancel()
        await processManager.stop(intent: intent)
        processManager.markStoppedAfterPlannedTermination()
    }

    public func stopForAppQuit() async {
        recoveryTask?.cancel()
        stableWindowTask?.cancel()
        await processManager.stopForAppQuit()
        processManager.markStoppedAfterPlannedTermination()
    }

    public func retryAfterFailure() async -> BackendRestartResult {
        await restart(intent: .plannedRestart)
    }

    public func runtimeEventJSON() -> String? {
        guard let lastEvent,
              let data = try? JSONEncoder().encode(lastEvent)
        else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    public func confirmLoadedModels(
        _ models: [BackendLoadedModelInfo],
        parameterConfirmedModelIDs: Set<String> = []
    ) {
        let config = configStore.load()
        let captured = BackendRecoverySnapshot.capture(
            config: config,
            scanner: scanner,
            parameterStore: parameterStore
        )
        let capturedByID = Dictionary(uniqueKeysWithValues: captured.models.map { ($0.id, $0) })
        let existingByID = Dictionary(
            uniqueKeysWithValues: confirmedSnapshot.models.map { ($0.id, $0) }
        )
        let confirmedModels = models.map { model -> BackendRecoveryModel in
            let shouldRefreshParameters = parameterConfirmedModelIDs.contains(model.id)
                || parameterConfirmedModelIDs.contains(model.model)
                || parameterConfirmedModelIDs.contains(model.path)
            var confirmed = if shouldRefreshParameters {
                capturedByID[model.id] ?? existingByID[model.id]
            } else {
                existingByID[model.id] ?? capturedByID[model.id]
            }
            if confirmed == nil {
                confirmed = BackendRecoveryModel(
                    id: model.id,
                    modelDir: model.path,
                    isDefault: model.isDefault,
                    pinned: model.pinned,
                    maxCacheCap: nil,
                    mtpModelDir: model.mtpModelDir,
                    mtpDraftTokens: model.mtpDraftTokens,
                    promptLookup: model.promptLookup,
                    samplingDefaults: .empty
                )
            }
            confirmed?.id = model.id
            confirmed?.modelDir = model.path
            confirmed?.isDefault = model.isDefault
            confirmed?.pinned = model.pinned
            confirmed?.mtpModelDir = model.mtpModelDir
            confirmed?.mtpDraftTokens = model.mtpDraftTokens
            confirmed?.promptLookup = model.promptLookup
            return confirmed!
        }
        confirmedSnapshot = BackendRecoverySnapshot(config: config, models: confirmedModels)
    }

    private func launchAndRestore(
        snapshot: BackendRecoverySnapshot,
        launchState: BackendProcessState,
        incidentID: UUID?,
        isAutomaticRecovery: Bool
    ) async throws -> BackendRestartResult {
        let launchID = try processManager.startProcess(initialState: launchState)
        defer {
            resumeLaunchCompletionWaiters(for: launchID)
        }
        do {
            try await readinessWaiter(snapshot.config, policy.readinessTimeout)
        } catch {
            if processManager.currentLaunchID == launchID, processManager.isRunning {
                await processManager.stop(
                    intent: isAutomaticRecovery
                        ? .recoveryFailureCleanup
                        : .startupFailureCleanup
                )
            }
            guard !Task.isCancelled else {
                throw CancellationError()
            }
            processManager.transition(to: .failed, error: error.localizedDescription)
            publish(
                BackendRuntimeEvent(
                    phase: .failed,
                    runtimeState: .failed,
                    incidentID: incidentID,
                    launchID: launchID,
                    detail: "Backend health check failed: \(error.localizedDescription)",
                    logTail: activeIncident?.record.logTail,
                    canRetry: true,
                    processHealthy: false
                )
            )
            throw BackendRuntimeSupervisorError.readinessFailed(error.localizedDescription)
        }

        guard processManager.currentLaunchID == launchID, processManager.isRunning else {
            throw CancellationError()
        }

        let result = await restartCoordinator.restore(snapshot)
        guard processManager.currentLaunchID == launchID, processManager.isRunning else {
            throw CancellationError()
        }
        applyRecoveryResult(result, incidentID: incidentID, launchID: launchID)
        return result
    }

    private func applyRecoveryResult(
        _ result: BackendRestartResult,
        incidentID: UUID?,
        launchID: UUID
    ) {
        if result.success {
            processManager.transition(to: .running)
            if incidentID != nil {
                publish(
                    BackendRuntimeEvent(
                        phase: .recovered,
                        runtimeState: .running,
                        incidentID: incidentID,
                        launchID: launchID,
                        recoveryAttempt: activeIncident?.record.recoveryAttempt ?? 0,
                        recoveredModels: result.loadedModels,
                        detail: "Backend and models recovered successfully.",
                        logTail: activeIncident?.record.logTail,
                        canRetry: false,
                        processHealthy: true
                    )
                )
            }
            return
        }

        let partial = !result.loadedModels.isEmpty
        let nextState: BackendProcessState = partial ? .degraded : .failed
        processManager.transition(to: nextState, error: result.error)
        publish(
            BackendRuntimeEvent(
                phase: .failed,
                runtimeState: nextState,
                incidentID: incidentID,
                launchID: launchID,
                recoveryAttempt: activeIncident?.record.recoveryAttempt ?? 0,
                recoveredModels: result.loadedModels,
                failedModels: result.failedModels,
                failures: result.failures,
                detail: result.error ?? result.status,
                logTail: activeIncident?.record.logTail,
                canRetry: result.failures.isEmpty || result.failures.contains(where: \.retryable),
                processHealthy: true
            )
        )
    }

    private func processDidTerminate(_ termination: BackendProcessTermination) {
        if termination.stopIntent.isPlanned {
            processManager.transition(to: .stopped)
            return
        }

        if var activeIncident,
           activeIncident.record.recoveryAttempt >= policy.maximumAutomaticRecoveryAttempts {
            recoveryTask?.cancel()
            stableWindowTask?.cancel()
            breakerActive = true
            activeIncident.record.recoveryResult = "breaker"
            activeIncident.record.error = "Backend exited again before the stable window completed."
            self.activeIncident = activeIncident
            lastIncident = activeIncident.record
            processManager.transition(
                to: .failed,
                error: "Crash-loop breaker stopped automatic recovery."
            )
            publish(
                BackendRuntimeEvent(
                    phase: .breaker,
                    runtimeState: .failed,
                    incidentID: activeIncident.record.id,
                    launchID: termination.launchID,
                    pid: termination.pid,
                    terminationStatus: termination.terminationStatus,
                    terminationReason: termination.terminationReason,
                    recoveryAttempt: activeIncident.record.recoveryAttempt,
                    detail: "Backend crashed again inside the stable window. Automatic recovery is disabled.",
                    logTail: BackendIncidentRecord.sanitizedLogTail(termination.logTail),
                    canRetry: true,
                    processHealthy: false
                )
            )
            persistIncident(activeIncident.record)
            return
        }

        let snapshot = currentSnapshot()
        var record = BackendIncidentRecord(termination: termination, snapshot: snapshot)
        activeIncident = ActiveBackendIncident(record: record, snapshot: snapshot)
        lastIncident = record
        processManager.transition(to: .recovering)
        publish(
            BackendRuntimeEvent(
                phase: .exited,
                runtimeState: .recovering,
                incidentID: record.id,
                launchID: termination.launchID,
                pid: termination.pid,
                terminationStatus: termination.terminationStatus,
                terminationReason: termination.terminationReason,
                detail: Self.terminationDetail(termination),
                logTail: record.logTail,
                canRetry: false,
                processHealthy: false
            )
        )
        persistIncident(record)

        let incidentID = record.id
        recoveryTask = Task { [weak self] in
            guard let self else {
                return
            }
            await self.sleep(self.policy.automaticRecoveryDelay)
            guard !Task.isCancelled,
                  var active = self.activeIncident,
                  active.record.id == incidentID,
                  !self.breakerActive
            else {
                return
            }
            active.record.recoveryAttempt += 1
            record = active.record
            self.activeIncident = active
            self.processManager.transition(to: .recovering)
            self.publish(
                BackendRuntimeEvent(
                    phase: .recovering,
                    runtimeState: .recovering,
                    incidentID: incidentID,
                    recoveryAttempt: record.recoveryAttempt,
                    detail: "Starting backend and restoring the confirmed model snapshot.",
                    logTail: record.logTail,
                    canRetry: false,
                    processHealthy: false
                )
            )
            self.persistIncident(record)

            do {
                let result = try await self.launchAndRestore(
                    snapshot: active.snapshot,
                    launchState: .recovering,
                    incidentID: incidentID,
                    isAutomaticRecovery: true
                )
                guard !Task.isCancelled,
                      var completed = self.activeIncident,
                      completed.record.id == incidentID,
                      !self.breakerActive
                else {
                    return
                }
                completed.record.recoveredModels = result.loadedModels
                completed.record.failedModels = result.failedModels
                completed.record.failures = result.failures
                completed.record.recoveryResult = result.success
                    ? "recovered"
                    : result.loadedModels.isEmpty ? "failed" : "degraded"
                completed.record.error = result.error
                self.activeIncident = completed
                self.lastIncident = completed.record
                self.persistIncident(completed.record)
                if result.success || !result.loadedModels.isEmpty {
                    self.startStableWindow(for: incidentID)
                }
            } catch {
                guard !Task.isCancelled,
                      var failed = self.activeIncident,
                      failed.record.id == incidentID,
                      !self.breakerActive
                else {
                    return
                }
                failed.record.recoveryResult = "failed"
                failed.record.error = error.localizedDescription
                self.activeIncident = failed
                self.lastIncident = failed.record
                self.persistIncident(failed.record)
            }
        }
    }

    private func startStableWindow(for incidentID: UUID) {
        stableWindowTask?.cancel()
        stableWindowTask = Task { [weak self] in
            guard let self else {
                return
            }
            await self.sleep(self.policy.stableWindow)
            guard !Task.isCancelled,
                  self.activeIncident?.record.id == incidentID,
                  !self.breakerActive
            else {
                return
            }
            self.activeIncident = nil
        }
    }

    private func clearBreakerForExplicitStart() {
        breakerActive = false
        activeIncident = nil
        recoveryTask?.cancel()
        stableWindowTask?.cancel()
    }

    private func currentSnapshot() -> BackendRecoverySnapshot {
        var snapshot = confirmedSnapshot
        snapshot.capturedAt = Date()
        var launchConfig = configStore.load()
        launchConfig.defaultModel = snapshot.config.defaultModel
        launchConfig.loadedModels = snapshot.config.loadedModels
        launchConfig.pinnedModels = snapshot.config.pinnedModels
        snapshot.config = launchConfig
        return snapshot
    }

    private func persistIncident(_ record: BackendIncidentRecord) {
        do {
            try incidentStore.upsert(record)
        } catch {
            IronMLXAppLogger.error("Failed to persist backend incident: \(error)")
        }
    }

    private func publish(_ event: BackendRuntimeEvent) {
        lastEvent = event
        notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: self)
    }

    private func waitForLaunchCompletion(_ launchID: UUID) async {
        guard processManager.currentLaunchID == launchID,
              state == .starting || state == .recovering
        else {
            return
        }
        await withCheckedContinuation { continuation in
            launchCompletionWaiters[launchID, default: []].append(continuation)
        }
    }

    private func resumeLaunchCompletionWaiters(for launchID: UUID) {
        let waiters = launchCompletionWaiters.removeValue(forKey: launchID) ?? []
        waiters.forEach { $0.resume() }
    }

    nonisolated private static func terminationDetail(
        _ termination: BackendProcessTermination
    ) -> String {
        if termination.terminationReason == "uncaught_signal" {
            return "signal \(termination.terminationStatus)"
        }
        return "exit code \(termination.terminationStatus)"
    }
}

private struct ActiveBackendIncident {
    var record: BackendIncidentRecord
    var snapshot: BackendRecoverySnapshot
}

public enum BackendRuntimeSupervisorError: LocalizedError, Equatable {
    case readinessFailed(String)
    case launchFailed(String)

    public var errorDescription: String? {
        switch self {
        case .readinessFailed(let detail):
            return "Backend did not become healthy: \(detail)"
        case .launchFailed(let detail):
            return "Backend launch failed: \(detail)"
        }
    }
}
