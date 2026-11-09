import Darwin
import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func plannedStopIntentsEnterStoppingThenStoppedWithoutRecovery() async throws {
    let intents: [BackendStopIntent] = [
        .userStop,
        .appQuit,
        .benchmarkExclusive,
        .schedulerProfileGeneration,
    ]

    for intent in intents {
        let harness = try runtimeHarness()
        let recorder = BackendStateRecorder(manager: harness.processManager)
        harness.notificationCenter.addObserver(
            recorder,
            selector: #selector(BackendStateRecorder.runtimeDidChange(_:)),
            name: .ironMLXBackendRuntimeDidChange,
            object: harness.processManager
        )

        try await harness.supervisor.ensureRunning()
        if intent == .appQuit {
            await harness.supervisor.stopForAppQuit()
        } else {
            await harness.supervisor.stop(intent: intent)
        }

        #expect(recorder.states.contains(.stopping), "\(intent.rawValue): \(recorder.states)")
        #expect(recorder.states.last == .stopped)
        #expect(harness.processManager.lastTermination?.stopIntent == intent)
        #expect(harness.supervisor.lastIncident == nil)
        #expect(harness.incidentStore.records().isEmpty)
        #expect(harness.launchCount.value == 1)
        harness.notificationCenter.removeObserver(recorder)
    }
}

@Test @MainActor
func plannedRestartCreatesNewGenerationWithoutOldTerminationPollution() async throws {
    let harness = try runtimeHarness()

    try await harness.supervisor.ensureRunning()
    let firstLaunchID = try #require(harness.processManager.currentLaunchID)
    let firstGeneration = try #require(harness.processManager.currentGeneration)
    let firstPID = try #require(harness.processManager.currentProcessIdentifier)

    let result = await harness.supervisor.restart(intent: .plannedRestart)
    let secondLaunchID = try #require(harness.processManager.currentLaunchID)
    let secondGeneration = try #require(harness.processManager.currentGeneration)
    let secondPID = try #require(harness.processManager.currentProcessIdentifier)

    #expect(result.success)
    #expect(firstLaunchID != secondLaunchID)
    #expect(secondGeneration > firstGeneration)
    #expect(firstPID != secondPID)
    #expect(harness.processManager.lastTermination?.launchID == firstLaunchID)
    #expect(harness.processManager.lastTermination?.stopIntent == .plannedRestart)
    #expect(harness.supervisor.state == .running)
    #expect(harness.supervisor.lastIncident == nil)
    #expect(harness.launchCount.value == 2)

    await harness.supervisor.stop(intent: .userStop)
}

@Test @MainActor
func restartClassifiesMissingBundledRuntimeForDashboardLocalization() async throws {
    let root = try runtimeTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    let processManager = BackendProcessManager(
        configStore: configStore,
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            throw BundledRuntimeLayoutError.missingFile(
                root.appendingPathComponent("IronMLX.app/Contents/Helpers/ironmlx").path
            )
        }
    )
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: FixedModelRestorer(result: .successful(port: 9068)),
        incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
        readinessWaiter: { _, _ in }
    )

    let result = await supervisor.restart(intent: .plannedRestart)

    #expect(!result.success)
    #expect(result.errorCode == "bundled_runtime_invalid")
}

@Test @MainActor
func processReadyCallbackRunsBeforeModelRestoreCompletes() async throws {
    let restorer = GatedModelRestorer()
    let harness = try runtimeHarness(restorer: restorer)
    let readyCount = MainActorCounter()

    let startup = Task {
        try await harness.supervisor.ensureRunning {
            readyCount.value += 1
        }
    }
    let deadline = Date().addingTimeInterval(2)
    while await !restorer.didStart, Date() < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }

    #expect(await restorer.didStart)
    #expect(readyCount.value == 1)
    #expect(harness.supervisor.isRunning)
    #expect(harness.supervisor.state == .starting)

    await restorer.release()
    try await startup.value
    #expect(harness.supervisor.state == .running)
    await harness.supervisor.stop(intent: .userStop)
}

@Test @MainActor
func benchmarkAndSchedulerPlannedRestartsPreserveLifecycleSemantics() async throws {
    for intent in [BackendStopIntent.benchmarkExclusive, .schedulerProfileGeneration] {
        let harness = try runtimeHarness()
        try await harness.supervisor.ensureRunning()
        let firstLaunchID = try #require(harness.processManager.currentLaunchID)

        let result = await harness.supervisor.restart(intent: intent)

        #expect(result.success)
        #expect(harness.supervisor.state == .running)
        #expect(harness.processManager.currentLaunchID != firstLaunchID)
        #expect(harness.processManager.lastTermination?.stopIntent == intent)
        #expect(harness.supervisor.lastIncident == nil)
        #expect(harness.launchCount.value == 2)
        await harness.supervisor.stop(intent: .userStop)
    }
}

@Test @MainActor
func appQuitForceKillRemainsPlannedAndCannotRestartBackend() async throws {
    let root = try runtimeTemporaryDirectory()
    let marker = root.appendingPathComponent("ready")
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    let launchCount = MainActorCounter()
    let processManager = BackendProcessManager(
        configStore: configStore,
        terminator: BackendProcessTerminator(sleep: { _ in }),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            launchCount.value += 1
            return BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/usr/bin/python3"),
                arguments: [
                    "-c",
                    "import pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).write_text('ready');time.sleep(30)",
                    marker.path,
                ]
            )
        }
    )
    let incidentStore = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: FixedModelRestorer(result: .successful(port: 9068)),
        incidentStore: incidentStore,
        readinessWaiter: { _, _ in
            let deadline = Date().addingTimeInterval(2)
            while !FileManager.default.fileExists(atPath: marker.path), Date() < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            guard FileManager.default.fileExists(atPath: marker.path) else {
                throw RuntimeSupervisorTestError.readiness
            }
        }
    )

    try await supervisor.ensureRunning()
    await supervisor.stopForAppQuit()

    #expect(supervisor.state == .stopped)
    #expect(!supervisor.isRunning)
    #expect(processManager.lastTermination?.stopIntent == .appQuit)
    #expect(processManager.lastTermination?.terminationReason == "uncaught_signal")
    #expect(processManager.lastTermination?.terminationStatus == SIGKILL)
    #expect(supervisor.lastIncident == nil)
    #expect(incidentStore.records().isEmpty)
    #expect(launchCount.value == 1)
}

@Test @MainActor
func delayedOldTerminationCallbackCannotOverwriteNewLaunch() async throws {
    let root = try runtimeTemporaryDirectory()
    let counter = MainActorCounter()
    let manager = BackendProcessManager(
        configStore: AppConfigStore(url: root.appendingPathComponent("app_config.json")),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            counter.value += 1
            return BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["30"]
            )
        }
    )

    let firstLaunchID = try manager.startProcess()
    let firstPID = try #require(manager.currentProcessIdentifier)
    #expect(Darwin.kill(pid_t(firstPID), SIGKILL) == 0)

    let deadline = Date().addingTimeInterval(2)
    while manager.isRunning, Date() < deadline {
        usleep(1_000)
    }
    #expect(!manager.isRunning)

    let secondLaunchID = try manager.startProcess()
    let secondPID = try #require(manager.currentProcessIdentifier)
    #expect(firstLaunchID != secondLaunchID)
    #expect(firstPID != secondPID)

    try await Task.sleep(for: .milliseconds(50))

    #expect(manager.currentLaunchID == secondLaunchID)
    #expect(manager.currentProcessIdentifier == secondPID)
    #expect(manager.state == .starting)
    #expect(manager.lastError == nil)

    await manager.stop(intent: .userStop)
    manager.markStoppedAfterPlannedTermination()
}

@Test @MainActor
func processRunFailureEntersFailedWithoutClaimingRunning() async throws {
    let root = try runtimeTemporaryDirectory()
    let manager = BackendProcessManager(
        configStore: AppConfigStore(url: root.appendingPathComponent("app_config.json")),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            BackendProcessLaunchPlan(
                processURL: root.appendingPathComponent("missing-backend"),
                arguments: []
            )
        }
    )
    let supervisor = BackendRuntimeSupervisor(
        processManager: manager,
        configStore: AppConfigStore(url: root.appendingPathComponent("app_config.json")),
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: FixedModelRestorer(result: .successful(port: 9068)),
        incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
        readinessWaiter: { _, _ in }
    )

    await #expect(throws: (any Error).self) {
        try await supervisor.ensureRunning()
    }
    #expect(supervisor.state == .failed)
    #expect(!supervisor.isRunning)
    #expect(manager.currentLaunchID == nil)
    #expect(manager.lastError?.isEmpty == false)
}

@Test @MainActor
func instanceConflictStopsAutomaticRecoveryAndPublishesStableFailure() async throws {
    let root = try runtimeTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    let launchCount = MainActorCounter()
    let processManager = BackendProcessManager(
        configStore: configStore,
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            launchCount.value += 1
            return BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: [
                    "-c",
                    "printf '%s\\n' 'Error: ironmlx_instance_already_running: lock held' >&2; exit 73",
                ]
            )
        }
    )
    let incidentStore = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: FixedModelRestorer(result: .successful(port: 9068)),
        incidentStore: incidentStore,
        policy: BackendRecoveryPolicy(
            maximumAutomaticRecoveryAttempts: 3,
            automaticRecoveryDelay: 0,
            stableWindow: 60,
            readinessTimeout: 1
        ),
        readinessWaiter: { _, _ in
            try await Task.sleep(for: .milliseconds(100))
            throw RuntimeSupervisorTestError.readiness
        }
    )
    let readyCount = MainActorCounter()

    do {
        try await supervisor.ensureRunning {
            readyCount.value += 1
        }
        Issue.record("Expected the second backend launch to fail")
    } catch let error as BackendRuntimeSupervisorError {
        #expect(error == .instanceAlreadyRunning)
    }
    try await Task.sleep(for: .milliseconds(100))

    #expect(supervisor.state == .failed)
    #expect(!supervisor.isRunning)
    #expect(supervisor.lastError?.contains("already running") == true)
    #expect(supervisor.lastEvent?.phase == .failed)
    #expect(supervisor.lastEvent?.errorCode == .instanceAlreadyRunning)
    #expect(supervisor.lastEvent?.canRetry == false)
    #expect(supervisor.lastIncident == nil)
    #expect(incidentStore.records().isEmpty)
    #expect(launchCount.value == 1)
    #expect(readyCount.value == 0)
}

@Test @MainActor
func instanceConflictDuringCrashRecoveryStopsWithoutAnotherLaunch() async throws {
    let root = try runtimeTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    let launchCount = MainActorCounter()
    let processManager = BackendProcessManager(
        configStore: configStore,
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            launchCount.value += 1
            if launchCount.value == 1 {
                return BackendProcessLaunchPlan(
                    processURL: URL(fileURLWithPath: "/bin/sleep"),
                    arguments: ["30"]
                )
            }
            return BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: [
                    "-c",
                    "printf '%s\\n' 'Error: ironmlx_instance_already_running: lock held' >&2; exit 73",
                ]
            )
        }
    )
    let incidentStore = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: FixedModelRestorer(result: .successful(port: 9068)),
        incidentStore: incidentStore,
        policy: BackendRecoveryPolicy(
            maximumAutomaticRecoveryAttempts: 3,
            automaticRecoveryDelay: 0,
            stableWindow: 60,
            readinessTimeout: 1
        ),
        readinessWaiter: { _, _ in
            try await Task.sleep(for: .milliseconds(100))
        }
    )

    try await supervisor.ensureRunning()
    let firstPID = try #require(processManager.currentProcessIdentifier)
    #expect(Darwin.kill(pid_t(firstPID), SIGKILL) == 0)
    let deadline = Date().addingTimeInterval(3)
    while supervisor.lastEvent?.errorCode != .instanceAlreadyRunning, Date() < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    try await Task.sleep(for: .milliseconds(100))

    #expect(supervisor.state == .failed)
    #expect(!supervisor.isRunning)
    #expect(supervisor.lastEvent?.errorCode == .instanceAlreadyRunning)
    #expect(supervisor.lastEvent?.canRetry == false)
    #expect(supervisor.lastIncident?.recoveryResult == "failed")
    #expect(supervisor.lastIncident?.recoverySteps.last?.action == .automaticRecoveryStopped)
    #expect(incidentStore.records().last?.recoveryResult == "failed")
    #expect(launchCount.value == 2)
}

@Test @MainActor
func staleInstanceConflictLogCannotMisclassifyANewerLaunch() async throws {
    let root = try runtimeTemporaryDirectory()
    let logStore = IronMLXLogStore(rootURL: root.appendingPathComponent("logs"))
    try logStore.appendLine(
        "Error: ironmlx_instance_already_running: stale failure",
        to: .backend
    )
    let manager = BackendProcessManager(
        configStore: AppConfigStore(url: root.appendingPathComponent("app_config.json")),
        logStore: logStore,
        launchPlanProvider: {
            BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "printf '%s\\n' 'new launch failure' >&2; exit 17"]
            )
        }
    )

    _ = try manager.startProcess()
    let deadline = Date().addingTimeInterval(2)
    while manager.lastTermination == nil, Date() < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }

    let termination = try #require(manager.lastTermination)
    #expect(termination.failureCode == nil)
    #expect(termination.logTail.contains("new launch failure"))
    #expect(!termination.logTail.contains("stale failure"))
}

@Test @MainActor
func readinessFailureTerminatesUnhealthyProcessAndEntersFailed() async throws {
    let harness = try runtimeHarness(
        readinessWaiter: { _, _ in
            throw RuntimeSupervisorTestError.readiness
        }
    )

    await #expect(throws: BackendRuntimeSupervisorError.self) {
        try await harness.supervisor.ensureRunning()
    }

    #expect(harness.supervisor.state == .failed)
    #expect(!harness.supervisor.isRunning)
    #expect(harness.processManager.currentLaunchID == nil)
    #expect(harness.processManager.lastTermination?.stopIntent == .startupFailureCleanup)
    #expect(harness.supervisor.lastEvent?.phase == .failed)
    #expect(harness.supervisor.lastEvent?.errorCode == .backendReadinessFailed)
    #expect(harness.supervisor.lastEvent?.processHealthy == false)
    #expect(
        BackendRuntimeSupervisorError.readinessFailed("health payload invalid").failureCode
            == .backendReadinessFailed
    )
}

@Test @MainActor
func backendExitDuringStartupFailsAtOnceWithoutAutomaticRecovery() async throws {
    let harness = try runtimeHarness(
        launchPlan: { _ in
            BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: [
                    "-c",
                    "sleep 0.2; printf '%s\\n' 'Error: fixed DFlash2 budget 7 exceeds supported 3' >&2; exit 1",
                ]
            )
        },
        // A launch that never answers health, like a DFlash2 warmup.
        readinessWaiter: { _, _ in
            try await Task.sleep(for: .seconds(60))
            throw RuntimeSupervisorTestError.readiness
        }
    )
    #expect(BackendRecoveryPolicy().readinessTimeout == 300)

    let started = Date()
    do {
        try await harness.supervisor.ensureRunning()
        Issue.record("Expected the exiting backend to fail its start")
    } catch let error as BackendRuntimeSupervisorError {
        #expect(error.failureCode == .backendReadinessFailed)
        #expect(error.localizedDescription.contains("exited during startup (exit code 1)"))
    }
    // Well inside the 60-second readiness wait, even on a loaded test host.
    #expect(Date().timeIntervalSince(started) < 30)
    // Automatic recovery would relaunch after its one-second delay.
    try await Task.sleep(for: .milliseconds(1500))

    #expect(harness.launchCount.value == 1)
    #expect(harness.supervisor.state == .failed)
    #expect(harness.supervisor.lastEvent?.phase == .failed)
    #expect(harness.supervisor.lastEvent?.errorCode == .backendReadinessFailed)
    #expect(harness.supervisor.lastEvent?.terminationStatus == 1)
    #expect(harness.supervisor.lastEvent?.logTail?.contains("exceeds supported 3") == true)
    #expect(harness.supervisor.lastEvent?.canRetry == true)
    #expect(harness.supervisor.lastIncident == nil)
    #expect(harness.incidentStore.records().isEmpty)
}

@Test @MainActor
func recoveryLaunchThatExitsDuringStartupIsNotRelaunched() async throws {
    let readinessCalls = CallCounter()
    let harness = try runtimeHarness(
        policy: BackendRecoveryPolicy(
            maximumAutomaticRecoveryAttempts: 3,
            automaticRecoveryDelay: 0,
            stableWindow: 60
        ),
        launchPlan: { launchNumber in
            launchNumber == 1
                ? BackendProcessLaunchPlan(
                    processURL: URL(fileURLWithPath: "/bin/sleep"),
                    arguments: ["30"]
                )
                : BackendProcessLaunchPlan(
                    processURL: URL(fileURLWithPath: "/bin/sh"),
                    arguments: ["-c", "sleep 0.2; exit 1"]
                )
        },
        // Only the first launch becomes healthy; the recovery launch exits
        // while it is still starting.
        readinessWaiter: { _, _ in
            if await readinessCalls.next() > 1 {
                try await Task.sleep(for: .seconds(60))
            }
        }
    )
    try await harness.supervisor.ensureRunning()

    let firstPID = try #require(harness.processManager.currentProcessIdentifier)
    #expect(Darwin.kill(pid_t(firstPID), SIGKILL) == 0)
    let deadline = Date().addingTimeInterval(5)
    while harness.supervisor.lastIncident?.recoveryResult != "failed", Date() < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    try await Task.sleep(for: .milliseconds(500))

    // The crash is recovered once; the recovery launch's startup exit is not.
    #expect(harness.launchCount.value == 2)
    #expect(harness.supervisor.state == .failed)
    #expect(harness.supervisor.lastIncident?.recoveryAttempt == 1)
    #expect(harness.supervisor.lastIncident?.recoveryResult == "failed")
    #expect(
        harness.supervisor.lastIncident?.error?.contains("exited during startup") == true
    )
    #expect(harness.incidentStore.records().count == 1)
}

@Test @MainActor
func partialModelRestoreKeepsHealthyBackendExplicitlyDegraded() async throws {
    let result = BackendRestartResult(
        success: false,
        status: "models_partially_loaded",
        port: 9068,
        model: "org/first",
        modelLoaded: true,
        loadedModels: ["org/first"],
        failedModels: ["org/second"],
        error: "org/second failed to load"
    )
    let harness = try runtimeHarness(
        restorer: FixedModelRestorer(result: result)
    )

    try await harness.supervisor.ensureRunning()

    #expect(harness.supervisor.state == .degraded)
    #expect(harness.supervisor.isRunning)
    #expect(harness.supervisor.lastEvent?.phase == .failed)
    #expect(harness.supervisor.lastEvent?.runtimeState == .degraded)
    #expect(harness.supervisor.lastEvent?.failedModels == ["org/second"])
    #expect(harness.supervisor.lastEvent?.processHealthy == true)

    await harness.supervisor.stop(intent: .userStop)
}

@Test @MainActor
func completeModelRestoreFailureIsExplicitWhileHealthyBackendStaysAvailable() async throws {
    let result = BackendRestartResult(
        success: false,
        status: "model_load_failed",
        port: 9068,
        model: "org/failed",
        failedModels: ["org/failed"],
        error: "model restore failed"
    )
    let harness = try runtimeHarness(
        restorer: FixedModelRestorer(result: result)
    )

    try await harness.supervisor.ensureRunning()

    #expect(harness.supervisor.state == .failed)
    #expect(harness.supervisor.isRunning)
    #expect(harness.supervisor.lastError == "model restore failed")
    #expect(harness.supervisor.lastEvent?.runtimeState == .failed)
    #expect(harness.supervisor.lastEvent?.processHealthy == true)
    #expect(harness.supervisor.lastEvent?.canRetry == true)

    await harness.supervisor.stop(intent: .userStop)
}

@Test @MainActor
func crashRecoveryUsesOnlyLastConfirmedModelsAndTheirRuntimeParameters() async throws {
    let root = try runtimeTemporaryDirectory()
    let firstModel = "org/confirmed"
    let unconfirmedModel = "org/in-flight"
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(
        AppConfig(
            defaultModel: firstModel,
            loadedModels: [firstModel],
            pinnedModels: [firstModel]
        )
    )
    let parameterStore = ModelParameterStore(
        url: root.appendingPathComponent("model_params.json")
    )
    try parameterStore.save(
        ModelParameters(
            modelID: firstModel,
            maxTokens: "8192",
            temperature: "0.7",
            topP: "0.9",
            topK: "40",
            repeatPenalty: "1.1",
            promptLookupEnabled: true,
            promptLookupCrossRequest: true
        )
    )
    let recorder = SnapshotRecordingRestorer()
    let processManager = BackendProcessManager(
        configStore: configStore,
        terminator: BackendProcessTerminator(sleep: { _ in }),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["30"]
            )
        }
    )
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: parameterStore,
        restartCoordinator: recorder,
        incidentStore: BackendIncidentStore(
            url: root.appendingPathComponent("incidents.json")
        ),
        policy: BackendRecoveryPolicy(
            maximumAutomaticRecoveryAttempts: 1,
            automaticRecoveryDelay: 0,
            stableWindow: 60,
            readinessTimeout: 1
        ),
        readinessWaiter: { _, _ in }
    )

    try await supervisor.ensureRunning()
    supervisor.confirmLoadedModels(
        [
            BackendLoadedModelInfo(
                id: firstModel,
                model: firstModel,
                path: "/models/confirmed",
                architecture: "llm",
                isDefault: true,
                maxPositionEmbeddings: 32_768,
                pinned: true,
                mtpEnabled: true,
                mtpModelDir: "/models/confirmed-mtp",
                mtpDraftTokens: 3,
                promptLookup: .crossRequest
            ),
        ],
        parameterConfirmedModelIDs: [firstModel]
    )
    var changedConfig = configStore.load()
    changedConfig.loadedModels = [firstModel, unconfirmedModel]
    configStore.save(changedConfig)

    let crashedPID = try #require(processManager.currentProcessIdentifier)
    #expect(Darwin.kill(pid_t(crashedPID), SIGKILL) == 0)
    try await waitForRuntimeSnapshotCount(recorder, count: 2)

    let snapshots = await recorder.snapshots
    let recoverySnapshot = try #require(snapshots.last)
    let model = try #require(recoverySnapshot.models.first)
    #expect(recoverySnapshot.models.map(\.id) == [firstModel])
    #expect(!recoverySnapshot.models.map(\.id).contains(unconfirmedModel))
    #expect(recoverySnapshot.config.restoredModelReferences == [firstModel])
    #expect(recoverySnapshot.config.defaultModelReference == firstModel)
    #expect(recoverySnapshot.config.pinnedModelReferences == [firstModel])
    #expect(model.isDefault)
    #expect(model.pinned)
    #expect(model.maxCacheCap == 8192)
    #expect(model.mtpModelDir == "/models/confirmed-mtp")
    #expect(model.mtpDraftTokens == 3)
    #expect(model.promptLookup == .crossRequest)
    #expect(model.samplingDefaults.temperature == 0.7)
    #expect(model.samplingDefaults.topP == 0.9)
    #expect(model.samplingDefaults.topK == 40)
    #expect(model.samplingDefaults.repetitionPenalty == 1.1)
    #expect(supervisor.lastIncident?.modelsBeforeCrash == [firstModel])

    await supervisor.stop(intent: .userStop)
}

@Test @MainActor
func cancelledRestoreIsNotRestoredAgain() async throws {
    let root = try runtimeTemporaryDirectory()
    let kept = "org/kept"
    let cancelled = "org/cancelled"
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig(defaultModel: kept, loadedModels: [kept, cancelled]))
    let restorer = CancellingModelRestorer(cancelled: cancelled)
    let processManager = BackendProcessManager(
        configStore: configStore,
        terminator: BackendProcessTerminator(sleep: { _ in }),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["30"]
            )
        }
    )
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        restartCoordinator: restorer,
        incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
        // Its loaded-models notification must not reach other tests' Dashboards.
        notificationCenter: NotificationCenter(),
        policy: BackendRecoveryPolicy(
            maximumAutomaticRecoveryAttempts: 1,
            automaticRecoveryDelay: 0,
            stableWindow: 60,
            readinessTimeout: 1
        ),
        readinessWaiter: { _, _ in }
    )

    try await supervisor.ensureRunning()

    // The next App launch reads the config.
    #expect(configStore.load().restoredModelReferences == [kept])
    // An automatic recovery restores from the confirmed snapshot.
    let crashedPID = try #require(processManager.currentProcessIdentifier)
    #expect(Darwin.kill(pid_t(crashedPID), SIGKILL) == 0)
    try await waitForCancellingRestorerCount(restorer, count: 2)
    let recovery = try #require(await restorer.snapshots.last)
    #expect(recovery.models.map(\.id) == [kept])

    await supervisor.stop(intent: .userStop)
}

private actor CancellingModelRestorer: BackendModelRestoring {
    let cancelled: String
    private(set) var snapshots: [BackendRecoverySnapshot] = []

    init(cancelled: String) {
        self.cancelled = cancelled
    }

    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        snapshots.append(snapshot)
        let ids = snapshot.models.map(\.id)
        let loaded = ids.filter { $0 != cancelled }
        var result = BackendRestartResult(
            success: true,
            status: loaded.isEmpty ? "restarted" : "models_loaded",
            port: snapshot.config.port,
            model: loaded.first,
            modelLoaded: !loaded.isEmpty,
            loadedModels: loaded
        )
        result.cancelledModels = ids.filter { $0 == cancelled }
        return result
    }
}

@MainActor
private func waitForCancellingRestorerCount(_ restorer: CancellingModelRestorer, count: Int) async throws {
    for _ in 0..<500 {
        if await restorer.snapshots.count >= count { return }
        try await Task.sleep(for: .milliseconds(10))
    }
    Issue.record("restorer was called \(await restorer.snapshots.count) times, expected \(count)")
}

private struct RuntimeHarness {
    let supervisor: BackendRuntimeSupervisor
    let processManager: BackendProcessManager
    let incidentStore: BackendIncidentStore
    let notificationCenter: NotificationCenter
    let launchCount: MainActorCounter
}

@MainActor
private func runtimeHarness(
    restorer: any BackendModelRestoring = FixedModelRestorer(
        result: .successful(port: 9068)
    ),
    policy: BackendRecoveryPolicy = BackendRecoveryPolicy(),
    launchPlan: @escaping @MainActor (_ launchNumber: Int) -> BackendProcessLaunchPlan = { _ in
        BackendProcessLaunchPlan(
            processURL: URL(fileURLWithPath: "/bin/sleep"),
            arguments: ["30"]
        )
    },
    readinessWaiter: @escaping BackendRuntimeSupervisor.ReadinessWaiter = { _, _ in }
) throws -> RuntimeHarness {
    let root = try runtimeTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    let notificationCenter = NotificationCenter()
    let launchCount = MainActorCounter()
    let processManager = BackendProcessManager(
        configStore: configStore,
        terminator: BackendProcessTerminator(sleep: { _ in }),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        notificationCenter: notificationCenter,
        launchPlanProvider: {
            launchCount.value += 1
            return launchPlan(launchCount.value)
        }
    )
    let incidentStore = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: restorer,
        incidentStore: incidentStore,
        notificationCenter: notificationCenter,
        policy: policy,
        readinessWaiter: readinessWaiter
    )
    return RuntimeHarness(
        supervisor: supervisor,
        processManager: processManager,
        incidentStore: incidentStore,
        notificationCenter: notificationCenter,
        launchCount: launchCount
    )
}

private struct FixedModelRestorer: BackendModelRestoring {
    let result: BackendRestartResult

    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        result
    }
}

private actor GatedModelRestorer: BackendModelRestoring {
    private(set) var didStart = false
    private var released = false
    private var continuation: CheckedContinuation<Void, Never>?

    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        didStart = true
        if !released {
            await withCheckedContinuation { continuation in
                self.continuation = continuation
            }
        }
        return BackendRestartResult(success: true, status: "restarted", port: snapshot.config.port)
    }

    func release() {
        released = true
        continuation?.resume()
        continuation = nil
    }
}

private actor SnapshotRecordingRestorer: BackendModelRestoring {
    private(set) var snapshots: [BackendRecoverySnapshot] = []

    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        snapshots.append(snapshot)
        return BackendRestartResult(
            success: true,
            status: "models_loaded",
            port: snapshot.config.port,
            model: snapshot.models.first(where: \.isDefault)?.id,
            modelLoaded: !snapshot.models.isEmpty,
            loadedModels: snapshot.models.map(\.id)
        )
    }
}

private extension BackendRestartResult {
    static func successful(port: UInt16) -> BackendRestartResult {
        BackendRestartResult(success: true, status: "restarted", port: port)
    }
}

@MainActor
private final class MainActorCounter {
    var value = 0
}

private actor CallCounter {
    private var count = 0

    func next() -> Int {
        count += 1
        return count
    }
}

@MainActor
private final class BackendStateRecorder: NSObject {
    private let manager: BackendProcessManager
    private(set) var states: [BackendProcessState] = []

    init(manager: BackendProcessManager) {
        self.manager = manager
    }

    @objc func runtimeDidChange(_ notification: Notification) {
        states.append(manager.state)
    }
}

private enum RuntimeSupervisorTestError: LocalizedError {
    case readiness

    var errorDescription: String? {
        "healthz did not become ready"
    }
}

private func waitForRuntimeSnapshotCount(
    _ recorder: SnapshotRecordingRestorer,
    count: Int,
    timeout: TimeInterval = 3
) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while await recorder.snapshots.count < count, Date() < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    let observedCount = await recorder.snapshots.count
    if observedCount < count {
        Issue.record("Timed out waiting for \(count) recovery snapshots; observed \(observedCount)")
    }
}

private func runtimeTemporaryDirectory() throws -> URL {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent(
            "ironmlx-runtime-supervisor-\(UUID().uuidString)",
            isDirectory: true
        )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
