import Foundation
import Testing

@testable import IronMLXAppCore

// MARK: - Fakes

/// The backend's load registry as the center sees it through the client.
private actor FakeModelLoadClient: BackendModelLoadTracking {
    enum StartScript {
        case acceptSent(state: ModelLoadState = .queued)
        case accept(loadID: String, state: ModelLoadState = .queued)
        case complete(BackendModelAdminResponse)
        case fail(Error)
        /// Holds the start request open until `releaseStart()`, then accepts.
        case gated(loadID: String)
        /// Registers the load, fails it in `stage` and answers `503 code`,
        /// like a full engine pool.
        case refuseAfterRegistering(code: String, stage: ModelLoadStage)
    }

    private(set) var loads: [BackendLoadActivity] = []
    private(set) var fetchCount = 0
    private(set) var cancelledLoadIDs: [String] = []
    private(set) var startRequests: [BackendLoadModelRequest] = []
    private var startScripts: [StartScript]
    private var cancelResult: BackendLoadCancelResult = .cancelling
    private let loadedModels: [BackendLoadedModelInfo]
    private let loadsProvider: (@Sendable () -> [BackendLoadActivity])?
    private var startGate: CheckedContinuation<Void, Never>?
    private(set) var startIsGated = false

    init(
        loads: [BackendLoadActivity] = [],
        startScripts: [StartScript] = [],
        loadedModels: [BackendLoadedModelInfo] = [],
        loadsProvider: (@Sendable () -> [BackendLoadActivity])? = nil
    ) {
        self.loads = loads
        self.startScripts = startScripts
        self.loadedModels = loadedModels
        self.loadsProvider = loadsProvider
    }

    func setLoads(_ loads: [BackendLoadActivity]) {
        self.loads = loads
    }

    func fetchModelLoads() async throws -> [BackendLoadActivity] {
        fetchCount += 1
        return loadsProvider?() ?? loads
    }

    /// Answers like the backend: a finished load cannot be cancelled.
    func cancelModelLoad(loadID: String) async throws -> BackendLoadCancelResult {
        cancelledLoadIDs.append(loadID)
        guard let load = loads.first(where: { $0.loadID == loadID }) else { return .notFound }
        return load.state.isTerminal ? .finished : cancelResult
    }

    func releaseStart() {
        startGate?.resume()
        startGate = nil
    }

    func startModelLoad(_ request: BackendLoadModelRequest) async throws -> BackendModelLoadStart {
        startRequests.append(request)
        let script = startScripts.isEmpty ? .acceptSent() : startScripts.removeFirst()
        switch script {
        case let .acceptSent(state):
            return .accepted(loadID: request.loadID ?? "", state: state)
        case let .accept(loadID, state):
            return .accepted(loadID: loadID, state: state)
        case let .complete(response):
            return .completed(response)
        case let .fail(error):
            throw error
        case let .gated(loadID):
            startIsGated = true
            await withCheckedContinuation { startGate = $0 }
            startIsGated = false
            return .accepted(loadID: loadID, state: .queued)
        case let .refuseAfterRegistering(code, stage):
            loads.append(BackendLoadActivity(
                loadID: request.loadID ?? "",
                model: request.model,
                variant: .plain,
                kind: .load,
                trigger: .admin,
                state: .failed,
                stage: stage,
                completedStages: [.init(stage: .queued, elapsedMs: 1), .init(stage: stage, elapsedMs: 2)],
                startedUnixMs: 1_800_000_000_000,
                finishedUnixMs: 1_800_000_000_003,
                message: "refused",
                code: code
            ))
            throw BackendAPIError.serverResponse(
                statusCode: 503,
                body: #"{"success":false,"status":"error","code":"\#(code)","loaded_models":[],"error":"refused"}"#
            )
        }
    }

    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        loadedModels
    }
}

/// Hands each poll interval to the test, which releases it explicitly.
private actor ManualSleeper {
    private(set) var requests: [Duration] = []
    private var pending: [CheckedContinuation<Void, Never>] = []

    func sleep(_ duration: Duration) async {
        requests.append(duration)
        await withCheckedContinuation { pending.append($0) }
    }

    func releaseOne() {
        guard !pending.isEmpty else { return }
        pending.removeFirst().resume()
    }
}

/// Simulated time: every poll interval passes at once.
private final class SimulatedClock: @unchecked Sendable {
    private let lock = NSLock()
    private var elapsed: Duration = .zero
    private var base = Date(timeIntervalSince1970: 1_800_000_000)

    var now: Date {
        lock.withLock { base.addingTimeInterval(seconds(elapsed)) }
    }

    var elapsedTime: Duration {
        lock.withLock { elapsed }
    }

    func advance(_ duration: Duration) {
        lock.withLock { elapsed += duration }
    }

    private func seconds(_ duration: Duration) -> TimeInterval {
        let components = duration.components
        return TimeInterval(components.seconds) + TimeInterval(components.attoseconds) / 1e18
    }
}

@MainActor
private func makeCenter(
    _ client: FakeModelLoadClient,
    sleeper: ManualSleeper? = nil,
    clock: SimulatedClock = SimulatedClock(),
    autoAdvance: Bool = false,
    notificationCenter: NotificationCenter = NotificationCenter()
) -> ModelLoadActivityCenter {
    ModelLoadActivityCenter(
        client: { client },
        notificationCenter: notificationCenter,
        pollSleep: { duration in
            if let sleeper {
                await sleeper.sleep(duration)
            } else if autoAdvance {
                clock.advance(duration)
                await Task.yield()
            } else {
                try? await Task.sleep(for: .milliseconds(5))
            }
        },
        now: { clock.now },
        retentionSleep: { _ in try? await Task.sleep(for: .seconds(3_600)) }
    )
}

private func backendLoad(
    _ loadID: String,
    model: String = "mlx-community/Tiny-4bit",
    state: ModelLoadState,
    stage: ModelLoadStage,
    kind: ModelLoadKind = .load,
    variant: ModelLoadVariant = .plain,
    trigger: BackendLoadTrigger = .admin,
    completed: [BackendLoadActivity.CompletedStage] = [],
    bytesDone: UInt64 = 0,
    bytesTotal: UInt64 = 0,
    message: String? = nil,
    code: String? = nil
) -> BackendLoadActivity {
    BackendLoadActivity(
        loadID: loadID,
        model: model,
        variant: variant,
        kind: kind,
        trigger: trigger,
        state: state,
        stage: stage,
        completedStages: completed,
        bytesDone: bytesDone,
        bytesTotal: bytesTotal,
        startedUnixMs: 1_800_000_000_000,
        finishedUnixMs: state.isTerminal ? 1_800_000_001_000 : nil,
        message: message,
        code: code
    )
}

private func eventually(
    _ timeout: Duration = .seconds(5),
    _ condition: @MainActor () async -> Bool
) async {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while clock.now < deadline {
        if await condition() { return }
        try? await Task.sleep(for: .milliseconds(1))
    }
}

private let model = "mlx-community/Tiny-4bit"

// MARK: - State merge and polling

@Test @MainActor
func merges_app_stage_then_backend_stages() async throws {
    let client = FakeModelLoadClient()
    let sleeper = ManualSleeper()
    let clock = SimulatedClock()
    let center = makeCenter(client, sleeper: sleeper, clock: clock)

    let ticket = center.begin(model: model, variant: .plain, origin: .dashboard)
    center.enterAppStage(ticket, .verifyingFiles)
    clock.advance(.milliseconds(1_500))
    center.enterAppStage(ticket, .waitingForBackend)
    clock.advance(.milliseconds(250))

    var activity = try #require(center.activities.first)
    #expect(activity.state == .loading)
    #expect(activity.stage == .waitingForBackend)
    #expect(activity.completedStages == [ModelLoadCompletedStage(stage: .verifyingFiles, elapsedMs: 1_500)])
    #expect(activity.origin == .dashboard)
    #expect(activity.autoOpenPanel)
    #expect(activity.cancellable)

    await client.setLoads([
        backendLoad(
            "L1",
            state: .loading,
            stage: .readingWeights,
            completed: [
                .init(stage: .queued, elapsedMs: 10),
                .init(stage: .preparingMemory, elapsedMs: 20),
            ],
            bytesDone: 5,
            bytesTotal: 10
        ),
    ])
    center.attachBackendLoad(ticket, loadID: "L1")
    await eventually { await sleeper.requests.count >= 1 }
    await sleeper.releaseOne()
    await eventually { await client.fetchCount >= 2 }
    await eventually { center.activities.first?.stage == .readingWeights }

    activity = try #require(center.activities.first)
    #expect(center.activities.count == 1)
    #expect(activity.stage == .readingWeights)
    #expect(activity.completedStages.map(\.stage) == [
        .verifyingFiles, .waitingForBackend, .queued, .preparingMemory,
    ])
    #expect(activity.completedStages[1].elapsedMs == 250)
    #expect(activity.bytesDone == 5)
    #expect(activity.bytesTotal == 10)
    #expect(activity.origin == .dashboard)
}

@Test @MainActor
func polls_every_500ms_while_active_and_stops_when_idle() async throws {
    let client = FakeModelLoadClient(loads: [backendLoad("L1", state: .loading, stage: .readingWeights)])
    let sleeper = ManualSleeper()
    let center = makeCenter(client, sleeper: sleeper)
    #expect(!center.isPolling)

    let ticket = center.begin(model: model, variant: .plain, origin: .chat)
    center.attachBackendLoad(ticket, loadID: "L1")
    await eventually { await sleeper.requests.count == 1 }
    await sleeper.releaseOne()
    await eventually { await sleeper.requests.count == 2 }
    #expect(await sleeper.requests == [.milliseconds(500), .milliseconds(500)])

    await client.setLoads([backendLoad("L1", state: .loaded, stage: .startingEngine)])
    center.finish(ticket, .loaded)
    await sleeper.releaseOne()
    await eventually { !center.isPolling }

    #expect(!center.isPolling)
    #expect(await sleeper.requests == [.milliseconds(500), .milliseconds(500)])
    let fetches = await client.fetchCount
    try await Task.sleep(for: .milliseconds(50))
    #expect(await client.fetchCount == fetches)
}

@Test @MainActor
func polls_every_2s_when_dashboard_visible_and_idle() async throws {
    let client = FakeModelLoadClient()
    let sleeper = ManualSleeper()
    let center = makeCenter(client, sleeper: sleeper)

    center.setDashboardVisible(true)
    await eventually { await sleeper.requests.count == 1 }
    await sleeper.releaseOne()
    await eventually { await sleeper.requests.count == 2 }
    #expect(await sleeper.requests == [.seconds(2), .seconds(2)])
    #expect(center.isPolling)

    center.setDashboardVisible(false)
    await sleeper.releaseOne()
    await eventually { !center.isPolling }
    #expect(!center.isPolling)
    #expect(await client.fetchCount == 2)
}

@Test @MainActor
func async_load_waits_for_registry_result_beyond_60_seconds() async throws {
    let clock = SimulatedClock()
    let client = FakeModelLoadClient(
        startScripts: [.accept(loadID: "L90")],
        loadedModels: [BackendLoadedModelInfo(
            id: model, model: model, path: "/models/tiny", architecture: "llama",
            isDefault: true, maxPositionEmbeddings: 4_096
        )],
        loadsProvider: {
            clock.elapsedTime < .seconds(90)
                ? [backendLoad("L90", state: .loading, stage: .readingWeights)]
                : [backendLoad("L90", state: .loaded, stage: .startingEngine)]
        }
    )
    let center = makeCenter(client, clock: clock, autoAdvance: true)

    let response = try await center.loadModel(
        BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
        origin: .chat
    )

    #expect(response.success)
    #expect(response.status == "loaded")
    #expect(response.loadedModels.map(\.id) == [model])
    #expect(clock.elapsedTime >= .seconds(90))
    let sent = try #require(await client.startRequests.first)
    #expect(sent.async == true)
    #expect(sent.loadID?.isEmpty == false)
    #expect(center.activities.first?.state == .loaded)
}

@Test @MainActor
func async_load_result_carries_the_backend_warning() async throws {
    var loaded = backendLoad("W1", state: .loaded, stage: .startingEngine)
    loaded.warningCode = "scheduler_profile_hardware_mismatch"
    loaded.warning = "This model's scheduler profile was generated on other hardware."
    let client = FakeModelLoadClient(
        loads: [loaded],
        startScripts: [.accept(loadID: "W1")]
    )
    let center = makeCenter(client)

    let response = try await center.loadModel(
        BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
        origin: .dashboard
    )

    #expect(response.warningCode == "scheduler_profile_hardware_mismatch")
    #expect(response.warning == "This model's scheduler profile was generated on other hardware.")
    // The Dashboard's load callback reads these fields for its info toast.
    let json = try #require(String(data: JSONEncoder().encode(response), encoding: .utf8))
    #expect(json.contains(#""warning_code":"scheduler_profile_hardware_mismatch""#))
}

/// The backend forgets a finished load 30 s after it ends; a poll that comes
/// later only sees the load gone.
private func expiringLoad(_ clock: SimulatedClock) -> @Sendable () -> [BackendLoadActivity] {
    {
        clock.elapsedTime < .seconds(2)
            ? [backendLoad("GONE", state: .loading, stage: .readingWeights)]
            : []
    }
}

@Test @MainActor
func expired_load_is_confirmed_through_loaded_models() async throws {
    let clock = SimulatedClock()
    let client = FakeModelLoadClient(
        startScripts: [.accept(loadID: "GONE")],
        loadedModels: [BackendLoadedModelInfo(
            id: model, model: model, path: "/models/tiny", architecture: "llama",
            isDefault: true, maxPositionEmbeddings: 4_096
        )],
        loadsProvider: expiringLoad(clock)
    )
    let center = makeCenter(client, clock: clock, autoAdvance: true)

    let response = try await center.loadModel(
        BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
        origin: .dashboard
    )

    #expect(response.success)
    #expect(response.status == "loaded")
    #expect(response.loadedModels.map(\.id) == [model])
}

@Test @MainActor
func expired_load_without_the_model_fails_with_a_localized_code() async throws {
    let clock = SimulatedClock()
    let client = FakeModelLoadClient(
        startScripts: [.accept(loadID: "GONE")],
        loadsProvider: expiringLoad(clock)
    )
    let center = makeCenter(client, clock: clock, autoAdvance: true)

    do {
        _ = try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
            origin: .dashboard
        )
        Issue.record("the load must not report success")
    } catch let error as ModelLoadFailedError {
        #expect(error.code == "load_result_unknown")
        #expect(DashboardErrorClassifier.code(for: error) == "load_result_unknown")
    }
    #expect(center.activities.first?.state == .failed)
    #expect(center.activities.first?.code == "load_result_unknown")
}

// MARK: - Cancellation

@Test @MainActor
func cancel_in_app_stage_cancels_task() async throws {
    let client = FakeModelLoadClient()
    let center = makeCenter(client)
    let load = Task { @MainActor in
        try await center.track(model: model, variant: .plain, origin: .dashboard) { ticket in
            center.enterAppStage(ticket, .verifyingFiles)
            try await Task.sleep(for: .seconds(60))
            return 1
        }
    }
    await eventually { center.activities.first?.stage == .verifyingFiles }

    await center.cancel(model: model)

    await #expect(throws: ModelLoadCancelledError.self) {
        _ = try await load.value
    }
    let activity = try #require(center.activities.first)
    #expect(activity.state == .cancelled)
    #expect(activity.code == ModelLoadCancelledError.loadCancelledCode)
    #expect(!activity.cancellable)
    #expect(await client.cancelledLoadIDs.isEmpty)
}

@Test @MainActor
func cancel_in_backend_stage_calls_cancel_endpoint() async throws {
    let client = FakeModelLoadClient(startScripts: [.accept(loadID: "L1")])
    await client.setLoads([backendLoad("L1", state: .loading, stage: .readingWeights)])
    let center = makeCenter(client)
    let load = Task { @MainActor in
        try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: false),
            origin: .dashboard
        )
    }
    await eventually { center.activities.first?.stage == .readingWeights }

    await center.cancel(model: model)
    #expect(await client.cancelledLoadIDs == ["L1"])
    #expect(center.activities.first?.state == .cancelling)

    await client.setLoads([
        backendLoad("L1", state: .cancelled, stage: .readingWeights, code: "load_cancelled"),
    ])
    await #expect(throws: ModelLoadCancelledError.self) {
        _ = try await load.value
    }
    let activity = try #require(center.activities.first)
    #expect(activity.state == .cancelled)
    #expect(activity.code == ModelLoadCancelledError.loadCancelledCode)
}

@Test @MainActor
func load_in_progress_waits_instead_of_failing() async throws {
    let conflict = BackendAPIError.serverResponse(
        statusCode: 409,
        body: #"{"success":false,"status":"error","code":"load_in_progress","loaded_models":[],"error":"busy","load_id":"OTHER"}"#
    )
    let loaded = BackendModelAdminResponse(
        success: true, status: "loaded", code: nil, model: model,
        loadedModels: [], warningCode: nil, warning: nil, error: nil
    )
    let client = FakeModelLoadClient(startScripts: [.fail(conflict), .complete(loaded)])
    await client.setLoads([backendLoad("OTHER", state: .loading, stage: .readingWeights)])
    let center = makeCenter(client)
    let load = Task { @MainActor in
        try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
            origin: .starter
        )
    }
    await eventually { center.activities.first?.stage == .readingWeights }
    #expect(center.activities.first?.state == .loading)
    #expect(center.activities.first?.origin == .starter)
    #expect(await client.startRequests.count == 1)

    await client.setLoads([backendLoad("OTHER", state: .loaded, stage: .startingEngine)])
    let response = try await load.value

    #expect(response.success)
    let requests = await client.startRequests
    #expect(requests.count == 2)
    #expect(requests[0].loadID != requests[1].loadID)
    #expect(center.activities.first?.state == .loaded)
}

private let loadInProgressConflict = BackendAPIError.serverResponse(
    statusCode: 409,
    body: #"{"success":false,"status":"error","code":"load_in_progress","loaded_models":[],"error":"busy","load_id":"OTHER"}"#
)

@Test @MainActor
func retry_after_load_in_progress_never_publishes_the_other_loads_result() async throws {
    let client = FakeModelLoadClient(startScripts: [.fail(loadInProgressConflict), .gated(loadID: "MINE")])
    await client.setLoads([backendLoad("OTHER", state: .loading, stage: .readingWeights)])
    let notifications = NotificationCenter()
    let center = makeCenter(client, notificationCenter: notifications)
    var published: [ModelLoadActivity] = []
    let observer = notifications.addObserver(
        forName: .ironMLXModelLoadsDidChange, object: center, queue: nil
    ) { _ in
        MainActor.assumeIsolated {
            if let activity = center.activities.first { published.append(activity) }
        }
    }
    defer { notifications.removeObserver(observer) }
    let load = Task { @MainActor in
        try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
            origin: .chat
        )
    }
    await eventually { center.activities.first?.stage == .readingWeights }

    // The other load fails; this one then starts its own load.
    await client.setLoads([
        backendLoad("OTHER", state: .failed, stage: .readingWeights, message: "other boom", code: "other_code"),
    ])
    await eventually { await client.startIsGated }
    #expect(center.activities.first?.state == .loading)
    #expect(center.activities.first?.cancellable == true)
    await client.setLoads([backendLoad("MINE", state: .loading, stage: .readingWeights)])
    await client.releaseStart()
    await eventually { center.activities.first?.stage == .readingWeights }
    await client.setLoads([backendLoad("MINE", state: .loaded, stage: .startingEngine)])
    _ = try await load.value

    let beforeFinish = published.dropLast()
    #expect(!beforeFinish.isEmpty)
    #expect(beforeFinish.allSatisfy { !$0.state.isTerminal })
    #expect(published.allSatisfy { $0.message != "other boom" && $0.code != "other_code" })
    #expect(published.last?.state == .loaded)
}

@Test @MainActor
func cancel_while_retry_start_in_flight_cancels_the_new_load() async throws {
    let client = FakeModelLoadClient(startScripts: [.fail(loadInProgressConflict), .gated(loadID: "MINE")])
    await client.setLoads([backendLoad("OTHER", state: .loading, stage: .readingWeights)])
    let center = makeCenter(client)
    let load = Task { @MainActor in
        try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
            origin: .chat
        )
    }
    await eventually { center.activities.first?.stage == .readingWeights }
    await client.setLoads([backendLoad("OTHER", state: .loaded, stage: .startingEngine)])
    await eventually { await client.startIsGated }

    await center.cancel(model: model)
    #expect(center.activities.first?.state == .cancelling)
    await client.setLoads([backendLoad("MINE", state: .loading, stage: .readingWeights)])
    await client.releaseStart()
    await eventually { await client.cancelledLoadIDs.contains("MINE") }

    #expect(await client.cancelledLoadIDs == ["MINE"])
    await client.setLoads([backendLoad("MINE", state: .cancelled, stage: .readingWeights, code: "load_cancelled")])
    await #expect(throws: ModelLoadCancelledError.self) {
        _ = try await load.value
    }
    #expect(center.activities.first?.state == .cancelled)
}

@Test @MainActor
func failure_persists_until_acknowledged_for_app_loads() async throws {
    let clock = SimulatedClock()
    let client = FakeModelLoadClient(startScripts: [.accept(loadID: "L1")])
    await client.setLoads([
        backendLoad("L1", state: .failed, stage: .readingWeights, message: "out of memory", code: "gpu_memory_insufficient"),
    ])
    let center = makeCenter(client, clock: clock)

    do {
        _ = try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true),
            origin: .restore
        )
        Issue.record("Expected the load to fail")
    } catch let error as ModelLoadFailedError {
        #expect(error.code == "gpu_memory_insufficient")
        #expect(error.localizedDescription == "out of memory")
    }

    await client.setLoads([])
    clock.advance(.seconds(120))
    center.setDashboardVisible(true)
    await eventually { await client.fetchCount >= 3 }
    let activity = try #require(center.activities.first)
    #expect(activity.state == .failed)
    #expect(activity.message == "out of memory")
    #expect(activity.code == "gpu_memory_insufficient")

    center.acknowledgeFailure(model: model)
    #expect(center.activities.isEmpty)
    center.setDashboardVisible(false)
}

@Test @MainActor
func external_load_appears_only_while_retained() async throws {
    let client = FakeModelLoadClient()
    await client.setLoads([
        backendLoad("E1", model: "other/Model", state: .loading, stage: .readingWeights, trigger: .request),
    ])
    let sleeper = ManualSleeper()
    let center = makeCenter(client, sleeper: sleeper)

    center.setDashboardVisible(true)
    await eventually { center.activities.count == 1 }
    let external = try #require(center.activities.first)
    #expect(external.origin == .external)
    #expect(external.model == "other/Model")
    #expect(external.state == .loading)
    #expect(!external.autoOpenPanel)
    await eventually { await sleeper.requests.count == 1 }
    #expect(await sleeper.requests == [.milliseconds(500)])

    await client.setLoads([
        backendLoad("E1", model: "other/Model", state: .failed, stage: .readingWeights, trigger: .request, message: "boom"),
    ])
    await sleeper.releaseOne()
    await eventually { center.activities.first?.state == .failed }
    #expect(center.activities.first?.state == .failed)

    await client.setLoads([])
    await sleeper.releaseOne()
    await eventually { center.activities.isEmpty }
    #expect(center.activities.isEmpty)
    center.setDashboardVisible(false)
    await sleeper.releaseOne()
}

@Test @MainActor
func reload_cancel_reports_previous_version_kept() async throws {
    let client = FakeModelLoadClient(startScripts: [.accept(loadID: "R1")])
    await client.setLoads([backendLoad("R1", state: .loading, stage: .readingWeights, kind: .reload)])
    let center = makeCenter(client)
    let load = Task { @MainActor in
        try await center.loadModel(
            BackendLoadModelRequest(model: model, modelDir: "/models/tiny-v2", setDefault: true),
            origin: .dashboard,
            kind: .reload
        )
    }
    await eventually { center.activities.first?.stage == .readingWeights }
    #expect(center.activities.first?.kind == .reload)
    #expect(center.activities.first?.autoOpenPanel == false)

    await center.cancel(model: model)
    await client.setLoads([
        backendLoad("R1", state: .cancelled, stage: .readingWeights, kind: .reload, code: "load_cancelled"),
    ])

    do {
        _ = try await load.value
        Issue.record("Expected the reload to be cancelled")
    } catch let error as ModelLoadCancelledError {
        #expect(error.code == ModelLoadCancelledError.reloadCancelledCode)
        #expect(error.previousVersionKept)
    }
    let activity = try #require(center.activities.first)
    #expect(activity.state == .cancelled)
    #expect(activity.code == ModelLoadCancelledError.reloadCancelledCode)
}

/// A parameter reload (the Dashboard's settings save) ends and publishes its
/// result on the first poll that sees the backend finish, so the model row
/// never keeps the reload state after the backend is done.
@Test @MainActor
func parameter_reload_publishes_its_result_on_the_poll_that_sees_it_finish() async throws {
    let client = FakeModelLoadClient(startScripts: [.accept(loadID: "R1")])
    await client.setLoads([backendLoad("R1", state: .loading, stage: .readingWeights, kind: .reload)])
    let sleeper = ManualSleeper()
    let notifications = NotificationCenter()
    let center = makeCenter(client, sleeper: sleeper, notificationCenter: notifications)
    var published: [ModelLoadState] = []
    let observer = notifications.addObserver(
        forName: .ironMLXModelLoadsDidChange, object: center, queue: nil
    ) { _ in
        MainActor.assumeIsolated {
            if let state = center.activities.first?.state { published.append(state) }
        }
    }
    defer { notifications.removeObserver(observer) }
    let reload = Task { @MainActor in
        try await center.track(model: model, variant: .plain, origin: .dashboard, kind: .reload) { ticket in
            center.enterAppStage(ticket, .verifyingFiles)
            return try await center.loadThroughBackend(
                ticket,
                request: BackendLoadModelRequest(
                    model: model, modelDir: "/models/tiny", setDefault: true, reloadWhenIdle: true
                )
            )
        }
    }
    await eventually { center.activities.first?.stage == .readingWeights }
    let sleepsBefore = await sleeper.requests.count

    await client.setLoads([backendLoad("R1", state: .loaded, stage: .startingEngine, kind: .reload)])
    await sleeper.releaseOne()
    let response = try await reload.value

    #expect(response.status == "reloaded")
    #expect(center.activities.first?.state == .loaded)
    #expect(center.activities.first?.kind == .reload)
    #expect(published.last == .loaded)
    // No further poll interval had to pass.
    #expect(await sleeper.requests.count <= sleepsBefore + 1)
}

@Test @MainActor
func failure_in_an_app_stage_keeps_that_stage() async throws {
    let center = makeCenter(FakeModelLoadClient())
    await #expect(throws: ModelLoadFailedError.self) {
        try await center.track(model: model, variant: .plain, origin: .dashboard) { ticket in
            center.enterAppStage(ticket, .verifyingFiles)
            throw ModelLoadFailedError(message: "checksum mismatch", code: "model_snapshot_invalid")
        }
    }
    let activity = try #require(center.activities.first)
    #expect(activity.state == .failed)
    #expect(activity.stage == .verifyingFiles)
    #expect(activity.completedStages.isEmpty)
}

@Test @MainActor
func refused_start_reports_the_backend_stage_it_failed_in() async throws {
    let client = FakeModelLoadClient(startScripts: [
        .refuseAfterRegistering(code: "max_loaded_models_reached", stage: .preparingMemory),
    ])
    let center = makeCenter(client)
    await #expect(throws: (any Error).self) {
        try await center.track(model: model, variant: .plain, origin: .dashboard) { ticket in
            center.enterAppStage(ticket, .syncingModels)
            return try await center.loadThroughBackend(
                ticket,
                request: BackendLoadModelRequest(model: model, modelDir: "/models/tiny", setDefault: true)
            )
        }
    }
    let activity = try #require(center.activities.first)
    #expect(center.activities.count == 1, "the refused load is this load, not an external one")
    #expect(activity.state == .failed)
    #expect(activity.code == "max_loaded_models_reached")
    #expect(activity.stage == .preparingMemory)
    #expect(activity.completedStages.map(\.stage).first == .syncingModels)
}

/// Registers models like a backend that answers each registration slowly.
private actor SlowModelRegistrar: BackendModelLoading {
    private(set) var registered: [String] = []

    func updateASRSettings(model: String, settings: BackendASRSettings) async throws {}

    func waitUntilReady(timeout: TimeInterval) async throws {}

    func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool?,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults,
        audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        registered.append(model)
        // Ends with `CancellationError` once cancelled, as a URL request does.
        try await Task.sleep(for: .seconds(60))
        return BackendModelAdminResponse(
            success: true, status: "registered", code: nil, model: model,
            loadedModels: [], warningCode: nil, warning: nil, error: nil
        )
    }

    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        promptLookup: BackendPromptLookupConfig?,
        reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults,
        audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        throw CancellationError()
    }
}

@Test @MainActor
func cancel_while_syncing_models_ends_the_load_at_once() async throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-load-sync-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for name in ["Sync-A-4bit", "Sync-B-4bit", "Sync-C-4bit"] {
        _ = try writeVerifiedTestSnapshot(
            root: root,
            repoID: "mlx-community/\(name)",
            files: ["config.json": Data("{}".utf8), "model.safetensors": Data("weights".utf8)]
        )
    }
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let localModels = scanner.scan(loadedModels: [], pinnedModels: [], mtpEnabledModels: [])
    #expect(localModels.count == 3)
    let registrar = SlowModelRegistrar()
    let notifications = NotificationCenter()
    let center = makeCenter(FakeModelLoadClient(), notificationCenter: notifications)
    var stages: [ModelLoadStage] = []
    let observer = notifications.addObserver(
        forName: .ironMLXModelLoadsDidChange, object: center, queue: nil
    ) { _ in
        MainActor.assumeIsolated {
            if let stage = center.activities.first?.stage, stages.last != stage { stages.append(stage) }
        }
    }
    defer { notifications.removeObserver(observer) }

    let load = Task { @MainActor in
        try await center.track(model: model, variant: .plain, origin: .dashboard) { ticket in
            try await center.prepareBackend(
                ticket,
                startBackend: {},
                syncModels: {
                    await LocalModelBackendRegistrar.register(
                        localModels: localModels,
                        defaultModel: nil,
                        scanner: scanner,
                        parameterStore: parameterStore,
                        activeKvOffloadEnabled: false,
                        preservePinnedState: true,
                        client: registrar
                    )
                }
            )
            try Task.checkCancellation()
            return 0
        }
    }
    await eventually { await registrar.registered.count == 1 }
    #expect(stages == [.waitingForBackend, .syncingModels])

    await center.cancel(model: model)
    // The load ends right away; the registration under way does not hold it.
    #expect(center.activities.first?.state == .cancelled)
    #expect(center.activities.first?.code == ModelLoadCancelledError.loadCancelledCode)
    await #expect(throws: ModelLoadCancelledError.self) {
        _ = try await load.value
    }
    // No model after the cancelled registration is registered.
    #expect(await registrar.registered.count == 1)
}

// MARK: - DFlash2

@MainActor
private final class StartupBlockingBackend: BackendRuntimeManaging {
    var isRunning = true
    var state: BackendProcessState = .running
    var currentLaunchID: UUID? = UUID()
    var lastError: String?
    var lastEvent: BackendRuntimeEvent?
    var lastIncident: BackendIncidentRecord?
    private(set) var calls: [String] = []
    private(set) var configsAtRestart: [AppConfig] = []
    private let configStore: AppConfigStore

    init(configStore: AppConfigStore) {
        self.configStore = configStore
    }

    func ensureRunning() async throws {}

    func restart(intent: BackendStopIntent) async -> BackendRestartResult {
        calls.append("restart:\(intent.rawValue)")
        configsAtRestart.append(configStore.load())
        if calls.filter({ $0.hasPrefix("restart:") }).count == 1 {
            // A DFlash2 launch loading its weights until it is cancelled.
            try? await Task.sleep(for: .seconds(60))
            return BackendRestartResult(success: false, status: "restart_failed", port: 1, error: "cancelled")
        }
        return BackendRestartResult(success: true, status: "models_loaded", port: 1)
    }

    func stop(intent: BackendStopIntent) async {
        calls.append("stop:\(intent.rawValue)")
    }

    func stopForAppQuit() async {}

    func confirmLoadedModels(_ models: [BackendLoadedModelInfo], parameterConfirmedModelIDs: Set<String>) {}

    func refreshConfirmedSnapshot() {
        calls.append("refreshConfirmedSnapshot")
    }
}

private let dflash2TargetID = "mlx-community/Qwen3.8-27B-4bit"
private let dflash2DraftID = "z-lab/Qwen3.8-27B-DFlash2"

@Test @MainActor
func dflash2_cancel_from_dashboard_restores_previous_config() async throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-load-activity-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    let previousConfig = AppConfig(port: 1, defaultModel: "other/Model", loadedModels: ["other/Model"])
    configStore.save(previousConfig)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let previousParameters = parameterStore.parameters(for: dflash2TargetID)
    // What the Dashboard records before the switch starts.
    try parameterStore.recordDFlash2LoadPreference(modelID: dflash2TargetID, enabled: true, dflash2ModelID: dflash2DraftID)
    var dflash2Config = previousConfig
    dflash2Config.loadedModels = [dflash2TargetID]
    dflash2Config.defaultModel = dflash2TargetID
    configStore.save(dflash2Config)

    let backend = StartupBlockingBackend(configStore: configStore)
    // The DFlash2 launch reports its own load while it starts.
    let client = FakeModelLoadClient(loads: [
        backendLoad(
            "startup",
            model: dflash2TargetID,
            state: .loading,
            stage: .readingWeights,
            variant: .dflash2,
            trigger: .startup,
            bytesDone: 4,
            bytesTotal: 16
        ),
    ])
    let center = makeCenter(client)
    let flow = DFlash2ActivationFlow(
        backend: backend,
        configStore: configStore,
        parameterStore: parameterStore,
        loadActivity: center,
        verify: {}
    )
    let run = Task { @MainActor in
        try await flow.run(
            model: dflash2TargetID,
            kind: .load,
            previousConfig: previousConfig,
            previousParameters: previousParameters
        )
    }
    await eventually { center.activities.first?.stage == .readingWeights }
    let loading = try #require(center.activities.first)
    #expect(loading.variant == .dflash2)
    #expect(loading.origin == .dashboard)
    #expect(loading.completedStages.map(\.stage) == [.verifyingFiles, .stoppingBackend, .restartingBackend])
    #expect(loading.bytesDone == 4)
    #expect(loading.bytesTotal == 16)

    await center.cancel(model: dflash2TargetID)

    do {
        _ = try await run.value
        Issue.record("Expected the DFlash2 switch to be cancelled")
    } catch let error as ModelLoadCancelledError {
        #expect(error.code == ModelLoadCancelledError.dflash2RestoredCode)
    }
    // The starting server has no cancel endpoint; the App stops the launch.
    #expect(await client.cancelledLoadIDs.isEmpty)
    #expect(configStore.load() == previousConfig)
    #expect(parameterStore.parameters(for: dflash2TargetID) == previousParameters)
    #expect(backend.calls.filter { $0.hasPrefix("restart:") }.count == 2)
    #expect(backend.calls.contains("stop:plannedRestart"))
    // The second restart brings back the models loaded before the switch.
    #expect(backend.configsAtRestart.last?.restoredModelReferences == ["other/Model"])
    let activity = try #require(center.activities.first)
    #expect(activity.state == .cancelled)
    #expect(activity.code == ModelLoadCancelledError.dflash2RestoredCode)
}

private actor RecordingRestorer: BackendModelRestoring {
    private(set) var snapshots: [BackendRecoverySnapshot] = []

    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        snapshots.append(snapshot)
        return BackendRestartResult(
            success: true,
            status: snapshot.models.isEmpty ? "restarted" : "models_loaded",
            port: snapshot.config.port,
            loadedModels: snapshot.models.map(\.id)
        )
    }
}

private actor ReadinessCounter {
    private(set) var calls = 0

    func next() -> Int {
        calls += 1
        return calls
    }
}

@MainActor
private final class LaunchCounter {
    var value = 0
}

@Test @MainActor
func dflash2_cancel_during_startup_restore_restarts_plain_without_models() async throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-load-activity-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try writeVerifiedTestSnapshot(
        root: root,
        repoID: dflash2TargetID,
        files: [
            "config.json": Data(loadActivityDFlash2TargetConfig.utf8),
            "model.safetensors": Data("target-weights".utf8),
        ]
    )
    try writeVerifiedTestSnapshot(
        root: root,
        repoID: dflash2DraftID,
        files: [
            "config.json": Data(loadActivityDFlash2DraftConfig.utf8),
            "model.safetensors": Data("draft-weights".utf8),
        ]
    )
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig(port: 1, defaultModel: dflash2TargetID, loadedModels: [dflash2TargetID]))
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try parameterStore.save(ModelParameters(modelID: dflash2TargetID, dflash2Enabled: true, dflash2ModelID: dflash2DraftID))
    let scanner = LocalModelScanner(rootURL: root)

    let launches = LaunchCounter()
    let notificationCenter = NotificationCenter()
    let processManager = BackendProcessManager(
        configStore: configStore,
        terminator: BackendProcessTerminator(sleep: { _ in }),
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        notificationCenter: notificationCenter,
        launchPlanProvider: {
            launches.value += 1
            return BackendProcessLaunchPlan(processURL: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"])
        }
    )
    let restorer = RecordingRestorer()
    let readiness = ReadinessCounter()
    let center = makeCenter(FakeModelLoadClient())
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: scanner,
        parameterStore: parameterStore,
        restartCoordinator: restorer,
        incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
        notificationCenter: notificationCenter,
        readinessWaiter: { _, _ in
            if await readiness.next() == 1 {
                // The DFlash2 launch is still reading weights.
                try await Task.sleep(for: .seconds(60))
            }
        },
        loadActivity: center
    )

    let start = Task { @MainActor in try await supervisor.ensureRunning() }
    await eventually { center.activities.first?.stage == .restartingBackend }
    let restoring = try #require(center.activities.first)
    #expect(restoring.model == dflash2TargetID)
    #expect(restoring.variant == .dflash2)
    #expect(restoring.origin == .restore)
    #expect(restoring.cancellable)

    await center.cancel(model: dflash2TargetID)
    try await start.value

    #expect(launches.value == 2)
    let snapshots = await restorer.snapshots
    #expect(snapshots.count == 1)
    #expect(snapshots.first?.models.isEmpty == true)
    #expect(configStore.load().restoredModelReferences.isEmpty)
    #expect(configStore.load().defaultModelReference == nil)
    // The DFlash2 preference stays for the next manual load.
    #expect(parameterStore.parameters(for: dflash2TargetID)?.dflash2Enabled == true)
    #expect(parameterStore.parameters(for: dflash2TargetID)?.dflash2ModelID == dflash2DraftID)
    #expect(supervisor.state == .running)
    let activity = try #require(center.activities.first)
    #expect(activity.state == .cancelled)
    #expect(activity.code == ModelLoadCancelledError.dflash2StartupCode)

    await supervisor.stop(intent: .userStop)
}

private let loadActivityDFlash2TargetConfig = """
{
  "model_type": "qwen3_5",
  "quantization": {
    "mode": "affine",
    "bits": 4,
    "group_size": 64
  },
  "text_config": {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 0.000001,
    "rope_parameters": {
      "rope_type": "default",
      "rope_theta": 10000000
    }
  }
}
"""

private let loadActivityDFlash2DraftConfig = """
{
  "architectures": ["DFlash2DraftModel"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_act": "silu",
  "attention_bias": false,
  "is_causal": false,
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "vocab_size": 248320,
  "max_position_embeddings": 262144,
  "head_dim": 128,
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "rms_norm_eps": 0.000001,
  "rope_parameters": {
    "rope_type": "default",
    "rope_theta": 10000000
  },
  "sliding_window": 2048,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61]
  }
}
"""
