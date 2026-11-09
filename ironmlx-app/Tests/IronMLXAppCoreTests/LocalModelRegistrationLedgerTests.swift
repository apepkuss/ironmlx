import Foundation
import Testing

@testable import IronMLXAppCore

private let modelA = "mlx-community/Ledger-A-4bit"
private let modelB = "mlx-community/Ledger-B-4bit"
private let modelC = "mlx-community/Ledger-C-4bit"

private struct LedgerFixture {
    let root: URL
    let scanner: LocalModelScanner
    let parameterStore: ModelParameterStore
    let client = CountingRegistrationClient()
    let ledger = LocalModelRegistrationLedger()

    init(config: String = "{}") throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ironmlx-registration-ledger-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for repoID in [modelA, modelB, modelC] {
            try writeVerifiedTestSnapshot(
                root: root,
                repoID: repoID,
                files: ["config.json": Data(config.utf8), "model.safetensors": Data("weights".utf8)]
            )
        }
        scanner = LocalModelScanner(rootURL: root)
        parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    }

    var models: [LocalModel] {
        scanner.scan(loadedModels: [], pinnedModels: [], mtpEnabledModels: [])
    }

    /// What a backend start sends: every model, with its pinned state.
    func registerAll(launchID: UUID, defaultModel: String? = nil) async -> [BackendModelRecoveryFailure] {
        await LocalModelBackendRegistrar.register(
            localModels: models, defaultModel: defaultModel, scanner: scanner,
            parameterStore: parameterStore, activeKvOffloadEnabled: false,
            client: client, ledger: ledger, launchID: launchID, onlyChanged: false
        )
    }

    /// What a Dashboard rescan sends.
    func rescan(
        launchID: UUID,
        defaultModel: String? = nil,
        activeKvOffload: Bool = false
    ) async -> [BackendModelRecoveryFailure] {
        await LocalModelBackendRegistrar.register(
            localModels: models, defaultModel: defaultModel, scanner: scanner,
            parameterStore: parameterStore, activeKvOffloadEnabled: activeKvOffload,
            preservePinnedState: true,
            client: client, ledger: ledger, launchID: launchID, onlyChanged: true,
            pass: .rescan
        )
    }

    /// The models whose registration was computed since `before`.
    func computed(since before: [String: Int]) -> [String] {
        ledger.computationCounts
            .filter { $0.value > before[$0.key, default: 0] }
            .map(\.key)
            .sorted()
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}

@Test @MainActor func dashboardLoadRegistersOnlyTheModelBeingLoaded() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    #expect(fixture.models.count == 3)

    let failures = await LocalModelBackendRegistrar.registerLoadTarget(
        modelB,
        config: AppConfig(port: 1, defaultModel: modelB, pinnedModels: [modelB]),
        scanner: fixture.scanner,
        parameterStore: fixture.parameterStore,
        client: fixture.client,
        ledger: fixture.ledger,
        launchID: UUID()
    )

    #expect(failures.isEmpty)
    let registered = await fixture.client.registered
    #expect(registered.map(\.model) == [modelB])
    // The default flag is sent as before; the pinned state is left to the backend.
    #expect(registered.first?.setDefault == true)
    #expect(registered.first?.pinned == nil)
}

@Test @MainActor func rescanRegistersOnlyModelsWhoseRegistrationChanged() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()

    #expect(await fixture.registerAll(launchID: launch, defaultModel: modelA).isEmpty)
    #expect(await fixture.client.registered.count == 3)

    // Nothing changed since the backend start.
    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelA).isEmpty)
    #expect(await fixture.client.registered.isEmpty)

    // A new context length changes the max cache cap of one model.
    try fixture.parameterStore.save(ModelParameters(modelID: modelB, maxTokens: "4096"))
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelA).isEmpty)
    #expect(await fixture.client.registered.map(\.model) == [modelB])

    // A new sampling default changes one model.
    await fixture.client.reset()
    try fixture.parameterStore.save(ModelParameters(modelID: modelC, temperature: "0.3"))
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelA).isEmpty)
    #expect(await fixture.client.registered.map(\.model) == [modelC])

    // A new default model changes the flag of the old and the new default.
    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelB).isEmpty)
    let registered = await fixture.client.registered
    #expect(registered.map(\.model) == [modelA, modelB])
    #expect(registered.map(\.setDefault) == [false, true])

    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelB).isEmpty)
    #expect(await fixture.client.registered.isEmpty)
}

@Test @MainActor func newBackendLaunchRegistersEverythingAgain() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let first = UUID()
    #expect(await fixture.registerAll(launchID: first).isEmpty)
    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: first).isEmpty)
    #expect(await fixture.client.registered.isEmpty)

    #expect(await fixture.rescan(launchID: UUID()).isEmpty)
    #expect(await fixture.client.registered.map(\.model) == [modelA, modelB, modelC])
}

@Test @MainActor func failedRegistrationIsRetriedOnTheNextRescan() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()
    await fixture.client.fail(modelB)

    let failures = await fixture.registerAll(launchID: launch)
    #expect(failures.map(\.model) == [modelB])

    await fixture.client.reset()
    await fixture.client.recover(modelB)
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(await fixture.client.registered.map(\.model) == [modelB])

    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(await fixture.client.registered.isEmpty)
}

@Test @MainActor func backendRestoreSeedsTheRecordForTheNextRescan() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()
    let client = fixture.client
    let coordinator = BackendRestartCoordinator(
        scanner: fixture.scanner,
        parameterStore: fixture.parameterStore,
        clientFactory: { _, _ in client },
        registrationLedger: fixture.ledger
    )
    let config = AppConfig(port: 1, defaultModel: modelA)
    let snapshot = BackendRecoverySnapshot.capture(
        config: config,
        scanner: fixture.scanner,
        parameterStore: fixture.parameterStore
    )

    _ = await coordinator.restore(snapshot, launchID: launch)
    #expect(await client.registered.map(\.model) == [modelA, modelB, modelC])

    await client.reset()
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelA).isEmpty)
    #expect(await client.registered.isEmpty)
}

@Test @MainActor func restoreThenDashboardRescanWithTheAppConfigSendsNothing() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()
    try fixture.parameterStore.save(ModelParameters(modelID: modelA, maxTokens: "8192", temperature: "0.6"))
    try fixture.parameterStore.save(ModelParameters(modelID: modelC, topP: "0.9"))
    // A default and a pinned model, as the App config holds them.
    let config = AppConfig(port: 1, defaultModel: modelA, pinnedModels: [modelC])
    let coordinator = BackendRestartCoordinator(
        scanner: fixture.scanner,
        parameterStore: fixture.parameterStore,
        clientFactory: { [client = fixture.client] _, _ in client },
        registrationLedger: fixture.ledger
    )
    _ = await coordinator.restore(
        BackendRecoverySnapshot.capture(config: config, scanner: fixture.scanner, parameterStore: fixture.parameterStore),
        launchID: launch
    )
    #expect(await fixture.client.registered.count == 3)
    await fixture.client.reset()

    // What `DashboardBridge.sendScannedModels` scans and registers.
    let scanned = fixture.scanner.scan(
        loadedModels: [],
        pinnedModels: Set(config.pinnedModelReferences),
        mtpEnabledModels: [],
        dflash2EnabledModels: []
    )
    await LocalModelBackendRegistrar.register(
        localModels: scanned,
        defaultModel: config.defaultModelReference,
        scanner: fixture.scanner,
        parameterStore: fixture.parameterStore,
        activeKvOffloadEnabled: config.activeKvOffload == true,
        preservePinnedState: true,
        client: fixture.client,
        ledger: fixture.ledger,
        launchID: launch,
        onlyChanged: true,
        pass: .rescan
    )
    #expect(await fixture.client.registered.isEmpty)
}

@Test @MainActor func overlappingRescansEndWithTheLaterRegistration() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    await fixture.client.reset()

    try fixture.parameterStore.save(ModelParameters(modelID: modelA, maxTokens: "2048"))
    await fixture.client.holdNextRegistration()
    let first = Task { @MainActor in await fixture.rescan(launchID: launch) }
    await eventually { await fixture.client.isHolding }
    // The parameters change while the first pass is sending its registration.
    try fixture.parameterStore.save(ModelParameters(modelID: modelA, maxTokens: "4096"))
    let second = Task { @MainActor in await fixture.rescan(launchID: launch) }
    await eventually {
        if fixture.ledger.waitingRegistrations == 1 { return true }
        return await !fixture.client.registered.isEmpty
    }
    await fixture.client.releaseHeldRegistration()
    #expect(await first.value.isEmpty)
    #expect(await second.value.isEmpty)

    // The backend ends with the later registration ...
    let registered = await fixture.client.registered
    #expect(registered.map(\.model) == [modelA, modelA])
    #expect(registered.map(\.maxCacheCap) == [2048, 4096])
    // ... and the record says so: nothing is left to send.
    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(await fixture.client.registered.isEmpty)
}

@Test @MainActor func loadTargetRegistrationDoesNotWaitForABackgroundPass() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()

    // A background pass over every model, held while it sends the first one.
    await fixture.client.holdNextRegistration()
    let background = Task { @MainActor in await fixture.rescan(launchID: launch) }
    await eventually { await fixture.client.isHolding }

    let loadDone = Flag()
    let load = Task { @MainActor in
        let failures = await LocalModelBackendRegistrar.registerLoadTarget(
            modelB,
            config: AppConfig(port: 1),
            scanner: fixture.scanner,
            parameterStore: fixture.parameterStore,
            client: fixture.client,
            ledger: fixture.ledger,
            launchID: launch
        )
        loadDone.set()
        return failures
    }
    await eventually(.seconds(2)) { loadDone.value }
    #expect(loadDone.value, "the load-target registration finished while the pass was still sending")
    #expect(await fixture.client.registered.map(\.model) == [modelB])

    await fixture.client.releaseHeldRegistration()
    #expect(await load.value.isEmpty)
    #expect(await background.value.isEmpty)
    // The pass skips the model the load already registered.
    #expect(await fixture.client.registered.map(\.model) == [modelB, modelA, modelC])
}

@MainActor
private final class Flag {
    private(set) var value = false
    func set() { value = true }
}

@Test @MainActor func appLoadMakesTheNextRescanRegisterThatModelAgain() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(await fixture.client.registered.isEmpty)

    // A one-off load setting the saved parameters do not have.
    let center = ModelLoadActivityCenter(
        client: { CompletingLoadClient() },
        notificationCenter: NotificationCenter(),
        registrationLedger: fixture.ledger
    )
    let modelDir = try #require(fixture.scanner.resolveModelPath(for: modelB))
    _ = try await center.loadModel(
        BackendLoadModelRequest(
            model: modelB,
            modelDir: modelDir,
            setDefault: false,
            maxCacheCap: 1024,
            pinned: false,
            reloadWhenIdle: false
        ),
        origin: .dashboard
    )

    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(await fixture.client.registered.map(\.model) == [modelB])
}

@Test @MainActor func forgottenModelIsRegisteredAgainOnTheNextRescan() async throws {
    let fixture = try LedgerFixture()
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    await fixture.client.reset()

    fixture.ledger.forget(model: modelC)

    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(await fixture.client.registered.map(\.model) == [modelC])
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

/// Answers every load start with a finished load.
private struct CompletingLoadClient: BackendModelLoadTracking {
    func fetchModelLoads() async throws -> [BackendLoadActivity] { [] }
    func cancelModelLoad(loadID: String) async throws -> BackendLoadCancelResult { .notFound }
    func startModelLoad(_ request: BackendLoadModelRequest) async throws -> BackendModelLoadStart {
        .completed(BackendModelAdminResponse(
            success: true, status: "loaded", code: nil, model: request.model,
            loadedModels: [], warningCode: nil, warning: nil, error: nil
        ))
    }
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] { [] }
}

private struct RegistrationCall: Equatable {
    var model: String
    var setDefault: Bool
    var pinned: Bool?
    var maxCacheCap: Int?
    var modelDir = ""
    var mtpModelDir: String?
    var audio: BackendAudioResources?
}

private actor CountingRegistrationClient: BackendModelLoading {
    private(set) var registered: [RegistrationCall] = []
    private var failing: Set<String> = []
    private var holdNext = false
    private var held: CheckedContinuation<Void, Never>?

    func reset() { registered = [] }
    func fail(_ model: String) { failing.insert(model) }
    func recover(_ model: String) { failing.remove(model) }
    /// The next registration reaches the backend only after `releaseHeldRegistration()`.
    func holdNextRegistration() { holdNext = true }
    var isHolding: Bool { held != nil }
    func releaseHeldRegistration() {
        held?.resume()
        held = nil
    }

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
        if holdNext {
            holdNext = false
            await withCheckedContinuation { held = $0 }
        }
        registered.append(RegistrationCall(
            model: model, setDefault: setDefault, pinned: pinned, maxCacheCap: maxCacheCap,
            modelDir: modelDir, mtpModelDir: mtpModelDir, audio: audio
        ))
        if failing.contains(model) {
            throw BackendAPIError.serverResponse(statusCode: 503, body: #"{"error":"busy","code":"backend_unavailable"}"#)
        }
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

private let contextConfig = #"{"model_type":"llama","max_position_embeddings":262144}"#

@Test @MainActor func unchangedRescanComputesNoRegistration() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    #expect(fixture.computed(since: [:]) == [modelA, modelB, modelC])

    var before = fixture.ledger.computationCounts
    await fixture.client.reset()
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(fixture.computed(since: before) == [])
    #expect(await fixture.client.registered.isEmpty)

    before = fixture.ledger.computationCounts
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(fixture.computed(since: before) == [])

    // A new backend launch sends every model again, from the cached values.
    #expect(await fixture.rescan(launchID: UUID()).isEmpty)
    #expect(fixture.computed(since: before) == [])
    #expect(await fixture.client.registered.map(\.model) == [modelA, modelB, modelC])
}

@Test @MainActor func changedInputsRecomputeOnlyTheAffectedModels() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch, defaultModel: modelA).isEmpty)
    #expect(await fixture.rescan(launchID: launch, defaultModel: modelA).isEmpty)

    func step(
        defaultModel: String = modelA,
        activeKvOffload: Bool = false
    ) async -> (computed: [String], sent: [RegistrationCall]) {
        let before = fixture.ledger.computationCounts
        await fixture.client.reset()
        #expect(await fixture.rescan(
            launchID: launch,
            defaultModel: defaultModel,
            activeKvOffload: activeKvOffload
        ).isEmpty)
        return (fixture.computed(since: before), await fixture.client.registered)
    }

    // Saved parameters of one model.
    try fixture.parameterStore.save(ModelParameters(modelID: modelB, maxTokens: "4096"))
    var result = await step()
    #expect(result.computed == [modelB])
    #expect(result.sent.map(\.model) == [modelB])
    #expect(result.sent.map(\.maxCacheCap) == [4096])

    // A file of one model is touched: recomputed, but its registration is the same.
    let snapshotC = try #require(fixture.scanner.resolveModelPath(for: modelC))
    try FileManager.default.setAttributes(
        [.modificationDate: Date(timeIntervalSinceNow: 60)],
        ofItemAtPath: URL(fileURLWithPath: snapshotC).appendingPathComponent("config.json").path
    )
    result = await step()
    #expect(result.computed == [modelC])
    #expect(result.sent.isEmpty)

    // A file of one model changes size and content.
    try Data(#"{"model_type":"llama","max_position_embeddings":16384}"#.utf8)
        .write(to: URL(fileURLWithPath: snapshotC).appendingPathComponent("config.json"))
    try refreshTestSnapshotManifest(at: URL(fileURLWithPath: snapshotC))
    result = await step()
    #expect(result.computed == [modelC])
    #expect(result.sent.map(\.model) == [modelC])
    #expect(result.sent.map(\.maxCacheCap) == [16384])

    // The default model moves from A to B.
    result = await step(defaultModel: modelB)
    #expect(result.computed == [modelA, modelB])
    #expect(result.sent.map(\.model) == [modelA, modelB])
    #expect(result.sent.map(\.setDefault) == [false, true])

    // Active KV offload changes the default cap of models without a saved one.
    result = await step(defaultModel: modelB, activeKvOffload: true)
    #expect(result.computed == [modelA, modelC])
    #expect(result.sent.map(\.model) == [modelA])
    #expect(result.sent.map(\.maxCacheCap) == [262144])

    result = await step(defaultModel: modelB, activeKvOffload: true)
    #expect(result.computed == [])
    #expect(result.sent.isEmpty)
}

@Test @MainActor func loadTargetRegistrationIgnoresTheCacheForItsModel() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    #expect(await fixture.rescan(launchID: launch).isEmpty)

    let before = fixture.ledger.computationCounts
    await fixture.client.reset()
    let failures = await LocalModelBackendRegistrar.registerLoadTarget(
        modelB,
        config: AppConfig(port: 1),
        scanner: fixture.scanner,
        parameterStore: fixture.parameterStore,
        client: fixture.client,
        ledger: fixture.ledger,
        launchID: launch
    )
    #expect(failures.isEmpty)
    #expect(fixture.computed(since: before) == [modelB])
}

@Test @MainActor func overlappingRescanRequestsRunOnePassAndOneRerun() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    await fixture.client.reset()

    func request() async {
        await LocalModelBackendRegistrar.rescan(
            localModels: fixture.models,
            defaultModel: nil,
            scanner: fixture.scanner,
            parameterStore: fixture.parameterStore,
            activeKvOffloadEnabled: false,
            client: fixture.client,
            ledger: fixture.ledger,
            launchID: launch
        )
    }

    try fixture.parameterStore.save(ModelParameters(modelID: modelA, maxTokens: "2048"))
    await fixture.client.holdNextRegistration()
    let first = Task { @MainActor in await request() }
    await eventually { await fixture.client.isHolding }

    // Three more requests while the first pass is sending; each returns at once.
    for maxTokens in ["3072", "4096", "8192"] {
        try fixture.parameterStore.save(ModelParameters(modelID: modelA, maxTokens: maxTokens))
        await request()
    }
    #expect(fixture.ledger.rescanPassesRun == 1)

    await fixture.client.releaseHeldRegistration()
    await first.value
    // One pass, then one rerun that sends the latest registration.
    #expect(fixture.ledger.rescanPassesRun == 2)
    let registered = await fixture.client.registered
    #expect(registered.map(\.model) == [modelA, modelA])
    #expect(registered.map(\.maxCacheCap) == [2048, 8192])
}

private func qwen35Config(modelType: String, mtpLayers: Int) -> String {
    """
    {"model_type": "\(modelType)", "text_config": {"hidden_size": 2560, "intermediate_size": 9728,
     "num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
     "vocab_size": 151936, "rms_norm_eps": 0.000001, "attention_bias": false, "tie_word_embeddings": false,
     "full_attention_interval": 4, "linear_num_value_heads": 16, "linear_num_key_heads": 16,
     "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
     "mtp_num_hidden_layers": \(mtpLayers), "max_position_embeddings": 262144}}
    """
}

@Test @MainActor func registrationPassScansTheModelStoreAtMostOnce() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let base = "mlx-community/Qwen3.5-4B-MLX-4bit"
    let draft = "mlx-community/Qwen3.5-4B-MTP-4bit"
    for (repoID, config) in [
        (base, qwen35Config(modelType: "qwen3_5", mtpLayers: 0)),
        (draft, qwen35Config(modelType: "qwen3_5_mtp", mtpLayers: 1)),
    ] {
        try writeVerifiedTestSnapshot(
            root: fixture.root,
            repoID: repoID,
            files: ["config.json": Data(config.utf8), "model.safetensors": Data("weights".utf8)]
        )
    }
    // MTP on, no saved draft: the draft is the first compatible candidate.
    try fixture.parameterStore.save(ModelParameters(modelID: base, mtpEnabled: true))
    let counter = FullScanCounter()
    var scanner = fixture.scanner
    scanner.onFullScan = { counter.increment() }
    let launch = UUID()

    // A backend start: one scan, then registering every model scans nothing more.
    let models = scanner.scan(loadedModels: [], pinnedModels: [], mtpEnabledModels: [])
    #expect(counter.value == 1)
    counter.reset()
    let failures = await LocalModelBackendRegistrar.register(
        localModels: models, defaultModel: nil, scanner: scanner,
        parameterStore: fixture.parameterStore, activeKvOffloadEnabled: false,
        client: fixture.client, ledger: fixture.ledger, launchID: launch, pass: .restore
    )
    #expect(failures.isEmpty)
    #expect(counter.value == 0)
    let restored = await fixture.client.registered
    let restoredBase = try #require(restored.first { $0.model == base })
    #expect(restoredBase.mtpModelDir == scanner.resolveModelPath(for: draft))

    // Loads register their target without scanning every model.
    await fixture.client.reset()
    counter.reset()
    for target in [modelA, base] {
        #expect(await LocalModelBackendRegistrar.registerLoadTarget(
            target, config: AppConfig(port: 1), scanner: scanner,
            parameterStore: fixture.parameterStore, client: fixture.client,
            ledger: fixture.ledger, launchID: launch
        ).isEmpty)
        if target == modelA {
            #expect(counter.value == 0)
        }
    }
    #expect(counter.value <= 1)
    // ... and send what the backend start sent, the pinned state aside.
    var expected = restored.filter { [modelA, base].contains($0.model) }
    for index in expected.indices {
        expected[index].pinned = nil
    }
    #expect(await fixture.client.registered == expected)
}

@Test @MainActor func inPlaceRewriteKeepingSizeAndModificationTimeRecomputesTheModel() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    #expect(await fixture.rescan(launchID: launch).isEmpty)

    // Rewrite C's config in place (same inode, same size) and put its
    // modification time back to the nanosecond.
    let config = URL(fileURLWithPath: try #require(fixture.scanner.resolveModelPath(for: modelC)))
        .appendingPathComponent("config.json")
    let kept = try rewriteInPlaceKeepingModificationTime(config)
    #expect(kept)

    let counts = fixture.ledger.computationCounts
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(fixture.computed(since: counts) == [modelC])
}

@Test @MainActor func rescanForgetsComputedRegistrationsOfModelsThatAreGone() async throws {
    let fixture = try LedgerFixture(config: contextConfig)
    defer { fixture.remove() }
    let launch = UUID()
    #expect(await fixture.registerAll(launchID: launch).isEmpty)
    #expect(await fixture.rescan(launchID: launch).isEmpty)
    #expect(fixture.ledger.computedRegistrationModels == [modelA, modelB, modelC])

    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: fixture.root,
        provider: .huggingFace,
        repoID: modelC
    )
    try FileManager.default.removeItem(at: repository)
    #expect(fixture.models.map(\.id) == [modelA, modelB])
    #expect(await fixture.rescan(launchID: launch).isEmpty)

    #expect(fixture.ledger.computedRegistrationModels == [modelA, modelB])
    #expect(Set(fixture.ledger.computationCounts.keys) == [modelA, modelB])
}

/// Writes `url`'s own contents back over it in place, then restores its
/// access and modification times exactly. Returns whether inode, size and
/// modification time are unchanged afterwards.
private func rewriteInPlaceKeepingModificationTime(_ url: URL) throws -> Bool {
    var before = stat()
    guard stat(url.path, &before) == 0 else { return false }
    let contents = try Data(contentsOf: url)
    let handle = try FileHandle(forWritingTo: url)
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: contents)
    try handle.close()
    let times = [before.st_atimespec, before.st_mtimespec]
    let restored = times.withUnsafeBufferPointer {
        utimensat(AT_FDCWD, url.path, $0.baseAddress, 0) == 0
    }
    var after = stat()
    guard restored, stat(url.path, &after) == 0 else { return false }
    return after.st_ino == before.st_ino
        && after.st_size == before.st_size
        && after.st_mtimespec.tv_sec == before.st_mtimespec.tv_sec
        && after.st_mtimespec.tv_nsec == before.st_mtimespec.tv_nsec
}
