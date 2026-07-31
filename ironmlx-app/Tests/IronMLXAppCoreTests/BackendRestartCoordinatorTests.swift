import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor func restartDefaultModelRegistersUnloadedDefaultWithoutLoading() async throws {
    let (root, snapshot) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let loader = FakeRestartModelLoader()
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )

    let recoverySnapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(port: 9068, defaultModel: "mlx-community/Tiny-4bit"),
        scanner: scanner,
        parameterStore: parameterStore
    )
    let result = await coordinator.restore(recoverySnapshot)

    #expect(result.success)
    #expect(!result.modelLoaded)
    #expect(result.model == "mlx-community/Tiny-4bit")
    #expect(result.loadedModels == [])
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "register:mlx-community/Tiny-4bit:\(snapshot.path):true:nil:false:nil:nil",
    ], "\(loaderCalls)")
}

@Test @MainActor func restartDefaultModelReportsLoadFailure() async throws {
    let (root, _) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let loader = FakeRestartModelLoader(
        loadError: BackendAPIError.serverResponse(
            statusCode: 400,
            body: #"{"success":false,"status":"error","error":"memory budget exceeded"}"#
        )
    )
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )

    let recoverySnapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Tiny-4bit",
            loadedModels: ["mlx-community/Tiny-4bit"]
        ),
        scanner: scanner,
        parameterStore: parameterStore
    )
    let result = await coordinator.restore(recoverySnapshot)

    #expect(!result.success)
    #expect(!result.modelLoaded)
    #expect(result.model == "mlx-community/Tiny-4bit")
    #expect(result.error == "memory budget exceeded")
    #expect(result.failedModels == ["mlx-community/Tiny-4bit"])
    #expect(await loader.calls.count == 2)
}

@Test @MainActor func restartDefaultModelPreservesBackendErrorCode() async throws {
    let (root, _) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let loader = FakeRestartModelLoader(
        loadError: BackendAPIError.serverResponse(
            statusCode: 503,
            body: #"{"success":false,"status":"error","code":"max_loaded_models_reached","error":"Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model."}"#
        )
    )
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )

    let recoverySnapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Tiny-4bit",
            loadedModels: ["mlx-community/Tiny-4bit"]
        ),
        scanner: scanner,
        parameterStore: parameterStore
    )
    let result = await coordinator.restore(recoverySnapshot)

    #expect(result.errorCode == "max_loaded_models_reached")
    #expect(result.error == "Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model.")
    #expect(result.failures.count == 1)
    #expect(result.failures.first?.stage == .loading)
    #expect(result.failures.first?.retryable == false)
    #expect(result.failures.first?.action == .unloadOtherModels)
}

@Test func recoveryFailureClassifierMapsMemoryErrorsToUsefulActions() {
    let gpuFailure = BackendRecoveryFailureClassifier.failure(
        model: "mlx-community/Tiny-4bit",
        stage: .loading,
        error: BackendAPIError.serverResponse(
            statusCode: 503,
            body: #"{"code":"gpu_memory_insufficient","error":"Not enough GPU memory."}"#
        )
    )
    let kvFailure = BackendRecoveryFailureClassifier.failure(
        model: "mlx-community/Tiny-4bit",
        stage: .loading,
        error: BackendAPIError.serverResponse(
            statusCode: 400,
            body: #"{"code":"kv_memory_budget_exceeded","error":"KV cache budget exceeded."}"#
        )
    )

    #expect(gpuFailure.retryable)
    #expect(gpuFailure.action == .unloadOtherModels)
    #expect(!kvFailure.retryable)
    #expect(kvFailure.action == .reviewRuntimeSettings)
}

@Test @MainActor func restartDefaultModelRestoresMultipleLoadedModels() async throws {
    let (root, firstSnapshot) = try restartModelRoot(repoID: "mlx-community/First-4bit")
    let (_, secondSnapshot) = try restartModelRoot(repoID: "mlx-community/Second-4bit", root: root)
    let loader = FakeRestartModelLoader()
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )

    let recoverySnapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Second-4bit",
            loadedModels: [
                "mlx-community/First-4bit",
                "mlx-community/Second-4bit",
            ]
        ),
        scanner: scanner,
        parameterStore: parameterStore
    )
    let result = await coordinator.restore(recoverySnapshot)

    #expect(result.success)
    #expect(result.modelLoaded)
    #expect(result.model == "mlx-community/Second-4bit")
    #expect(result.loadedModels == [
        "mlx-community/Second-4bit",
        "mlx-community/First-4bit",
    ])
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "register:mlx-community/First-4bit:\(firstSnapshot.path):false:nil:false:nil:nil",
        "register:mlx-community/Second-4bit:\(secondSnapshot.path):true:nil:false:nil:nil",
        "load:mlx-community/Second-4bit:\(secondSnapshot.path):true:nil:false:nil:nil",
        "load:mlx-community/First-4bit:\(firstSnapshot.path):false:nil:false:nil:nil",
    ], "\(loaderCalls)")
}

@Test @MainActor func restartDefaultModelRegistersLocalModelsWithoutLoadingWhenNoRestoredModel() async throws {
    let (root, snapshot) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let loader = FakeRestartModelLoader()
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )

    let recoverySnapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(port: 9068),
        scanner: scanner,
        parameterStore: parameterStore
    )
    let result = await coordinator.restore(recoverySnapshot)

    #expect(result.success)
    #expect(!result.modelLoaded)
    #expect(result.loadedModels.isEmpty)
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "register:mlx-community/Tiny-4bit:\(snapshot.path):false:nil:false:nil:nil",
    ], "\(loaderCalls)")
}

@Test @MainActor func restartUsesFirstConfirmedModelAsDefaultWhenNoPreferenceExists() async throws {
    let (root, snapshot) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let loader = FakeRestartModelLoader()
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(
        url: root.appendingPathComponent("model_params.json")
    )
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )
    let recoverySnapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(
            port: 9068,
            loadedModels: ["mlx-community/Tiny-4bit"]
        ),
        scanner: scanner,
        parameterStore: parameterStore
    )

    let result = await coordinator.restore(recoverySnapshot)

    #expect(result.success)
    #expect(result.model == "mlx-community/Tiny-4bit")
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "register:mlx-community/Tiny-4bit:\(snapshot.path):false:nil:false:nil:nil",
        "load:mlx-community/Tiny-4bit:\(snapshot.path):true:nil:false:nil:nil",
    ], "\(loaderCalls)")
}

@Test @MainActor
func registrationFailureForModelOutsideRecoverySnapshotDoesNotPolluteRecovery() async throws {
    let (root, _) = try restartModelRoot(repoID: "mlx-community/Default-4bit")
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "mlx-community/DiffusionGemma-MXFP4",
        files: [
            "config.json": Data(
                #"{"model_type":"diffusion_gemma","vision_config":{"hidden_size":1152},"quantization":{"mode":"mxfp4","bits":4,"group_size":32}}"#.utf8
            ),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    let registrationError = BackendAPIError.serverResponse(
        statusCode: 400,
        body: #"{"success":false,"status":"error","code":"unknown_backend_error","error":"registration failed"}"#
    )
    let loader = FakeRestartModelLoader(
        registrationErrors: ["mlx-community/DiffusionGemma-MXFP4": registrationError]
    )
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let coordinator = BackendRestartCoordinator(
        scanner: scanner,
        parameterStore: parameterStore,
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restore(
        BackendRecoverySnapshot.capture(
            config: AppConfig(
                port: 9068,
                defaultModel: "mlx-community/Default-4bit"
            ),
            scanner: scanner,
            parameterStore: parameterStore
        )
    )

    #expect(result.success)
    #expect(result.failedModels.isEmpty)
    #expect(result.failures.isEmpty)
}

@Test @MainActor
func diffusionGemmaRecoverySnapshotOmitsCausalOnlySettings() throws {
    let modelID = "mlx-community/DiffusionGemma-MXFP4"
    let root = try restartTemporaryDirectory()
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: modelID,
        files: [
            "config.json": Data(
                #"{"model_type":"diffusion_gemma","vision_config":{"hidden_size":1152},"quantization":{"mode":"mxfp4","bits":4,"group_size":32}}"#.utf8
            ),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try parameterStore.save(
        ModelParameters(
            modelID: modelID,
            maxTokens: "4096",
            temperature: "0.7",
            topP: "0.8",
            topK: "40",
            repeatPenalty: "1.1",
            mtpEnabled: true,
            promptLookupEnabled: true
        )
    )

    let snapshot = BackendRecoverySnapshot.capture(
        config: AppConfig(
            port: 9068,
            loadedModels: [modelID],
            pinnedModels: [modelID]
        ),
        scanner: scanner,
        parameterStore: parameterStore
    )
    let model = try #require(snapshot.models.first)

    #expect(model.maxCacheCap == nil)
    #expect(model.mtpModelDir == nil)
    #expect(model.promptLookup == nil)
    #expect(model.samplingDefaults.temperature == 0.7)
    #expect(model.samplingDefaults.topP == nil)
    #expect(model.samplingDefaults.topK == nil)
    #expect(model.samplingDefaults.repetitionPenalty == nil)
    #expect(model.capabilities?.runtimeKind == "block_diffusion")
}

private func restartModelRoot(repoID: String, root existingRoot: URL? = nil) throws -> (root: URL, snapshot: URL) {
    let root: URL
    if let existingRoot {
        root = existingRoot
    } else {
        root = try restartTemporaryDirectory()
    }
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: repoID,
        files: [
            "config.json": Data("{}".utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    return (root, snapshot)
}

private func restartTemporaryDirectory() throws -> URL {
    let url = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-restart-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private actor FakeRestartModelLoader: BackendModelLoading {
    private let loadError: Error?
    private let registrationErrors: [String: Error]
    private(set) var calls: [String] = []

    init(loadError: Error? = nil, registrationErrors: [String: Error] = [:]) {
        self.loadError = loadError
        self.registrationErrors = registrationErrors
    }

    func waitUntilReady(timeout: TimeInterval) async throws {
        calls.append("waitUntilReady")
    }

    func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        mtpModelDir: String?,
        mtpDraftTokens: Int?,
        promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        calls.append("register:\(model):\(modelDir):\(setDefault):\(maxCacheCap.map(String.init) ?? "nil"):\(pinned):\(mtpModelDir ?? "nil"):\(mtpDraftTokens.map(String.init) ?? "nil")")
        if let error = registrationErrors[model] {
            throw error
        }
        return BackendModelAdminResponse(
            success: true,
            status: "registered",
            code: nil,
            model: model,
            loadedModels: [],
            warningCode: nil,
            warning: nil,
            error: nil
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
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        calls.append("load:\(model):\(modelDir):\(setDefault):\(maxCacheCap.map(String.init) ?? "nil"):\(pinned):\(mtpModelDir ?? "nil"):\(mtpDraftTokens.map(String.init) ?? "nil")")
        if let loadError {
            throw loadError
        }
        return BackendModelAdminResponse(
            success: true,
            status: "loaded",
            code: nil,
            model: model,
            loadedModels: [
                BackendLoadedModelInfo(
                    id: model,
                    model: model,
                    path: modelDir,
                    architecture: "llm",
                    isDefault: true,
                    maxPositionEmbeddings: 4096,
                    pinned: pinned
                ),
            ],
            warningCode: nil,
            warning: nil,
            error: nil
        )
    }
}
