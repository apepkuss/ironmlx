import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor func restartDefaultModelRegistersUnloadedDefaultWithoutLoading() async throws {
    let (root, snapshot) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let backend = FakeRestartBackend()
    let loader = FakeRestartModelLoader()
    let coordinator = BackendRestartCoordinator(
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restartDefaultModel(
        config: AppConfig(port: 9068, defaultModel: "mlx-community/Tiny-4bit"),
        backend: backend
    )

    #expect(result.success)
    #expect(!result.modelLoaded)
    #expect(result.model == "mlx-community/Tiny-4bit")
    #expect(result.loadedModels == [])
    #expect(backend.calls == [
        "stop",
        "start",
    ])
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "waitUntilReady",
        "register:mlx-community/Tiny-4bit:\(snapshot.path):true:nil",
    ], "\(loaderCalls)")
}

@Test @MainActor func restartDefaultModelReportsLoadFailure() async throws {
    let (root, _) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let backend = FakeRestartBackend()
    let loader = FakeRestartModelLoader(
        loadError: BackendAPIError.serverResponse(
            statusCode: 400,
            body: #"{"success":false,"status":"error","error":"memory budget exceeded"}"#
        )
    )
    let coordinator = BackendRestartCoordinator(
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restartDefaultModel(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Tiny-4bit",
            loadedModels: ["mlx-community/Tiny-4bit"]
        ),
        backend: backend
    )

    #expect(!result.success)
    #expect(!result.modelLoaded)
    #expect(result.model == "mlx-community/Tiny-4bit")
    #expect(result.error == "memory budget exceeded")
    #expect(result.failedModels == ["mlx-community/Tiny-4bit"])
    #expect(await loader.calls.count == 3)
}

@Test @MainActor func restartDefaultModelPreservesBackendErrorCode() async throws {
    let (root, _) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let backend = FakeRestartBackend()
    let loader = FakeRestartModelLoader(
        loadError: BackendAPIError.serverResponse(
            statusCode: 503,
            body: #"{"success":false,"status":"error","code":"max_loaded_models_reached","error":"Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model."}"#
        )
    )
    let coordinator = BackendRestartCoordinator(
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restartDefaultModel(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Tiny-4bit",
            loadedModels: ["mlx-community/Tiny-4bit"]
        ),
        backend: backend
    )

    #expect(result.errorCode == "max_loaded_models_reached")
    #expect(result.error == "Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model.")
}

@Test @MainActor func restartDefaultModelRestoresMultipleLoadedModels() async throws {
    let (root, firstSnapshot) = try restartModelRoot(repoID: "mlx-community/First-4bit")
    let (_, secondSnapshot) = try restartModelRoot(repoID: "mlx-community/Second-4bit", root: root)
    let backend = FakeRestartBackend()
    let loader = FakeRestartModelLoader()
    let coordinator = BackendRestartCoordinator(
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restartDefaultModel(
        config: AppConfig(
            port: 9068,
            defaultModel: "mlx-community/Second-4bit",
            loadedModels: [
                "mlx-community/First-4bit",
                "mlx-community/Second-4bit",
            ]
        ),
        backend: backend
    )

    #expect(result.success)
    #expect(result.modelLoaded)
    #expect(result.model == "mlx-community/Second-4bit")
    #expect(result.loadedModels == [
        "mlx-community/Second-4bit",
        "mlx-community/First-4bit",
    ])
    #expect(backend.calls == [
        "stop",
        "start",
    ])
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "waitUntilReady",
        "register:mlx-community/First-4bit:\(firstSnapshot.path):false:nil",
        "register:mlx-community/Second-4bit:\(secondSnapshot.path):true:nil",
        "load:mlx-community/Second-4bit:\(secondSnapshot.path):true:nil",
        "load:mlx-community/First-4bit:\(firstSnapshot.path):false:nil",
    ], "\(loaderCalls)")
}

@Test @MainActor func restartDefaultModelRegistersLocalModelsWithoutLoadingWhenNoRestoredModel() async throws {
    let (root, snapshot) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let backend = FakeRestartBackend()
    let loader = FakeRestartModelLoader()
    let coordinator = BackendRestartCoordinator(
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restartDefaultModel(
        config: AppConfig(port: 9068),
        backend: backend
    )

    #expect(result.success)
    #expect(!result.modelLoaded)
    #expect(result.loadedModels.isEmpty)
    #expect(backend.calls == [
        "stop",
        "start",
    ])
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "waitUntilReady",
        "register:mlx-community/Tiny-4bit:\(snapshot.path):false:nil",
    ], "\(loaderCalls)")
}

private func restartModelRoot(repoID: String, root existingRoot: URL? = nil) throws -> (root: URL, snapshot: URL) {
    let root: URL
    if let existingRoot {
        root = existingRoot
    } else {
        root = try restartTemporaryDirectory()
    }
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    return (root, snapshot)
}

private func restartTemporaryDirectory() throws -> URL {
    let url = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-restart-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@MainActor
private final class FakeRestartBackend: BackendProcessManaging {
    private(set) var calls: [String] = []
    private(set) var isRunning: Bool = false

    func start() throws {
        calls.append("start")
        isRunning = true
    }

    func stop() {
        calls.append("stop")
        isRunning = false
    }
}

private actor FakeRestartModelLoader: BackendModelLoading {
    private let loadError: Error?
    private(set) var calls: [String] = []

    init(loadError: Error? = nil) {
        self.loadError = loadError
    }

    func waitUntilReady(timeout: TimeInterval) async throws {
        calls.append("waitUntilReady")
    }

    func registerModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        samplingDefaults: BackendSamplingDefaults
    ) async throws -> BackendModelAdminResponse {
        calls.append("register:\(model):\(modelDir):\(setDefault):\(maxCacheCap.map(String.init) ?? "nil")")
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
        maxCacheCap: Int?
    ) async throws -> BackendModelAdminResponse {
        calls.append("load:\(model):\(modelDir):\(setDefault):\(maxCacheCap.map(String.init) ?? "nil")")
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
                    maxPositionEmbeddings: 4096
                ),
            ],
            warningCode: nil,
            warning: nil,
            error: nil
        )
    }
}
