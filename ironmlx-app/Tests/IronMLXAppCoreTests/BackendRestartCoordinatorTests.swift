import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor func restartDefaultModelLoadsPersistedDefaultModel() async throws {
    let (root, snapshot) = try restartModelRoot(repoID: "mlx-community/Tiny-4bit")
    let backend = FakeRestartBackend()
    let loader = FakeRestartModelLoader()
    let coordinator = BackendRestartCoordinator(
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        clientFactory: { _, _ in loader }
    )

    let result = await coordinator.restartDefaultModel(
        config: AppConfig(port: 9068, lastModel: "mlx-community/Tiny-4bit"),
        backend: backend
    )

    #expect(result.success)
    #expect(result.modelLoaded)
    #expect(result.model == "mlx-community/Tiny-4bit")
    #expect(backend.calls == [
        "stop",
        "start:mlx-community/Tiny-4bit",
    ])
    let loaderCalls = await loader.calls
    #expect(loaderCalls == [
        "waitUntilReady",
        "load:mlx-community/Tiny-4bit:\(snapshot.path):true:nil",
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
        config: AppConfig(port: 9068, lastModel: "mlx-community/Tiny-4bit"),
        backend: backend
    )

    #expect(!result.success)
    #expect(!result.modelLoaded)
    #expect(result.model == "mlx-community/Tiny-4bit")
    #expect(result.error == "memory budget exceeded")
    #expect(await loader.calls.count == 2)
}

private func restartModelRoot(repoID: String) throws -> (root: URL, snapshot: URL) {
    let root = try restartTemporaryDirectory()
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

    func start(modelReference: String) throws {
        calls.append("start:\(modelReference)")
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
            warning: nil,
            error: nil
        )
    }
}
