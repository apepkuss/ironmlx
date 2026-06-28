import Foundation
import Testing

@testable import IronMLXAppCore

@Test func benchmarkExclusivePreflightRejectsActiveRequests() async throws {
    let client = MockBenchmarkModelClient(
        health: healthSnapshot(active: 1, queued: 0),
        loadedModels: [
            loadedModel("mlx-community/Old-4bit", isDefault: true),
        ]
    )
    let coordinator = BenchmarkExclusiveSessionCoordinator()

    await #expect(throws: BenchmarkExclusiveSessionError.activeRequests(active: 1, queued: 0)) {
        _ = try await coordinator.preflight(
            client: client,
            targetModel: "mlx-community/New-4bit"
        )
    }
}

@Test func benchmarkExclusivePrepareUnloadsNonTargetModelsAndSetsTargetDefault() async throws {
    let client = MockBenchmarkModelClient(
        health: healthSnapshot(active: 0, queued: 0),
        loadedModels: [
            loadedModel("mlx-community/Old-4bit", path: "/models/old", isDefault: true),
            loadedModel("mlx-community/New-4bit", path: "/models/new", isDefault: false),
        ]
    )
    let coordinator = BenchmarkExclusiveSessionCoordinator()

    let prepared = try await coordinator.prepare(
        client: client,
        targetModel: "mlx-community/New-4bit",
        targetModelPath: "/models/new"
    )

    #expect(prepared.success)
    #expect(prepared.unloadedModels == ["mlx-community/Old-4bit"])
    #expect(await client.calls == [
        "fetchHealthz",
        "fetchLoadedModels",
        "unload:mlx-community/Old-4bit:/models/old",
        "load:mlx-community/New-4bit:/models/new:true",
        "setDefault:mlx-community/New-4bit",
    ])
    #expect(await client.loadedIDs == ["mlx-community/New-4bit"])
    #expect(await client.defaultID == "mlx-community/New-4bit")
}

@Test func benchmarkExclusiveRestoreRestoresLoadedModelsAndDefaultModel() async throws {
    let client = MockBenchmarkModelClient(
        health: healthSnapshot(active: 0, queued: 0),
        loadedModels: [
            loadedModel("mlx-community/Old-4bit", path: "/models/old", isDefault: true),
        ]
    )
    let coordinator = BenchmarkExclusiveSessionCoordinator()

    _ = try await coordinator.prepare(
        client: client,
        targetModel: "mlx-community/New-4bit",
        targetModelPath: "/models/new"
    )
    let restored = try await coordinator.restore(client: client)

    #expect(restored.success)
    #expect(restored.restoredModels == ["mlx-community/Old-4bit"])
    #expect(restored.unloadedModels == ["mlx-community/New-4bit"])
    #expect(await client.loadedIDs == ["mlx-community/Old-4bit"])
    #expect(await client.defaultID == "mlx-community/Old-4bit")
}

private actor MockBenchmarkModelClient: BackendModelManaging {
    private var health: HealthzSnapshot
    private var loadedModelsByID: [String: BackendLoadedModelInfo]
    private var defaultModelID: String?
    private(set) var calls: [String] = []

    init(health: HealthzSnapshot, loadedModels: [BackendLoadedModelInfo]) {
        self.health = health
        self.loadedModelsByID = Dictionary(uniqueKeysWithValues: loadedModels.map { ($0.id, $0) })
        self.defaultModelID = loadedModels.first(where: \.isDefault)?.id
    }

    var loadedIDs: [String] {
        loadedModelsByID.keys.sorted()
    }

    var defaultID: String? {
        defaultModelID
    }

    func fetchHealthz() async throws -> HealthzSnapshot {
        calls.append("fetchHealthz")
        return health
    }

    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        calls.append("fetchLoadedModels")
        return loadedModelsByID.values
            .map { model in
                var updated = model
                updated.isDefault = model.id == defaultModelID
                return updated
            }
            .sorted { $0.id < $1.id }
    }

    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?
    ) async throws -> BackendModelAdminResponse {
        calls.append("load:\(model):\(modelDir):\(setDefault)")
        loadedModelsByID[model] = loadedModel(model, path: modelDir, isDefault: setDefault)
        if setDefault {
            defaultModelID = model
        }
        return adminResponse(status: "loaded")
    }

    func unloadModel(model: String, modelDir: String?) async throws -> BackendModelAdminResponse {
        calls.append("unload:\(model):\(modelDir ?? "")")
        loadedModelsByID.removeValue(forKey: model)
        if defaultModelID == model {
            defaultModelID = nil
        }
        return adminResponse(status: "unloaded")
    }

    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse {
        calls.append("setDefault:\(model)")
        defaultModelID = model
        return adminResponse(status: "default_set")
    }

    private func adminResponse(status: String) -> BackendModelAdminResponse {
        BackendModelAdminResponse(
            success: true,
            status: status,
            code: nil,
            model: nil,
            loadedModels: [],
            warningCode: nil,
            warning: nil,
            error: nil
        )
    }
}

private func loadedModel(
    _ id: String,
    path: String? = nil,
    isDefault: Bool
) -> BackendLoadedModelInfo {
    BackendLoadedModelInfo(
        id: id,
        model: id,
        path: path ?? "/models/\(id.replacingOccurrences(of: "/", with: "-"))",
        architecture: "qwen",
        isDefault: isDefault,
        maxPositionEmbeddings: 32768
    )
}

private func healthSnapshot(active: Int, queued: Int) -> HealthzSnapshot {
    HealthzSnapshot(
        status: "healthy",
        uptimeSecs: 10,
        model: .init(name: "test", maxPositionEmbeddings: 32768),
        scheduler: .init(
            bMax: 4,
            bActive: active,
            bQueued: queued,
            queueMax: 32,
            admissionQueueFullCount: 0,
            memoryBudgetExceededCount: 0
        ),
        memory: .init(
            totalRamBytes: 64 * 1024 * 1024 * 1024,
            freeRamBytes: 32 * 1024 * 1024 * 1024,
            kvCacheActiveBytes: 0,
            kvCacheSoftLimitBytes: 0,
            mlxTotalBytes: nil,
            mlxMaxRecommendedBytes: nil,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            mlxMemoryLimitBytes: 0
        ),
        activeKvOffload: .init(
            enabled: false,
            mode: "disabled",
            storageDir: nil,
            residentPages: 0,
            offloadedPages: 0,
            loadingPages: 0,
            dirtyPages: 0,
            parkedRequests: 0,
            offloadedBytes: 0,
            swapOutCount: 0,
            swapInCount: 0,
            swapErrorCount: 0,
            lastSwapOutUs: 0,
            lastSwapInUs: 0,
            supportedCacheKinds: [],
            notApplicableCacheKinds: []
        ),
        deviceName: "Apple Test",
        version: "test"
    )
}
