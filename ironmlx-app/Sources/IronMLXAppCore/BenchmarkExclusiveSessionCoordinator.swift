import Foundation

public typealias BenchmarkModelPathValidator =
    @Sendable (_ modelID: String, _ modelPath: String) async throws -> String

public actor BenchmarkExclusiveSessionCoordinator {
    private var session: BenchmarkExclusiveSession?

    public init() {}

    public func preflight(
        client: BackendModelManaging,
        targetModel: String
    ) async throws -> BenchmarkExclusivePreflightResult {
        let health = try await client.fetchHealthz()
        try ensureIdle(health)

        let loadedModels = try await client.fetchLoadedModels()
        let nonTargetModels = loadedModels
            .filter { !$0.pinned && !matches(model: $0, targetModel: targetModel) }
            .map(\.id)
            .sorted()
        let loadedIDs = loadedModels.map(\.id).sorted()

        return BenchmarkExclusivePreflightResult(
            success: true,
            targetModel: targetModel,
            activeRequests: health.scheduler.bActive,
            queuedRequests: health.scheduler.bQueued,
            loadedModels: loadedIDs,
            defaultModel: loadedModels.first(where: \.isDefault)?.id,
            nonTargetModels: nonTargetModels,
            willUnloadCount: nonTargetModels.count
        )
    }

    public func prepare(
        client: BackendModelManaging,
        targetModel: String,
        targetModelPath: String,
        validateModelPath: BenchmarkModelPathValidator? = nil
    ) async throws -> BenchmarkExclusivePrepareResult {
        guard session == nil else {
            throw BenchmarkExclusiveSessionError.alreadyRunning
        }

        let health = try await client.fetchHealthz()
        try ensureIdle(health)

        let loadedModels = try await client.fetchLoadedModels()
        let validatedTargetPath = try await validateModelPath?(targetModel, targetModelPath)
            ?? targetModelPath
        let targetWasPinned = loadedModels.first {
            matches(model: $0, targetModel: targetModel, targetModelPath: validatedTargetPath)
        }?.pinned ?? false
        let targetPromptLookup = loadedModels.first {
            matches(model: $0, targetModel: targetModel, targetModelPath: validatedTargetPath)
        }?.promptLookup
        let snapshot = BenchmarkExclusiveSession(
            targetModel: targetModel,
            targetModelPath: validatedTargetPath,
            loadedModels: loadedModels.map(BenchmarkLoadedModelSnapshot.init),
            defaultModel: loadedModels.first(where: \.isDefault)?.id
        )
        session = snapshot

        let nonTargetModels = loadedModels
            .filter {
                !$0.pinned
                    && !matches(
                        model: $0,
                        targetModel: targetModel,
                        targetModelPath: validatedTargetPath
                    )
            }
            .sorted { $0.id < $1.id }
        var unloaded: [String] = []

        do {
            for model in nonTargetModels {
                _ = try await client.unloadModel(model: model.id, modelDir: model.path)
                unloaded.append(model.id)
            }

            _ = try await client.loadModel(
                model: targetModel,
                modelDir: validatedTargetPath,
                setDefault: true,
                maxCacheCap: nil,
                pinned: targetWasPinned,
                promptLookup: targetPromptLookup
            )
            _ = try await client.setDefaultModel(targetModel)

            return BenchmarkExclusivePrepareResult(
                success: true,
                targetModel: targetModel,
                unloadedModels: unloaded
            )
        } catch {
            _ = try? await restoreFromSession(
                snapshot,
                client: client,
                validateModelPath: validateModelPath
            )
            session = nil
            throw error
        }
    }

    public func canRunBenchmark(targetModel: String) -> Bool {
        session?.targetModel == targetModel
    }

    public func restore(
        client: BackendModelManaging,
        validateModelPath: BenchmarkModelPathValidator? = nil
    ) async throws -> BenchmarkExclusiveRestoreResult {
        guard let activeSession = session else {
            return BenchmarkExclusiveRestoreResult(
                success: true,
                status: "not_active",
                restoredModels: [],
                unloadedModels: [],
                defaultModel: nil,
                warnings: []
            )
        }

        defer { session = nil }
        let result = try await restoreFromSession(
            activeSession,
            client: client,
            validateModelPath: validateModelPath
        )
        return result
    }

    private func restoreFromSession(
        _ activeSession: BenchmarkExclusiveSession,
        client: BackendModelManaging,
        validateModelPath: BenchmarkModelPathValidator?
    ) async throws -> BenchmarkExclusiveRestoreResult {
        let snapshotByID = Dictionary(uniqueKeysWithValues: activeSession.loadedModels.map { ($0.id, $0) })
        var currentByID = Dictionary(uniqueKeysWithValues: (try await client.fetchLoadedModels()).map { ($0.id, $0) })
        var unloaded: [String] = []
        var restored: [String] = []
        var warnings: [String] = []

        for model in currentByID.values.sorted(by: { $0.id < $1.id }) where snapshotByID[model.id] == nil && !model.pinned {
            do {
                _ = try await client.unloadModel(model: model.id, modelDir: model.path)
                unloaded.append(model.id)
                currentByID.removeValue(forKey: model.id)
            } catch {
                warnings.append("Failed to unload \(model.id): \(error.localizedDescription)")
            }
        }

        for model in activeSession.loadedModels.sorted(by: { $0.id < $1.id }) where currentByID[model.id] == nil {
            do {
                let modelPath = try await validateModelPath?(model.id, model.path) ?? model.path
                _ = try await client.loadModel(
                    model: model.id,
                    modelDir: modelPath,
                    setDefault: false,
                    maxCacheCap: nil,
                    pinned: model.pinned,
                    promptLookup: model.promptLookup
                )
                restored.append(model.id)
                currentByID[model.id] = model.backendInfo(isDefault: false)
            } catch {
                warnings.append("Failed to restore \(model.id): \(error.localizedDescription)")
            }
        }

        if let defaultModel = activeSession.defaultModel {
            do {
                _ = try await client.setDefaultModel(defaultModel)
            } catch {
                warnings.append("Failed to restore default model \(defaultModel): \(error.localizedDescription)")
            }
        }

        return BenchmarkExclusiveRestoreResult(
            success: warnings.isEmpty,
            status: warnings.isEmpty ? "restored" : "restored_with_warnings",
            restoredModels: restored,
            unloadedModels: unloaded,
            defaultModel: activeSession.defaultModel,
            warnings: warnings
        )
    }

    private func ensureIdle(_ health: HealthzSnapshot) throws {
        let active = health.scheduler.bActive
        let queued = health.scheduler.bQueued
        if active > 0 || queued > 0 {
            throw BenchmarkExclusiveSessionError.activeRequests(active: active, queued: queued)
        }
    }

    private func matches(
        model: BackendLoadedModelInfo,
        targetModel: String,
        targetModelPath: String? = nil
    ) -> Bool {
        model.id == targetModel
            || model.model == targetModel
            || targetModelPath.map { model.path == $0 } == true
    }
}

public struct BenchmarkExclusivePreflightResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var targetModel: String
    public var activeRequests: Int
    public var queuedRequests: Int
    public var loadedModels: [String]
    public var defaultModel: String?
    public var nonTargetModels: [String]
    public var willUnloadCount: Int

    enum CodingKeys: String, CodingKey {
        case success
        case targetModel = "target_model"
        case activeRequests = "active_requests"
        case queuedRequests = "queued_requests"
        case loadedModels = "loaded_models"
        case defaultModel = "default_model"
        case nonTargetModels = "non_target_models"
        case willUnloadCount = "will_unload_count"
    }
}

public struct BenchmarkExclusivePrepareResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var targetModel: String
    public var unloadedModels: [String]

    enum CodingKeys: String, CodingKey {
        case success
        case targetModel = "target_model"
        case unloadedModels = "unloaded_models"
    }
}

public struct BenchmarkExclusiveRestoreResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var restoredModels: [String]
    public var unloadedModels: [String]
    public var defaultModel: String?
    public var warnings: [String]

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case restoredModels = "restored_models"
        case unloadedModels = "unloaded_models"
        case defaultModel = "default_model"
        case warnings
    }
}

public enum BenchmarkExclusiveSessionError: LocalizedError, Equatable, Sendable {
    case activeRequests(active: Int, queued: Int)
    case alreadyRunning
    case notPrepared

    public var errorDescription: String? {
        switch self {
        case .activeRequests(let active, let queued):
            return "Benchmark cannot start while requests are running or queued. Active: \(active), queued: \(queued)."
        case .alreadyRunning:
            return "A benchmark session is already running."
        case .notPrepared:
            return "Benchmark exclusive session has not been prepared."
        }
    }

    public var code: String {
        switch self {
        case .activeRequests:
            return "benchmark_active_requests"
        case .alreadyRunning:
            return "benchmark_already_running"
        case .notPrepared:
            return "benchmark_not_prepared"
        }
    }
}

private struct BenchmarkExclusiveSession: Equatable, Sendable {
    var targetModel: String
    var targetModelPath: String
    var loadedModels: [BenchmarkLoadedModelSnapshot]
    var defaultModel: String?
}

private struct BenchmarkLoadedModelSnapshot: Equatable, Sendable {
    var id: String
    var model: String
    var path: String
    var architecture: String
    var isDefault: Bool
    var pinned: Bool
    var maxPositionEmbeddings: Int
    var promptLookup: BackendPromptLookupConfig?

    init(_ model: BackendLoadedModelInfo) {
        self.id = model.id
        self.model = model.model
        self.path = model.path
        self.architecture = model.architecture
        self.isDefault = model.isDefault
        self.pinned = model.pinned
        self.maxPositionEmbeddings = model.maxPositionEmbeddings
        self.promptLookup = model.promptLookup
    }

    func backendInfo(isDefault: Bool) -> BackendLoadedModelInfo {
        BackendLoadedModelInfo(
            id: id,
            model: model,
            path: path,
            architecture: architecture,
            isDefault: isDefault,
            maxPositionEmbeddings: maxPositionEmbeddings,
            pinned: pinned,
            promptLookup: promptLookup
        )
    }
}
