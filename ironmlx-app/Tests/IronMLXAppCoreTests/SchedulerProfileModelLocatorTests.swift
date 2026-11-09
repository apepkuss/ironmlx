import Foundation
import Testing

@testable import IronMLXAppCore

private let profileModel = "mlx-community/Profile-Lock-4bit"

private func profileModelRoot() throws -> (root: URL, snapshot: URL) {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-profile-locator-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: profileModel,
        files: ["config.json": Data("{}".utf8), "model.safetensors": Data("weights".utf8)]
    )
    return (root, snapshot)
}

private func holdRepositoryLock(root: URL) throws -> ModelRepositoryLock {
    try ModelDownloadStore(rootURL: root).acquireRepositoryLock(provider: .huggingFace, repoID: profileModel)
}

/// Counts retry waits; the test decides what happens during each one.
private final class RetryWaits: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private var onWait: (Int) -> Void

    init(onWait: @escaping (Int) -> Void = { _ in }) {
        self.onWait = onWait
    }

    var waits: Int { lock.withLock { count } }

    func wait(_: Duration) async throws {
        let current = lock.withLock {
            count += 1
            return count
        }
        onWait(current)
    }
}

@Test func profilePreviewFindsTheModelWhileItsRepositoryIsLocked() throws {
    let (root, snapshot) = try profileModelRoot()
    defer { try? FileManager.default.removeItem(at: root) }
    let scanner = LocalModelScanner(rootURL: root)
    let lock = try holdRepositoryLock(root: root)
    defer { withExtendedLifetime(lock) {} }
    // The locked verification would not get through now.
    #expect(throws: ModelRepositoryLockError.self) {
        try scanner.verifiedModelPath(for: profileModel)
    }

    #expect(try SchedulerProfileModelLocator.previewPath(for: profileModel, scanner: scanner) == snapshot.path)
}

@Test func profileGenerationWaitsForABriefRepositoryLock() async throws {
    let (root, snapshot) = try profileModelRoot()
    defer { try? FileManager.default.removeItem(at: root) }
    let scanner = LocalModelScanner(rootURL: root)
    nonisolated(unsafe) var lock: ModelRepositoryLock? = try holdRepositoryLock(root: root)
    let waits = RetryWaits { count in
        if count == 2 {
            lock = nil
        }
    }

    let path = try await SchedulerProfileModelLocator.generationPath(
        for: profileModel,
        scanner: scanner,
        fullChecksum: false,
        wait: waits.wait
    )

    #expect(path == snapshot.path)
    #expect(waits.waits == 2)
    #expect(lock == nil)
}

@Test func profileGenerationReportsABusyModelAfterItsRetries() async throws {
    let (root, _) = try profileModelRoot()
    defer { try? FileManager.default.removeItem(at: root) }
    let scanner = LocalModelScanner(rootURL: root)
    let lock = try holdRepositoryLock(root: root)
    defer { withExtendedLifetime(lock) {} }
    let waits = RetryWaits()

    do {
        _ = try await SchedulerProfileModelLocator.generationPath(
            for: profileModel,
            scanner: scanner,
            fullChecksum: false,
            wait: waits.wait
        )
        Issue.record("Expected the model to be busy")
    } catch let error as SchedulerProfileModelError {
        #expect(error == .busy(profileModel))
        #expect(error.code == SchedulerProfileModelError.busyCode)
        #expect(error.localizedDescription.contains("in use"))
    }
    #expect(waits.waits == SchedulerProfileModelLocator.busyAttempts - 1)
}

@Test func profileGenerationSurfacesASnapshotVerificationError() async throws {
    let (root, snapshot) = try profileModelRoot()
    defer { try? FileManager.default.removeItem(at: root) }
    let scanner = LocalModelScanner(rootURL: root)
    // The weights no longer match the manifest.
    try Data("other weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    let reason = try #require(scanner.readiness(for: profileModel)?.message)

    do {
        _ = try await SchedulerProfileModelLocator.generationPath(
            for: profileModel,
            scanner: scanner,
            fullChecksum: false,
            wait: RetryWaits().wait
        )
        Issue.record("Expected a verification error")
    } catch {
        #expect(error.localizedDescription.contains(reason))
        #expect(!error.localizedDescription.contains("not found"))
    }
    #expect(throws: SchedulerProfileModelError.self) {
        try SchedulerProfileModelLocator.previewPath(for: profileModel, scanner: scanner)
    }
}

@Test func profileGenerationStillReportsAMissingModel() async throws {
    let (root, _) = try profileModelRoot()
    defer { try? FileManager.default.removeItem(at: root) }
    let scanner = LocalModelScanner(rootURL: root)

    await #expect(throws: SchedulerProfileModelError.notFound("mlx-community/Absent-4bit")) {
        try await SchedulerProfileModelLocator.generationPath(
            for: "mlx-community/Absent-4bit",
            scanner: scanner,
            fullChecksum: false,
            wait: RetryWaits().wait
        )
    }
}

private func profileQwen35Config(modelType: String, mtpLayers: Int) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": {
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "full_attention_interval": 4,
        "linear_num_value_heads": 16,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": \(mtpLayers),
        "max_position_embeddings": 262144
      }
    }
    """
}

@Test func profilePreviewCountsTheSameJobsAsGenerationForAnMtpModel() async throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-profile-mtp-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let base = "mlx-community/Qwen3.5-4B-MLX-4bit"
    let mtp = "mlx-community/Qwen3.5-4B-MTP-4bit"
    for (repoID, config) in [
        (base, profileQwen35Config(modelType: "qwen3_5", mtpLayers: 0)),
        (mtp, profileQwen35Config(modelType: "qwen3_5_mtp", mtpLayers: 1)),
    ] {
        try writeVerifiedTestSnapshot(
            root: root,
            repoID: repoID,
            files: ["config.json": Data(config.utf8), "model.safetensors": Data("weights".utf8)]
        )
    }
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try parameterStore.save(ModelParameters(modelID: base, mtpEnabled: true))

    func jobs(_ target: SchedulerProfileModelTarget) -> Int? {
        SchedulerProfileGenerationPreview(
            request: SchedulerProfileGenerationRequest(
                model: base,
                modelPath: target.modelPath,
                calibrationLevel: "standard",
                mtpModelDir: target.mtpRuntime?.modelDir,
                mtpDraftTokens: target.mtpRuntime?.draftTokens
            )
        ).totalJobs
    }

    // The preview reads the draft model without its repository lock.
    let preview = try {
        let mtpLock = try ModelDownloadStore(rootURL: root).acquireRepositoryLock(provider: .huggingFace, repoID: mtp)
        defer { withExtendedLifetime(mtpLock) {} }
        return try SchedulerProfileModelLocator.previewTarget(
            for: base,
            scanner: scanner,
            parameterStore: parameterStore
        )
    }()
    let generation = try await SchedulerProfileModelLocator.generationTarget(
        for: base,
        scanner: scanner,
        parameterStore: parameterStore,
        fullChecksum: false,
        wait: RetryWaits().wait
    )

    #expect(preview.mtpRuntime?.modelID == mtp)
    #expect(preview.mtpRuntime == generation.mtpRuntime)
    #expect(jobs(preview) == jobs(generation))
    // The Qwen MTP matrix: b_max 1 and 2 only.
    #expect(jobs(preview) == 16)
}
