import Foundation
import Testing

@testable import IronMLXAppCore

@Test func localModelScannerRejectsIncompleteShardedSnapshot() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--Sharded-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("""
    {"weight_map":{"layer.0":"model-00001-of-00002.safetensors","layer.1":"model-00002-of-00002.safetensors"}}
    """.utf8).write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
    try Data("partial".utf8).write(to: snapshot.appendingPathComponent("model-00001-of-00002.safetensors"))

    let scanner = LocalModelScanner(rootURL: root)

    #expect(scanner.scan().isEmpty)

    try Data("complete".utf8).write(to: snapshot.appendingPathComponent("model-00002-of-00002.safetensors"))

    #expect(scanner.scan().map(\.repoID) == ["mlx-community/Sharded-4bit"])
}

@Test func localModelScannerReadsContextWindowFromConfig() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--LongContext-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("""
    {
      "model_type": "qwen3_5",
      "text_config": {
        "max_position_embeddings": 262144
      }
    }
    """.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let scanner = LocalModelScanner(rootURL: root)

    #expect(scanner.scan().first?.maxPositionEmbeddings == 262144)
}

@Test func localModelScannerCountsSymlinkedSnapshotTargetsForModelSize() throws {
    let root = try temporaryDirectory()
    let modelRoot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--Symlinked-4bit", isDirectory: true)
    let snapshot = modelRoot
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    let blobs = modelRoot.appendingPathComponent("blobs", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)

    let configBlob = blobs.appendingPathComponent("config-blob")
    let weightBlob = blobs.appendingPathComponent("weight-blob")
    try Data("{}".utf8).write(to: configBlob)
    try Data(repeating: 1, count: 2 * 1_048_576).write(to: weightBlob)
    try FileManager.default.createSymbolicLink(
        atPath: snapshot.appendingPathComponent("config.json").path,
        withDestinationPath: "../../blobs/config-blob"
    )
    try FileManager.default.createSymbolicLink(
        atPath: snapshot.appendingPathComponent("model.safetensors").path,
        withDestinationPath: "../../blobs/weight-blob"
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.sizeMB >= 2.0)
}

@Test func localModelScannerMarksPinnedLoadedModels() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Pinned-4bit",
        configJSON: "{}"
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan(
        loadedModels: ["mlx-community/Pinned-4bit"],
        pinnedModels: ["mlx-community/Pinned-4bit"],
        mtpEnabledModels: []
    ).first)

    #expect(model.loaded)
    #expect(model.pinned)
}

@Test func localModelScannerAttachesCompatibleMtpWeightsToBaseModel() throws {
    let root = try temporaryDirectory()
    let base = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        configJSON: qwen35Config(modelType: "qwen3_5", mtpLayers: 0)
    )
    let mtp = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Qwen3.5-4B-MTP-4bit",
        configJSON: qwen35Config(modelType: "qwen3_5_mtp", mtpLayers: 1)
    )

    let models = LocalModelScanner(rootURL: root).scan()

    #expect(models.map(\.id) == ["mlx-community/Qwen3.5-4B-MLX-4bit"])
    let model = try #require(models.first)
    #expect(model.type == "llm")
    #expect(model.mtp?.status == "available")
    #expect(model.mtp?.candidates.map(\.id) == ["mlx-community/Qwen3.5-4B-MTP-4bit"])
    #expect(
        URL(fileURLWithPath: try #require(model.mtp?.candidates.first?.path))
            .resolvingSymlinksInPath()
            .path == mtp.resolvingSymlinksInPath().path
    )
    #expect(base.lastPathComponent == "main")
}

@Test func localModelScannerAttachesCompatibleMoeMtpWeightsWithoutDenseIntermediateSize() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        configJSON: qwen35MoeConfig(modelType: "qwen3_5_moe", mtpLayers: 0)
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
        configJSON: qwen35MoeConfig(modelType: "qwen3_5_mtp", mtpLayers: 1)
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.id == "mlx-community/Qwen3.6-35B-A3B-4bit")
    #expect(model.mtp?.status == "available")
    #expect(model.mtp?.candidates.map(\.id) == ["mlx-community/Qwen3.6-35B-A3B-MTP-4bit"])
}

@Test func localModelScannerMarksQwenModelIncompatibleWhenOnlyMismatchedMtpExists() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        configJSON: qwen35Config(modelType: "qwen3_5", mtpLayers: 0, hiddenSize: 2560)
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Qwen3.5-8B-MTP-4bit",
        configJSON: qwen35Config(modelType: "qwen3_5_mtp", mtpLayers: 1, hiddenSize: 4096)
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.mtp?.status == "incompatible")
    #expect(model.mtp?.candidates.isEmpty == true)
    #expect(model.mtp?.incompatibleCandidates.map(\.id) == ["mlx-community/Qwen3.5-8B-MTP-4bit"])
}

func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@discardableResult
private func writeSnapshot(
    root: URL,
    repoID: String,
    configJSON: String
) throws -> URL {
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(configJSON.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    return snapshot
}

private func qwen35Config(
    modelType: String,
    mtpLayers: Int,
    hiddenSize: Int = 2560
) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": {
        "hidden_size": \(hiddenSize),
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

private func qwen35MoeConfig(
    modelType: String,
    mtpLayers: Int,
    hiddenSize: Int = 2048,
    moeIntermediateSize: Int = 512
) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": \(hiddenSize),
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "full_attention_interval": 4,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": \(moeIntermediateSize),
        "shared_expert_intermediate_size": \(moeIntermediateSize),
        "mtp_num_hidden_layers": \(mtpLayers),
        "max_position_embeddings": 262144
      }
    }
    """
}
