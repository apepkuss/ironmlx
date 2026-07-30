import Foundation
import Testing

@testable import IronMLXAppCore

@Test func localModelScannerSurfacesIncompleteShardedSnapshot() throws {
    let root = try temporaryDirectory()
    let index = Data("""
    {"weight_map":{"layer.0":"model-00001-of-00002.safetensors","layer.1":"model-00002-of-00002.safetensors"}}
    """.utf8)
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "mlx-community/Sharded-4bit",
        files: [
            "config.json": Data("{}".utf8),
            "model.safetensors.index.json": index,
            "model-00001-of-00002.safetensors": Data("partial".utf8),
        ]
    )

    let scanner = LocalModelScanner(rootURL: root)

    let incomplete = try #require(scanner.scan().first)
    #expect(incomplete.readiness?.status == "incomplete")
    #expect(incomplete.readiness?.missingFiles == ["model-00002-of-00002.safetensors"])
    #expect(scanner.resolveModelPath(for: "mlx-community/Sharded-4bit") == nil)

    try Data("complete".utf8).write(to: snapshot.appendingPathComponent("model-00002-of-00002.safetensors"))
    try refreshTestSnapshotManifest(at: snapshot)

    #expect(scanner.scan().map(\.repoID) == ["mlx-community/Sharded-4bit"])
    #expect(scanner.scan().first?.readiness?.status == "ready")
    #expect(scanner.resolveModelPath(for: "mlx-community/Sharded-4bit") != nil)
}

@Test func localModelScannerReportsSupportedWeightQuantizationKinds() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Dense-bf16",
        configJSON: #"{"torch_dtype":"bfloat16"}"#
    )
    for bits in [2, 4, 5, 6, 8] {
        _ = try writeSnapshot(
            root: root,
            repoID: "mlx-community/Affine-\(bits)bit",
            configJSON: #"{"quantization":{"group_size":64,"bits":\#(bits),"mode":"affine"}}"#
        )
    }
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/MXFP4",
        configJSON: #"{"quantization":{"group_size":32,"bits":4,"mode":"mxfp4"}}"#
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/MXFP8",
        configJSON: #"{"quantization_config":{"group_size":32,"bits":8,"mode":"mxfp8"}}"#
    )
    let optiq = try writeSnapshot(
        root: root,
        repoID: "mlx-community/OptiQ-4bit",
        configJSON: #"{"quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#
    )
    try Data("""
    {
      "method": "optiq_mixed_precision",
      "per_layer": {
        "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 4},
        "model.layers.0.self_attn.k_proj": {"group_size": 64, "bits": 8}
      }
    }
    """.utf8).write(to: optiq.appendingPathComponent("optiq_metadata.json"))

    let models = Dictionary(uniqueKeysWithValues: LocalModelScanner(rootURL: root).scan().map { ($0.id, $0) })

    #expect(models["mlx-community/Dense-bf16"]?.quantization?.label == "bf16")
    for bits in [2, 4, 5, 6, 8] {
        let model = try #require(models["mlx-community/Affine-\(bits)bit"])
        #expect(model.quantization?.kind == "affine")
        #expect(model.quantization?.bits == bits)
        #expect(model.quantization?.label == "affine \(bits)-bit")
        #expect(model.readiness?.status == "ready")
    }
    #expect(models["mlx-community/MXFP4"]?.quantization?.label == "MXFP4")
    #expect(models["mlx-community/MXFP8"]?.quantization?.label == "MXFP8")
    #expect(models["mlx-community/OptiQ-4bit"]?.quantization?.kind == "optiq")
    #expect(models["mlx-community/OptiQ-4bit"]?.quantization?.mixedBits == [4, 8])
    #expect(models["mlx-community/OptiQ-4bit"]?.quantization?.label == "OptiQ 4/8-bit")
}

@Test func localModelScannerReportsModelCapabilityTypes() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Text-LLM",
        configJSON: #"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Vision-LM",
        configJSON: """
        {
          "model_type": "gemma4",
          "architectures": ["Gemma4ForConditionalGeneration"],
          "text_config": {"model_type": "gemma4_text"},
          "vision_config": {"model_type": "gemma4_vision", "patch_size": 16}
        }
        """
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Text-Embedding",
        configJSON: #"{"pipeline_tag":"feature-extraction","architectures":["BertModel"]}"#
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Text-Reranker",
        configJSON: #"{"pipeline_tag":"text-ranking","architectures":["XLMRobertaForSequenceClassification"]}"#
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Speech-ASR",
        configJSON: #"{"model_type":"whisper","pipeline_tag":"automatic-speech-recognition"}"#
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Speech-TTS",
        configJSON: #"{"model_type":"vits","pipeline_tag":"text-to-speech"}"#
    )

    let models = Dictionary(uniqueKeysWithValues: LocalModelScanner(rootURL: root).scan().map { ($0.id, $0) })

    #expect(models["mlx-community/Text-LLM"]?.type == "llm")
    #expect(models["mlx-community/Text-LLM"]?.readiness?.status == "ready")
    #expect(models["mlx-community/Vision-LM"]?.type == "vlm")
    #expect(models["mlx-community/Vision-LM"]?.readiness?.status == "ready")
    for (id, type) in [
        ("mlx-community/Text-Embedding", "embedding"),
        ("mlx-community/Text-Reranker", "reranker"),
        ("mlx-community/Speech-ASR", "asr"),
        ("mlx-community/Speech-TTS", "tts"),
    ] {
        let model = try #require(models[id])
        #expect(model.type == type)
        #expect(model.readiness?.status == "unsupported")
        #expect(model.readiness?.reasonCode == "unsupported_model_type")
    }
}

@Test func localModelScannerMarksMissingOptiqSidecarIncomplete() throws {
    let root = try temporaryDirectory()
    let snapshot = try writeSnapshot(
        root: root,
        repoID: "mlx-community/Gemma-OptiQ",
        configJSON: """
        {
          "quantization": {"group_size": 64, "bits": 4, "mode": "optiq"},
          "optiq_vision": {"sidecar": "optiq/optiq_vision.safetensors"}
        }
        """
    )
    try Data("""
    {
      "method": "optiq_mixed_precision",
      "per_layer": {
        "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 4}
      }
    }
    """.utf8).write(to: snapshot.appendingPathComponent("optiq_metadata.json"))
    try refreshTestSnapshotManifest(at: snapshot)

    let scanner = LocalModelScanner(rootURL: root)
    let incomplete = try #require(scanner.scan().first)

    #expect(incomplete.readiness?.status == "incomplete")
    #expect(incomplete.readiness?.missingFiles == ["optiq/optiq_vision.safetensors"])
    #expect(scanner.resolveModelPath(for: "mlx-community/Gemma-OptiQ") == nil)

    let sidecar = snapshot.appendingPathComponent("optiq", isDirectory: true)
        .appendingPathComponent("optiq_vision.safetensors")
    try FileManager.default.createDirectory(at: sidecar.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("vision".utf8).write(to: sidecar)
    try refreshTestSnapshotManifest(at: snapshot)

    let ready = try #require(scanner.scan().first)
    #expect(ready.readiness?.status == "ready")
    #expect(scanner.resolveModelPath(for: "mlx-community/Gemma-OptiQ") != nil)
}

@Test func localModelScannerMarksUnsupportedQuantizationMode() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/NVFP4",
        configJSON: #"{"quantization":{"group_size":16,"bits":4,"mode":"nvfp4"}}"#
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.quantization?.kind == "unknown")
    #expect(model.readiness?.status == "unsupported")
    #expect(model.readiness?.reasonCode == "unsupported_quantization")
}

@Test func localModelScannerReadsContextWindowFromConfig() throws {
    let root = try temporaryDirectory()
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "mlx-community/LongContext-4bit",
        files: [
            "config.json": Data("""
            {
              "model_type": "qwen3_5",
              "text_config": {
                "max_position_embeddings": 262144
              }
            }
            """.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )

    let scanner = LocalModelScanner(rootURL: root)

    #expect(scanner.scan().first?.maxPositionEmbeddings == 262144)
}

@Test func localModelScannerCountsSymlinkedSnapshotTargetsForModelSize() throws {
    let root = try temporaryDirectory()
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "mlx-community/Symlinked-4bit",
        files: [
            "config.json": Data("{}".utf8),
            "model.safetensors": Data(repeating: 1, count: 2 * 1_048_576),
        ]
    )
    let modelRoot = snapshot.deletingLastPathComponent().deletingLastPathComponent()
    let blobs = modelRoot.appendingPathComponent("blobs", isDirectory: true)
    try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)

    let configBlob = blobs.appendingPathComponent("config-blob")
    let weightBlob = blobs.appendingPathComponent("weight-blob")
    try FileManager.default.moveItem(at: snapshot.appendingPathComponent("config.json"), to: configBlob)
    try FileManager.default.moveItem(at: snapshot.appendingPathComponent("model.safetensors"), to: weightBlob)
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
    #expect(base.lastPathComponent == String(repeating: "a", count: 40))
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

@Test func localModelScannerAttachesCompatibleGemma4AssistantWeights() throws {
    let root = try temporaryDirectory()
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/gemma-4-e4b-it-4bit",
        configJSON: gemma4BaseConfig(modelType: "gemma4", textModelType: "gemma4_text", hiddenSize: 2560)
    )
    _ = try writeSnapshot(
        root: root,
        repoID: "mlx-community/gemma-4-E4B-it-qat-assistant-4bit",
        configJSON: gemma4AssistantConfig(modelType: "gemma4_assistant", textModelType: "gemma4_text", backboneHiddenSize: 2560)
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.id == "mlx-community/gemma-4-e4b-it-4bit")
    #expect(model.type == "llm")
    #expect(model.mtp?.status == "available")
    #expect(model.mtp?.candidates.map(\.id) == ["mlx-community/gemma-4-E4B-it-qat-assistant-4bit"])
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
    try writeVerifiedTestSnapshot(
        root: root,
        repoID: repoID,
        files: [
            "config.json": Data(configJSON.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
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

private func gemma4BaseConfig(
    modelType: String,
    textModelType: String,
    hiddenSize: Int
) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": {
        "model_type": "\(textModelType)",
        "hidden_size": \(hiddenSize),
        "num_hidden_layers": 42,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 18,
        "hidden_size_per_layer_input": 256,
        "layer_types": [
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention"
        ],
        "tie_word_embeddings": true,
        "max_position_embeddings": 131072
      }
    }
    """
}

private func gemma4AssistantConfig(
    modelType: String,
    textModelType: String,
    backboneHiddenSize: Int
) -> String {
    """
    {
      "model_type": "\(modelType)",
      "backbone_hidden_size": \(backboneHiddenSize),
      "use_ordered_embeddings": true,
      "num_centroids": 2048,
      "centroid_intermediate_top_k": 32,
      "tie_word_embeddings": true,
      "text_config": {
        "model_type": "\(textModelType)",
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "intermediate_size": 2048,
        "num_attention_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "vocab_size": 262144,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 4,
        "hidden_size_per_layer_input": 0,
        "layer_types": [
          "sliding_attention", "sliding_attention",
          "sliding_attention", "full_attention"
        ],
        "tie_word_embeddings": true
      }
    }
    """
}
