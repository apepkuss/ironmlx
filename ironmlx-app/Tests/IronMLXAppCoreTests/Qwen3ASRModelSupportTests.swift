import Foundation
import Testing
@testable import IronMLXAppCore

func asrConfig(large: Bool) -> [String: Any] {
    ["model_type": "qwen3_asr", "quantization": ["mode": "affine", "bits": 8, "group_size": 64],
     "thinker_config": [
        "audio_config": ["d_model": large ? 1024 : 896, "encoder_layers": large ? 24 : 18, "num_mel_bins": 128],
        "text_config": ["hidden_size": large ? 2048 : 1024, "num_hidden_layers": 28,
                        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936],
     ]]
}
private struct ASRPreflight: ModelMetadataPreflighting {
    var accepts = true
    var role = "asr"
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        if !accepts { throw ModelMetadataPreflightError.rejected("unsupported Qwen3-ASR configuration") }
        return ModelMetadataPreflightResult(modelType: "qwen3_asr", artifactRole: role,
            quantization: ModelQuantizationPreflight(mode: "affine", bits: 8, groupSize: 64, overrideCount: 0))
    }
}

@Test(arguments: [false, true])
func qwen3ASROldSnapshotBecomesLoadableWithoutRedownloading(large: Bool) async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let repo = "mlx-community/Qwen3-ASR-\(large ? "1.7B" : "0.6B")-8bit"
    let snapshot = try writeVerifiedTestSnapshot(root: root, repoID: repo, files: [
        "config.json": try JSONSerialization.data(withJSONObject: asrConfig(large: large)),
        "tokenizer_config.json": Data("{}".utf8), "preprocessor_config.json": Data("{}".utf8),
        "vocab.json": Data("{}".utf8), "merges.txt": Data("#version: 0.2".utf8),
        "model.safetensors": Data("weights".utf8),
    ])
    let verifier = ModelSnapshotVerifier()
    var manifest = try verifier.loadManifest(at: snapshot)
    manifest.compatibility.modelType = "qwen3_asr"
    manifest.compatibility.artifactRole = "download_only"
    manifest.compatibility.downloadOnly = true
    manifest.compatibility.runtimeSupportError = "unsupported model_type: qwen3_asr"
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
    await #expect(throws: ModelMetadataPreflightError.self) {
        try await Qwen3ASRModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: ASRPreflight(accepts: false))
    }
    #expect(try verifier.loadManifest(at: snapshot).compatibility.downloadOnly == true)
    try await Qwen3ASRModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: ASRPreflight())
    let upgraded = try verifier.verify(snapshot: snapshot)
    #expect(upgraded.files == manifest.files)
    #expect(upgraded.compatibility.artifactRole == "asr")
    #expect(upgraded.compatibility.downloadOnly == nil)
    let model = try #require(LocalModelScanner(rootURL: root).model(for: repo))
    #expect(model.type == "asr")
    #expect(model.readiness?.isLoadable == true)
    #expect(model.capabilities?.runtimeKind == "asr")
    #expect(model.capabilities?.supportsKvCache == false)
    #expect(model.capabilities?.supportsAudio == true)
}


@Test
func qwen3ForcedAlignerSnapshotUsesAlignmentTask() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    var config = asrConfig(large: false)
    var thinker = config["thinker_config"] as! [String: Any]
    var audio = thinker["audio_config"] as! [String: Any]
    var text = thinker["text_config"] as! [String: Any]
    audio["d_model"] = 1024
    audio["encoder_layers"] = 24
    text["vocab_size"] = 152064
    thinker["audio_config"] = audio
    thinker["text_config"] = text
    thinker["model_type"] = "qwen3_forced_aligner"
    thinker["classify_num"] = 5000
    config["thinker_config"] = thinker
    config["timestamp_token_id"] = 151705
    config["timestamp_segment_time"] = 80
    let repo = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    let snapshot = try writeVerifiedTestSnapshot(root: root, repoID: repo, files: [
        "config.json": try JSONSerialization.data(withJSONObject: config),
        "tokenizer_config.json": Data("{}".utf8), "preprocessor_config.json": Data("{}".utf8),
        "vocab.json": Data("{}".utf8), "merges.txt": Data("#version: 0.2".utf8), "model.safetensors": Data("weights".utf8),
    ])
    var manifest = try ModelSnapshotVerifier().loadManifest(at: snapshot)
    manifest.compatibility.modelType = "qwen3_asr"
    manifest.compatibility.downloadOnly = true
    manifest.compatibility.artifactRole = "download_only"
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
    try await Qwen3ASRModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: ASRPreflight(role: "alignment"))
    let model = try #require(LocalModelScanner(rootURL: root).model(for: repo))
    #expect(model.readiness?.isLoadable == true)
    #expect(model.type == "alignment")
    #expect(model.capabilities?.runtimeKind == "alignment")
    #expect(model.capabilities?.supportsStreaming == false)
    #expect(model.asrDefaults == nil)
}
