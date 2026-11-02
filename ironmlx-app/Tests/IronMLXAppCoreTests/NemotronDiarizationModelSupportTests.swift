import Foundation
import Testing
@testable import IronMLXAppCore

private struct DiarizationPreflight: ModelMetadataPreflighting {
    var accepts = true
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        guard accepts else { throw ModelMetadataPreflightError.rejected("unsupported Nemotron profile") }
        return ModelMetadataPreflightResult(modelType: "nemotron_diarization", artifactRole: "diarization")
    }
}
@Test
func nemotronDiarizationSnapshotPromotionPreservesFilesAndNativeGate() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let repo = "mlx-community/Nemotron-3-Diarization"
    let config: [String: Any] = ["model_type":"nemotron_diarization","dtype":"bfloat16","num_speakers":8,"output_subsampling_factor":1,
        "encoder_config":["d_model":512,"n_layers":31,"n_heads":8,"feat_in":128],
        "processor_config":["sampling_rate":16000,"n_fft":512,"hop_length":160],
        "modules_config":["num_speakers":8,"fc_d_model":512,"tf_d_model":192]]
    #expect(NemotronDiarizationModelSupport.isSupported(config))
    var quantized = config
    quantized["quantization"] = ["bits":8]
    #expect(!NemotronDiarizationModelSupport.isSupported(quantized))
    let snapshot = try writeVerifiedTestSnapshot(root: root, repoID: repo, files: [
        "config.json": try JSONSerialization.data(withJSONObject: config), "model.safetensors": Data("weights".utf8)])
    let verifier = ModelSnapshotVerifier()
    var manifest = try verifier.loadManifest(at: snapshot)
    manifest.compatibility.modelType = "nemotron_diarization"
    manifest.compatibility.artifactRole = "download_only"
    manifest.compatibility.downloadOnly = true
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
    await #expect(throws: ModelMetadataPreflightError.self) {
        try await NemotronDiarizationModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: DiarizationPreflight(accepts: false))
    }
    #expect(try verifier.loadManifest(at: snapshot).compatibility.downloadOnly == true)
    try await NemotronDiarizationModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: DiarizationPreflight())
    let updated = try verifier.verify(snapshot: snapshot)
    #expect(updated.files == manifest.files)
    #expect(updated.compatibility.downloadOnly == nil)
    #expect(updated.compatibility.artifactRole == "diarization")
    let scanner = LocalModelScanner(rootURL: root)
    let model = try #require(scanner.model(for: repo))
    #expect(model.type == "diarization")
    #expect(model.readiness?.isLoadable == true)
    #expect(model.capabilities?.runtimeKind == "diarization")
    #expect(model.capabilities?.supportsStreaming == true)
    #expect(model.capabilities?.supportsAudio == true)
    #expect(model.capabilities?.supportsKvCache == false)
    #expect(model.asrDefaults == nil)
}
