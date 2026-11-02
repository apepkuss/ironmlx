import Foundation

enum NemotronDiarizationModelSupport {
    static func isSupported(_ config: [String: Any]) -> Bool {
        guard config["model_type"] as? String == "nemotron_diarization",
              config["dtype"] as? String == "bfloat16",
              config["num_speakers"] as? Int == 8,
              config["output_subsampling_factor"] as? Int == 1,
              config["quantization"] == nil, config["quantization_config"] == nil,
              let encoder = config["encoder_config"] as? [String: Any],
              let processor = config["processor_config"] as? [String: Any],
              let head = config["modules_config"] as? [String: Any]
        else { return false }
        return encoder["d_model"] as? Int == 512 && encoder["n_layers"] as? Int == 31
            && encoder["n_heads"] as? Int == 8 && encoder["feat_in"] as? Int == 128
            && processor["sampling_rate"] as? Int == 16000 && processor["n_fft"] as? Int == 512
            && processor["hop_length"] as? Int == 160 && head["num_speakers"] as? Int == 8
            && head["fc_d_model"] as? Int == 512 && head["tf_d_model"] as? Int == 192
    }

    /// Verified file-only weights are reused, with native preflight as the promotion gate.
    static func upgradeDownloadOnlySnapshot(at path: String,
        preflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight()) async throws {
        let directory = URL(fileURLWithPath: path)
        let verifier = ModelSnapshotVerifier()
        guard var manifest = try? verifier.loadManifest(at: directory),
              manifest.compatibility.modelType == "nemotron_diarization",
              manifest.compatibility.downloadOnly == true else { return }
        _ = try verifier.verifyStructure(snapshot: directory)
        let result = try await preflight.validate(metadataDirectory: directory)
        guard result.modelType == "nemotron_diarization", result.artifactRole == "diarization" else {
            throw ModelMetadataPreflightError.rejected("Invalid Nemotron diarization preflight result.")
        }
        manifest.compatibility = ModelSnapshotCompatibility(modelType: result.modelType, artifactRole: result.artifactRole)
        try ModelDownloadStore.atomicWrite(manifest, to: directory.appendingPathComponent(ModelSnapshotManifest.filename))
    }
}
