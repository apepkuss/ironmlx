import Foundation

enum Qwen3ASRModelSupport {
    static func isSupported(_ config: [String: Any]) -> Bool {
        guard config["model_type"] as? String == "qwen3_asr",
              let thinker = config["thinker_config"] as? [String: Any],
              let audio = thinker["audio_config"] as? [String: Any],
              let text = thinker["text_config"] as? [String: Any],
              let quant = config["quantization"] as? [String: Any],
              quant["mode"] as? String == "affine", quant["bits"] as? Int == 8,
              quant["group_size"] as? Int == 64,
              audio["num_mel_bins"] as? Int == 128,
              text["num_hidden_layers"] as? Int == 28,
              text["num_attention_heads"] as? Int == 16,
              text["num_key_value_heads"] as? Int == 8,
              text["head_dim"] as? Int == 128,
              text["vocab_size"] as? Int == 151936 else { return false }
        return (audio["d_model"] as? Int == 896 && audio["encoder_layers"] as? Int == 18
            && text["hidden_size"] as? Int == 1024)
            || (audio["d_model"] as? Int == 1024 && audio["encoder_layers"] as? Int == 24
                && text["hidden_size"] as? Int == 2048)
    }

    /// Preserve verified files; only a successful native preflight can promote an old snapshot.
    static func upgradeDownloadOnlySnapshot(
        at path: String,
        preflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight()
    ) async throws {
        let directory = URL(fileURLWithPath: path)
        let verifier = ModelSnapshotVerifier()
        guard var manifest = try? verifier.loadManifest(at: directory),
              manifest.compatibility.modelType == "qwen3_asr",
              manifest.compatibility.downloadOnly == true else { return }
        _ = try verifier.verifyStructure(snapshot: directory)
        let result = try await preflight.validate(metadataDirectory: directory)
        guard result.modelType == "qwen3_asr", result.artifactRole == "asr" else {
            throw ModelMetadataPreflightError.rejected("Invalid Qwen3-ASR preflight result.")
        }
        manifest.compatibility = ModelSnapshotCompatibility(
            modelType: result.modelType, artifactRole: result.artifactRole,
            quantizationMode: result.quantization?.mode,
            quantizationBits: result.quantization?.bits,
            quantizationGroupSize: result.quantization?.groupSize
        )
        try ModelDownloadStore.atomicWrite(manifest, to: directory.appendingPathComponent(ModelSnapshotManifest.filename))
    }
}
