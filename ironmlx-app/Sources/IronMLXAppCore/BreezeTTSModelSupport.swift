import Foundation

/// Breeze is self-contained: the separate Qwen speech codec is required in both variants.
enum BreezeTTSModelSupport {
    static let requiredMetadata = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                                   "audio_tokenizer/config.json", "audio_tokenizer/preprocessor_config.json"]
    static let codecWeight = "audio_tokenizer/model.safetensors"

    static func isSupported(_ config: [String: Any]) -> Bool {
        guard config["model_type"] as? String == "breeze_tts",
              config["audio_num_codebooks"] as? Int == 16,
              let backbone = config["backbone_config"] as? [String: Any],
              backbone["model_type"] as? String == "qwen3",
              backbone["num_hidden_layers"] as? Int == 28,
              backbone["hidden_size"] as? Int == 2048,
              let depth = config["depth_decoder_config"] as? [String: Any],
              depth["num_hidden_layers"] as? Int == 12,
              let text = config["text_encoder_config"] as? [String: Any],
              text["model_type"] as? String == "t5gemma2_text",
              text["num_hidden_layers"] as? Int == 26 else { return false }
        for key in ["quantization", "quantization_config"] {
            if let quant = config[key] {
                guard let quant = quant as? [String: Any], quant["mode"] as? String == "mxfp8",
                      quant["bits"] as? Int == 8, quant["group_size"] as? Int == 32 else { return false }
            }
        }
        return true
    }

    static func missingFiles(in directory: URL) -> [String] {
        (requiredMetadata + [codecWeight]).filter {
            let path = directory.appendingPathComponent($0).path
            return !FileManager.default.isReadableFile(atPath: path)
                || ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.int64Value ?? 0) <= 0
        }
    }

    static func upgradeDownloadOnlySnapshot(at path: String,
        preflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight()) async throws {
        let directory = URL(fileURLWithPath: path)
        let verifier = ModelSnapshotVerifier()
        guard var manifest = try? verifier.loadManifest(at: directory),
              manifest.compatibility.modelType == "breeze_tts", manifest.compatibility.downloadOnly == true else { return }
        _ = try verifier.verifyStructure(snapshot: directory)
        guard missingFiles(in: directory).isEmpty,
              manifest.files.contains(where: { $0.path == codecWeight }) else {
            throw ModelMetadataPreflightError.rejected("Breeze speech codec is missing from the verified snapshot. Download this model again to complete its resources.")
        }
        let result = try await preflight.validate(metadataDirectory: directory)
        guard result.modelType == "breeze_tts", result.artifactRole == "tts" else {
            throw ModelMetadataPreflightError.rejected("Invalid Breeze TTS preflight result.")
        }
        manifest.compatibility = ModelSnapshotCompatibility(modelType: result.modelType, artifactRole: result.artifactRole,
            quantizationMode: result.quantization?.mode, quantizationBits: result.quantization?.bits,
            quantizationGroupSize: result.quantization?.groupSize)
        try ModelDownloadStore.atomicWrite(manifest, to: directory.appendingPathComponent(ModelSnapshotManifest.filename))
    }
}
