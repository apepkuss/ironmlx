import Foundation

/// Repository download contracts are separate from inference backend support.
/// IndexTTS uses named components and a tiktoken vocabulary, not an LLM shard set.
struct TTSModelDownloadProfile: Sendable {
    let weightPaths: Set<String>
    let auxiliaryPaths: Set<String>
    let externalResources: [String]

    static func inspect(directory: URL, files: [RemoteModelFile]) throws -> Self? {
        let manifestURL = directory.appendingPathComponent("model_manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL))
        guard let manifest = object as? [String: Any],
              manifest["model_family"] as? String == "IndexTTS" else { return nil }
        func reject(_ message: String) -> ModelMetadataPreflightError {
            .rejected("Invalid IndexTTS download metadata: \(message)")
        }
        let config = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json"))
        ) as? [String: Any]
        guard manifest["format_version"] as? Int == 1,
              manifest["model_version"] as? String == "2.5",
              config?["version"] as? Double == 2.5,
              let components = manifest["components"] as? [String: [String: Any]],
              let tokenizer = manifest["tokenizer"] as? [String: Any],
              tokenizer["type"] as? String == "tiktoken",
              let vocabulary = tokenizer["filename"] as? String,
              let dataset = config?["dataset"] as? [String: Any],
              dataset["tokenizer_type"] as? String == "tiktoken",
              dataset["bpe_model"] as? String == vocabulary,
              let dependencies = manifest["required_auxiliary_resources"] as? [String],
              !dependencies.isEmpty,
              dependencies.allSatisfy(ModelDownloadService.isCanonicalHuggingFaceRepoID)
        else { throw reject("unsupported version, tokenizer, or component schema") }
        let byPath = Dictionary(uniqueKeysWithValues: files.map { ($0.path, $0) })
        var weights: Set<String> = ["model.safetensors"] // w2v-BERT semantic front-end
        for name in ["gpt", "codec", "s2mel", "bigvgan"] {
            guard let component = components[name],
                  let path = component["file"] as? String,
                  let bytes = component["bytes"] as? Int64,
                  let remote = byPath[path], remote.isWeight, remote.size == bytes,
                  weights.insert(path).inserted
            else { throw reject("missing or inconsistent \(name) component") }
        }
        var auxiliary: Set<String> = [vocabulary, "config.yaml"]
        guard vocabulary.hasSuffix(".tiktoken") else { throw reject("invalid vocabulary") }
        for key in ["spk_matrix", "emo_matrix", "w2v_stat"] {
            guard let path = config?[key] as? String, path.hasSuffix(".pt") else {
                throw reject("missing \(key)")
            }
            auxiliary.insert(path)
        }
        // Preserve repository notices alongside the checkpoint, without executing auxiliary files.
        auxiliary.formUnion(files.filter {
            $0.path.lowercased() == "readme.md" || $0.path.lowercased().hasPrefix("license")
        }.map(\.path))
        for path in weights.union(auxiliary) {
            _ = try ModelSnapshotVerifier.safeFileURL(path: path, beneath: directory)
            guard let file = byPath[path], file.size > 0 else {
                throw reject("missing or empty file \(path)")
            }
        }
        return Self(weightPaths: weights, auxiliaryPaths: auxiliary, externalResources: dependencies)
    }

    var compatibility: ModelMetadataPreflightResult {
        ModelMetadataPreflightResult(modelType: "indextts2_5", artifactRole: "tts", quantization: nil)
    }

    /// Conservative queue reservation before the immutable manifest has been downloaded.
    static func isPotentialAuxiliary(_ file: RemoteModelFile) -> Bool {
        let name = file.path.lowercased()
        return name.hasSuffix(".tiktoken") || name.hasSuffix(".pt") || name.hasSuffix(".yaml")
            || name == "readme.md" || name.hasPrefix("license")
    }
}
