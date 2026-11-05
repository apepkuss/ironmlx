import Foundation

/// Select each component's own weight set. A root index describes the main
/// network, not independent codecs, encoders, or optional prediction heads.
struct ModelWeightSelection {
    let requiredPaths: Set<String>
    let optionalPaths: Set<String>
    let requiredMetadataPaths: Set<String>

    var paths: Set<String> { requiredPaths.union(optionalPaths) }

    init(directory: URL, inventory: Set<String>) throws {
        func safe(_ path: String) throws -> URL {
            try ModelSnapshotVerifier.safeFileURL(path: path, beneath: directory)
        }
        func object(_ path: String) throws -> [String: Any] {
            guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: safe(path))) as? [String: Any] else {
                throw ModelSnapshotVerificationError.manifestInvalid("Invalid component metadata: \(path).")
            }
            return value
        }
        func weights(in component: String, required: Bool) throws -> Set<String> {
            let prefix = component.isEmpty ? "" : component + "/"
            let index = prefix + "model.safetensors.index.json"
            if inventory.contains(index) {
                guard let map = try object(index)["weight_map"] as? [String: String], !map.isEmpty else {
                    throw ModelSnapshotVerificationError.manifestInvalid("Invalid weight_map in \(index).")
                }
                return try Set(map.values.map { value in
                    let path = prefix + value
                    _ = try safe(path)
                    guard path.hasSuffix(".safetensors") else {
                        throw ModelSnapshotVerificationError.manifestInvalid("Invalid weight path in \(index).")
                    }
                    return path
                })
            }
            let single = prefix + "model.safetensors"
            if inventory.contains(single) || required { return [single] }
            guard !component.isEmpty else { return [] }
            return Set(inventory.filter {
                $0.hasPrefix(prefix) && $0.hasSuffix(".safetensors")
                    && !$0.dropFirst(prefix.count).contains("/")
            })
        }

        let config = inventory.contains("config.json") ? try object("config.json") : [:]
        let modelType = config["model_type"] as? String
        let codec: String?
        switch modelType {
        case "breeze_tts": codec = "audio_tokenizer"
        case "qwen3_tts": codec = "speech_tokenizer"
        default: codec = nil
        }
        var metadata: Set<String> = []
        var required = try weights(in: "", required: false)
        if required.isEmpty {
            // Pipeline layouts such as Qwen-Image have no root weight set.
            required.formUnion(inventory.filter { $0.hasSuffix(".safetensors") })
        }
        var components = Set(inventory.filter {
            $0.hasSuffix("/config.json") || $0.hasSuffix("/model.safetensors.index.json")
        }.map { ($0 as NSString).deletingLastPathComponent })
        if let codec {
            components.insert(codec)
            metadata.insert(codec + "/config.json")
        }
        for component in components {
            required.formUnion(try weights(in: component, required: component == codec))
        }
        if let vision = config["optiq_vision"] as? [String: Any],
           let path = vision["sidecar"] as? String, !path.isEmpty {
            required.insert(path)
        }
        var optional: Set<String> = []
        if let path = config["mtp_file"] as? String, !path.isEmpty {
            _ = try safe(path)
            // Missing optional MTP weights do not prevent base-model downloads.
            if inventory.contains(path) { optional.insert(path) }
        }
        for path in required.union(optional) {
            _ = try safe(path)
            guard inventory.contains(path), path.hasSuffix(".safetensors") else {
                throw ModelSnapshotVerificationError.fileMissing(path)
            }
        }
        for path in metadata {
            _ = try safe(path)
            guard inventory.contains(path) else { throw ModelSnapshotVerificationError.fileMissing(path) }
        }
        requiredPaths = required
        optionalPaths = optional.subtracting(required)
        requiredMetadataPaths = metadata
    }
}
