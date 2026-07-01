import Foundation

public struct LocalModel: Codable, Equatable, Sendable {
    public var id: String
    public var repoID: String
    public var source: String
    public var type: String
    public var sizeMB: Double
    public var loaded: Bool
    public var pinned: Bool
    public var maxPositionEmbeddings: Int?
    public var generationDefaults: BackendSamplingDefaults?
    public var mtp: LocalModelMtpInfo?

    public init(
        id: String,
        repoID: String,
        source: String,
        type: String = "llm",
        sizeMB: Double,
        loaded: Bool = false,
        pinned: Bool = false,
        maxPositionEmbeddings: Int? = nil,
        generationDefaults: BackendSamplingDefaults? = nil,
        mtp: LocalModelMtpInfo? = nil
    ) {
        self.id = id
        self.repoID = repoID
        self.source = source
        self.type = type
        self.sizeMB = sizeMB
        self.loaded = loaded
        self.pinned = pinned
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.generationDefaults = generationDefaults
        self.mtp = mtp
    }

    enum CodingKeys: String, CodingKey {
        case id
        case repoID = "repo_id"
        case source
        case type
        case sizeMB = "size_mb"
        case loaded
        case pinned
        case maxPositionEmbeddings = "max_position_embeddings"
        case generationDefaults = "generation_defaults"
        case mtp
    }
}

private enum LocalModelArtifactKind {
    case base
    case mtp
}

private struct LocalModelArtifact {
    var model: LocalModel
    var kind: LocalModelArtifactKind
    var path: URL
    var signature: MtpCompatibilitySignature?

    var mtpCandidate: LocalMtpCandidate {
        LocalMtpCandidate(
            id: model.id,
            repoID: model.repoID,
            source: model.source,
            sizeMB: model.sizeMB,
            path: path.path,
            reasonCode: signature == nil ? "mtp_invalid_config" : nil
        )
    }
}

private extension LocalModel {
    func artifact(
        kind: LocalModelArtifactKind,
        path: URL,
        signature: MtpCompatibilitySignature?
    ) -> LocalModelArtifact {
        LocalModelArtifact(model: self, kind: kind, path: path, signature: signature)
    }
}

private struct MtpCompatibilitySignature: Equatable {
    var supportsMtp: Bool
    var family: String
    var hiddenSize: Int?
    var intermediateSize: Int?
    var moeIntermediateSize: Int?
    var sharedExpertIntermediateSize: Int?
    var numExperts: Int?
    var numExpertsPerTok: Int?
    var normTopkProb: Bool
    var numAttentionHeads: Int?
    var numKeyValueHeads: Int?
    var headDim: Int?
    var vocabSize: Int?
    var rmsNormEps: Double?
    var attentionBias: Bool
    var tieWordEmbeddings: Bool
    var fullAttentionInterval: Int?
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var hasSlidingAttention: Bool
    var hasFullAttention: Bool

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.family == rhs.family
            && lhs.hiddenSize == rhs.hiddenSize
            && lhs.intermediateSize == rhs.intermediateSize
            && lhs.moeIntermediateSize == rhs.moeIntermediateSize
            && lhs.sharedExpertIntermediateSize == rhs.sharedExpertIntermediateSize
            && lhs.numExperts == rhs.numExperts
            && lhs.numExpertsPerTok == rhs.numExpertsPerTok
            && lhs.normTopkProb == rhs.normTopkProb
            && lhs.numAttentionHeads == rhs.numAttentionHeads
            && lhs.numKeyValueHeads == rhs.numKeyValueHeads
            && lhs.headDim == rhs.headDim
            && lhs.vocabSize == rhs.vocabSize
            && lhs.rmsNormEps == rhs.rmsNormEps
            && lhs.attentionBias == rhs.attentionBias
            && lhs.tieWordEmbeddings == rhs.tieWordEmbeddings
            && lhs.fullAttentionInterval == rhs.fullAttentionInterval
            && lhs.linearNumValueHeads == rhs.linearNumValueHeads
            && lhs.linearNumKeyHeads == rhs.linearNumKeyHeads
            && lhs.linearKeyHeadDim == rhs.linearKeyHeadDim
            && lhs.linearValueHeadDim == rhs.linearValueHeadDim
            && lhs.linearConvKernelDim == rhs.linearConvKernelDim
            && lhs.hasSlidingAttention == rhs.hasSlidingAttention
            && lhs.hasFullAttention == rhs.hasFullAttention
    }

    func isMtpCandidateCompatible(withBase base: Self) -> Bool {
        guard base.supportsMtp, !supportsMtp else {
            return false
        }
        return family == base.family
            && hiddenSize == base.hiddenSize
            && intermediateSize == base.intermediateSize
            && moeIntermediateSize == base.moeIntermediateSize
            && sharedExpertIntermediateSize == base.sharedExpertIntermediateSize
            && numExperts == base.numExperts
            && numExpertsPerTok == base.numExpertsPerTok
            && normTopkProb == base.normTopkProb
            && numAttentionHeads == base.numAttentionHeads
            && numKeyValueHeads == base.numKeyValueHeads
            && headDim == base.headDim
            && vocabSize == base.vocabSize
            && rmsNormEps == base.rmsNormEps
            && attentionBias == base.attentionBias
            && tieWordEmbeddings == base.tieWordEmbeddings
            && fullAttentionInterval == base.fullAttentionInterval
            && linearNumValueHeads == base.linearNumValueHeads
            && linearNumKeyHeads == base.linearNumKeyHeads
            && linearKeyHeadDim == base.linearKeyHeadDim
            && linearValueHeadDim == base.linearValueHeadDim
            && linearConvKernelDim == base.linearConvKernelDim
            && (!base.hasSlidingAttention || hasSlidingAttention)
            && (!base.hasFullAttention || hasFullAttention)
    }
}

public struct LocalModelMtpInfo: Codable, Equatable, Sendable {
    public var status: String
    public var enabled: Bool
    public var candidates: [LocalMtpCandidate]
    public var incompatibleCandidates: [LocalMtpCandidate]

    public init(
        status: String,
        enabled: Bool = false,
        candidates: [LocalMtpCandidate] = [],
        incompatibleCandidates: [LocalMtpCandidate] = []
    ) {
        self.status = status
        self.enabled = enabled
        self.candidates = candidates
        self.incompatibleCandidates = incompatibleCandidates
    }

    enum CodingKeys: String, CodingKey {
        case status
        case enabled
        case candidates
        case incompatibleCandidates = "incompatible_candidates"
    }
}

public struct LocalMtpCandidate: Codable, Equatable, Sendable {
    public var id: String
    public var repoID: String
    public var source: String
    public var sizeMB: Double
    public var path: String
    public var reasonCode: String?

    public init(
        id: String,
        repoID: String,
        source: String,
        sizeMB: Double,
        path: String,
        reasonCode: String? = nil
    ) {
        self.id = id
        self.repoID = repoID
        self.source = source
        self.sizeMB = sizeMB
        self.path = path
        self.reasonCode = reasonCode
    }

    enum CodingKeys: String, CodingKey {
        case id
        case repoID = "repo_id"
        case source
        case sizeMB = "size_mb"
        case path
        case reasonCode = "reason_code"
    }
}

public struct LocalModelScanner: Sendable {
    public var rootURL: URL

    public init(rootURL: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".ironmlx", isDirectory: true)) {
        self.rootURL = rootURL
    }

    public func scan(loadedModel: String? = nil) -> [LocalModel] {
        let loadedModels = loadedModel.map { Set([$0]) } ?? []
        return scan(loadedModels: loadedModels)
    }

    public func scan(loadedModels: Set<String>) -> [LocalModel] {
        scan(loadedModels: loadedModels, pinnedModels: [], mtpEnabledModels: [])
    }

    public func scan(loadedModels: Set<String>, mtpEnabledModels: Set<String>) -> [LocalModel] {
        scan(loadedModels: loadedModels, pinnedModels: [], mtpEnabledModels: mtpEnabledModels)
    }

    public func scan(loadedModels: Set<String>, pinnedModels: Set<String>, mtpEnabledModels: Set<String>) -> [LocalModel] {
        var artifacts: [LocalModelArtifact] = []
        artifacts += scanCacheDirectory(
            rootURL.appendingPathComponent("models", isDirectory: true),
            source: "hf",
            loadedModels: loadedModels,
            pinnedModels: pinnedModels
        )
        artifacts += scanCacheDirectory(
            rootURL.appendingPathComponent("models-ms", isDirectory: true),
            source: "ms",
            loadedModels: loadedModels,
            pinnedModels: pinnedModels
        )

        let mtpArtifacts = artifacts.filter { $0.kind == .mtp }
        let mtpCandidates = mtpArtifacts.map(\.mtpCandidate)
        return artifacts
            .filter { $0.kind == .base }
            .map { artifact in
                var model = artifact.model
                if let baseSignature = artifact.signature, baseSignature.supportsMtp {
                    let compatible = mtpArtifacts
                        .filter { $0.signature?.isMtpCandidateCompatible(withBase: baseSignature) == true }
                        .map(\.mtpCandidate)
                        .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
                    let incompatible = mtpCandidates
                        .filter { candidate in !compatible.contains(where: { $0.id == candidate.id }) }
                        .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
                    if !compatible.isEmpty {
                        let enabled = mtpEnabledModels.contains(model.id)
                        model.mtp = LocalModelMtpInfo(
                            status: enabled ? "enabled" : "available",
                            enabled: enabled,
                            candidates: compatible
                        )
                    } else if !incompatible.isEmpty {
                        model.mtp = LocalModelMtpInfo(status: "incompatible", incompatibleCandidates: incompatible)
                    }
                }
                return model
            }
            .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
    }

    public func resolveModelPath(for reference: String) -> String? {
        let direct = URL(fileURLWithPath: NSString(string: reference).expandingTildeInPath)
        if FileManager.default.fileExists(atPath: direct.path) {
            return direct.path
        }

        let dirName = "models--" + reference.replacingOccurrences(of: "/", with: "--")
        let roots = [
            rootURL.appendingPathComponent("models", isDirectory: true),
            rootURL.appendingPathComponent("models-ms", isDirectory: true),
        ]
        for root in roots {
            let snapshots = root
                .appendingPathComponent(dirName, isDirectory: true)
                .appendingPathComponent("snapshots", isDirectory: true)
            if let snapshot = firstCompleteSnapshot(in: snapshots) {
                return snapshot.path
            }
        }
        return nil
    }

    public func maxPositionEmbeddings(for reference: String) -> Int? {
        guard let path = resolveModelPath(for: reference) else {
            return nil
        }
        return maxPositionEmbeddings(in: URL(fileURLWithPath: path))
    }

    public func mtpCandidates(for reference: String) -> [LocalMtpCandidate] {
        scan(loadedModels: []).first(where: { $0.id == reference })?.mtp?.candidates ?? []
    }

    private func scanCacheDirectory(
        _ cacheURL: URL,
        source: String,
        loadedModels: Set<String>,
        pinnedModels: Set<String>
    ) -> [LocalModelArtifact] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: cacheURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return entries.compactMap { entry in
            let name = entry.lastPathComponent
            guard name.hasPrefix("models--") else {
                return nil
            }
            let id = String(name.dropFirst("models--".count))
                .replacingOccurrences(of: "--", with: "/")
            let snapshots = entry.appendingPathComponent("snapshots", isDirectory: true)
            guard let snapshot = firstCompleteSnapshot(in: snapshots) else {
                return nil
            }
            let sizeMB = directorySize(snapshot) / 1_048_576.0
            let loaded = loadedModels.contains(id) || loadedModels.contains(snapshot.path)
            let pinned = loaded && (pinnedModels.contains(id) || pinnedModels.contains(snapshot.path))
            let config = configJSON(in: snapshot)
            let kind = artifactKind(config)
            let signature = mtpCompatibilitySignature(config)
            return LocalModel(
                id: id,
                repoID: id,
                source: source,
                type: kind == .mtp ? "mtp" : "llm",
                sizeMB: sizeMB,
                loaded: loaded,
                pinned: pinned,
                maxPositionEmbeddings: maxPositionEmbeddings(in: snapshot),
                generationDefaults: generationDefaults(in: snapshot)
            ).artifact(kind: kind, path: snapshot, signature: signature)
        }
    }

    private func configJSON(in snapshot: URL) -> [String: Any]? {
        let config = snapshot.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }

    private func artifactKind(_ config: [String: Any]?) -> LocalModelArtifactKind {
        guard let modelType = config?["model_type"] as? String else {
            return .base
        }
        switch modelType {
        case "qwen3_5_mtp", "gemma4_assistant", "gemma4_unified_assistant":
            return .mtp
        default:
            return .base
        }
    }

    private func mtpCompatibilitySignature(_ config: [String: Any]?) -> MtpCompatibilitySignature? {
        guard let config,
              let modelType = config["model_type"] as? String,
              let text = config["text_config"] as? [String: Any] else {
            return nil
        }
        if modelType == "gemma4"
            || modelType == "gemma4_unified"
            || modelType == "gemma4_assistant"
            || modelType == "gemma4_unified_assistant" {
            return gemma4MtpCompatibilitySignature(config, modelType: modelType, text: text)
        }
        guard modelType == "qwen3_5" || modelType == "qwen3_5_moe" || modelType == "qwen3_5_mtp" else {
            return nil
        }
        return MtpCompatibilitySignature(
            supportsMtp: modelType == "qwen3_5" || modelType == "qwen3_5_moe",
            family: qwenMtpFamily(modelType: modelType, text: text),
            hiddenSize: intValue(text["hidden_size"]),
            intermediateSize: intValue(text["intermediate_size"]),
            moeIntermediateSize: intValue(text["moe_intermediate_size"]),
            sharedExpertIntermediateSize: intValue(text["shared_expert_intermediate_size"]),
            numExperts: intValue(text["num_experts"]),
            numExpertsPerTok: intValue(text["num_experts_per_tok"]),
            normTopkProb: boolValue(text["norm_topk_prob"]) ?? true,
            numAttentionHeads: intValue(text["num_attention_heads"]),
            numKeyValueHeads: intValue(text["num_key_value_heads"]),
            headDim: intValue(text["head_dim"]),
            vocabSize: intValue(text["vocab_size"]),
            rmsNormEps: doubleValue(text["rms_norm_eps"]),
            attentionBias: boolValue(text["attention_bias"]) ?? false,
            tieWordEmbeddings: boolValue(text["tie_word_embeddings"]) ?? false,
            fullAttentionInterval: intValue(text["full_attention_interval"]),
            linearNumValueHeads: intValue(text["linear_num_value_heads"]) ?? 0,
            linearNumKeyHeads: intValue(text["linear_num_key_heads"]) ?? 0,
            linearKeyHeadDim: intValue(text["linear_key_head_dim"]) ?? 0,
            linearValueHeadDim: intValue(text["linear_value_head_dim"]) ?? 0,
            linearConvKernelDim: intValue(text["linear_conv_kernel_dim"]) ?? 0,
            hasSlidingAttention: false,
            hasFullAttention: false
        )
    }

    private func qwenMtpFamily(modelType: String, text: [String: Any]) -> String {
        if modelType == "qwen3_5_moe" || intValue(text["num_experts"]) != nil {
            return "qwen3_5_moe"
        }
        return "qwen3_5"
    }

    private func gemma4MtpCompatibilitySignature(
        _ config: [String: Any],
        modelType: String,
        text: [String: Any]
    ) -> MtpCompatibilitySignature? {
        let isAssistant = modelType == "gemma4_assistant" || modelType == "gemma4_unified_assistant"
        if isAssistant {
            guard intValue(text["num_kv_shared_layers"]) == intValue(text["num_hidden_layers"]) else {
                return nil
            }
        }
        let family: String
        switch modelType {
        case "gemma4", "gemma4_assistant":
            family = "gemma4"
        case "gemma4_unified", "gemma4_unified_assistant":
            family = "gemma4_unified"
        default:
            return nil
        }
        let layerTypes = stringArray(text["layer_types"])
        return MtpCompatibilitySignature(
            supportsMtp: !isAssistant,
            family: family,
            hiddenSize: isAssistant ? intValue(config["backbone_hidden_size"]) : intValue(text["hidden_size"]),
            intermediateSize: nil,
            moeIntermediateSize: nil,
            sharedExpertIntermediateSize: nil,
            numExperts: nil,
            numExpertsPerTok: nil,
            normTopkProb: true,
            numAttentionHeads: nil,
            numKeyValueHeads: nil,
            headDim: nil,
            vocabSize: intValue(text["vocab_size"]),
            rmsNormEps: nil,
            attentionBias: false,
            tieWordEmbeddings: boolValue(text["tie_word_embeddings"]) ?? boolValue(config["tie_word_embeddings"]) ?? true,
            fullAttentionInterval: nil,
            linearNumValueHeads: 0,
            linearNumKeyHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelDim: 0,
            hasSlidingAttention: layerTypes.contains("sliding_attention"),
            hasFullAttention: layerTypes.contains("full_attention")
        )
    }

    private func maxPositionEmbeddings(in snapshot: URL) -> Int? {
        let config = snapshot.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let value = intValue(json["max_position_embeddings"]) {
            return value
        }
        if let textConfig = json["text_config"] as? [String: Any],
           let value = intValue(textConfig["max_position_embeddings"]) {
            return value
        }
        return nil
    }

    private func generationDefaults(in snapshot: URL) -> BackendSamplingDefaults? {
        let config = snapshot.appendingPathComponent("generation_config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let defaults = BackendSamplingDefaults(
            temperature: positiveDouble(json["temperature"]),
            topP: probability(json["top_p"]),
            topK: positiveIntValue(json["top_k"]),
            repetitionPenalty: positiveDouble(json["repetition_penalty"])
        )
        if defaults.temperature == nil,
           defaults.topP == nil,
           defaults.topK == nil,
           defaults.repetitionPenalty == nil {
            return nil
        }
        return defaults
    }

    private func intValue(_ value: Any?) -> Int? {
        if let int = value as? Int {
            return int
        }
        if let double = value as? Double {
            return Int(double)
        }
        if let string = value as? String {
            return Int(string.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    private func positiveIntValue(_ value: Any?) -> Int? {
        guard let parsed = intValue(value), parsed > 0 else {
            return nil
        }
        return parsed
    }

    private func doubleValue(_ value: Any?) -> Double? {
        if let double = value as? Double {
            return double
        }
        if let int = value as? Int {
            return Double(int)
        }
        if let string = value as? String {
            return Double(string.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    private func boolValue(_ value: Any?) -> Bool? {
        if let bool = value as? Bool {
            return bool
        }
        if let int = value as? Int {
            return int != 0
        }
        if let string = value as? String {
            switch string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true", "1", "yes":
                return true
            case "false", "0", "no":
                return false
            default:
                return nil
            }
        }
        return nil
    }

    private func stringArray(_ value: Any?) -> [String] {
        value as? [String] ?? []
    }

    private func positiveDouble(_ value: Any?) -> Double? {
        guard let parsed = doubleValue(value), parsed > 0 else {
            return nil
        }
        return parsed
    }

    private func probability(_ value: Any?) -> Double? {
        guard let parsed = positiveDouble(value), parsed <= 1 else {
            return nil
        }
        return parsed
    }

    private func firstCompleteSnapshot(in snapshotsURL: URL) -> URL? {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshotsURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return entries.sorted { $0.lastPathComponent < $1.lastPathComponent }
            .first { snapshotIsComplete($0) }
    }

    private func snapshotIsComplete(_ url: URL) -> Bool {
        let config = url.appendingPathComponent("config.json")
        guard FileManager.default.isReadableFile(atPath: config.path),
              let files = try? FileManager.default.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
        ) else {
            return false
        }
        let singleFile = files.contains { $0.lastPathComponent == "model.safetensors" }
        let index = url.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.isReadableFile(atPath: index.path) {
            let shards = safetensorsShards(from: index)
            let shardsComplete = !shards.isEmpty && shards.allSatisfy {
                FileManager.default.isReadableFile(atPath: url.appendingPathComponent($0).path)
            }
            return shardsComplete || singleFile
        }
        return singleFile || files.contains { $0.pathExtension == "safetensors" }
    }

    private func safetensorsShards(from index: URL) -> [String] {
        guard let data = try? Data(contentsOf: index),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: Any] else {
            return []
        }
        return Array(Set(weightMap.values.compactMap { $0 as? String })).sorted()
    }

    private func directorySize(_ url: URL) -> Double {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total: UInt64 = 0
        var countedTargets = Set<String>()
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(
                forKeys: [.fileSizeKey, .isDirectoryKey, .isSymbolicLinkKey]
            ), values.isDirectory != true else {
                continue
            }

            let sizeURL = values.isSymbolicLink == true ? fileURL.resolvingSymlinksInPath() : fileURL
            let countedKey = sizeURL.path
            if countedTargets.contains(countedKey) {
                continue
            }
            countedTargets.insert(countedKey)

            if let targetValues = try? sizeURL.resourceValues(forKeys: [.fileSizeKey]),
               let size = targetValues.fileSize {
                total += UInt64(size)
            }
        }
        return Double(total)
    }
}
