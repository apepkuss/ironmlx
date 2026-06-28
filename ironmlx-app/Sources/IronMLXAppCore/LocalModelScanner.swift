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

    public init(
        id: String,
        repoID: String,
        source: String,
        type: String = "llm",
        sizeMB: Double,
        loaded: Bool = false,
        pinned: Bool = false,
        maxPositionEmbeddings: Int? = nil,
        generationDefaults: BackendSamplingDefaults? = nil
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
        var models: [LocalModel] = []
        models += scanCacheDirectory(rootURL.appendingPathComponent("models", isDirectory: true), source: "hf", loadedModels: loadedModels)
        models += scanCacheDirectory(rootURL.appendingPathComponent("models-ms", isDirectory: true), source: "ms", loadedModels: loadedModels)
        return models.sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
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

    private func scanCacheDirectory(_ cacheURL: URL, source: String, loadedModels: Set<String>) -> [LocalModel] {
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
            return LocalModel(
                id: id,
                repoID: id,
                source: source,
                sizeMB: sizeMB,
                loaded: loaded,
                maxPositionEmbeddings: maxPositionEmbeddings(in: snapshot),
                generationDefaults: generationDefaults(in: snapshot)
            )
        }
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
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            if let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
               let size = values.fileSize {
                total += UInt64(size)
            }
        }
        return Double(total)
    }
}
