import Foundation

public struct LocalModelDeletionResult: Codable, Equatable {
    public var deleted: [String]
    public var clearedDefault: Bool

    enum CodingKeys: String, CodingKey {
        case deleted
        case clearedDefault = "cleared_default"
    }
}

public struct LocalModelDeletionService {
    public var rootURL: URL
    public var configStore: AppConfigStore
    private let fileManager: FileManager

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".ironmlx", isDirectory: true),
        configStore: AppConfigStore,
        fileManager: FileManager = .default
    ) {
        self.rootURL = rootURL
        self.configStore = configStore
        self.fileManager = fileManager
    }

    public func deleteModels(_ modelIDs: [String]) throws -> LocalModelDeletionResult {
        let normalizedIDs = modelIDs
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        var deleted: [String] = []
        for id in normalizedIDs {
            var removed = false
            for cacheSubdirectory in ["models", "models-ms"] {
                let directory = cacheDirectory(for: id, cacheSubdirectory: cacheSubdirectory)
                if fileManager.fileExists(atPath: directory.path) {
                    try fileManager.removeItem(at: directory)
                    removed = true
                }
            }
            if removed {
                deleted.append(id)
            }
        }

        var config = configStore.load()
        let clearedDefault = config.lastModel.map { normalizedIDs.contains($0) } ?? false
        if clearedDefault {
            config.lastModel = nil
            configStore.save(config)
        }

        return LocalModelDeletionResult(deleted: deleted, clearedDefault: clearedDefault)
    }

    private func cacheDirectory(for modelID: String, cacheSubdirectory: String) -> URL {
        rootURL
            .appendingPathComponent(cacheSubdirectory, isDirectory: true)
            .appendingPathComponent("models--" + modelID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }
}
