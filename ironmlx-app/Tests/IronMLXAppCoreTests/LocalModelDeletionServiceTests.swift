import Foundation
import Testing

@testable import IronMLXAppCore

@Test func deletingLocalModelsRemovesCacheDirectoriesAndClearsDeletedDefault() throws {
    let root = try temporaryDirectory()
    let configURL = root
        .appendingPathComponent("config", isDirectory: true)
        .appendingPathComponent("app_config.json")
    let configStore = AppConfigStore(url: configURL)
    configStore.save(AppConfig(
        defaultModel: "mlx-community/Tiny-4bit",
        loadedModels: [
            "mlx-community/Tiny-4bit",
            "mlx-community/Other-4bit",
        ]
    ))
    let hfDirectory = try createCachedModel(root: root, cacheSubdirectory: "models", repoID: "mlx-community/Tiny-4bit")
    let msDirectory = try createCachedModel(root: root, cacheSubdirectory: "models-ms", repoID: "mlx-community/Tiny-4bit")

    let result = try LocalModelDeletionService(rootURL: root, configStore: configStore)
        .deleteModels(["mlx-community/Tiny-4bit"])

    #expect(result.deleted == ["mlx-community/Tiny-4bit"])
    #expect(!FileManager.default.fileExists(atPath: hfDirectory.path))
    #expect(!FileManager.default.fileExists(atPath: msDirectory.path))
    #expect(configStore.load().defaultModel == "mlx-community/Other-4bit")
    #expect(configStore.load().loadedModels == ["mlx-community/Other-4bit"])
}

@MainActor
@Test func dashboardBridgeRegistersModelDeletionHandler() {
    #expect(DashboardBridge.handlerNames.contains("deleteModels"))
}

private func createCachedModel(root: URL, cacheSubdirectory: String, repoID: String) throws -> URL {
    let directory = root
        .appendingPathComponent(cacheSubdirectory, isDirectory: true)
        .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    let snapshot = directory
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    return directory
}
