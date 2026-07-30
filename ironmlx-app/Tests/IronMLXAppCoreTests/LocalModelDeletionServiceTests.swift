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
    let hfDirectory = try createCachedModel(root: root, provider: .huggingFace, repoID: "mlx-community/Tiny-4bit")
    let msDirectory = try createCachedModel(root: root, provider: .modelScope, repoID: "mlx-community/Tiny-4bit")

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
    #expect(DashboardBridge.handlerNames.contains("verifyModelIntegrity"))
}

private func createCachedModel(root: URL, provider: ModelRepositoryProvider, repoID: String) throws -> URL {
    _ = try writeVerifiedTestSnapshot(
        root: root,
        provider: provider,
        repoID: repoID,
        files: [
            "config.json": Data("{}".utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    return try ModelRepositoryLayout.repositoryRoot(rootURL: root, provider: provider, repoID: repoID)
}
