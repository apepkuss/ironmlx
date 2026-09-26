import Foundation
import Testing

@testable import IronMLXAppCore

private struct ImportPreflight: ModelMetadataPreflighting {
    func validate(metadataDirectory: URL) async throws -> ModelMetadataPreflightResult {
        try JSONDecoder().decode(ModelMetadataPreflightResult.self, from: Data(#"{"model_type":"llama","artifact_role":"base"}"#.utf8))
    }
}

@MainActor
@Test func dashboardRegistersModelImportHandlers() {
    #expect(DashboardBridge.handlerNames.contains("chooseModelImport"))
    #expect(DashboardBridge.handlerNames.contains("confirmModelImport"))
    #expect(DashboardBridge.handlerNames.contains("cancelModelImport"))
}

@Test func importsFlatFolderAsManagedStandaloneSnapshot() async throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("external/My Model", isDirectory: true)
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data(#"{"model_type":"llama"}"#.utf8).write(to: source.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
    try Data("model weights".utf8).write(to: source.appendingPathComponent("model.safetensors"))
    let managed = root.appendingPathComponent("managed", isDirectory: true)
    let service = ModelImportService(rootURL: managed, preflight: ImportPreflight())

    let preview = try await service.inspect(source)
    #expect(preview.provider == .standalone)
    #expect(preview.destination.path.contains("/models/standalone/"))
    let result = try await service.importModel(preview)

    #expect(result.repoID == preview.repoID)
    #expect(FileManager.default.fileExists(atPath: source.appendingPathComponent("model.safetensors").path))
    #expect(FileManager.default.fileExists(atPath: result.snapshot.appendingPathComponent(ModelSnapshotManifest.filename).path))
    let scanner = LocalModelScanner(rootURL: managed)
    let imported = try #require(scanner.scan().first)
    #expect(imported.id == result.repoID)
    #expect(imported.source == "standalone")
    #expect(imported.readiness?.isLoadable == true)
    #expect(scanner.resolveModelPath(for: result.repoID) == result.snapshot.path)
    let versions = try ModelVersionManagementService(rootURL: managed).versions(
        provider: .standalone, repoID: result.repoID, loadedModelPaths: []
    )
    #expect(versions.versions.count == 1)
    #expect(versions.versions.first?.isActive == true)
    let integrity = try await ModelIntegrityVerificationService(rootURL: managed).verify(repoID: result.repoID)
    #expect(integrity.state == "verified")
    let configStore = AppConfigStore(url: managed.appendingPathComponent("config/app_config.json"))
    let deletion = try LocalModelDeletionService(rootURL: managed, configStore: configStore)
        .deleteModels([result.repoID])
    #expect(deletion.deleted == [result.repoID])
    #expect(!FileManager.default.fileExists(atPath: result.snapshot.path))
}

@Test func importingHFCacheSnapshotCopiesBlobContents() async throws {
    let root = try temporaryDirectory()
    let repository = root.appendingPathComponent("models--org--sample", isDirectory: true)
    let blob = repository.appendingPathComponent("blobs/weights")
    let snapshot = repository.appendingPathComponent("snapshots/" + String(repeating: "a", count: 40), isDirectory: true)
    try FileManager.default.createDirectory(at: blob.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("weights from blob".utf8).write(to: blob)
    try Data(#"{"model_type":"llama"}"#.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: snapshot.appendingPathComponent("tokenizer.json"))
    try FileManager.default.createSymbolicLink(
        at: snapshot.appendingPathComponent("model.safetensors"),
        withDestinationURL: blob
    )
    let managed = root.appendingPathComponent("managed", isDirectory: true)
    let service = ModelImportService(rootURL: managed, preflight: ImportPreflight())

    let preview = try await service.inspect(snapshot)
    #expect(preview.displayName == "org/sample")
    let result = try await service.importModel(preview)
    try FileManager.default.removeItem(at: repository)

    #expect(try Data(contentsOf: result.snapshot.appendingPathComponent("model.safetensors")) == Data("weights from blob".utf8))
    #expect(LocalModelScanner(rootURL: managed).scan().first?.readiness?.isLoadable == true)
}

@Test func sourceMutationFailsWithoutPublishingImport() async throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("external/model", isDirectory: true)
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data(#"{"model_type":"llama"}"#.utf8).write(to: source.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
    try Data("original".utf8).write(to: source.appendingPathComponent("model.safetensors"))
    let managed = root.appendingPathComponent("managed", isDirectory: true)
    let service = ModelImportService(rootURL: managed, preflight: ImportPreflight())
    let preview = try await service.inspect(source)
    try Data("changed weight".utf8).write(to: source.appendingPathComponent("model.safetensors"))

    await #expect(throws: ModelImportError.self) { try await service.importModel(preview) }
    #expect(LocalModelScanner(rootURL: managed).scan().isEmpty)
}

@Test func importRejectsFileLinksOutsideSelectedModel() async throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source", isDirectory: true)
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data(#"{"model_type":"llama"}"#.utf8).write(to: source.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
    let outside = root.appendingPathComponent("outside.safetensors")
    try Data("outside".utf8).write(to: outside)
    try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("model.safetensors"),
        withDestinationURL: outside)
    let service = ModelImportService(rootURL: root.appendingPathComponent("managed"), preflight: ImportPreflight())

    await #expect(throws: ModelImportError.self) { try await service.inspect(source) }
    #expect(try Data(contentsOf: outside) == Data("outside".utf8))
}

@Test func verifiedIronMLXSnapshotPreservesProviderIdentity() async throws {
    let root = try temporaryDirectory()
    let sourceRoot = root.appendingPathComponent("original", isDirectory: true)
    let snapshot = try writeVerifiedTestSnapshot(
        root: sourceRoot,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "tokenizer.json": Data("{}".utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    let managed = root.appendingPathComponent("managed", isDirectory: true)
    let service = ModelImportService(rootURL: managed, preflight: ImportPreflight())
    let preview = try await service.inspect(snapshot)

    #expect(preview.provider == .huggingFace)
    #expect(preview.repoID == "org/model")
    let result = try await service.importModel(preview)
    #expect(LocalModelScanner(rootURL: managed).scan().first?.id == "org/model")
    #expect(result.alreadyPresent == false)
    #expect(try await service.importModel(preview).alreadyPresent == true)
}
