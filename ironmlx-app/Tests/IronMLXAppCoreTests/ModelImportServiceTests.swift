import Foundation
import Testing

@testable import IronMLXAppCore

private struct ImportPreflight: ModelMetadataPreflighting {
    func validate(metadataDirectory: URL) async throws -> ModelMetadataPreflightResult {
        try JSONDecoder().decode(ModelMetadataPreflightResult.self, from: Data(#"{"model_type":"llama","artifact_role":"base"}"#.utf8))
    }
}

private struct RejectingImportPreflight: ModelMetadataPreflighting {
    var message = "Error: unsupported model_type: qwen4_exp (expected 'llama')"

    func validate(metadataDirectory: URL) async throws -> ModelMetadataPreflightResult {
        throw ModelMetadataPreflightError.rejected(message)
    }
}

private func writeImportSource(in root: URL, config: String = #"{"model_type":"qwen4_exp"}"#) throws -> URL {
    let source = root.appendingPathComponent("external/model", isDirectory: true)
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data(config.utf8).write(to: source.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
    try Data("weights".utf8).write(to: source.appendingPathComponent("model.safetensors"))
    return source
}

@Test(arguments: ["Error: unsupported model_type: qwen4_exp (expected 'llama')",
                  "Error: unsupported quantization.mode: future_mode",
                  "Error: unsupported quantization.bits: 3",
                  "Error: unsupported quantization.group_size: 16"])
func unsupportedLocalModelsImportOnlyAfterExplicitConfirmation(message: String) async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = try writeImportSource(in: root)
    let managed = root.appendingPathComponent("managed")
    let service = ModelImportService(rootURL: managed, preflight: RejectingImportPreflight(message: message))
    let preview = try await service.inspect(source)
    #expect(preview.downloadOnly)
    #expect(preview.runtimeSupportError == message)
    await #expect(throws: ModelImportError.self) { try await service.importModel(preview) }
    #expect(!FileManager.default.fileExists(atPath: preview.destination.path))

    let result = try await service.importModel(preview, allowUnsupported: true)
    let manifest = try ModelSnapshotVerifier().verify(snapshot: result.snapshot)
    #expect(manifest.provider == .standalone)
    #expect(manifest.compatibility.downloadOnly == true)
    #expect(manifest.compatibility.runtimeSupportError == message)
    let scanner = LocalModelScanner(rootURL: managed)
    let imported = try #require(scanner.scan().first)
    #expect(imported.readiness?.status == "unsupported")
    #expect(imported.readiness?.reasonCode == "download_only_model")
    #expect(imported.readiness?.isLoadable == false)
    #expect(scanner.resolveModelPath(for: result.repoID) == nil)
    #expect(scanner.resolveModelPath(for: result.snapshot.path) == nil)
    #expect(try await ModelIntegrityVerificationService(rootURL: managed).verify(repoID: result.repoID).state == "verified")
    #expect(try Data(contentsOf: source.appendingPathComponent("model.safetensors")) == Data("weights".utf8))
}

@Test(arguments: ["Invalid config.json", "Missing tokenizer", "unsupported architecture",
                  "Error: DFlash2 config missing integer num_target_layers",
                  "Error: DFlash2 target_layer_ids must be unique and strictly increasing"])
func localImportDoesNotOverrideMalformedOrUnclassifiedMetadata(message: String) async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = try writeImportSource(in: root)
    let service = ModelImportService(rootURL: root.appendingPathComponent("managed"),
                                     preflight: RejectingImportPreflight(message: message))
    await #expect(throws: ModelMetadataPreflightError.self) { try await service.inspect(source) }
}

@Test(arguments: ["tokenizer", "weights", "shard", "index", "config"])
func fileOnlyImportStillRequiresCompleteModelFiles(gate: String) async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = try writeImportSource(in: root)
    switch gate {
    case "tokenizer": try FileManager.default.removeItem(at: source.appendingPathComponent("tokenizer.json"))
    case "weights": try FileManager.default.removeItem(at: source.appendingPathComponent("model.safetensors"))
    case "shard":
        try Data(#"{"weight_map":{"tensor":"missing.safetensors"}}"#.utf8)
            .write(to: source.appendingPathComponent("model.safetensors.index.json"))
    case "index":
        try Data(#"{"weight_map":{}}"#.utf8)
            .write(to: source.appendingPathComponent("model.safetensors.index.json"))
    default: try Data("invalid JSON".utf8).write(to: source.appendingPathComponent("config.json"))
    }
    let managed = root.appendingPathComponent("managed")
    let service = ModelImportService(rootURL: managed, preflight: RejectingImportPreflight())
    await #expect(throws: (any Error).self) { try await service.inspect(source) }
    #expect(LocalModelScanner(rootURL: managed).scan().isEmpty)
}

@Test func fileOnlyImportPreservesDFlash2DraftRoleWithoutTokenizer() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = try writeImportSource(in: root,
        config: #"{"model_type":"qwen3","architectures":["DFlash2DraftModel"]}"#)
    try FileManager.default.removeItem(at: source.appendingPathComponent("tokenizer.json"))
    let managed = root.appendingPathComponent("managed")
    let service = ModelImportService(rootURL: managed, preflight: RejectingImportPreflight(
        message: "Error: unsupported DFlash2 configuration: target_layer_ids count 8 differs from draft layer count 6"))
    let preview = try await service.inspect(source)
    #expect(preview.compatibility.artifactRole == ModelArtifactRole.dflash2Drafter)
    let result = try await service.importModel(preview, allowUnsupported: true)
    let manifest = try ModelSnapshotVerifier().verify(snapshot: result.snapshot)
    #expect(manifest.compatibility.artifactRole == ModelArtifactRole.dflash2Drafter)
    #expect(manifest.compatibility.downloadOnly == true)
    #expect(LocalModelScanner(rootURL: managed).resolveModelPath(for: result.repoID) == nil)
}

@Test func fileOnlyImportChecksExistingSnapshotHashesBeforePublishing() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let snapshot = try writeVerifiedTestSnapshot(root: root.appendingPathComponent("original"), repoID: "org/model",
        files: ["config.json": Data(#"{"model_type":"qwen4_exp"}"#.utf8),
                "tokenizer.json": Data("{}".utf8), "model.safetensors": Data("weights".utf8)])
    // Keep the size unchanged so structural verification alone cannot detect damage.
    try Data("damaged".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    let managed = root.appendingPathComponent("managed")
    let service = ModelImportService(rootURL: managed, preflight: RejectingImportPreflight())
    let preview = try await service.inspect(snapshot)
    await #expect(throws: ModelImportError.self) { try await service.importModel(preview, allowUnsupported: true) }
    #expect(LocalModelScanner(rootURL: managed).scan().isEmpty)
}

@Test(arguments: ["nemotron_diarization", "qwen3_asr"])
func fileOnlyAudioImportsUseTheSameRequiredMetadataAsDownloads(modelType: String) async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = try writeImportSource(in: root, config: "{\"model_type\":\"\(modelType)\"}")
    try FileManager.default.removeItem(at: source.appendingPathComponent("tokenizer.json"))
    if modelType == "qwen3_asr" {
        for path in ["tokenizer_config.json", "preprocessor_config.json", "vocab.json", "merges.txt"] {
            try Data("{}".utf8).write(to: source.appendingPathComponent(path))
        }
    }
    let managed = root.appendingPathComponent("managed")
    let service = ModelImportService(rootURL: managed, preflight: RejectingImportPreflight(
        message: "Error: unsupported quantization.mode: future_mode"))
    let preview = try await service.inspect(source)
    let result = try await service.importModel(preview, allowUnsupported: true)
    #expect(try ModelSnapshotVerifier().verify(snapshot: result.snapshot).compatibility.downloadOnly == true)
    if modelType == "qwen3_asr" {
        try FileManager.default.removeItem(at: source.appendingPathComponent("preprocessor_config.json"))
        await #expect(throws: ModelImportError.self) { try await service.inspect(source) }
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
