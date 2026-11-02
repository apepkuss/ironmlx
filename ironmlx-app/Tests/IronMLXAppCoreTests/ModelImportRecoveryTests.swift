import Foundation
import Testing

@testable import IronMLXAppCore

private func interruptedImportFixture(
    root: URL,
    provider: ModelRepositoryProvider = .standalone,
    fileOnly: Bool = false
) throws -> (service: ModelImportService, staging: URL, final: URL, ref: URL) {
    let repoID = provider == .standalone ? "standalone/recoverable-model" : "org/recoverable-model"
    let final = try writeVerifiedTestSnapshot(root: root, provider: provider, repoID: repoID,
        files: ["config.json": Data((fileOnly ? #"{"model_type":"qwen4_exp"}"# : #"{"model_type":"llama"}"#).utf8),
                "tokenizer.json": Data("{}".utf8), "model.safetensors": Data("weights".utf8)])
    if fileOnly {
        var manifest = try ModelSnapshotVerifier().loadManifest(at: final)
        manifest.compatibility.modelType = "qwen4_exp"
        manifest.compatibility.artifactRole = "download_only"
        manifest.compatibility.downloadOnly = true
        manifest.compatibility.runtimeSupportError = "Error: unsupported model_type: qwen4_exp"
        try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: final)
    }
    let repository = final.deletingLastPathComponent().deletingLastPathComponent()
    let ref = repository.appendingPathComponent("refs/\(provider.mutableRevision)")
    try FileManager.default.removeItem(at: ref)
    let staging = repository.appendingPathComponent(".imports/\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: staging.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.moveItem(at: final, to: staging)
    return (ModelImportService(rootURL: root), staging, final, ref)
}

@Test(arguments: ModelRepositoryProvider.allCases, [false, true])
func interruptedImportRecoveryVerifiesAndPublishesWithoutRecopying(provider: ModelRepositoryProvider, fileOnly: Bool) async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try interruptedImportFixture(root: root, provider: provider, fileOnly: fileOnly)
    let before = try ModelSnapshotVerifier.fileIdentity(of: fixture.staging.appendingPathComponent("model.safetensors"))
    let pending = try #require(fixture.service.interruptedImports().first)
    let result = try await fixture.service.recoverImport(pending)
    #expect(result.snapshot == fixture.final)
    #expect(!FileManager.default.fileExists(atPath: fixture.staging.path))
    #expect(try ModelSnapshotVerifier.fileIdentity(of: fixture.final.appendingPathComponent("model.safetensors")) == before)
    let manifest = try ModelSnapshotVerifier().verify(snapshot: fixture.final)
    #expect(try String(contentsOf: fixture.ref, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) == manifest.commitSHA)
    #expect(try ModelSnapshotVerifier().loadIntegrityRecord(at: fixture.final).state == .verified)
    #expect(LocalModelScanner(rootURL: root).readiness(for: fixture.final.path)?.isLoadable == !fileOnly)
    #expect(try fixture.service.interruptedImports().isEmpty)
}

@Test func interruptedImportRecoveryRepairsRefAfterSnapshotRename() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try interruptedImportFixture(root: root, fileOnly: true)
    let manifest = try ModelSnapshotVerifier().loadManifest(at: fixture.staging)
    let publication = fixture.staging.appendingPathExtension("json")
    try ModelDownloadStore.atomicWrite(manifest, to: publication)
    try FileManager.default.moveItem(at: fixture.staging, to: fixture.final)
    let pending = try #require(fixture.service.interruptedImports().first)
    let result = try await fixture.service.recoverImport(pending)
    #expect(result.alreadyPresent)
    #expect(try String(contentsOf: fixture.ref, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) == manifest.commitSHA)
    #expect(!FileManager.default.fileExists(atPath: publication.path))
    #expect(try fixture.service.interruptedImports().isEmpty)
}

@Test func interruptedImportRecoveryRejectsCorruptionAndPreservesTheCopy() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try interruptedImportFixture(root: root, fileOnly: true)
    try Data("damaged".utf8).write(to: fixture.staging.appendingPathComponent("model.safetensors"))
    let pending = try #require(fixture.service.interruptedImports().first)
    await #expect(throws: ModelSnapshotVerificationError.self) { try await fixture.service.recoverImport(pending) }
    #expect(FileManager.default.fileExists(atPath: fixture.staging.path))
    #expect(!FileManager.default.fileExists(atPath: fixture.ref.path))
    #expect(!FileManager.default.fileExists(atPath: fixture.final.path))
}

@Test func interruptedImportRecoveryReportsAnIncompleteCopyWithoutDeletingIt() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try interruptedImportFixture(root: root)
    try FileManager.default.removeItem(at: fixture.staging.appendingPathComponent(ModelSnapshotManifest.filename))
    let pending = try #require(fixture.service.interruptedImports().first)
    await #expect(throws: ModelImportError.self) { try await fixture.service.recoverImport(pending) }
    #expect(FileManager.default.fileExists(atPath: fixture.staging.appendingPathComponent("model.safetensors").path))
    #expect(!FileManager.default.fileExists(atPath: fixture.ref.path))
}

@Test func interruptedImportRecoveryCanBeCancelledWithoutLosingTheCopy() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try interruptedImportFixture(root: root)
    let pending = try #require(fixture.service.interruptedImports().first)
    let task = Task {
        try await fixture.service.recoverImport(pending) { _ in
            withUnsafeCurrentTask { $0?.cancel() }
        }
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(FileManager.default.fileExists(atPath: fixture.staging.path))
    #expect(!FileManager.default.fileExists(atPath: fixture.ref.path))
}

@Test func interruptedImportRecoveryRejectsRedirectedDirectories() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try interruptedImportFixture(root: root)
    let pending = try #require(fixture.service.interruptedImports().first)
    let external = root.appendingPathComponent("external")
    try FileManager.default.moveItem(at: fixture.staging, to: external)
    try FileManager.default.createSymbolicLink(at: fixture.staging, withDestinationURL: external)
    await #expect(throws: ModelImportError.self) { try await fixture.service.recoverImport(pending) }
    #expect(FileManager.default.fileExists(atPath: external.appendingPathComponent("model.safetensors").path))
    #expect(!FileManager.default.fileExists(atPath: fixture.ref.path))
}
