import CryptoKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Test func snapshotWithoutManifestIsUnverifiedAndCannotResolve() throws {
    let root = try temporaryDirectory()
    let repoID = "org/unverified"
    let commit = String(repeating: "d", count: 40)
    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: .huggingFace,
        repoID: repoID
    )
    let snapshot = repository.appendingPathComponent("snapshots/\(commit)", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(#"{"model_type":"llama"}"#.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    try ModelDownloadStore.atomicWrite(
        Data((commit + "\n").utf8),
        to: repository.appendingPathComponent("refs/main")
    )

    let scanner = LocalModelScanner(rootURL: root)
    let model = try #require(scanner.scan().first)

    #expect(model.readiness?.status == "unverified")
    #expect(model.readiness?.reasonCode == "snapshot_unverified")
    #expect(scanner.resolveModelPath(for: repoID) == nil)
}

@Test func legacyRuntimeDirectoryIsNotRecognized() throws {
    let root = try temporaryDirectory()
    let legacy = root
        .appendingPathComponent("models/models--org--legacy/snapshots/main", isDirectory: true)
    try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: legacy.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: legacy.appendingPathComponent("model.safetensors"))

    #expect(LocalModelScanner(rootURL: root).scan().isEmpty)
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: "org/legacy") == nil)
}

@Test func unmanifestedRuntimeFileMakesSnapshotUnverified() throws {
    let root = try temporaryDirectory()
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    try Data("extra".utf8).write(to: snapshot.appendingPathComponent("extra.safetensors"))

    #expect(throws: ModelSnapshotVerificationError.self) {
        try LocalModelScanner(rootURL: root).verifiedModelPath(for: "org/model")
    }
}

@Test func malformedIntegrityRecordCannotBypassCorruptState() throws {
    let root = try temporaryDirectory()
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    try Data("{invalid".utf8).write(
        to: snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
    )

    #expect(throws: ModelSnapshotVerificationError.self) {
        try LocalModelScanner(rootURL: root).verifiedModelPath(for: "org/model")
    }
}

@Test func resourcePreflightRejectsInsufficientDiskBeforeDownload() {
    let preflight = ModelResourcePreflight(
        weightBytes: 8 * 1_024 * 1_024 * 1_024,
        remainingDownloadBytes: 6 * 1_024 * 1_024 * 1_024,
        availableDiskBytes: 7 * 1_024 * 1_024 * 1_024,
        physicalMemoryBytes: 64 * 1_024 * 1_024 * 1_024
    )

    #expect(throws: ModelResourcePreflightError.self) {
        try preflight.validate()
    }
}

@Test func resourcePreflightRejectsInsufficientEstimatedMemory() {
    let preflight = ModelResourcePreflight(
        weightBytes: 32 * 1_024 * 1_024 * 1_024,
        remainingDownloadBytes: 0,
        availableDiskBytes: 64 * 1_024 * 1_024 * 1_024,
        physicalMemoryBytes: 32 * 1_024 * 1_024 * 1_024
    )

    #expect(throws: ModelResourcePreflightError.self) {
        try preflight.validate()
    }
}

@Test func interruptedJournalRecoversWithoutDeletingPartial() throws {
    let root = try temporaryDirectory()
    let store = ModelDownloadStore(rootURL: root)
    let commit = String(repeating: "e", count: 40)
    let journal = ModelDownloadJournal(
        provider: .huggingFace,
        repoID: "org/model",
        requestedRevision: "main",
        commitSHA: commit,
        phase: .downloading,
        progressBytes: 4,
        totalBytes: 10,
        currentFile: "model.safetensors"
    )
    try store.writeJournal(journal)
    let staging = try store.prepareStaging(provider: .huggingFace, repoID: "org/model", commitSHA: commit)
    let partial = staging.appendingPathComponent("model.safetensors.partial")
    try Data("1234".utf8).write(to: partial)

    let recovered = try #require(store.recoverInterruptedJournals().first)

    #expect(recovered.phase == .interrupted)
    #expect(recovered.progressBytes == 4)
    #expect(try Data(contentsOf: partial) == Data("1234".utf8))
}

@Test func repositoryLockRejectsConcurrentWriter() throws {
    let root = try temporaryDirectory()
    let store = ModelDownloadStore(rootURL: root)
    let first = try store.acquireRepositoryLock(provider: .huggingFace, repoID: "org/model")
    defer { withExtendedLifetime(first) {} }

    #expect(throws: ModelRepositoryLockError.self) {
        try store.acquireRepositoryLock(provider: .huggingFace, repoID: "org/model")
    }
}

@Test func immediateIntegrityVerificationMarksCorruptAndCanVerifyARepair() async throws {
    let root = try temporaryDirectory()
    let original = Data("weights".utf8)
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": original,
        ]
    )
    try Data("damaged".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    let service = ModelIntegrityVerificationService(rootURL: root)

    let corrupt = try await service.verify(repoID: "org/model")

    #expect(corrupt.state == "corrupt")
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: "org/model") == nil)

    try original.write(to: snapshot.appendingPathComponent("model.safetensors"))
    let repaired = try await service.verify(repoID: "org/model")

    #expect(repaired.state == "verified")
    #expect(repaired.verifiedAt != nil)
    #expect(repaired.verifiedAt?.contains("T") == true)
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: "org/model") == snapshot.path)
}

@Test func publishVerificationRehashesWhenValidatedFileIdentityChanges() throws {
    let root = try temporaryDirectory()
    let content = Data("weights".utf8)
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": content,
        ]
    )
    let verifier = ModelSnapshotVerifier()
    let manifest = try verifier.loadManifest(at: snapshot)
    let file = snapshot.appendingPathComponent("model.safetensors")
    let validation = ModelValidatedFile(
        path: "model.safetensors",
        sha256: SHA256.hash(data: content).map { String(format: "%02x", $0) }.joined(),
        identity: try ModelSnapshotVerifier.fileIdentity(of: file)
    )
    try Data("damaged".utf8).write(to: file)

    #expect(throws: ModelSnapshotVerificationError.self) {
        try verifier.verifyForPublish(
            snapshot: snapshot,
            manifest: manifest,
            validations: [validation]
        )
    }
}

@Test func cancellationOrNetworkFailureKeepsIdentityBoundPartial() async throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let content = Data("0123456789".utf8)
    let identity = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: String(repeating: "f", count: 40),
        path: "model.safetensors",
        expectedSize: Int64(content.count),
        expectedSHA256: SHA256.hash(data: content).map { String(format: "%02x", $0) }.joined(),
        etag: nil
    )
    let client = InterruptingHTTPClient(prefix: Data(content.prefix(4)))

    await #expect(throws: CancellationError.self) {
        try await ResumableFileDownloader(httpClient: client).download(
            ResumableDownloadRequest(
                urlRequest: URLRequest(url: URL(string: "https://example.test/model")!),
                identity: identity,
                destination: destination
            )
        )
    }

    let partial = destination.appendingPathExtension("partial")
    #expect(try Data(contentsOf: partial) == content.prefix(4))
    let stored = try JSONDecoder().decode(
        ModelPartialIdentity.self,
        from: Data(contentsOf: partial.appendingPathExtension("meta.json"))
    )
    #expect(stored.provider == identity.provider)
    #expect(stored.repoID == identity.repoID)
    #expect(stored.commitSHA == identity.commitSHA)
    #expect(stored.path == identity.path)
    #expect(stored.expectedSize == identity.expectedSize)
    #expect(stored.expectedSHA256 == identity.expectedSHA256)
    #expect(stored.etag == "\"stable\"")
}

@Test func completedPartialWithWrongChecksumIsRemoved() async throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let expected = Data("weights".utf8)
    let actual = Data("damaged".utf8)
    let identity = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: String(repeating: "f", count: 40),
        path: "model.safetensors",
        expectedSize: Int64(expected.count),
        expectedSHA256: SHA256.hash(data: expected).map { String(format: "%02x", $0) }.joined(),
        etag: "\"stable\""
    )

    await #expect(throws: ResumableDownloadError.self) {
        try await ResumableFileDownloader(httpClient: FixedBodyHTTPClient(body: actual)).download(
            ResumableDownloadRequest(
                urlRequest: URLRequest(url: URL(string: "https://example.test/model")!),
                identity: identity,
                destination: destination
            )
        )
    }

    let partial = destination.appendingPathExtension("partial")
    #expect(!FileManager.default.fileExists(atPath: partial.path))
    #expect(!FileManager.default.fileExists(
        atPath: partial.appendingPathExtension("meta.json").path
    ))
}

private struct InterruptingHTTPClient: ModelDownloadHTTPClient {
    var prefix: Data

    func data(for _: URLRequest) async throws -> (Data, HTTPURLResponse) {
        throw URLError(.unsupportedURL)
    }

    func stream(
        for request: URLRequest,
        onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        try await onResponse(
            HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["ETag": "\"stable\""]
            )!
        )
        try await onData(prefix)
        throw CancellationError()
    }
}

private struct FixedBodyHTTPClient: ModelDownloadHTTPClient {
    var body: Data

    func data(for _: URLRequest) async throws -> (Data, HTTPURLResponse) {
        throw URLError(.unsupportedURL)
    }

    func stream(
        for request: URLRequest,
        onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        try await onResponse(
            HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["ETag": "\"stable\""]
            )!
        )
        try await onData(body)
    }
}
