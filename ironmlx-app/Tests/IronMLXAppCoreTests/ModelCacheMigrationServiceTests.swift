import CryptoKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Test func migrationValidatesThenRenamesWithoutDownloadingWeightsAndAddsMissingRef() async throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let commit = String(repeating: "1", count: 40)
    let config = Data(#"{"model_type":"llama"}"#.utf8)
    let weights = Data("weights".utf8)
    let legacy = root.appendingPathComponent("models/models--org--model", isDirectory: true)
    let snapshot = legacy.appendingPathComponent("snapshots/\(commit)", isDirectory: true)
    let blobs = legacy.appendingPathComponent("blobs", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
    try config.write(to: blobs.appendingPathComponent("config"))
    try weights.write(to: blobs.appendingPathComponent("weights"))
    try FileManager.default.createSymbolicLink(
        atPath: snapshot.appendingPathComponent("config.json").path,
        withDestinationPath: "../../blobs/config"
    )
    try FileManager.default.createSymbolicLink(
        atPath: snapshot.appendingPathComponent("model.safetensors").path,
        withDestinationPath: "../../blobs/weights"
    )

    let client = MigrationHTTPClient()
    client.responses["https://huggingface.co/api/models/org/model/revision/\(commit)?blobs=true"] =
        repositoryInfo(
            commit: commit,
            files: [
                ("config.json", config),
                ("model.safetensors", weights),
            ]
        )
    let service = ModelCacheMigrationService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: MigrationMetadataPreflight()
    )

    let result = try await service.migrate(repoID: repoID)

    #expect(result.status == "migrated")
    #expect(client.streamRequestCount == 0)
    #expect(!FileManager.default.fileExists(atPath: legacy.path))
    let destination = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: .huggingFace,
        repoID: repoID
    )
    #expect(try String(contentsOf: destination.appendingPathComponent("refs/main"), encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines) == commit)
    _ = try ModelSnapshotVerifier().verify(
        snapshot: destination.appendingPathComponent("snapshots/\(commit)", isDirectory: true),
        expectedProvider: .huggingFace,
        expectedRepoID: repoID
    )
}

@Test func migrationRefusesDamagedLocalFileAndDoesNotRename() async throws {
    let root = try temporaryDirectory()
    let repoID = "org/damaged"
    let commit = String(repeating: "2", count: 40)
    let expected = Data("expected".utf8)
    let legacy = root.appendingPathComponent("models/models--org--damaged", isDirectory: true)
    let snapshot = legacy.appendingPathComponent("snapshots/\(commit)", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(#"{"model_type":"llama"}"#.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("damaged".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
    try ModelDownloadStore.atomicWrite(Data((commit + "\n").utf8), to: legacy.appendingPathComponent("refs/main"))

    let client = MigrationHTTPClient()
    client.responses["https://huggingface.co/api/models/org/damaged/revision/\(commit)?blobs=true"] =
        repositoryInfo(
            commit: commit,
            files: [
                ("config.json", Data(#"{"model_type":"llama"}"#.utf8)),
                ("model.safetensors", expected),
            ]
        )
    let service = ModelCacheMigrationService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: MigrationMetadataPreflight()
    )

    await #expect(throws: ModelSnapshotVerificationError.self) {
        try await service.migrate(repoID: repoID)
    }
    #expect(FileManager.default.fileExists(atPath: legacy.path))
    #expect(!FileManager.default.fileExists(
        atPath: root.appendingPathComponent("models/huggingface/org--damaged").path
    ))
}

private struct MigrationMetadataPreflight: ModelMetadataPreflighting {
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        ModelMetadataPreflightResult(modelType: "llama", artifactRole: "base", quantization: nil)
    }
}

private final class MigrationHTTPClient: ModelDownloadHTTPClient, @unchecked Sendable {
    var responses: [String: Data] = [:]
    private(set) var streamRequestCount = 0
    private let lock = NSLock()

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        guard let url = request.url,
              let data = lock.withLock({ responses[url.absoluteString] })
        else {
            throw URLError(.fileDoesNotExist)
        }
        return (
            data,
            HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
        )
    }

    func stream(
        for _: URLRequest,
        onResponse _: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData _: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        lock.withLock { streamRequestCount += 1 }
        throw URLError(.dataNotAllowed)
    }
}

private func repositoryInfo(commit: String, files: [(String, Data)]) -> Data {
    let siblings = files.map { path, data in
        [
            "rfilename": path,
            "size": data.count,
            "lfs": [
                "size": data.count,
                "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            ],
        ] as [String: Any]
    }
    return try! JSONSerialization.data(
        withJSONObject: [
            "sha": commit,
            "siblings": siblings,
        ]
    )
}
