import Foundation
import Testing

@testable import IronMLXAppCore

/// Opt-in network acceptance against the real Hugging Face checkpoint and a Release Bundle helper.
@Test func layaRealHuggingFaceDownloadAndSnapshotRecognition() async throws {
    guard let helperPath = ProcessInfo.processInfo.environment["IRONMLX_LAYA_LIVE_HELPER"],
          let rootPath = ProcessInfo.processInfo.environment["IRONMLX_LAYA_LIVE_ROOT"]
    else { return }
    let helper = URL(fileURLWithPath: helperPath)
    let root = URL(fileURLWithPath: rootPath, isDirectory: true)
    let client = URLSessionModelDownloadHTTPClient()
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: IronMLXModelMetadataPreflight(executableURL: helper),
        fileDownloader: ProviderModelFileDownloader(httpClient: client, executableURL: helper),
        telemetryLogger: { _ in }
    )
    let repoID = "aac6fef/laya-multilingual-mlx"
    let result = await service.downloadHuggingFace(repoID: repoID, token: nil)
    #expect(result.success, "\(result)")

    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root, provider: .huggingFace, repoID: repoID
    )
    let revision = try String(
        contentsOf: repository.appendingPathComponent("refs/main"), encoding: .utf8
    ).trimmingCharacters(in: .whitespacesAndNewlines)
    #expect(revision == "f2b4faf51023039425946074e2cf1361d2db11d5")
    let snapshot = repository.appendingPathComponent("snapshots/\(revision)")
    let manifest = try ModelSnapshotVerifier().verify(
        snapshot: snapshot, expectedProvider: .huggingFace, expectedRepoID: repoID
    )
    #expect(manifest.compatibility.artifactRole == "decision")
    #expect(manifest.files.contains { $0.path == "model.safetensors" })
    #expect(manifest.files.contains { $0.path == "tokenizer/tokenizer.json" })
    let scanner = LocalModelScanner(rootURL: root)
    let model = try #require(scanner.scan().first { $0.id == repoID })
    #expect(model.type == "decision")
    #expect(model.capabilities?.runtimeKind == "decision")
    #expect(model.readiness?.isLoadable == true)
    print("Laya live App download snapshot: \(snapshot.path)")
}
