import Foundation
import Testing

@testable import IronMLXAppCore

/// Opt-in acceptance of runnable ASR snapshots through the real App download service.
@Test(.enabled(if: ProcessInfo.processInfo.environment["IRONMLX_QWEN3_ASR_LIVE_ROOT"] != nil), arguments: ["0.6B", "1.7B"])
func qwen3ASRLiveDownloadUsingAppService(size: String) async throws {
    let environment = ProcessInfo.processInfo.environment
    let root = URL(fileURLWithPath: try #require(environment["IRONMLX_QWEN3_ASR_LIVE_ROOT"]), isDirectory: true)
    let helper = URL(fileURLWithPath: try #require(environment["IRONMLX_QWEN3_ASR_LIVE_HELPER"]))
    let client = URLSessionModelDownloadHTTPClient()
    let service = ModelDownloadService(
        rootURL: root, httpClient: client,
        metadataPreflight: IronMLXModelMetadataPreflight(executableURL: helper),
        fileDownloader: ProviderModelFileDownloader(httpClient: client, executableURL: helper)
    )
    let repoID = "mlx-community/Qwen3-ASR-\(size)-8bit"
    let first = await service.downloadHuggingFace(repoID: repoID, token: nil)
    try #require(first.success, "\(first)")
    let status = try #require(await service.downloadStatuses().first { $0.repoID == repoID })
    #expect(status.status == "completed", "\(status)")
    let revision = try #require(status.commitSHA)
    let snapshot = try ModelDownloadStore(rootURL: root).snapshotURL(provider: .huggingFace, repoID: repoID, commitSHA: revision)
    let manifest = try ModelSnapshotVerifier().verify(snapshot: snapshot, expectedProvider: .huggingFace, expectedRepoID: repoID)
    #expect(manifest.compatibility.modelType == "qwen3_asr")
    #expect(manifest.compatibility.downloadOnly != true)
    #expect(manifest.compatibility.artifactRole == "asr")
    for path in ["vocab.json", "merges.txt", "tokenizer_config.json", "preprocessor_config.json", "model.safetensors"] {
        #expect(manifest.files.contains { $0.path == path })
    }
    let scanner = LocalModelScanner(rootURL: root)
    #expect(scanner.model(for: repoID)?.type == "asr")
    #expect(scanner.readiness(for: repoID)?.isLoadable == true)
    #expect(scanner.downloadedSnapshotDirectory(for: repoID) == snapshot)
    print("Verified Qwen3-ASR App download: \(snapshot.path), bytes=\(manifest.files.reduce(Int64(0)) { $0 + $1.size })")
}
