import Foundation
import Testing

@testable import IronMLXAppCore

/// Opt-in acceptance of runnable diarization snapshots using the real App downloader.
@Test(.enabled(if: ProcessInfo.processInfo.environment["IRONMLX_NEMOTRON_DOWNLOAD_ROOT"] != nil))
func nemotronDiarizationLiveDownloadUsingAppService() async throws {
    let environment = ProcessInfo.processInfo.environment
    let root = URL(fileURLWithPath: try #require(environment["IRONMLX_NEMOTRON_DOWNLOAD_ROOT"]), isDirectory: true)
    let helper = URL(fileURLWithPath: try #require(environment["IRONMLX_NEMOTRON_DOWNLOAD_HELPER"]))
    let client = URLSessionModelDownloadHTTPClient()
    let service = ModelDownloadService(
        rootURL: root, httpClient: client,
        metadataPreflight: IronMLXModelMetadataPreflight(executableURL: helper),
        fileDownloader: ProviderModelFileDownloader(httpClient: client, executableURL: helper)
    )
    let repoID = "mlx-community/Nemotron-3-Diarization"
    let first = await service.downloadHuggingFace(repoID: repoID, token: nil)
    if !first.success {
        try #require(first.code == "unsupported_model_metadata", "\(first)")
        let resume = await service.resumeDownload(provider: .huggingFace, repoID: repoID, downloadOnly: true)
        try #require(resume.success, "\(resume)")
        let deadline = ContinuousClock.now + .seconds(300)
        while ContinuousClock.now < deadline {
            let queue = await service.downloadQueueSnapshot()
            if queue.activeCount == 0 && queue.queuedCount == 0 { break }
            try await Task.sleep(for: .milliseconds(100))
        }
    }
    let status = try #require(await service.downloadStatuses().first { $0.repoID == repoID })
    try #require(status.status == "completed", "\(status)")
    #expect(status.downloadOnly != true)
    let revision = try #require(status.commitSHA)
    let snapshot = try ModelDownloadStore(rootURL: root).snapshotURL(provider: .huggingFace, repoID: repoID, commitSHA: revision)
    let manifest = try ModelSnapshotVerifier().verify(snapshot: snapshot, expectedProvider: .huggingFace, expectedRepoID: repoID)
    #expect(manifest.compatibility.modelType == "nemotron_diarization")
    #expect(manifest.compatibility.downloadOnly != true)
    #expect(manifest.compatibility.runtimeSupportError == nil)
    #expect(manifest.files.contains { $0.path == "config.json" })
    #expect(manifest.files.contains { $0.path == "model.safetensors" })
    #expect(!manifest.files.contains { $0.path == "tokenizer.json" })
    let scanner = LocalModelScanner(rootURL: root)
    #expect(scanner.downloadedSnapshotDirectory(for: repoID) == snapshot)
    #expect(manifest.compatibility.artifactRole == "diarization")
    #expect(scanner.readiness(for: repoID)?.isLoadable == true)
    #expect(scanner.resolveModelPath(for: repoID) == snapshot.path)
    print("Verified runnable Nemotron diarization download: \(snapshot.path), bytes=\(manifest.files.reduce(Int64(0)) { $0 + $1.size })")
}
