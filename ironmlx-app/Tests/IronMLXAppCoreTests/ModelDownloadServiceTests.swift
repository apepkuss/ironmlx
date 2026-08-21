import CryptoKit
import Foundation
import Testing

@testable import IronMLXAppCore

private let testCommit = String(repeating: "a", count: 40)

@Test func urlSessionHTTPClientStreamsDelegateDataBlocksWithoutByteIteration() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ChunkedDownloadURLProtocol.self]
    let client = URLSessionModelDownloadHTTPClient(sessionConfiguration: configuration)
    let collector = StreamingResponseCollector()

    try await client.stream(
        for: URLRequest(url: URL(string: "https://stream.test/model.safetensors")!),
        onResponse: { response in
            await collector.record(statusCode: response.statusCode)
        },
        onData: { data in
            try Task.checkCancellation()
            await collector.append(data)
        }
    )

    let result = await collector.result()
    #expect(result.statusCode == 200)
    #expect(result.body == ChunkedDownloadURLProtocol.body)
    #expect(result.callbackCount > 0)
}

@Test func urlSessionHTTPClientStopsAfterChunkHandlerFailure() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ChunkedDownloadURLProtocol.self]
    let client = URLSessionModelDownloadHTTPClient(sessionConfiguration: configuration)
    let collector = StreamingResponseCollector()

    await #expect(throws: StreamingResponseTestError.self) {
        try await client.stream(
            for: URLRequest(url: URL(string: "https://stream.test/model.safetensors")!),
            onResponse: { _ in },
            onData: { data in
                await collector.append(data)
                throw StreamingResponseTestError.rejectedChunk
            }
        )
    }

    let result = await collector.result()
    #expect(result.callbackCount == 1)
}

@Test func urlSessionHTTPClientCancelsAnActiveDelegateStream() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [SuspendedDownloadURLProtocol.self]
    let client = URLSessionModelDownloadHTTPClient(sessionConfiguration: configuration)
    let firstChunk = StreamingSignal()
    let download = Task {
        try await client.stream(
            for: URLRequest(url: URL(string: "https://stream.test/large-model.safetensors")!),
            onResponse: { _ in },
            onData: { _ in
                await firstChunk.signal()
            }
        )
    }

    await firstChunk.wait()
    download.cancel()

    await #expect(throws: CancellationError.self) {
        try await download.value
    }
}

@Test func rustHuggingFaceTransferCancellationTerminatesChildProcess() async throws {
    let root = try temporaryDirectory()
    let executable = root.appendingPathComponent("fake-ironmlx")
    try Data("""
    #!/bin/sh
    trap 'exit 0' TERM INT
    printf '%s\\n' '{"type":"progress","bytes":1,"total":10}'
    while true; do sleep 1; done
    """.utf8).write(to: executable)
    try FileManager.default.setAttributes(
        [.posixPermissions: NSNumber(value: Int16(0o700))],
        ofItemAtPath: executable.path
    )
    let destination = root.appendingPathComponent("model.safetensors")
    let request = ResumableDownloadRequest(
        urlRequest: URLRequest(url: URL(string: "https://huggingface.co/org/model/resolve/\(testCommit)/model.safetensors")!),
        identity: ModelPartialIdentity(
            provider: .huggingFace,
            repoID: "org/model",
            commitSHA: testCommit,
            path: "model.safetensors",
            expectedSize: 10,
            expectedSHA256: String(repeating: "b", count: 64),
            etag: nil
        ),
        destination: destination
    )
    let started = StreamingSignal()
    let download = Task {
        try await RustHuggingFaceFileDownloader(executableURL: executable).download(
            request,
            progress: { bytes in
                if bytes > 0 {
                    await started.signal()
                }
            }
        )
    }

    await started.wait()
    download.cancel()

    await #expect(throws: CancellationError.self) {
        try await download.value
    }
    #expect(!FileManager.default.fileExists(atPath: destination.path))
}

@Test func rustHuggingFacePartialReportsOnlyIdentityBoundCommittedBytes() throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let cache = destination.appendingPathExtension("hf-transfer")
    let etag = String(repeating: "c", count: 64)
    let identity = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: testCommit,
        path: "model.safetensors",
        expectedSize: 10,
        expectedSHA256: String(repeating: "b", count: 64),
        etag: nil
    )
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    try Data("""
    {
      "version": 1,
      "provider": "huggingface",
      "repo_id": "org/model",
      "commit_sha": "\(testCommit)",
      "path": "model.safetensors",
      "expected_size": 10,
      "expected_sha256": "\(identity.expectedSHA256)",
      "etag": "\(etag)"
    }
    """.utf8).write(to: cache.appendingPathComponent("identity.json"))
    let partial = cache
        .appendingPathComponent("models--org--model/blobs", isDirectory: true)
        .appendingPathComponent(etag)
        .appendingPathExtension("sync.part")
    try FileManager.default.createDirectory(
        at: partial.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try Data().write(to: partial)
    let handle = try FileHandle(forWritingTo: partial)
    try handle.truncate(atOffset: 18)
    try handle.seek(toOffset: 10)
    try handle.write(contentsOf: UInt64(4).littleEndianData)
    try handle.close()

    #expect(
        RustHuggingFaceFileDownloader.recoverableBytes(
            destination: destination,
            identity: identity
        ) == 4
    )
    var changed = identity
    changed.commitSHA = String(repeating: "d", count: 40)
    #expect(
        RustHuggingFaceFileDownloader.recoverableBytes(
            destination: destination,
            identity: changed
        ) == 0
    )
}

@MainActor
@Test func dashboardBridgeRegistersModelDownloadHandlers() {
    #expect(DashboardBridge.handlerNames.contains("downloadModel"))
    #expect(DashboardBridge.handlerNames.contains("cancelModelDownload"))
    #expect(DashboardBridge.handlerNames.contains("searchHF"))
    #expect(DashboardBridge.handlerNames.contains("cancelHFSearch"))
    #expect(DashboardBridge.handlerNames.contains("listModelVersions"))
    #expect(DashboardBridge.handlerNames.contains("activateModelVersion"))
    #expect(DashboardBridge.handlerNames.contains("deleteModelVersions"))
}

@Test func huggingFaceSearchUsesMlxFilterAndSort() async throws {
    let client = FakeModelDownloadHTTPClient()
    client.dataResponses["https://huggingface.co/api/models?search=qwen&sort=downloads&direction=-1&limit=20&filter=mlx&full=true"] =
        .success(Data("""
        [{"id":"mlx-community/Qwen3-0.6B-4bit","sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","downloads":123,"likes":4,"pipeline_tag":"text-generation"}]
        """.utf8))
    let service = ModelDownloadService(rootURL: try temporaryDirectory(), httpClient: client)

    let results = try await service.searchHuggingFace(
        query: "qwen",
        sort: "downloads",
        token: "hf_search"
    )

    #expect(results.map(\.id) == ["mlx-community/Qwen3-0.6B-4bit"])
    #expect(results.first?.sha == String(repeating: "a", count: 40))
    #expect(client.dataRequests == [
        "https://huggingface.co/api/models?search=qwen&sort=downloads&direction=-1&limit=20&filter=mlx&full=true",
    ])
    #expect(client.dataAuthorizationHeaders == ["Bearer hf_search"])
}

@Test func huggingFaceSearchResolvesMissingCommitFromRepositoryDetails() async throws {
    let client = FakeModelDownloadHTTPClient()
    let searchURL =
        "https://huggingface.co/api/models?search=minicpm&sort=downloads&direction=-1&limit=20&filter=mlx&full=true"
    let detailURL = "https://huggingface.co/api/models/mlx-community/MiniCPM-V-4.6-bf16?blobs=true"
    client.dataResponses[searchURL] = .success(Data("""
    [{"id":"mlx-community/MiniCPM-V-4.6-bf16","downloads":123}]
    """.utf8))
    client.dataResponses[detailURL] = .success(Data("""
    {"sha":"\(testCommit)","siblings":[]}
    """.utf8))
    let service = ModelDownloadService(rootURL: try temporaryDirectory(), httpClient: client)

    let results = try await service.searchHuggingFace(
        query: "minicpm",
        sort: "downloads",
        token: nil
    )

    #expect(results.first?.sha == testCommit)
    #expect(client.dataRequests == [searchURL, detailURL])
}

@Test func huggingFaceSearchPropagatesCancellationWhileResolvingMissingCommit() async throws {
    let resolutionStarted = StreamingSignal()
    let client = CancellableHuggingFaceSearchHTTPClient(
        resolutionStarted: resolutionStarted
    )
    let service = ModelDownloadService(
        rootURL: try temporaryDirectory(),
        httpClient: client
    )
    let search = Task {
        try await service.searchHuggingFace(
            query: "minicpm",
            sort: "downloads",
            token: nil
        )
    }

    await resolutionStarted.wait()
    search.cancel()

    await #expect(throws: CancellationError.self) {
        _ = try await search.value
    }
}

@Test func huggingFaceCanonicalRepoIDUsesExactResolutionWithoutFuzzySearch() async throws {
    let client = FakeModelDownloadHTTPClient()
    let repoID = "mlx-community/MiniCPM-V-4.6-bf16"
    let detailURL = "https://huggingface.co/api/models/\(repoID)?blobs=true"
    client.dataResponses[detailURL] = .success(Data("""
    {"sha":"\(testCommit)","siblings":[]}
    """.utf8))
    let service = ModelDownloadService(rootURL: try temporaryDirectory(), httpClient: client)

    let results = try await service.searchHuggingFace(
        query: repoID,
        sort: "downloads",
        token: "hf_private"
    )

    #expect(results.map(\.id) == [repoID])
    #expect(results.first?.sha == testCommit)
    #expect(client.dataRequests == [detailURL])
    #expect(client.dataAuthorizationHeaders == ["Bearer hf_private"])
}

@Test func huggingFaceCanonicalRepoIDFailureDoesNotFallBackToFuzzySearch() async throws {
    let client = FakeModelDownloadHTTPClient()
    let repoID = "mlx-community/Does-Not-Exist"
    let detailURL = "https://huggingface.co/api/models/\(repoID)?blobs=true"
    client.dataResponses[detailURL] = .failure(ModelDownloadHTTPError(statusCode: 404))
    let service = ModelDownloadService(rootURL: try temporaryDirectory(), httpClient: client)

    await #expect(throws: RepositoryResolutionError.self) {
        _ = try await service.searchHuggingFace(
            query: repoID,
            sort: "downloads",
            token: nil
        )
    }

    #expect(client.dataRequests == [detailURL])
}

@Test func huggingFaceDownloadPinsCommitVerifiesAndAtomicallyPublishes() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let config = Data(#"{"model_type":"llama"}"#.utf8)
    let weights = Data("weights".utf8)
    configureHuggingFace(
        client,
        repoID: "mlx-community/Tiny-4bit",
        files: [
            ("config.json", config, nil),
            ("tokenizer.json", Data("{}".utf8), nil),
            ("model.safetensors", weights, sha256(weights)),
        ]
    )
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: AcceptingMetadataPreflight(),
        fileDownloader: ResumableFileDownloader(httpClient: client),
        telemetryLogger: { _ in }
    )

    let result = await service.downloadHuggingFace(repoID: "mlx-community/Tiny-4bit", token: nil)

    #expect(result.success)
    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: .huggingFace,
        repoID: "mlx-community/Tiny-4bit"
    )
    let snapshot = repository
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent(testCommit, isDirectory: true)
    let manifest = try ModelSnapshotVerifier().verify(
        snapshot: snapshot,
        expectedProvider: .huggingFace,
        expectedRepoID: "mlx-community/Tiny-4bit"
    )
    #expect(manifest.commitSHA == testCommit)
    #expect(try String(contentsOf: repository.appendingPathComponent("refs/main"), encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines) == testCommit)
    #expect(client.streamRequests.allSatisfy { $0.url.contains("/resolve/\(testCommit)/") })
    #expect(!FileManager.default.fileExists(
        atPath: repository.appendingPathComponent(".downloads/\(testCommit)/snapshot").path
    ))
}

@Test func dflash2DraftDownloadAllowsMissingTokenizerAndPublishesAuxiliaryArtifact() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let repoID = "z-lab/Qwen3.8-27B-DFlash2"
    let config = Data(#"{"architectures":["DFlash2DraftModel"],"dflash_config":{"block_size":8},"model_type":"qwen3"}"#.utf8)
    let weights = Data("dflash2-weights".utf8)
    configureHuggingFace(
        client,
        repoID: repoID,
        files: [
            ("config.json", config, nil),
            ("model.safetensors", weights, sha256(weights)),
        ]
    )
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: AcceptingDFlash2MetadataPreflight(),
        fileDownloader: ResumableFileDownloader(httpClient: client),
        telemetryLogger: { _ in }
    )

    let result = await service.downloadHuggingFace(repoID: repoID, token: nil)

    #expect(result.success)
    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: .huggingFace,
        repoID: repoID
    )
    let snapshot = repository
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent(testCommit, isDirectory: true)
    let manifest = try ModelSnapshotVerifier().verify(
        snapshot: snapshot,
        expectedProvider: .huggingFace,
        expectedRepoID: repoID
    )
    #expect(manifest.compatibility.modelType == "qwen3")
    #expect(manifest.compatibility.artifactRole == ModelArtifactRole.dflash2Drafter)
    #expect(!FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("tokenizer.json").path))
    #expect(client.streamRequests.contains { $0.url.hasSuffix("model.safetensors") })
    let scanner = LocalModelScanner(rootURL: root)
    #expect(scanner.scan().isEmpty)
    #expect(scanner.resolveModelPath(for: repoID) == nil)
}

@Test func baseDownloadWithoutTokenizerIsRejectedBeforeWeightRequest() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let repoID = "mlx-community/Base-Without-Tokenizer"
    let weights = Data("weights".utf8)
    configureHuggingFace(
        client,
        repoID: repoID,
        files: [
            ("config.json", Data(#"{"model_type":"llama"}"#.utf8), nil),
            ("model.safetensors", weights, sha256(weights)),
        ]
    )
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: AcceptingMetadataPreflight(),
        fileDownloader: ResumableFileDownloader(httpClient: client),
        telemetryLogger: { _ in }
    )

    let result = await service.downloadHuggingFace(repoID: repoID, token: nil)

    #expect(!result.success)
    #expect(result.code == "repo_missing_metadata")
    #expect(!client.streamRequests.contains { $0.url.hasSuffix("model.safetensors") })
}

@Test func huggingFaceDownloadRepairsAKnownCorruptCommitWithoutDeletingItBeforePublish() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let repoID = "mlx-community/Tiny-4bit"
    let weights = Data("weights".utf8)
    configureHuggingFace(
        client,
        repoID: repoID,
        files: [
            ("config.json", Data(#"{"model_type":"llama"}"#.utf8), nil),
            ("tokenizer.json", Data("{}".utf8), nil),
            ("model.safetensors", weights, sha256(weights)),
        ]
    )
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: AcceptingMetadataPreflight(),
        fileDownloader: ResumableFileDownloader(httpClient: client),
        telemetryLogger: { _ in }
    )
    #expect((await service.downloadHuggingFace(repoID: repoID, token: nil)).success)
    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: .huggingFace,
        repoID: repoID
    )
    let snapshot = repository
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent(testCommit, isDirectory: true)
    try Data("damaged".utf8).write(
        to: snapshot.appendingPathComponent("model.safetensors")
    )
    try ModelDownloadStore.atomicWrite(
        ModelSnapshotIntegrityRecord(
            provider: .huggingFace,
            repoID: repoID,
            commitSHA: testCommit,
            state: .corrupt,
            error: "test damage"
        ),
        to: snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
    )

    let repaired = await service.downloadHuggingFace(repoID: repoID, token: nil)

    #expect(repaired.success)
    _ = try ModelSnapshotVerifier().verify(
        snapshot: snapshot,
        expectedProvider: .huggingFace,
        expectedRepoID: repoID
    )
    #expect(
        try Data(contentsOf: snapshot.appendingPathComponent("model.safetensors"))
            == weights
    )
    #expect(
        try FileManager.default.contentsOfDirectory(
            atPath: repository.appendingPathComponent(".trash", isDirectory: true).path
        ).count == 1
    )
}

@Test func unsupportedMetadataIsRejectedBeforeWeightRequest() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let config = Data(#"{"model_type":"unsupported"}"#.utf8)
    let weights = Data("weights".utf8)
    configureHuggingFace(
        client,
        repoID: "mlx-community/Unsupported",
        files: [
            ("config.json", config, nil),
            ("tokenizer.json", Data("{}".utf8), nil),
            ("model.safetensors", weights, sha256(weights)),
        ]
    )
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: RejectingMetadataPreflight(),
        telemetryLogger: { _ in }
    )

    let result = await service.downloadHuggingFace(repoID: "mlx-community/Unsupported", token: nil)

    #expect(!result.success)
    #expect(result.code == "unsupported_model_metadata")
    #expect(!client.streamRequests.contains { $0.url.hasSuffix("model.safetensors") })
}

@Test func modelDownloadQueueRunsThreeTasksAndKeepsTheFourthInFIFOOrder() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let preflight = BlockingMetadataPreflight()
    let repoIDs = (1 ... 4).map { "org/queue-model-\($0)" }
    for repoID in repoIDs {
        configureTinyHuggingFace(client, repoID: repoID)
    }
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: preflight,
        fileDownloader: ResumableFileDownloader(httpClient: client),
        maxConcurrentDownloads: 3,
        telemetryLogger: { _ in }
    )

    for repoID in repoIDs {
        let response = await service.startHuggingFaceDownload(repoID: repoID, token: nil)
        #expect(response.success)
    }
    try await waitForDownloadCondition {
        let snapshot = await service.downloadQueueSnapshot()
        let preflightCalls = await preflight.totalCalls()
        return snapshot.activeCount == 3
            && snapshot.queuedCount == 1
            && preflightCalls == 3
    }
    let queued = await service.downloadQueueSnapshot().tasks.first { $0.repoID == repoIDs[3] }
    #expect(queued?.status == ModelDownloadPhase.queued.rawValue)
    #expect(queued?.queuePosition == 1)
    #expect(await preflight.maximumConcurrentCalls() == 3)

    await preflight.releaseOne()
    try await waitForDownloadCondition {
        await preflight.totalCalls() == 4
    }
    #expect(await preflight.maximumConcurrentCalls() == 3)

    await preflight.releaseAll()
    try await waitForDownloadCondition {
        let snapshot = await service.downloadQueueSnapshot()
        return snapshot.activeCount == 0
            && snapshot.queuedCount == 0
            && snapshot.tasks.filter { $0.status == ModelDownloadPhase.completed.rawValue }.count == 4
    }
    #expect(await service.clearFinishedDownloads() == 4)
    #expect(await service.downloadQueueSnapshot().tasks.isEmpty)
}

@Test func modelDownloadQueueAggregatesDiskReservationsAcrossTasks() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let preflight = BlockingMetadataPreflight()
    let first = "org/disk-first"
    let second = "org/disk-second"
    let repositoryBytes = configureTinyHuggingFace(client, repoID: first)
    _ = configureTinyHuggingFace(client, repoID: second)
    let capacity = ModelResourcePreflight.diskSafetyBytes + repositoryBytes + 1
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: preflight,
        fileDownloader: ResumableFileDownloader(httpClient: client),
        maxConcurrentDownloads: 3,
        availableCapacityProvider: { _ in capacity },
        telemetryLogger: { _ in }
    )

    #expect(await service.startHuggingFaceDownload(repoID: first, token: nil).success)
    #expect(await service.startHuggingFaceDownload(repoID: second, token: nil).success)
    try await waitForDownloadCondition {
        let task = await service.downloadQueueSnapshot().tasks.first { $0.repoID == second }
        return task?.status == ModelDownloadPhase.rejected.rawValue
    }
    let rejected = await service.downloadQueueSnapshot().tasks.first { $0.repoID == second }
    #expect(rejected?.errorCode == "insufficient_disk")
    #expect(await service.clearFinishedDownloads() == 1)
    let afterClear = await service.downloadQueueSnapshot()
    #expect(afterClear.tasks.contains { $0.repoID == first && ModelDownloadPhase(rawValue: $0.status)?.isActive == true })
    #expect(!afterClear.tasks.contains { $0.repoID == second })

    await service.cancelAllDownloads()
    await preflight.releaseAll()
    try await waitForDownloadCondition {
        await service.downloadQueueSnapshot().activeCount == 0
    }
}

@Test func modelDownloadRestartReminderPersistsIdentityWithoutToken() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    let preflight = BlockingMetadataPreflight()
    let repoID = "org/private-model"
    configureTinyHuggingFace(client, repoID: repoID)
    let service = ModelDownloadService(
        rootURL: root,
        httpClient: client,
        metadataPreflight: preflight,
        fileDownloader: ResumableFileDownloader(httpClient: client),
        telemetryLogger: { _ in }
    )

    #expect(await service.startHuggingFaceDownload(repoID: repoID, token: "hf_secret_value").success)
    try await waitForDownloadCondition {
        await service.downloadQueueSnapshot().activeCount == 1
    }
    await service.cancelAllDownloads()
    await preflight.releaseAll()
    try await waitForDownloadCondition {
        await service.downloadQueueSnapshot().activeCount == 0
    }

    let reminderURL = root.appendingPathComponent("state/model-download-reminders.json")
    let persisted = try String(contentsOf: reminderURL, encoding: .utf8)
    #expect(!persisted.contains("hf_secret_value"))
    #expect(persisted.contains(#""used_credential" : true"#))

    let relaunched = ModelDownloadService(rootURL: root, httpClient: client)
    let relaunchedSnapshot = await relaunched.downloadQueueSnapshot()
    let reminders = relaunchedSnapshot.recoveryReminders
    #expect(reminders.count == 1)
    #expect(reminders.first?.repoID == repoID)
    #expect(reminders.first?.usedCredential == true)
    #expect(relaunchedSnapshot.tasks.isEmpty)
}

@Test func resumableDownloaderUsesRangeAndKeepsValidPartialAcrossRestart() async throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let full = Data("0123456789".utf8)
    let identity = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: testCommit,
        path: "model.safetensors",
        expectedSize: Int64(full.count),
        expectedSHA256: sha256(full),
        etag: "\"stable\""
    )
    let partial = destination.appendingPathExtension("partial")
    try full.prefix(4).write(to: partial)
    try ModelDownloadStore.atomicWrite(identity, to: partial.appendingPathExtension("meta.json"))
    let client = FakeModelDownloadHTTPClient()
    client.streamResponses["https://example.test/model"] = StreamFixture(
        data: Data(full.dropFirst(4)),
        statusCode: 206,
        headers: [
            "Content-Range": "bytes 4-9/10",
            "ETag": "\"stable\"",
        ]
    )
    let downloader = ResumableFileDownloader(httpClient: client)

    _ = try await downloader.download(
        ResumableDownloadRequest(
            urlRequest: URLRequest(url: URL(string: "https://example.test/model")!),
            identity: identity,
            destination: destination
        )
    )

    #expect(try Data(contentsOf: destination) == full)
    #expect(client.streamRequests.single?.range == "bytes=4-")
    #expect(client.streamRequests.single?.ifRange == "\"stable\"")
}

@Test func changedETagCannotAppendToOldPartial() async throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let full = Data("0123456789".utf8)
    let identity = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: testCommit,
        path: "model.safetensors",
        expectedSize: Int64(full.count),
        expectedSHA256: sha256(full),
        etag: "\"old\""
    )
    let partial = destination.appendingPathExtension("partial")
    try full.prefix(4).write(to: partial)
    try ModelDownloadStore.atomicWrite(identity, to: partial.appendingPathExtension("meta.json"))
    let client = FakeModelDownloadHTTPClient()
    client.streamResponses["https://example.test/model"] = StreamFixture(
        data: Data(full.dropFirst(4)),
        statusCode: 206,
        headers: [
            "Content-Range": "bytes 4-9/10",
            "ETag": "\"new\"",
        ]
    )

    await #expect(throws: ResumableDownloadError.self) {
        try await ResumableFileDownloader(httpClient: client).download(
            ResumableDownloadRequest(
                urlRequest: URLRequest(url: URL(string: "https://example.test/model")!),
                identity: identity,
                destination: destination
            )
        )
    }
    #expect(try Data(contentsOf: partial) == full.prefix(4))
}

@Test func responseETagPersistedFromEarlierAttemptProtectsResume() async throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let full = Data("0123456789".utf8)
    let requestIdentity = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: testCommit,
        path: "model.safetensors",
        expectedSize: Int64(full.count),
        expectedSHA256: sha256(full),
        etag: nil
    )
    var storedIdentity = requestIdentity
    storedIdentity.etag = "\"observed-old\""
    let partial = destination.appendingPathExtension("partial")
    try full.prefix(4).write(to: partial)
    try ModelDownloadStore.atomicWrite(
        storedIdentity,
        to: partial.appendingPathExtension("meta.json")
    )
    let client = FakeModelDownloadHTTPClient()
    client.streamResponses["https://example.test/model"] = StreamFixture(
        data: Data(full.dropFirst(4)),
        statusCode: 206,
        headers: [
            "Content-Range": "bytes 4-9/10",
            "ETag": "\"observed-new\"",
        ]
    )

    await #expect(throws: ResumableDownloadError.self) {
        try await ResumableFileDownloader(httpClient: client).download(
            ResumableDownloadRequest(
                urlRequest: URLRequest(url: URL(string: "https://example.test/model")!),
                identity: requestIdentity,
                destination: destination
            )
        )
    }

    #expect(client.streamRequests.single?.range == "bytes=4-")
    #expect(client.streamRequests.single?.ifRange == "\"observed-old\"")
    #expect(try Data(contentsOf: partial) == full.prefix(4))
}

@Test func changedCommitIdentityDiscardsOldPartialBeforeRequest() async throws {
    let root = try temporaryDirectory()
    let destination = root.appendingPathComponent("model.safetensors")
    let full = Data("new-content".utf8)
    let old = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: String(repeating: "b", count: 40),
        path: "model.safetensors",
        expectedSize: 10,
        expectedSHA256: String(repeating: "0", count: 64),
        etag: nil
    )
    let partial = destination.appendingPathExtension("partial")
    try Data("old".utf8).write(to: partial)
    try ModelDownloadStore.atomicWrite(old, to: partial.appendingPathExtension("meta.json"))
    let client = FakeModelDownloadHTTPClient()
    client.streamResponses["https://example.test/model"] = StreamFixture(data: full)
    let current = ModelPartialIdentity(
        provider: .huggingFace,
        repoID: "org/model",
        commitSHA: testCommit,
        path: "model.safetensors",
        expectedSize: Int64(full.count),
        expectedSHA256: sha256(full),
        etag: nil
    )

    _ = try await ResumableFileDownloader(httpClient: client).download(
        ResumableDownloadRequest(
            urlRequest: URLRequest(url: URL(string: "https://example.test/model")!),
            identity: current,
            destination: destination
        )
    )

    #expect(client.streamRequests.single?.range == nil)
    #expect(try Data(contentsOf: destination) == full)
}

@Test func modelScopeResolverPinsMasterToAdvertisedCommitBeforeListingFiles() async throws {
    let client = FakeModelDownloadHTTPClient()
    let commit = String(repeating: "b", count: 40)
    client.dataResponses[
        "https://www.modelscope.test/org/model.git/info/refs?service=git-upload-pack"
    ] = .success(Data("003f\(commit) refs/heads/master\u{0}report-status\n".utf8))
    client.dataResponses[
        "https://api.modelscope.test/org/model/repo/files?Revision=\(commit)&Recursive=true"
    ] = .success(Data("""
    {
      "Success": true,
      "Data": {
        "Files": [
          {"Path":"config.json","Type":"blob","Size":2,"Sha256":"\(sha256(Data("{}".utf8)))"},
          {"Path":"model.safetensors","Type":"blob","Size":7,"Sha256":"\(sha256(Data("weights".utf8)))"}
        ]
      }
    }
    """.utf8))
    let resolver = ModelRepositoryResolver(
        httpClient: client,
        huggingFaceEndpoint: URL(string: "https://huggingface.test")!,
        modelScopeAPIEndpoint: URL(string: "https://api.modelscope.test")!,
        modelScopeGitEndpoint: URL(string: "https://www.modelscope.test")!
    )

    let repository = try await resolver.resolve(provider: .modelScope, repoID: "org/model", token: nil)

    #expect(repository.commitSHA == commit)
    #expect(repository.requestedRevision == "master")
    #expect(repository.files.map(\.path) == ["config.json", "model.safetensors"])
    let request = try resolver.request(for: repository.files[0], repository: repository, token: nil)
    #expect(request.url?.absoluteString.contains("Revision=\(commit)") == true)
}

@Test func fullLoadVerificationRejectsCorruptedPublishedShardAndMarksItCorrupt() throws {
    let root = try temporaryDirectory()
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        provider: .huggingFace,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    try Data("damaged".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    #expect(throws: ModelSnapshotVerificationError.self) {
        try LocalModelScanner(rootURL: root).verifiedModelPath(
            for: "org/model",
            fullChecksum: true
        )
    }
    let record = try ModelSnapshotVerifier().loadIntegrityRecord(at: snapshot)
    #expect(record.state == .corrupt)
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: "org/model") == nil)
}

@Test func defaultLoadVerificationUsesFastStructuralChecksOnly() throws {
    let root = try temporaryDirectory()
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        provider: .huggingFace,
        repoID: "org/model",
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    try Data("damaged".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let path = try LocalModelScanner(rootURL: root).verifiedModelPath(for: "org/model")

    #expect(path == snapshot.path)
}

private struct AcceptingMetadataPreflight: ModelMetadataPreflighting {
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        ModelMetadataPreflightResult(
            modelType: "llama",
            artifactRole: "base",
            quantization: nil
        )
    }
}

private struct AcceptingDFlash2MetadataPreflight: ModelMetadataPreflighting {
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        ModelMetadataPreflightResult(
            modelType: "qwen3",
            artifactRole: ModelArtifactRole.dflash2Drafter,
            quantization: nil
        )
    }
}

private struct RejectingMetadataPreflight: ModelMetadataPreflighting {
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        throw ModelMetadataPreflightError.rejected("unsupported architecture")
    }
}

private struct StreamFixture {
    var data: Data
    var statusCode = 200
    var headers: [String: String] = [:]
}

private struct RecordedStreamRequest {
    var url: String
    var range: String?
    var ifRange: String?
}

private final class FakeModelDownloadHTTPClient: ModelDownloadHTTPClient, @unchecked Sendable {
    var dataResponses: [String: Result<Data, Error>] = [:]
    var streamResponses: [String: StreamFixture] = [:]
    private(set) var dataRequests: [String] = []
    private(set) var dataAuthorizationHeaders: [String?] = []
    private(set) var streamRequests: [RecordedStreamRequest] = []
    private let lock = NSLock()

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let key = try requestKey(request)
        lock.withLock {
            dataRequests.append(key)
            dataAuthorizationHeaders.append(request.value(forHTTPHeaderField: "Authorization"))
        }
        guard let result = lock.withLock({ dataResponses[key] }) else {
            throw URLError(.fileDoesNotExist)
        }
        return (try result.get(), response(for: request, statusCode: 200))
    }

    func stream(
        for request: URLRequest,
        onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        let key = try requestKey(request)
        lock.withLock {
            streamRequests.append(
                RecordedStreamRequest(
                    url: key,
                    range: request.value(forHTTPHeaderField: "Range"),
                    ifRange: request.value(forHTTPHeaderField: "If-Range")
                )
            )
        }
        guard let fixture = lock.withLock({ streamResponses[key] }) else {
            throw URLError(.fileDoesNotExist)
        }
        try await onResponse(
            response(
                for: request,
                statusCode: fixture.statusCode,
                headers: fixture.headers
            )
        )
        try await onData(fixture.data)
    }

    private func requestKey(_ request: URLRequest) throws -> String {
        try #require(request.url).absoluteString
    }

    private func response(
        for request: URLRequest,
        statusCode: Int,
        headers: [String: String] = [:]
    ) -> HTTPURLResponse {
        HTTPURLResponse(
            url: request.url!,
            statusCode: statusCode,
            httpVersion: nil,
            headerFields: headers
        )!
    }
}

private final class CancellableHuggingFaceSearchHTTPClient:
    ModelDownloadHTTPClient,
    @unchecked Sendable
{
    private let resolutionStarted: StreamingSignal

    init(resolutionStarted: StreamingSignal) {
        self.resolutionStarted = resolutionStarted
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        if request.url?.query?.contains("search=") == true {
            return (
                Data(
                    #"[{"id":"mlx-community/MiniCPM-V-4.6-bf16","downloads":123}]"#
                        .utf8
                ),
                response
            )
        }

        await resolutionStarted.signal()
        try await Task.sleep(nanoseconds: 30_000_000_000)
        throw URLError(.timedOut)
    }

    func stream(
        for _: URLRequest,
        onResponse _: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData _: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        throw URLError(.unsupportedURL)
    }
}

private actor StreamingResponseCollector {
    private var statusCode: Int?
    private var body = Data()
    private var callbackCount = 0

    func record(statusCode: Int) {
        self.statusCode = statusCode
    }

    func append(_ data: Data) {
        body.append(data)
        callbackCount += 1
    }

    func result() -> (statusCode: Int?, body: Data, callbackCount: Int) {
        (statusCode, body, callbackCount)
    }
}

private enum StreamingResponseTestError: Error {
    case rejectedChunk
}

private actor StreamingSignal {
    private var continuation: CheckedContinuation<Void, Never>?
    private var signalled = false

    func signal() {
        signalled = true
        continuation?.resume()
        continuation = nil
    }

    func wait() async {
        guard !signalled else {
            return
        }
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }
}

private actor BlockingMetadataPreflight: ModelMetadataPreflighting {
    private var calls = 0
    private var activeCalls = 0
    private var maximumCalls = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        calls += 1
        activeCalls += 1
        maximumCalls = max(maximumCalls, activeCalls)
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
        activeCalls -= 1
        try Task.checkCancellation()
        return ModelMetadataPreflightResult(
            modelType: "llama",
            artifactRole: "base",
            quantization: nil
        )
    }

    func totalCalls() -> Int {
        calls
    }

    func maximumConcurrentCalls() -> Int {
        maximumCalls
    }

    func releaseOne() {
        guard !waiters.isEmpty else {
            return
        }
        waiters.removeFirst().resume()
    }

    func releaseAll() {
        let pending = waiters
        waiters.removeAll()
        for waiter in pending {
            waiter.resume()
        }
    }
}

private enum DownloadQueueTestError: Error {
    case timedOut
}

private func waitForDownloadCondition(
    attempts: Int = 300,
    condition: @escaping @Sendable () async -> Bool
) async throws {
    for _ in 0 ..< attempts {
        if await condition() {
            return
        }
        try await Task.sleep(for: .milliseconds(10))
    }
    throw DownloadQueueTestError.timedOut
}

private extension UInt64 {
    var littleEndianData: Data {
        var value = littleEndian
        return Data(bytes: &value, count: MemoryLayout<UInt64>.size)
    }
}

private final class ChunkedDownloadURLProtocol: URLProtocol, @unchecked Sendable {
    static let chunks = [
        Data(repeating: 0x11, count: 32 * 1_024),
        Data(repeating: 0x22, count: 48 * 1_024),
        Data(repeating: 0x33, count: 64 * 1_024),
    ]
    static let body = chunks.reduce(into: Data()) { $0.append($1) }

    override class func canInit(with _: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": String(Self.body.count)]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        for chunk in Self.chunks {
            client?.urlProtocol(self, didLoad: chunk)
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private final class SuspendedDownloadURLProtocol: URLProtocol, @unchecked Sendable {
    private var completionWorkItem: DispatchWorkItem?

    override class func canInit(with _: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(repeating: 0x44, count: 64 * 1_024))
        let workItem = DispatchWorkItem { [weak self] in
            guard let self else {
                return
            }
            client?.urlProtocolDidFinishLoading(self)
        }
        completionWorkItem = workItem
        DispatchQueue.global().asyncAfter(deadline: .now() + 10, execute: workItem)
    }

    override func stopLoading() {
        completionWorkItem?.cancel()
        completionWorkItem = nil
    }
}

private func configureHuggingFace(
    _ client: FakeModelDownloadHTTPClient,
    repoID: String,
    files: [(path: String, data: Data, sha256: String?)]
) {
    let siblings = files.map { file -> [String: Any] in
        var value: [String: Any] = [
            "rfilename": file.path,
            "size": file.data.count,
            "blobId": gitBlobSHA1(file.data),
        ]
        if let sha256 = file.sha256 {
            value["lfs"] = ["sha256": sha256, "size": file.data.count]
        }
        return value
    }
    let info: [String: Any] = ["sha": testCommit, "siblings": siblings]
    let api = "https://huggingface.co/api/models/\(repoID)?blobs=true"
    client.dataResponses[api] = .success(try! JSONSerialization.data(withJSONObject: info))
    for file in files {
        let url = "https://huggingface.co/\(repoID)/resolve/\(testCommit)/\(file.path)"
        if file.sha256 == nil {
            client.dataResponses[url] = .success(file.data)
        } else {
            client.streamResponses[url] = StreamFixture(data: file.data)
        }
    }
}

@discardableResult
private func configureTinyHuggingFace(
    _ client: FakeModelDownloadHTTPClient,
    repoID: String
) -> Int64 {
    let files: [(path: String, data: Data, sha256: String?)] = [
        ("config.json", Data(#"{"model_type":"llama"}"#.utf8), nil),
        ("tokenizer.json", Data("{}".utf8), nil),
        ("model.safetensors", Data("weights".utf8), sha256(Data("weights".utf8))),
    ]
    configureHuggingFace(client, repoID: repoID, files: files)
    return files.reduce(Int64(0)) { $0 + Int64($1.data.count) }
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func gitBlobSHA1(_ data: Data) -> String {
    var digest = Insecure.SHA1()
    digest.update(data: Data("blob \(data.count)\0".utf8))
    digest.update(data: data)
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
}

private extension Array {
    var single: Element? {
        count == 1 ? self[0] : nil
    }
}
