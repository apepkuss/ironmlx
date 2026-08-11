import Foundation
import Testing
import ZIPFoundation

@testable import IronMLXAppCore

@Suite("Privacy-safe diagnostic bundle", .serialized)
struct DiagnosticBundleServiceTests {
    @Test @MainActor
    func offlineBundleHasStableEntriesSchemaAndNoSentinels() async throws {
        let root = try temporaryDirectory()
        let logsRoot = root.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logsRoot, withIntermediateDirectories: true)
        let sentinels = [
            "PRIVATE_PROMPT_BUNDLE_SENTINEL", "HF_TOKEN_BUNDLE_SENTINEL",
            "LAN_KEY_BUNDLE_SENTINEL", "AUTH_BUNDLE_SENTINEL",
            NSUserName(), NSHomeDirectory(),
        ].filter { !$0.isEmpty }
        let logText = """
        Authorization: Bearer AUTH_BUNDLE_SENTINEL
        hf_token=HF_TOKEN_BUNDLE_SENTINEL
        lan_api_key=LAN_KEY_BUNDLE_SENTINEL
        {"prompt":"PRIVATE_PROMPT_BUNDLE_SENTINEL"}
        model=\(NSHomeDirectory())/.ironmlx/models/local
        """
        try logText.write(to: logsRoot.appendingPathComponent("app.log"), atomically: true, encoding: .utf8)
        try logText.write(to: logsRoot.appendingPathComponent("backend.log"), atomically: true, encoding: .utf8)
        let incidentStore = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
        try incidentStore.upsert(
            BackendIncidentRecord(
                termination: BackendProcessTermination(
                    occurredAt: Date(timeIntervalSince1970: 1),
                    launchID: UUID(),
                    generation: 1,
                    pid: 42,
                    terminationStatus: 9,
                    terminationReason: "uncaught_signal",
                    stopIntent: .unexpected,
                    logTail: logText
                ),
                snapshot: BackendRecoverySnapshot(config: AppConfig(), models: [])
            )
        )
        let service = DiagnosticBundleService(
            scanner: LocalModelScanner(rootURL: root),
            versionService: ModelVersionManagementService(rootURL: root),
            incidentStore: incidentStore,
            logStore: IronMLXLogStore(rootURL: logsRoot),
            now: { Date(timeIntervalSince1970: 123) }
        )

        let artifact = try await service.collect(config: AppConfig(), backendRunning: false)
        let archive = try Archive(data: artifact.archiveData, accessMode: .read)

        #expect(archive.map(\.path) == DiagnosticBundleService.entryOrder)
        #expect(artifact.archiveData.count <= DiagnosticBundleLimits().maximumArchiveBytes)
        #expect(artifact.manifest.schemaVersion == 1)
        #expect(!artifact.manifest.backendOnline)
        #expect(artifact.manifest.entries.keys.count == DiagnosticBundleService.entryOrder.count)

        var allBytes = Data()
        for entry in archive {
            var entryData = Data()
            _ = try archive.extract(entry, consumer: { entryData.append($0) })
            allBytes.append(entryData)
            if entry.path.hasSuffix(".json") {
                _ = try JSONSerialization.jsonObject(with: entryData)
            }
        }
        let allText = String(decoding: allBytes, as: UTF8.self)
        for sentinel in sentinels {
            #expect(!allText.contains(sentinel), "archive leaked \(sentinel)")
        }
        let runtimeJSON = try #require(
            JSONSerialization.jsonObject(with: try extract("runtime-health.json", from: archive)) as? [String: Any]
        )
        #expect(runtimeJSON["status"] as? String == "unavailable")
        #expect(runtimeJSON["error_code"] as? String == "backend_offline")
    }

    @Test @MainActor
    func oversizedLogsAreTruncatedAndManifestRecordsIt() async throws {
        let root = try temporaryDirectory()
        let logsRoot = root.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logsRoot, withIntermediateDirectories: true)
        try String(repeating: "safe diagnostic line\n", count: 10_000).write(
            to: logsRoot.appendingPathComponent("backend.log"), atomically: true, encoding: .utf8
        )
        var limits = DiagnosticBundleLimits()
        limits.backendLogBytes = 4_096
        limits.backendLogLines = 50
        let service = DiagnosticBundleService(
            scanner: LocalModelScanner(rootURL: root),
            versionService: ModelVersionManagementService(rootURL: root),
            incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
            logStore: IronMLXLogStore(rootURL: logsRoot),
            limits: limits
        )

        let artifact = try await service.collect(config: AppConfig(), backendRunning: false)

        #expect(artifact.manifest.entries["logs/backend.log"]?.truncated == true)
        #expect(artifact.manifest.entries["logs/backend.log"]?.bytes ?? 0 <= limits.backendLogBytes)
        #expect(artifact.manifest.contentTruncated)
    }

    @Test @MainActor
    func healthTimeoutIsStableAndHealthModelPathsNeverEnterArchive() async throws {
        let root = try temporaryDirectory()
        let timeoutService = DiagnosticBundleService(
            scanner: LocalModelScanner(rootURL: root),
            versionService: ModelVersionManagementService(rootURL: root),
            incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
            logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
            healthFetcher: { _, _ in throw URLError(.timedOut) }
        )
        let timeoutArtifact = try await timeoutService.collect(config: AppConfig(), backendRunning: true)
        let timeoutArchive = try Archive(data: timeoutArtifact.archiveData, accessMode: .read)
        let timeoutJSON = try #require(
            JSONSerialization.jsonObject(with: try extract("runtime-health.json", from: timeoutArchive))
                as? [String: Any]
        )
        #expect(timeoutJSON["error_code"] as? String == "health_timeout")
        #expect(!timeoutArtifact.manifest.backendOnline)

        let pathSentinel = "/Users/PRIVATE_HEALTH_USER/.ironmlx/models/HEALTH_PATH_SENTINEL"
        let healthService = DiagnosticBundleService(
            scanner: LocalModelScanner(rootURL: root),
            versionService: ModelVersionManagementService(rootURL: root),
            incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents-2.json")),
            logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
            healthFetcher: { _, _ in Self.healthSnapshot(modelPath: pathSentinel) }
        )
        let artifact = try await healthService.collect(config: AppConfig(), backendRunning: true)
        let healthArchive = try Archive(data: artifact.archiveData, accessMode: .read)
        var extracted = Data()
        for entry in healthArchive {
            _ = try healthArchive.extract(entry, consumer: { extracted.append($0) })
        }
        #expect(!String(decoding: extracted, as: UTF8.self).contains("HEALTH_PATH_SENTINEL"))
        #expect(artifact.manifest.backendOnline)
    }

    @Test func securePublisherUsesAtomicTemporaryAndCleansFailures() throws {
        let root = try temporaryDirectory()
        let destination = root.appendingPathComponent("diagnostics.zip")
        try DiagnosticArchivePublisher().publish(Data("zip".utf8), to: destination, maximumBytes: 10)

        #expect(try Data(contentsOf: destination) == Data("zip".utf8))
        let modes = try FileManager.default.attributesOfItem(atPath: destination.path)
        #expect((modes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).allSatisfy { !$0.hasSuffix(".tmp") })

        let missing = root.appendingPathComponent("missing/diagnostics.zip")
        #expect(throws: (any Error).self) {
            try DiagnosticArchivePublisher().publish(Data("zip".utf8), to: missing, maximumBytes: 10)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).allSatisfy { !$0.hasSuffix(".tmp") })
    }

    private func extract(_ path: String, from archive: Archive) throws -> Data {
        let entry = try #require(archive[path])
        var data = Data()
        _ = try archive.extract(entry, consumer: { data.append($0) })
        return data
    }

    private static func healthSnapshot(modelPath: String) -> HealthzSnapshot {
        let activeKV = HealthzSnapshot.ActiveKvOffloadInfo(
            enabled: false, status: "idle", active: false, degraded: false,
            mode: "disabled", storageDir: modelPath, residentPages: 0, offloadedPages: 0,
            loadingPages: 0, dirtyPages: 0, parkedRequests: 0, offloadedBytes: 0,
            swapOutCount: 0, swapInCount: 0, swapErrorCount: 0, lastSwapOutUs: 0,
            lastSwapInUs: 0, supportedCacheKinds: [], notApplicableCacheKinds: []
        )
        return HealthzSnapshot(
            status: "ok",
            mode: "multi",
            models: [
                BackendLoadedModelInfo(
                    id: "mlx-community/Safe-4bit", model: "mlx-community/Safe-4bit",
                    path: modelPath, architecture: "qwen", isDefault: true,
                    maxPositionEmbeddings: 32_768
                ),
            ],
            uptimeSecs: 12,
            model: .init(name: "mlx-community/Safe-4bit", maxPositionEmbeddings: 32_768),
            scheduler: .init(
                bMax: 4, bActive: 1, bQueued: 0, queueMax: 16,
                admissionQueueFullCount: 0, memoryBudgetExceededCount: 0
            ),
            memory: .init(
                totalRamBytes: 1, freeRamBytes: 1, kvCacheActiveBytes: 0,
                kvCacheSoftLimitBytes: 0, kvCacheLogicalCapTokens: 0,
                kvCacheResidentCapTokens: 0, kvCacheBudgetPolicy: "automatic",
                mlxTotalBytes: 1, mlxMaxRecommendedBytes: 1, mlxActiveBytes: 0,
                mlxCacheBytes: 0, mlxPeakBytes: 0, mlxMemoryLimitBytes: 1
            ),
            activeKvOffload: activeKV,
            deviceName: "Apple Silicon",
            version: "0.1.0"
        )
    }
}
