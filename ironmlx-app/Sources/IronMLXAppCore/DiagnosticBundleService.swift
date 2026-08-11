import Foundation
import ZIPFoundation

public enum DiagnosticBundleError: LocalizedError, Equatable {
    case cancelled
    case uncompressedCapacityExceeded
    case archiveCapacityExceeded
    case archiveUnavailable

    public var errorDescription: String? {
        switch self {
        case .cancelled: "Diagnostic export was cancelled."
        case .uncompressedCapacityExceeded: "Diagnostic content exceeded its hard capacity."
        case .archiveCapacityExceeded: "Diagnostic archive exceeded its hard capacity."
        case .archiveUnavailable: "Diagnostic ZIP creation failed."
        }
    }
}

public struct DiagnosticBundleArtifact: Sendable {
    public var archiveData: Data
    public var manifest: DiagnosticBundleManifest
}

@MainActor
public final class DiagnosticBundleService {
    public typealias ArchiveMaker = @Sendable ([String: Data], Int) throws -> Data
    nonisolated public static let entryOrder = [
        "manifest.json",
        "system.json",
        "runtime-health.json",
        "models.json",
        "incidents.json",
        "logs/app.log",
        "logs/backend.log",
    ]

    private let scanner: LocalModelScanner
    private let versionService: ModelVersionManagementService
    private let incidentStore: BackendIncidentStore
    private let logStore: IronMLXLogStore
    private let systemProfiler: DiagnosticSystemProfiler
    private let limits: DiagnosticBundleLimits
    private let now: @Sendable () -> Date
    private let healthFetcher: @Sendable (AppConfig, TimeInterval) async throws -> HealthzSnapshot
    private let archiveMaker: ArchiveMaker

    public init(
        scanner: LocalModelScanner = LocalModelScanner(),
        versionService: ModelVersionManagementService = ModelVersionManagementService(),
        incidentStore: BackendIncidentStore = BackendIncidentStore(),
        logStore: IronMLXLogStore = IronMLXLogStore(),
        systemProfiler: DiagnosticSystemProfiler = DiagnosticSystemProfiler(),
        limits: DiagnosticBundleLimits = DiagnosticBundleLimits(),
        now: @escaping @Sendable () -> Date = Date.init,
        healthFetcher: @escaping @Sendable (AppConfig, TimeInterval) async throws -> HealthzSnapshot = {
            config, timeout in
            try await BackendAPIClient(host: config.host, port: config.port).fetchHealthz(timeout: timeout)
        },
        archiveMaker: @escaping ArchiveMaker = { try DiagnosticBundleService.makeArchive(entries: $0, maximumBytes: $1) }
    ) {
        self.scanner = scanner
        self.versionService = versionService
        self.incidentStore = incidentStore
        self.logStore = logStore
        self.systemProfiler = systemProfiler
        self.limits = limits
        self.now = now
        self.healthFetcher = healthFetcher
        self.archiveMaker = archiveMaker
    }

    public func collect(config: AppConfig, backendRunning: Bool) async throws -> DiagnosticBundleArtifact {
        try Task.checkCancellation()
        let identity = DiagnosticBuildIdentity()
        let generatedAt = now()
        let healthResult = await collectHealth(config: config, backendRunning: backendRunning)
        let health = healthResult.snapshot
        let runtime = runtimeHealth(from: health, errorCode: healthResult.errorCode)

        let scanner = scanner
        let versionService = versionService
        let logStore = logStore
        let limits = limits
        async let models = Task.detached(priority: .utility) {
            DiagnosticModelInventoryCollector(scanner: scanner, versionService: versionService).collect(
                loadedModels: health?.models ?? [],
                maximumVersions: limits.maximumModelVersions,
                maximumBytes: limits.modelsBytes
            )
        }.value
        async let logTails = Task.detached(priority: .utility) {
            (
                logStore.diagnosticTail(
                    from: .app,
                    maxLines: limits.appLogLines,
                    maxBytes: limits.appLogBytes
                ),
                logStore.diagnosticTail(
                    from: .backend,
                    maxLines: limits.backendLogLines,
                    maxBytes: limits.backendLogBytes
                )
            )
        }.value

        let incidentsData = try incidentStore.exportData(matching: BackendIncidentQuery())
        let system = systemProfiler.snapshot(buildIdentity: identity)
        let (modelInventory, tails) = await (models, logTails)
        try Task.checkCancellation()

        var entries: [String: Data] = [:]
        entries["system.json"] = try boundedJSON(system, maximumBytes: limits.systemBytes)
        entries["runtime-health.json"] = try boundedJSON(runtime, maximumBytes: limits.runtimeHealthBytes)
        entries["models.json"] = try boundedJSON(modelInventory, maximumBytes: limits.modelsBytes)
        entries["incidents.json"] = DiagnosticPrivacy.sanitizedData(
            incidentsData,
            maximumBytes: limits.incidentsBytes
        )
        entries["logs/app.log"] = Data(DiagnosticPrivacy.sanitizedLog(
            tails.0.text,
            maximumBytes: limits.appLogBytes
        ).utf8)
        entries["logs/backend.log"] = Data(DiagnosticPrivacy.sanitizedLog(
            tails.1.text,
            maximumBytes: limits.backendLogBytes
        ).utf8)

        var fileManifest: [String: DiagnosticFileManifest] = [
            "system.json": .init(status: "generated", bytes: entries["system.json"]!.count, truncated: false),
            "runtime-health.json": .init(
                status: health == nil ? "generated_without_backend" : "generated",
                bytes: entries["runtime-health.json"]!.count,
                truncated: false
            ),
            "models.json": .init(
                status: modelInventory.status,
                bytes: entries["models.json"]!.count,
                truncated: modelInventory.truncated
            ),
            "incidents.json": .init(
                status: "generated",
                bytes: entries["incidents.json"]!.count,
                truncated: String(decoding: incidentsData, as: UTF8.self).contains(#""truncated" : true"#)
            ),
            "logs/app.log": .init(status: tails.0.status, bytes: entries["logs/app.log"]!.count, truncated: tails.0.truncated),
            "logs/backend.log": .init(status: tails.1.status, bytes: entries["logs/backend.log"]!.count, truncated: tails.1.truncated),
        ]
        var manifest = DiagnosticBundleManifest(
            schemaVersion: DiagnosticBundleManifest.schemaVersion,
            generatedAt: generatedAt,
            appVersion: identity.appVersion,
            appBuild: identity.appBuild,
            backendVersion: health?.version,
            ironMLXSourceCommit: identity.sourceCommit,
            mlxCommit: identity.mlxCommit,
            distributionChannel: identity.distributionChannel,
            sourceTreeState: identity.sourceTreeState,
            developerIDStatus: system.developerIDStatus,
            notarizationStatus: system.notarizationStatus,
            backendOnline: health != nil,
            entries: fileManifest,
            totalUncompressedBytes: entries.values.reduce(0) { $0 + $1.count },
            contentTruncated: fileManifest.values.contains(where: \.truncated)
        )
        var manifestData = try boundedJSON(manifest, maximumBytes: limits.manifestBytes)
        for _ in 0..<3 {
            fileManifest["manifest.json"] = .init(status: "generated", bytes: manifestData.count, truncated: false)
            manifest.entries = fileManifest
            manifest.totalUncompressedBytes = entries.values.reduce(manifestData.count) { $0 + $1.count }
            manifestData = try boundedJSON(manifest, maximumBytes: limits.manifestBytes)
        }
        entries["manifest.json"] = manifestData
        guard manifest.totalUncompressedBytes <= limits.maximumUncompressedBytes else {
            throw DiagnosticBundleError.uncompressedCapacityExceeded
        }

        let archiveMaker = archiveMaker
        let archiveTask = Task.detached(priority: .utility) {
            try archiveMaker(entries, limits.maximumArchiveBytes)
        }
        let archiveData = try await withTaskCancellationHandler {
            try await archiveTask.value
        } onCancel: {
            archiveTask.cancel()
        }
        try Task.checkCancellation()
        return DiagnosticBundleArtifact(archiveData: archiveData, manifest: manifest)
    }

    private func collectHealth(
        config: AppConfig,
        backendRunning: Bool
    ) async -> (snapshot: HealthzSnapshot?, errorCode: String?) {
        guard backendRunning else { return (nil, "backend_offline") }
        do {
            return (try await healthFetcher(config, limits.healthTimeout), nil)
        } catch let error as URLError where error.code == .timedOut {
            return (nil, "health_timeout")
        } catch is DecodingError {
            return (nil, "invalid_health_response")
        } catch {
            return (nil, "health_request_failed")
        }
    }

    private func runtimeHealth(from health: HealthzSnapshot?, errorCode: String?) -> DiagnosticRuntimeHealth {
        guard let health else {
            return DiagnosticRuntimeHealth(
                status: "unavailable",
                errorCode: errorCode,
                backendVersion: nil,
                mode: nil,
                deviceName: nil,
                uptimeSeconds: nil,
                scheduler: nil,
                memory: nil,
                activeKV: nil,
                loadedModels: []
            )
        }
        return DiagnosticRuntimeHealth(
            status: health.status,
            errorCode: nil,
            backendVersion: health.version,
            mode: health.mode,
            deviceName: health.deviceName,
            uptimeSeconds: health.uptimeSecs,
            scheduler: health.scheduler,
            memory: health.memory,
            activeKV: DiagnosticActiveKV(health.activeKvOffload),
            loadedModels: health.models.map {
                DiagnosticLoadedModel(
                    id: DiagnosticPrivacy.stableModelReference($0.id),
                    repoID: DiagnosticPrivacy.stableModelReference($0.model),
                    architecture: $0.architecture,
                    runtimeKind: $0.capabilities.runtimeKind,
                    runtimeState: $0.runtimeState,
                    scheduler: $0.scheduler,
                    isDefault: $0.isDefault,
                    pinned: $0.pinned,
                    mtpEnabled: $0.mtpEnabled,
                    activeRequests: $0.activeRequests,
                    queuedRequests: $0.queuedRequests,
                    queueCapacity: $0.queueCapacity
                )
            }
        )
    }

    private func boundedJSON<T: Encodable>(_ value: T, maximumBytes: Int) throws -> Data {
        let data = try JSONEncoder.ironMLXDiagnostic.encode(value)
        guard data.count <= maximumBytes else { throw DiagnosticBundleError.uncompressedCapacityExceeded }
        return DiagnosticPrivacy.sanitizedData(data, maximumBytes: maximumBytes)
    }

    nonisolated public static func makeArchive(
        entries: [String: Data],
        maximumBytes: Int
    ) throws -> Data {
        let archive = try Archive(accessMode: .create)
        let fixedDate = Date(timeIntervalSince1970: 315_532_800)
        for path in entryOrder {
            try Task.checkCancellation()
            guard let data = entries[path] else { throw DiagnosticBundleError.archiveUnavailable }
            try archive.addEntry(
                with: path,
                type: .file,
                uncompressedSize: Int64(data.count),
                modificationDate: fixedDate,
                permissions: 0o600,
                compressionMethod: .deflate,
                bufferSize: 32_768,
                provider: { position, size in
                    try Task.checkCancellation()
                    let start = Int(position)
                    return data.subdata(in: start..<min(start + size, data.count))
                }
            )
        }
        guard let data = archive.data else { throw DiagnosticBundleError.archiveUnavailable }
        guard data.count <= maximumBytes else { throw DiagnosticBundleError.archiveCapacityExceeded }
        return data
    }
}
