import Foundation

public struct DiagnosticModelInventoryCollector: Sendable {
    public var scanner: LocalModelScanner
    public var versionService: ModelVersionManagementService

    public init(
        scanner: LocalModelScanner = LocalModelScanner(),
        versionService: ModelVersionManagementService = ModelVersionManagementService()
    ) {
        self.scanner = scanner
        self.versionService = versionService
    }

    public func collect(
        loadedModels: [BackendLoadedModelInfo],
        maximumVersions: Int,
        maximumBytes: Int
    ) -> DiagnosticModelInventory {
        let loadedReferences = Set(loadedModels.flatMap { [$0.id, $0.model] })
        let pinned = Set(loadedModels.filter(\.pinned).flatMap { [$0.id, $0.model] })
        let mtp = Set(loadedModels.filter(\.mtpEnabled).flatMap { [$0.id, $0.model] })
        let dflash2 = Set(loadedModels.filter { $0.dflash2?.enabled == true }.flatMap {
            [$0.id, $0.model]
        })
        let localModels = scanner.scan(
            loadedModels: loadedReferences,
            pinnedModels: pinned,
            mtpEnabledModels: mtp,
            dflash2EnabledModels: dflash2
        )
        var records: [DiagnosticModelVersion] = []
        var metadataFailed = false
        for model in localModels {
            guard records.count < maximumVersions else { break }
            guard let provider = provider(for: model.source) else {
                metadataFailed = true
                records.append(unavailableRecord(model: model, provider: "unknown"))
                continue
            }
            do {
                let list = try versionService.versions(
                    provider: provider,
                    repoID: model.repoID,
                    loadedModelPaths: Set(loadedModels.map(\.path))
                )
                if list.versions.isEmpty {
                    records.append(unavailableRecord(model: model, provider: provider.rawValue))
                } else {
                    records.append(contentsOf: list.versions.prefix(maximumVersions - records.count).map { version in
                        DiagnosticModelVersion(
                            provider: provider.rawValue,
                            repoID: DiagnosticPrivacy.stableModelReference(model.repoID),
                            commitSHA: version.commitSHA,
                            requestedRevision: version.requestedRevision,
                            quantization: model.quantization,
                            modelType: model.type,
                            architecture: model.architecture,
                            runtimeKind: model.capabilities?.runtimeKind,
                            supportsVision: model.capabilities?.supportsVision,
                            supportsMTP: model.capabilities?.supportsMtp,
                            isLoaded: version.isLoaded || (model.loaded && version.isActive),
                            isActiveRevision: version.isActive,
                            integrityStatus: version.integrityState.rawValue,
                            verifiedAt: version.verifiedAt,
                            metadataStatus: "available"
                        )
                    })
                }
            } catch {
                metadataFailed = true
                records.append(unavailableRecord(model: model, provider: provider.rawValue))
            }
        }

        var truncated = records.count >= maximumVersions && localModels.count > records.count
        var inventory = DiagnosticModelInventory(
            status: metadataFailed ? "partial" : "available",
            truncated: truncated,
            models: records
        )
        while let data = try? JSONEncoder.ironMLXDiagnostic.encode(inventory),
              data.count > maximumBytes,
              !inventory.models.isEmpty {
            inventory.models.removeLast()
            inventory.truncated = true
            truncated = true
        }
        if truncated { inventory.truncated = true }
        return inventory
    }

    private func unavailableRecord(model: LocalModel, provider: String) -> DiagnosticModelVersion {
        DiagnosticModelVersion(
            provider: provider,
            repoID: DiagnosticPrivacy.stableModelReference(model.repoID),
            commitSHA: nil,
            requestedRevision: nil,
            quantization: model.quantization,
            modelType: model.type,
            architecture: model.architecture,
            runtimeKind: model.capabilities?.runtimeKind,
            supportsVision: model.capabilities?.supportsVision,
            supportsMTP: model.capabilities?.supportsMtp,
            isLoaded: model.loaded,
            isActiveRevision: false,
            integrityStatus: model.integrity?.state ?? "unavailable",
            verifiedAt: nil,
            metadataStatus: "unavailable"
        )
    }

    private func provider(for source: String) -> ModelRepositoryProvider? {
        switch source {
        case "hf": .huggingFace
        case "ms": .modelScope
        default: nil
        }
    }
}
