import Darwin
import Foundation

public enum ModelVersionIntegrityState: String, Codable, Sendable {
    case verified
    case corrupt
    case unverified
}

public struct ModelVersionInfo: Codable, Equatable, Sendable {
    public var provider: String
    public var repoID: String
    public var commitSHA: String
    public var requestedRevision: String?
    public var publishedAt: Date?
    public var verifiedAt: Date?
    public var sizeBytes: Int64
    public var integrityState: ModelVersionIntegrityState
    public var isActive: Bool
    public var isLoaded: Bool
    public var isReferenced: Bool
    public var canActivate: Bool
    public var canDelete: Bool
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case provider
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case requestedRevision = "requested_revision"
        case publishedAt = "published_at"
        case verifiedAt = "verified_at"
        case sizeBytes = "size_bytes"
        case integrityState = "integrity_state"
        case isActive = "is_active"
        case isLoaded = "is_loaded"
        case isReferenced = "is_referenced"
        case canActivate = "can_activate"
        case canDelete = "can_delete"
        case error
    }
}

public struct ModelVersionList: Codable, Equatable, Sendable {
    public var provider: String
    public var repoID: String
    public var activeCommitSHA: String?
    public var versions: [ModelVersionInfo]
    public var reclaimableBytes: Int64

    enum CodingKeys: String, CodingKey {
        case provider
        case repoID = "repo_id"
        case activeCommitSHA = "active_commit_sha"
        case versions
        case reclaimableBytes = "reclaimable_bytes"
    }
}

public struct ModelVersionActivationResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var provider: String
    public var repoID: String
    public var previousCommitSHA: String?
    public var activeCommitSHA: String

    enum CodingKeys: String, CodingKey {
        case success
        case provider
        case repoID = "repo_id"
        case previousCommitSHA = "previous_commit_sha"
        case activeCommitSHA = "active_commit_sha"
    }
}

public struct ModelVersionDeletionResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var provider: String
    public var repoID: String
    public var deletedCommitSHAs: [String]
    public var reclaimedBytes: Int64

    enum CodingKeys: String, CodingKey {
        case success
        case provider
        case repoID = "repo_id"
        case deletedCommitSHAs = "deleted_commit_shas"
        case reclaimedBytes = "reclaimed_bytes"
    }
}

public enum ModelSearchLocalState: String, Codable, Sendable {
    case available
    case exists
    case updateAvailable = "update_available"
    case localInactive = "local_inactive"
    case repair
    case identityUnavailable = "identity_unavailable"
}

public enum ModelVersionManagementError: LocalizedError {
    case repositoryUnavailable(String)
    case invalidCommit(String)
    case versionUnavailable(String)
    case versionUnverified(String)
    case versionReferenced(String)
    case versionLoaded(String)

    public var errorDescription: String? {
        switch self {
        case .repositoryUnavailable(let repoID):
            "Model repository \(repoID) is not available locally."
        case .invalidCommit(let commitSHA):
            "Invalid model commit SHA \(commitSHA)."
        case .versionUnavailable(let commitSHA):
            "Model version \(commitSHA) is not available."
        case .versionUnverified(let commitSHA):
            "Model version \(commitSHA) is not verified and cannot be activated."
        case .versionReferenced(let commitSHA):
            "Model version \(commitSHA) is still referenced and cannot be deleted."
        case .versionLoaded(let commitSHA):
            "Model version \(commitSHA) is currently loaded and cannot be deleted."
        }
    }
}

public struct ModelVersionManagementService: @unchecked Sendable {
    public let rootURL: URL
    private let fileManager: FileManager

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true),
        fileManager: FileManager = .default
    ) {
        self.rootURL = rootURL
        self.fileManager = fileManager
    }

    public func versions(
        provider: ModelRepositoryProvider,
        repoID: String,
        loadedModelPaths: Set<String> = []
    ) throws -> ModelVersionList {
        let store = ModelDownloadStore(rootURL: rootURL)
        let repository = try store.repositoryRoot(provider: provider, repoID: repoID)
        let snapshots = repository.appendingPathComponent("snapshots", isDirectory: true)
        let references = refTargets(repository: repository)
        let activeCommit = references[provider.mutableRevision]
        let loadedPaths = Set(loadedModelPaths.map(canonicalPath))
        let directories = try fileManager.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        let versions = directories.compactMap { snapshot -> ModelVersionInfo? in
            let commitSHA = snapshot.lastPathComponent.lowercased()
            guard ModelSnapshotVerifier.isCommitSHA(commitSHA),
                (try? snapshot.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            else {
                return nil
            }
            return inspectVersion(
                snapshot: snapshot,
                provider: provider,
                repoID: repoID,
                commitSHA: commitSHA,
                activeCommit: activeCommit,
                referencedCommits: Set(references.values),
                loadedPaths: loadedPaths
            )
        }
        .sorted {
            if $0.isActive != $1.isActive {
                return $0.isActive
            }
            return ($0.publishedAt ?? .distantPast) > ($1.publishedAt ?? .distantPast)
        }
        return ModelVersionList(
            provider: provider.rawValue,
            repoID: repoID,
            activeCommitSHA: activeCommit,
            versions: versions,
            reclaimableBytes: versions.filter(\.canDelete).reduce(0) { $0 + $1.sizeBytes }
        )
    }

    public func searchLocalState(
        provider: ModelRepositoryProvider,
        repoID: String,
        remoteCommitSHA: String?,
        loadedModelPaths: Set<String> = []
    ) -> (state: ModelSearchLocalState, localCommitSHA: String?) {
        guard
            let list = try? versions(
                provider: provider,
                repoID: repoID,
                loadedModelPaths: loadedModelPaths
            ), !list.versions.isEmpty
        else {
            return (.available, nil)
        }
        guard let remoteCommit = remoteCommitSHA?.lowercased(),
              ModelSnapshotVerifier.isCommitSHA(remoteCommit)
        else {
            return (.identityUnavailable, list.activeCommitSHA)
        }
        if let exact = list.versions.first(where: { $0.commitSHA == remoteCommit }) {
            if exact.integrityState != .verified {
                return (.repair, list.activeCommitSHA)
            }
            return exact.isActive
                ? (.exists, exact.commitSHA)
                : (.localInactive, list.activeCommitSHA)
        }
        return (.updateAvailable, list.activeCommitSHA)
    }

    public func activate(
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHA rawCommitSHA: String,
        fullChecksum: Bool
    ) throws -> ModelVersionActivationResult {
        let commitSHA = try validatedCommit(rawCommitSHA)
        let store = ModelDownloadStore(rootURL: rootURL)
        let repository = try store.repositoryRoot(provider: provider, repoID: repoID)
        guard fileManager.fileExists(atPath: repository.path) else {
            throw ModelVersionManagementError.repositoryUnavailable(repoID)
        }
        let lock = try store.acquireRepositoryLock(provider: provider, repoID: repoID)
        defer { withExtendedLifetime(lock) {} }
        let snapshot = try store.snapshotURL(
            provider: provider,
            repoID: repoID,
            commitSHA: commitSHA
        )
        guard fileManager.fileExists(atPath: snapshot.path) else {
            throw ModelVersionManagementError.versionUnavailable(commitSHA)
        }
        let verifier = ModelSnapshotVerifier()
        let manifest: ModelSnapshotManifest
        if fullChecksum {
            do {
                manifest = try verifier.verify(
                    snapshot: snapshot,
                    expectedProvider: provider,
                    expectedRepoID: repoID
                )
                try ModelDownloadStore.atomicWrite(
                    ModelSnapshotIntegrityRecord(
                        provider: provider,
                        repoID: repoID,
                        commitSHA: commitSHA,
                        state: .verified,
                        verifiedAt: Date()
                    ),
                    to: snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
                )
            } catch {
                try? ModelDownloadStore.atomicWrite(
                    ModelSnapshotIntegrityRecord(
                        provider: provider,
                        repoID: repoID,
                        commitSHA: commitSHA,
                        state: .corrupt,
                        error: error.localizedDescription
                    ),
                    to: snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
                )
                throw error
            }
        } else {
            manifest = try verifier.verifyStructure(
                snapshot: snapshot,
                expectedProvider: provider,
                expectedRepoID: repoID
            )
            guard let record = try? verifier.loadIntegrityRecord(at: snapshot),
                record.provider == provider,
                record.repoID == repoID,
                record.commitSHA == commitSHA,
                record.state == .verified
            else {
                throw ModelVersionManagementError.versionUnverified(commitSHA)
            }
        }
        let previousCommit = refTargets(repository: repository)[provider.mutableRevision]
        try store.updateRef(for: manifest)
        return ModelVersionActivationResult(
            success: true,
            provider: provider.rawValue,
            repoID: repoID,
            previousCommitSHA: previousCommit,
            activeCommitSHA: commitSHA
        )
    }

    public func deleteVersions(
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHAs rawCommitSHAs: [String],
        loadedModelPaths: Set<String>
    ) throws -> ModelVersionDeletionResult {
        let commitSHAs = try Array(
            Set(rawCommitSHAs.map { try validatedCommit($0) })
        ).sorted()
        let store = ModelDownloadStore(rootURL: rootURL)
        let repository = try store.repositoryRoot(provider: provider, repoID: repoID)
        guard fileManager.fileExists(atPath: repository.path) else {
            throw ModelVersionManagementError.repositoryUnavailable(repoID)
        }

        var trashURLs: [URL] = []
        var reclaimedBytes: Int64 = 0
        do {
            let repositoryLock = try store.acquireRepositoryLock(provider: provider, repoID: repoID)
            defer { withExtendedLifetime(repositoryLock) {} }
            let referencedCommits = Set(refTargets(repository: repository).values)
            let loadedPaths = Set(loadedModelPaths.map(canonicalPath))
            var useLocks: [ModelSnapshotUseLock] = []
            var deletionPlan: [(commitSHA: String, snapshot: URL, trash: URL, size: Int64)] = []
            let trashRoot = repository.appendingPathComponent(".trash", isDirectory: true)
            try fileManager.createDirectory(at: trashRoot, withIntermediateDirectories: true)

            for commitSHA in commitSHAs {
                guard !referencedCommits.contains(commitSHA) else {
                    throw ModelVersionManagementError.versionReferenced(commitSHA)
                }
                let snapshot = try store.snapshotURL(
                    provider: provider,
                    repoID: repoID,
                    commitSHA: commitSHA
                )
                guard fileManager.fileExists(atPath: snapshot.path) else {
                    throw ModelVersionManagementError.versionUnavailable(commitSHA)
                }
                guard !loadedPaths.contains(canonicalPath(snapshot.path)) else {
                    throw ModelVersionManagementError.versionLoaded(commitSHA)
                }
                useLocks.append(
                    try store.acquireSnapshotUseLock(
                        provider: provider,
                        repoID: repoID,
                        commitSHA: commitSHA,
                        exclusive: true
                    )
                )
                let trash = trashRoot.appendingPathComponent(
                    "\(commitSHA)-\(UUID().uuidString)",
                    isDirectory: true
                )
                deletionPlan.append(
                    (
                        commitSHA: commitSHA,
                        snapshot: snapshot,
                        trash: trash,
                        size: directorySize(snapshot)
                    )
                )
            }

            for item in deletionPlan {
                reclaimedBytes += item.size
                guard rename(item.snapshot.path, item.trash.path) == 0 else {
                    let renameError = errno
                    for moved in deletionPlan.prefix(trashURLs.count).reversed() {
                        _ = rename(moved.trash.path, moved.snapshot.path)
                    }
                    try? ModelDownloadStore.syncDirectory(
                        repository.appendingPathComponent("snapshots", isDirectory: true)
                    )
                    try? ModelDownloadStore.syncDirectory(trashRoot)
                    trashURLs.removeAll()
                    reclaimedBytes = 0
                    throw POSIXError(POSIXErrorCode(rawValue: renameError) ?? .EIO)
                }
                trashURLs.append(item.trash)
            }
            try ModelDownloadStore.syncDirectory(
                repository.appendingPathComponent("snapshots", isDirectory: true)
            )
            try ModelDownloadStore.syncDirectory(trashRoot)
            withExtendedLifetime(useLocks) {}
        }
        for trash in trashURLs {
            try? fileManager.removeItem(at: trash)
        }
        return ModelVersionDeletionResult(
            success: true,
            provider: provider.rawValue,
            repoID: repoID,
            deletedCommitSHAs: commitSHAs,
            reclaimedBytes: reclaimedBytes
        )
    }

    public func purgeTrash() {
        for provider in ModelRepositoryProvider.allCases {
            let providerRoot = ModelRepositoryLayout.providerRoot(
                rootURL: rootURL,
                provider: provider
            )
            guard
                let repositories = try? fileManager.contentsOfDirectory(
                    at: providerRoot,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles]
                )
            else {
                continue
            }
            for repository in repositories {
                guard let repoID = repoID(fromRepositoryName: repository.lastPathComponent),
                    let lock = try? ModelDownloadStore(rootURL: rootURL).acquireRepositoryLock(
                        provider: provider,
                        repoID: repoID
                    )
                else {
                    continue
                }
                let trash = repository.appendingPathComponent(".trash", isDirectory: true)
                try? fileManager.removeItem(at: trash)
                withExtendedLifetime(lock) {}
            }
        }
    }

    private func inspectVersion(
        snapshot: URL,
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHA: String,
        activeCommit: String?,
        referencedCommits: Set<String>,
        loadedPaths: Set<String>
    ) -> ModelVersionInfo {
        let verifier = ModelSnapshotVerifier()
        var manifest: ModelSnapshotManifest?
        var integrityState = ModelVersionIntegrityState.unverified
        var verifiedAt: Date?
        var error: String?
        do {
            manifest = try verifier.verifyStructure(
                snapshot: snapshot,
                expectedProvider: provider,
                expectedRepoID: repoID,
                allowKnownCorrupt: true
            )
            if let record = try? verifier.loadIntegrityRecord(at: snapshot),
                record.provider == provider,
                record.repoID == repoID,
                record.commitSHA == commitSHA
            {
                integrityState = record.state == .verified ? .verified : .corrupt
                verifiedAt = record.verifiedAt
                error = record.error
            }
        } catch let inspectionError {
            if fileManager.fileExists(
                atPath: snapshot.appendingPathComponent(ModelSnapshotManifest.filename).path
            ) {
                integrityState = .corrupt
            }
            error = inspectionError.localizedDescription
        }
        let isActive = activeCommit == commitSHA
        let isLoaded = loadedPaths.contains(canonicalPath(snapshot.path))
        let isReferenced = referencedCommits.contains(commitSHA)
        let sizeBytes = manifest?.files.reduce(0) { $0 + $1.size } ?? directorySize(snapshot)
        return ModelVersionInfo(
            provider: provider.rawValue,
            repoID: repoID,
            commitSHA: commitSHA,
            requestedRevision: manifest?.requestedRevision,
            publishedAt: manifest?.publishedAt,
            verifiedAt: verifiedAt,
            sizeBytes: sizeBytes,
            integrityState: integrityState,
            isActive: isActive,
            isLoaded: isLoaded,
            isReferenced: isReferenced,
            canActivate: !isActive && integrityState == .verified && manifest != nil,
            canDelete: !isReferenced && !isLoaded,
            error: error
        )
    }

    private func validatedCommit(_ rawCommitSHA: String) throws -> String {
        let commitSHA =
            rawCommitSHA
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard ModelSnapshotVerifier.isCommitSHA(commitSHA) else {
            throw ModelVersionManagementError.invalidCommit(rawCommitSHA)
        }
        return commitSHA
    }

    private func refTargets(repository: URL) -> [String: String] {
        let refs = repository.appendingPathComponent("refs", isDirectory: true)
        guard
            let entries = try? fileManager.contentsOfDirectory(
                at: refs,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return [:]
        }
        var result: [String: String] = [:]
        for entry in entries {
            guard let data = try? Data(contentsOf: entry),
                let raw = String(data: data, encoding: .utf8)
            else {
                continue
            }
            let commitSHA = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard ModelSnapshotVerifier.isCommitSHA(commitSHA) else {
                continue
            }
            result[entry.lastPathComponent] = commitSHA
        }
        return result
    }

    private func canonicalPath(_ value: String) -> String {
        URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath().path
    }

    private func repoID(fromRepositoryName name: String) -> String? {
        guard let separator = name.range(of: "--"),
            !name[..<separator.lowerBound].isEmpty,
            !name[separator.upperBound...].isEmpty,
            name[separator.upperBound...].range(of: "--") == nil
        else {
            return nil
        }
        return "\(name[..<separator.lowerBound])/\(name[separator.upperBound...])"
    }

    private func directorySize(_ directory: URL) -> Int64 {
        guard
            let enumerator = fileManager.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return 0
        }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            guard
                let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                values.isRegularFile == true,
                let size = values.fileSize
            else {
                continue
            }
            let next = total.addingReportingOverflow(Int64(size))
            total = next.overflow ? Int64.max : next.partialValue
        }
        return total
    }
}
