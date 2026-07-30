import CryptoKit
import Darwin
import Foundation

public enum ModelCacheMigrationPhase: String, Codable, Sendable {
    case validating
    case manifested
    case renamed
    case completed
    case failed
}

public struct ModelCacheMigrationJournal: Codable, Sendable {
    public var version = 1
    public var repoID: String
    public var commitSHA: String?
    public var phase: ModelCacheMigrationPhase
    public var error: String?
    public var updatedAt = Date()

    enum CodingKeys: String, CodingKey {
        case version
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case phase
        case error
        case updatedAt = "updated_at"
    }
}

public struct ModelCacheMigrationResult: Codable, Sendable {
    public var repoID: String
    public var commitSHA: String
    public var fileCount: Int
    public var weightBytes: Int64
    public var status: String

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case fileCount = "file_count"
        case weightBytes = "weight_bytes"
        case status
    }
}

public struct ModelCacheMigrationService: Sendable {
    private let rootURL: URL
    private let resolver: ModelRepositoryResolver
    private let metadataPreflight: any ModelMetadataPreflighting
    private let token: String?

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true),
        httpClient: any ModelDownloadHTTPClient = URLSessionModelDownloadHTTPClient(),
        huggingFaceEndpoint: URL = URL(string: "https://huggingface.co")!,
        metadataPreflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight(),
        token: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
    ) {
        self.rootURL = rootURL
        resolver = ModelRepositoryResolver(
            httpClient: httpClient,
            huggingFaceEndpoint: huggingFaceEndpoint,
            modelScopeAPIEndpoint: URL(string: "https://modelscope.cn/api/v1/models")!
        )
        self.metadataPreflight = metadataPreflight
        self.token = token
    }

    public func migrateAll(
        progress: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> [ModelCacheMigrationResult] {
        let legacyRoot = rootURL.appendingPathComponent("models", isDirectory: true)
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: legacyRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        let repositories = entries.compactMap(Self.legacyRepoID).sorted()
        var results: [ModelCacheMigrationResult] = []
        for repoID in repositories {
            progress("validating \(repoID)")
            do {
                let result = try await migrate(repoID: repoID, progress: progress)
                results.append(result)
            } catch {
                try? writeJournal(
                    ModelCacheMigrationJournal(
                        repoID: repoID,
                        phase: .failed,
                        error: error.localizedDescription
                    )
                )
                throw error
            }
        }
        return results
    }

    public func migrate(
        repoID: String,
        progress: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> ModelCacheMigrationResult {
        let legacy = legacyRepository(repoID: repoID)
        let destination = try ModelRepositoryLayout.repositoryRoot(
            rootURL: rootURL,
            provider: .huggingFace,
            repoID: repoID
        )
        let lock = try ModelRepositoryLock(lockURL: migrationLockURL(repoID: repoID))
        defer { withExtendedLifetime(lock) {} }

        if FileManager.default.fileExists(atPath: destination.path),
           !FileManager.default.fileExists(atPath: legacy.path) {
            let snapshot = try activeSnapshot(repository: destination)
            let manifest = try ModelSnapshotVerifier().verifyStructure(
                snapshot: snapshot,
                expectedProvider: .huggingFace,
                expectedRepoID: repoID
            )
            try writeJournal(
                ModelCacheMigrationJournal(
                    repoID: repoID,
                    commitSHA: manifest.commitSHA,
                    phase: .completed
                )
            )
            return ModelCacheMigrationResult(
                repoID: repoID,
                commitSHA: manifest.commitSHA,
                fileCount: manifest.files.count,
                weightBytes: manifest.resources.weightBytes,
                status: "already_migrated"
            )
        }
        guard FileManager.default.fileExists(atPath: legacy.path) else {
            throw ModelCacheMigrationError.legacyRepositoryMissing(repoID)
        }
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ModelCacheMigrationError.destinationExists(destination.path)
        }

        let commitSHA = try localCommit(repository: legacy)
        var journal = ModelCacheMigrationJournal(
            repoID: repoID,
            commitSHA: commitSHA,
            phase: .validating
        )
        try writeJournal(journal)

        let remote = try await resolver.resolveHuggingFace(
            repoID: repoID,
            revision: commitSHA,
            token: token
        )
        guard remote.commitSHA == commitSHA else {
            throw ModelCacheMigrationError.commitMismatch(local: commitSHA, remote: remote.commitSHA)
        }
        let snapshot = legacy
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(commitSHA, isDirectory: true)
        let compatibility = try await metadataPreflight.validate(metadataDirectory: snapshot)
        let validationResult = try validateLocalFiles(
            snapshot: snapshot,
            remoteFiles: remote.files,
            repoID: repoID,
            progress: progress
        )
        let files = validationResult.files
        let weightBytes = files
            .filter { $0.path.hasSuffix(".safetensors") }
            .reduce(Int64(0)) { $0 + $1.size }
        let resources = ModelSnapshotResources(
            weightBytes: weightBytes,
            estimatedPeakMemoryBytes: weightBytes + max(512 * 1_024 * 1_024, weightBytes / 10)
        )
        let manifest = ModelSnapshotManifest(
            provider: .huggingFace,
            repoID: repoID,
            requestedRevision: ModelRepositoryProvider.huggingFace.mutableRevision,
            commitSHA: commitSHA,
            files: files,
            compatibility: ModelSnapshotCompatibility(
                modelType: compatibility.modelType,
                artifactRole: compatibility.artifactRole,
                quantizationMode: compatibility.quantization?.mode,
                quantizationBits: compatibility.quantization?.bits,
                quantizationGroupSize: compatibility.quantization?.groupSize
            ),
            resources: resources
        )
        try ModelDownloadStore(rootURL: rootURL).writeManifest(manifest, to: snapshot)
        try ModelSnapshotVerifier().verifyForPublish(
            snapshot: snapshot,
            manifest: manifest,
            validations: validationResult.validations
        )
        try ModelDownloadStore.atomicWrite(
            ModelSnapshotIntegrityRecord(
                provider: .huggingFace,
                repoID: repoID,
                commitSHA: commitSHA,
                state: .verified,
                verifiedAt: Date()
            ),
            to: snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
        )
        try writeMainRef(repository: legacy, commitSHA: commitSHA)
        journal.phase = .manifested
        journal.updatedAt = Date()
        try writeJournal(journal)

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard rename(legacy.path, destination.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        try syncDirectory(destination.deletingLastPathComponent())
        journal.phase = .renamed
        journal.updatedAt = Date()
        try writeJournal(journal)

        let published = try activeSnapshot(repository: destination)
        _ = try ModelSnapshotVerifier().verifyStructure(
            snapshot: published,
            expectedProvider: .huggingFace,
            expectedRepoID: repoID
        )
        journal.phase = .completed
        journal.updatedAt = Date()
        try writeJournal(journal)
        progress("completed \(repoID)")
        return ModelCacheMigrationResult(
            repoID: repoID,
            commitSHA: commitSHA,
            fileCount: files.count,
            weightBytes: weightBytes,
            status: "migrated"
        )
    }

    private func validateLocalFiles(
        snapshot: URL,
        remoteFiles: [RemoteModelFile],
        repoID: String,
        progress: @escaping @Sendable (String) -> Void
    ) throws -> (files: [ModelSnapshotFile], validations: [ModelValidatedFile]) {
        let local = try localFiles(in: snapshot)
        let remoteByPath = Dictionary(uniqueKeysWithValues: remoteFiles.map { ($0.path, $0) })
        let localPaths = Set(local.map(\.path))
        let remotePaths = Set(remoteByPath.keys)
        guard localPaths == remotePaths else {
            throw ModelCacheMigrationError.inventoryMismatch(
                missing: remotePaths.subtracting(localPaths).sorted(),
                unexpected: localPaths.subtracting(remotePaths).sorted()
            )
        }

        var verified: [ModelSnapshotFile] = []
        var validations: [ModelValidatedFile] = []
        for item in local {
            guard let remote = remoteByPath[item.path] else {
                throw ModelCacheMigrationError.identityMissing(item.path)
            }
            let actualSize = try Self.fileSize(item.url)
            guard actualSize == remote.size else {
                throw ModelSnapshotVerificationError.sizeMismatch(
                    path: item.path,
                    expected: remote.size,
                    actual: actualSize
                )
            }
            progress("hashing \(repoID)/\(item.path)")
            let identityBefore = try ModelSnapshotVerifier.fileIdentity(of: item.url)
            let hashes = try Self.hashes(
                of: item.url,
                size: actualSize,
                includeGitBlob: remote.sha256 == nil
            )
            let identityAfter = try ModelSnapshotVerifier.fileIdentity(of: item.url)
            guard identityBefore == identityAfter else {
                throw ModelSnapshotVerificationError.fileChangedDuringVerification(item.path)
            }
            if let expected = remote.sha256 {
                guard hashes.sha256 == expected.lowercased() else {
                    throw ModelSnapshotVerificationError.checksumMismatch(
                        path: item.path,
                        expected: expected,
                        actual: hashes.sha256
                    )
                }
            } else {
                guard let expected = remote.blobID?.lowercased(),
                      ModelSnapshotVerifier.isCommitSHA(expected),
                      hashes.gitBlobSHA1 == expected
                else {
                    throw ModelCacheMigrationError.identityMismatch(item.path)
                }
            }
            verified.append(
                ModelSnapshotFile(
                    path: item.path,
                    size: actualSize,
                    sha256: hashes.sha256,
                    etag: remote.etag,
                    blobID: remote.blobID
                )
            )
            validations.append(
                ModelValidatedFile(
                    path: item.path,
                    sha256: hashes.sha256,
                    identity: identityAfter
                )
            )
        }
        return (
            verified.sorted { $0.path < $1.path },
            validations.sorted { $0.path < $1.path }
        )
    }

    private func localFiles(in snapshot: URL) throws -> [(path: String, url: URL)] {
        guard let enumerator = FileManager.default.enumerator(
            at: snapshot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: []
        ) else {
            throw ModelCacheMigrationError.snapshotMissing(snapshot.path)
        }
        var files: [(String, URL)] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isDirectoryKey])
            if values.isDirectory == true
                || url.lastPathComponent == ModelSnapshotManifest.filename
                || url.lastPathComponent == ModelSnapshotIntegrityRecord.filename {
                continue
            }
            let components = url.pathComponents
            guard enumerator.level > 0, enumerator.level <= components.count else {
                throw ModelCacheMigrationError.identityMissing(url.path)
            }
            files.append((components.suffix(enumerator.level).joined(separator: "/"), url))
        }
        return files.sorted { $0.0 < $1.0 }
    }

    private func localCommit(repository: URL) throws -> String {
        let main = repository
            .appendingPathComponent("refs", isDirectory: true)
            .appendingPathComponent("main")
        if let data = try? Data(contentsOf: main),
           let string = String(data: data, encoding: .utf8) {
            let commit = string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard ModelSnapshotVerifier.isCommitSHA(commit) else {
                throw ModelCacheMigrationError.invalidCommit(commit)
            }
            return commit
        }
        let snapshots = repository.appendingPathComponent("snapshots", isDirectory: true)
        let entries = try FileManager.default.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ).filter { ModelSnapshotVerifier.isCommitSHA($0.lastPathComponent) }
        guard entries.count == 1 else {
            throw ModelCacheMigrationError.commitUnavailable(repository.lastPathComponent)
        }
        return entries[0].lastPathComponent.lowercased()
    }

    private func activeSnapshot(repository: URL) throws -> URL {
        let commit = try localCommit(repository: repository)
        let snapshot = repository
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(commit, isDirectory: true)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw ModelCacheMigrationError.snapshotMissing(snapshot.path)
        }
        return snapshot
    }

    private func writeMainRef(repository: URL, commitSHA: String) throws {
        try ModelDownloadStore.atomicWrite(
            Data((commitSHA + "\n").utf8),
            to: repository
                .appendingPathComponent("refs", isDirectory: true)
                .appendingPathComponent("main")
        )
    }

    private func writeJournal(_ journal: ModelCacheMigrationJournal) throws {
        try ModelDownloadStore.atomicWrite(journal, to: migrationJournalURL(repoID: journal.repoID))
    }

    private func legacyRepository(repoID: String) -> URL {
        rootURL
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }

    private func migrationLockURL(repoID: String) -> URL {
        rootURL
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(".migration-locks", isDirectory: true)
            .appendingPathComponent(repoID.replacingOccurrences(of: "/", with: "--") + ".lock")
    }

    private func migrationJournalURL(repoID: String) -> URL {
        rootURL
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(".migrations", isDirectory: true)
            .appendingPathComponent(repoID.replacingOccurrences(of: "/", with: "--") + ".json")
    }

    private static func legacyRepoID(_ url: URL) -> String? {
        let name = url.lastPathComponent
        guard name.hasPrefix("models--") else {
            return nil
        }
        let identity = String(name.dropFirst("models--".count))
        guard let separator = identity.range(of: "--") else {
            return nil
        }
        return String(identity[..<separator.lowerBound]) + "/" + String(identity[separator.upperBound...])
    }

    private static func fileSize(_ url: URL) throws -> Int64 {
        try ModelSnapshotVerifier.fileSize(of: url)
    }

    private static func hashes(
        of url: URL,
        size: Int64,
        includeGitBlob: Bool
    ) throws -> (sha256: String, gitBlobSHA1: String?) {
        var sha256 = SHA256()
        var gitBlob = Insecure.SHA1()
        if includeGitBlob {
            gitBlob.update(data: Data("blob \(size)\0".utf8))
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        while true {
            let data = try handle.read(upToCount: 4 * 1_024 * 1_024) ?? Data()
            if data.isEmpty {
                break
            }
            sha256.update(data: data)
            if includeGitBlob {
                gitBlob.update(data: data)
            }
        }
        let sha256String = sha256.finalize().map { String(format: "%02x", $0) }.joined()
        let gitBlobString = includeGitBlob
            ? gitBlob.finalize().map { String(format: "%02x", $0) }.joined()
            : nil
        return (sha256String, gitBlobString)
    }

    private func syncDirectory(_ url: URL) throws {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { Darwin.close(descriptor) }
        guard fsync(descriptor) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }
}

public enum ModelCacheMigrationError: LocalizedError {
    case legacyRepositoryMissing(String)
    case destinationExists(String)
    case invalidCommit(String)
    case commitUnavailable(String)
    case commitMismatch(local: String, remote: String)
    case snapshotMissing(String)
    case inventoryMismatch(missing: [String], unexpected: [String])
    case identityMissing(String)
    case identityMismatch(String)

    public var errorDescription: String? {
        switch self {
        case let .legacyRepositoryMissing(repoID):
            "Legacy repository is missing: \(repoID)."
        case let .destinationExists(path):
            "Migration destination already exists: \(path)."
        case let .invalidCommit(commit):
            "Local ref is not a full commit SHA: \(commit)."
        case let .commitUnavailable(repository):
            "Cannot determine an unambiguous commit for \(repository)."
        case let .commitMismatch(local, remote):
            "Remote commit \(remote) does not match local commit \(local)."
        case let .snapshotMissing(path):
            "Snapshot is missing: \(path)."
        case let .inventoryMismatch(missing, unexpected):
            "Local/remote file inventory mismatch; missing=\(missing), unexpected=\(unexpected)."
        case let .identityMissing(path):
            "Remote identity metadata is missing for \(path)."
        case let .identityMismatch(path):
            "Remote identity validation failed for \(path)."
        }
    }
}
