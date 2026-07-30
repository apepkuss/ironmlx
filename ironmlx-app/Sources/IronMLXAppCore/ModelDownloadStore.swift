import Darwin
import Foundation

public enum ModelDownloadPhase: String, Codable, Sendable {
    case resolving
    case preflighting
    case downloading
    case verifying
    case publishing
    case completed
    case cancelled
    case interrupted
    case rejected
    case corrupt

    var isActive: Bool {
        switch self {
        case .resolving, .preflighting, .downloading, .verifying, .publishing:
            true
        case .completed, .cancelled, .interrupted, .rejected, .corrupt:
            false
        }
    }
}

public struct ModelDownloadJournal: Codable, Equatable, Sendable {
    public var version: Int
    public var provider: ModelRepositoryProvider
    public var repoID: String
    public var requestedRevision: String
    public var commitSHA: String
    public var phase: ModelDownloadPhase
    public var progressBytes: Int64
    public var totalBytes: Int64
    public var currentFile: String?
    public var error: String?
    public var errorCode: String?
    public var updatedAt: Date

    public init(
        provider: ModelRepositoryProvider,
        repoID: String,
        requestedRevision: String,
        commitSHA: String,
        phase: ModelDownloadPhase,
        progressBytes: Int64 = 0,
        totalBytes: Int64 = 0,
        currentFile: String? = nil,
        error: String? = nil,
        errorCode: String? = nil,
        updatedAt: Date = Date()
    ) {
        version = 1
        self.provider = provider
        self.repoID = repoID
        self.requestedRevision = requestedRevision
        self.commitSHA = commitSHA
        self.phase = phase
        self.progressBytes = progressBytes
        self.totalBytes = totalBytes
        self.currentFile = currentFile
        self.error = error
        self.errorCode = errorCode
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case version
        case provider
        case repoID = "repo_id"
        case requestedRevision = "requested_revision"
        case commitSHA = "commit_sha"
        case phase
        case progressBytes = "progress_bytes"
        case totalBytes = "total_bytes"
        case currentFile = "current_file"
        case error
        case errorCode = "error_code"
        case updatedAt = "updated_at"
    }
}

public struct ModelPartialIdentity: Codable, Equatable, Sendable {
    public var version: Int
    public var provider: ModelRepositoryProvider
    public var repoID: String
    public var commitSHA: String
    public var path: String
    public var expectedSize: Int64
    public var expectedSHA256: String
    public var etag: String?

    public init(
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHA: String,
        path: String,
        expectedSize: Int64,
        expectedSHA256: String,
        etag: String?
    ) {
        version = 1
        self.provider = provider
        self.repoID = repoID
        self.commitSHA = commitSHA
        self.path = path
        self.expectedSize = expectedSize
        self.expectedSHA256 = expectedSHA256.lowercased()
        self.etag = etag
    }

    enum CodingKeys: String, CodingKey {
        case version
        case provider
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case path
        case expectedSize = "expected_size"
        case expectedSHA256 = "expected_sha256"
        case etag
    }
}

public enum ModelRepositoryLockError: LocalizedError {
    case busy
    case openFailed(Int32)
    case lockFailed(Int32)

    public var errorDescription: String? {
        switch self {
        case .busy:
            "Another process is already modifying this model repository."
        case let .openFailed(code):
            "Unable to open model repository lock: errno \(code)."
        case let .lockFailed(code):
            "Unable to acquire model repository lock: errno \(code)."
        }
    }
}

public final class ModelRepositoryLock: @unchecked Sendable {
    private let descriptor: Int32

    public init(lockURL: URL) throws {
        try FileManager.default.createDirectory(
            at: lockURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let descriptor = Darwin.open(lockURL.path, O_CREAT | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw ModelRepositoryLockError.openFailed(errno)
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let code = errno
            Darwin.close(descriptor)
            if code == EWOULDBLOCK {
                throw ModelRepositoryLockError.busy
            }
            throw ModelRepositoryLockError.lockFailed(code)
        }
        self.descriptor = descriptor
    }

    deinit {
        flock(descriptor, LOCK_UN)
        Darwin.close(descriptor)
    }
}

public enum ModelSnapshotUseLockError: LocalizedError {
    case busy
    case openFailed(Int32)
    case lockFailed(Int32)

    public var errorDescription: String? {
        switch self {
        case .busy:
            "The model snapshot is currently in use."
        case let .openFailed(code):
            "Unable to open model snapshot use lock: errno \(code)."
        case let .lockFailed(code):
            "Unable to acquire model snapshot use lock: errno \(code)."
        }
    }
}

public final class ModelSnapshotUseLock: @unchecked Sendable {
    private let descriptor: Int32

    public init(lockURL: URL, exclusive: Bool) throws {
        try FileManager.default.createDirectory(
            at: lockURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let descriptor = Darwin.open(lockURL.path, O_CREAT | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw ModelSnapshotUseLockError.openFailed(errno)
        }
        let operation = (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB
        guard flock(descriptor, operation) == 0 else {
            let code = errno
            Darwin.close(descriptor)
            if code == EWOULDBLOCK {
                throw ModelSnapshotUseLockError.busy
            }
            throw ModelSnapshotUseLockError.lockFailed(code)
        }
        self.descriptor = descriptor
    }

    deinit {
        flock(descriptor, LOCK_UN)
        Darwin.close(descriptor)
    }
}

public struct ModelDownloadStore: Sendable {
    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public func repositoryRoot(provider: ModelRepositoryProvider, repoID: String) throws -> URL {
        try ModelRepositoryLayout.repositoryRoot(rootURL: rootURL, provider: provider, repoID: repoID)
    }

    public func snapshotURL(provider: ModelRepositoryProvider, repoID: String, commitSHA: String) throws -> URL {
        try repositoryRoot(provider: provider, repoID: repoID)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(commitSHA, isDirectory: true)
    }

    public func downloadRoot(provider: ModelRepositoryProvider, repoID: String, commitSHA: String) throws -> URL {
        try repositoryRoot(provider: provider, repoID: repoID)
            .appendingPathComponent(".downloads", isDirectory: true)
            .appendingPathComponent(commitSHA, isDirectory: true)
    }

    public func stagingSnapshotURL(provider: ModelRepositoryProvider, repoID: String, commitSHA: String) throws -> URL {
        try downloadRoot(provider: provider, repoID: repoID, commitSHA: commitSHA)
            .appendingPathComponent("snapshot", isDirectory: true)
    }

    public func journalURL(provider: ModelRepositoryProvider, repoID: String, commitSHA: String) throws -> URL {
        try downloadRoot(provider: provider, repoID: repoID, commitSHA: commitSHA)
            .appendingPathComponent("state.json")
    }

    public func acquireRepositoryLock(provider: ModelRepositoryProvider, repoID: String) throws -> ModelRepositoryLock {
        let url = try repositoryRoot(provider: provider, repoID: repoID)
            .appendingPathComponent(".locks", isDirectory: true)
            .appendingPathComponent("repository.lock")
        return try ModelRepositoryLock(lockURL: url)
    }

    public func snapshotUseLockURL(
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHA: String
    ) throws -> URL {
        guard ModelSnapshotVerifier.isCommitSHA(commitSHA) else {
            throw ModelSnapshotVerificationError.identityMismatch(
                "snapshot use lock requires a full commit SHA"
            )
        }
        return try repositoryRoot(provider: provider, repoID: repoID)
            .appendingPathComponent(".locks", isDirectory: true)
            .appendingPathComponent("\(commitSHA.lowercased()).use.lock")
    }

    public func acquireSnapshotUseLock(
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHA: String,
        exclusive: Bool
    ) throws -> ModelSnapshotUseLock {
        try ModelSnapshotUseLock(
            lockURL: snapshotUseLockURL(
                provider: provider,
                repoID: repoID,
                commitSHA: commitSHA
            ),
            exclusive: exclusive
        )
    }

    public func prepareStaging(provider: ModelRepositoryProvider, repoID: String, commitSHA: String) throws -> URL {
        let staging = try stagingSnapshotURL(provider: provider, repoID: repoID, commitSHA: commitSHA)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        return staging
    }

    public func writeJournal(_ journal: ModelDownloadJournal) throws {
        let url = try journalURL(
            provider: journal.provider,
            repoID: journal.repoID,
            commitSHA: journal.commitSHA
        )
        try Self.atomicWrite(journal, to: url)
    }

    public func writeManifest(_ manifest: ModelSnapshotManifest, to snapshot: URL) throws {
        try Self.atomicWrite(
            manifest,
            to: snapshot.appendingPathComponent(ModelSnapshotManifest.filename)
        )
    }

    public func publish(_ manifest: ModelSnapshotManifest) throws -> URL {
        let staging = try stagingSnapshotURL(
            provider: manifest.provider,
            repoID: manifest.repoID,
            commitSHA: manifest.commitSHA
        )
        let destination = try snapshotURL(
            provider: manifest.provider,
            repoID: manifest.repoID,
            commitSHA: manifest.commitSHA
        )
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            let existing: ModelSnapshotManifest
            do {
                existing = try ModelSnapshotVerifier().verifyStructure(
                    snapshot: destination,
                    expectedProvider: manifest.provider,
                    expectedRepoID: manifest.repoID
                )
                let record = try ModelSnapshotVerifier().loadIntegrityRecord(at: destination)
                guard record.provider == manifest.provider,
                      record.repoID == manifest.repoID,
                      record.commitSHA == manifest.commitSHA,
                      record.state == .verified
                else {
                    throw ModelSnapshotVerificationError.identityMismatch(
                        "published commit does not have a matching verified integrity record"
                    )
                }
            } catch {
                return try replaceInvalidSnapshot(
                    at: destination,
                    with: staging,
                    manifest: manifest
                )
            }
            guard existing.version == manifest.version,
                  existing.provider == manifest.provider,
                  existing.repoID == manifest.repoID,
                  existing.requestedRevision == manifest.requestedRevision,
                  existing.commitSHA == manifest.commitSHA,
                  existing.files == manifest.files,
                  existing.compatibility == manifest.compatibility,
                  existing.resources == manifest.resources
            else {
                throw ModelSnapshotVerificationError.identityMismatch(
                    "published commit exists with a different manifest"
                )
            }
            try updateRef(for: manifest)
            return destination
        }
        guard rename(staging.path, destination.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        try Self.syncDirectory(destination.deletingLastPathComponent())
        try updateRef(for: manifest)
        return destination
    }

    private func replaceInvalidSnapshot(
        at destination: URL,
        with staging: URL,
        manifest: ModelSnapshotManifest
    ) throws -> URL {
        let useLock = try acquireSnapshotUseLock(
            provider: manifest.provider,
            repoID: manifest.repoID,
            commitSHA: manifest.commitSHA,
            exclusive: true
        )
        defer { withExtendedLifetime(useLock) {} }
        guard renameatx_np(
            AT_FDCWD,
            staging.path,
            AT_FDCWD,
            destination.path,
            UInt32(RENAME_SWAP)
        ) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        do {
            try Self.syncDirectory(destination.deletingLastPathComponent())
            try updateRef(for: manifest)
        } catch {
            _ = renameatx_np(
                AT_FDCWD,
                staging.path,
                AT_FDCWD,
                destination.path,
                UInt32(RENAME_SWAP)
            )
            try? Self.syncDirectory(destination.deletingLastPathComponent())
            throw error
        }

        let repository = try repositoryRoot(
            provider: manifest.provider,
            repoID: manifest.repoID
        )
        let trashRoot = repository.appendingPathComponent(".trash", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: trashRoot,
            withIntermediateDirectories: true
        )
        let trash = trashRoot.appendingPathComponent(
            "\(manifest.commitSHA)-repair-\(UUID().uuidString)",
            isDirectory: true
        )
        if rename(staging.path, trash.path) == 0 {
            try? Self.syncDirectory(staging.deletingLastPathComponent())
            try? Self.syncDirectory(trashRoot)
        }
        return destination
    }

    public func updateRef(for manifest: ModelSnapshotManifest) throws {
        let ref = try repositoryRoot(provider: manifest.provider, repoID: manifest.repoID)
            .appendingPathComponent("refs", isDirectory: true)
            .appendingPathComponent(manifest.requestedRevision)
        try Self.atomicWrite(Data((manifest.commitSHA + "\n").utf8), to: ref)
    }

    public func recoverInterruptedJournals() -> [ModelDownloadJournal] {
        var recovered: [ModelDownloadJournal] = []
        for provider in ModelRepositoryProvider.allCases {
            let root = ModelRepositoryLayout.providerRoot(rootURL: rootURL, provider: provider)
            guard let repositories = try? FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }
            for repository in repositories {
                let downloads = repository.appendingPathComponent(".downloads", isDirectory: true)
                guard let commits = try? FileManager.default.contentsOfDirectory(
                    at: downloads,
                    includingPropertiesForKeys: nil
                ) else {
                    continue
                }
                for commit in commits {
                    let url = commit.appendingPathComponent("state.json")
                    guard let data = try? Data(contentsOf: url),
                          var journal = try? JSONDecoder().decode(ModelDownloadJournal.self, from: data)
                    else {
                        continue
                    }
                    if journal.phase.isActive {
                        journal.phase = .interrupted
                        journal.errorCode = "process_interrupted"
                        journal.error = "The previous process exited before the download completed."
                        journal.updatedAt = Date()
                        try? writeJournal(journal)
                    }
                    recovered.append(journal)
                }
            }
        }
        return recovered
    }

    public static func atomicWrite<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try atomicWrite(try encoder.encode(value), to: url)
    }

    public static func atomicWrite(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        do {
            let handle = try FileHandle(forWritingTo: temporary)
            try handle.write(contentsOf: data)
            try handle.synchronize()
            try handle.close()
            guard rename(temporary.path, url.path) == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            try syncDirectory(url.deletingLastPathComponent())
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    static func syncDirectory(_ url: URL) throws {
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
