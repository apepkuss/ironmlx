import Foundation

public struct ModelIntegrityStatus: Codable, Equatable, Sendable {
    public var repoID: String
    public var state: String
    public var progressPct: Double
    public var currentFile: String?
    public var verifiedAt: String?
    public var error: String?

    public init(
        repoID: String,
        state: String,
        progressPct: Double = 0,
        currentFile: String? = nil,
        verifiedAt: Date? = nil,
        error: String? = nil
    ) {
        self.repoID = repoID
        self.state = state
        self.progressPct = progressPct
        self.currentFile = currentFile
        self.verifiedAt = verifiedAt.map { ISO8601DateFormatter().string(from: $0) }
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case state
        case progressPct = "progress_pct"
        case currentFile = "current_file"
        case verifiedAt = "verified_at"
        case error
    }
}

public enum ModelIntegrityVerificationError: LocalizedError {
    case busy(String)
    case snapshotUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case let .busy(repoID):
            "Integrity verification is already running for \(repoID)."
        case let .snapshotUnavailable(repoID):
            "No active verified snapshot is available for \(repoID)."
        }
    }
}

public actor ModelIntegrityVerificationService {
    private let rootURL: URL
    private var activeModels: Set<String> = []
    private var statuses: [String: ModelIntegrityStatus] = [:]

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
    ) {
        self.rootURL = rootURL
    }

    public func status(for repoID: String) -> ModelIntegrityStatus? {
        statuses[repoID]
    }

    public func verify(
        repoID: String,
        progress: @escaping @Sendable (ModelIntegrityStatus) -> Void = { _ in }
    ) throws -> ModelIntegrityStatus {
        guard !activeModels.contains(repoID) else {
            throw ModelIntegrityVerificationError.busy(repoID)
        }
        let location = try activeSnapshot(repoID: repoID)
        let lock = try ModelDownloadStore(rootURL: rootURL).acquireRepositoryLock(
            provider: location.provider,
            repoID: repoID
        )
        defer { withExtendedLifetime(lock) {} }
        activeModels.insert(repoID)
        defer { activeModels.remove(repoID) }

        var current = ModelIntegrityStatus(repoID: repoID, state: "verifying")
        statuses[repoID] = current
        progress(current)

        do {
            let manifest = try ModelSnapshotVerifier().verify(
                snapshot: location.snapshot,
                expectedProvider: location.provider,
                expectedRepoID: repoID
            ) { path, completedBytes, totalBytes in
                progress(
                    ModelIntegrityStatus(
                        repoID: repoID,
                        state: "verifying",
                        progressPct: totalBytes > 0
                            ? min(100, Double(completedBytes) / Double(totalBytes) * 100)
                            : 100,
                        currentFile: path
                    )
                )
            }
            let verifiedAt = Date()
            try ModelDownloadStore.atomicWrite(
                ModelSnapshotIntegrityRecord(
                    provider: manifest.provider,
                    repoID: manifest.repoID,
                    commitSHA: manifest.commitSHA,
                    state: .verified,
                    verifiedAt: verifiedAt
                ),
                to: location.snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
            )
            current = ModelIntegrityStatus(
                repoID: repoID,
                state: "verified",
                progressPct: 100,
                verifiedAt: verifiedAt
            )
            statuses[repoID] = current
            progress(current)
            return current
        } catch {
            try? ModelDownloadStore.atomicWrite(
                ModelSnapshotIntegrityRecord(
                    provider: location.provider,
                    repoID: repoID,
                    commitSHA: location.commitSHA,
                    state: .corrupt,
                    error: error.localizedDescription
                ),
                to: location.snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
            )
            current = ModelIntegrityStatus(
                repoID: repoID,
                state: "corrupt",
                error: error.localizedDescription
            )
            statuses[repoID] = current
            progress(current)
            return current
        }
    }

    private func activeSnapshot(
        repoID: String
    ) throws -> (provider: ModelRepositoryProvider, commitSHA: String, snapshot: URL) {
        for provider in ModelRepositoryProvider.allCases {
            guard let repository = try? ModelRepositoryLayout.repositoryRoot(
                rootURL: rootURL,
                provider: provider,
                repoID: repoID
            ) else {
                continue
            }
            let ref = repository
                .appendingPathComponent("refs", isDirectory: true)
                .appendingPathComponent(provider.mutableRevision)
            guard let data = try? Data(contentsOf: ref),
                  let raw = String(data: data, encoding: .utf8)
            else {
                continue
            }
            let commitSHA = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard ModelSnapshotVerifier.isCommitSHA(commitSHA) else {
                continue
            }
            let snapshot = repository
                .appendingPathComponent("snapshots", isDirectory: true)
                .appendingPathComponent(commitSHA, isDirectory: true)
            guard FileManager.default.fileExists(atPath: snapshot.path) else {
                continue
            }
            _ = try ModelSnapshotVerifier().verifyStructure(
                snapshot: snapshot,
                expectedProvider: provider,
                expectedRepoID: repoID,
                allowKnownCorrupt: true
            )
            return (provider, commitSHA, snapshot)
        }
        throw ModelIntegrityVerificationError.snapshotUnavailable(repoID)
    }
}
