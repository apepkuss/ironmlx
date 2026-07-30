import Foundation
import Testing

@testable import IronMLXAppCore

private let versionCommitA = String(repeating: "a", count: 40)
private let versionCommitB = String(repeating: "b", count: 40)
private let versionCommitC = String(repeating: "c", count: 40)

@Test func modelVersionListDistinguishesActiveLoadedAndReclaimableSnapshots() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let first = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )

    let list = try ModelVersionManagementService(rootURL: root).versions(
        provider: .huggingFace,
        repoID: repoID,
        loadedModelPaths: [first.path]
    )

    #expect(list.activeCommitSHA == versionCommitB)
    #expect(list.versions.map(\.commitSHA) == [versionCommitB, versionCommitA])
    let active = try #require(list.versions.first { $0.isActive })
    #expect(active.integrityState == .verified)
    #expect(!active.canDelete)
    let loaded = try #require(list.versions.first { $0.isLoaded })
    #expect(loaded.commitSHA == versionCommitA)
    #expect(!loaded.canDelete)
    #expect(list.reclaimableBytes == 0)
}

@Test func modelVersionActivationAtomicallySwitchesTheActiveRef() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let first = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )

    let result = try ModelVersionManagementService(rootURL: root).activate(
        provider: .huggingFace,
        repoID: repoID,
        commitSHA: versionCommitA,
        fullChecksum: false
    )

    #expect(result.previousCommitSHA == versionCommitB)
    #expect(result.activeCommitSHA == versionCommitA)
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: repoID) == first.path)
}

@Test func modelVersionActivationRejectsKnownCorruptSnapshot() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let first = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )
    try ModelDownloadStore.atomicWrite(
        ModelSnapshotIntegrityRecord(
            provider: .huggingFace,
            repoID: repoID,
            commitSHA: versionCommitA,
            state: .corrupt,
            error: "damaged"
        ),
        to: first.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
    )

    #expect(throws: ModelSnapshotVerificationError.self) {
        try ModelVersionManagementService(rootURL: root).activate(
            provider: .huggingFace,
            repoID: repoID,
            commitSHA: versionCommitA,
            fullChecksum: false
        )
    }
}

@Test func modelVersionActivationFullVerificationMarksDamagedSnapshotCorrupt() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let first = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )
    try Data("other".utf8).write(
        to: first.appendingPathComponent("model.safetensors")
    )

    #expect(throws: ModelSnapshotVerificationError.self) {
        try ModelVersionManagementService(rootURL: root).activate(
            provider: .huggingFace,
            repoID: repoID,
            commitSHA: versionCommitA,
            fullChecksum: true
        )
    }

    let record = try ModelSnapshotVerifier().loadIntegrityRecord(at: first)
    #expect(record.state == .corrupt)
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: repoID)?.hasSuffix(versionCommitB) == true)
}

@Test func modelVersionDeletionHonorsSnapshotUseLeaseThenRemovesInactiveCommit() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let first = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )
    let store = ModelDownloadStore(rootURL: root)
    let service = ModelVersionManagementService(rootURL: root)

    do {
        let useLease = try store.acquireSnapshotUseLock(
            provider: .huggingFace,
            repoID: repoID,
            commitSHA: versionCommitA,
            exclusive: false
        )
        #expect(throws: ModelSnapshotUseLockError.self) {
            try service.deleteVersions(
                provider: .huggingFace,
                repoID: repoID,
                commitSHAs: [versionCommitA],
                loadedModelPaths: []
            )
        }
        withExtendedLifetime(useLease) {}
    }

    let result = try service.deleteVersions(
        provider: .huggingFace,
        repoID: repoID,
        commitSHAs: [versionCommitA],
        loadedModelPaths: []
    )

    #expect(result.deletedCommitSHAs == [versionCommitA])
    #expect(result.reclaimedBytes > 0)
    #expect(!FileManager.default.fileExists(atPath: first.path))
    #expect(LocalModelScanner(rootURL: root).resolveModelPath(for: repoID) != nil)
}

@Test func modelVersionDeletionPreflightsEverySelectedSnapshotBeforeRenamingAny() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    let first = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    let second = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitC,
        weights: "current"
    )
    let store = ModelDownloadStore(rootURL: root)
    let useLease = try store.acquireSnapshotUseLock(
        provider: .huggingFace,
        repoID: repoID,
        commitSHA: versionCommitB,
        exclusive: false
    )

    #expect(throws: ModelSnapshotUseLockError.self) {
        try ModelVersionManagementService(rootURL: root).deleteVersions(
            provider: .huggingFace,
            repoID: repoID,
            commitSHAs: [versionCommitA, versionCommitB],
            loadedModelPaths: []
        )
    }

    #expect(FileManager.default.fileExists(atPath: first.path))
    #expect(FileManager.default.fileExists(atPath: second.path))
    withExtendedLifetime(useLease) {}
}

@Test func modelSearchLocalStateSeparatesExactInactiveAndNewRemoteCommits() throws {
    let root = try temporaryDirectory()
    let repoID = "org/model"
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "first"
    )
    _ = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitB,
        weights: "second"
    )
    let service = ModelVersionManagementService(rootURL: root)

    let active = service.searchLocalState(
        provider: .huggingFace,
        repoID: repoID,
        remoteCommitSHA: versionCommitB
    )
    let inactive = service.searchLocalState(
        provider: .huggingFace,
        repoID: repoID,
        remoteCommitSHA: versionCommitA
    )
    let newer = service.searchLocalState(
        provider: .huggingFace,
        repoID: repoID,
        remoteCommitSHA: String(repeating: "c", count: 40)
    )
    let unknown = service.searchLocalState(
        provider: .huggingFace,
        repoID: repoID,
        remoteCommitSHA: nil
    )

    #expect(active.state == .exists)
    #expect(inactive.state == .localInactive)
    #expect(newer.state == .updateAvailable)
    #expect(unknown.state == .identityUnavailable)
}

@Test func publishingAReplacementAtomicallySwapsOutAKnownCorruptSnapshot() throws {
    let root = try temporaryDirectory()
    let sourceRoot = try temporaryDirectory()
    let repoID = "org/model"
    let corrupt = try writeVersionSnapshot(
        root: root,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "corrupt"
    )
    try ModelDownloadStore.atomicWrite(
        ModelSnapshotIntegrityRecord(
            provider: .huggingFace,
            repoID: repoID,
            commitSHA: versionCommitA,
            state: .corrupt,
            error: "damaged"
        ),
        to: corrupt.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
    )
    let replacement = try writeVersionSnapshot(
        root: sourceRoot,
        repoID: repoID,
        commitSHA: versionCommitA,
        weights: "replacement"
    )
    let manifest = try ModelSnapshotVerifier().verifyStructure(
        snapshot: replacement,
        expectedProvider: .huggingFace,
        expectedRepoID: repoID
    )
    let store = ModelDownloadStore(rootURL: root)
    let staging = try store.prepareStaging(
        provider: .huggingFace,
        repoID: repoID,
        commitSHA: versionCommitA
    )
    for item in try FileManager.default.contentsOfDirectory(
        at: replacement,
        includingPropertiesForKeys: nil
    ) {
        try FileManager.default.copyItem(
            at: item,
            to: staging.appendingPathComponent(item.lastPathComponent)
        )
    }

    let published = try store.publish(manifest)

    #expect(
        try String(
            contentsOf: published.appendingPathComponent("model.safetensors"),
            encoding: .utf8
        ) == "replacement"
    )
    let trash = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: .huggingFace,
        repoID: repoID
    ).appendingPathComponent(".trash", isDirectory: true)
    #expect(
        try FileManager.default.contentsOfDirectory(atPath: trash.path).count == 1
    )
}

private func writeVersionSnapshot(
    root: URL,
    repoID: String,
    commitSHA: String,
    weights: String
) throws -> URL {
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: repoID,
        files: [
            "config.json": Data(#"{"model_type":"llama"}"#.utf8),
            "model.safetensors": Data(weights.utf8),
        ],
        commitSHA: commitSHA
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
    return snapshot
}
