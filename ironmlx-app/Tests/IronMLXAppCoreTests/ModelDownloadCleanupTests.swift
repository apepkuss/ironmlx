import Foundation
import Testing

@testable import IronMLXAppCore

private let cleanupCommit = String(repeating: "a", count: 40)
private let completedCommit = String(repeating: "b", count: 40)

@Test func downloadCleanupSupportsASymlinkConfiguredAsTheStorageRoot() throws {
    let parent = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: parent) }
    let target = parent.appendingPathComponent("real-storage")
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    let link = parent.appendingPathComponent("configured-root")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
    let store = ModelDownloadStore(rootURL: link)
    let staging = try store.prepareStaging(provider: .huggingFace, repoID: "org/cleanup", commitSHA: cleanupCommit)
    try store.clearIncompleteDownloads(provider: .huggingFace, repoID: "org/cleanup")
    #expect(!FileManager.default.fileExists(atPath: staging.path))
    #expect(FileManager.default.fileExists(atPath: target.path))
}

@Test(arguments: ModelRepositoryProvider.allCases)
func downloadCleanupOnlyRemovesIncompleteDataForTheSelectedRepository(provider: ModelRepositoryProvider) throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelDownloadStore(rootURL: root)
    let repoID = "org/cleanup"
    for source in ModelRepositoryProvider.allCases {
        for (commit, phase) in [(cleanupCommit, ModelDownloadPhase.interrupted), (completedCommit, .completed)] {
            let staging = try store.prepareStaging(provider: source, repoID: repoID, commitSHA: commit)
            try Data("staging".utf8).write(to: staging.appendingPathComponent("weight.partial"))
            try store.writeJournal(.init(
                provider: source, repoID: repoID, requestedRevision: "main", commitSHA: commit, phase: phase
            ))
        }
    }
    let snapshot = try store.snapshotURL(provider: provider, repoID: repoID, commitSHA: completedCommit)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    let weight = snapshot.appendingPathComponent("model.safetensors")
    try Data("installed".utf8).write(to: weight)
    let staging = try store.stagingSnapshotURL(provider: provider, repoID: repoID, commitSHA: cleanupCommit)
    try FileManager.default.createSymbolicLink(at: staging.appendingPathComponent("linked-model"), withDestinationURL: snapshot)
    // An interrupted attempt can have created its staging directory before the
    // first journal write. It belongs to the same repository and is also removed.
    let unjournaled = try store.prepareStaging(provider: provider, repoID: repoID, commitSHA: String(repeating: "c", count: 40))
    let unrelated = try store.prepareStaging(provider: provider, repoID: "org/other", commitSHA: cleanupCommit)

    try store.clearIncompleteDownloads(provider: provider, repoID: repoID)

    #expect(!FileManager.default.fileExists(atPath: staging.deletingLastPathComponent().path))
    #expect(!FileManager.default.fileExists(atPath: unjournaled.deletingLastPathComponent().path))
    #expect(FileManager.default.fileExists(atPath: unrelated.path))
    #expect(try Data(contentsOf: weight) == Data("installed".utf8))
    let completed = try store.downloadRoot(provider: provider, repoID: repoID, commitSHA: completedCommit)
    #expect(FileManager.default.fileExists(atPath: completed.appendingPathComponent("state.json").path))
    for source in ModelRepositoryProvider.allCases where source != provider {
        let untouched = try store.stagingSnapshotURL(provider: source, repoID: repoID, commitSHA: cleanupCommit)
        #expect(FileManager.default.fileExists(atPath: untouched.path))
    }
    try store.clearIncompleteDownloads(provider: provider, repoID: repoID)
}

@Test(arguments: ["models", "models/huggingface/org--cleanup", "models/huggingface/org--cleanup/.downloads", "models/huggingface/org--cleanup/.downloads/" + cleanupCommit])
func downloadCleanupRejectsRedirectedDirectories(relativePath: String) throws {
    let root = try temporaryDirectory()
    let outside = try temporaryDirectory()
    defer {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: outside)
    }
    let store = ModelDownloadStore(rootURL: root)
    _ = try store.prepareStaging(provider: .huggingFace, repoID: "org/cleanup", commitSHA: cleanupCommit)
    let sentinel = outside.appendingPathComponent("keep")
    try Data("keep".utf8).write(to: sentinel)
    let link = root.appendingPathComponent(relativePath)
    try FileManager.default.removeItem(at: link)
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)

    #expect(throws: ModelSnapshotVerificationError.self) {
        try store.clearIncompleteDownloads(provider: .huggingFace, repoID: "org/cleanup")
    }
    #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
}

@Test func downloadCleanupValidatesAllJournalsBeforeDeleting() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelDownloadStore(rootURL: root)
    let repoID = "org/cleanup"
    let valid = try store.prepareStaging(provider: .huggingFace, repoID: repoID, commitSHA: cleanupCommit)
    let invalid = try store.prepareStaging(provider: .huggingFace, repoID: repoID, commitSHA: completedCommit)
    let journal = ModelDownloadJournal(
        provider: .modelScope, repoID: "org/different", requestedRevision: "master",
        commitSHA: completedCommit, phase: .cancelled
    )
    try ModelDownloadStore.atomicWrite(journal, to: invalid.deletingLastPathComponent().appendingPathComponent("state.json"))
    #expect(throws: ModelSnapshotVerificationError.self) {
        try store.clearIncompleteDownloads(provider: .huggingFace, repoID: repoID)
    }
    #expect(FileManager.default.fileExists(atPath: valid.path))
    #expect(FileManager.default.fileExists(atPath: invalid.path))
}
