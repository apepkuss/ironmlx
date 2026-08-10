import Foundation
import Testing

@testable import IronMLXAppCore

@Test(arguments: [
    ConfigurationFileCoordinator.TransactionCheckpoint.journalPersisted,
    .lkgCommitted,
    .activeCommitted,
])
func configurationTransactionFailureRestoresActiveAndLKG(
    checkpoint: ConfigurationFileCoordinator.TransactionCheckpoint
) throws {
    let root = try configurationTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let active = root.appendingPathComponent("app_config.json")
    let oldActive = Data("old-active".utf8)
    let oldLKG = Data("old-lkg".utf8)
    try oldActive.write(to: active)
    let layout = ConfigurationFileLayout(activeURL: active)
    try FileManager.default.createDirectory(
        at: layout.recoveryDirectoryURL,
        withIntermediateDirectories: true
    )
    try oldLKG.write(to: layout.lkgURL)
    let coordinator = ConfigurationFileCoordinator(
        activeURL: active,
        fileManager: .default,
        checkpoint: { current in
            if current == checkpoint {
                throw CocoaError(.fileWriteUnknown)
            }
        }
    )

    #expect(throws: (any Error).self) {
        try coordinator.commitActiveAndLKG(Data("candidate".utf8))
    }

    #expect(try Data(contentsOf: active) == oldActive)
    #expect(try Data(contentsOf: layout.lkgURL) == oldLKG)
    #expect(!FileManager.default.fileExists(atPath: layout.transactionJournalURL.path))
}

@Test func interruptedConfigurationTransactionRollsBackFromJournal() throws {
    let root = try configurationTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let active = root.appendingPathComponent("model_params.json")
    let coordinator = ConfigurationFileCoordinator(activeURL: active, fileManager: .default)
    let layout = coordinator.layout
    try FileManager.default.createDirectory(
        at: layout.recoveryDirectoryURL,
        withIntermediateDirectories: true
    )
    let oldActive = Data("old-active".utf8)
    let oldLKG = Data("old-lkg".utf8)
    let candidate = Data("candidate".utf8)
    try candidate.write(to: active)
    try candidate.write(to: layout.lkgURL)
    let activeRollbackName = ".active.rollback-test"
    let lkgRollbackName = ".lkg.rollback-test"
    try oldActive.write(
        to: layout.recoveryDirectoryURL.appendingPathComponent(activeRollbackName)
    )
    try oldLKG.write(
        to: layout.recoveryDirectoryURL.appendingPathComponent(lkgRollbackName)
    )
    let journal: [String: Any] = [
        "active_existed": true,
        "lkg_existed": true,
        "active_rollback_name": activeRollbackName,
        "lkg_rollback_name": lkgRollbackName,
        "candidate_sha256": "not-the-installed-candidate",
    ]
    try JSONSerialization.data(withJSONObject: journal)
        .write(to: layout.transactionJournalURL)

    try coordinator.recoverInterruptedTransactionIfNeeded()

    #expect(try Data(contentsOf: active) == oldActive)
    #expect(try Data(contentsOf: layout.lkgURL) == oldLKG)
    #expect(!FileManager.default.fileExists(atPath: layout.transactionJournalURL.path))
}

@Test func configurationEvidenceIsReadOnlyAndLimitedToOneV0Source() throws {
    let root = try configurationTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let coordinator = ConfigurationFileCoordinator(
        activeURL: root.appendingPathComponent("app_config.json"),
        fileManager: .default
    )
    let original = Data("first-v0".utf8)

    let evidence = try coordinator.preservePreMigration(original, schemaVersion: 0)
    let duplicate = try coordinator.preservePreMigration(original, schemaVersion: 0)

    #expect(evidence == duplicate)
    #expect(try Data(contentsOf: evidence) == original)
    #expect(try posixPermissions(evidence) == 0o400)
    #expect(try posixPermissions(coordinator.layout.recoveryDirectoryURL) == 0o700)
    #expect(throws: ConfigurationPersistenceError.self) {
        _ = try coordinator.preservePreMigration(Data("different-v0".utf8), schemaVersion: 0)
    }
}

@Test func configurationAtomicWriteRestrictsPermissionsAndNeverLeavesTemporaryFile() throws {
    let root = try configurationTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let active = root.appendingPathComponent("app_config.json")
    let coordinator = ConfigurationFileCoordinator(activeURL: active, fileManager: .default)

    try coordinator.commitActiveAndLKG(Data("candidate".utf8))

    #expect(try posixPermissions(active) == 0o600)
    #expect(try posixPermissions(coordinator.layout.lkgURL) == 0o600)
    let files = try FileManager.default.contentsOfDirectory(
        at: coordinator.layout.recoveryDirectoryURL,
        includingPropertiesForKeys: nil
    )
    #expect(files.map(\.lastPathComponent) == [coordinator.layout.lkgURL.lastPathComponent])
}

private func configurationTemporaryDirectory() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-configuration-persistence-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func posixPermissions(_ url: URL) throws -> Int {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    return try #require(attributes[.posixPermissions] as? Int)
}
