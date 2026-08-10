import CryptoKit
import Darwin
import Foundation

enum ConfigurationPersistenceError: LocalizedError, Equatable {
    case invalidTopLevel
    case invalidSchemaVersion
    case unsupportedSchemaVersion(found: Int, supported: Int)
    case unexpectedKeys([String])
    case invalidValue(String)
    case transactionRecoveryFailed(URL)
    case preMigrationEvidenceConflict(URL)
    case lkgUnavailable(URL)

    var errorDescription: String? {
        switch self {
        case .invalidTopLevel:
            "Configuration must be a JSON object."
        case .invalidSchemaVersion:
            "Configuration schema_version must be an integer."
        case let .unsupportedSchemaVersion(found, supported):
            "Configuration schema version \(found) is newer than supported version \(supported)."
        case let .unexpectedKeys(keys):
            "Configuration contains unsupported keys: \(keys.sorted().joined(separator: ", "))."
        case let .invalidValue(field):
            "Configuration field is invalid: \(field)."
        case let .transactionRecoveryFailed(url):
            "Configuration transaction recovery failed: \(url.path)."
        case let .preMigrationEvidenceConflict(url):
            "A different pre-migration configuration is already preserved: \(url.path)."
        case let .lkgUnavailable(url):
            "Last-known-good configuration is unavailable or invalid: \(url.path)."
        }
    }
}

struct ConfigurationFileLayout: Sendable {
    let activeURL: URL
    let recoveryDirectoryURL: URL
    let lkgURL: URL
    let transactionJournalURL: URL

    init(activeURL: URL) {
        self.activeURL = activeURL
        let recoveryDirectory = activeURL.deletingLastPathComponent()
            .appendingPathComponent("recovery", isDirectory: true)
        recoveryDirectoryURL = recoveryDirectory
        let baseName = activeURL.deletingPathExtension().lastPathComponent
        let suffix = activeURL.pathExtension.isEmpty ? "" : ".\(activeURL.pathExtension)"
        lkgURL = recoveryDirectory.appendingPathComponent("\(baseName).lkg\(suffix)")
        transactionJournalURL = recoveryDirectory.appendingPathComponent(".\(baseName).transaction.json")
    }

    func preMigrationURL(schemaVersion: Int, digest: String) -> URL {
        let baseName = activeURL.deletingPathExtension().lastPathComponent
        let suffix = activeURL.pathExtension.isEmpty ? "" : ".\(activeURL.pathExtension)"
        return recoveryDirectoryURL.appendingPathComponent(
            "\(baseName).pre-migration-v\(schemaVersion)-\(digest)\(suffix)"
        )
    }
}

private struct ConfigurationTransactionJournal: Codable {
    let activeExisted: Bool
    let lkgExisted: Bool
    let activeRollbackName: String
    let lkgRollbackName: String
    let candidateSHA256: String

    enum CodingKeys: String, CodingKey {
        case activeExisted = "active_existed"
        case lkgExisted = "lkg_existed"
        case activeRollbackName = "active_rollback_name"
        case lkgRollbackName = "lkg_rollback_name"
        case candidateSHA256 = "candidate_sha256"
    }
}

final class ConfigurationFileCoordinator: @unchecked Sendable {
    enum TransactionCheckpoint: Equatable, Sendable {
        case journalPersisted
        case lkgCommitted
        case activeCommitted
    }

    let layout: ConfigurationFileLayout
    private let fileManager: FileManager
    private let lock = NSRecursiveLock()
    private let checkpoint: (TransactionCheckpoint) throws -> Void

    init(
        activeURL: URL,
        fileManager: FileManager,
        checkpoint: @escaping (TransactionCheckpoint) throws -> Void = { _ in }
    ) {
        layout = ConfigurationFileLayout(activeURL: activeURL)
        self.fileManager = fileManager
        self.checkpoint = checkpoint
    }

    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    func recoverInterruptedTransactionIfNeeded() throws {
        guard fileManager.fileExists(atPath: layout.transactionJournalURL.path) else {
            return
        }
        let journalData = try Data(contentsOf: layout.transactionJournalURL)
        let journal = try JSONDecoder().decode(ConfigurationTransactionJournal.self, from: journalData)
        let activeDigest = try digestIfPresent(layout.activeURL)
        let lkgDigest = try digestIfPresent(layout.lkgURL)
        if activeDigest == journal.candidateSHA256, lkgDigest == journal.candidateSHA256 {
            try cleanupTransaction(journal)
            return
        }
        do {
            try restoreTarget(
                layout.activeURL,
                existed: journal.activeExisted,
                rollbackName: journal.activeRollbackName
            )
            try restoreTarget(
                layout.lkgURL,
                existed: journal.lkgExisted,
                rollbackName: journal.lkgRollbackName
            )
            try cleanupTransaction(journal)
        } catch {
            throw ConfigurationPersistenceError.transactionRecoveryFailed(layout.transactionJournalURL)
        }
    }

    func commitActiveAndLKG(_ data: Data) throws {
        try prepareDirectories()
        try recoverInterruptedTransactionIfNeeded()
        let transactionID = UUID().uuidString
        let activeRollback = ".\(layout.activeURL.lastPathComponent).rollback-\(transactionID)"
        let lkgRollback = ".\(layout.lkgURL.lastPathComponent).rollback-\(transactionID)"
        let journal = ConfigurationTransactionJournal(
            activeExisted: fileManager.fileExists(atPath: layout.activeURL.path),
            lkgExisted: fileManager.fileExists(atPath: layout.lkgURL.path),
            activeRollbackName: activeRollback,
            lkgRollbackName: lkgRollback,
            candidateSHA256: Self.sha256(data)
        )
        let activeRollbackURL = layout.recoveryDirectoryURL.appendingPathComponent(activeRollback)
        let lkgRollbackURL = layout.recoveryDirectoryURL.appendingPathComponent(lkgRollback)
        if journal.activeExisted {
            try atomicWrite(try Data(contentsOf: layout.activeURL), to: activeRollbackURL, permissions: 0o600)
        }
        if journal.lkgExisted {
            try atomicWrite(try Data(contentsOf: layout.lkgURL), to: lkgRollbackURL, permissions: 0o600)
        }
        let journalData = try JSONEncoder.configurationPersistence.encode(journal)
        try atomicWrite(journalData, to: layout.transactionJournalURL, permissions: 0o600)

        do {
            try checkpoint(.journalPersisted)
            try atomicWrite(data, to: layout.lkgURL, permissions: 0o600)
            try checkpoint(.lkgCommitted)
            try atomicWrite(data, to: layout.activeURL, permissions: 0o600)
            try checkpoint(.activeCommitted)
            guard try Data(contentsOf: layout.activeURL) == data,
                  try Data(contentsOf: layout.lkgURL) == data else {
                throw CocoaError(.fileReadCorruptFile)
            }
            try cleanupTransaction(journal)
        } catch {
            do {
                try restoreTarget(
                    layout.activeURL,
                    existed: journal.activeExisted,
                    rollbackName: activeRollback
                )
                try restoreTarget(
                    layout.lkgURL,
                    existed: journal.lkgExisted,
                    rollbackName: lkgRollback
                )
                try cleanupTransaction(journal)
            } catch {
                throw ConfigurationPersistenceError.transactionRecoveryFailed(layout.transactionJournalURL)
            }
            throw error
        }
    }

    func refreshLKG(_ data: Data) throws {
        try prepareDirectories()
        if fileManager.fileExists(atPath: layout.lkgURL.path),
           try Data(contentsOf: layout.lkgURL) == data {
            return
        }
        try atomicWrite(data, to: layout.lkgURL, permissions: 0o600)
    }

    func preservePreMigration(_ data: Data, schemaVersion: Int) throws -> URL {
        try prepareDirectories()
        let digest = Self.sha256(data)
        let candidate = layout.preMigrationURL(schemaVersion: schemaVersion, digest: digest)
        let baseName = layout.activeURL.deletingPathExtension().lastPathComponent
        let prefix = "\(baseName).pre-migration-v\(schemaVersion)-"
        let existing = try fileManager.contentsOfDirectory(
            at: layout.recoveryDirectoryURL,
            includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.hasPrefix(prefix) }
        let canonicalCandidate = candidate.resolvingSymlinksInPath().standardizedFileURL
        if let other = existing.first(where: {
            $0.resolvingSymlinksInPath().standardizedFileURL != canonicalCandidate
        }) {
            throw ConfigurationPersistenceError.preMigrationEvidenceConflict(other)
        }
        if fileManager.fileExists(atPath: candidate.path) {
            guard try Data(contentsOf: candidate) == data else {
                throw ConfigurationPreservationError.contentMismatch(candidate)
            }
            return candidate
        }
        try atomicWrite(data, to: candidate, permissions: 0o400)
        return candidate
    }

    func atomicWrite(_ data: Data, to url: URL, permissions: Int16) throws {
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        let descriptor = Darwin.open(
            temporary.path,
            O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC,
            mode_t(permissions)
        )
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        do {
            try data.withUnsafeBytes { buffer in
                var offset = 0
                while offset < buffer.count {
                    let written = Darwin.write(
                        descriptor,
                        buffer.baseAddress!.advanced(by: offset),
                        buffer.count - offset
                    )
                    guard written > 0 else {
                        throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                    }
                    offset += written
                }
            }
            guard fsync(descriptor) == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            guard Darwin.close(descriptor) == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            guard rename(temporary.path, url.path) == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            try fileManager.setAttributes([.posixPermissions: permissions], ofItemAtPath: url.path)
            try Self.syncDirectory(url.deletingLastPathComponent())
        } catch {
            Darwin.close(descriptor)
            try? fileManager.removeItem(at: temporary)
            throw error
        }
    }

    private func prepareDirectories() throws {
        try fileManager.createDirectory(
            at: layout.activeURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(
            at: layout.recoveryDirectoryURL,
            withIntermediateDirectories: true
        )
        try fileManager.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: layout.recoveryDirectoryURL.path
        )
    }

    private func restoreTarget(_ target: URL, existed: Bool, rollbackName: String) throws {
        let rollback = layout.recoveryDirectoryURL.appendingPathComponent(rollbackName)
        if existed {
            guard fileManager.fileExists(atPath: rollback.path) else {
                throw ConfigurationPersistenceError.transactionRecoveryFailed(layout.transactionJournalURL)
            }
            try atomicWrite(try Data(contentsOf: rollback), to: target, permissions: 0o600)
        } else if fileManager.fileExists(atPath: target.path) {
            try fileManager.removeItem(at: target)
            try Self.syncDirectory(target.deletingLastPathComponent())
        }
    }

    private func cleanupTransaction(_ journal: ConfigurationTransactionJournal) throws {
        let files = [
            layout.recoveryDirectoryURL.appendingPathComponent(journal.activeRollbackName),
            layout.recoveryDirectoryURL.appendingPathComponent(journal.lkgRollbackName),
            layout.transactionJournalURL,
        ]
        for url in files where fileManager.fileExists(atPath: url.path) {
            try fileManager.removeItem(at: url)
        }
        try Self.syncDirectory(layout.recoveryDirectoryURL)
    }

    private func digestIfPresent(_ url: URL) throws -> String? {
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        return Self.sha256(try Data(contentsOf: url))
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func syncDirectory(_ url: URL) throws {
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

enum ConfigurationJSON {
    static func object(from data: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ConfigurationPersistenceError.invalidTopLevel
        }
        return object
    }

    static func schemaVersion(in object: [String: Any]) throws -> Int? {
        guard let value = object["schema_version"] else {
            return nil
        }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.rounded() == number.doubleValue else {
            return nil
        }
        return number.intValue
    }

    static func requireKeys(_ actual: Set<String>, allowed: Set<String>) throws {
        let unexpected = actual.subtracting(allowed)
        guard unexpected.isEmpty else {
            throw ConfigurationPersistenceError.unexpectedKeys(Array(unexpected))
        }
    }
}

private extension JSONEncoder {
    static var configurationPersistence: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}
