import CryptoKit
import Darwin
import Foundation

public enum ModelRepositoryProvider: String, Codable, CaseIterable, Sendable {
    case huggingFace = "huggingface"
    case modelScope = "modelscope"

    var mutableRevision: String {
        switch self {
        case .huggingFace: "main"
        case .modelScope: "master"
        }
    }
}

public struct ModelSnapshotFile: Codable, Equatable, Sendable {
    public var path: String
    public var size: Int64
    public var sha256: String
    public var etag: String?
    public var blobID: String?

    public init(path: String, size: Int64, sha256: String, etag: String? = nil, blobID: String? = nil) {
        self.path = path
        self.size = size
        self.sha256 = sha256.lowercased()
        self.etag = etag
        self.blobID = blobID
    }

    enum CodingKeys: String, CodingKey {
        case path
        case size
        case sha256
        case etag
        case blobID = "blob_id"
    }
}

public struct ModelSnapshotCompatibility: Codable, Equatable, Sendable {
    public var modelType: String
    public var artifactRole: String
    public var quantizationMode: String?
    public var quantizationBits: Int?
    public var quantizationGroupSize: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case artifactRole = "artifact_role"
        case quantizationMode = "quantization_mode"
        case quantizationBits = "quantization_bits"
        case quantizationGroupSize = "quantization_group_size"
    }
}

public struct ModelSnapshotResources: Codable, Equatable, Sendable {
    public var weightBytes: Int64
    public var estimatedPeakMemoryBytes: Int64

    enum CodingKeys: String, CodingKey {
        case weightBytes = "weight_bytes"
        case estimatedPeakMemoryBytes = "estimated_peak_memory_bytes"
    }
}

public struct ModelSnapshotManifest: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let filename = ".ironmlx-snapshot.json"

    public var version: Int
    public var provider: ModelRepositoryProvider
    public var repoID: String
    public var requestedRevision: String
    public var commitSHA: String
    public var files: [ModelSnapshotFile]
    public var compatibility: ModelSnapshotCompatibility
    public var resources: ModelSnapshotResources
    public var publishedAt: Date

    public init(
        provider: ModelRepositoryProvider,
        repoID: String,
        requestedRevision: String,
        commitSHA: String,
        files: [ModelSnapshotFile],
        compatibility: ModelSnapshotCompatibility,
        resources: ModelSnapshotResources,
        publishedAt: Date = Date()
    ) {
        version = Self.currentVersion
        self.provider = provider
        self.repoID = repoID
        self.requestedRevision = requestedRevision
        self.commitSHA = commitSHA.lowercased()
        self.files = files.sorted { $0.path < $1.path }
        self.compatibility = compatibility
        self.resources = resources
        self.publishedAt = publishedAt
    }

    enum CodingKeys: String, CodingKey {
        case version
        case provider
        case repoID = "repo_id"
        case requestedRevision = "requested_revision"
        case commitSHA = "commit_sha"
        case files
        case compatibility
        case resources
        case publishedAt = "published_at"
    }
}

public struct ModelSnapshotFileIdentity: Codable, Equatable, Sendable {
    public var device: UInt64
    public var inode: UInt64
    public var size: Int64
    public var modifiedSeconds: Int64
    public var modifiedNanoseconds: Int64

    enum CodingKeys: String, CodingKey {
        case device
        case inode
        case size
        case modifiedSeconds = "modified_seconds"
        case modifiedNanoseconds = "modified_nanoseconds"
    }
}

public struct ModelValidatedFile: Equatable, Sendable {
    public var path: String
    public var sha256: String
    public var etag: String?
    public var identity: ModelSnapshotFileIdentity

    public init(
        path: String,
        sha256: String,
        etag: String? = nil,
        identity: ModelSnapshotFileIdentity
    ) {
        self.path = path
        self.sha256 = sha256.lowercased()
        self.etag = etag
        self.identity = identity
    }
}

public enum ModelSnapshotIntegrityState: String, Codable, Sendable {
    case verified
    case corrupt
}

public struct ModelSnapshotIntegrityRecord: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let filename = ".ironmlx-integrity.json"

    public var version: Int
    public var provider: ModelRepositoryProvider
    public var repoID: String
    public var commitSHA: String
    public var state: ModelSnapshotIntegrityState
    public var verifiedAt: Date?
    public var error: String?

    public init(
        provider: ModelRepositoryProvider,
        repoID: String,
        commitSHA: String,
        state: ModelSnapshotIntegrityState,
        verifiedAt: Date? = nil,
        error: String? = nil
    ) {
        version = Self.currentVersion
        self.provider = provider
        self.repoID = repoID
        self.commitSHA = commitSHA
        self.state = state
        self.verifiedAt = verifiedAt
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case version
        case provider
        case repoID = "repo_id"
        case commitSHA = "commit_sha"
        case state
        case verifiedAt = "verified_at"
        case error
    }
}

public enum ModelSnapshotVerificationError: LocalizedError, Equatable {
    case manifestMissing
    case manifestInvalid(String)
    case identityMismatch(String)
    case fileMissing(String)
    case unexpectedFile(String)
    case sizeMismatch(path: String, expected: Int64, actual: Int64)
    case checksumMismatch(path: String, expected: String, actual: String)
    case knownCorrupt(String)
    case fileChangedDuringVerification(String)

    public var errorDescription: String? {
        switch self {
        case .manifestMissing:
            "Model snapshot is unverified because \(ModelSnapshotManifest.filename) is missing."
        case let .manifestInvalid(detail):
            "Model snapshot manifest is invalid: \(detail)"
        case let .identityMismatch(detail):
            "Model snapshot identity does not match: \(detail)"
        case let .fileMissing(path):
            "Model snapshot is missing \(path)."
        case let .unexpectedFile(path):
            "Model snapshot contains unmanifested file \(path)."
        case let .sizeMismatch(path, expected, actual):
            "Model snapshot file \(path) has size \(actual), expected \(expected)."
        case let .checksumMismatch(path, expected, actual):
            "Model snapshot file \(path) has SHA-256 \(actual), expected \(expected)."
        case let .knownCorrupt(detail):
            "Model snapshot is marked as corrupt: \(detail)"
        case let .fileChangedDuringVerification(path):
            "Model snapshot file \(path) changed while it was being verified."
        }
    }
}

public struct ModelSnapshotVerifier: Sendable {
    public typealias Progress = @Sendable (
        _ path: String,
        _ completedBytes: Int64,
        _ totalBytes: Int64
    ) -> Void

    public init() {}

    public func loadManifest(at snapshot: URL) throws -> ModelSnapshotManifest {
        let url = snapshot.appendingPathComponent(ModelSnapshotManifest.filename)
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            throw ModelSnapshotVerificationError.manifestMissing
        }
        do {
            let manifest = try JSONDecoder().decode(ModelSnapshotManifest.self, from: Data(contentsOf: url))
            guard manifest.version == ModelSnapshotManifest.currentVersion else {
                throw ModelSnapshotVerificationError.manifestInvalid("unsupported version \(manifest.version)")
            }
            guard Self.isCommitSHA(manifest.commitSHA) else {
                throw ModelSnapshotVerificationError.manifestInvalid("commit_sha is not a full hexadecimal SHA")
            }
            guard manifest.files.map(\.path) == manifest.files.map(\.path).sorted(),
                  Set(manifest.files.map(\.path)).count == manifest.files.count
            else {
                throw ModelSnapshotVerificationError.manifestInvalid("file paths must be unique and sorted")
            }
            guard manifest.requestedRevision == manifest.provider.mutableRevision else {
                throw ModelSnapshotVerificationError.manifestInvalid(
                    "requested_revision does not match the provider active ref"
                )
            }
            guard manifest.resources.weightBytes >= 0,
                  manifest.resources.estimatedPeakMemoryBytes >= 0
            else {
                throw ModelSnapshotVerificationError.manifestInvalid(
                    "resource sizes must be non-negative"
                )
            }
            for file in manifest.files {
                guard file.size >= 0,
                      file.sha256.count == 64,
                      file.sha256.allSatisfy(\.isHexDigit)
                else {
                    throw ModelSnapshotVerificationError.manifestInvalid(
                        "invalid size or SHA-256 for \(file.path)"
                    )
                }
                _ = try Self.safeFileURL(
                    path: file.path,
                    beneath: URL(fileURLWithPath: "/snapshot", isDirectory: true)
                )
            }
            return manifest
        } catch let error as ModelSnapshotVerificationError {
            throw error
        } catch {
            throw ModelSnapshotVerificationError.manifestInvalid(error.localizedDescription)
        }
    }

    public func verify(
        snapshot: URL,
        expectedProvider: ModelRepositoryProvider? = nil,
        expectedRepoID: String? = nil,
        requireCommitDirectory: Bool = true,
        progress: Progress? = nil
    ) throws -> ModelSnapshotManifest {
        let manifest = try loadManifest(at: snapshot)
        try verifyIdentity(
            manifest,
            snapshot: snapshot,
            expectedProvider: expectedProvider,
            expectedRepoID: expectedRepoID,
            requireCommitDirectory: requireCommitDirectory
        )
        try verifyInventory(manifest, snapshot: snapshot)
        let totalBytes = manifest.files.reduce(Int64(0)) { total, file in
            let result = total.addingReportingOverflow(file.size)
            return result.overflow ? Int64.max : result.partialValue
        }
        var completedBytes: Int64 = 0
        for file in manifest.files {
            let fileURL = try Self.safeFileURL(path: file.path, beneath: snapshot)
            try verifyFileStructure(file, at: fileURL)
            let before = try Self.fileIdentity(of: fileURL)
            let fileBase = completedBytes
            let actualHash = try Self.sha256(of: fileURL) { fileBytes in
                progress?(file.path, fileBase + fileBytes, totalBytes)
            }
            let after = try Self.fileIdentity(of: fileURL)
            guard before == after else {
                throw ModelSnapshotVerificationError.fileChangedDuringVerification(file.path)
            }
            guard actualHash == file.sha256.lowercased() else {
                throw ModelSnapshotVerificationError.checksumMismatch(
                    path: file.path,
                    expected: file.sha256,
                    actual: actualHash
                )
            }
            completedBytes += file.size
            progress?(file.path, completedBytes, totalBytes)
        }
        return manifest
    }

    public func verifyStructure(
        snapshot: URL,
        expectedProvider: ModelRepositoryProvider? = nil,
        expectedRepoID: String? = nil,
        requireCommitDirectory: Bool = true,
        allowKnownCorrupt: Bool = false
    ) throws -> ModelSnapshotManifest {
        let manifest = try loadManifest(at: snapshot)
        try verifyIdentity(
            manifest,
            snapshot: snapshot,
            expectedProvider: expectedProvider,
            expectedRepoID: expectedRepoID,
            requireCommitDirectory: requireCommitDirectory
        )
        for file in manifest.files {
            let fileURL = try Self.safeFileURL(path: file.path, beneath: snapshot)
            try verifyFileStructure(file, at: fileURL)
        }
        try verifyInventory(manifest, snapshot: snapshot)
        let integrityURL = snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
        if !allowKnownCorrupt,
           FileManager.default.fileExists(atPath: integrityURL.path) {
            let record = try loadIntegrityRecord(at: snapshot)
            guard record.provider == manifest.provider,
                  record.repoID == manifest.repoID,
                  record.commitSHA == manifest.commitSHA
            else {
                throw ModelSnapshotVerificationError.identityMismatch(
                    "integrity record does not match the snapshot manifest"
                )
            }
            if record.state == .corrupt {
                throw ModelSnapshotVerificationError.knownCorrupt(
                    record.error ?? "checksum verification failed"
                )
            }
        }
        return manifest
    }

    public func verifyForPublish(
        snapshot: URL,
        manifest: ModelSnapshotManifest,
        validations: [ModelValidatedFile]
    ) throws {
        _ = try verifyStructure(
            snapshot: snapshot,
            expectedProvider: manifest.provider,
            expectedRepoID: manifest.repoID,
            requireCommitDirectory: false
        )
        let byPath = Dictionary(uniqueKeysWithValues: validations.map { ($0.path, $0) })
        for file in manifest.files {
            let fileURL = try Self.safeFileURL(path: file.path, beneath: snapshot)
            if let validation = byPath[file.path],
               validation.sha256 == file.sha256.lowercased(),
               validation.identity == (try Self.fileIdentity(of: fileURL)) {
                continue
            }
            let before = try Self.fileIdentity(of: fileURL)
            let actualHash = try Self.sha256(of: fileURL)
            let after = try Self.fileIdentity(of: fileURL)
            guard before == after else {
                throw ModelSnapshotVerificationError.fileChangedDuringVerification(file.path)
            }
            guard actualHash == file.sha256.lowercased() else {
                throw ModelSnapshotVerificationError.checksumMismatch(
                    path: file.path,
                    expected: file.sha256,
                    actual: actualHash
                )
            }
        }
    }

    public func loadIntegrityRecord(at snapshot: URL) throws -> ModelSnapshotIntegrityRecord {
        let url = snapshot.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
        let record = try JSONDecoder().decode(
            ModelSnapshotIntegrityRecord.self,
            from: Data(contentsOf: url)
        )
        guard record.version == ModelSnapshotIntegrityRecord.currentVersion else {
            throw ModelSnapshotVerificationError.manifestInvalid(
                "unsupported integrity record version \(record.version)"
            )
        }
        return record
    }

    public static func sha256(
        of url: URL,
        progress: (@Sendable (Int64) -> Void)? = nil
    ) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        var completedBytes: Int64 = 0
        var nextProgressBytes: Int64 = 64 * 1_024 * 1_024
        while true {
            try Task.checkCancellation()
            let data = try handle.read(upToCount: 4 * 1_024 * 1_024) ?? Data()
            if data.isEmpty {
                break
            }
            digest.update(data: data)
            completedBytes += Int64(data.count)
            if completedBytes >= nextProgressBytes {
                progress?(completedBytes)
                nextProgressBytes = completedBytes + 64 * 1_024 * 1_024
            }
        }
        progress?(completedBytes)
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func fileSize(of url: URL) throws -> Int64 {
        try fileIdentity(of: url).size
    }

    public static func fileIdentity(of url: URL) throws -> ModelSnapshotFileIdentity {
        var info = stat()
        let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { Darwin.close(descriptor) }
        guard fstat(descriptor, &info) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return ModelSnapshotFileIdentity(
            device: UInt64(info.st_dev),
            inode: UInt64(info.st_ino),
            size: info.st_size,
            modifiedSeconds: Int64(info.st_mtimespec.tv_sec),
            modifiedNanoseconds: Int64(info.st_mtimespec.tv_nsec)
        )
    }

    public static func isCommitSHA(_ value: String) -> Bool {
        value.count == 40 && value.allSatisfy(\.isHexDigit)
    }

    public static func safeFileURL(path: String, beneath root: URL) throws -> URL {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.split(separator: "/", omittingEmptySubsequences: false).contains(where: {
                  $0.isEmpty || $0 == "." || $0 == ".."
              })
        else {
            throw ModelSnapshotVerificationError.manifestInvalid("unsafe repository path \(path)")
        }
        return path.split(separator: "/").reduce(root) {
            $0.appendingPathComponent(String($1), isDirectory: false)
        }
    }

    private func verifyIdentity(
        _ manifest: ModelSnapshotManifest,
        snapshot: URL,
        expectedProvider: ModelRepositoryProvider?,
        expectedRepoID: String?,
        requireCommitDirectory: Bool
    ) throws {
        guard !requireCommitDirectory || snapshot.lastPathComponent == manifest.commitSHA else {
            throw ModelSnapshotVerificationError.identityMismatch(
                "directory \(snapshot.lastPathComponent) != commit \(manifest.commitSHA)"
            )
        }
        if let expectedProvider, manifest.provider != expectedProvider {
            throw ModelSnapshotVerificationError.identityMismatch(
                "provider \(manifest.provider.rawValue) != \(expectedProvider.rawValue)"
            )
        }
        if let expectedRepoID, manifest.repoID != expectedRepoID {
            throw ModelSnapshotVerificationError.identityMismatch(
                "repo \(manifest.repoID) != \(expectedRepoID)"
            )
        }
    }

    private func verifyFileStructure(_ file: ModelSnapshotFile, at fileURL: URL) throws {
        guard FileManager.default.isReadableFile(atPath: fileURL.path) else {
            throw ModelSnapshotVerificationError.fileMissing(file.path)
        }
        let actualSize = try Self.fileSize(of: fileURL)
        guard actualSize == file.size else {
            throw ModelSnapshotVerificationError.sizeMismatch(
                path: file.path,
                expected: file.size,
                actual: actualSize
            )
        }
    }

    private func verifyInventory(
        _ manifest: ModelSnapshotManifest,
        snapshot: URL
    ) throws {
        guard let enumerator = FileManager.default.enumerator(
            at: snapshot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: []
        ) else {
            throw ModelSnapshotVerificationError.manifestInvalid(
                "unable to enumerate snapshot files"
            )
        }
        let expected = Set(manifest.files.map(\.path))
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isDirectoryKey])
            if values.isDirectory == true
                || url.lastPathComponent == ModelSnapshotManifest.filename
                || url.lastPathComponent == ModelSnapshotIntegrityRecord.filename {
                continue
            }
            let components = url.pathComponents
            guard enumerator.level > 0, enumerator.level <= components.count else {
                throw ModelSnapshotVerificationError.manifestInvalid(
                    "unable to resolve snapshot-relative path for \(url.path)"
                )
            }
            let relative = components.suffix(enumerator.level).joined(separator: "/")
            guard expected.contains(relative) else {
                throw ModelSnapshotVerificationError.unexpectedFile(relative)
            }
        }
    }
}

public enum ModelRepositoryLayout {
    public static func repositoryName(repoID: String) throws -> String {
        let parts = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2,
              parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\\") })
        else {
            throw ModelSnapshotVerificationError.manifestInvalid("repo_id must be organization/model")
        }
        return "\(parts[0])--\(parts[1])"
    }

    public static func providerRoot(rootURL: URL, provider: ModelRepositoryProvider) -> URL {
        rootURL
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(provider.rawValue, isDirectory: true)
    }

    public static func repositoryRoot(rootURL: URL, provider: ModelRepositoryProvider, repoID: String) throws -> URL {
        try providerRoot(rootURL: rootURL, provider: provider)
            .appendingPathComponent(repositoryName(repoID: repoID), isDirectory: true)
    }
}
