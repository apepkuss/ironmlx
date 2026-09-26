import CryptoKit
import Darwin
import Foundation

public enum ModelImportError: LocalizedError {
    case invalidSource(String)
    case unsupported(String)
    case insufficientDisk(required: Int64, available: Int64)
    case sourceChanged(String)
    case destinationConflict(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidSource(detail): "Invalid model directory: \(detail)"
        case let .unsupported(detail): "This model cannot be imported: \(detail)"
        case let .insufficientDisk(required, available):
            "Insufficient disk space: \(required) bytes required, \(available) bytes available."
        case let .sourceChanged(path): "The source model changed during import: \(path)"
        case let .destinationConflict(path): "A different model snapshot already exists at \(path)."
        }
    }
}

public struct ModelImportPreview: Sendable {
    public let source: URL
    public let displayName: String
    public let provider: ModelRepositoryProvider
    public let repoID: String
    public let destination: URL
    public let fileCount: Int
    public let totalBytes: Int64
    public let availableBytes: Int64?
    public let compatibility: ModelMetadataPreflightResult
    public let externalResources: [String]?
    fileprivate let files: [InputFile]
    fileprivate let existingManifest: ModelSnapshotManifest?
}

public struct ModelImportResult: Sendable {
    public let repoID: String
    public let snapshot: URL
    public let alreadyPresent: Bool
}

public struct ModelImportProgress: Codable, Sendable {
    public let phase: String
    public let completedBytes: Int64
    public let totalBytes: Int64
    public let currentFile: String
}

fileprivate struct InputFile: Sendable {
    let path: String
    let url: URL
    let identity: ModelSnapshotFileIdentity
}

public struct ModelImportService: Sendable {
    public let rootURL: URL
    private let preflight: any ModelMetadataPreflighting

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true),
        preflight: any ModelMetadataPreflighting = IronMLXModelMetadataPreflight()
    ) {
        self.rootURL = rootURL
        self.preflight = preflight
    }

    public func inspect(_ source: URL) async throws -> ModelImportPreview {
        try Task.checkCancellation()
        let source = try Self.physicalURL(source.standardizedFileURL)
        let managedRoot = (try? Self.physicalURL(rootURL))?
            .appendingPathComponent("models", isDirectory: true)
            ?? rootURL.appendingPathComponent("models", isDirectory: true)
        guard !source.pathComponents.starts(with: managedRoot.pathComponents) else {
            throw ModelImportError.invalidSource("Choose a directory outside the IronMLX model library.")
        }
        guard try source.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
            throw ModelImportError.invalidSource("Choose a model snapshot or model folder.")
        }
        guard FileManager.default.isReadableFile(atPath: source.appendingPathComponent("config.json").path) else {
            throw ModelImportError.invalidSource("config.json is missing. Select the snapshot directory, not its cache parent.")
        }
        let verifier = ModelSnapshotVerifier()
        let manifestURL = source.appendingPathComponent(ModelSnapshotManifest.filename)
        let manifest = FileManager.default.fileExists(atPath: manifestURL.path)
            ? try verifier.verifyStructure(snapshot: source, requireCommitDirectory: false) : nil
        let native = manifest.flatMap { item -> ModelSnapshotManifest? in
            item.provider == .standalone ? nil : item
        }
        let files = try collectFiles(in: source, manifest: manifest)
        try Task.checkCancellation()
        let remoteFiles = files.map { file in
            RemoteModelFile(path: file.path, size: file.identity.size, sha256: nil, blobID: nil, etag: nil)
        }
        let ttsProfile = try TTSModelDownloadProfile.inspect(directory: source, files: remoteFiles)
        let compatibility: ModelMetadataPreflightResult
        if let ttsProfile {
            compatibility = ttsProfile.compatibility
        } else {
            compatibility = try await preflight.validate(metadataDirectory: source)
        }
        try Task.checkCancellation()
        guard !files.isEmpty, files.contains(where: { $0.path.hasSuffix(".safetensors") }) else {
            throw ModelImportError.invalidSource("No safetensors model weights were found.")
        }
        let tokenizerPath = compatibility.artifactRole == "decision"
            ? "tokenizer/tokenizer.json" : "tokenizer.json"
        if ttsProfile == nil, compatibility.artifactRole != ModelArtifactRole.dflash2Drafter,
           !files.contains(where: { $0.path == tokenizerPath }) {
            throw ModelImportError.invalidSource("\(tokenizerPath) is missing.")
        }
        let displayName = native?.displayName ?? native?.repoID ?? Self.cacheModelName(source) ?? source.lastPathComponent
        let totalBytes = try files.reduce(Int64(0)) { total, file in
            let sum = total.addingReportingOverflow(file.identity.size)
            guard !sum.overflow else { throw ModelImportError.invalidSource("Model size overflows the supported range.") }
            return sum.partialValue
        }
        let provider = native?.provider ?? .standalone
        let repoID: String
        if let native {
            repoID = native.repoID
        } else {
            let slug = Self.slug(displayName)
            repoID = "standalone/\(slug)-\(UUID().uuidString.prefix(8).lowercased())"
        }
        let destination = try ModelRepositoryLayout.repositoryRoot(
            rootURL: rootURL, provider: provider, repoID: repoID
        )
        var capacityURL = managedRoot
        while !FileManager.default.fileExists(atPath: capacityURL.path),
              capacityURL.deletingLastPathComponent() != capacityURL {
            capacityURL = capacityURL.deletingLastPathComponent()
        }
        let capacity = try? capacityURL.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey
        ])
        let available = capacity?.volumeAvailableCapacityForImportantUsage
            ?? capacity?.volumeAvailableCapacity.map { Int64($0) }
        let resourceReservation = ttsProfile == nil ? 0 : AudioResourcePreparationService.diskReservationBytes
        let required = totalBytes.addingReportingOverflow(ModelResourcePreflight.diskSafetyBytes)
        let withResources = required.partialValue.addingReportingOverflow(resourceReservation)
        let requiredBytes = required.overflow || withResources.overflow ? Int64.max : withResources.partialValue
        if let available, requiredBytes > available {
            throw ModelImportError.insufficientDisk(required: requiredBytes, available: available)
        }
        return ModelImportPreview(
            source: source,
            displayName: displayName,
            provider: provider,
            repoID: repoID,
            destination: destination,
            fileCount: files.count,
            totalBytes: totalBytes,
            availableBytes: available,
            compatibility: compatibility,
            externalResources: ttsProfile?.externalResources ?? native?.compatibility.externalResources,
            files: files,
            existingManifest: native
        )
    }

    public func importModel(
        _ preview: ModelImportPreview,
        progress: @escaping @Sendable (ModelImportProgress) -> Void = { _ in }
    ) async throws -> ModelImportResult {
        let store = ModelDownloadStore(rootURL: rootURL)
        try ensureManagedPathIsNotRedirected(preview)
        let currentFiles = try collectFiles(in: preview.source, manifest: preview.existingManifest)
        guard currentFiles.map(\.path) == preview.files.map(\.path),
              currentFiles.map(\.identity) == preview.files.map(\.identity) else {
            throw ModelImportError.sourceChanged("file inventory")
        }
        let lock = try store.acquireRepositoryLock(provider: preview.provider, repoID: preview.repoID)
        defer { withExtendedLifetime(lock) {} }
        let repository = preview.destination
        let staging = repository.appendingPathComponent(".imports", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        var copiedBytes: Int64 = 0
        var copied: [ModelSnapshotFile] = []
        var contentDigest = SHA256()
        for file in preview.files {
            try Task.checkCancellation()
            let before = try ModelSnapshotVerifier.fileIdentity(of: file.url)
            guard before == file.identity else { throw ModelImportError.sourceChanged(file.path) }
            let target = try ModelSnapshotVerifier.safeFileURL(path: file.path, beneath: staging)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard FileManager.default.createFile(atPath: target.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
            let input = try FileHandle(forReadingFrom: file.url)
            let output = try FileHandle(forWritingTo: target)
            var digest = SHA256()
            do {
                while true {
                    try Task.checkCancellation()
                    let chunk = try input.read(upToCount: 4 * 1_024 * 1_024) ?? Data()
                    if chunk.isEmpty { break }
                    try output.write(contentsOf: chunk)
                    digest.update(data: chunk)
                    copiedBytes += Int64(chunk.count)
                    progress(ModelImportProgress(phase: "copying", completedBytes: copiedBytes,
                        totalBytes: preview.totalBytes, currentFile: file.path))
                }
                try output.synchronize()
                try input.close()
                try output.close()
            } catch {
                try? input.close()
                try? output.close()
                throw error
            }
            guard try ModelSnapshotVerifier.fileIdentity(of: file.url) == before else {
                throw ModelImportError.sourceChanged(file.path)
            }
            let sha = digest.finalize().map { String(format: "%02x", $0) }.joined()
            let sourceEntry = preview.existingManifest?.files.first(where: { $0.path == file.path })
            if let expected = sourceEntry,
               expected.sha256 != sha {
                throw ModelImportError.sourceChanged(file.path)
            }
            copied.append(ModelSnapshotFile(path: file.path, size: before.size, sha256: sha,
                etag: sourceEntry?.etag, blobID: sourceEntry?.blobID))
            contentDigest.update(data: Data(file.path.utf8))
            contentDigest.update(data: Data([0]))
            contentDigest.update(data: Data(sha.utf8))
        }
        guard copiedBytes == preview.totalBytes else {
            throw ModelImportError.sourceChanged("file inventory")
        }
        let afterFiles = try collectFiles(in: preview.source, manifest: preview.existingManifest)
        guard afterFiles.map(\.path) == preview.files.map(\.path),
              afterFiles.map(\.identity) == preview.files.map(\.identity) else {
            throw ModelImportError.sourceChanged("file inventory")
        }
        let contentID = contentDigest.finalize().map { String(format: "%02x", $0) }.joined()
        // Standalone snapshots use a content-derived 40-hex identity in the existing snapshot schema.
        let snapshotID = preview.existingManifest?.commitSHA ?? String(contentID.prefix(40))
        let weightBytes = copied.filter { $0.path.hasSuffix(".safetensors") }
            .reduce(Int64(0)) { $0 + $1.size }
        let manifest = ModelSnapshotManifest(
            provider: preview.provider,
            repoID: preview.repoID,
            requestedRevision: preview.provider.mutableRevision,
            commitSHA: snapshotID,
            files: copied,
            compatibility: ModelSnapshotCompatibility(
                modelType: preview.compatibility.modelType,
                artifactRole: preview.compatibility.artifactRole,
                quantizationMode: preview.compatibility.quantization?.mode,
                quantizationBits: preview.compatibility.quantization?.bits,
                quantizationGroupSize: preview.compatibility.quantization?.groupSize,
                externalResources: preview.externalResources
            ),
            resources: ModelSnapshotResources(
                weightBytes: weightBytes,
                estimatedPeakMemoryBytes: weightBytes + max(512 * 1_024 * 1_024, weightBytes / 10)
            ),
            displayName: preview.provider == .standalone ? preview.displayName : preview.existingManifest?.displayName
        )
        try store.writeManifest(manifest, to: staging)
        _ = try ModelSnapshotVerifier().verify(snapshot: staging,
            expectedProvider: preview.provider, expectedRepoID: preview.repoID,
            requireCommitDirectory: false
        ) { path, completed, total in
            progress(ModelImportProgress(phase: "verifying", completedBytes: completed,
                totalBytes: total, currentFile: path))
        }
        if preview.compatibility.artifactRole == "tts" {
            _ = try await AudioResourcePreparationService(rootURL: rootURL).prepare(
                snapshot: staging, token: nil
            ) { name, _, _ in
                progress(ModelImportProgress(phase: "preparing", completedBytes: 1,
                    totalBytes: 1, currentFile: name))
            }
        }
        let readiness = LocalModelScanner(rootURL: rootURL).readiness(for: staging.path)
        guard readiness?.isLoadable == true else {
            throw ModelImportError.unsupported(readiness?.message ?? "The copied snapshot is not ready to load.")
        }
        try ModelDownloadStore.atomicWrite(
            ModelSnapshotIntegrityRecord(
                provider: preview.provider,
                repoID: preview.repoID,
                commitSHA: snapshotID,
                state: .verified,
                verifiedAt: Date()
            ),
            to: staging.appendingPathComponent(ModelSnapshotIntegrityRecord.filename)
        )
        let final = try store.snapshotURL(provider: preview.provider, repoID: preview.repoID, commitSHA: snapshotID)
        try FileManager.default.createDirectory(at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: final.path) {
            let existing = try ModelSnapshotVerifier().verify(snapshot: final,
                expectedProvider: preview.provider, expectedRepoID: preview.repoID)
            guard existing.files == manifest.files else {
                throw ModelImportError.destinationConflict(final.path)
            }
            try store.updateRef(for: existing)
            return ModelImportResult(repoID: preview.repoID, snapshot: final, alreadyPresent: true)
        }
        try Task.checkCancellation()
        guard rename(staging.path, final.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        do {
            try ModelDownloadStore.syncDirectory(final.deletingLastPathComponent())
            try store.updateRef(for: manifest)
        } catch {
            try? FileManager.default.removeItem(at: final)
            throw error
        }
        return ModelImportResult(repoID: preview.repoID, snapshot: final, alreadyPresent: false)
    }

    private func collectFiles(in source: URL, manifest: ModelSnapshotManifest?) throws -> [InputFile] {
        let paths: [String]
        if let manifest {
            paths = manifest.files.map(\.path)
        } else {
            guard let enumerator = FileManager.default.enumerator(
                at: source, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: []
            ) else { throw ModelImportError.invalidSource("The folder cannot be read.") }
            var discovered: [String] = []
            for case let url as URL in enumerator {
                let relative = url.pathComponents.suffix(enumerator.level).joined(separator: "/")
                if relative == ".git" || relative == ".cache" {
                    enumerator.skipDescendants()
                    continue
                }
                if [".DS_Store", ModelSnapshotManifest.filename, ModelSnapshotIntegrityRecord.filename]
                    .contains(url.lastPathComponent) { continue }
                let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                if values.isSymbolicLink == true,
                   (try? url.resolvingSymlinksInPath().resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    throw ModelImportError.invalidSource("Linked directories are not supported: \(relative)")
                }
                if values.isDirectory == true { continue }
                discovered.append(relative)
            }
            paths = discovered.sorted()
        }
        let allowed = source.deletingLastPathComponent().lastPathComponent == "snapshots"
            && source.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent.hasPrefix("models--")
            ? source.deletingLastPathComponent().deletingLastPathComponent() : source
        let allowedPath = try Self.physicalURL(allowed).pathComponents
        return try paths.map { path in
            let url = try ModelSnapshotVerifier.safeFileURL(path: path, beneath: source)
            let physical = try Self.physicalURL(url)
            guard physical.pathComponents.starts(with: allowedPath) else {
                throw ModelImportError.invalidSource("A linked file escapes the selected model: \(path)")
            }
            let values = try physical.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else {
                throw ModelImportError.invalidSource("Not a regular file: \(path)")
            }
            return InputFile(path: path, url: url, identity: try ModelSnapshotVerifier.fileIdentity(of: url))
        }
    }

    private static func slug(_ value: String) -> String {
        let characters = value.lowercased().unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) && scalar.isASCII ? Character(String(scalar)) : "-"
        }
        let collapsed = String(characters).split(separator: "-").joined(separator: "-")
        return String(collapsed.prefix(48)).isEmpty ? "model" : String(collapsed.prefix(48))
    }

    private static func cacheModelName(_ snapshot: URL) -> String? {
        let snapshots = snapshot.deletingLastPathComponent()
        let repository = snapshots.deletingLastPathComponent().lastPathComponent
        guard snapshots.lastPathComponent == "snapshots", repository.hasPrefix("models--") else { return nil }
        let identity = String(repository.dropFirst("models--".count))
        guard let separator = identity.range(of: "--"),
              separator.lowerBound != identity.startIndex,
              separator.upperBound != identity.endIndex else { return nil }
        return String(identity[..<separator.lowerBound]) + "/" + String(identity[separator.upperBound...])
    }

    private static func physicalURL(_ url: URL) throws -> URL {
        guard let path = realpath(url.path, nil) else {
            throw ModelImportError.invalidSource("A source file is missing or unreadable: \(url.path)")
        }
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path))
    }

    private func ensureManagedPathIsNotRedirected(_ preview: ModelImportPreview) throws {
        let modelRoot = rootURL.appendingPathComponent("models", isDirectory: true)
        let providerRoot = ModelRepositoryLayout.providerRoot(rootURL: rootURL, provider: preview.provider)
        for path in [modelRoot, providerRoot, preview.destination,
                     preview.destination.appendingPathComponent(".imports"),
                     preview.destination.appendingPathComponent("snapshots")] {
            var info = stat()
            if lstat(path.path, &info) == 0 {
                guard info.st_mode & S_IFMT != S_IFLNK else {
                    throw ModelImportError.invalidSource("The model library contains a redirected directory: \(path.path)")
                }
            } else if errno != ENOENT {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        }
    }
}
