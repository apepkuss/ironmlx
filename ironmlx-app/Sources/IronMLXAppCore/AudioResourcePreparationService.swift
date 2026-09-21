import Darwin
import Foundation
import ZIPFoundation

public struct BackendAudioResources: Codable, Equatable, Sendable {
    public var derivedResources: String
    public var resourceLock: String
    public var wetextFsts: String
    public var unidicDir: String

    public init(derivedResources: String, resourceLock: String, wetextFsts: String, unidicDir: String) {
        self.derivedResources = derivedResources
        self.resourceLock = resourceLock
        self.wetextFsts = wetextFsts
        self.unidicDir = unidicDir
    }

    enum CodingKeys: String, CodingKey {
        case derivedResources = "derived_resources"
        case resourceLock = "resource_lock"
        case wetextFsts = "wetext_fsts"
        case unidicDir = "unidic_dir"
    }
}

public protocol AudioResourcePreparing: Sendable {
    func prepare(
        snapshot: URL, token: String?,
        progress: @escaping @Sendable (String, Int64, Int64) async -> Void
    ) async throws -> BackendAudioResources
}

/// Provision data beside immutable snapshots. Downloaded archives are pinned;
/// neither package installation nor execution of upstream code is involved.
struct AudioResourcePreparationService: AudioResourcePreparing {
    /// Includes archive cache, extracted dictionaries, new and replaced derived data.
    static let diskReservationBytes: Int64 = 512 * 1_024 * 1_024
    let rootURL: URL
    var httpClient: any ModelDownloadHTTPClient = URLSessionModelDownloadHTTPClient()
    var huggingFaceEndpoint = URL(string: "https://huggingface.co")!

    private struct Receipt: Codable {
        var configuration: BackendAudioResources
        var files: [String: ModelSnapshotFileIdentity]
    }

    /// Fast scan check. Full hashes are checked during preparation and again by
    /// the native loader. An edited/replaced resource requires preparation again.
    func readyConfiguration() throws -> BackendAudioResources {
        let profile = try AudioResourceProfile()
        let directory = directory(for: profile)
        let receipt = try JSONDecoder().decode(Receipt.self, from: Data(contentsOf:
            directory.appendingPathComponent("configuration.json")))
        guard receipt.configuration == configuration(at: directory),
              Set(receipt.files.keys) == Set(resourcePaths(profile)) else {
            throw AudioResourceError.invalid("resource configuration is incomplete")
        }
        for (path, identity) in receipt.files {
            guard try ModelSnapshotVerifier.fileIdentity(of: AudioResourceProfile.path(path, in: directory)) == identity else {
                throw AudioResourceError.invalid("resource changed: \(path)")
            }
        }
        return receipt.configuration
    }

    private func resourcePaths(_ profile: AudioResourceProfile) -> [String] {
        (profile.outputs + profile.files).map { "derived/" + $0.path }
            + profile.text.flatMap { $0.members.map { "text/" + $0.path } }
            + ["derived/manifest.json", "sources.json"]
    }

    private func writeReceipt(at directory: URL, destination: URL, profile: AudioResourceProfile) throws {
        var identities: [String: ModelSnapshotFileIdentity] = [:]
        for path in resourcePaths(profile) {
            identities[path] = try ModelSnapshotVerifier.fileIdentity(of: AudioResourceProfile.path(path, in: directory))
        }
        try JSONEncoder().encode(Receipt(configuration: configuration(at: destination), files: identities))
            .write(to: directory.appendingPathComponent("configuration.json"), options: .atomic)
    }

    func directory(for profile: AudioResourceProfile) -> URL {
        rootURL.appendingPathComponent("audio/indextts25/\(profile.source.revision)", isDirectory: true)
    }

    func configuration(at directory: URL) -> BackendAudioResources {
        BackendAudioResources(
            derivedResources: directory.appendingPathComponent("derived").path,
            resourceLock: directory.appendingPathComponent("sources.json").path,
            wetextFsts: directory.appendingPathComponent("text/wetext/fsts").path,
            unidicDir: directory.appendingPathComponent("text/unidic-lite-1.0.8/unidic_lite/dicdir").path
        )
    }

    func verify(at directory: URL, profile: AudioResourceProfile) throws {
        try profile.verifyDerived(at: directory.appendingPathComponent("derived"))
        for package in profile.text {
            for file in package.members {
                try file.verify(at: AudioResourceProfile.path(file.path, in: directory.appendingPathComponent("text")))
            }
        }
        guard try Data(contentsOf: directory.appendingPathComponent("sources.json")) == profile.resourceLock else {
            throw AudioResourceError.invalid("resource lock differs from bundled profile")
        }
    }

    /// Existing valid resources are usable offline; partial downloads survive retry.
    /// A process lock prevents concurrent writers and is released even after a crash.
    func prepare(
        snapshot: URL,
        token: String? = nil,
        progress: @escaping @Sendable (String, Int64, Int64) async -> Void = { _, _, _ in }
    ) async throws -> BackendAudioResources {
        let profile = try AudioResourceProfile()
        // Verify the complete runtime profile, including vocabulary and model config.
        // Snapshot download integrity alone does not establish model compatibility.
        for file in profile.source.files {
            try file.verify(at: AudioResourceProfile.path(file.path, in: snapshot))
        }
        let destination = directory(for: profile)
        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let lockURL = destination.appendingPathExtension("lock")
        let lock = open(lockURL.path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
        guard lock >= 0 else { throw AudioResourceError.invalid("cannot open preparation lock") }
        defer { close(lock) }
        guard flock(lock, LOCK_EX | LOCK_NB) == 0 else {
            throw AudioResourceError.invalid("preparation is already running; retry after it finishes")
        }
        defer { _ = flock(lock, LOCK_UN) }
        do {
            try verify(at: destination, profile: profile)
            try writeReceipt(at: destination, destination: destination, profile: profile)
            return configuration(at: destination)
        } catch is CancellationError { throw CancellationError() }
        catch { /* Missing or damaged resources are rebuilt from verified sources. */ }

        let cache = parent.appendingPathComponent("downloads/\(profile.source.revision)", isDirectory: true)
        let stage = parent.appendingPathComponent(".prepare-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: stage) }
        var roots = ["snapshot": snapshot]
        for source in profile.auxiliary {
            let role: String
            switch source.repository {
            case "funasr/campplus": role = "campplus"
            case "facebook/w2v-bert-2.0": role = "w2v"
            default: throw AudioResourceError.invalid("unknown auxiliary source")
            }
            let root = cache.appendingPathComponent(role)
            roots[role] = root
            for file in source.files {
                let url = huggingFaceEndpoint.appendingPathComponent(source.repository)
                    .appendingPathComponent("resolve/\(source.revision)/\(file.path)")
                try await download(file, url: url, destination: AudioResourceProfile.path(file.path, in: root),
                                   repository: source.repository, revision: source.revision, token: token, progress: progress)
            }
        }
        await progress("Preparing reference encoder resources", 0, 1)
        try profile.convert(roots: roots, destination: stage.appendingPathComponent("derived"))
        let text = stage.appendingPathComponent("text", isDirectory: true)
        try FileManager.default.createDirectory(at: text, withIntermediateDirectories: true)
        for package in profile.text {
            let archive = cache.appendingPathComponent(package.artifact.url.lastPathComponent)
            let file = AudioResourceProfile.File(path: archive.lastPathComponent, bytes: package.artifact.size,
                                                sha256: package.sha256)
            try await download(file, url: package.artifact.url, destination: archive,
                               repository: "text/\(package.name)", revision: package.sha256, token: nil, progress: progress)
            try await extract(package, archive: archive, to: text)
        }
        try profile.resourceLock.write(to: stage.appendingPathComponent("sources.json"), options: .atomic)
        try verify(at: stage, profile: profile)
        try writeReceipt(at: stage, destination: destination, profile: profile)
        try Task.checkCancellation()
        // Only this exact managed profile is replaced; immutable model snapshots
        // and other versions are never modified. A failed publish restores the old directory.
        let old = parent.appendingPathComponent(".replaced-\(UUID().uuidString)")
        let hadOld = FileManager.default.fileExists(atPath: destination.path)
        if hadOld { try FileManager.default.moveItem(at: destination, to: old) }
        do { try FileManager.default.moveItem(at: stage, to: destination) }
        catch {
            if hadOld { try? FileManager.default.moveItem(at: old, to: destination) }
            throw error
        }
        if hadOld { try? FileManager.default.removeItem(at: old) }
        await progress("Audio resources ready", 1, 1)
        return configuration(at: destination)
    }

    private func download(
        _ file: AudioResourceProfile.File, url: URL, destination: URL,
        repository: String, revision: String, token: String?,
        progress: @escaping @Sendable (String, Int64, Int64) async -> Void
    ) async throws {
        try Task.checkCancellation()
        var request = URLRequest(url: url)
        if let token, !token.isEmpty { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        _ = try await ResumableFileDownloader(httpClient: httpClient).download(
            ResumableDownloadRequest(
                urlRequest: request,
                identity: ModelPartialIdentity(provider: .huggingFace, repoID: repository,
                                               commitSHA: revision, path: file.path,
                                               expectedSize: file.bytes, expectedSHA256: file.sha256, etag: nil),
                destination: destination
            ),
            progress: { bytes in await progress(file.path, bytes, file.bytes) }
        )
        try file.verify(at: destination)
    }

    private func extract(_ package: AudioResourceProfile.TextResource, archive: URL, to directory: URL) async throws {
        for member in package.members { _ = try AudioResourceProfile.path(member.path, in: directory) }
        if archive.pathExtension == "whl" {
            let zip = try Archive(url: archive, accessMode: .read)
            for member in package.members {
                try Task.checkCancellation()
                guard let entry = zip[member.path], entry.type == .file, entry.uncompressedSize == member.bytes else {
                    throw AudioResourceError.invalid("invalid text archive member: \(member.path)")
                }
                _ = try zip.extract(entry, to: AudioResourceProfile.path(member.path, in: directory))
            }
            // Preserve the wheel's distribution metadata and licensing alongside its data.
            for entry in zip where entry.type == .file && entry.path.contains(".dist-info/") {
                _ = try zip.extract(entry, to: AudioResourceProfile.path(entry.path, in: directory))
            }
        } else {
            // macOS ships bsdtar. Only the pinned member list is extracted from a
            // hash-verified archive; package setup scripts are data, never executed.
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            process.arguments = ["-xzf", archive.path, "-C", directory.path, "--"] + package.members.map(\.path)
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            try process.run()
            do {
                while process.isRunning { try await Task.sleep(for: .milliseconds(50)) }
            } catch {
                if process.isRunning { process.terminate() }
                process.waitUntilExit()
                throw error
            }
            guard process.terminationStatus == 0 else { throw AudioResourceError.invalid("text archive extraction failed") }
        }
        for file in package.members { try file.verify(at: AudioResourceProfile.path(file.path, in: directory)) }
    }
}
