import CryptoKit
import Foundation

@testable import IronMLXAppCore

@discardableResult
func writeVerifiedTestSnapshot(
    root: URL,
    provider: ModelRepositoryProvider = .huggingFace,
    repoID: String,
    files: [String: Data],
    commitSHA: String = String(repeating: "a", count: 40)
) throws -> URL {
    let repository = try ModelRepositoryLayout.repositoryRoot(
        rootURL: root,
        provider: provider,
        repoID: repoID
    )
    let snapshot = repository
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent(commitSHA, isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    for (path, data) in files {
        let url = try ModelSnapshotVerifier.safeFileURL(path: path, beneath: snapshot)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
    }
    let entries = files.map { path, data in
        ModelSnapshotFile(
            path: path,
            size: Int64(data.count),
            sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        )
    }
    let manifest = ModelSnapshotManifest(
        provider: provider,
        repoID: repoID,
        requestedRevision: provider.mutableRevision,
        commitSHA: commitSHA,
        files: entries,
        compatibility: ModelSnapshotCompatibility(
            modelType: "llama",
            artifactRole: "base",
            quantizationMode: nil,
            quantizationBits: nil,
            quantizationGroupSize: nil
        ),
        resources: ModelSnapshotResources(
            weightBytes: entries.filter { $0.path.hasSuffix(".safetensors") }.reduce(0) { $0 + $1.size },
            estimatedPeakMemoryBytes: 0
        )
    )
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
    try ModelDownloadStore.atomicWrite(
        Data((commitSHA + "\n").utf8),
        to: repository
            .appendingPathComponent("refs", isDirectory: true)
            .appendingPathComponent(provider.mutableRevision)
    )
    return snapshot
}

func refreshTestSnapshotManifest(at snapshot: URL) throws {
    let verifier = ModelSnapshotVerifier()
    var manifest = try verifier.loadManifest(at: snapshot)
    guard let enumerator = FileManager.default.enumerator(
        at: snapshot,
        includingPropertiesForKeys: [.isDirectoryKey]
    ) else {
        throw CocoaError(.fileReadUnknown)
    }
    var files: [ModelSnapshotFile] = []
    for case let url as URL in enumerator {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey])
        if values.isDirectory == true
            || url.lastPathComponent == ModelSnapshotManifest.filename
            || url.lastPathComponent == ModelSnapshotIntegrityRecord.filename {
            continue
        }
        let components = url.pathComponents
        let path = components.suffix(enumerator.level).joined(separator: "/")
        let data = try Data(contentsOf: url)
        files.append(
            ModelSnapshotFile(
                path: path,
                size: Int64(data.count),
                sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            )
        )
    }
    manifest.files = files.sorted { $0.path < $1.path }
    try ModelDownloadStore(rootURL: snapshot).writeManifest(manifest, to: snapshot)
}
