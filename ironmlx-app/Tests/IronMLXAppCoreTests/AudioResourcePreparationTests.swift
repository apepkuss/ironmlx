import CryptoKit
import Foundation
import Testing
import ZIPFoundation

@testable import IronMLXAppCore

private struct AudioConversionFixture {
    let root: URL
    let archive: URL
    let output: URL
    let bytes = Data([0, 0, 0, 128, 255, 255, 255, 127])
    let header = Data(repeating: 0, count: 8)
    var recipe: [String: Any]

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        archive = root.appendingPathComponent("weights.pt")
        output = root.appendingPathComponent("derived")
        let zip = try Archive(url: archive, accessMode: .create)
        let content = bytes
        try zip.addEntry(with: "weights/data/0", type: .file, uncompressedSize: Int64(content.count),
                         provider: { offset, count in content.subdata(in: Int(offset)..<Int(offset) + count) })
        func record(_ name: String, _ data: Data) -> [String: Any] {
            ["path": name, "bytes": data.count, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
        }
        var input = record("weights.pt", try Data(contentsOf: archive))
        input["role"] = "snapshot"
        recipe = [
            "version": 1,
            "sources": ["source": ["repository": "test/model", "revision": "test", "files": [input]],
                        "auxiliary_sources": [], "text_resources": []],
            "manifest": ["inputs": [input], "files": [], "components": ["weights.safetensors": record("weights.safetensors", header + bytes)]],
            "components": [["path": "weights.safetensors", "header": header.base64EncodedString(),
                            "blocks": [["role": "snapshot", "archive": "weights.pt", "member": "weights/data/0", "bytes": bytes.count]]]],
        ]
    }

    func profile() throws -> AudioResourceProfile {
        try AudioResourceProfile(data: JSONSerialization.data(withJSONObject: recipe))
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

@Test func audioResourceConversionPreservesEveryStorageBitAndVerifiesOutputs() throws {
    let fixture = try AudioConversionFixture()
    defer { fixture.remove() }
    let profile = try fixture.profile()
    try profile.convert(roots: ["snapshot": fixture.root], destination: fixture.output)
    #expect(try Data(contentsOf: fixture.output.appendingPathComponent("weights.safetensors")) == fixture.header + fixture.bytes)
    try profile.verifyDerived(at: fixture.output)
    try Data(repeating: 1, count: 16).write(to: fixture.output.appendingPathComponent("weights.safetensors"))
    #expect(throws: AudioResourceError.self) { try profile.verifyDerived(at: fixture.output) }
}

@Test func audioResourceConversionRejectsChangedSourceBeforeCreatingOutput() throws {
    let fixture = try AudioConversionFixture()
    defer { fixture.remove() }
    let profile = try fixture.profile()
    var changed = try Data(contentsOf: fixture.archive)
    changed[changed.count - 1] ^= 1
    try changed.write(to: fixture.archive)
    #expect(throws: AudioResourceError.self) {
        try profile.convert(roots: ["snapshot": fixture.root], destination: fixture.output)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test func audioResourceConversionRejectsMissingOrMismatchedStorage() throws {
    var fixture = try AudioConversionFixture()
    defer { fixture.remove() }
    fixture.recipe["components"] = [["path": "weights.safetensors", "header": "", "blocks": [
        ["role": "snapshot", "archive": "weights.pt", "member": "weights/data/1", "bytes": 8]
    ]]]
    #expect(throws: AudioResourceError.self) {
        try fixture.profile().convert(roots: ["snapshot": fixture.root], destination: fixture.output)
    }
}

@Test func audioResourceProfileRejectsPathTraversal() throws {
    #expect(throws: (any Error).self) {
        try AudioResourceProfile.path("../outside", in: URL(fileURLWithPath: "/tmp/audio"))
    }
}

@Test func audioResourceBundledRecipeMatchesNativeSourceProfiles() throws {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let profile = try AudioResourceProfile()
    let source = try Data(contentsOf: repository.appendingPathComponent("ironmlx-audio/resources/indextts25/sources.json"))
    #expect(try JSONSerialization.jsonObject(with: source) as? NSDictionary
            == JSONSerialization.jsonObject(with: profile.resourceLock) as? NSDictionary)
    #expect(profile.components.reduce(0) { $0 + $1.blocks.count } == 941)
    let manifest = try #require(JSONSerialization.jsonObject(with: profile.manifest) as? [String: Any])
    let pinned = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
        repository.appendingPathComponent("ironmlx-audio/resources/indextts25/derived.json"))) as? [String: Any])
    for (key, value) in pinned where key != "components" {
        #expect(NSDictionary(dictionary: [key: value]) == NSDictionary(dictionary: [key: manifest[key] as Any]))
    }
    let components = try #require(manifest["components"] as? [String: [String: Any]])
    let expected = try #require(pinned["components"] as? [String: [String: Any]])
    for (name, spec) in expected {
        for (key, value) in spec {
            #expect(NSDictionary(dictionary: [key: value]) == NSDictionary(dictionary: [key: components[name]?[key] as Any]))
        }
    }
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["IRONMLX_AUDIO_PREPARE_SNAPSHOT"] != nil))
func audioResourceRealNetworkPreparationAndOfflineReuse() async throws {
    let env = ProcessInfo.processInfo.environment
    let snapshot = URL(fileURLWithPath: try #require(env["IRONMLX_AUDIO_PREPARE_SNAPSHOT"]))
    let root = URL(fileURLWithPath: try #require(env["IRONMLX_AUDIO_PREPARE_ROOT"]))
    let service = AudioResourcePreparationService(rootURL: root)
    let config = try await service.prepare(snapshot: snapshot)
    #expect(try service.readyConfiguration() == config)
    let offline = AudioResourcePreparationService(rootURL: root, httpClient: OfflineAudioHTTPClient())
    #expect(try await offline.prepare(snapshot: snapshot) == config)
    // A changed resource must disappear from readiness and be repaired entirely
    // from verified download caches without contacting the network.
    let damaged = URL(fileURLWithPath: config.derivedResources).appendingPathComponent("auxiliary.safetensors")
    var data = try Data(contentsOf: damaged)
    data[data.count - 1] ^= 1
    try data.write(to: damaged)
    #expect(throws: AudioResourceError.self) { try service.readyConfiguration() }
    #expect(try await offline.prepare(snapshot: snapshot) == config)
    #expect(try offline.readyConfiguration() == config)
    try service.verify(at: service.directory(for: AudioResourceProfile()), profile: AudioResourceProfile())
    // With no cache, a network failure must never publish partial readiness.
    let failedRoot = root.appendingPathComponent("offline-failure")
    let failed = AudioResourcePreparationService(rootURL: failedRoot, httpClient: OfflineAudioHTTPClient())
    await #expect(throws: (any Error).self) { try await failed.prepare(snapshot: snapshot) }
    #expect(!FileManager.default.fileExists(atPath: failed.directory(for: try AudioResourceProfile()).path))
    let cancelled = AudioResourcePreparationService(rootURL: root.appendingPathComponent("cancelled"),
                                                     httpClient: CancelledAudioHTTPClient())
    await #expect(throws: CancellationError.self) { try await cancelled.prepare(snapshot: snapshot) }
    let cancelledDestination = cancelled.directory(for: try AudioResourceProfile())
    #expect(!FileManager.default.fileExists(atPath: cancelledDestination.path))
    #expect(try FileManager.default.contentsOfDirectory(atPath: cancelledDestination.deletingLastPathComponent().path)
        .allSatisfy { !$0.hasPrefix(".prepare-") })
    #expect(try service.readyConfiguration() == config)
    print("Verified native audio resources at \(root.path)")
}

private struct OfflineAudioHTTPClient: ModelDownloadHTTPClient {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) { throw URLError(.notConnectedToInternet) }
    func stream(for request: URLRequest,
                onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
                onData: @escaping @Sendable (Data) async throws -> Void) async throws {
        throw URLError(.notConnectedToInternet)
    }
}

private struct CancelledAudioHTTPClient: ModelDownloadHTTPClient {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) { throw CancellationError() }
    func stream(for request: URLRequest,
                onResponse: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
                onData: @escaping @Sendable (Data) async throws -> Void) async throws {
        throw CancellationError()
    }
}
