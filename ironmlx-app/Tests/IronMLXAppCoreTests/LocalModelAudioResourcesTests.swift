import Foundation
import Testing

@testable import IronMLXAppCore

/// Counts the full scans a scanner performs.
final class FullScanCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
    func reset() { lock.withLock { count = 0 } }
}

private enum AudioOutcome: Equatable {
    case value(BackendAudioResources?)
    case error(String)

    init(_ body: () throws -> BackendAudioResources?) {
        do {
            self = .value(try body())
        } catch {
            self = .error(String(describing: error))
        }
    }
}

/// `audioResources(for:execution:)` as it was written before it stopped
/// scanning every model: the reference for the comparison below.
private func fullScanAudioResources(
    _ scanner: LocalModelScanner,
    for reference: String,
    execution: BackendAudioExecutionSettings?
) throws -> BackendAudioResources? {
    guard let model = scanner.scan(loadedModels: []).first(where: { $0.id == reference }),
          model.type == "tts" else { return nil }
    guard model.readiness?.isLoadable == true else {
        throw AudioResourceError.invalid(model.readiness?.message ?? "model is not ready")
    }
    var resources = try AudioResourcePreparationService(rootURL: scanner.rootURL).readyConfiguration()
    resources.execution = execution
    return resources
}

/// An IndexTTS 2.5 snapshot whose manifest matches the bundled profile. The
/// weights are sparse files of the profile's sizes.
func writeIndexTTSSnapshot(root: URL, repoID: String, matchingProfile: Bool) throws {
    let profile = try AudioResourceProfile()
    let snapshot = try writeVerifiedTestSnapshot(
        root: root,
        repoID: repoID,
        files: ["config.json": Data("{}".utf8), "model.safetensors": Data("weights".utf8)]
    )
    let fileManager = FileManager.default
    var entries: [ModelSnapshotFile] = []
    if matchingProfile {
        try fileManager.removeItem(at: snapshot.appendingPathComponent("config.json"))
        try fileManager.removeItem(at: snapshot.appendingPathComponent("model.safetensors"))
        for file in profile.source.files {
            let url = try ModelSnapshotVerifier.safeFileURL(path: file.path, beneath: snapshot)
            try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if file.path == "config.json" {
                var json = Data("{}".utf8)
                json.append(Data(repeating: UInt8(ascii: " "), count: Int(file.bytes) - json.count))
                try json.write(to: url)
            } else {
                fileManager.createFile(atPath: url.path, contents: nil)
                let handle = try FileHandle(forWritingTo: url)
                try handle.truncate(atOffset: UInt64(file.bytes))
                try handle.close()
            }
            entries.append(ModelSnapshotFile(path: file.path, size: file.bytes, sha256: file.sha256))
        }
    }
    let verifier = ModelSnapshotVerifier()
    var manifest = try verifier.loadManifest(at: snapshot)
    if matchingProfile {
        manifest.files = entries
    }
    manifest.compatibility = ModelSnapshotCompatibility(
        modelType: "indextts2_5",
        artifactRole: "tts",
        quantizationMode: nil,
        quantizationBits: nil,
        quantizationGroupSize: nil
    )
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
}

private struct AudioReceipt: Codable {
    var configuration: BackendAudioResources
    var files: [String: ModelSnapshotFileIdentity]
}

/// Prepared speech resources as `readyConfiguration()` expects them.
@discardableResult
func writePreparedAudioResources(root: URL) throws -> URL {
    let profile = try AudioResourceProfile()
    let service = AudioResourcePreparationService(rootURL: root)
    let directory = service.directory(for: profile)
    let paths = (profile.outputs + profile.files).map { "derived/" + $0.path }
        + profile.text.flatMap { $0.members.map { "text/" + $0.path } }
        + ["derived/manifest.json", "sources.json"]
    var identities: [String: ModelSnapshotFileIdentity] = [:]
    for path in paths {
        let url = try AudioResourceProfile.path(path, in: directory)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(path.utf8).write(to: url)
        identities[path] = try ModelSnapshotVerifier.fileIdentity(of: url)
    }
    let receipt = directory.appendingPathComponent("configuration.json")
    try JSONEncoder().encode(AudioReceipt(configuration: service.configuration(at: directory), files: identities))
        .write(to: receipt)
    return receipt
}

@Test func audioResourcesMatchAFullScanForEveryModelKindWithoutScanning() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-audio-resources-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let configs: [(String, String)] = [
        ("org/llm", #"{"model_type":"llama"}"#),
        ("org/embedding", #"{"model_type":"bert","architectures":["BertModel"]}"#),
        ("org/asr", #"{"model_type":"qwen3_asr","architectures":["Qwen3ASRForConditionalGeneration"]}"#),
        ("org/aligner", #"{"model_type":"qwen3_asr","thinker_config":{"model_type":"qwen3_forced_aligner"}}"#),
        ("org/diarization", #"{"model_type":"nemotron_diarization","num_speakers":8}"#),
        ("org/mtp-draft", #"{"model_type":"qwen3_5_mtp","text_config":{"mtp_num_hidden_layers":1}}"#),
        ("org/kokoro-tts", #"{"model_type":"kokoro","architectures":["KokoroTTS"]}"#),
    ]
    for (repoID, config) in configs {
        try writeVerifiedTestSnapshot(
            root: root,
            repoID: repoID,
            files: [
                "config.json": Data(config.utf8),
                "model.safetensors": Data("weights".utf8),
                "tokenizer.json": Data("{}".utf8),
                "tokenizer_config.json": Data("{}".utf8),
                "preprocessor_config.json": Data("{}".utf8),
            ]
        )
    }
    // The same repository under a second provider, a standalone model, and
    // a DFlash2 target with its draft.
    try writeVerifiedTestSnapshot(
        root: root,
        provider: .modelScope,
        repoID: "org/llm",
        files: ["config.json": Data(#"{"model_type":"llama"}"#.utf8), "model.safetensors": Data("w".utf8)]
    )
    try writeVerifiedTestSnapshot(
        root: root,
        provider: .standalone,
        repoID: "standalone/local-llm",
        files: ["config.json": Data(#"{"model_type":"llama"}"#.utf8), "model.safetensors": Data("w".utf8)]
    )
    for (repoID, config) in [("org/dflash2-target", dflash2Target), ("org/dflash2-draft", dflash2Draft)] {
        try writeVerifiedTestSnapshot(
            root: root,
            repoID: repoID,
            files: ["config.json": Data(config.utf8), "model.safetensors": Data("weights".utf8)]
        )
    }
    try writeIndexTTSSnapshot(root: root, repoID: "org/indextts-ready", matchingProfile: true)
    try writeIndexTTSSnapshot(root: root, repoID: "org/indextts-other-profile", matchingProfile: false)
    let receipt = try writePreparedAudioResources(root: root)

    let counter = FullScanCounter()
    var scanner = LocalModelScanner(rootURL: root)
    let reference = scanner
    scanner.onFullScan = { counter.increment() }
    let ids = configs.map(\.0) + [
        "standalone/local-llm", "org/dflash2-target", "org/dflash2-draft",
        "org/indextts-ready", "org/indextts-other-profile", "org/missing",
    ]
    // The fixture holds what the comparison claims to cover.
    let scanned = Dictionary(
        reference.scan(loadedModels: []).map { ($0.id, $0) },
        uniquingKeysWith: { first, _ in first }
    )
    let standaloneSource = scanned["standalone/local-llm"]?.source
    let dflash2Drafts = scanned["org/dflash2-target"]?.dflash2?.candidates.map(\.id)
    let ttsLoadable = scanned["org/indextts-ready"]?.readiness?.isLoadable
    #expect(standaloneSource == "standalone")
    #expect(dflash2Drafts == ["org/dflash2-draft"])
    #expect(ttsLoadable == true)
    let executions: [BackendAudioExecutionSettings?] = [nil, BackendAudioExecutionSettings(queueTimeoutMS: 1500)]

    var outcomes: [AudioOutcome] = []
    for phase in ["prepared", "resources removed"] {
        if phase == "resources removed" {
            try FileManager.default.removeItem(at: receipt)
        }
        for id in ids {
            for execution in executions {
                let expected = AudioOutcome { try fullScanAudioResources(reference, for: id, execution: execution) }
                counter.reset()
                let actual = AudioOutcome { try scanner.audioResources(for: id, execution: execution) }
                #expect(actual == expected, "\(phase) \(id)")
                #expect(counter.value == 0, "\(phase) \(id) scanned every model")
                outcomes.append(actual)
            }
        }
    }
    // The comparison covers a ready TTS model, an unready one and the nil case.
    #expect(outcomes.contains { if case .value(let value) = $0 { value?.execution?.queueTimeoutMS == 1500 } else { false } })
    #expect(outcomes.contains { if case .error = $0 { true } else { false } })
    #expect(outcomes.contains(.value(nil)))
}

private let dflash2Target = """
{"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
 "text_config": {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64, "vocab_size": 248320,
  "max_position_embeddings": 262144, "rms_norm_eps": 0.000001,
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000000}, "tie_word_embeddings": false}}
"""

private let dflash2Draft = """
{"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "dtype": "bfloat16", "hidden_act": "silu",
 "attention_bias": false, "is_causal": false, "hidden_size": 5120, "intermediate_size": 17408,
 "vocab_size": 248320, "max_position_embeddings": 262144, "head_dim": 128, "num_attention_heads": 32,
 "num_hidden_layers": 5, "num_key_value_heads": 8, "num_target_layers": 64, "rms_norm_eps": 0.000001,
 "rope_parameters": {"rope_type": "default", "rope_theta": 10000000}, "sliding_window": 2048,
 "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
  "sliding_attention"],
 "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248070,
  "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61]}}
"""

private actor AudioRecordingClient: BackendModelLoading {
    private(set) var audio: [String: BackendAudioResources?] = [:]

    func updateASRSettings(model: String, settings: BackendASRSettings) async throws {}
    func waitUntilReady(timeout: TimeInterval) async throws {}
    func registerModel(
        model: String, modelDir: String, setDefault: Bool, maxCacheCap: Int?, pinned: Bool?,
        mtpModelDir: String?, mtpDraftTokens: Int?, promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults, audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        self.audio[model] = audio
        return BackendModelAdminResponse(
            success: true, status: "registered", code: nil, model: model,
            loadedModels: [], warningCode: nil, warning: nil, error: nil
        )
    }
    func loadModel(
        model: String, modelDir: String, setDefault: Bool, maxCacheCap: Int?, pinned: Bool,
        mtpModelDir: String?, mtpDraftTokens: Int?, promptLookup: BackendPromptLookupConfig?,
        reloadWhenIdle: Bool, samplingDefaults: BackendSamplingDefaults, audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        throw CancellationError()
    }
}

@Test func onlyTTSRegistrationsCheckTheSpeechResources() async throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-audio-registration-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for repoID in ["org/llm-a", "org/llm-b"] {
        try writeVerifiedTestSnapshot(
            root: root,
            repoID: repoID,
            files: ["config.json": Data(#"{"model_type":"llama"}"#.utf8), "model.safetensors": Data("w".utf8)]
        )
    }
    try writeIndexTTSSnapshot(root: root, repoID: "org/indextts", matchingProfile: true)
    try writePreparedAudioResources(root: root)
    let checks = FullScanCounter()
    var scanner = LocalModelScanner(rootURL: root)
    scanner.onAudioResourceCheck = { checks.increment() }
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let client = AudioRecordingClient()

    let failures = await LocalModelBackendRegistrar.register(
        localModels: scanner.scan(loadedModels: []),
        defaultModel: nil,
        scanner: scanner,
        parameterStore: parameterStore,
        activeKvOffloadEnabled: false,
        client: client,
        ledger: LocalModelRegistrationLedger(),
        launchID: UUID(),
        pass: .restore
    )

    #expect(failures.isEmpty)
    // One check, for the TTS model; it sends what the check returns.
    #expect(checks.value == 1)
    let sent = await client.audio
    #expect(Set(sent.keys) == ["org/llm-a", "org/llm-b", "org/indextts"])
    #expect(sent["org/llm-a"] == .some(nil))
    #expect(sent["org/indextts"] == .some(try scanner.audioResources(for: "org/indextts")))
    #expect(sent["org/indextts"]??.derivedResources.isEmpty == false)
}
