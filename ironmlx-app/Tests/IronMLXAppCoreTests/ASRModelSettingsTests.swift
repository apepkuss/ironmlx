import Foundation
import Testing
@testable import IronMLXAppCore

// Existing model-loading doubles do not implement the ASR control endpoint.
// Fail if a test unexpectedly tries to use it; production uses BackendAPIClient.
extension BackendModelLoading {
    func updateASRSettings(model: String, settings: BackendASRSettings) async throws {
        throw BackendAPIError.serverResponse(statusCode: 501, body: "ASR settings not implemented by this test double")
    }
}

@Test func asrSettingsPersistIndependentlyAndRejectInvalidValues() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    let small = "mlx-community/Qwen3-ASR-0.6B-8bit"
    let large = "mlx-community/Qwen3-ASR-1.7B-8bit"
    let custom = BackendASRSettings(maxAudioDurationSeconds: 1800, maxOutputTokens: 2048)
    try store.save(ModelParameters(modelID: small, asr: .qwen3Defaults))
    try store.save(ModelParameters(modelID: large, asr: custom))
    let reloaded = ModelParameterStore(url: url)
    #expect(reloaded.parameters(for: small)?.asr == .qwen3Defaults)
    #expect(reloaded.parameters(for: large)?.asr == custom)
    for settings in [
        BackendASRSettings(maxAudioDurationSeconds: 0, maxOutputTokens: 1024),
        BackendASRSettings(maxAudioDurationSeconds: 7201, maxOutputTokens: 1024),
        BackendASRSettings(maxAudioDurationSeconds: 1200, maxOutputTokens: 0),
        BackendASRSettings(maxAudioDurationSeconds: 1200, maxOutputTokens: 8193)
    ] {
        #expect(throws: ConfigurationPersistenceError.self) {
            try store.save(ModelParameters(modelID: small, asr: settings))
        }
        #expect(store.parameters(for: small)?.asr == .qwen3Defaults)
    }
    try store.save(ModelParameters(modelID: small))
    #expect(store.parameters(for: small)?.asr == nil)
    #expect(store.parameters(for: large)?.asr == custom)
}

@Test @MainActor func asrRegistrationRestoresPerModelSettings() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let model = "mlx-community/Qwen3-ASR-0.6B-8bit"
    let scanner = LocalModelScanner(rootURL: root)
    let store = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let settings = BackendASRSettings(maxAudioDurationSeconds: 2400, maxOutputTokens: 2048)
    try store.save(ModelParameters(modelID: model, asr: settings))
    // Reuse a verified snapshot produced by the ASR support test fixtures.
    let snapshot = try writeVerifiedTestSnapshot(root: root, repoID: model, files: [
        "config.json": try JSONSerialization.data(withJSONObject: asrConfig(large: false)),
        "tokenizer_config.json": Data("{}".utf8), "preprocessor_config.json": Data("{}".utf8),
        "vocab.json": Data("{}".utf8), "merges.txt": Data("#version: 0.2".utf8),
        "model.safetensors": Data("weights".utf8)
    ])
    var manifest = try ModelSnapshotVerifier().loadManifest(at: snapshot)
    manifest.compatibility.modelType = "qwen3_asr"
    manifest.compatibility.artifactRole = "asr"
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
    let models = scanner.scan(loadedModels: [], pinnedModels: [], mtpEnabledModels: [])
    #expect(models.first?.asrDefaults == .qwen3Defaults)
    let client = ASRSettingsLoader()
    let failures = await LocalModelBackendRegistrar.register(localModels: models, defaultModel: nil,
        scanner: scanner, parameterStore: store, activeKvOffloadEnabled: false, client: client)
    #expect(failures.isEmpty)
    #expect(await client.applied[model] == settings)
}

private actor ASRSettingsLoader: BackendModelLoading {
    var applied: [String: BackendASRSettings] = [:]
    func updateASRSettings(model: String, settings: BackendASRSettings) async throws { applied[model] = settings }
    func waitUntilReady(timeout: TimeInterval) async throws {}
    func registerModel(model: String, modelDir: String, setDefault: Bool, maxCacheCap: Int?, pinned: Bool?,
        mtpModelDir: String?, mtpDraftTokens: Int?, promptLookup: BackendPromptLookupConfig?,
        samplingDefaults: BackendSamplingDefaults, audio: BackendAudioResources?, decision: BackendDecisionSettings?) async throws -> BackendModelAdminResponse {
        BackendModelAdminResponse(success: true, status: "registered", model: model, loadedModels: [])
    }
    func loadModel(model: String, modelDir: String, setDefault: Bool, maxCacheCap: Int?, pinned: Bool,
        mtpModelDir: String?, mtpDraftTokens: Int?, promptLookup: BackendPromptLookupConfig?, reloadWhenIdle: Bool,
        samplingDefaults: BackendSamplingDefaults, audio: BackendAudioResources?, decision: BackendDecisionSettings?) async throws -> BackendModelAdminResponse {
        throw BackendAPIError.serverResponse(statusCode: 501, body: "No weight loading in this settings test")
    }
}
