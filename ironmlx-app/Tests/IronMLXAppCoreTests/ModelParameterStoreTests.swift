import Foundation
import Testing

@testable import IronMLXAppCore

@Test func modelParameterStorePersistsMaxTokensByModelID() throws {
    let root = try temporaryDirectory()
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    let params = ModelParameters(
        modelID: "mlx-community/LongContext-4bit",
        alias: "Long Context",
        modelType: "llm",
        contextSize: "262144",
        maxTokens: "65536",
        temperature: "0.7",
        topP: "0.95",
        topK: "40",
        repeatPenalty: "1.05"
    )

    try store.save(params)

    let loaded = try ModelParameterStore(url: url).loadAll()
    #expect(loaded["mlx-community/LongContext-4bit"]?.maxTokens == "65536")
    #expect(loaded["mlx-community/LongContext-4bit"]?.maxCacheCap == 65536)
}

@Test func modelParameterStoreIgnoresNonPositiveMaxTokensForBackendCap() throws {
    let params = ModelParameters(
        modelID: "mlx-community/Tiny-4bit",
        maxTokens: "0"
    )

    #expect(params.maxCacheCap == nil)
}

@Test func modelParametersExposeSamplingDefaultsForBackendLoad() throws {
    let params = ModelParameters(
        modelID: "mlx-community/Tiny-4bit",
        temperature: "0.7",
        topP: "0.8",
        topK: "40",
        repeatPenalty: "1.1"
    )

    #expect(params.samplingDefaults.temperature == 0.7)
    #expect(params.samplingDefaults.topP == 0.8)
    #expect(params.samplingDefaults.topK == 40)
    #expect(params.samplingDefaults.repetitionPenalty == 1.1)
}

@Test func modelParameterStorePersistsMtpRuntimePreference() throws {
    let root = try temporaryDirectory()
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    let params = ModelParameters(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        mtpEnabled: true,
        mtpModelID: "mlx-community/Qwen3.5-4B-MTP-4bit",
        mtpDraftTokens: "2"
    )

    try store.save(params)

    let loaded = try ModelParameterStore(url: url).loadAll()
    #expect(loaded["mlx-community/Qwen3.5-4B-MLX-4bit"]?.mtpEnabled == true)
    #expect(loaded["mlx-community/Qwen3.5-4B-MLX-4bit"]?.mtpModelID == "mlx-community/Qwen3.5-4B-MTP-4bit")
    #expect(loaded["mlx-community/Qwen3.5-4B-MLX-4bit"]?.mtpDraftTokensValue == 2)
}

@Test func modelParameterStoreRecordsMtpLoadPreferenceAndPreservesExistingParameters() throws {
    let root = try temporaryDirectory()
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    try store.save(ModelParameters(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        maxTokens: "32768",
        temperature: "0.7",
        mtpDraftTokens: "3"
    ))

    try store.recordMtpLoadPreference(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        enabled: true,
        mtpModelID: "mlx-community/Qwen3.5-4B-MTP-4bit"
    )

    let loaded = try ModelParameterStore(url: url).loadAll()
    let params = try #require(loaded["mlx-community/Qwen3.5-4B-MLX-4bit"])
    #expect(params.maxTokens == "32768")
    #expect(params.temperature == "0.7")
    #expect(params.mtpEnabled == true)
    #expect(params.mtpModelID == "mlx-community/Qwen3.5-4B-MTP-4bit")
    #expect(params.mtpDraftTokens == "3")
}

@Test func modelParameterStoreRecordsModelOnlyLoadPreferenceWithoutClearingMtpSelection() throws {
    let root = try temporaryDirectory()
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    try store.save(ModelParameters(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        mtpEnabled: true,
        mtpModelID: "mlx-community/Qwen3.5-4B-MTP-4bit"
    ))

    try store.recordMtpLoadPreference(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        enabled: false,
        mtpModelID: nil
    )

    let loaded = try ModelParameterStore(url: url).loadAll()
    let params = try #require(loaded["mlx-community/Qwen3.5-4B-MLX-4bit"])
    #expect(params.mtpEnabled == false)
    #expect(params.mtpModelID == "mlx-community/Qwen3.5-4B-MTP-4bit")
}

@Test func localModelScannerReadsGenerationConfigSamplingDefaults() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--SamplerDefaults-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(#"{"max_position_embeddings":32768}"#.utf8)
        .write(to: snapshot.appendingPathComponent("config.json"))
    try Data(#"{"temperature":0.65,"top_p":0.9,"top_k":32,"repetition_penalty":1.08}"#.utf8)
        .write(to: snapshot.appendingPathComponent("generation_config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.generationDefaults?.temperature == 0.65)
    #expect(model.generationDefaults?.topP == 0.9)
    #expect(model.generationDefaults?.topK == 32)
    #expect(model.generationDefaults?.repetitionPenalty == 1.08)
}

@Test func modelLoadParametersPreferSavedMaxTokensOverModelContextWindow() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--LongContext-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(#"{"max_position_embeddings":262144}"#.utf8)
        .write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let store = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    try store.save(ModelParameters(modelID: "mlx-community/LongContext-4bit", maxTokens: "65536"))

    let maxCacheCap = ModelLoadParameters.maxCacheCap(
        for: "mlx-community/LongContext-4bit",
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: store,
        activeKvOffloadEnabled: false
    )

    #expect(maxCacheCap == 65536)
}

@Test func modelLoadParametersUseSafeDefaultForLongContextWithoutActiveKvOffload() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--LongContext-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(#"{"max_position_embeddings":262144}"#.utf8)
        .write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let maxCacheCap = ModelLoadParameters.maxCacheCap(
        for: "mlx-community/LongContext-4bit",
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        activeKvOffloadEnabled: false
    )

    #expect(maxCacheCap == 32768)
}

@Test func modelLoadParametersAllowFullLongContextWithActiveKvOffload() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--LongContext-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data(#"{"max_position_embeddings":262144}"#.utf8)
        .write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let maxCacheCap = ModelLoadParameters.maxCacheCap(
        for: "mlx-community/LongContext-4bit",
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        activeKvOffloadEnabled: true
    )

    #expect(maxCacheCap == 262144)
}
