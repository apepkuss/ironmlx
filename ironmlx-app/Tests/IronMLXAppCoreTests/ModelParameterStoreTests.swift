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

@Test func modelParameterStorePersistsValidatedDFlash2Configuration() throws {
    let root = try temporaryDirectory()
    let store = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let parameters = ModelParameters(
        modelID: "mlx-community/Qwen3.8-27B-4bit",
        dflash2Enabled: true,
        dflash2ModelID: "z-lab/Qwen3.8-27B-DFlash2",
        dflash2BlockSize: "4",
        dflash2DraftBits: "8",
        dflash2TensorBatchMaxWidth: "6"
    )

    try store.save(parameters)

    let loaded = try #require(store.parameters(for: parameters.modelID))
    #expect(loaded.dflash2Enabled == true)
    #expect(loaded.dflash2BlockSizeValue == 4)
    #expect(loaded.dflash2DraftBitsValue == 8)
    #expect(loaded.dflash2TensorBatchMaxWidthValue == 6)
}

@Test func modelParameterStoreRejectsInvalidDFlash2TensorBatchWidth() throws {
    let root = try temporaryDirectory()
    let store = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let parameters = ModelParameters(
        modelID: "mlx-community/Qwen3.8-27B-4bit",
        dflash2TensorBatchMaxWidth: "0"
    )

    #expect(
        throws: ConfigurationPersistenceError.invalidValue(
            "dflash2_tensor_batch_max_width"
        )
    ) {
        try store.save(parameters)
    }
}

@Test func modelParameterStoreRejectsConflictingDFlash2Acceleration() throws {
    let root = try temporaryDirectory()
    let store = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let parameters = ModelParameters(
        modelID: "mlx-community/Qwen3.8-27B-4bit",
        mtpEnabled: true,
        dflash2Enabled: true,
        dflash2ModelID: "z-lab/Qwen3.8-27B-DFlash2"
    )

    #expect(throws: ConfigurationPersistenceError.invalidValue("dflash2_acceleration_conflict")) {
        try store.save(parameters)
    }
}

@Test func modelParameterV0MigratesAllFieldsAndCreatesIndependentLKG() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let expected = ModelParameters(
        modelID: "mlx-community/Full-4bit", alias: "Full", modelType: "llm",
        contextSize: "131072", maxTokens: "32768", temperature: "0.7", topP: "0.95",
        topK: "40", repeatPenalty: "1.05", mtpEnabled: true,
        mtpModelID: "mlx-community/Full-MTP-4bit", mtpDraftTokens: "3",
        promptLookupEnabled: true, promptLookupCrossRequest: true
    )
    let v0 = try JSONEncoder().encode([expected.modelID: expected])
    try v0.write(to: url)

    let store = ModelParameterStore(url: url)
    let migrated = try store.loadAll()

    #expect(migrated == [expected.modelID: expected])
    let active = try #require(
        JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
    )
    #expect(active["schema_version"] as? Int == 1)
    #expect((active["models"] as? [String: Any])?.keys.contains(expected.modelID) == true)
    let layout = ConfigurationFileLayout(activeURL: url)
    #expect(try Data(contentsOf: layout.lkgURL) == Data(contentsOf: url))
    let evidence = try FileManager.default.contentsOfDirectory(
        at: layout.recoveryDirectoryURL,
        includingPropertiesForKeys: nil
    ).filter { $0.lastPathComponent.contains("pre-migration-v0-") }
    #expect(evidence.count == 1)
    #expect(try Data(contentsOf: evidence[0]) == v0)
}

@Test func modelParameterV0AllowsModelIDNamedSchemaVersion() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let parameters = ModelParameters(modelID: "schema_version", maxTokens: "4096")
    try JSONEncoder().encode(["schema_version": parameters]).write(to: url)

    let loaded = try ModelParameterStore(url: url).loadAll()

    #expect(loaded["schema_version"] == parameters)
}

@Test func modelParameterV1DoesNotMigrateAgain() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    try store.save(ModelParameters(modelID: "stable", maxTokens: "2048"))
    let before = try Data(contentsOf: url)

    #expect(try store.loadAll()["stable"]?.maxTokens == "2048")

    #expect(try Data(contentsOf: url) == before)
    let files = try FileManager.default.contentsOfDirectory(
        at: ConfigurationFileLayout(activeURL: url).recoveryDirectoryURL,
        includingPropertiesForKeys: nil
    )
    #expect(!files.contains { $0.lastPathComponent.contains("pre-migration") })
}

@Test func modelParameterFutureVersionIsRejectedWithoutCorruptionCopy() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let future = Data(#"{"schema_version":7,"models":{}}"#.utf8)
    try future.write(to: url)
    let store = ModelParameterStore(url: url)

    #expect(throws: ConfigurationPersistenceError.unsupportedSchemaVersion(found: 7, supported: 1)) {
        _ = try store.loadAll()
    }
    let issue = try #require(store.recoveryIssue)
    #expect(issue.reason == .unsupportedVersion(found: 7, supported: 1))
    #expect(issue.preservedURL == nil)
    #expect(try Data(contentsOf: url) == future)
    #expect(throws: ConfigurationRecoveryWriteError.unresolvedIssue(url, "configuration_version_unsupported")) {
        try store.save(ModelParameters(modelID: "new"))
    }
}

@Test func modelParameterCorruptionRestoresExplicitlyFromLKG() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let expected = ModelParameters(modelID: "mlx-community/Restore-4bit", maxTokens: "8192")
    try ModelParameterStore(url: url).save(expected)
    try Data("broken".utf8).write(to: url)
    let store = ModelParameterStore(url: url)
    #expect(throws: (any Error).self) { _ = try store.loadAll() }
    #expect(try #require(store.recoveryIssue).hasValidLKG)

    try store.restoreFromLKG()

    #expect(try store.loadAll()[expected.modelID] == expected)
}

@Test func concurrentModelParameterSavesPreserveEveryModelAndValidEnvelope() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    let errors = ConcurrentConfigurationErrorBox()

    DispatchQueue.concurrentPerform(iterations: 24) { index in
        do {
            try store.save(ModelParameters(modelID: "model-\(index)", maxTokens: "\(index + 1)"))
        } catch {
            errors.append(error)
        }
    }

    #expect(errors.isEmpty)
    #expect(try store.loadAll().count == 24)
    let object = try #require(
        JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
    )
    #expect(object["schema_version"] as? Int == 1)
    #expect((object["models"] as? [String: Any])?.count == 24)
}

@Test func modelParameterStorePreservesCorruptFileAndRequiresExplicitReset() throws {
    let root = try temporaryDirectory()
    let url = root.appendingPathComponent("model_params.json")
    let corruptData = Data(#"{"mlx-community/Broken": }"#.utf8)
    try corruptData.write(to: url)
    let store = ModelParameterStore(url: url)

    #expect(throws: (any Error).self) {
        _ = try store.loadAll()
    }
    let issue = try #require(store.recoveryIssue)
    let preservedURL = try #require(issue.preservedURL)

    #expect(issue.kind == .modelParameters)
    #expect(try Data(contentsOf: url) == corruptData)
    #expect(try Data(contentsOf: preservedURL) == corruptData)
    #expect(store.parameters(for: "mlx-community/Broken") == nil)
    #expect(store.jsonString() == "{}")
    #expect(throws: ConfigurationRecoveryWriteError.unresolvedIssue(url, "configuration_recovery_required")) {
        try store.save(ModelParameters(modelID: "mlx-community/New"))
    }
    #expect(try Data(contentsOf: url) == corruptData)

    try store.resetAfterCorruption()

    #expect(store.recoveryIssue == nil)
    #expect(try store.loadAll().isEmpty)
    #expect(try Data(contentsOf: preservedURL) == corruptData)
}

@Test @MainActor
func configurationRecoveryManagerInspectsAndResetsBothStores() throws {
    let root = try temporaryDirectory()
    let configURL = root.appendingPathComponent("app_config.json")
    let parametersURL = root.appendingPathComponent("model_params.json")
    try Data("bad-app-config".utf8).write(to: configURL)
    try Data("bad-model-parameters".utf8).write(to: parametersURL)
    let configStore = AppConfigStore(url: configURL)
    let parameterStore = ModelParameterStore(url: parametersURL)
    let manager = ConfigurationRecoveryManager(
        appConfigStore: configStore,
        modelParameterStore: parameterStore
    )

    manager.inspect()

    #expect(manager.hasIssues)
    #expect(Set(manager.issues.map(\.kind)) == [.appConfig, .modelParameters])

    try manager.resetAffectedConfigurations()

    #expect(!manager.hasIssues)
    #expect(configStore.recoveryIssue == nil)
    #expect(parameterStore.recoveryIssue == nil)
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

@Test func modelParameterStorePersistsCrossRequestPromptLookupPreference() throws {
    let root = try temporaryDirectory()
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    let params = ModelParameters(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        promptLookupEnabled: true,
        promptLookupCrossRequest: true
    )

    try store.save(params)

    let loaded = try #require(
        ModelParameterStore(url: url)
            .loadAll()["mlx-community/Qwen3.5-4B-MLX-4bit"]
    )
    #expect(loaded.promptLookupEnabled == true)
    #expect(loaded.promptLookupCrossRequest == true)
    #expect(loaded.promptLookupConfig == .crossRequest)
}

@Test func disabledPromptLookupDoesNotCreateBackendConfig() {
    let params = ModelParameters(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        promptLookupEnabled: false,
        promptLookupCrossRequest: true
    )

    #expect(params.promptLookupConfig == nil)
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
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: "mlx-community/SamplerDefaults-4bit",
        files: [
            "config.json": Data(#"{"max_position_embeddings":32768}"#.utf8),
            "generation_config.json": Data(
                #"{"temperature":0.65,"top_p":0.9,"top_k":32,"repetition_penalty":1.08}"#.utf8
            ),
            "model.safetensors": Data("weights".utf8),
        ]
    )

    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(model.generationDefaults?.temperature == 0.65)
    #expect(model.generationDefaults?.topP == 0.9)
    #expect(model.generationDefaults?.topK == 32)
    #expect(model.generationDefaults?.repetitionPenalty == 1.08)
}

@Test func modelLoadParametersPreferSavedMaxTokensOverModelContextWindow() throws {
    let root = try temporaryDirectory()
    _ = try writeLongContextSnapshot(root: root)

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
    _ = try writeLongContextSnapshot(root: root)

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
    _ = try writeLongContextSnapshot(root: root)

    let maxCacheCap = ModelLoadParameters.maxCacheCap(
        for: "mlx-community/LongContext-4bit",
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json")),
        activeKvOffloadEnabled: true
    )

    #expect(maxCacheCap == 262144)
}

private func writeLongContextSnapshot(root: URL) throws -> URL {
    try writeVerifiedTestSnapshot(
        root: root,
        repoID: "mlx-community/LongContext-4bit",
        files: [
            "config.json": Data(#"{"max_position_embeddings":262144}"#.utf8),
            "model.safetensors": Data("weights".utf8),
        ]
    )
}

private final class ConcurrentConfigurationErrorBox: @unchecked Sendable {
    private let lock = NSLock()
    private var errors: [Error] = []

    var isEmpty: Bool {
        lock.withLock { errors.isEmpty }
    }

    func append(_ error: Error) {
        lock.withLock { errors.append(error) }
    }
}
