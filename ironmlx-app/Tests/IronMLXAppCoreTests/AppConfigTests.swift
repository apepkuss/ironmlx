import Foundation
import Testing

@testable import IronMLXAppCore

@Test(arguments: [
    (["en-US"], "en"),
    (["zh-CN"], "zh-Hans"),
    (["zh-Hans-SG"], "zh-Hans"),
    (["zh-TW"], "zh-Hant"),
    (["zh-Hant-HK"], "zh-Hant"),
    (["zh-MO"], "zh-Hant"),
    (["ja-JP"], "ja"),
    (["ko-KR"], "ko"),
    (["fr-FR"], "en"),
    (["fr-FR", "ja-JP"], "en"),
])
func appLanguageResolverMatchesSupportedMacOSPreferences(
    preferredLanguages: [String],
    expected: String
) {
    #expect(AppLanguageResolver.resolve(preferredLanguages: preferredLanguages) == expected)
}

@Test func appConfigStorePersistsSystemLanguageOnlyWhenConfigIsMissing() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-language-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let configURL = root.appendingPathComponent("app_config.json")
    let store = AppConfigStore(
        url: configURL,
        preferredLanguages: { ["zh-Hant-TW"] }
    )

    let initial = store.load()

    #expect(initial.language == "zh-Hant")
    #expect(FileManager.default.fileExists(atPath: configURL.path))
    #expect(store.load().language == "zh-Hant")

    store.save(AppConfig(language: "ko"))
    let existingConfigStore = AppConfigStore(
        url: configURL,
        preferredLanguages: { ["ja-JP"] }
    )

    #expect(existingConfigStore.load().language == "ko")
}

@Test func appConfigStorePreservesCorruptFileAndBlocksImplicitOverwrite() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-config-corrupt-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let configURL = root.appendingPathComponent("app_config.json")
    let corruptData = Data(#"{"host": "127.0.0.1", "port": }"#.utf8)
    try corruptData.write(to: configURL)
    let store = AppConfigStore(
        url: configURL,
        preferredLanguages: { ["ja-JP"] }
    )

    let fallback = store.load()
    let issue = try #require(store.recoveryIssue)

    #expect(fallback.language == "ja")
    #expect(issue.kind == .appConfig)
    #expect(issue.sourceURL == configURL)
    #expect(try Data(contentsOf: configURL) == corruptData)
    let preservedURL = try #require(issue.preservedURL)
    #expect(try Data(contentsOf: preservedURL) == corruptData)

    let didSave = store.save(AppConfig(language: "ko"))

    #expect(!didSave)
    #expect(try Data(contentsOf: configURL) == corruptData)
    #expect(store.recoveryIssue != nil)
}

@Test func appConfigStoreRequiresExplicitResetAndRetainsPreservedCopy() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-config-reset-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let configURL = root.appendingPathComponent("app_config.json")
    let corruptData = Data("not-json".utf8)
    try corruptData.write(to: configURL)
    let store = AppConfigStore(
        url: configURL,
        preferredLanguages: { ["zh-Hant-TW"] }
    )

    _ = store.load()
    let preservedURL = try #require(store.recoveryIssue?.preservedURL)
    try store.resetAfterCorruption()

    #expect(store.recoveryIssue == nil)
    #expect(store.load().language == "zh-Hant")
    #expect(try Data(contentsOf: preservedURL) == corruptData)
}

@Test func appConfigStoreRefusesResetWhenPreservedCopyIsMissing() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-config-missing-backup-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let configURL = root.appendingPathComponent("app_config.json")
    let corruptData = Data("not-json".utf8)
    try corruptData.write(to: configURL)
    let store = AppConfigStore(url: configURL)

    _ = store.load()
    let preservedURL = try #require(store.recoveryIssue?.preservedURL)
    try FileManager.default.removeItem(at: preservedURL)

    #expect(throws: ConfigurationRecoveryResetError.preservedCopyMissing(configURL)) {
        try store.resetAfterCorruption()
    }
    #expect(try Data(contentsOf: configURL) == corruptData)
    #expect(store.recoveryIssue != nil)
}

@Test func appConfigStoreCreatesOnePreservedCopyUnderConcurrentDetection() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-config-concurrent-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let configURL = root.appendingPathComponent("app_config.json")
    let corruptData = Data("not-json".utf8)
    try corruptData.write(to: configURL)
    let store = AppConfigStore(url: configURL)

    DispatchQueue.concurrentPerform(iterations: 32) { _ in
        _ = store.load()
    }

    let recoveryDirectory = root.appendingPathComponent("recovery", isDirectory: true)
    let preservedFiles = try FileManager.default.contentsOfDirectory(
        at: recoveryDirectory,
        includingPropertiesForKeys: nil
    )
    #expect(preservedFiles.count == 1)
    #expect(try Data(contentsOf: preservedFiles[0]) == corruptData)
    let issueURL = try #require(store.recoveryIssue?.preservedURL)
    #expect(
        issueURL.resolvingSymlinksInPath()
            == preservedFiles[0].resolvingSymlinksInPath()
    )
}

@Test func appConfigDecodesDashboardAndSchedulerSettings() throws {
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "auto_start": false,
      "default_model": "mlx-community/Qwen3-0.6B-4bit",
      "loaded_models": ["mlx-community/Other-4bit", "mlx-community/Qwen3-0.6B-4bit"],
      "pinned_models": ["mlx-community/Qwen3-0.6B-4bit"],
      "language": "zh",
      "theme": "dark",
      "log_level": "debug",
      "mem_limit_total": 64,
      "mem_limit_model": 40,
      "mem_total_auto": false,
      "mem_total": 64,
      "mem_model_auto": false,
      "mem_model": 40,
      "max_sequences": 6,
      "prefill_chunk_size": 1024,
      "admission_deadline_ms": 9,
      "admission_queue_max": 24,
      "max_cache_cap": 32768,
      "decode_cadence_mid_chunk_cap": 128,
      "scheduler_profile": "/tmp/scheduler-profile.json",
      "scheduler_autotune_report": true
    }
    """

    let config = try JSONDecoder().decode(AppConfig.self, from: Data(json.utf8))

    #expect(config.defaultModel == "mlx-community/Qwen3-0.6B-4bit")
    #expect(config.loadedModels == ["mlx-community/Other-4bit", "mlx-community/Qwen3-0.6B-4bit"])
    #expect(config.pinnedModels == ["mlx-community/Qwen3-0.6B-4bit"])
    #expect(config.restoredModelReferences == ["mlx-community/Qwen3-0.6B-4bit", "mlx-community/Other-4bit"])
    #expect(config.language == "zh")
    #expect(config.logLevel == "debug")
    #expect(config.memLimitTotal == 64)
    #expect(config.memLimitModel == 40)
    #expect(config.memTotalAuto == false)
    #expect(config.memTotal == 64)
    #expect(config.memModelAuto == false)
    #expect(config.memModel == 40)
    #expect(config.maxSequences == 6)
    #expect(config.prefillChunkSize == 1024)
    #expect(config.admissionDeadlineMs == 9)
    #expect(config.admissionQueueMax == 24)
    #expect(config.maxCacheCap == 32768)
    #expect(config.decodeCadenceMidChunkCap == 128)
    #expect(config.schedulerProfile == "/tmp/scheduler-profile.json")
    #expect(config.schedulerAutotuneReport == true)
}

@Test func appConfigPersistsPinnedModelsAndClearsManualUnload() {
    var config = AppConfig(
        loadedModels: [
            "mlx-community/Alpha-4bit",
            "mlx-community/Beta-4bit",
        ],
        pinnedModels: ["mlx-community/Alpha-4bit"]
    )

    #expect(config.pinnedModelReferences == ["mlx-community/Alpha-4bit"])
    config.recordPinnedModel("mlx-community/Beta-4bit", pinned: true)
    #expect(config.pinnedModelReferences == [
        "mlx-community/Alpha-4bit",
        "mlx-community/Beta-4bit",
    ])

    config.recordUnloadedModel("mlx-community/Alpha-4bit")

    #expect(config.loadedModels == ["mlx-community/Beta-4bit"])
    #expect(config.pinnedModelReferences == ["mlx-community/Beta-4bit"])
}

@Test func backendLoadModelRequestEncodesOptionalMaxCacheCap() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/LongContext-4bit",
        modelDir: "/models/long",
        setDefault: true,
        maxCacheCap: 65536,
        pinned: true
    )

    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["model"] as? String == "mlx-community/LongContext-4bit")
    #expect(object["model_dir"] as? String == "/models/long")
    #expect(object["set_default"] as? Bool == true)
    #expect(object["max_cache_cap"] as? Int == 65536)
    #expect(object["pinned"] as? Bool == true)
}

@Test func backendLoadModelRequestEncodesIdleReloadAndSamplingDefaults() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/LongContext-4bit",
        modelDir: "/models/long",
        setDefault: true,
        maxCacheCap: 65536,
        reloadWhenIdle: true,
        deferWhenBusy: false,
        samplingDefaults: BackendSamplingDefaults(
            temperature: 0.7,
            topP: 0.8,
            topK: 40,
            repetitionPenalty: 1.1
        )
    )

    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["reload_when_idle"] as? Bool == true)
    #expect(object["defer_when_busy"] as? Bool == false)
    #expect(object["temperature"] as? Double == 0.7)
    #expect(object["top_p"] as? Double == 0.8)
    #expect(object["top_k"] as? Int == 40)
    #expect(object["repetition_penalty"] as? Double == 1.1)
}

@Test func backendLoadedModelInfoDecodesPinnedState() throws {
    let data = Data("""
    {
      "id": "mlx-community/Pinned-4bit",
      "model": "mlx-community/Pinned-4bit",
      "path": "/models/pinned",
      "architecture": "llm",
      "default": false,
      "max_position_embeddings": 4096,
      "pinned": true,
      "runtime_kind": "causal",
      "supports_streaming": true,
      "supports_vision": false,
      "supports_mtp": false,
      "supports_prompt_lookup": true,
      "supports_speculative_decoding": false,
      "supports_kv_cache": true,
      "supported_sampling_parameters": ["max_tokens", "temperature"],
      "runtime_state": "loaded",
      "active_requests": 0,
      "queued_requests": 0,
      "queue_capacity": 8,
      "usage": {
        "cumulative_tokens": 42,
        "input_tokens": 30,
        "output_tokens": 12,
        "prefix_cache": {
          "hit_tokens": 10,
          "eligible_tokens": 25
        }
      },
      "active_kv_offload": {
        "enabled": true,
        "status": "active",
        "active": true,
        "degraded": false,
        "mode": "request_preemption_hot_cold_tiering",
        "storage_dir": "/tmp/model-active-kv",
        "resident_pages": 4,
        "offloaded_pages": 3,
        "loading_pages": 0,
        "dirty_pages": 1,
        "parked_requests": 2,
        "offloaded_bytes": 1048576,
        "swap_out_count": 5,
        "swap_in_count": 4,
        "swap_error_count": 0,
        "last_swap_out_us": 123,
        "last_swap_in_us": 98,
        "supported_cache_kinds": ["full_attention_paged"],
        "not_applicable_cache_kinds": ["gated_delta_linear"]
      }
    }
    """.utf8)

    let info = try JSONDecoder().decode(BackendLoadedModelInfo.self, from: data)

    #expect(info.pinned)
    #expect(info.usage.cumulativeTokens == 42)
    #expect(info.usage.inputTokens == 30)
    #expect(info.usage.outputTokens == 12)
    #expect(info.usage.prefixCache?.hitTokens == 10)
    #expect(info.usage.prefixCache?.eligibleTokens == 25)
    #expect(info.activeKvOffload?.status == "active")
    #expect(info.activeKvOffload?.parkedRequests == 2)
    #expect(info.activeKvOffload?.offloadedBytes == 1_048_576)
}

@Test func restoredModelReferencesExcludeUnloadedDefaultModel() {
    let config = AppConfig(
        defaultModel: "mlx-community/Default-4bit",
        loadedModels: [
            "mlx-community/Other-4bit",
            "mlx-community/Third-4bit",
        ]
    )

    #expect(config.defaultModelReference == "mlx-community/Default-4bit")
    #expect(config.restoredModelReferences == [
        "mlx-community/Other-4bit",
        "mlx-community/Third-4bit",
    ])
}
