import Foundation
import Testing

@testable import IronMLXAppCore

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
      "pinned": true
    }
    """.utf8)

    let info = try JSONDecoder().decode(BackendLoadedModelInfo.self, from: data)

    #expect(info.pinned)
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

@Test func appLaunchRestoreKeepsUnloadedDefaultModelPreference() {
    let config = AppConfig(defaultModel: "mlx-community/Default-4bit")
    let backendLoadedModels = [
        BackendLoadedModelInfo(
            id: "mlx-community/Other-4bit",
            model: "mlx-community/Other-4bit",
            path: "/models/other",
            architecture: "llm",
            isDefault: true,
            maxPositionEmbeddings: 4096
        ),
        BackendLoadedModelInfo(
            id: "mlx-community/Third-4bit",
            model: "mlx-community/Third-4bit",
            path: "/models/third",
            architecture: "llm",
            isDefault: false,
            maxPositionEmbeddings: 4096
        ),
    ]

    let defaultModel = AppDelegate.defaultModelForLaunchRestore(
        config: config,
        backendLoadedModels: backendLoadedModels
    )

    #expect(defaultModel == "mlx-community/Default-4bit")
}

@Test func appLaunchRestoreUsesBackendDefaultWhenNoDefaultPreferenceExists() {
    let backendLoadedModels = [
        BackendLoadedModelInfo(
            id: "mlx-community/Other-4bit",
            model: "mlx-community/Other-4bit",
            path: "/models/other",
            architecture: "llm",
            isDefault: true,
            maxPositionEmbeddings: 4096
        ),
    ]

    let defaultModel = AppDelegate.defaultModelForLaunchRestore(
        config: AppConfig(),
        backendLoadedModels: backendLoadedModels
    )

    #expect(defaultModel == "mlx-community/Other-4bit")
}
