import Foundation
import Testing

@testable import IronMLXAppCore

@Test func appConfigDecodesDashboardAndSchedulerSettings() throws {
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "auto_start": false,
      "last_model": "mlx-community/Qwen3-0.6B-4bit",
      "language": "zh",
      "theme": "dark",
      "log_level": "debug",
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

    #expect(config.lastModel == "mlx-community/Qwen3-0.6B-4bit")
    #expect(config.language == "zh")
    #expect(config.logLevel == "debug")
    #expect(config.maxSequences == 6)
    #expect(config.prefillChunkSize == 1024)
    #expect(config.admissionDeadlineMs == 9)
    #expect(config.admissionQueueMax == 24)
    #expect(config.maxCacheCap == 32768)
    #expect(config.decodeCadenceMidChunkCap == 128)
    #expect(config.schedulerProfile == "/tmp/scheduler-profile.json")
    #expect(config.schedulerAutotuneReport == true)
}

@Test func backendLoadModelRequestEncodesOptionalMaxCacheCap() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/LongContext-4bit",
        modelDir: "/models/long",
        setDefault: true,
        maxCacheCap: 65536
    )

    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["model"] as? String == "mlx-community/LongContext-4bit")
    #expect(object["model_dir"] as? String == "/models/long")
    #expect(object["set_default"] as? Bool == true)
    #expect(object["max_cache_cap"] as? Int == 65536)
}

@Test func backendLoadModelRequestEncodesIdleReloadAndSamplingDefaults() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/LongContext-4bit",
        modelDir: "/models/long",
        setDefault: true,
        maxCacheCap: 65536,
        reloadWhenIdle: true,
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
    #expect(object["temperature"] as? Double == 0.7)
    #expect(object["top_p"] as? Double == 0.8)
    #expect(object["top_k"] as? Int == 40)
    #expect(object["repetition_penalty"] as? Double == 1.1)
}
