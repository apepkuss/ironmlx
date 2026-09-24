import Foundation
import Testing

@testable import IronMLXAppCore

@Test func backendAPIClientUsesLoopbackForWildcardBindHost() {
    let client = BackendAPIClient(host: "0.0.0.0", port: 9068)

    #expect(BackendAPIClient.connectableHost(for: "0.0.0.0") == "127.0.0.1")
    #expect(BackendAPIClient.connectableHost(for: "::") == "127.0.0.1")
    #expect(BackendAPIClient.connectableHost(for: "127.0.0.1") == "127.0.0.1")
    #expect(client.host == "127.0.0.1")
}

@Test func backendLoadModelRequestEncodesMtpSettings() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/Qwen3.5-4B-MLX-4bit",
        modelDir: "/models/qwen",
        setDefault: true,
        maxCacheCap: 65536,
        mtpModelDir: "/models/qwen-mtp",
        mtpDraftTokens: 2,
        reloadWhenIdle: false,
        samplingDefaults: .empty
    )

    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["mtp_model_dir"] as? String == "/models/qwen-mtp")
    #expect(object["mtp_draft_tokens"] as? Int == 2)
}

@Test func backendLoadModelRequestEncodesDefaultOutputBudgetSeparatelyFromCacheCap() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/Model-4bit",
        modelDir: "/models/model",
        setDefault: true,
        maxCacheCap: 65_536,
        samplingDefaults: BackendSamplingDefaults(defaultMaxOutputTokens: 8_192)
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["max_cache_cap"] as? Int == 65_536)
    #expect(object["default_max_output_tokens"] as? Int == 8_192)
}

@Test func backendLoadModelRequestEncodesCrossRequestPromptLookupSettings() throws {
    let request = BackendLoadModelRequest(
        model: "mlx-community/Qwen3.5-4B-MLX-4bit",
        modelDir: "/models/qwen",
        setDefault: true,
        promptLookup: .crossRequest
    )

    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let promptLookup = try #require(object["prompt_lookup"] as? [String: Any])

    #expect(promptLookup["min_ngram"] as? Int == 2)
    #expect(promptLookup["max_ngram"] as? Int == 4)
    #expect(promptLookup["max_draft_tokens"] as? Int == 4)
    #expect(promptLookup["history_window_tokens"] as? Int == 32 * 1_024)
    #expect(promptLookup["max_index_entries"] as? Int == 64 * 1_024)
    #expect(promptLookup["cross_request"] as? Bool == true)
}

@Test func backendPromptLookupClearRequestEncodesOptionalModel() throws {
    let targeted = try JSONEncoder().encode(
        BackendPromptLookupClearRequest(model: "mlx-community/Qwen3.5-4B-MLX-4bit")
    )
    let targetedObject = try #require(
        JSONSerialization.jsonObject(with: targeted) as? [String: Any]
    )
    #expect(targetedObject["model"] as? String == "mlx-community/Qwen3.5-4B-MLX-4bit")

    let all = try JSONEncoder().encode(BackendPromptLookupClearRequest(model: nil))
    let allObject = try #require(JSONSerialization.jsonObject(with: all) as? [String: Any])
    #expect(allObject["model"] == nil)
}
