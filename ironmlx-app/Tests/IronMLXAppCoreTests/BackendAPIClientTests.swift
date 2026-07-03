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
