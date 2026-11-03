import Foundation
import Testing
@testable import IronMLXAppCore

@Test func pagingHealthSurvivesLoadedModelBridgeRoundTrip() throws {
    let cache: [String: Any] = ["budget_bytes": 1024, "charged_bytes": 512, "payload_bytes": 256,
        "peak_charged_bytes": 768, "entries": 2, "pinned_bytes": 0, "phases": []]
    let source: [String: Any] = ["instance_id": "instance-one", "mode": "ssd", "started_unix_ms": 100,
        "sampled_unix_ms": 200, "elapsed_ms": 100, "experts": cache, "ple": cache]
    let paging = try JSONDecoder().decode(BackendPagingInfo.self, from: JSONSerialization.data(withJSONObject: source))
    let model = BackendLoadedModelInfo(id: "flash", paging: paging, model: "flash", path: "", architecture: "qwen4_exp", isDefault: true, maxPositionEmbeddings: 4096)
    let data = try JSONEncoder().encode(model)
    let restored = try JSONDecoder().decode(BackendLoadedModelInfo.self, from: data)
    #expect(restored.paging == paging)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let encodedPaging = try #require(json["paging"] as? [String: Any])
    #expect(encodedPaging["instance_id"] as? String == "instance-one")
    #expect(encodedPaging["sampled_unix_ms"] as? Int == 200)
}
