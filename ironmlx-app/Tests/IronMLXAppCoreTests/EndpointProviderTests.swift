import Foundation
import Testing

@testable import IronMLXAppCore

@Test func endpointPayloadKeepsDashboardShape() throws {
    let payload = EndpointPayload(host: "127.0.0.1", port: 9068)
    let data = try JSONEncoder().encode(payload)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["host"] as? String == "127.0.0.1")
    #expect(object["loopback"] as? String == "127.0.0.1")
    #expect(object["port"] as? Int == 9068)
    #expect((object["interfaces"] as? [Any])?.isEmpty == true)
}
