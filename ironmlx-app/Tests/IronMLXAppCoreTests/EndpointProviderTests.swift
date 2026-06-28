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

@Test func endpointPayloadPublishesLanInterfacesForWildcardHost() throws {
    let payload = EndpointPayload(
        host: "0.0.0.0",
        port: 9068,
        interfaces: [
            EndpointPayload.NetworkInterface(name: "en0", ip: "192.168.1.24")
        ]
    )
    let data = try JSONEncoder().encode(payload)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let interfaces = try #require(object["interfaces"] as? [[String: Any]])

    #expect(object["host"] as? String == "0.0.0.0")
    #expect(object["loopback"] as? String == "127.0.0.1")
    #expect(interfaces.count == 1)
    #expect(interfaces[0]["name"] as? String == "en0")
    #expect(interfaces[0]["ip"] as? String == "192.168.1.24")
}

@Test func endpointPayloadRecognizesWildcardHosts() {
    #expect(EndpointPayload.shouldPublishNetworkInterfaces(for: "0.0.0.0"))
    #expect(EndpointPayload.shouldPublishNetworkInterfaces(for: "::"))
    #expect(!EndpointPayload.shouldPublishNetworkInterfaces(for: "127.0.0.1"))
    #expect(!EndpointPayload.shouldPublishNetworkInterfaces(for: "localhost"))
}
