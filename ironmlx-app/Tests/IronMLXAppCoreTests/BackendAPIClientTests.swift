import Testing

@testable import IronMLXAppCore

@Test func backendAPIClientUsesLoopbackForWildcardBindHost() {
    let client = BackendAPIClient(host: "0.0.0.0", port: 9068)

    #expect(BackendAPIClient.connectableHost(for: "0.0.0.0") == "127.0.0.1")
    #expect(BackendAPIClient.connectableHost(for: "::") == "127.0.0.1")
    #expect(BackendAPIClient.connectableHost(for: "127.0.0.1") == "127.0.0.1")
    #expect(client.host == "127.0.0.1")
}
