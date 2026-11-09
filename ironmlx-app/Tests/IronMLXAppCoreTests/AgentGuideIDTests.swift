import Foundation

import Testing

@testable import IronMLXAppCore

@Test func guideIDsMatchDashboardPage() {
    #expect(Set(AgentGuideID.allCases.map(\.rawValue)) == ["dsh", "opencode", "hermes", "omp", "dsh-desktop"])
}

@Test func hfEndpointOverrideAcceptsOnlyHTTPURLs() {
    #expect(HuggingFaceEndpointOverride.parse("http://127.0.0.1:9") == URL(string: "http://127.0.0.1:9"))
    #expect(HuggingFaceEndpointOverride.parse(" https://hf.invalid ") == URL(string: "https://hf.invalid"))
    #expect(HuggingFaceEndpointOverride.parse(nil) == nil)
    #expect(HuggingFaceEndpointOverride.parse("") == nil)
    #expect(HuggingFaceEndpointOverride.parse("ftp://example.com") == nil)
    #expect(HuggingFaceEndpointOverride.parse("example.com") == nil)
    #expect(HuggingFaceEndpointOverride.parse("https://") == nil)
}
