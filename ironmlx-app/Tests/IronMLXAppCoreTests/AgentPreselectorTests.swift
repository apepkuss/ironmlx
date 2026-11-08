import Foundation

import Testing

@testable import IronMLXAppCore

@Test func noneInstalledChoosesDeepSeekHarness() {
    #expect(AgentPreselector.choose(installed: []) == .dsh)
}

@Test func onlyOpenCodeInstalledChoosesOpenCode() {
    #expect(AgentPreselector.choose(installed: [.opencode]) == .opencode)
}

@Test func dshPreferredWhenSeveralInstalled() {
    #expect(AgentPreselector.choose(installed: [.omp, .hermes, .dsh, .opencode]) == .dsh)
    #expect(AgentPreselector.choose(installed: [.dshDesktop, .omp, .hermes]) == .hermes)
    #expect(AgentPreselector.choose(installed: [.dshDesktop, .omp]) == .omp)
    #expect(AgentPreselector.choose(installed: [.dshDesktop]) == .dshDesktop)
}

@Test func installedCodesTreatUnofficialCLIAsMissing() {
    #expect(!AgentInstallationChecker.isInstalled("not_installed"))
    #expect(!AgentInstallationChecker.isInstalled("cli_unrecognized"))
    #expect(AgentInstallationChecker.isInstalled("ready"))
    #expect(AgentInstallationChecker.isInstalled("version_too_old"))
    #expect(AgentInstallationChecker.isInstalled("version_unknown"))
}

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
