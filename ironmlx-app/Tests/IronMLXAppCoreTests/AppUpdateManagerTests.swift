import Foundation
import Testing

@testable import IronMLXAppCore

@Test
func developmentUpdateConfigurationRequiresLoopbackHTTPSAndSignatures() throws {
    let configuration = try AppUpdateConfiguration(infoDictionary: validUpdateInfo())

    #expect(configuration.feedURL.absoluteString == "https://127.0.0.1:8443/appcast.xml")
    #expect(configuration.publicEdKey == "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
}

@Test
func developmentUpdateConfigurationRejectsRemoteFeed() {
    var info = validUpdateInfo()
    info["SUFeedURL"] = "https://updates.ironmlx.example/appcast.xml"

    #expect(throws: AppUpdateConfigurationError.invalidDevelopmentFeed(
        "https://updates.ironmlx.example/appcast.xml"
    )) {
        _ = try AppUpdateConfiguration(infoDictionary: info)
    }
}

@Test
func developmentUpdateConfigurationRejectsUnsignedFeed() {
    var info = validUpdateInfo()
    info["SURequireSignedFeed"] = false

    #expect(throws: AppUpdateConfigurationError.signatureVerificationNotRequired) {
        _ = try AppUpdateConfiguration(infoDictionary: info)
    }
}

@Test
func developmentUpdateConfigurationRejectsInvalidPublicKey() {
    var info = validUpdateInfo()
    info["SUPublicEDKey"] = "not-an-ed25519-public-key"

    #expect(throws: AppUpdateConfigurationError.invalidPublicEdKey) {
        _ = try AppUpdateConfiguration(infoDictionary: info)
    }
}

@Test
func developmentUpdateConfigurationRequiresAutomaticDownloads() {
    var info = validUpdateInfo()
    info["SUAutomaticallyUpdate"] = false

    #expect(throws: AppUpdateConfigurationError.automaticUpdatesNotEnabled) {
        _ = try AppUpdateConfiguration(infoDictionary: info)
    }
}

@Test
func developmentUpdateConfigurationRejectsStableChannelInPhaseOne() {
    var info = validUpdateInfo()
    info[AppUpdateConfiguration.channelKey] = "stable"

    #expect(throws: AppUpdateConfigurationError.unsupportedChannel("stable")) {
        _ = try AppUpdateConfiguration(infoDictionary: info)
    }
}

private func validUpdateInfo() -> [String: Any] {
    [
        AppUpdateConfiguration.channelKey: AppUpdateConfiguration.developmentChannel,
        "SUFeedURL": "https://127.0.0.1:8443/appcast.xml",
        "SUPublicEDKey": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "SUEnableAutomaticChecks": true,
        "SUAutomaticallyUpdate": true,
        "SUEnableSystemProfiling": false,
        "SURequireSignedFeed": true,
        "SUVerifyUpdateBeforeExtraction": true,
    ]
}
