import Foundation
import Testing

@testable import IronMLXAppCore

@Test
@MainActor
func sparkleLanguageFollowsAppLanguage() {
    #expect(SparkleAppUpdateManager.sparkleLanguage(for: "zh") == "zh_CN")
    #expect(SparkleAppUpdateManager.sparkleLanguage(for: "zh-Hans") == "zh_CN")
    #expect(SparkleAppUpdateManager.sparkleLanguage(for: "zh-Hant") == "zh_TW")
    #expect(SparkleAppUpdateManager.sparkleLanguage(for: "en") == "en")
}

@Test
@MainActor
func sparkleLocalizationCanChangeBetweenUpdateChecks() throws {
    let bundle = try #require(Bundle(identifier: "org.sparkle-project.Sparkle"))
    for (language, expected) in [
        ("en", "Update Error!"),
        ("zh-Hans", "\u{66f4}\u{65b0}\u{9519}\u{8bef}\u{ff01}"),
        ("ja", "\u{30a2}\u{30c3}\u{30d7}\u{30c7}\u{30fc}\u{30c8}\u{30a8}\u{30e9}\u{30fc}!"),
        ("en", "Update Error!"),
    ] {
        SparkleAppUpdateManager.applySparkleLanguage(for: language)
        #expect(
            bundle.localizedString(forKey: "Update Error!", value: nil, table: "Sparkle")
                == expected
        )
    }
}

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
func publicUpdateChannelsAcceptRemoteSignedFeeds() throws {
    for channel in ["stable", "release-candidate"] {
        var info = validUpdateInfo()
        info[AppUpdateConfiguration.channelKey] = channel
        info["SUFeedURL"] = "https://updates.example/\(channel).xml"
        let configuration = try AppUpdateConfiguration(infoDictionary: info)
        #expect(configuration.channel == channel)
    }
}

@Test
func publicUpdateChannelsRejectLoopbackAndCredentialURLs() {
    for feed in ["https://127.0.0.1/feed.xml", "http://updates.example/feed.xml",
                 "https://user:secret@updates.example/feed.xml", "https://updates.example/feed.xml#fragment"] {
        var info = validUpdateInfo()
        info[AppUpdateConfiguration.channelKey] = "stable"
        info["SUFeedURL"] = feed
        #expect(throws: AppUpdateConfigurationError.invalidPublicFeed(feed)) {
            _ = try AppUpdateConfiguration(infoDictionary: info)
        }
    }
}

@Test
func unknownUpdateChannelIsRejected() {
    var info = validUpdateInfo()
    info[AppUpdateConfiguration.channelKey] = "unknown"
    #expect(throws: AppUpdateConfigurationError.unsupportedChannel("unknown")) {
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
