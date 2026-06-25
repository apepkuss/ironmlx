import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func dashboardSettingsPayloadParsesOneAsMaxSequencesInteger() throws {
    let existing = AppConfig(
        host: "127.0.0.1",
        port: 9068,
        language: "zh-Hans",
        maxSequences: 16
    )
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "language": "zh-Hans",
      "max_sequences": 1
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.maxSequences == 1)
}

@MainActor
@Test func dashboardSettingsPayloadAllowsCacheAndKVQuantTogether() throws {
    let existing = AppConfig(cacheEnable: true, kvQuant: "none")
    let json = """
    {
      "cache_enable": true,
      "cache_dir": "/tmp/cache",
      "kv_quant": "k3v4"
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.cacheEnable == true)
    #expect(config.cacheDir == "/tmp/cache")
    #expect(config.kvQuant == "k3v4")
}

@MainActor
@Test func dashboardSettingsPayloadParsesActiveKVOffloadToggle() throws {
    let existing = AppConfig(activeKvOffload: false)
    let json = """
    {
      "active_kv_offload": true
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.activeKvOffload == true)
}
