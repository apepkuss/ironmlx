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

@MainActor
@Test func dashboardSettingsPayloadParsesMemoryLimits() throws {
    let existing = AppConfig(
        memLimitTotal: nil,
        memLimitModel: nil,
        memTotalAuto: true,
        memTotal: 24,
        memModelAuto: true,
        memModel: 22
    )
    let json = """
    {
      "mem_limit_total": 64,
      "mem_limit_model": 40,
      "mem_total_auto": false,
      "mem_total": 64,
      "mem_model_auto": false,
      "mem_model": 40
    }
    """

    let config = try DashboardBridge.config(applyingSettingsJSON: json, to: existing)

    #expect(config.memLimitTotal == 64)
    #expect(config.memLimitModel == 40)
    #expect(config.memTotalAuto == false)
    #expect(config.memTotal == 64)
    #expect(config.memModelAuto == false)
    #expect(config.memModel == 40)
}

@Test func dashboardLoadDoesNotMakeSecondModelDefaultUnlessUserSelectedIt() {
    let config = AppConfig(
        defaultModel: "mlx-community/Existing-4bit",
        loadedModels: ["mlx-community/Existing-4bit"]
    )

    let shouldSetDefault = DashboardBridge.shouldSetDefaultWhenLoadingModel(
        "mlx-community/New-4bit",
        config: config,
        currentLoadedModelCount: 1
    )

    #expect(shouldSetDefault == false)
}

@Test func dashboardLoadMakesModelDefaultWhenItIsTheOnlyLoadedModelOrExplicitDefault() {
    let firstModelConfig = AppConfig()
    let explicitDefaultConfig = AppConfig(
        defaultModel: "mlx-community/New-4bit",
        loadedModels: ["mlx-community/Existing-4bit"]
    )

    #expect(DashboardBridge.shouldSetDefaultWhenLoadingModel(
        "mlx-community/New-4bit",
        config: firstModelConfig,
        currentLoadedModelCount: 0
    ))
    #expect(DashboardBridge.shouldSetDefaultWhenLoadingModel(
        "mlx-community/New-4bit",
        config: explicitDefaultConfig,
        currentLoadedModelCount: 1
    ))
}
