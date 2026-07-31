import Foundation
import Testing

@testable import IronMLXAppCore

@Test func serveArgumentsStartAppDaemonWithoutModel() {
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068
    )

    #expect(config.arguments == [
        "serve",
        "--host", "127.0.0.1",
        "--port", "9068",
        "--network-mode", "local",
    ])
}

@Test func serveArgumentsIncludePersistedBackendRuntimeSettings() {
    let options = BackendLaunchOptions(
        prefillChunkSize: 1024,
        bMax: 4,
        admissionDeadlineMs: 9,
        admissionQueueMax: 24,
        maxCacheCap: 32768,
        decodeCadenceMidChunkCap: 128,
        schedulerProfile: "/tmp/scheduler-profile.json",
        schedulerAutotuneReport: true
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments == [
        "serve",
        "--host", "127.0.0.1",
        "--port", "9068",
        "--network-mode", "local",
        "--prefill-chunk-size", "1024",
        "--max-sequences", "4",
        "--admission-deadline-ms", "9",
        "--admission-queue-max", "24",
        "--max-cache-cap", "32768",
        "--decode-cadence-mid-chunk-cap", "128",
        "--scheduler-profile", "/tmp/scheduler-profile.json",
        "--scheduler-autotune-report",
    ])
}

@Test func lanServeArgumentsUseSelectedAddressAndStdinBootstrap() {
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        networkMode: "lan",
        lanHost: "192.168.1.24",
        securityBootstrapStdin: true
    )

    #expect(config.arguments == [
        "serve", "--host", "127.0.0.1", "--port", "9068",
        "--network-mode", "lan", "--lan-host", "192.168.1.24",
        "--security-bootstrap-stdin",
    ])
}

@Test func backendLaunchOptionsMapDashboardMaxSequencesToBMax() {
    let config = AppConfig(maxSequences: 6)

    #expect(BackendLaunchOptions(config: config).bMax == 6)
}

@Test func backendLaunchOptionsPreferDashboardMaxSequencesOverStaleInternalBMax() throws {
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "language": "zh-Hans",
      "max_sequences": 1,
      "b_max": 16
    }
    """
    let config = try JSONDecoder().decode(AppConfig.self, from: Data(json.utf8))

    #expect(BackendLaunchOptions(config: config).bMax == 1)
}

@Test func serveArgumentsUseMaxSequencesForDashboardMaxSequences() {
    let options = BackendLaunchOptions(config: AppConfig(maxSequences: 6))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--max-sequences"))
    #expect(!config.arguments.contains("--b-max"))
}

@Test func serveArgumentsIncludeMaxLoadedModelsForEnginePoolCapacity() {
    let options = BackendLaunchOptions(config: AppConfig(maxModels: 3))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--max-loaded-models"))
    #expect(config.arguments.contains("3"))
}

@Test func serveArgumentsIncludeConfiguredMemoryLimits() {
    let options = BackendLaunchOptions(
        config: AppConfig(memLimitTotal: 64, memLimitModel: 40)
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--memory-limit-total-gb"))
    #expect(config.arguments.contains("64"))
    #expect(config.arguments.contains("--memory-limit-model-gb"))
    #expect(config.arguments.contains("40"))
}

@Test func serveArgumentsOmitAutomaticMemoryLimits() {
    let options = BackendLaunchOptions(
        config: AppConfig(memLimitTotal: 0, memLimitModel: 0)
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(!config.arguments.contains("--memory-limit-total-gb"))
    #expect(!config.arguments.contains("--memory-limit-model-gb"))
}

@Test func serveArgumentsIncludeModelTTLWhenEnabled() {
    let options = BackendLaunchOptions(config: AppConfig(modelTtlMinutes: 30))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--model-ttl-minutes"))
    #expect(config.arguments.contains("30"))
}

@Test func backendLaunchOptionsUseDefaultModelTTLForAppMode() {
    let options = BackendLaunchOptions(config: AppConfig())

    #expect(options.modelTtlMinutes == BackendLaunchOptions.defaultModelTtlMinutes)
}

@Test func serveArgumentsOmitModelTTLWhenDisabled() {
    let options = BackendLaunchOptions(config: AppConfig(modelTtlMinutes: 0))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(!config.arguments.contains("--model-ttl-minutes"))
}

@Test func serveArgumentsIncludePersistedKVQuantSetting() {
    let options = BackendLaunchOptions(config: AppConfig(cacheEnable: false, kvQuant: "turbo4"))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--kv-quant"))
    #expect(config.arguments.contains("turbo4"))
}

@Test func serveArgumentsOmitActiveKVOffloadByDefault() {
    let options = BackendLaunchOptions(config: AppConfig())
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(!config.arguments.contains("--active-kv-offload"))
}

@Test func serveArgumentsIncludeActiveKVOffloadWhenEnabled() {
    let options = BackendLaunchOptions(config: AppConfig(activeKvOffload: true))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--active-kv-offload"))
}

@Test func serveArgumentsMapLegacyAdaptiveKVQuantToK3V4() {
    let options = BackendLaunchOptions(config: AppConfig(cacheEnable: false, kvQuant: "adaptive"))
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.suffix(2) == ["--kv-quant", "k3v4"])
}

@Test func backendLaunchOptionsAllowCacheAndKVQuantTogether() {
    let options = BackendLaunchOptions(
        config: AppConfig(cacheEnable: true, cacheDir: "/tmp/cache", kvQuant: "k3v4")
    )

    #expect(options.validationError == nil)
    #expect(options.isValid == true)
}

@Test func serveArgumentsIncludeCacheAndKVQuantFlagsTogether() {
    let options = BackendLaunchOptions(
        config: AppConfig(cacheEnable: true, cacheDir: "/tmp/cache", kvQuant: "k3v4")
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--paged-prefix-cache-dir"))
    #expect(config.arguments.contains("/tmp/cache"))
    #expect(config.arguments.contains("--prefix-lru-cache-max-bytes"))
    #expect(config.arguments.contains("--ssd-prefix-cache-max-gb"))
    #expect(config.arguments.suffix(2) == ["--kv-quant", "k3v4"])
}

@Test func serveArgumentsIncludeManualHotCacheLimit() {
    let options = BackendLaunchOptions(
        config: AppConfig(
            hotCache: 4,
            cacheEnable: true,
            cacheDir: "/tmp/ironmlx-prefix-cache",
            kvQuant: "none"
        ),
        physicalMemoryBytes: 128 * 1024 * 1024 * 1024
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--paged-prefix-cache-dir"))
    #expect(config.arguments.contains("/tmp/ironmlx-prefix-cache"))
    #expect(config.arguments.contains("--prefix-lru-cache-max-bytes"))
    #expect(config.arguments.contains(String(4 * 1024 * 1024 * 1024)))
}

@Test func serveArgumentsIncludeColdCacheLimitWhenCacheEnabled() {
    let options = BackendLaunchOptions(
        config: AppConfig(
            hotCache: 4,
            coldCache: 12,
            cacheEnable: true,
            cacheDir: "/tmp/ironmlx-prefix-cache",
            kvQuant: "none"
        ),
        physicalMemoryBytes: 128 * 1024 * 1024 * 1024
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(config.arguments.contains("--ssd-prefix-cache-max-gb"))
    #expect(config.arguments.contains("12"))
}

@Test func backendLaunchOptionsUseDefaultColdCacheLimit() {
    let options = BackendLaunchOptions(
        config: AppConfig(hotCache: 0, cacheEnable: true, cacheDir: "/tmp/cache"),
        physicalMemoryBytes: 128 * 1024 * 1024 * 1024
    )

    #expect(options.ssdPrefixCacheMaxGB == 10)
}

@Test func coldCacheCapacityPolicyUsesDiskSpaceAndReserve() {
    let gib = BackendLaunchOptions.bytesPerGigabyte

    #expect(ColdCacheCapacityPolicy.maximumGigabytes(availableBytes: 30 * gib) == 15)
    #expect(ColdCacheCapacityPolicy.maximumGigabytes(availableBytes: 80 * gib) == 40)
    #expect(ColdCacheCapacityPolicy.maximumGigabytes(availableBytes: 300 * gib) == 100)
    #expect(ColdCacheCapacityPolicy.maximumGigabytes(availableBytes: 12 * gib) == 2)
    #expect(ColdCacheCapacityPolicy.maximumGigabytes(availableBytes: nil) == 100)
}

@Test func backendLaunchOptionsUseAutoHotCacheLimit() {
    let options = BackendLaunchOptions(
        config: AppConfig(hotCache: 0, cacheEnable: true, cacheDir: "/tmp/cache"),
        physicalMemoryBytes: 128 * 1024 * 1024 * 1024
    )

    #expect(options.prefixLruCacheMaxBytes == 8 * 1024 * 1024 * 1024)
}

@Test func serveArgumentsOmitPrefixCacheWhenCacheDisabled() {
    let options = BackendLaunchOptions(
        config: AppConfig(hotCache: 4, cacheEnable: false, cacheDir: "/tmp/cache"),
        physicalMemoryBytes: 128 * 1024 * 1024 * 1024
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        options: options
    )

    #expect(!config.arguments.contains("--paged-prefix-cache-dir"))
    #expect(!config.arguments.contains("--prefix-lru-cache-max-bytes"))
    #expect(!config.arguments.contains("--ssd-prefix-cache-max-gb"))
}
