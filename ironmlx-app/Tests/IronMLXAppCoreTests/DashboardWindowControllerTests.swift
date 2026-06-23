import AppKit
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func dashboardWindowSupportsNativeFullscreenWithoutInitialFullscreenStyle() {
    #expect(DashboardWindowController.dashboardWindowStyleMask.contains(.resizable))
    #expect(!DashboardWindowController.dashboardWindowStyleMask.contains(.fullScreen))
    #expect(DashboardWindowController.dashboardWindowCollectionBehavior.contains(.fullScreenPrimary))
}

@MainActor
@Test func dashboardBootstrapIncludesPersistedRuntimeSettings() throws {
    let script = try DashboardWindowController.bootstrapScript(
        config: AppConfig(
            port: 9068,
            lastModel: "mlx-community/Qwen3.5-4B-MLX-4bit",
            language: "zh-Hans",
            kvQuant: "k3v4",
            maxSequences: 1,
            maxModels: 2,
            initCacheBlocks: 4,
            modelTtlMinutes: 15
        ),
        route: .status
    )

    #expect(script.contains("window.__IRONMLX_APP_CONFIG__"))
    #expect(script.contains(#""max_sequences":1"#))
    #expect(script.contains(#""max_models":2"#))
    #expect(script.contains(#""init_cache_blocks":4"#))
    #expect(script.contains(#""model_ttl_minutes":15"#))
    #expect(script.contains("window.__IRONMLX_AUTO_HOT_CACHE_BYTES__"))
    #expect(script.contains("window.__IRONMLX_COLD_CACHE_CAPACITY__"))
    #expect(script.contains(#""default_gb":10"#))
    #expect(script.contains(#""min_gb":1"#))
}
