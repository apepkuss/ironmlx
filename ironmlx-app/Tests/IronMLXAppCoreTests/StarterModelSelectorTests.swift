import Foundation
import Testing

@testable import IronMLXAppCore

@Suite("Starter model selector")
struct StarterModelSelectorTests {
    private static let gib: UInt64 = 1_024 * 1_024 * 1_024

    @Test(arguments: [
        (UInt64(8), "qwen35-2b-4bit"),
        (UInt64(16), "qwen35-4b-affine4"),
        (UInt64(32), "qwen35-4b-affine4"),
        (UInt64(48), "qwen35-4b-affine4"),
    ])
    func initialRecommendationFollowsMemoryTier(memoryGiB: UInt64, expected: String) throws {
        let catalog = try SupportedModelCatalog.bundled()
        let recommendation = try #require(
            StarterModelSelector.initial(catalog: catalog, physicalMemoryBytes: memoryGiB * Self.gib)
        )
        #expect(recommendation.entry.id == expected)
        #expect(recommendation.isUpgrade == false)
        #expect(recommendation.pinnedFiles == catalog.starter?.files[expected])
    }

    @Test func upgradeOfferedAt48GiBForStarterModel() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let upgrade = try #require(StarterModelSelector.upgrade(
            catalog: catalog, physicalMemoryBytes: 48 * Self.gib, currentModelID: "qwen35-4b-affine4"
        ))
        #expect(upgrade.entry.id == "qwen36-35b-a3b-affine4")
        #expect(upgrade.isUpgrade)
        #expect(upgrade.pinnedFiles == catalog.starter?.files["qwen36-35b-a3b-affine4"])
        // The backend reports the repository ID for downloaded models.
        #expect(StarterModelSelector.upgrade(
            catalog: catalog, physicalMemoryBytes: 64 * Self.gib,
            currentModelID: "mlx-community/Qwen3.5-4B-MLX-4bit"
        )?.entry.id == "qwen36-35b-a3b-affine4")
    }

    @Test func noUpgradeBelow48GiB() throws {
        let catalog = try SupportedModelCatalog.bundled()
        #expect(StarterModelSelector.upgrade(
            catalog: catalog, physicalMemoryBytes: 32 * Self.gib, currentModelID: "qwen35-4b-affine4"
        ) == nil)
    }

    @Test func noUpgradeWhenAlreadyOnUpgradeModel() throws {
        let catalog = try SupportedModelCatalog.bundled()
        #expect(StarterModelSelector.upgrade(
            catalog: catalog, physicalMemoryBytes: 64 * Self.gib, currentModelID: "qwen36-35b-a3b-affine4"
        ) == nil)
        #expect(StarterModelSelector.upgrade(
            catalog: catalog, physicalMemoryBytes: 64 * Self.gib,
            currentModelID: "mlx-community/Qwen3.6-35B-A3B-4bit"
        ) == nil)
    }

    @Test func noUpgradeForANonStarterModel() throws {
        let catalog = try SupportedModelCatalog.bundled()
        #expect(StarterModelSelector.upgrade(
            catalog: catalog, physicalMemoryBytes: 64 * Self.gib, currentModelID: "org/custom-model"
        ) == nil)
    }
}
