import Foundation

/// A catalogue model offered to a beginner together with its pinned file identities.
struct StarterRecommendation: Equatable, Sendable {
    let entry: SupportedModelCatalog.Entry
    let pinnedFiles: [PinnedModelFile]
    let isUpgrade: Bool
}

/// Applies the hand-picked starter tiers from the bundled catalogue; it never
/// ranks models by itself.
enum StarterModelSelector {
    private static let bytesPerGiB: UInt64 = 1_024 * 1_024 * 1_024

    static func initial(
        catalog: SupportedModelCatalog,
        physicalMemoryBytes: UInt64
    ) -> StarterRecommendation? {
        guard let starter = catalog.starter,
              let tier = starter.tiers
                .filter({ meets($0, physicalMemoryBytes) })
                .max(by: { $0.minPhysicalMemoryGiB < $1.minPhysicalMemoryGiB })
        else { return nil }
        return recommendation(for: tier, in: catalog, starter: starter, isUpgrade: false)
    }

    /// Offered only when memory allows it and the current model is one of the
    /// initial starter models. `currentModelID` may be the catalogue entry ID or
    /// the repository ID the backend reports.
    static func upgrade(
        catalog: SupportedModelCatalog,
        physicalMemoryBytes: UInt64,
        currentModelID: String
    ) -> StarterRecommendation? {
        guard let starter = catalog.starter, meets(starter.upgrade, physicalMemoryBytes) else { return nil }
        let entries = Dictionary(catalog.entries.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let currentIsStarter = starter.tiers.contains { tier in
            guard let entry = entries[tier.entryId] else { return false }
            return entry.id == currentModelID || entry.hfRepo == currentModelID
        }
        guard currentIsStarter else { return nil }
        return recommendation(for: starter.upgrade, in: catalog, starter: starter, isUpgrade: true)
    }

    private static func meets(_ tier: SupportedModelCatalog.Starter.Tier, _ memory: UInt64) -> Bool {
        let required = UInt64(max(0, tier.minPhysicalMemoryGiB)).multipliedReportingOverflow(by: bytesPerGiB)
        return !required.overflow && memory >= required.partialValue
    }

    private static func recommendation(
        for tier: SupportedModelCatalog.Starter.Tier,
        in catalog: SupportedModelCatalog,
        starter: SupportedModelCatalog.Starter,
        isUpgrade: Bool
    ) -> StarterRecommendation? {
        guard let entry = catalog.entries.first(where: { $0.id == tier.entryId }),
              let files = starter.files[tier.entryId], !files.isEmpty
        else { return nil }
        return StarterRecommendation(entry: entry, pinnedFiles: files, isUpgrade: isUpgrade)
    }
}
