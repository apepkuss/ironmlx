import Foundation

/// One repository file pinned by the bundled catalogue, verified regardless of download host.
public struct PinnedModelFile: Codable, Equatable, Sendable {
    public let path: String
    public let size: Int64
    public let sha256: String

    public init(path: String, size: Int64, sha256: String) {
        self.path = path
        self.size = size
        self.sha256 = sha256
    }
}

/// Curated repository variants shipped with this App, not an architecture allowlist.
struct SupportedModelCatalog: Codable, Sendable {
    /// Hand-picked beginner models by physical memory; `files` pins each entry's
    /// complete file list at its `metadataRevision`.
    struct Starter: Codable, Sendable {
        struct Tier: Codable, Equatable, Sendable {
            let minPhysicalMemoryGiB: Int
            let entryId: String
        }

        let tiers: [Tier]
        let upgrade: Tier
        let files: [String: [PinnedModelFile]]
    }

    struct Entry: Codable, Equatable, Sendable {
        let id: String
        let modelId: String
        let name: String
        let parameters: String?
        let weightFormat: String
        let variantLabel: String
        let weightBytes: Int64
        let downloadBytes: Int64
        let category: String
        let capabilities: [String]
        let status: String
        let noteKey: String?
        let hfRepo: String
        let msRepo: String?
        let metadataRevision: String
        let msMetadataRevision: String?
        let evidence: [String]
    }

    let version: Int
    let updatedAt: String
    let starter: Starter?
    let entries: [Entry]

    struct MemoryHints: Encodable {
        let availableBytes: Int64
        let estimatedBytes: [String: Int64]
    }

    func memoryHints(physicalMemoryBytes: UInt64) -> MemoryHints {
        let memory = Int64(clamping: physicalMemoryBytes)
        let estimates = entries.map { entry in
            let preflight = ModelResourcePreflight(
                weightBytes: entry.weightBytes,
                remainingDownloadBytes: 0,
                availableDiskBytes: nil,
                physicalMemoryBytes: memory
            )
            return (entry.id, preflight.estimatedPeakMemoryBytes)
        }
        return MemoryHints(
            availableBytes: max(0, memory - ModelResourcePreflight.memorySafetyBytes),
            estimatedBytes: Dictionary(estimates, uniquingKeysWith: { first, _ in first })
        )
    }

    static func bundled() throws -> Self {
        guard let url = IronMLXAppResourceResolver.url(
            forResource: "supported-models", withExtension: "json"
        ) else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}
