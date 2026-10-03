import Foundation

/// Curated repository variants shipped with this App, not an architecture allowlist.
struct SupportedModelCatalog: Codable, Sendable {
    struct Entry: Codable, Sendable {
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
