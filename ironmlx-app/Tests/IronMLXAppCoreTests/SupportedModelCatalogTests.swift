import Foundation
import JavaScriptCore
import Testing

@testable import IronMLXAppCore

@Suite("Supported model catalogue")
struct SupportedModelCatalogTests {
    @Test("bundled catalogue is injected into the offline dashboard")
    @MainActor
    func bootstrapContainsBundledCatalog() throws {
        let catalog = try SupportedModelCatalog.bundled()
        #expect(catalog.entries.count == 59)
        let script = try DashboardWindowController.bootstrapScript(config: AppConfig(), route: .status)
        let context = JSContext()!
        context.evaluateScript("var window = {};\n" + script)
        #expect(context.exception == nil)
        #expect(
            context.evaluateScript("window.__IRONMLX_SUPPORTED_MODELS__.entries.length")?.toInt32()
                == Int32(catalog.entries.count)
        )
        #expect(
            context.evaluateScript("window.__IRONMLX_CATALOG_MEMORY_HINTS__.availableBytes > 0")?.toBool() == true
        )
    }

    @Test("catalogue records repository evidence and separates auxiliary artifacts")
    func catalogEvidenceAndRoles() throws {
        let catalog = try SupportedModelCatalog.bundled()
        #expect(Set(catalog.entries.map(\.id)).count == catalog.entries.count)
        for entry in catalog.entries {
            #expect(entry.weightBytes > 0 && entry.downloadBytes >= entry.weightBytes)
            #expect(!entry.modelId.isEmpty && !entry.variantLabel.isEmpty)
            #expect(ModelSnapshotVerifier.isCommitSHA(entry.metadataRevision))
            #expect(!entry.evidence.isEmpty)
            #expect(["verified", "compatible"].contains(entry.status))
            #expect(entry.hfRepo.split(separator: "/").count == 2)
            #expect(entry.msRepo == nil || entry.msRepo?.split(separator: "/").count == 2)
            if entry.msRepo != nil {
                #expect(ModelSnapshotVerifier.isCommitSHA(entry.msMetadataRevision ?? ""))
            }
            if entry.category == "assistant" {
                #expect(entry.capabilities == ["assistant"])
                #expect(entry.noteKey != nil)
            }
        }
        let laya = try #require(catalog.entries.first { $0.id == "laya" })
        #expect(laya.msRepo == nil, "Do not invent a mirror for a repository missing on ModelScope")
    }

    @Test("model groups distinguish same-bit formats and separate assistant artifacts")
    func quantizationGroups() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let groups = Dictionary(grouping: catalog.entries, by: \.modelId)
        #expect(groups.count == 29)
        #expect(groups["qwen35-2b"]?.map(\.variantLabel) == ["Affine · 4 bit", "Affine · 5 bit", "Affine · 6 bit", "OptiQ · 4 bit"])
        #expect(groups["qwen38-27b"]?.map(\.variantLabel) == ["Affine · 4 bit", "Affine · 8 bit"])
        for entries in groups.values {
            #expect(Set(entries.map(\.variantLabel)).count == entries.count)
            #expect(Set(entries.map(\.name)).count == 1)
            #expect(Set(entries.map(\.category)).count == 1)
        }
        #expect(groups["dflash2-qwen38"]?.count == 1)
        #expect(groups["dflash2-qwen36-35b-a3b"]?.map(\.variantLabel) == ["BF16"])
    }

    @Test("memory hints reuse the download resource preflight policy")
    func memoryHintsUsePreflight() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let memory: UInt64 = 16 * 1_024 * 1_024 * 1_024
        let hints = catalog.memoryHints(physicalMemoryBytes: memory)
        #expect(hints.availableBytes == Int64(memory) - ModelResourcePreflight.memorySafetyBytes)
        for entry in catalog.entries {
            let preflight = ModelResourcePreflight(
                weightBytes: entry.weightBytes, remainingDownloadBytes: 0,
                availableDiskBytes: nil, physicalMemoryBytes: Int64(memory)
            )
            #expect(hints.estimatedBytes[entry.id] == preflight.estimatedPeakMemoryBytes)
        }
        #expect(catalog.memoryHints(physicalMemoryBytes: 0).availableBytes == 0)
    }
}
