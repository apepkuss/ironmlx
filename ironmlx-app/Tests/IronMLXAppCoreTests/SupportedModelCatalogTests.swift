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
        #expect(catalog.entries.count == 63)
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
        let asr = catalog.entries.filter { $0.category == "stt" }
        #expect(Set(asr.map(\.hfRepo)) == Set(["mlx-community/Qwen3-ASR-0.6B-8bit", "mlx-community/Qwen3-ASR-1.7B-8bit"]))
        #expect(asr.allSatisfy { $0.capabilities == ["stt"] && $0.msRepo == nil })
        let laya = try #require(catalog.entries.first { $0.id == "laya" })
        #expect(laya.msRepo == nil, "Do not invent a mirror for a repository missing on ModelScope")
    }

    @Test("model groups distinguish same-bit formats and separate assistant artifacts")
    func quantizationGroups() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let groups = Dictionary(grouping: catalog.entries, by: \.modelId)
        #expect(groups.count == 33)
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

    @Test("starter entries exist and pin every file")
    func starterEntriesExistAndPinEveryFile() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let starter = try #require(catalog.starter)
        let entries = Dictionary(uniqueKeysWithValues: catalog.entries.map { ($0.id, $0) })
        #expect(starter.tiers.map(\.minPhysicalMemoryGiB) == [0, 16])
        let ids = starter.tiers.map(\.entryId) + [starter.upgrade.entryId]
        #expect(ids == ["qwen35-2b-4bit", "qwen35-4b-affine4", "qwen36-35b-a3b-affine4"])
        for id in ids {
            let entry = try #require(entries[id])
            let files = try #require(starter.files[id])
            #expect(!files.isEmpty)
            #expect(Set(files.map(\.path)).count == files.count)
            for file in files {
                #expect(file.sha256.count == 64)
                #expect(file.sha256.allSatisfy { "0123456789abcdef".contains($0) })
                #expect(file.size >= 0)
            }
            #expect(files.reduce(Int64(0)) { $0 + $1.size } == entry.downloadBytes)
        }
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
