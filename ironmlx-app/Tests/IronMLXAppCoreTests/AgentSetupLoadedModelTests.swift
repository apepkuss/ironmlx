import Foundation
import Testing
@testable import IronMLXAppCore

private let target = "mlx-community/Qwen3.8-27B-4bit"

@MainActor
@Test func agentSetupConfirmsExclusiveDFlash2TargetFromHealthWithoutAdminAPI() async throws {
    // The exclusive DFlash2 server answers /admin/api/models/loaded with 404.
    let noAdminAPI: () async throws -> [BackendLoadedModelInfo] = {
        Issue.record("exclusive DFlash2 serving has no loaded-model admin API")
        throw URLError(.badServerResponse)
    }
    let ready = exclusiveDFlash2Health(model: target, status: "healthy")
    let found = try await DashboardBridge.loadedAgentModel(
        target, dflash2Target: target, health: { ready }, loadedModels: noAdminAPI)
    #expect(found?.id == target)
    #expect(found?.capabilities.runtimeKind == "causal")

    for unready in [exclusiveDFlash2Health(model: target, status: "loading"),
                    exclusiveDFlash2Health(model: "mlx-community/other", status: "healthy"),
                    exclusiveDFlash2Health(model: target, status: "healthy", dflash2: false)] {
        #expect(try await DashboardBridge.loadedAgentModel(
            target, dflash2Target: target, health: { unready }, loadedModels: noAdminAPI) == nil)
    }
}

@MainActor
@Test func agentSetupUsesLoadedModelListOutsideExclusiveDFlash2() async throws {
    let causal = LegacyHealthAdapter().legacyStatus(
        from: exclusiveDFlash2Health(model: target, status: "healthy")).runtimeModels
    let found = try await DashboardBridge.loadedAgentModel(
        target, dflash2Target: nil,
        health: { Issue.record("ordinary serving reads the loaded-model list"); throw URLError(.badServerResponse) },
        loadedModels: { causal })
    #expect(found?.id == target)
    #expect(try await DashboardBridge.loadedAgentModel(
        "mlx-community/not-loaded", dflash2Target: nil,
        health: { throw URLError(.badServerResponse) }, loadedModels: { causal }) == nil)
}

private func exclusiveDFlash2Health(model name: String, status: String, dflash2: Bool = true) -> HealthzSnapshot {
    var snapshot = HealthzSnapshot(
        status: status,
        uptimeSecs: 10,
        model: .init(name: name, maxPositionEmbeddings: 262_144),
        scheduler: .init(bMax: 4, bActive: 0, bQueued: 0, queueMax: 32,
                         admissionQueueFullCount: 0, memoryBudgetExceededCount: 0),
        memory: .init(totalRamBytes: 128 << 30, freeRamBytes: 64 << 30, kvCacheActiveBytes: 0,
                      kvCacheSoftLimitBytes: 0, kvCacheLogicalCapTokens: 262_144,
                      kvCacheResidentCapTokens: 262_144, kvCacheBudgetPolicy: "full_resident",
                      mlxTotalBytes: nil, mlxMaxRecommendedBytes: nil, mlxActiveBytes: 0,
                      mlxCacheBytes: 0, mlxPeakBytes: 0, mlxMemoryLimitBytes: 0),
        activeKvOffload: .init(enabled: false, status: "disabled", active: false, degraded: false,
                               mode: "disabled", storageDir: nil, residentPages: 0, offloadedPages: 0,
                               loadingPages: 0, dirtyPages: 0, parkedRequests: 0, offloadedBytes: 0,
                               swapOutCount: 0, swapInCount: 0, swapErrorCount: 0, lastSwapOutUs: 0,
                               lastSwapInUs: 0, supportedCacheKinds: [], notApplicableCacheKinds: []),
        deviceName: "Apple Test",
        version: "test"
    )
    snapshot.dflash2 = .init(
        enabled: dflash2, checkpointBlockSize: 8, blockSize: 8, maxDraftTokens: 7, draftBudgetChanges: 0,
        currentDraftBudget: 7, draftQuantizationBits: 4, requests: 0, windows: 0, draftedTokens: 0,
        acceptedDraftTokens: 0, rollbackCount: 0, tensorBatchWindows: 0, tensorBatchDivergentSplits: 0,
        tensorBatchGroupsCreated: 0, tensorBatchWidthLimit: 4, tensorBatchMaxWidth: 4, sampledRequests: 0,
        exactSamplingWindows: 0, exactAcceptanceDraws: 0, exactResidualCorrections: 0, exactBonusSamples: 0,
        samplingUs: 0, latestGenerationTPS: 0, latestAcceptanceRate: 0, peakMemoryBytes: 0,
        prefixCacheEnabled: false, prefixCacheMaxBytes: 0, prefixCacheEntries: 0, prefixCacheBytes: 0,
        prefixCacheHits: 0, prefixCacheMisses: 0, prefixCacheSaves: 0, prefixCacheEvictions: 0,
        prefixCacheHitTokens: 0,
        runtimeUsage: BackendModelRuntimeUsage(cumulativeTokens: 0, inputTokens: 0, outputTokens: 0))
    return snapshot
}
