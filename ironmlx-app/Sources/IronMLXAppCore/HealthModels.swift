import Foundation

public struct HealthzSnapshot: Codable, Equatable, Sendable {
    public var status: String
    public var mode: String = "single"
    public var models: [BackendLoadedModelInfo] = []
    public var uptimeSecs: UInt64
    public var model: ModelInfo
    public var scheduler: SchedulerInfo
    public var memory: MemoryInfo
    public var dflash2: DFlash2Info?
    public var activeKvOffload: ActiveKvOffloadInfo
    public var deviceName: String?
    public var version: String

    enum CodingKeys: String, CodingKey {
        case status
        case mode
        case models
        case uptimeSecs = "uptime_secs"
        case model
        case scheduler
        case memory
        case dflash2
        case activeKvOffload = "active_kv_offload"
        case deviceName = "device_name"
        case version
    }

    public struct ModelInfo: Codable, Equatable, Sendable {
        public var name: String
        public var maxPositionEmbeddings: Int

        enum CodingKeys: String, CodingKey {
            case name
            case maxPositionEmbeddings = "max_position_embeddings"
        }
    }

    public struct SchedulerInfo: Codable, Equatable, Sendable {
        public var bMax: Int
        public var bActive: Int
        public var bQueued: Int
        public var queueMax: Int
        public var admissionQueueFullCount: UInt64
        public var memoryBudgetExceededCount: UInt64

        enum CodingKeys: String, CodingKey {
            case bMax = "b_max"
            case bActive = "b_active"
            case bQueued = "b_queued"
            case queueMax = "queue_max"
            case admissionQueueFullCount = "admission_queue_full_count"
            case memoryBudgetExceededCount = "memory_budget_exceeded_count"
        }
    }

    public struct MemoryInfo: Codable, Equatable, Sendable {
        public var totalRamBytes: UInt64
        public var freeRamBytes: UInt64
        public var kvCacheActiveBytes: UInt64
        public var kvCacheSoftLimitBytes: UInt64
        public var kvCacheLogicalCapTokens: Int
        public var kvCacheResidentCapTokens: Int
        public var kvCacheBudgetPolicy: String
        public var mlxTotalBytes: UInt64?
        public var mlxMaxRecommendedBytes: UInt64?
        public var mlxActiveBytes: UInt64
        public var mlxCacheBytes: UInt64
        public var mlxPeakBytes: UInt64
        public var mlxMemoryLimitBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case totalRamBytes = "total_ram_bytes"
            case freeRamBytes = "free_ram_bytes"
            case kvCacheActiveBytes = "kv_cache_active_bytes"
            case kvCacheSoftLimitBytes = "kv_cache_soft_limit_bytes"
            case kvCacheLogicalCapTokens = "kv_cache_logical_cap_tokens"
            case kvCacheResidentCapTokens = "kv_cache_resident_cap_tokens"
            case kvCacheBudgetPolicy = "kv_cache_budget_policy"
            case mlxTotalBytes = "mlx_total_bytes"
            case mlxMaxRecommendedBytes = "mlx_max_recommended_bytes"
            case mlxActiveBytes = "mlx_active_bytes"
            case mlxCacheBytes = "mlx_cache_bytes"
            case mlxPeakBytes = "mlx_peak_bytes"
            case mlxMemoryLimitBytes = "mlx_memory_limit_bytes"
        }
    }

    public struct ActiveKvOffloadInfo: Codable, Equatable, Sendable {
        public var enabled: Bool
        public var status: String
        public var active: Bool
        public var degraded: Bool
        public var mode: String
        public var storageDir: String?
        public var residentPages: UInt64
        public var offloadedPages: UInt64
        public var loadingPages: UInt64
        public var dirtyPages: UInt64
        public var parkedRequests: UInt64
        public var offloadedBytes: UInt64
        public var swapOutCount: UInt64
        public var swapInCount: UInt64
        public var swapErrorCount: UInt64
        public var lastSwapOutUs: UInt64
        public var lastSwapInUs: UInt64
        public var supportedCacheKinds: [String]
        public var notApplicableCacheKinds: [String]

        enum CodingKeys: String, CodingKey {
            case enabled
            case status
            case active
            case degraded
            case mode
            case storageDir = "storage_dir"
            case residentPages = "resident_pages"
            case offloadedPages = "offloaded_pages"
            case loadingPages = "loading_pages"
            case dirtyPages = "dirty_pages"
            case parkedRequests = "parked_requests"
            case offloadedBytes = "offloaded_bytes"
            case swapOutCount = "swap_out_count"
            case swapInCount = "swap_in_count"
            case swapErrorCount = "swap_error_count"
            case lastSwapOutUs = "last_swap_out_us"
            case lastSwapInUs = "last_swap_in_us"
            case supportedCacheKinds = "supported_cache_kinds"
            case notApplicableCacheKinds = "not_applicable_cache_kinds"
        }
    }

    public struct DFlash2Info: Codable, Equatable, Sendable {
        public var enabled: Bool
        public var blockSize: Int?
        public var draftQuantizationBits: Int?
        public var requests: UInt64
        public var windows: UInt64
        public var draftedTokens: UInt64
        public var acceptedDraftTokens: UInt64
        public var rollbackCount: UInt64
        public var tensorBatchWindows: UInt64
        public var tensorBatchDivergentSplits: UInt64
        public var tensorBatchGroupsCreated: UInt64
        public var tensorBatchWidthLimit: Int
        public var tensorBatchMaxWidth: Int
        public var sampledRequests: UInt64
        public var exactSamplingWindows: UInt64
        public var exactAcceptanceDraws: UInt64
        public var exactResidualCorrections: UInt64
        public var exactBonusSamples: UInt64
        public var samplingUs: UInt64
        public var latestGenerationTPS: Double
        public var latestAcceptanceRate: Double
        public var peakMemoryBytes: UInt64
        public var prefixCacheEnabled: Bool
        public var prefixCacheMaxBytes: UInt64?
        public var prefixCacheEntries: UInt64
        public var prefixCacheBytes: UInt64
        public var prefixCacheHits: UInt64
        public var prefixCacheMisses: UInt64
        public var prefixCacheSaves: UInt64
        public var prefixCacheEvictions: UInt64
        public var prefixCacheHitTokens: UInt64
        public var runtimeUsage: BackendModelRuntimeUsage

        enum CodingKeys: String, CodingKey {
            case enabled
            case blockSize = "block_size"
            case draftQuantizationBits = "draft_quantization_bits"
            case requests
            case windows
            case draftedTokens = "drafted_tokens"
            case acceptedDraftTokens = "accepted_draft_tokens"
            case rollbackCount = "rollback_count"
            case tensorBatchWindows = "tensor_batch_windows"
            case tensorBatchDivergentSplits = "tensor_batch_divergent_splits"
            case tensorBatchGroupsCreated = "tensor_batch_groups_created"
            case tensorBatchWidthLimit = "tensor_batch_width_limit"
            case tensorBatchMaxWidth = "tensor_batch_max_width"
            case sampledRequests = "sampled_requests"
            case exactSamplingWindows = "exact_sampling_windows"
            case exactAcceptanceDraws = "exact_acceptance_draws"
            case exactResidualCorrections = "exact_residual_corrections"
            case exactBonusSamples = "exact_bonus_samples"
            case samplingUs = "sampling_us"
            case latestGenerationTPS = "latest_generation_tps"
            case latestAcceptanceRate = "latest_acceptance_rate"
            case peakMemoryBytes = "peak_memory_bytes"
            case prefixCacheEnabled = "prefix_cache_enabled"
            case prefixCacheMaxBytes = "prefix_cache_max_bytes"
            case prefixCacheEntries = "prefix_cache_entries"
            case prefixCacheBytes = "prefix_cache_bytes"
            case prefixCacheHits = "prefix_cache_hits"
            case prefixCacheMisses = "prefix_cache_misses"
            case prefixCacheSaves = "prefix_cache_saves"
            case prefixCacheEvictions = "prefix_cache_evictions"
            case prefixCacheHitTokens = "prefix_cache_hit_tokens"
            case runtimeUsage = "runtime_usage"
        }
    }
}

public struct LegacyHealthStatus: Codable, Equatable, Sendable {
    public var startedAt: UInt64
    public var model: String
    public var memory: LegacyMemory
    public var totalTokens: UInt64
    public var cachedTokens: UInt64
    public var cacheHitRate: Double
    public var activeKvOffload: HealthzSnapshot.ActiveKvOffloadInfo
    public var dflash2: HealthzSnapshot.DFlash2Info?
    public var deviceName: String?
    public var runtimeModels: [BackendLoadedModelInfo]

    enum CodingKeys: String, CodingKey {
        case startedAt = "started_at"
        case model
        case memory
        case totalTokens = "total_tokens"
        case cachedTokens = "cached_tokens"
        case cacheHitRate = "cache_hit_rate"
        case activeKvOffload = "active_kv_offload"
        case dflash2
        case deviceName = "device_name"
        case runtimeModels = "runtime_models"
    }
}

public struct LegacyMemory: Codable, Equatable, Sendable {
    public var activeMB: Double
    public var cacheMB: Double
    public var peakMB: Double
    public var totalMB: Double?
    public var maxMB: Double?

    enum CodingKeys: String, CodingKey {
        case activeMB = "active_mb"
        case cacheMB = "cache_mb"
        case peakMB = "peak_mb"
        case totalMB = "total_mb"
        case maxMB = "max_mb"
    }
}

public struct LegacyHealthAdapter {
    public var statusNow: Date

    public init(statusNow: Date = Date()) {
        self.statusNow = statusNow
    }

    public func legacyStatus(from snapshot: HealthzSnapshot) -> LegacyHealthStatus {
        let nowSeconds = UInt64(statusNow.timeIntervalSince1970)
        let startedAt = nowSeconds > snapshot.uptimeSecs ? nowSeconds - snapshot.uptimeSecs : 0
        let memory = snapshot.memory

        var runtimeModels = snapshot.models
        if runtimeModels.isEmpty, let dflash2 = snapshot.dflash2, dflash2.enabled {
            runtimeModels = [
                BackendLoadedModelInfo(
                    id: snapshot.model.name,
                    model: snapshot.model.name,
                    path: "",
                    architecture: "qwen3_5",
                    isDefault: true,
                    maxPositionEmbeddings: snapshot.model.maxPositionEmbeddings,
                    dflash2: dflash2,
                    capabilities: BackendModelCapabilities(
                        runtimeKind: "causal",
                        supportsStreaming: true,
                        supportsVision: false,
                        supportsMtp: false,
                        supportsPromptLookup: false,
                        supportsSpeculativeDecoding: true,
                        supportsKvCache: true,
                        supportedSamplingParameters: [
                            "max_tokens", "temperature", "top_p", "top_k",
                            "repetition_penalty", "seed",
                        ]
                    ),
                    scheduler: "dflash2",
                    activeRequests: snapshot.scheduler.bActive,
                    queuedRequests: snapshot.scheduler.bQueued,
                    queueCapacity: snapshot.scheduler.queueMax,
                    usage: dflash2.runtimeUsage
                ),
            ]
        }
        return LegacyHealthStatus(
            startedAt: startedAt,
            model: snapshot.model.name,
            memory: LegacyMemory(
                activeMB: bytesToMegabytes(memory.mlxActiveBytes),
                cacheMB: bytesToMegabytes(memory.mlxCacheBytes),
                peakMB: bytesToMegabytes(memory.mlxPeakBytes),
                totalMB: memory.mlxTotalBytes.map(bytesToMegabytes),
                maxMB: memory.mlxMaxRecommendedBytes.map(bytesToMegabytes)
            ),
            totalTokens: 0,
            cachedTokens: 0,
            cacheHitRate: 0,
            activeKvOffload: snapshot.activeKvOffload,
            dflash2: snapshot.dflash2,
            deviceName: snapshot.deviceName,
            runtimeModels: runtimeModels
        )
    }

    private func bytesToMegabytes(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_048_576.0
    }
}
