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
    public var paging: BackendPagingInfo? = nil
    public var prefixPersistence: PrefixPersistenceInfo? = nil
    public var ane: AneInfo? = nil
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
        case paging
        case prefixPersistence = "prefix_persistence"
        case ane
        case activeKvOffload = "active_kv_offload"
        case deviceName = "device_name"
        case version
    }

    public struct AneInfo: Codable, Equatable, Sendable {
        public var runtimeError: String? = nil
        public var currentModel: AneModelInfo? = nil
        public var mode: String
        public var chip: String?
        public var qualified: Bool
        public var available: Bool
        public var reason: String
        public var models: [AneModelInfo]
        enum CodingKeys: String, CodingKey {
            case runtimeError = "runtime_error"
            case currentModel = "current_model"
            case mode, chip, qualified, available, reason, models
        }
    }
    public struct AneModelInfo: Codable, Equatable, Sendable {
        public var liveSurfaceBytes: UInt64? = nil
        public var model: String
        public var reason: String
        public var completedJobs: UInt64
        enum CodingKeys: String, CodingKey {
            case model, reason
            case completedJobs = "completed_jobs"
            case liveSurfaceBytes = "live_surface_bytes"
        }
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
        public var checkpointBlockSize: Int?
        public var blockSize: Int?
        public var maxDraftTokens: Int?
        public var draftBudgetChanges: UInt64?
        public var currentDraftBudget: Int?
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
        public var treeMaxNodes: Int? = nil
        public var m5Profile: M5ProfileInfo? = nil

        enum CodingKeys: String, CodingKey {
            case enabled
            case checkpointBlockSize = "checkpoint_block_size"
            case blockSize = "block_size"
            case maxDraftTokens = "max_draft_tokens"
            case draftBudgetChanges = "draft_budget_changes"
            case currentDraftBudget = "current_draft_budget"
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
            case treeMaxNodes = "tree_max_nodes"
            case m5Profile = "m5_profile"
        }
    }

    /// Process DFlash2 profile for Apple GPU generation 17+ (M5).
    public struct M5ProfileInfo: Codable, Equatable, Sendable {
        public var installed: Bool
        /// `auto` or `off`.
        public var mode: String?
        /// `active`, `disabled`, `not_dflash2`, `unsupported_gpu` or `gpu_unknown`.
        public var status: String?
        public var architecture: String?
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
    public var prefixPersistence: PrefixPersistenceInfo? = nil
    public var ane: HealthzSnapshot.AneInfo? = nil
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
        case ane
        case prefixPersistence = "prefix_persistence"
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
                    ane: snapshot.ane?.currentModel,
                    paging: snapshot.paging,
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
            prefixPersistence: snapshot.prefixPersistence,
            ane: snapshot.ane,
            deviceName: snapshot.deviceName,
            runtimeModels: runtimeModels
        )
    }

    private func bytesToMegabytes(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_048_576.0
    }
}

public struct BackendPagingPhase: Codable, Equatable, Sendable {
    public var arithmeticSplits: UInt64? = nil
    public var requestWidth: UInt64? = nil
    public var budgetSplits: UInt64? = nil
    public var prefetchScheduled: UInt64? = nil
    public var prefetchUnused: UInt64? = nil
    public var phase: String
    public var hitRate: Double?
    public var hits: UInt64
    public var misses: UInt64
    public var evictions: UInt64
    public var readBytes: UInt64
    public var readUs: UInt64
    public var packUs: UInt64
    public var lockWaitUs: UInt64
    public var routeCount: UInt64
    public var batches: UInt64
    public var microbatches: UInt64
    public var maxWidth: UInt64
    public var scanAdmissions: UInt64
    public var prefetchHits: UInt64
    enum CodingKeys: String, CodingKey {
        case arithmeticSplits = "arithmetic_splits"
        case requestWidth = "request_width"
        case budgetSplits = "budget_splits"
        case prefetchScheduled = "prefetch_scheduled"
        case prefetchUnused = "prefetch_unused"
        case phase
        case hitRate = "hit_rate"
        case hits
        case misses
        case evictions
        case readBytes = "read_bytes"
        case readUs = "read_us"
        case packUs = "pack_us"
        case lockWaitUs = "lock_wait_us"
        case routeCount = "route_count"
        case batches
        case microbatches
        case maxWidth = "max_width"
        case scanAdmissions = "scan_admissions"
        case prefetchHits = "prefetch_hits"
    }
}
public struct BackendPagingCache: Codable, Equatable, Sendable {
    public var pendingBytes: UInt64? = nil
    public var budgetBytes: UInt64
    public var chargedBytes: UInt64
    public var payloadBytes: UInt64
    public var peakChargedBytes: UInt64
    public var entries: UInt64
    public var pinnedBytes: UInt64
    public var phases: [BackendPagingPhase]
    enum CodingKeys: String, CodingKey {
        case pendingBytes = "pending_bytes"
        case budgetBytes = "budget_bytes"
        case chargedBytes = "charged_bytes"
        case payloadBytes = "payload_bytes"
        case peakChargedBytes = "peak_charged_bytes"
        case entries
        case pinnedBytes = "pinned_bytes"
        case phases
    }
}
public struct BackendPagingFallback: Codable, Equatable, Sendable {
    public var reason: String
    public var calls: UInt64
}
public struct BackendPagingInfo: Codable, Equatable, Sendable {
    public var fallbacks: [BackendPagingFallback]? = nil
    public var instanceId: String
    public var mode: String
    public var startedUnixMs: UInt64
    public var sampledUnixMs: UInt64
    public var elapsedMs: UInt64
    public var experts: BackendPagingCache
    public var ple: BackendPagingCache
    enum CodingKeys: String, CodingKey {
        case fallbacks
        case instanceId = "instance_id"
        case mode
        case startedUnixMs = "started_unix_ms"
        case sampledUnixMs = "sampled_unix_ms"
        case elapsedMs = "elapsed_ms"
        case experts
        case ple
    }
}

/// Process-wide optional snapshot; older backends omit this field.
public struct PrefixPersistenceInfo: Codable, Equatable, Sendable {
    public struct Profile: Codable, Equatable, Sendable {
        public var tier: String
        public var budgetBytes: UInt64
        public var maxJobs: Int
        public var chunkBytes: UInt64
        enum CodingKeys: String, CodingKey { case tier; case budgetBytes = "budget_bytes"; case maxJobs = "max_jobs"; case chunkBytes = "chunk_bytes" }
    }
    public var profile: Profile
    public var scope: String
    public var pendingJobs: Int
    public var reservedPeakBytes: UInt64
    public var frozenBytes: UInt64
    public var queuedTotal: UInt64
    public var committedTotal: UInt64
    public var skippedTotal: UInt64
    public var cancelledTotal: UInt64
    public var failedTotal: UInt64
    public var hotMemoryHits: UInt64
    public var pendingSnapshotHits: UInt64
    public var ssdHits: UInt64
    public var lastStatus: String
    public var lastReason: String
    public var lastMode: String
    public var lastModel: String
    public var throttled: Bool
    enum CodingKeys: String, CodingKey {
        case profile, scope, throttled
        case pendingJobs = "pending_jobs", reservedPeakBytes = "reserved_peak_bytes", frozenBytes = "frozen_bytes"
        case queuedTotal = "queued_total", committedTotal = "committed_total", skippedTotal = "skipped_total"
        case cancelledTotal = "cancelled_total", failedTotal = "failed_total"
        case hotMemoryHits = "hot_memory_hits", pendingSnapshotHits = "pending_snapshot_hits", ssdHits = "ssd_hits"
        case lastStatus = "last_status", lastReason = "last_reason", lastMode = "last_mode", lastModel = "last_model"
    }
}
