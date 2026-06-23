import Foundation

public struct HealthzSnapshot: Codable, Equatable, Sendable {
    public var status: String
    public var uptimeSecs: UInt64
    public var model: ModelInfo
    public var scheduler: SchedulerInfo
    public var memory: MemoryInfo
    public var deviceName: String?
    public var version: String

    enum CodingKeys: String, CodingKey {
        case status
        case uptimeSecs = "uptime_secs"
        case model
        case scheduler
        case memory
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
            case mlxTotalBytes = "mlx_total_bytes"
            case mlxMaxRecommendedBytes = "mlx_max_recommended_bytes"
            case mlxActiveBytes = "mlx_active_bytes"
            case mlxCacheBytes = "mlx_cache_bytes"
            case mlxPeakBytes = "mlx_peak_bytes"
            case mlxMemoryLimitBytes = "mlx_memory_limit_bytes"
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
    public var deviceName: String?

    enum CodingKeys: String, CodingKey {
        case startedAt = "started_at"
        case model
        case memory
        case totalTokens = "total_tokens"
        case cachedTokens = "cached_tokens"
        case cacheHitRate = "cache_hit_rate"
        case deviceName = "device_name"
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
            deviceName: snapshot.deviceName
        )
    }

    private func bytesToMegabytes(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_048_576.0
    }
}
