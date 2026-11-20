import Foundation

/// A string enum that decodes values it does not know as `unknown`, so a newer backend
/// never breaks an older App.
public protocol RuntimeMetricsLenientEnum: RawRepresentable, Decodable where RawValue == String {
    static var unknown: Self { get }
}

public extension RuntimeMetricsLenientEnum {
    init(from decoder: any Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? .unknown
    }
}

public enum RuntimeActivityState: String, RuntimeMetricsLenientEnum, Equatable, Sendable {
    case idle, prefilling, generating, loading, unknown
}

public enum RuntimeHealthStatus: String, RuntimeMetricsLenientEnum, Equatable, Sendable {
    case healthy, degraded, down, unknown
}

public enum RuntimePressureLevel: String, RuntimeMetricsLenientEnum, Equatable, Sendable {
    case normal, soft, hard, emergency, unknown
}

public enum RuntimeModelState: String, RuntimeMetricsLenientEnum, Equatable, Sendable {
    case loading, loaded, failed, unknown
}

/// One `metrics` event from `GET /admin/api/metrics/stream` (schema version 1).
///
/// Every field except `schemaVersion` may be missing; missing values decode as 0, empty or nil.
public struct RuntimeMetricsSnapshot: Decodable, Equatable, Sendable {
    public struct Backend: Decodable, Equatable, Sendable {
        public var version: String
        public var startedUnixMs: Int64
        public var deviceName: String

        public init(version: String = "", startedUnixMs: Int64 = 0, deviceName: String = "") {
            self.version = version
            self.startedUnixMs = startedUnixMs
            self.deviceName = deviceName
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                version: try c.decodeIfPresent(String.self, forKey: .version) ?? "",
                startedUnixMs: try c.decodeIfPresent(Int64.self, forKey: .startedUnixMs) ?? 0,
                deviceName: try c.decodeIfPresent(String.self, forKey: .deviceName) ?? ""
            )
        }

        private enum CodingKeys: String, CodingKey {
            case version, startedUnixMs, deviceName
        }
    }

    public struct Health: Decodable, Equatable, Sendable {
        public var status: RuntimeHealthStatus
        /// Raw `HealthDegradedReason` names, kept verbatim so unknown reasons can still be shown.
        public var degradedReasons: [String]

        public init(status: RuntimeHealthStatus = .unknown, degradedReasons: [String] = []) {
            self.status = status
            self.degradedReasons = degradedReasons
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                status: try c.decodeIfPresent(RuntimeHealthStatus.self, forKey: .status) ?? .unknown,
                degradedReasons: try c.decodeIfPresent([String].self, forKey: .degradedReasons) ?? []
            )
        }

        private enum CodingKeys: String, CodingKey {
            case status, degradedReasons
        }
    }

    public struct Activity: Decodable, Equatable, Sendable {
        public var state: RuntimeActivityState
        public var activeRequests: Int
        public var prefillingRequests: Int
        public var queuedRequests: Int

        public init(
            state: RuntimeActivityState = .unknown,
            activeRequests: Int = 0,
            prefillingRequests: Int = 0,
            queuedRequests: Int = 0
        ) {
            self.state = state
            self.activeRequests = activeRequests
            self.prefillingRequests = prefillingRequests
            self.queuedRequests = queuedRequests
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                state: try c.decodeIfPresent(RuntimeActivityState.self, forKey: .state) ?? .unknown,
                activeRequests: try c.decodeIfPresent(Int.self, forKey: .activeRequests) ?? 0,
                prefillingRequests: try c.decodeIfPresent(Int.self, forKey: .prefillingRequests) ?? 0,
                queuedRequests: try c.decodeIfPresent(Int.self, forKey: .queuedRequests) ?? 0
            )
        }

        private enum CodingKeys: String, CodingKey {
            case state, activeRequests, prefillingRequests, queuedRequests
        }
    }

    public struct Performance: Decodable, Equatable, Sendable {
        public var liveDecodeTokensPerSecond: Double?
        public var prefillTokensPerSecond: Double?
        public var decodeTokensPerSecond: Double?
        public var ttftMs: Double?
        public var windowSeconds: Int

        public init(
            liveDecodeTokensPerSecond: Double? = nil,
            prefillTokensPerSecond: Double? = nil,
            decodeTokensPerSecond: Double? = nil,
            ttftMs: Double? = nil,
            windowSeconds: Int = 0
        ) {
            self.liveDecodeTokensPerSecond = liveDecodeTokensPerSecond
            self.prefillTokensPerSecond = prefillTokensPerSecond
            self.decodeTokensPerSecond = decodeTokensPerSecond
            self.ttftMs = ttftMs
            self.windowSeconds = windowSeconds
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                liveDecodeTokensPerSecond: try c.decodeIfPresent(Double.self, forKey: .liveDecodeTokensPerSecond),
                prefillTokensPerSecond: try c.decodeIfPresent(Double.self, forKey: .prefillTokensPerSecond),
                decodeTokensPerSecond: try c.decodeIfPresent(Double.self, forKey: .decodeTokensPerSecond),
                ttftMs: try c.decodeIfPresent(Double.self, forKey: .ttftMs),
                windowSeconds: try c.decodeIfPresent(Int.self, forKey: .windowSeconds) ?? 0
            )
        }

        private enum CodingKeys: String, CodingKey {
            case liveDecodeTokensPerSecond, prefillTokensPerSecond, decodeTokensPerSecond, ttftMs, windowSeconds
        }
    }

    public struct Memory: Decodable, Equatable, Sendable {
        public var pressureLevel: RuntimePressureLevel
        public var processBytes: Int64
        public var ceilingBytes: Int64
        public var kvCacheBytes: Int64
        public var mlxCacheBytes: Int64
        public var telemetryDegraded: Bool

        public init(
            pressureLevel: RuntimePressureLevel = .unknown,
            processBytes: Int64 = 0,
            ceilingBytes: Int64 = 0,
            kvCacheBytes: Int64 = 0,
            mlxCacheBytes: Int64 = 0,
            telemetryDegraded: Bool = false
        ) {
            self.pressureLevel = pressureLevel
            self.processBytes = processBytes
            self.ceilingBytes = ceilingBytes
            self.kvCacheBytes = kvCacheBytes
            self.mlxCacheBytes = mlxCacheBytes
            self.telemetryDegraded = telemetryDegraded
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                pressureLevel: try c.decodeIfPresent(RuntimePressureLevel.self, forKey: .pressureLevel) ?? .unknown,
                processBytes: try c.decodeIfPresent(Int64.self, forKey: .processBytes) ?? 0,
                ceilingBytes: try c.decodeIfPresent(Int64.self, forKey: .ceilingBytes) ?? 0,
                kvCacheBytes: try c.decodeIfPresent(Int64.self, forKey: .kvCacheBytes) ?? 0,
                mlxCacheBytes: try c.decodeIfPresent(Int64.self, forKey: .mlxCacheBytes) ?? 0,
                telemetryDegraded: try c.decodeIfPresent(Bool.self, forKey: .telemetryDegraded) ?? false
            )
        }

        private enum CodingKeys: String, CodingKey {
            case pressureLevel, processBytes, ceilingBytes, kvCacheBytes, mlxCacheBytes, telemetryDegraded
        }
    }

    public struct PrefixCache: Decodable, Equatable, Sendable {
        public var hitTokens: Int64
        public var eligibleTokens: Int64

        public init(hitTokens: Int64 = 0, eligibleTokens: Int64 = 0) {
            self.hitTokens = hitTokens
            self.eligibleTokens = eligibleTokens
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                hitTokens: try c.decodeIfPresent(Int64.self, forKey: .hitTokens) ?? 0,
                eligibleTokens: try c.decodeIfPresent(Int64.self, forKey: .eligibleTokens) ?? 0
            )
        }

        private enum CodingKeys: String, CodingKey {
            case hitTokens, eligibleTokens
        }
    }

    public struct Model: Decodable, Equatable, Sendable {
        public var id: String
        /// Backend `runtime_kind`: `causal`, `block_diffusion`, `tts`, `asr`, ... kept as a raw string.
        public var kind: String
        public var state: RuntimeModelState
        /// `normal` / `dflash2` / `mtp`; only present for `causal` models.
        public var mode: String?
        public var isDefault: Bool
        public var activeRequests: Int
        public var queuedRequests: Int
        public var queueCapacity: Int?
        public var liveDecodeTokensPerSecond: Double?
        public var lastError: String?

        public init(
            id: String = "",
            kind: String = "",
            state: RuntimeModelState = .unknown,
            mode: String? = nil,
            isDefault: Bool = false,
            activeRequests: Int = 0,
            queuedRequests: Int = 0,
            queueCapacity: Int? = nil,
            liveDecodeTokensPerSecond: Double? = nil,
            lastError: String? = nil
        ) {
            self.id = id
            self.kind = kind
            self.state = state
            self.mode = mode
            self.isDefault = isDefault
            self.activeRequests = activeRequests
            self.queuedRequests = queuedRequests
            self.queueCapacity = queueCapacity
            self.liveDecodeTokensPerSecond = liveDecodeTokensPerSecond
            self.lastError = lastError
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                id: try c.decodeIfPresent(String.self, forKey: .id) ?? "",
                kind: try c.decodeIfPresent(String.self, forKey: .kind) ?? "",
                state: try c.decodeIfPresent(RuntimeModelState.self, forKey: .state) ?? .unknown,
                mode: try c.decodeIfPresent(String.self, forKey: .mode),
                isDefault: try c.decodeIfPresent(Bool.self, forKey: .isDefault) ?? false,
                activeRequests: try c.decodeIfPresent(Int.self, forKey: .activeRequests) ?? 0,
                queuedRequests: try c.decodeIfPresent(Int.self, forKey: .queuedRequests) ?? 0,
                queueCapacity: try c.decodeIfPresent(Int.self, forKey: .queueCapacity),
                liveDecodeTokensPerSecond: try c.decodeIfPresent(Double.self, forKey: .liveDecodeTokensPerSecond),
                lastError: try c.decodeIfPresent(String.self, forKey: .lastError)
            )
        }

        private enum CodingKeys: String, CodingKey {
            case id, kind, state, mode, isDefault, activeRequests, queuedRequests, queueCapacity
            case liveDecodeTokensPerSecond, lastError
        }
    }

    public var schemaVersion: Int
    public var seq: UInt64
    public var unixMs: Int64
    public var backend: Backend
    public var health: Health
    public var activity: Activity
    public var performance: Performance
    public var memory: Memory
    public var prefixCache: PrefixCache
    public var models: [Model]

    public init(
        schemaVersion: Int = 1,
        seq: UInt64 = 0,
        unixMs: Int64 = 0,
        backend: Backend = Backend(),
        health: Health = Health(),
        activity: Activity = Activity(),
        performance: Performance = Performance(),
        memory: Memory = Memory(),
        prefixCache: PrefixCache = PrefixCache(),
        models: [Model] = []
    ) {
        self.schemaVersion = schemaVersion
        self.seq = seq
        self.unixMs = unixMs
        self.backend = backend
        self.health = health
        self.activity = activity
        self.performance = performance
        self.memory = memory
        self.prefixCache = prefixCache
        self.models = models
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            schemaVersion: try c.decode(Int.self, forKey: .schemaVersion),
            seq: try c.decodeIfPresent(UInt64.self, forKey: .seq) ?? 0,
            unixMs: try c.decodeIfPresent(Int64.self, forKey: .unixMs) ?? 0,
            backend: try c.decodeIfPresent(Backend.self, forKey: .backend) ?? Backend(),
            health: try c.decodeIfPresent(Health.self, forKey: .health) ?? Health(),
            activity: try c.decodeIfPresent(Activity.self, forKey: .activity) ?? Activity(),
            performance: try c.decodeIfPresent(Performance.self, forKey: .performance) ?? Performance(),
            memory: try c.decodeIfPresent(Memory.self, forKey: .memory) ?? Memory(),
            prefixCache: try c.decodeIfPresent(PrefixCache.self, forKey: .prefixCache) ?? PrefixCache(),
            models: try c.decodeIfPresent([Model].self, forKey: .models) ?? []
        )
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, seq, unixMs, backend, health, activity, performance, memory, prefixCache, models
    }

    /// The decoder for the snake_case wire format.
    public static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    public static func decode(from data: Data) throws -> RuntimeMetricsSnapshot {
        try makeDecoder().decode(RuntimeMetricsSnapshot.self, from: data)
    }
}
