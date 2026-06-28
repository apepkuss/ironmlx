import Foundation

public struct BenchmarkRequest: Codable, Equatable, Sendable {
    public var model: String
    public var modelPath: String
    public var promptTokens: Int
    public var maxTokens: Int
    public var batchSize: Int

    public init(
        model: String,
        modelPath: String,
        promptTokens: Int,
        maxTokens: Int,
        batchSize: Int
    ) {
        self.model = model
        self.modelPath = modelPath
        self.promptTokens = max(1, promptTokens)
        self.maxTokens = max(1, maxTokens)
        self.batchSize = max(1, batchSize)
    }

    enum CodingKeys: String, CodingKey {
        case model
        case modelPath = "model_path"
        case promptTokens = "prompt_tokens"
        case maxTokens = "max_tokens"
        case batchSize = "batch_size"
    }
}

public struct BenchmarkPlan: Equatable, Sendable {
    public var processURL: URL
    public var arguments: [String]

    public init(
        ironBenchURL: URL,
        request: BenchmarkRequest,
        host: String,
        port: UInt16,
        durationSeconds: Int = 10
    ) {
        self.processURL = ironBenchURL
        let targetHost = BackendAPIClient.connectableHost(for: host)
        var arguments = [
            "--target", "ironmlx=http://\(targetHost):\(port)",
            "--model-dir", request.modelPath,
            "--model", request.model,
            "--prompt-len", String(request.promptTokens),
            "--max-tokens", String(request.maxTokens),
            "--format", "json",
            "--timeout", "300",
        ]
        if request.batchSize <= 1 {
            arguments += [
                "--runs", "1",
                "--warmup", "0",
            ]
        } else {
            arguments += [
                "--concurrent", String(request.batchSize),
                "--duration", String(max(1, durationSeconds)),
                "--warmup-duration", "0",
            ]
        }
        if ironBenchURL.path == "/usr/bin/env" {
            self.arguments = ["iron-bench"] + arguments
        } else {
            self.arguments = arguments
        }
    }
}

public struct BenchmarkResult: Codable, Equatable, Sendable {
    public var success: Bool
    public var model: String
    public var batchSize: Int
    public var promptTokens: Int
    public var maxTokens: Int
    public var ttftMs: Double
    public var tpotMs: Double
    public var tgTps: Double
    public var ppTps: Double
    public var totalMs: Double
    public var totalThroughput: Double
    public var memoryPeakMB: Double?
    public var requestCount: Int?
    public var mode: String

    enum CodingKeys: String, CodingKey {
        case success
        case model
        case batchSize = "batch_size"
        case promptTokens = "prompt_tokens"
        case maxTokens = "max_tokens"
        case ttftMs = "ttft_ms"
        case tpotMs = "tpot_ms"
        case tgTps = "tg_tps"
        case ppTps = "pp_tps"
        case totalMs = "total_ms"
        case totalThroughput = "total_throughput"
        case memoryPeakMB = "memory_peak_mb"
        case requestCount = "request_count"
        case mode
    }

    public static func parse(
        ironBenchJSON data: Data,
        request: BenchmarkRequest,
        memoryPeakMB: Double?
    ) throws -> BenchmarkResult {
        if let concurrent = try? JSONDecoder().decode(ConcurrentIronBenchOutput.self, from: data),
           let cell = concurrent.cells.first {
            let ttft = cell.ttftMs.p50
            let tpot = cell.itlMs.p50
            let throughput = cell.aggregate.tokensPerSec
            let ppTps = ttft > 0 ? Double(request.promptTokens) / (ttft / 1000.0) : 0
            return BenchmarkResult(
                success: true,
                model: request.model,
                batchSize: request.batchSize,
                promptTokens: request.promptTokens,
                maxTokens: request.maxTokens,
                ttftMs: ttft,
                tpotMs: tpot,
                tgTps: throughput,
                ppTps: ppTps,
                totalMs: cell.wallDurationSeconds * 1000.0,
                totalThroughput: throughput,
                memoryPeakMB: memoryPeakMB,
                requestCount: cell.requestCount,
                mode: "concurrent"
            )
        }

        let sequential = try JSONDecoder().decode(SequentialIronBenchOutput.self, from: data)
        guard let stats = sequential.stats.first else {
            throw BenchmarkError.missingStats
        }
        return BenchmarkResult(
            success: true,
            model: request.model,
            batchSize: request.batchSize,
            promptTokens: request.promptTokens,
            maxTokens: request.maxTokens,
            ttftMs: stats.ttftMsMedian,
            tpotMs: stats.tpotMsMedian,
            tgTps: stats.tgTpsMedian,
            ppTps: stats.ppTpsMedian,
            totalMs: stats.e2eSecondsMedian * 1000.0,
            totalThroughput: stats.tgTpsMedian,
            memoryPeakMB: memoryPeakMB,
            requestCount: stats.runCount,
            mode: "sequential"
        )
    }
}

public struct BenchmarkService: Sendable {
    private let ironBenchURL: URL
    private let durationSeconds: Int

    public init(
        ironBenchURL: URL? = nil,
        durationSeconds: Int = 10
    ) {
        let backendURL = BackendBinaryResolver.resolve()
        self.ironBenchURL = ironBenchURL
            ?? BackendBinaryResolver.resolveIronBenchBinary(near: backendURL)
            ?? URL(fileURLWithPath: "/usr/bin/env")
        self.durationSeconds = durationSeconds
    }

    public func run(
        request: BenchmarkRequest,
        host: String,
        port: UInt16,
        client: BackendAPIClient
    ) async throws -> BenchmarkResult {
        let plan = BenchmarkPlan(
            ironBenchURL: ironBenchURL,
            request: request,
            host: host,
            port: port,
            durationSeconds: durationSeconds
        )
        let process = Process()
        process.executableURL = plan.processURL
        process.arguments = plan.arguments
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()
        var memoryPeakMB: Double?
        while process.isRunning {
            memoryPeakMB = await sampleMemoryPeakMB(client: client, current: memoryPeakMB)
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        process.waitUntilExit()
        memoryPeakMB = await sampleMemoryPeakMB(client: client, current: memoryPeakMB)

        let output = stdout.fileHandleForReading.readDataToEndOfFile()
        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
        if process.terminationStatus != 0 {
            let detail = String(data: errorData, encoding: .utf8)
                ?? String(data: output, encoding: .utf8)
                ?? "iron-bench failed"
            throw BenchmarkError.processFailed(status: process.terminationStatus, detail: detail)
        }
        return try BenchmarkResult.parse(
            ironBenchJSON: output,
            request: request,
            memoryPeakMB: memoryPeakMB
        )
    }

    private func sampleMemoryPeakMB(client: BackendAPIClient, current: Double?) async -> Double? {
        guard let snapshot = try? await client.fetchHealthz() else {
            return current
        }
        let active = Double(snapshot.memory.mlxActiveBytes) / 1_048_576.0
        let peak = Double(snapshot.memory.mlxPeakBytes) / 1_048_576.0
        return max(current ?? 0, active, peak)
    }
}

public enum BenchmarkError: LocalizedError, Equatable {
    case missingStats
    case processFailed(status: Int32, detail: String)

    public var errorDescription: String? {
        switch self {
        case .missingStats:
            return "iron-bench did not return benchmark stats."
        case .processFailed(let status, let detail):
            return "iron-bench exited with status \(status): \(detail)"
        }
    }
}

private struct SequentialIronBenchOutput: Decodable {
    var stats: [SequentialStats]
}

private struct SequentialStats: Decodable {
    var runCount: Int?
    var ttftMsMedian: Double
    var tgTpsMedian: Double
    var tpotMsMedian: Double
    var ppTpsMedian: Double
    var e2eSecondsMedian: Double

    enum CodingKeys: String, CodingKey {
        case runCount = "n_runs"
        case ttftMsMedian = "ttft_ms_median"
        case tgTpsMedian = "tg_tps_median"
        case tpotMsMedian = "tpot_ms_median"
        case ppTpsMedian = "pp_tps_median"
        case e2eSecondsMedian = "e2e_s_median"
    }
}

private struct ConcurrentIronBenchOutput: Decodable {
    var mode: String
    var cells: [ConcurrentCell]
}

private struct ConcurrentCell: Decodable {
    var wallDurationSeconds: Double
    var requestCount: Int
    var ttftMs: PercentileMetric
    var itlMs: PercentileMetric
    var aggregate: AggregateMetric

    enum CodingKeys: String, CodingKey {
        case wallDurationSeconds = "wall_duration_s"
        case requestCount = "n_requests"
        case ttftMs = "ttft_ms"
        case itlMs = "itl_ms"
        case aggregate
    }
}

private struct PercentileMetric: Decodable {
    var p50: Double
    var p95: Double
}

private struct AggregateMetric: Decodable {
    var tokensPerSec: Double

    enum CodingKeys: String, CodingKey {
        case tokensPerSec = "tokens_per_sec"
    }
}
