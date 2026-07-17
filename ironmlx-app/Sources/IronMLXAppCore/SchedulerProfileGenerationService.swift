import Foundation

public enum SchedulerProfileCalibrationLevel: String, Codable, Sendable {
    case quick
    case standard
    case full

    fileprivate static func normalized(_ value: String) -> Self {
        switch value.trimmingCharacters(in: .whitespacesAndNewlines) {
        case Self.quick.rawValue:
            return .quick
        case Self.full.rawValue:
            return .full
        default:
            return .standard
        }
    }
}

public struct SchedulerProfileGenerationRequest: Codable, Equatable, Sendable {
    public var model: String
    public var modelPath: String
    public var selectionProfile: String
    public var calibrationLevel: SchedulerProfileCalibrationLevel
    public var mtpModelDir: String?
    public var mtpDraftTokens: Int?
    public var kvQuant: String
    public var pagedPrefixCacheDir: String?
    public var prefixLruCacheMaxBytes: Int?
    public var ssdPrefixCacheMaxGB: Int?
    public var activeKvOffload: Bool
    public var memoryLimitTotalGB: Int?
    public var memoryLimitModelGB: Int?
    public var maxCacheCap: Int

    public init(
        model: String,
        modelPath: String,
        selectionProfile: String = "agent-long-prompt",
        calibrationLevel: String = "standard",
        mtpModelDir: String? = nil,
        mtpDraftTokens: Int? = nil,
        kvQuant: String = "none",
        pagedPrefixCacheDir: String? = nil,
        prefixLruCacheMaxBytes: Int? = nil,
        ssdPrefixCacheMaxGB: Int? = nil,
        activeKvOffload: Bool = false,
        memoryLimitTotalGB: Int? = nil,
        memoryLimitModelGB: Int? = nil,
        maxCacheCap: Int = 32768
    ) {
        self.model = model
        self.modelPath = modelPath
        self.selectionProfile = Self.normalizedSelectionProfile(selectionProfile)
        self.calibrationLevel = SchedulerProfileCalibrationLevel.normalized(calibrationLevel)
        self.mtpModelDir = mtpModelDir
        self.mtpDraftTokens = mtpDraftTokens
        self.kvQuant = BackendLaunchOptions.normalizedKVQuant(kvQuant) ?? "none"
        self.pagedPrefixCacheDir = pagedPrefixCacheDir
        self.prefixLruCacheMaxBytes = prefixLruCacheMaxBytes
        self.ssdPrefixCacheMaxGB = ssdPrefixCacheMaxGB
        self.activeKvOffload = activeKvOffload
        self.memoryLimitTotalGB = memoryLimitTotalGB
        self.memoryLimitModelGB = memoryLimitModelGB
        self.maxCacheCap = max(maxCacheCap, 1)
    }

    private static func normalizedSelectionProfile(_ value: String) -> String {
        switch value.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "balanced":
            return "balanced"
        default:
            return "agent-long-prompt"
        }
    }

    fileprivate var expectedJobCount: Int {
        let cacheStateCount = pagedPrefixCacheDir == nil ? 1 : 2
        return calibrationCandidateCount * calibrationConcurrency.count * cacheStateCount
    }

    fileprivate var estimatedDurationRange: ClosedRange<Int> {
        let baseline: (jobs: Int, minimumSeconds: Int, maximumSeconds: Int) = switch calibrationLevel {
        case .quick:
            (16, 8 * 60, 15 * 60)
        case .standard:
            (48, 40 * 60, 60 * 60)
        case .full:
            (96, 2 * 60 * 60, 150 * 60)
        }
        let scale = Double(expectedJobCount) / Double(baseline.jobs)
        return Self.roundUpToMinute(Double(baseline.minimumSeconds) * scale) ...
            Self.roundUpToMinute(Double(baseline.maximumSeconds) * scale)
    }

    private static func roundUpToMinute(_ seconds: Double) -> Int {
        Int(ceil(seconds / 60)) * 60
    }

    fileprivate var calibrationArguments: [String] {
        var arguments: [String] = []
        for candidate in calibrationCandidates {
            arguments += ["--candidate", candidate]
        }
        arguments += ["--concurrency", calibrationConcurrency.map(String.init).joined(separator: ",")]
        switch calibrationLevel {
        case .quick:
            arguments += [
                "--runs", "2",
                "--warmup", "1",
                "--duration", "10",
                "--warmup-duration", "2",
            ]
        case .standard:
            arguments += [
                "--runs", "3",
                "--warmup", "1",
                "--duration", "15",
                "--warmup-duration", "3",
            ]
        case .full:
            break
        }
        return arguments
    }

    private var calibrationCandidateCount: Int {
        switch calibrationLevel {
        case .quick:
            return 4
        case .standard, .full:
            return candidateBMaxValues.count * 4
        }
    }

    private var calibrationConcurrency: [Int] {
        switch calibrationLevel {
        case .quick:
            return [1, 2]
        case .standard:
            return [1, 4]
        case .full:
            return [1, 2, 4, 8]
        }
    }

    private var calibrationCandidates: [String] {
        switch calibrationLevel {
        case .quick:
            let highBMax = candidateBMaxValues.last ?? 1
            return [
                candidate(bMax: 1, prefillChunkSize: 1024, decodeCadenceCap: 128),
                candidate(bMax: 1, prefillChunkSize: 2048, decodeCadenceCap: 256),
                candidate(bMax: highBMax, prefillChunkSize: 1024, decodeCadenceCap: 256),
                candidate(bMax: highBMax, prefillChunkSize: 2048, decodeCadenceCap: 128),
            ]
        case .standard:
            return candidateBMaxValues.flatMap { bMax in
                [1024, 2048].flatMap { prefillChunkSize in
                    [128, 256].map { decodeCadenceCap in
                        candidate(
                            bMax: bMax,
                            prefillChunkSize: prefillChunkSize,
                            decodeCadenceCap: decodeCadenceCap
                        )
                    }
                }
            }
        case .full:
            return []
        }
    }

    private var candidateBMaxValues: [Int] {
        usesQwenMtpMatrix ? [1, 2] : [1, 2, 4]
    }

    private var usesQwenMtpMatrix: Bool {
        guard mtpModelDir != nil,
              let data = try? Data(
                  contentsOf: URL(fileURLWithPath: modelPath, isDirectory: true)
                      .appendingPathComponent("config.json")
              ),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return false
        }
        let textConfig = config["text_config"] as? [String: Any]
        let modelTypes = [config["model_type"] as? String, textConfig?["model_type"] as? String]
            .compactMap { $0 }
        return modelTypes.contains("qwen3_5")
            || modelTypes.contains("qwen3_5_moe")
            || modelTypes.contains("qwen3_5_text")
            || modelTypes.contains("qwen3_5_moe_text")
    }

    private func candidate(bMax: Int, prefillChunkSize: Int, decodeCadenceCap: Int) -> String {
        "b_max=\(bMax),prefill_chunk_size=\(prefillChunkSize),admission_deadline_ms=5," +
            "admission_queue_max=32,max_cache_cap=\(maxCacheCap)," +
            "decode_cadence_mid_chunk_cap=\(decodeCadenceCap)"
    }
}

public struct SchedulerProfileGenerationPreview: Codable, Equatable, Sendable {
    public var success: Bool
    public var requestToken: Int?
    public var model: String?
    public var selectionProfile: String?
    public var calibrationLevel: SchedulerProfileCalibrationLevel?
    public var totalJobs: Int?
    public var estimatedMinSeconds: Int?
    public var estimatedMaxSeconds: Int?
    public var error: String?

    public init(request: SchedulerProfileGenerationRequest, requestToken: Int? = nil) {
        let duration = request.estimatedDurationRange
        success = true
        self.requestToken = requestToken
        model = request.model
        selectionProfile = request.selectionProfile
        calibrationLevel = request.calibrationLevel
        totalJobs = request.expectedJobCount
        estimatedMinSeconds = duration.lowerBound
        estimatedMaxSeconds = duration.upperBound
    }

    public static func failed(error: String, requestToken: Int? = nil) -> Self {
        Self(success: false, requestToken: requestToken, error: error)
    }

    private init(success: Bool, requestToken: Int?, error: String) {
        self.success = success
        self.requestToken = requestToken
        model = nil
        selectionProfile = nil
        calibrationLevel = nil
        totalJobs = nil
        estimatedMinSeconds = nil
        estimatedMaxSeconds = nil
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case success
        case requestToken = "request_token"
        case model
        case selectionProfile = "selection_profile"
        case calibrationLevel = "calibration_level"
        case totalJobs = "total_jobs"
        case estimatedMinSeconds = "estimated_min_seconds"
        case estimatedMaxSeconds = "estimated_max_seconds"
        case error
    }
}

public struct SchedulerProfileGenerationPlan: Equatable, Sendable {
    public var processURL: URL
    public var arguments: [String]
    public var outputDirectoryURL: URL
    public var runtimeProfileURL: URL
    public var logURL: URL

    public init(
        executableURL: URL,
        ironBenchURL: URL?,
        request: SchedulerProfileGenerationRequest,
        outputRootURL: URL,
        timestamp: Date = Date()
    ) {
        let directoryName = Self.outputDirectoryName(model: request.model, timestamp: timestamp)
        let outputDirectory = outputRootURL.appendingPathComponent(directoryName, isDirectory: true)
        let runtimeProfile = outputDirectory.appendingPathComponent("scheduler-profile.json")
        let processURL = executableURL
        var arguments: [String]
        if executableURL.path == "/usr/bin/env" {
            arguments = ["ironmlx", "scheduler-autotune", "calibrate"]
        } else {
            arguments = ["scheduler-autotune", "calibrate"]
        }
        arguments += [
            "--model", request.modelPath,
            "--model-name", request.model,
        ]
        if let ironBenchURL {
            arguments += ["--iron-bench-bin", ironBenchURL.path]
        }
        arguments += [
            "--output-dir", outputDirectory.path,
            "--write-profile", runtimeProfile.path,
            "--selection-profile", request.selectionProfile,
            "--kv-quant", request.kvQuant,
            "--max-cache-cap", String(request.maxCacheCap),
        ]
        arguments += request.calibrationArguments
        if let mtpModelDir = request.mtpModelDir {
            arguments += ["--mtp-model-dir", mtpModelDir]
        }
        if let mtpDraftTokens = request.mtpDraftTokens {
            arguments += ["--mtp-draft-tokens", String(mtpDraftTokens)]
        }
        if let pagedPrefixCacheDir = request.pagedPrefixCacheDir {
            arguments += ["--paged-prefix-cache-dir", pagedPrefixCacheDir]
        }
        if let prefixLruCacheMaxBytes = request.prefixLruCacheMaxBytes {
            arguments += ["--prefix-lru-cache-max-bytes", String(prefixLruCacheMaxBytes)]
        }
        if let ssdPrefixCacheMaxGB = request.ssdPrefixCacheMaxGB {
            arguments += ["--ssd-prefix-cache-max-gb", String(ssdPrefixCacheMaxGB)]
        }
        if request.activeKvOffload {
            arguments.append("--active-kv-offload")
        }
        if let memoryLimitTotalGB = request.memoryLimitTotalGB {
            arguments += ["--memory-limit-total-gb", String(memoryLimitTotalGB)]
        }
        if let memoryLimitModelGB = request.memoryLimitModelGB {
            arguments += ["--memory-limit-model-gb", String(memoryLimitModelGB)]
        }

        self.processURL = processURL
        self.arguments = arguments
        self.outputDirectoryURL = outputDirectory
        self.runtimeProfileURL = runtimeProfile
        self.logURL = outputDirectory.appendingPathComponent("calibrate.log")
    }

    private static func outputDirectoryName(model: String, timestamp: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let name = model.split(separator: "/").last.map(String.init) ?? model
        let slug = name
            .map { char -> Character in
                char.isLetter || char.isNumber || char == "-" || char == "_" ? char : "-"
            }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
        return "\(slug.isEmpty ? "model" : slug)-\(formatter.string(from: timestamp))"
    }
}

public struct SchedulerProfileGenerationStatus: Codable, Equatable, Sendable {
    public var state: String
    public var success: Bool
    public var model: String?
    public var modelPath: String?
    public var selectionProfile: String?
    public var calibrationLevel: SchedulerProfileCalibrationLevel?
    public var startedAt: TimeInterval?
    public var finishedAt: TimeInterval?
    public var outputDirectory: String?
    public var runtimeProfile: String?
    public var storedRuntimeProfile: String?
    public var logTail: String?
    public var exitCode: Int32?
    public var error: String?
    public var completedJobs: Int?
    public var totalJobs: Int?
    public var elapsedSeconds: TimeInterval?
    public var estimatedRemainingSeconds: TimeInterval?
    public var currentStage: String?
    public var currentJob: String?

    public static let idle = SchedulerProfileGenerationStatus(
        state: "idle",
        success: true
    )

    public static func running(
        request: SchedulerProfileGenerationRequest,
        outputDirectory: String,
        startedAt: Date = Date()
    ) -> SchedulerProfileGenerationStatus {
        SchedulerProfileGenerationStatus(
            state: "running",
            success: true,
            model: request.model,
            modelPath: request.modelPath,
            selectionProfile: request.selectionProfile,
            calibrationLevel: request.calibrationLevel,
            startedAt: startedAt.timeIntervalSince1970,
            outputDirectory: outputDirectory,
            completedJobs: 0,
            totalJobs: request.expectedJobCount
        )
    }

    public static func completed(
        request: SchedulerProfileGenerationRequest,
        exitCode: Int32,
        logTail: String,
        outputDirectory: String,
        startedAt: TimeInterval? = nil,
        finishedAt: Date = Date()
    ) -> SchedulerProfileGenerationStatus {
        let succeeded = exitCode == 0
        return SchedulerProfileGenerationStatus(
            state: succeeded ? "succeeded" : "failed",
            success: succeeded,
            model: request.model,
            modelPath: request.modelPath,
            selectionProfile: request.selectionProfile,
            calibrationLevel: request.calibrationLevel,
            startedAt: startedAt,
            finishedAt: finishedAt.timeIntervalSince1970,
            outputDirectory: outputDirectory,
            runtimeProfile: Self.value(after: "runtime_profile:", in: logTail),
            storedRuntimeProfile: Self.value(after: "stored_runtime_profile:", in: logTail),
            logTail: logTail,
            exitCode: exitCode,
            error: succeeded ? nil : "scheduler-autotune calibrate exited with code \(exitCode)"
        )
    }

    public static func cancelled(
        request: SchedulerProfileGenerationRequest,
        logTail: String,
        outputDirectory: String,
        startedAt: TimeInterval? = nil,
        finishedAt: Date = Date()
    ) -> SchedulerProfileGenerationStatus {
        SchedulerProfileGenerationStatus(
            state: "cancelled",
            success: false,
            model: request.model,
            modelPath: request.modelPath,
            selectionProfile: request.selectionProfile,
            calibrationLevel: request.calibrationLevel,
            startedAt: startedAt,
            finishedAt: finishedAt.timeIntervalSince1970,
            outputDirectory: outputDirectory,
            logTail: logTail
        )
    }

    public static func failed(
        request: SchedulerProfileGenerationRequest?,
        error: String,
        logTail: String? = nil,
        startedAt: TimeInterval? = nil,
        outputDirectory: String? = nil,
        finishedAt: Date = Date()
    ) -> SchedulerProfileGenerationStatus {
        SchedulerProfileGenerationStatus(
            state: "failed",
            success: false,
            model: request?.model,
            modelPath: request?.modelPath,
            selectionProfile: request?.selectionProfile,
            calibrationLevel: request?.calibrationLevel,
            startedAt: startedAt,
            finishedAt: finishedAt.timeIntervalSince1970,
            outputDirectory: outputDirectory,
            logTail: logTail,
            error: error
        )
    }

    enum CodingKeys: String, CodingKey {
        case state
        case success
        case model
        case modelPath = "model_path"
        case selectionProfile = "selection_profile"
        case calibrationLevel = "calibration_level"
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case outputDirectory = "output_directory"
        case runtimeProfile = "runtime_profile"
        case storedRuntimeProfile = "stored_runtime_profile"
        case logTail = "log_tail"
        case exitCode = "exit_code"
        case error
        case completedJobs = "completed_jobs"
        case totalJobs = "total_jobs"
        case elapsedSeconds = "elapsed_seconds"
        case estimatedRemainingSeconds = "estimated_remaining_seconds"
        case currentStage = "current_stage"
        case currentJob = "current_job"
    }

    private static func value(after key: String, in text: String) -> String? {
        text.split(separator: "\n")
            .compactMap { line -> String? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard trimmed.hasPrefix(key) else {
                    return nil
                }
                let value = trimmed.dropFirst(key.count).trimmingCharacters(in: .whitespacesAndNewlines)
                return value.isEmpty ? nil : value
            }
            .last
    }

    mutating func refreshProgress(logTail: String, now: Date = Date()) {
        guard let outputDirectory else {
            return
        }
        self.logTail = logTail.isEmpty ? nil : logTail
        if let startedAt {
            elapsedSeconds = max(0, now.timeIntervalSince1970 - startedAt)
        }

        let outputURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        let manifestURL = outputURL.appendingPathComponent("run-order.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(SchedulerProfileRunOrder.self, from: data)
        else {
            return
        }

        totalJobs = manifest.jobs.count
        let completed = manifest.jobs.filter {
            FileManager.default.fileExists(
                atPath: outputURL.appendingPathComponent($0.outputJSON).path
            )
        }.count
        completedJobs = completed

        if let current = manifest.jobs.first(where: {
            !FileManager.default.fileExists(
                atPath: outputURL.appendingPathComponent($0.outputJSON).path
            )
        }) {
            currentJob = "candidate \(current.candidateIndex) · concurrency \(current.concurrency) · \(current.cacheState)"
        } else {
            currentJob = nil
        }

        currentStage = logTail
            .split(separator: "\n")
            .reversed()
            .map(String.init)
            .first(where: { $0.contains("[scheduler-autotune]") })

        let completedDurations = logTail
            .split(separator: "\n")
            .compactMap { line -> Double? in
                guard line.contains("stage=completed"),
                      let suffix = line.range(of: "elapsed_s=")?.upperBound
                else {
                    return nil
                }
                return Double(line[suffix...].trimmingCharacters(in: .whitespacesAndNewlines))
            }
        if !completedDurations.isEmpty, completed < manifest.jobs.count {
            let average = completedDurations.reduce(0, +) / Double(completedDurations.count)
            estimatedRemainingSeconds = average * Double(manifest.jobs.count - completed)
        } else if completed == manifest.jobs.count {
            estimatedRemainingSeconds = 0
        }
    }
}

private struct SchedulerProfileRunOrder: Decodable {
    var jobs: [SchedulerProfileRunOrderJob]
}

private struct SchedulerProfileRunOrderJob: Decodable {
    var candidateIndex: Int
    var concurrency: Int
    var cacheState: String
    var outputJSON: String

    enum CodingKeys: String, CodingKey {
        case candidateIndex = "candidate_idx"
        case concurrency
        case cacheState = "cache_state"
        case outputJSON = "output_json"
    }
}

public final class SchedulerProfileGenerationService: @unchecked Sendable {
    private let outputRootURL: URL
    private let lock = NSLock()
    private var status: SchedulerProfileGenerationStatus = .idle
    private var activeProcess: Process?
    private var cancelRequested = false

    public init(
        outputRootURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("reports", isDirectory: true)
            .appendingPathComponent("scheduler-autotune", isDirectory: true)
    ) {
        self.outputRootURL = outputRootURL
    }

    public func currentStatus() -> SchedulerProfileGenerationStatus {
        var snapshot = lock.withLock { status }
        if ["running", "cancelling"].contains(snapshot.state),
           let outputDirectory = snapshot.outputDirectory {
            let logURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
                .appendingPathComponent("calibrate.log")
            let tail = Self.logTail(from: logURL)
            snapshot.refreshProgress(logTail: tail)
        }
        return snapshot
    }

    public func cancel() -> SchedulerProfileGenerationStatus {
        let process = lock.withLock { () -> Process? in
            guard status.state == "running" else {
                return nil
            }
            cancelRequested = true
            status.state = "cancelling"
            status.success = false
            status.error = nil
            return activeProcess
        }
        process?.interrupt()
        return currentStatus()
    }

    public func start(
        executableURL: URL = BackendBinaryResolver.resolve(),
        ironBenchURL: URL? = nil,
        request: SchedulerProfileGenerationRequest,
        completion: @escaping @Sendable (SchedulerProfileGenerationStatus) -> Void = { _ in }
    ) -> SchedulerProfileGenerationStatus {
        let runningStatus = lock.withLock { () -> SchedulerProfileGenerationStatus? in
            guard !["running", "cancelling"].contains(status.state) else {
                return nil
            }
            let plan = SchedulerProfileGenerationPlan(
                executableURL: executableURL,
                ironBenchURL: ironBenchURL ?? BackendBinaryResolver.resolveIronBenchBinary(near: executableURL),
                request: request,
                outputRootURL: outputRootURL
            )
            let next = SchedulerProfileGenerationStatus.running(
                request: request,
                outputDirectory: plan.outputDirectoryURL.path
            )
            cancelRequested = false
            status = next
            DispatchQueue.global(qos: .utility).async {
                self.run(plan: plan, request: request, startedAt: next.startedAt, completion: completion)
            }
            return next
        }

        return runningStatus ?? currentStatus()
    }

    private func run(
        plan: SchedulerProfileGenerationPlan,
        request: SchedulerProfileGenerationRequest,
        startedAt: TimeInterval?,
        completion: @escaping @Sendable (SchedulerProfileGenerationStatus) -> Void
    ) {
        do {
            try FileManager.default.createDirectory(
                at: plan.outputDirectoryURL,
                withIntermediateDirectories: true
            )
            try writeCommandHeader(plan)
            let logHandle = try Self.openLogHandle(plan.logURL)
            defer {
                try? logHandle.close()
            }

            let process = Process()
            process.executableURL = plan.processURL
            process.arguments = plan.arguments
            process.standardOutput = logHandle
            process.standardError = logHandle
            try process.run()
            let shouldInterrupt = lock.withLock { () -> Bool in
                activeProcess = process
                return cancelRequested
            }
            if shouldInterrupt {
                process.interrupt()
            }
            process.waitUntilExit()
            let logTail = Self.logTail(from: plan.logURL)
            let wasCancelled = lock.withLock { cancelRequested }
            var next = if wasCancelled && process.terminationStatus != 0 {
                SchedulerProfileGenerationStatus.cancelled(
                    request: request,
                    logTail: logTail,
                    outputDirectory: plan.outputDirectoryURL.path,
                    startedAt: startedAt
                )
            } else {
                SchedulerProfileGenerationStatus.completed(
                    request: request,
                    exitCode: process.terminationStatus,
                    logTail: logTail,
                    outputDirectory: plan.outputDirectoryURL.path,
                    startedAt: startedAt
                )
            }
            next.refreshProgress(logTail: logTail)
            lock.withLock {
                activeProcess = nil
                cancelRequested = false
                status = next
            }
            completion(next)
        } catch {
            let logTail = Self.logTail(from: plan.logURL)
            let next = SchedulerProfileGenerationStatus.failed(
                request: request,
                error: error.localizedDescription,
                logTail: logTail,
                startedAt: startedAt,
                outputDirectory: plan.outputDirectoryURL.path
            )
            lock.withLock {
                activeProcess = nil
                cancelRequested = false
                status = next
            }
            completion(next)
        }
    }

    private func writeCommandHeader(_ plan: SchedulerProfileGenerationPlan) throws {
        let command = ([plan.processURL.path] + plan.arguments)
            .map { $0.contains(" ") ? "\"\($0)\"" : $0 }
            .joined(separator: " ")
        try "[ironmlx-app] \(command)\n\n".write(to: plan.logURL, atomically: true, encoding: .utf8)
    }

    private static func openLogHandle(_ url: URL) throws -> FileHandle {
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        return handle
    }

    private static func logTail(from url: URL, maxBytes: Int = 24_576) -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return ""
        }
        defer {
            try? handle.close()
        }
        let size = (try? handle.seekToEnd()) ?? 0
        let offset = size > UInt64(maxBytes) ? size - UInt64(maxBytes) : 0
        try? handle.seek(toOffset: offset)
        let data = (try? handle.readToEnd()) ?? Data()
        return String(data: data, encoding: .utf8) ?? ""
    }
}

extension BackendBinaryResolver {
    public static func resolveIronBenchBinary(near executableURL: URL) -> URL? {
        let environment = ProcessInfo.processInfo.environment
        if let explicit = environment["IRON_BENCH_BIN"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit)
        }
        if executableURL.path == "/usr/bin/env" {
            return nil
        }
        let sibling = executableURL.deletingLastPathComponent().appendingPathComponent("iron-bench")
        if FileManager.default.isExecutableFile(atPath: sibling.path) {
            return sibling
        }
        return sibling
    }
}
