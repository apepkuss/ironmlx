import Foundation

public struct SchedulerProfileGenerationRequest: Codable, Equatable, Sendable {
    public var model: String
    public var modelPath: String
    public var selectionProfile: String

    public init(model: String, modelPath: String, selectionProfile: String = "agent-long-prompt") {
        self.model = model
        self.modelPath = modelPath
        self.selectionProfile = Self.normalizedSelectionProfile(selectionProfile)
    }

    private static func normalizedSelectionProfile(_ value: String) -> String {
        switch value.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "balanced":
            return "balanced"
        default:
            return "agent-long-prompt"
        }
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
        ]

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
    public var startedAt: TimeInterval?
    public var finishedAt: TimeInterval?
    public var outputDirectory: String?
    public var runtimeProfile: String?
    public var storedRuntimeProfile: String?
    public var logTail: String?
    public var exitCode: Int32?
    public var error: String?

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
            startedAt: startedAt.timeIntervalSince1970,
            outputDirectory: outputDirectory
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
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case outputDirectory = "output_directory"
        case runtimeProfile = "runtime_profile"
        case storedRuntimeProfile = "stored_runtime_profile"
        case logTail = "log_tail"
        case exitCode = "exit_code"
        case error
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
}

public final class SchedulerProfileGenerationService: @unchecked Sendable {
    private let outputRootURL: URL
    private let lock = NSLock()
    private var status: SchedulerProfileGenerationStatus = .idle
    private var activeProcess: Process?

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
        if snapshot.state == "running",
           let outputDirectory = snapshot.outputDirectory {
            let logURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
                .appendingPathComponent("calibrate.log")
            let tail = Self.logTail(from: logURL)
            snapshot.logTail = tail.isEmpty ? nil : tail
        }
        return snapshot
    }

    public func start(
        executableURL: URL = BackendBinaryResolver.resolve(),
        ironBenchURL: URL? = nil,
        request: SchedulerProfileGenerationRequest,
        completion: @escaping @Sendable (SchedulerProfileGenerationStatus) -> Void = { _ in }
    ) -> SchedulerProfileGenerationStatus {
        let runningStatus = lock.withLock { () -> SchedulerProfileGenerationStatus? in
            guard status.state != "running" else {
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
            lock.withLock {
                activeProcess = process
            }
            try process.run()
            process.waitUntilExit()
            let logTail = Self.logTail(from: plan.logURL)
            let next = SchedulerProfileGenerationStatus.completed(
                request: request,
                exitCode: process.terminationStatus,
                logTail: logTail,
                outputDirectory: plan.outputDirectoryURL.path,
                startedAt: startedAt
            )
            lock.withLock {
                activeProcess = nil
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
