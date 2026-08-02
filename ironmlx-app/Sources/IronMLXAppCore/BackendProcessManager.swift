import Foundation

public enum BackendProcessState: String, Codable, Equatable, Sendable {
    case stopped
    case starting
    case running
    case stopping
    case recovering
    case degraded
    case failed
}

public enum BackendStopIntent: String, Codable, Equatable, Sendable {
    case userStop
    case appQuit
    case plannedRestart
    case benchmarkExclusive
    case schedulerProfileGeneration
    case startupFailureCleanup
    case recoveryFailureCleanup
    case unexpected

    public var isPlanned: Bool {
        self != .unexpected
    }
}

public struct BackendProcessTermination: Codable, Equatable, Sendable {
    public var occurredAt: Date
    public var launchID: UUID
    public var generation: UInt64
    public var pid: Int32
    public var terminationStatus: Int32
    public var terminationReason: String
    public var stopIntent: BackendStopIntent
    public var logTail: String

    public init(
        occurredAt: Date,
        launchID: UUID,
        generation: UInt64,
        pid: Int32,
        terminationStatus: Int32,
        terminationReason: String,
        stopIntent: BackendStopIntent,
        logTail: String
    ) {
        self.occurredAt = occurredAt
        self.launchID = launchID
        self.generation = generation
        self.pid = pid
        self.terminationStatus = terminationStatus
        self.terminationReason = terminationReason
        self.stopIntent = stopIntent
        self.logTail = logTail
    }
}

public struct BackendProcessLaunchPlan: Sendable {
    public var processURL: URL
    public var arguments: [String]
    public var command: String
    public var standardInputData: Data?

    public init(
        processURL: URL,
        arguments: [String],
        command: String? = nil,
        standardInputData: Data? = nil
    ) {
        self.processURL = processURL
        self.arguments = arguments
        self.command = command ?? ([processURL.path] + arguments).joined(separator: " ")
        self.standardInputData = standardInputData
    }
}

@MainActor
public final class BackendProcessManager {
    public typealias LaunchPlanProvider = @MainActor () throws -> BackendProcessLaunchPlan
    public typealias ProcessFactory = @MainActor () -> Process
    public typealias TerminationObserver = @MainActor (BackendProcessTermination) -> Void

    public private(set) var state: BackendProcessState = .stopped
    public private(set) var lastError: String?
    public private(set) var currentLaunchID: UUID?
    public private(set) var currentGeneration: UInt64?
    public private(set) var lastTermination: BackendProcessTermination?
    public var terminationObserver: TerminationObserver?

    private var currentLaunch: ManagedBackendLaunch?
    private var managedLaunches: [UUID: ManagedBackendLaunch] = [:]
    private var launchGeneration: UInt64 = 0
    private var terminationWaiters: [UUID: [CheckedContinuation<Void, Never>]] = [:]
    private let terminator: BackendProcessTerminator
    private let logStore: IronMLXLogStore
    private let notificationCenter: NotificationCenter
    private let launchPlanProvider: LaunchPlanProvider
    private let processFactory: ProcessFactory

    public init(
        configStore: AppConfigStore = .shared,
        scanner _: LocalModelScanner = LocalModelScanner(),
        terminator: BackendProcessTerminator = BackendProcessTerminator(),
        logStore: IronMLXLogStore = IronMLXLogStore(),
        notificationCenter: NotificationCenter = .default,
        securityStore: LANSecurityMaterialStore = .shared,
        launchPlanProvider: LaunchPlanProvider? = nil,
        processFactory: @escaping ProcessFactory = Process.init
    ) {
        self.terminator = terminator
        self.logStore = logStore
        self.notificationCenter = notificationCenter
        self.processFactory = processFactory
        self.launchPlanProvider = launchPlanProvider ?? {
            let config = configStore.load()
            let runtime = try BackendBinaryResolver.resolveValidatedRuntime()
            let options = BackendLaunchOptions(config: config)
            if let validationError = options.validationError {
                throw BackendProcessError.invalidLaunchConfiguration(validationError)
            }
            let launch = BackendLaunchConfiguration(
                executableURL: runtime.backendURL,
                metallibURL: runtime.metallibURL,
                host: "127.0.0.1",
                port: config.port,
                options: options,
                networkMode: config.isLANMode ? "lan" : "local",
                lanHost: config.lanHost,
                securityBootstrapStdin: config.isLANMode
            )
            if let validationError = launch.validationError {
                throw BackendProcessError.invalidLaunchConfiguration(validationError)
            }
            let standardInputData: Data?
            if config.isLANMode {
                guard let credentialID = config.lanCredentialID,
                      let lanHost = config.lanHost
                else {
                    throw BackendProcessError.invalidLaunchConfiguration(.lanSecurityMaterialMissing)
                }
                standardInputData = try securityStore.bootstrap(
                    credentialID: credentialID,
                    lanHost: lanHost
                )
            } else {
                standardInputData = nil
            }
            let (processURL, arguments) = BackendBinaryResolver.resolvedExecutableAndArguments(
                for: launch
            )
            return BackendProcessLaunchPlan(
                processURL: processURL,
                arguments: arguments,
                standardInputData: standardInputData
            )
        }
    }

    public var isRunning: Bool {
        currentLaunch?.process.isRunning == true
    }

    public var currentProcessIdentifier: Int32? {
        guard let currentLaunch, currentLaunch.process.isRunning else {
            return nil
        }
        return currentLaunch.process.processIdentifier
    }

    @discardableResult
    public func startProcess(
        initialState: BackendProcessState = .starting
    ) throws -> UUID {
        if let currentLaunch, currentLaunch.process.isRunning {
            return currentLaunch.launchID
        }

        let plan: BackendProcessLaunchPlan
        do {
            plan = try launchPlanProvider()
#if IRONMLX_APP_BUNDLE
            let runtime = try BackendBinaryResolver.resolveValidatedRuntime()
            guard plan.processURL.standardizedFileURL == runtime.backendURL.standardizedFileURL else {
                throw BackendProcessError.externalHelperNotAllowed(plan.processURL.path)
            }
            guard plan.arguments.starts(with: ["--mlx-metallib", runtime.metallibURL.path]) else {
                throw BackendProcessError.bundledMetallibArgumentRequired
            }
#endif
        } catch {
            transition(to: .failed, error: error.localizedDescription)
            throw error
        }

        do {
            try logStore.prepareLog(
                .backend,
                sessionHeader: IronMLXAppLogger.backendSessionHeader(command: plan.command)
            )
            let logHandle = try logStore.openFileForAppend(.backend)
            let process = processFactory()
            process.executableURL = plan.processURL
            process.arguments = plan.arguments
            process.environment = BundledChildProcessEnvironment.sanitized()
            process.standardOutput = logHandle
            process.standardError = logHandle
            let standardInputPipe = plan.standardInputData.map { _ in Pipe() }
            process.standardInput = standardInputPipe

            launchGeneration &+= 1
            let launchID = UUID()
            let generation = launchGeneration
            let launch = ManagedBackendLaunch(
                launchID: launchID,
                generation: generation,
                process: process,
                logHandle: logHandle
            )
            process.terminationHandler = { [weak self] terminatedProcess in
                let reason = Self.terminationReasonName(terminatedProcess.terminationReason)
                let status = terminatedProcess.terminationStatus
                let pid = terminatedProcess.processIdentifier
                Task { @MainActor [weak self] in
                    self?.handleTermination(
                        launchID: launchID,
                        generation: generation,
                        pid: pid,
                        status: status,
                        reason: reason
                    )
                }
            }

            currentLaunch = launch
            managedLaunches[launchID] = launch
            currentLaunchID = launchID
            currentGeneration = generation
            transition(to: initialState)
            do {
                try process.run()
                if let data = plan.standardInputData,
                   let standardInputPipe {
                    try standardInputPipe.fileHandleForWriting.write(contentsOf: data)
                    try standardInputPipe.fileHandleForWriting.close()
                }
            } catch {
                process.terminationHandler = nil
                if process.isRunning {
                    process.terminate()
                }
                closeLogHandle(for: launch)
                managedLaunches.removeValue(forKey: launchID)
                currentLaunch = nil
                currentLaunchID = nil
                currentGeneration = nil
                transition(to: .failed, error: error.localizedDescription)
                throw error
            }
            return launchID
        } catch {
            if state != .failed {
                transition(to: .failed, error: error.localizedDescription)
            }
            throw error
        }
    }

    public func stop(
        intent: BackendStopIntent,
        forceAfter gracePeriod: TimeInterval = 1.0
    ) async {
        guard let launch = currentLaunch else {
            transition(to: .stopped)
            return
        }
        launch.stopIntent = intent
        transition(to: .stopping)
        guard launch.process.isRunning else {
            await waitForTermination(of: launch.launchID)
            return
        }

        await terminator.stop(launch.process, forceAfter: gracePeriod)
        await waitForTermination(of: launch.launchID)
    }

    public func stopForAppQuit() async {
        await stop(intent: .appQuit, forceAfter: 0.5)
    }

    public func transition(to nextState: BackendProcessState, error: String? = nil) {
        state = nextState
        lastError = error
        notificationCenter.post(name: .ironMLXBackendRuntimeDidChange, object: self)
    }

    public func markStoppedAfterPlannedTermination() {
        transition(to: .stopped)
    }

    public static func backendLogURL() -> URL {
        IronMLXLogStore().url(for: .backend)
    }

    private func handleTermination(
        launchID: UUID,
        generation: UInt64,
        pid: Int32,
        status: Int32,
        reason: String
    ) {
        guard let launch = managedLaunches.removeValue(forKey: launchID) else {
            resumeTerminationWaiters(for: launchID)
            return
        }

        closeLogHandle(for: launch)
        guard launch.generation == generation,
              currentLaunch?.launchID == launchID,
              currentLaunch?.generation == generation
        else {
            resumeTerminationWaiters(for: launchID)
            return
        }

        let termination = BackendProcessTermination(
            occurredAt: Date(),
            launchID: launchID,
            generation: generation,
            pid: pid,
            terminationStatus: status,
            terminationReason: reason,
            stopIntent: launch.stopIntent,
            logTail: logStore.tailText(from: .backend, maxLines: 200, maxBytes: 65_536)
        )
        lastTermination = termination
        currentLaunch = nil
        currentLaunchID = nil
        currentGeneration = nil
        if launch.stopIntent.isPlanned {
            transition(to: .stopped)
        } else {
            transition(
                to: .failed,
                error: "Backend terminated with \(reason) status \(status)."
            )
        }
        resumeTerminationWaiters(for: launchID)
        terminationObserver?(termination)
    }

    private func waitForTermination(of launchID: UUID) async {
        guard managedLaunches[launchID] != nil else {
            return
        }
        await withCheckedContinuation { continuation in
            terminationWaiters[launchID, default: []].append(continuation)
        }
    }

    private func resumeTerminationWaiters(for launchID: UUID) {
        let waiters = terminationWaiters.removeValue(forKey: launchID) ?? []
        waiters.forEach { $0.resume() }
    }

    private func closeLogHandle(for launch: ManagedBackendLaunch) {
        guard !launch.logClosed else {
            return
        }
        launch.logClosed = true
        try? launch.logHandle.synchronize()
        try? launch.logHandle.close()
    }

    nonisolated private static func terminationReasonName(
        _ reason: Process.TerminationReason
    ) -> String {
        switch reason {
        case .exit:
            return "exit"
        case .uncaughtSignal:
            return "uncaught_signal"
        @unknown default:
            return "unknown"
        }
    }
}

private final class ManagedBackendLaunch {
    let launchID: UUID
    let generation: UInt64
    let process: Process
    let logHandle: FileHandle
    var stopIntent: BackendStopIntent = .unexpected
    var logClosed = false

    init(
        launchID: UUID,
        generation: UInt64,
        process: Process,
        logHandle: FileHandle
    ) {
        self.launchID = launchID
        self.generation = generation
        self.process = process
        self.logHandle = logHandle
    }
}

@frozen public enum BackendProcessError: LocalizedError {
    case invalidLaunchConfiguration(BackendLaunchValidationError)
    case externalHelperNotAllowed(String)
    case bundledMetallibArgumentRequired

    public var errorDescription: String? {
        switch self {
        case .invalidLaunchConfiguration(let error):
            return error.message
        case .externalHelperNotAllowed(let path):
            return "Release App cannot launch a helper outside its App Bundle: \(path)"
        case .bundledMetallibArgumentRequired:
            return "Release App must pass its bundled mlx.metallib to the backend helper."
        }
    }
}
