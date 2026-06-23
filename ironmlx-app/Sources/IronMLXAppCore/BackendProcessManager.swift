import Foundation

public enum BackendProcessState: String, Equatable {
    case stopped
    case starting
    case running
    case failed
}

@MainActor
public final class BackendProcessManager {
    public private(set) var state: BackendProcessState = .stopped
    public private(set) var lastError: String?

    private var process: Process?
    private let configStore: AppConfigStore
    private let scanner: LocalModelScanner
    private let terminator: BackendProcessTerminator
    private static let logStore = IronMLXLogStore()

    public init(
        configStore: AppConfigStore = .shared,
        scanner: LocalModelScanner = LocalModelScanner(),
        terminator: BackendProcessTerminator = BackendProcessTerminator()
    ) {
        self.configStore = configStore
        self.scanner = scanner
        self.terminator = terminator
    }

    public var isRunning: Bool {
        process?.isRunning == true
    }

    public func start(modelReference: String) throws {
        guard !modelReference.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BackendProcessError.missingModel
        }
        if isRunning {
            return
        }

        var config = configStore.load()
        let executable = BackendBinaryResolver.resolve()
        let options = BackendLaunchOptions(config: config)
        if let validationError = options.validationError {
            state = .failed
            lastError = validationError.message
            throw BackendProcessError.invalidLaunchConfiguration(validationError)
        }
        let launch = BackendLaunchConfiguration(
            executableURL: executable,
            host: config.host,
            port: config.port,
            options: options
        )
        let (processURL, arguments) = BackendBinaryResolver.resolvedExecutableAndArguments(
            for: launch
        )
        let command = ([processURL.path] + arguments).joined(separator: " ")
        try Self.logStore.prepareLog(
            .backend,
            sessionHeader: IronMLXAppLogger.backendSessionHeader(command: command)
        )

        let nextProcess = Process()
        nextProcess.executableURL = processURL
        nextProcess.arguments = arguments
        let backendLogHandle = try Self.logStore.openFileForAppend(.backend)
        nextProcess.standardOutput = backendLogHandle
        nextProcess.standardError = backendLogHandle

        state = .starting
        lastError = nil
        do {
            try nextProcess.run()
            process = nextProcess
            state = .running
            config.lastModel = modelReference
            configStore.save(config)
        } catch {
            state = .failed
            lastError = error.localizedDescription
            throw error
        }
    }

    public func stop() {
        stop(forceAfter: 1.0)
    }

    public func stopForAppQuit() {
        stop(forceAfter: 0.5)
    }

    private func stop(forceAfter gracePeriod: TimeInterval) {
        guard let process else {
            state = .stopped
            return
        }
        if process.isRunning {
            terminator.stop(process, forceAfter: gracePeriod)
        }
        self.process = nil
        state = .stopped
    }

    public static func backendLogURL() -> URL {
        logStore.url(for: .backend)
    }
}

public enum BackendProcessError: LocalizedError {
    case missingModel
    case invalidLaunchConfiguration(BackendLaunchValidationError)

    public var errorDescription: String? {
        switch self {
        case .missingModel:
            return "No model is configured."
        case .invalidLaunchConfiguration(let error):
            return error.message
        }
    }
}
