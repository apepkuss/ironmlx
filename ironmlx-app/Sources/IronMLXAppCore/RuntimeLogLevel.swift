import Foundation

public enum AppLogLevel: String, Codable, CaseIterable, Sendable {
    case all = "ALL", debug = "DEBUG", info = "INFO", warning = "WARNING", error = "ERROR"

    public init?(setting: String) {
        switch setting.uppercased() {
        case "TRACE": self = .all
        case "WARN": self = .warning
        default:
            guard let value = Self(rawValue: setting.uppercased()) else { return nil }
            self = value
        }
    }

    public static func saved(_ value: String?) -> Self {
        value.flatMap { Self(setting: $0) } ?? .info
    }

    func includes(_ level: AppLogLevel) -> Bool {
        let order: [Self] = [.all, .debug, .info, .warning, .error]
        return order.firstIndex(of: level)! >= order.firstIndex(of: self)!
    }
}

public struct BackendLogLevelSnapshot: Codable, Equatable, Sendable {
    public var level: AppLogLevel?
    public var revision: UInt64
    public var processID: UInt32
    enum CodingKeys: String, CodingKey { case level, revision; case processID = "process_id" }
}

public protocol BackendLogLevelControlling: Sendable {
    func logLevel() async throws -> BackendLogLevelSnapshot
    func setLogLevel(_ level: AppLogLevel, expected: BackendLogLevelSnapshot) async throws -> BackendLogLevelSnapshot
}

extension BackendAPIClient: BackendLogLevelControlling {
    public func logLevel() async throws -> BackendLogLevelSnapshot {
        try await logLevelRequest(body: nil)
    }

    public func setLogLevel(_ level: AppLogLevel, expected: BackendLogLevelSnapshot) async throws -> BackendLogLevelSnapshot {
        let data = try JSONSerialization.data(withJSONObject: [
            "level": level.rawValue,
            "expected_revision": expected.revision,
            "expected_process_id": expected.processID
        ])
        return try await logLevelRequest(body: data)
    }

    private func logLevelRequest(body: Data?) async throws -> BackendLogLevelSnapshot {
        // This management endpoint is deliberately absent from the LAN listener.
        let url = URL(string: "http://127.0.0.1:\(port)/admin/api/log-level")!
        var request = URLRequest(url: url, timeoutInterval: 5)
        if let body {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw RuntimeLogLevelError.backendUnavailable
        }
        return try JSONDecoder().decode(BackendLogLevelSnapshot.self, from: data)
    }
}

public enum RuntimeLogLevelError: Error {
    case busy, backendUnavailable, backendChanged, persistenceFailed, rollbackFailed
}

/// Applies a single setting without committing other Dashboard edits.
@MainActor
public final class RuntimeLogLevelController {
    private let configStore: AppConfigStore
    private let backend: any BackendRuntimeManaging
    private let clientFactory: (UInt16) -> any BackendLogLevelControlling
    private let applyAppLevel: (AppLogLevel) -> Void
    private let persistLevel: (AppLogLevel) -> Bool
    private var applying = false

    public init(
        configStore: AppConfigStore,
        backend: any BackendRuntimeManaging,
        clientFactory: @escaping (UInt16) -> any BackendLogLevelControlling = {
            BackendAPIClient(host: "127.0.0.1", port: $0)
        },
        applyAppLevel: @escaping (AppLogLevel) -> Void = { IronMLXAppLogger.setMinimumLevel($0) },
        persistLevel: ((AppLogLevel) -> Bool)? = nil
    ) {
        self.configStore = configStore
        self.backend = backend
        self.clientFactory = clientFactory
        self.applyAppLevel = applyAppLevel
        self.persistLevel = persistLevel ?? { level in configStore.update { $0.logLevel = level.rawValue } }
    }

    public func apply(_ level: AppLogLevel) async throws {
        guard !applying else { throw RuntimeLogLevelError.busy }
        applying = true
        defer { applying = false }
        let config = configStore.load()
        guard configStore.recoveryIssue == nil else { throw RuntimeLogLevelError.persistenceFailed }
        guard ![.starting, .stopping, .recovering].contains(backend.state) else { throw RuntimeLogLevelError.busy }
        let launchID = backend.currentLaunchID
        let client = clientFactory(config.port)
        var previous: BackendLogLevelSnapshot?
        var attempted = false
        do {
            if backend.isRunning {
                let snapshot = try await client.logLevel()
                guard snapshot.level != nil else { throw RuntimeLogLevelError.backendUnavailable }
                previous = snapshot
                guard backend.currentLaunchID == launchID else { throw RuntimeLogLevelError.backendChanged }
                attempted = true
                let applied = try await client.setLogLevel(level, expected: snapshot)
                guard applied.level == level, applied.processID == snapshot.processID,
                      applied.revision == snapshot.revision + 1,
                      backend.currentLaunchID == launchID, backend.isRunning else {
                    throw RuntimeLogLevelError.backendChanged
                }
            }
            // Reload inside update: another setting may have changed during HTTP awaits.
            guard persistLevel(level) else {
                throw RuntimeLogLevelError.persistenceFailed
            }
            applyAppLevel(level)
            IronMLXAppLogger.debug("Runtime log level applied: \(level.rawValue)")
            IronMLXAppLogger.trace("Runtime log level application completed")
        } catch {
            if attempted, let previous, let oldLevel = previous.level,
               backend.isRunning, backend.currentLaunchID == launchID {
                do {
                    // Reconcile a response lost after the backend applied the update.
                    let current = try await client.logLevel()
                    if current == previous {
                        // No backend change occurred.
                    } else if current.processID == previous.processID,
                              current.revision == previous.revision + 1, current.level == level {
                        _ = try await client.setLogLevel(oldLevel, expected: current)
                    } else {
                        throw RuntimeLogLevelError.rollbackFailed
                    }
                } catch {
                    throw RuntimeLogLevelError.rollbackFailed
                }
            }
            throw error
        }
    }
}
