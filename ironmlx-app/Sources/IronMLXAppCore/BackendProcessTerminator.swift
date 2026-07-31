import Darwin
import Foundation

@MainActor
public protocol BackendChildProcess: AnyObject {
    var isRunning: Bool { get }
    var processIdentifier: Int32 { get }

    func terminate()
}

extension Process: BackendChildProcess {}

public struct BackendProcessTerminator {
    public typealias KillSignal = (pid_t, Int32) -> Int32
    public typealias Sleep = (TimeInterval) async -> Void

    private let killSignal: KillSignal
    private let sleep: Sleep

    public init(
        killSignal: @escaping KillSignal = Darwin.kill,
        sleep: @escaping Sleep = { interval in
            guard interval > 0 else {
                return
            }
            try? await Task.sleep(for: .seconds(interval))
        }
    ) {
        self.killSignal = killSignal
        self.sleep = sleep
    }

    @MainActor
    public func stop(
        _ process: BackendChildProcess,
        forceAfter gracePeriod: TimeInterval = 0.5
    ) async {
        process.terminate()

        if gracePeriod > 0, process.isRunning {
            await sleep(gracePeriod)
        }

        if process.isRunning {
            _ = killSignal(pid_t(process.processIdentifier), SIGKILL)
        }
    }
}
