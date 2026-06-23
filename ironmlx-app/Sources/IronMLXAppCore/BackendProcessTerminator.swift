import Darwin
import Foundation

public protocol BackendChildProcess: AnyObject {
    var isRunning: Bool { get }
    var processIdentifier: Int32 { get }

    func terminate()
    func waitUntilExit()
}

extension Process: BackendChildProcess {}

public struct BackendProcessTerminator {
    public typealias KillSignal = (pid_t, Int32) -> Int32

    private let killSignal: KillSignal
    private let sleep: (TimeInterval) -> Void

    public init(
        killSignal: @escaping KillSignal = Darwin.kill,
        sleep: @escaping (TimeInterval) -> Void = Thread.sleep(forTimeInterval:)
    ) {
        self.killSignal = killSignal
        self.sleep = sleep
    }

    public func stop(_ process: BackendChildProcess, forceAfter gracePeriod: TimeInterval = 0.5) {
        process.terminate()

        if gracePeriod > 0 {
            let deadline = Date().addingTimeInterval(gracePeriod)
            while process.isRunning && Date() < deadline {
                let remaining = max(0, deadline.timeIntervalSinceNow)
                sleep(min(0.05, remaining))
            }
        }

        if process.isRunning {
            _ = killSignal(pid_t(process.processIdentifier), SIGKILL)
        }

        process.waitUntilExit()
    }
}
