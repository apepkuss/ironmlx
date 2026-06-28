import Darwin
import Testing

@testable import IronMLXAppCore

private class FakeBackendChildProcess: BackendChildProcess {
    var isRunning: Bool
    let processIdentifier: Int32
    private(set) var terminateCalled = false
    private(set) var waitCalled = false

    init(processIdentifier: Int32 = 1234, isRunning: Bool = true) {
        self.processIdentifier = processIdentifier
        self.isRunning = isRunning
    }

    func terminate() {
        terminateCalled = true
    }

    func waitUntilExit() {
        waitCalled = true
    }
}

@Test func appQuitTerminatesAndForceKillsStillRunningBackend() {
    let child = FakeBackendChildProcess()
    var kills: [(pid: pid_t, signal: Int32)] = []
    let terminator = BackendProcessTerminator(
        killSignal: { pid, signal in
            kills.append((pid, signal))
            child.isRunning = false
            return 0
        },
        sleep: { _ in }
    )

    terminator.stop(child, forceAfter: 0)

    #expect(child.terminateCalled)
    #expect(kills.count == 1)
    #expect(kills.first?.pid == 1234)
    #expect(kills.first?.signal == SIGKILL)
    #expect(child.waitCalled)
}

@Test func appQuitDoesNotForceKillBackendThatStopsGracefully() {
    final class GracefulChildProcess: FakeBackendChildProcess {
        override func terminate() {
            super.terminate()
            isRunning = false
        }
    }

    let child = GracefulChildProcess()
    var killCount = 0
    let terminator = BackendProcessTerminator(
        killSignal: { _, _ in
            killCount += 1
            return 0
        },
        sleep: { _ in }
    )

    terminator.stop(child, forceAfter: 0)

    #expect(child.terminateCalled)
    #expect(killCount == 0)
    #expect(child.waitCalled)
}
