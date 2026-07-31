import Darwin
import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func realHelperKill9RecoversOnceThenTripsBreakerAndAllowsManualRetry() async throws {
    let root = try crashRecoveryTemporaryDirectory()
    let modelID = "mlx-community/DiffusionGemma-Crash-Recovery-MXFP4"
    _ = try writeVerifiedTestSnapshot(
        root: root,
        repoID: modelID,
        files: [
            "config.json": Data(
                #"{"model_type":"diffusion_gemma","vision_config":{"hidden_size":1152},"quantization":{"mode":"mxfp4","bits":4,"group_size":32}}"#.utf8
            ),
            "model.safetensors": Data("weights".utf8),
        ]
    )
    let port = try availableLoopbackPort()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(
        AppConfig(
            host: "127.0.0.1",
            port: port,
            defaultModel: modelID,
            loadedModels: [modelID],
            pinnedModels: [modelID]
        )
    )
    let helperURL = try #require(
        Bundle.module.url(
            forResource: "backend_crash_helper",
            withExtension: "py",
            subdirectory: "Fixtures"
        )
            ?? Bundle.module.url(
                forResource: "backend_crash_helper",
                withExtension: "py"
            )
    )
    let logStore = IronMLXLogStore(rootURL: root.appendingPathComponent("logs"))
    let incidentStore = BackendIncidentStore(
        url: root.appendingPathComponent("incidents.json")
    )
    var launchCount = 0
    let processManager = BackendProcessManager(
        configStore: configStore,
        logStore: logStore,
        launchPlanProvider: {
            launchCount += 1
            return BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/usr/bin/python3"),
                arguments: [helperURL.path, "--port", String(port)]
            )
        }
    )
    let scanner = LocalModelScanner(rootURL: root)
    let parameterStore = ModelParameterStore(
        url: root.appendingPathComponent("model_params.json")
    )
    try parameterStore.save(
        ModelParameters(
            modelID: modelID,
            maxTokens: "4096",
            temperature: "0.7",
            topP: "0.8",
            topK: "40",
            repeatPenalty: "1.1",
            mtpEnabled: true,
            promptLookupEnabled: true
        )
    )
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: scanner,
        parameterStore: parameterStore,
        incidentStore: incidentStore,
        policy: BackendRecoveryPolicy(
            maximumAutomaticRecoveryAttempts: 1,
            automaticRecoveryDelay: 0,
            stableWindow: 60,
            readinessTimeout: 5
        )
    )
    defer {
        if let pid = processManager.currentProcessIdentifier {
            _ = Darwin.kill(pid_t(pid), SIGKILL)
        }
    }

    try await supervisor.ensureRunning()
    #expect(supervisor.state == .running)
    #expect(launchCount == 1)

    let firstPID = try #require(processManager.currentProcessIdentifier)
    let workTask = Task {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/work")!
        )
        request.httpMethod = "POST"
        request.httpBody = Data("{}".utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        _ = try? await URLSession.shared.data(for: request)
    }
    try await waitForCrashRecoveryCondition {
        logStore.tailText(from: .backend).contains("helper work started")
    }

    #expect(Darwin.kill(pid_t(firstPID), SIGKILL) == 0)
    try await waitForCrashRecoveryCondition(timeout: 8) {
        supervisor.lastEvent?.phase == .recovered
            && processManager.currentProcessIdentifier != nil
            && processManager.currentProcessIdentifier != firstPID
    }
    workTask.cancel()

    #expect(supervisor.state == .running)
    #expect(launchCount == 2)
    #expect(supervisor.lastIncident?.recoveryAttempt == 1)
    #expect(supervisor.lastIncident?.recoveredModels == [modelID])
    #expect(supervisor.lastIncident?.failures.isEmpty == true)

    let recoveredPID = try #require(processManager.currentProcessIdentifier)
    #expect(Darwin.kill(pid_t(recoveredPID), SIGKILL) == 0)
    try await waitForCrashRecoveryCondition {
        supervisor.lastEvent?.phase == .breaker
    }

    #expect(supervisor.state == .failed)
    #expect(!supervisor.isRunning)
    #expect(launchCount == 2)
    #expect(supervisor.lastIncident?.recoveryResult == "breaker")
    #expect(supervisor.lastIncident?.terminationStatus == SIGKILL)
    #expect(supervisor.lastIncident?.terminationReason == "uncaught_signal")
    #expect(supervisor.lastIncident?.logTail.contains("helper") == true)
    #expect(incidentStore.records().last?.recoveryResult == "breaker")

    let retry = await supervisor.retryAfterFailure()
    #expect(retry.success)
    #expect(supervisor.state == .running)
    #expect(launchCount == 3)
}

@Test @MainActor
func plannedUserStopDoesNotCreateIncidentOrRestart() async throws {
    let root = try crashRecoveryTemporaryDirectory()
    let configStore = AppConfigStore(url: root.appendingPathComponent("app_config.json"))
    configStore.save(AppConfig())
    var launchCount = 0
    let processManager = BackendProcessManager(
        configStore: configStore,
        logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
        launchPlanProvider: {
            launchCount += 1
            return BackendProcessLaunchPlan(
                processURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["30"]
            )
        }
    )
    let incidentStore = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
    let supervisor = BackendRuntimeSupervisor(
        processManager: processManager,
        configStore: configStore,
        scanner: LocalModelScanner(rootURL: root),
        parameterStore: ModelParameterStore(
            url: root.appendingPathComponent("model_params.json")
        ),
        restartCoordinator: EmptyModelRestorer(),
        incidentStore: incidentStore,
        readinessWaiter: { _, _ in }
    )

    try await supervisor.ensureRunning()
    await supervisor.stop(intent: .userStop)

    #expect(supervisor.state == .stopped)
    #expect(launchCount == 1)
    #expect(supervisor.lastIncident == nil)
    #expect(incidentStore.records().isEmpty)
}

private struct EmptyModelRestorer: BackendModelRestoring {
    func restore(_ snapshot: BackendRecoverySnapshot) async -> BackendRestartResult {
        BackendRestartResult(
            success: true,
            status: "restarted",
            port: snapshot.config.port
        )
    }
}

private func crashRecoveryTemporaryDirectory() throws -> URL {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent(
            "ironmlx-backend-crash-recovery-\(UUID().uuidString)",
            isDirectory: true
        )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func availableLoopbackPort() throws -> UInt16 {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else {
        throw POSIXError(.ENOTSOCK)
    }
    defer {
        Darwin.close(descriptor)
    }

    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = 0
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let bindResult = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindResult == 0 else {
        throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EADDRINUSE)
    }

    var boundAddress = sockaddr_in()
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    let nameResult = withUnsafeMutablePointer(to: &boundAddress) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            getsockname(descriptor, $0, &length)
        }
    }
    guard nameResult == 0 else {
        throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EINVAL)
    }
    return UInt16(bigEndian: boundAddress.sin_port)
}

@MainActor
private func waitForCrashRecoveryCondition(
    timeout: TimeInterval = 5,
    condition: @escaping () -> Bool
) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if condition() {
            return
        }
        try await Task.sleep(for: .milliseconds(20))
    }
    Issue.record("Timed out waiting for backend crash recovery condition")
}
