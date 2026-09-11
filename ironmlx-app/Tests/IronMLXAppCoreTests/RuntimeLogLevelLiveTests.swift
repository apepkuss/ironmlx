import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
@Test(.enabled(if: ProcessInfo.processInfo.environment["IRONMLX_LOG_CONTROL_LIVE"] == "1"))
func runtimeLogLevelLiveHelperAndStreamingInference() async throws {
    let env = ProcessInfo.processInfo.environment
    let helper = try #require(env["IRONMLX_LOG_CONTROL_HELPER"])
    let metallib = try #require(env["IRONMLX_LOG_CONTROL_METALLIB"])
    let modelPath = try #require(env["IRONMLX_LOG_CONTROL_MODEL"])
    let root = URL(fileURLWithPath: try #require(env["IRONMLX_LOG_CONTROL_EVIDENCE"]))
    let port = try #require(UInt16(env["IRONMLX_LOG_CONTROL_PORT"] ?? "19078"))
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let store = AppConfigStore(url: root.appendingPathComponent("config.json"))
    try #require(store.save(AppConfig(port: port, logLevel: "INFO", modelTtlMinutes: 0)))
    let logs = IronMLXLogStore(rootURL: root.appendingPathComponent("logs"))
    let appWriter = AppDiagnosticLogWriter(store: logs)
    let manager = BackendProcessManager(logStore: logs, launchPlanProvider: {
        BackendProcessLaunchPlan(processURL: URL(fileURLWithPath: helper),
            arguments: ["--mlx-metallib", metallib, "serve", "--host", "127.0.0.1", "--port", String(port), "--max-sequences", "1"],
            logLevel: .saved(store.load().logLevel))
    })
    let runtime = BackendRuntimeSupervisor(processManager: manager, configStore: store,
        parameterStore: ModelParameterStore(url: root.appendingPathComponent("params.json")),
        incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")))
    let client = BackendAPIClient(host: "127.0.0.1", port: port)
    let control = RuntimeLogLevelController(configStore: store, backend: runtime, applyAppLevel: { level in
        appWriter.setMinimumLevel(level)
        appWriter.write(line: "app_debug_\(level.rawValue)", level: .debug)
        appWriter.write(line: "app_trace_\(level.rawValue)", level: .all)
    })
    do {
        try await runtime.ensureRunning()
        let pid = try #require(manager.currentProcessIdentifier)
        let initial = try await client.logLevel()
        try #require(initial.level == .info)
        for (level, revision, processID, expectedStatus) in [
            ("INVALID", initial.revision, initial.processID, 422),
            ("DEBUG", initial.revision + 1, initial.processID, 409),
            ("DEBUG", initial.revision, UInt32(0), 409)
        ] {
            var invalid = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/admin/api/log-level")!)
            invalid.httpMethod = "POST"
            invalid.setValue("application/json", forHTTPHeaderField: "Content-Type")
            invalid.httpBody = try JSONSerialization.data(withJSONObject: [
                "level": level, "expected_revision": revision, "expected_process_id": processID
            ])
            let (_, response) = try await URLSession.shared.data(for: invalid)
            try #require((response as? HTTPURLResponse)?.statusCode == expectedStatus)
            try #require(try await client.logLevel() == initial)
        }
        let model = "log-level-validation"
        _ = try await client.loadModel(model: model, modelDir: modelPath, setDefault: true, maxCacheCap: 4096)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!, timeoutInterval: 120)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model,
            "messages": [["role": "user", "content": "List the integers from 1 to 1000, separated by spaces. Do not stop early."]],
            "max_tokens": 2048, "temperature": 0, "stream": true])
        var contentSeen = false
        var finished = false
        var doneSeen = false
        let stream = Task { @MainActor in
            defer { finished = true }
            let (bytes, response) = try await URLSession.shared.bytes(for: request)
            try #require((response as? HTTPURLResponse)?.statusCode == 200)
            for try await line in bytes.lines {
                if line == "data: [DONE]" { doneSeen = true }
                if line.hasPrefix("data: "), let data = line.dropFirst(6).data(using: .utf8),
                   let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = object["choices"] as? [[String: Any]],
                   let delta = choices.first?["delta"] as? [String: Any],
                   let content = delta["content"] as? String, !content.isEmpty { contentSeen = true }
            }
        }
        defer { stream.cancel() }
        let deadline = Date().addingTimeInterval(60)
        while !contentSeen && !finished && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        try #require(contentSeen && !finished)
        var overlapped = 0
        for level in [AppLogLevel.all, .debug, .warning, .error, .info] {
            let before = (try? Data(contentsOf: logs.url(for: .backend)).count) ?? 0
            if !finished { overlapped += 1 }
            try await control.apply(level)
            try #require(manager.currentProcessIdentifier == pid)
            try #require(try await client.logLevel().level == level)
            let output = try Data(contentsOf: logs.url(for: .backend))
            let segment = String(decoding: output.dropFirst(before), as: UTF8.self)
            try #require(segment.contains("runtime log filter applied") == (level == .debug || level == .all))
            try #require(segment.contains("runtime log filter trace enabled") == (level == .all))
        }
        try await stream.value
        try #require(doneSeen && overlapped > 0)
        let appLog = try String(contentsOf: logs.url(for: .app), encoding: .utf8)
        try #require(appLog.contains("app_debug_ALL") && appLog.contains("app_trace_ALL") && appLog.contains("app_debug_DEBUG"))
        try #require(!appLog.contains("app_trace_DEBUG") && !appLog.contains("app_debug_WARNING") && !appLog.contains("app_debug_ERROR") && !appLog.contains("app_debug_INFO"))
        try await control.apply(.debug)
        await runtime.stopForAppQuit()
        try #require(!manager.isRunning)
        try await runtime.ensureRunning()
        try #require(manager.currentProcessIdentifier != pid)
        try #require(try await client.logLevel().level == .debug)
        try #require(store.load().logLevel == "DEBUG")
        await runtime.stopForAppQuit()
        try #require(!manager.isRunning)
        let result: [String: Any] = ["passed": true, "pid_during_updates": pid, "updates_during_stream": overlapped,
            "stream_completed": doneSeen, "levels": ["ALL", "DEBUG", "WARNING", "ERROR", "INFO"],
            "restart_restored": "DEBUG", "helper_stopped": !manager.isRunning]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("result.json"))
    } catch {
        await runtime.stopForAppQuit()
        throw error
    }
}
