import Foundation
import Testing
import Darwin
import WebKit

@testable import IronMLXAppCore

@Test @MainActor func audioDashboardUsesReadinessForLoadAndResourceRepairActions() async throws {
    let url = try #require(IronMLXAppResourceResolver.url(forResource: "dashboard2", withExtension: "html"))
    let web = WKWebView()
    web.loadHTMLString(try String(contentsOf: url, encoding: .utf8), baseURL: url.deletingLastPathComponent())
    var ready = false
    for _ in 0..<100 {
        if (try? await web.evaluateJavaScript("typeof renderModelLoadActions === 'function'")) as? Bool == true {
            ready = true
            break
        }
        try await Task.sleep(for: .milliseconds(50))
    }
    #expect(ready)
    let result = try #require(try await web.evaluateJavaScript("""
        (() => {
          currentLang = 'en';
          const ready = {id:'tts',type:'tts',readiness:{status:'ready'},integrity:{state:'verified'}};
          const missing = {...ready,readiness:{status:'incomplete',reason_code:'audio_resources_missing'}};
          return {
            load: renderModelLoadActions(ready,'tts','tts','action-load','Load',false),
            repair: renderModelLoadActions(missing,'tts','tts','action-load','Load',false),
            status: modelStatusPresentation(ready,false).text
          };
        })()
        """) as? [String: String])
    #expect(result["load"]?.contains("toggleModelLoad") == true)
    #expect(result["load"]?.contains("disabled") == false)
    #expect(result["repair"]?.contains("startDownload") == true)
    #expect(result["status"]?.contains("unavailable") == false)
}

@Test func audioLoadRequestCarriesResourcesWithoutStaleLanguageModelOverrides() throws {
    let execution = BackendAudioExecutionSettings(
        queueTimeoutMS: 30_000,
        firstAudioTimeoutMS: 90_000,
        executionTimeoutMS: 600_000,
        slowConsumerTimeoutMS: 15_000,
        maxOutputFrames: 6_615_000,
        segmentTokens: 96
    )
    let audio = BackendAudioResources(derivedResources: "/audio/derived", resourceLock: "/audio/sources.json",
                                      wetextFsts: "/audio/fsts", unidicDir: "/audio/dictionary",
                                      execution: execution)
    let request = BackendLoadModelRequest(
        model: "tts", modelDir: "/snapshot", setDefault: false, maxCacheCap: 8192,
        mtpModelDir: "/old-mtp", mtpDraftTokens: 4, promptLookup: .crossRequest,
        samplingDefaults: BackendSamplingDefaults(temperature: 0.9, topK: 20), audio: audio
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let config = try #require(object["audio"] as? [String: Any])
    #expect(config["derived_resources"] as? String == audio.derivedResources)
    #expect(config["resource_lock"] as? String == audio.resourceLock)
    #expect(config["wetext_fsts"] as? String == audio.wetextFsts)
    #expect(config["unidic_dir"] as? String == audio.unidicDir)
    let encodedExecution = try #require(config["execution"] as? [String: Any])
    #expect(encodedExecution["queue_timeout_ms"] as? Int == execution.queueTimeoutMS)
    #expect(encodedExecution["first_audio_timeout_ms"] as? Int == execution.firstAudioTimeoutMS)
    #expect(encodedExecution["execution_timeout_ms"] as? Int == execution.executionTimeoutMS)
    #expect(encodedExecution["slow_consumer_timeout_ms"] as? Int == execution.slowConsumerTimeoutMS)
    #expect(encodedExecution["max_output_frames"] as? Int == execution.maxOutputFrames)
    #expect(encodedExecution["segment_tokens"] as? Int == execution.segmentTokens)
    for key in ["max_cache_cap", "mtp_model_dir", "mtp_draft_tokens", "prompt_lookup", "temperature", "top_k"] {
        #expect(object[key] == nil)
    }
    #expect(try JSONDecoder().decode(BackendLoadModelRequest.self, from: data) == request)
}

@Test func audioRecoveryConfigurationRoundTripsAndLegacySnapshotsStillDecode() throws {
    let audio = BackendAudioResources(derivedResources: "/audio/derived", resourceLock: "/audio/sources.json",
                                      wetextFsts: "/audio/fsts", unidicDir: "/audio/dictionary")
    let model = BackendRecoveryModel(id: "tts", modelDir: "/snapshot", isDefault: false, pinned: true,
                                     maxCacheCap: nil, mtpModelDir: nil, mtpDraftTokens: nil,
                                     promptLookup: nil, samplingDefaults: .empty, audio: audio)
    let snapshot = BackendRecoverySnapshot(config: AppConfig(loadedModels: ["tts"]), models: [model])
    let data = try JSONEncoder().encode(snapshot)
    #expect(try JSONDecoder().decode(BackendRecoverySnapshot.self, from: data) == snapshot)
    var legacy = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(model)) as? [String: Any])
    legacy.removeValue(forKey: "audio")
    #expect(try JSONDecoder().decode(BackendRecoveryModel.self,
                                    from: JSONSerialization.data(withJSONObject: legacy)).audio == nil)
}

/// Opt-in acceptance against real resources downloaded by the App service into
/// an isolated root. The helper and metallib must both come from a Release Bundle.
@Test(.enabled(if: ProcessInfo.processInfo.environment["IRONMLX_AUDIO_APP_BUNDLE"] != nil))
@MainActor func audioAppLoadsAndRestoresFromSavedConfigurationWithBundle() async throws {
    let env = ProcessInfo.processInfo.environment
    let bundle = URL(fileURLWithPath: try #require(env["IRONMLX_AUDIO_APP_BUNDLE"]))
    let root = URL(fileURLWithPath: try #require(env["IRONMLX_TEST_DOWNLOAD_ROOT"]))
    let reference = try Data(contentsOf: URL(fileURLWithPath: try #require(env["IRONMLX_AUDIO_APP_REFERENCE"])))
    let id = "mlx-community/IndexTTS-2.5-fp16"
    let scanner = LocalModelScanner(rootURL: root)
    #expect(scanner.model(for: id)?.readiness?.isLoadable == true)
    let parameterStore = ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    let configStore = AppConfigStore(url: root.appendingPathComponent("config.json"))
    var config = AppConfig(port: try audioTestPort(), defaultModel: id, loadedModels: [id], pinnedModels: [id])
    #expect(configStore.save(config))
    for iteration in 0..<2 {
        // A new store/decoder and process on each pass exercises persisted state.
        config = AppConfigStore(url: configStore.url).load()
        let saved = BackendRecoverySnapshot.capture(config: config, scanner: scanner, parameterStore: parameterStore)
        let record = root.appendingPathComponent("audio-recovery.json")
        try JSONEncoder().encode(saved).write(to: record)
        let restored = try JSONDecoder().decode(BackendRecoverySnapshot.self, from: Data(contentsOf: record))
        #expect(restored.models.first?.audio != nil)
        let process = Process()
        process.executableURL = bundle.appendingPathComponent("Contents/Helpers/ironmlx")
        process.arguments = ["--mlx-metallib", bundle.appendingPathComponent("Contents/Resources/mlx.metallib").path,
                             "serve", "--port", String(config.port)]
        process.environment = env.filter { !$0.key.hasPrefix("MLX_") && !$0.key.hasPrefix("DYLD_") && !$0.key.hasPrefix("PYTHON") }
        process.environment?["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        process.currentDirectoryURL = root
        let log = root.appendingPathComponent("backend-\(iteration).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        process.standardOutput = handle
        process.standardError = handle
        try process.run()
        defer { if process.isRunning { process.terminate(); process.waitUntilExit() } }
        let client = BackendAPIClient(host: config.host, port: config.port)
        try await client.waitUntilReady(timeout: 15)
        if iteration == 0 {
            var unloaded = config
            unloaded.loadedModels = []
            unloaded.pinnedModels = []
            unloaded.defaultModel = nil
            #expect(configStore.save(unloaded))
            try await audioLoadThroughDashboard(client: client, configStore: configStore,
                                                scanner: scanner, parameterStore: parameterStore, model: id)
            #expect(configStore.load().restoredModelReferences.contains(id))
        } else {
            let coordinator = BackendRestartCoordinator(scanner: scanner, parameterStore: parameterStore)
            let loaded = await coordinator.restore(restored)
            try #require(loaded.success, "\(loaded)")
        }
        let models = try await client.fetchLoadedModels()
        #expect(models.contains { $0.id == id && $0.pinned })
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(config.port)/v1/audio/speech")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 180
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let streaming = iteration == 1
        let text = streaming
            ? String(repeating: "这是独立数据目录中的流式语音测试。应用自动准备模型资源，重新启动后仍然可以合成语音。", count: 5)
            : "这是应用自动配置后的语音测试。"
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": id, "input": text, "ref_audio": reference.base64EncodedString(),
            "response_format": streaming ? "pcm" : "wav", "stream": streaming,
        ])
        let started = Date()
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        var received = Data()
        var first: Date?
        for try await byte in bytes {
            if first == nil { first = Date() }
            received.append(byte)
        }
        let completed = Date()
        #expect(received.count > 44)
        if streaming {
            #expect(received.count % 2 == 0)
            #expect(completed.timeIntervalSince(try #require(first)) > 1)
        } else {
            #expect(received.prefix(4) == Data("RIFF".utf8))
        }
        try received.write(to: root.appendingPathComponent(streaming ? "app-speech.pcm" : "app-speech.wav"))
        print("App acceptance pass \(iteration): \(received.count) bytes; first \(first!.timeIntervalSince(started))s; complete \(completed.timeIntervalSince(started))s")
        process.terminate()
        process.waitUntilExit()
    }
}

@MainActor private func audioLoadThroughDashboard(
    client: BackendAPIClient, configStore: AppConfigStore, scanner: LocalModelScanner,
    parameterStore: ModelParameterStore, model: String
) async throws {
    let web = WKWebView()
    let backend = AudioTestRunningBackend(client: client)
    let bridge = DashboardBridge(webView: web, configStore: configStore, backend: backend, scanner: scanner,
                                 downloadService: ModelDownloadService(rootURL: scanner.rootURL),
                                 parameterStore: parameterStore)
    web.configuration.userContentController.add(bridge, name: "loadModel")
    web.configuration.userContentController.add(bridge, name: "fetchAPIPost")
    defer {
        web.configuration.userContentController.removeScriptMessageHandler(forName: "loadModel")
        web.configuration.userContentController.removeScriptMessageHandler(forName: "fetchAPIPost")
    }
    let html = try #require(IronMLXAppResourceResolver.url(forResource: "dashboard2", withExtension: "html"))
    web.loadHTMLString(try String(contentsOf: html, encoding: .utf8), baseURL: html.deletingLastPathComponent())
    for _ in 0..<100 {
        if (try? await web.evaluateJavaScript("typeof toggleModelLoad === 'function'")) as? Bool == true { break }
        try await Task.sleep(for: .milliseconds(50))
    }
    let models = try String(decoding: JSONEncoder().encode(scanner.scan()), as: UTF8.self)
    let id = try String(decoding: JSONEncoder().encode(model), as: UTF8.self)
    _ = try await web.evaluateJavaScript("""
        onLocalModelsScanned(JSON.stringify(\(models)));
        window.__AUDIO_LOAD_RESULT__ = null;
        const originalAudioLoadCallback = onModelLoaded;
        onModelLoaded = function(value) {
          window.__AUDIO_LOAD_RESULT__ = JSON.parse(value);
          originalAudioLoadCallback(value);
        };
        toggleModelLoad(\(id), document.createElement('button'));
        """)
    var result: [String: Any]?
    for _ in 0..<600 {
        result = try await web.evaluateJavaScript("window.__AUDIO_LOAD_RESULT__") as? [String: Any]
        if result != nil { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    #expect(result?["success"] as? Bool == true, "Dashboard result: \(String(describing: result))")
    #expect(backend.confirmedModels.contains(model))
    _ = try await web.evaluateJavaScript("togglePin(\(id), document.createElement('button'))")
    for _ in 0..<100 {
        if configStore.load().pinnedModelReferences.contains(model) { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    #expect(configStore.load().pinnedModelReferences.contains(model))
}

/// Process ownership stays with the test; all model-management HTTP calls and
/// persistence still run through the production DashboardBridge.
@MainActor private final class AudioTestRunningBackend: BackendRuntimeManaging {
    let client: BackendAPIClient
    var isRunning: Bool { true }
    var state: BackendProcessState { .running }
    var currentLaunchID: UUID? = UUID()
    var lastError: String? { nil }
    var lastEvent: BackendRuntimeEvent? { nil }
    var lastIncident: BackendIncidentRecord? { nil }
    var confirmedModels: [String] = []
    init(client: BackendAPIClient) { self.client = client }
    func ensureRunning() async throws { try await client.waitUntilReady() }
    func restart(intent: BackendStopIntent) async -> BackendRestartResult {
        BackendRestartResult(success: false, status: "test_owns_process", port: client.port)
    }
    func stop(intent: BackendStopIntent) async {}
    func stopForAppQuit() async {}
    func confirmLoadedModels(_ models: [BackendLoadedModelInfo], parameterConfirmedModelIDs: Set<String>) {
        confirmedModels = models.map(\.id)
    }
    func refreshConfirmedSnapshot() {}
}

private func audioTestPort() throws -> UInt16 {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else { throw POSIXError(.ENOTSOCK) }
    defer { Darwin.close(descriptor) }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let bound = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bound == 0 else { throw POSIXError(.EADDRINUSE) }
    var size = socklen_t(MemoryLayout<sockaddr_in>.size)
    let result = withUnsafeMutablePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &size) }
    }
    guard result == 0 else { throw POSIXError(.EINVAL) }
    return UInt16(bigEndian: address.sin_port)
}
