import AppKit
import Foundation
import WebKit

@MainActor
public final class DashboardBridge: NSObject, WKScriptMessageHandler {
    public enum SettingsValidationError: Error, Equatable {
        case prefixCacheConflictsWithKVQuant
    }

    private weak var webView: WKWebView?
    private let configStore: AppConfigStore
    private let backend: BackendProcessManager
    private let scanner: LocalModelScanner
    private let downloadService: ModelDownloadService
    private let deletionService: LocalModelDeletionService
    private let profileGenerationService: SchedulerProfileGenerationService
    private let benchmarkService: BenchmarkService
    private let benchmarkSessionCoordinator: BenchmarkExclusiveSessionCoordinator
    private let restartCoordinator: BackendRestartCoordinator
    private let parameterStore: ModelParameterStore

    public init(
        webView: WKWebView,
        configStore: AppConfigStore,
        backend: BackendProcessManager,
        scanner: LocalModelScanner = LocalModelScanner(),
        downloadService: ModelDownloadService = ModelDownloadService(),
        deletionService: LocalModelDeletionService? = nil,
        profileGenerationService: SchedulerProfileGenerationService = SchedulerProfileGenerationService(),
        benchmarkService: BenchmarkService = BenchmarkService(),
        benchmarkSessionCoordinator: BenchmarkExclusiveSessionCoordinator = BenchmarkExclusiveSessionCoordinator(),
        restartCoordinator: BackendRestartCoordinator? = nil,
        parameterStore: ModelParameterStore = .shared
    ) {
        self.webView = webView
        self.configStore = configStore
        self.backend = backend
        self.scanner = scanner
        self.downloadService = downloadService
        self.deletionService = deletionService ?? LocalModelDeletionService(configStore: configStore)
        self.profileGenerationService = profileGenerationService
        self.benchmarkService = benchmarkService
        self.benchmarkSessionCoordinator = benchmarkSessionCoordinator
        self.parameterStore = parameterStore
        self.restartCoordinator = restartCoordinator ?? BackendRestartCoordinator(
            scanner: scanner,
            parameterStore: parameterStore
        )
        super.init()
    }

    public static let handlerNames = [
        "fetchAPI",
        "fetchAPIPost",
        "fetchAPIDelete",
        "setLanguage",
        "setTheme",
        "setDefaultModel",
        "deleteModels",
        "saveSettings",
        "restartServer",
        "loadModel",
        "forceLoadModel",
        "unloadModel",
        "downloadModel",
        "searchHF",
        "scanLocalModels",
        "syncLoadedModels",
        "getAppLogs",
        "dashboardLog",
        "generateSchedulerProfile",
        "refreshSchedulerProfileStatus",
        "saveModelParams",
        "openOpenClawChat",
        "openOpenClawDashboard",
        "checkOpenClaw",
        "checkIronHermes",
    ]

    public func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        let name = message.name
        let body = message.body

        switch name {
        case "fetchAPI":
            handleFetch(path: stringBody(body))
        case "fetchAPIPost":
            handlePost(json: stringBody(body))
        case "fetchAPIDelete":
            sendFetchResult(path: stringBody(body), jsonString: "null")
        case "setLanguage":
            updateConfig { $0.language = stringBody(body) }
            notifyMenuLanguageDidChange()
        case "setTheme":
            let value = stringBody(body)
            updateConfig { $0.theme = value == "system" ? nil : value }
        case "setDefaultModel":
            let model = stringBody(body).trimmingCharacters(in: .whitespacesAndNewlines)
            updateConfig { $0.lastModel = model }
            setBackendDefaultModelIfLoaded(model)
        case "deleteModels":
            deleteModels(json: stringBody(body))
        case "saveSettings":
            saveSettings(json: stringBody(body))
        case "restartServer":
            restartBackend()
        case "loadModel", "forceLoadModel":
            loadBackendModel(modelReference: stringBody(body), callback: .modelLoaded)
        case "unloadModel":
            unloadBackendModel(modelReference: stringBody(body), callback: .modelUnloaded)
        case "downloadModel":
            startHuggingFaceDownload(json: stringBody(body))
        case "searchHF":
            searchHuggingFace(json: stringBody(body))
        case "scanLocalModels":
            sendScannedModels()
        case "syncLoadedModels":
            syncLoadedModels()
        case "getAppLogs":
            sendAppLogs()
        case "dashboardLog":
            logDashboardMessage(json: stringBody(body))
        case "generateSchedulerProfile":
            generateSchedulerProfile(json: stringBody(body))
        case "refreshSchedulerProfileStatus":
            sendSchedulerProfileStatus()
        case "saveModelParams":
            saveModelParams(json: stringBody(body))
        case "openOpenClawChat", "openOpenClawDashboard":
            NSWorkspace.shared.open(URL(string: "http://127.0.0.1:18789")!)
        case "checkOpenClaw":
            sendJavaScript("onOpenClawStatus(\(Self.jsStringLiteral("{\"installed\":false,\"gatewayRunning\":false}")))")
        case "checkIronHermes":
            sendJavaScript("onIronHermesStatus(\(Self.jsStringLiteral("{\"installed\":false,\"running\":false}")))")
        default:
            break
        }
    }

    private func handleFetch(path: String) {
        switch path {
        case "/health":
            let config = configStore.load()
            let host = config.host
            let port = config.port
            Task {
                do {
                    let client = BackendAPIClient(host: host, port: port)
                    let healthz = try await client.fetchHealthz()
                    let legacy = LegacyHealthAdapter(statusNow: Date()).legacyStatus(from: healthz)
                    let json = try Self.jsonString(legacy)
                    await MainActor.run {
                        self.sendFetchResult(path: path, jsonString: json)
                    }
                } catch {
                    await MainActor.run {
                        self.sendFetchResult(path: path, jsonString: "null")
                    }
                }
            }
        case "/admin/api/endpoints":
            let config = configStore.load()
            let payload = EndpointPayload(host: config.host, port: config.port)
            sendFetchResult(path: path, jsonString: (try? Self.jsonString(payload)) ?? "null")
        case "/admin/api/models/local":
            Task {
                let loadedModels = await self.loadedModelReferences()
                let models = self.scanner.scan(loadedModels: loadedModels)
                let benchmarkModels = models.map {
                    BenchmarkModel(repoID: $0.repoID, loaded: $0.loaded)
                }
                let json = (try? Self.jsonString(benchmarkModels)) ?? "[]"
                await MainActor.run {
                    self.sendFetchResult(path: path, jsonString: json)
                }
            }
        case "/admin/api/models/downloads":
            Task {
                let statuses = await self.downloadService.downloadStatuses()
                let json = (try? Self.jsonString(statuses)) ?? "[]"
                await MainActor.run {
                    self.sendFetchResult(path: path, jsonString: json)
                }
            }
        case "/admin/api/logs":
            sendFetchResult(path: path, jsonString: logText(from: .backend))
        default:
            sendFetchResult(path: path, jsonString: emptyPayload(for: path))
        }
    }

    private func handlePost(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(APIPostPayload.self, from: data)
        else {
            return
        }

        switch payload.path {
        case "/admin/api/models/load":
            if let model = payload.body["model"]?.stringValue
                ?? payload.body["model_dir"]?.stringValue
                ?? payload.body["repo_id"]?.stringValue {
                loadBackendModel(modelReference: model, callback: .fetch(path: payload.path))
            } else {
                sendFetchResult(path: payload.path, jsonString: "null")
            }
        case "/admin/api/models/ms/download":
            if let repoID = payload.body["repo_id"]?.stringValue {
                startModelScopeDownload(repoID: repoID, path: payload.path)
            } else {
                sendFetchResult(path: payload.path, jsonString: "{\"success\":false,\"status\":\"error\",\"error\":\"missing repo_id\"}")
            }
        case "/admin/api/benchmark/preflight":
            preflightBenchmark(payload: payload)
        case "/admin/api/benchmark/prepare":
            prepareBenchmark(payload: payload)
        case "/admin/api/benchmark":
            runBenchmark(payload: payload)
        case "/admin/api/benchmark/restore":
            restoreBenchmark(path: payload.path)
        case "/admin/api/cache/capacity":
            sendColdCacheCapacity(payload: payload)
        default:
            sendFetchResult(path: payload.path, jsonString: emptyPayload(for: payload.path))
        }
    }

    private func sendColdCacheCapacity(payload: APIPostPayload) {
        let directory = payload.body["cache_dir"]?.stringValue
            ?? payload.body["dir"]?.stringValue
            ?? BackendLaunchOptions.defaultPagedPrefixCacheDirectory
        let capacity = ColdCacheCapacityPolicy.capacity(forDirectoryPath: directory)
        let json = (try? Self.jsonString(capacity)) ?? "{\"min_gb\":1,\"max_gb\":100,\"default_gb\":10,\"reserve_gb\":10}"
        sendFetchResult(path: payload.path, jsonString: json)
    }

    private func preflightBenchmark(payload: APIPostPayload) {
        guard let target = benchmarkTarget(from: payload) else {
            sendFetchResult(path: payload.path, jsonString: benchmarkTargetUnavailableJSON())
            return
        }
        let config = configStore.load()
        let client = BackendAPIClient(host: config.host, port: config.port)

        Task {
            do {
                let result = try await benchmarkSessionCoordinator.preflight(
                    client: client,
                    targetModel: target.model
                )
                let json = try Self.jsonString(result)
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: json)
                }
            } catch {
                let fallback = BenchmarkExclusivePreflightResult(
                    success: true,
                    targetModel: target.model,
                    activeRequests: 0,
                    queuedRequests: 0,
                    loadedModels: [],
                    defaultModel: nil,
                    nonTargetModels: [],
                    willUnloadCount: 0
                )
                let json = (try? Self.jsonString(fallback)) ?? "{\"success\":true}"
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: json)
                }
            }
        }
    }

    private func prepareBenchmark(payload: APIPostPayload) {
        guard let target = benchmarkTarget(from: payload) else {
            sendFetchResult(path: payload.path, jsonString: benchmarkTargetUnavailableJSON())
            return
        }
        let config = configStore.load()
        let client = BackendAPIClient(host: config.host, port: config.port)

        Task {
            do {
                try await MainActor.run {
                    try self.backend.start(modelReference: target.model)
                }
                try await client.waitUntilReady()
                let result = try await benchmarkSessionCoordinator.prepare(
                    client: client,
                    targetModel: target.model,
                    targetModelPath: target.path
                )
                let json = try Self.jsonString(result)
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: json)
                    self.sendScannedModels()
                }
            } catch {
                let json = self.benchmarkExclusiveErrorJSON(error)
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: json)
                    self.sendScannedModels()
                }
            }
        }
    }

    private func runBenchmark(payload: APIPostPayload) {
        guard let model = payload.body["model"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty,
              let modelPath = scanner.resolveModelPath(for: model)
        else {
            sendFetchResult(
                path: payload.path,
                jsonString: "{\"success\":false,\"error\":\"Benchmark model is not available locally.\"}"
            )
            return
        }
        let request = BenchmarkRequest(
            model: model,
            modelPath: modelPath,
            promptTokens: payload.body["prompt_tokens"]?.intValue ?? 1024,
            maxTokens: payload.body["max_tokens"]?.intValue ?? 128,
            batchSize: payload.body["batch_size"]?.intValue ?? 1
        )
        let config = configStore.load()
        let client = BackendAPIClient(host: config.host, port: config.port)

        Task {
            do {
                guard await benchmarkSessionCoordinator.canRunBenchmark(targetModel: model) else {
                    throw BenchmarkExclusiveSessionError.notPrepared
                }
                let result = try await benchmarkService.run(
                    request: request,
                    host: config.host,
                    port: config.port,
                    client: client
                )
                let json = try Self.jsonString(result)
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: json)
                }
            } catch {
                let failure = ErrorPayload(success: false, error: error.localizedDescription)
                let json = (try? Self.jsonString(failure)) ?? "{\"success\":false,\"error\":\"Benchmark failed.\"}"
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: json)
                }
            }
        }
    }

    private func restoreBenchmark(path: String) {
        let config = configStore.load()
        let client = BackendAPIClient(host: config.host, port: config.port)

        Task {
            do {
                let result = try await benchmarkSessionCoordinator.restore(client: client)
                let json = try Self.jsonString(result)
                await MainActor.run {
                    if result.status != "not_active" {
                        self.updateConfig { $0.lastModel = result.defaultModel }
                    }
                    self.sendFetchResult(path: path, jsonString: json)
                    self.sendScannedModels()
                }
            } catch {
                let json = self.benchmarkExclusiveErrorJSON(error)
                await MainActor.run {
                    self.sendFetchResult(path: path, jsonString: json)
                    self.sendScannedModels()
                }
            }
        }
    }

    private func startHuggingFaceDownload(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(HuggingFaceDownloadPayload.self, from: data)
        else {
            let result = ModelDownloadCompletion(
                success: false,
                message: nil,
                error: "Invalid HuggingFace download request.",
                code: "hf_download_failed",
                repoID: nil
            )
            sendDownloadComplete(result)
            return
        }

        Task {
            let result = await downloadService.downloadHuggingFace(
                repoID: payload.repoID,
                token: payload.token,
                progress: { [weak self] progress in
                    await MainActor.run {
                        self?.sendJavaScript(
                            "onDownloadProgress(\(progress.percent), \(Self.jsStringLiteral(progress.filename)))"
                        )
                    }
                }
            )
            await MainActor.run {
                self.sendDownloadComplete(result)
                if result.success {
                    self.sendScannedModels()
                }
            }
        }
    }

    private func searchHuggingFace(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(HuggingFaceSearchPayload.self, from: data)
        else {
            sendJavaScript("onSearchResults(\(Self.jsStringLiteral("[]")))")
            return
        }

        Task {
            do {
                let results = try await downloadService.searchHuggingFace(
                    query: payload.query,
                    sort: payload.sort
                )
                let json = (try? Self.jsonString(results)) ?? "[]"
                await MainActor.run {
                    self.sendJavaScript("onSearchResults(\(Self.jsStringLiteral(json)))")
                }
            } catch {
                await MainActor.run {
                    self.sendJavaScript("onSearchResults(\(Self.jsStringLiteral("[]")))")
                }
            }
        }
    }

    private func startModelScopeDownload(repoID: String, path: String) {
        Task {
            let response = await downloadService.startModelScopeDownload(repoID: repoID)
            let json = (try? Self.jsonString(response)) ?? "{\"success\":false,\"status\":\"error\"}"
            await MainActor.run {
                self.sendFetchResult(path: path, jsonString: json)
            }
        }
    }

    private func deleteModels(json: String) {
        guard let data = json.data(using: .utf8),
              let modelIDs = try? JSONDecoder().decode([String].self, from: data)
        else {
            sendJavaScript("showToast(\(Self.jsStringLiteral("Invalid model deletion request.")), 'warn')")
            return
        }

        do {
            let result = try deletionService.deleteModels(modelIDs)
            let resultJSON = (try? Self.jsonString(result)) ?? "{}"
            let defaultSync = result.clearedDefault ? "window.__DEFAULT_MODEL__ = '';" : ""
            sendJavaScript("\(defaultSync)onModelsDeleted(\(Self.jsStringLiteral(resultJSON)))")
        } catch {
            sendJavaScript(
                "showToast(\(Self.jsStringLiteral("Failed to delete model: \(error.localizedDescription)")), 'warn')"
            )
        }
    }

    private func sendDownloadComplete(_ result: ModelDownloadCompletion) {
        let json = (try? Self.jsonString(result)) ?? "{\"success\":false,\"error\":\"Download failed.\"}"
        sendJavaScript("onDownloadComplete(\(Self.jsStringLiteral(json)))")
    }

    private func loadBackendModel(modelReference: String, callback: ModelOperationCallback) {
        let model = modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            deliverModelOperationResult(error: "No model is configured.", callback: callback)
            return
        }
        let config = configStore.load()
        let resolvedModel = scanner.resolveModelPath(for: model) ?? model
        let maxCacheCap = ModelLoadParameters.maxCacheCap(
            for: model,
            scanner: scanner,
            parameterStore: parameterStore
        )
        Task {
            do {
                try await MainActor.run {
                    try self.backend.start(modelReference: model)
                }
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                let response = try await client.loadModel(
                    model: model,
                    modelDir: resolvedModel,
                    setDefault: true,
                    maxCacheCap: maxCacheCap,
                    reloadWhenIdle: false,
                    samplingDefaults: self.parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
                )
                let json = try Self.jsonString(response)
                await MainActor.run {
                    self.deliverModelOperationResult(jsonString: json, callback: callback)
                    self.sendScannedModels()
                }
            } catch {
                await MainActor.run {
                    self.deliverBackendError(error, callback: callback)
                }
            }
        }
    }

    private func unloadBackendModel(modelReference: String, callback: ModelOperationCallback) {
        let model = modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            deliverModelOperationResult(error: "No model is configured.", callback: callback)
            return
        }
        let config = configStore.load()
        let resolvedModel = scanner.resolveModelPath(for: model)
        Task {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                let response = try await client.unloadModel(model: model, modelDir: resolvedModel)
                let json = try Self.jsonString(response)
                await MainActor.run {
                    self.deliverModelOperationResult(jsonString: json, callback: callback)
                    self.sendScannedModels()
                }
            } catch {
                await MainActor.run {
                    self.deliverBackendError(error, callback: callback)
                }
            }
        }
    }

    private func restartBackend() {
        let config = configStore.load()
        Task {
            let result = await self.restartCoordinator.restartDefaultModel(
                config: config,
                backend: self.backend
            )
            let json = (try? Self.jsonString(result)) ?? "{\"success\":false,\"status\":\"restart_failed\"}"
            await MainActor.run {
                self.sendJavaScript("onServerRestarted(\(Self.jsStringLiteral(json)))")
                self.sendScannedModels()
            }
        }
    }

    private func saveSettings(json: String) {
        let config: AppConfig
        do {
            config = try Self.config(applyingSettingsJSON: json, to: configStore.load())
        } catch SettingsValidationError.prefixCacheConflictsWithKVQuant {
            sendJavaScript("onSettingsSaved(\(Self.jsStringLiteral("{\"status\":\"error\",\"code\":\"cache_turboquant_conflict\"}")))")
            return
        } catch {
            return
        }

        configStore.save(config)
        notifyMenuLanguageDidChange()
        sendJavaScript("onSettingsSaved(\(Self.jsStringLiteral("{\"status\":\"ok\",\"needs_restart\":true}")))")
    }

    static func config(applyingSettingsJSON json: String, to existing: AppConfig) throws -> AppConfig {
        guard let data = json.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return existing
        }

        var config = existing
        if let host = object["host"] as? String, !host.isEmpty {
            config.host = host
        }
        if let port = object["port"] as? UInt16 {
            config.port = port
        } else if let port = intValue(object, "port"), (1...65535).contains(port) {
            config.port = UInt16(port)
        } else if let port = object["port"] as? String, let parsed = UInt16(port) {
            config.port = parsed
        }
        if let language = object["language"] as? String, !language.isEmpty {
            config.language = language
        }
        if let theme = object["theme"] as? String {
            config.theme = theme == "system" ? nil : theme
        }
        if let logLevel = stringValue(object, "log_level"), !logLevel.isEmpty {
            config.logLevel = logLevel
        }
        config.memLimitTotal = intValue(object, "mem_limit_total") ?? config.memLimitTotal
        config.memLimitModel = intValue(object, "mem_limit_model") ?? config.memLimitModel
        config.memTotalAuto = boolValue(object, "mem_total_auto") ?? config.memTotalAuto
        config.memTotal = intValue(object, "mem_total") ?? config.memTotal
        config.memModelAuto = boolValue(object, "mem_model_auto") ?? config.memModelAuto
        config.hotCache = intValue(object, "hot_cache") ?? config.hotCache
        config.coldCache = intValue(object, "cold_cache") ?? config.coldCache
        config.cacheEnable = boolValue(object, "cache_enable") ?? config.cacheEnable
        config.cacheDir = stringValue(object, "cache_dir") ?? config.cacheDir
        config.kvQuant = stringValue(object, "kv_quant") ?? config.kvQuant
        config.maxSequences = intValue(object, "max_sequences") ?? config.maxSequences
        config.bMax = nil
        config.maxModels = intValue(object, "max_models") ?? config.maxModels
        config.initCacheBlocks = intValue(object, "init_cache_blocks") ?? config.initCacheBlocks
        config.modelTtlMinutes = intValue(object, "model_ttl_minutes") ?? config.modelTtlMinutes
        config.distributedBackend = stringValue(object, "distributed_backend") ?? config.distributedBackend
        config.parallelMode = stringValue(object, "parallel_mode") ?? config.parallelMode
        config.prefillChunkSize = intValue(object, "prefill_chunk_size") ?? config.prefillChunkSize
        config.admissionDeadlineMs = intValue(object, "admission_deadline_ms") ?? config.admissionDeadlineMs
        config.admissionQueueMax = intValue(object, "admission_queue_max") ?? config.admissionQueueMax
        config.maxCacheCap = intValue(object, "max_cache_cap") ?? config.maxCacheCap
        config.decodeCadenceMidChunkCap = intValue(object, "decode_cadence_mid_chunk_cap") ?? config.decodeCadenceMidChunkCap
        if let schedulerProfile = stringValue(object, "scheduler_profile") {
            let trimmed = schedulerProfile.trimmingCharacters(in: .whitespacesAndNewlines)
            config.schedulerProfile = trimmed.isEmpty ? nil : trimmed
        }
        config.schedulerAutotuneReport = boolValue(object, "scheduler_autotune_report") ?? config.schedulerAutotuneReport
        let launchOptions = BackendLaunchOptions(config: config)
        if launchOptions.validationError == .prefixCacheConflictsWithKVQuant {
            throw SettingsValidationError.prefixCacheConflictsWithKVQuant
        }
        return config
    }

    private static func intValue(_ object: [String: Any], _ key: String) -> Int? {
        guard let value = object[key], CFGetTypeID(value as CFTypeRef) != CFBooleanGetTypeID() else {
            return nil
        }
        if let int = value as? Int {
            return int
        }
        if let double = value as? Double {
            return Int(double)
        }
        if let string = value as? String {
            return Int(string.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    private static func boolValue(_ object: [String: Any], _ key: String) -> Bool? {
        guard let value = object[key] else {
            return nil
        }
        if let bool = value as? Bool {
            return bool
        }
        if let int = value as? Int {
            return int != 0
        }
        if let string = value as? String {
            switch string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true", "1", "yes":
                return true
            case "false", "0", "no":
                return false
            default:
                return nil
            }
        }
        return nil
    }

    private static func stringValue(_ object: [String: Any], _ key: String) -> String? {
        guard let value = object[key] else {
            return nil
        }
        if let string = value as? String {
            return string
        }
        if let int = value as? Int {
            return String(int)
        }
        if let double = value as? Double {
            return String(double)
        }
        if let bool = value as? Bool {
            return bool ? "true" : "false"
        }
        return nil
    }

    private func setBackendDefaultModelIfLoaded(_ modelReference: String) {
        let model = modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty, backend.isRunning else {
            return
        }
        let config = configStore.load()
        Task {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                _ = try await client.setDefaultModel(model)
            } catch {
                IronMLXAppLogger.error("Failed to set backend default model: \(error)")
            }
        }
    }

    private func saveModelParams(json: String) {
        guard let data = json.data(using: .utf8),
              let parameters = try? JSONDecoder().decode(ModelParameters.self, from: data) else {
            sendJavaScript("showToast(\(Self.jsStringLiteral("Invalid model parameters.")), 'warn')")
            return
        }
        do {
            try parameterStore.save(parameters)
            sendModelParameters()
            scheduleModelParameterReloadIfNeeded(parameters)
        } catch {
            sendJavaScript(
                "showToast(\(Self.jsStringLiteral("Failed to save model parameters: \(error.localizedDescription)")), 'warn')"
            )
        }
    }

    private func scheduleModelParameterReloadIfNeeded(_ parameters: ModelParameters) {
        let model = parameters.modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty, backend.isRunning else {
            return
        }
        let config = configStore.load()
        Task {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                let loadedModels = try await client.fetchLoadedModels()
                guard let loaded = loadedModels.first(where: { candidate in
                    candidate.id == model || candidate.model == model || candidate.path == model
                }) else {
                    return
                }
                let resolvedModel = scanner.resolveModelPath(for: model) ?? loaded.path
                let response = try await client.loadModel(
                    model: model,
                    modelDir: resolvedModel,
                    setDefault: loaded.isDefault,
                    maxCacheCap: parameters.maxCacheCap,
                    reloadWhenIdle: true,
                    samplingDefaults: parameters.samplingDefaults
                )
                let json = try Self.jsonString(response)
                await MainActor.run {
                    self.sendJavaScript("onModelParamsReloaded(\(Self.jsStringLiteral(json)))")
                    self.sendScannedModels()
                }
            } catch {
                let response = BackendModelAdminResponse(
                    success: false,
                    status: "error",
                    model: nil,
                    loadedModels: [],
                    warning: nil,
                    error: error.localizedDescription
                )
                let json = (try? Self.jsonString(response)) ?? "{\"success\":false,\"status\":\"error\"}"
                await MainActor.run {
                    self.sendJavaScript("onModelParamsReloaded(\(Self.jsStringLiteral(json)))")
                }
            }
        }
    }

    private func syncLoadedModels() {
        Task {
            let loaded = await self.loadedModelReferences()
            let json = (try? Self.jsonString(Array(loaded).sorted())) ?? "[]"
            await MainActor.run {
                self.sendJavaScript("onLoadedModelsSynced(\(Self.jsStringLiteral(json)))")
            }
        }
    }

    private func loadedModelReferences() async -> Set<String> {
        let config = configStore.load()
        var loaded = Set<String>()
        if backend.isRunning {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                let models = try await client.fetchLoadedModels()
                for model in models {
                    loaded.insert(model.id)
                    loaded.insert(model.model)
                    loaded.insert(model.path)
                }
            } catch {
                IronMLXAppLogger.error("Failed to fetch loaded models: \(error)")
            }
        }
        if loaded.isEmpty, let lastModel = config.lastModel {
            loaded.insert(lastModel)
        }
        return loaded
    }

    private func sendScannedModels() {
        Task {
            let loadedModels = await self.loadedModelReferences()
            let models = self.scanner.scan(loadedModels: loadedModels)
            let json = (try? Self.jsonString(models)) ?? "[]"
            await MainActor.run {
                self.sendModelParameters()
                self.sendJavaScript("onLocalModelsScanned(\(Self.jsStringLiteral(json)))")
            }
        }
    }

    private func sendModelParameters() {
        sendJavaScript("onModelParamsLoaded(\(Self.jsStringLiteral(parameterStore.jsonString())))")
    }

    private func deliverModelOperationResult(jsonString: String, callback: ModelOperationCallback) {
        switch callback {
        case .modelLoaded:
            sendJavaScript("onModelLoaded(\(Self.jsStringLiteral(jsonString)))")
        case .modelUnloaded:
            sendJavaScript("onModelUnloaded(\(Self.jsStringLiteral(jsonString)))")
        case .fetch(let path):
            sendFetchResult(path: path, jsonString: jsonString)
        case .none:
            break
        }
    }

    private func deliverModelOperationResult(error: String, callback: ModelOperationCallback) {
        let payload = BridgeErrorResponse(success: false, status: "error", error: error)
        let json = (try? Self.jsonString(payload)) ?? "{\"success\":false,\"status\":\"error\"}"
        deliverModelOperationResult(jsonString: json, callback: callback)
    }

    private func deliverBackendError(_ error: Error, callback: ModelOperationCallback) {
        if case BackendAPIError.serverResponse(statusCode: _, body: let body) = error,
           let body,
           !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            deliverModelOperationResult(jsonString: body, callback: callback)
            return
        }
        deliverModelOperationResult(error: error.localizedDescription, callback: callback)
    }

    private func sendAppLogs() {
        sendJavaScript("receiveAppLogs(\(Self.jsStringLiteral(logText(from: .app))))")
    }

    private func logDashboardMessage(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(DashboardLogPayload.self, from: data) else {
            IronMLXAppLogger.warning("Dashboard: \(json)")
            return
        }
        let message = "Dashboard: \(payload.message)"
        switch payload.level.uppercased() {
        case "ERROR":
            IronMLXAppLogger.error(message)
        case "WARN", "WARNING":
            IronMLXAppLogger.warning(message)
        default:
            IronMLXAppLogger.info(message)
        }
    }

    private func generateSchedulerProfile(json: String) {
        let current = profileGenerationService.currentStatus()
        guard current.state != "running" else {
            sendSchedulerProfileStatus(current)
            return
        }
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(SchedulerProfileGenerationPayload.self, from: data)
        else {
            sendSchedulerProfileStatus(
                SchedulerProfileGenerationStatus.failed(
                    request: nil,
                    error: "Invalid scheduler profile generation request."
                )
            )
            return
        }
        let model = payload.model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            sendSchedulerProfileStatus(
                SchedulerProfileGenerationStatus.failed(
                    request: nil,
                    error: "No model is selected."
                )
            )
            return
        }
        guard let modelPath = scanner.resolveModelPath(for: model) else {
            sendSchedulerProfileStatus(
                SchedulerProfileGenerationStatus.failed(
                    request: nil,
                    error: "Model not found locally: \(model)"
                )
            )
            return
        }

        let shouldRestartBackend = backend.isRunning
        if shouldRestartBackend {
            backend.stop()
        }
        let request = SchedulerProfileGenerationRequest(
            model: model,
            modelPath: modelPath,
            selectionProfile: payload.selectionProfile ?? "agent-long-prompt"
        )
        let started = profileGenerationService.start(request: request) { [weak self] status in
            Task { @MainActor in
                guard let self else {
                    return
                }
                self.sendSchedulerProfileStatus(status)
                if shouldRestartBackend {
                    self.restartBackend()
                }
            }
        }
        sendSchedulerProfileStatus(started)
    }

    private func sendSchedulerProfileStatus(_ status: SchedulerProfileGenerationStatus? = nil) {
        let payload = status ?? profileGenerationService.currentStatus()
        let json = (try? Self.jsonString(payload)) ?? "{\"state\":\"failed\",\"success\":false}"
        sendJavaScript("onSchedulerProfileStatus(\(Self.jsStringLiteral(json)))")
    }

    private func sendFetchResult(path: String, jsonString: String) {
        sendJavaScript(
            "onApiFetchResult(\(Self.jsStringLiteral(path)), \(Self.jsStringLiteral(jsonString)))"
        )
    }

    private func sendJavaScript(_ source: String) {
        webView?.evaluateJavaScript(source) { _, error in
            if let error {
                IronMLXAppLogger.error("Dashboard JavaScript error: \(error)")
            }
        }
    }

    private func updateConfig(_ mutate: (inout AppConfig) -> Void) {
        var config = configStore.load()
        mutate(&config)
        configStore.save(config)
    }

    private func notifyMenuLanguageDidChange() {
        NotificationCenter.default.post(name: .ironMLXMenuLanguageDidChange, object: self)
    }

    private func logText(from file: IronMLXLogFile) -> String {
        IronMLXLogStore().tailText(from: file)
    }

    private func emptyPayload(for path: String) -> String {
        if path.contains("discovered") || path.contains("invitations") || path.contains("downloads") {
            return "[]"
        }
        if path.contains("cluster/status") {
            return "{\"status\":\"unavailable\"}"
        }
        return "null"
    }

    private func benchmarkTarget(from payload: APIPostPayload) -> (model: String, path: String)? {
        guard let model = payload.body["model"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty,
              let path = scanner.resolveModelPath(for: model)
        else {
            return nil
        }
        return (model, path)
    }

    private func benchmarkTargetUnavailableJSON() -> String {
        let payload = BenchmarkExclusiveErrorPayload(
            success: false,
            code: "benchmark_model_unavailable",
            error: "Benchmark model is not available locally.",
            activeRequests: nil,
            queuedRequests: nil
        )
        return (try? Self.jsonString(payload)) ?? "{\"success\":false,\"error\":\"Benchmark model is not available locally.\"}"
    }

    private func benchmarkExclusiveErrorJSON(_ error: Error) -> String {
        if let benchmarkError = error as? BenchmarkExclusiveSessionError {
            let activeRequests: Int?
            let queuedRequests: Int?
            if case .activeRequests(let active, let queued) = benchmarkError {
                activeRequests = active
                queuedRequests = queued
            } else {
                activeRequests = nil
                queuedRequests = nil
            }
            let payload = BenchmarkExclusiveErrorPayload(
                success: false,
                code: benchmarkError.code,
                error: benchmarkError.localizedDescription,
                activeRequests: activeRequests,
                queuedRequests: queuedRequests
            )
            return (try? Self.jsonString(payload)) ?? "{\"success\":false,\"error\":\"\(benchmarkError.localizedDescription)\"}"
        }

        let payload = BenchmarkExclusiveErrorPayload(
            success: false,
            code: "benchmark_exclusive_failed",
            error: error.localizedDescription,
            activeRequests: nil,
            queuedRequests: nil
        )
        return (try? Self.jsonString(payload)) ?? "{\"success\":false,\"error\":\"Benchmark failed.\"}"
    }

    private func stringBody(_ body: Any) -> String {
        if let string = body as? String {
            return string
        }
        return String(describing: body)
    }

    private struct BenchmarkModel: Codable {
        var repoID: String
        var loaded: Bool

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case loaded
        }
    }

    private enum ModelOperationCallback {
        case modelLoaded
        case modelUnloaded
        case fetch(path: String)
        case none
    }

    private struct BridgeErrorResponse: Encodable {
        var success: Bool
        var status: String
        var error: String
    }

    private struct HuggingFaceDownloadPayload: Decodable {
        var repoID: String
        var token: String?

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case token
        }
    }

    private struct HuggingFaceSearchPayload: Decodable {
        var query: String
        var sort: String
    }

    private struct SchedulerProfileGenerationPayload: Decodable {
        var model: String
        var selectionProfile: String?

        enum CodingKeys: String, CodingKey {
            case model
            case selectionProfile = "selection_profile"
        }
    }

    private struct DashboardLogPayload: Decodable {
        var level: String
        var message: String
    }

    private struct APIPostPayload: Decodable {
        var path: String
        var body: [String: JSONValue]
    }

    private struct ErrorPayload: Encodable {
        var success: Bool
        var error: String
    }

    private struct BenchmarkExclusiveErrorPayload: Encodable {
        var success: Bool
        var code: String
        var error: String
        var activeRequests: Int?
        var queuedRequests: Int?

        enum CodingKeys: String, CodingKey {
            case success
            case code
            case error
            case activeRequests = "active_requests"
            case queuedRequests = "queued_requests"
        }
    }

    private enum JSONValue: Decodable {
        case string(String)
        case number(Double)
        case bool(Bool)
        case object([String: JSONValue])
        case array([JSONValue])
        case null

        var stringValue: String? {
            switch self {
            case .string(let value):
                return value
            case .number(let value):
                return String(value)
            default:
                return nil
            }
        }

        var intValue: Int? {
            switch self {
            case .number(let value):
                return Int(value)
            case .string(let value):
                return Int(value)
            default:
                return nil
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .null
            } else if let value = try? container.decode(Bool.self) {
                self = .bool(value)
            } else if let value = try? container.decode(Double.self) {
                self = .number(value)
            } else if let value = try? container.decode(String.self) {
                self = .string(value)
            } else if let value = try? container.decode([String: JSONValue].self) {
                self = .object(value)
            } else {
                self = .array(try container.decode([JSONValue].self))
            }
        }
    }

    static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        let data = try encoder.encode(value)
        return String(data: data, encoding: .utf8) ?? "null"
    }

    static func jsStringLiteral(_ value: String) -> String {
        let data = try? JSONEncoder().encode(value)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
    }
}
