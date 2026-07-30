import AppKit
import Foundation
import WebKit

@MainActor
public final class DashboardBridge: NSObject, WKScriptMessageHandler {
    private weak var webView: WKWebView?
    private let configStore: AppConfigStore
    private let backend: BackendProcessManager
    private let scanner: LocalModelScanner
    private let downloadService: ModelDownloadService
    private let deletionService: LocalModelDeletionService
    private let integrityService: ModelIntegrityVerificationService
    private let versionService: ModelVersionManagementService
    private let profileGenerationService: SchedulerProfileGenerationService
    private let benchmarkService: BenchmarkService
    private let benchmarkSessionCoordinator: BenchmarkExclusiveSessionCoordinator
    private let restartCoordinator: BackendRestartCoordinator
    private let parameterStore: ModelParameterStore
    private let notificationCenter: NotificationCenter
    private var huggingFaceSearchTask: Task<Void, Never>?

    public init(
        webView: WKWebView,
        configStore: AppConfigStore,
        backend: BackendProcessManager,
        scanner: LocalModelScanner = LocalModelScanner(),
        downloadService: ModelDownloadService = ModelDownloadService(),
        deletionService: LocalModelDeletionService? = nil,
        integrityService: ModelIntegrityVerificationService = ModelIntegrityVerificationService(),
        versionService: ModelVersionManagementService = ModelVersionManagementService(),
        profileGenerationService: SchedulerProfileGenerationService = SchedulerProfileGenerationService(),
        benchmarkService: BenchmarkService = BenchmarkService(),
        benchmarkSessionCoordinator: BenchmarkExclusiveSessionCoordinator = BenchmarkExclusiveSessionCoordinator(),
        restartCoordinator: BackendRestartCoordinator? = nil,
        parameterStore: ModelParameterStore = .shared,
        notificationCenter: NotificationCenter = .default
    ) {
        self.webView = webView
        self.configStore = configStore
        self.backend = backend
        self.scanner = scanner
        self.downloadService = downloadService
        self.deletionService = deletionService ?? LocalModelDeletionService(configStore: configStore)
        self.integrityService = integrityService
        self.versionService = versionService
        self.profileGenerationService = profileGenerationService
        self.benchmarkService = benchmarkService
        self.benchmarkSessionCoordinator = benchmarkSessionCoordinator
        self.parameterStore = parameterStore
        self.notificationCenter = notificationCenter
        self.restartCoordinator = restartCoordinator ?? BackendRestartCoordinator(
            scanner: scanner,
            parameterStore: parameterStore
        )
        super.init()
        notificationCenter.addObserver(
            self,
            selector: #selector(loadedModelsDidChange(_:)),
            name: .ironMLXLoadedModelsDidChange,
            object: nil
        )
    }

    deinit {
        huggingFaceSearchTask?.cancel()
        notificationCenter.removeObserver(self)
        let downloadService = downloadService
        Task {
            await downloadService.cancelAllDownloads()
        }
    }

    public func cancelAllDownloads() {
        Task {
            await downloadService.cancelAllDownloads()
        }
    }

    public static let handlerNames = [
        "fetchAPI",
        "fetchAPIPost",
        "fetchAPIDelete",
        "setLanguage",
        "setTheme",
        "setDefaultModel",
        "deleteModels",
        "verifyModelIntegrity",
        "listModelVersions",
        "activateModelVersion",
        "deleteModelVersions",
        "saveSettings",
        "restartServer",
        "loadModel",
        "forceLoadModel",
        "unloadModel",
        "downloadModel",
        "cancelModelDownload",
        "searchHF",
        "cancelHFSearch",
        "scanLocalModels",
        "syncLoadedModels",
        "getAppLogs",
        "dashboardLog",
        "previewSchedulerProfileGeneration",
        "generateSchedulerProfile",
        "cancelSchedulerProfileGeneration",
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
            updateConfig { $0.defaultModel = model }
            setBackendDefaultModelIfLoaded(model)
        case "deleteModels":
            deleteModels(json: stringBody(body))
        case "verifyModelIntegrity":
            verifyModelIntegrity(repoID: stringBody(body))
        case "listModelVersions":
            listModelVersions(json: stringBody(body))
        case "activateModelVersion":
            activateModelVersion(json: stringBody(body))
        case "deleteModelVersions":
            deleteModelVersions(json: stringBody(body))
        case "saveSettings":
            saveSettings(json: stringBody(body))
        case "restartServer":
            restartBackend()
        case "loadModel", "forceLoadModel":
            loadBackendModel(instruction: Self.modelLoadInstruction(from: body), callback: .modelLoaded)
        case "unloadModel":
            unloadBackendModel(modelReference: stringBody(body), callback: .modelUnloaded)
        case "downloadModel":
            startHuggingFaceDownload(json: stringBody(body))
        case "cancelModelDownload":
            cancelModelDownload(json: stringBody(body))
        case "searchHF":
            searchHuggingFace(json: stringBody(body))
        case "cancelHFSearch":
            cancelHuggingFaceSearch()
        case "scanLocalModels":
            sendScannedModels()
        case "syncLoadedModels":
            syncLoadedModels()
        case "getAppLogs":
            sendAppLogs()
        case "dashboardLog":
            logDashboardMessage(json: stringBody(body))
        case "previewSchedulerProfileGeneration":
            previewSchedulerProfileGeneration(json: stringBody(body))
        case "generateSchedulerProfile":
            generateSchedulerProfile(json: stringBody(body))
        case "cancelSchedulerProfileGeneration":
            cancelSchedulerProfileGeneration()
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
                let state = await self.loadedModelState()
                let models = self.scanner.scan(
                    loadedModels: state.references,
                    pinnedModels: state.pinnedModels,
                    mtpEnabledModels: state.mtpEnabledModels
                )
                let benchmarkModels = models.filter { $0.readiness?.isLoadable != false }.map {
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
                loadBackendModel(
                    instruction: ModelLoadInstruction(modelReference: model, useMtp: nil, mtpModelID: nil),
                    callback: .fetch(path: payload.path)
                )
            } else {
                sendFetchResult(path: payload.path, jsonString: "null")
            }
        case "/admin/api/prompt-lookup/clear":
            clearSharedPromptLookup(path: payload.path)
        case "/v1/models/pin":
            setBackendPinnedModel(payload: payload, pinned: true)
        case "/v1/models/unpin":
            setBackendPinnedModel(payload: payload, pinned: false)
        case "/admin/api/models/ms/download":
            if let repoID = payload.body["repo_id"]?.stringValue {
                startModelScopeDownload(repoID: repoID, path: payload.path)
            } else {
                sendFetchResult(path: payload.path, jsonString: "{\"success\":false,\"status\":\"error\",\"error\":\"missing repo_id\"}")
            }
        case "/admin/api/models/download/cancel":
            let repoID = payload.body["repo_id"]?.stringValue ?? ""
            let provider = payload.body["provider"]?.stringValue ?? ModelRepositoryProvider.huggingFace.rawValue
            cancelModelDownload(
                repoID: repoID,
                providerName: provider,
                path: payload.path
            )
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

    private func clearSharedPromptLookup(path: String) {
        let config = configStore.load()
        Task {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                let response = try await client.clearSharedPromptLookup()
                let json = try Self.jsonString(response)
                await MainActor.run {
                    self.sendFetchResult(path: path, jsonString: json)
                }
            } catch {
                let errorJSON = self.backendErrorJSON(error)
                await MainActor.run {
                    self.sendFetchResult(path: path, jsonString: errorJSON)
                }
            }
        }
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
                    try self.backend.start()
                }
                try await client.waitUntilReady()
                let result = try await benchmarkSessionCoordinator.prepare(
                    client: client,
                    targetModel: target.model,
                    targetModelPath: target.path,
                    validateModelPath: self.benchmarkModelPathValidator(config: config)
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
        let config = configStore.load()
        guard let model = payload.body["model"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty,
              let modelPath = try? scanner.verifiedModelPath(for: model)
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
                let result = try await benchmarkSessionCoordinator.restore(
                    client: client,
                    validateModelPath: self.benchmarkModelPathValidator(config: config)
                )
                let json = try Self.jsonString(result)
                await MainActor.run {
                    if result.status != "not_active" {
                        self.updateConfig {
                            $0.replaceLoadedModels(result.restoredModels, defaultModel: result.defaultModel)
                        }
                        notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: self)
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
            return
        }

        cancelHuggingFaceSearch()
        let downloadService = downloadService
        let versionService = versionService
        huggingFaceSearchTask = Task { [weak self] in
            do {
                var results = try await downloadService.searchHuggingFace(
                    query: payload.query,
                    sort: payload.sort,
                    token: payload.token
                )
                try Task.checkCancellation()
                guard let self else {
                    return
                }
                let backendModels = await backendLoadedModels()
                try Task.checkCancellation()
                let loadedPaths = Set(backendModels.map(\.path))
                for index in results.indices {
                    try Task.checkCancellation()
                    let repoID = results[index].modelId ?? results[index].id
                    let local = versionService.searchLocalState(
                        provider: .huggingFace,
                        repoID: repoID,
                        remoteCommitSHA: results[index].sha,
                        loadedModelPaths: loadedPaths
                    )
                    results[index].localState = local.state
                    results[index].localCommitSHA = local.localCommitSHA
                }
                let json = (try? Self.jsonString(results)) ?? "[]"
                try Task.checkCancellation()
                self.sendJavaScript(
                    "onSearchResults(\(payload.requestID), \(Self.jsStringLiteral(json)))"
                )
            } catch {
                guard !Task.isCancelled else {
                    return
                }
                let code: String?
                if let resolutionError = error as? RepositoryResolutionError {
                    switch resolutionError {
                    case .notFound:
                        code = "repo_not_found"
                    case .invalidCommit, .incompleteMetadata:
                        code = nil
                    }
                } else {
                    code = nil
                }
                let result = ModelDownloadCompletion(
                    success: false,
                    message: nil,
                    error: error.localizedDescription,
                    code: code,
                    repoID: payload.query
                )
                let json = (try? Self.jsonString(result))
                    ?? #"{"success":false,"error":"HuggingFace search failed."}"#
                self?.sendJavaScript(
                    "onSearchError(\(payload.requestID), \(Self.jsStringLiteral(json)))"
                )
            }
        }
    }

    private func cancelHuggingFaceSearch() {
        huggingFaceSearchTask?.cancel()
        huggingFaceSearchTask = nil
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

    private func cancelModelDownload(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(ModelDownloadCancellationPayload.self, from: data)
        else {
            return
        }
        cancelModelDownload(repoID: payload.repoID, providerName: payload.provider, path: nil)
    }

    private func cancelModelDownload(repoID: String, providerName: String, path: String?) {
        guard let provider = ModelRepositoryProvider(rawValue: providerName) else {
            if let path {
                sendFetchResult(path: path, jsonString: "{\"success\":false,\"code\":\"invalid_download_identity\"}")
            }
            return
        }
        Task {
            let cancelled = await downloadService.cancelDownload(
                provider: provider,
                repoID: repoID
            )
            let json = cancelled
                ? "{\"success\":true,\"status\":\"cancelling\"}"
                : "{\"success\":false,\"code\":\"download_not_active\"}"
            await MainActor.run {
                if let path {
                    self.sendFetchResult(path: path, jsonString: json)
                }
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
            syncLoadedModels()
            sendJavaScript("\(defaultSync)onModelsDeleted(\(Self.jsStringLiteral(resultJSON)))")
        } catch {
            sendJavaScript(
                "showToast(\(Self.jsStringLiteral("Failed to delete model: \(error.localizedDescription)")), 'warn')"
            )
        }
    }

    private func verifyModelIntegrity(repoID: String) {
        let repoID = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repoID.isEmpty else {
            return
        }
        Task {
            do {
                let result = try await integrityService.verify(repoID: repoID) { [weak self] status in
                    guard status.state == "verifying" else {
                        return
                    }
                    let json = (try? Self.jsonString(status)) ?? "{}"
                    Task { @MainActor in
                        self?.sendJavaScript(
                            "onModelIntegrityStatus(\(Self.jsStringLiteral(json)))"
                        )
                    }
                }
                let json = (try? Self.jsonString(result)) ?? "{}"
                await MainActor.run {
                    self.sendJavaScript(
                        "onModelIntegrityStatus(\(Self.jsStringLiteral(json)))"
                    )
                    self.sendScannedModels()
                }
            } catch {
                let status = ModelIntegrityStatus(
                    repoID: repoID,
                    state: "error",
                    error: error.localizedDescription
                )
                let json = (try? Self.jsonString(status)) ?? "{}"
                await MainActor.run {
                    self.sendJavaScript(
                        "onModelIntegrityStatus(\(Self.jsStringLiteral(json)))"
                    )
                }
            }
        }
    }

    private func listModelVersions(json: String) {
        guard let payload = decodeVersionRepositoryPayload(json),
              let provider = ModelRepositoryProvider(rawValue: payload.provider)
        else {
            sendModelVersionOperationError("Invalid model version request.")
            return
        }
        Task {
            let loadedModels = await backendLoadedModels()
            let loadedPaths = Set(loadedModels.map(\.path))
            let repoID = payload.repoID
            do {
                let service = versionService
                let list = try await Task.detached {
                    try service.versions(
                        provider: provider,
                        repoID: repoID,
                        loadedModelPaths: loadedPaths
                    )
                }.value
                let result = try Self.jsonString(list)
                await MainActor.run {
                    self.sendJavaScript(
                        "onModelVersions(\(Self.jsStringLiteral(result)))"
                    )
                }
            } catch {
                await MainActor.run {
                    self.sendModelVersionOperationError(error.localizedDescription)
                }
            }
        }
    }

    private func activateModelVersion(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(ModelVersionActivationPayload.self, from: data),
              let provider = ModelRepositoryProvider(rawValue: payload.provider)
        else {
            sendModelVersionOperationError("Invalid model version activation request.")
            return
        }
        let config = configStore.load()
        let repoID = payload.repoID
        let commitSHA = payload.commitSHA
        let fullChecksum = config.verifyModelOnLoad == true
        Task {
            let service = versionService
            do {
                let loadedModels = try await requiredBackendLoadedModels()
                let loaded = loadedModel(
                    provider: provider,
                    repoID: repoID,
                    in: loadedModels
                )
                let activation = try await Task.detached {
                    try service.activate(
                        provider: provider,
                        repoID: repoID,
                        commitSHA: commitSHA,
                        fullChecksum: fullChecksum
                    )
                }.value
                var reloadStatus: String?
                if let loaded {
                    do {
                        let client = BackendAPIClient(host: config.host, port: config.port)
                        let targetPath = try await scanner.verifiedModelPathAsync(
                            for: repoID,
                            fullChecksum: false
                        )
                        let parameters = parameterStore.parameters(for: loaded.id)
                            ?? parameterStore.parameters(for: repoID)
                        let response = try await client.loadModel(
                            model: loaded.id,
                            modelDir: targetPath,
                            setDefault: loaded.isDefault,
                            maxCacheCap: parameters?.maxCacheCap,
                            pinned: loaded.pinned,
                            mtpModelDir: loaded.mtpModelDir,
                            mtpDraftTokens: loaded.mtpDraftTokens,
                            promptLookup: loaded.promptLookup,
                            reloadWhenIdle: true,
                            deferWhenBusy: false,
                            samplingDefaults: parameters?.samplingDefaults ?? .empty
                        )
                        reloadStatus = response.status
                        await MainActor.run {
                            self.persistBackendLoadedModels(response.loadedModels)
                        }
                    } catch {
                        if let previousCommit = activation.previousCommitSHA {
                            _ = try? await Task.detached {
                                try service.activate(
                                    provider: provider,
                                    repoID: repoID,
                                    commitSHA: previousCommit,
                                    fullChecksum: false
                                )
                            }.value
                        }
                        throw error
                    }
                }
                let result = ModelVersionBridgeOperationResult(
                    success: true,
                    provider: provider.rawValue,
                    repoID: repoID,
                    activeCommitSHA: activation.activeCommitSHA,
                    deletedCommitSHAs: nil,
                    reclaimedBytes: nil,
                    reloadStatus: reloadStatus,
                    error: nil
                )
                let resultJSON = try Self.jsonString(result)
                await MainActor.run {
                    self.sendJavaScript(
                        "onModelVersionOperation(\(Self.jsStringLiteral(resultJSON)))"
                    )
                    self.sendScannedModels()
                    self.listModelVersions(json: json)
                }
            } catch {
                await MainActor.run {
                    self.sendModelVersionOperationError(error.localizedDescription)
                    self.listModelVersions(json: json)
                    self.sendScannedModels()
                }
            }
        }
    }

    private func deleteModelVersions(json: String) {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(ModelVersionDeletionPayload.self, from: data),
              let provider = ModelRepositoryProvider(rawValue: payload.provider)
        else {
            sendModelVersionOperationError("Invalid model version deletion request.")
            return
        }
        let repoID = payload.repoID
        let commitSHAs = payload.commitSHAs
        Task {
            let service = versionService
            do {
                let loadedModels = try await requiredBackendLoadedModels()
                let loadedPaths = Set(loadedModels.map(\.path))
                let deletion = try await Task.detached {
                    try service.deleteVersions(
                        provider: provider,
                        repoID: repoID,
                        commitSHAs: commitSHAs,
                        loadedModelPaths: loadedPaths
                    )
                }.value
                let result = ModelVersionBridgeOperationResult(
                    success: true,
                    provider: provider.rawValue,
                    repoID: repoID,
                    activeCommitSHA: nil,
                    deletedCommitSHAs: deletion.deletedCommitSHAs,
                    reclaimedBytes: deletion.reclaimedBytes,
                    reloadStatus: nil,
                    error: nil
                )
                let resultJSON = try Self.jsonString(result)
                await MainActor.run {
                    self.sendJavaScript(
                        "onModelVersionOperation(\(Self.jsStringLiteral(resultJSON)))"
                    )
                    self.sendScannedModels()
                    self.listModelVersions(json: json)
                }
            } catch {
                await MainActor.run {
                    self.sendModelVersionOperationError(error.localizedDescription)
                    self.listModelVersions(json: json)
                }
            }
        }
    }

    private func sendModelVersionOperationError(_ message: String) {
        let result = ModelVersionBridgeOperationResult(
            success: false,
            provider: nil,
            repoID: nil,
            activeCommitSHA: nil,
            deletedCommitSHAs: nil,
            reclaimedBytes: nil,
            reloadStatus: nil,
            error: message
        )
        let json = (try? Self.jsonString(result)) ?? #"{"success":false}"#
        sendJavaScript("onModelVersionOperation(\(Self.jsStringLiteral(json)))")
    }

    private func sendDownloadComplete(_ result: ModelDownloadCompletion) {
        let json = (try? Self.jsonString(result)) ?? "{\"success\":false,\"error\":\"Download failed.\"}"
        sendJavaScript("onDownloadComplete(\(Self.jsStringLiteral(json)))")
    }

    private func loadBackendModel(instruction: ModelLoadInstruction, callback: ModelOperationCallback) {
        let model = instruction.modelReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            deliverModelOperationResult(error: "No model is configured.", callback: callback)
            return
        }
        if let readiness = scanner.readiness(for: model), !readiness.isLoadable {
            let detail = readiness.message ?? "Model snapshot is not ready to load."
            deliverModelOperationResult(error: detail, callback: callback)
            return
        }
        let config = configStore.load()
        let pinned = config.pinnedModelReferences.contains(model)
        let resolvedModel: String
        do {
            resolvedModel = try scanner.verifiedModelPath(
                for: model,
                fullChecksum: false
            )
        } catch {
            deliverModelOperationResult(error: error.localizedDescription, callback: callback)
            return
        }
        let maxCacheCap = ModelLoadParameters.maxCacheCap(
            for: model,
            scanner: scanner,
            parameterStore: parameterStore,
            activeKvOffloadEnabled: config.activeKvOffload == true
        )
        let promptLookup = promptLookupConfig(for: model, instruction: instruction)
        let mtpRuntime: ModelMtpRuntime?
        do {
            mtpRuntime = try ModelMtpRuntimeResolver.runtime(
                for: model,
                useMtp: instruction.useMtp,
                explicitMtpModelID: instruction.mtpModelID,
                scanner: scanner,
                parameterStore: parameterStore,
                fullChecksum: false
            )
        } catch {
            deliverModelOperationResult(error: error.localizedDescription, callback: callback)
            return
        }
        Task {
            do {
                let loadModelPath: String
                let loadMtpRuntime: ModelMtpRuntime?
                if config.verifyModelOnLoad == true {
                    loadModelPath = try await self.scanner.verifiedModelPathAsync(
                        for: model,
                        fullChecksum: true
                    )
                    loadMtpRuntime = try await ModelMtpRuntimeResolver.runtimeAsync(
                        for: model,
                        useMtp: instruction.useMtp,
                        explicitMtpModelID: instruction.mtpModelID,
                        scanner: self.scanner,
                        parameterStore: self.parameterStore,
                        fullChecksum: true
                    )
                } else {
                    loadModelPath = resolvedModel
                    loadMtpRuntime = mtpRuntime
                }
                try await MainActor.run {
                    try self.backend.start()
                }
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                await self.registerLocalModels(config: config, client: client)
                let loadedModels = try await client.fetchLoadedModels()
                let setDefault = Self.shouldSetDefaultWhenLoadingModel(
                    model,
                    config: config,
                    currentLoadedModelCount: loadedModels.count
                )
                let response = try await client.loadModel(
                    model: model,
                    modelDir: loadModelPath,
                    setDefault: setDefault,
                    maxCacheCap: maxCacheCap,
                    pinned: pinned,
                    mtpModelDir: loadMtpRuntime?.modelDir,
                    mtpDraftTokens: loadMtpRuntime?.draftTokens,
                    promptLookup: promptLookup,
                    reloadWhenIdle: false,
                    samplingDefaults: self.parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
                )
                let json = try Self.jsonString(response)
                await MainActor.run {
                    self.persistMtpLoadPreferenceIfRequested(
                        model: model,
                        instruction: instruction,
                        mtpRuntime: loadMtpRuntime
                    )
                    self.persistBackendLoadedModels(response.loadedModels)
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
                    self.persistBackendLoadedModels(response.loadedModels)
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

    private func setBackendPinnedModel(payload: APIPostPayload, pinned: Bool) {
        guard let model = payload.body["model"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty
        else {
            sendFetchResult(
                path: payload.path,
                jsonString: "{\"success\":false,\"status\":\"error\",\"error\":\"missing model\"}"
            )
            return
        }
        let config = configStore.load()
        Task {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                try await client.waitUntilReady()
                let response = pinned
                    ? try await client.pinModel(model: model)
                    : try await client.unpinModel(model: model)
                let modelID = Self.loadedModelID(matching: model, in: response.loadedModels) ?? model
                let json = try Self.jsonString(
                    PinModelBridgeResponse(
                        success: response.success,
                        status: response.success ? "ok" : response.status,
                        model: modelID,
                        pinned: pinned,
                        loadedModels: response.loadedModels,
                        error: response.error
                    )
                )
                await MainActor.run {
                    if response.success {
                        self.persistBackendLoadedModels(response.loadedModels)
                        self.updateConfig { $0.recordPinnedModel(modelID, pinned: pinned) }
                    }
                    self.sendFetchResult(path: payload.path, jsonString: json)
                    self.sendScannedModels()
                }
            } catch {
                let errorJSON = self.backendErrorJSON(error)
                await MainActor.run {
                    self.sendFetchResult(path: payload.path, jsonString: errorJSON)
                }
            }
        }
    }

    private static func loadedModelID(matching model: String, in loadedModels: [BackendLoadedModelInfo]) -> String? {
        loadedModels.first { candidate in
            candidate.id == model || candidate.model == model || candidate.path == model
        }?.id
    }

    nonisolated static func shouldSetDefaultWhenLoadingModel(
        _ model: String,
        config: AppConfig,
        currentLoadedModelCount: Int
    ) -> Bool {
        guard let model = AppConfig.normalizedModelReference(model) else {
            return false
        }
        if config.defaultModelReference == model {
            return true
        }
        return currentLoadedModelCount == 0
    }

    private func persistMtpLoadPreferenceIfRequested(
        model: String,
        instruction: ModelLoadInstruction,
        mtpRuntime: ModelMtpRuntime?
    ) {
        guard let useMtp = instruction.useMtp else {
            return
        }
        do {
            try parameterStore.recordMtpLoadPreference(
                modelID: model,
                enabled: useMtp,
                mtpModelID: mtpRuntime?.modelID ?? instruction.mtpModelID
            )
        } catch {
            IronMLXAppLogger.error("Failed to persist MTP load preference for \(model): \(error)")
        }
    }

    private func promptLookupConfig(
        for model: String,
        instruction: ModelLoadInstruction
    ) -> BackendPromptLookupConfig? {
        let saved = parameterStore.parameters(for: model)
        guard instruction.usePromptLookup ?? saved?.promptLookupEnabled ?? false else {
            return nil
        }
        return BackendPromptLookupConfig(
            crossRequest: instruction.crossRequestPromptLookup
                ?? saved?.promptLookupCrossRequest
                ?? false
        )
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
                if result.success {
                    self.updateConfig {
                        $0.replaceLoadedModels(result.loadedModels, defaultModel: result.model)
                    }
                    notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: self)
                }
                self.sendJavaScript("onServerRestarted(\(Self.jsStringLiteral(json)))")
                self.sendScannedModels()
            }
        }
    }

    private func saveSettings(json: String) {
        let existing = configStore.load()
        let config: AppConfig
        do {
            config = try Self.config(applyingSettingsJSON: json, to: existing)
        } catch {
            return
        }

        configStore.save(config)
        notifyMenuLanguageDidChange()
        let needsRestart = Self.backendRestartRequired(from: existing, to: config)
        let response = #"{"status":"ok","needs_restart":\#(needsRestart)}"#
        sendJavaScript("onSettingsSaved(\(Self.jsStringLiteral(response)))")
    }

    static func backendRestartRequired(from existing: AppConfig, to updated: AppConfig) -> Bool {
        existing.host != updated.host
            || existing.port != updated.port
            || existing.memLimitTotal != updated.memLimitTotal
            || existing.memLimitModel != updated.memLimitModel
            || existing.memTotalAuto != updated.memTotalAuto
            || existing.memTotal != updated.memTotal
            || existing.memModelAuto != updated.memModelAuto
            || existing.memModel != updated.memModel
            || existing.hotCache != updated.hotCache
            || existing.coldCache != updated.coldCache
            || existing.cacheEnable != updated.cacheEnable
            || existing.cacheDir != updated.cacheDir
            || existing.kvQuant != updated.kvQuant
            || existing.activeKvOffload != updated.activeKvOffload
            || existing.maxSequences != updated.maxSequences
            || existing.maxModels != updated.maxModels
            || existing.modelTtlMinutes != updated.modelTtlMinutes
            || existing.distributedBackend != updated.distributedBackend
            || existing.parallelMode != updated.parallelMode
            || existing.prefillChunkSize != updated.prefillChunkSize
            || existing.admissionDeadlineMs != updated.admissionDeadlineMs
            || existing.admissionQueueMax != updated.admissionQueueMax
            || existing.maxCacheCap != updated.maxCacheCap
            || existing.decodeCadenceMidChunkCap != updated.decodeCadenceMidChunkCap
            || existing.schedulerProfile != updated.schedulerProfile
            || existing.schedulerAutotuneReport != updated.schedulerAutotuneReport
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
        config.memModel = intValue(object, "mem_model") ?? config.memModel
        config.hotCache = intValue(object, "hot_cache") ?? config.hotCache
        config.coldCache = intValue(object, "cold_cache") ?? config.coldCache
        config.cacheEnable = boolValue(object, "cache_enable") ?? config.cacheEnable
        config.cacheDir = stringValue(object, "cache_dir") ?? config.cacheDir
        config.kvQuant = stringValue(object, "kv_quant") ?? config.kvQuant
        config.activeKvOffload = boolValue(object, "active_kv_offload") ?? config.activeKvOffload
        config.maxSequences = intValue(object, "max_sequences") ?? config.maxSequences
        config.bMax = nil
        config.maxModels = intValue(object, "max_models") ?? config.maxModels
        config.modelTtlMinutes = intValue(object, "model_ttl_minutes") ?? config.modelTtlMinutes
        config.verifyModelOnLoad = boolValue(object, "verify_model_on_load") ?? config.verifyModelOnLoad
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
                let response: BackendModelAdminResponse
                if let resolvedModel = try? self.scanner.verifiedModelPath(for: model) {
                    let mtpRuntime = try? ModelMtpRuntimeResolver.runtime(
                        for: model,
                        useMtp: nil,
                        scanner: self.scanner,
                        parameterStore: self.parameterStore
                    )
                    response = try await client.registerModel(
                        model: model,
                        modelDir: resolvedModel,
                        setDefault: true,
                        maxCacheCap: ModelLoadParameters.maxCacheCap(
                            for: model,
                            scanner: self.scanner,
                            parameterStore: self.parameterStore,
                            activeKvOffloadEnabled: config.activeKvOffload == true
                        ),
                        pinned: config.pinnedModelReferences.contains(model),
                        mtpModelDir: mtpRuntime?.modelDir,
                        mtpDraftTokens: mtpRuntime?.draftTokens,
                        promptLookup: self.parameterStore.parameters(for: model)?.promptLookupConfig,
                        samplingDefaults: self.parameterStore.parameters(for: model)?.samplingDefaults ?? .empty
                    )
                } else {
                    response = try await client.setDefaultModel(model)
                }
                await MainActor.run {
                    self.persistBackendLoadedModels(response.loadedModels, preferredDefaultModel: model)
                }
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
                let resolvedModel = try await scanner.verifiedModelPathAsync(
                    for: model,
                    fullChecksum: config.verifyModelOnLoad == true
                )
                let mtpRuntime = try await ModelMtpRuntimeResolver.runtimeAsync(
                    for: model,
                    useMtp: nil,
                    scanner: scanner,
                    parameterStore: parameterStore,
                    fullChecksum: config.verifyModelOnLoad == true
                )
                let response = try await client.loadModel(
                    model: model,
                    modelDir: resolvedModel,
                    setDefault: loaded.isDefault,
                    maxCacheCap: parameters.maxCacheCap,
                    pinned: loaded.pinned,
                    mtpModelDir: mtpRuntime?.modelDir,
                    mtpDraftTokens: mtpRuntime?.draftTokens,
                    promptLookup: parameters.promptLookupConfig,
                    reloadWhenIdle: true,
                    samplingDefaults: parameters.samplingDefaults
                )
                let json = try Self.jsonString(response)
                await MainActor.run {
                    self.persistBackendLoadedModels(response.loadedModels)
                    self.sendJavaScript("onModelParamsReloaded(\(Self.jsStringLiteral(json)))")
                    self.sendScannedModels()
                }
            } catch {
                let response = BackendModelAdminResponse(
                    success: false,
                    status: "error",
                    code: nil,
                    model: nil,
                    loadedModels: [],
                    warningCode: nil,
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
        await loadedModelState().references
    }

    private func backendLoadedModels() async -> [BackendLoadedModelInfo] {
        guard backend.isRunning else {
            return []
        }
        let config = configStore.load()
        do {
            return try await BackendAPIClient(host: config.host, port: config.port)
                .fetchLoadedModels()
        } catch {
            IronMLXAppLogger.error("Failed to fetch loaded model paths: \(error)")
            return []
        }
    }

    private func requiredBackendLoadedModels() async throws -> [BackendLoadedModelInfo] {
        guard backend.isRunning else {
            return []
        }
        let config = configStore.load()
        return try await BackendAPIClient(host: config.host, port: config.port)
            .fetchLoadedModels()
    }

    private func loadedModel(
        provider: ModelRepositoryProvider,
        repoID: String,
        in models: [BackendLoadedModelInfo]
    ) -> BackendLoadedModelInfo? {
        guard let repository = try? ModelRepositoryLayout.repositoryRoot(
            rootURL: versionService.rootURL,
            provider: provider,
            repoID: repoID
        ) else {
            return nil
        }
        let snapshotsRoot = repository
            .appendingPathComponent("snapshots", isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath()
            .path + "/"
        return models.first { model in
            let path = URL(fileURLWithPath: model.path)
                .standardizedFileURL
                .resolvingSymlinksInPath()
                .path
            return path.hasPrefix(snapshotsRoot)
        }
    }

    private func loadedModelState() async -> (references: Set<String>, pinnedModels: Set<String>, mtpEnabledModels: Set<String>) {
        let config = configStore.load()
        var loaded = Set<String>()
        var pinned = Set<String>()
        var mtpEnabled = Set<String>()
        if backend.isRunning {
            do {
                let client = BackendAPIClient(host: config.host, port: config.port)
                let models = try await client.fetchLoadedModels()
                syncPersistedLoadedModelsIfNeeded(models)
                for model in models {
                    loaded.insert(model.id)
                    loaded.insert(model.model)
                    loaded.insert(model.path)
                    if model.pinned {
                        pinned.insert(model.id)
                        pinned.insert(model.model)
                        pinned.insert(model.path)
                    }
                    if model.mtpEnabled {
                        mtpEnabled.insert(model.id)
                    }
                }
                return (loaded, pinned, mtpEnabled)
            } catch {
                IronMLXAppLogger.error("Failed to fetch loaded models: \(error)")
            }
        }
        for model in config.restoredModelReferences {
            loaded.insert(model)
        }
        for model in config.pinnedModelReferences {
            pinned.insert(model)
        }
        return (loaded, pinned, mtpEnabled)
    }

    private func syncPersistedLoadedModelsIfNeeded(_ models: [BackendLoadedModelInfo]) {
        let backendLoaded = AppConfig.normalizedModelReferences(models.map(\.id))
        let backendPinned = AppConfig.normalizedModelReferences(models.filter(\.pinned).map(\.id))
        let config = configStore.load()
        let persistedLoaded = AppConfig.normalizedModelReferences(config.loadedModels ?? [])
        let persistedPinned = config.pinnedModelReferences
        let backendDefault = AppConfig.normalizedModelReference(models.first(where: \.isDefault)?.id)
        let defaultChanged = backendDefault != nil && backendDefault != config.defaultModelReference
        guard Set(backendLoaded) != Set(persistedLoaded)
            || Set(backendPinned) != Set(persistedPinned)
            || defaultChanged
        else {
            return
        }
        persistBackendLoadedModels(models)
    }

    private func persistBackendLoadedModels(
        _ models: [BackendLoadedModelInfo],
        preferredDefaultModel: String? = nil
    ) {
        let loaded = models.map(\.id)
        let backendDefault = models.first(where: \.isDefault)?.id
        let currentDefault = configStore.load().defaultModelReference
        let defaultModel = backendDefault ?? preferredDefaultModel ?? currentDefault
        updateConfig {
            $0.replaceLoadedModels(loaded, defaultModel: backendDefault)
            $0.replacePinnedModels(models.filter(\.pinned).map(\.id))
            if let defaultModel {
                $0.defaultModel = defaultModel
            }
        }
        notificationCenter.post(name: .ironMLXLoadedModelsDidChange, object: self)
        if let defaultModel {
            sendJavaScript("window.__DEFAULT_MODEL__ = \(Self.jsStringLiteral(defaultModel));")
        }
    }

    private func sendScannedModels() {
        Task {
            let state = await self.loadedModelState()
            let models = self.scanner.scan(
                loadedModels: state.references,
                pinnedModels: state.pinnedModels,
                mtpEnabledModels: state.mtpEnabledModels
            )
            let config = self.configStore.load()
            let dashboardModels = self.modelsWithEffectiveMaxTokens(
                models,
                activeKvOffloadEnabled: config.activeKvOffload == true
            )
            if self.backend.isRunning {
                let client = BackendAPIClient(host: config.host, port: config.port)
                await self.registerLocalModels(models: models, config: config, client: client)
            }
            let json = (try? Self.jsonString(dashboardModels)) ?? "[]"
            await MainActor.run {
                self.sendModelParameters()
                self.sendJavaScript("onLocalModelsScanned(\(Self.jsStringLiteral(json)))")
            }
        }
    }

    private func modelsWithEffectiveMaxTokens(
        _ models: [LocalModel],
        activeKvOffloadEnabled: Bool
    ) -> [LocalModel] {
        models.map { model in
            var model = model
            model.effectiveMaxTokens = ModelLoadParameters.effectiveMaxCacheCap(
                savedMaxCacheCap: parameterStore.parameters(for: model.id)?.maxCacheCap,
                contextWindow: model.maxPositionEmbeddings,
                activeKvOffloadEnabled: activeKvOffloadEnabled
            )
            return model
        }
    }

    private func registerLocalModels(
        models: [LocalModel]? = nil,
        config: AppConfig,
        client: BackendAPIClient
    ) async {
        let configPinned = Set(config.pinnedModelReferences)
        let localModels = models ?? scanner.scan(loadedModels: [], pinnedModels: configPinned, mtpEnabledModels: [])
        await LocalModelBackendRegistrar.register(
            localModels: localModels,
            defaultModel: config.defaultModelReference,
            scanner: scanner,
            parameterStore: parameterStore,
            activeKvOffloadEnabled: config.activeKvOffload == true,
            client: client
        )
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

    private func backendErrorJSON(_ error: Error) -> String {
        if case BackendAPIError.serverResponse(statusCode: _, body: let body) = error,
           let body,
           !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return body
        }
        let payload = BridgeErrorResponse(success: false, status: "error", error: error.localizedDescription)
        return (try? Self.jsonString(payload)) ?? "{\"success\":false,\"status\":\"error\"}"
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
        guard current.state != "running", current.state != "cancelling" else {
            sendSchedulerProfileStatus(current)
            return
        }
        Task {
            do {
                let request = try await schedulerProfileGenerationRequest(
                    json: json,
                    fullChecksum: configStore.load().verifyModelOnLoad == true
                )
                let shouldRestartBackend = backend.isRunning
                if shouldRestartBackend {
                    backend.stop()
                }
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
            } catch {
                sendSchedulerProfileStatus(
                    SchedulerProfileGenerationStatus.failed(
                        request: nil,
                        error: error.localizedDescription
                    )
                )
            }
        }
    }

    private func previewSchedulerProfileGeneration(json: String) {
        let payload = json.data(using: .utf8)
            .flatMap { try? JSONDecoder().decode(SchedulerProfileGenerationPayload.self, from: $0) }
        Task {
            do {
                sendSchedulerProfilePreview(
                    SchedulerProfileGenerationPreview(
                        request: try await schedulerProfileGenerationRequest(
                            json: json,
                            fullChecksum: false
                        ),
                        requestToken: payload?.requestToken
                    )
                )
            } catch {
                sendSchedulerProfilePreview(
                    .failed(error: error.localizedDescription, requestToken: payload?.requestToken)
                )
            }
        }
    }

    private func schedulerProfileGenerationRequest(
        json: String,
        fullChecksum: Bool
    ) async throws -> SchedulerProfileGenerationRequest {
        guard let data = json.data(using: .utf8),
              let payload = try? JSONDecoder().decode(SchedulerProfileGenerationPayload.self, from: data)
        else {
            throw SchedulerProfileGenerationRequestError.invalidRequest
        }
        let model = payload.model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            throw SchedulerProfileGenerationRequestError.noModelSelected
        }
        let config = configStore.load()
        guard let modelPath = try? await scanner.verifiedModelPathAsync(
            for: model,
            fullChecksum: fullChecksum
        ) else {
            throw SchedulerProfileGenerationRequestError.modelNotFound(model)
        }

        let launchOptions = BackendLaunchOptions(config: config)
        let mtpRuntime = try await ModelMtpRuntimeResolver.runtimeAsync(
            for: model,
            useMtp: nil,
            scanner: scanner,
            parameterStore: parameterStore,
            fullChecksum: fullChecksum
        )
        let maxCacheCap = ModelLoadParameters.maxCacheCap(
            for: model,
            scanner: scanner,
            parameterStore: parameterStore,
            activeKvOffloadEnabled: launchOptions.activeKvOffload
        ) ?? ModelLoadParameters.conservativeLongContextCap

        return SchedulerProfileGenerationRequest(
            model: model,
            modelPath: modelPath,
            selectionProfile: payload.selectionProfile ?? "agent-long-prompt",
            calibrationLevel: payload.calibrationLevel ?? "standard",
            mtpModelDir: mtpRuntime?.modelDir,
            mtpDraftTokens: mtpRuntime?.draftTokens,
            kvQuant: launchOptions.kvQuant ?? "none",
            pagedPrefixCacheDir: launchOptions.pagedPrefixCacheDir,
            prefixLruCacheMaxBytes: launchOptions.prefixLruCacheMaxBytes,
            ssdPrefixCacheMaxGB: launchOptions.ssdPrefixCacheMaxGB,
            activeKvOffload: launchOptions.activeKvOffload,
            memoryLimitTotalGB: launchOptions.memoryLimitTotalGB,
            memoryLimitModelGB: launchOptions.memoryLimitModelGB,
            maxCacheCap: maxCacheCap
        )
    }

    private func cancelSchedulerProfileGeneration() {
        sendSchedulerProfileStatus(profileGenerationService.cancel())
    }

    private func sendSchedulerProfileStatus(_ status: SchedulerProfileGenerationStatus? = nil) {
        let payload = status ?? profileGenerationService.currentStatus()
        let json = (try? Self.jsonString(payload)) ?? "{\"state\":\"failed\",\"success\":false}"
        sendJavaScript("onSchedulerProfileStatus(\(Self.jsStringLiteral(json)))")
    }

    private func sendSchedulerProfilePreview(_ preview: SchedulerProfileGenerationPreview) {
        let json = (try? Self.jsonString(preview)) ?? "{\"success\":false}"
        sendJavaScript("onSchedulerProfilePreview(\(Self.jsStringLiteral(json)))")
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
        notificationCenter.post(name: .ironMLXMenuLanguageDidChange, object: self)
    }

    @objc private func loadedModelsDidChange(_ notification: Notification) {
        if let object = notification.object as AnyObject?,
           object === self {
            return
        }
        sendScannedModels()
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
              let path = try? scanner.verifiedModelPath(for: model)
        else {
            return nil
        }
        return (model, path)
    }

    private func benchmarkModelPathValidator(
        config: AppConfig
    ) -> BenchmarkModelPathValidator? {
        guard config.verifyModelOnLoad == true else {
            return nil
        }
        let scanner = scanner
        return { modelID, _ in
            try await scanner.verifiedModelPathAsync(for: modelID, fullChecksum: true)
        }
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

    private func decodeVersionRepositoryPayload(_ json: String) -> ModelVersionRepositoryPayload? {
        guard let data = json.data(using: .utf8) else {
            return nil
        }
        return try? JSONDecoder().decode(ModelVersionRepositoryPayload.self, from: data)
    }

    private static func modelLoadInstruction(from body: Any) -> ModelLoadInstruction {
        if let dictionary = body as? [String: Any] {
            return ModelLoadInstruction(
                modelReference: stringValue(dictionary, "model")
                    ?? stringValue(dictionary, "model_id")
                    ?? "",
                useMtp: boolValue(dictionary, "use_mtp"),
                mtpModelID: stringValue(dictionary, "mtp_model_id"),
                usePromptLookup: boolValue(dictionary, "use_prompt_lookup"),
                crossRequestPromptLookup: boolValue(
                    dictionary,
                    "cross_request_prompt_lookup"
                )
            )
        }
        let text = body as? String ?? String(describing: body)
        if let data = text.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(ModelLoadInstructionPayload.self, from: data) {
            return ModelLoadInstruction(
                modelReference: decoded.model,
                useMtp: decoded.useMtp,
                mtpModelID: decoded.mtpModelID,
                usePromptLookup: decoded.usePromptLookup,
                crossRequestPromptLookup: decoded.crossRequestPromptLookup
            )
        }
        return ModelLoadInstruction(modelReference: text, useMtp: nil, mtpModelID: nil)
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

    private struct ModelLoadInstruction {
        var modelReference: String
        var useMtp: Bool?
        var mtpModelID: String?
        var usePromptLookup: Bool?
        var crossRequestPromptLookup: Bool?

        init(
            modelReference: String,
            useMtp: Bool?,
            mtpModelID: String?,
            usePromptLookup: Bool? = nil,
            crossRequestPromptLookup: Bool? = nil
        ) {
            self.modelReference = modelReference
            self.useMtp = useMtp
            self.mtpModelID = mtpModelID
            self.usePromptLookup = usePromptLookup
            self.crossRequestPromptLookup = crossRequestPromptLookup
        }
    }

    private struct ModelLoadInstructionPayload: Decodable {
        var model: String
        var useMtp: Bool?
        var mtpModelID: String?
        var usePromptLookup: Bool?
        var crossRequestPromptLookup: Bool?

        enum CodingKeys: String, CodingKey {
            case model
            case useMtp = "use_mtp"
            case mtpModelID = "mtp_model_id"
            case usePromptLookup = "use_prompt_lookup"
            case crossRequestPromptLookup = "cross_request_prompt_lookup"
        }
    }

    private struct BridgeErrorResponse: Encodable {
        var success: Bool
        var status: String
        var error: String
    }

    private struct PinModelBridgeResponse: Encodable {
        var success: Bool
        var status: String
        var model: String
        var pinned: Bool
        var loadedModels: [BackendLoadedModelInfo]
        var error: String?

        enum CodingKeys: String, CodingKey {
            case success
            case status
            case model
            case pinned
            case loadedModels = "loaded_models"
            case error
        }
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
        var token: String?
        var requestID: Int

        enum CodingKeys: String, CodingKey {
            case query
            case sort
            case token
            case requestID = "request_id"
        }
    }

    private struct ModelVersionRepositoryPayload: Decodable {
        var provider: String
        var repoID: String

        enum CodingKeys: String, CodingKey {
            case provider
            case repoID = "repo_id"
        }
    }

    private struct ModelVersionActivationPayload: Decodable {
        var provider: String
        var repoID: String
        var commitSHA: String

        enum CodingKeys: String, CodingKey {
            case provider
            case repoID = "repo_id"
            case commitSHA = "commit_sha"
        }
    }

    private struct ModelVersionDeletionPayload: Decodable {
        var provider: String
        var repoID: String
        var commitSHAs: [String]

        enum CodingKeys: String, CodingKey {
            case provider
            case repoID = "repo_id"
            case commitSHAs = "commit_shas"
        }
    }

    private struct ModelVersionBridgeOperationResult: Encodable {
        var success: Bool
        var provider: String?
        var repoID: String?
        var activeCommitSHA: String?
        var deletedCommitSHAs: [String]?
        var reclaimedBytes: Int64?
        var reloadStatus: String?
        var error: String?

        enum CodingKeys: String, CodingKey {
            case success
            case provider
            case repoID = "repo_id"
            case activeCommitSHA = "active_commit_sha"
            case deletedCommitSHAs = "deleted_commit_shas"
            case reclaimedBytes = "reclaimed_bytes"
            case reloadStatus = "reload_status"
            case error
        }
    }

    private struct ModelDownloadCancellationPayload: Decodable {
        var repoID: String
        var provider: String

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case provider
        }
    }

    private struct SchedulerProfileGenerationPayload: Decodable {
        var model: String
        var selectionProfile: String?
        var calibrationLevel: String?
        var requestToken: Int?

        enum CodingKeys: String, CodingKey {
            case model
            case selectionProfile = "selection_profile"
            case calibrationLevel = "calibration_level"
            case requestToken = "request_token"
        }
    }

    private enum SchedulerProfileGenerationRequestError: LocalizedError {
        case invalidRequest
        case noModelSelected
        case modelNotFound(String)

        var errorDescription: String? {
            switch self {
            case .invalidRequest:
                "Invalid scheduler profile generation request."
            case .noModelSelected:
                "No model is selected."
            case let .modelNotFound(model):
                "Model not found locally: \(model)"
            }
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

    nonisolated static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        let data = try encoder.encode(value)
        return String(data: data, encoding: .utf8) ?? "null"
    }

    nonisolated static func jsStringLiteral(_ value: String) -> String {
        let data = try? JSONEncoder().encode(value)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
    }
}
