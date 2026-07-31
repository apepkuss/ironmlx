import Foundation

public enum BackendRecoveryStage: String, Codable, Equatable, Sendable {
    case registration
    case verification
    case loading
    case readiness
}

public enum BackendRecoveryAction: String, Codable, Equatable, Sendable {
    case retry
    case verifyModel = "verify_model"
    case redownloadModel = "redownload_model"
    case unloadOtherModels = "unload_other_models"
    case reviewConfiguration = "review_configuration"
    case reviewRuntimeSettings = "review_runtime_settings"
    case viewLogs = "view_logs"
}

public struct BackendModelRecoveryFailure: Codable, Equatable, Sendable {
    public var model: String
    public var stage: BackendRecoveryStage
    public var code: String
    public var message: String
    public var retryable: Bool
    public var action: BackendRecoveryAction

    public init(
        model: String,
        stage: BackendRecoveryStage,
        code: String,
        message: String,
        retryable: Bool,
        action: BackendRecoveryAction
    ) {
        self.model = model
        self.stage = stage
        self.code = code
        self.message = message
        self.retryable = retryable
        self.action = action
    }
}

enum BackendRecoveryFailureClassifier {
    static func failure(
        model: String,
        stage: BackendRecoveryStage,
        error: Error
    ) -> BackendModelRecoveryFailure {
        if let verification = error as? ModelSnapshotVerificationError {
            return verificationFailure(model: model, error: verification)
        }
        let details = backendDetails(from: error)
        let code = details.code ?? "unknown_backend_error"
        let policy = policy(for: code)
        return BackendModelRecoveryFailure(
            model: model,
            stage: stage,
            code: code,
            message: details.message,
            retryable: policy.retryable,
            action: policy.action
        )
    }

    static func missingPath(model: String) -> BackendModelRecoveryFailure {
        BackendModelRecoveryFailure(
            model: model,
            stage: .registration,
            code: "model_path_not_found",
            message: "The local model path could not be resolved.",
            retryable: false,
            action: .redownloadModel
        )
    }

    private static func verificationFailure(
        model: String,
        error: ModelSnapshotVerificationError
    ) -> BackendModelRecoveryFailure {
        let code: String
        let action: BackendRecoveryAction
        switch error {
        case .fileMissing:
            code = "model_file_missing"
            action = .redownloadModel
        case .checksumMismatch, .knownCorrupt, .sizeMismatch:
            code = "model_snapshot_corrupt"
            action = .redownloadModel
        case .manifestMissing, .manifestInvalid, .identityMismatch, .unexpectedFile,
             .fileChangedDuringVerification:
            code = "model_verification_failed"
            action = .verifyModel
        }
        return BackendModelRecoveryFailure(
            model: model,
            stage: .verification,
            code: code,
            message: error.localizedDescription,
            retryable: false,
            action: action
        )
    }

    private static func policy(
        for code: String
    ) -> (retryable: Bool, action: BackendRecoveryAction) {
        switch code {
        case "max_loaded_models_reached":
            (false, .unloadOtherModels)
        case "gpu_memory_insufficient":
            (true, .unloadOtherModels)
        case "model_memory_limit_exceeded", "total_memory_limit_exceeded",
             "kv_memory_budget_exceeded":
            (false, .reviewRuntimeSettings)
        case "diffusion_gemma_mtp_unsupported",
             "diffusion_gemma_prompt_lookup_unsupported",
             "diffusion_gemma_kv_cache_unsupported",
             "diffusion_gemma_sampling_parameter_unsupported":
            (false, .reviewConfiguration)
        case "model_path_not_found", "model_directory_not_found", "model_file_missing",
             "model_snapshot_corrupt":
            (false, .redownloadModel)
        case "memory_budget_exceeded", "backend_unavailable", "model_reload_busy":
            (true, .retry)
        default:
            (true, .viewLogs)
        }
    }

    private static func backendDetails(from error: Error) -> (message: String, code: String?) {
        if case BackendAPIError.serverResponse(statusCode: _, body: let body) = error,
           let body,
           let data = body.data(using: .utf8),
           let payload = try? JSONDecoder().decode(BackendErrorPayload.self, from: data) {
            let message = payload.error?.trimmingCharacters(in: .whitespacesAndNewlines)
            return (
                message.flatMap { $0.isEmpty ? nil : $0 } ?? error.localizedDescription,
                payload.code
            )
        }
        return (error.localizedDescription, nil)
    }

    private struct BackendErrorPayload: Decodable {
        var error: String?
        var code: String?
    }
}

public enum LocalModelBackendRegistrar {
    @discardableResult
    public static func register(
        localModels: [LocalModel],
        defaultModel: String?,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        activeKvOffloadEnabled: Bool,
        client: any BackendModelLoading
    ) async -> [BackendModelRecoveryFailure] {
        let defaultModel = AppConfig.normalizedModelReference(defaultModel)
        var failed: [BackendModelRecoveryFailure] = []
        for model in localModels {
            guard model.readiness?.isLoadable != false else {
                continue
            }
            guard let modelDir = scanner.resolveModelPath(for: model.id) else {
                failed.append(BackendRecoveryFailureClassifier.missingPath(model: model.id))
                IronMLXAppLogger.error("Failed to register local model \(model.id): model path not found")
                continue
            }
            do {
                let capabilities = model.capabilities
                let mtpRuntime = capabilities?.supportsMtp != false
                    ? try? ModelMtpRuntimeResolver.runtime(
                        for: model.id,
                        useMtp: nil,
                        scanner: scanner,
                        parameterStore: parameterStore
                    )
                    : nil
                let parameters = parameterStore.parameters(for: model.id)
                _ = try await client.registerModel(
                    model: model.id,
                    modelDir: modelDir,
                    setDefault: model.id == defaultModel,
                    maxCacheCap: capabilities?.supportsKvCache != false
                        ? ModelLoadParameters.maxCacheCap(
                            for: model.id,
                            scanner: scanner,
                            parameterStore: parameterStore,
                            activeKvOffloadEnabled: activeKvOffloadEnabled
                        )
                        : nil,
                    pinned: model.pinned,
                    mtpModelDir: mtpRuntime?.modelDir,
                    mtpDraftTokens: mtpRuntime?.draftTokens,
                    promptLookup: capabilities?.supportsPromptLookup != false
                        ? parameters?.promptLookupConfig
                        : nil,
                    samplingDefaults: (parameters?.samplingDefaults ?? .empty)
                        .filtered(for: capabilities)
                )
            } catch {
                failed.append(
                    BackendRecoveryFailureClassifier.failure(
                        model: model.id,
                        stage: .registration,
                        error: error
                    )
                )
                IronMLXAppLogger.error("Failed to register local model \(model.id): \(error)")
            }
        }
        return failed
    }
}
