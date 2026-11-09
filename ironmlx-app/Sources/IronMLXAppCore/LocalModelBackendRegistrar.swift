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

public enum BackendRecoveryFailureReason: String, Codable, Equatable, Sendable {
    case memoryInsufficient = "memory_insufficient"
    case modelLimitReached = "model_limit_reached"
    case modelFilesMissing = "model_files_missing"
    case modelSnapshotInvalid = "model_snapshot_invalid"
    case incompatibleConfiguration = "incompatible_configuration"
    case unknownModelLoadFailure = "unknown_model_load_failure"
    case crashLoopBreaker = "crash_loop_breaker"
}

public struct BackendModelRecoveryFailure: Codable, Equatable, Sendable {
    public var model: String
    public var stage: BackendRecoveryStage
    public var code: String
    public var message: String
    public var retryable: Bool
    public var action: BackendRecoveryAction
    public var reason: BackendRecoveryFailureReason

    public init(
        model: String,
        stage: BackendRecoveryStage,
        code: String,
        message: String,
        retryable: Bool,
        action: BackendRecoveryAction,
        reason: BackendRecoveryFailureReason? = nil
    ) {
        self.model = model
        self.stage = stage
        self.code = code
        self.message = message
        self.retryable = retryable
        self.action = action
        self.reason = reason ?? BackendRecoveryFailureClassifier.reason(for: code)
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case stage
        case code
        case message
        case retryable
        case action
        case reason
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        stage = try container.decode(BackendRecoveryStage.self, forKey: .stage)
        code = try container.decode(String.self, forKey: .code)
        message = try container.decode(String.self, forKey: .message)
        retryable = try container.decode(Bool.self, forKey: .retryable)
        action = try container.decode(BackendRecoveryAction.self, forKey: .action)
        reason = try container.decodeIfPresent(
            BackendRecoveryFailureReason.self,
            forKey: .reason
        ) ?? BackendRecoveryFailureClassifier.reason(for: code)
    }
}

enum BackendRecoveryFailureClassifier {
    static func reason(for code: String) -> BackendRecoveryFailureReason {
        switch code {
        case "memory_budget_exceeded", "gpu_memory_insufficient",
             "model_memory_limit_exceeded", "total_memory_limit_exceeded",
             "kv_memory_budget_exceeded", "insufficient_memory":
            .memoryInsufficient
        case "max_loaded_models_reached":
            .modelLimitReached
        case "model_path_not_found", "model_directory_not_found", "model_file_missing",
             "model_not_found", "mtp_model_not_found", "mtp_base_model_not_found":
            .modelFilesMissing
        case "model_snapshot_corrupt", "model_verification_failed", "snapshot_corrupt",
             "snapshot_unverified":
            .modelSnapshotInvalid
        case let value where value.hasPrefix("diffusion_gemma_"):
            .incompatibleConfiguration
        case let value where value.hasPrefix("mtp_"):
            .incompatibleConfiguration
        case let value where value.hasPrefix("unsupported_"):
            .incompatibleConfiguration
        case "incompatible", "model_name_mismatch":
            .incompatibleConfiguration
        default:
            .unknownModelLoadFailure
        }
    }

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
        // A load tracked through the backend registry reports its code there.
        if let failed = error as? ModelLoadFailedError {
            return (failed.message, failed.code)
        }
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

/// Why a registration pass runs; named in its log line.
public enum LocalModelRegistrationPass: String, Sendable {
    /// A backend start registers every model.
    case restore
    /// A Dashboard rescan registers the models that changed.
    case rescan
    /// A Dashboard load registers its target first.
    case loadTarget = "load-target"
    case other
}

public enum LocalModelBackendRegistrar {
    /// Registers `localModels` with the backend. With a `ledger` and the
    /// backend's `launchID`, each sent registration is recorded for that
    /// launch; `onlyChanged` then skips models the launch already has with the
    /// same registration. A failed registration is not recorded. Registrations
    /// of the same model run one at a time; different models do not wait for
    /// each other.
    @discardableResult
    public static func register(
        localModels: [LocalModel],
        defaultModel: String?,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        activeKvOffloadEnabled: Bool,
        preservePinnedState: Bool = false,
        client: any BackendModelLoading,
        ledger: LocalModelRegistrationLedger? = nil,
        launchID: UUID? = nil,
        onlyChanged: Bool = false,
        pass: LocalModelRegistrationPass = .other
    ) async -> [BackendModelRecoveryFailure] {
        let started = ContinuousClock.now
        let defaultModel = AppConfig.normalizedModelReference(defaultModel)
        let gate = launchID == nil ? nil : ledger
        // Only rescans take a registration computed by an earlier pass; a
        // backend start and a load compute theirs afresh (and keep it).
        let readsComputed = pass == .rescan
        if readsComputed, let ledger {
            // A rescan lists every local model: drop what is kept for the
            // models that are gone.
            ledger.keepComputedRegistrations(of: Set(localModels.map(\.id)))
        }
        var failed: [BackendModelRecoveryFailure] = []
        var sent = 0
        var skipped = 0
        var computedCount = 0
        var cachedCount = 0
        for model in localModels {
            // A cancelled load stops here instead of finishing the list.
            if Task.isCancelled {
                break
            }
            guard model.readiness?.isLoadable != false else {
                continue
            }
            await gate?.beginRegistration(of: model.id)
            defer { gate?.endRegistration(of: model.id) }
            if Task.isCancelled {
                break
            }
            // Read before the registration, so a change made while it is
            // computed shows up as changed inputs on the next pass.
            let inputs: LocalModelRegistrationInputs?
            if ledger != nil {
                inputs = LocalModelRegistrationInputs.read(
                    for: model,
                    defaultModel: defaultModel,
                    scanner: scanner,
                    parameterStore: parameterStore,
                    activeKvOffloadEnabled: activeKvOffloadEnabled
                )
            } else {
                inputs = nil
            }
            do {
                var registration: LocalModelRegistration
                if readsComputed, let ledger, let inputs,
                   let cached = ledger.computedRegistration(of: model.id, inputs: inputs) {
                    registration = cached
                    cachedCount += 1
                } else {
                    guard let modelDir = scanner.resolveModelPath(for: model.id) else {
                        failed.append(BackendRecoveryFailureClassifier.missingPath(model: model.id))
                        IronMLXAppLogger.error("Failed to register local model \(model.id): model path not found")
                        continue
                    }
                    try await Qwen3ASRModelSupport.upgradeDownloadOnlySnapshot(at: modelDir)
                    try await NemotronDiarizationModelSupport.upgradeDownloadOnlySnapshot(at: modelDir)
                    ledger?.recordComputation(of: model.id)
                    computedCount += 1
                    let computed = try self.registration(
                        for: model,
                        modelDir: modelDir,
                        defaultModel: defaultModel,
                        scanner: scanner,
                        parameterStore: parameterStore,
                        activeKvOffloadEnabled: activeKvOffloadEnabled
                    )
                    registration = computed.registration
                    // A draft lookup can fail for a moment (its repository
                    // is locked): such a result is not kept.
                    if let ledger, let inputs,
                       computed.mtpModelID == inputs.mtpModelID,
                       inputs == LocalModelRegistrationInputs.read(
                           for: model,
                           defaultModel: defaultModel,
                           scanner: scanner,
                           parameterStore: parameterStore,
                           activeKvOffloadEnabled: activeKvOffloadEnabled
                       ) {
                        ledger.storeComputedRegistration(registration, of: model.id, inputs: inputs)
                    }
                }
                if preservePinnedState {
                    registration.pinned = nil
                }
                var claimed: (ledger: LocalModelRegistrationLedger, launchID: UUID, value: LocalModelRegistration)?
                if let ledger, let launchID {
                    guard let value = ledger.claim(
                        registration,
                        model: model.id,
                        launchID: launchID,
                        force: !onlyChanged
                    ) else {
                        skipped += 1
                        continue
                    }
                    claimed = (ledger, launchID, value)
                }
                do {
                    try await send(registration, model: model.id, client: client)
                    sent += 1
                } catch {
                    if let claimed {
                        claimed.ledger.release(claimed.value, model: model.id, launchID: claimed.launchID)
                    }
                    throw error
                }
            } catch {
                if Task.isCancelled {
                    break
                }
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
        let elapsed = ContinuousClock.now - started
        let milliseconds = elapsed.components.seconds * 1000
            + elapsed.components.attoseconds / 1_000_000_000_000_000
        IronMLXAppLogger.info(
            "Model registration pass: kind=\(pass.rawValue) sent=\(sent) skipped=\(skipped) "
                + "failed=\(failed.count) computed=\(computedCount) cached=\(cachedCount) "
                + "duration_ms=\(milliseconds)"
        )
        return failed
    }

    /// A Dashboard rescan: registers the models whose registration changed
    /// for `launchID`. While a rescan pass runs, a new request does not start
    /// another; the latest request runs once after it.
    public static func rescan(
        localModels: [LocalModel],
        defaultModel: String?,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        activeKvOffloadEnabled: Bool,
        client: any BackendModelLoading,
        ledger: LocalModelRegistrationLedger,
        launchID: UUID?
    ) async {
        await ledger.runCoalescedRescan {
            await register(
                localModels: localModels,
                defaultModel: defaultModel,
                scanner: scanner,
                parameterStore: parameterStore,
                activeKvOffloadEnabled: activeKvOffloadEnabled,
                preservePinnedState: true,
                client: client,
                ledger: ledger,
                launchID: launchID,
                onlyChanged: true,
                pass: .rescan
            )
        }
    }

    /// Registers the model a Dashboard load is about to load, and only that
    /// one: the load request carries the model's own settings, and the other
    /// local models play no part in it.
    @discardableResult
    public static func registerLoadTarget(
        _ model: String,
        config: AppConfig,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        client: any BackendModelLoading,
        ledger: LocalModelRegistrationLedger? = nil,
        launchID: UUID? = nil
    ) async -> [BackendModelRecoveryFailure] {
        // Only the target's own repository is read. Without the pairing a
        // full scan adds, its MTP draft (when on and not saved) is looked up
        // by the resolver as before.
        let target = scanner.scannedModelWithoutPairing(
            for: model,
            pinnedModels: Set(config.pinnedModelReferences)
        )
        return await register(
            localModels: target.map { [$0] } ?? [],
            defaultModel: config.defaultModelReference,
            scanner: scanner,
            parameterStore: parameterStore,
            activeKvOffloadEnabled: config.activeKvOffload == true,
            preservePinnedState: true,
            client: client,
            ledger: ledger,
            launchID: launchID,
            pass: .loadTarget
        )
    }

    private static func registration(
        for model: LocalModel,
        modelDir: String,
        defaultModel: String?,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        activeKvOffloadEnabled: Bool
    ) throws -> (registration: LocalModelRegistration, mtpModelID: String?) {
        let capabilities = model.capabilities
        let mtpRuntime = capabilities?.supportsMtp != false
            ? try? ModelMtpRuntimeResolver.runtime(
                for: model.id,
                useMtp: nil,
                // The caller's scan already paired the model with its drafts.
                mtpCandidates: model.mtp.map(\.candidates),
                scanner: scanner,
                parameterStore: parameterStore
            )
            : nil
        let parameters = parameterStore.parameters(for: model.id)
        let registration = LocalModelRegistration(
            modelDir: modelDir,
            setDefault: model.id == defaultModel && !["embedding", "asr", "alignment", "diarization"].contains(capabilities?.runtimeKind ?? ""),
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
                .filtered(for: capabilities),
            // `audioResources` is nil for every type but TTS; the caller's
            // scan already read this snapshot's type.
            audio: model.type == "tts"
                ? try scanner.audioResources(for: model.id, execution: parameters?.audioExecution)
                : nil,
            decision: parameters?.decision,
            asr: capabilities?.runtimeKind == "asr" ? parameters?.asr ?? .qwen3Defaults : nil
        )
        return (registration, mtpRuntime?.modelID)
    }

    private static func send(
        _ registration: LocalModelRegistration,
        model: String,
        client: any BackendModelLoading
    ) async throws {
        _ = try await client.registerModel(
            model: model,
            modelDir: registration.modelDir,
            setDefault: registration.setDefault,
            maxCacheCap: registration.maxCacheCap,
            pinned: registration.pinned,
            mtpModelDir: registration.mtpModelDir,
            mtpDraftTokens: registration.mtpDraftTokens,
            promptLookup: registration.promptLookup,
            samplingDefaults: registration.samplingDefaults,
            audio: registration.audio,
            decision: registration.decision
        )
        if let asr = registration.asr {
            try await client.updateASRSettings(model: model, settings: asr)
        }
    }
}
