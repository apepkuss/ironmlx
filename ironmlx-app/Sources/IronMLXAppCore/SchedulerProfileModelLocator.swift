import Foundation

/// Why a scheduler profile cannot use a model.
enum SchedulerProfileModelError: LocalizedError, Equatable {
    case notFound(String)
    /// The model exists but its snapshot cannot be loaded.
    case unavailable(model: String, reason: String)
    /// Another operation holds the model's repository lock.
    case busy(String)

    static let busyCode = "model_busy"

    /// The code the Dashboard shows a localized message for.
    var code: String? {
        if case .busy = self {
            return Self.busyCode
        }
        return nil
    }

    var errorDescription: String? {
        switch self {
        case let .notFound(model):
            "Model not found locally: \(model)"
        case let .unavailable(model, reason):
            "Model \(model) cannot be used: \(reason)"
        case let .busy(model):
            "Model \(model) is in use by another operation. Try again shortly."
        }
    }
}

/// The model files a scheduler profile request runs on.
struct SchedulerProfileModelTarget: Equatable, Sendable {
    var modelPath: String
    var mtpRuntime: ModelMtpRuntime?
}

/// Finds the model a scheduler profile preview or generation works on.
enum SchedulerProfileModelLocator {
    /// What the preview estimates with, read without any repository lock.
    static func previewTarget(
        for model: String,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore
    ) throws -> SchedulerProfileModelTarget {
        // The MTP draft decides the b_max matrix, so the estimate needs it too.
        SchedulerProfileModelTarget(
            modelPath: try previewPath(for: model, scanner: scanner),
            mtpRuntime: ModelMtpRuntimeResolver.unverifiedRuntime(
                for: model,
                scanner: scanner,
                parameterStore: parameterStore
            )
        )
    }

    /// What generation runs on: the verified model and MTP draft.
    static func generationTarget(
        for model: String,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool,
        wait: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) async throws -> SchedulerProfileModelTarget {
        let modelPath = try await generationPath(
            for: model,
            scanner: scanner,
            fullChecksum: fullChecksum,
            wait: wait
        )
        let mtpRuntime = try await ModelMtpRuntimeResolver.runtimeAsync(
            for: model,
            useMtp: nil,
            scanner: scanner,
            parameterStore: parameterStore,
            fullChecksum: fullChecksum
        )
        return SchedulerProfileModelTarget(modelPath: modelPath, mtpRuntime: mtpRuntime)
    }

    static let busyAttempts = 10
    static let busyRetryDelay = Duration.milliseconds(200)

    /// The preview only estimates the work, so it reads the snapshot without
    /// taking the repository lock that loads and downloads hold.
    static func previewPath(for model: String, scanner: LocalModelScanner) throws -> String {
        guard let path = scanner.resolveModelPath(for: model) else {
            throw unavailable(model, scanner: scanner)
        }
        return path
    }

    /// Generation verifies the snapshot under the repository lock. A lock held
    /// by another operation is waited for briefly before the model is
    /// reported busy; other errors are reported as they are.
    static func generationPath(
        for model: String,
        scanner: LocalModelScanner,
        fullChecksum: Bool,
        wait: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) async throws -> String {
        guard scanner.resolveModelPath(for: model) != nil else {
            throw unavailable(model, scanner: scanner)
        }
        for attempt in 1...busyAttempts {
            do {
                return try await scanner.verifiedModelPathAsync(for: model, fullChecksum: fullChecksum)
            } catch ModelRepositoryLockError.busy {
                guard attempt < busyAttempts else {
                    break
                }
                try await wait(busyRetryDelay)
            }
        }
        throw SchedulerProfileModelError.busy(model)
    }

    private static func unavailable(_ model: String, scanner: LocalModelScanner) -> SchedulerProfileModelError {
        guard let readiness = scanner.readiness(for: model), !readiness.isLoadable else {
            return .notFound(model)
        }
        return .unavailable(
            model: model,
            reason: readiness.message ?? readiness.reasonCode ?? readiness.status
        )
    }
}
