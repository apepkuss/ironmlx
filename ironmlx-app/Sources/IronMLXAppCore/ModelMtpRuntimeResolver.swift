import Foundation

public struct ModelMtpRuntime: Equatable, Sendable {
    public var modelID: String
    public var modelDir: String
    public var draftTokens: Int?

    public init(modelID: String, modelDir: String, draftTokens: Int? = nil) {
        self.modelID = modelID
        self.modelDir = modelDir
        self.draftTokens = draftTokens
    }
}

public enum ModelMtpRuntimeError: LocalizedError, Equatable {
    case noCompatibleMtp(model: String)
    case mtpPathNotFound(model: String)

    public var errorDescription: String? {
        switch self {
        case .noCompatibleMtp(let model):
            return "No compatible MTP weights are available for \(model)."
        case .mtpPathNotFound(let model):
            return "MTP weights are not available locally: \(model)."
        }
    }
}

public enum ModelMtpRuntimeResolver {
    public static func runtimeAsync(
        for modelID: String,
        useMtp: Bool?,
        explicitMtpModelID: String? = nil,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool = false
    ) async throws -> ModelMtpRuntime? {
        try await Task.detached(priority: .userInitiated) {
            try runtime(
                for: modelID,
                useMtp: useMtp,
                explicitMtpModelID: explicitMtpModelID,
                scanner: scanner,
                parameterStore: parameterStore,
                fullChecksum: fullChecksum
            )
        }.value
    }

    /// `mtpCandidates` are the model's compatible drafts from a scan the
    /// caller already made; `nil` scans for them when they are needed.
    public static func runtime(
        for modelID: String,
        useMtp: Bool?,
        explicitMtpModelID: String? = nil,
        mtpCandidates: [LocalMtpCandidate]? = nil,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool = false
    ) throws -> ModelMtpRuntime? {
        guard let selected = try selectedMtpModelID(
            for: modelID,
            useMtp: useMtp,
            explicitMtpModelID: explicitMtpModelID,
            mtpCandidates: mtpCandidates,
            scanner: scanner,
            parameterStore: parameterStore
        ) else {
            return nil
        }
        guard let modelDir = try? scanner.verifiedModelPath(
            for: selected,
            fullChecksum: fullChecksum
        ) else {
            throw ModelMtpRuntimeError.mtpPathNotFound(model: selected)
        }
        return ModelMtpRuntime(
            modelID: selected,
            modelDir: modelDir,
            draftTokens: parameterStore.parameters(for: modelID)?.mtpDraftTokensValue
        )
    }

    /// The MTP runtime the saved parameters select, read without verifying the
    /// draft snapshot under its repository lock. `nil` when MTP is off or the
    /// draft model does not resolve.
    static func unverifiedRuntime(
        for modelID: String,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore
    ) -> ModelMtpRuntime? {
        guard let selected = try? selectedMtpModelID(
            for: modelID,
            useMtp: nil,
            explicitMtpModelID: nil,
            scanner: scanner,
            parameterStore: parameterStore
        ),
            let modelDir = scanner.resolveModelPath(for: selected)
        else {
            return nil
        }
        return ModelMtpRuntime(
            modelID: selected,
            modelDir: modelDir,
            draftTokens: parameterStore.parameters(for: modelID)?.mtpDraftTokensValue
        )
    }

    /// The draft model a load uses: the explicit choice, the saved one, or the
    /// first compatible candidate. `nil` when MTP is off.
    private static func selectedMtpModelID(
        for modelID: String,
        useMtp: Bool?,
        explicitMtpModelID: String?,
        mtpCandidates: [LocalMtpCandidate]? = nil,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore
    ) throws -> String? {
        let parameters = parameterStore.parameters(for: modelID)
        guard useMtp ?? (parameters?.mtpEnabled == true) else {
            return nil
        }
        guard let selected = normalized(explicitMtpModelID)
            ?? normalized(parameters?.mtpModelID)
            ?? (mtpCandidates ?? scanner.mtpCandidates(for: modelID)).first?.id
        else {
            throw ModelMtpRuntimeError.noCompatibleMtp(model: modelID)
        }
        return selected
    }

    private static func normalized(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }
}
