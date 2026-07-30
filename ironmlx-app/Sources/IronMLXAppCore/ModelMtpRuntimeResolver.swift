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

    public static func runtime(
        for modelID: String,
        useMtp: Bool?,
        explicitMtpModelID: String? = nil,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool = false
    ) throws -> ModelMtpRuntime? {
        let parameters = parameterStore.parameters(for: modelID)
        let shouldUseMtp = useMtp ?? (parameters?.mtpEnabled == true)
        guard shouldUseMtp else {
            return nil
        }

        let selected = normalized(explicitMtpModelID)
            ?? normalized(parameters?.mtpModelID)
            ?? scanner.mtpCandidates(for: modelID).first?.id
        guard let selected else {
            throw ModelMtpRuntimeError.noCompatibleMtp(model: modelID)
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
            draftTokens: parameters?.mtpDraftTokensValue
        )
    }

    private static func normalized(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }
}
