import Foundation

public struct ModelDFlash2Runtime: Equatable, Sendable {
    public var targetModelID: String
    public var targetModelDir: String
    public var draftModelID: String
    public var draftModelDir: String
    public var blockSize: Int
    public var draftBits: Int
    public var tensorBatchMaxWidth: Int?
    public var maxCacheCap: Int?

    public init(
        targetModelID: String,
        targetModelDir: String,
        draftModelID: String,
        draftModelDir: String,
        blockSize: Int,
        draftBits: Int,
        tensorBatchMaxWidth: Int? = nil,
        maxCacheCap: Int? = nil
    ) {
        self.targetModelID = targetModelID
        self.targetModelDir = targetModelDir
        self.draftModelID = draftModelID
        self.draftModelDir = draftModelDir
        self.blockSize = blockSize
        self.draftBits = draftBits
        self.tensorBatchMaxWidth = tensorBatchMaxWidth
        self.maxCacheCap = maxCacheCap
    }
}

public enum ModelDFlash2RuntimeError: LocalizedError, Equatable {
    case targetPathNotFound(model: String)
    case noCompatibleDraft(model: String)
    case draftPathNotFound(model: String)
    case incompatibleAccelerationConfiguration(model: String)

    public var errorDescription: String? {
        switch self {
        case .targetPathNotFound(let model):
            return "DFlash2 target model is not available locally: \(model)."
        case .noCompatibleDraft(let model):
            return "No compatible DFlash2 draft is available for \(model)."
        case .draftPathNotFound(let model):
            return "DFlash2 draft is not available locally: \(model)."
        case .incompatibleAccelerationConfiguration(let model):
            return "DFlash2 cannot be combined with MTP or repeated-text acceleration for \(model)."
        }
    }
}

public enum ModelDFlash2RuntimeResolver {
    public static func runtimeAsync(
        for modelID: String,
        useDFlash2: Bool?,
        explicitDraftModelID: String? = nil,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool = false
    ) async throws -> ModelDFlash2Runtime? {
        try await Task.detached(priority: .userInitiated) {
            try runtime(
                for: modelID,
                useDFlash2: useDFlash2,
                explicitDraftModelID: explicitDraftModelID,
                scanner: scanner,
                parameterStore: parameterStore,
                fullChecksum: fullChecksum
            )
        }.value
    }

    public static func runtime(
        for modelID: String,
        useDFlash2: Bool?,
        explicitDraftModelID: String? = nil,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        fullChecksum: Bool = false
    ) throws -> ModelDFlash2Runtime? {
        let parameters = parameterStore.parameters(for: modelID)
        let shouldUseDFlash2 = useDFlash2 ?? (parameters?.dflash2Enabled == true)
        guard shouldUseDFlash2 else {
            return nil
        }
        if parameters?.mtpEnabled == true || parameters?.promptLookupEnabled == true {
            throw ModelDFlash2RuntimeError.incompatibleAccelerationConfiguration(model: modelID)
        }
        guard let targetModelDir = try? scanner.verifiedModelPath(
            for: modelID,
            fullChecksum: fullChecksum
        ) else {
            throw ModelDFlash2RuntimeError.targetPathNotFound(model: modelID)
        }
        let selected = normalized(explicitDraftModelID)
            ?? normalized(parameters?.dflash2ModelID)
            ?? scanner.dflash2Candidates(for: modelID).first?.id
        guard let selected else {
            throw ModelDFlash2RuntimeError.noCompatibleDraft(model: modelID)
        }
        guard scanner.dflash2Candidates(for: modelID).contains(where: { $0.id == selected }) else {
            throw ModelDFlash2RuntimeError.noCompatibleDraft(model: modelID)
        }
        guard let draftModelDir = try? scanner.verifiedDFlash2DraftPath(
            for: selected,
            fullChecksum: fullChecksum
        ) else {
            throw ModelDFlash2RuntimeError.draftPathNotFound(model: selected)
        }
        return ModelDFlash2Runtime(
            targetModelID: modelID,
            targetModelDir: targetModelDir,
            draftModelID: selected,
            draftModelDir: draftModelDir,
            blockSize: parameters?.dflash2BlockSizeValue ?? 4,
            draftBits: parameters?.dflash2DraftBitsValue ?? 4,
            tensorBatchMaxWidth: parameters?.dflash2TensorBatchMaxWidthValue,
            maxCacheCap: ModelLoadParameters.maxCacheCap(
                for: modelID,
                scanner: scanner,
                parameterStore: parameterStore,
                activeKvOffloadEnabled: false
            )
        )
    }

    private static func normalized(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }
}
