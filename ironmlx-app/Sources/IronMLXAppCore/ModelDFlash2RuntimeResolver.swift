import Foundation

public struct ModelDFlash2Runtime: Equatable, Sendable {
    public static let qualifiedAutomaticBlockSizeCap = 8

    public var targetModelID: String
    public var targetModelDir: String
    public var draftModelID: String
    public var draftModelDir: String
    public var checkpointBlockSize: Int
    public var blockSize: Int
    public var draftBits: Int
    public var tensorBatchMaxWidth: Int?
    public var maxCacheCap: Int?

    public init(
        targetModelID: String,
        targetModelDir: String,
        draftModelID: String,
        draftModelDir: String,
        checkpointBlockSize: Int,
        blockSize: Int,
        draftBits: Int,
        tensorBatchMaxWidth: Int? = nil,
        maxCacheCap: Int? = nil
    ) {
        self.targetModelID = targetModelID
        self.targetModelDir = targetModelDir
        self.draftModelID = draftModelID
        self.draftModelDir = draftModelDir
        self.checkpointBlockSize = checkpointBlockSize
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
    case blockSizeExceedsCheckpoint(model: String, requested: Int, checkpoint: Int)
    case incompatibleAccelerationConfiguration(model: String)
    case draftPrecisionNotQualified(model: String, bits: Int)

    public var errorDescription: String? {
        switch self {
        case .targetPathNotFound(let model):
            return "DFlash2 target model is not available locally: \(model)."
        case .noCompatibleDraft(let model):
            return "No compatible DFlash2 draft is available for \(model)."
        case .draftPathNotFound(let model):
            return "DFlash2 draft is not available locally: \(model)."
        case .blockSizeExceedsCheckpoint(let model, let requested, let checkpoint):
            return "DFlash2 block size \(requested) exceeds \(model)'s checkpoint block size \(checkpoint)."
        case .incompatibleAccelerationConfiguration(let model):
            return "DFlash2 cannot be combined with MTP or repeated-text acceleration for \(model)."
        case .draftPrecisionNotQualified(let model, let bits):
            let precision = bits == 0 ? "BF16" : "\(bits)-bit"
            return "DFlash2 draft precision \(precision) is not qualified for \(model)."
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
        let info = scanner.dflash2Info(for: modelID)
        let candidates = info?.candidates ?? []
        guard let candidate = candidates.first(where: { $0.id == selected }),
              let checkpointBlockSize = candidate.blockSize
        else {
            throw ModelDFlash2RuntimeError.noCompatibleDraft(model: modelID)
        }
        guard let draftModelDir = try? scanner.verifiedDFlash2DraftPath(
            for: selected,
            fullChecksum: fullChecksum
        ) else {
            throw ModelDFlash2RuntimeError.draftPathNotFound(model: selected)
        }
        let requestedBlockSize = parameters?.dflash2BlockSizeValue
        if let requestedBlockSize, requestedBlockSize > checkpointBlockSize {
            throw ModelDFlash2RuntimeError.blockSizeExceedsCheckpoint(
                model: selected,
                requested: requestedBlockSize,
                checkpoint: checkpointBlockSize
            )
        }
        let resolvedBlockSize = requestedBlockSize
            ?? min(checkpointBlockSize, ModelDFlash2Runtime.qualifiedAutomaticBlockSizeCap)
        // Draft precision is a target qualification. Without an explicit
        // choice the target's first option applies (4-bit for both families);
        // Qwen3.6 MoE targets accept 4-bit and BF16 only.
        let draftBitsOptions = info?.draftBitsOptions ?? LocalModelDFlash2Info.denseDraftBitsOptions
        let draftBits: Int
        if let explicitBits = parameters?.dflash2DraftBitsExplicitValue {
            guard draftBitsOptions.contains(explicitBits) else {
                throw ModelDFlash2RuntimeError.draftPrecisionNotQualified(model: modelID, bits: explicitBits)
            }
            draftBits = explicitBits
        } else {
            draftBits = draftBitsOptions.first ?? 4
        }
        return ModelDFlash2Runtime(
            targetModelID: modelID,
            targetModelDir: targetModelDir,
            draftModelID: selected,
            draftModelDir: draftModelDir,
            checkpointBlockSize: checkpointBlockSize,
            blockSize: resolvedBlockSize,
            draftBits: draftBits,
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
