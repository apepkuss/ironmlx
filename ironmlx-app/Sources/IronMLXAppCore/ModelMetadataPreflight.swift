import Foundation

enum ModelArtifactRole {
    static let dflash2Drafter = "dflash2_drafter"
}

public protocol ModelMetadataPreflighting: Sendable {
    func validate(metadataDirectory: URL) async throws -> ModelMetadataPreflightResult
}

public struct ModelMetadataPreflightResult: Decodable, Equatable, Sendable {
    public var modelType: String
    public var artifactRole: String
    public var quantization: ModelQuantizationPreflight?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case artifactRole = "artifact_role"
        case quantization
    }
}

public struct ModelQuantizationPreflight: Decodable, Equatable, Sendable {
    public var mode: String
    public var bits: Int
    public var groupSize: Int
    public var overrideCount: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case bits
        case groupSize = "group_size"
        case overrideCount = "override_count"
    }
}

public struct IronMLXModelMetadataPreflight: ModelMetadataPreflighting {
    private let executableURL: URL

    public init(executableURL: URL = BackendBinaryResolver.resolve()) {
        self.executableURL = executableURL
    }

    public func validate(metadataDirectory: URL) async throws -> ModelMetadataPreflightResult {
        let executableURL = self.executableURL
        return try await Task.detached(priority: .utility) {
            try BackendBinaryResolver.validateBundledRuntimeIfNeeded(for: executableURL)
            let process = Process()
            let standardOutput = Pipe()
            let standardError = Pipe()
            process.executableURL = executableURL
            process.arguments = BackendBinaryResolver.helperArguments(
                [
                    "model-preflight",
                    "--metadata-dir", metadataDirectory.path,
                ],
                executableURL: executableURL
            )
            process.environment = BundledChildProcessEnvironment.sanitized()
            process.standardOutput = standardOutput
            process.standardError = standardError
            try process.run()
            process.waitUntilExit()
            let output = standardOutput.fileHandleForReading.readDataToEndOfFile()
            let errorOutput = standardError.fileHandleForReading.readDataToEndOfFile()
            guard process.terminationStatus == 0 else {
                let detail = String(data: errorOutput, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                throw ModelMetadataPreflightError.rejected(
                    detail?.isEmpty == false ? detail! : "ironmlx model-preflight rejected the metadata."
                )
            }
            do {
                return try JSONDecoder().decode(ModelMetadataPreflightResult.self, from: output)
            } catch {
                throw ModelMetadataPreflightError.invalidResponse(error.localizedDescription)
            }
        }.value
    }
}

public enum ModelMetadataPreflightError: LocalizedError {
    case rejected(String)
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case let .rejected(detail):
            detail
        case let .invalidResponse(detail):
            "ironmlx model-preflight returned invalid JSON: \(detail)"
        }
    }
}

public struct ModelResourcePreflight: Sendable {
    public static let diskSafetyBytes: Int64 = 2 * 1_024 * 1_024 * 1_024
    public static let memorySafetyBytes: Int64 = 2 * 1_024 * 1_024 * 1_024

    public var weightBytes: Int64
    public var remainingDownloadBytes: Int64
    public var availableDiskBytes: Int64?
    public var physicalMemoryBytes: Int64

    public var estimatedPeakMemoryBytes: Int64 {
        let overhead = max(512 * 1_024 * 1_024, weightBytes / 10)
        let result = weightBytes.addingReportingOverflow(overhead)
        return result.overflow ? Int64.max : result.partialValue
    }

    public var requiredDiskBytes: Int64 {
        let result = remainingDownloadBytes.addingReportingOverflow(Self.diskSafetyBytes)
        return result.overflow ? Int64.max : result.partialValue
    }

    public func validate() throws {
        if let availableDiskBytes,
           requiredDiskBytes > availableDiskBytes {
            throw ModelResourcePreflightError.insufficientDisk(
                required: requiredDiskBytes,
                available: availableDiskBytes
            )
        }
        let usableMemory = max(0, physicalMemoryBytes - Self.memorySafetyBytes)
        if estimatedPeakMemoryBytes > usableMemory {
            throw ModelResourcePreflightError.insufficientMemory(
                estimated: estimatedPeakMemoryBytes,
                available: usableMemory
            )
        }
    }
}

public enum ModelResourcePreflightError: LocalizedError {
    case insufficientDisk(required: Int64, available: Int64)
    case insufficientMemory(estimated: Int64, available: Int64)

    public var errorDescription: String? {
        switch self {
        case let .insufficientDisk(required, available):
            "Insufficient disk space: \(required) bytes required, \(available) bytes available."
        case let .insufficientMemory(estimated, available):
            "Model is expected to require \(estimated) bytes, exceeding the \(available)-byte memory budget."
        }
    }
}
