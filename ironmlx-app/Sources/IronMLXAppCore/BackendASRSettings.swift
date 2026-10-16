import Foundation

/// ASR admission and per-window decoding settings; independent of chat sampling.
public struct BackendASRSettings: Codable, Equatable, Sendable {
    // Both supported Qwen3-ASR profiles share the official chunk/evaluation policy.
    public static let qwen3Defaults = BackendASRSettings(maxAudioDurationSeconds: 1200, maxOutputTokens: 1024)
    public var maxAudioDurationSeconds: Int
    public var maxOutputTokens: Int

    public init(maxAudioDurationSeconds: Int, maxOutputTokens: Int) {
        self.maxAudioDurationSeconds = maxAudioDurationSeconds
        self.maxOutputTokens = maxOutputTokens
    }

    enum CodingKeys: String, CodingKey {
        case maxAudioDurationSeconds = "max_audio_duration_seconds"
        case maxOutputTokens = "max_output_tokens"
    }

    public func validate() throws {
        guard (1...7200).contains(maxAudioDurationSeconds) else {
            throw ConfigurationPersistenceError.invalidValue("asr.max_audio_duration_seconds")
        }
        guard (1...8192).contains(maxOutputTokens) else {
            throw ConfigurationPersistenceError.invalidValue("asr.max_output_tokens")
        }
    }
}
