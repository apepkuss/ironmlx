import Foundation

public struct BackendDecisionSettings: Codable, Equatable, Sendable {
    public enum Dtype: String, Codable, Sendable {
        case float16
        case float32
    }
    public enum Device: String, Codable, Sendable { case auto, gpu, cpu }
    public var device: Device
    public var compile: Bool
    public var padToMultiple: Int?
    public var dtype: Dtype
    public var batchSize: Int
    public var cachePrompts: Bool

    public init(dtype: Dtype = .float16, batchSize: Int = 16, cachePrompts: Bool = false,
                device: Device = .auto, compile: Bool = false, padToMultiple: Int? = nil) {
        self.device = device
        self.compile = compile
        self.padToMultiple = padToMultiple
        self.dtype = dtype
        self.batchSize = batchSize
        self.cachePrompts = cachePrompts
    }

    enum CodingKeys: String, CodingKey {
        case device, compile
        case padToMultiple = "pad_to_multiple"
        case dtype
        case batchSize = "batch_size"
        case cachePrompts = "cache_prompts"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        dtype = try container.decodeIfPresent(Dtype.self, forKey: .dtype) ?? .float16
        batchSize = try container.decodeIfPresent(Int.self, forKey: .batchSize) ?? 16
        cachePrompts = try container.decodeIfPresent(Bool.self, forKey: .cachePrompts) ?? false
        device = try container.decodeIfPresent(Device.self, forKey: .device) ?? .auto
        compile = try container.decodeIfPresent(Bool.self, forKey: .compile) ?? false
        padToMultiple = try container.decodeIfPresent(Int.self, forKey: .padToMultiple)
    }

    public func validate() throws {
        if let padToMultiple, !(1 ... 1024).contains(padToMultiple) {
            throw ConfigurationPersistenceError.invalidValue("decision.pad_to_multiple")
        }
        guard (1 ... 256).contains(batchSize) else {
            throw ConfigurationPersistenceError.invalidValue("decision.batch_size")
        }
    }
}
