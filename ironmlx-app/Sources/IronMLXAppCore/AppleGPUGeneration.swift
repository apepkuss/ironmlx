import Foundation
import Metal

/// Apple GPU generation of this Mac, read from the Metal architecture name
/// (for example `applegpu_g17s` → 17). The backend enables the M5 DFlash2
/// profile, and with it Neural Engine acceleration, only on generation 17+.
public enum AppleGPUGeneration {
    /// First generation that runs the M5 DFlash2 profile (Apple M5).
    public static let m5ProfileMinimumGeneration = 17

    /// Generation number from an architecture name, or nil when the name is not an Apple GPU.
    public static func parse(architectureName: String) -> Int? {
        let name = architectureName.lowercased()
        guard let range = name.range(of: "applegpu_g") else { return nil }
        let digits = name[range.upperBound...].prefix { $0.isNumber }
        return digits.isEmpty ? nil : Int(digits)
    }

    /// Generation of the system default GPU, or nil when it cannot be determined.
    public static func current() -> Int? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return parse(architectureName: device.architecture.name)
    }
}
