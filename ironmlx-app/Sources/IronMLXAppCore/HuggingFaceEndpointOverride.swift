import Foundation

/// Debug builds only: `IRONMLX_HF_ENDPOINT` replaces the huggingface.co base URL,
/// so a tester can simulate the official host being unreachable. Release builds
/// ignore it. hf-mirror.com is never overridden.
public enum HuggingFaceEndpointOverride {
    static let environmentKey = "IRONMLX_HF_ENDPOINT"
    static let defaultEndpoint = URL(string: "https://huggingface.co")!

    /// Accepts an http or https URL with a host; anything else is ignored.
    static func parse(_ value: String?) -> URL? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              let url = URL(string: value),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty
        else { return nil }
        return url
    }

    /// Read once at startup.
    public static let official: URL = {
        #if DEBUG
        return parse(ProcessInfo.processInfo.environment[environmentKey]) ?? defaultEndpoint
        #else
        return defaultEndpoint
        #endif
    }()
}
