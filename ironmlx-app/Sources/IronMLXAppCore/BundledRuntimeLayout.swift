import Foundation

public enum BundledRuntimeLayoutError: Error, Equatable, LocalizedError {
    case missingFile(String)
    case nonRegularFile(String)
    case nonExecutableFile(String)
    case symbolicLinkNotAllowed(String)
    case pathEscapesBundle(String)

    public var errorDescription: String? {
        switch self {
        case let .missingFile(path):
            return "Required bundled file is missing: \(path)"
        case let .nonRegularFile(path):
            return "Required bundled path is not a regular file: \(path)"
        case let .nonExecutableFile(path):
            return "Required bundled helper is not executable: \(path)"
        case let .symbolicLinkNotAllowed(path):
            return "Bundled runtime files must not be symbolic links: \(path)"
        case let .pathEscapesBundle(path):
            return "Resolved runtime path escapes the App Bundle: \(path)"
        }
    }
}

public struct BundledRuntimeLayout: Equatable, Sendable {
    public let bundleURL: URL
    public let backendURL: URL
    public let ironBenchURL: URL
    public let metallibURL: URL
    public let dashboardURL: URL
    public let appIconURL: URL
    public let menuBarIconURL: URL
    public let menuBarIcon2xURL: URL
    public let logoURL: URL
    public let sidebarLogo2xURL: URL

    public static func expected(bundleURL: URL = Bundle.main.bundleURL) -> Self {
        let contentsURL = bundleURL.appendingPathComponent("Contents", isDirectory: true)
        let helpersURL = contentsURL.appendingPathComponent("Helpers", isDirectory: true)
        let resourcesURL = contentsURL.appendingPathComponent("Resources", isDirectory: true)
        return Self(
            bundleURL: bundleURL,
            backendURL: helpersURL.appendingPathComponent("ironmlx", isDirectory: false),
            ironBenchURL: helpersURL.appendingPathComponent("iron-bench", isDirectory: false),
            metallibURL: resourcesURL.appendingPathComponent("mlx.metallib", isDirectory: false),
            dashboardURL: resourcesURL.appendingPathComponent("dashboard2.html", isDirectory: false),
            appIconURL: resourcesURL.appendingPathComponent("AppIcon.icns", isDirectory: false),
            menuBarIconURL: resourcesURL.appendingPathComponent("menubar-icon.png", isDirectory: false),
            menuBarIcon2xURL: resourcesURL.appendingPathComponent("menubar-icon@2x.png", isDirectory: false),
            logoURL: resourcesURL.appendingPathComponent("logo.png", isDirectory: false),
            sidebarLogo2xURL: resourcesURL.appendingPathComponent("sidebar-logo@2x.png", isDirectory: false)
        )
    }

    public static func resolve(
        bundleURL: URL = Bundle.main.bundleURL,
        fileManager: FileManager = .default
    ) throws -> Self {
        let layout = expected(bundleURL: bundleURL)
        try validate(layout.backendURL, inside: bundleURL, executable: true, fileManager: fileManager)
        try validate(layout.ironBenchURL, inside: bundleURL, executable: true, fileManager: fileManager)
        for resourceURL in [
            layout.metallibURL,
            layout.dashboardURL,
            layout.appIconURL,
            layout.menuBarIconURL,
            layout.menuBarIcon2xURL,
            layout.logoURL,
            layout.sidebarLogo2xURL,
        ] {
            try validate(resourceURL, inside: bundleURL, executable: false, fileManager: fileManager)
        }
        return layout
    }

    private static func validate(
        _ url: URL,
        inside bundleURL: URL,
        executable: Bool,
        fileManager: FileManager
    ) throws {
        let candidate = url.standardizedFileURL
        guard fileManager.fileExists(atPath: candidate.path) else {
            throw BundledRuntimeLayoutError.missingFile(candidate.path)
        }

        let values = try candidate.resourceValues(forKeys: [
            .isRegularFileKey,
            .isSymbolicLinkKey,
        ])
        if values.isSymbolicLink == true {
            throw BundledRuntimeLayoutError.symbolicLinkNotAllowed(candidate.path)
        }
        guard values.isRegularFile == true else {
            throw BundledRuntimeLayoutError.nonRegularFile(candidate.path)
        }

        let resolvedBundle = bundleURL.standardizedFileURL.resolvingSymlinksInPath()
        let resolvedCandidate = candidate.resolvingSymlinksInPath()
        let bundlePrefix = resolvedBundle.path.hasSuffix("/")
            ? resolvedBundle.path
            : resolvedBundle.path + "/"
        guard resolvedCandidate.path.hasPrefix(bundlePrefix) else {
            throw BundledRuntimeLayoutError.pathEscapesBundle(candidate.path)
        }

        if executable, !fileManager.isExecutableFile(atPath: candidate.path) {
            throw BundledRuntimeLayoutError.nonExecutableFile(candidate.path)
        }
    }
}

public enum BundledChildProcessEnvironment {
    public static func sanitized(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        environment.filter { key, _ in
            !key.hasPrefix("MLX_") && !key.hasPrefix("DYLD_")
        }
    }
}

enum IronMLXAppResourceResolver {
    static func url(forResource name: String, withExtension extensionName: String) -> URL? {
#if IRONMLX_APP_BUNDLE
        let url = BundledRuntimeLayout.expected().bundleURL
            .appendingPathComponent("Contents/Resources", isDirectory: true)
            .appendingPathComponent("\(name).\(extensionName)", isDirectory: false)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
#else
        return Bundle.module.url(forResource: name, withExtension: extensionName)
#endif
    }
}
