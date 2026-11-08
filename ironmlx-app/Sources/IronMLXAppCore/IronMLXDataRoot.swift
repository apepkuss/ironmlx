import Foundation

/// The single root of every App and backend data path: models, config, chats,
/// logs, incidents, caches, voices, scheduler profiles. Defaults to
/// `~/.ironmlx`.
///
/// Debug builds only: `IRONMLX_DEBUG_DATA_ROOT` (an absolute path, `~`
/// expanded) replaces the root, so an end-to-end test can run the App as a new
/// user against an empty directory. Release builds ignore it. Unit-test
/// processes without the variable get a fresh temporary root instead of the
/// real one. The backend instance lock is not under this root; it stays in the
/// real `~/.ironmlx`.
public enum IronMLXDataRoot {
    static let environmentKey = "IRONMLX_DEBUG_DATA_ROOT"
    static let directoryName = ".ironmlx"

    public struct Resolution: Equatable, Sendable {
        public let url: URL
        public let isOverridden: Bool

        /// The path of `relativePath` under the root as the App writes it into
        /// settings and backend flags: `~/.ironmlx/<relativePath>` for the
        /// default root, so existing values stay byte-identical, and an
        /// absolute path for an override.
        public func displayPath(_ relativePath: String) -> String {
            if isOverridden {
                return url.appendingPathComponent(relativePath).path
            }
            return "~/\(IronMLXDataRoot.directoryName)/\(relativePath)"
        }
    }

    static func defaultURL(home: URL) -> URL {
        home.appendingPathComponent(directoryName, isDirectory: true)
    }

    /// Accepts a non-empty absolute path, expanding a leading `~` or `~/`
    /// against `home`; anything else is ignored.
    static func parse(_ value: String?, home: URL) -> URL? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else { return nil }
        let expanded: String
        if trimmed == "~" {
            expanded = home.path
        } else if trimmed.hasPrefix("~/") {
            expanded = home.path + String(trimmed.dropFirst())
        } else {
            expanded = trimmed
        }
        guard expanded.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
    }

    /// `testRoot` is the root of a test process when no valid override is set,
    /// so unit tests never resolve the real `~/.ironmlx`.
    static func resolve(
        environmentValue: String?,
        home: URL,
        testRoot: URL? = nil,
        log: (String) -> Void
    ) -> Resolution {
        let fallback = testRoot.map { Resolution(url: $0, isOverridden: true) }
            ?? Resolution(url: defaultURL(home: home), isOverridden: false)
        guard let environmentValue else {
            return fallback
        }
        guard let override = parse(environmentValue, home: home) else {
            log("\(environmentKey) ignored: expected a non-empty absolute path, got \"\(environmentValue)\"")
            return fallback
        }
        log("\(environmentKey) active: IronMLX data root is \(override.path)")
        return Resolution(url: override, isOverridden: true)
    }

    /// True inside an XCTest or swift-testing runner; never true for the App
    /// or the migration tool.
    static var isTestProcess: Bool {
        let name = ProcessInfo.processInfo.processName
        return name == "xctest" || name == "swiftpm-testing-helper" || NSClassFromString("XCTestCase") != nil
    }

    /// A fresh per-process directory under the system temporary directory.
    private static func makeTestRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "ironmlx-tests-\(ProcessInfo.processInfo.processIdentifier)-\(UUID().uuidString)",
                isDirectory: true
            )
            .standardizedFileURL
    }

    /// Read once at startup.
    public static let current: Resolution = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let testRoot = isTestProcess ? makeTestRoot() : nil
        #if DEBUG
        let environmentValue = ProcessInfo.processInfo.environment[environmentKey]
        #else
        let environmentValue: String? = nil
        #endif
        return resolve(
            environmentValue: environmentValue,
            home: home,
            testRoot: testRoot,
            log: { NSLog("%@", $0) }
        )
    }()

    public static var url: URL { current.url }

    public static func url(appending relativePath: String, isDirectory: Bool = true) -> URL {
        url.appendingPathComponent(relativePath, isDirectory: isDirectory)
    }
}
