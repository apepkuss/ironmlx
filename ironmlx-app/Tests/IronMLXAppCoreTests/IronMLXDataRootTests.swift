import Foundation
import Testing

@testable import IronMLXAppCore

private let fakeHome = URL(fileURLWithPath: "/Users/tester", isDirectory: true)

@Test func dataRootDefaultsToDotIronmlxUnderHome() {
    let resolution = IronMLXDataRoot.resolve(environmentValue: nil, home: fakeHome) { _ in }

    #expect(resolution.url.path == "/Users/tester/.ironmlx")
    #expect(!resolution.isOverridden)
    #expect(resolution.displayPath("cache/paged_prefix_cache") == "~/.ironmlx/cache/paged_prefix_cache")
}

@Test func dataRootOverrideAcceptsAbsolutePath() {
    var logs: [String] = []
    let resolution = IronMLXDataRoot.resolve(environmentValue: "/tmp/ironmlx-e2e/", home: fakeHome) {
        logs.append($0)
    }

    #expect(resolution.url.path == "/tmp/ironmlx-e2e")
    #expect(resolution.isOverridden)
    #expect(resolution.displayPath("cache/paged_prefix_cache") == "/tmp/ironmlx-e2e/cache/paged_prefix_cache")
    #expect(logs.count == 1)
}

@Test func dataRootOverrideExpandsTilde() {
    #expect(IronMLXDataRoot.parse("~/scratch/root", home: fakeHome)?.path == "/Users/tester/scratch/root")
    #expect(IronMLXDataRoot.parse("~", home: fakeHome)?.path == "/Users/tester")
    #expect(IronMLXDataRoot.parse("  /tmp/x  ", home: fakeHome)?.path == "/tmp/x")
}

@Test func dataRootOverrideIgnoresEmptyOrRelativeValues() {
    #expect(IronMLXDataRoot.parse(nil, home: fakeHome) == nil)
    #expect(IronMLXDataRoot.parse("", home: fakeHome) == nil)
    #expect(IronMLXDataRoot.parse("   ", home: fakeHome) == nil)
    #expect(IronMLXDataRoot.parse("relative/root", home: fakeHome) == nil)
    #expect(IronMLXDataRoot.parse("~other/root", home: fakeHome) == nil)

    var logs: [String] = []
    let resolution = IronMLXDataRoot.resolve(environmentValue: "relative/root", home: fakeHome) {
        logs.append($0)
    }
    #expect(resolution.url.path == "/Users/tester/.ironmlx")
    #expect(!resolution.isOverridden)
    #expect(logs.count == 1)
    #expect(logs.first?.contains("IRONMLX_DEBUG_DATA_ROOT") == true)
}

@Test func testProcessUsesATemporaryRootWhenTheVariableIsUnset() {
    let testRoot = URL(fileURLWithPath: "/tmp/ironmlx-tests-1", isDirectory: true)
    let resolution = IronMLXDataRoot.resolve(environmentValue: nil, home: fakeHome, testRoot: testRoot) { _ in }

    #expect(resolution.url.path == "/tmp/ironmlx-tests-1")
    #expect(resolution.isOverridden)
    #expect(resolution.displayPath("logs") == "/tmp/ironmlx-tests-1/logs")

    let explicit = IronMLXDataRoot.resolve(environmentValue: "/tmp/e2e", home: fakeHome, testRoot: testRoot) { _ in }
    #expect(explicit.url.path == "/tmp/e2e")

    let invalid = IronMLXDataRoot.resolve(environmentValue: "relative", home: fakeHome, testRoot: testRoot) { _ in }
    #expect(invalid.url.path == "/tmp/ironmlx-tests-1")
}

/// The test runner must never resolve the real `~/.ironmlx`, whether or not
/// `IRONMLX_DEBUG_DATA_ROOT` is set.
@Test func thisTestProcessNeverResolvesTheRealDataRoot() {
    #expect(IronMLXDataRoot.isTestProcess)
    let real = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".ironmlx", isDirectory: true).standardizedFileURL.path
    #expect(IronMLXDataRoot.url.standardizedFileURL.path != real)
    #expect(!IronMLXDataRoot.url.standardizedFileURL.path.hasPrefix(real + "/"))
    #expect(IronMLXLogStore.defaultRootURL().standardizedFileURL.path != real + "/logs")
}

@MainActor @Test func appDataPathsAreDerivedFromTheResolvedDataRoot() {
    let root = IronMLXDataRoot.url.standardizedFileURL.path + "/"
    let paths = [
        AppConfigStore.defaultConfigURL(),
        BackendIncidentStore.defaultURL(),
        IronMLXLogStore.defaultRootURL(),
        ModelParameterStore.defaultURL(),
        ChatStore.defaultRoot,
        LocalModelScanner().rootURL.appendingPathComponent("x"),
    ]
    for path in paths {
        #expect(path.standardizedFileURL.path.hasPrefix(root), Comment(rawValue: "\(path.path) is outside \(root)"))
    }
    #expect(
        BackendLaunchOptions.defaultPagedPrefixCacheDirectory
            == IronMLXDataRoot.current.displayPath("cache/paged_prefix_cache")
    )
}

@Test func serveArgumentsPassEveryDataRootedPathExplicitly() {
    let root = URL(fileURLWithPath: "/tmp/ironmlx-data", isDirectory: true)
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        metallibURL: URL(fileURLWithPath: "/tmp/mlx.metallib"),
        host: "127.0.0.1",
        port: 9068,
        dataRootURL: root
    )

    #expect(config.arguments == [
        "--mlx-metallib", "/tmp/mlx.metallib",
        "--data-root", "/tmp/ironmlx-data",
        "serve",
        "--host", "127.0.0.1",
        "--port", "9068",
        "--network-mode", "local",
        "--voice-profile-dir", "/tmp/ironmlx-data/audio/voices",
        "--active-kv-offload-dir", "/tmp/ironmlx-data/cache/active_kv_offload",
    ])
}

@Test func dflash2ServeArgumentsAlsoPassDataRootedPaths() {
    let root = URL(fileURLWithPath: "/tmp/ironmlx-data", isDirectory: true)
    let runtime = ModelDFlash2Runtime(
        targetModelID: "target",
        targetModelDir: "/models/target",
        draftModelID: "draft",
        draftModelDir: "/models/draft",
        checkpointBlockSize: 8,
        blockSize: 4,
        draftBits: 8,
        tensorBatchMaxWidth: 6,
        maxCacheCap: 65_536
    )
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        dflash2Runtime: runtime,
        dataRootURL: root
    )

    #expect(Array(config.arguments.prefix(3)) == ["--data-root", "/tmp/ironmlx-data", "serve"])
    #expect(config.arguments.contains("/tmp/ironmlx-data/audio/voices"))
    #expect(config.arguments.contains("/tmp/ironmlx-data/cache/active_kv_offload"))
}

/// Every `.ironmlx` path component in the App sources must come from
/// `IronMLXDataRoot`. A new hard-coded root would silently escape
/// `IRONMLX_DEBUG_DATA_ROOT` and touch the real `~/.ironmlx` during tests.
@Test func sourcesDoNotHardCodeTheDataRoot() throws {
    let sources = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Sources", isDirectory: true)
    let resolverFile = "IronMLXAppCore/IronMLXDataRoot.swift"
    // Each entry: relative file path, the allowed line (trimmed), how it is
    // matched, and why. `.exact` must equal the whole trimmed line; `.keyedText`
    // must start with the given key and contain the given text.
    enum Match {
        case exact(String)
        case keyedText(key: String, text: String)

        func matches(_ trimmedLine: String) -> Bool {
            switch self {
            case .exact(let line):
                return trimmedLine == line
            case .keyedText(let key, let text):
                return trimmedLine.hasPrefix(key) && trimmedLine.contains(text)
            }
        }
    }
    let html = "IronMLXAppCore/Resources/dashboard2.html"
    let allowlist: [(file: String, match: Match, reason: String)] = [
        ("IronMLXAppCore/Chat/ChatStore.swift",
         .exact("/// Chat history and chat preferences under `~/.ironmlx/chats/`: one JSON file"),
         "doc comment describing the default location"),
        ("IronMLXAppCore/Chat/ChatSessionController+Live.swift",
         .exact("/// The App's wiring: history in `~/.ironmlx/chats/`, requests to the"),
         "doc comment describing the default location"),
        (html,
         .exact(#"<div class="settings-row-right"><input type="text" value="~/.ironmlx/cache/paged_prefix_cache" style="min-width:260px;" id="cfg-cache-dir"></div>"#),
         "static placeholder replaced on load by the injected default cache dir"),
        (html,
         .exact("setTextInput('cfg-cache-dir', appConfig.cache_dir, window.__IRONMLX_DEFAULT_CACHE_DIR__ || '~/.ironmlx/cache/paged_prefix_cache');"),
         "fallback only when the App did not inject the default cache dir"),
        (html,
         .keyedText(key: "err_backend_readiness_failed: \"", text: "~/.ironmlx/logs/backend.log"),
         "user-facing error copy naming the default log location"),
        (html,
         .keyedText(key: "err_backend_load_timeout: \"", text: "~/.ironmlx/logs/backend.log"),
         "user-facing error copy naming the default log location"),
    ]
    var allowlistHits = Array(repeating: 0, count: allowlist.count)
    let pattern = try NSRegularExpression(pattern: #"(?<![A-Za-z0-9_])\.ironmlx(?![A-Za-z0-9_-])"#)
    let enumerator = try #require(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
    var violations: [String] = []
    var scannedFiles = 0
    for case let fileURL as URL in enumerator {
        let ext = fileURL.pathExtension
        guard ["swift", "html", "js", "json", "css"].contains(ext) else { continue }
        let relative = String(fileURL.standardizedFileURL.path.dropFirst(sources.standardizedFileURL.path.count + 1))
        guard relative != resolverFile else { continue }
        guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { continue }
        scannedFiles += 1
        for (index, line) in text.components(separatedBy: "\n").enumerated() {
            let range = NSRange(line.startIndex..., in: line)
            guard pattern.firstMatch(in: line, range: range) != nil else { continue }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let entry = allowlist.indices.first(where: {
                allowlist[$0].file == relative && allowlist[$0].match.matches(trimmed)
            }) {
                allowlistHits[entry] += 1
            } else {
                violations.append("\(relative):\(index + 1): \(trimmed)")
            }
        }
    }
    for (entry, hits) in zip(allowlist, allowlistHits) {
        #expect(hits > 0, "Dead allowlist entry in \(entry.file): \(entry.match) (\(entry.reason))")
    }
    #expect(scannedFiles > 50)
    #expect(violations.isEmpty, "Use IronMLXDataRoot instead of a hard-coded .ironmlx root:\n\(violations.joined(separator: "\n"))")
}
