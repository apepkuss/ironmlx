import AppKit
import Foundation
import Yams

/// Writes a dedicated profile shared by Hermes CLI and local Desktop installations.
@MainActor
struct HermesIntegrationService {
    nonisolated static let minimumVersion = "0.20.0"
    nonisolated static let installationURL = "https://hermes-agent.nousresearch.com/docs/getting-started/installation"
    static let desktopIdentifiers = ["com.nousresearch.hermes", "com.nousresearch.hermes-bundled"]

    struct Result: Codable {
        var success = false
        var code: String
        var installedVersion: String?
        var minimumVersion = HermesIntegrationService.minimumVersion
        var configurationPath: String?
        var backupPaths: [String] = []
    }

    struct DesktopInstallation {
        let bundleIdentifier: String
        let version: String
        var hasLocalRuntime = true
    }

    var hermesHome = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".hermes")
    var findExecutable: () async -> URL? = { await AgentCLI.locateExecutable(named: "hermes") }
    var versionOutput: (URL) async throws -> String = {
        try await AgentCLI.commandOutput($0, arguments: ["--version"])
    }
    var desktopInstallations: () -> [DesktopInstallation] = { findDesktopInstallations() }
    var environment = ProcessInfo.processInfo.environment

    func inspect() async -> Result {
        var candidates: [String?] = []
        if let executable = await findExecutable() {
            candidates.append((try? await versionOutput(executable)).flatMap(Self.parseCLIVersion))
        }
        for desktop in desktopInstallations() where Self.desktopIdentifiers.contains(desktop.bundleIdentifier) {
            if desktop.hasLocalRuntime {
                candidates.append(Self.parseVersion(desktop.version))
            } else if candidates.isEmpty {
                // A bootstrap shell alone cannot serve the local profile yet.
                candidates.append(nil)
            }
        }
        if let version = candidates.compactMap({ $0 }).first(where: {
            $0.compare(Self.minimumVersion, options: .numeric) != .orderedAscending
        }) {
            return Result(success: true, code: "ready", installedVersion: version)
        }
        guard !candidates.isEmpty else { return Result(code: "not_installed") }
        if let version = candidates.compactMap({ $0 }).first {
            return Result(code: "version_too_old", installedVersion: version)
        }
        return Result(code: "version_unknown")
    }

    nonisolated static func parseVersion(_ value: String) -> String? {
        guard value.range(of: #"^[0-9]+\.[0-9]+\.[0-9]+(?:\+[A-Za-z0-9.-]+)?$"#,
                          options: .regularExpression) != nil else { return nil }
        return String(value.split(separator: "+", maxSplits: 1)[0])
    }

    nonisolated static func parseCLIVersion(_ output: String) -> String? {
        // Official --version starts with this identity; the rest describes the installation.
        let first = output.components(separatedBy: .newlines).first ?? ""
        guard first.hasPrefix("Hermes Agent v") else { return nil }
        let version = first.dropFirst("Hermes Agent v".count).split(separator: " ").first.map(String.init) ?? ""
        return parseVersion(version)
    }

    static func findDesktopInstallations() -> [DesktopInstallation] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        var urls = desktopIdentifiers.compactMap { NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) }
        for directory in [URL(fileURLWithPath: "/Applications"), home.appendingPathComponent("Applications")] {
            urls += ["Hermes.app", "Hermes Agent.app"].map { directory.appendingPathComponent($0) }
        }
        return urls.compactMap(readDesktopInstallation)
    }

    static func readDesktopInstallation(at url: URL) -> DesktopInstallation? {
        let fm = FileManager.default
        guard let bundle = Bundle(url: url), let identifier = bundle.bundleIdentifier,
              desktopIdentifiers.contains(identifier), let executable = bundle.executableURL,
              fm.isExecutableFile(atPath: executable.path), let resources = bundle.resourceURL else { return nil }
        var installation = DesktopInstallation(bundleIdentifier: identifier,
            version: bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "",
            hasLocalRuntime: false)
        let stampURL = resources.appendingPathComponent("install-stamp.json")
        guard let data = try? Data(contentsOf: stampURL),
              let stamp = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              stamp["payload"] as? String == "bundled", let runtime = stamp["runtime"] as? [String: Any],
              let commands = runtime["commands"] as? [String: String], let shim = commands["hermes"],
              let repo = runtime["repoDir"] as? String else { return installation }
        let payload = resources.appendingPathComponent("agent-payload").resolvingSymlinksInPath()
        let launcher = payload.appendingPathComponent(shim).resolvingSymlinksInPath()
        let main = payload.appendingPathComponent(repo).appendingPathComponent("hermes_cli/main.py").resolvingSymlinksInPath()
        installation.hasLocalRuntime = launcher.path.hasPrefix(payload.path + "/")
            && main.path.hasPrefix(payload.path + "/") && fm.isExecutableFile(atPath: launcher.path)
            && fm.fileExists(atPath: main.path)
        installation = DesktopInstallation(bundleIdentifier: identifier,
            version: stamp["baseVersion"] as? String ?? installation.version,
            hasLocalRuntime: installation.hasLocalRuntime)
        return installation
    }

    func configure(modelID: String, contextWindow: Int, port: UInt16) async -> Result {
        var result = await inspect()
        guard result.success else { return result }
        result.success = false
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              modelID != "MODEL_ID", port > 0, contextWindow > 0 else {
            result.code = "model_unavailable"
            return result
        }
        guard contextWindow >= 64000 else {
            result.code = "context_too_small"
            return result
        }
        let customHome = environment["HERMES_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard (customHome.isEmpty || NSString(string: customHome).expandingTildeInPath == hermesHome.path),
              (environment["HERMES_DATA_DIR_SUFFIX"] ?? "").isEmpty else {
            result.code = "custom_home"
            return result
        }
        let fm = FileManager.default
        let profiles = hermesHome.appendingPathComponent("profiles")
        let profile = profiles.appendingPathComponent("ironmlx")
        let url = profile.appendingPathComponent("config.yaml")
        guard !fm.fileExists(atPath: profiles.appendingPathComponent(".deleted/ironmlx").path),
              profile.resolvingSymlinksInPath().path == hermesHome.resolvingSymlinksInPath().appendingPathComponent("profiles/ironmlx").path,
              url.resolvingSymlinksInPath().path == profile.resolvingSymlinksInPath().appendingPathComponent("config.yaml").path else {
            result.code = "profile_conflict"
            return result
        }
        do {
            let old = fm.fileExists(atPath: url.path) ? try Data(contentsOf: url) : nil
            let data = try mergedConfiguration(old, modelID: modelID, contextWindow: contextWindow, port: port)
            if old != data {
                if let old {
                    let backup = url.appendingPathExtension("ironmlx-backup-\(UUID().uuidString)")
                    try old.write(to: backup, options: .atomic)
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backup.path)
                    result.backupPaths.append(backup.path)
                }
                let existed = fm.fileExists(atPath: profile.path)
                try fm.createDirectory(at: profile, withIntermediateDirectories: true,
                                       attributes: [.posixPermissions: 0o700])
                do {
                    try data.write(to: url, options: .atomic)
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                } catch {
                    if let old {
                        try old.write(to: url, options: .atomic)
                        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                    } else if !existed {
                        try fm.removeItem(at: profile)
                    } else if fm.fileExists(atPath: url.path) {
                        try fm.removeItem(at: url)
                    }
                    throw error
                }
            }
            result.success = true
            result.code = "configured"
            result.configurationPath = url.path
        } catch {
            result.code = "configuration_failed"
        }
        return result
    }

    private enum ConfigurationError: Error { case invalidDocument }

    private func mergedConfiguration(_ data: Data?, modelID: String, contextWindow: Int, port: UInt16) throws -> Data {
        let loaded: Any?
        if let data {
            guard let text = String(data: data, encoding: .utf8) else { throw ConfigurationError.invalidDocument }
            loaded = try Yams.load(yaml: text)
        } else { loaded = nil }
        guard loaded == nil || loaded is [String: Any] else { throw ConfigurationError.invalidDocument }
        var document = loaded as? [String: Any] ?? [:]
        func mapping(_ value: Any?) throws -> [String: Any] {
            guard value == nil || value is [String: Any] else { throw ConfigurationError.invalidDocument }
            return value as? [String: Any] ?? [:]
        }
        var model = try mapping(document["model"])
        var providers = try mapping(document["providers"])
        var provider = try mapping(providers["ironmlx-responses"])
        let endpoint = "http://127.0.0.1:\(port)/v1"
        model.merge(["default": modelID, "provider": "custom:ironmlx-responses", "context_length": contextWindow,
                     "base_url": endpoint, "api_mode": "codex_responses"]) { _, new in new }
        provider.merge(["api": endpoint, "base_url": endpoint, "transport": "codex_responses",
                        "api_mode": "codex_responses", "discover_models": false, "enabled": true]) { _, new in new }
        document["model"] = model
        providers["ironmlx-responses"] = provider
        document["providers"] = providers
        if let data, let previous = loaded as? NSDictionary, previous.isEqual(to: document) { return data }
        return Data(try Yams.dump(object: document, sortKeys: true).utf8)
    }
}
