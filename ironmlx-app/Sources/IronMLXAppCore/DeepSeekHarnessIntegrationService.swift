import AppKit
import Foundation
import Yams

/// Official DeepSeek Harness only. Community DSH Desktop owns a separate home and bundle ID.
@MainActor
struct DeepSeekHarnessIntegrationService {
    nonisolated static let bundleIdentifier = "com.deepseek.dsh"
    nonisolated static let minimumVersion = "0.2.1-rc.2"
    nonisolated static let credentialReference = "IRONMLX_HARNESS_API_KEY"

    enum Mode: String { case cli, desktop }
    typealias Model = DSHDesktopIntegrationService.Model
    struct Result: Codable {
        var success = false
        var code: String
        var installedVersion: String?
        var minimumVersion = DeepSeekHarnessIntegrationService.minimumVersion
        var configurationPaths: [String] = []
        var backupPaths: [String] = []
    }

    var harnessHome = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".dsh")
    var environment = ProcessInfo.processInfo.environment
    var findExecutable: () async -> URL? = { await AgentCLI.locateExecutable(named: "dsh") }
    var versionOutput: (URL) async throws -> String = {
        // npm's executable uses /usr/bin/env node; Finder does not inherit Homebrew/nvm PATH.
        try await AgentCLI.commandOutput(URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-lc", "exec \"$1\" --version", "ironmlx-dsh-version", $0.path])
    }
    var loginShellHome: () async throws -> String = {
        try await AgentCLI.commandOutput(URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-lc", "printf '%s' \"${DSH_HOME-}\""])
    }
    var applicationURLs: () -> [URL] = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return NSWorkspace.shared.urlsForApplications(withBundleIdentifier: bundleIdentifier) + [
            URL(fileURLWithPath: "/Applications/DeepSeek Harness.app"),
            home.appendingPathComponent("Applications/DeepSeek Harness.app"),
        ]
    }
    var isDesktopRunning: () -> Bool = {
        !NSRunningApplication.runningApplications(withBundleIdentifier: bundleIdentifier).isEmpty
    }

    func inspect(mode: Mode) async -> Result {
        var versions: [String?] = []
        switch mode {
        case .cli:
            if let executable = await findExecutable() {
                // A command called dsh, or a bundled community runtime, is not proof of an official installation.
                guard Self.isOfficialCLI(executable) else { return Result(code: "cli_unrecognized") }
                versions.append((try? await versionOutput(executable))?.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        case .desktop:
            versions = applicationURLs().compactMap { url -> String? in
                guard let info = Self.applicationInfo(url),
                      info["CFBundleIdentifier"] as? String == Self.bundleIdentifier else { return nil }
                return info["CFBundleShortVersionString"] as? String ?? "unknown"
            }.map(Optional.some)
        }
        guard !versions.isEmpty else { return Result(code: "not_installed") }
        if let version = versions.compactMap({ $0 }).first(where: Self.supports) {
            return Result(success: true, code: "ready", installedVersion: version)
        }
        if let version = versions.compactMap({ $0 }).first(where: { Self.versionParts($0) != nil }) {
            return Result(code: "version_too_old", installedVersion: version)
        }
        return Result(code: "version_unknown")
    }

    private static func applicationInfo(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url.appendingPathComponent("Contents/Info.plist")) else { return nil }
        return try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any]
    }

    static func isOfficialCLI(_ url: URL) -> Bool {
        var parent = url.resolvingSymlinksInPath().deletingLastPathComponent()
        while parent.path != "/" {
            if parent.pathExtension == "app" {
                return applicationInfo(parent)?["CFBundleIdentifier"] as? String == bundleIdentifier
            }
            parent.deleteLastPathComponent()
        }
        parent = url.resolvingSymlinksInPath().deletingLastPathComponent()
        while parent.path != "/" {
            if let data = try? Data(contentsOf: parent.appendingPathComponent("package.json")),
               let package = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               package["name"] as? String == "@deepseek-ai/dsh",
               let bin = package["bin"] as? [String: String], let entry = bin["dsh"] {
                return parent.appendingPathComponent(entry).resolvingSymlinksInPath()
                    == url.resolvingSymlinksInPath()
            }
            parent.deleteLastPathComponent()
        }
        return false
    }

    // The verified adapter targets the published RC format; accept stable and subsequent RC releases.
    // Development/test/alpha builds are deliberately not treated as released installations.
    private static func versionParts(_ value: String) -> [Int]? {
        guard value.range(of: #"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-rc\.(0|[1-9][0-9]*))?$"#,
                          options: .regularExpression) != nil else { return nil }
        let parts = value.replacingOccurrences(of: "-rc.", with: ".").split(separator: ".")
        var numbers = parts.compactMap { Int($0) }
        guard numbers.count == parts.count else { return nil }
        if numbers.count == 3 { numbers.append(Int.max) }
        return numbers
    }

    static func supports(_ version: String) -> Bool {
        guard let candidate = versionParts(version), let minimum = versionParts(minimumVersion) else { return false }
        return !candidate.lexicographicallyPrecedes(minimum)
    }

    func configure(mode: Mode, model: Model, port: UInt16) async -> Result {
        var result = await inspect(mode: mode)
        guard result.success else { return result }
        result.success = false
        // Only the official default home is supported automatically. Never follow DSH_HOME into community data.
        var homeOverrides = [environment["DSH_HOME"] ?? ""]
        if mode == .cli {
            guard let shellHome = try? await loginShellHome() else { result.code = "home_unknown"; return result }
            homeOverrides.append(shellHome)
        }
        let defaultHome = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".dsh").path
        guard homeOverrides.allSatisfy({ value in
            let value = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty || value == defaultHome || value == "~/.dsh"
        }) else { result.code = "custom_home"; return result }
        guard !isDesktopRunning() else { result.code = "close_app"; return result }
        guard !model.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, model.id != "MODEL_ID",
              model.contextWindow > 0, model.maxTokens > 0, model.maxTokens <= model.contextWindow, port > 0
        else { result.code = "model_unavailable"; return result }

        let fm = FileManager.default
        let profiles = mode == .cli ? ["web", "headless"] : ["desktop"]
        let patches = profiles.map { harnessHome.appendingPathComponent("profiles/\($0)/cordis.patch.yml") }
        let credentials = harnessHome.appendingPathComponent(".credentials.yaml")
        do {
            let homePatch = harnessHome.appendingPathComponent("cordis.patch.yml")
            for url in patches + [credentials, homePatch, harnessHome.appendingPathComponent("settings.yaml")] {
                try checkPath(url)
            }
            guard !fm.fileExists(atPath: harnessHome.appendingPathComponent("settings.yaml").path) else {
                result.code = "initialize_app"; return result
            }
            if let homeRows = try rows(readIfPresent(homePatch)), Self.containsManagedEntry(homeRows) {
                result.code = "home_conflict"; return result
            }
            // A live host may rewrite the shared credential document; require it to be stopped first.
            for profile in ["desktop", "web", "headless"] {
                let lock = harnessHome.appendingPathComponent("profiles/\(profile)/lock")
                try checkPath(lock)
                if fm.fileExists(atPath: lock.path) { result.code = "close_app"; return result }
            }
            var writes: [(URL, Data?, Data)] = []
            for url in patches {
                try validateProfile(url.deletingLastPathComponent())
                let old = try readIfPresent(url)
                writes.append((url, old, try mergedPatch(old, model: model, port: port)))
            }
            let oldCredentials = try readIfPresent(credentials)
            writes.append((credentials, oldCredentials, try mergedCredentials(oldCredentials)))
            writes = writes.filter { $0.1 != $0.2 }
            let backupID = UUID().uuidString
            for (url, old, _) in writes {
                if let old {
                    let backup = url.appendingPathExtension("ironmlx-backup-\(backupID)")
                    try old.write(to: backup, options: .atomic)
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backup.path)
                    result.backupPaths.append(backup.path)
                }
            }
            var written: [(URL, Data?)] = []
            do {
                for (url, old, data) in writes {
                    try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                           attributes: [.posixPermissions: 0o700])
                    try data.write(to: url, options: .atomic)
                    written.append((url, old))
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                }
            } catch {
                for (url, old) in written.reversed() {
                    if let old {
                        try old.write(to: url, options: .atomic)
                        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                    } else { try fm.removeItem(at: url) }
                }
                throw error
            }
            result.success = true
            result.code = "configured"
            result.configurationPaths = (patches + [credentials]).map(\.path)
        } catch {
            result.code = "configuration_failed"
        }
        return result
    }

    private enum ConfigurationError: Error { case invalidDocument, unsafePath }

    private func checkPath(_ url: URL) throws {
        var path = url
        while path.path.hasPrefix(harnessHome.path + "/") || path.path == harnessHome.path {
            if let attributes = try? FileManager.default.attributesOfItem(atPath: path.path) {
                guard attributes[.type] as? FileAttributeType != .typeSymbolicLink else {
                    throw ConfigurationError.unsafePath
                }
            }
            if path.path == harnessHome.path { break }
            path.deleteLastPathComponent()
        }
    }

    private func validateProfile(_ directory: URL) throws {
        let manifest = directory.appendingPathComponent("package.json")
        try checkPath(manifest)
        guard let data = try readIfPresent(manifest) else { return } // Official first launch initializes it.
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let dsh = json["dsh"] as? [String: Any], let profile = dsh["profile"] as? [String: Any],
              let bundles = profile["bundles"] as? [String], bundles.contains("@deepseek-ai/dsh-base") else {
            throw ConfigurationError.invalidDocument
        }
    }

    private func readIfPresent(_ url: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try Data(contentsOf: url)
    }

    private func yaml(_ data: Data?) throws -> Any? {
        guard let data else { return nil }
        guard let text = String(data: data, encoding: .utf8) else { throw ConfigurationError.invalidDocument }
        return try Yams.load(yaml: text)
    }

    private func rows(_ data: Data?) throws -> [[String: Any]]? {
        guard let loaded = try yaml(data) else { return nil }
        guard let rows = loaded as? [[String: Any]] else { throw ConfigurationError.invalidDocument }
        return rows
    }

    private static func containsManagedEntry(_ rows: [[String: Any]]) -> Bool {
        rows.contains { row in
            ["llm-pi-ai", "agent-default-model", "credentials"].contains(row["id"] as? String ?? "")
                || containsManagedEntry(row["config"] as? [[String: Any]] ?? [])
                || containsManagedEntry(row["insert"] as? [[String: Any]] ?? [])
        }
    }

    private func mergedPatch(_ data: Data?, model: Model, port: UInt16) throws -> Data {
        let loaded = try rows(data)
        var rows = loaded ?? []
        for row in rows {
            guard row["id"] as? String != "credentials",
                  !Self.containsManagedEntry(row["config"] as? [[String: Any]] ?? []),
                  !Self.containsManagedEntry(row["insert"] as? [[String: Any]] ?? []) else {
                throw ConfigurationError.invalidDocument
            }
        }
        for id in ["llm-pi-ai", "agent-default-model"] {
            let indices = rows.indices.filter { rows[$0]["id"] as? String == id }
            guard indices.count <= 1 else { throw ConfigurationError.invalidDocument }
            let index = indices.first ?? rows.count
            if index == rows.count { rows.append(["id": id]) }
            guard rows[index]["disabled"] == nil || rows[index]["disabled"] as? Bool == false,
                  rows[index]["name"] == nil || rows[index]["name"] as? String == "@deepseek-ai/dsh-\(id)" else {
                throw ConfigurationError.invalidDocument
            }
            var config = try dictionary(rows[index]["config"])
            if id == "agent-default-model" {
                config["provider"] = "ironmlx-local"
                config["model"] = model.id
                config.removeValue(forKey: "reasoningEffort")
            } else {
                var providers = try dictionary(config["providers"])
                var provider = try dictionary(providers["ironmlx-local"])
                guard provider["models"] == nil || provider["models"] is [[String: Any]] else {
                    throw ConfigurationError.invalidDocument
                }
                var models = provider["models"] as? [[String: Any]] ?? []
                var selected = models.first { $0["id"] as? String == model.id } ?? [:]
                selected.merge([
                    "id": model.id, "name": model.id, "contextWindow": model.contextWindow,
                    "maxTokens": min(4096, model.maxTokens), "reasoningEfforts": false,
                ]) { _, new in new }
                models.removeAll { $0["id"] as? String == model.id }
                models.append(selected)
                provider.merge([
                    "displayName": "IronMLX", "api": "openai-responses",
                    "baseURL": "http://127.0.0.1:\(port)/v1", "apiKeyEnv": Self.credentialReference,
                    "cacheRetention": "none", "transport": "sse", "models": models,
                ]) { _, new in new }
                // Retained request overrides must not route this local provider through an old proxy/key.
                for key in ["apiKey", "headers", "baseURLEnv", "apiKeyEnvFallbacks", "reasoning", "modelOverrides"] {
                    provider.removeValue(forKey: key)
                }
                providers["ironmlx-local"] = provider
                config["providers"] = providers
            }
            rows[index]["config"] = config
        }
        if let data, let previous = loaded as NSArray?, previous.isEqual(to: rows) { return data }
        return Data(try Yams.dump(object: rows, sortKeys: true).utf8)
    }

    private func dictionary(_ value: Any?) throws -> [String: Any] {
        guard let value else { return [:] }
        guard let result = value as? [String: Any] else { throw ConfigurationError.invalidDocument }
        return result
    }

    private func mergedCredentials(_ data: Data?) throws -> Data {
        let loaded = try yaml(data)
        var document = try dictionary(loaded)
        guard document.isEmpty || document["version"] as? Int == 1 else { throw ConfigurationError.invalidDocument }
        guard document["refs"] == nil || document["refs"] is [String: String] else {
            throw ConfigurationError.invalidDocument
        }
        var refs = document["refs"] as? [String: String] ?? [:]
        if refs[Self.credentialReference] == "local", let data { return data }
        refs[Self.credentialReference] = "local"
        document["version"] = 1
        document["refs"] = refs
        return Data(try Yams.dump(object: document, sortKeys: true).utf8)
    }
}
