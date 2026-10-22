import AppKit
import Foundation
import Yams

/// Community DSH Desktop only. Keep this contract separate from DeepSeek's official app.
@MainActor
struct DSHDesktopIntegrationService {
    nonisolated static let bundleIdentifier = "io.dsh.desktop"
    // First configuration format supported by this adapter (Harness 0.2.1-rc.2).
    nonisolated static let minimumVersion = "0.12.1"
    nonisolated static let downloadURL = "https://dshdesktop.com/en/"
    nonisolated static let credentialReference = "IRONMLX_DSH_DESKTOP_API_KEY"

    struct Result: Codable {
        var success = false
        var code: String
        var installedVersion: String?
        var minimumVersion = DSHDesktopIntegrationService.minimumVersion
        var downloadURL = DSHDesktopIntegrationService.downloadURL
        var configurationPath: String?
        var backupPaths: [String] = []
    }

    struct Model {
        let id: String
        let contextWindow: Int
        let maxTokens: Int
    }

    var applicationURLs: () -> [URL] = {
        let fm = FileManager.default
        return NSWorkspace.shared.urlsForApplications(withBundleIdentifier: bundleIdentifier) + [
            URL(fileURLWithPath: "/Applications/DSH Desktop.app"),
            fm.homeDirectoryForCurrentUser.appendingPathComponent("Applications/DSH Desktop.app"),
        ]
    }
    var isRunning: () -> Bool = {
        !NSRunningApplication.runningApplications(withBundleIdentifier: bundleIdentifier).isEmpty
    }
    var harnessHome = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/dsh-desktop/harness")

    func inspect() -> Result {
        let versions = applicationURLs().compactMap { url -> String? in
            guard let data = try? Data(contentsOf: url.appendingPathComponent("Contents/Info.plist")),
                  let info = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
                  info["CFBundleIdentifier"] as? String == Self.bundleIdentifier
            else { return nil }
            return info["CFBundleShortVersionString"] as? String ?? "unknown"
        }
        guard !versions.isEmpty else { return Result(code: "not_installed") }
        let supported = versions.filter { Self.supports($0) }.sorted {
            $0.compare($1, options: .numeric) == .orderedDescending
        }
        if let version = supported.first {
            return Result(success: true, code: "ready", installedVersion: version)
        }
        let version = versions.sorted { $0.compare($1, options: .numeric) == .orderedDescending }.first ?? "unknown"
        return Result(
            code: Self.versionComponents(version) == nil ? "version_unknown" : "version_too_old",
            installedVersion: version
        )
    }

    private static func versionComponents(_ version: String) -> [Int]? {
        let parts = version.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3 else { return nil }
        let numbers = parts.compactMap { part -> Int? in
            guard !part.isEmpty, part.allSatisfy({ $0.isASCII && $0.isNumber }) else { return nil }
            return Int(part)
        }
        return numbers.count == 3 ? numbers : nil
    }

    static func supports(_ version: String) -> Bool {
        guard let candidate = versionComponents(version), let minimum = versionComponents(minimumVersion) else {
            return false
        }
        return !candidate.lexicographicallyPrecedes(minimum)
    }

    /// Recheck installation before reading or creating any configuration, even for direct bridge calls.
    func configure(model: Model, port: UInt16) -> Result {
        var result = inspect()
        guard result.success else { return result }
        result.success = false
        guard !isRunning() else { result.code = "close_app"; return result }
        guard !model.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              model.id != "MODEL_ID", model.contextWindow > 0, model.maxTokens > 0,
              model.maxTokens <= model.contextWindow, port > 0
        else { result.code = "model_unavailable"; return result }

        let fm = FileManager.default
        let patchURL = harnessHome.appendingPathComponent("profiles/web/cordis.patch.yml")
        let credentialsURL = harnessHome.appendingPathComponent(".credentials.yaml")
        // Legacy settings are migrated by Desktop itself; do not race that migration.
        guard !fm.fileExists(atPath: harnessHome.appendingPathComponent("settings.yaml").path) else {
            result.code = "initialize_app"
            return result
        }
        do {
            let oldPatch = try readIfPresent(patchURL)
            let oldCredentials = try readIfPresent(credentialsURL)
            let patch = try mergedPatch(oldPatch, model: model, port: port)
            let credentials = try mergedCredentials(oldCredentials)
            let writes = [(credentialsURL, oldCredentials, credentials), (patchURL, oldPatch, patch)]
                .filter { $0.1 != $0.2 }
            let backupID = UUID().uuidString
            // Prepare every backup before changing either file.
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
                    }
                    else { try fm.removeItem(at: url) }
                }
                throw error
            }
            result.success = true
            result.code = "configured"
            result.configurationPath = patchURL.path
        } catch {
            result.code = "configuration_failed"
        }
        return result
    }

    private func readIfPresent(_ url: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try Data(contentsOf: url)
    }

    private enum ConfigurationError: Error { case invalidDocument }

    private func yaml(_ data: Data?) throws -> Any? {
        guard let data else { return nil }
        guard let text = String(data: data, encoding: .utf8) else { throw ConfigurationError.invalidDocument }
        return try Yams.load(yaml: text)
    }

    private func mergedPatch(_ data: Data?, model: Model, port: UInt16) throws -> Data {
        let loaded = try yaml(data)
        guard loaded == nil || loaded is [[String: Any]] else { throw ConfigurationError.invalidDocument }
        var rows = loaded as? [[String: Any]] ?? []
        let indices = rows.indices.filter { rows[$0]["id"] as? String == "llm-pi-ai" }
        guard indices.count <= 1 else { throw ConfigurationError.invalidDocument }
        let index = indices.first ?? rows.count
        if index == rows.count { rows.append(["id": "llm-pi-ai"]) }
        if let existing = rows[index]["config"], !(existing is [String: Any]) {
            throw ConfigurationError.invalidDocument
        }
        var config = rows[index]["config"] as? [String: Any] ?? [:]
        if let existing = config["providers"], !(existing is [String: Any]) {
            throw ConfigurationError.invalidDocument
        }
        var providers = config["providers"] as? [String: Any] ?? [:]
        if let existing = providers["ironmlx-local"], !(existing is [String: Any]) {
            throw ConfigurationError.invalidDocument
        }
        var provider = providers["ironmlx-local"] as? [String: Any] ?? [:]
        if let existing = provider["models"], !(existing is [[String: Any]]) {
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
            "cacheRetention": "none", "models": models,
        ]) { _, new in new }
        providers["ironmlx-local"] = provider
        config["providers"] = providers
        rows[index]["config"] = config
        let encoded = Data(try Yams.dump(object: rows, sortKeys: true).utf8)
        // Repeated configuration should neither rewrite files nor generate redundant backups.
        if let data, let previous = loaded as? NSArray,
           previous.isEqual(to: rows) { return data }
        return encoded
    }

    private func mergedCredentials(_ data: Data?) throws -> Data {
        let loaded = try yaml(data)
        guard loaded == nil || loaded is [String: Any] else { throw ConfigurationError.invalidDocument }
        var document = loaded as? [String: Any] ?? [:]
        if !document.isEmpty, document["version"] as? Int != 1 { throw ConfigurationError.invalidDocument }
        if let existing = document["refs"], !(existing is [String: String]) {
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
