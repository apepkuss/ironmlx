import AppKit
import Foundation

/// Registers a native Chat provider without changing OpenCode's default model or starting a session.
@MainActor
struct OpenCodeIntegrationService {
    nonisolated static let minimumVersion = "2.2.5"
    nonisolated static let installationURL = "https://opencode.ai/v2/docs/"
    nonisolated static let providerPackage = "@opencode/ai/providers/openai-compatible"

    struct Result: Codable {
        var success = false
        var code: String
        var installedVersion: String?
        var minimumVersion = OpenCodeIntegrationService.minimumVersion
        var configurationPath: String?
        var backupPaths: [String] = []
    }

    struct Model {
        let id: String
        let contextWindow: Int
        let maxTokens: Int
        var supportsVision = false
        var reasoning: LocalModelReasoningMetadata?
    }

    var home = FileManager.default.homeDirectoryForCurrentUser
    var environment = ProcessInfo.processInfo.environment
    var findExecutable: () async -> URL? = {
        if let executable = await AgentCLI.locateExecutable(named: "opencode") { return executable }
        let executable = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".opencode/bin/opencode")
        return FileManager.default.isExecutableFile(atPath: executable.path) ? executable : nil
    }
    var findDesktopExecutable: () -> URL? = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let applications = [
            NSWorkspace.shared.urlForApplication(withBundleIdentifier: "ai.opencode.desktop"),
            home.appendingPathComponent("Applications/OpenCode.app"),
            URL(fileURLWithPath: "/Applications/OpenCode.app"),
        ].compactMap { $0 }
        return applications.lazy.compactMap { desktopExecutable(in: $0) }.first
    }
    var versionOutput: (URL) async throws -> String = {
        try await AgentCLI.commandOutput($0, arguments: ["--version"])
    }

    func inspect() async -> Result {
        let executables = [await findExecutable(), findDesktopExecutable()].compactMap { $0 }
        guard !executables.isEmpty else { return Result(code: "not_installed") }
        var failure = Result(code: "version_unknown")
        for executable in executables {
            guard let output = try? await versionOutput(executable), let version = Self.parseVersion(output) else { continue }
            let supported = version.compare(Self.minimumVersion, options: .numeric) != .orderedAscending
            let result = Result(success: supported, code: supported ? "ready" : "version_too_old", installedVersion: version)
            if supported { return result }
            if failure.installedVersion == nil || version.compare(failure.installedVersion!, options: .numeric) == .orderedDescending {
                failure = result
            }
        }
        return failure
    }

    static func desktopExecutable(in application: URL) -> URL? {
        guard Bundle(url: application)?.bundleIdentifier == "ai.opencode.desktop" else { return nil }
        let executable = application.appendingPathComponent("Contents/Resources/opencode-cli")
        return FileManager.default.isExecutableFile(atPath: executable.path) ? executable : nil
    }

    nonisolated static func parseVersion(_ output: String) -> String? {
        let line = output.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = line.hasPrefix("opencode v") ? String(line.dropFirst("opencode v".count)) : line
        guard value.range(of: #"^[0-9]+\.[0-9]+\.[0-9]+$"#, options: .regularExpression) != nil else { return nil }
        return value
    }

    func configure(model: Model, port: UInt16) async -> Result {
        var result = await inspect()
        guard result.success else { return result }
        result.success = false
        guard !model.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !model.id.contains("#"), model.id != "MODEL_ID", model.contextWindow > 0,
              model.maxTokens > 0, model.maxTokens <= model.contextWindow, port > 0 else {
            result.code = "model_unavailable"
            return result
        }
        // Inline/explicit configs load after global files. Their process-relative paths cannot be inferred by the App.
        guard ["OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT"].allSatisfy({ environment[$0]?.isEmpty != false }) else {
            result.code = "custom_configuration"
            return result
        }
        do {
            let directory: URL
            if let custom = environment["OPENCODE_CONFIG_DIR"], !custom.isEmpty {
                guard custom.hasPrefix("/") else { throw ConfigurationError.invalidDocument }
                directory = URL(fileURLWithPath: custom, isDirectory: true)
            } else if let xdg = environment["XDG_CONFIG_HOME"], !xdg.isEmpty {
                guard xdg.hasPrefix("/") else { throw ConfigurationError.invalidDocument }
                directory = URL(fileURLWithPath: xdg, isDirectory: true).appendingPathComponent("opencode")
            } else {
                directory = home.appendingPathComponent(".config/opencode")
            }
            let fm = FileManager.default
            let json = directory.appendingPathComponent("opencode.json")
            let jsonc = directory.appendingPathComponent("opencode.jsonc")
            let target = fm.fileExists(atPath: jsonc.path) ? jsonc : fm.fileExists(atPath: json.path) ? json : jsonc
            var changes: [(url: URL, old: Data?, updated: Data)] = []
            // Both global files load. Clear obsolete IronMLX transport overrides in either one before adding the
            // selected model to the higher-priority file; unrelated providers and model definitions stay untouched.
            for url in [json, jsonc] where url == target || fm.fileExists(atPath: url.path) {
                let old = fm.fileExists(atPath: url.path) ? try Data(contentsOf: url) : nil
                var document = try OpenCodeJSONC(data: old ?? Data("{}\n".utf8))
                if let legacy = try document.object["provider"] as? [String: Any], legacy["ironmlx"] != nil {
                    result.code = "legacy_configuration"
                    return result
                }
                let providers = try document.dictionary(at: ["providers"])
                guard url == target || providers["ironmlx"] != nil else { continue }
                let provider = try document.dictionary(at: ["providers", "ironmlx"])
                _ = try document.dictionary(at: ["providers", "ironmlx", "models"])
                for key in ["canonical", "env", "headers", "compatibility"] {
                    try document.remove(at: ["providers", "ironmlx", key])
                }
                try document.set(provider["name"] as? String ?? "IronMLX", at: ["providers", "ironmlx", "name"])
                try document.set(Self.providerPackage, at: ["providers", "ironmlx", "package"])
                try document.set(["baseURL": "http://127.0.0.1:\(port)/v1", "apiKey": "", "provider": "ironmlx"],
                                 at: ["providers", "ironmlx", "settings"])
                try document.set(["chat_template_kwargs": ["enable_thinking": false]],
                                 at: ["providers", "ironmlx", "body"])
                let previous = try document.dictionary(at: ["providers", "ironmlx", "models", model.id])
                var definition: [String: Any] = [
                    "name": previous["name"] as? String ?? model.id,
                    "capabilities": ["tools": true, "input": model.supportsVision ? ["text", "image"] : ["text"],
                                     "output": ["text"]],
                    "limit": ["context": model.contextWindow, "output": min(4096, model.maxTokens)],
                ]
                if let reasoning = model.reasoning, !reasoning.nativeEfforts.isEmpty {
                    var capabilities = definition["capabilities"] as! [String: Any]
                    capabilities["reasoning"] = ["supported": true, "effort": reasoning.nativeEfforts,
                                                  "toggle": reasoning.supportsDisable] as [String: Any]
                    definition["capabilities"] = capabilities
                    definition["compatibility"] = ["reasoningField": "reasoning_content", "requireReasoning": true] as [String: Any]
                    var variants: [[String: Any]] = reasoning.nativeEfforts.map { effort in
                        ["id": effort, "settings": ["reasoningEffort": effort],
                         "body": ["chat_template_kwargs": ["enable_thinking": true, "preserve_thinking": true]]]
                    }
                    if reasoning.supportsDisable {
                        variants.insert(["id": "none", "body": ["chat_template_kwargs": ["enable_thinking": false]]], at: 0)
                    }
                    definition["variants"] = variants
                }
                try document.set(definition, at: ["providers", "ironmlx", "models", model.id])
                let updated = Data(document.text.utf8)
                if old != updated { changes.append((url, old, updated)) }
            }
            for change in changes where change.old != nil {
                let backup = change.url.appendingPathExtension("ironmlx-backup-\(UUID().uuidString)")
                try change.old!.write(to: backup, options: .atomic)
                try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backup.path)
                result.backupPaths.append(backup.path)
            }
            try fm.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            var written: [(url: URL, old: Data?)] = []
            do {
                for change in changes {
                    try change.updated.write(to: change.url, options: .atomic)
                    written.append((change.url, change.old))
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: change.url.path)
                }
            } catch {
                for change in written.reversed() {
                    if let old = change.old { try? old.write(to: change.url, options: .atomic) }
                    else { try? fm.removeItem(at: change.url) }
                }
                throw error
            }
            result.success = true
            result.code = "configured"
            result.configurationPath = target.path
        } catch {
            result.code = "configuration_failed"
        }
        return result
    }

    private enum ConfigurationError: Error { case invalidDocument }
}

/// Edits only owned JSONC values. Unrelated bytes, including comments and trailing commas, are retained.
private struct OpenCodeJSONC {
    enum Error: Swift.Error { case invalidDocument }
    var text: String
    var object: [String: Any] {
        get throws {
            guard let object = try JSONSerialization.jsonObject(with: Data(text.utf8), options: .json5Allowed) as? [String: Any] else {
                throw Error.invalidDocument
            }
            return object
        }
    }

    init(data: Data) throws {
        guard let text = String(data: data, encoding: .utf8) else { throw Error.invalidDocument }
        self.text = text
        _ = try object
    }

    func dictionary(at path: [String]) throws -> [String: Any] {
        var value: Any = try object
        for key in path {
            guard let dictionary = value as? [String: Any] else { throw Error.invalidDocument }
            guard let child = dictionary[key] else { return [:] }
            value = child
        }
        guard let dictionary = value as? [String: Any] else { throw Error.invalidDocument }
        return dictionary
    }

    mutating func set(_ value: Any, at path: [String]) throws {
        try edit(path: path, value: value)
    }

    mutating func remove(at path: [String]) throws {
        try edit(path: path, value: nil)
    }

    private mutating func edit(path: [String], value: Any?) throws {
        var bytes = Array(text.utf8)
        var start = 0
        for (depth, key) in path.enumerated() {
            let object = try Self.members(bytes, start: start)
            let matches = object.entries.filter { $0.key == key }
            guard matches.count <= 1 else { throw Error.invalidDocument }
            if let match = matches.first {
                if depth < path.count - 1 { start = match.value.lowerBound; continue }
                if let value {
                    let data = try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
                    bytes.replaceSubrange(match.value, with: data)
                } else {
                    var end = match.value.upperBound
                    try Self.skipTrivia(bytes, index: &end)
                    if bytes[end] == 44 { end += 1 }
                    else { end = match.value.upperBound }
                    bytes.removeSubrange(match.start..<end)
                }
                text = String(decoding: bytes, as: UTF8.self)
                _ = try self.object
                return
            }
            guard var nested = value else { return }
            for child in path.dropFirst(depth + 1).reversed() { nested = [child: nested] }
            let encodedKey = try JSONSerialization.data(withJSONObject: key, options: .fragmentsAllowed)
            let encodedValue = try JSONSerialization.data(withJSONObject: nested, options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
            var insertion: [UInt8] = []
            if let last = object.entries.last {
                var after = last.value.upperBound
                try Self.skipTrivia(bytes, index: &after)
                if bytes[after] != 44 { insertion.append(44) }
                // Insert before trailing comments so a line comment cannot swallow the separator.
                start = last.value.upperBound
                if bytes[after] == 44 { start = after + 1 }
            } else { start = object.open + 1 }
            insertion += Array("\n".utf8) + encodedKey + Array(": ".utf8) + encodedValue + Array("\n".utf8)
            bytes.insert(contentsOf: insertion, at: start)
            text = String(decoding: bytes, as: UTF8.self)
            _ = try self.object
            return
        }
    }

    private struct Member {
        let key: String
        let start: Int
        let value: Range<Int>
    }

    private static func members(_ bytes: [UInt8], start: Int) throws -> (open: Int, entries: [Member]) {
        var index = start
        try skipTrivia(bytes, index: &index)
        guard index < bytes.count, bytes[index] == 123 else { throw Error.invalidDocument }
        let open = index
        index += 1
        var entries: [Member] = []
        while true {
            try skipTrivia(bytes, index: &index)
            guard index < bytes.count else { throw Error.invalidDocument }
            if bytes[index] == 125 { return (open, entries) }
            let keyStart = index
            guard bytes[index] == 34 else { throw Error.invalidDocument }
            try skipValue(bytes, index: &index)
            guard let key = try JSONSerialization.jsonObject(with: Data(bytes[keyStart..<index]), options: .fragmentsAllowed) as? String else {
                throw Error.invalidDocument
            }
            try skipTrivia(bytes, index: &index)
            guard index < bytes.count, bytes[index] == 58 else { throw Error.invalidDocument }
            index += 1
            try skipTrivia(bytes, index: &index)
            let valueStart = index
            try skipValue(bytes, index: &index)
            entries.append(Member(key: key, start: keyStart, value: valueStart..<index))
            try skipTrivia(bytes, index: &index)
            guard index < bytes.count else { throw Error.invalidDocument }
            if bytes[index] == 44 { index += 1 }
            else if bytes[index] != 125 { throw Error.invalidDocument }
        }
    }

    private static func skipValue(_ bytes: [UInt8], index: inout Int) throws {
        guard index < bytes.count else { throw Error.invalidDocument }
        if bytes[index] == 34 || bytes[index] == 39 {
            let quote = bytes[index]
            index += 1
            while index < bytes.count {
                if bytes[index] == 92 { index += 2; continue }
                if bytes[index] == quote { index += 1; return }
                index += 1
            }
            throw Error.invalidDocument
        }
        if bytes[index] == 123 || bytes[index] == 91 {
            let close: UInt8 = bytes[index] == 123 ? 125 : 93
            index += 1
            while index < bytes.count {
                try skipTrivia(bytes, index: &index)
                guard index < bytes.count else { throw Error.invalidDocument }
                if bytes[index] == close { index += 1; return }
                if bytes[index] == 44 || bytes[index] == 58 { index += 1 }
                else { try skipValue(bytes, index: &index) }
            }
            throw Error.invalidDocument
        }
        let start = index
        while index < bytes.count, ![UInt8(44), 125, 93, 58, 32, 9, 10, 13, 47].contains(bytes[index]) { index += 1 }
        guard index > start else { throw Error.invalidDocument }
    }

    private static func skipTrivia(_ bytes: [UInt8], index: inout Int) throws {
        while index < bytes.count {
            if [UInt8(32), 9, 10, 13].contains(bytes[index]) { index += 1; continue }
            guard bytes[index] == 47, index + 1 < bytes.count else { return }
            if bytes[index + 1] == 47 {
                index += 2
                while index < bytes.count, bytes[index] != 10 { index += 1 }
            } else if bytes[index + 1] == 42 {
                index += 2
                while index + 1 < bytes.count, !(bytes[index] == 42 && bytes[index + 1] == 47) { index += 1 }
                guard index + 1 < bytes.count else { throw Error.invalidDocument }
                index += 2
            } else { return }
        }
    }
}
