import Darwin
import Foundation
import Yams

/// Configures the default oh-my-pi profile without changing model roles or launching an agent.
@MainActor
struct OMPIntegrationService {
    nonisolated static let minimumVersion = "18.1.16"
    nonisolated static let installationURL = "https://omp.sh/docs/quickstart"

    struct Result: Codable {
        var success = false
        var code: String
        var installedVersion: String?
        var minimumVersion = OMPIntegrationService.minimumVersion
        var configurationPath: String?
        var backupPaths: [String] = []
    }

    struct Installation {
        let result: Result
        let executableURL: URL?
    }

    struct Model {
        let id: String
        let contextWindow: Int
        let maxTokens: Int
    }

    var agentHome = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".omp/agent")
    var findExecutable: () async -> URL? = { await OMPIntegrationService.locateExecutable() }
    var versionOutput: (URL) async throws -> String = {
        try await OMPIntegrationService.commandOutput($0, arguments: ["--version"])
    }

    func inspect() async -> Installation {
        guard let executable = await findExecutable() else {
            return Installation(result: Result(code: "not_installed"), executableURL: nil)
        }
        guard let output = try? await versionOutput(executable),
              let version = Self.parseVersion(output) else {
            return Installation(result: Result(code: "version_unknown"), executableURL: executable)
        }
        let supported = version.compare(Self.minimumVersion, options: .numeric) != .orderedAscending
        return Installation(result: Result(success: supported, code: supported ? "ready" : "version_too_old",
                                           installedVersion: version), executableURL: executable)
    }

    nonisolated static func parseVersion(_ output: String) -> String? {
        let value = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.range(of: #"^omp/[0-9]+\.[0-9]+\.[0-9]+$"#, options: .regularExpression) != nil else { return nil }
        return String(value.dropFirst(4))
    }

    /// Called after installation and live model checks. Rechecks the CLI before touching files.
    func configure(model: Model, port: UInt16) async -> Result {
        var result = await inspect().result
        guard result.success else { return result }
        result.success = false
        guard !model.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              model.id != "MODEL_ID", model.contextWindow > 0, model.maxTokens > 0,
              model.maxTokens <= model.contextWindow, port > 0 else {
            result.code = "model_unavailable"
            return result
        }
        let fm = FileManager.default
        let primary = agentHome.appendingPathComponent("models.yml")
        let alternative = agentHome.appendingPathComponent("models.yaml")
        let url = fm.fileExists(atPath: primary.path) ? primary
            : fm.fileExists(atPath: alternative.path) ? alternative : primary
        guard fm.fileExists(atPath: url.path)
                || !fm.fileExists(atPath: agentHome.appendingPathComponent("models.json").path) else {
            result.code = "initialize_cli"
            return result
        }
        do {
            let old = fm.fileExists(atPath: url.path) ? try Data(contentsOf: url) : nil
            let data = try mergedConfiguration(old, model: model, port: port)
            if old != data {
                if let old {
                    let backup = url.appendingPathExtension("ironmlx-backup-\(UUID().uuidString)")
                    try old.write(to: backup, options: .atomic)
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backup.path)
                    result.backupPaths.append(backup.path)
                }
                try fm.createDirectory(at: agentHome, withIntermediateDirectories: true,
                                       attributes: [.posixPermissions: 0o700])
                do {
                    try data.write(to: url, options: .atomic)
                    try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                } catch {
                    if let old {
                        try old.write(to: url, options: .atomic)
                        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
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
    private enum CommandError: Error { case timedOut, failed(Int32), outputTooLarge, invalidOutput }

    private func mergedConfiguration(_ data: Data?, model: Model, port: UInt16) throws -> Data {
        var loaded: Any?
        if let data {
            guard let text = String(data: data, encoding: .utf8) else { throw ConfigurationError.invalidDocument }
            loaded = try Yams.load(yaml: text)
        }
        guard loaded == nil || loaded is [String: Any] else { throw ConfigurationError.invalidDocument }
        var document = loaded as? [String: Any] ?? [:]
        guard document.keys.allSatisfy({ $0 == "providers" }) else { throw ConfigurationError.invalidDocument }
        if let value = document["providers"], !(value is [String: Any]) { throw ConfigurationError.invalidDocument }
        var providers = document["providers"] as? [String: Any] ?? [:]
        if let value = providers["ironmlx"], !(value is [String: Any]) { throw ConfigurationError.invalidDocument }
        var provider = providers["ironmlx"] as? [String: Any] ?? [:]
        if let value = provider["modelOverrides"], !(value is [String: Any]) { throw ConfigurationError.invalidDocument }
        var overrides = provider["modelOverrides"] as? [String: Any] ?? [:]
        if let value = overrides[model.id], !(value is [String: Any]) { throw ConfigurationError.invalidDocument }
        var selected = overrides[model.id] as? [String: Any] ?? [:]
        selected.merge(["contextWindow": model.contextWindow, "maxTokens": min(4096, model.maxTokens),
                        "reasoning": false]) { _, new in new }
        overrides[model.id] = selected
        provider.merge(["baseUrl": "http://127.0.0.1:\(port)/v1", "auth": "none", "api": "openai-responses",
                        "discovery": ["type": "openai-models-list"], "modelOverrides": overrides]) { _, new in new }
        providers["ironmlx"] = provider
        document["providers"] = providers
        if let data, let previous = loaded as? NSDictionary, previous.isEqual(to: document) { return data }
        return Data(try Yams.dump(object: document, sortKeys: true).utf8)
    }

    nonisolated static func locateExecutable() async -> URL? {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let paths = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for directory in paths + ["/opt/homebrew/bin", "/usr/local/bin", home.appendingPathComponent(".local/bin").path,
                                   home.appendingPathComponent(".bun/bin").path, home.appendingPathComponent(".npm-global/bin").path] {
            guard directory.hasPrefix("/") else { continue }
            let candidate = URL(fileURLWithPath: directory).appendingPathComponent("omp")
            if fm.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        // Finder-launched apps do not inherit interactive shell PATH additions.
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"
        guard ["/bin/zsh", "/bin/bash"].contains(shell),
              let path = try? await commandOutput(URL(fileURLWithPath: shell), arguments: ["-lc", "command -v omp"])
                .trimmingCharacters(in: .whitespacesAndNewlines),
              path.hasPrefix("/"), !path.contains("\n"), fm.isExecutableFile(atPath: path) else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// Bounded, off-main-thread probes. Files avoid pipe-buffer deadlocks from CLI startup output.
    nonisolated static func commandOutput(_ executable: URL, arguments: [String], timeout: TimeInterval = 5,
                                           environment: [String: String]? = nil,
                                           maximumOutputBytes: Int = 16384) async throws -> String {
        try await Task.detached(priority: .utility) {
            let fm = FileManager.default
            let directory = fm.temporaryDirectory.appendingPathComponent("ironmlx-omp-\(UUID().uuidString)")
            try fm.createDirectory(at: directory, withIntermediateDirectories: false,
                                   attributes: [.posixPermissions: 0o700])
            defer { try? fm.removeItem(at: directory) }
            let outputURL = directory.appendingPathComponent("stdout")
            fm.createFile(atPath: outputURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
            let output = try FileHandle(forWritingTo: outputURL)
            defer { try? output.close() }
            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            process.environment = environment
            process.currentDirectoryURL = fm.homeDirectoryForCurrentUser
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = output
            process.standardError = FileHandle.nullDevice
            try process.run()
            let deadline = Date().addingTimeInterval(timeout)
            while process.isRunning && Date() < deadline { usleep(20_000) }
            if process.isRunning {
                process.terminate()
                let grace = Date().addingTimeInterval(0.25)
                while process.isRunning && Date() < grace { usleep(20_000) }
                if process.isRunning { kill(process.processIdentifier, SIGKILL) }
                process.waitUntilExit()
                throw CommandError.timedOut
            }
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { throw CommandError.failed(process.terminationStatus) }
            guard let size = try fm.attributesOfItem(atPath: outputURL.path)[.size] as? NSNumber,
                  size.intValue <= maximumOutputBytes else { throw CommandError.outputTooLarge }
            guard let text = String(data: try Data(contentsOf: outputURL), encoding: .utf8) else {
                throw CommandError.invalidOutput
            }
            return text
        }.value
    }
}
