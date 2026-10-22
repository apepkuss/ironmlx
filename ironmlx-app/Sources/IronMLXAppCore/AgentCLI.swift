import Darwin
import Foundation

/// Shared, bounded CLI discovery and probes for Dashboard integrations.
enum AgentCLI {
    private enum CommandError: Error { case timedOut, failed(Int32), outputTooLarge, invalidOutput }

    nonisolated static func locateExecutable(named name: String) async -> URL? {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let paths = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for directory in paths + ["/opt/homebrew/bin", "/usr/local/bin", home.appendingPathComponent(".local/bin").path,
                                   home.appendingPathComponent(".bun/bin").path, home.appendingPathComponent(".npm-global/bin").path] {
            guard directory.hasPrefix("/") else { continue }
            let candidate = URL(fileURLWithPath: directory).appendingPathComponent(name)
            if fm.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        // Finder-launched apps do not inherit interactive shell PATH additions.
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"
        guard ["/bin/zsh", "/bin/bash"].contains(shell),
              let path = try? await commandOutput(URL(fileURLWithPath: shell), arguments: ["-lc", "command -v \"$1\"", "ironmlx-agent-probe", name])
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
            let directory = fm.temporaryDirectory.appendingPathComponent("ironmlx-agent-\(UUID().uuidString)")
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
