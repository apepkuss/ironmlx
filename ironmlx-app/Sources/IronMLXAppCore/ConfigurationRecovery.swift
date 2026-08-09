import AppKit
import CryptoKit
import Foundation

public struct ConfigurationRecoveryIssue: Equatable, Sendable {
    public enum Kind: String, Equatable, Sendable {
        case appConfig
        case modelParameters

        var displayName: String {
            switch self {
            case .appConfig:
                "Application configuration"
            case .modelParameters:
                "Model parameters"
            }
        }
    }

    public let kind: Kind
    public let sourceURL: URL
    public let preservedURL: URL?
    public let errorDescription: String
    public let preservationErrorDescription: String?
}

final class ConfigurationRecoveryState: @unchecked Sendable {
    private let lock = NSLock()
    private var storedIssue: ConfigurationRecoveryIssue?

    var issue: ConfigurationRecoveryIssue? {
        lock.lock()
        defer { lock.unlock() }
        return storedIssue
    }

    @discardableResult
    func recordIfNeeded(_ makeIssue: () -> ConfigurationRecoveryIssue) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard storedIssue == nil else {
            return false
        }
        storedIssue = makeIssue()
        return true
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        storedIssue = nil
    }
}

enum ConfigurationCorruptionPreserver {
    static func makeIssue(
        kind: ConfigurationRecoveryIssue.Kind,
        sourceURL: URL,
        data: Data?,
        error: Error,
        fileManager: FileManager
    ) -> ConfigurationRecoveryIssue {
        var preservedURL: URL?
        var preservationErrorDescription: String?

        if let data {
            do {
                let recoveryDirectory = sourceURL.deletingLastPathComponent()
                    .appendingPathComponent("recovery", isDirectory: true)
                try fileManager.createDirectory(
                    at: recoveryDirectory,
                    withIntermediateDirectories: true
                )
                let digest = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }
                    .joined()
                let sourceName = sourceURL.deletingPathExtension().lastPathComponent
                let sourceExtension = sourceURL.pathExtension
                let suffix = sourceExtension.isEmpty ? "" : ".\(sourceExtension)"
                let candidate = recoveryDirectory.appendingPathComponent(
                    "\(sourceName).corrupt-\(digest)\(suffix)"
                )
                if fileManager.fileExists(atPath: candidate.path) {
                    guard try Data(contentsOf: candidate) == data else {
                        throw ConfigurationPreservationError.contentMismatch(candidate)
                    }
                } else {
                    try data.write(to: candidate, options: .withoutOverwriting)
                    try fileManager.setAttributes(
                        [.posixPermissions: 0o400],
                        ofItemAtPath: candidate.path
                    )
                }
                preservedURL = candidate
            } catch {
                preservationErrorDescription = error.localizedDescription
            }
        }

        return ConfigurationRecoveryIssue(
            kind: kind,
            sourceURL: sourceURL,
            preservedURL: preservedURL,
            errorDescription: error.localizedDescription,
            preservationErrorDescription: preservationErrorDescription
        )
    }
}

enum ConfigurationPreservationError: LocalizedError {
    case contentMismatch(URL)

    var errorDescription: String? {
        switch self {
        case let .contentMismatch(url):
            "Existing preserved configuration does not match its content hash: \(url.path)"
        }
    }
}

public enum ConfigurationRecoveryResetError: LocalizedError, Equatable {
    case preservedCopyMissing(URL)

    public var errorDescription: String? {
        switch self {
        case let .preservedCopyMissing(url):
            "Refusing to reset configuration because its preserved copy is missing: \(url.path)"
        }
    }
}

@MainActor
public protocol ConfigurationRecoveryManaging: AnyObject {
    var hasIssues: Bool { get }

    func inspect()
    func presentRecovery(_ sender: Any?)
}

@MainActor
public final class DisabledConfigurationRecoveryManager: ConfigurationRecoveryManaging {
    public let hasIssues = false

    public init() {}

    public func inspect() {}
    public func presentRecovery(_ sender: Any?) {}
}

@MainActor
public final class ConfigurationRecoveryManager: ConfigurationRecoveryManaging {
    private let appConfigStore: AppConfigStore
    private let modelParameterStore: ModelParameterStore
    private let workspace: NSWorkspace

    public var issues: [ConfigurationRecoveryIssue] {
        [appConfigStore.recoveryIssue, modelParameterStore.recoveryIssue].compactMap { $0 }
    }

    public var hasIssues: Bool {
        !issues.isEmpty
    }

    public init(
        appConfigStore: AppConfigStore,
        modelParameterStore: ModelParameterStore,
        workspace: NSWorkspace = .shared
    ) {
        self.appConfigStore = appConfigStore
        self.modelParameterStore = modelParameterStore
        self.workspace = workspace
    }

    public func inspect() {
        _ = appConfigStore.load()
        _ = try? modelParameterStore.loadAll()
    }

    public func resetAffectedConfigurations() throws {
        if appConfigStore.recoveryIssue != nil {
            try appConfigStore.resetAfterCorruption()
        }
        if modelParameterStore.recoveryIssue != nil {
            try modelParameterStore.resetAfterCorruption()
        }
    }

    public func presentRecovery(_ sender: Any?) {
        let currentIssues = issues
        guard !currentIssues.isEmpty else {
            return
        }

        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "IronMLX configuration needs recovery"
        alert.informativeText = recoveryMessage(for: currentIssues)
        alert.addButton(withTitle: "Reset Affected Configuration")
        alert.addButton(withTitle: "Show Preserved Files")
        alert.addButton(withTitle: "Cancel")
        alert.buttons[0].isEnabled = currentIssues.allSatisfy { $0.preservedURL != nil }

        switch alert.runModal() {
        case .alertFirstButtonReturn:
            do {
                try resetAffectedConfigurations()
            } catch {
                showResetFailure(error)
            }
        case .alertSecondButtonReturn:
            for issue in currentIssues {
                workspace.activateFileViewerSelecting([issue.preservedURL ?? issue.sourceURL])
            }
        default:
            break
        }
    }

    private func recoveryMessage(for issues: [ConfigurationRecoveryIssue]) -> String {
        var lines = [
            "IronMLX could not read the following configuration. Normal writes are blocked until you explicitly recover it.",
            "",
        ]
        for issue in issues {
            lines.append("\(issue.kind.displayName): \(issue.sourceURL.path)")
            lines.append("Error: \(issue.errorDescription)")
            if let preservedURL = issue.preservedURL {
                lines.append("Preserved copy: \(preservedURL.path)")
            } else {
                lines.append("The original file remains untouched at its current path.")
            }
            if let preservationError = issue.preservationErrorDescription {
                lines.append("Could not create an additional preserved copy: \(preservationError)")
            }
            lines.append("")
        }
        lines.append("Resetting replaces only the affected active file. Any preserved copy remains available for inspection.")
        return lines.joined(separator: "\n")
    }

    private func showResetFailure(_ error: Error) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "Configuration reset failed"
        alert.informativeText = error.localizedDescription
        alert.addButton(withTitle: "Close")
        alert.runModal()
    }
}
