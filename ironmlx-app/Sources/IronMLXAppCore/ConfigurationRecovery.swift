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

    public enum Reason: Equatable, Sendable {
        case corruption
        case migrationFailed(from: Int, to: Int)
        case unsupportedVersion(found: Int, supported: Int)
    }

    public let kind: Kind
    public let sourceURL: URL
    public let preservedURL: URL?
    public let lkgURL: URL
    public let lkgErrorDescription: String?
    public let reason: Reason
    public let errorDescription: String
    public let preservationErrorDescription: String?

    public var hasValidLKG: Bool {
        lkgErrorDescription == nil
    }

    public var dashboardErrorCode: String {
        switch reason {
        case .corruption:
            "configuration_recovery_required"
        case .migrationFailed:
            "configuration_migration_failed"
        case .unsupportedVersion:
            "configuration_version_unsupported"
        }
    }
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
        fileManager: FileManager,
        lkgURL: URL,
        lkgErrorDescription: String?,
        reason: ConfigurationRecoveryIssue.Reason = .corruption
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
                try fileManager.setAttributes(
                    [.posixPermissions: 0o700],
                    ofItemAtPath: recoveryDirectory.path
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
            lkgURL: lkgURL,
            lkgErrorDescription: lkgErrorDescription,
            reason: reason,
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
    case unsupportedVersion(URL)

    public var errorDescription: String? {
        switch self {
        case let .preservedCopyMissing(url):
            "Refusing to reset configuration because its preserved copy is missing: \(url.path)"
        case let .unsupportedVersion(url):
            "Refusing to replace configuration from a newer IronMLX version: \(url.path)"
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

    public func restoreAffectedConfigurationsFromLKG() throws {
        if appConfigStore.recoveryIssue != nil {
            try appConfigStore.restoreFromLKG()
        }
        if modelParameterStore.recoveryIssue != nil {
            try modelParameterStore.restoreFromLKG()
        }
    }

    public func presentRecovery(_ sender: Any?) {
        let currentIssues = issues
        guard !currentIssues.isEmpty else {
            return
        }

        let text = ConfigurationRecoveryText(language: appConfigStore.load().language)

        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .critical
        let hasUnsupportedVersion = currentIssues.contains {
            if case .unsupportedVersion = $0.reason {
                return true
            }
            return false
        }
        alert.messageText = hasUnsupportedVersion ? text.unsupportedTitle : text.recoveryTitle
        alert.informativeText = recoveryMessage(for: currentIssues, text: text)
        if hasUnsupportedVersion {
            alert.addButton(withTitle: text.showFiles)
            alert.addButton(withTitle: text.cancel)
            alert.addButton(withTitle: text.quit)
            switch alert.runModal() {
            case .alertFirstButtonReturn:
                showFiles(for: currentIssues)
            case .alertThirdButtonReturn:
                NSApp.terminate(nil)
            default:
                break
            }
            return
        }

        alert.addButton(withTitle: text.restoreLKG)
        alert.addButton(withTitle: text.reset)
        alert.addButton(withTitle: text.showFiles)
        alert.addButton(withTitle: text.cancel)
        alert.buttons[0].isEnabled = currentIssues.allSatisfy(\.hasValidLKG)
        alert.buttons[1].isEnabled = currentIssues.allSatisfy { $0.preservedURL != nil }

        switch alert.runModal() {
        case .alertFirstButtonReturn:
            do {
                try restoreAffectedConfigurationsFromLKG()
            } catch {
                showOperationFailure(error, text: text)
            }
        case .alertSecondButtonReturn:
            do {
                try resetAffectedConfigurations()
            } catch {
                showOperationFailure(error, text: text)
            }
        case .alertThirdButtonReturn:
            showFiles(for: currentIssues)
        default:
            break
        }
    }

    func recoveryMessage(
        for issues: [ConfigurationRecoveryIssue],
        text: ConfigurationRecoveryText
    ) -> String {
        var lines = [
            text.blockedIntroduction,
            "",
        ]
        for issue in issues {
            lines.append("\(text.displayName(for: issue.kind)): \(issue.sourceURL.path)")
            switch issue.reason {
            case .corruption:
                lines.append(text.corruptionReason)
            case let .migrationFailed(from, to):
                lines.append(text.migrationReason(from: from, to: to))
            case let .unsupportedVersion(found, supported):
                lines.append(text.versionReason(found: found, supported: supported))
            }
            lines.append("\(text.errorLabel): \(issue.errorDescription)")
            if let preservedURL = issue.preservedURL {
                lines.append("\(text.preservedLabel): \(preservedURL.path)")
            } else {
                lines.append(text.originalUntouched)
            }
            if let preservationError = issue.preservationErrorDescription {
                lines.append("\(text.preservationFailed): \(preservationError)")
            }
            if let lkgError = issue.lkgErrorDescription {
                lines.append("\(text.lkgUnavailable): \(lkgError)")
            } else {
                lines.append("\(text.lkgLabel): \(issue.lkgURL.path)")
            }
            lines.append("")
        }
        lines.append(text.resetBoundary)
        return lines.joined(separator: "\n")
    }

    private func showFiles(for issues: [ConfigurationRecoveryIssue]) {
        var urls: [URL] = []
        for issue in issues {
            urls.append(issue.sourceURL)
            if let preservedURL = issue.preservedURL {
                urls.append(preservedURL)
            }
            if fileManager.fileExists(atPath: issue.lkgURL.path) {
                urls.append(issue.lkgURL)
            }
        }
        let grouped = Dictionary(grouping: urls, by: { $0.deletingLastPathComponent() })
        for selected in grouped.values {
            workspace.activateFileViewerSelecting(selected)
        }
    }

    private var fileManager: FileManager {
        .default
    }

    private func showOperationFailure(_ error: Error, text: ConfigurationRecoveryText) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = text.operationFailed
        alert.informativeText = error.localizedDescription
        alert.addButton(withTitle: text.close)
        alert.runModal()
    }
}

struct ConfigurationRecoveryText: Equatable {
    let language: String

    init(language: String) {
        switch language {
        case "zh", "zh-Hans":
            self.language = "zh-Hans"
        case "zh-Hant", "ja", "ko":
            self.language = language
        default:
            self.language = "en"
        }
    }

    var recoveryTitle: String { value(en: "IronMLX configuration needs recovery", zhHans: "IronMLX 配置需要恢复", zhHant: "IronMLX 設定需要恢復", ja: "IronMLX の設定を復旧する必要があります", ko: "IronMLX 설정 복구가 필요합니다") }
    var unsupportedTitle: String { value(en: "IronMLX configuration is from a newer version", zhHans: "IronMLX 配置来自更高版本", zhHant: "IronMLX 設定來自較新版本", ja: "IronMLX の設定は新しいバージョンで作成されています", ko: "IronMLX 설정이 더 최신 버전에서 생성되었습니다") }
    var restoreLKG: String { value(en: "Restore Last-Known-Good", zhHans: "从最后有效配置恢复", zhHant: "從最後有效設定恢復", ja: "最後の有効な設定から復旧", ko: "마지막 유효 설정에서 복구") }
    var reset: String { value(en: "Reset Affected Configuration", zhHans: "重置受影响配置", zhHant: "重設受影響設定", ja: "影響を受けた設定をリセット", ko: "영향받은 설정 재설정") }
    var showFiles: String { value(en: "Show Original, Preserved Copy, and LKG", zhHans: "显示原文件、保留副本和 LKG", zhHant: "顯示原始檔、保留副本和 LKG", ja: "元ファイル、保存コピー、LKG を表示", ko: "원본, 보존 사본 및 LKG 보기") }
    var cancel: String { value(en: "Cancel", zhHans: "取消", zhHant: "取消", ja: "キャンセル", ko: "취소") }
    var quit: String { value(en: "Quit IronMLX", zhHans: "退出 IronMLX", zhHant: "結束 IronMLX", ja: "IronMLX を終了", ko: "IronMLX 종료") }
    var close: String { value(en: "Close", zhHans: "关闭", zhHant: "關閉", ja: "閉じる", ko: "닫기") }
    var operationFailed: String { value(en: "Configuration recovery failed", zhHans: "配置恢复失败", zhHant: "設定恢復失敗", ja: "設定の復旧に失敗しました", ko: "설정 복구에 실패했습니다") }
    var blockedIntroduction: String { value(en: "Configuration writes are blocked until the following issue is explicitly resolved.", zhHans: "在明确解决以下问题前，配置写入已被阻止。", zhHant: "在明確解決以下問題前，設定寫入已被阻止。", ja: "次の問題を明示的に解決するまで、設定への書き込みはブロックされます。", ko: "다음 문제를 명시적으로 해결할 때까지 설정 쓰기가 차단됩니다.") }
    var corruptionReason: String { value(en: "The active configuration is damaged or invalid.", zhHans: "活动配置已损坏或无效。", zhHant: "使用中的設定已損壞或無效。", ja: "現在の設定が破損しているか無効です。", ko: "현재 설정이 손상되었거나 유효하지 않습니다.") }
    var errorLabel: String { value(en: "Error", zhHans: "错误", zhHant: "錯誤", ja: "エラー", ko: "오류") }
    var preservedLabel: String { value(en: "Preserved copy", zhHans: "保留副本", zhHant: "保留副本", ja: "保存済みコピー", ko: "보존된 사본") }
    var originalUntouched: String { value(en: "The original file remains untouched at its current path.", zhHans: "原文件仍保留在当前路径且未被修改。", zhHant: "原始檔案仍保留在目前路徑且未被修改。", ja: "元のファイルは現在の場所に変更せず保持されています。", ko: "원본 파일은 현재 경로에 변경 없이 유지됩니다.") }
    var preservationFailed: String { value(en: "Could not preserve an additional copy", zhHans: "无法创建额外保留副本", zhHant: "無法建立額外保留副本", ja: "追加の保存コピーを作成できませんでした", ko: "추가 보존 사본을 만들 수 없습니다") }
    var lkgUnavailable: String { value(en: "Last-known-good configuration is unavailable", zhHans: "最后有效配置不可用", zhHant: "最後有效設定無法使用", ja: "最後の有効な設定を利用できません", ko: "마지막 유효 설정을 사용할 수 없습니다") }
    var lkgLabel: String { value(en: "Last-known-good", zhHans: "最后有效配置", zhHant: "最後有效設定", ja: "最後の有効な設定", ko: "마지막 유효 설정") }
    var resetBoundary: String { value(en: "Restoring or resetting replaces only affected active files. Evidence files are retained.", zhHans: "恢复或重置只替换受影响的活动文件，所有证据文件均会保留。", zhHant: "恢復或重設只會取代受影響的使用中檔案，所有證據檔案都會保留。", ja: "復旧またはリセットでは影響を受けた現在のファイルだけを置き換え、証拠ファイルは保持します。", ko: "복구 또는 재설정은 영향받은 현재 파일만 교체하며 증거 파일은 유지합니다.") }

    func migrationReason(from: Int, to: Int) -> String {
        value(en: "Configuration migration from v\(from) to v\(to) failed.", zhHans: "配置从 v\(from) 迁移到 v\(to) 失败。", zhHant: "設定從 v\(from) 移轉到 v\(to) 失敗。", ja: "設定の v\(from) から v\(to) への移行に失敗しました。", ko: "설정을 v\(from)에서 v\(to)로 마이그레이션하지 못했습니다.")
    }

    func versionReason(found: Int, supported: Int) -> String {
        value(en: "This file uses schema v\(found), but this IronMLX supports up to v\(supported). Update IronMLX before continuing.", zhHans: "此文件使用 schema v\(found)，当前 IronMLX 最高支持 v\(supported)。请先更新 IronMLX。", zhHant: "此檔案使用 schema v\(found)，目前 IronMLX 最高支援 v\(supported)。請先更新 IronMLX。", ja: "このファイルは schema v\(found) を使用していますが、この IronMLX は v\(supported) まで対応しています。先に IronMLX を更新してください。", ko: "이 파일은 schema v\(found)을 사용하지만 현재 IronMLX는 v\(supported)까지 지원합니다. 먼저 IronMLX를 업데이트하세요.")
    }

    func displayName(for kind: ConfigurationRecoveryIssue.Kind) -> String {
        switch kind {
        case .appConfig:
            value(en: "Application configuration", zhHans: "应用配置", zhHant: "應用程式設定", ja: "アプリケーション設定", ko: "애플리케이션 설정")
        case .modelParameters:
            value(en: "Model parameters", zhHans: "模型参数", zhHant: "模型參數", ja: "モデルパラメータ", ko: "모델 매개변수")
        }
    }

    private func value(en: String, zhHans: String, zhHant: String, ja: String, ko: String) -> String {
        switch language {
        case "zh-Hans": zhHans
        case "zh-Hant": zhHant
        case "ja": ja
        case "ko": ko
        default: en
        }
    }
}
