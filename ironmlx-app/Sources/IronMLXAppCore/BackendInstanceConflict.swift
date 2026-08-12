import AppKit
import Foundation

public enum BackendRuntimeFailureCode: String, Codable, Equatable, Sendable {
    case instanceAlreadyRunning = "ironmlx_instance_already_running"

    public static func detect(in logText: String) -> Self? {
        logText.contains(Self.instanceAlreadyRunning.rawValue) ? .instanceAlreadyRunning : nil
    }
}

public struct BackendInstanceConflictText: Equatable, Sendable {
    public var title: String
    public var message: String
    public var dismiss: String

    public init(title: String, message: String, dismiss: String) {
        self.title = title
        self.message = message
        self.dismiss = dismiss
    }
}

public enum BackendInstanceConflictPresentation {
    public nonisolated static func text(language: String) -> BackendInstanceConflictText {
        switch normalizedLanguage(language) {
        case "zh":
            BackendInstanceConflictText(
                title: "已有 IronMLX 实例正在运行",
                message: "同一 macOS 用户只能运行一个 IronMLX 后端。请先退出现有实例，然后重试。",
                dismiss: "确定"
            )
        case "zh-Hant":
            BackendInstanceConflictText(
                title: "已有 IronMLX 執行個體正在執行",
                message: "同一 macOS 使用者只能執行一個 IronMLX 後端。請先結束現有執行個體，然後重試。",
                dismiss: "確定"
            )
        case "ja":
            BackendInstanceConflictText(
                title: "IronMLX はすでに実行中です",
                message: "同じ macOS ユーザーが実行できる IronMLX バックエンドは1つだけです。既存のインスタンスを終了してから再試行してください。",
                dismiss: "OK"
            )
        case "ko":
            BackendInstanceConflictText(
                title: "IronMLX가 이미 실행 중입니다",
                message: "동일한 macOS 사용자는 IronMLX 백엔드를 하나만 실행할 수 있습니다. 기존 인스턴스를 종료한 후 다시 시도하십시오.",
                dismiss: "확인"
            )
        default:
            BackendInstanceConflictText(
                title: "IronMLX Is Already Running",
                message: "Only one IronMLX backend can run for the same macOS user. Quit the existing instance, then try again.",
                dismiss: "OK"
            )
        }
    }

    @MainActor
    public static func presentAlert(language: String) {
        let text = text(language: language)
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = text.title
        alert.informativeText = text.message
        alert.addButton(withTitle: text.dismiss)
        alert.runModal()
    }

    nonisolated private static func normalizedLanguage(_ language: String) -> String {
        switch language.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "zh", "zh-Hans": "zh"
        case "zh-Hant": "zh-Hant"
        case "ja": "ja"
        case "ko": "ko"
        default: "en"
        }
    }
}
