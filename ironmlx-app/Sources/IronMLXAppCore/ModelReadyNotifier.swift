import Foundation
import UserNotifications

/// The system notification centre as the starter flow uses it; replaced in tests.
@MainActor
protocol ModelReadyNotificationPosting: AnyObject {
    /// Shows the permission prompt if the user has not decided yet.
    func requestAuthorization() async -> Bool
    func isAuthorized() async -> Bool
    func postModelReady(modelID: String, title: String, body: String) async
}

enum ModelReadyAction: Equatable {
    /// The no-model window is still open: it turns into chat.
    case openChat
    case postNotification
    /// Notifications are not allowed: the menu bar row is the only signal.
    case menuOnly
}

/// Tells the user that the starter model is ready when its window is closed.
/// Permission is asked only once the user leaves the downloading window, so
/// the prompt makes sense where it appears.
@MainActor
final class ModelReadyNotifier {
    nonisolated static let modelIDKey = "ironmlx.modelID"

    private let center: (any ModelReadyNotificationPosting)?
    private var permissionRequested = false
    /// Called with the model ID when the user clicks the notification.
    var onOpenChat: ((String) -> Void)?

    init(center: (any ModelReadyNotificationPosting)?) {
        self.center = center
    }

    /// The real centre, or nil when the process has no App bundle (tests and
    /// `swift run`), where `UNUserNotificationCenter` cannot be used.
    static func systemCenter() -> UserNotificationCenterPoster? {
        Bundle.main.bundleIdentifier == nil ? nil : UserNotificationCenterPoster()
    }

    static func readyAction(windowVisible: Bool, notificationsAllowed: Bool) -> ModelReadyAction {
        if windowVisible { return .openChat }
        return notificationsAllowed ? .postNotification : .menuOnly
    }

    /// The user chose "Continue in Background" or closed the downloading window.
    func userLeftDownloadWindow() async {
        guard let center, !permissionRequested else { return }
        permissionRequested = true
        let granted = await center.requestAuthorization()
        IronMLXAppLogger.info("Model ready notifications \(granted ? "allowed" : "not allowed")")
    }

    func modelReady(modelID: String, modelName: String, windowVisible: Bool, language: String) async -> ModelReadyAction {
        var allowed = false
        if !windowVisible, let center {
            allowed = await center.isAuthorized()
        }
        let action = Self.readyAction(windowVisible: windowVisible, notificationsAllowed: allowed)
        if action == .postNotification, let center {
            let text = Self.text(modelName: modelName, language: language)
            await center.postModelReady(modelID: modelID, title: text.title, body: text.body)
        }
        return action
    }

    func handleNotificationResponse(userInfo: [AnyHashable: Any]) {
        guard let modelID = userInfo[Self.modelIDKey] as? String, !modelID.isEmpty else { return }
        onOpenChat?(modelID)
    }

    static func text(modelName: String, language: String) -> (title: String, body: String) {
        switch language {
        case "zh", "zh-Hans":
            ("模型已就绪", "\(modelName) 已下载并加载好，点按开始对话。")
        case "zh-Hant":
            ("模型已就緒", "\(modelName) 已下載並載入完成，點按開始對話。")
        case "ja":
            ("モデルの準備ができました", "\(modelName) のダウンロードと読み込みが完了しました。クリックしてチャットを始めましょう。")
        case "ko":
            ("모델이 준비되었습니다", "\(modelName) 다운로드와 로드가 끝났습니다. 클릭해서 대화를 시작하세요.")
        default:
            ("Model ready", "\(modelName) is downloaded and loaded. Click to start chatting.")
        }
    }
}

/// `UNUserNotificationCenter` behind the protocol; also its delegate, so a
/// click reaches the notifier and banners show while the App is active.
@MainActor
final class UserNotificationCenterPoster: NSObject, ModelReadyNotificationPosting, UNUserNotificationCenterDelegate {
    private let center = UNUserNotificationCenter.current()
    var onResponse: ((_ userInfo: [AnyHashable: Any]) -> Void)?

    override init() {
        super.init()
        center.delegate = self
    }

    func requestAuthorization() async -> Bool {
        do {
            return try await center.requestAuthorization(options: [.alert, .sound])
        } catch {
            IronMLXAppLogger.error("Notification permission request failed: \(error)")
            return false
        }
    }

    func isAuthorized() async -> Bool {
        let status = await center.notificationSettings().authorizationStatus
        return status == .authorized || status == .provisional
    }

    func postModelReady(modelID: String, title: String, body: String) async {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        content.userInfo = [ModelReadyNotifier.modelIDKey: modelID]
        let request = UNNotificationRequest(identifier: "ironmlx.model-ready", content: content, trigger: nil)
        do {
            try await center.add(request)
        } catch {
            IronMLXAppLogger.error("Model ready notification failed: \(error)")
        }
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let modelID = response.notification.request.content.userInfo[ModelReadyNotifier.modelIDKey] as? String
        await MainActor.run {
            self.onResponse?(modelID.map { [ModelReadyNotifier.modelIDKey: $0] } ?? [:])
        }
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }
}
