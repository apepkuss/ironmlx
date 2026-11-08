import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
private final class FakeNotificationCenter: ModelReadyNotificationPosting {
    var authorized: Bool
    var grantOnRequest: Bool
    private(set) var authorizationRequests = 0
    private(set) var posted: [(modelID: String, title: String, body: String)] = []

    init(authorized: Bool, grantOnRequest: Bool = true) {
        self.authorized = authorized
        self.grantOnRequest = grantOnRequest
    }

    func requestAuthorization() async -> Bool {
        authorizationRequests += 1
        authorized = authorized || grantOnRequest
        return authorized
    }

    func isAuthorized() async -> Bool { authorized }

    func postModelReady(modelID: String, title: String, body: String) async {
        posted.append((modelID, title, body))
    }
}

@MainActor
@Suite("Model ready notifier")
struct ModelReadyNotifierTests {
    private let model = "mlx-community/Qwen3.5-4B-MLX-4bit"

    @Test func readyActionDependsOnWindowThenPermission() {
        #expect(ModelReadyNotifier.readyAction(windowVisible: true, notificationsAllowed: true) == .openChat)
        #expect(ModelReadyNotifier.readyAction(windowVisible: true, notificationsAllowed: false) == .openChat)
        #expect(ModelReadyNotifier.readyAction(windowVisible: false, notificationsAllowed: true) == .postNotification)
        #expect(ModelReadyNotifier.readyAction(windowVisible: false, notificationsAllowed: false) == .menuOnly)
    }

    @Test func readyWithWindowOpenNeverPostsANotification() async {
        let center = FakeNotificationCenter(authorized: true)
        let notifier = ModelReadyNotifier(center: center)
        let action = await notifier.modelReady(modelID: model, modelName: "Qwen 3.5 4B", windowVisible: true, language: "en")
        #expect(action == .openChat)
        #expect(center.posted.isEmpty)
    }

    @Test func readyWithWindowClosedPostsWhenAllowed() async {
        let center = FakeNotificationCenter(authorized: true)
        let notifier = ModelReadyNotifier(center: center)
        let action = await notifier.modelReady(
            modelID: model, modelName: "Qwen 3.5 4B", windowVisible: false, language: "zh-Hans"
        )
        #expect(action == .postNotification)
        #expect(center.posted.map(\.modelID) == [model])
        #expect(center.posted.first?.title == "模型已就绪")
        #expect(center.posted.first?.body.contains("Qwen 3.5 4B") == true)
    }

    @Test func deniedPermissionLeavesOnlyTheMenuRow() async {
        let center = FakeNotificationCenter(authorized: false, grantOnRequest: false)
        let notifier = ModelReadyNotifier(center: center)
        await notifier.userLeftDownloadWindow()
        let action = await notifier.modelReady(modelID: model, modelName: "Qwen 3.5 4B", windowVisible: false, language: "en")
        #expect(action == .menuOnly)
        #expect(center.posted.isEmpty)
        // Without a notification centre (no App bundle) the menu row is the only signal.
        let unbundled = ModelReadyNotifier(center: nil)
        #expect(await unbundled.modelReady(modelID: model, modelName: "Qwen", windowVisible: false, language: "en") == .menuOnly)
    }

    @Test func permissionIsRequestedOnlyWhenLeavingTheDownloadWindowAndOnce() async {
        let center = FakeNotificationCenter(authorized: false)
        let notifier = ModelReadyNotifier(center: center)
        _ = await notifier.modelReady(modelID: model, modelName: "Qwen", windowVisible: true, language: "en")
        #expect(center.authorizationRequests == 0)
        await notifier.userLeftDownloadWindow()
        await notifier.userLeftDownloadWindow()
        #expect(center.authorizationRequests == 1)
        #expect(await notifier.modelReady(modelID: model, modelName: "Qwen", windowVisible: false, language: "en")
            == .postNotification)
    }

    @Test func notificationClickOpensChatForItsModel() {
        let notifier = ModelReadyNotifier(center: nil)
        var opened: [String] = []
        notifier.onOpenChat = { opened.append($0) }
        notifier.handleNotificationResponse(userInfo: [ModelReadyNotifier.modelIDKey: model])
        notifier.handleNotificationResponse(userInfo: ["other": "value"])
        #expect(opened == [model])
    }

    @Test func notificationCopyExistsForEveryLanguage() {
        for language in ["en", "zh-Hans", "zh-Hant", "ja", "ko"] {
            let text = ModelReadyNotifier.text(modelName: "Qwen 3.5 4B", language: language)
            #expect(!text.title.isEmpty)
            #expect(text.body.contains("Qwen 3.5 4B"))
        }
        #expect(ModelReadyNotifier.text(modelName: "X", language: "en").title == "Model ready")
    }
}
