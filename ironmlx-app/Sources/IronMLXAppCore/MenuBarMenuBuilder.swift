import AppKit
import Foundation

public struct MenuBarMenuSnapshot: Equatable {
    public var state: BackendProcessState
    public var modelNames: [String]
    public var openClawInstalled: Bool
    public var openClawGatewayConfigured: Bool
    public var ironHermesInstalled: Bool
    public var updatesEnabled: Bool
    public var language: String

    public init(
        state: BackendProcessState,
        modelNames: [String],
        openClawInstalled: Bool,
        openClawGatewayConfigured: Bool,
        ironHermesInstalled: Bool,
        updatesEnabled: Bool,
        language: String = "en"
    ) {
        self.state = state
        self.modelNames = modelNames
        self.openClawInstalled = openClawInstalled
        self.openClawGatewayConfigured = openClawGatewayConfigured
        self.ironHermesInstalled = ironHermesInstalled
        self.updatesEnabled = updatesEnabled
        self.language = language
    }
}

@MainActor
public enum MenuBarMenuBuilder {
    nonisolated static func snapshotModelNames(
        cached: [String]?,
        config: AppConfig,
        state: BackendProcessState
    ) -> [String] {
        guard state == .running || state == .starting else {
            return []
        }
        if let cached {
            return cached
        }
        return config.restoredModelReferences
    }

    nonisolated static func modelNames(from loadedModels: [BackendLoadedModelInfo], fallback: String? = nil) -> [String] {
        var seen = Set<String>()
        var names: [String] = []
        for model in loadedModels {
            let name = firstNonEmpty(model.id, model.model, model.path)
            guard let name, seen.insert(name).inserted else {
                continue
            }
            names.append(name)
        }
        if names.isEmpty, let fallback = trimmedNonEmpty(fallback) {
            names.append(fallback)
        }
        return names
    }

    public static func makeMenu(snapshot: MenuBarMenuSnapshot, target: AnyObject?) -> NSMenu {
        let menu = NSMenu()
        menu.autoenablesItems = false

        addStatus(snapshot.state, language: snapshot.language, to: menu)
        addModelRows(snapshot.modelNames, to: menu)
        menu.addItem(.separator())

        menu.addItem(
            item(
                localized(.dashboard, language: snapshot.language),
                #selector(MenuBarController.openDashboard(_:)),
                "d",
                target: target,
                symbol: "square.grid.2x2"
            )
        )
        menu.addItem(.separator())

        if addAgentItems(snapshot, target: target, to: menu) {
            menu.addItem(.separator())
        }

        addBackendControlItems(snapshot.state, language: snapshot.language, target: target, to: menu)
        menu.addItem(.separator())

        let updates = item(
            localized(.updates, language: snapshot.language),
            #selector(MenuBarController.checkForUpdates(_:)),
            "",
            target: target,
            symbol: "arrow.down.circle"
        )
        updates.isEnabled = snapshot.updatesEnabled
        menu.addItem(updates)
        menu.addItem(.separator())

        menu.addItem(
            item(
                localized(.quit, language: snapshot.language),
                #selector(MenuBarController.quit(_:)),
                "q",
                target: target,
                symbol: "power"
            )
        )
        return menu
    }

    private static func addStatus(_ state: BackendProcessState, language: String, to menu: NSMenu) {
        let title: String
        let color: NSColor
        switch state {
        case .running:
            title = localized(.statusRunning, language: language)
            color = .systemGreen
        case .starting:
            title = localized(.statusStarting, language: language)
            color = .systemYellow
        case .failed:
            title = localized(.statusFailed, language: language)
            color = .systemRed
        case .stopped:
            title = localized(.statusStopped, language: language)
            color = .systemRed
        }
        let status = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        status.isEnabled = false
        status.image = symbolImage("circle.fill", color: color)
        menu.addItem(status)
    }

    private static func addModelRows(_ modelNames: [String], to menu: NSMenu) {
        let rows = modelNames.isEmpty ? ["—"] : modelNames
        for modelName in rows {
            let shortName = modelName.split(separator: "/").last.map(String.init) ?? modelName
            let item = NSMenuItem(title: shortName, action: nil, keyEquivalent: "")
            item.isEnabled = false
            item.image = symbolImage("cube")
            menu.addItem(item)
        }
    }

    private static func addAgentItems(_ snapshot: MenuBarMenuSnapshot, target: AnyObject?, to menu: NSMenu) -> Bool {
        var didAddItem = false
        if snapshot.openClawInstalled {
            let openClaw = item(
                localized(.chatOpenClaw, language: snapshot.language),
                #selector(MenuBarController.openOpenClawChat(_:)),
                "",
                target: target,
                symbol: "bubble.left.and.text.bubble.right"
            )
            openClaw.isEnabled = snapshot.openClawGatewayConfigured
            menu.addItem(openClaw)
            didAddItem = true
        }

        if snapshot.ironHermesInstalled {
            menu.addItem(
                item(
                    localized(.chatIronHermes, language: snapshot.language),
                    #selector(MenuBarController.openIronHermes(_:)),
                    "",
                    target: target,
                    symbol: "bubble.left.and.bubble.right"
                )
            )
            didAddItem = true
        }
        return didAddItem
    }

    private static func addBackendControlItems(
        _ state: BackendProcessState,
        language: String,
        target: AnyObject?,
        to menu: NSMenu
    ) {
        if state == .running || state == .starting {
            menu.addItem(
                item(
                    localized(.stop, language: language),
                    #selector(MenuBarController.stopServer(_:)),
                    "",
                    target: target,
                    symbol: "stop.fill"
                )
            )
        } else {
            menu.addItem(
                item(
                    localized(.start, language: language),
                    #selector(MenuBarController.startServer(_:)),
                    "",
                    target: target,
                    symbol: "play.fill"
                )
            )
        }

        let restart = item(
            localized(.restart, language: language),
            #selector(MenuBarController.restartServer(_:)),
            "",
            target: target,
            symbol: "arrow.clockwise"
        )
        restart.isEnabled = state == .running
        menu.addItem(restart)
    }

    nonisolated private static func localized(_ key: MenuTextKey, language: String) -> String {
        switch (normalizedLanguage(language), key) {
        case ("zh", .dashboard):
            return "仪表盘"
        case ("zh", .chatOpenClaw):
            return "与 OpenClaw 聊天"
        case ("zh", .chatIronHermes):
            return "与 IronHermes 聊天"
        case ("zh", .stop):
            return "停止服务"
        case ("zh", .start):
            return "启动服务"
        case ("zh", .restart):
            return "重启服务"
        case ("zh", .updates):
            return "检查更新..."
        case ("zh", .quit):
            return "退出"
        case ("zh", .statusRunning):
            return "服务器运行中"
        case ("zh", .statusStarting):
            return "服务器启动中"
        case ("zh", .statusFailed):
            return "服务器启动失败"
        case ("zh", .statusStopped):
            return "服务器已停止"

        case ("zh-Hant", .dashboard):
            return "儀表板"
        case ("zh-Hant", .chatOpenClaw):
            return "與 OpenClaw 聊天"
        case ("zh-Hant", .chatIronHermes):
            return "與 IronHermes 聊天"
        case ("zh-Hant", .stop):
            return "停止服務"
        case ("zh-Hant", .start):
            return "啟動服務"
        case ("zh-Hant", .restart):
            return "重啟服務"
        case ("zh-Hant", .updates):
            return "檢查更新..."
        case ("zh-Hant", .quit):
            return "退出"
        case ("zh-Hant", .statusRunning):
            return "伺服器執行中"
        case ("zh-Hant", .statusStarting):
            return "伺服器啟動中"
        case ("zh-Hant", .statusFailed):
            return "伺服器啟動失敗"
        case ("zh-Hant", .statusStopped):
            return "伺服器已停止"

        case ("ja", .dashboard):
            return "ダッシュボード"
        case ("ja", .chatOpenClaw):
            return "OpenClaw でチャット"
        case ("ja", .chatIronHermes):
            return "IronHermes とチャット"
        case ("ja", .stop):
            return "サーバー停止"
        case ("ja", .start):
            return "サーバー開始"
        case ("ja", .restart):
            return "サーバー再起動"
        case ("ja", .updates):
            return "アップデート確認..."
        case ("ja", .quit):
            return "終了"
        case ("ja", .statusRunning):
            return "サーバー: 実行中"
        case ("ja", .statusStarting):
            return "サーバー: 起動中"
        case ("ja", .statusFailed):
            return "サーバー: 失敗"
        case ("ja", .statusStopped):
            return "サーバー: 停止中"

        case ("ko", .dashboard):
            return "대시보드"
        case ("ko", .chatOpenClaw):
            return "OpenClaw로 채팅"
        case ("ko", .chatIronHermes):
            return "IronHermes와 채팅"
        case ("ko", .stop):
            return "서버 정지"
        case ("ko", .start):
            return "서버 시작"
        case ("ko", .restart):
            return "서버 재시작"
        case ("ko", .updates):
            return "업데이트 확인..."
        case ("ko", .quit):
            return "종료"
        case ("ko", .statusRunning):
            return "서버: 실행 중"
        case ("ko", .statusStarting):
            return "서버: 시작 중"
        case ("ko", .statusFailed):
            return "서버: 실패"
        case ("ko", .statusStopped):
            return "서버: 정지됨"

        default:
            return englishText(key)
        }
    }

    nonisolated private static func normalizedLanguage(_ language: String) -> String {
        switch language.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "zh", "zh-Hans":
            return "zh"
        case "zh-Hant":
            return "zh-Hant"
        case "ja":
            return "ja"
        case "ko":
            return "ko"
        default:
            return "en"
        }
    }

    nonisolated private static func englishText(_ key: MenuTextKey) -> String {
        switch key {
        case .dashboard:
            return "Dashboard"
        case .chatOpenClaw:
            return "Chat with OpenClaw"
        case .chatIronHermes:
            return "Chat with IronHermes"
        case .stop:
            return "Stop Server"
        case .start:
            return "Start Server"
        case .restart:
            return "Restart Server"
        case .updates:
            return "Check for Updates..."
        case .quit:
            return "Quit"
        case .statusRunning:
            return "Server: Running"
        case .statusStarting:
            return "Server: Starting"
        case .statusFailed:
            return "Server: Failed"
        case .statusStopped:
            return "Server: Stopped"
        }
    }

    nonisolated private static func firstNonEmpty(_ values: String?...) -> String? {
        for value in values {
            if let trimmed = trimmedNonEmpty(value) {
                return trimmed
            }
        }
        return nil
    }

    nonisolated private static func trimmedNonEmpty(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private static func item(
        _ title: String,
        _ action: Selector,
        _ keyEquivalent: String,
        target: AnyObject?,
        symbol: String
    ) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
        item.target = target
        item.image = symbolImage(symbol)
        return item
    }

    private static func symbolImage(_ name: String, color: NSColor? = nil) -> NSImage? {
        guard let image = NSImage(systemSymbolName: name, accessibilityDescription: nil) else {
            return nil
        }
        guard let color else {
            return image
        }
        let configuration = NSImage.SymbolConfiguration(hierarchicalColor: color)
        return image.withSymbolConfiguration(configuration)
    }
}

private enum MenuTextKey {
    case dashboard
    case chatOpenClaw
    case chatIronHermes
    case stop
    case start
    case restart
    case updates
    case quit
    case statusRunning
    case statusStarting
    case statusFailed
    case statusStopped
}
