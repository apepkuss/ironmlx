import AppKit
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func menuBarMenuKeepsOriginalStoppedServerItems() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .stopped,
            modelNames: [],
            updatesEnabled: false,
            language: "en"
        ),
        target: nil
    )

    #expect(menuTitles(menu) == [
        "Server: Stopped",
        "—",
        "-",
        "Chat",
        "Dashboard",
        "Show Floating HUD",
        "Show Speed in Menu Bar",
        "-",
        "Start Server",
        "Restart Server",
        "-",
        "Check for Updates...",
        "-",
        "Quit",
    ])
    #expect(menu.item(withTitle: "Server: Stopped")?.isEnabled == false)
    #expect(menu.item(withTitle: "—")?.isEnabled == false)
    #expect(menu.item(withTitle: "Dashboard")?.keyEquivalent == "d")
    #expect(menu.item(withTitle: "Start Server")?.isEnabled == true)
    #expect(menu.item(withTitle: "Restart Server")?.isEnabled == false)
    #expect(menu.item(withTitle: "Check for Updates...")?.isEnabled == false)
}

@MainActor
@Test func menuBarMenuKeepsRunningServerItemsWithModels() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: ["mlx-community/Qwen3-0.6B-4bit"],
            updatesEnabled: false,
            language: "en"
        ),
        target: nil
    )

    #expect(menuTitles(menu) == [
        "Server: Running",
        "Qwen3-0.6B-4bit",
        "-",
        "Chat",
        "Dashboard",
        "Show Floating HUD",
        "Show Speed in Menu Bar",
        "-",
        "Stop Server",
        "Restart Server",
        "-",
        "Check for Updates...",
        "-",
        "Quit",
    ])
    #expect(menu.item(withTitle: "Qwen3-0.6B-4bit")?.isEnabled == false)
    #expect(menu.item(withTitle: "Stop Server")?.isEnabled == true)
    #expect(menu.item(withTitle: "Restart Server")?.isEnabled == true)
}

@MainActor
@Test func menuBarMenuRendersMultipleLoadedModels() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: [
                "mlx-community/Qwen3.5-35B-A3B-4bit",
                "mlx-community/Qwen3.5-4B-MLX-4bit",
            ],
            updatesEnabled: false,
            language: "zh-Hans"
        ),
        target: nil
    )

    #expect(menuTitles(menu).prefix(3) == [
        "服务器运行中",
        "Qwen3.5-35B-A3B-4bit",
        "Qwen3.5-4B-MLX-4bit",
    ])
}

@MainActor
@Test func menuBarMenuUsesSimplifiedChineseForDashboardLanguagePreference() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: ["mlx-community/Qwen3-0.6B-4bit"],
            updatesEnabled: false,
            language: "zh-Hans"
        ),
        target: nil
    )

    #expect(menuTitles(menu) == [
        "服务器运行中",
        "Qwen3-0.6B-4bit",
        "-",
        "对话",
        "仪表盘",
        "显示悬浮 HUD",
        "在菜单栏显示速度",
        "-",
        "停止服务",
        "重启服务",
        "-",
        "检查更新...",
        "-",
        "退出",
    ])
}

@MainActor
@Test func menuBarMenuUsesJapaneseForDashboardLanguagePreference() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .stopped,
            modelNames: [],
            updatesEnabled: false,
            language: "ja"
        ),
        target: nil
    )

    #expect(menuTitles(menu) == [
        "サーバー: 停止中",
        "—",
        "-",
        "チャット",
        "ダッシュボード",
        "フローティング HUD を表示",
        "メニューバーに速度を表示",
        "-",
        "サーバー開始",
        "サーバー再起動",
        "-",
        "アップデート確認...",
        "-",
        "終了",
    ])
}

@MainActor
@Test func menuBarMenuExposesLocalizedConfigurationRecoveryEntry() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .stopped,
            modelNames: [],
            updatesEnabled: false,
            configurationRecoveryAvailable: true,
            language: "zh-Hans"
        ),
        target: nil
    )

    #expect(menu.item(withTitle: "配置恢复...") != nil)
}

@Test func menuBarSnapshotModelNamesFallBackToPersistedLoadedModels() {
    let names = MenuBarMenuBuilder.snapshotModelNames(
        cached: nil,
        config: AppConfig(
            defaultModel: "mlx-community/Qwen3.5-35B-A3B-4bit",
            loadedModels: [
                "mlx-community/Qwen3.5-4B-MLX-4bit",
                "mlx-community/Qwen3.5-35B-A3B-4bit",
            ]
        ),
        state: .running
    )

    #expect(names == [
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        "mlx-community/Qwen3.5-4B-MLX-4bit",
    ])
}

@Test func menuBarSnapshotModelNamesUseConfirmedEmptyBackendList() {
    let names = MenuBarMenuBuilder.snapshotModelNames(
        cached: [],
        config: AppConfig(
            defaultModel: "mlx-community/Qwen3.5-35B-A3B-4bit",
            loadedModels: [
                "mlx-community/Qwen3.5-4B-MLX-4bit",
                "mlx-community/Qwen3.5-35B-A3B-4bit",
            ]
        ),
        state: .running
    )

    #expect(names == [])
}

@Test func menuBarModelNamesPreferBackendLoadedModelsInOrder() {
    let names = MenuBarMenuBuilder.modelNames(
        from: [
            BackendLoadedModelInfo(
                id: "mlx-community/Qwen3-0.6B-4bit",
                model: "Qwen3",
                path: "/tmp/qwen3",
                architecture: "qwen3",
                isDefault: true,
                maxPositionEmbeddings: 32_768
            ),
            BackendLoadedModelInfo(
                id: "mlx-community/Phi-3.5-mini-4bit",
                model: "Phi",
                path: "/tmp/phi",
                architecture: "phi3",
                isDefault: false,
                maxPositionEmbeddings: 131_072
            ),
        ],
        fallback: "mlx-community/Fallback-4bit"
    )

    #expect(names == [
        "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/Phi-3.5-mini-4bit",
    ])
}

@MainActor
@Test func chatItemIsFirstActionItem() {
    let titles = ["en": "Chat", "zh-Hans": "对话", "zh-Hant": "對話", "ja": "チャット", "ko": "채팅"]
    for (language, title) in titles {
        let menu = MenuBarMenuBuilder.makeMenu(
            snapshot: MenuBarMenuSnapshot(state: .running, modelNames: [], updatesEnabled: false, language: language),
            target: nil
        )
        let first = menu.items.first { $0.action != nil }
        #expect(first?.title == title)
        #expect(first?.action == #selector(MenuBarController.openChat(_:)))
        #expect(first?.isEnabled == true)
    }
}

@MainActor
@Test func downloadingStarterShowsProgressRow() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: [],
            updatesEnabled: false,
            language: "zh-Hans",
            starterDownload: .downloading(source: .hfMirror, fraction: 0.428, bytes: 428, total: 1_000, bytesPerSecond: 10),
            starterModelName: "Qwen3.5 4B"
        ),
        target: nil
    )
    let row = menu.item(withTitle: "正在下载 Qwen3.5 4B · 42%")
    #expect(row != nil)
    #expect(row?.isEnabled == false)
    #expect(row?.action == nil)
    #expect(menuTitles(menu).prefix(6) == ["服务器运行中", "—", "-", "对话", "正在下载 Qwen3.5 4B · 42%", "仪表盘"])

    for (language, title) in [
        "en": "Downloading Qwen3.5 4B · 42%", "zh-Hant": "正在下載 Qwen3.5 4B · 42%",
        "ja": "Qwen3.5 4B をダウンロード中 · 42%", "ko": "Qwen3.5 4B 다운로드 중 · 42%",
    ] {
        #expect(MenuBarMenuBuilder.starterRowTitle(
            .downloading(source: .huggingFace, fraction: 0.42, bytes: 42, total: 100, bytesPerSecond: nil),
            modelName: "Qwen3.5 4B", language: language
        ) == title)
    }
    #expect(MenuBarMenuBuilder.starterRowTitle(.idle, modelName: "Qwen3.5 4B", language: "en") == nil)
}

@Test func starterLoadFailureIsNotCalledAnUnfinishedDownload() {
    var titles: Set<String> = []
    for language in ["en", "zh-Hans", "zh-Hant", "ja", "ko"] {
        let loadFailed = MenuBarMenuBuilder.starterRowTitle(
            .failed(code: StarterDownloadCoordinator.modelLoadFailedCode), modelName: "Qwen3.5 4B", language: language
        )
        let downloadFailed = MenuBarMenuBuilder.starterRowTitle(
            .failed(code: "network_unreachable"), modelName: "Qwen3.5 4B", language: language
        )
        #expect(loadFailed != nil)
        #expect(loadFailed != downloadFailed, "\(language)")
        #expect(loadFailed?.contains("Qwen3.5 4B") == true)
        titles.insert(loadFailed ?? "")
    }
    #expect(titles.count == 5)
    #expect(MenuBarMenuBuilder.starterRowTitle(
        .failed(code: StarterDownloadCoordinator.modelLoadFailedCode), modelName: "Qwen3.5 4B", language: "zh-Hans"
    ) == "Qwen3.5 4B 已下载，但没能加载")
}

@MainActor
@Test func readyStarterShowsStartChatRow() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: ["mlx-community/Qwen3.5-4B-MLX-4bit"],
            updatesEnabled: false,
            language: "zh-Hans",
            starterDownload: .ready(modelID: "mlx-community/Qwen3.5-4B-MLX-4bit"),
            starterModelName: "Qwen3.5 4B"
        ),
        target: nil
    )
    let row = menu.item(withTitle: "模型已就绪，开始对话")
    #expect(row?.action == #selector(MenuBarController.openChat(_:)))
    #expect(row?.isEnabled == true)
}

@MainActor
@Test func menuInsertsStatusCardFirstWhenProvided() {
    let snapshot = MenuBarMenuSnapshot(
        state: .running,
        modelNames: ["mlx-community/Qwen3-0.6B-4bit"],
        updatesEnabled: false,
        language: "en"
    )
    let card = NSView()
    let plain = MenuBarMenuBuilder.makeMenu(snapshot: snapshot, target: nil)
    let menu = MenuBarMenuBuilder.makeMenu(snapshot: snapshot, target: nil, statusCard: card)

    #expect(menu.items[0].view === card)
    #expect(menu.items[1].isSeparatorItem)
    #expect(Array(menuTitles(menu).dropFirst(2)) == menuTitles(plain))
    #expect(plain.items.allSatisfy { $0.view == nil })
}

@MainActor
@Test func glanceTogglesReflectPreferences() {
    let menu = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: [],
            updatesEnabled: false,
            language: "en",
            hudVisible: true,
            menuBarSpeedEnabled: false,
            hudHotKeyDisplay: nil
        ),
        target: nil
    )
    let hud = menu.items.first { $0.action == #selector(MenuBarController.toggleHUD(_:)) }
    let speed = menu.items.first { $0.action == #selector(MenuBarController.toggleMenuBarSpeed(_:)) }
    #expect(hud?.title == "Show Floating HUD")
    #expect(hud?.state == .on)
    #expect(hud?.keyEquivalent == "")
    #expect(speed?.title == "Show Speed in Menu Bar")
    #expect(speed?.state == .off)

    // Defaults: HUD off, speed on; a registered hotkey is shown on the HUD item.
    let defaults = MenuBarMenuBuilder.makeMenu(
        snapshot: MenuBarMenuSnapshot(
            state: .running,
            modelNames: [],
            updatesEnabled: false,
            language: "zh-Hans",
            hudHotKeyDisplay: "⌃⌥⌘H"
        ),
        target: nil
    )
    let defaultHUD = defaults.items.first { $0.action == #selector(MenuBarController.toggleHUD(_:)) }
    let defaultSpeed = defaults.items.first { $0.action == #selector(MenuBarController.toggleMenuBarSpeed(_:)) }
    #expect(defaultHUD?.title == "显示悬浮 HUD")
    #expect(defaultHUD?.state == .off)
    #expect(defaultHUD?.keyEquivalent == "h")
    #expect(defaultHUD?.keyEquivalentModifierMask == [.control, .option, .command])
    #expect(defaultSpeed?.state == .on)
}

/// The rows the user sees: hidden items are left out.
private func menuTitles(_ menu: NSMenu) -> [String] {
    menu.items.filter { !$0.isHidden }.map { $0.isSeparatorItem ? "-" : $0.title }
}

/// "HUD Click-Through" sits right under "Show Floating HUD" and only while the HUD is on, since
/// with click-through on the HUD's own context menu cannot be reached.
@MainActor
@Test func hudClickThroughItemShowsOnlyWhileHUDIsOn() throws {
    func menu(hudVisible: Bool, clickThrough: Bool, language: String = "en") -> NSMenu {
        MenuBarMenuBuilder.makeMenu(
            snapshot: MenuBarMenuSnapshot(
                state: .running,
                modelNames: [],
                updatesEnabled: false,
                language: language,
                hudVisible: hudVisible,
                hudClickThrough: clickThrough
            ),
            target: nil
        )
    }
    let hidden = menu(hudVisible: false, clickThrough: true)
    #expect(!menuTitles(hidden).contains("HUD Click-Through"))
    let showHUD = try #require(menuTitles(hidden).firstIndex(of: "Show Floating HUD"))
    #expect(menuTitles(hidden)[showHUD + 1] == "Show Speed in Menu Bar")

    let shown = menu(hudVisible: true, clickThrough: true)
    let titles = menuTitles(shown)
    let index = try #require(titles.firstIndex(of: "Show Floating HUD"))
    #expect(titles[index + 1] == "HUD Click-Through")
    #expect(titles[index + 2] == "Show Speed in Menu Bar")
    let item = shown.items.first { $0.action == #selector(MenuBarController.toggleHUDClickThrough(_:)) }
    #expect(item?.state == .on)
    #expect(item?.isHidden == false)
    #expect(menu(hudVisible: true, clickThrough: false).items.first {
        $0.action == #selector(MenuBarController.toggleHUDClickThrough(_:))
    }?.state == .off)

    #expect(menuTitles(menu(hudVisible: true, clickThrough: false, language: "zh-Hans")).contains("HUD 点击穿透"))
}
