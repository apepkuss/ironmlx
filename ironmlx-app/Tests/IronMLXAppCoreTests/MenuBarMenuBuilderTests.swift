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

private func menuTitles(_ menu: NSMenu) -> [String] {
    menu.items.map { $0.isSeparatorItem ? "-" : $0.title }
}
