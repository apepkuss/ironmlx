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

private func menuTitles(_ menu: NSMenu) -> [String] {
    menu.items.map { $0.isSeparatorItem ? "-" : $0.title }
}
