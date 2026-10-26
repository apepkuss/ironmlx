import AppKit
import Foundation

/// Opens an existing official Desktop without writing configuration or interrupting tasks.
@MainActor
struct HarnessChatService {
    enum Issue: Equatable {
        case notInstalled, incompatibleVersion, localOnly, backendUnavailable
        case modelUnavailable, configurationRequired, launchFailed
    }

    var integration = DeepSeekHarnessIntegrationService()
    var runningApplications: () -> [URL] = {
        NSRunningApplication.runningApplications(withBundleIdentifier: DeepSeekHarnessIntegrationService.bundleIdentifier)
            .compactMap(\.bundleURL)
    }
    var loadedModelIDs: (AppConfig) async throws -> [String] = { config in
        let client = BackendAPIClient(host: config.host, port: config.port)
        return try await client.fetchLoadedModels().filter { $0.capabilities.runtimeKind == "causal" }.map(\.id)
    }
    var openApplication: (URL) async throws -> Void = { url in
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        _ = try await NSWorkspace.shared.openApplication(at: url, configuration: configuration)
    }

    /// nil means the requested application was opened or its existing window was activated.
    func open(config: AppConfig, backendState: BackendProcessState) async -> Issue? {
        var service = integration
        let running = runningApplications()
        // Preserve application ownership: never open a different copy behind a running instance.
        if !running.isEmpty { service.applicationURLs = { running } }
        guard let url = service.desktopApplicationURL() else {
            return await service.inspect(mode: .desktop).code == "not_installed" ? .notInstalled : .incompatibleVersion
        }
        guard !config.isLANMode else { return .localOnly }
        guard backendState == .running || backendState == .degraded else { return .backendUnavailable }
        let models: [String]
        do { models = try await loadedModelIDs(config) }
        catch { return .backendUnavailable }
        guard !models.isEmpty else { return .modelUnavailable }
        guard service.desktopConfigurationIsReady(modelIDs: models, port: config.port) else {
            return .configurationRequired
        }
        do { try await openApplication(url) }
        catch { return .launchFailed }
        return nil
    }
}

struct HarnessChatText {
    let language: String

    static func menu(language: String) -> String { HarnessChatText(language: language).values[0] }
    static func tooltip(language: String) -> String { HarnessChatText(language: language).tooltip }
    var tooltip: String { values[1] }
    var dashboard: String { values[2] }
    var download: String { values[3] }
    var cancel: String { values[4] }

    func message(_ issue: HarnessChatService.Issue) -> String {
        let index: Int = switch issue {
        case .notInstalled: 5
        case .incompatibleVersion: 6
        case .localOnly: 7
        case .backendUnavailable: 8
        case .modelUnavailable: 9
        case .configurationRequired: 10
        case .launchFailed: 11
        }
        return values[index]
    }

    private var values: [String] {
        switch language {
        case "zh", "zh-Hans":
            ["与 DSH 对话", "打开 DeepSeek Harness", "打开仪表盘", "前往下载", "取消",
             "尚未安装官方 DeepSeek Harness Desktop。请安装后重试；也可重新运行 IronMLX 安装器并选择该组件。",
             "已安装的官方 DeepSeek Harness Desktop 版本不兼容或无法识别。请在 Agent 页面查看版本要求。",
             "此入口使用 IronMLX 本机模式。请切换为本机模式后，在 Agent 页面完成一键配置。",
             "IronMLX 推理服务尚未就绪。请在仪表盘启动服务后重试。",
             "尚未加载可用于对话的模型。请先加载对话 LLM。",
             "请在 Agent → DeepSeek Harness → 官方桌面版中使用“一键配置”，连接当前服务和已加载的模型，然后重试。",
             "无法打开 DeepSeek Harness。请检查应用是否仍在原位置，并在 Agent 页面检查安装情况。"]
        case "zh-Hant":
            ["與 DSH 對話", "開啟 DeepSeek Harness", "開啟儀表板", "前往下載", "取消",
             "尚未安裝官方 DeepSeek Harness Desktop。請安裝後重試；也可重新執行 IronMLX 安裝程式並選取該元件。",
             "已安裝的官方 DeepSeek Harness Desktop 版本不相容或無法辨識。請在 Agent 頁面查看版本要求。",
             "此入口使用 IronMLX 本機模式。請切換為本機模式後，在 Agent 頁面完成一鍵設定。",
             "IronMLX 推論服務尚未就緒。請在儀表板啟動服務後重試。",
             "尚未載入可用於對話的模型。請先載入對話 LLM。",
             "請在 Agent → DeepSeek Harness → 官方桌面版中使用「一鍵設定」，連接目前服務和已載入的模型，然後重試。",
             "無法開啟 DeepSeek Harness。請檢查應用程式是否仍在原位置，並在 Agent 頁面檢查安裝狀態。"]
        case "ja":
            ["DSH とチャット", "DeepSeek Harness を開く", "ダッシュボードを開く", "ダウンロード", "キャンセル",
             "公式の DeepSeek Harness Desktop がインストールされていません。インストールして再試行してください。IronMLX インストーラでこのコンポーネントを選択することもできます。",
             "公式の DeepSeek Harness Desktop のバージョンが非対応、または確認できません。Agent ページで対応バージョンを確認してください。",
             "この機能には IronMLX のローカルモードが必要です。ローカルモードに切り替え、Agent ページでワンクリック設定を実行してください。",
             "IronMLX 推論サービスの準備ができていません。ダッシュボードでサービスを起動して再試行してください。",
             "対話用モデルが読み込まれていません。対話用 LLM を読み込んでください。",
             "Agent → DeepSeek Harness → 公式デスクトップ版でワンクリック設定を実行し、現在のサービスとモデルを接続して再試行してください。",
             "DeepSeek Harness を開けません。アプリの場所を確認し、Agent ページでインストール状態を確認してください。"]
        case "ko":
            ["DSH와 대화", "DeepSeek Harness 열기", "대시보드 열기", "다운로드", "취소",
             "공식 DeepSeek Harness Desktop이 설치되지 않았습니다. 설치 후 다시 시도하세요. IronMLX 설치 프로그램에서 이 구성 요소를 선택할 수도 있습니다.",
             "설치된 공식 DeepSeek Harness Desktop 버전이 호환되지 않거나 확인할 수 없습니다. Agent 페이지에서 지원 버전을 확인하세요.",
             "이 기능은 IronMLX 로컬 모드를 사용합니다. 로컬 모드로 전환하고 Agent 페이지에서 원클릭 설정을 실행하세요.",
             "IronMLX 추론 서비스가 준비되지 않았습니다. 대시보드에서 서비스를 시작한 후 다시 시도하세요.",
             "대화용 모델이 로드되지 않았습니다. 대화용 LLM을 먼저 로드하세요.",
             "Agent → DeepSeek Harness → 공식 데스크톱에서 원클릭 설정으로 현재 서비스와 모델을 연결한 후 다시 시도하세요.",
             "DeepSeek Harness를 열 수 없습니다. 앱의 위치와 Agent 페이지의 설치 상태를 확인하세요."]
        default:
            ["Chat with DSH", "Open DeepSeek Harness", "Open Dashboard", "Download", "Cancel",
             "Official DeepSeek Harness Desktop is not installed. Install it and try again, or rerun the IronMLX installer and select this component.",
             "The installed official DeepSeek Harness Desktop version is unsupported or unrecognized. Check the version requirements on the Agent page.",
             "This entry uses IronMLX local mode. Switch to local mode and use one-click setup on the Agent page.",
             "The IronMLX inference service is not ready. Start it from the Dashboard and try again.",
             "No chat model is loaded. Load a conversational LLM first.",
             "Use one-click setup under Agent → DeepSeek Harness → Official desktop to connect the current service and loaded model, then try again.",
             "DeepSeek Harness could not be opened. Check its location and installation status on the Agent page."]
        }
    }
}
