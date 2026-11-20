import Foundation

/// Text for the menu bar readout, the menu status card and the HUD.
public enum GlanceText: CaseIterable, Sendable {
    case stateGenerating
    case statePrefilling
    case stateLoading
    case stateIdle
    case stateStarting
    case backendNotRunning
    case reconnecting
    /// Format string; `%@` is the formatted duration.
    case uptime
    case decodeUnit
    case window60
    case prefill
    case prefillShort
    case ttft
    case active
    case queued
    case memory
    case memoryModels
    case memoryKV
    case memoryCache
    case prefixHit
    case showHUD
    /// The menu bar item; the HUD's own context menu uses `hudClickThrough`.
    case menuHUDClickThrough
    case menuBarSpeed
    case hudDensity
    case hudMini
    case hudStandard
    case hudAutoHide
    case hudClickThrough
    case hudResetPosition
    case hudClose

    public static func localized(_ key: GlanceText, language: String) -> String {
        key.translations.text(for: language)
    }

    /// A short title for a backend `HealthDegradedReason`; unknown reasons are returned verbatim.
    public static func reasonTitle(_ reason: String, language: String) -> String {
        reasonTranslations(reason)?.text(for: language) ?? reason
    }

    /// Whether `reason` has a translated title (an older App may get reasons it does not know).
    static func isKnownReason(_ reason: String) -> Bool {
        reasonTranslations(reason) != nil
    }

    private var translations: GlanceTranslations {
        switch self {
        case .stateGenerating:
            .init(en: "Generating", zh: "正在生成", zhHant: "正在生成", ja: "生成中", ko: "생성 중")
        case .statePrefilling:
            .init(en: "Prefilling", zh: "预填充中", zhHant: "預填充中", ja: "プリフィル中", ko: "프리필 중")
        case .stateLoading:
            .init(en: "Loading", zh: "正在加载", zhHant: "正在載入", ja: "読み込み中", ko: "로드 중")
        case .stateIdle:
            .init(en: "Idle", zh: "空闲", zhHant: "閒置", ja: "アイドル", ko: "유휴")
        case .stateStarting:
            .init(en: "Starting", zh: "正在启动", zhHant: "正在啟動", ja: "起動中", ko: "시작 중")
        case .backendNotRunning:
            .init(
                en: "Server not running", zh: "后端未运行", zhHant: "後端未執行",
                ja: "サーバー未起動", ko: "서버 실행 안 됨"
            )
        case .reconnecting:
            .init(en: "Reconnecting", zh: "正在重新连接", zhHant: "正在重新連線", ja: "再接続中", ko: "다시 연결 중")
        case .uptime:
            .init(en: "Up %@", zh: "运行 %@", zhHant: "執行 %@", ja: "稼働 %@", ko: "가동 %@")
        case .decodeUnit:
            .init(
                en: "tok/s decode", zh: "tok/s 解码", zhHant: "tok/s 解碼",
                ja: "tok/s デコード", ko: "tok/s 디코드"
            )
        case .window60:
            .init(en: "Last 60 s", zh: "近 60 秒", zhHant: "近 60 秒", ja: "直近 60 秒", ko: "최근 60초")
        case .prefill, .prefillShort:
            .init(en: "Prefill", zh: "预填充", zhHant: "預填充", ja: "プリフィル", ko: "프리필")
        case .ttft:
            .init(en: "TTFT", zh: "首字延迟", zhHant: "首字延遲", ja: "初回トークン遅延", ko: "첫 토큰 지연")
        case .active:
            .init(en: "Active", zh: "进行中", zhHant: "進行中", ja: "処理中", ko: "처리 중")
        case .queued:
            .init(en: "Queued", zh: "排队", zhHant: "排隊", ja: "待機", ko: "대기")
        case .memory:
            .init(en: "Memory", zh: "内存", zhHant: "記憶體", ja: "メモリ", ko: "메모리")
        case .memoryModels:
            .init(
                en: "Models & other", zh: "模型与其他", zhHant: "模型與其他",
                ja: "モデルとその他", ko: "모델 및 기타"
            )
        case .memoryKV:
            .init(en: "KV", zh: "KV", zhHant: "KV", ja: "KV", ko: "KV")
        case .memoryCache:
            .init(en: "Cache", zh: "缓存", zhHant: "快取", ja: "キャッシュ", ko: "캐시")
        case .prefixHit:
            .init(
                en: "Prefix cache hits", zh: "前缀缓存命中", zhHant: "前綴快取命中",
                ja: "プレフィックスキャッシュヒット", ko: "프리픽스 캐시 적중"
            )
        case .showHUD:
            .init(
                en: "Show Floating HUD", zh: "显示悬浮 HUD", zhHant: "顯示浮動 HUD",
                ja: "フローティング HUD を表示", ko: "플로팅 HUD 표시"
            )
        case .menuHUDClickThrough:
            .init(
                en: "HUD Click-Through", zh: "HUD 点击穿透", zhHant: "HUD 點擊穿透",
                ja: "HUD のクリックを透過", ko: "HUD 클릭 통과"
            )
        case .menuBarSpeed:
            .init(
                en: "Show Speed in Menu Bar", zh: "在菜单栏显示速度", zhHant: "在選單列顯示速度",
                ja: "メニューバーに速度を表示", ko: "메뉴 막대에 속도 표시"
            )
        case .hudDensity:
            .init(en: "Density", zh: "密度", zhHant: "密度", ja: "表示密度", ko: "밀도")
        case .hudMini:
            .init(en: "Mini", zh: "迷你", zhHant: "迷你", ja: "ミニ", ko: "미니")
        case .hudStandard:
            .init(en: "Standard", zh: "标准", zhHant: "標準", ja: "標準", ko: "표준")
        case .hudAutoHide:
            .init(
                en: "Hide When Idle", zh: "空闲时自动隐藏", zhHant: "閒置時自動隱藏",
                ja: "アイドル時に自動で隠す", ko: "유휴 시 자동 숨기기"
            )
        case .hudClickThrough:
            .init(en: "Click Through", zh: "点击穿透", zhHant: "點擊穿透", ja: "クリックを透過", ko: "클릭 통과")
        case .hudResetPosition:
            .init(en: "Reset Position", zh: "重置位置", zhHant: "重設位置", ja: "位置をリセット", ko: "위치 재설정")
        case .hudClose:
            .init(en: "Close", zh: "关闭", zhHant: "關閉", ja: "閉じる", ko: "닫기")
        }
    }

    private static func reasonTranslations(_ reason: String) -> GlanceTranslations? {
        switch reason {
        case "process_memory_soft", "process_memory_hard", "process_memory_emergency":
            .init(en: "Memory pressure", zh: "内存紧张", zhHant: "記憶體緊張", ja: "メモリ逼迫", ko: "메모리 부족")
        case "scheduler_queue_high":
            .init(en: "Queue backed up", zh: "排队过多", zhHant: "排隊過多", ja: "キュー混雑", ko: "대기열 과다")
        case "model_failed":
            .init(
                en: "Model failed to load", zh: "模型加载失败", zhHant: "模型載入失敗",
                ja: "モデルの読み込みに失敗", ko: "모델 로드 실패"
            )
        case "kv_cache_near_soft_limit":
            .init(
                en: "KV cache near limit", zh: "KV 接近上限", zhHant: "KV 接近上限",
                ja: "KV キャッシュ上限間近", ko: "KV 캐시 한도 근접"
            )
        case "process_memory_telemetry_degraded":
            .init(
                en: "Memory telemetry unavailable", zh: "内存遥测异常", zhHant: "記憶體遙測異常",
                ja: "メモリ計測不可", ko: "메모리 측정 불가"
            )
        case "prefix_store_backpressured":
            .init(
                en: "Prefix cache writes backed up", zh: "前缀缓存写入积压", zhHant: "前綴快取寫入積壓",
                ja: "プレフィックスキャッシュ書き込み滞留", ko: "프리픽스 캐시 쓰기 지연"
            )
        case "active_kv_offload_degraded":
            .init(
                en: "KV offload degraded", zh: "KV 卸载降级", zhHant: "KV 卸載降級",
                ja: "KV オフロード低下", ko: "KV 오프로드 저하"
            )
        default:
            nil
        }
    }
}

private struct GlanceTranslations {
    var en: String
    var zh: String
    var zhHant: String
    var ja: String
    var ko: String

    func text(for language: String) -> String {
        switch MenuBarMenuBuilder.normalizedLanguage(language) {
        case "zh": zh
        case "zh-Hant": zhHant
        case "ja": ja
        case "ko": ko
        default: en
        }
    }
}
