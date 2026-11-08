import AppKit
import SwiftUI

/// State and actions behind the no-model window.
@MainActor
final class NoModelWindowModel: ObservableObject {
    @Published var state: StarterDownloadState = .idle
    @Published var language: String
    let recommendation: StarterRecommendation?
    let physicalMemoryBytes: UInt64
    var onDownload: () -> Void = {}
    var onChooseModel: () -> Void = {}
    var onClose: () -> Void = {}
    var onCancelDownload: () -> Void = {}

    init(recommendation: StarterRecommendation?, physicalMemoryBytes: UInt64, language: String) {
        self.recommendation = recommendation
        self.physicalMemoryBytes = physicalMemoryBytes
        self.language = language
    }

    var text: NoModelWindowText { NoModelWindowText(language: language) }

    /// Probing, downloading, paused and loading all show the progress view;
    /// so does ready, until the window closes and chat opens.
    var showsProgress: Bool { Self.showsProgress(state) }

    /// Still on its way: closing the window leaves a download behind.
    var isDownloading: Bool { showsProgress && !state.isReady }

    nonisolated static func showsProgress(_ state: StarterDownloadState) -> Bool {
        switch state {
        case .probing, .downloading, .paused, .loading, .ready: true
        case .idle, .failed: false
        }
    }
}

/// The small native window shown instead of chat when there is no
/// conversation model: it offers the recommended model, then shows its download.
@MainActor
final class NoModelWindowController: NSObject, NSWindowDelegate {
    let model: NoModelWindowModel
    private var window: NSWindow?
    /// Called when the window closes while a download runs (including
    /// "Continue in Background"), which is when notification permission is asked.
    var onClosedWhileDownloading: () -> Void = {}
    /// Shared with chat and the Dashboard; nil leaves the activation policy alone.
    let activation: AppWindowActivation?

    init(model: NoModelWindowModel, activation: AppWindowActivation? = nil) {
        self.model = model
        self.activation = activation
        super.init()
        model.onClose = { [weak self] in self?.close() }
    }

    var isVisible: Bool { window?.isVisible == true }

    func show(language: String) {
        model.language = language
        let window = window ?? makeWindow()
        self.window = window
        if !window.isVisible { window.center() }
        activation?.shown(.noModel)
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    func close() {
        window?.close()
    }

    func update(_ state: StarterDownloadState) {
        model.state = state
    }

    func windowWillClose(_ notification: Notification) {
        activation?.hidden(.noModel)
        if model.isDownloading { onClosedWhileDownloading() }
    }

    private func makeWindow() -> NSWindow {
        let hosting = NSHostingController(rootView: NoModelView(model: model))
        hosting.sizingOptions = [.preferredContentSize]
        let window = NSWindow(contentViewController: hosting)
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.title = "IronMLX"
        window.isMovableByWindowBackground = true
        window.isReleasedWhenClosed = false
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true
        window.delegate = self
        return window
    }
}

private struct NoModelView: View {
    @ObservedObject var model: NoModelWindowModel

    var body: some View {
        let text = model.text
        VStack(spacing: 0) {
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .fill(Color.accentColor)
                .frame(width: 48, height: 48)
                .overlay(Text("Fe").font(.system(size: 19, weight: .bold)).foregroundStyle(.white))
                .padding(.bottom, 12)
            if model.showsProgress {
                progressContent(text)
            } else {
                offerContent(text)
            }
        }
        .multilineTextAlignment(.center)
        .padding(.horizontal, 22)
        .padding(.top, 30)
        .padding(.bottom, 18)
        .frame(width: 360)
        .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private func offerContent(_ text: NoModelWindowText) -> some View {
        Text(text.offerTitle).font(.system(size: 15, weight: .semibold)).padding(.bottom, 6)
        Text(text.offerDescription).font(.system(size: 12.5)).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.bottom, 14)
        if let recommendation = model.recommendation {
            Card {
                HStack(alignment: .firstTextBaseline) {
                    Text(recommendation.entry.name).font(.system(size: 13.5, weight: .semibold))
                    Spacer()
                    Text(NoModelWindowText.gigabytes(recommendation.entry.downloadBytes, digits: 1))
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Text(text.reason(physicalMemoryBytes: model.physicalMemoryBytes, capabilities: recommendation.entry.capabilities))
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 4)
                HStack(spacing: 6) {
                    ForEach(text.tags(for: recommendation.entry.capabilities), id: \.self) { tag in
                        Text(tag).font(.system(size: 11)).foregroundStyle(.secondary)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 4))
                    }
                }
                .padding(.top, 7)
            }
            .padding(.bottom, 16)
        }
        if case let .failed(code) = model.state {
            Text(text.failure(code: code))
                .font(.system(size: 12)).foregroundStyle(Color(nsColor: .systemRed))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, 10)
        }
        if model.recommendation != nil {
            WideButton(title: model.state.isFailed ? text.retry : text.downloadRecommended, style: .primary,
                       action: model.onDownload)
        }
        WideButton(title: text.chooseModel, style: .secondary, action: model.onChooseModel)
        WideButton(title: text.notNow, style: .link, action: model.onClose)
    }

    @ViewBuilder
    private func progressContent(_ text: NoModelWindowText) -> some View {
        let name = model.recommendation?.entry.name ?? ""
        Text(text.preparingTitle(name)).font(.system(size: 15, weight: .semibold)).padding(.bottom, 6)
        Text(text.preparingDescription).font(.system(size: 12.5)).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.bottom, 14)
        Card {
            HStack(alignment: .firstTextBaseline) {
                Text(text.phase(model.state)).font(.system(size: 13.5, weight: .semibold))
                Spacer()
                if let progress = model.state.progress {
                    Text(NoModelWindowText.percent(progress.fraction)).font(.system(size: 12)).foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            progressBar
                .padding(.top, 10).padding(.bottom, 6)
            if let progress = model.state.progress {
                HStack {
                    Text("\(NoModelWindowText.downloadedSize(progress.bytes)) / \(NoModelWindowText.gigabytes(progress.total, digits: 1))")
                    Spacer()
                    Text(text.rateAndRemaining(bytes: progress.bytes, total: progress.total,
                                               bytesPerSecond: progress.bytesPerSecond))
                }
                .font(.system(size: 11.5)).foregroundStyle(.secondary).monospacedDigit()
            }
            if let source = text.sourceLine(model.state) {
                Text(source).font(.system(size: 11.5)).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 10)
            }
        }
        .padding(.bottom, 16)
        // Ready: chat opens and this window closes on its own.
        if model.state.isReady {
            EmptyView()
        } else if model.state.isPaused {
            WideButton(title: text.resume, style: .primary, action: model.onDownload)
        } else {
            WideButton(title: text.continueInBackground, style: .secondary, action: model.onClose)
        }
        if model.state != .loading, !model.state.isReady {
            WideButton(title: text.cancelDownload, style: .link, action: model.onCancelDownload)
        }
        if !model.state.isPaused, !model.state.isReady {
            Text(text.notificationNote).font(.system(size: 11)).foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 8)
        }
    }

    @ViewBuilder
    private var progressBar: some View {
        switch model.state {
        case .downloading, .paused:
            ProgressBar(fraction: model.state.progress?.fraction ?? 0)
        case .ready:
            ProgressBar(fraction: 1)
        default:
            ProgressView().progressViewStyle(.linear)
        }
    }
}

private struct Card<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) { content }
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 13).padding(.vertical, 11)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.09), lineWidth: 0.5))
    }
}

private struct WideButton: View {
    enum Style { case primary, secondary, link }
    let title: String
    let style: Style
    let action: () -> Void

    var body: some View {
        switch style {
        case .primary:
            Button(title, action: action).buttonStyle(FilledButtonStyle(prominent: true))
                .keyboardShortcut(.defaultAction).padding(.top, 8)
        case .secondary:
            Button(title, action: action).buttonStyle(FilledButtonStyle(prominent: false)).padding(.top, 8)
        case .link:
            Button(action: action) {
                Text(title).font(.system(size: 13, weight: .medium)).foregroundStyle(Color.accentColor)
                    .frame(maxWidth: .infinity).padding(.vertical, 6).contentShape(Rectangle())
            }
            .buttonStyle(.plain).padding(.top, 2)
        }
    }
}

/// Full-width buttons drawn the same in key and inactive windows, as in the mockup.
private struct FilledButtonStyle: ButtonStyle {
    let prominent: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: prominent ? .semibold : .medium))
            .foregroundStyle(prominent ? Color.white : Color.primary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 7)
            .background(
                prominent ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(Color.primary.opacity(0.09)),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .opacity(configuration.isPressed ? 0.75 : 1)
            .contentShape(Rectangle())
    }
}

/// A determinate bar that keeps the accent colour when the window is inactive.
private struct ProgressBar: View {
    let fraction: Double

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.1))
                Capsule().fill(Color.accentColor)
                    .frame(width: proxy.size.width * max(0, min(1, fraction)))
            }
        }
        .frame(height: 6)
        .accessibilityElement()
        .accessibilityValue(NoModelWindowText.percent(fraction))
    }
}

private extension StarterDownloadState {
    var isFailed: Bool {
        if case .failed = self { return true }
        return false
    }

    var isPaused: Bool {
        if case .paused = self { return true }
        return false
    }

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// Every string the no-model window shows, in the five Dashboard languages.
struct NoModelWindowText {
    let language: String

    init(language: String) {
        switch language.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "zh", "zh-Hans": self.language = "zh-Hans"
        case "zh-Hant", "ja", "ko": self.language = language.trimmingCharacters(in: .whitespacesAndNewlines)
        default: self.language = "en"
        }
    }

    private func pick(_ en: String, _ zh: String, _ zht: String, _ ja: String, _ ko: String) -> String {
        switch language {
        case "zh-Hans": zh
        case "zh-Hant": zht
        case "ja": ja
        case "ko": ko
        default: en
        }
    }

    var offerTitle: String {
        pick("No chat model yet", "还没有可以对话的模型", "還沒有可以對話的模型",
             "チャットできるモデルがまだありません", "아직 대화할 수 있는 모델이 없습니다")
    }

    var offerDescription: String {
        pick("Download one model to get started. Everything runs on this Mac.",
             "下载一个模型就能开始。所有内容都在这台 Mac 上运行。",
             "下載一個模型就能開始。所有內容都在這台 Mac 上執行。",
             "モデルを 1 つダウンロードすれば始められます。すべてこの Mac 上で動作します。",
             "모델 하나만 내려받으면 바로 시작할 수 있습니다. 모든 처리는 이 Mac에서 이루어집니다.")
    }

    /// Why the model suits this Mac: memory first, then what it can do.
    func reason(physicalMemoryBytes: UInt64, capabilities: [String]) -> String {
        let gib = Int((Double(physicalMemoryBytes) / Double(1 << 30)).rounded())
        let small = gib < 16
        let vision = capabilities.contains("vision")
        switch language {
        case "zh-Hans":
            return "适合你这台 \(gib) GB 内存的 Mac：" + (small ? "占用内存少、回复快" : "回复快、中英文都好")
                + (vision ? "，还能看图。" : "。")
        case "zh-Hant":
            return "適合你這台 \(gib) GB 記憶體的 Mac：" + (small ? "佔用記憶體少、回覆快" : "回覆快、中英文都好")
                + (vision ? "，還能看圖。" : "。")
        case "ja":
            return "メモリ \(gib) GB のこの Mac に向いています。"
                + (small ? "メモリ消費が少なく、返答も速いモデルです。" : "返答が速く、多くの言語に対応しています。")
                + (vision ? "画像も読み取れます。" : "")
        case "ko":
            return "메모리 \(gib)GB인 이 Mac에 알맞습니다. "
                + (small ? "메모리를 적게 쓰고 답변이 빠릅니다." : "답변이 빠르고 여러 언어를 지원합니다.")
                + (vision ? " 이미지도 이해합니다." : "")
        default:
            return "Suits your \(gib) GB Mac: " + (small ? "light on memory and quick to reply" : "quick replies and strong in English and Chinese")
                + (vision ? ", and it can read images." : ".")
        }
    }

    func tags(for capabilities: [String]) -> [String] {
        var tags: [String] = []
        if capabilities.contains("text") { tags.append(pick("Chat", "对话", "對話", "チャット", "대화")) }
        if capabilities.contains("vision") { tags.append(pick("Images", "看图", "看圖", "画像", "이미지")) }
        if capabilities.contains("reasoning") {
            tags.append(pick("Thinking", "可开启思考", "可開啟思考", "思考モード", "생각 모드"))
        }
        return tags
    }

    var downloadRecommended: String {
        pick("Download Recommended Model", "下载推荐模型", "下載推薦模型", "おすすめモデルをダウンロード", "추천 모델 다운로드")
    }

    var chooseModel: String { pick("Choose a Model Myself", "自己选择模型", "自己選擇模型", "自分でモデルを選ぶ", "직접 모델 선택") }
    var notNow: String { pick("Not Now", "以后再说", "以後再說", "あとで", "나중에") }
    var retry: String { pick("Try Again", "重试", "重試", "再試行", "다시 시도") }
    var resume: String { pick("Resume Download", "继续下载", "繼續下載", "ダウンロードを再開", "다운로드 재개") }

    var continueInBackground: String {
        pick("Continue in Background", "在后台继续", "在背景繼續", "バックグラウンドで続ける", "백그라운드에서 계속")
    }

    var cancelDownload: String {
        pick("Cancel Download", "取消下载", "取消下載", "ダウンロードをキャンセル", "다운로드 취소")
    }

    func preparingTitle(_ name: String) -> String {
        pick("Preparing \(name)", "正在准备 \(name)", "正在準備 \(name)", "\(name) を準備中", "\(name) 준비 중")
    }

    var preparingDescription: String {
        pick("If this window stays open, chat opens when it's done. You can close it; you'll be notified when the model is ready.",
             "完成后会直接进入对话；关掉这个窗口也没关系，准备好后会通知你。",
             "完成後會直接進入對話；關掉這個視窗也沒關係，準備好後會通知你。",
             "このウインドウを開いたままなら、完了後そのままチャットに進みます。閉じても大丈夫です。準備ができたら通知でお知らせします。",
             "이 창을 열어 두면 완료 후 바로 대화로 넘어갑니다. 창을 닫아도 괜찮습니다. 준비되면 알림으로 알려 드립니다.")
    }

    /// The one-line reason for the permission prompt that may follow.
    var notificationNote: String {
        pick("If you continue in the background, IronMLX asks to notify you when the model is ready.",
             "在后台继续时，IronMLX 会请求通知权限，以便下载完成后提醒你。",
             "在背景繼續時，IronMLX 會請求通知權限，以便下載完成後提醒你。",
             "バックグラウンドで続けると、完了をお知らせするために通知の許可を求めます。",
             "백그라운드에서 계속하면 완료를 알려 드리기 위해 알림 권한을 요청합니다.")
    }

    func phase(_ state: StarterDownloadState) -> String {
        switch state {
        case .probing:
            pick("Choosing a download source", "正在选择下载源", "正在選擇下載來源", "ダウンロード元を選択中", "다운로드 서버 선택 중")
        case .paused:
            pick("Paused", "已暂停", "已暫停", "一時停止中", "일시 정지됨")
        case .loading:
            pick("Loading the model", "正在加载模型", "正在載入模型", "モデルを読み込み中", "모델 불러오는 중")
        case .ready:
            pick("Ready. Opening chat…", "已就绪，正在打开对话…", "已就緒，正在開啟對話…",
                 "準備ができました。チャットを開いています…", "준비되었습니다. 대화를 여는 중…")
        case .idle, .downloading, .failed:
            pick("Downloading", "下载中", "下載中", "ダウンロード中", "다운로드 중")
        }
    }

    /// Only a fresh download from the mirror explains why; a resumed one keeps
    /// its host whatever the network is like now.
    func sourceLine(_ state: StarterDownloadState) -> String? {
        let source: DownloadSource
        let explain: Bool
        switch state {
        case let .downloading(value, _, _, _, _):
            source = value
            explain = true
        case let .paused(value, _, _):
            source = value
            explain = false
        case .idle, .probing, .loading, .ready, .failed:
            return nil
        }
        let prefix = pick("Source: ", "来源：", "來源：", "ダウンロード元：", "출처: ")
        guard source == .hfMirror else { return prefix + "huggingface.co" }
        guard explain else { return prefix + "hf-mirror.com" }
        return prefix + pick("hf-mirror.com (huggingface.co is unreachable right now)",
                             "hf-mirror.com（huggingface.co 当前无法访问）",
                             "hf-mirror.com（huggingface.co 目前無法連線）",
                             "hf-mirror.com（huggingface.co に現在接続できません）",
                             "hf-mirror.com (지금은 huggingface.co에 연결할 수 없음)")
    }

    func rateAndRemaining(bytes: Int64, total: Int64, bytesPerSecond: Double?) -> String {
        guard let rate = bytesPerSecond, rate > 0 else { return "" }
        // Below 1 MB/s show KB/s, so a slow link never reads "0.0 MB/s".
        let speed = rate < 1_000_000
            ? "\(max(1, Int((rate / 1_000).rounded(.down)))) KB/s"
            : String(format: "%.1f MB/s", rate / 1_000_000)
        let seconds = Double(max(0, total - bytes)) / rate
        let remaining: String
        if seconds < 60 {
            remaining = pick("under a minute", "不到 1 分钟", "不到 1 分鐘", "1 分未満", "1분 미만")
        } else {
            let minutes = Int((seconds / 60).rounded(.up))
            remaining = pick("about \(minutes) min", "约 \(minutes) 分钟", "約 \(minutes) 分鐘", "約 \(minutes) 分", "약 \(minutes)분")
        }
        return "\(speed) · \(remaining)"
    }

    /// Short, plain copy for each starter failure; every one offers a retry.
    /// Raw failure codes are never shown.
    func failure(code: String) -> String {
        switch code {
        case StarterDownloadCoordinator.networkUnreachableCode:
            pick("Can't reach huggingface.co or hf-mirror.com. Check your network and try again.",
                 "连不上 huggingface.co，也连不上 hf-mirror.com。请检查网络后重试。",
                 "無法連線 huggingface.co，也無法連線 hf-mirror.com。請檢查網路後重試。",
                 "huggingface.co にも hf-mirror.com にも接続できません。ネットワークを確認して再試行してください。",
                 "huggingface.co와 hf-mirror.com 모두에 연결할 수 없습니다. 네트워크를 확인한 뒤 다시 시도하세요.")
        case "download_failed":
            pick("The download was interrupted. Check your network and try again.",
                 "下载中断了。请检查网络后重试。",
                 "下載中斷了。請檢查網路後重試。",
                 "ダウンロードが中断されました。ネットワークを確認して再試行してください。",
                 "다운로드가 중단되었습니다. 네트워크를 확인한 뒤 다시 시도하세요.")
        case ModelDownloadService.pinnedHashMismatchCode:
            pick("The downloaded files didn't match their checksums, so they weren't used. Try again, or switch networks if it keeps happening.",
                 "下载的文件和校验值对不上，没有使用。请重试；反复出现时换个网络再试。",
                 "下載的檔案與校驗值不符，沒有使用。請重試；反覆出現時換個網路再試。",
                 "ダウンロードしたファイルがチェックサムと一致しないため、使用しませんでした。再試行し、繰り返す場合は別のネットワークでお試しください。",
                 "내려받은 파일이 체크섬과 일치하지 않아 사용하지 않았습니다. 다시 시도하고, 계속되면 다른 네트워크에서 시도하세요.")
        case "starter_download_busy":
            pick("This model is already downloading in the Dashboard. Wait for it to finish, or cancel it there and try again.",
                 "这个模型正在 Dashboard 里下载。等它完成，或在 Dashboard 里取消后再重试。",
                 "這個模型正在 Dashboard 裡下載。等它完成，或在 Dashboard 裡取消後再重試。",
                 "このモデルはダッシュボードでダウンロード中です。完了を待つか、ダッシュボードでキャンセルしてから再試行してください。",
                 "이 모델은 대시보드에서 다운로드 중입니다. 끝날 때까지 기다리거나 대시보드에서 취소한 뒤 다시 시도하세요.")
        case StarterDownloadCoordinator.modelLoadFailedCode:
            pick("The model downloaded but couldn't be loaded. Try again, or open the Dashboard to see why.",
                 "模型已经下载好，但没能加载。请重试，或打开 Dashboard 查看原因。",
                 "模型已下載完成，但無法載入。請重試，或開啟 Dashboard 查看原因。",
                 "モデルはダウンロードできましたが、読み込めませんでした。再試行するか、ダッシュボードで原因を確認してください。",
                 "모델은 내려받았지만 불러오지 못했습니다. 다시 시도하거나 대시보드에서 원인을 확인하세요.")
        case "insufficient_disk":
            pick("There isn't enough disk space for this model. Free up some space and try again.",
                 "磁盘空间不够放下这个模型。清理一些空间后重试。",
                 "磁碟空間不足以存放這個模型。清出一些空間後重試。",
                 "このモデルを保存するディスク容量が足りません。空き容量を増やしてから再試行してください。",
                 "이 모델을 저장할 디스크 공간이 부족합니다. 공간을 확보한 뒤 다시 시도하세요.")
        case "insufficient_memory":
            pick("This Mac doesn't have enough free memory for this model right now. Quit some apps and try again.",
                 "这台 Mac 现在的可用内存不够加载这个模型。关掉一些应用后重试。",
                 "這台 Mac 目前的可用記憶體不足以載入這個模型。關閉一些應用程式後重試。",
                 "現在この Mac の空きメモリが足りません。いくつかのアプリを終了してから再試行してください。",
                 "지금 이 Mac의 사용 가능한 메모리가 부족합니다. 앱 몇 개를 종료한 뒤 다시 시도하세요.")
        default:
            // The code itself only goes to the log (StarterDownloadCoordinator).
            pick("Something went wrong while getting the model ready. Try again.",
                 "准备模型时出了点问题。请重试。",
                 "準備模型時出了點問題。請重試。",
                 "モデルの準備中に問題が発生しました。再試行してください。",
                 "모델을 준비하는 중 문제가 생겼습니다. 다시 시도하세요.")
        }
    }

    /// Downloaded bytes, in MB below 1 GB so a slow link visibly moves.
    static func downloadedSize(_ bytes: Int64) -> String {
        bytes < 1_000_000_000
            ? String(format: "%.1f MB", Double(max(0, bytes)) / 1_000_000)
            : gigabytes(bytes, digits: 2)
    }

    static func gigabytes(_ bytes: Int64, digits: Int) -> String {
        String(format: "%.\(digits)f GB", Double(max(0, bytes)) / 1_000_000_000)
    }

    static func percent(_ fraction: Double) -> String {
        "\(Int((max(0, min(1, fraction)) * 100).rounded(.down)))%"
    }
}
