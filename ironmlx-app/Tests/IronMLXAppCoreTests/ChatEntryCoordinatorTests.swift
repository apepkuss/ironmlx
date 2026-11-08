import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Suite("Chat entry decision")
struct ChatEntryCoordinatorTests {
    private static let starter = "mlx-community/Qwen3.5-4B-MLX-4bit"
    private static let other = "mlx-community/Qwen3.5-2B-4bit"
    private static let tts = "mlx-community/Kokoro-82M-bf16"
    private static let decision = "mlx-community/Laya-Decision"
    private static let conversation: Set<String> = [starter, other]

    private func loaded(_ id: String, isDefault: Bool = false) -> BackendLoadedModelInfo {
        BackendLoadedModelInfo(
            id: id, model: id, path: "/models/\(id)", architecture: "qwen3_5",
            isDefault: isDefault, maxPositionEmbeddings: 32_768
        )
    }

    private func local(_ id: String) -> LocalModel {
        LocalModel(id: id, repoID: id, source: "hf", sizeMB: 1_000)
    }

    private func decide(
        loaded: [BackendLoadedModelInfo] = [],
        local: [LocalModel] = [],
        defaultModel: String? = nil,
        starter: StarterDownloadState = .idle
    ) -> ChatEntryDecision {
        ChatEntryCoordinator.decide(
            loaded: loaded,
            localModels: local,
            defaultModel: defaultModel,
            starterState: starter,
            isConversationModel: { Self.conversation.contains($0) }
        )
    }

    @Test func loadedDefaultConversationModelOpensChat() {
        let decision = decide(
            loaded: [loaded(Self.other), loaded(Self.starter, isDefault: true)],
            local: [local(Self.other), local(Self.starter)],
            defaultModel: Self.starter
        )
        #expect(decision == .openChat(modelID: Self.starter, needsLoad: false))
    }

    @Test func loadedNonDefaultConversationModelOpensChatWhenDefaultNotLoaded() {
        let decision = decide(
            loaded: [loaded(Self.tts, isDefault: true), loaded(Self.other)],
            local: [local(Self.starter), local(Self.other)],
            defaultModel: Self.starter
        )
        #expect(decision == .openChat(modelID: Self.other, needsLoad: false))
    }

    @Test func onlyTTSAndDecisionLoadedTreatedAsNoModel() {
        let decision = decide(
            loaded: [loaded(Self.tts, isDefault: true), loaded(Self.decision)],
            local: [local(Self.tts), local(Self.decision)],
            defaultModel: Self.tts
        )
        #expect(decision == .showNoModel)
    }

    @Test func downloadedButNotLoadedOpensChatWithLoad() {
        let preferDefault = decide(
            loaded: [loaded(Self.tts)],
            local: [local(Self.other), local(Self.starter)],
            defaultModel: Self.starter
        )
        #expect(preferDefault == .openChat(modelID: Self.starter, needsLoad: true))

        let anyConversation = decide(local: [local(Self.tts), local(Self.other)], defaultModel: Self.tts)
        #expect(anyConversation == .openChat(modelID: Self.other, needsLoad: true))
    }

    @Test func withoutADefaultTheStarterThenTheSmallestModelIsLoaded() {
        let big = LocalModel(id: "big/Model-120B", repoID: "big/Model-120B", source: "hf", sizeMB: 70_000)
        let small = LocalModel(id: Self.other, repoID: Self.other, source: "hf", sizeMB: 1_500)
        let starter = LocalModel(id: Self.starter, repoID: Self.starter, source: "hf", sizeMB: 3_000)
        func decide(_ local: [LocalModel], defaultModel: String? = nil) -> ChatEntryDecision {
            ChatEntryCoordinator.decide(
                loaded: [], localModels: local, defaultModel: defaultModel, starterState: .idle,
                isConversationModel: { _ in true }, starterModelIDs: [Self.starter]
            )
        }
        // Scan order never decides: the starter first, then the smallest model.
        #expect(decide([big, small, starter]) == .openChat(modelID: Self.starter, needsLoad: true))
        #expect(decide([big, small]) == .openChat(modelID: Self.other, needsLoad: true))
        // The user's default still wins.
        #expect(decide([small, big, starter], defaultModel: big.id) == .openChat(modelID: big.id, needsLoad: true))
    }

    @Test func starterModelIDsComeFromEveryStarterTier() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let ids = ChatEntryCoordinator.starterModelIDs(catalog: catalog)
        let starter = try #require(catalog.starter)
        for tier in starter.tiers {
            let entry = try #require(catalog.entries.first { $0.id == tier.entryId })
            #expect(ids.contains(entry.hfRepo))
        }
        #expect(ChatEntryCoordinator.starterModelIDs(catalog: nil).isEmpty)
    }

    @Test func unloadableLocalModelIsSkipped() {
        var broken = local(Self.starter)
        broken.readiness = LocalModelReadiness(status: "incomplete", reasonCode: "missing_files")
        #expect(decide(local: [broken]) == .showNoModel)
    }

    @Test func starterDownloadingShowsProgress() {
        let states: [StarterDownloadState] = [
            .probing,
            .downloading(source: .hfMirror, fraction: 0.42, bytes: 42, total: 100, bytesPerSecond: 5),
            .loading,
        ]
        for state in states {
            #expect(decide(starter: state) == .showDownloadProgress)
        }
        // Loading the finished starter must not race a second load from chat.
        #expect(decide(local: [local(Self.starter)], defaultModel: Self.starter, starter: .loading)
            == .showDownloadProgress)
    }

    @Test func pausedStarterDownloadShowsProgressNotNoModel() throws {
        // After a quit mid-download the service restores the starter task as a
        // reminder that keeps the host it used; the App replays it as paused.
        let snapshot = ModelDownloadQueueSnapshot(
            maxConcurrent: 3, activeCount: 0, queuedCount: 0, tasks: [],
            recoveryReminders: [Self.reminder(Self.starter, previousStatus: "downloading", mirror: true)]
        )
        let restored = try #require(ChatEntryCoordinator.pausedStarterSource(in: snapshot, repoID: Self.starter))
        #expect(restored == .hfMirror)
        #expect(decide(starter: .paused(source: restored)) == .showDownloadProgress)
    }

    @Test func pausedStarterTaskKeepsItsOriginalSource() {
        var paused = ModelDownloadStatus(repoID: Self.starter, provider: "huggingface", status: "paused", progressPct: 40)
        paused.useHfMirror = true
        var other = ModelDownloadStatus(repoID: Self.other, provider: "huggingface", status: "paused", progressPct: 10)
        other.useHfMirror = false
        let snapshot = ModelDownloadQueueSnapshot(
            maxConcurrent: 3, activeCount: 0, queuedCount: 0, tasks: [other, paused], recoveryReminders: []
        )
        #expect(ChatEntryCoordinator.pausedStarterSource(in: snapshot, repoID: Self.starter) == .hfMirror)

        paused.useHfMirror = nil
        let official = ModelDownloadQueueSnapshot(
            maxConcurrent: 3, activeCount: 0, queuedCount: 0, tasks: [paused], recoveryReminders: []
        )
        #expect(ChatEntryCoordinator.pausedStarterSource(in: official, repoID: Self.starter) == .huggingFace)

        paused.status = "failed"
        let failed = ModelDownloadQueueSnapshot(
            maxConcurrent: 3, activeCount: 0, queuedCount: 0, tasks: [paused], recoveryReminders: []
        )
        #expect(ChatEntryCoordinator.pausedStarterSource(in: failed, repoID: Self.starter) == nil)
    }

    @Test func nothingShowsNoModel() {
        #expect(decide() == .showNoModel)
        #expect(decide(starter: .failed(code: "network_unreachable")) == .showNoModel)
    }

    @Test func conversationModelsFollowCatalogueCategoryThenLocalType() throws {
        let catalog = try SupportedModelCatalog.bundled()
        var tts = local("someone/custom-tts")
        tts.type = "tts"
        var decisionModel = local("someone/laya")
        decisionModel.capabilities = BackendModelCapabilities(
            runtimeKind: "decision", supportsStreaming: false, supportsVision: false, supportsMtp: false,
            supportsPromptLookup: false, supportsSpeculativeDecoding: false, supportsKvCache: false,
            supportedSamplingParameters: []
        )
        var vlm = local("someone/custom-vlm")
        vlm.type = "vlm"
        let isConversation = ChatEntryCoordinator.conversationModelPredicate(
            catalog: catalog,
            localModels: [tts, decisionModel, vlm, local("someone/custom-llm")],
            loaded: []
        )
        #expect(isConversation(Self.starter))
        #expect(!isConversation(try #require(catalog.entries.first { $0.category == "tts" }?.hfRepo)))
        #expect(!isConversation(try #require(catalog.entries.first { $0.category == "embedding" }?.hfRepo)))
        #expect(isConversation("someone/custom-llm"))
        #expect(isConversation("someone/custom-vlm"))
        #expect(!isConversation("someone/custom-tts"))
        #expect(!isConversation("someone/laya"))
        #expect(!isConversation("someone/unknown"))
    }

    @Test func starterReadyPersistsDefaultAndLoadedModels() {
        var config = AppConfig(defaultModel: nil, loadedModels: [])
        ChatEntryCoordinator.persistStarterModel(
            Self.starter,
            loaded: [loaded(Self.tts), loaded(Self.starter, isDefault: true)],
            into: &config
        )
        #expect(config.defaultModelReference == Self.starter)
        #expect(config.restoredModelReferences == [Self.starter, Self.tts])
        #expect(AppLaunchPlanner().plan(config: config, localModels: []).backendModelReferences.first == Self.starter)
    }

    @Test func starterReadyWithoutLoadedListKeepsPinnedAndLoadedModels() {
        var config = AppConfig(
            defaultModel: Self.other,
            loadedModels: [Self.other, Self.tts],
            pinnedModels: [Self.other, "someone/not-loaded"]
        )
        ChatEntryCoordinator.persistStarterModel(Self.starter, loaded: nil, into: &config)
        #expect(config.defaultModelReference == Self.starter)
        #expect(config.loadedModels == [Self.other, Self.tts, Self.starter])
        #expect(config.pinnedModels == [Self.other, "someone/not-loaded"])

        // Persisting twice does not duplicate the model.
        ChatEntryCoordinator.persistStarterModel(Self.starter, loaded: nil, into: &config)
        #expect(config.loadedModels == [Self.other, Self.tts, Self.starter])
    }

    @Test func starterReadyKeepsPinsTheLoadedListDoesNotReport() {
        var config = AppConfig(
            defaultModel: Self.other,
            loadedModels: [Self.other, Self.tts],
            pinnedModels: [Self.other]
        )
        // Like chat's default change: a pin the backend list leaves out stays.
        ChatEntryCoordinator.persistStarterModel(
            Self.starter,
            loaded: [loaded(Self.other), loaded(Self.tts), loaded(Self.starter, isDefault: true)],
            into: &config
        )
        #expect(config.defaultModelReference == Self.starter)
        #expect(config.pinnedModels == [Self.other])
    }

    private static func reminder(_ repoID: String, previousStatus: String, mirror: Bool) -> ModelDownloadRecoveryReminder {
        var reminder = ModelDownloadRecoveryReminder(
            provider: .huggingFace, repoID: repoID, queueOrder: 0,
            previousStatus: previousStatus, usedCredential: false
        )
        reminder.useHfMirror = mirror
        return reminder
    }
}

@Suite("No-model window copy")
struct NoModelWindowTextTests {
    private static let languages = ["en", "zh-Hans", "zh-Hant", "ja", "ko"]

    @Test func everyStarterFailureHasItsOwnCopyInEveryLanguage() {
        let codes = ["network_unreachable", "download_failed", "starter_hash_mismatch", "starter_download_busy",
                     "model_load_failed", "insufficient_disk", "insufficient_memory"]
        for language in Self.languages {
            let text = NoModelWindowText(language: language)
            let generic = text.failure(code: "something_else")
            let messages = codes.map { text.failure(code: $0) }
            #expect(Set(messages).count == codes.count, "\(language)")
            #expect(!messages.contains(generic), "\(language)")
            // Raw codes stay in the log; the window shows plain words only.
            for message in messages + [generic] {
                #expect(!message.contains("_"), "\(language): \(message)")
                #expect(!message.contains("something"), "\(language): \(message)")
            }
        }
        #expect(NoModelWindowText(language: "en").failure(code: "download_failed")
            == "The download was interrupted. Check your network and try again.")
        #expect(NoModelWindowText(language: "zh-Hans").failure(code: "unexpected_code")
            == "准备模型时出了点问题。请重试。")
        #expect(NoModelWindowText(language: "zh-Hans").failure(code: "network_unreachable")
            == "连不上 huggingface.co，也连不上 hf-mirror.com。请检查网络后重试。")
    }

    @Test func readyKeepsTheProgressViewUntilChatOpens() {
        #expect(NoModelWindowModel.showsProgress(.ready(modelID: "m")))
        let downloading = StarterDownloadState.downloading(source: .huggingFace, fraction: 0.5, bytes: 1, total: 2, bytesPerSecond: nil)
        var phrases: Set<String> = []
        for language in Self.languages {
            let text = NoModelWindowText(language: language)
            let ready = text.phase(.ready(modelID: "m"))
            #expect(ready != text.phase(downloading), "\(language)")
            #expect(ready != text.offerTitle, "\(language)")
            phrases.insert(ready)
        }
        #expect(phrases.count == Self.languages.count)
        #expect(NoModelWindowText(language: "zh-Hans").phase(.ready(modelID: "m")) == "已就绪，正在打开对话…")
    }

    @MainActor
    @Test func closingTheWindowWhenReadyIsNotLeavingADownload() {
        let controller = NoModelWindowController(model: NoModelWindowModel(
            recommendation: nil, physicalMemoryBytes: 16 << 30, language: "en"
        ))
        var leftDownload = 0
        controller.onClosedWhileDownloading = { leftDownload += 1 }
        controller.update(.ready(modelID: "m"))
        controller.windowWillClose(Notification(name: NSWindow.willCloseNotification))
        #expect(leftDownload == 0)
        controller.update(.loading)
        controller.windowWillClose(Notification(name: NSWindow.willCloseNotification))
        #expect(leftDownload == 1)
    }

    @Test func sourceLineExplainsTheMirrorOnlyForAFreshDownload() {
        let zh = NoModelWindowText(language: "zh-Hans")
        #expect(zh.sourceLine(.downloading(source: .hfMirror, fraction: 0.4, bytes: 4, total: 10, bytesPerSecond: nil))
            == "来源：hf-mirror.com（huggingface.co 当前无法访问）")
        #expect(zh.sourceLine(.downloading(source: .huggingFace, fraction: 0.4, bytes: 4, total: 10, bytesPerSecond: nil))
            == "来源：huggingface.co")
        #expect(zh.sourceLine(.paused(source: .hfMirror)) == "来源：hf-mirror.com")
        #expect(zh.sourceLine(.idle) == nil)
    }

    @Test func pausedProgressShowsTheBytesOnDiskWithoutARate() throws {
        let paused = try #require(StarterDownloadState.paused(source: .huggingFace, bytes: 226_000_000, total: 3_061_129_077).progress)
        #expect(paused.bytes == 226_000_000 && paused.total == 3_061_129_077 && paused.bytesPerSecond == nil)
        #expect(NoModelWindowText.percent(paused.fraction) == "7%")
        #expect(StarterDownloadState.paused(source: .huggingFace).progress?.fraction == 0)
        #expect(StarterDownloadState.loading.progress == nil)
    }

    @Test func downloadingCopyPromisesChatOnlyWhileTheWindowIsOpen() {
        #expect(NoModelWindowText(language: "zh-Hans").preparingDescription
            == "完成后会直接进入对话；关掉这个窗口也没关系，准备好后会通知你。")
        for language in Self.languages {
            let text = NoModelWindowText(language: language).preparingDescription
            #expect(!text.isEmpty)
            #expect(text != NoModelWindowText(language: "en").preparingDescription || language == "en")
        }
        #expect(NoModelWindowText(language: "en").preparingDescription.contains("notified"))
    }

    @Test func recommendationCardMatchesTheMockup() throws {
        let catalog = try SupportedModelCatalog.bundled()
        let starter = try #require(StarterModelSelector.initial(catalog: catalog, physicalMemoryBytes: 16 << 30))
        let zh = NoModelWindowText(language: "zh-Hans")
        #expect(zh.reason(physicalMemoryBytes: 16 << 30, capabilities: starter.entry.capabilities)
            == "适合你这台 16 GB 内存的 Mac：回复快、中英文都好，还能看图。")
        #expect(zh.tags(for: starter.entry.capabilities) == ["对话", "看图", "可开启思考"])
        #expect(zh.rateAndRemaining(bytes: 1_260_000_000, total: 3_000_000_000, bytesPerSecond: 18_400_000)
            == "18.4 MB/s · 约 2 分钟")
        // Without a meaningful rate sample only bytes and percent show.
        #expect(zh.rateAndRemaining(bytes: 1_260_000_000, total: 3_000_000_000, bytesPerSecond: nil) == "")
        // Slow links show KB/s, never "0.0 MB/s".
        #expect(zh.rateAndRemaining(bytes: 2_900_000_000, total: 3_000_000_000, bytesPerSecond: 45_000)
            == "45 KB/s · 约 38 分钟")
        #expect(zh.rateAndRemaining(bytes: 0, total: 3_000_000_000, bytesPerSecond: 300).hasPrefix("1 KB/s"))
        // Downloaded bytes move visibly on slow links: MB below 1 GB.
        #expect(NoModelWindowText.downloadedSize(226_440_000) == "226.4 MB")
        #expect(NoModelWindowText.downloadedSize(1_260_000_000) == "1.26 GB")
        #expect(NoModelWindowText.gigabytes(1_260_000_000, digits: 2) == "1.26 GB")
        for language in Self.languages {
            let text = NoModelWindowText(language: language)
            #expect(text.reason(physicalMemoryBytes: 8 << 30, capabilities: ["text"]).contains("8"))
            #expect(!text.offerTitle.isEmpty && !text.notificationNote.isEmpty)
        }
    }
}
