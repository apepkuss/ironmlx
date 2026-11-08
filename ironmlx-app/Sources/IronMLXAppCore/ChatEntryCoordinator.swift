import Foundation

/// What the menu bar "Chat" item, a notification click or the Dashboard's
/// "Start chatting" step opens.
enum ChatEntryDecision: Equatable {
    case openChat(modelID: String, needsLoad: Bool)
    case showDownloadProgress
    case showNoModel
}

/// Implemented by the chat window, which opens on `modelID` and loads it first
/// when `needsLoad` is true.
@MainActor
protocol ChatWindowPresenting: AnyObject {
    func showChat(modelID: String, needsLoad: Bool)
}

enum ChatEntryCoordinator {
    /// Order: a loaded conversation model (the default first), then a starter
    /// download in progress, then a downloaded conversation model: the
    /// default, else a catalogue starter model (`starterModelIDs`), else the
    /// smallest one, so chat never loads a huge model just because the scan
    /// listed it first. A starter that is still loading wins over
    /// "downloaded" so chat never starts a second load of the same model.
    static func decide(
        loaded: [BackendLoadedModelInfo],
        localModels: [LocalModel],
        defaultModel: String?,
        starterState: StarterDownloadState,
        isConversationModel: (String) -> Bool,
        starterModelIDs: Set<String> = []
    ) -> ChatEntryDecision {
        let defaultModel = AppConfig.normalizedModelReference(defaultModel)
        let loadedConversation = loaded.filter { isConversationModel($0.id) }
        if !loadedConversation.isEmpty {
            let chosen = loadedConversation.first { $0.id == defaultModel }
                ?? loadedConversation.first(where: \.isDefault)
                ?? loadedConversation[0]
            return .openChat(modelID: chosen.id, needsLoad: false)
        }
        switch starterState {
        case .probing, .downloading, .paused, .loading:
            return .showDownloadProgress
        case .idle, .ready, .failed:
            break
        }
        let downloaded = localModels.filter {
            isConversationModel($0.id) && $0.readiness?.isLoadable != false
        }
        let smallestFirst = downloaded.sorted { $0.sizeMB < $1.sizeMB }
        if let chosen = downloaded.first(where: { $0.id == defaultModel })
            ?? smallestFirst.first(where: { starterModelIDs.contains($0.id) || starterModelIDs.contains($0.repoID) })
            ?? smallestFirst.first {
            return .openChat(modelID: chosen.id, needsLoad: true)
        }
        return .showNoModel
    }

    /// The repositories and IDs of every starter tier in the catalogue.
    static func starterModelIDs(catalog: SupportedModelCatalog?) -> Set<String> {
        guard let catalog, let starter = catalog.starter else { return [] }
        var ids: Set<String> = []
        for tier in starter.tiers {
            guard let entry = catalog.entries.first(where: { $0.id == tier.entryId }) else { continue }
            ids.formUnion([entry.id, entry.hfRepo] + [entry.msRepo].compactMap { $0 })
        }
        return ids
    }

    /// A conversation model has catalogue category `text` or `vision` with the
    /// `text` capability. Models outside the catalogue are judged by the type
    /// the local scan found, then by what the backend reports for them.
    static func conversationModelPredicate(
        catalog: SupportedModelCatalog?,
        localModels: [LocalModel],
        loaded: [BackendLoadedModelInfo]
    ) -> (String) -> Bool {
        var catalogued: [String: Bool] = [:]
        for entry in catalog?.entries ?? [] {
            let isConversation = ["text", "vision"].contains(entry.category) && entry.capabilities.contains("text")
            for key in [entry.hfRepo, entry.msRepo, entry.id].compactMap({ $0 }) where catalogued[key] == nil {
                catalogued[key] = isConversation
            }
        }
        var local: [String: Bool] = [:]
        for model in localModels {
            let isConversation = ["llm", "vlm", "block_diffusion_vlm"].contains(model.type)
                && isConversationRuntime(model.capabilities?.runtimeKind)
            local[model.id] = local[model.id] ?? isConversation
            local[model.repoID] = local[model.repoID] ?? isConversation
        }
        var backend: [String: Bool] = [:]
        for model in loaded where backend[model.id] == nil {
            backend[model.id] = isConversationRuntime(model.capabilities.runtimeKind)
        }
        return { id in catalogued[id] ?? local[id] ?? backend[id] ?? false }
    }

    private static func isConversationRuntime(_ kind: String?) -> Bool {
        guard let kind else { return true }
        return kind == "causal" || kind == "block_diffusion"
    }

    /// The host a starter download left paused on: a paused task, or a task
    /// the last App session was running when it quit (kept as a reminder).
    static func pausedStarterSource(in snapshot: ModelDownloadQueueSnapshot, repoID: String) -> DownloadSource? {
        let provider = ModelRepositoryProvider.huggingFace
        if let task = snapshot.tasks.first(where: {
            $0.provider == provider.rawValue && $0.repoID == repoID
        }) {
            guard task.status == ModelDownloadPhase.paused.rawValue else { return nil }
            return task.useHfMirror == true ? .hfMirror : .huggingFace
        }
        if let reminder = snapshot.recoveryReminders.first(where: { $0.provider == provider && $0.repoID == repoID }) {
            return reminder.useHfMirror == true ? .hfMirror : .huggingFace
        }
        return nil
    }

    /// Records a loaded starter model the way a Dashboard load does, so the
    /// next launch restores it as the default. A pin is never dropped while
    /// its model is still loaded, even when the backend's list does not report
    /// the pin. Without the backend's loaded list (`nil`), only adds the model
    /// and makes it the default; pinned models are never touched then.
    static func persistStarterModel(_ modelID: String, loaded: [BackendLoadedModelInfo]?, into config: inout AppConfig) {
        guard let loaded else {
            var references = AppConfig.normalizedModelReferences(config.loadedModels ?? [])
            if !references.contains(modelID) { references.append(modelID) }
            config.loadedModels = references
            config.defaultModel = modelID
            return
        }
        let pins = config.pinnedModelReferences
        let ids = loaded.map(\.id)
        config.replaceLoadedModels(ids.contains(modelID) ? ids : ids + [modelID], defaultModel: modelID)
        config.replacePinnedModels(pins + loaded.filter(\.pinned).map(\.id))
        config.defaultModel = modelID
    }
}
