import Foundation

// MARK: - Vocabulary

/// Where an App load started. Backend loads the App did not start are `external`.
public enum ModelLoadOrigin: String, Codable, Equatable, Sendable {
    case dashboard
    case chat
    case starter
    case restore
    case benchmark
    case external
}

/// Steps the App performs before or around a backend load; the backend cannot see them.
public enum ModelLoadAppStage: String, Codable, Equatable, Sendable {
    case verifyingFiles = "verifying_files"
    /// The backend is started (if needed) and answers.
    case waitingForBackend = "waiting_for_backend"
    /// Local models are registered with the backend before the load.
    case syncingModels = "syncing_models"
    case stoppingBackend = "stopping_backend"
    case restartingBackend = "restarting_backend"
}

public enum ModelLoadVariant: String, Codable, Equatable, Sendable {
    case plain
    case mtp
    case dflash2
}

public enum ModelLoadKind: String, Codable, Equatable, Sendable {
    case load
    case reload
}

/// Same values as the backend's `LoadState`.
public enum ModelLoadState: String, Codable, Equatable, Sendable {
    case queued
    case loading
    case cancelling
    case loaded
    case failed
    case cancelled

    public var isTerminal: Bool {
        self == .loaded || self == .failed || self == .cancelled
    }
}

/// The App stages followed by the backend's `LoadStage` values, in load order.
public enum ModelLoadStage: String, Codable, Equatable, Sendable {
    case verifyingFiles = "verifying_files"
    case waitingForBackend = "waiting_for_backend"
    case syncingModels = "syncing_models"
    case stoppingBackend = "stopping_backend"
    case restartingBackend = "restarting_backend"
    case queued
    case preparingMemory = "preparing_memory"
    case readingWeights = "reading_weights"
    case buildingModel = "building_model"
    case readingDraftWeights = "reading_draft_weights"
    case startingEngine = "starting_engine"
    case warmingUp = "warming_up"

    init(_ stage: ModelLoadAppStage) {
        self = ModelLoadStage(rawValue: stage.rawValue)!
    }
}

/// The backend's `LoadTriggerKind`.
public enum BackendLoadTrigger: String, Codable, Equatable, Sendable {
    case admin
    case openaiControl = "openai_control"
    case request
    case deferredReload = "deferred_reload"
    case startup
}

/// One entry of `GET /admin/api/models/loads` (`LoadActivity` in
/// `ironmlx-runtime/src/core/load_registry.rs`).
public struct BackendLoadActivity: Codable, Equatable, Sendable {
    public struct CompletedStage: Codable, Equatable, Sendable {
        public var stage: ModelLoadStage
        public var elapsedMs: UInt64

        public init(stage: ModelLoadStage, elapsedMs: UInt64) {
            self.stage = stage
            self.elapsedMs = elapsedMs
        }

        enum CodingKeys: String, CodingKey {
            case stage
            case elapsedMs = "elapsed_ms"
        }
    }

    public var loadID: String
    public var model: String
    public var variant: ModelLoadVariant
    public var kind: ModelLoadKind
    public var trigger: BackendLoadTrigger
    public var state: ModelLoadState
    public var stage: ModelLoadStage
    public var completedStages: [CompletedStage]
    public var bytesDone: UInt64
    public var bytesTotal: UInt64
    public var startedUnixMs: UInt64
    public var finishedUnixMs: UInt64?
    public var message: String?
    public var code: String?
    /// A caveat about a loaded model (`scheduler_profile_hardware_mismatch`, `scheduler_profile_unusable`).
    public var warningCode: String?
    public var warning: String?

    public init(
        loadID: String,
        model: String,
        variant: ModelLoadVariant,
        kind: ModelLoadKind,
        trigger: BackendLoadTrigger,
        state: ModelLoadState,
        stage: ModelLoadStage,
        completedStages: [CompletedStage] = [],
        bytesDone: UInt64 = 0,
        bytesTotal: UInt64 = 0,
        startedUnixMs: UInt64,
        finishedUnixMs: UInt64? = nil,
        message: String? = nil,
        code: String? = nil,
        warningCode: String? = nil,
        warning: String? = nil
    ) {
        self.loadID = loadID
        self.model = model
        self.variant = variant
        self.kind = kind
        self.trigger = trigger
        self.state = state
        self.stage = stage
        self.completedStages = completedStages
        self.bytesDone = bytesDone
        self.bytesTotal = bytesTotal
        self.startedUnixMs = startedUnixMs
        self.finishedUnixMs = finishedUnixMs
        self.message = message
        self.code = code
        self.warningCode = warningCode
        self.warning = warning
    }

    enum CodingKeys: String, CodingKey {
        case loadID = "load_id"
        case model
        case variant
        case kind
        case trigger
        case state
        case stage
        case completedStages = "completed_stages"
        case bytesDone = "bytes_done"
        case bytesTotal = "bytes_total"
        case startedUnixMs = "started_unix_ms"
        case finishedUnixMs = "finished_unix_ms"
        case message
        case code
        case warningCode = "warning_code"
        case warning
    }
}

/// Body of `GET /admin/api/models/loads`. An entry this App cannot read (a
/// newer backend's stage, say) is skipped instead of hiding every load.
struct BackendLoadActivityList: Decodable {
    var loads: [BackendLoadActivity]

    private enum CodingKeys: String, CodingKey {
        case loads
    }

    private struct Skipped: Decodable {
        init(from decoder: Decoder) throws {}
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        var entries = try container.nestedUnkeyedContainer(forKey: .loads)
        var loads: [BackendLoadActivity] = []
        while !entries.isAtEnd {
            if let load = try? entries.decode(BackendLoadActivity.self) {
                loads.append(load)
            } else {
                _ = try entries.decode(Skipped.self)
            }
        }
        self.loads = loads
    }
}

public struct ModelLoadCompletedStage: Codable, Equatable, Sendable {
    public var stage: ModelLoadStage
    public var elapsedMs: UInt64

    public init(stage: ModelLoadStage, elapsedMs: UInt64) {
        self.stage = stage
        self.elapsedMs = elapsedMs
    }
}

/// The merged state of one model's load that the Dashboard renders
/// (`onModelLoadsChanged`). `startedAt` is Unix milliseconds. A cancelled
/// load's `code` says what it left behind (`ModelLoadCancelledError` codes).
public struct ModelLoadActivity: Codable, Equatable, Sendable {
    public var model: String
    public var variant: ModelLoadVariant
    public var kind: ModelLoadKind
    public var origin: ModelLoadOrigin
    public var state: ModelLoadState
    public var stage: ModelLoadStage?
    public var completedStages: [ModelLoadCompletedStage]
    public var bytesDone: UInt64
    public var bytesTotal: UInt64
    public var startedAt: UInt64
    public var message: String?
    public var code: String?
    public var cancellable: Bool
    public var autoOpenPanel: Bool
}

/// Payload of the Dashboard callback `onModelLoadsChanged(json)`.
struct ModelLoadActivityList: Encodable {
    var loads: [ModelLoadActivity]
}

public struct ModelLoadTicket: Hashable, Sendable {
    public let id: UUID
    public let model: String
}

public enum ModelLoadOutcome: Equatable, Sendable {
    case loaded
    case failed(message: String, code: String?)
    case cancelled(code: String)
    /// Ends the ticket without a result to show (for example a reload the
    /// backend deferred until the model is idle).
    case dismissed
}

public enum BackendLoadCancelResult: Equatable, Sendable {
    case cancelling
    case notFound
    case finished
    case finishing
}

/// How `POST /admin/api/models/load` with `async: true` answered.
public enum BackendModelLoadStart: Sendable {
    /// `202`: the load runs in the backend under `loadID`, which differs from
    /// the requested id when the request joined an identical load.
    case accepted(loadID: String, state: ModelLoadState)
    /// `200`: nothing left to wait for (`loaded`, `reloaded`, `already_loaded`
    /// or `reload_deferred`).
    case completed(BackendModelAdminResponse)
}

/// What the center needs from the backend.
public protocol BackendModelLoadTracking: Sendable {
    func fetchModelLoads() async throws -> [BackendLoadActivity]
    func cancelModelLoad(loadID: String) async throws -> BackendLoadCancelResult
    func startModelLoad(_ request: BackendLoadModelRequest) async throws -> BackendModelLoadStart
    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo]
}

public struct ModelLoadCancelledError: LocalizedError, Equatable, DashboardErrorCodeProviding {
    /// A new load was cancelled; the model is not loaded.
    public static let loadCancelledCode = "load_cancelled"
    /// A reload was cancelled; the version loaded before stays in use.
    public static let reloadCancelledCode = "reload_cancelled"
    /// A Dashboard switch to DFlash2 was cancelled; the previous configuration
    /// was restored and the backend restarted with it.
    public static let dflash2RestoredCode = "dflash2_cancelled_restored"
    /// Restoring a DFlash2 model at App start or backend restart was
    /// cancelled; the backend runs without models and the preference stays.
    public static let dflash2StartupCode = "dflash2_startup_cancelled"

    public var code: String

    public init(code: String) {
        self.code = code
    }

    public var previousVersionKept: Bool {
        code == Self.reloadCancelledCode
    }

    public var dashboardErrorCode: String {
        code
    }

    public var errorDescription: String? {
        switch code {
        case Self.reloadCancelledCode:
            return "The reload was cancelled; the previous version stays loaded."
        case Self.dflash2RestoredCode:
            return "The DFlash2 load was cancelled; the previous configuration was restored."
        case Self.dflash2StartupCode:
            return "The DFlash2 load was cancelled; the backend started without models."
        default:
            return "The model load was cancelled."
        }
    }
}

public struct ModelLoadFailedError: LocalizedError, Equatable, DashboardErrorCodeProviding {
    /// The backend stopped listing the load before the App saw its result,
    /// and the model is not loaded.
    public static let loadResultUnknownCode = "load_result_unknown"

    public var message: String
    public var code: String?

    public init(message: String, code: String?) {
        self.message = message
        self.code = code
    }

    public var dashboardErrorCode: String {
        code ?? "operation_failed"
    }

    public var errorDescription: String? {
        message
    }
}

public enum ModelLoadWaitError: LocalizedError, Equatable {
    case loadNotFound(String)
    case backendUnreachable

    public var errorDescription: String? {
        switch self {
        case .loadNotFound(let loadID):
            return "The backend no longer reports model load \(loadID)."
        case .backendUnreachable:
            return "The backend stopped answering while the model was loading."
        }
    }
}

/// The code and load named by an error body of `POST /admin/api/models/load`.
struct BackendLoadErrorBody: Decodable {
    var code: String?
    var loadID: String?
    var error: String?

    enum CodingKeys: String, CodingKey {
        case code
        case loadID = "load_id"
        case error
    }

    init?(_ error: Error) {
        guard case BackendAPIError.serverResponse(_, let body) = error,
              let data = body?.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(Self.self, from: data)
        else { return nil }
        self = decoded
    }
}

// MARK: - Center

/// App-wide record of model loads: the App stages of loads the App starts,
/// merged with the backend's load registry, which it polls while anything is
/// loading (and slowly while the Dashboard is visible, to see loads others
/// start). Every App load entry point goes through here.
@MainActor
public final class ModelLoadActivityCenter {
    public typealias ClientProvider = @Sendable () -> any BackendModelLoadTracking
    public typealias Sleep = @Sendable (Duration) async -> Void
    /// The default sleep, defined here rather than as a closure in the
    /// default arguments: those are compiled into each caller, and with
    /// Swift 6.3 such a caller-compiled closure aborted the poll task
    /// ("freed pointer was not the last allocation").
    public static let taskSleep: Sleep = { try? await Task.sleep(for: $0) }

    static let activePollInterval: Duration = .milliseconds(500)
    static let idlePollInterval: Duration = .seconds(2)
    /// Loaded and cancelled App loads stay listed as long as the backend
    /// keeps finished entries, so the Dashboard can show the result.
    static let finishedRetention: TimeInterval = 30
    /// A waited-for load the backend stops listing fails the wait after this
    /// many polls (fewer once it has been seen).
    static let missingPollsBeforeSeen = 6
    static let missingPollsAfterSeen = 3
    /// About ten seconds of failed polls while a wait is pending.
    static let unreachablePolls = 20
    static let maximumStartAttempts = 6

    private struct Record {
        let ticket: ModelLoadTicket
        var variant: ModelLoadVariant
        var kind: ModelLoadKind
        let origin: ModelLoadOrigin
        let startedAt: Date
        var autoOpenPanel: Bool
        var appStage: ModelLoadAppStage?
        var appStageStartedAt: Date
        var appCompleted: [ModelLoadCompletedStage] = []
        var backendLoadID: String?
        /// The id of a start request still in flight.
        var startingLoadID: String?
        var backend: BackendLoadActivity?
        /// A DFlash2 launch's own load: the starting server has no cancel
        /// endpoint, so a cancel stops the App task (which stops the launch).
        var adoptedStartupLoad = false
        var cancelRequested = false
        var cancelHandler: (@MainActor () -> Void)?
        var cancelTask: (@MainActor () -> Void)?
        var dismissOnSuccess = false
        var finished: (state: ModelLoadState, message: String?, code: String?, at: Date)?
    }

    private struct Waiter {
        let loadID: String
        let continuation: CheckedContinuation<BackendLoadActivity, Error>
        var seen = false
        var missingPolls = 0
    }

    private let clientProvider: ClientProvider
    private let notificationCenter: NotificationCenter
    private let pollSleep: Sleep
    private let now: @Sendable () -> Date
    private let retentionSleep: Sleep
    /// An App load changes the model's backend configuration, so the
    /// registration recorded for it no longer applies.
    private let registrationLedger: LocalModelRegistrationLedger
    private var records: [Record] = []
    private var waiters: [UUID: Waiter] = [:]
    private var backendLoads: [BackendLoadActivity] = []
    /// Loads that belong to App records, so they never show as external.
    private var appLoadIDs: Set<String> = []
    private var externalCancelling: Set<String> = []
    private var dashboardVisible = false
    private var consecutiveFetchFailures = 0
    private var pollTask: Task<Void, Never>?
    private var pollGeneration = 0
    private var sleepingInterval: Duration?
    private var published: [ModelLoadActivity] = []

    public init(
        client: @escaping ClientProvider,
        notificationCenter: NotificationCenter = .default,
        pollSleep: @escaping Sleep = ModelLoadActivityCenter.taskSleep,
        now: @escaping @Sendable () -> Date = { Date() },
        retentionSleep: @escaping Sleep = ModelLoadActivityCenter.taskSleep,
        registrationLedger: LocalModelRegistrationLedger = .shared
    ) {
        self.clientProvider = client
        self.notificationCenter = notificationCenter
        self.pollSleep = pollSleep
        self.now = now
        self.retentionSleep = retentionSleep
        self.registrationLedger = registrationLedger
    }

    /// Talks to the backend the config names at the time of each call.
    public convenience init(configStore: AppConfigStore, notificationCenter: NotificationCenter = .default) {
        self.init(
            client: {
                let config = configStore.load()
                return BackendAPIClient(host: config.host, port: config.port)
            },
            notificationCenter: notificationCenter
        )
    }

    var isPolling: Bool {
        pollTask != nil
    }

    // MARK: Ticket lifecycle

    /// Starts tracking an App load. A new load replaces a failure still shown
    /// for the same model. Only a Dashboard load (not a reload) opens the panel.
    public func begin(
        model: String,
        variant: ModelLoadVariant,
        origin: ModelLoadOrigin,
        kind: ModelLoadKind = .load
    ) -> ModelLoadTicket {
        let ticket = ModelLoadTicket(id: UUID(), model: model)
        records.removeAll { $0.ticket.model == model && $0.finished != nil }
        let date = now()
        records.append(Record(
            ticket: ticket,
            variant: variant,
            kind: kind,
            origin: origin,
            startedAt: date,
            autoOpenPanel: origin == .dashboard && kind == .load,
            appStageStartedAt: date
        ))
        publishIfChanged()
        ensurePolling()
        return ticket
    }

    public func enterAppStage(_ ticket: ModelLoadTicket, _ stage: ModelLoadAppStage) {
        guard let index = activeIndex(ticket) else { return }
        completeAppStage(at: index)
        records[index].appStage = stage
        records[index].appStageStartedAt = now()
        publishIfChanged()
    }

    /// The backend now runs this ticket's load as `loadID`.
    public func attachBackendLoad(_ ticket: ModelLoadTicket, loadID: String) {
        guard let index = activeIndex(ticket) else { return }
        completeAppStage(at: index)
        records[index].backendLoadID = loadID
        records[index].startingLoadID = nil
        records[index].backend = backendLoads.first { $0.loadID == loadID }
        appLoadIDs.insert(loadID)
        if records[index].cancelRequested, records[index].cancelHandler == nil {
            // Cancelled while the start request was in flight.
            Task { await self.requestBackendCancel(loadID: loadID, ticket: ticket) }
        }
        publishIfChanged()
        ensurePolling()
    }

    /// Replaces the default cancel behaviour (cancel the App task, or the
    /// backend load once attached) for loads that need their own, such as a
    /// DFlash2 backend launch.
    public func setCancelHandler(_ ticket: ModelLoadTicket, _ handler: @escaping @MainActor () -> Void) {
        guard let index = activeIndex(ticket) else { return }
        records[index].cancelHandler = handler
    }

    /// Ends a ticket; the first result wins. A failure keeps the App stage it
    /// happened in as the current stage, so the panel can mark it.
    public func finish(_ ticket: ModelLoadTicket, _ outcome: ModelLoadOutcome) {
        guard let index = activeIndex(ticket) else { return }
        if case .failed = outcome {} else {
            completeAppStage(at: index)
        }
        let date = now()
        switch outcome {
        case .dismissed:
            records.remove(at: index)
        case .loaded where records[index].dismissOnSuccess:
            records.remove(at: index)
        case .loaded:
            records[index].finished = (.loaded, nil, nil, date)
        case let .failed(message, code):
            records[index].finished = (.failed, message, code, date)
        case let .cancelled(code):
            records[index].finished = (.cancelled, nil, code, date)
        }
        if case .failed = outcome {} else {
            scheduleRetentionExpiry()
        }
        publishIfChanged()
    }

    public func hasActiveLoad(model: String) -> Bool {
        records.contains { $0.ticket.model == model && $0.finished == nil }
    }

    public func acknowledgeFailure(model: String) {
        records.removeAll { $0.ticket.model == model && $0.finished?.state == .failed }
        publishIfChanged()
    }

    /// The Dashboard opened the progress panel for `model`; it does not open
    /// on its own again for this load.
    public func markPanelOpened(model: String) {
        for index in records.indices where records[index].ticket.model == model {
            records[index].autoOpenPanel = false
        }
        publishIfChanged()
    }

    public func setDashboardVisible(_ visible: Bool) {
        dashboardVisible = visible
        ensurePolling()
    }

    // MARK: Cancellation

    /// Cancels the load shown for `model`: an App stage cancels its task, a
    /// backend load goes to the cancel endpoint, and a load with its own
    /// handler (DFlash2) runs that.
    public func cancel(model: String) async {
        let active = records.filter { $0.ticket.model == model && $0.finished == nil }
        if active.isEmpty {
            await cancelExternalLoad(model: model)
            return
        }
        for record in active {
            let ticket = record.ticket
            guard let index = activeIndex(ticket), !records[index].cancelRequested else { continue }
            records[index].cancelRequested = true
            publishIfChanged()
            if let handler = records[index].cancelHandler {
                handler()
            } else if let loadID = records[index].backendLoadID, !records[index].adoptedStartupLoad {
                await requestBackendCancel(loadID: loadID, ticket: ticket)
            } else if records[index].startingLoadID != nil {
                // `attachBackendLoad` cancels it once the backend names it.
            } else {
                records[index].cancelTask?()
                if records[index].variant != .dflash2, !records[index].adoptedStartupLoad {
                    // Nothing reached the backend: the load ends now, not
                    // when the App step under way notices the cancel. A
                    // DFlash2 switch first restores the previous backend.
                    finish(ticket, .cancelled(code: cancelCode(ticket)))
                }
            }
        }
        ensurePolling()
    }

    private func cancelExternalLoad(model: String) async {
        guard let load = backendLoads.first(where: {
            $0.model == model && !$0.state.isTerminal && !appLoadIDs.contains($0.loadID)
        }) else { return }
        externalCancelling.insert(load.loadID)
        publishIfChanged()
        let result = try? await clientProvider().cancelModelLoad(loadID: load.loadID)
        if result != .cancelling {
            externalCancelling.remove(load.loadID)
        }
        publishIfChanged()
        ensurePolling()
    }

    private func requestBackendCancel(loadID: String, ticket: ModelLoadTicket) async {
        let result: BackendLoadCancelResult?
        do {
            result = try await clientProvider().cancelModelLoad(loadID: loadID)
        } catch {
            IronMLXAppLogger.error("Could not cancel model load \(loadID): \(error.localizedDescription)")
            result = nil
        }
        guard let index = activeIndex(ticket) else { return }
        switch result {
        case .cancelling:
            break
        case .notFound:
            // The backend does not know the load; stop waiting for it.
            records[index].cancelTask?()
        case .finished, .finishing, nil:
            // Too late (or unreachable): the load runs to its end.
            records[index].cancelRequested = false
        }
        publishIfChanged()
        ensurePolling()
    }

    // MARK: Loading

    /// Runs `operation` as one tracked App load: its errors end the ticket as
    /// failed, a user cancel ends it as cancelled (thrown as
    /// `ModelLoadCancelledError`), success ends it as loaded.
    public func track<T: Sendable>(
        model: String,
        variant: ModelLoadVariant,
        origin: ModelLoadOrigin,
        kind: ModelLoadKind = .load,
        operation: @escaping @MainActor @Sendable (ModelLoadTicket) async throws -> T
    ) async throws -> T {
        let ticket = begin(model: model, variant: variant, origin: origin, kind: kind)
        let task = Task { @MainActor in try await operation(ticket) }
        if let index = activeIndex(ticket) {
            records[index].cancelTask = { task.cancel() }
        }
        do {
            let value = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            finish(ticket, .loaded)
            return value
        } catch {
            let (outcome, thrown) = outcome(of: error, ticket: ticket)
            finish(ticket, outcome)
            throw thrown
        }
    }

    /// The backend's configuration for `model` changed outside a
    /// registration pass (an unload); the next rescan registers it again.
    nonisolated public func forgetRegistration(of model: String) {
        registrationLedger.forget(model: model)
    }

    /// The App stages before a Dashboard load: the backend comes up
    /// (`waiting_for_backend`), then local models are registered with it
    /// (`syncing_models`). A cancel stops after the step under way.
    public func prepareBackend(
        _ ticket: ModelLoadTicket,
        startBackend: @MainActor () async throws -> Void,
        syncModels: @MainActor () async -> Void
    ) async throws {
        enterAppStage(ticket, .waitingForBackend)
        try await startBackend()
        try Task.checkCancellation()
        enterAppStage(ticket, .syncingModels)
        await syncModels()
        try Task.checkCancellation()
    }

    /// Loads `request` as a tracked App load.
    public func loadModel(
        _ request: BackendLoadModelRequest,
        origin: ModelLoadOrigin,
        kind: ModelLoadKind = .load
    ) async throws -> BackendModelAdminResponse {
        let variant: ModelLoadVariant = request.mtpModelDir == nil ? .plain : .mtp
        return try await track(model: request.model, variant: variant, origin: origin, kind: kind) { ticket in
            try await self.loadThroughBackend(ticket, request: request)
        }
    }

    /// Starts `request` as an asynchronous backend load under a fresh id and
    /// waits for its registry result, so no HTTP request stays open for the
    /// whole load. A load of the same model with other settings is waited
    /// for, then this one starts.
    public func loadThroughBackend(
        _ ticket: ModelLoadTicket,
        request: BackendLoadModelRequest
    ) async throws -> BackendModelAdminResponse {
        // The load (or reload) sets the model's backend configuration, so the
        // next rescan registers the saved one again, as it did before the record.
        defer { registrationLedger.forget(model: request.model) }
        let client = clientProvider()
        for _ in 0..<Self.maximumStartAttempts {
            try Task.checkCancellation()
            if cancelRequested(ticket) {
                throw ModelLoadCancelledError(code: cancelCode(ticket))
            }
            let loadID = UUID().uuidString.lowercased()
            var attempt = request
            attempt.loadID = loadID
            attempt.async = true
            setStartingLoadID(ticket, loadID)
            let start: BackendModelLoadStart
            do {
                start = try await client.startModelLoad(attempt)
            } catch {
                setStartingLoadID(ticket, nil)
                guard let body = BackendLoadErrorBody(error) else { throw error }
                switch body.code {
                case "invalid_load_id", "load_id_in_use":
                    continue
                case "load_in_progress":
                    guard let other = body.loadID else { throw error }
                    attachBackendLoad(ticket, loadID: other)
                    do {
                        _ = try await waitForBackendLoad(loadID: other)
                    } catch ModelLoadWaitError.loadNotFound {
                        // That load ended and its entry expired; start this one.
                    }
                    if cancelRequested(ticket) {
                        throw ModelLoadCancelledError(code: cancelCode(ticket))
                    }
                    detachBackendLoad(ticket)
                    continue
                case ModelLoadCancelledError.loadCancelledCode:
                    throw ModelLoadCancelledError(code: cancelCode(ticket))
                default:
                    await adoptRefusedLoad(ticket, loadID: loadID, client: client)
                    throw error
                }
            }
            switch start {
            case .completed(let response):
                setStartingLoadID(ticket, nil)
                if response.status == "reload_deferred" {
                    // The reload runs later as its own (`deferred_reload`) load.
                    setDismissOnSuccess(ticket)
                }
                return response
            case .accepted(let acceptedID, _):
                attachBackendLoad(ticket, loadID: acceptedID)
                let result: BackendLoadActivity
                do {
                    result = try await waitForBackendLoad(loadID: acceptedID)
                } catch ModelLoadWaitError.loadNotFound {
                    return try await confirmExpiredLoad(ticket, request: request, client: client)
                }
                switch result.state {
                case .loaded:
                    return BackendModelAdminResponse(
                        success: true,
                        status: result.kind == .reload ? "reloaded" : "loaded",
                        code: nil,
                        model: request.model,
                        loadedModels: try await client.fetchLoadedModels(),
                        warningCode: result.warningCode,
                        warning: result.warning,
                        error: nil
                    )
                case .failed:
                    throw ModelLoadFailedError(
                        message: result.message ?? "The model failed to load.",
                        code: result.code
                    )
                default:
                    if cancelRequested(ticket) || acceptedID == loadID {
                        throw ModelLoadCancelledError(code: cancelCode(ticket))
                    }
                    // A joined load someone else cancelled: start this one.
                    detachBackendLoad(ticket)
                    continue
                }
            }
        }
        throw ModelLoadFailedError(
            message: "The model load could not be started.",
            code: "load_start_failed"
        )
    }

    /// The backend refused this start. The App steps are over; when it had
    /// registered the load first (a full engine pool, say), its entry under
    /// this ticket's id names the stage the load failed in.
    private func adoptRefusedLoad(
        _ ticket: ModelLoadTicket,
        loadID: String,
        client: any BackendModelLoadTracking
    ) async {
        let refused = try? await client.fetchModelLoads().first {
            $0.loadID == loadID && $0.state.isTerminal
        }
        guard let index = activeIndex(ticket) else { return }
        completeAppStage(at: index)
        if let refused {
            records[index].backendLoadID = loadID
            records[index].backend = refused
            appLoadIDs.insert(loadID)
        }
    }

    /// The backend no longer lists this ticket's load (its result outlived the
    /// registry's retention while polling stalled). A new load that left the
    /// model loaded succeeded; a reload cannot be told apart from the version
    /// it replaced, and nothing else proves a result.
    private func confirmExpiredLoad(
        _ ticket: ModelLoadTicket,
        request: BackendLoadModelRequest,
        client: any BackendModelLoadTracking
    ) async throws -> BackendModelAdminResponse {
        let kind = activeIndex(ticket).flatMap { records[$0].backend?.kind } ?? .load
        if kind == .load {
            let loaded = try await client.fetchLoadedModels()
            if loaded.contains(where: { $0.id == request.model || $0.model == request.model }) {
                return BackendModelAdminResponse(
                    success: true,
                    status: "loaded",
                    code: nil,
                    model: request.model,
                    loadedModels: loaded,
                    warningCode: nil,
                    warning: nil,
                    error: nil
                )
            }
        }
        throw ModelLoadFailedError(
            message: "The backend no longer reports this model load, and its result is unknown.",
            code: ModelLoadFailedError.loadResultUnknownCode
        )
    }

    /// Waits until the backend reports `loadID` finished (loaded, failed or
    /// cancelled), polling its registry meanwhile.
    public func waitForBackendLoad(loadID: String) async throws -> BackendLoadActivity {
        let waiterID = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                guard !Task.isCancelled else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                waiters[waiterID] = Waiter(loadID: loadID, continuation: continuation)
                ensurePolling()
            }
        } onCancel: {
            Task { @MainActor in
                self.resumeWaiter(waiterID, with: .failure(CancellationError()))
            }
        }
    }

    // MARK: Published state

    public var activities: [ModelLoadActivity] {
        let date = now()
        var candidates: [(activity: ModelLoadActivity, isApp: Bool)] = []
        for record in records {
            if let finished = record.finished, finished.state != .failed,
               date.timeIntervalSince(finished.at) >= Self.finishedRetention {
                continue
            }
            candidates.append((activity(for: record), true))
        }
        for load in backendLoads where !appLoadIDs.contains(load.loadID) {
            candidates.append((externalActivity(for: load), false))
        }
        var chosen: [String: (activity: ModelLoadActivity, isApp: Bool)] = [:]
        for candidate in candidates {
            guard let current = chosen[candidate.activity.model] else {
                chosen[candidate.activity.model] = candidate
                continue
            }
            if Self.priority(candidate) > Self.priority(current) {
                chosen[candidate.activity.model] = candidate
            }
        }
        return chosen.values.map(\.activity).sorted {
            ($0.startedAt, $0.model) < ($1.startedAt, $1.model)
        }
    }

    /// Active beats finished, App beats external, newer beats older.
    private static func priority(
        _ candidate: (activity: ModelLoadActivity, isApp: Bool)
    ) -> (Int, Int, UInt64) {
        (candidate.activity.state.isTerminal ? 0 : 1, candidate.isApp ? 1 : 0, candidate.activity.startedAt)
    }

    private func activity(for record: Record) -> ModelLoadActivity {
        let backend = record.backend
        // Only `finish` ends an App load: the backend entry it waits on may
        // be another load whose result is not this one's (it then retries).
        let backendFinished = record.finished == nil && backend?.state.isTerminal == true
        let state: ModelLoadState
        if let finished = record.finished {
            state = finished.state
        } else if record.cancelRequested {
            state = .cancelling
        } else if let backend, !backendFinished {
            state = backend.state
        } else {
            state = .loading
        }
        let stage = backend?.stage ?? record.appStage.map(ModelLoadStage.init)
        let backendCompleted = backend?.completedStages.map {
            ModelLoadCompletedStage(stage: $0.stage, elapsedMs: $0.elapsedMs)
        } ?? []
        return ModelLoadActivity(
            model: record.ticket.model,
            variant: backend?.variant ?? record.variant,
            kind: backend?.kind ?? record.kind,
            origin: record.origin,
            state: state,
            stage: stage,
            completedStages: record.appCompleted + backendCompleted,
            bytesDone: backend?.bytesDone ?? 0,
            bytesTotal: backend?.bytesTotal ?? 0,
            startedAt: Self.unixMilliseconds(record.startedAt),
            message: record.finished?.message ?? (backendFinished ? nil : backend?.message),
            code: record.finished?.code ?? (backendFinished ? nil : backend?.code),
            cancellable: state == .queued || state == .loading,
            autoOpenPanel: record.autoOpenPanel
        )
    }

    private func externalActivity(for load: BackendLoadActivity) -> ModelLoadActivity {
        let state = externalCancelling.contains(load.loadID) && !load.state.isTerminal
            ? .cancelling
            : load.state
        return ModelLoadActivity(
            model: load.model,
            variant: load.variant,
            kind: load.kind,
            origin: .external,
            state: state,
            stage: load.stage,
            completedStages: load.completedStages.map {
                ModelLoadCompletedStage(stage: $0.stage, elapsedMs: $0.elapsedMs)
            },
            bytesDone: load.bytesDone,
            bytesTotal: load.bytesTotal,
            startedAt: load.startedUnixMs,
            message: load.message,
            code: load.code,
            cancellable: state == .queued || state == .loading,
            autoOpenPanel: false
        )
    }

    private func publishIfChanged() {
        let current = activities
        guard current != published else { return }
        published = current
        notificationCenter.post(name: .ironMLXModelLoadsDidChange, object: self)
    }

    private func scheduleRetentionExpiry() {
        let retentionSleep = retentionSleep
        Task { [weak self] in
            await retentionSleep(.seconds(Self.finishedRetention))
            self?.publishIfChanged()
        }
    }

    // MARK: Polling

    private func nextPollInterval() -> Duration? {
        let loading = !waiters.isEmpty
            || records.contains { $0.finished == nil }
            || backendLoads.contains { !$0.state.isTerminal }
        if loading {
            return Self.activePollInterval
        }
        return dashboardVisible ? Self.idlePollInterval : nil
    }

    private func ensurePolling() {
        guard let interval = nextPollInterval() else { return }
        if pollTask != nil {
            // A slow idle wait would delay a load that just started.
            guard let sleepingInterval, sleepingInterval > interval else { return }
            pollTask?.cancel()
        }
        pollGeneration += 1
        sleepingInterval = nil
        let generation = pollGeneration
        pollTask = Task { [weak self] in
            await self?.pollLoop(generation: generation)
        }
    }

    private func pollLoop(generation: Int) async {
        defer {
            if pollGeneration == generation {
                pollTask = nil
                sleepingInterval = nil
            }
        }
        while !Task.isCancelled {
            await pollOnce()
            guard !Task.isCancelled, let interval = nextPollInterval() else { return }
            sleepingInterval = interval
            await pollSleep(interval)
            if pollGeneration == generation {
                sleepingInterval = nil
            }
            guard !Task.isCancelled, nextPollInterval() != nil else { return }
        }
    }

    private func pollOnce() async {
        let client = clientProvider()
        do {
            let loads = try await client.fetchModelLoads()
            consecutiveFetchFailures = 0
            apply(loads)
        } catch {
            consecutiveFetchFailures += 1
            if consecutiveFetchFailures >= Self.unreachablePolls {
                consecutiveFetchFailures = 0
                backendLoads = []
                for id in Array(waiters.keys) {
                    resumeWaiter(id, with: .failure(ModelLoadWaitError.backendUnreachable))
                }
                publishIfChanged()
            }
        }
    }

    private func apply(_ loads: [BackendLoadActivity]) {
        backendLoads = loads
        externalCancelling.formIntersection(loads.filter { !$0.state.isTerminal }.map(\.loadID))
        for index in records.indices where records[index].finished == nil {
            if let loadID = records[index].backendLoadID {
                if let load = loads.first(where: { $0.loadID == loadID }) {
                    records[index].backend = load
                }
            } else if records[index].variant == .dflash2,
                      records[index].appStage == .restartingBackend
                        || records[index].appStage == .waitingForBackend,
                      let load = startupLoad(for: records[index], in: loads) {
                // A DFlash2 launch reports its load without an App-chosen id.
                completeAppStage(at: index)
                records[index].backendLoadID = load.loadID
                records[index].backend = load
                records[index].adoptedStartupLoad = true
                appLoadIDs.insert(load.loadID)
            }
        }
        for (id, waiter) in waiters {
            if let load = loads.first(where: { $0.loadID == waiter.loadID }) {
                if load.state.isTerminal {
                    resumeWaiter(id, with: .success(load))
                } else {
                    waiters[id]?.seen = true
                    waiters[id]?.missingPolls = 0
                }
                continue
            }
            waiters[id]?.missingPolls += 1
            let limit = waiter.seen ? Self.missingPollsAfterSeen : Self.missingPollsBeforeSeen
            if waiter.missingPolls + 1 >= limit {
                resumeWaiter(id, with: .failure(ModelLoadWaitError.loadNotFound(waiter.loadID)))
            }
        }
        publishIfChanged()
    }

    private func startupLoad(for record: Record, in loads: [BackendLoadActivity]) -> BackendLoadActivity? {
        let earliest = Self.unixMilliseconds(record.startedAt.addingTimeInterval(-5))
        return loads.first {
            $0.trigger == .startup && $0.model == record.ticket.model && $0.startedUnixMs >= earliest
        }
    }

    private func resumeWaiter(_ id: UUID, with result: Result<BackendLoadActivity, Error>) {
        guard let waiter = waiters.removeValue(forKey: id) else { return }
        waiter.continuation.resume(with: result)
    }

    // MARK: Helpers

    private func activeIndex(_ ticket: ModelLoadTicket) -> Int? {
        records.firstIndex { $0.ticket.id == ticket.id && $0.finished == nil }
    }

    private func completeAppStage(at index: Int) {
        guard let stage = records[index].appStage else { return }
        let elapsed = max(0, now().timeIntervalSince(records[index].appStageStartedAt))
        records[index].appCompleted.append(
            ModelLoadCompletedStage(stage: ModelLoadStage(stage), elapsedMs: UInt64((elapsed * 1_000).rounded()))
        )
        records[index].appStage = nil
    }

    private func cancelRequested(_ ticket: ModelLoadTicket) -> Bool {
        activeIndex(ticket).map { records[$0].cancelRequested } ?? false
    }

    /// Whether the user cancelled this ticket, also once a cancel ended it.
    private func userCancelled(_ ticket: ModelLoadTicket) -> Bool {
        records.first { $0.ticket.id == ticket.id }?.cancelRequested ?? false
    }

    private func cancelCode(_ ticket: ModelLoadTicket) -> String {
        guard let index = records.firstIndex(where: { $0.ticket.id == ticket.id }) else {
            return ModelLoadCancelledError.loadCancelledCode
        }
        let kind = records[index].backend?.kind ?? records[index].kind
        return kind == .reload
            ? ModelLoadCancelledError.reloadCancelledCode
            : ModelLoadCancelledError.loadCancelledCode
    }

    /// Forgets a load this ticket only waited on, before it starts its own:
    /// that load's result is not this ticket's, and a cancel must reach the
    /// new load rather than the finished one.
    private func detachBackendLoad(_ ticket: ModelLoadTicket) {
        guard let index = activeIndex(ticket) else { return }
        records[index].backendLoadID = nil
        records[index].backend = nil
        records[index].adoptedStartupLoad = false
        publishIfChanged()
    }

    private func setStartingLoadID(_ ticket: ModelLoadTicket, _ loadID: String?) {
        guard let index = activeIndex(ticket) else { return }
        records[index].startingLoadID = loadID
    }

    private func setDismissOnSuccess(_ ticket: ModelLoadTicket) {
        guard let index = activeIndex(ticket) else { return }
        records[index].dismissOnSuccess = true
    }

    /// How an error thrown by a tracked operation ends its ticket, and what
    /// the caller sees: a user cancel always surfaces as `ModelLoadCancelledError`.
    private func outcome(of error: Error, ticket: ModelLoadTicket) -> (ModelLoadOutcome, Error) {
        if let cancelled = error as? ModelLoadCancelledError {
            return (.cancelled(code: cancelled.code), cancelled)
        }
        if userCancelled(ticket) {
            let cancelled = ModelLoadCancelledError(code: cancelCode(ticket))
            return (.cancelled(code: cancelled.code), cancelled)
        }
        if error is CancellationError {
            // The caller went away; the backend may still finish the load,
            // which then shows like any other load.
            if let index = activeIndex(ticket), let loadID = records[index].backendLoadID {
                appLoadIDs.remove(loadID)
            }
            return (.dismissed, error)
        }
        if let failed = error as? ModelLoadFailedError {
            return (.failed(message: failed.message, code: failed.code), error)
        }
        if let body = BackendLoadErrorBody(error) {
            return (.failed(message: body.error ?? error.localizedDescription, code: body.code), error)
        }
        return (.failed(message: error.localizedDescription, code: DashboardErrorClassifier.code(for: error)), error)
    }

    /// Whether `error` ended a load because the user cancelled it.
    public nonisolated static func isUserCancellation(_ error: Error) -> Bool {
        error is ModelLoadCancelledError
    }

    private static func unixMilliseconds(_ date: Date) -> UInt64 {
        UInt64(max(0, (date.timeIntervalSince1970 * 1_000).rounded()))
    }
}

// MARK: - Model manager adapter

/// A model manager whose loads go through the center, so starter and chat
/// loads show their progress and survive long loads.
struct LoadTrackingModelManager: BackendModelManaging {
    let base: any BackendModelManaging
    let loadActivity: ModelLoadActivityCenter
    let origin: ModelLoadOrigin

    func fetchHealthz() async throws -> HealthzSnapshot {
        try await base.fetchHealthz()
    }

    func loadModel(
        model: String,
        modelDir: String,
        setDefault: Bool,
        maxCacheCap: Int?,
        pinned: Bool,
        promptLookup: BackendPromptLookupConfig?,
        audio: BackendAudioResources?,
        decision: BackendDecisionSettings?
    ) async throws -> BackendModelAdminResponse {
        try await loadActivity.loadModel(
            BackendLoadModelRequest(
                model: model,
                modelDir: modelDir,
                setDefault: setDefault,
                maxCacheCap: maxCacheCap,
                pinned: pinned,
                promptLookup: promptLookup,
                reloadWhenIdle: false,
                audio: audio,
                decision: decision
            ),
            origin: origin
        )
    }

    func unloadModel(model: String, modelDir: String?) async throws -> BackendModelAdminResponse {
        let response = try await base.unloadModel(model: model, modelDir: modelDir)
        loadActivity.forgetRegistration(of: model)
        return response
    }

    func setDefaultModel(_ model: String) async throws -> BackendModelAdminResponse {
        try await base.setDefaultModel(model)
    }

    func pinModel(model: String) async throws -> BackendModelAdminResponse {
        try await base.pinModel(model: model)
    }

    func unpinModel(model: String) async throws -> BackendModelAdminResponse {
        try await base.unpinModel(model: model)
    }

    func fetchLoadedModels() async throws -> [BackendLoadedModelInfo] {
        try await base.fetchLoadedModels()
    }

    func clearSharedPromptLookup(model: String?) async throws -> BackendPromptLookupClearResponse {
        try await base.clearSharedPromptLookup(model: model)
    }
}
