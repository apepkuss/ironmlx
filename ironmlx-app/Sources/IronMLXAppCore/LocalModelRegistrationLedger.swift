import Foundation

/// Everything one registration sends for a local model. Two equal values
/// register the model the same way.
struct LocalModelRegistration: Equatable, Sendable {
    var modelDir: String
    var setDefault: Bool
    var maxCacheCap: Int?
    /// `nil` leaves the backend's pinned state as it is.
    var pinned: Bool?
    var mtpModelDir: String?
    var mtpDraftTokens: Int?
    var promptLookup: BackendPromptLookupConfig?
    var samplingDefaults: BackendSamplingDefaults
    var audio: BackendAudioResources?
    var decision: BackendDecisionSettings?
    /// Sent after the registration for ASR models.
    var asr: BackendASRSettings?
}

/// What each local model was last registered with in one backend launch, so a
/// rescan registers only the models whose registration changed. It lives in
/// memory only; another launch starts with nothing recorded. It also keeps
/// each model's last computed registration with the inputs it was computed
/// from, so a rescan whose inputs are unchanged does not compute it again.
public final class LocalModelRegistrationLedger: @unchecked Sendable {
    public static let shared = LocalModelRegistrationLedger()

    private let lock = NSLock()
    private var launchID: UUID?
    private var registrations: [String: LocalModelRegistration] = [:]
    private var modelsInFlight: Set<String> = []
    private var modelWaiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    public init() {}

    /// Waits until no other pass is registering `model`. Registrations of one
    /// model run one at a time and in arrival order, so a registration that
    /// reads newer parameters reaches the backend after one that read older
    /// ones; other models are not held up.
    func beginRegistration(of model: String) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let runNow = lock.withLock {
                if modelsInFlight.contains(model) {
                    modelWaiters[model, default: []].append(continuation)
                    return false
                }
                modelsInFlight.insert(model)
                return true
            }
            if runNow {
                continuation.resume()
            }
        }
    }

    /// Ends the registration of `model` and starts the next one waiting for it.
    func endRegistration(of model: String) {
        let next = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            guard var waiters = modelWaiters[model], !waiters.isEmpty else {
                modelsInFlight.remove(model)
                modelWaiters[model] = nil
                return nil
            }
            let next = waiters.removeFirst()
            modelWaiters[model] = waiters.isEmpty ? nil : waiters
            return next
        }
        next?.resume()
    }

    /// Records `registration` as sent to `launchID` before it is sent, so
    /// concurrent rescans do not send it twice. Returns what is recorded, or
    /// `nil` when the launch already has this registration and `force` is
    /// false. A registration that leaves the pinned state alone keeps the
    /// recorded one.
    func claim(
        _ registration: LocalModelRegistration,
        model: String,
        launchID: UUID,
        force: Bool
    ) -> LocalModelRegistration? {
        lock.withLock {
            if self.launchID != launchID {
                self.launchID = launchID
                registrations = [:]
            }
            var effective = registration
            if effective.pinned == nil {
                effective.pinned = registrations[model]?.pinned
            }
            if !force, registrations[model] == effective {
                return nil
            }
            registrations[model] = effective
            return effective
        }
    }

    /// Forgets a claimed registration that failed, so the next rescan sends it
    /// again. A newer claim for the model is kept.
    func release(_ registration: LocalModelRegistration, model: String, launchID: UUID) {
        lock.withLock {
            guard self.launchID == launchID, registrations[model] == registration else {
                return
            }
            registrations[model] = nil
        }
    }

    /// Forgets what `model` was registered with, so the next rescan sends
    /// its registration again.
    public func forget(model: String) {
        lock.withLock {
            registrations[model] = nil
        }
    }

    private var computations: [String: Int] = [:]
    private var rescanPasses = 0
    private var computed: [String: (inputs: LocalModelRegistrationInputs, registration: LocalModelRegistration)] = [:]

    func recordComputation(of model: String) {
        lock.withLock { computations[model, default: 0] += 1 }
    }

    /// The registration last computed for `model` from these same inputs.
    func computedRegistration(
        of model: String,
        inputs: LocalModelRegistrationInputs
    ) -> LocalModelRegistration? {
        lock.withLock {
            guard let entry = computed[model], entry.inputs == inputs else {
                return nil
            }
            return entry.registration
        }
    }

    /// Keeps `registration` as what `inputs` compute to for `model`. It does
    /// not depend on the backend launch, so a new launch keeps it.
    func storeComputedRegistration(
        _ registration: LocalModelRegistration,
        of model: String,
        inputs: LocalModelRegistrationInputs
    ) {
        lock.withLock { computed[model] = (inputs, registration) }
    }

    private var rescanRunning = false
    private var pendingRescan: (@Sendable () async -> Void)?

    /// Runs `pass` unless a rescan pass is running. Then it is kept as the
    /// rerun that pass starts when it ends, replacing an earlier kept one, and
    /// this call returns at once: overlapping rescans make one pass plus one
    /// rerun with the latest request.
    ///
    /// The kept rerun runs in the task of the call that started the running
    /// pass. If that task is cancelled, the rerun still starts but stops at its
    /// first model check, so the request it carries is lost until the next
    /// rescan. `DashboardBridge.sendScannedModels` starts each rescan in its
    /// own unstructured `Task` that nothing keeps or cancels, so this does not
    /// happen today.
    func runCoalescedRescan(_ pass: @escaping @Sendable () async -> Void) async {
        var next: (@Sendable () async -> Void)? = lock.withLock {
            if rescanRunning {
                pendingRescan = pass
                return nil
            }
            rescanRunning = true
            return pass
        }
        while let current = next {
            lock.withLock { rescanPasses += 1 }
            await current()
            next = lock.withLock {
                let pending = pendingRescan
                pendingRescan = nil
                if pending == nil {
                    rescanRunning = false
                }
                return pending
            }
        }
    }

    /// How many times each model's registration was computed.
    var computationCounts: [String: Int] {
        lock.withLock { computations }
    }

    /// Forgets the computed registrations and computation counts of every
    /// model not in `models`.
    func keepComputedRegistrations(of models: Set<String>) {
        lock.withLock {
            computed = computed.filter { models.contains($0.key) }
            computations = computations.filter { models.contains($0.key) }
        }
    }

    /// Models with a kept computed registration.
    var computedRegistrationModels: Set<String> {
        lock.withLock { Set(computed.keys) }
    }

    /// Rescan passes run so far.
    var rescanPassesRun: Int {
        lock.withLock { rescanPasses }
    }

    /// Registrations waiting for another registration of the same model.
    var waitingRegistrations: Int {
        lock.withLock { modelWaiters.values.reduce(0) { $0 + $1.count } }
    }
}
