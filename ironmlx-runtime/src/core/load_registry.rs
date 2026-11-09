//! In-flight and recently finished model loads: stages, weight progress and
//! cancellation, shared by every load path so clients can watch and cancel
//! them.
//!
//! A load is registered with [`LoadRegistry::begin`] before it waits for the
//! pool's load gate. The returned [`LoadHandle`] reports stages, exposes the
//! [`LoadObserver`] the weight loaders report bytes to, and records the
//! result. Finished entries stay listed for [`FINISHED_RETENTION_MS`].

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::watch;

pub use ironmlx_lm::core::load_observer::{LoadCancelled, LoadObserver, WeightComponent};

/// How long a finished load stays in [`LoadRegistry::list`].
pub const FINISHED_RETENTION_MS: u64 = 30_000;

/// Returned when a model already has a load in progress whose configuration
/// differs from the one requested.
#[derive(Debug, thiserror::Error)]
#[error("model load in progress")]
pub struct LoadInProgress {
    pub load_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadVariant {
    Plain,
    Mtp,
    Dflash2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadKind {
    Load,
    Reload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadTriggerKind {
    Admin,
    #[serde(rename = "openai_control")]
    OpenAiControl,
    Request,
    DeferredReload,
    Startup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadState {
    Queued,
    Loading,
    Cancelling,
    Loaded,
    Failed,
    Cancelled,
}

impl LoadState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Loaded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadStage {
    Queued,
    PreparingMemory,
    ReadingWeights,
    BuildingModel,
    ReadingDraftWeights,
    StartingEngine,
    WarmingUp,
}

impl LoadStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::PreparingMemory => "preparing_memory",
            Self::ReadingWeights => "reading_weights",
            Self::BuildingModel => "building_model",
            Self::ReadingDraftWeights => "reading_draft_weights",
            Self::StartingEngine => "starting_engine",
            Self::WarmingUp => "warming_up",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CompletedStage {
    pub stage: LoadStage,
    pub elapsed_ms: u64,
}

/// Registration of a new load.
#[derive(Debug, Clone)]
pub struct LoadBegin {
    /// Client-chosen id; a UUID is generated when absent or already in use.
    pub load_id: Option<String>,
    pub model: String,
    pub variant: LoadVariant,
    pub kind: LoadKind,
    pub trigger: LoadTriggerKind,
    /// Identifies the requested configuration, so a second request for the
    /// same model can join an identical load or be refused.
    pub config_fingerprint: String,
    /// Reported with the result once the model loads.
    pub warning: Option<LoadWarning>,
}

/// A note about a successful load, e.g. that no scheduler profile matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadWarning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LoadActivity {
    pub load_id: String,
    pub model: String,
    pub variant: LoadVariant,
    pub kind: LoadKind,
    pub trigger: LoadTriggerKind,
    pub state: LoadState,
    pub stage: LoadStage,
    pub completed_stages: Vec<CompletedStage>,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub message: Option<String>,
    pub code: Option<String>,
    /// Set on a loaded load whose configuration has a caveat.
    pub warning_code: Option<String>,
    pub warning: Option<String>,
    #[serde(skip)]
    pub config_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LoadCancelError {
    #[error("model load not found")]
    NotFound,
    #[error("model load already finished")]
    Finished,
    #[error("model load is finishing and can no longer be cancelled")]
    Finishing,
}

/// Wall clock used for timestamps, stage timing and retention.
pub trait LoadClock: Send + Sync + 'static {
    fn now_unix_ms(&self) -> u64;
}

struct SystemClock;

impl LoadClock for SystemClock {
    fn now_unix_ms(&self) -> u64 {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis())
            .unwrap_or_default();
        millis.min(u128::from(u64::MAX)) as u64
    }
}

pub struct LoadRegistry {
    shared: Arc<Shared>,
}

struct Shared {
    clock: Arc<dyn LoadClock>,
    entries: Mutex<Vec<Entry>>,
}

struct Entry {
    activity: LoadActivity,
    warning: Option<LoadWarning>,
    observer: Arc<LoadObserver>,
    stage_started_unix_ms: u64,
    /// Set once the load can no longer be cancelled.
    finishing: bool,
    weights: WeightTally,
    cancel: watch::Sender<bool>,
    finished: watch::Sender<Option<LoadActivity>>,
}

const COMPONENTS: [WeightComponent; 2] = [WeightComponent::Main, WeightComponent::Draft];

fn component_index(component: WeightComponent) -> usize {
    match component {
        WeightComponent::Main => 0,
        WeightComponent::Draft => 1,
    }
}

/// Bytes across every weight loader of one load. The observer only reports
/// the component currently loading, so finished components are remembered
/// here; components not opened yet count with their directory estimate.
#[derive(Debug, Default)]
struct WeightTally {
    planned_total: u64,
    estimate: [Option<u64>; 2],
    exact: [Option<u64>; 2],
    done: [u64; 2],
    reported_done: u64,
}

impl WeightTally {
    fn observe(&mut self, observer: &LoadObserver) {
        let snapshot = observer.snapshot();
        let Some(component) = snapshot.component else {
            return;
        };
        let index = component_index(component);
        // Components load one after another: a later one starting means the
        // earlier ones finished, even if their last chunk was never sampled.
        for earlier in 0..index {
            if let Some(exact) = self.exact[earlier] {
                self.done[earlier] = exact;
            }
        }
        if self.exact[index] != Some(snapshot.bytes_total) {
            self.done[index] = 0;
        }
        self.exact[index] = Some(snapshot.bytes_total);
        self.done[index] = self.done[index].max(snapshot.bytes_done);
    }

    fn planned(&self) -> u64 {
        let estimated: u64 = self.estimate.iter().flatten().sum();
        self.planned_total.max(estimated)
    }

    fn totals(&mut self) -> (u64, u64) {
        let mut exact_total = 0u64;
        let mut seen_estimate = 0u64;
        for component in COMPONENTS {
            let index = component_index(component);
            if let Some(exact) = self.exact[index] {
                exact_total = exact_total.saturating_add(exact);
                seen_estimate = seen_estimate.saturating_add(self.estimate[index].unwrap_or(exact));
            }
        }
        let total = exact_total.saturating_add(self.planned().saturating_sub(seen_estimate));
        let done = self.done.iter().sum::<u64>().min(total);
        self.reported_done = self.reported_done.max(done);
        (self.reported_done, total.max(self.reported_done))
    }

    /// Every opened component is complete; nothing else will be read.
    fn complete(&mut self) -> (u64, u64) {
        if self.exact.iter().all(Option::is_none) {
            let planned = self.planned();
            self.reported_done = self.reported_done.max(planned);
            return (self.reported_done, self.reported_done);
        }
        for index in 0..self.exact.len() {
            if let Some(exact) = self.exact[index] {
                self.done[index] = exact;
            }
        }
        let total = self.exact.iter().flatten().sum::<u64>();
        self.reported_done = self.reported_done.max(total);
        (self.reported_done, self.reported_done)
    }
}

impl Entry {
    fn is_terminal(&self) -> bool {
        self.activity.state.is_terminal()
    }

    fn refresh(&mut self) {
        if self.is_terminal() {
            return;
        }
        self.weights.observe(&self.observer);
        let (done, total) = self.weights.totals();
        self.activity.bytes_done = done;
        self.activity.bytes_total = total;
    }

    fn close_stage(&mut self, now_unix_ms: u64) -> u64 {
        let elapsed_ms = now_unix_ms.saturating_sub(self.stage_started_unix_ms);
        self.activity.completed_stages.push(CompletedStage {
            stage: self.activity.stage,
            elapsed_ms,
        });
        self.stage_started_unix_ms = now_unix_ms;
        elapsed_ms
    }
}

impl Shared {
    fn entries(&self) -> MutexGuard<'_, Vec<Entry>> {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let now = self.clock.now_unix_ms();
        entries.retain(|entry| {
            entry
                .activity
                .finished_unix_ms
                .is_none_or(|finished| now.saturating_sub(finished) < FINISHED_RETENTION_MS)
        });
        entries
    }

    fn with_entry<R>(&self, load_id: &str, f: impl FnOnce(&mut Entry, u64) -> R) -> Option<R> {
        let mut entries = self.entries();
        let now = self.clock.now_unix_ms();
        entries
            .iter_mut()
            .find(|entry| entry.activity.load_id == load_id)
            .map(|entry| f(entry, now))
    }

    fn enter_stage(&self, load_id: &str, stage: LoadStage) {
        self.with_entry(load_id, |entry, now| {
            if entry.is_terminal() {
                return;
            }
            entry.refresh();
            if entry.activity.stage != stage {
                let finished = entry.activity.stage;
                let elapsed_ms = entry.close_stage(now);
                log_stage_finished(&entry.activity, finished, elapsed_ms);
                entry.activity.stage = stage;
            }
            if entry.activity.state == LoadState::Queued && stage != LoadStage::Queued {
                entry.activity.state = LoadState::Loading;
            }
        });
    }

    fn finish(&self, load_id: &str, state: LoadState, message: Option<String>, code: &str) {
        self.with_entry(load_id, |entry, now| {
            if entry.is_terminal() {
                return;
            }
            entry.weights.observe(&entry.observer);
            let (done, total) = if state == LoadState::Loaded {
                entry.weights.complete()
            } else {
                entry.weights.totals()
            };
            entry.activity.bytes_done = done;
            entry.activity.bytes_total = total;
            let finished = entry.activity.stage;
            let elapsed_ms = entry.close_stage(now);
            log_stage_finished(&entry.activity, finished, elapsed_ms);
            entry.activity.state = state;
            entry.activity.finished_unix_ms = Some(now);
            entry.activity.message = message;
            entry.activity.code = (state != LoadState::Loaded).then(|| code.to_string());
            if state == LoadState::Loaded {
                if let Some(warning) = entry.warning.take() {
                    entry.activity.warning_code = Some(warning.code);
                    entry.activity.warning = Some(warning.message);
                }
            }
            tracing::info!(
                model = %entry.activity.model,
                load_id = %entry.activity.load_id,
                state = ?state,
                total_ms = now.saturating_sub(entry.activity.started_unix_ms),
                "model load finished"
            );
            let activity = entry.activity.clone();
            entry.finished.send_replace(Some(activity));
        });
    }
}

fn log_stage_finished(activity: &LoadActivity, stage: LoadStage, elapsed_ms: u64) {
    let total_ms: u64 = activity
        .completed_stages
        .iter()
        .map(|stage| stage.elapsed_ms)
        .sum();
    tracing::info!(
        model = %activity.model,
        load_id = %activity.load_id,
        stage = stage.as_str(),
        elapsed_ms,
        total_ms,
        "model load stage finished"
    );
}

impl LoadRegistry {
    pub fn new() -> Arc<Self> {
        Self::with_clock(Arc::new(SystemClock))
    }

    pub fn with_clock(clock: Arc<dyn LoadClock>) -> Arc<Self> {
        Arc::new(Self {
            shared: Arc::new(Shared {
                clock,
                entries: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Register a queued load.
    pub fn begin(&self, spec: LoadBegin) -> LoadHandle {
        let mut entries = self.shared.entries();
        self.insert(&mut entries, spec)
    }

    /// Register a queued load unless `spec.model` already has one in
    /// progress, which is returned instead.
    pub fn try_begin(&self, spec: LoadBegin) -> Result<LoadHandle, Box<LoadActivity>> {
        let mut entries = self.shared.entries();
        if let Some(existing) = entries
            .iter_mut()
            .find(|entry| !entry.is_terminal() && entry.activity.model == spec.model)
        {
            existing.refresh();
            return Err(Box::new(existing.activity.clone()));
        }
        Ok(self.insert(&mut entries, spec))
    }

    fn insert(&self, entries: &mut Vec<Entry>, spec: LoadBegin) -> LoadHandle {
        let load_id = match spec.load_id {
            Some(id)
                if !id.is_empty() && !entries.iter().any(|entry| entry.activity.load_id == id) =>
            {
                id
            }
            Some(id) => {
                let generated = uuid::Uuid::new_v4().to_string();
                tracing::warn!(requested = %id, load_id = %generated, "model load id already in use; generated a new id");
                generated
            }
            None => uuid::Uuid::new_v4().to_string(),
        };
        let now = self.shared.clock.now_unix_ms();
        let observer = LoadObserver::new();
        entries.push(Entry {
            activity: LoadActivity {
                load_id: load_id.clone(),
                model: spec.model,
                variant: spec.variant,
                kind: spec.kind,
                trigger: spec.trigger,
                state: LoadState::Queued,
                stage: LoadStage::Queued,
                completed_stages: Vec::new(),
                bytes_done: 0,
                bytes_total: 0,
                started_unix_ms: now,
                finished_unix_ms: None,
                message: None,
                code: None,
                warning_code: None,
                warning: None,
                config_fingerprint: spec.config_fingerprint,
            },
            warning: spec.warning,
            observer: observer.clone(),
            stage_started_unix_ms: now,
            finishing: false,
            weights: WeightTally::default(),
            cancel: watch::Sender::new(false),
            finished: watch::Sender::new(None),
        });
        LoadHandle {
            tracker: LoadTracker {
                shared: self.shared.clone(),
                load_id,
                observer,
            },
            finished: false,
        }
    }

    /// Loads in progress and loads finished within the retention window, by
    /// start time.
    pub fn list(&self) -> Vec<LoadActivity> {
        let mut entries = self.shared.entries();
        let mut loads: Vec<LoadActivity> = entries
            .iter_mut()
            .map(|entry| {
                entry.refresh();
                entry.activity.clone()
            })
            .collect();
        loads.sort_by_key(|load| load.started_unix_ms);
        loads
    }

    /// Ask a load to stop at its next cancellation point.
    pub fn cancel(&self, load_id: &str) -> Result<(), LoadCancelError> {
        self.shared
            .with_entry(load_id, |entry, _| {
                if entry.is_terminal() {
                    return Err(LoadCancelError::Finished);
                }
                if entry.finishing {
                    return Err(LoadCancelError::Finishing);
                }
                entry.activity.state = LoadState::Cancelling;
                entry.observer.cancel();
                entry.cancel.send_replace(true);
                tracing::info!(model = %entry.activity.model, load_id, "model load cancel requested");
                Ok(())
            })
            .unwrap_or(Err(LoadCancelError::NotFound))
    }

    /// The load of `model_id` that has not finished yet, if any.
    pub fn in_progress_for(&self, model_id: &str) -> Option<LoadActivity> {
        let mut entries = self.shared.entries();
        entries
            .iter_mut()
            .find(|entry| !entry.is_terminal() && entry.activity.model == model_id)
            .map(|entry| {
                entry.refresh();
                entry.activity.clone()
            })
    }

    /// Wait until `load_id` finishes and return its final activity; `None`
    /// when the id is unknown (or its retention already expired).
    pub async fn wait(&self, load_id: &str) -> Option<LoadActivity> {
        let mut finished = self
            .shared
            .with_entry(load_id, |entry, _| entry.finished.subscribe())?;
        let result = finished.wait_for(Option::is_some).await;
        match result {
            Ok(activity) => activity.clone(),
            Err(_) => None,
        }
    }
}

/// Cloneable access to one registered load for the code doing the work: stage
/// reports, weight planning and cancellation checks. Finishing the load is
/// reserved to its [`LoadHandle`].
#[derive(Clone)]
pub struct LoadTracker {
    shared: Arc<Shared>,
    load_id: String,
    observer: Arc<LoadObserver>,
}

impl std::fmt::Debug for LoadTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadTracker")
            .field("load_id", &self.load_id)
            .finish_non_exhaustive()
    }
}

impl LoadTracker {
    pub fn load_id(&self) -> &str {
        &self.load_id
    }

    /// Progress sink and cancellation flag to pass to the weight loaders.
    pub fn observer(&self) -> Arc<LoadObserver> {
        self.observer.clone()
    }

    pub fn enter_stage(&self, stage: LoadStage) {
        self.shared.enter_stage(&self.load_id, stage);
    }

    /// Directory estimate of every weight component of this load, used until
    /// the loaders report exact sizes.
    pub fn set_planned_total(&self, bytes: u64) {
        self.shared.with_entry(&self.load_id, |entry, _| {
            entry.weights.planned_total = bytes;
        });
    }

    /// Directory estimate of one component; replaced by the loader's exact
    /// size once that component opens.
    pub fn set_component_estimate(&self, component: WeightComponent, bytes: u64) {
        self.shared.with_entry(&self.load_id, |entry, _| {
            entry.weights.estimate[component_index(component)] = Some(bytes);
        });
    }

    pub fn is_cancelled(&self) -> bool {
        self.observer.is_cancelled()
    }

    /// Cancellation point: `Err(LoadCancelled)` once cancelled.
    pub fn checkpoint(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            return Err(LoadCancelled.into());
        }
        Ok(())
    }

    /// Last cancellation point: enter `stage` and refuse cancellation from now
    /// on, unless the load was already cancelled.
    pub fn begin_finishing(&self, stage: LoadStage) -> crate::Result<()> {
        let cancelled = self
            .shared
            .with_entry(&self.load_id, |entry, _| {
                if entry.observer.is_cancelled() {
                    return true;
                }
                entry.finishing = true;
                false
            })
            .unwrap_or_else(|| self.is_cancelled());
        if cancelled {
            return Err(LoadCancelled.into());
        }
        self.enter_stage(stage);
        Ok(())
    }

    /// Resolves once the load is cancelled.
    pub async fn cancelled(&self) {
        let Some(mut cancel) = self
            .shared
            .with_entry(&self.load_id, |entry, _| entry.cancel.subscribe())
        else {
            return;
        };
        let _ = cancel.wait_for(|cancelled| *cancelled).await;
    }
}

/// Owner of one registered load. Exactly one `finish_*` records its result;
/// dropping it unfinished records the load as aborted.
#[derive(Debug)]
pub struct LoadHandle {
    tracker: LoadTracker,
    finished: bool,
}

impl LoadHandle {
    pub fn load_id(&self) -> &str {
        self.tracker.load_id()
    }

    pub fn tracker(&self) -> LoadTracker {
        self.tracker.clone()
    }

    pub fn observer(&self) -> Arc<LoadObserver> {
        self.tracker.observer()
    }

    pub fn enter_stage(&self, stage: LoadStage) {
        self.tracker.enter_stage(stage);
    }

    pub fn set_planned_total(&self, bytes: u64) {
        self.tracker.set_planned_total(bytes);
    }

    pub fn finish_loaded(mut self) {
        self.finish(LoadState::Loaded, None, "");
    }

    /// Record `error`; a [`LoadCancelled`] error records a cancellation.
    /// A failure is recorded with the code and user-facing message the
    /// admin API gives the same error; the raw error chain goes to the log.
    pub fn finish_failed(mut self, error: &anyhow::Error) {
        if error.downcast_ref::<LoadCancelled>().is_some() {
            self.finish(LoadState::Cancelled, None, "load_cancelled");
            return;
        }
        tracing::warn!(load_id = %self.load_id(), "model load failed: {error:#}");
        let classified = crate::core::load_error::classify_load_error(error);
        self.finish(
            LoadState::Failed,
            Some(classified.message),
            classified.code.unwrap_or("load_failed"),
        );
    }

    pub fn finish_cancelled(mut self) {
        self.finish(LoadState::Cancelled, None, "load_cancelled");
    }

    fn finish(&mut self, state: LoadState, message: Option<String>, code: &str) {
        self.finished = true;
        self.tracker
            .shared
            .finish(&self.tracker.load_id, state, message, code);
    }
}

impl Drop for LoadHandle {
    fn drop(&mut self) {
        if !self.finished {
            self.finish(
                LoadState::Failed,
                Some("model load ended without a result".to_string()),
                "load_aborted",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct ManualClock(AtomicU64);

    impl ManualClock {
        fn new(now_unix_ms: u64) -> Arc<Self> {
            Arc::new(Self(AtomicU64::new(now_unix_ms)))
        }

        fn advance(&self, ms: u64) {
            self.0.fetch_add(ms, Ordering::SeqCst);
        }
    }

    impl LoadClock for ManualClock {
        fn now_unix_ms(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn spec(model: &str) -> LoadBegin {
        LoadBegin {
            load_id: None,
            model: model.to_string(),
            variant: LoadVariant::Plain,
            kind: LoadKind::Load,
            trigger: LoadTriggerKind::Admin,
            config_fingerprint: format!("{model}-config"),
            warning: None,
        }
    }

    fn only(registry: &LoadRegistry) -> LoadActivity {
        let mut loads = registry.list();
        assert_eq!(loads.len(), 1, "{loads:?}");
        loads.remove(0)
    }

    #[test]
    fn queued_load_is_listed_and_cancellable() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));

        let queued = only(&registry);
        assert_eq!(queued.load_id, handle.load_id());
        assert_eq!(queued.model, "alpha");
        assert_eq!(queued.state, LoadState::Queued);
        assert_eq!(queued.stage, LoadStage::Queued);
        assert_eq!(queued.started_unix_ms, 1_000);
        assert_eq!(queued.finished_unix_ms, None);

        assert_eq!(registry.cancel(handle.load_id()), Ok(()));
        assert!(handle.observer().is_cancelled());
        assert!(handle.tracker().is_cancelled());
        assert_eq!(only(&registry).state, LoadState::Cancelling);
        // Repeating a cancel the load has not acted on yet is harmless.
        assert_eq!(registry.cancel(handle.load_id()), Ok(()));

        handle.finish_cancelled();
        let cancelled = only(&registry);
        assert_eq!(cancelled.state, LoadState::Cancelled);
        assert_eq!(cancelled.code.as_deref(), Some("load_cancelled"));
        assert_eq!(cancelled.finished_unix_ms, Some(1_000));
    }

    #[test]
    fn finished_entry_is_retained_for_30_seconds() {
        let clock = ManualClock::new(1_000);
        let registry = LoadRegistry::with_clock(clock.clone());
        let handle = registry.begin(spec("alpha"));
        handle.finish_loaded();

        clock.advance(29_999);
        assert_eq!(only(&registry).state, LoadState::Loaded);
        clock.advance(1);
        assert!(registry.list().is_empty());
    }

    #[test]
    fn cancel_finished_returns_finished() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        let load_id = handle.load_id().to_string();
        handle.finish_loaded();

        assert_eq!(registry.cancel(&load_id), Err(LoadCancelError::Finished));
        assert_eq!(registry.cancel("missing"), Err(LoadCancelError::NotFound));

        let starting = registry.begin(spec("beta"));
        starting
            .tracker()
            .begin_finishing(LoadStage::StartingEngine)
            .expect("not cancelled");
        assert_eq!(
            registry.cancel(starting.load_id()),
            Err(LoadCancelError::Finishing)
        );
        assert!(!starting.observer().is_cancelled());
    }

    #[test]
    fn begin_finishing_refuses_a_cancelled_load() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        registry.cancel(handle.load_id()).expect("cancel");

        let error = handle
            .tracker()
            .begin_finishing(LoadStage::StartingEngine)
            .expect_err("cancelled load must not start its engine");
        assert!(error.downcast_ref::<LoadCancelled>().is_some());
        assert!(handle.tracker().checkpoint().is_err());
    }

    #[test]
    fn dropped_handle_marks_aborted() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        drop(registry.begin(spec("alpha")));

        let aborted = only(&registry);
        assert_eq!(aborted.state, LoadState::Failed);
        assert_eq!(aborted.code.as_deref(), Some("load_aborted"));
        assert_eq!(aborted.finished_unix_ms, Some(1_000));
    }

    #[test]
    fn in_progress_for_matches_model_id() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let alpha = registry.begin(spec("alpha"));
        let beta = registry.begin(spec("beta"));

        let found = registry
            .in_progress_for("alpha")
            .expect("alpha in progress");
        assert_eq!(found.load_id, alpha.load_id());
        assert_eq!(found.config_fingerprint, "alpha-config");
        assert!(registry.in_progress_for("gamma").is_none());

        alpha.finish_loaded();
        assert!(registry.in_progress_for("alpha").is_none());
        assert_eq!(
            registry.in_progress_for("beta").map(|load| load.load_id),
            Some(beta.load_id().to_string())
        );
        // A second load of an in-progress model is refused, not registered.
        let existing = registry
            .try_begin(spec("beta"))
            .expect_err("beta already loading");
        assert_eq!(existing.load_id, beta.load_id());
        assert_eq!(registry.list().len(), 2);
    }

    #[test]
    fn completed_stages_record_elapsed() {
        let clock = ManualClock::new(1_000);
        let registry = LoadRegistry::with_clock(clock.clone());
        let handle = registry.begin(spec("alpha"));
        clock.advance(5);
        handle.enter_stage(LoadStage::PreparingMemory);
        assert_eq!(only(&registry).state, LoadState::Loading);
        clock.advance(10);
        handle.enter_stage(LoadStage::ReadingWeights);
        clock.advance(20);
        handle.finish_loaded();

        let loaded = only(&registry);
        assert_eq!(loaded.state, LoadState::Loaded);
        assert_eq!(
            loaded.completed_stages,
            vec![
                CompletedStage {
                    stage: LoadStage::Queued,
                    elapsed_ms: 5
                },
                CompletedStage {
                    stage: LoadStage::PreparingMemory,
                    elapsed_ms: 10
                },
                CompletedStage {
                    stage: LoadStage::ReadingWeights,
                    elapsed_ms: 20
                },
            ]
        );
        assert_eq!(loaded.finished_unix_ms, Some(1_035));
    }

    #[test]
    fn weight_bytes_accumulate_across_components_and_never_decrease() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        let tracker = handle.tracker();
        tracker.set_component_estimate(WeightComponent::Main, 1_000);
        tracker.set_component_estimate(WeightComponent::Draft, 100);
        let observer = handle.observer();
        let bytes = |registry: &LoadRegistry| {
            let load = only(registry);
            (load.bytes_done, load.bytes_total)
        };

        assert_eq!(bytes(&registry), (0, 1_100));
        // The main loader's exact size replaces its directory estimate.
        observer.begin_weights(WeightComponent::Main, 900);
        observer.add_weight_bytes(300);
        assert_eq!(bytes(&registry), (300, 1_000));
        // The draft loader resets the observer; finished main bytes stay.
        observer.begin_weights(WeightComponent::Draft, 80);
        assert_eq!(bytes(&registry), (900, 980));
        observer.add_weight_bytes(40);
        assert_eq!(bytes(&registry), (940, 980));
        handle.finish_loaded();
        assert_eq!(bytes(&registry), (980, 980));
    }

    #[test]
    fn planned_total_is_a_placeholder_until_components_open() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        handle.set_planned_total(2_000);
        assert_eq!(only(&registry).bytes_total, 2_000);
        handle
            .observer()
            .begin_weights(WeightComponent::Main, 1_500);
        handle.observer().add_weight_bytes(1_500);
        let load = only(&registry);
        assert_eq!((load.bytes_done, load.bytes_total), (1_500, 2_000));
        handle.finish_loaded();
        let load = only(&registry);
        assert_eq!((load.bytes_done, load.bytes_total), (1_500, 1_500));
    }

    #[test]
    fn client_supplied_load_id_is_used() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let mut begin = spec("alpha");
        begin.load_id = Some("client-id".to_string());
        let handle = registry.begin(begin);
        assert_eq!(handle.load_id(), "client-id");
        assert_eq!(only(&registry).load_id, "client-id");

        // A colliding id is replaced rather than shadowing a live entry.
        let mut duplicate = spec("beta");
        duplicate.load_id = Some("client-id".to_string());
        let second = registry.begin(duplicate);
        assert_ne!(second.load_id(), "client-id");
    }

    #[test]
    fn failed_load_records_message() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        handle.finish_failed(&anyhow::anyhow!("disk on fire").context("loading alpha"));
        let failed = only(&registry);
        assert_eq!(failed.state, LoadState::Failed);
        assert_eq!(failed.code.as_deref(), Some("load_failed"));
        assert_eq!(
            failed.message.as_deref(),
            Some("loading alpha: disk on fire")
        );

        // A cancellation reported as an error still ends as cancelled.
        let handle = registry.begin(spec("beta"));
        handle.finish_failed(&anyhow::Error::from(LoadCancelled).context("loading beta"));
        assert_eq!(
            registry
                .list()
                .into_iter()
                .find(|load| load.model == "beta")
                .map(|load| load.state),
            Some(LoadState::Cancelled)
        );
    }

    #[test]
    fn failed_load_records_the_classified_reason() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        handle.finish_failed(
            &anyhow::anyhow!(
                "engine pool capacity reached: max_loaded_models=1, no idle lazy engine can be evicted"
            )
            .context("loading engine model `alpha`"),
        );
        let failed = only(&registry);
        assert_eq!(failed.code.as_deref(), Some("max_loaded_models_reached"));
        assert_eq!(
            failed.message.as_deref(),
            Some(crate::core::load_error::MAX_LOADED_MODELS_REACHED_MESSAGE)
        );
    }

    #[test]
    fn loaded_load_reports_its_warning() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let mut begin = spec("alpha");
        begin.warning = Some(LoadWarning {
            code: "scheduler_profile_unusable".to_string(),
            message: "default profile".to_string(),
        });
        let handle = registry.begin(begin);
        // Only the result carries the warning.
        assert_eq!(only(&registry).warning_code, None);
        handle.finish_loaded();
        let loaded = only(&registry);
        assert_eq!(
            loaded.warning_code.as_deref(),
            Some("scheduler_profile_unusable")
        );
        assert_eq!(loaded.warning.as_deref(), Some("default profile"));
        let value = serde_json::to_value(&loaded).expect("serialize");
        assert_eq!(value["warning_code"], "scheduler_profile_unusable");
    }

    #[test]
    fn activity_serializes_as_snake_case() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let mut begin = spec("alpha");
        begin.variant = LoadVariant::Dflash2;
        begin.trigger = LoadTriggerKind::DeferredReload;
        let handle = registry.begin(begin);
        handle.enter_stage(LoadStage::ReadingDraftWeights);
        let value = serde_json::to_value(only(&registry)).expect("serialize");
        assert_eq!(value["variant"], "dflash2");
        assert_eq!(value["kind"], "load");
        assert_eq!(value["trigger"], "deferred_reload");
        assert_eq!(value["state"], "loading");
        assert_eq!(value["stage"], "reading_draft_weights");
        assert_eq!(value["completed_stages"][0]["stage"], "queued");
        assert!(value.get("config_fingerprint").is_none());
        drop(handle);
    }

    #[tokio::test]
    async fn wait_returns_the_terminal_activity() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        let load_id = handle.load_id().to_string();
        let waiter = {
            let registry = registry.clone();
            let load_id = load_id.clone();
            tokio::spawn(async move { registry.wait(&load_id).await })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        handle.finish_loaded();
        let finished = waiter.await.expect("join").expect("known load");
        assert_eq!(finished.state, LoadState::Loaded);
        assert_eq!(
            registry.wait(&load_id).await.map(|load| load.state),
            Some(LoadState::Loaded)
        );
        assert!(registry.wait("missing").await.is_none());
    }

    #[tokio::test]
    async fn tracker_cancelled_resolves_on_cancel() {
        let registry = LoadRegistry::with_clock(ManualClock::new(1_000));
        let handle = registry.begin(spec("alpha"));
        let tracker = handle.tracker();
        let waiting = tokio::spawn(async move { tracker.cancelled().await });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        registry.cancel(handle.load_id()).expect("cancel");
        tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .expect("cancellation wakes the tracker")
            .expect("join");
        handle.finish_cancelled();
    }
}
