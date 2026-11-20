//! Pure computation of the runtime-metrics snapshot shown by the App's
//! glance surfaces (menu bar, card, HUD).
//!
//! Nothing here blocks or shells out: every source reads atomics, the usage
//! counters, or an already-sampled governor snapshot.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{watch, Notify};

use crate::core::cache::process_async_prefix_store_queue;
use crate::core::process_memory::{
    global_process_memory_governor, MemoryGovernorSnapshot, PressureLevel,
    SharedProcessMemoryGovernor,
};
use crate::core::runtime_health::{
    classify_process_status, scheduler_degraded_reasons, HealthDegradedReason, HealthStatus,
    SchedulerHealthCollector,
};
use crate::core::runtime_usage::{
    ModelRuntimeUsageCounters, ModelRuntimeUsageSnapshot, PrefixCacheUsageSnapshot,
};

const BYTE_CHANGE_THRESHOLD: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityState {
    Idle,
    Prefilling,
    Generating,
    Loading,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Loading,
    Loaded,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelMode {
    Normal,
    Dflash2,
    Mtp,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BackendInfo {
    pub version: String,
    pub started_unix_ms: u64,
    pub device_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HealthView {
    pub status: HealthStatus,
    pub degraded_reasons: Vec<HealthDegradedReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActivityView {
    pub state: ActivityState,
    pub active_requests: u64,
    pub prefilling_requests: u64,
    pub queued_requests: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceView {
    pub live_decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub window_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryView {
    pub pressure_level: PressureLevel,
    pub process_bytes: u64,
    pub ceiling_bytes: u64,
    pub kv_cache_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub telemetry_degraded: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelView {
    pub id: String,
    pub kind: String,
    pub state: ModelState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ModelMode>,
    pub is_default: bool,
    pub active_requests: u64,
    pub queued_requests: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_capacity: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_decode_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeMetricsSnapshot {
    pub schema_version: u32,
    pub seq: u64,
    pub unix_ms: u64,
    pub backend: BackendInfo,
    pub health: HealthView,
    pub activity: ActivityView,
    pub performance: PerformanceView,
    pub memory: MemoryView,
    pub prefix_cache: PrefixCacheUsageSnapshot,
    pub models: Vec<ModelView>,
}

/// One reading of a loaded model's live counters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelMetricsReading {
    pub active_requests: u64,
    pub queued_requests: u64,
    pub queue_capacity: Option<u64>,
    pub decoding_requests: u64,
    pub kv_cache_bytes: u64,
    pub kv_cache_soft_limit_bytes: u64,
    pub active_kv_offload_degraded: bool,
    /// `Some` only for language models (`causal`, `block_diffusion`).
    pub usage: Option<ModelRuntimeUsageSnapshot>,
}

pub trait ModelMetricsSource: Send + Sync {
    /// `None` means the engine no longer exists.
    fn read(&self) -> Option<ModelMetricsReading>;

    fn attach_request_waker(&self, _waker: &Arc<Notify>) {}
}

/// Reads a causal-LM scheduler's atomics and usage counters. Never calls
/// `SchedulerHealthCollector::snapshot()`, which runs `vm_stat`.
pub struct CausalMetricsSource {
    health: Arc<SchedulerHealthCollector>,
    usage: Arc<ModelRuntimeUsageCounters>,
}

impl CausalMetricsSource {
    pub fn new(
        health: Arc<SchedulerHealthCollector>,
        usage: Arc<ModelRuntimeUsageCounters>,
    ) -> Self {
        Self { health, usage }
    }
}

impl ModelMetricsSource for CausalMetricsSource {
    fn read(&self) -> Option<ModelMetricsReading> {
        let health = &self.health;
        Some(ModelMetricsReading {
            active_requests: health.b_active.load(Ordering::Relaxed),
            queued_requests: health.b_queued.load(Ordering::Relaxed),
            queue_capacity: Some(health.queue_max as u64),
            decoding_requests: self.usage.decoding_requests(),
            kv_cache_bytes: health.kv_cache_active_bytes.load(Ordering::Relaxed) as u64,
            kv_cache_soft_limit_bytes: health.kv_cache_soft_limit_bytes as u64,
            active_kv_offload_degraded: health.active_kv_offload.snapshot().degraded,
            usage: Some(self.usage.snapshot(true)),
        })
    }

    fn attach_request_waker(&self, waker: &Arc<Notify>) {
        self.usage.attach_request_waker(waker.clone());
    }
}

pub(crate) struct HubEntryView<'a> {
    pub id: &'a str,
    pub kind: &'a str,
    pub state: ModelState,
    pub mode: Option<ModelMode>,
    pub is_default: bool,
    pub last_error: Option<&'a str>,
    pub reading: Option<&'a ModelMetricsReading>,
}

fn is_language_kind(kind: &str) -> bool {
    matches!(kind, "causal" | "block_diffusion")
}

pub(crate) fn build_snapshot(
    entries: &[HubEntryView],
    retired_prefix: PrefixCacheUsageSnapshot,
    governor: &MemoryGovernorSnapshot,
    prefix_store_backpressured: bool,
    backend: &BackendInfo,
    seq: u64,
    unix_ms: u64,
) -> RuntimeMetricsSnapshot {
    let mut active_requests = 0_u64;
    let mut queued_requests = 0_u64;
    let mut prefilling_requests = 0_u64;
    let mut decoding_language = 0_u64;
    let mut active_non_language = 0_u64;
    let mut kv_cache_bytes = 0_u64;
    let mut any_loading = false;
    let mut any_failed = false;
    let mut live_sum: Option<f64> = None;
    let mut prefix = retired_prefix;
    let mut scheduler_reasons: Vec<HealthDegradedReason> = Vec::new();
    // Language model with the most completed requests in its window.
    let mut window_best: Option<(&str, &ModelMetricsReading)> = None;
    let mut models = Vec::with_capacity(entries.len());

    for entry in entries {
        let language = is_language_kind(entry.kind);
        any_loading |= entry.state == ModelState::Loading;
        any_failed |= entry.state == ModelState::Failed;
        let mut live = None;
        if let Some(reading) = entry.reading {
            active_requests += reading.active_requests;
            queued_requests += reading.queued_requests;
            kv_cache_bytes += reading.kv_cache_bytes;
            if let Some(cache) = reading.usage.as_ref().and_then(|u| u.prefix_cache) {
                prefix.hit_tokens += cache.hit_tokens;
                prefix.eligible_tokens += cache.eligible_tokens;
            }
            if language {
                prefilling_requests += reading
                    .active_requests
                    .saturating_sub(reading.decoding_requests);
                decoding_language += reading.decoding_requests;
                live = reading
                    .usage
                    .as_ref()
                    .and_then(|u| u.performance.live_decode_tokens_per_second);
                if let Some(rate) = live {
                    *live_sum.get_or_insert(0.0) += rate;
                }
                for reason in scheduler_degraded_reasons(
                    reading.queued_requests as usize,
                    reading.queue_capacity.unwrap_or(0) as usize,
                    reading.kv_cache_bytes as usize,
                    reading.kv_cache_soft_limit_bytes as usize,
                    reading.active_kv_offload_degraded,
                    false,
                ) {
                    if !scheduler_reasons.contains(&reason) {
                        scheduler_reasons.push(reason);
                    }
                }
                if let Some(usage) = &reading.usage {
                    let completed = usage.performance.completed_requests;
                    let better = match window_best {
                        None => completed > 0,
                        Some((best_id, best)) => {
                            let best_completed = best
                                .usage
                                .as_ref()
                                .map_or(0, |u| u.performance.completed_requests);
                            completed > best_completed
                                || (completed == best_completed
                                    && completed > 0
                                    && entry.id < best_id)
                        }
                    };
                    if better {
                        window_best = Some((entry.id, reading));
                    }
                }
            } else {
                active_non_language += reading.active_requests;
            }
        }
        models.push(ModelView {
            id: entry.id.to_string(),
            kind: entry.kind.to_string(),
            state: entry.state,
            mode: entry.mode,
            is_default: entry.is_default,
            active_requests: entry.reading.map_or(0, |r| r.active_requests),
            queued_requests: entry.reading.map_or(0, |r| r.queued_requests),
            queue_capacity: entry.reading.and_then(|r| r.queue_capacity),
            live_decode_tokens_per_second: live,
            last_error: entry.last_error.map(str::to_string),
        });
    }

    let state = if any_loading {
        ActivityState::Loading
    } else if decoding_language > 0 || active_non_language > 0 {
        ActivityState::Generating
    } else if prefilling_requests > 0 {
        ActivityState::Prefilling
    } else {
        ActivityState::Idle
    };

    let window = window_best.and_then(|(_, reading)| reading.usage.as_ref());
    let performance = PerformanceView {
        live_decode_tokens_per_second: live_sum,
        prefill_tokens_per_second: window.and_then(|u| u.performance.prefill_tokens_per_second),
        decode_tokens_per_second: window.and_then(|u| u.performance.decode_tokens_per_second),
        ttft_ms: window.and_then(|u| u.performance.ttft_ms),
        window_seconds: window.map_or(60, |u| u.performance.window_seconds),
    };

    if prefix_store_backpressured {
        scheduler_reasons.push(HealthDegradedReason::PrefixStoreBackpressured);
    }
    if any_failed {
        scheduler_reasons.push(HealthDegradedReason::ModelFailed);
    }
    // Declaration order, so the output does not depend on entry order.
    scheduler_reasons.sort();
    let (status, degraded_reasons) = classify_process_status(governor, scheduler_reasons);

    RuntimeMetricsSnapshot {
        schema_version: 1,
        seq,
        unix_ms,
        backend: backend.clone(),
        health: HealthView {
            status,
            degraded_reasons,
        },
        activity: ActivityView {
            state,
            active_requests,
            prefilling_requests,
            queued_requests,
        },
        performance,
        memory: MemoryView {
            pressure_level: governor.pressure_level,
            process_bytes: governor.current_usage_bytes as u64,
            ceiling_bytes: governor.effective_ceiling_bytes as u64,
            kv_cache_bytes,
            mlx_cache_bytes: governor.mlx_cache_bytes as u64,
            telemetry_degraded: governor.telemetry_degraded,
        },
        prefix_cache: prefix,
        models,
    }
}

fn quantize(value: Option<f64>, steps_per_unit: f64) -> Option<i64> {
    value.map(|v| (v * steps_per_unit).round() as i64)
}

fn bytes_changed(prev: u64, next: u64) -> bool {
    prev.abs_diff(next) > BYTE_CHANGE_THRESHOLD
}

/// The governor's effective ceiling follows free system memory and moves by
/// hundreds of MiB while idle; only a move beyond 5% of the last pushed value
/// (and never less than the byte threshold) counts.
fn ceiling_changed(prev: u64, next: u64) -> bool {
    prev.abs_diff(next) > BYTE_CHANGE_THRESHOLD.max(prev / 20)
}

/// Whether `next` differs from the last pushed snapshot `prev` enough to push.
pub(crate) fn is_meaningful_change(
    prev: &RuntimeMetricsSnapshot,
    next: &RuntimeMetricsSnapshot,
) -> bool {
    let speed = |v| quantize(v, 10.0);
    let (pp, np) = (&prev.performance, &next.performance);
    if speed(pp.live_decode_tokens_per_second) != speed(np.live_decode_tokens_per_second)
        || speed(pp.prefill_tokens_per_second) != speed(np.prefill_tokens_per_second)
        || speed(pp.decode_tokens_per_second) != speed(np.decode_tokens_per_second)
        || quantize(pp.ttft_ms, 1.0) != quantize(np.ttft_ms, 1.0)
        || pp.window_seconds != np.window_seconds
    {
        return true;
    }
    let (pm, nm) = (&prev.memory, &next.memory);
    if pm.pressure_level != nm.pressure_level
        || pm.telemetry_degraded != nm.telemetry_degraded
        || bytes_changed(pm.process_bytes, nm.process_bytes)
        || ceiling_changed(pm.ceiling_bytes, nm.ceiling_bytes)
        || bytes_changed(pm.kv_cache_bytes, nm.kv_cache_bytes)
        || bytes_changed(pm.mlx_cache_bytes, nm.mlx_cache_bytes)
    {
        return true;
    }
    if prev.schema_version != next.schema_version
        || prev.backend != next.backend
        || prev.health != next.health
        || prev.activity != next.activity
        || prev.prefix_cache != next.prefix_cache
        || prev.models.len() != next.models.len()
    {
        return true;
    }
    prev.models.iter().zip(&next.models).any(|(a, b)| {
        let mut a = a.clone();
        let mut b = b.clone();
        let (la, lb) = (
            speed(a.live_decode_tokens_per_second.take()),
            speed(b.live_decode_tokens_per_second.take()),
        );
        la != lb || a != b
    })
}

/// How the hub obtains the memory governor snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySampling {
    /// Another loop (`start_memory_governor_monitor`) keeps the governor
    /// sampled; the hub only reads its cached snapshot.
    Cached,
    /// Nothing else samples the governor (DFlash2 exclusive mode): the hub
    /// samples it once before every computation, so the cached snapshot does
    /// not go stale and report degraded telemetry while idle.
    SampleOnTick,
}

const ACTIVE_TICK: Duration = Duration::from_secs(1);
const IDLE_TICK: Duration = Duration::from_secs(5);

struct HubEntry {
    kind: &'static str,
    state: ModelState,
    mode: Option<ModelMode>,
    last_error: Option<String>,
    source: Option<Arc<dyn ModelMetricsSource>>,
    /// Last reading `source` returned, used when the engine is already gone.
    last_reading: Option<ModelMetricsReading>,
}

impl HubEntry {
    fn new(kind: &'static str, state: ModelState, mode: Option<ModelMode>) -> Self {
        Self {
            kind,
            state,
            mode,
            last_error: None,
            source: None,
            last_reading: None,
        }
    }

    /// Read the source, remembering the reading. When the engine is gone,
    /// keep only the cached prefix-cache totals so the process-wide sum does
    /// not dip before the entry is removed.
    fn read(&mut self) -> Option<ModelMetricsReading> {
        let source = self.source.as_ref()?;
        if let Some(reading) = source.read() {
            self.last_reading = Some(reading.clone());
            return Some(reading);
        }
        let prefix_cache = self
            .last_reading
            .as_ref()
            .and_then(|r| r.usage.as_ref())
            .and_then(|u| u.prefix_cache);
        prefix_cache.map(|prefix_cache| ModelMetricsReading {
            usage: Some(ModelRuntimeUsageSnapshot {
                prefix_cache: Some(prefix_cache),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Final prefix-cache totals of an entry that is being replaced or
    /// removed: a fresh read if the engine still answers, else the last one.
    fn final_prefix_cache(&mut self) -> Option<PrefixCacheUsageSnapshot> {
        self.read()
            .and_then(|reading| reading.usage)
            .and_then(|usage| usage.prefix_cache)
    }
}

fn same_source(a: &Arc<dyn ModelMetricsSource>, b: &Arc<dyn ModelMetricsSource>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

#[derive(Default)]
struct Registry {
    /// Ordered by id, so the snapshot does not depend on registration order.
    entries: BTreeMap<String, HubEntry>,
    default_model: Option<String>,
    /// Prefix-cache totals of models that were unloaded or reloaded.
    retired_prefix: PrefixCacheUsageSnapshot,
}

impl Registry {
    fn retire(&mut self, mut entry: HubEntry) {
        if entry.state != ModelState::Loaded {
            return;
        }
        if let Some(totals) = entry.final_prefix_cache() {
            self.retired_prefix.hit_tokens += totals.hit_tokens;
            self.retired_prefix.eligible_tokens += totals.eligible_tokens;
        }
    }

    fn replace(&mut self, id: &str, entry: HubEntry) {
        let Some(old) = self.entries.insert(id.to_string(), entry) else {
            return;
        };
        let new_source = self.entries[id].source.as_ref();
        let same = matches!((&old.source, new_source), (Some(a), Some(b)) if same_source(a, b));
        if !same {
            self.retire(old);
        }
    }

    fn build(
        &mut self,
        governor: &MemoryGovernorSnapshot,
        prefix_store_backpressured: bool,
        backend: &BackendInfo,
        unix_ms: u64,
    ) -> RuntimeMetricsSnapshot {
        let readings: Vec<Option<ModelMetricsReading>> =
            self.entries.values_mut().map(HubEntry::read).collect();
        let views: Vec<HubEntryView> = self
            .entries
            .iter()
            .zip(&readings)
            .map(|((id, entry), reading)| HubEntryView {
                id,
                kind: entry.kind,
                state: entry.state,
                mode: entry.mode,
                is_default: self.default_model.as_deref() == Some(id.as_str()),
                last_error: entry.last_error.as_deref(),
                reading: reading.as_ref(),
            })
            .collect();
        build_snapshot(
            &views,
            self.retired_prefix,
            governor,
            prefix_store_backpressured,
            backend,
            0,
            unix_ms,
        )
    }
}

/// State the sampling task shares with the hub without keeping it alive.
struct HubSignals {
    /// Wakes the sampling task before its next tick.
    waker: Arc<Notify>,
    /// Signals that a subscriber was added (or that the hub was dropped).
    subscribers: Notify,
    /// Latest published snapshot; `seq == 0` until the first publish.
    sender: watch::Sender<Arc<RuntimeMetricsSnapshot>>,
    /// Whether a sampling task is running; cleared when it exits, including
    /// by panic, so the next `subscribe` starts a new one.
    task_running: AtomicBool,
    /// Bumped every time a sampling task exits. Open streams end on it:
    /// without a task their receivers would keep the last frame forever.
    task_exits: watch::Sender<u64>,
}

/// Lives inside the sampling task; its drop runs on every exit path.
struct SamplingTaskGuard(Arc<HubSignals>);

impl Drop for SamplingTaskGuard {
    fn drop(&mut self) {
        self.0.task_running.store(false, Ordering::Release);
        self.0
            .task_exits
            .send_modify(|exits| *exits = exits.wrapping_add(1));
    }
}

/// Process-wide source of runtime-metrics snapshots: keeps the model
/// registry, recomputes the snapshot on a 1 s (busy) / 5 s (idle) cadence
/// while anyone subscribes, and publishes only meaningful changes.
pub struct RuntimeMetricsHub {
    memory: MemorySampling,
    backend: BackendInfo,
    governor: SharedProcessMemoryGovernor,
    prefix_store_backpressured: fn() -> bool,
    registry: Mutex<Registry>,
    signals: Arc<HubSignals>,
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn empty_snapshot(backend: &BackendInfo) -> RuntimeMetricsSnapshot {
    build_snapshot(
        &[],
        PrefixCacheUsageSnapshot::default(),
        &MemoryGovernorSnapshot::default(),
        false,
        backend,
        0,
        0,
    )
}

impl RuntimeMetricsHub {
    pub fn new(memory: MemorySampling) -> Arc<Self> {
        Self::with_environment(
            memory,
            global_process_memory_governor(),
            || process_async_prefix_store_queue().is_backpressured(),
            mlx::memory::snapshot().device_name,
        )
    }

    fn with_environment(
        memory: MemorySampling,
        governor: SharedProcessMemoryGovernor,
        prefix_store_backpressured: fn() -> bool,
        device_name: Option<String>,
    ) -> Arc<Self> {
        let backend = BackendInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            started_unix_ms: now_unix_ms(),
            device_name,
        };
        // Placeholder (`seq == 0`): `subscribe` publishes a computed snapshot
        // before it hands out a receiver, so no subscriber ever sees it.
        let (sender, _) = watch::channel(Arc::new(empty_snapshot(&backend)));
        Arc::new(Self {
            memory,
            backend,
            governor,
            prefix_store_backpressured,
            registry: Mutex::new(Registry::default()),
            signals: Arc::new(HubSignals {
                waker: Arc::new(Notify::new()),
                subscribers: Notify::new(),
                sender,
                task_running: AtomicBool::new(false),
                task_exits: watch::Sender::new(0),
            }),
        })
    }

    fn registry(&self) -> MutexGuard<'_, Registry> {
        self.registry.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Recompute before the next tick (request start, registry change,
    /// memory pressure change).
    pub fn wake(&self) {
        self.signals.waker.notify_one();
    }

    pub fn set_loading(&self, id: &str, kind: &'static str, mode: Option<ModelMode>) {
        self.registry()
            .replace(id, HubEntry::new(kind, ModelState::Loading, mode));
        self.wake();
    }

    pub fn set_loaded(
        &self,
        id: &str,
        kind: &'static str,
        mode: Option<ModelMode>,
        source: Arc<dyn ModelMetricsSource>,
    ) {
        source.attach_request_waker(&self.signals.waker);
        let mut entry = HubEntry::new(kind, ModelState::Loaded, mode);
        entry.source = Some(source);
        self.registry().replace(id, entry);
        self.wake();
    }

    /// The entry stays, with `error`, until the next `set_loading` or `remove`.
    pub fn set_failed(&self, id: &str, kind: &'static str, mode: Option<ModelMode>, error: String) {
        let mut entry = HubEntry::new(kind, ModelState::Failed, mode);
        entry.last_error = Some(error);
        self.registry().replace(id, entry);
        self.wake();
    }

    pub fn remove(&self, id: &str) {
        {
            let mut registry = self.registry();
            if let Some(entry) = registry.entries.remove(id) {
                registry.retire(entry);
            }
        }
        self.wake();
    }

    pub fn set_default_model(&self, id: Option<&str>) {
        self.registry().default_model = id.map(str::to_string);
        self.wake();
    }

    /// Receiver of the published snapshots. Its current value is computed
    /// now, and the sampling task runs while any receiver is alive; a task
    /// that has died is started again here.
    pub fn subscribe(self: &Arc<Self>) -> watch::Receiver<Arc<RuntimeMetricsSnapshot>> {
        self.refresh();
        let receiver = self.signals.sender.subscribe();
        if !self.signals.task_running.swap(true, Ordering::AcqRel) {
            let guard = SamplingTaskGuard(Arc::clone(&self.signals));
            tokio::spawn(run_sampling_task(Arc::downgrade(self), guard));
        }
        self.signals.subscribers.notify_one();
        receiver
    }

    /// Resolves when the sampling task running at (or started after) this
    /// call exits, after which receivers get no new snapshots. Take it
    /// before `subscribe` and end the stream on it, so the client reconnects
    /// and its `subscribe` starts a new task.
    pub fn sampling_stopped(&self) -> impl Future<Output = ()> + Send + 'static {
        let mut exits = self.signals.task_exits.subscribe();
        async move {
            let _ = exits.changed().await;
        }
    }

    /// One computation for the single-shot JSON endpoint; never published.
    pub fn snapshot_now(&self) -> RuntimeMetricsSnapshot {
        let mut snapshot = self.compute();
        snapshot.seq = self.signals.sender.borrow().seq;
        snapshot
    }

    fn governor_snapshot(&self) -> MemoryGovernorSnapshot {
        if self.memory == MemorySampling::SampleOnTick {
            self.governor.sample_process();
        }
        self.governor.snapshot()
    }

    fn compute(&self) -> RuntimeMetricsSnapshot {
        let governor = self.governor_snapshot();
        let backpressured = (self.prefix_store_backpressured)();
        self.registry()
            .build(&governor, backpressured, &self.backend, now_unix_ms())
    }

    /// Compute and publish if this is the first snapshot or it differs
    /// meaningfully from the last published one. Returns whether the
    /// runtime is busy (requests in flight or a model loading).
    fn refresh(&self) -> bool {
        let next = self.compute();
        let busy = next.activity.state != ActivityState::Idle;
        let mut next = Some(next);
        self.signals.sender.send_if_modified(|current| {
            let mut next = next
                .take()
                .expect("send_if_modified calls the closure once");
            if current.seq != 0 && !is_meaningful_change(current, &next) {
                return false;
            }
            next.seq = current.seq + 1;
            *current = Arc::new(next);
            true
        });
        busy
    }
}

impl Drop for RuntimeMetricsHub {
    fn drop(&mut self) {
        // Wake the sampling task wherever it waits, so it sees the hub is
        // gone and exits (dropping the sender closes every receiver).
        self.signals.subscribers.notify_one();
        self.signals.waker.notify_one();
    }
}

async fn run_sampling_task(hub: Weak<RuntimeMetricsHub>, guard: SamplingTaskGuard) {
    let signals = &guard.0;
    loop {
        if hub.strong_count() == 0 {
            return;
        }
        if signals.sender.receiver_count() == 0 {
            // No timer while nobody listens.
            signals.subscribers.notified().await;
            continue;
        }
        let Some(strong) = hub.upgrade() else {
            return;
        };
        let busy = strong.refresh();
        drop(strong);
        let tick = if busy { ACTIVE_TICK } else { IDLE_TICK };
        tokio::select! {
            _ = tokio::time::sleep(tick) => {}
            _ = signals.waker.notified() => {}
            _ = signals.sender.closed() => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::core::runtime_usage::ModelRuntimePerformanceSnapshot;

    const MIB: u64 = 1024 * 1024;

    fn backend() -> BackendInfo {
        BackendInfo {
            version: "0.5.0".into(),
            started_unix_ms: 1,
            device_name: Some("Apple M5 Max".into()),
        }
    }

    fn lm_reading(active: u64, decoding: u64) -> ModelMetricsReading {
        ModelMetricsReading {
            active_requests: active,
            decoding_requests: decoding,
            usage: Some(ModelRuntimeUsageSnapshot::default()),
            ..Default::default()
        }
    }

    fn with_perf(
        mut reading: ModelMetricsReading,
        completed: usize,
        live: Option<f64>,
        decode: Option<f64>,
    ) -> ModelMetricsReading {
        let usage = reading.usage.as_mut().unwrap();
        usage.performance = ModelRuntimePerformanceSnapshot {
            completed_requests: completed,
            live_decode_tokens_per_second: live,
            decode_tokens_per_second: decode,
            prefill_tokens_per_second: Some(1000.0),
            ttft_ms: Some(200.0),
            ..Default::default()
        };
        reading
    }

    fn entry<'a>(
        id: &'a str,
        kind: &'a str,
        state: ModelState,
        reading: Option<&'a ModelMetricsReading>,
    ) -> HubEntryView<'a> {
        HubEntryView {
            id,
            kind,
            state,
            mode: None,
            is_default: false,
            last_error: None,
            reading,
        }
    }

    fn build(entries: &[HubEntryView]) -> RuntimeMetricsSnapshot {
        build_snapshot(
            entries,
            PrefixCacheUsageSnapshot::default(),
            &MemoryGovernorSnapshot::default(),
            false,
            &backend(),
            1,
            1,
        )
    }

    #[test]
    fn activity_prefers_loading_then_generating_then_prefilling() {
        let decoding = lm_reading(1, 1);
        let loading = entry("a", "causal", ModelState::Loading, None);
        let running = entry("b", "causal", ModelState::Loaded, Some(&decoding));
        let both = [loading, running];
        assert_eq!(build(&both).activity.state, ActivityState::Loading);

        let running = entry("b", "causal", ModelState::Loaded, Some(&decoding));
        assert_eq!(build(&[running]).activity.state, ActivityState::Generating);

        let prefill = lm_reading(2, 0);
        let snap = build(&[entry("b", "causal", ModelState::Loaded, Some(&prefill))]);
        assert_eq!(snap.activity.state, ActivityState::Prefilling);
        assert_eq!(snap.activity.prefilling_requests, 2);

        let idle = lm_reading(0, 0);
        let snap = build(&[entry("b", "causal", ModelState::Loaded, Some(&idle))]);
        assert_eq!(snap.activity.state, ActivityState::Idle);
    }

    #[test]
    fn non_language_activity_counts_as_generating_without_speed() {
        let reading = ModelMetricsReading {
            active_requests: 1,
            ..Default::default()
        };
        let snap = build(&[entry("t", "tts", ModelState::Loaded, Some(&reading))]);
        assert_eq!(snap.activity.state, ActivityState::Generating);
        assert_eq!(snap.activity.active_requests, 1);
        assert_eq!(snap.performance.live_decode_tokens_per_second, None);
    }

    #[test]
    fn window_metrics_come_from_model_with_most_completed_requests() {
        let a = with_perf(lm_reading(1, 1), 3, Some(10.0), Some(40.0));
        let b = with_perf(lm_reading(1, 1), 5, Some(20.0), Some(50.0));
        let snap = build(&[
            entry("a", "causal", ModelState::Loaded, Some(&a)),
            entry("b", "causal", ModelState::Loaded, Some(&b)),
        ]);
        assert_eq!(snap.performance.decode_tokens_per_second, Some(50.0));
        assert_eq!(snap.performance.live_decode_tokens_per_second, Some(30.0));

        let none = lm_reading(0, 0);
        let snap = build(&[entry("a", "causal", ModelState::Loaded, Some(&none))]);
        assert_eq!(snap.performance.decode_tokens_per_second, None);
        assert_eq!(snap.performance.ttft_ms, None);
    }

    #[test]
    fn window_tie_prefers_smallest_id() {
        let a = with_perf(lm_reading(0, 0), 4, None, Some(40.0));
        let b = with_perf(lm_reading(0, 0), 4, None, Some(50.0));
        let snap = build(&[
            entry("b", "causal", ModelState::Loaded, Some(&b)),
            entry("a", "causal", ModelState::Loaded, Some(&a)),
        ]);
        assert_eq!(snap.performance.decode_tokens_per_second, Some(40.0));
    }

    #[test]
    fn failed_entry_adds_model_failed_and_keeps_last_error() {
        let mut failed = entry("asr", "asr", ModelState::Failed, None);
        failed.last_error = Some("model weights not found");
        let snap = build(&[failed]);
        assert_eq!(snap.health.status, HealthStatus::Degraded);
        assert!(snap
            .health
            .degraded_reasons
            .contains(&HealthDegradedReason::ModelFailed));
        assert_eq!(
            snap.models[0].last_error.as_deref(),
            Some("model weights not found")
        );
    }

    #[test]
    fn queue_and_kv_reasons_match_classify_status() {
        let reading = ModelMetricsReading {
            queued_requests: 2,
            queue_capacity: Some(4),
            kv_cache_bytes: 95,
            kv_cache_soft_limit_bytes: 100,
            usage: Some(ModelRuntimeUsageSnapshot::default()),
            ..Default::default()
        };
        // The default governor snapshot reports degraded telemetry; clear it so
        // only scheduler reasons remain.
        let governor = MemoryGovernorSnapshot {
            telemetry_degraded: false,
            ..Default::default()
        };
        let snap = build_snapshot(
            &[entry("a", "causal", ModelState::Loaded, Some(&reading))],
            PrefixCacheUsageSnapshot::default(),
            &governor,
            false,
            &backend(),
            1,
            1,
        );
        assert_eq!(
            snap.health.degraded_reasons,
            vec![
                HealthDegradedReason::SchedulerQueueHigh,
                HealthDegradedReason::KvCacheNearSoftLimit
            ]
        );
    }

    #[test]
    fn reasons_are_ordered_by_declaration_not_entry_order() {
        let a = ModelMetricsReading {
            kv_cache_bytes: 95,
            kv_cache_soft_limit_bytes: 100,
            usage: Some(ModelRuntimeUsageSnapshot::default()),
            ..Default::default()
        };
        let b = ModelMetricsReading {
            queued_requests: 2,
            queue_capacity: Some(4),
            usage: Some(ModelRuntimeUsageSnapshot::default()),
            ..Default::default()
        };
        let governor = MemoryGovernorSnapshot {
            telemetry_degraded: false,
            ..Default::default()
        };
        let snap = build_snapshot(
            &[
                entry("a", "causal", ModelState::Loaded, Some(&a)),
                entry("b", "causal", ModelState::Loaded, Some(&b)),
            ],
            PrefixCacheUsageSnapshot::default(),
            &governor,
            false,
            &backend(),
            1,
            1,
        );
        assert_eq!(
            snap.health.degraded_reasons,
            vec![
                HealthDegradedReason::SchedulerQueueHigh,
                HealthDegradedReason::KvCacheNearSoftLimit
            ]
        );
    }

    #[test]
    fn prefix_cache_adds_retired_totals() {
        let mut reading = lm_reading(0, 0);
        reading.usage.as_mut().unwrap().prefix_cache = Some(PrefixCacheUsageSnapshot {
            hit_tokens: 5,
            eligible_tokens: 10,
        });
        let snap = build_snapshot(
            &[entry("a", "causal", ModelState::Loaded, Some(&reading))],
            PrefixCacheUsageSnapshot {
                hit_tokens: 100,
                eligible_tokens: 200,
            },
            &MemoryGovernorSnapshot::default(),
            false,
            &backend(),
            1,
            1,
        );
        assert_eq!(
            snap.prefix_cache,
            PrefixCacheUsageSnapshot {
                hit_tokens: 105,
                eligible_tokens: 210
            }
        );
    }

    fn base_snapshot() -> RuntimeMetricsSnapshot {
        let reading = with_perf(lm_reading(1, 1), 1, Some(42.8), Some(44.0));
        build(&[entry("a", "causal", ModelState::Loaded, Some(&reading))])
    }

    #[test]
    fn meaningful_change_thresholds() {
        let prev = base_snapshot();

        let mut next = prev.clone();
        next.seq += 1;
        next.unix_ms += 1000;
        assert!(!is_meaningful_change(&prev, &next));

        let mut next = prev.clone();
        next.memory.process_bytes += 63 * MIB;
        assert!(!is_meaningful_change(&prev, &next));
        next.memory.process_bytes = prev.memory.process_bytes + 65 * MIB;
        assert!(is_meaningful_change(&prev, &next));

        let mut a = prev.clone();
        a.performance.live_decode_tokens_per_second = Some(42.81);
        let mut b = prev.clone();
        b.performance.live_decode_tokens_per_second = Some(42.84);
        assert!(!is_meaningful_change(&a, &b));
        a.performance.live_decode_tokens_per_second = Some(42.8);
        b.performance.live_decode_tokens_per_second = Some(42.9);
        assert!(is_meaningful_change(&a, &b));

        let mut a = prev.clone();
        a.performance.ttft_ms = Some(312.0);
        let mut b = prev.clone();
        b.performance.ttft_ms = None;
        assert!(is_meaningful_change(&a, &b));

        let mut next = prev.clone();
        next.memory.pressure_level = PressureLevel::Soft;
        assert!(is_meaningful_change(&prev, &next));

        // The ceiling follows free system memory: only a move beyond 5% of
        // the last pushed ceiling (and at least 64 MiB) counts.
        let gib = 1024 * MIB;
        let mut a = prev.clone();
        a.memory.ceiling_bytes = 100 * gib;
        let mut b = a.clone();
        b.memory.ceiling_bytes = 100 * gib + 1229 * MIB;
        assert!(!is_meaningful_change(&a, &b));
        b.memory.ceiling_bytes = 106 * gib;
        assert!(is_meaningful_change(&a, &b));
        a.memory.ceiling_bytes = gib;
        b.memory.ceiling_bytes = gib + 63 * MIB;
        assert!(!is_meaningful_change(&a, &b));
        b.memory.ceiling_bytes = gib + 65 * MIB;
        assert!(is_meaningful_change(&a, &b));
    }

    fn key_paths(value: &serde_json::Value, prefix: &str, out: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    out.insert(path.clone());
                    key_paths(child, &path, out);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    key_paths(item, &format!("{prefix}[]"), out);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn snapshot_json_keys_match_app_fixture() {
        let snapshot = RuntimeMetricsSnapshot {
            schema_version: 1,
            seq: 1842,
            unix_ms: 1_763_280_000_000,
            backend: backend(),
            health: HealthView {
                status: HealthStatus::Degraded,
                degraded_reasons: vec![HealthDegradedReason::SchedulerQueueHigh],
            },
            activity: ActivityView {
                state: ActivityState::Generating,
                active_requests: 2,
                prefilling_requests: 0,
                queued_requests: 1,
            },
            performance: PerformanceView {
                live_decode_tokens_per_second: Some(42.8),
                prefill_tokens_per_second: Some(1840.0),
                decode_tokens_per_second: Some(44.1),
                ttft_ms: Some(312.0),
                window_seconds: 60,
            },
            memory: MemoryView {
                pressure_level: PressureLevel::Normal,
                process_bytes: 1,
                ceiling_bytes: 2,
                kv_cache_bytes: 3,
                mlx_cache_bytes: 4,
                telemetry_degraded: false,
            },
            prefix_cache: PrefixCacheUsageSnapshot {
                hit_tokens: 1,
                eligible_tokens: 2,
            },
            models: vec![
                ModelView {
                    id: "m".into(),
                    kind: "causal".into(),
                    state: ModelState::Loaded,
                    mode: Some(ModelMode::Dflash2),
                    is_default: true,
                    active_requests: 2,
                    queued_requests: 1,
                    queue_capacity: Some(2),
                    live_decode_tokens_per_second: Some(42.8),
                    last_error: None,
                },
                ModelView {
                    id: "asr".into(),
                    kind: "asr".into(),
                    state: ModelState::Failed,
                    mode: None,
                    is_default: false,
                    active_requests: 0,
                    queued_requests: 0,
                    queue_capacity: None,
                    live_decode_tokens_per_second: None,
                    last_error: Some("boom".into()),
                },
            ],
        };
        let mut actual = BTreeSet::new();
        key_paths(&serde_json::to_value(&snapshot).unwrap(), "", &mut actual);

        let path = format!(
            "{}/../ironmlx-app/Tests/IronMLXAppCoreTests/Fixtures/runtime-metrics-v1.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut expected = BTreeSet::new();
        key_paths(&fixture, "", &mut expected);
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod hub_tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;
    use crate::core::process_memory::{
        MemoryGovernorConfig, MemoryTelemetry, ProcessMemoryGovernor,
    };
    use crate::core::runtime_usage::ModelRuntimePerformanceSnapshot;

    struct FakeSource {
        reading: Mutex<Option<ModelMetricsReading>>,
        reads: AtomicUsize,
        panic_next_read: AtomicBool,
    }

    impl FakeSource {
        fn new(reading: ModelMetricsReading) -> Arc<Self> {
            Arc::new(Self {
                reading: Mutex::new(Some(reading)),
                reads: AtomicUsize::new(0),
                panic_next_read: AtomicBool::new(false),
            })
        }

        fn set(&self, reading: ModelMetricsReading) {
            *self.reading.lock().unwrap() = Some(reading);
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }

        /// The next `read` panics, killing whichever caller made it.
        fn panic_on_next_read(&self) {
            self.panic_next_read.store(true, Ordering::SeqCst);
        }
    }

    impl ModelMetricsSource for FakeSource {
        fn read(&self) -> Option<ModelMetricsReading> {
            if self.panic_next_read.swap(false, Ordering::SeqCst) {
                panic!("fake source read failure (expected in this test)");
            }
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.reading.lock().unwrap().clone()
        }
    }

    fn idle_reading() -> ModelMetricsReading {
        ModelMetricsReading {
            usage: Some(ModelRuntimeUsageSnapshot::default()),
            ..Default::default()
        }
    }

    fn active_reading() -> ModelMetricsReading {
        ModelMetricsReading {
            active_requests: 1,
            ..idle_reading()
        }
    }

    fn prefix_reading(hit: u64, eligible: u64) -> ModelMetricsReading {
        let mut reading = idle_reading();
        reading.usage.as_mut().unwrap().prefix_cache = Some(PrefixCacheUsageSnapshot {
            hit_tokens: hit,
            eligible_tokens: eligible,
        });
        reading
    }

    fn decode_reading(decode: Option<f64>) -> ModelMetricsReading {
        let mut reading = idle_reading();
        reading.usage.as_mut().unwrap().performance = ModelRuntimePerformanceSnapshot {
            completed_requests: usize::from(decode.is_some()),
            decode_tokens_per_second: decode,
            window_seconds: 60,
            ..Default::default()
        };
        reading
    }

    /// A hub whose governor is never sampled (its snapshot stays the same on
    /// every tick) and whose prefix store is never backpressured.
    fn test_hub(memory: MemorySampling) -> Arc<RuntimeMetricsHub> {
        let governor =
            Arc::new(ProcessMemoryGovernor::new(MemoryGovernorConfig::default()).unwrap());
        RuntimeMetricsHub::with_environment(memory, governor, || false, None)
    }

    fn loaded_hub(reading: ModelMetricsReading) -> (Arc<RuntimeMetricsHub>, Arc<FakeSource>) {
        let hub = test_hub(MemorySampling::Cached);
        let source = FakeSource::new(reading);
        hub.set_loaded("m", "causal", Some(ModelMode::Normal), source.clone());
        (hub, source)
    }

    /// Let the sampling task run everything that is ready at the current
    /// (paused) instant.
    async fn settle() {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }

    async fn advance(duration: Duration) {
        tokio::time::advance(duration).await;
        settle().await;
    }

    #[tokio::test(start_paused = true)]
    async fn registration_lifecycle_loading_loaded_failed_removed() {
        let hub = test_hub(MemorySampling::Cached);
        hub.set_loading("m", "causal", Some(ModelMode::Normal));
        let snap = hub.snapshot_now();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.models[0].state, ModelState::Loading);
        assert_eq!(snap.models[0].mode, Some(ModelMode::Normal));
        assert_eq!(snap.activity.state, ActivityState::Loading);

        hub.set_loaded(
            "m",
            "causal",
            Some(ModelMode::Normal),
            FakeSource::new(idle_reading()),
        );
        let snap = hub.snapshot_now();
        assert_eq!(snap.models[0].state, ModelState::Loaded);
        assert_eq!(snap.activity.state, ActivityState::Idle);

        hub.set_failed("m", "causal", Some(ModelMode::Normal), "boom".to_string());
        let snap = hub.snapshot_now();
        assert_eq!(snap.models[0].state, ModelState::Failed);
        assert_eq!(snap.models[0].last_error.as_deref(), Some("boom"));

        hub.remove("m");
        assert!(hub.snapshot_now().models.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn default_model_is_flagged() {
        let (hub, _source) = loaded_hub(idle_reading());
        hub.set_default_model(Some("m"));
        assert!(hub.snapshot_now().models[0].is_default);
        hub.set_default_model(None);
        assert!(!hub.snapshot_now().models[0].is_default);
    }

    #[tokio::test(start_paused = true)]
    async fn prefix_totals_survive_reload_and_unload() {
        let (hub, _old) = loaded_hub(prefix_reading(7, 9));
        let expected = |hit, eligible| PrefixCacheUsageSnapshot {
            hit_tokens: hit,
            eligible_tokens: eligible,
        };
        assert_eq!(hub.snapshot_now().prefix_cache, expected(7, 9));

        hub.set_loading("m", "causal", Some(ModelMode::Normal));
        assert_eq!(hub.snapshot_now().prefix_cache, expected(7, 9));

        hub.set_loaded(
            "m",
            "causal",
            Some(ModelMode::Normal),
            FakeSource::new(prefix_reading(1, 2)),
        );
        assert_eq!(hub.snapshot_now().prefix_cache, expected(8, 11));

        hub.remove("m");
        assert_eq!(hub.snapshot_now().prefix_cache, expected(8, 11));
    }

    #[tokio::test(start_paused = true)]
    async fn subscriber_sees_a_computed_snapshot_immediately() {
        let (hub, _source) = loaded_hub(idle_reading());
        let rx = hub.subscribe();
        let first = rx.borrow().clone();
        assert_eq!(first.seq, 1);
        assert_eq!(first.models.len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn request_start_wakes_task_immediately() {
        let (hub, source) = loaded_hub(idle_reading());
        let mut rx = hub.subscribe();
        settle().await;
        rx.borrow_and_update();

        source.set(active_reading());
        hub.wake();
        tokio::time::timeout(Duration::from_millis(1), rx.changed())
            .await
            .expect("wake publishes without waiting for a tick")
            .unwrap();
        assert_eq!(rx.borrow().activity.active_requests, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn idle_ticks_every_five_seconds_and_active_every_second() {
        let (hub, source) = loaded_hub(idle_reading());
        let _rx = hub.subscribe();
        settle().await;

        let before = source.reads();
        advance(Duration::from_millis(4900)).await;
        assert_eq!(source.reads(), before);
        advance(Duration::from_millis(100)).await;
        assert_eq!(source.reads(), before + 1);

        source.set(active_reading());
        hub.wake();
        settle().await;
        for _ in 0..3 {
            let before = source.reads();
            advance(Duration::from_millis(999)).await;
            assert_eq!(source.reads(), before);
            advance(Duration::from_millis(1)).await;
            assert_eq!(source.reads(), before + 1);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn idle_without_change_publishes_nothing() {
        let (hub, source) = loaded_hub(idle_reading());
        let mut rx = hub.subscribe();
        settle().await;
        rx.borrow_and_update();
        let before = source.reads();
        for _ in 0..60 {
            advance(Duration::from_secs(1)).await;
            assert!(!rx.has_changed().unwrap());
        }
        assert_eq!(source.reads(), before + 12);
    }

    #[tokio::test(start_paused = true)]
    async fn window_expiry_publishes_once() {
        let (hub, source) = loaded_hub(decode_reading(Some(40.0)));
        let mut rx = hub.subscribe();
        settle().await;
        assert_eq!(
            rx.borrow_and_update().performance.decode_tokens_per_second,
            Some(40.0)
        );

        source.set(decode_reading(None));
        advance(Duration::from_secs(5)).await;
        assert!(rx.has_changed().unwrap());
        let seq = {
            let snap = rx.borrow_and_update();
            assert_eq!(snap.performance.decode_tokens_per_second, None);
            snap.seq
        };
        for _ in 0..12 {
            advance(Duration::from_secs(5)).await;
            assert!(!rx.has_changed().unwrap());
        }
        assert_eq!(rx.borrow().seq, seq);
    }

    #[tokio::test(start_paused = true)]
    async fn task_stops_reading_when_last_subscriber_drops() {
        let (hub, source) = loaded_hub(active_reading());
        let rx = hub.subscribe();
        settle().await;
        advance(Duration::from_secs(1)).await;
        drop(rx);
        settle().await;

        let before = source.reads();
        advance(Duration::from_secs(60)).await;
        assert_eq!(source.reads(), before);

        let _rx = hub.subscribe();
        settle().await;
        assert!(source.reads() > before);
        let resumed = source.reads();
        advance(Duration::from_secs(1)).await;
        assert_eq!(source.reads(), resumed + 1);
    }

    #[tokio::test(start_paused = true)]
    async fn next_subscribe_restarts_a_sampling_task_that_died() {
        let (hub, source) = loaded_hub(idle_reading());
        let mut rx = hub.subscribe();
        settle().await;
        rx.borrow_and_update();
        let stopped = hub.sampling_stopped();

        source.panic_on_next_read();
        hub.wake();
        settle().await;
        tokio::time::timeout(Duration::from_millis(1), stopped)
            .await
            .expect("open streams learn the task is gone");
        let before = source.reads();
        advance(Duration::from_secs(60)).await;
        assert_eq!(source.reads(), before, "the task is gone");

        let mut rx = hub.subscribe();
        rx.borrow_and_update();
        source.set(active_reading());
        hub.wake();
        tokio::time::timeout(Duration::from_millis(1), rx.changed())
            .await
            .expect("the restarted task publishes on wake")
            .unwrap();
        assert_eq!(rx.borrow().activity.active_requests, 1);
        let resumed = source.reads();
        advance(Duration::from_secs(1)).await;
        assert_eq!(source.reads(), resumed + 1, "busy ticks resume");
    }

    #[tokio::test(start_paused = true)]
    async fn task_exits_when_hub_drops() {
        let (hub, _source) = loaded_hub(idle_reading());
        let mut rx = hub.subscribe();
        settle().await;
        rx.borrow_and_update();
        drop(hub);
        settle().await;
        assert!(rx.changed().await.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn only_sample_on_tick_samples_the_governor() {
        for (memory, expect_sampled) in [
            (MemorySampling::Cached, false),
            (MemorySampling::SampleOnTick, true),
        ] {
            let governor =
                Arc::new(ProcessMemoryGovernor::new(MemoryGovernorConfig::default()).unwrap());
            let samples = Arc::new(AtomicUsize::new(0));
            let counter = samples.clone();
            governor.set_test_telemetry(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                MemoryTelemetry {
                    total_ram_bytes: 64 << 30,
                    phys_footprint_bytes: Some(1 << 30),
                    vm: None,
                    mlx_active_bytes: Some(0),
                    mlx_cache_bytes: Some(0),
                    metal_limit_bytes: Some(48 << 30),
                }
            });
            let hub = RuntimeMetricsHub::with_environment(memory, governor, || false, None);
            let snap = hub.snapshot_now();
            assert_eq!(samples.load(Ordering::SeqCst) > 0, expect_sampled);
            assert_eq!(snap.memory.process_bytes > 0, expect_sampled);
        }
    }
}
