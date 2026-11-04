//! Adaptive persistence for completed Flash prefixes. In-flight generation
//! state is never offloaded. All retained snapshots are owned CPU bytes.
use super::{
    prefix_store::{
        PagedPrefixLoadResult, PagedPrefixLoadStatus, PagedPrefixStore, PrefixLoadSource,
    },
    prefix_stream::{FrozenPrefix, WriteControl},
};
use crate::core::process_memory::{
    global_process_memory_governor, MemoryReservation, PressureLevel,
};
use anyhow::{Context, Result};
use ironmlx_lm::core::cache::prefix_payload::{PagedPrefixEntry, PagedPrefixKeySpec};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};
const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixSaveMode {
    Auto,
    #[default]
    Sync,
}
impl FromStr for PrefixSaveMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "sync" => Ok(Self::Sync),
            _ => Err("prefix save mode must be auto or sync".into()),
        }
    }
}
impl std::fmt::Display for PrefixSaveMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Sync => "sync",
        })
    }
}
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PrefixSaveProfile {
    pub tier: &'static str,
    pub budget_bytes: usize,
    pub max_jobs: usize,
    pub chunk_bytes: usize,
}
impl PrefixSaveProfile {
    fn capture_peak(self, payload: usize, tokens: usize) -> Option<usize> {
        payload
            .checked_mul(2)?
            .checked_add(self.chunk_bytes)?
            .checked_add(tokens.checked_mul(12)?)?
            .checked_add(MIB)
    }
    fn fits(self, peak: usize, headroom: usize, held: usize, legacy: usize) -> bool {
        peak <= self
            .budget_bytes
            .saturating_sub(held)
            .saturating_sub(legacy.saturating_mul(2))
            && peak <= headroom / 4
    }
    pub fn for_ram(ram: usize) -> Self {
        if ram <= 32 * GIB {
            Self {
                tier: "conservative",
                budget_bytes: 512 * MIB,
                max_jobs: 1,
                chunk_bytes: 8 * MIB,
            }
        } else if ram <= 64 * GIB {
            Self {
                tier: "balanced",
                budget_bytes: 2 * GIB,
                max_jobs: 1,
                chunk_bytes: 16 * MIB,
            }
        } else {
            Self {
                tier: "ample",
                budget_bytes: (ram / 32).min(8 * GIB),
                max_jobs: 2,
                chunk_bytes: 32 * MIB,
            }
        }
    }
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrefixPersistenceStats {
    pub scope: &'static str,
    pub pending_jobs: usize,
    pub reserved_peak_bytes: usize,
    pub frozen_bytes: usize,
    pub queued_total: u64,
    pub committed_total: u64,
    pub skipped_total: u64,
    pub cancelled_total: u64,
    pub failed_total: u64,
    pub hot_memory_hits: u64,
    pub pending_snapshot_hits: u64,
    pub ssd_hits: u64,
    pub capture_us: u64,
    pub write_us: u64,
    pub committed_bytes: u64,
    pub sync_saves: u64,
    pub sync_us: u64,
    pub last_status: String,
    pub last_reason: String,
    pub last_model: String,
    pub last_mode: String,
    pub effective_write_bytes_per_second: Option<f64>,
    pub effective_capture_bytes_per_second: Option<f64>,
    pub throttled: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct PrefixPersistenceSnapshot {
    pub profile: PrefixSaveProfile,
    #[serde(flatten)]
    pub stats: PrefixPersistenceStats,
}
struct Pending {
    queued_at: Instant,
    frozen: Arc<FrozenPrefix>,
    cancel: Arc<AtomicBool>,
    lease: Arc<MemoryReservation>,
}
struct Recent {
    root: PathBuf,
    spec: Arc<PagedPrefixKeySpec>,
}
#[derive(Default)]
struct Feedback {
    baseline_ms: f64,
    samples: u32,
    slow_windows: u32,
}
struct State {
    stats: PrefixPersistenceStats,
    pending: HashMap<(PathBuf, String), Pending>,
    recent: VecDeque<Recent>,
    last_pressure: Option<Instant>,
    feedback: HashMap<String, Feedback>,
    prefill: HashMap<String, (f64, u32)>,
    last_volume: Option<u64>,
    rate_updated: Option<Instant>,
    jobs: VecDeque<Job>,
    captures: HashMap<(PathBuf, String), (String, Arc<AtomicBool>)>,
}
struct Job {
    volume: u64,
    store: PagedPrefixStore,
    frozen: Arc<FrozenPrefix>,
    cancel: Arc<AtomicBool>,
    peak: usize,
    _lease: Arc<MemoryReservation>,
}
struct Controller {
    profile: PrefixSaveProfile,
    state: Arc<Mutex<State>>,
    wake: Arc<Condvar>,
}
static CONTROLLER: OnceLock<Controller> = OnceLock::new();
fn controller() -> &'static Controller {
    CONTROLLER.get_or_init(|| {
        let profile =
            PrefixSaveProfile::for_ram(crate::core::memory_budget::system_total_ram_bytes());
        let state = Arc::new(Mutex::new(State {
            stats: PrefixPersistenceStats {
                scope: "process",
                ..Default::default()
            },
            pending: HashMap::new(),
            recent: VecDeque::new(),
            last_pressure: None,
            feedback: HashMap::new(),
            prefill: HashMap::new(),
            last_volume: None,
            rate_updated: None,
            jobs: VecDeque::new(),
            captures: HashMap::new(),
        }));
        let wake = Arc::new(Condvar::new());
        let worker = Arc::clone(&state);
        let notify = Arc::clone(&wake);
        std::thread::Builder::new()
            .name("ironmlx-prefix-bytes".into())
            .spawn(move || writer_loop(profile, worker, notify))
            .expect("spawn byte-only prefix writer");
        Controller {
            profile,
            state,
            wake,
        }
    })
}
fn writer_loop(profile: PrefixSaveProfile, worker: Arc<Mutex<State>>, notify: Arc<Condvar>) {
    #[cfg(target_os = "macos")]
    if unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) } != 0
    {
        tracing::warn!("prefix writer could not set utility QoS; bounded cooperative throttling remains active");
    }
    loop {
        let job = {
            let mut state = worker.lock().expect("prefix state");
            loop {
                if let Some(job) = state.jobs.pop_front() {
                    break job;
                }
                state = notify.wait(state).expect("prefix state");
            }
        };
        let started = Instant::now();
        let bytes = job.frozen.spec.payload_bytes();
        let checkpoint = |n: usize| -> Result<()> {
            anyhow::ensure!(!job.cancel.load(Ordering::Acquire), "prefix_save_cancelled");
            let memory = global_process_memory_governor().sample_process();
            if memory.telemetry_degraded || memory.pressure_level != PressureLevel::Normal {
                worker.lock().expect("prefix state").last_pressure = Some(Instant::now());
            }
            if memory.telemetry_degraded
                || matches!(
                    memory.pressure_level,
                    PressureLevel::Hard | PressureLevel::Emergency
                )
            {
                job.cancel.store(true, Ordering::Release);
                anyhow::bail!("prefix_save_cancelled:memory_pressure");
            }
            let throttle = worker.lock().expect("prefix state").stats.throttled;
            if throttle && n > 0 {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        };
        {
            let mut state = worker.lock().expect("prefix state");
            state.stats.last_status = "writing".into();
            state.stats.last_mode = "auto".into();
            state.stats.last_model = job.frozen.spec.model_id.clone();
            state.stats.last_reason.clear();
        }
        let control = WriteControl {
            chunk_bytes: profile.chunk_bytes,
            checkpoint: &checkpoint,
        };
        let result = job.store.save_frozen(&job.frozen, &control);
        let key = (
            job.store.root().to_owned(),
            PagedPrefixStore::key_for(&job.frozen.spec),
        );
        let elapsed = started.elapsed().as_micros() as u64;
        let mut state = worker.lock().expect("prefix state");
        state.pending.remove(&key);
        state.stats.write_us = state.stats.write_us.saturating_add(elapsed);
        state.stats.last_model = job.frozen.spec.model_id.clone();
        state.stats.last_mode = "auto".into();
        match result {
            Ok(_) => {
                state.stats.committed_total += 1;
                state.stats.committed_bytes =
                    state.stats.committed_bytes.saturating_add(bytes as u64);
                state.stats.last_status = "committed".into();
                state.stats.last_reason.clear();
                if state.last_volume == Some(job.volume) {
                    let rate = bytes as f64 / (elapsed.max(1) as f64 / 1e6);
                    state.rate_updated = Some(Instant::now());
                    state.stats.effective_write_bytes_per_second = Some(
                        state
                            .stats
                            .effective_write_bytes_per_second
                            .map_or(rate, |old| 0.75 * old + 0.25 * rate),
                    );
                }
                remember(
                    &mut state,
                    job.store.root(),
                    Arc::new(job.frozen.spec.clone()),
                );
                tracing::info!(model=%job.frozen.spec.model_id, tokens=job.frozen.spec.cached_len, write_us=elapsed, bytes, "Flash prefix committed asynchronously");
            }
            Err(error) => {
                if job.cancel.load(Ordering::Acquire) {
                    state.stats.cancelled_total += 1;
                    state.stats.last_status = "cancelled".into();
                } else {
                    state.stats.failed_total += 1;
                    state.stats.last_status = "failed".into();
                }
                state.stats.last_reason = error.to_string();
                tracing::warn!(%error, model=%job.frozen.spec.model_id, "Flash background prefix save did not commit");
            }
        }
        let peak = job.peak;
        // Release bytes and the governor lease before publishing idle status.
        drop(job);
        state.stats.pending_jobs = state.stats.pending_jobs.saturating_sub(1);
        state.stats.reserved_peak_bytes = state.stats.reserved_peak_bytes.saturating_sub(peak);
        state.stats.frozen_bytes = state.stats.frozen_bytes.saturating_sub(bytes);
    }
}

fn remember(state: &mut State, root: &Path, spec: Arc<PagedPrefixKeySpec>) {
    state
        .recent
        .retain(|r| r.root != root || r.spec.model_id != spec.model_id);
    state.recent.push_back(Recent {
        root: root.to_owned(),
        spec,
    });
    while state.recent.len() > 8 {
        state.recent.pop_front();
    }
}
pub(super) fn admit_legacy(admit: impl FnOnce() -> bool) -> bool {
    let Some(c) = CONTROLLER.get() else {
        return admit();
    };
    let state = c.state.lock().expect("prefix state");
    if state.stats.pending_jobs > 0 {
        return false;
    }
    admit()
}
pub fn snapshot() -> PrefixPersistenceSnapshot {
    let c = controller();
    PrefixPersistenceSnapshot {
        profile: c.profile,
        stats: c.state.lock().expect("prefix state").stats.clone(),
    }
}
pub fn busy() -> bool {
    CONTROLLER
        .get()
        .is_some_and(|c| c.state.lock().expect("prefix state").stats.pending_jobs > 0)
}
/// Cancel model snapshots and immediately release queued (not active) jobs.
/// The active writer checks cancellation between bounded writes and lock waits.
pub fn cancel_model(model: &str) -> usize {
    cancel_model_for(model, "model_unloaded", true)
}
fn cancel_model_for(model: &str, reason: &str, unloading: bool) -> usize {
    if let Some(c) = CONTROLLER.get() {
        let mut s = c.state.lock().expect("prefix state");
        let mut count = 0;
        for pending in s.pending.values() {
            if pending.frozen.spec.model_id == model {
                count += usize::from(!pending.cancel.swap(true, Ordering::AcqRel));
            }
        }
        for (name, cancel) in s.captures.values() {
            if name == model {
                count += usize::from(!cancel.swap(true, Ordering::AcqRel));
            }
        }
        let mut cancelled = Vec::new();
        let mut keep = VecDeque::new();
        while let Some(job) = s.jobs.pop_front() {
            if job.frozen.spec.model_id == model {
                s.pending.remove(&(
                    job.store.root().to_owned(),
                    PagedPrefixStore::key_for(&job.frozen.spec),
                ));
                s.stats.pending_jobs = s.stats.pending_jobs.saturating_sub(1);
                s.stats.reserved_peak_bytes = s.stats.reserved_peak_bytes.saturating_sub(job.peak);
                s.stats.frozen_bytes = s
                    .stats
                    .frozen_bytes
                    .saturating_sub(job.frozen.spec.payload_bytes());
                s.stats.cancelled_total += 1;
                cancelled.push(job);
            } else {
                keep.push_back(job);
            }
        }
        s.jobs = keep;
        if !cancelled.is_empty() {
            s.stats.last_status = "cancelled".into();
            s.stats.last_reason = reason.into();
            s.stats.last_model = model.into();
        }
        if unloading {
            s.recent.retain(|r| r.spec.model_id != model);
            s.prefill.remove(model);
            s.feedback
                .retain(|k, _| !k.starts_with(&format!("{model}|")));
        } else {
            s.last_pressure = Some(Instant::now());
        }
        drop(cancelled);
        drop(s);
        count
    } else {
        0
    }
}

/// Foreground allocation takes precedence over optional persistence.
pub fn preempt_for_foreground() -> bool {
    let Some(c) = CONTROLLER.get() else {
        return false;
    };
    let models = {
        let s = c.state.lock().expect("prefix state");
        s.pending
            .values()
            .map(|p| p.frozen.spec.model_id.clone())
            .chain(s.captures.values().map(|(m, _)| m.clone()))
            .collect::<Vec<_>>()
    };
    if models.is_empty() {
        return false;
    }
    for model in models {
        cancel_model_for(&model, "foreground_priority", false);
    }
    wait_idle(Duration::from_millis(100));
    true
}
/// Shutdown callers may wait a bounded interval; completion of a request does
/// not imply that its queued persistence transaction has committed.
pub fn wait_idle(timeout: Duration) -> bool {
    let start = Instant::now();
    while busy() {
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

pub(super) fn invalidated(root: &Path, spec: &PagedPrefixKeySpec) {
    if let Some(c) = CONTROLLER.get() {
        let mut s = c.state.lock().expect("prefix state");
        s.recent
            .retain(|r| r.root != root || r.spec.model_id != spec.model_id);
    }
}
pub(super) fn disk_hit(root: &Path, spec: &PagedPrefixKeySpec) {
    if let Some(c) = CONTROLLER.get() {
        let mut s = c.state.lock().expect("prefix state");
        s.stats.ssd_hits += 1;
        remember(&mut s, root, Arc::new(spec.clone()));
    }
}
pub(crate) fn hot_hit() {
    if let Some(c) = CONTROLLER.get() {
        c.state.lock().expect("prefix state").stats.hot_memory_hits += 1;
    }
}
pub(super) fn pending(
    store: &PagedPrefixStore,
    spec: &PagedPrefixKeySpec,
) -> Result<Option<PagedPrefixLoadResult>> {
    let Some(c) = CONTROLLER.get() else {
        return Ok(None);
    };
    let key = PagedPrefixStore::key_for(spec);
    let frozen = {
        let s = c.state.lock().expect("prefix state");
        s.pending
            .get(&(store.root().to_owned(), key.clone()))
            .filter(|p| !p.cancel.load(Ordering::Acquire) && p.frozen.spec == *spec)
            .map(|p| (Arc::clone(&p.frozen), Arc::clone(&p.lease)))
    };
    let Some((frozen, _lease)) = frozen else {
        return Ok(None);
    };
    let _restore = global_process_memory_governor()
        .try_reserve(spec.payload_bytes(), "prefix_pending_restore")
        .map_err(|e| anyhow::anyhow!("pending restore budget: {e:?}"))?;
    let entry = frozen.thaw()?;
    c.state
        .lock()
        .expect("prefix state")
        .stats
        .pending_snapshot_hits += 1;
    Ok(Some(PagedPrefixLoadResult {
        key,
        status: PagedPrefixLoadStatus::Hit,
        source: Some(PrefixLoadSource::PendingSnapshot),
        stats: Some(entry.observability_stats(spec.cached_len)),
        entry: Some(entry),
    }))
}
pub(super) fn pending_lengths(root: &Path, max: i32) -> Vec<i32> {
    let Some(c) = CONTROLLER.get() else {
        return vec![];
    };
    let s = c.state.lock().expect("prefix state");
    s.pending
        .iter()
        .filter(|((r, _), p)| {
            r == root && !p.cancel.load(Ordering::Acquire) && p.frozen.spec.cached_len <= max
        })
        .map(|(_, p)| p.frozen.spec.cached_len)
        .collect()
}
pub(super) fn contains_pending(root: &Path, spec: &PagedPrefixKeySpec) -> bool {
    let Some(c) = CONTROLLER.get() else {
        return false;
    };
    let s = c.state.lock().expect("prefix state");
    s.pending
        .get(&(root.to_owned(), PagedPrefixStore::key_for(spec)))
        .is_some_and(|p| p.frozen.spec == *spec && !p.cancel.load(Ordering::Acquire))
}
fn skip(state: &mut State, reason: &str, model: &str) {
    state.stats.skipped_total += 1;
    state.stats.last_status = "skipped".into();
    state.stats.last_reason = reason.into();
    state.stats.last_model = model.into();
}
/// Conservative workload-matched decode feedback; insufficient samples never
/// justify raising IO concurrency. TTFT is retained separately from write time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn observe_request(
    model: &str,
    route: &str,
    tokens: u64,
    concurrency: usize,
    ttft_ms: f64,
    decode_ms: Option<f64>,
    overlapped: bool,
    cache_hit: bool,
) {
    let Some(c) = CONTROLLER.get() else { return };
    let mut s = c.state.lock().expect("prefix state");
    if concurrency == 1 && !cache_hit && !overlapped && ttft_ms.is_finite() {
        let p = s.prefill.entry(model.into()).or_default();
        p.0 = if p.1 == 0 {
            ttft_ms / tokens.max(1) as f64
        } else {
            0.8 * p.0 + 0.2 * ttft_ms / tokens.max(1) as f64
        };
        p.1 += 1;
    }
    let Some(ms) = decode_ms.filter(|ms| ms.is_finite() && *ms > 0.) else {
        return;
    };
    let key = format!(
        "{model}|{route}|{}|{concurrency}",
        tokens.max(1).next_power_of_two()
    );
    if s.feedback.len() > 128 {
        s.feedback.clear();
    }
    let f = s.feedback.entry(key).or_default();
    if !overlapped {
        f.baseline_ms = if f.samples == 0 {
            ms
        } else {
            0.8 * f.baseline_ms + 0.2 * ms
        };
        f.samples += 1;
        f.slow_windows = 0;
    } else if f.samples >= 3 {
        if ms > f.baseline_ms * 1.1 {
            f.slow_windows += 1;
        } else {
            f.slow_windows = 0;
        }
    }
    let slow = f.slow_windows >= 2;
    if slow {
        s.stats.throttled = true;
        s.last_pressure = Some(Instant::now());
    }
}
fn available_disk(root: &Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let absolute = if root.is_absolute() {
            root.to_owned()
        } else {
            std::env::current_dir().ok()?.join(root)
        };
        let mut p = absolute.as_path();
        while !p.exists() {
            p = p.parent()?;
        }
        let path = std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).ok()?;
        let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
            return None;
        }
        let stat = unsafe { stat.assume_init() };
        Some((
            u64::from(stat.f_bavail).saturating_mul(stat.f_frsize),
            std::fs::metadata(p).ok()?.dev(),
        ))
    }
    #[cfg(not(unix))]
    {
        let _ = root;
        None
    }
}
/// Persist a completed Flash prefix. `Some(key)` means committed in Sync,
/// or an owned snapshot accepted for background saving in Auto; it is not a
/// durability acknowledgement in Auto. `None` means optional saving was skipped.
/// Auto skips if it cannot safely own an immutable snapshot. Sync preserves
/// the blocking fallback. Neither mode offloads active KV/request state.
pub fn save(
    store: &PagedPrefixStore,
    spec: &PagedPrefixKeySpec,
    entry: &PagedPrefixEntry,
    mode: PrefixSaveMode,
    preferred: bool,
) -> Result<Option<String>> {
    let c = controller();
    let key = PagedPrefixStore::key_for(spec);
    let size = spec.payload_bytes();
    store.validate_capture(spec, entry)?;
    anyhow::ensure!(
        spec.main_layers
            .iter()
            .any(|l| l.kind == ironmlx_lm::core::cache::prefix_payload::PrefixLayerKind::Flash),
        "adaptive persistence requires a complete Flash prefix"
    );
    if mode == PrefixSaveMode::Sync {
        // Sync also budgets its reusable buffer and largest possible compact copy.
        global_process_memory_governor().refresh_process();
        let _lease = match global_process_memory_governor()
            .try_reserve(size.saturating_add(8 * MIB), "prefix_sync_buffer")
        {
            Ok(lease) => lease,
            Err(_) => {
                let mut s = c.state.lock().expect("prefix state");
                s.stats.last_mode = "sync".into();
                skip(&mut s, "memory_budget", &spec.model_id);
                return Ok(None);
            }
        };
        {
            let mut s = c.state.lock().expect("prefix state");
            s.stats.last_mode = "sync".into();
            s.stats.last_model = spec.model_id.clone();
            s.stats.last_status = "writing".into();
            s.stats.last_reason.clear();
        }
        let begin = Instant::now();
        let result = store.save_if_absent(spec, entry);
        let mut s = c.state.lock().expect("prefix state");
        s.stats.last_mode = "sync".into();
        s.stats.sync_us = s
            .stats
            .sync_us
            .saturating_add(begin.elapsed().as_micros() as u64);
        return match result {
            Ok((key, written)) => {
                if written {
                    s.stats.sync_saves += 1;
                    s.stats.committed_total += 1;
                    s.stats.committed_bytes = s.stats.committed_bytes.saturating_add(size as u64);
                }
                s.stats.last_status = "committed".into();
                s.stats.last_reason.clear();
                remember(&mut s, store.root(), Arc::new(spec.clone()));
                Ok(Some(key))
            }
            Err(e) => {
                s.stats.failed_total += 1;
                s.stats.last_status = "failed".into();
                s.stats.last_reason = e.to_string();
                Err(e)
            }
        };
    }
    let governor = global_process_memory_governor();
    let memory = governor.refresh_process();
    let peak = c
        .profile
        .capture_peak(size, spec.token_ids.len())
        .context("prefix capture reservation overflow")?;
    let headroom = memory
        .soft_watermark_bytes
        .saturating_sub(memory.current_usage_bytes)
        .saturating_sub(memory.reserved_bytes);
    let disk = available_disk(store.root());
    let mut s = c.state.lock().expect("prefix state");
    s.stats.last_mode = "auto".into();
    if !preferred {
        skip(&mut s, "non_preferred_boundary", &spec.model_id);
        return Ok(None);
    }
    if s.captures
        .contains_key(&(store.root().to_owned(), key.clone()))
    {
        skip(&mut s, "capture_in_progress", &spec.model_id);
        return Ok(None);
    }
    if s.pending
        .contains_key(&(store.root().to_owned(), key.clone()))
    {
        skip(&mut s, "coalesced", &spec.model_id);
        return Ok(Some(key));
    }
    // Do not repeat a nearly identical ordinary boundary; only reuse evidence
    // backed by an extant committed file or an owned pending snapshot.
    let adjacent_match = |other: &PagedPrefixKeySpec| {
        other.model_id == spec.model_id
            && other.entry_kind == spec.entry_kind
            && other.block_size == spec.block_size
            && other.kv_cache_profile == spec.kv_cache_profile
            && other.fingerprint == spec.fingerprint
            && spec.token_ids.starts_with(&other.token_ids)
            && spec.cached_len >= other.cached_len
            && spec.cached_len - other.cached_len <= 8
    };
    let adjacent = s.recent.iter().any(|r| {
        r.root == store.root()
            && adjacent_match(&r.spec)
            && store.contains_persisted(&r.spec).unwrap_or(false)
    }) || s.pending.iter().any(|((root, _), p)| {
        root == store.root() && !p.cancel.load(Ordering::Acquire) && adjacent_match(&p.frozen.spec)
    });
    if adjacent {
        skip(&mut s, "adjacent_boundary", &spec.model_id);
        return Ok(None);
    }
    if memory.telemetry_degraded || memory.pressure_level != PressureLevel::Normal {
        s.last_pressure = Some(Instant::now());
        skip(&mut s, "memory_pressure_or_telemetry", &spec.model_id);
        return Ok(None);
    }
    if s.last_pressure
        .is_some_and(|t| t.elapsed() < Duration::from_secs(5))
    {
        skip(&mut s, "recovery_hysteresis", &spec.model_id);
        return Ok(None);
    }
    if s.stats.throttled
        && s.last_pressure
            .is_some_and(|t| t.elapsed() >= Duration::from_secs(5))
    {
        s.stats.throttled = false;
        s.last_pressure = None;
    }
    let legacy = super::prefix_store::process_async_prefix_store_queue().stats();
    if !c.profile.fits(
        peak,
        headroom,
        s.stats.reserved_peak_bytes,
        legacy.pending_bytes,
    ) {
        skip(&mut s, "memory_budget", &spec.model_id);
        return Ok(None);
    }
    let Some((free, volume)) = disk else {
        skip(&mut s, "disk_telemetry", &spec.model_id);
        return Ok(None);
    };
    if free < (size as u64).saturating_add(64 * MIB as u64)
        || store
            .max_bytes()
            .is_some_and(|max| size.saturating_add(MIB) > max)
    {
        skip(&mut s, "disk_budget", &spec.model_id);
        return Ok(None);
    }
    if s.last_volume != Some(volume)
        || s.rate_updated
            .is_some_and(|t| t.elapsed() > Duration::from_secs(30))
    {
        s.stats.effective_write_bytes_per_second = None;
        s.rate_updated = None;
        s.last_volume = Some(volume);
    }
    let drain = s
        .stats
        .effective_write_bytes_per_second
        .map(|rate| s.stats.frozen_bytes as f64 / rate.max(1.))
        .unwrap_or(0.);
    let age = s
        .pending
        .values()
        .map(|p| p.queued_at.elapsed().as_secs_f64())
        .fold(0., f64::max);
    let drain = drain.max(age);
    let jobs_limit = if drain > 2. || s.stats.throttled {
        1
    } else {
        c.profile.max_jobs
    };
    if drain > 5. || s.stats.pending_jobs >= jobs_limit || legacy.pending_jobs > 0 {
        skip(&mut s, "writer_backpressure", &spec.model_id);
        return Ok(None);
    }
    let estimated_restore_ms = s
        .stats
        .effective_write_bytes_per_second
        .map(|r| size as f64 / r.max(1.) * 1000.)
        .unwrap_or(250.);
    let estimated_capture_ms = s
        .stats
        .effective_capture_bytes_per_second
        .map(|r| size as f64 / r.max(1.) * 1000.)
        .unwrap_or(50.);
    let useful = s
        .prefill
        .get(&spec.model_id)
        .filter(|(_, n)| *n >= 3)
        .map_or(spec.cached_len >= 256, |(ms, _)| {
            ms * spec.cached_len as f64 >= 2. * (estimated_restore_ms + estimated_capture_ms)
        });
    if !useful {
        skip(&mut s, "low_expected_benefit", &spec.model_id);
        return Ok(None);
    }
    let lease = match governor.try_reserve(peak, "prefix_snapshot") {
        Ok(lease) => Arc::new(lease),
        Err(_) => {
            skip(&mut s, "governor_reservation", &spec.model_id);
            return Ok(None);
        }
    };
    // Reserve before any CPU copy or MLX compact materialization. Register a
    // cancellation token before capture so unloading also interrupts CPU copies.
    let cancel = Arc::new(AtomicBool::new(false));
    s.captures.insert(
        (store.root().to_owned(), key.clone()),
        (spec.model_id.clone(), Arc::clone(&cancel)),
    );
    s.stats.pending_jobs += 1;
    s.stats.reserved_peak_bytes += peak;
    s.stats.last_status = "capturing".into();
    s.stats.last_model = spec.model_id.clone();
    s.stats.last_reason.clear();
    drop(s);
    let begin = Instant::now();
    let checkpoint = |_: usize| -> Result<()> {
        let memory = governor.sample_process();
        if memory.telemetry_degraded || memory.pressure_level != PressureLevel::Normal {
            c.state.lock().expect("prefix state").last_pressure = Some(Instant::now());
        }
        anyhow::ensure!(
            !cancel.load(Ordering::Acquire)
                && !memory.telemetry_degraded
                && !matches!(
                    memory.pressure_level,
                    PressureLevel::Hard | PressureLevel::Emergency
                ),
            "prefix_save_cancelled:capture"
        );
        Ok(())
    };
    let control = WriteControl {
        chunk_bytes: c.profile.chunk_bytes,
        checkpoint: &checkpoint,
    };
    let result = FrozenPrefix::capture(spec.clone(), entry, &control);
    let mut s = c.state.lock().expect("prefix state");
    s.captures.remove(&(store.root().to_owned(), key.clone()));
    let frozen = match result {
        Ok(frozen) if !cancel.load(Ordering::Acquire) => frozen,
        result => {
            s.stats.last_reason = result
                .as_ref()
                .err()
                .map(ToString::to_string)
                .unwrap_or_else(|| "save_cancelled".into());
            s.stats.pending_jobs -= 1;
            s.stats.reserved_peak_bytes -= peak;
            if cancel.load(Ordering::Acquire)
                || result
                    .as_ref()
                    .err()
                    .is_some_and(|e| e.to_string().starts_with("prefix_save_cancelled"))
            {
                s.stats.cancelled_total += 1;
                s.stats.last_status = "cancelled".into();
                return Ok(None);
            }
            s.stats.failed_total += 1;
            s.stats.last_status = "failed".into();
            return Err(result.err().expect("capture error"));
        }
    };
    let capture_rate = size as f64 / begin.elapsed().as_secs_f64().max(1e-6);
    s.stats.effective_capture_bytes_per_second = Some(
        s.stats
            .effective_capture_bytes_per_second
            .map_or(capture_rate, |r| 0.75 * r + 0.25 * capture_rate),
    );
    s.stats.capture_us = s
        .stats
        .capture_us
        .saturating_add(begin.elapsed().as_micros() as u64);
    s.stats.frozen_bytes += size;
    s.stats.last_status = "queued".into();
    s.stats.last_model = spec.model_id.clone();
    s.stats.last_reason.clear();
    s.pending.insert(
        (store.root().to_owned(), key.clone()),
        Pending {
            queued_at: begin,
            frozen: Arc::clone(&frozen),
            cancel: Arc::clone(&cancel),
            lease: Arc::clone(&lease),
        },
    );
    s.jobs.push_back(Job {
        volume,
        store: store.clone(),
        frozen,
        cancel,
        peak,
        _lease: lease,
    });
    s.stats.queued_total += 1;
    c.wake.notify_one();
    tracing::info!(model=%spec.model_id,tokens=spec.cached_len,capture_us=begin.elapsed().as_micros(),size,peak,"Flash prefix queued; not yet durable");
    Ok(Some(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_new_cache_root_uses_its_existing_volume_without_creating_files() {
        let path = PathBuf::from(format!("uncreated-prefix-{}", uuid::Uuid::new_v4()));
        assert!(!path.exists());
        assert!(available_disk(&path).is_some());
        assert!(!path.exists());
    }
    #[test]
    fn initial_budgets_are_process_caps_and_scale_above_64_gib() {
        for (ram, budget, jobs, chunk) in [
            (32, 512 * MIB, 1, 8 * MIB),
            (64, 2 * GIB, 1, 16 * MIB),
            (96, 3 * GIB, 2, 32 * MIB),
            (128, 4 * GIB, 2, 32 * MIB),
            (256, 8 * GIB, 2, 32 * MIB),
            (512, 8 * GIB, 2, 32 * MIB),
        ] {
            let profile = PrefixSaveProfile::for_ram(ram * GIB);
            assert_eq!(
                (profile.budget_bytes, profile.max_jobs, profile.chunk_bytes),
                (budget, jobs, chunk)
            );
        }
        assert_eq!(
            "auto".parse::<PrefixSaveMode>().unwrap(),
            PrefixSaveMode::Auto
        );
        assert_eq!(PrefixSaveMode::default(), PrefixSaveMode::Sync);
        assert!("async".parse::<PrefixSaveMode>().is_err());
    }
    #[test]
    fn snapshot_admission_checks_peak_held_budget_and_quarter_headroom() {
        let low = PrefixSaveProfile::for_ram(32 * GIB);
        let high = PrefixSaveProfile::for_ram(128 * GIB);
        let large = 583_598_570; // measured Stage 7 16K artifact, not a 32G hardware result
        assert!(!low.fits(low.capture_peak(large, 16385).unwrap(), 16 * GIB, 0, 0));
        let peak = high.capture_peak(large, 16385).unwrap();
        assert!(high.fits(peak, 16 * GIB, 0, 0));
        assert!(!high.fits(peak, 4 * GIB, 0, 0));
        assert!(!high.fits(peak, 32 * GIB, 3 * GIB, 0));
        assert!(!high.fits(peak, 32 * GIB, 2 * GIB, 512 * MIB));
        assert!(high.capture_peak(usize::MAX, 1).is_none());
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn pending_reuse_cancellation_and_atomic_background_commit() {
        use super::super::prefix_store::{flash_durable_tests::fixture, PrefixLoadSource};
        use ironmlx_lm::core::cache::prefix_payload::PrefixLayerPayload;
        use std::os::fd::AsRawFd;
        let root = std::env::temp_dir().join(format!("prefix-controller-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let (store, mut spec, mut entry) = fixture(&root);
        let store = store.with_max_bytes(16 * MIB);
        spec.model_id = format!("adaptive-{}", uuid::Uuid::new_v4());
        spec.cached_len = 512;
        spec.token_ids = vec![1; 512];
        let PrefixLayerPayload::Flash { tensors } = &mut entry.main_layers[0] else {
            unreachable!()
        };
        tensors[0] = mlx::Array::try_from((&[1_i32, 0, 0, 512, 0, 0][..], &[6][..])).unwrap();
        let governor = global_process_memory_governor();
        governor.refresh_process();
        let original_reserved = governor.snapshot().reserved_bytes;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(".writer.lock"))
            .unwrap();
        assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) }, 0);
        let key = save(&store, &spec, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .unwrap_or_else(|| panic!("not admitted: {:?}", snapshot()));
        assert!(busy());
        assert!(!root.join(&key).exists());
        assert!(store.cached_lengths_descending(512).unwrap().contains(&512));
        let pending = store.load_observed(&spec).unwrap();
        assert_eq!(pending.source, Some(PrefixLoadSource::PendingSnapshot));
        drop(pending);
        {
            let mut state = controller().state.lock().unwrap();
            for p in state.pending.values_mut() {
                p.queued_at = Instant::now() - Duration::from_secs(6);
            }
        }
        let mut stalled = spec.clone();
        stalled.model_id.push_str("-stalled");
        assert!(save(&store, &stalled, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .is_none());
        assert_eq!(
            snapshot().stats.last_reason,
            "writer_backpressure",
            "unknown initial bandwidth must not hide a stalled writer"
        );
        {
            let mut state = controller().state.lock().unwrap();
            for p in state.pending.values_mut() {
                p.queued_at = Instant::now();
            }
        }
        let mut queued_spec = spec.clone();
        queued_spec.model_id.push_str("-queued");
        save(&store, &queued_spec, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .expect("second queued snapshot admitted");
        assert_eq!(snapshot().stats.pending_jobs, 2);
        cancel_model(&queued_spec.model_id);
        assert_eq!(
            snapshot().stats.pending_jobs,
            1,
            "queued cancellation must release ownership immediately"
        );
        assert!(save(&store, &spec, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .is_some());
        let mut next = spec.clone();
        next.cached_len += 1;
        next.token_ids.push(1);
        assert!(save(&store, &next, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .is_none());
        assert_eq!(snapshot().stats.last_reason, "adjacent_boundary");
        cancel_model(&spec.model_id);
        assert!(wait_idle(Duration::from_secs(2)));
        assert!(!root.join(&key).exists());
        assert_eq!(governor.snapshot().reserved_bytes, original_reserved);
        save(&store, &spec, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .expect("admitted for foreground preemption");
        assert!(governor
            .try_reserve(usize::MAX / 2, "foreground_priority_probe")
            .is_err());
        assert!(wait_idle(Duration::from_secs(2)));
        assert_eq!(governor.snapshot().reserved_bytes, original_reserved);
        assert!(save(&store, &spec, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .is_none());
        assert_eq!(snapshot().stats.last_reason, "recovery_hysteresis");
        controller().state.lock().unwrap().last_pressure =
            Some(Instant::now() - Duration::from_secs(6));
        assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_UN) }, 0);
        drop(lock);
        save(&store, &spec, &entry, PrefixSaveMode::Auto, true)
            .unwrap()
            .expect("readmitted");
        assert!(wait_idle(Duration::from_secs(5)));
        assert_eq!(
            store.load_observed(&spec).unwrap().source,
            Some(PrefixLoadSource::Ssd)
        );
        assert_eq!(governor.snapshot().reserved_bytes, original_reserved);
        assert!(snapshot().stats.pending_snapshot_hits > 0);
        assert!(snapshot().stats.committed_total > 0);
        let mut different_contract = spec.clone();
        different_contract.block_size = 256;
        save(
            &store,
            &different_contract,
            &entry,
            PrefixSaveMode::Auto,
            true,
        )
        .unwrap()
        .expect("different block contract must not coalesce");
        assert!(wait_idle(Duration::from_secs(5)));
        assert!(store.contains_persisted(&different_contract).unwrap());
        std::fs::remove_dir_all(root).unwrap();
        cancel_model(&spec.model_id);
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn decode_feedback_requires_matching_workload_and_repeated_slow_windows() {
        let c = controller();
        let model = format!("feedback-{}", uuid::Uuid::new_v4());
        for _ in 0..3 {
            observe_request(&model, "ordinary", 4096, 1, 1000., Some(10.), false, false);
        }
        for _ in 0..2 {
            observe_request(&model, "ordinary", 4096, 4, 1000., Some(20.), true, false);
        }
        assert!(
            !snapshot().stats.throttled,
            "B4 cannot qualify a B1 regression"
        );
        observe_request(&model, "ordinary", 4096, 1, 1000., Some(12.), true, false);
        assert!(!snapshot().stats.throttled);
        observe_request(&model, "ordinary", 4096, 1, 1000., Some(12.), true, false);
        assert!(snapshot().stats.throttled);
        cancel_model(&model);
        let mut state = c.state.lock().unwrap();
        state.stats.throttled = false;
        state.last_pressure = None;
    }
}
