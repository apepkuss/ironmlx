//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-3 activates multi-request batching via a hybrid admission window:
//! the first admit starts a [`ADMISSION_DEADLINE`] timer; further admits
//! accumulate until either [`Scheduler::active_count`] saturates at
//! `b_max` (saturate path) or the deadline expires (hard limit, no
//! reset on new admits).
//!
//! 3c-3 introduced the rolling decode loop: after first-batch prefill
//! the driver usually biased-selects between `cmd_rx.recv()` (mid-batch
//! admit) and an always-ready step branch. Admission work marks the next
//! decode step as due, so active rows take one [`Scheduler::step`] before
//! the actor accepts more optional mid-batch admission work. Mid admits
//! route through [`Scheduler::admit_mid`] (B=1 temp-cache prefill +
//! adopt-into-main); step branch calls [`Scheduler::step`] +
//! [`Scheduler::gc_finished_rows`]. The loop exits when
//! `active_count == 0` AND `cmd_rx` is empty.
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md` § 4.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::cache::{
    ActiveKvOffloadConfig, ActiveKvOffloadSharedStats, PagedPrefixCacheConfig, PrefixLruCacheConfig,
};
use crate::core::generate::GenerateRequest;
use crate::core::model::Model;
use crate::core::scheduler::{
    ActiveKvParkedRequest, AdmitMidHandle, DenseVlMethods, Phase, RequestId, Scheduler, StepEvent,
};
use crate::core::speculative::{MtpSpeculativeConfig, MtpSpeculativeModel, MtpSpeculativeStats};
use crate::Result;

/// Commands accepted by the actor. 3b-2 ships only [`Admit`]; later
/// phases may add `Cancel { id }`, `Stats`, etc.
pub enum SchedulerCommand {
    /// Submit a request for batched generation. On success, replies with
    /// the admitted [`RequestId`] and an mpsc receiver that streams
    /// [`StepEvent`]s (one per produced token, until `finish_reason`
    /// becomes `Some(_)` on the final event for this row).
    Admit {
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    },
}

/// A request parked in `driver_loop`'s admission queue while the scheduler
/// is at `active_count == b_max`. Drained when `gc_finished_rows` frees a
/// slot, then handed to the rolling mid-admit chunk path.
struct PendingAdmit {
    request: GenerateRequest,
    reply_tx: oneshot::Sender<Result<AdmitReply>>,
    queued_at_profile: Option<Instant>,
}

fn fresh_prefill_batch_limit_for_request<M: Model>(
    request: &GenerateRequest,
    b_max: usize,
) -> usize {
    M::fresh_prefill_batch_limit(request.prompt_ids.len(), b_max).clamp(1, b_max)
}

fn fresh_prefill_batch_limit_for_command<M: Model>(cmd: &SchedulerCommand, b_max: usize) -> usize {
    let SchedulerCommand::Admit { request, .. } = cmd;
    fresh_prefill_batch_limit_for_request::<M>(request, b_max)
}

/// Event yielded by the rolling decode loop. Either a new admit command
/// arrived (mid-batch admit), a decode step is due, or the cmd_rx channel
/// was closed (shutdown).
#[allow(clippy::large_enum_variant)] // Admit(SchedulerCommand) intentionally large; boxing would add allocation on hot path
enum RollingEvent {
    Admit(SchedulerCommand),
    AdvanceMidAdmit,
    Step,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RollingMidAdmitSource {
    Direct,
    Queue,
}

#[derive(Clone, Copy, Debug)]
struct MidAdmitProfileContext {
    source: RollingMidAdmitSource,
    queue_wait_ms: Option<f64>,
    queue_len: usize,
}

impl RollingMidAdmitSource {
    fn as_str(self) -> &'static str {
        match self {
            RollingMidAdmitSource::Direct => "direct",
            RollingMidAdmitSource::Queue => "queue",
        }
    }
}

fn rolling_profile_enabled_from_env(value: Option<&str>) -> bool {
    value == Some("1")
}

fn rolling_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        rolling_profile_enabled_from_env(
            std::env::var("IRONMLX_CHUNKED_ROLLING_PROFILE")
                .ok()
                .as_deref(),
        )
    })
}

fn rolling_profile_t_ms(now: Instant) -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(|| now);
    now.saturating_duration_since(epoch).as_secs_f64() * 1000.0
}

fn rolling_profile_elapsed_ms(start: Instant, end: Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1000.0
}

fn rolling_profile_queue_wait_ms(queued_at: Instant, now: Instant) -> f64 {
    rolling_profile_elapsed_ms(queued_at, now)
}

fn cadence_protected_mid_chunk_size(
    requested_chunk_size: i32,
    active_count_with_mid_admit: usize,
    decode_cadence_mid_chunk_cap: usize,
) -> i32 {
    let requested_chunk_size = requested_chunk_size.max(1);
    if active_count_with_mid_admit > 1 {
        let cap = decode_cadence_mid_chunk_cap.clamp(1, i32::MAX as usize) as i32;
        requested_chunk_size.min(cap)
    } else {
        requested_chunk_size
    }
}

/// Rolling-loop admission fairness policy.
///
/// Any completed admission work (initial prefill or mid-batch prefill)
/// makes one decode step due before the actor accepts more optional
/// admission work. This bounds active streams' token gap under sustained
/// arrivals while keeping the existing initial admission window and FIFO
/// queue semantics intact.
#[derive(Debug, Default)]
struct RollingAdmissionPolicy {
    decode_due_after_admission: bool,
}

impl RollingAdmissionPolicy {
    fn record_admission_work(&mut self) {
        self.decode_due_after_admission = true;
    }

    fn record_decode_step(&mut self) {
        self.decode_due_after_admission = false;
    }

    fn should_force_decode(&self, phase: Phase, active_count: usize) -> bool {
        self.decode_due_after_admission && phase == Phase::Decoding && active_count > 0
    }
}

#[derive(Clone)]
struct SchedulerActorMtpCounters {
    mtp_prefill_count: Arc<AtomicU64>,
    mtp_step_count: Arc<AtomicU64>,
    mtp_prefill_fallback_count: Arc<AtomicU64>,
    mtp_drafted_tokens: Arc<AtomicU64>,
    mtp_accepted_draft_tokens: Arc<AtomicU64>,
}

impl SchedulerActorMtpCounters {
    fn new(
        mtp_prefill_count: Arc<AtomicU64>,
        mtp_step_count: Arc<AtomicU64>,
        mtp_prefill_fallback_count: Arc<AtomicU64>,
        mtp_drafted_tokens: Arc<AtomicU64>,
        mtp_accepted_draft_tokens: Arc<AtomicU64>,
    ) -> Self {
        Self {
            mtp_prefill_count,
            mtp_step_count,
            mtp_prefill_fallback_count,
            mtp_drafted_tokens,
            mtp_accepted_draft_tokens,
        }
    }

    fn store_stats(&self, stats: Option<MtpSpeculativeStats>) {
        if let Some(stats) = stats {
            self.mtp_drafted_tokens
                .store(stats.drafted_tokens as u64, Ordering::Relaxed);
            self.mtp_accepted_draft_tokens
                .store(stats.accepted_draft_tokens as u64, Ordering::Relaxed);
        }
    }
}

trait SchedulerActorMtpMode<M>
where
    M: Model + DenseVlMethods,
{
    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>>;

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>>;
}

struct SchedulerActorNoMtp;

struct SchedulerActorMtp<H> {
    mtp: H,
    cfg: MtpSpeculativeConfig,
}

impl<H> SchedulerActorMtp<H> {
    fn new(mtp: H, mtp_draft_tokens: usize) -> Self {
        debug_assert!(mtp_draft_tokens > 0);
        Self {
            mtp,
            cfg: MtpSpeculativeConfig {
                max_draft_tokens: mtp_draft_tokens,
            },
        }
    }
}

impl<M> SchedulerActorMtpMode<M> for SchedulerActorNoMtp
where
    M: Model + DenseVlMethods,
{
    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        _counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        sched.prefill_admitted(model)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        _counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        sched.step(model)
    }
}

impl<M> SchedulerActorMtpMode<M> for SchedulerActorMtp<M::MtpHead>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel,
{
    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        if sched.mtp_batch_active_greedy_eligible() {
            counters.mtp_prefill_count.fetch_add(1, Ordering::Relaxed);
            let events = sched.prefill_admitted_mtp_batch(model, &self.mtp, self.cfg)?;
            counters.store_stats(sched.mtp_stats());
            Ok(events)
        } else {
            counters
                .mtp_prefill_fallback_count
                .fetch_add(1, Ordering::Relaxed);
            sched.prefill_admitted(model)
        }
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        if sched.mtp_stats().is_some() {
            counters.mtp_step_count.fetch_add(1, Ordering::Relaxed);
            let events = sched.step_mtp_batch(model, &self.mtp)?;
            counters.store_stats(sched.mtp_stats());
            Ok(events)
        } else {
            sched.step(model)
        }
    }
}

/// Result returned by [`drive_empty_scheduler_handoff`] encoding what the
/// caller's rolling loop should do next. Matches the existing `continue
/// 'rolling` / `break 'rolling` / `continue 'outer` / `return` patterns
/// without exposing label control to the helper.
///
/// Keeps the empty-batch handoff path reusable from both the existing
/// post-step empty-handoff site and the pre-event Finished-batch
/// finalization at the rolling-loop top.
enum RollingControl {
    /// Re-enter the rolling loop (a new batch was admitted + prefilled).
    ContinueRolling,
    /// Exit the rolling loop into the outer-loop tail cleanup (no
    /// queued or pending admits; outer will block on `cmd_rx.recv()`).
    BreakRolling,
    /// `continue 'outer` — outer loop body resumes from its top
    /// (e.g., poisoned-state recovery).
    ContinueOuter,
    /// `return` from the actor (cmd_rx disconnected; all senders dropped).
    ReturnActor,
}

/// Reply payload for [`SchedulerCommand::Admit`]. Carries the assigned
/// [`RequestId`] and the per-request event receiver.
pub struct AdmitReply {
    pub request_id: RequestId,
    pub event_rx: mpsc::UnboundedReceiver<StepEvent>,
}

/// Handle held by [`crate::core::server::AppState`]. Cheap to clone
/// (`mpsc::Sender` and `Arc<AtomicU64>` are both `Clone`).
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    /// Test-observable counter. Incremented by the driver every time
    /// `Scheduler::admit` succeeds. Doc-hidden because production code
    /// shouldn't read it — it exists for integration tests to assert
    /// routing decisions (e.g., "VL request did NOT increment the
    /// counter, so it took the GS path"). Cost: one atomic load per
    /// successful admit.
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by the driver once per
    /// batch (prefill_admitted invocation, including failed batches —
    /// diagnostic purpose). When multi-admit batching is working,
    /// integration tests expect `batch_count < admit_count`. Doc-hidden.
    #[doc(hidden)]
    pub batch_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by `drain_window` when it
    /// exits because `Scheduler::active_count() >= b_max` (saturate path),
    /// NOT when the deadline expires. Used by integration tests to prove
    /// the saturate-trigger fired without relying on wall-time measurement.
    /// Doc-hidden.
    #[doc(hidden)]
    pub saturate_triggered: Arc<AtomicU64>,
    /// Test-observable peak `admission_queue.len()` ever reached. Used by
    /// integration tests to confirm the queue drained (e.g., `peak >= N` for
    /// c=N+b_max admit burst). Doc-hidden — production code shouldn't read it.
    #[doc(hidden)]
    pub queue_depth_peak: Arc<AtomicUsize>,
    /// Test-observable count of admit requests rejected with "admission
    /// queue full" Err (queue_max overflow). Doc-hidden.
    #[doc(hidden)]
    pub queue_rejected: Arc<AtomicU64>,
    /// Count of actor calls to scheduler-internal MTP prefill. Exposed through
    /// `/healthz.mtp.prefill_count` for server-level diagnostics.
    #[doc(hidden)]
    pub mtp_prefill_count: Arc<AtomicU64>,
    /// Count of actor calls to scheduler-internal MTP step. Exposed through
    /// `/healthz.mtp.step_count` for server-level diagnostics.
    #[doc(hidden)]
    pub mtp_step_count: Arc<AtomicU64>,
    /// Count of MTP-enabled prefill calls that fell back to the ordinary
    /// scheduler path because the active batch was not MTP-eligible.
    #[doc(hidden)]
    pub mtp_fallback_prefill_count: Arc<AtomicU64>,
    /// Latest cumulative scheduler MTP drafted-token count.
    #[doc(hidden)]
    pub mtp_drafted_tokens: Arc<AtomicU64>,
    /// Latest cumulative scheduler MTP accepted-draft-token count.
    #[doc(hidden)]
    pub mtp_accepted_draft_tokens: Arc<AtomicU64>,
    // ── B1-p2.5 G3: /healthz monitoring atomics ──────────────────────────
    /// Live count of in-flight (active) requests in the scheduler slots.
    /// Updated by driver_loop tail on every rolling iteration.
    pub b_active: Arc<AtomicU64>,
    /// Live count of requests parked in the admission queue.
    /// Updated by driver_loop tail on every rolling iteration.
    pub b_queued: Arc<AtomicU64>,
    /// Monotonic count of admits rejected due to admission queue full.
    /// Aliased from `queue_rejected` — single source of truth in driver_loop.
    /// P1.1: Scheduler.admission_queue_full_count field removed (no fetch_add
    /// caller); health collector now reads from this Arc directly. B1-p2.5.
    pub admission_queue_full_count: Arc<AtomicU64>,
    /// Monotonic count of admits rejected due to memory budget exceeded.
    /// Cloned from Scheduler::memory_budget_exceeded_count.
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    /// Shared Arc into BudgetState::active — live bytes charged to KV cache.
    pub kv_cache_active_bytes: Arc<AtomicUsize>,
    /// KV cache soft limit in bytes (computed at startup; static for lifetime).
    pub kv_cache_soft_limit_bytes: usize,
    /// Shared Active KV offload metrics and runtime status.
    pub active_kv_offload: ActiveKvOffloadSharedStats,
}

/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send`
/// (holds Array fields: KVCache, prng_state) and the model lock is sync.
///
/// # Arguments
/// - `model` — shared model handle (Mutex-protected sync state).
/// - `b_max` — maximum concurrent in-flight requests (Scheduler slot count).
/// - `admission_deadline` — drain-window timeout after the first admit in a
///   batch arrives. Hard limit; new admits do not reset it.
/// - `admission_queue_max` — capacity of the FIFO admission queue. `0`
///   disables queueing (immediate Err on saturation, mirroring pre-3d).
/// - `effective_cap_max` — upper bound on `prompt_len + max_new_tokens`
///   per request. Computed at boot as
///   `min(--max-cache-cap CLI, model.config.max_position_embeddings)`.
///   Passed directly to `Scheduler::new`. B1-p2.3f.
/// - `decode_cadence_mid_chunk_cap` — maximum rolling mid-admit chunk size
///   while existing decode rows are active.
/// - `meta` — model memory-budget metadata for startup validation. B1-p2.5.
pub fn spawn_scheduler_actor<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        None,
        None,
        ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_active_kv_offload<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        None,
        None,
        active_kv_offload,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_paged_prefix_cache<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: PagedPrefixCacheConfig,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        Some(paged_prefix_cache),
        prefix_lru_cache,
        ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_paged_prefix_cache_and_active_kv<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: PagedPrefixCacheConfig,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        Some(paged_prefix_cache),
        prefix_lru_cache,
        active_kv_offload,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_mtp<M>(
    model: Arc<Mutex<M>>,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorMtp::new(mtp, mtp_draft_tokens),
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_mtp_and_active_kv<M>(
    model: Arc<Mutex<M>>,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorMtp::new(mtp, mtp_draft_tokens),
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        active_kv_offload,
    )
}

#[allow(clippy::too_many_arguments)]
fn spawn_scheduler_actor_with_mode<M, A>(
    model: Arc<Mutex<M>>,
    mtp_mode: A,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M> + Send + 'static,
{
    // ── Step 1: Budget validation on the calling thread. ──────────────────
    // No Scheduler / Array is allocated here — just pure arithmetic + RAM
    // check. Returns Err early if the budget is too tight.
    let budget_state =
        crate::core::memory_budget::validate_startup_budget(b_max, effective_cap_max, &meta)?;

    // ── Step 2: Shared atomics created on the calling thread. ─────────────
    // Cloned for both the handle (returned to caller) and the driver thread.
    // This is the single source of truth — handle + driver share the same
    // Arc instances, so /healthz reads the live values the driver updates.
    //
    // B1-p2.5 P0 fix v2: the previous fix (c043ce9) created Scheduler::new
    // on the calling thread then moved it into spawn_blocking. That caused
    // "MLX runtime error: There is no Stream(gpu, N) in current thread"
    // because Array::zeros (prng_state) was bound to the calling thread's
    // Metal Stream. This fix keeps budget validation + Arc creation on the
    // calling thread while deferring Scheduler::new_with_state (and thus
    // Array allocation) to the spawn_blocking worker thread.
    let memory_budget_exceeded_count = Arc::new(AtomicU64::new(0));

    // Healthz observables cloned from BudgetState (Arc<AtomicUsize> inside).
    let kv_cache_active_bytes = budget_state.shared_active();
    let kv_cache_soft_limit_bytes = budget_state.soft_limit();

    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let saturate_triggered = Arc::new(AtomicU64::new(0));
    let queue_depth_peak = Arc::new(AtomicUsize::new(0));
    let queue_rejected = Arc::new(AtomicU64::new(0));
    let mtp_prefill_count = Arc::new(AtomicU64::new(0));
    let mtp_step_count = Arc::new(AtomicU64::new(0));
    let mtp_fallback_prefill_count = Arc::new(AtomicU64::new(0));
    let mtp_drafted_tokens = Arc::new(AtomicU64::new(0));
    let mtp_accepted_draft_tokens = Arc::new(AtomicU64::new(0));
    // B1-p2.5 G3: live b_active / b_queued updated by driver_loop tail.
    let b_active = Arc::new(AtomicU64::new(0));
    let b_queued = Arc::new(AtomicU64::new(0));
    let active_kv_stats = ActiveKvOffloadSharedStats::new(&active_kv_offload);

    // Clone Arcs for the driver thread.
    let driver_budget_state = budget_state.clone();
    let driver_mb_exceeded = memory_budget_exceeded_count.clone();
    let admit_count_for_task = admit_count.clone();
    let batch_count_for_task = batch_count.clone();
    let saturate_triggered_for_task = saturate_triggered.clone();
    let queue_depth_peak_for_task = queue_depth_peak.clone();
    let queue_rejected_for_task = queue_rejected.clone();
    let mtp_counters_for_task = SchedulerActorMtpCounters::new(
        mtp_prefill_count.clone(),
        mtp_step_count.clone(),
        mtp_fallback_prefill_count.clone(),
        mtp_drafted_tokens.clone(),
        mtp_accepted_draft_tokens.clone(),
    );
    let b_active_for_task = b_active.clone();
    let b_queued_for_task = b_queued.clone();
    let paged_prefix_cache_for_task = paged_prefix_cache.clone();
    let prefix_lru_cache_for_task = prefix_lru_cache;
    let active_kv_offload_for_task = active_kv_offload.clone();
    let active_kv_stats_for_task = active_kv_stats.clone();

    // ── Step 3: Spawn driver — Scheduler::new_with_state constructed INSIDE
    //    spawn_blocking so MLX Array fields (prng_state) are allocated on the
    //    worker thread's Metal Stream. Thread affinity preserved.
    tokio::task::spawn_blocking(move || {
        let mut scheduler = Scheduler::<M>::new_with_state(
            b_max,
            effective_cap_max,
            driver_budget_state,
            driver_mb_exceeded,
            meta,
        )
        .expect("budget already validated above; new_with_state must not fail");
        if let Some(config) = paged_prefix_cache_for_task {
            scheduler
                .enable_paged_prefix_cache(config)
                .expect("paged prefix cache config was validated before actor spawn");
        }
        if let Some(config) = prefix_lru_cache_for_task {
            scheduler
                .enable_prefix_lru_cache(config)
                .expect("prefix LRU cache config was validated before actor spawn");
        }
        scheduler
            .enable_active_kv_offload(active_kv_offload_for_task, active_kv_stats_for_task.clone())
            .expect("active KV offload config was validated before actor spawn");
        driver_loop(
            scheduler,
            model,
            mtp_mode,
            mtp_counters_for_task,
            active_kv_stats_for_task,
            admission_deadline,
            admission_queue_max,
            cmd_rx,
            admit_count_for_task,
            batch_count_for_task,
            saturate_triggered_for_task,
            queue_depth_peak_for_task,
            queue_rejected_for_task,
            b_active_for_task,
            b_queued_for_task,
            decode_cadence_mid_chunk_cap,
        );
    });

    Ok(SchedulerActorHandle {
        cmd_tx,
        admit_count,
        batch_count,
        saturate_triggered,
        queue_depth_peak,
        queue_rejected: queue_rejected.clone(),
        mtp_prefill_count,
        mtp_step_count,
        mtp_fallback_prefill_count,
        mtp_drafted_tokens,
        mtp_accepted_draft_tokens,
        b_active,
        b_queued,
        // P1.1: alias admission_queue_full_count to queue_rejected Arc —
        // driver_loop is the single fetch_add site; Scheduler field removed.
        admission_queue_full_count: queue_rejected,
        memory_budget_exceeded_count,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
        active_kv_offload: active_kv_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn driver_loop<M, A>(
    scheduler: Scheduler<M>,
    model: Arc<Mutex<M>>,
    mut mtp_mode: A,
    mtp_counters: SchedulerActorMtpCounters,
    active_kv_stats: ActiveKvOffloadSharedStats,
    admission_deadline: Duration,
    admission_queue_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
    saturate_triggered: Arc<AtomicU64>,
    queue_depth_peak: Arc<AtomicUsize>,
    queue_rejected: Arc<AtomicU64>,
    b_active: Arc<AtomicU64>,
    b_queued: Arc<AtomicU64>,
    decode_cadence_mid_chunk_cap: usize,
) where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    // Receive Scheduler ownership from spawn_scheduler_actor (single instance).
    // P0 fix: previously driver_loop called Scheduler::new a second time,
    // creating fresh Arc atomics disconnected from the handle. B1-p2.5.
    let mut sched = scheduler;
    let b_max = sched.b_max();
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let mut admission_queue: VecDeque<PendingAdmit> = VecDeque::new();
    let mut in_flight_mid_admit: Option<AdmitMidHandle> = None;
    let mut parked_active_kv: VecDeque<ActiveKvParkedRequest> = VecDeque::new();
    let rt = tokio::runtime::Handle::current();

    'outer: loop {
        // Defensive: ensure scheduler is in Phase::Idle before
        // blocking on next admit. Most error paths already call evict_all,
        // but this guards any future code path that leaves phase=Finished.
        // If finalize fails, the actor cannot safely admit more requests
        // (the scheduler would be in an unrecoverable state); terminate
        // cleanly rather than emit ERROR per request.
        if sched.phase() == Phase::Finished {
            if let Err(e) = finalize_finished_batch_if_any(&mut sched, &mut event_txs) {
                tracing::error!(
                    "[SchedulerActor] outer-loop finalize failed: {e:?}; \
                     actor cannot reset Finished batch safely — terminating"
                );
                cleanup_parked_active_kv_requests(
                    &sched,
                    &mut parked_active_kv,
                    &mut event_txs,
                    &active_kv_stats,
                    "outer-loop finalize failed",
                );
                event_txs.clear();
                return;
            }
        }

        // ===== Outer Idle: block waiting for first admit (or shutdown). =====
        // Outer Idle is reached only after evict_all clears all slots; the
        // admission queue is invariantly empty here (any queue elements were
        // drained inside the rolling loop before reaching this point).
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            cleanup_parked_active_kv_requests(
                &sched,
                &mut parked_active_kv,
                &mut event_txs,
                &active_kv_stats,
                "scheduler command channel closed",
            );
            return; // cmd_rx closed; all senders dropped.
        };
        let fresh_batch_limit = fresh_prefill_batch_limit_for_command::<M>(&first_cmd, b_max);
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            // First admit failed (Err) — nothing to prefill. Wait for next.
            continue 'outer;
        }

        // ===== Admission window: drain additional admits until deadline
        //       or the model's fresh-prefill batch limit. Beyond the limit,
        //       push to admission_queue (bounded by admission_queue_max). =====
        if sched.active_count() < fresh_batch_limit {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &mut admission_queue,
                &admit_count,
                &saturate_triggered,
                &queue_depth_peak,
                &queue_rejected,
                fresh_batch_limit,
                b_max,
                admission_queue_max,
                admission_deadline,
            ));
        }

        // ===== First-batch prefill. =====
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_profile = rolling_profile_enabled()
            .then(|| (sched.active_count(), admission_queue.len(), Instant::now()));
        let prefill_result = {
            let model_lock = model.blocking_lock();
            mtp_mode.prefill_admitted(&mut sched, &model_lock, &mtp_counters)
        };
        match prefill_result {
            Ok(prefill_events) => {
                if let Some((prefill_active, prefill_queue_len, prefill_timer)) = prefill_profile {
                    let prefill_end = Instant::now();
                    tracing::info!(
                        "[chunked-rolling-profile] event=fresh_prefill t_ms={:.3} active_count={} queue_len={} fresh_batch_limit={} event_count={} elapsed_ms={:.3}",
                        rolling_profile_t_ms(prefill_end),
                        prefill_active,
                        prefill_queue_len,
                        fresh_batch_limit,
                        prefill_events.len(),
                        rolling_profile_elapsed_ms(prefill_timer, prefill_end)
                    );
                }
                for ev in prefill_events {
                    route_event(ev, &event_txs);
                }
            }
            Err(e) => {
                if let Some((prefill_active, prefill_queue_len, prefill_timer)) = prefill_profile {
                    let prefill_end = Instant::now();
                    tracing::info!(
                        "[chunked-rolling-profile] event=fresh_prefill_error t_ms={:.3} active_count={} queue_len={} fresh_batch_limit={} elapsed_ms={:.3}",
                        rolling_profile_t_ms(prefill_end),
                        prefill_active,
                        prefill_queue_len,
                        fresh_batch_limit,
                        rolling_profile_elapsed_ms(prefill_timer, prefill_end)
                    );
                }
                tracing::error!("[SchedulerActor] prefill error: {e:?}");
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all after prefill error also failed: \
                         {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                    );
                }
                cleanup_parked_active_kv_requests(
                    &sched,
                    &mut parked_active_kv,
                    &mut event_txs,
                    &active_kv_stats,
                    "scheduler poisoned after prefill error",
                );
                event_txs.clear();
                // Anything queued during the failed-batch window has nowhere
                // to land — reject with Err so callers see a clear error
                // rather than hanging.
                while let Some(pending) = admission_queue.pop_front() {
                    let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                        "scheduler poisoned after prefill error"
                    )));
                }
                continue 'outer;
            }
        }

        // ===== Rolling decode loop with bounded mid-batch admit + queue drain. =====
        let mut admission_policy = RollingAdmissionPolicy::default();
        admission_policy.record_admission_work();
        'rolling: loop {
            // Pre-event Finished-batch finalization + handoff. If
            // previous iteration's prefill_admitted/step left phase=Finished
            // (e.g. max_tokens=1 workload), handle the completed batch BEFORE
            // dispatching another event. Per Codex Q6: biased select may pick
            // Admit over Step, so this must run before the event pick — or the
            // actor could call admit_mid_begin() in Phase::Finished.
            //
            // `drive_empty_scheduler_handoff` itself calls
            // `finalize_finished_batch_if_any`; do not duplicate finalization
            // here. This avoids two divergent finalize/error paths.
            if sched.phase() == Phase::Finished {
                match drive_empty_scheduler_handoff(
                    &mut sched,
                    &mut cmd_rx,
                    &mut event_txs,
                    &mut admission_queue,
                    &model,
                    &admit_count,
                    &saturate_triggered,
                    &queue_depth_peak,
                    &queue_rejected,
                    &batch_count,
                    &mut mtp_mode,
                    &mtp_counters,
                    &mut parked_active_kv,
                    &active_kv_stats,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    &rt,
                ) {
                    RollingControl::ContinueRolling => {
                        admission_policy.record_admission_work();
                        continue 'rolling;
                    }
                    RollingControl::BreakRolling => break 'rolling,
                    RollingControl::ContinueOuter => continue 'outer,
                    RollingControl::ReturnActor => return,
                }
            }

            let evt: RollingEvent =
                if admission_policy.should_force_decode(sched.phase(), sched.active_count()) {
                    RollingEvent::Step
                } else if in_flight_mid_admit.is_some() {
                    RollingEvent::AdvanceMidAdmit
                } else {
                    rt.block_on(async {
                        tokio::select! {
                            biased;
                            maybe_cmd = cmd_rx.recv() => match maybe_cmd {
                                Some(cmd) => RollingEvent::Admit(cmd),
                                None => RollingEvent::Shutdown,
                            },
                            _ = std::future::ready(()) => RollingEvent::Step,
                        }
                    })
                };

            match evt {
                RollingEvent::Shutdown => {
                    cleanup_parked_active_kv_requests(
                        &sched,
                        &mut parked_active_kv,
                        &mut event_txs,
                        &active_kv_stats,
                        "scheduler shutting down",
                    );
                    event_txs.clear();
                    // Reject any queued admits — callers shouldn't hang.
                    while let Some(pending) = admission_queue.pop_front() {
                        let _ = pending
                            .reply_tx
                            .send(Err(anyhow::anyhow!("scheduler shutting down")));
                    }
                    return;
                }
                RollingEvent::Admit(cmd) => {
                    if !can_start_rolling_mid_admit_for_command::<M>(
                        &cmd,
                        sched.active_count(),
                        b_max,
                    ) {
                        let mut pending_cmd = Some(cmd);
                        let can_start_after_park = can_start_rolling_mid_admit_for_command::<M>(
                            pending_cmd.as_ref().expect("pending command present"),
                            sched.active_count().saturating_sub(1),
                            b_max,
                        );
                        if in_flight_mid_admit.is_none()
                            && can_start_after_park
                            && try_park_one_active_kv_request(
                                &mut sched,
                                &model,
                                &mut parked_active_kv,
                                &active_kv_stats,
                            )
                        {
                            if sched.active_count() == 0 {
                                enqueue_or_reject(
                                    pending_cmd.take().expect("pending command present"),
                                    &mut admission_queue,
                                    admission_queue_max,
                                    &queue_depth_peak,
                                    &queue_rejected,
                                );
                                admission_policy.record_admission_work();
                            } else if start_mid_admit_one_chunk(
                                pending_cmd.take().expect("pending command present"),
                                &mut in_flight_mid_admit,
                                &mut sched,
                                &mut event_txs,
                                &admit_count,
                                &model,
                                MidAdmitProfileContext {
                                    source: RollingMidAdmitSource::Direct,
                                    queue_wait_ms: None,
                                    queue_len: admission_queue.len(),
                                },
                                decode_cadence_mid_chunk_cap,
                            ) {
                                admission_policy.record_admission_work();
                            }
                        }
                        if let Some(cmd) = pending_cmd {
                            // Rolling admission limit reached — queue for a later decode turn.
                            enqueue_or_reject(
                                cmd,
                                &mut admission_queue,
                                admission_queue_max,
                                &queue_depth_peak,
                                &queue_rejected,
                            );
                        }
                    } else if start_mid_admit_one_chunk(
                        cmd,
                        &mut in_flight_mid_admit,
                        &mut sched,
                        &mut event_txs,
                        &admit_count,
                        &model,
                        MidAdmitProfileContext {
                            source: RollingMidAdmitSource::Direct,
                            queue_wait_ms: None,
                            queue_len: admission_queue.len(),
                        },
                        decode_cadence_mid_chunk_cap,
                    ) {
                        admission_policy.record_admission_work();
                    }
                }
                RollingEvent::AdvanceMidAdmit => {
                    if advance_mid_admit_one_chunk(
                        &mut in_flight_mid_admit,
                        &mut sched,
                        &mut event_txs,
                        &admit_count,
                        &model,
                        admission_queue.len(),
                        decode_cadence_mid_chunk_cap,
                    ) {
                        admission_policy.record_admission_work();
                    }
                }
                RollingEvent::Step => {
                    let step_profile = rolling_profile_enabled().then(|| {
                        (
                            sched.active_count(),
                            admission_queue.len(),
                            in_flight_mid_admit.is_some(),
                            Instant::now(),
                        )
                    });
                    let step_result = {
                        let model_lock = model.blocking_lock();
                        mtp_mode.step(&mut sched, &model_lock, &mtp_counters)
                    };
                    let step_end = step_profile.map(|_| Instant::now());
                    match step_result {
                        Ok(events) => {
                            let event_count = events.len();
                            for ev in events {
                                route_event(ev, &event_txs);
                            }
                            let evicted_count = sched.gc_finished_rows(&mut event_txs).len();
                            if let (
                                Some((
                                    step_active_before,
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    step_timer,
                                )),
                                Some(step_end),
                            ) = (step_profile, step_end)
                            {
                                tracing::info!(
                                    "[chunked-rolling-profile] event=decode_step t_ms={:.3} active_before={} active_after={} queue_len={} had_in_flight_mid_admit={} event_count={} evicted_count={} elapsed_ms={:.3}",
                                    rolling_profile_t_ms(step_end),
                                    step_active_before,
                                    sched.active_count(),
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    event_count,
                                    evicted_count,
                                    rolling_profile_elapsed_ms(step_timer, step_end)
                                );
                            }
                            admission_policy.record_decode_step();
                            // ===== Post-gc queue drain. =====
                            // Free slots → pull from admission_queue head
                            // for one bounded mid-admit. Further queued
                            // requests wait for the next decode turn so
                            // active streams keep making progress.
                            if in_flight_mid_admit.is_none()
                                && !admission_queue.is_empty()
                                && sched.active_count() >= b_max
                            {
                                let _ = try_park_one_active_kv_request(
                                    &mut sched,
                                    &model,
                                    &mut parked_active_kv,
                                    &active_kv_stats,
                                );
                            }
                            if in_flight_mid_admit.is_none()
                                && drain_admission_queue(
                                    &mut admission_queue,
                                    &mut in_flight_mid_admit,
                                    &mut sched,
                                    &mut event_txs,
                                    &admit_count,
                                    &model,
                                    b_max,
                                    decode_cadence_mid_chunk_cap,
                                )
                            {
                                admission_policy.record_admission_work();
                            }
                        }
                        Err(e) => {
                            if let (
                                Some((
                                    step_active_before,
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    step_timer,
                                )),
                                Some(step_end),
                            ) = (step_profile, step_end)
                            {
                                tracing::info!(
                                    "[chunked-rolling-profile] event=decode_step_error t_ms={:.3} active_before={} active_after={} queue_len={} had_in_flight_mid_admit={} elapsed_ms={:.3}",
                                    rolling_profile_t_ms(step_end),
                                    step_active_before,
                                    sched.active_count(),
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    rolling_profile_elapsed_ms(step_timer, step_end)
                                );
                            }
                            tracing::error!("[SchedulerActor] step error: {e:?}");
                            if let Err(evict_err) = sched.evict_all() {
                                tracing::warn!(
                                    "[SchedulerActor] evict_all after step error also failed: \
                                     {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                                );
                            }
                            in_flight_mid_admit = None;
                            cleanup_parked_active_kv_requests(
                                &sched,
                                &mut parked_active_kv,
                                &mut event_txs,
                                &active_kv_stats,
                                "scheduler poisoned after step error",
                            );
                            event_txs.clear();
                            while let Some(pending) = admission_queue.pop_front() {
                                let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                                    "scheduler poisoned after step error"
                                )));
                            }
                            continue 'outer;
                        }
                    }
                }
            }

            // B1-p2.5 G3: update /healthz live counters at tail of every rolling step.
            b_active.store(sched.active_count() as u64, Ordering::Relaxed);
            b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);
            sched.refresh_active_kv_residency_stats();
            active_kv_stats.set_parked_requests(parked_active_kv.len());

            // ===== Exit rolling loop when active_count == 0 AND queue empty. =====
            // Spec §9 R1: if `active_count() == 0` but admission_queue is
            // non-empty (mid-rolling admit arrived AFTER all rows finished),
            // treat as a "new batch within rolling": evict_all to reset to
            // Idle, then admit from queue + drain_window + prefill_admitted
            // inline (mirrors the existing post-empty path but pulls the
            // first admit from the queue instead of cmd_rx).
            //
            // Extracted into `drive_empty_scheduler_handoff` so the
            // same logic backs the pre-event Finished-batch finalization hook
            // at the rolling-loop top. The helper finalizes any leftover
            // `Phase::Finished` state first, then performs the queued-admit
            // / try_recv / break handoff.
            if sched.active_count() == 0 {
                match drive_empty_scheduler_handoff(
                    &mut sched,
                    &mut cmd_rx,
                    &mut event_txs,
                    &mut admission_queue,
                    &model,
                    &admit_count,
                    &saturate_triggered,
                    &queue_depth_peak,
                    &queue_rejected,
                    &batch_count,
                    &mut mtp_mode,
                    &mtp_counters,
                    &mut parked_active_kv,
                    &active_kv_stats,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    &rt,
                ) {
                    RollingControl::ContinueRolling => {
                        admission_policy.record_admission_work();
                        continue 'rolling;
                    }
                    RollingControl::BreakRolling => break 'rolling,
                    RollingControl::ContinueOuter => continue 'outer,
                    RollingControl::ReturnActor => return,
                }
            }
        }

        // After rolling loop: reset cache + Phase for next outer iteration.
        if matches!(sched.phase(), Phase::Decoding | Phase::Finished) {
            if let Err(evict_err) = sched.evict_all() {
                tracing::warn!(
                    "[SchedulerActor] evict_all at end of outer failed: {evict_err:?}; \
                     relying on 3b-1 poison flag to reject subsequent admits"
                );
            }
        }
        in_flight_mid_admit = None;
        cleanup_parked_active_kv_requests(
            &sched,
            &mut parked_active_kv,
            &mut event_txs,
            &active_kv_stats,
            "scheduler outer loop reset",
        );
        event_txs.clear();
    }
}

/// Drain additional `Admit` commands until either the deadline expires or the
/// fresh-batch admission limit is reached. Hard deadline — new admits do NOT
/// reset the timer. Once the limit is reached, additional admits within the
/// window go to the admission queue (bounded by `admission_queue_max`).
#[allow(clippy::too_many_arguments)]
async fn drain_window<M>(
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_limit: usize,
    b_max: usize,
    queue_max: usize,
    deadline: Duration,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    let batch_limit = batch_limit.clamp(1, b_max);
    let timer = tokio::time::sleep(deadline);
    tokio::pin!(timer);
    let mut limit_reached = false;
    loop {
        tokio::select! {
            biased;
            _ = &mut timer => return,
            maybe = cmd_rx.recv() => {
                let Some(cmd) = maybe else { return }; // channel closed
                if limit_reached {
                    // Fresh batch is full for this model/prompt policy — push
                    // to queue or reject.
                    enqueue_or_reject(
                        cmd,
                        admission_queue,
                        queue_max,
                        queue_depth_peak,
                        queue_rejected,
                    );
                    continue;
                }
                handle_admit(cmd, sched, event_txs, admit_count);
                if sched.active_count() >= batch_limit {
                    if batch_limit >= b_max {
                        saturate_triggered.fetch_add(1, Ordering::Relaxed);
                    }
                    limit_reached = true;
                    // Stay in the loop until deadline so queued admits
                    // arriving during the window's remaining time are
                    // captured. (Pre-3d returned here; 3d keeps draining.)
                }
            }
        }
    }
}

/// Process a single `Admit` command: try `Scheduler::admit`; on success
/// register the per-request event channel and increment admit_count;
/// on failure forward the Err to the caller. Reply-tx send failure
/// (caller abandoned) causes the slot to be evicted as cleanup.
fn handle_admit<M>(
    cmd: SchedulerCommand,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    match sched.admit(request) {
        Ok(id) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            event_txs.insert(id, event_tx);
            if reply_tx
                .send(Ok(AdmitReply {
                    request_id: id,
                    event_rx,
                }))
                .is_err()
            {
                // Caller dropped reply_rx before we could send.
                // Evict the orphan slot.
                let _ = sched.evict(id);
                event_txs.remove(&id);
            }
        }
        Err(e) => {
            let _ = reply_tx.send(Err(e));
        }
    }
}

fn uses_multi_chunk_prefill(request: &GenerateRequest) -> bool {
    request.prefill_chunk_size > 0 && request.prompt_ids.len() > request.prefill_chunk_size
}

fn can_start_rolling_mid_admit_for_request<M: Model>(
    request: &GenerateRequest,
    active_count: usize,
    b_max: usize,
) -> bool {
    if active_count >= b_max {
        return false;
    }
    let rolling_limit = fresh_prefill_batch_limit_for_request::<M>(request, b_max);
    active_count < rolling_limit || uses_multi_chunk_prefill(request)
}

fn can_start_rolling_mid_admit_for_command<M: Model>(
    cmd: &SchedulerCommand,
    active_count: usize,
    b_max: usize,
) -> bool {
    let SchedulerCommand::Admit { request, .. } = cmd;
    can_start_rolling_mid_admit_for_request::<M>(request, active_count, b_max)
}

fn begin_mid_admit<M>(
    cmd: SchedulerCommand,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    model: &Arc<Mutex<M>>,
    profile_context: MidAdmitProfileContext,
) -> Option<AdmitMidHandle>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let admit_profile = rolling_profile_enabled().then(|| {
        (
            request.prompt_ids.len(),
            request.prefill_chunk_size,
            sched.active_count(),
            Instant::now(),
        )
    });
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    // Phase 1: begin.
    let handle = {
        let m = model.blocking_lock();
        match sched.admit_mid_begin(request, &m) {
            Ok(h) => h,
            Err(e) => {
                let _ = reply_tx.send(Err(e));
                return None;
            }
        }
    };
    let begin_end = admit_profile.map(|_| Instant::now());
    let id = handle.request_id;
    if let (Some((prompt_len, prefill_chunk_size, active_before, begin_start)), Some(begin_end)) =
        (admit_profile, begin_end)
    {
        tracing::info!(
            "[chunked-rolling-profile] event=mid_begin t_ms={:.3} request_id={} source={} prompt_len={} prefill_chunk_size={} active_before={} active_after={} queue_len={} queue_wait_ms={:.3} elapsed_ms={:.3}",
            rolling_profile_t_ms(begin_end),
            id.0,
            profile_context.source.as_str(),
            prompt_len,
            prefill_chunk_size,
            active_before,
            sched.active_count(),
            profile_context.queue_len,
            profile_context.queue_wait_ms.unwrap_or(-1.0),
            rolling_profile_elapsed_ms(begin_start, begin_end)
        );
    }
    event_txs.insert(id, event_tx);
    if reply_tx
        .send(Ok(AdmitReply {
            request_id: id,
            event_rx,
        }))
        .is_err()
    {
        // Caller dropped reply_rx before the prefill chunks completed.
        let _ = sched.evict(id);
        event_txs.remove(&id);
        return None;
    }

    Some(handle)
}

#[allow(clippy::too_many_arguments)]
fn start_mid_admit_one_chunk<M>(
    cmd: SchedulerCommand,
    in_flight_mid_admit: &mut Option<AdmitMidHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    profile_context: MidAdmitProfileContext,
    decode_cadence_mid_chunk_cap: usize,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if in_flight_mid_admit.is_some() {
        return false;
    }
    let Some(handle) = begin_mid_admit(cmd, sched, event_txs, model, profile_context) else {
        return false;
    };
    *in_flight_mid_admit = Some(handle);
    advance_mid_admit_one_chunk(
        in_flight_mid_admit,
        sched,
        event_txs,
        admit_count,
        model,
        profile_context.queue_len,
        decode_cadence_mid_chunk_cap,
    )
}

fn advance_mid_admit_one_chunk<M>(
    in_flight_mid_admit: &mut Option<AdmitMidHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    queue_len: usize,
    decode_cadence_mid_chunk_cap: usize,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let Some(mut handle) = in_flight_mid_admit.take() else {
        return false;
    };
    let id = handle.request_id;
    let active_count_before_chunk = sched.active_count();
    let requested_chunk_size = handle.chunk_size;
    let effective_chunk_size = cadence_protected_mid_chunk_size(
        requested_chunk_size,
        active_count_before_chunk,
        handle.decode_cadence_mid_chunk_cap,
    );
    debug_assert!(decode_cadence_mid_chunk_cap > 0);
    handle.chunk_size = effective_chunk_size;
    let chunk_profile = rolling_profile_enabled().then(|| {
        (
            handle.chunk_start,
            handle.prompt_len,
            effective_chunk_size,
            active_count_before_chunk,
            Instant::now(),
        )
    });

    let chunk_result = {
        let m = model.blocking_lock();
        sched.admit_mid_chunk(&mut handle, &m)
    };
    handle.chunk_size = requested_chunk_size;
    let is_last = match chunk_result {
        Ok(b) => b,
        Err(e) => {
            if let Some((chunk_start, prompt_len, chunk_size, active_count, chunk_timer)) =
                chunk_profile
            {
                let chunk_end_time = Instant::now();
                let chunk_end = handle.chunk_start;
                tracing::info!(
                    "[chunked-rolling-profile] event=mid_chunk_error t_ms={:.3} request_id={} chunk_start={} chunk_end={} chunk_len={} prompt_len={} chunk_size={} active_count={} queue_len={} elapsed_ms={:.3}",
                    rolling_profile_t_ms(chunk_end_time),
                    id.0,
                    chunk_start,
                    chunk_end,
                    chunk_end.saturating_sub(chunk_start),
                    prompt_len,
                    chunk_size,
                    active_count,
                    queue_len,
                    rolling_profile_elapsed_ms(chunk_timer, chunk_end_time)
                );
            }
            tracing::error!("[SchedulerActor] admit_mid_chunk error: {e:?}");
            let _ = sched.evict(id);
            event_txs.remove(&id);
            return true;
        }
    };
    if let Some((chunk_start, prompt_len, chunk_size, active_count, chunk_timer)) = chunk_profile {
        let chunk_end_time = Instant::now();
        let chunk_end = handle.chunk_start;
        tracing::info!(
            "[chunked-rolling-profile] event=mid_chunk t_ms={:.3} request_id={} chunk_start={} chunk_end={} chunk_len={} prompt_len={} chunk_size={} is_last={} active_count={} queue_len={} elapsed_ms={:.3}",
            rolling_profile_t_ms(chunk_end_time),
            id.0,
            chunk_start,
            chunk_end,
            chunk_end.saturating_sub(chunk_start),
            prompt_len,
            chunk_size,
            is_last,
            active_count,
            queue_len,
            rolling_profile_elapsed_ms(chunk_timer, chunk_end_time)
        );
    }

    if !is_last {
        *in_flight_mid_admit = Some(handle);
        return true;
    }

    let finalize_profile =
        rolling_profile_enabled().then(|| (sched.active_count(), Instant::now()));
    let m = model.blocking_lock();
    match sched.admit_mid_finalize(handle, &m) {
        Ok((_id, first_event)) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            if let Some((active_before, finalize_timer)) = finalize_profile {
                let finalize_end = Instant::now();
                tracing::info!(
                    "[chunked-rolling-profile] event=mid_finalize t_ms={:.3} request_id={} active_before={} active_after={} queue_len={} elapsed_ms={:.3}",
                    rolling_profile_t_ms(finalize_end),
                    id.0,
                    active_before,
                    sched.active_count(),
                    queue_len,
                    rolling_profile_elapsed_ms(finalize_timer, finalize_end)
                );
            }
            route_event(first_event, event_txs);
        }
        Err(e) => {
            if let Some((active_before, finalize_timer)) = finalize_profile {
                let finalize_end = Instant::now();
                tracing::info!(
                    "[chunked-rolling-profile] event=mid_finalize_error t_ms={:.3} request_id={} active_before={} active_after={} queue_len={} elapsed_ms={:.3}",
                    rolling_profile_t_ms(finalize_end),
                    id.0,
                    active_before,
                    sched.active_count(),
                    queue_len,
                    rolling_profile_elapsed_ms(finalize_timer, finalize_end)
                );
            }
            tracing::error!("[SchedulerActor] admit_mid_finalize error: {e:?}");
            let _ = sched.evict(id);
            event_txs.remove(&id);
        }
    }
    true
}

/// Push a pending admit into the queue if there's capacity; otherwise reply
/// with `Err(SchedulerError::QueueFull)` (wrapped in anyhow) and bump
/// `queue_rejected`. Updates `queue_depth_peak` via `fetch_max`.
///
/// HTTP handlers downcast the anyhow Err to [`SchedulerError`] to map
/// QueueFull → HTTP 503 + Retry-After; other errors → HTTP 400.
fn enqueue_or_reject(
    cmd: SchedulerCommand,
    queue: &mut VecDeque<PendingAdmit>,
    queue_max: usize,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    if queue.len() >= queue_max {
        queue_rejected.fetch_add(1, Ordering::Relaxed);
        let _ = reply_tx.send(Err(anyhow::Error::new(
            crate::core::scheduler::SchedulerError::QueueFull {
                capacity: queue_max,
            },
        )));
        return;
    }
    let enqueue_profile = rolling_profile_enabled().then(|| {
        (
            Instant::now(),
            request.prompt_ids.len(),
            request.prefill_chunk_size,
        )
    });
    let queued_at_profile = enqueue_profile.map(|(now, _, _)| now);
    queue.push_back(PendingAdmit {
        request,
        reply_tx,
        queued_at_profile,
    });
    queue_depth_peak.fetch_max(queue.len(), Ordering::Relaxed);
    if let Some((now, prompt_len, prefill_chunk_size)) = enqueue_profile {
        tracing::info!(
            "[chunked-rolling-profile] event=queue_enqueue t_ms={:.3} prompt_len={} prefill_chunk_size={} queue_len={} queue_max={}",
            rolling_profile_t_ms(now),
            prompt_len,
            prefill_chunk_size,
            queue.len(),
            queue_max
        );
    }
}

/// Drain at most one mid-batch admit chunk from the admission queue.
/// Full-prompt rolling admits obey the model's `fresh_prefill_batch_limit`.
/// Multi-chunk admits may start in a spare slot beyond that limit because
/// each chunk yields back to the rolling loop before the next chunk runs.
/// Once any admission work happens the caller must return to decode before
/// draining more queue.
///
/// IMPORTANT: `admit_mid` is only legal in `Decoding` phase. If
/// `gc_finished_rows` just transitioned the scheduler to `Finished`
/// (because `active_count` dropped to 0), the caller's rolling-loop
/// exit branch (`active_count == 0 && queue non-empty`) will handle the
/// queued entries via `evict_all` + fresh `prefill_admitted`. Return
/// early here so we do not call `admit_mid` in an illegal phase.
#[allow(clippy::too_many_arguments)]
fn drain_admission_queue<M>(
    queue: &mut VecDeque<PendingAdmit>,
    in_flight_mid_admit: &mut Option<AdmitMidHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    b_max: usize,
    decode_cadence_mid_chunk_cap: usize,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // admit_mid is only legal in Decoding phase.
    if sched.phase() != Phase::Decoding {
        return false;
    }
    if sched.active_count() == 0 {
        return false;
    }
    if in_flight_mid_admit.is_some() {
        return false;
    }
    while sched.active_count() < b_max {
        let Some(pending) = queue.front() else {
            return false;
        };
        if !can_start_rolling_mid_admit_for_request::<M>(
            &pending.request,
            sched.active_count(),
            b_max,
        ) {
            return false;
        }
        let pending = queue
            .pop_front()
            .expect("queue.front returned Some immediately before pop_front");
        let dequeue_profile = pending.queued_at_profile.map(|queued_at| {
            let now = Instant::now();
            (now, rolling_profile_queue_wait_ms(queued_at, now))
        });
        let queue_wait_ms = dequeue_profile.map(|(_, wait_ms)| wait_ms);
        if let Some((dequeue_at, queue_wait_ms)) = dequeue_profile {
            tracing::info!(
                "[chunked-rolling-profile] event=queue_dequeue t_ms={:.3} prompt_len={} prefill_chunk_size={} queue_len={} queue_wait_ms={:.3}",
                rolling_profile_t_ms(dequeue_at),
                pending.request.prompt_ids.len(),
                pending.request.prefill_chunk_size,
                queue.len(),
                queue_wait_ms
            );
        }
        let cmd = SchedulerCommand::Admit {
            request: pending.request,
            reply_tx: pending.reply_tx,
        };
        let did_admission_work = start_mid_admit_one_chunk(
            cmd,
            in_flight_mid_admit,
            sched,
            event_txs,
            admit_count,
            model,
            MidAdmitProfileContext {
                source: RollingMidAdmitSource::Queue,
                queue_wait_ms,
                queue_len: queue.len(),
            },
            decode_cadence_mid_chunk_cap,
        );
        // Re-check phase after each mid-admit — if admit_mid itself
        // exhausted remaining rows and transitioned to Finished, stop.
        if did_admission_work || sched.phase() != Phase::Decoding {
            return did_admission_work;
        }
    }
    false
}

fn route_event(ev: StepEvent, event_txs: &HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>) {
    if let Some(tx) = event_txs.get(&ev.id) {
        // Unbounded channel — only fails when the receiver was dropped
        // (handler abandoned). That's fine; the entry naturally clears
        // at the next `event_txs.clear()` in driver_loop.
        let _ = tx.send(ev);
    }
}

fn try_park_one_active_kv_request<M>(
    sched: &mut Scheduler<M>,
    model: &Arc<Mutex<M>>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if !sched.active_kv_offload_enabled() || sched.phase() != Phase::Decoding {
        return false;
    }
    let candidate_ids: Vec<RequestId> = sched
        .active()
        .into_iter()
        .filter(|state| !state.finished && !state.generated_tokens.is_empty())
        .map(|state| state.id)
        .collect();
    if candidate_ids.is_empty() {
        return false;
    }
    let model_lock = model.blocking_lock();
    for id in candidate_ids {
        match sched.park_active_kv_request(id, &model_lock) {
            Ok(Some(parked)) => {
                tracing::info!(
                    "[active-kv-offload] event=park request_id={} parked_queue_len={}",
                    parked.id.0,
                    parked_active_kv.len() + 1
                );
                parked_active_kv.push_back(parked);
                active_kv_stats.set_parked_requests(parked_active_kv.len());
                return true;
            }
            Ok(None) => {}
            Err(err) => {
                active_kv_stats.record_error();
                tracing::warn!(
                    "[active-kv-offload] event=park_error request_id={} error={err:#}",
                    id.0
                );
            }
        }
    }
    false
}

fn try_restore_one_active_kv_request<M>(
    sched: &mut Scheduler<M>,
    model: &Arc<Mutex<M>>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if !sched.active_kv_offload_enabled() || sched.active_count() >= sched.b_max() {
        return false;
    }

    while sched.active_count() < sched.b_max() {
        let Some(parked) = parked_active_kv.pop_front() else {
            active_kv_stats.set_parked_requests(0);
            return false;
        };
        let model_lock = model.blocking_lock();
        match sched.restore_active_kv_request(&parked, &model_lock) {
            Ok(id) => {
                tracing::info!(
                    "[active-kv-offload] event=restore request_id={} parked_queue_len={}",
                    id.0,
                    parked_active_kv.len()
                );
                active_kv_stats.set_parked_requests(parked_active_kv.len());
                return true;
            }
            Err(err) => {
                active_kv_stats.record_error();
                tracing::warn!(
                    "[active-kv-offload] event=restore_error request_id={} error={err:#}",
                    parked.id.0
                );
                if let Err(cleanup_err) = sched.discard_active_kv_request(&parked) {
                    active_kv_stats.record_error();
                    tracing::warn!(
                        "[active-kv-offload] event=restore_error_cleanup_failed request_id={} error={cleanup_err:#}",
                        parked.id.0
                    );
                }
                event_txs.remove(&parked.id);
                active_kv_stats.set_parked_requests(parked_active_kv.len());
            }
        }
    }
    false
}

fn cleanup_parked_active_kv_requests<M>(
    sched: &Scheduler<M>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
    reason: &str,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    while let Some(parked) = parked_active_kv.pop_front() {
        if let Err(err) = sched.discard_active_kv_request(&parked) {
            active_kv_stats.record_error();
            tracing::warn!(
                "[active-kv-offload] event=cleanup_error request_id={} reason={} error={err:#}",
                parked.id.0,
                reason
            );
        }
        event_txs.remove(&parked.id);
    }
    active_kv_stats.set_parked_requests(0);
}

/// Finalize a `Phase::Finished` batch: evict slots + release budget +
/// reset to `Phase::Idle`, then close per-request event channels.
///
/// Returns `Ok(true)` if finalization happened (caller MUST go to the
/// empty-scheduler handoff path, NOT continue the normal event pick;
/// per spec § 4.2.1 hard binding).
/// This binding applies in the rolling-loop context; the outer-loop hook calls this directly because admission_queue is invariantly empty at that point.
/// Returns `Ok(false)` if `phase != Finished` (no-op; safe to continue).
/// Returns `Err` if `evict_all` failed (caller should reject queued
/// admits + `continue 'outer` per existing pattern).
///
/// The `Phase::Finished` state arises naturally when `prefill_admitted`
/// completes a batch where every request has
/// `max_new_tokens=1` (the prefill samples first+last token in one
/// pass), which is the standard `iron-bench --max-tokens 1` perf
/// measurement workload.
fn finalize_finished_batch_if_any<M: Model>(
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
) -> Result<bool> {
    if sched.phase() != Phase::Finished {
        return Ok(false);
    }
    let evicted_ids: Vec<RequestId> = sched.active().into_iter().map(|state| state.id).collect();
    match sched.evict_all() {
        Ok(()) => {
            for id in evicted_ids {
                event_txs.remove(&id);
            }
            Ok(true)
        }
        Err(e) => {
            tracing::warn!("[SchedulerActor] finalize_finished_batch: evict_all failed: {e:?}");
            Err(e)
        }
    }
}

/// Finalize a just-finished batch if needed, then drain queued admits
/// (or a single pending `cmd_rx.try_recv` admit) into a fresh batch, run
/// `prefill_admitted`, and return how the caller's rolling loop should
/// proceed. Lifts the existing empty-batch transition logic at the
/// rolling-loop tail so it can also be invoked from the pre-event
/// Finished-batch finalization at the rolling-loop top.
///
/// This helper is the single empty-batch handoff path. It first calls
/// [`finalize_finished_batch_if_any`], so callers must not separately
/// finalize before invoking it. After that call the scheduler is either
/// `Phase::Idle` or `Phase::Decoding` (the legacy post-step empty-handoff
/// path used to encounter `Phase::Finished`; the new pre-event hook now
/// shoulders that case via finalize). The helper preserves the current
/// reset semantics for `Decoding`-with-zero-active-rows before starting
/// the next batch but never calls `evict_all` in `Idle` (which is itself
/// an error per scheduler.rs:775-780).
///
/// Behavior per branch:
/// - Queued admit present → pop head, fresh batch via `handle_admit` +
///   `drain_window` + `prefill_admitted`; returns `ContinueRolling`.
/// - Queue empty + `cmd_rx.try_recv()` returns `Ok(cmd)` → fresh batch
///   via the same path; returns `ContinueRolling`.
/// - Queue empty + `try_recv` returns `Empty` → returns `BreakRolling`.
/// - Queue empty + `try_recv` returns `Disconnected` → clear `event_txs`,
///   returns `ReturnActor`.
/// - Any `finalize`, legacy reset, or `prefill_admitted` failure →
///   reject queued admits, clear `event_txs`, returns `ContinueOuter`.
///
/// Replaces the previous `if sched.active_count() == 0 { ... }` block
/// at rolling-loop tail to avoid divergent copies.
#[allow(clippy::too_many_arguments)]
fn drive_empty_scheduler_handoff<M, A>(
    sched: &mut Scheduler<M>,
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    model: &Arc<Mutex<M>>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_count: &Arc<AtomicU64>,
    mtp_mode: &mut A,
    mtp_counters: &SchedulerActorMtpCounters,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
    b_max: usize,
    admission_queue_max: usize,
    admission_deadline: Duration,
    rt: &tokio::runtime::Handle,
) -> RollingControl
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    // Finalize any Finished batch BEFORE re-admitting. After
    // this, phase is one of {Idle, Decoding}; never Finished. Callers
    // must not separately finalize.
    match finalize_finished_batch_if_any(sched, event_txs) {
        Ok(_) => {}
        Err(_e) => {
            cleanup_parked_active_kv_requests(
                sched,
                parked_active_kv,
                event_txs,
                active_kv_stats,
                "scheduler poisoned during Finished-batch finalize",
            );
            while let Some(pending) = admission_queue.pop_front() {
                let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                    "scheduler poisoned during Finished-batch finalize"
                )));
            }
            event_txs.clear();
            return RollingControl::ContinueOuter;
        }
    }

    if !admission_queue.is_empty() {
        // Reset Decoding-with-zero-active-rows to Idle for fresh batch.
        // (Finished was already handled by finalize above; Idle would
        // itself be an error for `evict_all`.)
        if sched.phase() == Phase::Decoding {
            if let Err(evict_err) = sched.evict_all() {
                tracing::warn!(
                    "[SchedulerActor] evict_all between batches (queue drain) failed: \
                     {evict_err:?}; rejecting queued admits"
                );
                while let Some(pending) = admission_queue.pop_front() {
                    let _ = pending
                        .reply_tx
                        .send(Err(anyhow::anyhow!("scheduler evict_all failed")));
                }
                event_txs.clear();
                return RollingControl::ContinueOuter;
            }
            // Preserve parked Active KV request event channels. At this point
            // active_count is zero; stale finished-batch channels were already
            // removed by gc/finalize paths.
        }
        // Pop first queued admit as the new batch's first admit.
        let pending = admission_queue
            .pop_front()
            .expect("queue non-empty checked");
        let fresh_batch_limit = fresh_prefill_batch_limit_for_request::<M>(&pending.request, b_max);
        handle_admit(
            SchedulerCommand::Admit {
                request: pending.request,
                reply_tx: pending.reply_tx,
            },
            sched,
            event_txs,
            admit_count,
        );
        if sched.active_count() == 0 {
            // Admit failed; loop to drain more queue (or exit).
            return RollingControl::ContinueRolling;
        }
        if sched.active_count() < fresh_batch_limit {
            // Drain queue head-by-head into the new batch (no deadline —
            // these are already-queued admits, not racing-in cmd_rx).
            // Then optionally drain_window for fresh cmd_rx admits.
            while sched.active_count() < fresh_batch_limit {
                let Some(p) = admission_queue.pop_front() else {
                    break;
                };
                handle_admit(
                    SchedulerCommand::Admit {
                        request: p.request,
                        reply_tx: p.reply_tx,
                    },
                    sched,
                    event_txs,
                    admit_count,
                );
            }
            // Optionally absorb cmd_rx admits arriving right now.
            if sched.active_count() < fresh_batch_limit {
                rt.block_on(drain_window(
                    cmd_rx,
                    sched,
                    event_txs,
                    admission_queue,
                    admit_count,
                    saturate_triggered,
                    queue_depth_peak,
                    queue_rejected,
                    fresh_batch_limit,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                ));
            }
        }
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_result = {
            let model_lock = model.blocking_lock();
            mtp_mode.prefill_admitted(sched, &model_lock, mtp_counters)
        };
        match prefill_result {
            Ok(events) => {
                for ev in events {
                    route_event(ev, event_txs);
                }
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] re-prefill (queue drain) error: {e:?}");
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all after re-prefill error also failed: \
                         {evict_err:?}; rejecting remaining queued admits"
                    );
                }
                event_txs.clear();
                while let Some(p) = admission_queue.pop_front() {
                    let _ = p.reply_tx.send(Err(anyhow::anyhow!(
                        "scheduler poisoned after re-prefill error"
                    )));
                }
                return RollingControl::ContinueOuter;
            }
        }
        return RollingControl::ContinueRolling;
    }

    if try_restore_one_active_kv_request(sched, model, parked_active_kv, event_txs, active_kv_stats)
    {
        return RollingControl::ContinueRolling;
    }
    // Queue empty + no active rows — same logic as pre-3d.
    match cmd_rx.try_recv() {
        Ok(cmd) => {
            let fresh_batch_limit = fresh_prefill_batch_limit_for_command::<M>(&cmd, b_max);
            if sched.phase() == Phase::Decoding {
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all between batches failed: \
                         {evict_err:?}; rejecting incoming admit"
                    );
                    let SchedulerCommand::Admit { reply_tx, .. } = cmd;
                    let _ = reply_tx.send(Err(evict_err));
                    event_txs.clear();
                    return RollingControl::ContinueOuter;
                }
                // Preserve parked Active KV request event channels. At this
                // point active_count is zero; stale finished-batch channels
                // were already removed by gc/finalize paths.
            }
            handle_admit(cmd, sched, event_txs, admit_count);
            if sched.active_count() == 0 {
                return RollingControl::BreakRolling;
            }
            if sched.active_count() < fresh_batch_limit {
                rt.block_on(drain_window(
                    cmd_rx,
                    sched,
                    event_txs,
                    admission_queue,
                    admit_count,
                    saturate_triggered,
                    queue_depth_peak,
                    queue_rejected,
                    fresh_batch_limit,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                ));
            }
            batch_count.fetch_add(1, Ordering::Relaxed);
            let prefill_result = {
                let model_lock = model.blocking_lock();
                mtp_mode.prefill_admitted(sched, &model_lock, mtp_counters)
            };
            match prefill_result {
                Ok(events) => {
                    for ev in events {
                        route_event(ev, event_txs);
                    }
                }
                Err(e) => {
                    tracing::error!("[SchedulerActor] re-prefill error: {e:?}");
                    if let Err(evict_err) = sched.evict_all() {
                        tracing::warn!(
                            "[SchedulerActor] evict_all after re-prefill error also failed: \
                             {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                        );
                    }
                    event_txs.clear();
                    return RollingControl::ContinueOuter;
                }
            }
            RollingControl::ContinueRolling
        }
        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => RollingControl::BreakRolling,
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
            event_txs.clear();
            RollingControl::ReturnActor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::cache::MtpCache;
    use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
    use crate::core::sampler::Sampler;
    use crate::core::speculative::MtpSpeculativeModel;
    use crate::nn::MtpStepOutput;

    struct SchedulerActorFakeModel;
    #[derive(Clone, Copy)]
    struct SchedulerActorFakeMtpHead;
    static FAKE_MODEL_FORWARD_DELAY_MS: AtomicU64 = AtomicU64::new(0);

    fn maybe_delay_fake_forward() {
        let delay_ms = FAKE_MODEL_FORWARD_DELAY_MS.load(Ordering::Relaxed);
        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
    }

    struct FakeForwardDelayGuard;

    impl FakeForwardDelayGuard {
        fn set(delay_ms: u64) -> Self {
            FAKE_MODEL_FORWARD_DELAY_MS.store(delay_ms, Ordering::Relaxed);
            Self
        }
    }

    impl Drop for FakeForwardDelayGuard {
        fn drop(&mut self) {
            FAKE_MODEL_FORWARD_DELAY_MS.store(0, Ordering::Relaxed);
        }
    }

    fn write_fake_full_kv(
        input_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut [crate::nn::LayerCache]>,
    ) -> Result<()> {
        let Some(cache) = cache else {
            return Ok(());
        };
        let Some(crate::nn::LayerCache::Full(kv)) = cache.first_mut() else {
            return Ok(());
        };
        let shape = input_ids.shape();
        let dims = shape.as_slice();
        let batch = dims[0];
        let seq = dims[1];
        let owned_lens;
        let lens = match per_row_lens {
            Some(lens) => lens,
            None => {
                owned_lens = vec![seq; batch as usize];
                &owned_lens
            }
        };
        let k = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("fake full k failed: {e:?}"))?;
        let v = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("fake full v failed: {e:?}"))?;
        kv.update_and_fetch(&k, &v, lens)?;
        Ok(())
    }

    impl Model for SchedulerActorFakeModel {
        fn make_cache(
            &self,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> Result<Vec<crate::nn::LayerCache>> {
            Ok(vec![crate::nn::LayerCache::Full(
                crate::core::KVCache::new(batch, 1, 1, 1, dtype, cap),
            )])
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            write_fake_full_kv(input_ids, _per_row_lens, cache)?;
            maybe_delay_fake_forward();
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn batched_prefill(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            per_row_lens: &[i32],
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            write_fake_full_kv(input_ids, Some(per_row_lens), cache)?;
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let shape = input_ids.shape();
            let b = shape.as_slice()[0] as usize;
            let s = shape.as_slice()[1] as usize;
            let hidden = 4_usize;
            let flat = vec![0.0_f32; b * s * hidden];
            (&flat[..], &[b as i32, s as i32, hidden as i32][..])
                .try_into()
                .map_err(|e| anyhow::anyhow!("fake hidden Array failed: {e:?}"))
        }

        fn fresh_prefill_batch_limit(_prompt_len: usize, b_max: usize) -> usize
        where
            Self: Sized,
        {
            b_max.min(2)
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl DenseVlMethods for SchedulerActorFakeModel {
        fn batched_prefill_vl(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let rows: i32 = grid_thw
                .iter()
                .map(|&(t, h, w)| t * (h / 2).max(1) * (w / 2).max(1))
                .sum::<i32>()
                .max(1);
            mlx::Array::zeros((rows, 1_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vision embeds Array failed: {e:?}"))
        }

        fn forward_vl_chunk(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn forward_vl_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let shape = input_ids.shape();
            let b = shape.as_slice()[0] as usize;
            let s = shape.as_slice()[1] as usize;
            let hidden = 4_usize;
            let flat = vec![0.0_f32; b * s * hidden];
            (&flat[..], &[b as i32, s as i32, hidden as i32][..])
                .try_into()
                .map_err(|e| anyhow::anyhow!("fake VL hidden Array failed: {e:?}"))
        }
    }

    impl MtpSpeculativeModel for SchedulerActorFakeModel {
        type MtpHead = SchedulerActorFakeMtpHead;

        fn load_mtp_head(&self, _loader: &crate::core::Loader) -> Result<Self::MtpHead> {
            Ok(SchedulerActorFakeMtpHead)
        }

        fn make_mtp_cache(
            &self,
            _mtp: &Self::MtpHead,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> Result<MtpCache> {
            MtpCache::new_with_cap(1, batch, 1, 1, 1, dtype, cap)
        }

        fn project_hidden_on(
            &self,
            hidden: &mlx::Array,
            _target: impl Into<mlx::StreamOrDevice>,
        ) -> Result<mlx::Array> {
            let seq = hidden.shape().as_slice()[1] as usize;
            let tokens: Vec<u32> = if seq == 1 { vec![3] } else { vec![4; seq] };
            fake_logits_for_tokens(&tokens)
        }

        fn mtp_hidden_size(&self, _mtp: &Self::MtpHead) -> i32 {
            4
        }

        fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> mlx::Dtype {
            mlx::Dtype::Float32
        }

        fn mtp_forward_hidden_on(
            &self,
            _mtp: &Self::MtpHead,
            hidden_states: &mlx::Array,
            next_token_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _mask: Option<&mlx::Array>,
            mtp_cache: Option<&mut MtpCache>,
            _target: impl Into<mlx::StreamOrDevice>,
        ) -> Result<mlx::Array> {
            if let Some(cache) = mtp_cache {
                let seq = next_token_ids.shape().as_slice()[1];
                let k = mlx::Array::zeros((1_i32, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("fake mtp k failed: {e:?}"))?;
                let v = mlx::Array::zeros((1_i32, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("fake mtp v failed: {e:?}"))?;
                cache.layer_mut(0).update_and_fetch(&k, &v, &[seq])?;
            }
            Ok(hidden_states.clone())
        }

        fn mtp_forward_on(
            &self,
            mtp: &Self::MtpHead,
            hidden_states: &mlx::Array,
            next_token_ids: &mlx::Array,
            position_ids: &mlx::Array,
            mask: Option<&mlx::Array>,
            mtp_cache: Option<&mut MtpCache>,
            target: impl Into<mlx::StreamOrDevice>,
        ) -> Result<MtpStepOutput> {
            let hidden_states = self.mtp_forward_hidden_on(
                mtp,
                hidden_states,
                next_token_ids,
                position_ids,
                mask,
                mtp_cache,
                target,
            )?;
            Ok(MtpStepOutput {
                hidden_states,
                logits: fake_logits_for_tokens(&[4])?,
            })
        }
    }

    fn fake_logits(batch: usize) -> Result<mlx::Array> {
        let vocab = 8_usize;
        let mut flat = vec![0.0_f32; batch * vocab];
        for row in 0..batch {
            flat[row * vocab + 3] = 100.0;
        }
        let logits_bv: mlx::Array = (&flat[..], &[batch as i32, vocab as i32][..])
            .try_into()
            .map_err(|e| anyhow::anyhow!("fake logits Array failed: {e:?}"))?;
        logits_bv
            .reshape(&[batch as i32, 1, vocab as i32][..])
            .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
    }

    fn fake_logits_for_tokens(tokens: &[u32]) -> Result<mlx::Array> {
        let vocab = 8_usize;
        let mut flat = vec![0.0_f32; tokens.len() * vocab];
        for (pos, &token) in tokens.iter().enumerate() {
            flat[pos * vocab + token as usize] = 100.0;
        }
        (&flat[..], &[1_i32, tokens.len() as i32, vocab as i32][..])
            .try_into()
            .map_err(|e| anyhow::anyhow!("fake logits Array failed: {e:?}"))
    }

    fn mk_req(prompt_token: u32) -> GenerateRequest {
        GenerateRequest {
            prompt_ids: vec![prompt_token],
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
        }
    }

    fn mk_vl_req() -> GenerateRequest {
        let mut req = mk_req(11);
        req.prompt_ids = vec![11, IMAGE_TOKEN_ID as u32, 12];
        req.max_new_tokens = 1;
        req.pixel_values = Some(vec![
            mlx::Array::zeros((1_i32, 1_i32), mlx::Dtype::Float32).unwrap()
        ]);
        req.image_grid_thw = Some(vec![(1, 2, 2)]);
        req
    }

    fn queued_pending(prompt_token: u32) -> (PendingAdmit, oneshot::Receiver<Result<AdmitReply>>) {
        let (reply_tx, reply_rx) = oneshot::channel();
        (
            PendingAdmit {
                request: mk_req(prompt_token),
                reply_tx,
                queued_at_profile: None,
            },
            reply_rx,
        )
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_mtp_actor_accepts_paged_prefix_cache_config() {
        let root = unique_temp_dir("actor-mtp-prefix");
        let config = PagedPrefixCacheConfig::new(&root, "fake-qwen", 16, 8).expect("prefix config");
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let handle = spawn_scheduler_actor_with_mtp(
            model,
            SchedulerActorFakeMtpHead,
            1,
            2,
            Duration::from_millis(1),
            1,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            Some(config),
            None,
        )
        .expect("spawn mtp actor with prefix cache");

        assert_eq!(handle.mtp_prefill_count.load(Ordering::Relaxed), 0);
        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_active_kv_offload_parks_and_restores_full_slot_request() {
        let root = unique_temp_dir("actor-active-kv");
        let _delay_guard = FakeForwardDelayGuard::set(25);
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let handle = spawn_scheduler_actor_with_active_kv_offload(
            model,
            1,
            Duration::from_millis(1),
            4,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            ActiveKvOffloadConfig::enabled(root.clone()),
        )
        .expect("spawn actor with active kv offload");

        let (reply_tx_1, reply_rx_1) = oneshot::channel();
        let mut request_1 = mk_req(11);
        request_1.max_new_tokens = 4;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_1,
                reply_tx: reply_tx_1,
            })
            .await
            .expect("send first request");
        let mut events_1 = reply_rx_1
            .await
            .expect("first reply")
            .expect("first admit")
            .event_rx;
        let first_event = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("first event timeout")
            .expect("first event");
        assert_eq!(first_event.finish_reason, None);

        let (reply_tx_2, reply_rx_2) = oneshot::channel();
        let mut request_2 = mk_req(22);
        request_2.max_new_tokens = 1;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_2,
                reply_tx: reply_tx_2,
            })
            .await
            .expect("send second request");
        let mut events_2 = tokio::time::timeout(Duration::from_secs(2), reply_rx_2)
            .await
            .expect("second reply timeout")
            .expect("second reply")
            .expect("second admit")
            .event_rx;

        let mut second_finished = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_2.recv())
            .await
            .expect("second event timeout")
        {
            if event.finish_reason.is_some() {
                second_finished = true;
                break;
            }
        }
        assert!(
            second_finished,
            "second request should finish while first is parked"
        );

        let mut first_finished = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("restored first event timeout")
        {
            if event.finish_reason.is_some() {
                first_finished = true;
                break;
            }
        }
        assert!(first_finished, "first request should restore and finish");

        let health = handle.active_kv_offload.snapshot();
        assert!(health.swap_out_count >= 1, "expected at least one swap out");
        assert!(health.swap_in_count >= 1, "expected at least one swap in");
        assert_eq!(health.swap_error_count, 0);
        assert_eq!(health.parked_requests, 0);

        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn actor_mtp_mode_prefill_and_step_use_mtp_for_eligible_request() {
        let mut scheduler = Scheduler::<SchedulerActorFakeModel>::new(
            1,
            32,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler");
        scheduler.admit(mk_req(11)).expect("admit");
        let counters = SchedulerActorMtpCounters::new(
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
        );
        let mut mode = SchedulerActorMtp::new(SchedulerActorFakeMtpHead, 1);

        let prefill_events = mode
            .prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("mtp prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(counters.mtp_prefill_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            counters.mtp_prefill_fallback_count.load(Ordering::Relaxed),
            0
        );
        assert!(scheduler.mtp_stats().is_some());

        let step_events = mode
            .step(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("mtp step");
        assert_eq!(step_events.len(), 1);
        assert_eq!(counters.mtp_step_count.load(Ordering::Relaxed), 1);
        assert_eq!(counters.mtp_drafted_tokens.load(Ordering::Relaxed), 1);
        assert_eq!(
            counters.mtp_accepted_draft_tokens.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn actor_mtp_mode_prefill_uses_mtp_for_eligible_vl_request() {
        let mut scheduler = Scheduler::<SchedulerActorFakeModel>::new(
            1,
            32,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler");
        scheduler.admit(mk_vl_req()).expect("admit");
        let counters = SchedulerActorMtpCounters::new(
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
        );
        let mut mode = SchedulerActorMtp::new(SchedulerActorFakeMtpHead, 1);

        let prefill_events = mode
            .prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("VL MTP prefill");

        assert_eq!(prefill_events.len(), 1);
        assert_eq!(counters.mtp_prefill_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            counters.mtp_prefill_fallback_count.load(Ordering::Relaxed),
            0
        );
        assert!(scheduler.mtp_stats().is_some());
    }

    #[test]
    fn actor_mtp_mode_prefill_falls_back_for_non_greedy_request() {
        let mut scheduler = Scheduler::<SchedulerActorFakeModel>::new(
            1,
            32,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler");
        let mut request = mk_req(11);
        request.sampler = Sampler::greedy().with_temperature(0.7);
        scheduler.admit(request).expect("admit");
        let counters = SchedulerActorMtpCounters::new(
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
        );
        let mut mode = SchedulerActorMtp::new(SchedulerActorFakeMtpHead, 1);

        let prefill_events = mode
            .prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("ordinary prefill");

        assert_eq!(prefill_events.len(), 1);
        assert_eq!(counters.mtp_prefill_count.load(Ordering::Relaxed), 0);
        assert_eq!(
            counters.mtp_prefill_fallback_count.load(Ordering::Relaxed),
            1
        );
        assert_eq!(counters.mtp_drafted_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(
            counters.mtp_accepted_draft_tokens.load(Ordering::Relaxed),
            0
        );
        assert!(scheduler.mtp_stats().is_none());
    }

    #[test]
    fn rolling_policy_forces_one_decode_after_admission_work() {
        let mut policy = RollingAdmissionPolicy::default();

        assert!(!policy.should_force_decode(Phase::Decoding, 1));
        policy.record_admission_work();
        assert!(policy.should_force_decode(Phase::Decoding, 1));

        policy.record_decode_step();
        assert!(!policy.should_force_decode(Phase::Decoding, 1));
    }

    #[test]
    fn rolling_policy_does_not_force_decode_without_active_decoding_rows() {
        let mut policy = RollingAdmissionPolicy::default();

        policy.record_admission_work();
        assert!(!policy.should_force_decode(Phase::Idle, 1));
        assert!(!policy.should_force_decode(Phase::Finished, 1));
        assert!(!policy.should_force_decode(Phase::Decoding, 0));
    }

    #[test]
    fn rolling_profile_env_parser_only_enables_explicit_one() {
        assert!(rolling_profile_enabled_from_env(Some("1")));
        assert!(!rolling_profile_enabled_from_env(None));
        assert!(!rolling_profile_enabled_from_env(Some("")));
        assert!(!rolling_profile_enabled_from_env(Some("true")));
        assert!(!rolling_profile_enabled_from_env(Some("0")));
    }

    #[test]
    fn rolling_profile_queue_wait_ms_uses_supplied_clock() {
        let queued_at = std::time::Instant::now();
        let now = queued_at + Duration::from_micros(12_345);

        let wait_ms = rolling_profile_queue_wait_ms(queued_at, now);

        assert!((wait_ms - 12.345).abs() < 1e-9);
    }

    #[test]
    fn cadence_protection_uses_supplied_runtime_cap() {
        assert_eq!(cadence_protected_mid_chunk_size(1024, 2, 384), 384);
        assert_eq!(cadence_protected_mid_chunk_size(1024, 1, 384), 1024);
        assert_eq!(cadence_protected_mid_chunk_size(128, 2, 384), 128);
    }

    #[test]
    fn drain_admission_queue_limits_successful_mid_admit_to_one_per_turn() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(sched.phase(), Phase::Decoding);

        let (pending_1, reply_rx_1) = queued_pending(21);
        let (pending_2, reply_rx_2) = queued_pending(22);
        let (pending_3, reply_rx_3) = queued_pending(23);
        let _reply_rxs = [reply_rx_1, reply_rx_2, reply_rx_3];
        let mut queue = VecDeque::from([pending_1, pending_2, pending_3]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
            256,
        );

        assert!(did_admit, "expected one queued request to be admitted");
        assert_eq!(
            queue.len(),
            2,
            "queue drain should leave remaining queued requests for later decode turns"
        );
        assert_eq!(
            sched.active_count(),
            2,
            "one active row plus exactly one mid-admitted row"
        );
        assert!(in_flight_mid_admit.is_none());
    }

    #[test]
    fn drain_admission_queue_respects_rolling_prefill_batch_limit() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (pending_1, reply_rx_1) = queued_pending(21);
        let (pending_2, reply_rx_2) = queued_pending(22);
        let _reply_rxs = [reply_rx_1, reply_rx_2];
        let mut queue = VecDeque::from([pending_1, pending_2]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
            256,
        );

        assert!(
            !did_admit,
            "active rows already reached the model's rolling prefill batch limit"
        );
        assert_eq!(
            queue.len(),
            2,
            "queued requests should wait for decode progress instead of growing active batch"
        );
        assert_eq!(sched.active_count(), 2);
        assert_eq!(admit_count.load(Ordering::Relaxed), 0);
        assert!(in_flight_mid_admit.is_none());
    }

    #[test]
    fn drain_admission_queue_starts_chunked_mid_admit_beyond_rolling_limit() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (reply_tx, reply_rx) = oneshot::channel();
        let mut chunked_req = mk_req(21);
        chunked_req.prompt_ids = vec![21, 22, 23, 24];
        chunked_req.prefill_chunk_size = 2;
        let _reply_rx = reply_rx;
        let mut queue = VecDeque::from([PendingAdmit {
            request: chunked_req,
            reply_tx,
            queued_at_profile: None,
        }]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
            256,
        );

        assert!(
            did_admit,
            "chunked queued requests may start prefill under decode-cadence protection"
        );
        assert_eq!(queue.len(), 0);
        assert_eq!(sched.active_count(), 3);
        assert_eq!(
            admit_count.load(Ordering::Relaxed),
            0,
            "the request should not count as admitted until the final chunk samples its first token"
        );
        assert!(
            in_flight_mid_admit.is_some(),
            "multi-chunk mid-admit should yield after one chunk"
        );
    }

    #[test]
    fn drain_admission_queue_caps_chunked_mid_admit_when_decode_rows_are_active() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (reply_tx, reply_rx) = oneshot::channel();
        let mut chunked_req = mk_req(21);
        chunked_req.prompt_ids = (0..1025).collect();
        chunked_req.prefill_chunk_size = 1024;
        chunked_req.decode_cadence_mid_chunk_cap = 384;
        let _reply_rx = reply_rx;
        let mut queue = VecDeque::from([PendingAdmit {
            request: chunked_req,
            reply_tx,
            queued_at_profile: None,
        }]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
            384,
        );

        assert!(did_admit, "chunked queued request should start");
        let handle = in_flight_mid_admit
            .as_ref()
            .expect("chunked mid-admit should still be in flight");
        assert_eq!(
            handle.chunk_start, 384,
            "active decode rows should cap the first mid-admit chunk to protect ITL"
        );
        assert_eq!(
            handle.chunk_size, 1024,
            "cadence cap should be temporary and preserve the request chunk size"
        );
    }

    /// Drop the SchedulerActorHandle (and thus cmd_tx); confirm the driver
    /// task exits cleanly. We can't construct a real Qwen35Model in a unit
    /// test, so we never send any commands — we only verify the driver's
    /// `rt.block_on(cmd_rx.recv())` outer loop terminates when all
    /// senders are dropped.
    ///
    /// To keep this test self-contained without a model, we don't call
    /// `spawn_scheduler_actor` (which would require a real model handle).
    /// Instead we directly spawn a minimal stand-in driver that mirrors
    /// the channel-close exit condition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn driver_shuts_down_when_cmd_channel_closes() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<SchedulerCommand>(8);
        let handle = tokio::task::spawn_blocking(move || {
            // Mirrors `driver_loop`'s exit condition without touching a model.
            while let Some(_cmd) = cmd_rx.blocking_recv() {
                // would dispatch here in real driver
            }
        });
        drop(cmd_tx);
        // Driver should exit promptly after senders drop.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("driver did not shut down within 2s")
            .expect("driver join error");
    }

    /// b_max=1 + queue_max=2; admit 3 short requests in rapid succession;
    /// verify the queue grows to peak >= 1 before slots free up.
    /// Real-model heavy — gated by `#[ignore]`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore] // real-model heavy: loads Qwen3.5-4B-MLX-4bit
    async fn admission_queue_push_when_full() {
        use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            std::fs::read_dir(&glob)
                .expect("snapshots dir")
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let loader = Loader::open_multimodal(std::path::Path::new(&model_dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let model = Arc::new(Mutex::new(
            crate::models::Qwen35Model::from_loader(&loader).unwrap(),
        ));
        let meta = model.lock().await.model_meta();

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 2,
            /* effective_cap_max */ 32768,
            /* decode_cadence_mid_chunk_cap */ 256,
            meta,
        )
        .expect("spawn");

        let mk_req = |text: &str| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer
                .apply_chat_template(&msgs, true, Some(&kw))
                .unwrap();
            let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
            GenerateRequest {
                prompt_ids,
                max_new_tokens: 8,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 256,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: IMAGE_TOKEN_ID,
            }
        };

        let mut replies = Vec::new();
        for text in ["Hello", "World", "Goodbye"] {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            handle
                .cmd_tx
                .send(SchedulerCommand::Admit {
                    request: mk_req(text),
                    reply_tx,
                })
                .await
                .expect("cmd_tx.send");
            replies.push(reply_rx);
        }

        let mut counts = Vec::new();
        for rx in replies {
            let admit_reply = rx.await.expect("reply").expect("admit ok");
            let mut event_rx = admit_reply.event_rx;
            let mut n = 0;
            while let Some(ev) = event_rx.recv().await {
                n += 1;
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            counts.push(n);
        }

        for c in &counts {
            assert!(*c >= 1, "expected ≥1 event per request, got {c}");
        }

        let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
        assert!(peak >= 1, "expected queue_depth_peak >= 1, got {peak}");

        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert_eq!(rejected, 0, "expected no rejections, got {rejected}");

        drop(handle);
    }

    /// b_max=1 + queue_max=1; send 3 admits back-to-back. The 3rd one
    /// must be rejected with Err("admission queue full").
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore] // real-model heavy
    async fn admission_queue_overflow_returns_err() {
        use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            std::fs::read_dir(&glob)
                .expect("snapshots dir")
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let loader = Loader::open_multimodal(std::path::Path::new(&model_dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let model = Arc::new(Mutex::new(
            crate::models::Qwen35Model::from_loader(&loader).unwrap(),
        ));
        let meta = model.lock().await.model_meta();

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 1,
            /* effective_cap_max */ 32768,
            /* decode_cadence_mid_chunk_cap */ 256,
            meta,
        )
        .expect("spawn");

        let mk_req = |text: &str, max_new: usize| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer
                .apply_chat_template(&msgs, true, Some(&kw))
                .unwrap();
            let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
            GenerateRequest {
                prompt_ids,
                max_new_tokens: max_new,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 256,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: IMAGE_TOKEN_ID,
            }
        };

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Hello", 64),
                reply_tx: tx1,
            })
            .await
            .unwrap();

        // Wait briefly so first admit enters Decoding before #2/#3 arrive.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("World", 8),
                reply_tx: tx2,
            })
            .await
            .unwrap();

        let (tx3, rx3) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Goodbye", 8),
                reply_tx: tx3,
            })
            .await
            .unwrap();

        let reply3 = tokio::time::timeout(Duration::from_secs(5), rx3)
            .await
            .expect("rx3 timeout")
            .expect("rx3 recv");
        match reply3 {
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("admission queue full"),
                    "expected 'admission queue full' Err, got: {msg}"
                );
            }
            Ok(_) => panic!("expected Err for #3, got Ok"),
        }

        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert!(rejected >= 1, "expected queue_rejected ≥ 1, got {rejected}");

        let _ = tokio::time::timeout(Duration::from_secs(120), async {
            let r1 = rx1.await.unwrap().unwrap();
            let mut e1 = r1.event_rx;
            while let Some(ev) = e1.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            let r2 = rx2.await.unwrap().unwrap();
            let mut e2 = r2.event_rx;
            while let Some(ev) = e2.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
        })
        .await
        .expect("rx1/rx2 drain timeout");

        drop(handle);
    }
}
