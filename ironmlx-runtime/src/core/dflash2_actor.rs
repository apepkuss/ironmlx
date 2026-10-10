//! Isolated multi-sequence DFlash2 execution actor.
//!
//! The actor intentionally does not construct or drive [`crate::core::Scheduler`].
//! It owns the DFlash2 draft model and drives request-local DFlash2 streams on
//! one blocking worker so cache/PRNG state remains isolated and MLX stream
//! affinity is stable. Compatible rows execute in persistent B=N tensor groups;
//! unmatched or divergent rows retain the qualified B1 path. `b_max` controls
//! the number of concurrently active sequences.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Context;
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::dflash2::{
    DFlash2PrefixCache, DFlash2RaggedBatchCache, DFlash2TensorBatchCache,
    DFlash2TextGenerationStream,
};
use crate::core::dflash2_step_diagnostic::DFlash2StepDiagnostic;
use crate::core::generation_types::{GenerateEvent, GenerateRequest, RequestPriority};
use crate::core::memory_budget::BudgetState;
use crate::core::runtime_health::DFlash2RaggedLinearCounters;
use crate::core::scheduler::{RequestId, SchedulerError, StepEvent};
use crate::core::scheduler_actor::AdmitReply;
use crate::Result;
use ironmlx_lm::core::vision::DenseVlMethods;
use {ironmlx_lm::core::model::Model, ironmlx_lm::core::tokenizer::Tokenizer};
use {
    ironmlx_lm::models::dflash2::DFlash2DraftModel, ironmlx_lm::models::dflash2::DFlash2Target,
    ironmlx_lm::models::dflash2::DFlash2TargetCacheCost,
};

struct ActiveDFlash2Request<'m, M>
where
    M: DFlash2Target,
{
    request_id: RequestId,
    priority: RequestPriority,
    event_tx: mpsc::UnboundedSender<StepEvent>,
    stream: DFlash2TextGenerationStream<'m, M>,
    _memory_charge: DFlash2MemoryCharge,
}

#[derive(Default)]
struct DFlash2StepOutcome {
    events: Vec<GenerateEvent>,
    cancelled: bool,
    finished: bool,
    failure: Option<String>,
}

struct DFlash2TensorGroup {
    request_ids: Vec<RequestId>,
    cache: DFlash2TensorBatchCache,
}

fn tensor_group_positions<'m, M>(
    active: &[ActiveDFlash2Request<'m, M>],
    request_ids: &[RequestId],
) -> Option<Vec<usize>>
where
    M: DFlash2Target,
{
    request_ids
        .iter()
        .map(|request_id| {
            active
                .iter()
                .position(|request| request.request_id == *request_id)
        })
        .collect()
}

fn tensor_group_streams_mut<'a, 'm, M>(
    active: &'a mut [ActiveDFlash2Request<'m, M>],
    indices: &[usize],
) -> Result<Vec<&'a mut DFlash2TextGenerationStream<'m, M>>>
where
    M: DFlash2Target,
{
    let mut streams = Vec::with_capacity(indices.len());
    let mut tail = active;
    let mut base = 0_usize;
    for &index in indices {
        anyhow::ensure!(
            index >= base && index < base + tail.len(),
            "DFlash2 tensor group indices must be strictly increasing and in range"
        );
        let offset = index - base;
        let (_, from_row) = tail.split_at_mut(offset);
        let (row, remaining) = from_row
            .split_first_mut()
            .ok_or_else(|| anyhow::anyhow!("DFlash2 tensor group row is missing"))?;
        streams.push(&mut row.stream);
        tail = remaining;
        base = index + 1;
    }
    Ok(streams)
}

/// Tree/linear switch setting (on with the M5 profile).
const RAGGED_LINEAR_ENV: &str = ironmlx_core::m5_profile::settings::DFLASH2_RAGGED_LINEAR;
/// Ragged linear windows use verify width `draft_len + 1` for 2..=4 rows.
const RAGGED_LINEAR_MAX_WIDTH: usize = 4;
/// Diagnostic only: an extra limit (bytes) on the KV budget charge allowed
/// when reserving a ragged group cache, to exercise the budget fallback.
const RAGGED_BUDGET_LIMIT_ENV: &str = "IRONMLX_DIAGNOSTIC_DFLASH2_RAGGED_BUDGET_LIMIT_BYTES";

/// Rows sharing one batched target cache for ragged linear windows. Each
/// row's own target cache is stale while the group exists; the group cache
/// is scattered back before any member runs another window path or leaves.
struct DFlash2RaggedGroup {
    request_ids: Vec<RequestId>,
    cache: DFlash2RaggedBatchCache,
    /// KV budget charge of the batched cache; released when the group is
    /// dropped (scatter on membership change, finish, cancel, pause or
    /// failure).
    memory_charge: DFlash2MemoryCharge,
}

/// Charge the KV budget for a ragged group cache: `rows` sequences of
/// `cap_tokens` each, the same cost model as an admission. `limit` is the
/// diagnostic extra limit on the resulting active charge.
/// Process-memory headroom for a ragged group cache, mirroring admission:
/// refresh the governor sample first (decode does not sample, so a reserve
/// on stale telemetry would fail closed and force soft pressure for later
/// admissions), fall back without reserving unless pressure is normal, then
/// reserve. The caller commits the reservation once the cache exists.
fn reserve_ragged_group_headroom(
    governor: &crate::core::process_memory::SharedProcessMemoryGovernor,
    bytes: usize,
    refresh: impl FnOnce(
        &crate::core::process_memory::ProcessMemoryGovernor,
    ) -> crate::core::process_memory::MemoryGovernorSnapshot,
) -> Result<crate::core::process_memory::MemoryReservation> {
    let snapshot = refresh(governor);
    anyhow::ensure!(
        snapshot.pressure_level == crate::core::process_memory::PressureLevel::Normal
            && !snapshot.telemetry_degraded,
        "process memory pressure {:?} (telemetry degraded: {})",
        snapshot.pressure_level,
        snapshot.telemetry_degraded
    );
    governor
        .try_reserve(bytes, "dflash2_ragged_group")
        .map_err(|error| anyhow::anyhow!("{error}"))
}

fn reserve_ragged_group_memory(
    budget_state: &BudgetState,
    cache_cost: DFlash2TargetCacheCost,
    cap_tokens: usize,
    rows: usize,
    limit: Option<usize>,
) -> Result<DFlash2MemoryCharge> {
    let requested_bytes = cache_cost.request_bytes(cap_tokens).saturating_mul(rows);
    if let Some(limit) = limit {
        let active_bytes = budget_state.active_bytes();
        if active_bytes.saturating_add(requested_bytes) > limit {
            return Err(anyhow::Error::new(SchedulerError::MemoryBudgetExceeded {
                active_bytes,
                requested_bytes,
                soft_limit_bytes: limit,
            }));
        }
    }
    budget_state
        .try_admit_with_allowance(requested_bytes, 0)
        .map_err(|(active_bytes, requested_bytes, soft_limit_bytes)| {
            anyhow::Error::new(SchedulerError::MemoryBudgetExceeded {
                active_bytes,
                requested_bytes,
                soft_limit_bytes,
            })
        })?;
    Ok(DFlash2MemoryCharge {
        budget_state: budget_state.clone(),
        bytes: requested_bytes,
    })
}

/// Rows for one ragged linear window. `candidates` are `(active index,
/// draft_len, supported batch-width mask)` of rows at a window boundary that
/// may batch; members of the current group come first so an unchanged
/// membership keeps its batched cache. Returns the chosen active indices in
/// increasing order, or nothing when fewer than two rows can batch.
fn select_ragged_linear_rows(
    candidates: &[(usize, usize, u64)],
    current_group: &[usize],
    max_width: usize,
) -> Vec<usize> {
    let mut ordered = candidates
        .iter()
        .filter(|candidate| current_group.contains(&candidate.0))
        .chain(
            candidates
                .iter()
                .filter(|candidate| !current_group.contains(&candidate.0)),
        )
        .copied()
        .collect::<Vec<_>>();
    let Some(&(_, draft_len, _)) = ordered.first() else {
        return Vec::new();
    };
    ordered.retain(|candidate| candidate.1 == draft_len);
    let mut width = max_width.min(ordered.len());
    while width >= 2 {
        let supported = width < u64::BITS as usize
            && ordered[..width]
                .iter()
                .all(|candidate| candidate.2 & (1_u64 << width) != 0);
        if supported {
            break;
        }
        width -= 1;
    }
    if width < 2 {
        return Vec::new();
    }
    let mut chosen = ordered[..width]
        .iter()
        .map(|candidate| candidate.0)
        .collect::<Vec<_>>();
    chosen.sort_unstable();
    chosen
}

/// Active indices of the ragged group's rows that are still present (used
/// to fail them if the group cannot be scattered back).
fn ragged_member_positions<'m, M>(
    active: &[ActiveDFlash2Request<'m, M>],
    group: Option<&DFlash2RaggedGroup>,
) -> Vec<usize>
where
    M: DFlash2Target,
{
    group
        .map(|group| {
            group
                .request_ids
                .iter()
                .filter_map(|request_id| {
                    active
                        .iter()
                        .position(|request| request.request_id == *request_id)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn scatter_ragged_group<'m, M>(
    active: &mut [ActiveDFlash2Request<'m, M>],
    group: &mut Option<DFlash2RaggedGroup>,
    counters: &DFlash2RaggedLinearCounters,
) -> Result<()>
where
    M: DFlash2Target,
{
    let Some(group) = group.take() else {
        return Ok(());
    };
    counters.active_groups.store(0, Ordering::Relaxed);
    counters.reserved_bytes.store(0, Ordering::Relaxed);
    counters.groups_scattered.fetch_add(1, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let positions = tensor_group_positions(active, &group.request_ids)
        .ok_or_else(|| anyhow::anyhow!("DFlash2 ragged group lost a row"))?;
    let mut streams = tensor_group_streams_mut(active, &positions)?;
    group.cache.scatter_to_rows(&mut streams)?;
    drop(group.memory_charge);
    counters.scatter_us.fetch_add(
        u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    Ok(())
}

fn scatter_all_tensor_groups<'m, M>(
    active: &mut [ActiveDFlash2Request<'m, M>],
    tensor_groups: &mut Vec<DFlash2TensorGroup>,
) -> Result<()>
where
    M: DFlash2Target,
{
    while let Some(group) = tensor_groups.pop() {
        let positions = tensor_group_positions(active, &group.request_ids)
            .ok_or_else(|| anyhow::anyhow!("DFlash2 priority pause lost a tensor-group row"))?;
        let mut streams = tensor_group_streams_mut(active, &positions)?;
        group.cache.scatter_to_rows(&mut streams)?;
    }
    Ok(())
}

#[derive(Debug)]
struct DFlash2MemoryCharge {
    budget_state: BudgetState,
    bytes: usize,
}

impl DFlash2MemoryCharge {
    fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DFlash2MemoryCharge {
    fn drop(&mut self) {
        self.budget_state.release(self.bytes);
    }
}

fn reserve_dflash2_request_memory(
    budget_state: &BudgetState,
    cache_cost: DFlash2TargetCacheCost,
    token_cap: usize,
    background_priority_allowance: usize,
    memory_budget_exceeded_count: &AtomicU64,
) -> Result<DFlash2MemoryCharge> {
    let requested_bytes = cache_cost.request_bytes(token_cap);
    if let Err((active_bytes, requested_bytes, soft_limit_bytes)) =
        budget_state.try_admit_with_allowance(requested_bytes, background_priority_allowance)
    {
        memory_budget_exceeded_count.fetch_add(1, Ordering::Relaxed);
        return Err(anyhow::Error::new(SchedulerError::MemoryBudgetExceeded {
            active_bytes,
            requested_bytes,
            soft_limit_bytes,
        }));
    }
    Ok(DFlash2MemoryCharge {
        budget_state: budget_state.clone(),
        bytes: requested_bytes,
    })
}

fn discard_abandoned_queued_request(
    reply_tx: &oneshot::Sender<Result<AdmitReply>>,
    in_flight: &AtomicUsize,
) -> bool {
    if !reply_tx.is_closed() {
        return false;
    }
    in_flight.fetch_sub(1, Ordering::Release);
    true
}

#[derive(Clone)]
struct DFlash2ActorCounters {
    windows: Arc<AtomicU64>,
    drafted_tokens: Arc<AtomicU64>,
    accepted_draft_tokens: Arc<AtomicU64>,
    rollback_count: Arc<AtomicU64>,
    ordinary_windows: Arc<AtomicU64>,
    tree_windows: Arc<AtomicU64>,
    tree_drafted_nodes: Arc<AtomicU64>,
    tree_fallback_linear_windows: Arc<AtomicU64>,
    draft_budget_changes: Arc<AtomicU64>,
    current_draft_budget: Arc<AtomicUsize>,
    latest_adaptive_acceptance_ewma_bits: Arc<AtomicU64>,
    sampled_requests: Arc<AtomicU64>,
    exact_sampling_windows: Arc<AtomicU64>,
    exact_acceptance_draws: Arc<AtomicU64>,
    exact_residual_corrections: Arc<AtomicU64>,
    exact_bonus_samples: Arc<AtomicU64>,
    sampling_us: Arc<AtomicU64>,
    draft_build_us: Arc<AtomicU64>,
    draft_schedule_us: Arc<AtomicU64>,
    verify_build_us: Arc<AtomicU64>,
    projection_build_us: Arc<AtomicU64>,
    verify_schedule_us: Arc<AtomicU64>,
    host_sync_us: Arc<AtomicU64>,
    rollback_us: Arc<AtomicU64>,
    window_us: Arc<AtomicU64>,
    prefill_us: Arc<AtomicU64>,
    generation_us: Arc<AtomicU64>,
    latest_generation_tps_bits: Arc<AtomicU64>,
    latest_acceptance_rate_bits: Arc<AtomicU64>,
    peak_memory_bytes: Arc<AtomicUsize>,
}

impl DFlash2ActorCounters {
    fn record(&self, metrics: &crate::core::dflash2::DFlash2Metrics) {
        self.windows
            .fetch_add(metrics.windows as u64, Ordering::Relaxed);
        self.drafted_tokens
            .fetch_add(metrics.drafted_tokens as u64, Ordering::Relaxed);
        self.accepted_draft_tokens
            .fetch_add(metrics.accepted_draft_tokens as u64, Ordering::Relaxed);
        self.rollback_count
            .fetch_add(metrics.rollback_count as u64, Ordering::Relaxed);
        self.ordinary_windows
            .fetch_add(metrics.ordinary_windows as u64, Ordering::Relaxed);
        self.tree_windows
            .fetch_add(metrics.tree_windows as u64, Ordering::Relaxed);
        self.tree_drafted_nodes
            .fetch_add(metrics.tree_drafted_nodes as u64, Ordering::Relaxed);
        self.tree_fallback_linear_windows.fetch_add(
            metrics.tree_fallback_linear_windows as u64,
            Ordering::Relaxed,
        );
        self.draft_budget_changes
            .fetch_add(metrics.draft_budget_changes as u64, Ordering::Relaxed);
        self.current_draft_budget
            .store(metrics.current_draft_budget, Ordering::Relaxed);
        self.latest_adaptive_acceptance_ewma_bits.store(
            metrics.adaptive_acceptance_ewma.unwrap_or(0.0).to_bits(),
            Ordering::Relaxed,
        );
        if metrics.sampled {
            self.sampled_requests.fetch_add(1, Ordering::Relaxed);
        }
        self.exact_sampling_windows
            .fetch_add(metrics.exact_sampling_windows as u64, Ordering::Relaxed);
        self.exact_acceptance_draws
            .fetch_add(metrics.exact_acceptance_draws as u64, Ordering::Relaxed);
        self.exact_residual_corrections
            .fetch_add(metrics.exact_residual_corrections as u64, Ordering::Relaxed);
        self.exact_bonus_samples
            .fetch_add(metrics.exact_bonus_samples as u64, Ordering::Relaxed);
        self.sampling_us
            .fetch_add(metrics.sampling_us, Ordering::Relaxed);
        for (counter, value) in [
            (&self.draft_build_us, metrics.draft_build_us),
            (&self.draft_schedule_us, metrics.draft_schedule_us),
            (&self.verify_build_us, metrics.verify_build_us),
            (&self.projection_build_us, metrics.projection_build_us),
            (&self.verify_schedule_us, metrics.verify_schedule_us),
            (&self.host_sync_us, metrics.host_sync_us),
            (&self.rollback_us, metrics.rollback_us),
            (&self.window_us, metrics.window_us),
            (&self.prefill_us, metrics.prefill_us),
            (&self.generation_us, metrics.generation_us),
        ] {
            counter.fetch_add(value, Ordering::Relaxed);
        }
        self.latest_generation_tps_bits
            .store(metrics.generation_tps.to_bits(), Ordering::Relaxed);
        self.latest_acceptance_rate_bits
            .store(metrics.acceptance_rate.to_bits(), Ordering::Relaxed);
        self.peak_memory_bytes
            .fetch_max(metrics.peak_memory_bytes, Ordering::Relaxed);
    }
}

pub(super) enum DFlash2Command {
    Admit {
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    },
}

impl DFlash2Command {
    fn reply_is_closed(&self) -> bool {
        match self {
            Self::Admit { reply_tx, .. } => reply_tx.is_closed(),
        }
    }

    fn priority(&self) -> RequestPriority {
        match self {
            Self::Admit { request, .. } => request.priority,
        }
    }
}

fn push_pending_by_priority(pending: &mut VecDeque<DFlash2Command>, command: DFlash2Command) {
    if command.priority().is_background() {
        pending.push_back(command);
    } else {
        let foreground_end = pending
            .iter()
            .position(|queued| queued.priority().is_background())
            .unwrap_or(pending.len());
        pending.insert(foreground_end, command);
    }
}

fn prune_abandoned_pending_requests(
    pending: &mut VecDeque<DFlash2Command>,
    in_flight: &AtomicUsize,
    b_queued: &AtomicU64,
) -> usize {
    let mut removed = 0_usize;
    pending.retain(|command| {
        if !command.reply_is_closed() {
            return true;
        }
        in_flight.fetch_sub(1, Ordering::Release);
        b_queued.fetch_sub(1, Ordering::Relaxed);
        removed += 1;
        false
    });
    removed
}

pub(crate) struct DFlash2ActorConfig {
    pub(crate) block_size: usize,
    pub(crate) p2_options: crate::core::dflash2::DFlash2P2Options,
    pub(crate) b_max: usize,
    pub(crate) admission_deadline: std::time::Duration,
    pub(crate) tensor_batch_max_width: usize,
    pub(crate) admission_queue_max: usize,
    pub(crate) effective_cap_max: usize,
    pub(crate) budget_state: BudgetState,
    pub(crate) cache_cost: DFlash2TargetCacheCost,
    pub(crate) prefix_cache_max_bytes: Option<usize>,
    pub(crate) initial_draft_budget: usize,
    pub(crate) target_execution_fingerprint: String,
    pub(crate) verify_profile: String,
    /// Whether the target certifies any B>1 execution. When false, every
    /// request keeps B1 prefill, proposal and verify graphs.
    pub(crate) batched_execution: bool,
}

#[derive(Clone)]
pub struct DFlash2ActorHandle {
    cmd_tx: mpsc::UnboundedSender<DFlash2Command>,
    in_flight: Arc<AtomicUsize>,
    capacity: usize,
    b_max: usize,
    pub(crate) runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
    pub(crate) b_active: Arc<AtomicU64>,
    pub(crate) b_queued: Arc<AtomicU64>,
    pub(crate) background_paused: Arc<AtomicU64>,
    pub(crate) background_preemptions: Arc<AtomicU64>,
    pub(crate) background_resumes: Arc<AtomicU64>,
    pub(crate) admit_count: Arc<AtomicU64>,
    pub(crate) batch_count: Arc<AtomicU64>,
    pub(crate) admission_queue_full_count: Arc<AtomicU64>,
    pub(crate) memory_budget_exceeded_count: Arc<AtomicU64>,
    pub(crate) kv_cache_active_bytes: Arc<AtomicUsize>,
    pub(crate) kv_cache_soft_limit_bytes: usize,
    pub(crate) kv_cache_logical_cap_tokens: usize,
    pub(crate) kv_cache_resident_cap_tokens: usize,
    pub(crate) kv_cache_budget_policy: &'static str,
    pub(crate) windows: Arc<AtomicU64>,
    pub(crate) drafted_tokens: Arc<AtomicU64>,
    pub(crate) accepted_draft_tokens: Arc<AtomicU64>,
    pub(crate) rollback_count: Arc<AtomicU64>,
    pub(crate) ordinary_windows: Arc<AtomicU64>,
    pub(crate) tree_windows: Arc<AtomicU64>,
    pub(crate) tree_drafted_nodes: Arc<AtomicU64>,
    pub(crate) tree_fallback_linear_windows: Arc<AtomicU64>,
    pub(crate) draft_budget_changes: Arc<AtomicU64>,
    pub(crate) current_draft_budget: Arc<AtomicUsize>,
    pub(crate) latest_adaptive_acceptance_ewma_bits: Arc<AtomicU64>,
    pub(crate) tensor_batch_windows: Arc<AtomicU64>,
    pub(crate) tensor_batch_divergent_splits: Arc<AtomicU64>,
    pub(crate) tensor_batch_groups_created: Arc<AtomicU64>,
    pub(crate) tensor_batch_width_limit: usize,
    pub(crate) tensor_batch_max_width: Arc<AtomicUsize>,
    pub(crate) sampled_requests: Arc<AtomicU64>,
    pub(crate) exact_sampling_windows: Arc<AtomicU64>,
    pub(crate) exact_acceptance_draws: Arc<AtomicU64>,
    pub(crate) exact_residual_corrections: Arc<AtomicU64>,
    pub(crate) exact_bonus_samples: Arc<AtomicU64>,
    pub(crate) sampling_us: Arc<AtomicU64>,
    pub(crate) draft_build_us: Arc<AtomicU64>,
    pub(crate) draft_schedule_us: Arc<AtomicU64>,
    pub(crate) verify_build_us: Arc<AtomicU64>,
    pub(crate) projection_build_us: Arc<AtomicU64>,
    pub(crate) verify_schedule_us: Arc<AtomicU64>,
    pub(crate) host_sync_us: Arc<AtomicU64>,
    pub(crate) rollback_us: Arc<AtomicU64>,
    pub(crate) window_us: Arc<AtomicU64>,
    pub(crate) prefill_us: Arc<AtomicU64>,
    pub(crate) generation_us: Arc<AtomicU64>,
    pub(crate) verify_profile: String,
    pub(crate) prefix_fingerprint: String,
    pub(crate) latest_generation_tps_bits: Arc<AtomicU64>,
    pub(crate) latest_acceptance_rate_bits: Arc<AtomicU64>,
    pub(crate) peak_memory_bytes: Arc<AtomicUsize>,
    pub(crate) prefix_cache_enabled: bool,
    pub(crate) prefix_cache_max_bytes: Option<usize>,
    pub(crate) prefix_cache_entries: Arc<AtomicUsize>,
    pub(crate) prefix_cache_bytes: Arc<AtomicUsize>,
    pub(crate) prefix_cache_hits: Arc<AtomicU64>,
    pub(crate) prefix_cache_misses: Arc<AtomicU64>,
    pub(crate) prefix_cache_saves: Arc<AtomicU64>,
    pub(crate) prefix_cache_evictions: Arc<AtomicU64>,
    pub(crate) prefix_cache_hit_tokens: Arc<AtomicU64>,
    pub(crate) ragged_linear: DFlash2RaggedLinearCounters,
}

#[derive(Clone)]
struct DFlash2PrefixCacheCounters {
    entries: Arc<AtomicUsize>,
    bytes: Arc<AtomicUsize>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    saves: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    hit_tokens: Arc<AtomicU64>,
}

impl DFlash2PrefixCacheCounters {
    fn publish(&self, cache: Option<&DFlash2PrefixCache>) {
        let snapshot = cache.map(DFlash2PrefixCache::snapshot).unwrap_or_default();
        self.entries.store(snapshot.entries, Ordering::Relaxed);
        self.bytes.store(snapshot.bytes, Ordering::Relaxed);
        self.hits.store(snapshot.hits, Ordering::Relaxed);
        self.misses.store(snapshot.misses, Ordering::Relaxed);
        self.saves.store(snapshot.saves, Ordering::Relaxed);
        self.evictions.store(snapshot.evictions, Ordering::Relaxed);
    }
}

pub(super) enum DFlash2EnqueueError {
    QueueFull(anyhow::Error),
    Unavailable,
}

impl DFlash2ActorHandle {
    pub(super) fn enqueue(
        &self,
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    ) -> std::result::Result<(), DFlash2EnqueueError> {
        let mut observed = self.in_flight.load(Ordering::Acquire);
        loop {
            if observed >= self.capacity {
                self.admission_queue_full_count
                    .fetch_add(1, Ordering::Relaxed);
                return Err(DFlash2EnqueueError::QueueFull(anyhow::Error::new(
                    SchedulerError::QueueFull {
                        capacity: self.capacity.saturating_sub(self.b_max),
                    },
                )));
            }
            match self.in_flight.compare_exchange_weak(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => observed = current,
            }
        }

        self.b_queued.fetch_add(1, Ordering::Relaxed);
        if self
            .cmd_tx
            .send(DFlash2Command::Admit { request, reply_tx })
            .is_err()
        {
            self.b_queued.fetch_sub(1, Ordering::Relaxed);
            self.in_flight.fetch_sub(1, Ordering::Release);
            return Err(DFlash2EnqueueError::Unavailable);
        }
        Ok(())
    }
}

pub(crate) fn spawn_dflash2_actor<M>(
    model: Arc<Mutex<M>>,
    draft: DFlash2DraftModel,
    tokenizer: Arc<Tokenizer>,
    config: DFlash2ActorConfig,
    cold_materialization_tracker: Arc<crate::core::process_memory::ColdMaterializationTracker>,
) -> DFlash2ActorHandle
where
    M: Model + DenseVlMethods + DFlash2Target + Send + 'static,
{
    let DFlash2ActorConfig {
        block_size,
        p2_options,
        b_max,
        admission_deadline,
        tensor_batch_max_width,
        admission_queue_max,
        effective_cap_max,
        budget_state,
        cache_cost,
        prefix_cache_max_bytes,
        initial_draft_budget,
        target_execution_fingerprint,
        verify_profile,
        batched_execution,
    } = config;
    assert!(b_max > 0, "DFlash2 actor requires b_max > 0");
    assert!(
        (1..=b_max).contains(&tensor_batch_max_width),
        "DFlash2 tensor batch width limit must be in 1..=b_max"
    );
    let capacity = admission_queue_max.saturating_add(b_max);
    let ragged_linear_max_width = b_max.min(RAGGED_LINEAR_MAX_WIDTH);
    let ragged_linear_enabled = batched_execution
        && ragged_linear_max_width >= 2
        && ironmlx_core::m5_profile::flag(RAGGED_LINEAR_ENV);
    if ragged_linear_enabled {
        tracing::info!(
            target: "ironmlx::dflash2",
            max_width = ragged_linear_max_width,
            "DFlash2 tree/linear switch enabled"
        );
    }
    let ragged_linear = DFlash2RaggedLinearCounters {
        enabled: ragged_linear_enabled,
        max_width_limit: if ragged_linear_enabled {
            ragged_linear_max_width
        } else {
            0
        },
        ..Default::default()
    };
    let worker_ragged_linear = ragged_linear.clone();
    let ragged_budget_limit = std::env::var(RAGGED_BUDGET_LIMIT_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    if let Some(limit) = ragged_budget_limit {
        tracing::warn!(
            target: "ironmlx::dflash2",
            limit,
            "diagnostic DFlash2 ragged group budget limit enabled"
        );
    }
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
    let in_flight = Arc::new(AtomicUsize::new(0));
    let b_active = Arc::new(AtomicU64::new(0));
    let b_queued = Arc::new(AtomicU64::new(0));
    let background_paused = Arc::new(AtomicU64::new(0));
    let background_preemptions = Arc::new(AtomicU64::new(0));
    let background_resumes = Arc::new(AtomicU64::new(0));
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let admission_queue_full_count = Arc::new(AtomicU64::new(0));
    let memory_budget_exceeded_count = Arc::new(AtomicU64::new(0));
    let kv_cache_active_bytes = budget_state.shared_active();
    let kv_cache_soft_limit_bytes = budget_state.soft_limit();
    let kv_cache_logical_cap_tokens = budget_state.logical_cap();
    let kv_cache_resident_cap_tokens = budget_state.resident_cap();
    let kv_cache_budget_policy = budget_state.policy().name();
    let windows = Arc::new(AtomicU64::new(0));
    let drafted_tokens = Arc::new(AtomicU64::new(0));
    let accepted_draft_tokens = Arc::new(AtomicU64::new(0));
    let rollback_count = Arc::new(AtomicU64::new(0));
    let ordinary_windows = Arc::new(AtomicU64::new(0));
    let tree_windows = Arc::new(AtomicU64::new(0));
    let tree_drafted_nodes = Arc::new(AtomicU64::new(0));
    let tree_fallback_linear_windows = Arc::new(AtomicU64::new(0));
    let draft_budget_changes = Arc::new(AtomicU64::new(0));
    let current_draft_budget = Arc::new(AtomicUsize::new(initial_draft_budget));
    let latest_adaptive_acceptance_ewma_bits = Arc::new(AtomicU64::new(0_f64.to_bits()));
    let tensor_batch_windows = Arc::new(AtomicU64::new(0));
    let tensor_batch_divergent_splits = Arc::new(AtomicU64::new(0));
    let tensor_batch_groups_created = Arc::new(AtomicU64::new(0));
    let tensor_batch_observed_max_width = Arc::new(AtomicUsize::new(0));
    let sampled_requests = Arc::new(AtomicU64::new(0));
    let exact_sampling_windows = Arc::new(AtomicU64::new(0));
    let exact_acceptance_draws = Arc::new(AtomicU64::new(0));
    let exact_residual_corrections = Arc::new(AtomicU64::new(0));
    let exact_bonus_samples = Arc::new(AtomicU64::new(0));
    let sampling_us = Arc::new(AtomicU64::new(0));
    let draft_build_us = Arc::new(AtomicU64::new(0));
    let draft_schedule_us = Arc::new(AtomicU64::new(0));
    let verify_build_us = Arc::new(AtomicU64::new(0));
    let projection_build_us = Arc::new(AtomicU64::new(0));
    let verify_schedule_us = Arc::new(AtomicU64::new(0));
    let host_sync_us = Arc::new(AtomicU64::new(0));
    let rollback_us = Arc::new(AtomicU64::new(0));
    let window_us = Arc::new(AtomicU64::new(0));
    let prefill_us = Arc::new(AtomicU64::new(0));
    let generation_us = Arc::new(AtomicU64::new(0));
    let latest_generation_tps_bits = Arc::new(AtomicU64::new(0_f64.to_bits()));
    let latest_acceptance_rate_bits = Arc::new(AtomicU64::new(0_f64.to_bits()));
    let peak_memory_bytes = Arc::new(AtomicUsize::new(0));
    let prefix_cache_entries = Arc::new(AtomicUsize::new(0));
    let prefix_cache_bytes = Arc::new(AtomicUsize::new(0));
    let prefix_cache_hits = Arc::new(AtomicU64::new(0));
    let prefix_cache_misses = Arc::new(AtomicU64::new(0));
    let prefix_cache_saves = Arc::new(AtomicU64::new(0));
    let prefix_cache_evictions = Arc::new(AtomicU64::new(0));
    let prefix_cache_hit_tokens = Arc::new(AtomicU64::new(0));
    let worker_prefix_cache_counters = DFlash2PrefixCacheCounters {
        entries: Arc::clone(&prefix_cache_entries),
        bytes: Arc::clone(&prefix_cache_bytes),
        hits: Arc::clone(&prefix_cache_hits),
        misses: Arc::clone(&prefix_cache_misses),
        saves: Arc::clone(&prefix_cache_saves),
        evictions: Arc::clone(&prefix_cache_evictions),
        hit_tokens: Arc::clone(&prefix_cache_hit_tokens),
    };
    let runtime_usage = Arc::new(crate::core::runtime_usage::ModelRuntimeUsageCounters::default());
    let worker_runtime_usage = Arc::clone(&runtime_usage);
    let prefix_fingerprint = format!(
        "dflash2-prefix-v2:ironmlx={};backend=mlx;kernel-contract=row-exact-v1;prefill=scheduler-b1-chunk-v2-cold-single;target={};draft-dtype={};draft-hidden={};draft-layer-count={};target-layers={:?};sliding-window={};block-size={}",
        env!("CARGO_PKG_VERSION"),
        target_execution_fingerprint,
        draft.config().dtype,
        draft.config().hidden_size,
        draft.config().num_hidden_layers,
        draft.config().dflash_config.target_layer_ids,
        draft.config().sliding_window,
        block_size,
    );
    let health_prefix_fingerprint = prefix_fingerprint.clone();
    let worker_in_flight = Arc::clone(&in_flight);
    let worker_active = Arc::clone(&b_active);
    let worker_queued = Arc::clone(&b_queued);
    let worker_background_paused = Arc::clone(&background_paused);
    let worker_background_preemptions = Arc::clone(&background_preemptions);
    let worker_background_resumes = Arc::clone(&background_resumes);
    let worker_admit_count = Arc::clone(&admit_count);
    let worker_batch_count = Arc::clone(&batch_count);
    let worker_memory_budget_exceeded_count = Arc::clone(&memory_budget_exceeded_count);
    let worker_tensor_batch_windows = Arc::clone(&tensor_batch_windows);
    let worker_tensor_batch_divergent_splits = Arc::clone(&tensor_batch_divergent_splits);
    let worker_tensor_batch_groups_created = Arc::clone(&tensor_batch_groups_created);
    let worker_tensor_batch_max_width = Arc::clone(&tensor_batch_observed_max_width);
    let worker_counters = DFlash2ActorCounters {
        windows: Arc::clone(&windows),
        drafted_tokens: Arc::clone(&drafted_tokens),
        accepted_draft_tokens: Arc::clone(&accepted_draft_tokens),
        rollback_count: Arc::clone(&rollback_count),
        ordinary_windows: Arc::clone(&ordinary_windows),
        tree_windows: Arc::clone(&tree_windows),
        tree_drafted_nodes: Arc::clone(&tree_drafted_nodes),
        tree_fallback_linear_windows: Arc::clone(&tree_fallback_linear_windows),
        draft_budget_changes: Arc::clone(&draft_budget_changes),
        current_draft_budget: Arc::clone(&current_draft_budget),
        latest_adaptive_acceptance_ewma_bits: Arc::clone(&latest_adaptive_acceptance_ewma_bits),
        sampled_requests: Arc::clone(&sampled_requests),
        exact_sampling_windows: Arc::clone(&exact_sampling_windows),
        exact_acceptance_draws: Arc::clone(&exact_acceptance_draws),
        exact_residual_corrections: Arc::clone(&exact_residual_corrections),
        exact_bonus_samples: Arc::clone(&exact_bonus_samples),
        sampling_us: Arc::clone(&sampling_us),
        draft_build_us: Arc::clone(&draft_build_us),
        draft_schedule_us: Arc::clone(&draft_schedule_us),
        verify_build_us: Arc::clone(&verify_build_us),
        projection_build_us: Arc::clone(&projection_build_us),
        verify_schedule_us: Arc::clone(&verify_schedule_us),
        host_sync_us: Arc::clone(&host_sync_us),
        rollback_us: Arc::clone(&rollback_us),
        window_us: Arc::clone(&window_us),
        prefill_us: Arc::clone(&prefill_us),
        generation_us: Arc::clone(&generation_us),
        latest_generation_tps_bits: Arc::clone(&latest_generation_tps_bits),
        latest_acceptance_rate_bits: Arc::clone(&latest_acceptance_rate_bits),
        peak_memory_bytes: Arc::clone(&peak_memory_bytes),
    };

    tokio::task::spawn_blocking(move || {
        let model = model.blocking_lock();
        let mut prefix_cache = prefix_cache_max_bytes
            .map(DFlash2PrefixCache::new)
            .transpose()
            .expect("validated DFlash2 prefix cache capacity");
        let mut next_request_id = 1_u64;
        let mut active = Vec::<ActiveDFlash2Request<'_, M>>::with_capacity(b_max);
        let mut paused_background = VecDeque::<ActiveDFlash2Request<'_, M>>::with_capacity(b_max);
        let mut tensor_groups = Vec::<DFlash2TensorGroup>::with_capacity(b_max / 2);
        let mut ragged_group = None::<DFlash2RaggedGroup>;
        let mut token_id_diagnostic =
            crate::core::dflash2_token_diagnostic::DFlash2TokenIdDiagnostic::from_env();
        let mut step_diagnostic =
            crate::core::dflash2_step_diagnostic::DFlash2StepDiagnostic::from_env();
        let mut pending = VecDeque::<DFlash2Command>::new();
        let mut command_channel_open = true;

        loop {
            let forming_empty_batch = active.is_empty();
            let mut admission_window_waited = false;
            while command_channel_open {
                match cmd_rx.try_recv() {
                    Ok(command) => push_pending_by_priority(&mut pending, command),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        command_channel_open = false;
                        break;
                    }
                }
            }
            prune_abandoned_pending_requests(&mut pending, &worker_in_flight, &worker_queued);

            let foreground_pressure = active
                .iter()
                .any(|request| !request.priority.is_background())
                || pending
                    .iter()
                    .any(|command| !command.priority().is_background());
            if foreground_pressure
                && active
                    .iter()
                    .any(|request| request.priority.is_background())
            {
                if let Err(error) = scatter_all_tensor_groups(&mut active, &mut tensor_groups)
                    .and_then(|()| {
                        scatter_ragged_group(&mut active, &mut ragged_group, &worker_ragged_linear)
                    })
                {
                    tracing::error!(%error, "DFlash2 background priority scatter failed");
                    for request in active.drain(..).chain(paused_background.drain(..)) {
                        drop(request);
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                    }
                    while let Some(DFlash2Command::Admit { reply_tx, .. }) = pending.pop_front() {
                        let _ = reply_tx.send(Err(anyhow::anyhow!(
                            "DFlash2 background priority scatter failed"
                        )));
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                    }
                    worker_active.store(0, Ordering::Relaxed);
                    worker_background_paused.store(0, Ordering::Relaxed);
                    return;
                }
                let mut foreground = Vec::with_capacity(active.len());
                for request in active.drain(..) {
                    if request.priority.is_background() {
                        paused_background.push_back(request);
                        worker_background_preemptions.fetch_add(1, Ordering::Relaxed);
                    } else {
                        foreground.push(request);
                    }
                }
                active = foreground;
            } else if !foreground_pressure {
                while active.len() < b_max {
                    let Some(request) = paused_background.pop_front() else {
                        break;
                    };
                    if request.event_tx.is_closed() {
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                        continue;
                    }
                    active.push(request);
                    worker_background_resumes.fetch_add(1, Ordering::Relaxed);
                }
            }
            paused_background.retain(|request| {
                if request.event_tx.is_closed() {
                    worker_in_flight.fetch_sub(1, Ordering::Release);
                    false
                } else {
                    true
                }
            });
            worker_active.store(active.len() as u64, Ordering::Relaxed);
            worker_background_paused.store(paused_background.len() as u64, Ordering::Relaxed);

            while active.len() < b_max {
                if pending.is_empty() && active.is_empty() && command_channel_open {
                    match cmd_rx.blocking_recv() {
                        Some(command) => push_pending_by_priority(&mut pending, command),
                        None => command_channel_open = false,
                    }
                    prune_abandoned_pending_requests(
                        &mut pending,
                        &worker_in_flight,
                        &worker_queued,
                    );
                }

                if pending.is_empty()
                    && forming_empty_batch
                    && !active.is_empty()
                    && !admission_window_waited
                    && !admission_deadline.is_zero()
                {
                    std::thread::sleep(admission_deadline);
                    admission_window_waited = true;
                    while command_channel_open {
                        match cmd_rx.try_recv() {
                            Ok(command) => push_pending_by_priority(&mut pending, command),
                            Err(mpsc::error::TryRecvError::Empty) => break,
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                command_channel_open = false;
                                break;
                            }
                        }
                    }
                    prune_abandoned_pending_requests(
                        &mut pending,
                        &worker_in_flight,
                        &worker_queued,
                    );
                }

                if active
                    .iter()
                    .any(|request| !request.priority.is_background())
                    && pending
                        .front()
                        .is_some_and(|command| command.priority().is_background())
                {
                    break;
                }

                let Some(DFlash2Command::Admit { request, reply_tx }) = pending.pop_front() else {
                    break;
                };
                worker_queued.fetch_sub(1, Ordering::Relaxed);
                if discard_abandoned_queued_request(&reply_tx, &worker_in_flight) {
                    continue;
                }

                let required_total_tokens = request
                    .prompt_ids
                    .len()
                    .saturating_add(request.max_new_tokens);
                if required_total_tokens > effective_cap_max {
                    let error = SchedulerError::RequestTooLarge {
                        required_total_tokens,
                        input_tokens: request.prompt_ids.len(),
                        requested_max_output_tokens: request.max_new_tokens,
                        server_max_context_tokens: effective_cap_max,
                        max_allowed_output_tokens: effective_cap_max
                            .saturating_sub(request.prompt_ids.len()),
                    };
                    let _ = reply_tx.send(Err(anyhow::Error::new(error)));
                    worker_in_flight.fetch_sub(1, Ordering::Release);
                    continue;
                }
                if let Err(error) = DFlash2TextGenerationStream::<M>::validate_text_request(
                    &draft, &request, block_size,
                ) {
                    let _ = reply_tx.send(Err(error));
                    worker_in_flight.fetch_sub(1, Ordering::Release);
                    continue;
                }

                let memory_charge = match reserve_dflash2_request_memory(
                    &budget_state,
                    cache_cost,
                    crate::core::dflash2::dflash2_target_cache_tokens(
                        request.prompt_ids.len(),
                        request.max_new_tokens,
                        block_size,
                        p2_options.tree_max_nodes,
                    ),
                    if request.priority.is_background() {
                        0
                    } else {
                        paused_background
                            .iter()
                            .map(|request| request._memory_charge.bytes())
                            .fold(0usize, usize::saturating_add)
                    },
                    &worker_memory_budget_exceeded_count,
                ) {
                    Ok(charge) => charge,
                    Err(error) => {
                        let _ = reply_tx.send(Err(error));
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                        continue;
                    }
                };
                let governor = crate::core::process_memory::global_process_memory_governor();
                let mut snapshot = governor.sample_process();
                if super::dflash2::prefill_phase_diagnostic_enabled() {
                    tracing::warn!(
                        diagnostic = "dflash2_admission_pressure",
                        level = ?snapshot.pressure_level,
                        current_bytes = snapshot.current_usage_bytes,
                        soft_watermark_bytes = snapshot.soft_watermark_bytes,
                        ceiling_bytes = snapshot.effective_ceiling_bytes,
                        "diagnostic admission memory pressure; not a formal measurement"
                    );
                }
                if snapshot.pressure_level != crate::core::process_memory::PressureLevel::Normal {
                    let retain_ratio = match snapshot.pressure_level {
                        crate::core::process_memory::PressureLevel::Normal => 1.0,
                        crate::core::process_memory::PressureLevel::Soft => 0.5,
                        crate::core::process_memory::PressureLevel::Hard
                        | crate::core::process_memory::PressureLevel::Emergency => 0.0,
                    };
                    if let Some(cache) = prefix_cache.as_mut() {
                        let cache_bytes = cache.snapshot().bytes;
                        let target_bytes = (cache_bytes as f64 * retain_ratio) as usize;
                        let reclaimed_bytes = cache.shrink_to(target_bytes);
                        if reclaimed_bytes > 0 {
                            tracing::info!(
                                reclaimed_bytes,
                                ?snapshot.pressure_level,
                                "memory governor shrank DFlash2 prefix cache"
                            );
                        }
                    }
                    worker_prefix_cache_counters.publish(prefix_cache.as_ref());
                    mlx::transforms::clear_cache();
                    snapshot = governor.sample_process();
                }
                if snapshot.pressure_level != crate::core::process_memory::PressureLevel::Normal {
                    let error = SchedulerError::MemoryPressure {
                        level: snapshot.pressure_level,
                        current_bytes: snapshot.current_usage_bytes,
                        ceiling_bytes: snapshot.effective_ceiling_bytes,
                    };
                    let _ = reply_tx.send(Err(anyhow::Error::new(error)));
                    worker_in_flight.fetch_sub(1, Ordering::Release);
                    continue;
                }
                let governor_reservation =
                    match governor.try_reserve(memory_charge.bytes(), "dflash2_admission") {
                        Ok(reservation) => reservation,
                        Err(error) => {
                            let snapshot = governor.snapshot();
                            tracing::warn!(
                                error = %error,
                                "process memory governor rejected DFlash2 admission"
                            );
                            let error = SchedulerError::MemoryPressure {
                                level: snapshot.pressure_level,
                                current_bytes: snapshot.current_usage_bytes,
                                ceiling_bytes: snapshot.effective_ceiling_bytes,
                            };
                            let _ = reply_tx.send(Err(anyhow::Error::new(error)));
                            worker_in_flight.fetch_sub(1, Ordering::Release);
                            continue;
                        }
                    };

                let components =
                    crate::core::process_memory::MaterializationComponents::for_request(
                        false, true,
                    );
                let cold = match cold_materialization_tracker.begin(
                    components,
                    &crate::core::process_memory::global_process_memory_governor(),
                ) {
                    Ok(cold) => cold,
                    Err(_) => {
                        let snapshot = governor.snapshot();
                        let error = SchedulerError::ColdMaterializationUnsafe {
                            requested_bytes: components
                                .requested_bytes(cold_materialization_tracker.estimate()),
                            current_bytes: snapshot.current_usage_bytes,
                            target_bytes: snapshot.hard_watermark_bytes,
                        };
                        let _ = reply_tx.send(Err(anyhow::Error::new(error)));
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                        continue;
                    }
                };

                let request_id = RequestId(next_request_id);
                next_request_id = next_request_id.wrapping_add(1).max(1);
                let batch_prompt_len = request.prompt_ids.len();
                let batch_chunk_size = request.prefill_chunk_size;
                let batch_priority = request.priority;
                let mut admission_requests = vec![request];
                let mut admission_priorities = vec![batch_priority];
                let mut admission_replies = vec![reply_tx];
                let mut admission_request_ids = vec![request_id];
                let mut admission_memory_charges = vec![memory_charge];
                let mut admission_governor_reservations = vec![governor_reservation];

                // Prefix artifacts are request-local and may start at different
                // offsets. Cold misses without that feature can preserve the B1
                // prefill morphology in one equal-length B=N graph.
                while batched_execution
                    && prefix_cache.is_none()
                    && active.len() + admission_requests.len() < b_max
                    && pending.front().is_some_and(|command| match command {
                        DFlash2Command::Admit { request, .. } => {
                            request.prompt_ids.len() == batch_prompt_len
                                && request.prefill_chunk_size == batch_chunk_size
                                && request.priority == batch_priority
                        }
                    })
                {
                    let Some(DFlash2Command::Admit {
                        request: candidate,
                        reply_tx: candidate_reply,
                    }) = pending.pop_front()
                    else {
                        break;
                    };
                    worker_queued.fetch_sub(1, Ordering::Relaxed);
                    if discard_abandoned_queued_request(&candidate_reply, &worker_in_flight) {
                        continue;
                    }
                    let required_total_tokens = candidate
                        .prompt_ids
                        .len()
                        .saturating_add(candidate.max_new_tokens);
                    if required_total_tokens > effective_cap_max {
                        let error = SchedulerError::RequestTooLarge {
                            required_total_tokens,
                            input_tokens: candidate.prompt_ids.len(),
                            requested_max_output_tokens: candidate.max_new_tokens,
                            server_max_context_tokens: effective_cap_max,
                            max_allowed_output_tokens: effective_cap_max
                                .saturating_sub(candidate.prompt_ids.len()),
                        };
                        let _ = candidate_reply.send(Err(anyhow::Error::new(error)));
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                        continue;
                    }
                    if let Err(error) = DFlash2TextGenerationStream::<M>::validate_text_request(
                        &draft, &candidate, block_size,
                    ) {
                        let _ = candidate_reply.send(Err(error));
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                        continue;
                    }
                    let candidate_charge = match reserve_dflash2_request_memory(
                        &budget_state,
                        cache_cost,
                        crate::core::dflash2::dflash2_target_cache_tokens(
                            candidate.prompt_ids.len(),
                            candidate.max_new_tokens,
                            block_size,
                            p2_options.tree_max_nodes,
                        ),
                        if candidate.priority.is_background() {
                            0
                        } else {
                            paused_background
                                .iter()
                                .map(|request| request._memory_charge.bytes())
                                .fold(0usize, usize::saturating_add)
                        },
                        &worker_memory_budget_exceeded_count,
                    ) {
                        Ok(charge) => charge,
                        Err(error) => {
                            let _ = candidate_reply.send(Err(error));
                            worker_in_flight.fetch_sub(1, Ordering::Release);
                            continue;
                        }
                    };
                    let candidate_reservation =
                        match governor.try_reserve(candidate_charge.bytes(), "dflash2_admission") {
                            Ok(reservation) => reservation,
                            Err(error) => {
                                let snapshot = governor.snapshot();
                                tracing::warn!(
                                    error = %error,
                                    "process memory governor rejected batched DFlash2 admission"
                                );
                                let error = SchedulerError::MemoryPressure {
                                    level: snapshot.pressure_level,
                                    current_bytes: snapshot.current_usage_bytes,
                                    ceiling_bytes: snapshot.effective_ceiling_bytes,
                                };
                                let _ = candidate_reply.send(Err(anyhow::Error::new(error)));
                                worker_in_flight.fetch_sub(1, Ordering::Release);
                                continue;
                            }
                        };
                    let candidate_id = RequestId(next_request_id);
                    next_request_id = next_request_id.wrapping_add(1).max(1);
                    admission_requests.push(candidate);
                    admission_priorities.push(batch_priority);
                    admission_replies.push(candidate_reply);
                    admission_request_ids.push(candidate_id);
                    admission_memory_charges.push(candidate_charge);
                    admission_governor_reservations.push(candidate_reservation);
                }

                let streams_result = if admission_requests.len() > 1 {
                    DFlash2TextGenerationStream::new_scheduler_bn_text_only_with_cancellation(
                        &*model,
                        &draft,
                        &tokenizer,
                        admission_requests,
                        block_size,
                        p2_options,
                        &|_| false,
                    )
                    .context("initializing batched DFlash2 actor streams")
                } else {
                    let request = admission_requests
                        .pop()
                        .expect("one DFlash2 admission request is present");
                    DFlash2TextGenerationStream::new_scheduler_b1_text_only_with_cancellation(
                        &*model,
                        &draft,
                        &tokenizer,
                        request,
                        super::dflash2::DFlash2ExecutionOptions {
                            block_size,
                            p2: p2_options,
                        },
                        prefix_cache
                            .as_mut()
                            .map(|cache| (cache, prefix_fingerprint.as_str())),
                        &|| admission_replies[0].is_closed(),
                    )
                    .map(|stream| vec![stream])
                    .context("initializing DFlash2 actor stream")
                };
                let streams = match streams_result {
                    Ok(streams) => streams,
                    Err(error) => {
                        let message = format!("{error:#}");
                        for reply_tx in admission_replies {
                            let _ = reply_tx.send(Err(anyhow::anyhow!(message.clone())));
                            worker_in_flight.fetch_sub(1, Ordering::Release);
                        }
                        continue;
                    }
                };
                for stream in &streams {
                    worker_runtime_usage.record_prefix_cache_lookup(
                        stream.metrics().prompt_tokens as u64,
                        stream.prefix_cache_hit_tokens() as u64,
                    );
                    worker_prefix_cache_counters
                        .hit_tokens
                        .fetch_add(stream.prefix_cache_hit_tokens() as u64, Ordering::Relaxed);
                }
                worker_prefix_cache_counters.publish(prefix_cache.as_ref());
                // Prefill and first-token materialization completed while
                // constructing the stream. Mark the shared weights warm before
                // admitting another sequence; otherwise a second begin() would
                // wait on the same blocking worker.
                cold.commit();
                for reservation in admission_governor_reservations {
                    reservation.commit();
                }
                governor.refresh_process();

                for ((((request_id, reply_tx), stream), memory_charge), priority) in
                    admission_request_ids
                        .into_iter()
                        .zip(admission_replies)
                        .zip(streams)
                        .zip(admission_memory_charges)
                        .zip(admission_priorities)
                {
                    let (event_tx, event_rx) = mpsc::unbounded_channel();
                    if reply_tx
                        .send(Ok(AdmitReply {
                            request_id,
                            event_rx,
                        }))
                        .is_err()
                    {
                        worker_in_flight.fetch_sub(1, Ordering::Release);
                        continue;
                    }
                    worker_admit_count.fetch_add(1, Ordering::Relaxed);
                    active.push(ActiveDFlash2Request {
                        request_id,
                        priority,
                        event_tx,
                        stream,
                        _memory_charge: memory_charge,
                    });
                }
                worker_active.store(active.len() as u64, Ordering::Relaxed);
            }

            if active.is_empty() {
                if command_channel_open {
                    continue;
                }
                break;
            }

            worker_batch_count.fetch_add(1, Ordering::Relaxed);
            let batch_width = active.len();
            DFlash2StepDiagnostic::begin(&mut step_diagnostic, batch_width);
            // With two or more active rows and the switch on, every row
            // publishes its whole committed window so rows reach window
            // boundaries together. A lone row keeps the original one token
            // per step.
            let publish_whole_windows = ragged_linear_enabled && active.len() >= 2;
            let mut outcomes = (0..active.len())
                .map(|_| DFlash2StepOutcome::default())
                .collect::<Vec<_>>();

            for (index, request) in active.iter_mut().enumerate() {
                if request.event_tx.is_closed() {
                    outcomes[index].cancelled = true;
                    outcomes[index].finished = true;
                    continue;
                }
                loop {
                    match request.stream.next_token_deferred() {
                        Ok(Some(event)) => {
                            outcomes[index].finished = event.finish_reason.is_some();
                            outcomes[index].events.push(event);
                        }
                        Ok(None) => outcomes[index].finished = true,
                        Err(error) => {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(format!("{error:#}"));
                        }
                    }
                    if !publish_whole_windows
                        || outcomes[index].finished
                        || request.stream.pending_token_count() == 0
                    {
                        break;
                    }
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "next_tokens");
            // The stream already owns a materialized token before any draft
            // window is built. Publish it immediately so TTFT does not include
            // the following speculative draft/verify cycle.
            for (index, outcome) in outcomes.iter_mut().enumerate() {
                if outcome.failure.is_some() {
                    continue;
                }
                for event in &outcome.events {
                    if active[index]
                        .event_tx
                        .send(StepEvent {
                            id: active[index].request_id,
                            token: event.token,
                            finish_reason: event.finish_reason,
                        })
                        .is_err()
                    {
                        outcome.cancelled = true;
                        outcome.finished = true;
                        break;
                    }
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "publish");
            let mut keys = vec![None; active.len()];
            for (index, request) in active.iter().enumerate() {
                if outcomes[index].finished || outcomes[index].failure.is_some() {
                    continue;
                }
                match request.stream.tensor_batch_key() {
                    Ok(key) => keys[index] = key,
                    Err(error) => {
                        outcomes[index].finished = true;
                        outcomes[index].failure = Some(format!("{error:#}"));
                    }
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "keys");
            let mut claimed = vec![false; active.len()];
            let mut group_index = 0_usize;
            while group_index < tensor_groups.len() {
                let group = tensor_groups.remove(group_index);
                let Some(positions) = tensor_group_positions(&active, &group.request_ids) else {
                    continue;
                };
                if positions.iter().any(|&index| outcomes[index].finished) {
                    let mut streams = match tensor_group_streams_mut(&mut active, &positions) {
                        Ok(streams) => streams,
                        Err(error) => {
                            let error = format!("{error:#}");
                            for &index in &positions {
                                outcomes[index].finished = true;
                                outcomes[index].failure = Some(error.clone());
                            }
                            continue;
                        }
                    };
                    if let Err(error) = group.cache.scatter_to_rows(&mut streams) {
                        let error = format!("{error:#}");
                        for &index in &positions {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(error.clone());
                        }
                    }
                    continue;
                }
                let group_key = keys[positions[0]];
                if group_key.is_some_and(|key| key.is_ordinary_decode()) {
                    let mut streams = match tensor_group_streams_mut(&mut active, &positions) {
                        Ok(streams) => streams,
                        Err(error) => {
                            let error = format!("{error:#}");
                            for &index in &positions {
                                outcomes[index].finished = true;
                                outcomes[index].failure = Some(error.clone());
                            }
                            continue;
                        }
                    };
                    if let Err(error) = group.cache.scatter_to_rows(&mut streams) {
                        let error = format!("{error:#}");
                        for &index in &positions {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(error.clone());
                        }
                    }
                    continue;
                }
                if group_key.is_some_and(|key| key.supports_batch_width(positions.len()))
                    && positions.iter().all(|&index| keys[index] == group_key)
                {
                    for &index in &positions {
                        claimed[index] = true;
                    }
                    let mut streams = match tensor_group_streams_mut(&mut active, &positions) {
                        Ok(streams) => streams,
                        Err(error) => {
                            let error = format!("{error:#}");
                            for &index in &positions {
                                outcomes[index].finished = true;
                                outcomes[index].failure = Some(error.clone());
                            }
                            continue;
                        }
                    };
                    worker_tensor_batch_windows.fetch_add(1, Ordering::Relaxed);
                    worker_tensor_batch_max_width.fetch_max(positions.len(), Ordering::Relaxed);
                    match DFlash2TextGenerationStream::fill_deferred_window_bn(
                        &mut streams,
                        Some(group.cache),
                    ) {
                        Ok(Some(cache)) => {
                            tensor_groups.insert(
                                group_index,
                                DFlash2TensorGroup {
                                    request_ids: group.request_ids,
                                    cache,
                                },
                            );
                            group_index += 1;
                        }
                        Ok(None) => {
                            worker_tensor_batch_divergent_splits.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            let error = format!("{error:#}");
                            for &index in &positions {
                                outcomes[index].finished = true;
                                outcomes[index].failure = Some(error.clone());
                            }
                        }
                    }
                } else if positions.iter().all(|&index| keys[index].is_some()) {
                    let mut streams = match tensor_group_streams_mut(&mut active, &positions) {
                        Ok(streams) => streams,
                        Err(error) => {
                            let error = format!("{error:#}");
                            for &index in &positions {
                                outcomes[index].finished = true;
                                outcomes[index].failure = Some(error.clone());
                            }
                            continue;
                        }
                    };
                    if let Err(error) = group.cache.scatter_to_rows(&mut streams) {
                        let error = format!("{error:#}");
                        for &index in &positions {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(error.clone());
                        }
                    }
                } else {
                    for &index in &positions {
                        claimed[index] = true;
                    }
                    tensor_groups.insert(group_index, group);
                    group_index += 1;
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "tensor_groups");
            if ragged_linear_enabled && (active.len() >= 2 || ragged_group.is_some()) {
                let candidates = keys
                    .iter()
                    .enumerate()
                    .filter_map(|(index, key)| {
                        let key = (*key)?;
                        (!claimed[index]
                            && !outcomes[index].finished
                            && !key.is_ordinary_decode()
                            && active[index].stream.ragged_linear_eligible())
                        .then_some((index, key.draft_len(), key.supported_batch_widths()))
                    })
                    .collect::<Vec<_>>();
                if candidates.len() == 1 && active.len() > 1 {
                    worker_ragged_linear
                        .single_eligible_steps
                        .fetch_add(1, Ordering::Relaxed);
                }
                let current_positions = ragged_group
                    .as_ref()
                    .and_then(|group| tensor_group_positions(&active, &group.request_ids))
                    .unwrap_or_default();
                let selected = select_ragged_linear_rows(
                    &candidates,
                    &current_positions,
                    ragged_linear_max_width,
                );
                let reuse = ragged_group.is_some() && selected == current_positions;
                if !reuse {
                    let members = ragged_member_positions(&active, ragged_group.as_ref());
                    if let Err(error) =
                        scatter_ragged_group(&mut active, &mut ragged_group, &worker_ragged_linear)
                    {
                        let error = format!("{error:#}");
                        for &index in &members {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(error.clone());
                        }
                    }
                }
                DFlash2StepDiagnostic::mark(&mut step_diagnostic, "ragged_select_scatter");
                // A new group needs a KV budget charge (and process headroom)
                // for its batched cache before it is built; otherwise the
                // rows keep their original path this step.
                let mut reservation = None;
                if selected.len() >= 2
                    && selected.iter().all(|&index| !outcomes[index].finished)
                    && ragged_group.is_none()
                {
                    let cap_tokens = crate::core::dflash2::ragged_batch_cache_cap(
                        selected
                            .iter()
                            .map(|&index| active[index].stream.ragged_cache_tokens()),
                    );
                    match reserve_ragged_group_memory(
                        &budget_state,
                        cache_cost,
                        cap_tokens,
                        selected.len(),
                        ragged_budget_limit,
                    ) {
                        Ok(charge) => {
                            match reserve_ragged_group_headroom(
                                &crate::core::process_memory::global_process_memory_governor(),
                                charge.bytes(),
                                crate::core::process_memory::ProcessMemoryGovernor::sample_process,
                            ) {
                                Ok(governor) => reservation = Some((charge, governor)),
                                Err(error) => {
                                    tracing::debug!(%error, "DFlash2 ragged group: governor fallback");
                                    worker_ragged_linear
                                        .governor_fallbacks
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        Err(error) => {
                            tracing::debug!(%error, "DFlash2 ragged group: budget fallback");
                            worker_ragged_linear
                                .budget_fallbacks
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                DFlash2StepDiagnostic::mark(&mut step_diagnostic, "ragged_reserve");
                let runnable = ragged_group.is_some() || reservation.is_some();
                if runnable
                    && selected.len() >= 2
                    && selected.iter().all(|&index| !outcomes[index].finished)
                {
                    for &index in &selected {
                        claimed[index] = true;
                    }
                    let request_ids = selected
                        .iter()
                        .map(|&index| active[index].request_id)
                        .collect::<Vec<_>>();
                    let (cache, memory_charge, governor) = match ragged_group.take() {
                        Some(group) => (Some(group.cache), group.memory_charge, None),
                        None => {
                            let (charge, governor) =
                                reservation.take().expect("new ragged group is reserved");
                            (None, charge, Some(governor))
                        }
                    };
                    let built = cache.is_none();
                    let result =
                        tensor_group_streams_mut(&mut active, &selected).and_then(|mut streams| {
                            DFlash2TextGenerationStream::fill_ragged_linear_window_bn(
                                &mut streams,
                                cache,
                            )
                        });
                    match result {
                        Ok((cache, timing)) => {
                            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "ragged_window");
                            DFlash2StepDiagnostic::ragged(&mut step_diagnostic, built, &timing);
                            let width = selected.len();
                            let counters = &worker_ragged_linear;
                            counters.windows.fetch_add(1, Ordering::Relaxed);
                            counters.windows_by_width[width - 2].fetch_add(1, Ordering::Relaxed);
                            counters
                                .row_windows
                                .fetch_add(width as u64, Ordering::Relaxed);
                            counters.emitted_tokens.fetch_add(
                                timing.emitted.iter().sum::<usize>() as u64,
                                Ordering::Relaxed,
                            );
                            counters.accepted_draft_tokens.fetch_add(
                                timing.accepted.iter().sum::<usize>() as u64,
                                Ordering::Relaxed,
                            );
                            counters
                                .window_us
                                .fetch_add(timing.window_us, Ordering::Relaxed);
                            counters.max_width.fetch_max(width, Ordering::Relaxed);
                            if built {
                                counters.groups_built.fetch_add(1, Ordering::Relaxed);
                                counters
                                    .cache_build_us
                                    .fetch_add(timing.cache_build_us, Ordering::Relaxed);
                            }
                            counters.active_groups.store(1, Ordering::Relaxed);
                            counters
                                .reserved_bytes
                                .store(memory_charge.bytes(), Ordering::Relaxed);
                            if let Some(governor) = governor {
                                governor.commit();
                            }
                            ragged_group = Some(DFlash2RaggedGroup {
                                request_ids,
                                cache,
                                memory_charge,
                            });
                        }
                        Err(error) => {
                            drop(memory_charge);
                            worker_ragged_linear
                                .active_groups
                                .store(0, Ordering::Relaxed);
                            worker_ragged_linear
                                .reserved_bytes
                                .store(0, Ordering::Relaxed);
                            let error = format!("{error:#}");
                            for &index in &selected {
                                outcomes[index].finished = true;
                                outcomes[index].failure = Some(error.clone());
                            }
                        }
                    }
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "ragged");
            let mut ready = BTreeMap::new();
            for (index, key) in keys.iter().enumerate() {
                if !claimed[index] && !outcomes[index].finished {
                    if let Some(key) = key {
                        ready.entry(*key).or_insert_with(Vec::new).push(index);
                    }
                }
            }
            for compatible_indices in ready.values() {
                let mut offset = 0_usize;
                while offset < compatible_indices.len() {
                    let first_index = compatible_indices[offset];
                    let key = keys[first_index].expect("ready DFlash2 row has a batch key");
                    let chunk_limit = tensor_batch_max_width.min(compatible_indices.len() - offset);
                    let chunk_width = key.largest_supported_batch_width(chunk_limit);
                    let indices = &compatible_indices[offset..offset + chunk_width];
                    offset += chunk_width;
                    let ordinary_decode =
                        keys[indices[0]].is_some_and(|key| key.is_ordinary_decode());
                    DFlash2StepDiagnostic::note(
                        &mut step_diagnostic,
                        if ordinary_decode {
                            "ordinary_rows"
                        } else if indices.len() >= 2 {
                            "tensor_window_rows"
                        } else {
                            "tree_window_rows"
                        },
                        indices.len() as u64,
                    );
                    let result = if ordinary_decode {
                        indices
                            .iter()
                            .copied()
                            .try_for_each(|index| active[index].stream.fill_deferred_window_b1())
                    } else if indices.len() >= 2 {
                        let request_ids = indices
                            .iter()
                            .map(|&index| active[index].request_id)
                            .collect::<Vec<_>>();
                        tensor_group_streams_mut(&mut active, indices).and_then(|mut streams| {
                            worker_tensor_batch_windows.fetch_add(1, Ordering::Relaxed);
                            worker_tensor_batch_max_width
                                .fetch_max(indices.len(), Ordering::Relaxed);
                            match DFlash2TextGenerationStream::fill_deferred_window_bn(
                                &mut streams,
                                None,
                            ) {
                                Ok(Some(cache)) => {
                                    worker_tensor_batch_groups_created
                                        .fetch_add(1, Ordering::Relaxed);
                                    tensor_groups.push(DFlash2TensorGroup { request_ids, cache });
                                    Ok(())
                                }
                                Ok(None) => {
                                    worker_tensor_batch_divergent_splits
                                        .fetch_add(1, Ordering::Relaxed);
                                    Ok(())
                                }
                                Err(error) => Err(error),
                            }
                        })
                    } else {
                        active[indices[0]].stream.fill_deferred_window_b1()
                    };
                    if let Err(error) = result {
                        let error = format!("{error:#}");
                        for &index in indices {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(error.clone());
                        }
                    }
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "ready_windows");
            let mut group_index = 0_usize;
            while group_index < tensor_groups.len() {
                let Some(positions) =
                    tensor_group_positions(&active, &tensor_groups[group_index].request_ids)
                else {
                    tensor_groups.remove(group_index);
                    continue;
                };
                if positions.iter().all(|&index| !outcomes[index].finished) {
                    group_index += 1;
                    continue;
                }
                let group = tensor_groups.remove(group_index);
                let mut streams = match tensor_group_streams_mut(&mut active, &positions) {
                    Ok(streams) => streams,
                    Err(error) => {
                        let error = format!("{error:#}");
                        for &index in &positions {
                            outcomes[index].finished = true;
                            outcomes[index].failure = Some(error.clone());
                        }
                        continue;
                    }
                };
                if let Err(error) = group.cache.scatter_to_rows(&mut streams) {
                    let error = format!("{error:#}");
                    for &index in &positions {
                        outcomes[index].finished = true;
                        outcomes[index].failure = Some(error.clone());
                    }
                }
            }

            if ragged_group.as_ref().is_some_and(|group| {
                tensor_group_positions(&active, &group.request_ids)
                    .is_none_or(|positions| positions.iter().any(|&index| outcomes[index].finished))
            }) {
                let members = ragged_member_positions(&active, ragged_group.as_ref());
                if let Err(error) =
                    scatter_ragged_group(&mut active, &mut ragged_group, &worker_ragged_linear)
                {
                    let error = format!("{error:#}");
                    for &index in &members {
                        outcomes[index].finished = true;
                        outcomes[index].failure = Some(error.clone());
                    }
                }
            }

            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "group_scatter");
            for index in (0..active.len()).rev() {
                if !outcomes[index].finished {
                    continue;
                }
                let outcome = outcomes.remove(index);
                let completed = active.remove(index);
                let metrics = completed.stream.metrics();
                if let Some(cache) = prefix_cache.as_ref() {
                    tracing::debug!(
                        prefix_cache = ?cache.snapshot(),
                        prefix_cache_hit_tokens = completed.stream.prefix_cache_hit_tokens(),
                        "DFlash2 prefix cache request summary"
                    );
                }
                worker_counters.record(&metrics);
                if let Some(diagnostic) = token_id_diagnostic.as_mut() {
                    diagnostic.record(
                        completed.request_id.0,
                        completed.stream.prompt_token_ids(),
                        completed.stream.published_token_ids(),
                        outcome.cancelled,
                        outcome.failure.as_deref(),
                        serde_json::to_value(&metrics).unwrap_or_default(),
                    );
                }
                if metrics.ragged_linear_windows > 0 {
                    worker_ragged_linear
                        .requests_with_ragged_windows
                        .fetch_add(1, Ordering::Relaxed);
                }
                worker_ragged_linear
                    .tree_to_ragged_switches
                    .fetch_add(metrics.tree_to_ragged_switches as u64, Ordering::Relaxed);
                worker_ragged_linear
                    .ragged_to_tree_switches
                    .fetch_add(metrics.ragged_to_tree_switches as u64, Ordering::Relaxed);
                if let Some(error) = outcome.failure {
                    tracing::error!(
                        target: "ironmlx::dflash2",
                        request_id = completed.request_id.0,
                        batch_width,
                        error,
                        metrics = %serde_json::to_string(&metrics).unwrap_or_default(),
                        "DFlash2 request failed"
                    );
                } else {
                    tracing::info!(
                        target: "ironmlx::dflash2",
                        request_id = completed.request_id.0,
                        batch_width,
                        cancelled = outcome.cancelled,
                        metrics = %serde_json::to_string(&metrics).unwrap_or_default(),
                        "DFlash2 request completed"
                    );
                }
                worker_in_flight.fetch_sub(1, Ordering::Release);
                worker_active.store(active.len() as u64, Ordering::Relaxed);
            }
            DFlash2StepDiagnostic::mark(&mut step_diagnostic, "complete");
            DFlash2StepDiagnostic::finish(&mut step_diagnostic);
        }
    });

    DFlash2ActorHandle {
        cmd_tx,
        in_flight,
        capacity,
        b_max,
        runtime_usage,
        b_active,
        b_queued,
        background_paused,
        background_preemptions,
        background_resumes,
        admit_count,
        batch_count,
        admission_queue_full_count,
        memory_budget_exceeded_count,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
        kv_cache_logical_cap_tokens,
        kv_cache_resident_cap_tokens,
        kv_cache_budget_policy,
        windows,
        drafted_tokens,
        accepted_draft_tokens,
        rollback_count,
        ordinary_windows,
        tree_windows,
        tree_drafted_nodes,
        tree_fallback_linear_windows,
        draft_budget_changes,
        current_draft_budget,
        latest_adaptive_acceptance_ewma_bits,
        tensor_batch_windows,
        tensor_batch_divergent_splits,
        tensor_batch_groups_created,
        tensor_batch_width_limit: tensor_batch_max_width,
        tensor_batch_max_width: tensor_batch_observed_max_width,
        sampled_requests,
        exact_sampling_windows,
        exact_acceptance_draws,
        exact_residual_corrections,
        exact_bonus_samples,
        sampling_us,
        draft_build_us,
        draft_schedule_us,
        verify_build_us,
        projection_build_us,
        verify_schedule_us,
        host_sync_us,
        rollback_us,
        window_us,
        prefill_us,
        generation_us,
        verify_profile,
        prefix_fingerprint: health_prefix_fingerprint,
        latest_generation_tps_bits,
        latest_acceptance_rate_bits,
        peak_memory_bytes,
        prefix_cache_enabled: prefix_cache_max_bytes.is_some(),
        prefix_cache_max_bytes,
        prefix_cache_entries,
        prefix_cache_bytes,
        prefix_cache_hits,
        prefix_cache_misses,
        prefix_cache_saves,
        prefix_cache_evictions,
        prefix_cache_hit_tokens,
        ragged_linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_budget::KvBudgetPolicy;
    use ironmlx_core::sampler::Sampler;

    fn test_request() -> GenerateRequest {
        GenerateRequest {
            priority: Default::default(),
            prompt_ids: vec![1],
            max_new_tokens: 1,
            sampler: Sampler::greedy(),
            stop_token_ids: Vec::new(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 1,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248_056,
            constraint: None,
        }
    }

    const WIDTHS_1_TO_8: u64 = 0b1_1111_1110;

    #[test]
    fn ragged_linear_selection_needs_two_rows_and_never_waits() {
        assert!(select_ragged_linear_rows(&[], &[], 4).is_empty());
        // A single row at a window boundary keeps its own (tree) path.
        assert!(select_ragged_linear_rows(&[(3, 7, WIDTHS_1_TO_8)], &[], 4).is_empty());
        assert_eq!(
            select_ragged_linear_rows(&[(0, 7, WIDTHS_1_TO_8), (2, 7, WIDTHS_1_TO_8)], &[], 4),
            vec![0, 2]
        );
        // Width limit 1 (b_max 1) never batches.
        assert!(
            select_ragged_linear_rows(&[(0, 7, WIDTHS_1_TO_8), (1, 7, WIDTHS_1_TO_8)], &[], 1)
                .is_empty()
        );
    }

    #[test]
    fn ragged_linear_selection_caps_width_and_keeps_current_members() {
        let candidates = (0..6)
            .map(|index| (index, 7, WIDTHS_1_TO_8))
            .collect::<Vec<_>>();
        assert_eq!(
            select_ragged_linear_rows(&candidates, &[], 4),
            vec![0, 1, 2, 3]
        );
        // Members of the current group are kept first so the batched cache
        // can be reused; free slots go to the earliest other rows.
        assert_eq!(
            select_ragged_linear_rows(&candidates, &[2, 4, 5], 4),
            vec![0, 2, 4, 5]
        );
        assert_eq!(
            select_ragged_linear_rows(&candidates, &[1, 3], 2),
            vec![1, 3]
        );
    }

    #[test]
    fn ragged_linear_selection_respects_draft_len_and_qualified_widths() {
        // Rows with another draft length are not mixed into the batch.
        assert_eq!(
            select_ragged_linear_rows(
                &[
                    (0, 7, WIDTHS_1_TO_8),
                    (1, 3, WIDTHS_1_TO_8),
                    (2, 7, WIDTHS_1_TO_8)
                ],
                &[],
                4
            ),
            vec![0, 2]
        );
        // Width 3 unqualified: three candidates batch as two.
        let no_three = WIDTHS_1_TO_8 & !(1 << 3);
        assert_eq!(
            select_ragged_linear_rows(
                &[(0, 7, no_three), (1, 7, no_three), (2, 7, no_three)],
                &[],
                4
            ),
            vec![0, 1]
        );
        // No qualified batch width: every row keeps its own path.
        assert!(select_ragged_linear_rows(&[(0, 7, 1 << 1), (1, 7, 1 << 1)], &[], 4).is_empty());
    }

    #[test]
    fn ragged_group_charge_is_rows_times_cap_and_released_on_drop() {
        let budget = BudgetState::with_soft_limit(10_000, 100, 100, KvBudgetPolicy::FullResident);
        let cost = DFlash2TargetCacheCost {
            bytes_per_token: 10,
            fixed_bytes_per_sequence: 100,
        };
        let rows = reserve_dflash2_request_memory(&budget, cost, 40, 0, &AtomicU64::new(0))
            .expect("row admission fits");
        let group = reserve_ragged_group_memory(&budget, cost, 50, 4, None)
            .expect("group cache fits the budget");
        assert_eq!(group.bytes(), 4 * (50 * 10 + 100));
        assert_eq!(budget.active_bytes(), 500 + 2_400);
        drop(group);
        assert_eq!(budget.active_bytes(), 500);
        drop(rows);
        assert_eq!(budget.active_bytes(), 0);
    }

    #[test]
    fn ragged_group_reservation_fails_without_charging_when_budget_is_short() {
        let budget = BudgetState::with_soft_limit(2_500, 100, 100, KvBudgetPolicy::FullResident);
        let cost = DFlash2TargetCacheCost {
            bytes_per_token: 10,
            fixed_bytes_per_sequence: 100,
        };
        let rows = reserve_dflash2_request_memory(&budget, cost, 40, 0, &AtomicU64::new(0))
            .expect("row admission fits");
        let error = reserve_ragged_group_memory(&budget, cost, 50, 4, None)
            .expect_err("group cache exceeds the soft limit");
        assert!(matches!(
            error.downcast_ref::<SchedulerError>(),
            Some(SchedulerError::MemoryBudgetExceeded {
                active_bytes: 500,
                requested_bytes: 2_400,
                soft_limit_bytes: 2_500,
            })
        ));
        assert_eq!(budget.active_bytes(), 500);
        // The diagnostic limit rejects even when the budget itself has room.
        let roomy = BudgetState::with_soft_limit(1_000_000, 100, 100, KvBudgetPolicy::FullResident);
        assert!(reserve_ragged_group_memory(&roomy, cost, 50, 2, Some(1_000)).is_err());
        assert_eq!(roomy.active_bytes(), 0);
        assert!(reserve_ragged_group_memory(&roomy, cost, 50, 1, Some(1_000)).is_ok());
        assert_eq!(roomy.active_bytes(), 0);
        drop(rows);
    }

    fn governor_telemetry(usage: usize) -> crate::core::process_memory::MemoryTelemetry {
        const GIB: usize = 1 << 30;
        crate::core::process_memory::MemoryTelemetry {
            total_ram_bytes: 32 * GIB,
            phys_footprint_bytes: Some(usage),
            vm: Some(crate::core::process_memory::HostVmStatistics {
                free_bytes: 8 * GIB,
                inactive_bytes: 2 * GIB,
                active_bytes: 8 * GIB,
                wired_bytes: 8 * GIB,
            }),
            mlx_active_bytes: Some(usage.saturating_sub(1)),
            mlx_cache_bytes: Some(0),
            metal_limit_bytes: Some(24 * GIB),
        }
    }

    #[test]
    fn ragged_group_headroom_refreshes_stale_telemetry_before_reserving() {
        use crate::core::process_memory::{
            MemoryGovernorConfig, PressureLevel, ProcessMemoryGovernor,
        };
        const GIB: usize = 1 << 30;
        let governor = Arc::new(
            ProcessMemoryGovernor::new(MemoryGovernorConfig {
                poll_interval: std::time::Duration::from_millis(1),
                telemetry_stale_after: std::time::Duration::from_millis(2),
                ..MemoryGovernorConfig::default()
            })
            .expect("valid governor config"),
        );
        governor.update(governor_telemetry(4 * GIB));
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Stale telemetry: the refresh makes the reservation authoritative,
        // so it succeeds and leaves pressure normal for later admissions.
        let reservation = reserve_ragged_group_headroom(&governor, GIB, |g| {
            g.update(governor_telemetry(4 * GIB))
        })
        .expect("fresh sample with headroom reserves");
        assert_eq!(reservation.bytes(), GIB);
        reservation.commit();
        assert_eq!(governor.snapshot().pressure_level, PressureLevel::Normal);
        assert_eq!(governor.snapshot().reserved_bytes, 0);
        // Under pressure the group falls back without reserving anything.
        for _ in 0..3 {
            governor.update(governor_telemetry(23 * GIB));
        }
        assert!(reserve_ragged_group_headroom(&governor, GIB, |g| {
            g.update(governor_telemetry(23 * GIB))
        })
        .is_err());
        assert_eq!(governor.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn abandoned_queued_request_releases_in_flight_capacity_once() {
        let in_flight = AtomicUsize::new(1);
        let (live_tx, _live_rx) = oneshot::channel::<Result<AdmitReply>>();
        assert!(!discard_abandoned_queued_request(&live_tx, &in_flight));
        assert_eq!(in_flight.load(Ordering::Acquire), 1);

        let (abandoned_tx, abandoned_rx) = oneshot::channel::<Result<AdmitReply>>();
        drop(abandoned_rx);
        assert!(discard_abandoned_queued_request(&abandoned_tx, &in_flight));
        assert_eq!(in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn pending_queue_prunes_abandoned_request_without_waiting_for_active_capacity() {
        let in_flight = AtomicUsize::new(2);
        let b_queued = AtomicU64::new(2);
        let (live_tx, _live_rx) = oneshot::channel::<Result<AdmitReply>>();
        let (abandoned_tx, abandoned_rx) = oneshot::channel::<Result<AdmitReply>>();
        drop(abandoned_rx);
        let mut pending = VecDeque::from([
            DFlash2Command::Admit {
                request: test_request(),
                reply_tx: live_tx,
            },
            DFlash2Command::Admit {
                request: test_request(),
                reply_tx: abandoned_tx,
            },
        ]);

        assert_eq!(
            prune_abandoned_pending_requests(&mut pending, &in_flight, &b_queued),
            1
        );
        assert_eq!(pending.len(), 1);
        assert!(!pending
            .front()
            .expect("live request remains")
            .reply_is_closed());
        assert_eq!(in_flight.load(Ordering::Acquire), 1);
        assert_eq!(b_queued.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn pending_queue_is_fifo_within_priority_and_foreground_first() {
        let mut pending = VecDeque::new();
        let mut receivers = Vec::new();
        for (token, priority) in [
            (10, RequestPriority::Background),
            (20, RequestPriority::Foreground),
            (30, RequestPriority::Background),
            (40, RequestPriority::Foreground),
        ] {
            let (reply_tx, reply_rx) = oneshot::channel();
            let mut request = test_request();
            request.prompt_ids[0] = token;
            request.priority = priority;
            push_pending_by_priority(&mut pending, DFlash2Command::Admit { request, reply_tx });
            receivers.push(reply_rx);
        }

        assert_eq!(
            pending
                .iter()
                .map(|command| match command {
                    DFlash2Command::Admit { request, .. } => request.prompt_ids[0],
                })
                .collect::<Vec<_>>(),
            vec![20, 40, 10, 30]
        );
        drop(receivers);
    }

    #[test]
    fn request_memory_charge_rejects_aggregate_overcommit_and_releases_on_drop() {
        let budget = BudgetState::with_soft_limit(1_000, 100, 100, KvBudgetPolicy::FullResident);
        let cost = DFlash2TargetCacheCost {
            bytes_per_token: 10,
            fixed_bytes_per_sequence: 100,
        };
        let rejected = AtomicU64::new(0);
        let first = reserve_dflash2_request_memory(&budget, cost, 40, 0, &rejected)
            .expect("first request fits");
        assert_eq!(first.bytes(), 500);
        assert_eq!(budget.active_bytes(), 500);

        let error = reserve_dflash2_request_memory(&budget, cost, 50, 0, &rejected)
            .expect_err("aggregate charge exceeds soft limit");
        assert!(matches!(
            error.downcast_ref::<SchedulerError>(),
            Some(SchedulerError::MemoryBudgetExceeded {
                active_bytes: 500,
                requested_bytes: 600,
                soft_limit_bytes: 1_000,
            })
        ));
        assert_eq!(rejected.load(Ordering::Relaxed), 1);
        assert_eq!(budget.active_bytes(), 500);

        drop(first);
        assert_eq!(budget.active_bytes(), 0);
        let second = reserve_dflash2_request_memory(&budget, cost, 50, 0, &rejected)
            .expect("released charge makes room");
        assert_eq!(budget.active_bytes(), 600);
        drop(second);
        assert_eq!(budget.active_bytes(), 0);
    }

    #[test]
    fn foreground_replacement_allowance_preserves_truthful_resident_charge() {
        let budget = BudgetState::with_soft_limit(1_000, 100, 100, KvBudgetPolicy::FullResident);
        let cost = DFlash2TargetCacheCost {
            bytes_per_token: 10,
            fixed_bytes_per_sequence: 100,
        };
        let rejected = AtomicU64::new(0);
        let paused_background = reserve_dflash2_request_memory(&budget, cost, 90, 0, &rejected)
            .expect("background request fills the logical budget");
        assert_eq!(paused_background.bytes(), 1_000);

        let foreground =
            reserve_dflash2_request_memory(&budget, cost, 90, paused_background.bytes(), &rejected)
                .expect("foreground may replace the paused logical working set");
        assert_eq!(budget.active_bytes(), 2_000);
        assert_eq!(rejected.load(Ordering::Relaxed), 0);

        drop(foreground);
        drop(paused_background);
        assert_eq!(budget.active_bytes(), 0);
    }
}
