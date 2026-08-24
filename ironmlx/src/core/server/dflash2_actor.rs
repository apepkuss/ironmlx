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
    DFlash2PrefixCache, DFlash2TensorBatchCache, DFlash2TextGenerationStream,
};
use crate::core::generate::{GenerateEvent, GenerateRequest};
use crate::core::memory_budget::BudgetState;
use crate::core::scheduler::{DenseVlMethods, RequestId, SchedulerError, StepEvent};
use crate::core::server::scheduler_actor::AdmitReply;
use crate::core::{Model, Tokenizer};
use crate::models::dflash2::{DFlash2DraftModel, DFlash2Target, DFlash2TargetCacheCost};
use crate::Result;

struct ActiveDFlash2Request<'m, M>
where
    M: DFlash2Target,
{
    request_id: RequestId,
    event_tx: mpsc::UnboundedSender<StepEvent>,
    stream: DFlash2TextGenerationStream<'m, M>,
    _memory_charge: DFlash2MemoryCharge,
}

#[derive(Default)]
struct DFlash2StepOutcome {
    event: Option<GenerateEvent>,
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
    memory_budget_exceeded_count: &AtomicU64,
) -> Result<DFlash2MemoryCharge> {
    let requested_bytes = cache_cost.request_bytes(token_cap);
    if let Err((active_bytes, requested_bytes, soft_limit_bytes)) =
        budget_state.try_admit(requested_bytes)
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
    sampled_requests: Arc<AtomicU64>,
    exact_sampling_windows: Arc<AtomicU64>,
    exact_acceptance_draws: Arc<AtomicU64>,
    exact_residual_corrections: Arc<AtomicU64>,
    exact_bonus_samples: Arc<AtomicU64>,
    sampling_us: Arc<AtomicU64>,
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
    pub(crate) b_max: usize,
    pub(crate) admission_deadline: std::time::Duration,
    pub(crate) tensor_batch_max_width: usize,
    pub(crate) admission_queue_max: usize,
    pub(crate) effective_cap_max: usize,
    pub(crate) budget_state: BudgetState,
    pub(crate) cache_cost: DFlash2TargetCacheCost,
    pub(crate) prefix_cache_max_bytes: Option<usize>,
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
        b_max,
        admission_deadline,
        tensor_batch_max_width,
        admission_queue_max,
        effective_cap_max,
        budget_state,
        cache_cost,
        prefix_cache_max_bytes,
    } = config;
    assert!(b_max > 0, "DFlash2 actor requires b_max > 0");
    assert!(
        (1..=b_max).contains(&tensor_batch_max_width),
        "DFlash2 tensor batch width limit must be in 1..=b_max"
    );
    let capacity = admission_queue_max.saturating_add(b_max);
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
    let in_flight = Arc::new(AtomicUsize::new(0));
    let b_active = Arc::new(AtomicU64::new(0));
    let b_queued = Arc::new(AtomicU64::new(0));
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
        "dflash2-prefix-v1:draft-dtype={};draft-hidden={};draft-layer-count={};target-layers={:?};sliding-window={};block-size={}",
        draft.config().dtype,
        draft.config().hidden_size,
        draft.config().num_hidden_layers,
        draft.config().dflash_config.target_layer_ids,
        draft.config().sliding_window,
        block_size,
    );
    let worker_in_flight = Arc::clone(&in_flight);
    let worker_active = Arc::clone(&b_active);
    let worker_queued = Arc::clone(&b_queued);
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
        sampled_requests: Arc::clone(&sampled_requests),
        exact_sampling_windows: Arc::clone(&exact_sampling_windows),
        exact_acceptance_draws: Arc::clone(&exact_acceptance_draws),
        exact_residual_corrections: Arc::clone(&exact_residual_corrections),
        exact_bonus_samples: Arc::clone(&exact_bonus_samples),
        sampling_us: Arc::clone(&sampling_us),
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
        let mut tensor_groups = Vec::<DFlash2TensorGroup>::with_capacity(b_max / 2);
        let mut pending = VecDeque::<DFlash2Command>::new();
        let mut command_channel_open = true;

        loop {
            let forming_empty_batch = active.is_empty();
            let mut admission_window_waited = false;
            while command_channel_open {
                match cmd_rx.try_recv() {
                    Ok(command) => pending.push_back(command),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        command_channel_open = false;
                        break;
                    }
                }
            }
            prune_abandoned_pending_requests(&mut pending, &worker_in_flight, &worker_queued);

            while active.len() < b_max {
                if pending.is_empty() && active.is_empty() && command_channel_open {
                    match cmd_rx.blocking_recv() {
                        Some(command) => pending.push_back(command),
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
                            Ok(command) => pending.push_back(command),
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
                    required_total_tokens,
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
                let stream =
                    match DFlash2TextGenerationStream::new_scheduler_b1_text_only_with_cancellation(
                        &*model,
                        &draft,
                        &tokenizer,
                        request,
                        block_size,
                        prefix_cache
                            .as_mut()
                            .map(|cache| (cache, prefix_fingerprint.as_str())),
                        &|| reply_tx.is_closed(),
                    )
                    .context("initializing DFlash2 actor stream")
                    {
                        Ok(stream) => stream,
                        Err(error) => {
                            let _ = reply_tx.send(Err(error));
                            worker_in_flight.fetch_sub(1, Ordering::Release);
                            continue;
                        }
                    };
                worker_runtime_usage.record_prefix_cache_lookup(
                    stream.metrics().prompt_tokens as u64,
                    stream.prefix_cache_hit_tokens() as u64,
                );
                worker_prefix_cache_counters
                    .hit_tokens
                    .fetch_add(stream.prefix_cache_hit_tokens() as u64, Ordering::Relaxed);
                worker_prefix_cache_counters.publish(prefix_cache.as_ref());
                // Prefill and first-token materialization completed while
                // constructing the stream. Mark the shared weights warm before
                // admitting another sequence; otherwise a second begin() would
                // wait on the same blocking worker.
                cold.commit();
                governor_reservation.commit();
                governor.refresh_process();

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
                    event_tx,
                    stream,
                    _memory_charge: memory_charge,
                });
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
            let mut outcomes = (0..active.len())
                .map(|_| DFlash2StepOutcome::default())
                .collect::<Vec<_>>();

            for (index, request) in active.iter_mut().enumerate() {
                if request.event_tx.is_closed() {
                    outcomes[index].cancelled = true;
                    outcomes[index].finished = true;
                    continue;
                }
                match request.stream.next_token_deferred() {
                    Ok(Some(event)) => {
                        outcomes[index].finished = event.finish_reason.is_some();
                        outcomes[index].event = Some(event);
                    }
                    Ok(None) => outcomes[index].finished = true,
                    Err(error) => {
                        outcomes[index].finished = true;
                        outcomes[index].failure = Some(format!("{error:#}"));
                    }
                }
            }

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
                if group_key.is_some() && positions.iter().all(|&index| keys[index] == group_key) {
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

            let mut ready = BTreeMap::new();
            for (index, key) in keys.iter().enumerate() {
                if !claimed[index] && !outcomes[index].finished {
                    if let Some(key) = key {
                        ready.entry(*key).or_insert_with(Vec::new).push(index);
                    }
                }
            }
            for compatible_indices in ready.values() {
                for indices in compatible_indices.chunks(tensor_batch_max_width) {
                    let result = if indices.len() >= 2 {
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

            for (index, outcome) in outcomes.iter_mut().enumerate() {
                if outcome.failure.is_some() {
                    continue;
                }
                let Some(event) = outcome.event.as_ref() else {
                    continue;
                };
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
                }
            }

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_budget::KvBudgetPolicy;
    use crate::core::sampler::Sampler;

    fn test_request() -> GenerateRequest {
        GenerateRequest {
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
    fn request_memory_charge_rejects_aggregate_overcommit_and_releases_on_drop() {
        let budget = BudgetState::with_soft_limit(1_000, 100, 100, KvBudgetPolicy::FullResident);
        let cost = DFlash2TargetCacheCost {
            bytes_per_token: 10,
            fixed_bytes_per_sequence: 100,
        };
        let rejected = AtomicU64::new(0);
        let first = reserve_dflash2_request_memory(&budget, cost, 40, &rejected)
            .expect("first request fits");
        assert_eq!(first.bytes(), 500);
        assert_eq!(budget.active_bytes(), 500);

        let error = reserve_dflash2_request_memory(&budget, cost, 50, &rejected)
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
        let second = reserve_dflash2_request_memory(&budget, cost, 50, &rejected)
            .expect("released charge makes room");
        assert_eq!(budget.active_bytes(), 600);
        drop(second);
        assert_eq!(budget.active_bytes(), 0);
    }
}
