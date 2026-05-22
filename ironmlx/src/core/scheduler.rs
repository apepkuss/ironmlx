//! B1-p2.3a scheduler skeleton — per-request state + fixed-capacity admit/evict.
//!
//! Subsequent sub-phases extend this module:
//! - B1-p2.3b adds `Scheduler::step()` driving `model.forward_on([B, 1], ...)`
//!   and the HTTP server refactor.
//! - B1-p2.3c adds per-row KV cache offset tracking + per-row decode mask.
//! - B1-p2.3d adds an admission queue + preemption when `b_max` is full.
//! - B1-p2.3e adds per-row sampler invocation (temperature/top_k per row).
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md`.

use std::collections::HashMap;
use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use mlx::{Array, Dtype};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;

/// Typed scheduler-side errors that need HTTP-level discrimination.
///
/// Anyhow remains the default error type for internal Scheduler paths
/// (out-of-memory, prompt parsing, MLX failures, etc.). `SchedulerError`
/// only enumerates errors the HTTP server needs to map to non-400
/// responses. Wrap into anyhow via `anyhow::Error::new(SchedulerError::...)`
/// at the emit site; HTTP handlers downcast with
/// `err.downcast_ref::<SchedulerError>()`.
///
/// Replaces the pre-3e.3 string-match `err.to_string().contains("admission
/// queue full")` pattern (spec §9 R3 acknowledged-fragile).
#[derive(Error, Debug)]
pub enum SchedulerError {
    /// Admission queue rejected a request because the queue was already
    /// at `capacity`. Maps to HTTP 503 + Retry-After.
    #[error("admission queue full: capacity={capacity} reached")]
    QueueFull { capacity: usize },

    /// Request's `prompt_len + max_new_tokens` exceeds the server's
    /// effective cap_max (the smaller of `--max-cache-cap` CLI flag and
    /// the model's `max_position_embeddings`). Maps to HTTP 413
    /// Payload Too Large. B1-p2.3f.
    #[error("request too large: needs cap={needed} but server max_cache_cap={max}")]
    RequestTooLarge { needed: usize, max: usize },

    /// Admission gate: request's KV cache bytes plus active bytes would
    /// exceed the soft limit (85% of total budget). Maps to HTTP 503
    /// + Retry-After (T2). B1-p2.5.
    #[error(
        "memory budget exceeded: active {active_bytes} + requested {requested_bytes} > \
         soft limit {soft_limit_bytes}"
    )]
    MemoryBudgetExceeded {
        active_bytes: usize,
        requested_bytes: usize,
        soft_limit_bytes: usize,
    },
}

use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_decode_position_ids,
    build_per_row_decode_mask, build_position_ids, build_position_ids_batched,
    build_position_ids_vl, build_position_ids_vl_batched, count_image_pad, slice_logits_row,
    slice_pos_ids_axis2, slice_vision_embeds_rows, GenerateRequest,
};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::nn::LayerCache;

/// T4.5: process-wide once-only guard for the first-eval diagnostic span.
///
/// On the FIRST `prefill_admitted` call with an active P5h trace, the
/// `model_prefill_forward` body incurs cold-start cost from MLX JIT
/// compilation + Metal pipeline cache population that subsequent calls
/// amortize away. We emit a parallel `first_eval_amortized_cost`
/// diagnostic span (parent = root span, span_kind = "diagnostic") so the
/// T5 aggregator can report this cold-start cost as a separate column
/// without contaminating the exclusive parent-child tree (sum-to-root
/// invariants exclude diagnostic spans per spec § 2.5a).
///
/// `OnceLock::set(())` returns `Ok(())` on the first caller; concurrent
/// racers see `Err(())` and skip — race-safe single emission per process.
#[cfg(feature = "p5h-profile")]
static FIRST_EVAL_AMORTIZED_COST_FIRED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Convenience alias — avoids `clippy::type_complexity` on Vec<Option<&[...]>> sites.
type GridThwSlice<'a> = Option<&'a [(i32, i32, i32)]>;

/// Extension trait for VL-capable models, intentionally NOT part of `core::Model`
/// (per P5 spec §3.1 — VL methods stay inherent / extension-trait-only).
///
/// Only `Qwen35Model` implements this. `Scheduler<M>` methods that call VL
/// code paths (vision tower + cross-modal scatter + VL prefill) require
/// `M: Model + DenseVlMethods`; instantiating such methods with a non-VL model
/// (e.g., the future `Qwen35MoeModel`) is a compile-time error.
pub trait DenseVlMethods {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batched_prefill_vl(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&mlx::Array>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;

    fn compute_vision_embeds(
        &self,
        pixel_values: &mlx::Array,
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_chunk(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;
}

impl DenseVlMethods for crate::models::qwen3_5::Qwen35Model {
    fn batched_prefill_vl(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&mlx::Array>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        crate::models::qwen3_5::Qwen35Model::batched_prefill_vl(
            self,
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            per_row_pixel_values,
            per_row_grid_thw,
            image_token_id,
            cache,
            target,
        )
    }

    fn compute_vision_embeds(
        &self,
        pixel_values: &mlx::Array,
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        crate::models::qwen3_5::Qwen35Model::compute_vision_embeds(
            self,
            pixel_values,
            grid_thw,
            target,
        )
    }

    fn forward_vl_chunk(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        crate::models::qwen3_5::Qwen35Model::forward_vl_chunk(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )
    }
}

/// Opaque, monotonically-increasing identifier for an admitted request.
///
/// Never reused after the request is evicted — admitting another request into
/// the same `row_idx` produces a new `RequestId` value. This eliminates
/// stale-id bugs at the cost of a 64-bit counter (~10^19 IDs before overflow;
/// practically infinite).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// Scheduler lifecycle phase. The state machine is `Idle → Admitting →
/// Decoding → Finished → Idle`.
///
/// Transitions are driven by the scheduler methods:
/// - `admit()` from `Idle` → `Admitting`.
/// - `admit()` from `Admitting` → `Admitting`.
/// - `admit()` from `Decoding` → `Decoding` (mid-batch admit, 3c-3;
///   caller is responsible for prefilling the new slot via `admit_mid`).
/// - `evict()` from `Admitting` + `active_count==0` → `Idle`.
/// - `evict()` from `Decoding` + `active_count==0` → `Finished` (3c-3).
/// - `prefill_admitted()` from `Idle`/`Admitting` → `Decoding`.
/// - `step()` from `Decoding`: stays `Decoding` while ≥1 row unfinished,
///   transitions to `Finished` when all active rows are `finished`.
/// - `gc_finished_rows()` from `Decoding` + `active_count==0` → `Finished`
///   (3c-3, idempotent with `step`'s end-of-loop transition).
/// - `evict_all()` from `Decoding`/`Finished` → `Idle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Admitting,
    Decoding,
    Finished,
}

/// One per-row event emitted by [`Scheduler::step`].
///
/// Only rows that were not yet `finished` at the start of the step appear
/// in the event list. The step in which a row first transitions to
/// `finished` produces an event with `finish_reason = Some("stop"|"length")`.
/// Subsequent steps never emit anything for that row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepEvent {
    pub id: RequestId,
    pub token: u32,
    pub finish_reason: Option<&'static str>,
}

/// State shared across the three `admit_mid_*` calls that make up a
/// chunked mid-batch admit (B1-p2.3c+). Built by [`Scheduler::admit_mid_begin`],
/// mutated by [`Scheduler::admit_mid_chunk`] calls, consumed by
/// [`Scheduler::admit_mid_finalize`]. The caller
/// (`SchedulerActor::driver_loop`'s `handle_admit_mid_chunked`) owns this
/// between calls and interleaves `Scheduler::step` between chunks so
/// active rows continue emitting tokens during a long-prompt mid-batch
/// admit.
///
/// All fields except `request_id` are `pub(crate)` so the chunk loop
/// in `scheduler.rs` itself can read/write them; HTTP / actor code
/// treats the handle as opaque (only inspects `request_id`).
#[doc(hidden)]
pub struct AdmitMidHandle {
    /// Slot the admit reserved at `admit_mid_begin`.
    pub request_id: RequestId,
    pub(crate) row_idx: usize,
    /// Full prompt token ids cloned from `RequestState` so we can index
    /// per-chunk without re-borrowing slot state across calls.
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) prompt_len: i32,
    /// Per-chunk max token count; equals `req.prefill_chunk_size.max(1)`
    /// at construction, unless the VL R6 fallback forces single-chunk
    /// (image_pad straddles a chunk boundary — spec §4.6 NG7).
    pub(crate) chunk_size: i32,
    pub(crate) chunk_start: i32,
    /// B=1 temp KV cache; `temp_cache.offsets[0]` advances from `0` to
    /// `prompt_len` across the chunk loop.
    pub(crate) temp_cache: Vec<crate::nn::LayerCache>,
    pub(crate) is_vl: bool,
    pub(crate) image_token_id: i32,
    /// Pre-computed `[3, 1, prompt_len]` MRoPE position ids for the
    /// full prompt — sliced per chunk inside `admit_mid_chunk` to
    /// avoid rebuilding on each iteration. For VL it incorporates
    /// `image_spatial_merge_size` + `image_grid_thw` (consumed in
    /// `admit_mid_begin` when building this Array — no need to carry
    /// the inputs forward).
    pub(crate) position_ids_full: Array,
    /// Pre-computed full-prompt vision embeddings
    /// `[N_image_pad_total, hidden]` — only populated for VL requests.
    /// Each chunk slices the rows it consumes; `image_pad_consumed`
    /// tracks the running offset. `pixel_values` + `image_grid_thw`
    /// are consumed by `compute_vision_embeds` in `admit_mid_begin`
    /// and not carried forward in the handle.
    pub(crate) vision_embeds_full: Option<Array>,
    pub(crate) image_pad_consumed: usize,
    /// Last chunk's `[1, 1, vocab]` logits, captured only at the final
    /// chunk for first-token sampling in `admit_mid_finalize`.
    pub(crate) last_logits: Option<Array>,
}

/// Returns true if any `image_pad` run in `prompt_ids` would straddle
/// a chunk boundary at `chunk_size`. Used by `admit_mid_begin` to
/// detect the VL v1 fallback condition (spec §4.6 NG7 / §4.7 R6):
/// when an image's `image_pad` tokens span chunks, we'd need per-chunk
/// vision-arg slicing — deferred to v2. v1 forces single-chunk in
/// this case.
fn vl_image_pad_crosses_chunk_boundary(
    prompt_ids: &[u32],
    image_token_id: i32,
    chunk_size: i32,
) -> bool {
    if image_token_id < 0 || chunk_size <= 0 {
        return false;
    }
    let pad = image_token_id as u32;
    let cs = chunk_size as usize;
    let mut in_run = false;
    let mut run_start = 0usize;
    for (i, &t) in prompt_ids.iter().enumerate() {
        if t == pad {
            if !in_run {
                in_run = true;
                run_start = i;
            }
        } else if in_run {
            let run_end = i; // exclusive
            if run_start / cs != (run_end - 1) / cs {
                return true;
            }
            in_run = false;
        }
    }
    if in_run {
        let run_end = prompt_ids.len();
        if run_start / cs != (run_end - 1) / cs {
            return true;
        }
    }
    false
}

/// All per-request state the scheduler tracks. Pre-allocated at admit time
/// and held until eviction.
///
/// Fields are chosen to cover B1-p2.3b–3e needs without a later refactor.
/// VL fields (`pixel_values`, `image_grid_thw`, etc.) are intentionally
/// omitted from 3a — they get added in B1-p2.4 when VL B>1 lands.
#[derive(Debug)]
pub struct RequestState {
    /// Opaque token returned by [`Scheduler::admit`].
    pub id: RequestId,
    /// Position in the scheduler's slot vector (0..b_max). Fixed for the
    /// lifetime of this request — subsequent admits never relocate it.
    /// Used by 3b to index into the batched KV cache and per-row mask
    /// tensors.
    pub row_idx: usize,
    /// Original prompt token ids; copied from `GenerateRequest::prompt_ids`.
    pub prompt_ids: Vec<u32>,
    /// Decode-time tokens produced so far. Empty at admit. 3b pushes one
    /// token per `Scheduler::step()` per row.
    pub generated_tokens: Vec<u32>,
    /// Hard cap on tokens generated beyond the prompt.
    pub max_new_tokens: usize,
    /// Token ids that terminate the stream when produced; copied from
    /// `GenerateRequest::stop_token_ids`.
    pub stop_token_ids: Vec<u32>,
    /// Per-row sampler — cloned from the request's sampler at admit time so
    /// each row owns independent sampler state. Sampler is `Copy` post-3e.2;
    /// PRNG state lives in `Scheduler.prng_state` (centralized) — see
    /// `docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md`.
    pub sampler: Sampler,
    /// Effective KV-cache length for this row: starts at `prompt_ids.len()`
    /// and is incremented by 1 per decode step (3b). Used by 3c to build
    /// the per-row decode mask.
    pub real_len: i32,
    /// `false` at admit; 3b sets `true` on EOS / `max_new_tokens` reached.
    pub finished: bool,
    /// `"stop"` or `"length"` when `finished` is `true`; otherwise `None`.
    pub finish_reason: Option<&'static str>,

    // ─── B1-p2.4: VL fields, carried from GenerateRequest at admit ───
    /// Vision input. `None` for text-only rows. `Array` clone is mlx
    /// reference-counted — cheap. Lives until evict.
    pub pixel_values: Option<Array>,
    /// Per-image `(temporal, height, width)` grid sizes; same len as image
    /// count for this row. `None` ⇔ `pixel_values.is_none()`.
    pub image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    /// Spatial merge factor for image patches → embedding rows. Carried
    /// from `GenerateRequest::image_spatial_merge_size`. Unused if
    /// `pixel_values` is None.
    pub image_spatial_merge_size: i32,
    /// Tokenizer id of `<|image_pad|>`. Carried from
    /// `GenerateRequest::image_token_id`. Unused if `pixel_values` is None.
    pub image_token_id: i32,
    /// Per-request chunk size for chunked mid-batch prefill. Copied from
    /// `GenerateRequest::prefill_chunk_size` at admit time, clamped to i32
    /// and floored at 1. Used by `admit_mid_begin` to initialise
    /// `AdmitMidHandle::chunk_size`.
    pub prefill_chunk_size: i32,
    /// KV cache bytes charged to budget at admit time. Released on
    /// row completion / eviction. B1-p2.5.
    pub kv_bytes_admitted: usize,

    #[cfg(feature = "p5h-profile")]
    #[allow(dead_code)] // read by cloned_active_row_p5h_trace_and_root; set by T0a.6 handler
    pub(crate) p5h_trace: Option<crate::core::p5h::P5hTraceContext>,

    #[cfg(feature = "p5h-profile")]
    #[allow(dead_code)] // read by cloned_active_row_p5h_trace_and_root; set by T0a.6 handler
    pub(crate) p5h_root_span: Option<crate::core::p5h::SpanHandle>,
}

/// Read pre-write per-row offsets from the first Full-attention layer's
/// `KVCache`. Used by [`Scheduler::step`] to construct the per-row decode
/// mask before the forward.
///
/// All Full-attention layers advance their `KVCache.offsets()` in
/// lockstep across decode steps (per-row offsets diverge across rows
/// but NOT across layers for a given row). Any Full layer's offsets
/// view is equivalent — picking the first is arbitrary but consistent.
fn first_full_layer_offsets(cache: &[LayerCache]) -> Result<&[i32]> {
    cache
        .iter()
        .find_map(|c| match c {
            LayerCache::Full(kv) => Some(kv.offsets()),
            _ => None,
        })
        .ok_or_else(|| {
            anyhow!(
                "Scheduler::step: no Full-attention layer in cache; per-row offsets unavailable"
            )
        })
}

/// Fixed-capacity scheduler holding up to `b_max` in-flight requests.
///
/// 3a is single-threaded only — no `Send + Sync` impls. A later sub-phase
/// will decide whether to run the scheduler on the main runtime thread or
/// in `tokio::spawn_blocking`.
pub struct Scheduler<M: Model> {
    b_max: usize,
    slots: Vec<Option<RequestState>>,
    next_id: u64,
    phase: Phase,
    cache: Option<Vec<LayerCache>>,
    poisoned: bool,
    /// Upper bound on `prompt_len + max_new_tokens` per request, computed
    /// at boot as `min(cli_max_cache_cap, model.config.max_position_embeddings)`.
    /// `admit` and `admit_mid` reject requests exceeding this with
    /// [`SchedulerError::RequestTooLarge`]. B1-p2.3f.
    effective_cap_max: usize,
    /// Centralized per-row PRNG state, shape `[b_max, 2]` u32.
    /// Row `i` holds the PRNG key for slot `i`. Initialized to zeros;
    /// `init_row_prng` seeds a row on every new admission. (B1-p2.3e.2)
    pub(crate) prng_state: Array,
    /// Runtime memory budget tracker. Charged at admit; released on slot
    /// clear. Also holds the soft_limit for admission gating. (B1-p2.5)
    pub(crate) budget_state: crate::core::memory_budget::BudgetState,
    /// Snapshot of the model's memory-budget metadata, used to compute
    /// per-request KV byte cost in admit. (B1-p2.5)
    pub(crate) meta: crate::core::memory_budget::ModelMeta,
    /// Count of admits rejected by the memory budget gate. Used by T3
    /// /healthz. (B1-p2.5)
    pub(crate) memory_budget_exceeded_count: std::sync::Arc<std::sync::atomic::AtomicU64>,
    _marker: PhantomData<fn(&M) -> ()>,
}

impl<M: Model> std::fmt::Debug for Scheduler<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("b_max", &self.b_max)
            .field("slots", &self.slots)
            .field("next_id", &self.next_id)
            .field("phase", &self.phase)
            .field("cache_layers", &self.cache.as_ref().map(|c| c.len()))
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl<M: Model> Scheduler<M> {
    /// Construct a scheduler with `b_max` pre-allocated slots, all `None`.
    /// `effective_cap_max` is the hard upper bound on per-request
    /// `prompt_len + max_new_tokens` — admit gates reject requests beyond
    /// this with [`SchedulerError::RequestTooLarge`] (HTTP 413 downstream).
    ///
    /// Validates startup memory budget via `validate_startup_budget`; returns
    /// `Err(MemoryBudgetError)` if `b_max × effective_cap_max × per_token_bytes`
    /// exceeds available system RAM. (B1-p2.5)
    ///
    /// **Thread affinity**: `Array::zeros` for `prng_state` is allocated on
    /// the current thread's Metal Stream. Call this on the thread that will
    /// own the Scheduler. For actor spawning, prefer [`Scheduler::new_with_state`]
    /// so budget validation can happen on the calling thread while Array
    /// allocation happens on the worker thread.
    pub fn new(
        b_max: usize,
        effective_cap_max: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<Self, crate::core::memory_budget::MemoryBudgetError> {
        let budget_state =
            crate::core::memory_budget::validate_startup_budget(b_max, effective_cap_max, &meta)?;
        let memory_budget_exceeded_count =
            std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        Self::new_with_state(
            b_max,
            effective_cap_max,
            budget_state,
            memory_budget_exceeded_count,
            meta,
        )
    }

    /// Variant that accepts pre-created [`BudgetState`] and
    /// `memory_budget_exceeded_count` Arc. Used by [`spawn_scheduler_actor`]
    /// for thread-affinity-correct construction: budget validation happens on
    /// the calling thread; Array allocation (prng_state) happens inside
    /// `spawn_blocking` on the worker thread that will own the Scheduler.
    ///
    /// Callers are responsible for ensuring `budget_state` was produced by
    /// [`validate_startup_budget`] with consistent `b_max` / `effective_cap_max`.
    pub fn new_with_state(
        b_max: usize,
        effective_cap_max: usize,
        budget_state: crate::core::memory_budget::BudgetState,
        memory_budget_exceeded_count: std::sync::Arc<std::sync::atomic::AtomicU64>,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<Self, crate::core::memory_budget::MemoryBudgetError> {
        let mut slots = Vec::with_capacity(b_max);
        for _ in 0..b_max {
            slots.push(None);
        }
        // Initialize prng_state to zeros [b_max, 2] u32. Each slot's row is
        // seeded via init_row_prng on admission.
        // IMPORTANT: Array::zeros binds to the current thread's Metal Stream.
        // This method must be called on the thread that will drive the Scheduler.
        let prng_state =
            Array::zeros(&[b_max as i32, 2_i32][..], Dtype::Uint32).expect("prng_state zeros");
        Ok(Self {
            b_max,
            slots,
            next_id: 0,
            phase: Phase::Idle,
            cache: None,
            poisoned: false,
            effective_cap_max,
            prng_state,
            budget_state,
            meta,
            memory_budget_exceeded_count,
            _marker: PhantomData,
        })
    }

    /// Seed the PRNG state for `row_idx` from `seed`.
    ///
    /// Uses the to_vec + host-replace + try_from pattern to avoid
    /// `slice_update` (T0 measured at 274 ms/call in hot path).
    /// Called only on admission (not in hot path) so ~80 µs to_vec cost is OK.
    fn init_row_prng(&mut self, row_idx: usize, seed: u64) -> Result<()> {
        let b_max = self.prng_state.shape().as_slice()[0] as usize;
        anyhow::ensure!(
            row_idx < b_max,
            "init_row_prng: row_idx={row_idx} >= b_max={b_max}"
        );
        let key = mlx::random::key(seed)?;
        let key_host: Vec<u32> = key.to_vec()?;
        let mut new_host: Vec<u32> = self.prng_state.to_vec()?;
        new_host[row_idx * 2] = key_host[0];
        new_host[row_idx * 2 + 1] = key_host[1];
        self.prng_state = (new_host.as_slice(), &[b_max as i32, 2_i32][..]).try_into()?;
        Ok(())
    }

    /// Write an updated row key back into `prng_state` after sampling.
    ///
    /// Same to_vec + host-replace + try_from pattern as `init_row_prng`.
    fn write_row_prng(&mut self, row_idx: usize, row_key: &Array) -> Result<()> {
        let b_max = self.prng_state.shape().as_slice()[0] as usize;
        anyhow::ensure!(
            row_idx < b_max,
            "write_row_prng: row_idx={row_idx} >= b_max={b_max}"
        );
        let key_host: Vec<u32> = row_key.to_vec()?;
        let mut host: Vec<u32> = self.prng_state.to_vec()?;
        host[row_idx * 2] = key_host[0];
        host[row_idx * 2 + 1] = key_host[1];
        self.prng_state = (host.as_slice(), &[b_max as i32, 2_i32][..]).try_into()?;
        Ok(())
    }

    /// Maximum concurrent in-flight requests this scheduler can hold.
    pub fn b_max(&self) -> usize {
        self.b_max
    }

    /// Admit a new request. Walks `slots` for the first `None`, fills it
    /// with a freshly-constructed `RequestState`, and returns a new
    /// monotonically-increasing [`RequestId`]. Returns `Err` if the
    /// scheduler is full (B1-p2.3d will replace this with queueing).
    ///
    /// The request's sampler is **cloned** so each row has its own
    /// independent sampler state.
    pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
        self.ensure_not_poisoned()?;
        // B1-p2.3f: cap check before admission. Reject oversize requests
        // upfront rather than allocating a slot then failing at prefill.
        let cap_needed = req.prompt_ids.len().saturating_add(req.max_new_tokens);
        if cap_needed > self.effective_cap_max {
            return Err(anyhow::Error::new(SchedulerError::RequestTooLarge {
                needed: cap_needed,
                max: self.effective_cap_max,
            }));
        }
        if self.phase == Phase::Finished {
            return Err(anyhow!(
                "scheduler in Finished phase: cannot admit; call evict_all first"
            ));
        }
        // Idle / Admitting / Decoding all allow admit.
        //   Idle -> Admitting (first admit transitions below).
        //   Admitting -> Admitting (subsequent admits during window).
        //   Decoding -> Decoding (mid-batch admit; caller is responsible
        //     for prefilling the new slot via admit_mid in Task 4).
        let row_idx =
            self.slots.iter().position(|s| s.is_none()).ok_or_else(|| {
                anyhow!("scheduler full: no row available (b_max={})", self.b_max)
            })?;

        // B1-p2.5: memory budget admission gate.
        let row_cap = req.prompt_ids.len().saturating_add(req.max_new_tokens);
        let requested_bytes = crate::core::memory_budget::kv_cache_bytes(1, row_cap, &self.meta);
        if let Err((active, requested, soft_limit)) = self.budget_state.try_admit(requested_bytes) {
            self.memory_budget_exceeded_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(anyhow::Error::new(SchedulerError::MemoryBudgetExceeded {
                active_bytes: active,
                requested_bytes: requested,
                soft_limit_bytes: soft_limit,
            }));
        }

        let id = RequestId(self.next_id);
        self.next_id += 1;

        let prompt_len = req.prompt_ids.len();
        anyhow::ensure!(
            prompt_len <= i32::MAX as usize,
            "prompt too long: {} tokens exceeds i32::MAX",
            prompt_len
        );
        let real_len = prompt_len as i32;
        let state = RequestState {
            id,
            row_idx,
            prompt_ids: req.prompt_ids,
            generated_tokens: Vec::new(),
            max_new_tokens: req.max_new_tokens,
            stop_token_ids: req.stop_token_ids,
            sampler: req.sampler,
            real_len,
            finished: false,
            finish_reason: None,
            pixel_values: req.pixel_values,
            image_grid_thw: req.image_grid_thw,
            image_spatial_merge_size: req.image_spatial_merge_size,
            image_token_id: req.image_token_id,
            prefill_chunk_size: i32::try_from(req.prefill_chunk_size).unwrap_or(512).max(1),
            kv_bytes_admitted: requested_bytes,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: req.p5h_trace.clone(),
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: req.p5h_root_span.clone(),
        };
        let seed = state.sampler.seed;
        self.slots[row_idx] = Some(state);
        // Seed this row's PRNG state from the request's sampler seed.
        self.init_row_prng(row_idx, seed)?;
        if self.phase == Phase::Idle {
            self.phase = Phase::Admitting;
        }
        // Decoding stays Decoding (no transition on mid-batch admit).
        Ok(id)
    }

    /// Evict an in-flight request, freeing its slot for reuse. The slot
    /// index is freed but the [`RequestId`] is **never** reissued (the
    /// counter keeps incrementing).
    pub fn evict(&mut self, id: RequestId) -> Result<()> {
        self.ensure_not_poisoned()?;
        // 3c-3: evict allowed in all phases. Slot is cleared; main cache
        // state for this row stays in place (no resource leak; next
        // admit_mid into this slot overwrites via adopt_row_from).
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        // B1-p2.5: release budget before clearing the slot.
        if let Some(state) = self.slots[row_idx].take() {
            self.budget_state.release(state.kv_bytes_admitted);
        }
        // Phase transitions on evict:
        //   Admitting + active_count==0 -> Idle (pre-3c-3 behavior)
        //   Decoding  + active_count==0 -> Finished (NEW in 3c-3)
        //   Idle / Finished: no transition
        if self.active_count() == 0 {
            if self.phase == Phase::Admitting {
                self.phase = Phase::Idle;
            } else if self.phase == Phase::Decoding {
                self.phase = Phase::Finished;
            }
        }
        Ok(())
    }

    /// Number of occupied slots.
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Borrow every occupied slot's `RequestState`, in slot order.
    pub fn active(&self) -> Vec<&RequestState> {
        self.slots.iter().filter_map(|s| s.as_ref()).collect()
    }

    /// Look up by id. `None` if the id was never admitted or has been evicted.
    pub fn get(&self, id: RequestId) -> Option<&RequestState> {
        self.slots
            .iter()
            .find_map(|s| s.as_ref().filter(|r| r.id == id))
    }

    /// Mutable lookup by id.
    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut RequestState> {
        self.slots
            .iter_mut()
            .find_map(|s| s.as_mut().filter(|r| r.id == id))
    }

    /// `row_idx` of every occupied slot, in slot order. Used by 3b to
    /// build batched inputs.
    pub fn occupied_rows(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| s.as_ref().map(|_| idx))
            .collect()
    }

    /// Current scheduler phase. See [`Phase`] for the state machine.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Returns `Err` if the scheduler has been poisoned by a previous `Err`
    /// return from `prefill_admitted` or `step`. Call `evict_all` to recover.
    fn ensure_not_poisoned(&self) -> Result<()> {
        if self.poisoned {
            return Err(anyhow!(
                "scheduler poisoned by a previous Err; call evict_all to recover"
            ));
        }
        Ok(())
    }

    /// Free all in-flight rows and reset every layer cache to offset 0
    /// (preserves Array allocations for reuse). Only legal in
    /// `Decoding`/`Finished` phases. After this call the scheduler is back
    /// in `Idle` and ready to admit a new batch.
    ///
    /// `next_id` is **not** reset — the monotonic-no-reuse guarantee from
    /// 3a continues across batches.
    pub fn evict_all(&mut self) -> Result<()> {
        match self.phase {
            Phase::Decoding | Phase::Finished => {}
            Phase::Idle | Phase::Admitting => {
                return Err(anyhow!(
                    "evict_all illegal in {:?} phase: only Decoding/Finished are valid",
                    self.phase
                ));
            }
        }
        for slot in self.slots.iter_mut() {
            // B1-p2.5: release budget before clearing slot.
            if let Some(state) = slot.take() {
                self.budget_state.release(state.kv_bytes_admitted);
            }
        }
        // B1-p2.3f: drop the cache so the next prefill_admitted lazy-allocates
        // with cap matching the new batch's requirements. ~10ms re-alloc per
        // outer batch is negligible vs prefill GPU time (100s of ms to
        // seconds). Pre-3f kept the cache + reset offsets but locked the
        // first batch's cap forever — incompatible with dynamic cap.
        self.cache = None;
        self.phase = Phase::Idle;
        self.poisoned = false;
        Ok(())
    }

    /// Run batched prefill for every currently-admitted request. Only legal
    /// in `Idle`/`Admitting` phase with `active_count() >= 1`.
    ///
    /// Lazy-allocates the batched KV cache on first call (`b_max` rows;
    /// capacity = `min(max(prompt_len + max_new_tokens) over slots,
    /// effective_cap_max)`, bf16). Subsequent calls after `evict_all`
    /// allocate fresh — `evict_all` drops the cache (3f) so the next
    /// batch's cap is sized to its slots, not inherited from the prior batch.
    ///
    /// Builds right-padded `[B, T_max]` input_ids + `[3, B, T_max]`
    /// position_ids + `[B, 1, T_max, T_max]` attention mask + `[B, T_max]`
    /// linear mask, then calls `M::batched_prefill` via the `Model` trait.
    ///
    /// After prefill, samples the first token via a three-stage dispatch:
    /// Stage A collects per-row `sampler` refs + prompt histories (sentinel
    /// greedy + empty history for `None` slots so sample_batch sees a uniform
    /// `[B]` view without branching). Stage B reshapes `[B, 1, vocab]` →
    /// `[B, vocab]` and calls `sample_batch` once — coalescing all-greedy
    /// batches into a single GPU op rather than B serial kernel launches.
    /// Stage C distributes tokens to occupied rows, checks EOS / `max_new_tokens`,
    /// and emits one [`StepEvent`] per occupied row. Sentinel-row outputs are
    /// silently discarded. Transitions to `Decoding` (or `Finished` if every
    /// first token was EOS). See spec §4.5.
    pub fn prefill_admitted(&mut self, model: &M) -> Result<Vec<StepEvent>>
    where
        M: DenseVlMethods,
    {
        self.ensure_not_poisoned()?;
        match self.prefill_admitted_inner(model) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn prefill_admitted_inner(&mut self, model: &M) -> Result<Vec<StepEvent>>
    where
        M: DenseVlMethods,
    {
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "prefill_admitted illegal in {:?} phase: call evict_all first",
                    self.phase
                ));
            }
        }
        if self.active_count() == 0 {
            return Err(anyhow!("prefill_admitted: no admitted requests to prefill"));
        }

        // T0a.9: read trace ctx + root span from the (singleton) active row.
        // Placed AFTER the 0-active-row check so the existing error path is
        // unchanged in feature-on builds. Some(...) — openai.rs handler path
        // (T0a.6 populates both fields). None — any other entry path
        // (anthropic.rs / CLI / tests / scheduler_actor internals); SINK
        // quietly no-ops so non-openai code under `p5h-profile` still works.
        #[cfg(feature = "p5h-profile")]
        let p5h_trace = self.cloned_active_row_p5h_trace_and_root()?;

        // Build per-row prompt-length vector in slot order. None slots get
        // a synthetic length=1 so that build_position_ids_batched and the
        // mask builders accept the input (they assert > 0). The row stays
        // all-pad-zero in input_ids; attention masks treat its single
        // "real" column as pad K/V (zeroed by the model's batched_prefill
        // path), so active rows see no leakage from None slots.
        let prompt_lens: Vec<i32> = self
            .slots
            .iter()
            .map(|s| s.as_ref().map(|r| r.prompt_ids.len() as i32).unwrap_or(1))
            .collect();
        let max_len = prompt_lens.iter().copied().max().unwrap_or(0);
        if max_len <= 0 {
            return Err(anyhow!(
                "prefill_admitted: max prompt length is 0 — all admitted prompts are empty"
            ));
        }

        // Build [B, T_max] right-padded input_ids (pad value 0). Slot order
        // matches the slots vector — None rows become full-zero.
        let b = self.b_max;
        let t = max_len as usize;
        let mut flat: Vec<i32> = vec![0; b * t];
        for (row, slot) in self.slots.iter().enumerate() {
            if let Some(state) = slot {
                for (j, &tok) in state.prompt_ids.iter().enumerate() {
                    flat[row * t + j] = tok as i32;
                }
                // positions [state.prompt_ids.len() .. t] stay 0 (pad)
            }
        }
        let input_ids: Array = (&flat[..], &[b as i32, max_len][..])
            .try_into()
            .map_err(|e| anyhow!("input_ids try_into Array failed: {e:?}"))?;

        // B1-p2.4: detect any VL row. Dispatch determines both position_ids
        // builder and prefill entry point.
        let any_vl = self
            .slots
            .iter()
            .any(|s| s.as_ref().is_some_and(|r| r.pixel_values.is_some()));

        let attention_mask = build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16)?;
        let linear_attention_mask = build_batch_linear_mask(&prompt_lens, max_len)?;

        // Lazy-allocate the cache.
        // TODO: when a non-bf16 model lands, expose dtype via the `Model`
        // trait and thread it here.
        if self.cache.is_none() {
            // B1-p2.3f: dynamic cap = max(prompt_len + max_new_tokens) over
            // admitted slots, bounded by effective_cap_max (defense-in-depth;
            // admit gate already rejects oversize). min_cap=256 fallback if
            // all slots None (defensive — not reachable in production since
            // prefill_admitted asserts active_count() >= 1 earlier).
            let slots_max = self
                .slots
                .iter()
                .filter_map(|s| s.as_ref())
                .map(|r| {
                    let max_new_i32 = i32::try_from(r.max_new_tokens).unwrap_or(i32::MAX);
                    (r.prompt_ids.len() as i32).saturating_add(max_new_i32)
                })
                .max()
                .unwrap_or(256);
            // Logical cap: largest per-slot (prompt_len + max_new_tokens),
            // bounded by user's effective_cap_max. Then floored at
            // MIN_KV_CACHE_CAP_FOR_GPU_PERF to avoid the MLX Metal
            // kernel slow-path cliff for tight K/V buffer widths
            // (cap < ~256 → 100-300× decode-step slowdown).
            //
            // Order: floor LAST so it can exceed `effective_cap_max`
            // when the user-set cap_max is itself below the floor.
            // The floor is a physical-buffer concern; admit-gate
            // semantics use `cap_needed > effective_cap_max` based on
            // the user-requested size, not the physical KVCache cap.
            let cap = slots_max
                .min(self.effective_cap_max as i32)
                .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
            self.cache = Some(model.make_cache(b as i32, cap, Dtype::Bfloat16)?);
        }
        let cache_ref = self
            .cache
            .as_mut()
            .ok_or_else(|| anyhow!("cache missing after lazy-alloc — internal bug"))?;

        // Run batched prefill. Capture [B, 1, vocab] logits (sequence axis
        // already collapsed via slice_last_and_project) for first-token
        // sampling.
        //
        // T0a.9: wrap in `model_prefill_forward` span. Pattern (per Codex v11
        // P2 #5): capture Result without `?`, close span, then `?` the result
        // so the span closes on both Ok and Err paths. IIFE preserves the
        // existing block shape so default builds stay byte-identical.
        #[cfg(feature = "p5h-profile")]
        let mpf_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
            crate::core::p5h::open_p5h_span(ctx, Some(root_span), "model_prefill_forward")
        });

        // T4.5 (Codex Option S): open a `first_eval_amortized_cost` DIAGNOSTIC
        // span exactly once per process, wrapping the same MPF body. The
        // span_kind = "diagnostic" classification keeps it out of the T5
        // exclusive-tree sum invariants (per spec § 2.5a); T5 will report it
        // as a separate cold-start column. `OnceLock::set(()).is_ok()` wins
        // for the first caller only (race-safe). Only fires when the request
        // carries a P5h trace — non-OpenAI / non-streaming callers skip and
        // the OnceLock stays unset until a traced request arrives.
        //
        // NOTE for T5 aggregator: this adds `first_eval_amortized_cost` to
        // the closed set of permitted diagnostic span_names for routing_path
        // == "scheduler" (currently only `sse_write_role_chunk_diagnostic`).
        // Spec § 2.5a `diagnostic_allowed_by_routing` and the per-lane bucket
        // lists need to be extended accordingly.
        #[cfg(feature = "p5h-profile")]
        let first_eval_span = p5h_trace.as_ref().and_then(|(ctx, root_span)| {
            if FIRST_EVAL_AMORTIZED_COST_FIRED.set(()).is_ok() {
                Some((
                    ctx.clone(),
                    crate::core::p5h::open_p5h_span(
                        ctx,
                        Some(root_span),
                        "first_eval_amortized_cost",
                    ),
                ))
            } else {
                None
            }
        });

        let logits_result: anyhow::Result<Array> = (|| -> anyhow::Result<Array> {
            #[cfg(feature = "p5h-profile")]
            let _mpf_guard = match (p5h_trace.as_ref(), mpf_span.as_ref()) {
                (Some((ctx, _)), Some(mpf)) => Some(crate::core::p5h::P5hTraceGuard::enter(
                    ctx.clone(),
                    mpf.clone(),
                )),
                _ => None,
            };
            let logits = if any_vl {
                // Collect per-row prompt_ids (i32 conversion) + per-row vision args + tokenizer consts.
                let per_row_ids_i32: Vec<Vec<i32>> = self
                    .slots
                    .iter()
                    .map(|s| match s {
                        Some(r) => r.prompt_ids.iter().map(|&t| t as i32).collect(),
                        None => vec![0_i32], // synthetic length-1 zero row (matches prompt_lens fallback)
                    })
                    .collect();
                let per_row_ids_refs: Vec<&[i32]> =
                    per_row_ids_i32.iter().map(|v| v.as_slice()).collect();
                let per_row_grids_owned: Vec<Option<Vec<(i32, i32, i32)>>> = self
                    .slots
                    .iter()
                    .map(|s| s.as_ref().and_then(|r| r.image_grid_thw.clone()))
                    .collect();
                let per_row_grids: Vec<GridThwSlice<'_>> = per_row_grids_owned
                    .iter()
                    .map(|opt| opt.as_deref())
                    .collect();
                let per_row_pv: Vec<Option<&Array>> = self
                    .slots
                    .iter()
                    .map(|s| s.as_ref().and_then(|r| r.pixel_values.as_ref()))
                    .collect();

                // Tokenizer-defined constants from the first VL slot.
                let (img_token_id, merge_size) = self
                    .slots
                    .iter()
                    .find_map(|s| {
                        s.as_ref()
                            .filter(|r| r.pixel_values.is_some())
                            .map(|r| (r.image_token_id, r.image_spatial_merge_size))
                    })
                    .expect("any_vl == true implies at least one VL slot");

                let position_ids = build_position_ids_vl_batched(
                    &per_row_ids_refs,
                    &per_row_grids,
                    img_token_id,
                    merge_size,
                    max_len,
                )?;

                model.batched_prefill_vl(
                    &input_ids,
                    &position_ids,
                    &attention_mask,
                    &linear_attention_mask,
                    &prompt_lens,
                    &per_row_pv,
                    &per_row_grids,
                    img_token_id,
                    Some(cache_ref),
                    mlx::StreamOrDevice::default(),
                )?
            } else {
                let position_ids = build_position_ids_batched(&prompt_lens, max_len)?;
                model.batched_prefill(
                    &input_ids,
                    &position_ids,
                    &attention_mask,
                    &linear_attention_mask,
                    &prompt_lens,
                    Some(cache_ref),
                    mlx::StreamOrDevice::default(),
                )?
            };
            Ok(logits)
            // _mpf_guard drops here (end of IIFE scope) BEFORE close_p5h_span
            // below — stack-empty invariant required by P5hTraceGuard::drop.
        })();

        #[cfg(feature = "p5h-profile")]
        if let (Some((ctx, _)), Some(mpf)) = (p5h_trace.as_ref(), mpf_span) {
            crate::core::p5h::close_p5h_span(
                ctx,
                mpf,
                crate::core::p5h::monotonic_ns_public(),
                crate::core::p5h::SpanFields::default(),
            );
        }

        // T4.5: close the once-per-process `first_eval_amortized_cost`
        // diagnostic span (if it opened on this call). Closed AFTER MPF
        // close so the diagnostic interval covers the MPF body end-to-end.
        // Uses `close_p5h_span_diagnostic` to emit span_kind="diagnostic".
        #[cfg(feature = "p5h-profile")]
        if let Some((ctx, span)) = first_eval_span {
            crate::core::p5h::close_p5h_span_diagnostic(
                &ctx,
                span,
                crate::core::p5h::monotonic_ns_public(),
                crate::core::p5h::SpanFields::default(),
            );
        }

        let logits = logits_result?;

        // After per-row prefill, row i's cache is filled up to position
        // prompt_lens[i] - 1. The first decode step must use position
        // prompt_lens[i] for that row.
        for (slot, &plen) in self.slots.iter_mut().zip(prompt_lens.iter()) {
            if let Some(state) = slot.as_mut() {
                state.real_len = plen;
            }
        }

        // Sample first token per occupied row from logits[:, 0, :].
        // batched_prefill returns [B, 1, vocab]. Reshape to [B, vocab] for
        // sample_batch, then dispatch once to coalesce all-greedy batches.
        //
        // T0a.9: wrap reshape + Stage A + Stage B in `first_token_sampling`
        // span. No `P5hTraceGuard` here — body has no deep substep call sites
        // that use `try_with_p5h_span_from_current_trace`, so no guard
        // required. Stage C (distribute + termination) stays OUTSIDE the span.
        #[cfg(feature = "p5h-profile")]
        let fts_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
            crate::core::p5h::open_p5h_span(ctx, Some(root_span), "first_token_sampling")
        });

        let tokens_result: anyhow::Result<Vec<u32>> = (|| -> anyhow::Result<Vec<u32>> {
            let logits_shape = logits.shape();
            let vocab = logits_shape.as_slice()[2];
            let logits_bv = logits.reshape(&[b as i32, vocab][..]).map_err(|e| {
                anyhow!("prefill_admitted: reshape logits [B,1,vocab]->[B,vocab] failed: {e:?}")
            })?;

            // Stage A — collect per-row sampler refs + histories in slot order.
            // Sentinel covers None / pad rows; their tokens are discarded.
            let sentinel = Sampler::greedy();
            let mut row_samplers: Vec<&Sampler> = Vec::with_capacity(b);
            let mut row_histories: Vec<Vec<u32>> = Vec::with_capacity(b);
            for b_idx in 0..b {
                if let Some(state) = self.slots[b_idx].as_ref() {
                    row_samplers.push(&state.sampler);
                    row_histories.push(state.prompt_ids.clone());
                } else {
                    row_samplers.push(&sentinel);
                    row_histories.push(Vec::new());
                }
            }

            // Stage B — dispatch sample_batch once over [B, vocab].
            let history_refs: Vec<&[u32]> = row_histories.iter().map(|h| h.as_slice()).collect();
            let tokens = crate::core::sampler::sample_batch(
                &row_samplers,
                &logits_bv,
                &history_refs,
                &mut self.prng_state,
            )
            .map_err(|e| anyhow!("prefill_admitted: sample_batch failed: {e:?}"))?;
            Ok(tokens)
        })();

        #[cfg(feature = "p5h-profile")]
        if let (Some((ctx, _)), Some(fts)) = (p5h_trace.as_ref(), fts_span) {
            crate::core::p5h::close_p5h_span(
                ctx,
                fts,
                crate::core::p5h::monotonic_ns_public(),
                crate::core::p5h::SpanFields::default(),
            );
        }

        let tokens = tokens_result?;

        // Stage C — distribute tokens + termination per occupied row.
        let mut events: Vec<StepEvent> = Vec::new();
        for (b_idx, &token) in tokens.iter().enumerate() {
            if self.slots[b_idx].is_none() {
                continue;
            }
            let state = self.slots[b_idx].as_mut().expect("is_some checked above");

            state.generated_tokens.push(token);
            state.real_len += 1;

            if state.stop_token_ids.contains(&token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }

            events.push(StepEvent {
                id: state.id,
                token,
                finish_reason: state.finish_reason,
            });
        }

        // Phase transition: Finished if every occupied row already done (rare
        // corner case — first token of every prompt was EOS), else Decoding.
        let any_unfinished = self
            .slots
            .iter()
            .any(|s| matches!(s, Some(r) if !r.finished));
        self.phase = if any_unfinished {
            Phase::Decoding
        } else {
            Phase::Finished
        };

        Ok(events)
    }

    /// Advance every non-finished active row by exactly one decode token.
    /// Only legal in `Decoding` phase.
    ///
    /// Advance every non-finished active row by one decode token using a
    /// three-stage sample_batch dispatch rather than per-row sampler calls.
    /// Stage A collects `active_at_start` flags, then builds per-row sampler
    /// refs + token histories — sentinel greedy + empty history for pad /
    /// finished / mid-admit rows so sample_batch sees a uniform `[B]` view;
    /// sentinel tokens are discarded in Stage C, avoiding conditional dispatch
    /// inside the hot sampling path. Stage B packs `[B, 1]` input_ids, runs
    /// `forward_on`, reshapes `[B, 1, vocab]` → `[B, vocab]`, and calls
    /// `sample_batch` once — coalescing all-greedy batches into one GPU op.
    /// Stage C distributes tokens only to `active_at_start` rows, advances
    /// `real_len`, checks EOS / `max_new_tokens`, and collects events.
    ///
    /// Already-finished rows are still padded into the forward (lockstep
    /// cost — see spec §7). Only active-at-start rows appear in the returned
    /// event list. Transitions phase to `Finished` when all occupied rows
    /// are done.
    pub fn step(&mut self, model: &M) -> Result<Vec<StepEvent>> {
        self.ensure_not_poisoned()?;
        match self.step_inner(model) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn step_inner(&mut self, model: &M) -> Result<Vec<StepEvent>> {
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "step illegal in {:?} phase: call prefill_admitted first",
                self.phase
            ));
        }

        let b = self.b_max;

        // Capture which rows are eligible to step at the start of this
        // call. Eligible = `Some` slot AND not finished AND already has
        // at least one generated token. The `!generated_tokens.is_empty()`
        // guard excludes B1-p2.3c+ admit_mid chunk-loop rows: those have
        // an inserted `RequestState` (slot reserved by `admit_mid_begin`)
        // but no first token until `admit_mid_finalize` runs (which
        // adopts the temp cache + samples the first token). Treating
        // them as pad here makes the interleaved step a no-op for the
        // mid-admit row — the main cache slot stays empty until
        // finalize's adopt overwrites it.
        //
        // Already-finished rows are also padded into the forward
        // (lockstep cost — see spec §7).
        let active_at_start: Vec<bool> = self
            .slots
            .iter()
            .map(|s| matches!(s, Some(r) if !r.finished && !r.generated_tokens.is_empty()))
            .collect();

        // Build [B, 1] input_ids in slot order.
        // - For active rows: last generated token. `active_at_start`
        //   guarantees `generated_tokens` is non-empty so .last() unwrap
        //   is safe.
        // - For pad / mid-admit / finished rows: pad 0.
        let last_tokens: Vec<i32> = self
            .slots
            .iter()
            .zip(active_at_start.iter())
            .map(|(slot, &active)| {
                if active {
                    let r = slot.as_ref().expect("active implies Some");
                    *r.generated_tokens
                        .last()
                        .expect("active_at_start guarantees ≥1 generated token")
                        as i32
                } else {
                    0
                }
            })
            .collect();
        let input_ids: Array = (&last_tokens[..], &[b as i32, 1][..])
            .try_into()
            .map_err(|e| anyhow!("step: build input_ids Array failed: {e:?}"))?;

        // Build [3, B, 1] decode position ids. Active rows use real_len
        // (which is prompt_len + generated_count so far). Pad rows use 0.
        let per_row_pos: Vec<i32> = self
            .slots
            .iter()
            .zip(active_at_start.iter())
            .map(|(slot, &active)| {
                if active {
                    slot.as_ref().expect("active implies Some").real_len
                } else {
                    0
                }
            })
            .collect();
        let position_ids = build_decode_position_ids(&per_row_pos)?;

        // Per-row lens for decode: each active row writes 1 token; pad
        // rows (finished, mid-admit, or None slots) write 0 to skip the
        // K/V write. Mid-admit rows must skip so finalize's adopt_row_from
        // can cleanly install the prefilled state at offset 0.
        let per_row_lens: Vec<i32> = active_at_start
            .iter()
            .map(|&active| if active { 1 } else { 0 })
            .collect();

        let cache_ref = self
            .cache
            .as_mut()
            .ok_or_else(|| anyhow!("step: cache absent — was prefill_admitted called?"))?;

        // Build per-row decode mask BEFORE the forward — necessary so
        // SDPA correctly masks stale K/V cells for rows whose cache
        // offsets have diverged from max(offsets). Without the mask,
        // finished rows would attend to stale buffer-init zero K/V at
        // positions [offsets[i]..max_off], deflating their real-position
        // softmax weights. Outputs of finished rows are discarded by
        // this step, but the mask is also a prerequisite for 3c-3's
        // mid-batch admit/evict where slot reuse would expose
        // previously-written stale K/V to new admissions.
        //
        // Clone offsets into Vec to release the immutable borrow before
        // re-borrowing cache_ref mutably for the forward.
        let pre_offsets: Vec<i32> = first_full_layer_offsets(cache_ref)?.to_vec();
        let per_row_real_lens: Vec<i32> = pre_offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .collect();
        let max_real_len = per_row_real_lens
            .iter()
            .copied()
            .max()
            .expect("Decoding phase guarantees b_max >= 1 and per_row_real_lens is non-empty");
        let decode_mask =
            build_per_row_decode_mask(&per_row_real_lens, max_real_len, Dtype::Bfloat16)?;

        let logits = model.forward_on(
            &input_ids,
            &position_ids,
            Some(&per_row_lens),
            Some(&decode_mask),
            Some(cache_ref),
            mlx::StreamOrDevice::default(),
        )?;

        // logits shape: [B, 1, vocab]. Reshape to [B, vocab] for sample_batch.
        let logits_shape = logits.shape();
        let vocab = logits_shape.as_slice()[2];
        let logits_bv = logits
            .reshape(&[b as i32, vocab][..])
            .map_err(|e| anyhow!("step: reshape logits [B,1,vocab]->[B,vocab] failed: {e:?}"))?;

        // Stage A — collect per-row sampler refs + histories in slot order.
        // Sentinel covers pad / inactive rows; their tokens are discarded.
        let sentinel = Sampler::greedy();
        let mut row_samplers: Vec<&Sampler> = Vec::with_capacity(b);
        let mut row_histories: Vec<Vec<u32>> = Vec::with_capacity(b);
        for (b_idx, &was_active) in active_at_start.iter().enumerate() {
            if was_active {
                let state = self.slots[b_idx]
                    .as_ref()
                    .expect("active_at_start guaranteed Some");
                row_samplers.push(&state.sampler);
                let mut hist: Vec<u32> =
                    Vec::with_capacity(state.prompt_ids.len() + state.generated_tokens.len());
                hist.extend_from_slice(&state.prompt_ids);
                hist.extend_from_slice(&state.generated_tokens);
                row_histories.push(hist);
            } else {
                row_samplers.push(&sentinel);
                row_histories.push(Vec::new());
            }
        }

        // Stage B — dispatch sample_batch once over [B, vocab].
        let history_refs: Vec<&[u32]> = row_histories.iter().map(|h| h.as_slice()).collect();
        let tokens = crate::core::sampler::sample_batch(
            &row_samplers,
            &logits_bv,
            &history_refs,
            &mut self.prng_state,
        )
        .map_err(|e| anyhow!("step: sample_batch failed: {e:?}"))?;

        // Stage C — distribute tokens + termination per active row.
        let mut events: Vec<StepEvent> = Vec::new();
        for (b_idx, &was_active) in active_at_start.iter().enumerate() {
            if !was_active {
                continue;
            }
            let token = tokens[b_idx];
            let state = self.slots[b_idx]
                .as_mut()
                .expect("active_at_start guaranteed Some");

            state.generated_tokens.push(token);
            state.real_len += 1;

            // Termination: EOS check first, then max_new_tokens.
            if state.stop_token_ids.contains(&token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }

            events.push(StepEvent {
                id: state.id,
                token,
                finish_reason: state.finish_reason,
            });
        }

        // If every active slot is now finished, transition to Finished.
        let all_done = self
            .slots
            .iter()
            .all(|s| matches!(s, Some(r) if r.finished) || s.is_none());
        let any_present = self.slots.iter().any(|s| s.is_some());
        if all_done && any_present {
            self.phase = Phase::Finished;
        }

        Ok(events)
    }

    /// Raise every layer of the main cache's `cap` to `target_cap` if
    /// smaller. Used by `admit_mid_finalize` before adoption so that a
    /// longer mid-batch request can land into a cache that was sized
    /// for the original batch's `slots_max`.
    ///
    /// `KVCache::grow_cap` lifts the bound only; the physical K/V
    /// buffer is grown lazily by `adopt_row_from`'s `grow_to`.
    /// `GatedDeltaCache::grow_cap` is a pure i32 field update — its
    /// `conv_state` and `recurrent_state` shapes do not depend on
    /// `cap`. Both are no-ops if `target_cap <= layer.cap`.
    ///
    /// Errs only if the cache has not been lazy-allocated yet — which
    /// is impossible from `admit_mid_begin` (Decoding phase guarantees
    /// `prefill_admitted` already ran).
    fn grow_main_cache_to(&mut self, target_cap: i32) -> Result<()> {
        let cache = self.cache.as_mut().ok_or_else(|| {
            anyhow!("grow_main_cache_to: cache absent — internal bug (admit_mid in Decoding phase implies prefill_admitted ran)")
        })?;
        for layer in cache.iter_mut() {
            match layer {
                LayerCache::Full(kv) => kv.grow_cap(target_cap),
                LayerCache::Linear(gd) => gd.grow_cap(target_cap),
            }
        }
        Ok(())
    }

    /// Begin a chunked mid-batch admit (B1-p2.3c+).
    ///
    /// Reserves a slot, allocates a B=1 temp cache sized to
    /// `prompt_len + max_new_tokens` (floored at
    /// [`MIN_KV_CACHE_CAP_FOR_GPU_PERF`] for the same Metal-kernel reason
    /// as the main cache), and pre-computes the full-prompt MRoPE
    /// position ids so subsequent `admit_mid_chunk` calls can slice
    /// without rebuilding.
    ///
    /// Returns an [`AdmitMidHandle`] the caller passes to
    /// `admit_mid_chunk` (looped until `is_last=true`) and finally
    /// `admit_mid_finalize`. The caller interleaves
    /// [`Scheduler::step`] between chunks so active rows continue
    /// emitting tokens.
    ///
    /// # VL fallback (spec §4.6 NG7 / §4.7 R6)
    /// If the request has `image_pad` token runs that span a chunk
    /// boundary, this v1 implementation forces single-chunk path
    /// (`chunk_size = prompt_len`). Per-chunk vision slicing is a v2
    /// task. A warning is logged.
    ///
    /// # Errors
    /// - [`SchedulerError::RequestTooLarge`] when
    ///   `prompt_len + max_new_tokens > effective_cap_max`.
    /// - `phase != Decoding` (`admit_mid_begin` is only callable
    ///   mid-batch; use `admit` for fresh batches).
    /// - `scheduler full` when no slot is free (admission queue is
    ///   `driver_loop`'s concern; here we surface the raw error).
    /// - dtype / make_cache failures bubble up; the orphan slot is
    ///   rolled back via `evict` so the next `step()` does not panic.
    pub fn admit_mid_begin(&mut self, req: GenerateRequest, model: &M) -> Result<AdmitMidHandle>
    where
        M: DenseVlMethods,
    {
        self.ensure_not_poisoned()?;

        // Cap gate — mirror admit's, otherwise queue drain could push an
        // oversize request through.
        let cap_needed = req.prompt_ids.len().saturating_add(req.max_new_tokens);
        if cap_needed > self.effective_cap_max {
            return Err(anyhow::Error::new(SchedulerError::RequestTooLarge {
                needed: cap_needed,
                max: self.effective_cap_max,
            }));
        }
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "admit_mid_begin illegal in {:?} phase: only Decoding (use admit for Idle/Admitting)",
                self.phase
            ));
        }
        let row_idx =
            self.slots.iter().position(|s| s.is_none()).ok_or_else(|| {
                anyhow!("scheduler full: no row available (b_max={})", self.b_max)
            })?;

        // 1. Reserve slot via the relaxed admit() path. Phase stays Decoding.
        let id = self.admit(req)?;

        // From here on, any Err must roll back the slot.
        match self.admit_mid_begin_inner(id, row_idx, model) {
            Ok(h) => Ok(h),
            Err(e) => {
                let _ = self.evict(id);
                Err(e)
            }
        }
    }

    /// Body of `admit_mid_begin` separated so the caller can centralise
    /// rollback. Steps: extract per-row state, compute floored
    /// `cap_for_temp`, detect dtype, allocate temp_cache, pre-build
    /// full-prompt position ids, run VL R6 fallback detection.
    fn admit_mid_begin_inner(
        &mut self,
        id: RequestId,
        row_idx: usize,
        model: &M,
    ) -> Result<AdmitMidHandle>
    where
        M: DenseVlMethods,
    {
        let (
            prompt_ids,
            prompt_len_usz,
            max_new_tokens,
            pixel_values,
            image_grid_thw,
            image_token_id,
            image_spatial_merge_size,
            prefill_chunk_size,
        ) = {
            let state = self.slots[row_idx].as_ref().expect("admit inserted");
            (
                state.prompt_ids.clone(),
                state.prompt_ids.len(),
                state.max_new_tokens,
                state.pixel_values.clone(),
                state.image_grid_thw.clone(),
                state.image_token_id,
                state.image_spatial_merge_size,
                state.prefill_chunk_size,
            )
        };
        let prompt_len = prompt_len_usz as i32;
        // Saturate max_new to i32 then floor cap for GPU-perf (same reason
        // as prefill_admitted_inner main cache; spec §4.5.5).
        let max_new_i32 = i32::try_from(max_new_tokens).unwrap_or(i32::MAX);
        let cap_for_temp = prompt_len
            .saturating_add(max_new_i32)
            .max(prompt_len)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);

        // Dtype from main cache's first Full layer.
        let dtype = {
            let main_cache = self
                .cache
                .as_ref()
                .ok_or_else(|| anyhow!("admit_mid_begin: main cache absent"))?;
            main_cache
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(kv) => Some(kv.dtype()),
                    _ => None,
                })
                .unwrap_or(Dtype::Bfloat16)
        };

        let temp_cache = model.make_cache(1, cap_for_temp, dtype)?;

        let is_vl = pixel_values.is_some();

        // VL R6 fallback: if any image_pad run straddles a chunk
        // boundary, force single-chunk path (v1 does not slice vision
        // args per chunk).
        let mut chunk_size = prefill_chunk_size.max(1);
        if is_vl && vl_image_pad_crosses_chunk_boundary(&prompt_ids, image_token_id, chunk_size) {
            tracing::warn!(
                "[admit_mid_begin] VL request with image_pad spanning chunk boundary; \
                 forcing single-chunk (chunk_size={chunk_size} -> {prompt_len}); \
                 v2 will support per-chunk vision slicing",
            );
            chunk_size = prompt_len;
        }

        // Pre-build full-prompt MRoPE position ids in the B=1 single-stream
        // shape that `model.forward_on` / `model.forward_vl_chunk` expect:
        // `[3, 1, prompt_len]`. Chunked path slices axis 2 per chunk.
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
        let position_ids_full = if is_vl {
            build_position_ids_vl(
                &prompt_ids_i32,
                image_grid_thw
                    .as_deref()
                    .expect("is_vl implies image_grid_thw is Some"),
                image_token_id,
                image_spatial_merge_size,
            )?
        } else {
            build_position_ids(0, prompt_len)?
        };

        // For VL: pre-compute vision embeddings once (`[N_image_pad_total,
        // hidden]`). Each chunk slices the rows it consumes; tracking the
        // running offset via `image_pad_consumed`.
        let vision_embeds_full = if is_vl {
            let pv = pixel_values
                .as_ref()
                .expect("is_vl implies pixel_values is Some");
            let grids = image_grid_thw
                .as_deref()
                .expect("is_vl implies image_grid_thw is Some");
            Some(model.compute_vision_embeds(pv, grids, mlx::StreamOrDevice::default())?)
        } else {
            None
        };

        Ok(AdmitMidHandle {
            request_id: id,
            row_idx,
            prompt_ids,
            prompt_len,
            chunk_size,
            chunk_start: 0,
            temp_cache,
            is_vl,
            image_token_id,
            position_ids_full,
            vision_embeds_full,
            image_pad_consumed: 0,
            last_logits: None,
        })
    }

    /// Run one chunk of admit_mid prefill into `handle.temp_cache`.
    /// Returns `true` if this was the last chunk (`chunk_end ==
    /// prompt_len`); caller then proceeds to `admit_mid_finalize`.
    /// Otherwise caller should run one `Scheduler::step` against the
    /// main cache before the next `admit_mid_chunk` call so active rows
    /// continue emitting tokens (spec §4.5.5 chunk:step = 1:1).
    ///
    /// On the last chunk this method stashes the `[1, 1, vocab]` logits
    /// into `handle.last_logits` for first-token sampling in
    /// `admit_mid_finalize`.
    pub fn admit_mid_chunk(
        &mut self,
        handle: &mut AdmitMidHandle,
        model: &M,
    ) -> Result<bool /* is_last */>
    where
        M: DenseVlMethods,
    {
        self.ensure_not_poisoned()?;

        let chunk_end = handle
            .chunk_start
            .saturating_add(handle.chunk_size)
            .min(handle.prompt_len);
        let is_last = chunk_end == handle.prompt_len;
        let chunk_len = chunk_end - handle.chunk_start;
        if chunk_len <= 0 {
            return Err(anyhow!(
                "admit_mid_chunk: chunk_len <= 0 (chunk_start={}, chunk_end={})",
                handle.chunk_start,
                chunk_end
            ));
        }

        // Build chunk-local input_ids [1, chunk_len].
        let chunk_ids_u32 = &handle.prompt_ids[handle.chunk_start as usize..chunk_end as usize];
        let chunk_ids_i32: Vec<i32> = chunk_ids_u32.iter().map(|&t| t as i32).collect();
        let input_ids: Array = (&chunk_ids_i32[..], &[1_i32, chunk_len][..])
            .try_into()
            .map_err(|e| anyhow!("admit_mid_chunk: input_ids try_into Array failed: {e:?}"))?;

        // Slice axis 2 of the pre-built full-prompt position ids.
        // position_ids_full shape: [3, 1, prompt_len].
        let position_ids =
            slice_pos_ids_axis2(&handle.position_ids_full, handle.chunk_start, chunk_end)?;

        // Forward via the B=1 single-stream API (same path GS chunked
        // prefill uses). The pre-3c+ implementation went through
        // `batched_prefill` (a B>1 path) which adds per-row mask and
        // B-loop overhead that dominated cold + warm runs alike
        // (3c+ T4 finding: chunks ran 10-25× slower than expected with
        // batched_prefill). Going through `forward_on` removes the
        // overhead AND removes the need for caller-built attention /
        // linear masks — the B=1 path derives them internally from
        // input shape + cache state.
        //
        // - Last chunk: `forward_on` returns `[1, 1, vocab]` logits via
        //   lm_head, captured for first-token sampling in
        //   `admit_mid_finalize`.
        // - Intermediate chunk: text path uses `text().forward_on`
        //   (skips lm_head; we don't need logits), VL path uses
        //   `forward_vl_chunk` (always returns logits — we discard).
        //   Either way the result is `eval`-d before return so the
        //   chunk's lazy graph materialises here rather than ballooning
        //   into the interleaved `Scheduler::step` call. (GenerationStream
        //   chunked prefill at core/generate.rs ~line 1053 uses the
        //   same `eval(hidden)` pattern for the same reason.)
        let result_logits: Option<Array> = if handle.is_vl {
            // VL chunk: slice the rows of `vision_embeds_full` that
            // correspond to this chunk's `image_pad` token count.
            let k_i = count_image_pad(chunk_ids_u32, handle.image_token_id);
            let ve_slice = if k_i > 0 {
                let ve_full = handle
                    .vision_embeds_full
                    .as_ref()
                    .expect("is_vl implies vision_embeds_full was computed in admit_mid_begin");
                let start = handle.image_pad_consumed;
                let slice = slice_vision_embeds_rows(ve_full, start, start + k_i)?;
                handle.image_pad_consumed += k_i;
                Some(slice)
            } else {
                None
            };

            let logits = model.forward_vl_chunk(
                &input_ids,
                &position_ids,
                None, // per_row_lens (B=1 path derives from input shape)
                None, // decode_mask (prefill — model builds its own causal mask)
                Some(&mut handle.temp_cache),
                ve_slice.as_ref(),
                handle.image_token_id,
                mlx::StreamOrDevice::default(),
            )?;
            if is_last {
                Some(logits)
            } else {
                // T4.2 (Codex Option A): wrap the EXISTING explicit per-chunk
                // sync barrier in `mlx_eval_barrier` tree span. Parent context
                // is the active P5h trace stack top when one is active; the
                // centralized `try_with_p5h_span_from_current_trace` no-ops
                // when no trace is active (today the mid-admit chunked path
                // does not enter a `P5hTraceGuard`, so this site is inert
                // until / unless that plumbing is added in a future task).
                // No new eval is added — we wrap the existing call only.
                #[cfg(feature = "p5h-profile")]
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "mlx_eval_barrier",
                    crate::core::p5h::SpanFields::default,
                    || mlx::transforms::eval(&[&logits]).map_err(anyhow::Error::from),
                )?;
                #[cfg(not(feature = "p5h-profile"))]
                mlx::transforms::eval(&[&logits])?;
                None
            }
        } else if is_last {
            // Text last chunk: full forward returns logits.
            Some(model.forward_on(
                &input_ids,
                &position_ids,
                None,
                None,
                Some(&mut handle.temp_cache),
                mlx::StreamOrDevice::default(),
            )?)
        } else {
            // Text intermediate chunk: skip lm_head, just update KV cache.
            let hidden = model.forward_text_hidden(
                &input_ids,
                &position_ids,
                None,
                None,
                Some(&mut handle.temp_cache),
                mlx::StreamOrDevice::default(),
            )?;
            // T4.2 (Codex Option A): same `mlx_eval_barrier` wrap as the VL
            // non-last branch above — see that comment for rationale (existing
            // explicit per-chunk sync; no new eval added; no-ops without an
            // active P5h trace).
            #[cfg(feature = "p5h-profile")]
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "mlx_eval_barrier",
                crate::core::p5h::SpanFields::default,
                || mlx::transforms::eval(&[&hidden]).map_err(anyhow::Error::from),
            )?;
            #[cfg(not(feature = "p5h-profile"))]
            mlx::transforms::eval(&[&hidden])?;
            None
        };

        if let Some(logits) = result_logits {
            handle.last_logits = Some(logits);
        }
        handle.chunk_start = chunk_end;
        Ok(is_last)
    }

    /// Finalise a chunked mid-batch admit: grow the main cache to
    /// `temp_cache.cap` if needed (Option C from 3f), adopt
    /// `temp_cache` row 0 into `main_cache` at `handle.row_idx`,
    /// sample the new row's first token from `handle.last_logits`,
    /// then update the row's termination state.
    ///
    /// Returns `(request_id, first_event)`; caller routes the event
    /// to its `event_rx`.
    pub fn admit_mid_finalize(
        &mut self,
        handle: AdmitMidHandle,
        _model: &M,
    ) -> Result<(RequestId, StepEvent)> {
        self.ensure_not_poisoned()?;
        let AdmitMidHandle {
            request_id: id,
            row_idx,
            temp_cache,
            last_logits,
            prompt_ids,
            ..
        } = handle;

        let logits = last_logits
            .ok_or_else(|| anyhow!("admit_mid_finalize: last_logits absent (no chunks ran?)"))?;

        // Grow main cache cap from temp_cache.cap (3f Option C).
        let cap_for_temp = temp_cache
            .iter()
            .find_map(|c| match c {
                LayerCache::Full(kv) => Some(kv.cap()),
                _ => None,
            })
            .unwrap_or(0);
        self.grow_main_cache_to(cap_for_temp)?;

        // Adopt temp → main per layer.
        {
            let main_cache = self
                .cache
                .as_mut()
                .expect("cache asserted Some by Decoding phase");
            if main_cache.len() != temp_cache.len() {
                return Err(anyhow!(
                    "admit_mid_finalize: cache layer count mismatch ({} vs {})",
                    main_cache.len(),
                    temp_cache.len()
                ));
            }
            for (main_layer, temp_layer) in main_cache.iter_mut().zip(temp_cache.iter()) {
                match (main_layer, temp_layer) {
                    (LayerCache::Full(main_kv), LayerCache::Full(temp_kv)) => {
                        main_kv.adopt_row_from(temp_kv, row_idx, 0)?;
                    }
                    (LayerCache::Linear(main_gd), LayerCache::Linear(temp_gd)) => {
                        main_gd.adopt_row_from(temp_gd, row_idx, 0)?;
                    }
                    _ => return Err(anyhow!("admit_mid_finalize: cache layer kind mismatch")),
                }
            }
        }

        // Sample first generated token using centralized PRNG state.
        let row_logits = slice_logits_row(&logits, 0)?;
        let token = {
            let history: Vec<u32> = prompt_ids.clone();
            // Copy the sampler (Sampler: Copy post-3e.2) so we release the borrow on self.slots before
            // calling write_row_prng (which needs &mut self).
            let sampler = self.slots[row_idx]
                .as_ref()
                .expect("admit_mid_begin reserved the slot")
                .sampler;
            // Extract this row's PRNG key from centralized state.
            let prng_host: Vec<u32> = self.prng_state.to_vec()?;
            let row_bytes = &prng_host[row_idx * 2..(row_idx + 1) * 2];
            let mut row_key: Array = (row_bytes, &[2_i32][..]).try_into()?;
            let tok = sampler.sample(&row_logits, &history, &mut row_key)?;
            // Write updated key back (releases sampler borrow, &mut self OK now).
            self.write_row_prng(row_idx, &row_key)?;
            tok
        };

        // Update RequestState + termination.
        let state = self.slots[row_idx]
            .as_mut()
            .expect("admit_mid_begin reserved the slot");
        state.generated_tokens.push(token);
        state.real_len += 1;
        if state.stop_token_ids.contains(&token) {
            state.finished = true;
            state.finish_reason = Some("stop");
        } else if state.generated_tokens.len() >= state.max_new_tokens {
            state.finished = true;
            state.finish_reason = Some("length");
        }
        let finish_reason = state.finish_reason;

        Ok((
            id,
            StepEvent {
                id,
                token,
                finish_reason,
            },
        ))
    }

    /// Sweep finished rows: clear their slot, drop their event channel,
    /// and return the evicted IDs. Cache buffer entries for evicted
    /// slots stay in place — a subsequent `admit_mid` into the same
    /// slot overwrites via `adopt_row_from`.
    ///
    /// Phase transition: Decoding -> Finished if `active_count == 0`
    /// after the sweep. This duplicates `step_inner`'s end-of-loop
    /// Phase transition (both call `phase = Finished` when no active
    /// rows remain after the last row finishes). The duplication is
    /// idempotent: in the rolling decode loop (Task 4), step runs
    /// first (may transition Phase) and gc_finished_rows runs second
    /// (re-affirms transition + clears slots). Either path alone is
    /// correct; the redundancy is harmless and preserves backward
    /// compatibility with direct-step callers (3b-1 integration tests).
    ///
    /// Called by `SchedulerActor::driver_loop` after every successful
    /// `step` invocation in 3c-3's rolling decode loop (Task 4).
    ///
    /// Generic over the event-payload type `S` so the Scheduler does
    /// not need to import `StepEvent`'s concrete channel type; in
    /// production `S = StepEvent`.
    pub fn gc_finished_rows<S>(
        &mut self,
        event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<S>>,
    ) -> Vec<RequestId> {
        let mut evicted: Vec<RequestId> = Vec::new();
        for slot in self.slots.iter_mut() {
            if slot.as_ref().is_some_and(|s| s.finished) {
                // B1-p2.5: release budget on slot clear.
                if let Some(state) = slot.take() {
                    event_txs.remove(&state.id);
                    evicted.push(state.id);
                    self.budget_state.release(state.kv_bytes_admitted);
                }
            }
        }
        if self.phase == Phase::Decoding && self.active_count() == 0 {
            self.phase = Phase::Finished;
        }
        evicted
    }

    /// Test-only seam to flip the scheduler's phase without driving a
    /// model forward. Used to verify phase-guard error paths from unit
    /// tests; never called by production code.
    #[cfg(test)]
    pub(crate) fn force_phase(&mut self, p: Phase) {
        self.phase = p;
    }

    /// cfg(test)-only accessor: compute the cap that `prefill_admitted_inner`
    /// passes to `model.make_cache`, including the GPU-perf floor.
    /// Mirrors the production formula
    /// `slots_max.min(effective_cap_max).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF)`.
    /// Used by 3f unit tests to verify cap-formula correctness without
    /// invoking a real model.
    #[cfg(test)]
    pub(crate) fn computed_cap_for_prefill(&self) -> i32 {
        let slots_max = self
            .slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|r| {
                let max_new_i32 = i32::try_from(r.max_new_tokens).unwrap_or(i32::MAX);
                (r.prompt_ids.len() as i32).saturating_add(max_new_i32)
            })
            .max()
            .unwrap_or(256);
        slots_max
            .min(self.effective_cap_max as i32)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF)
    }
}

#[cfg(feature = "p5h-profile")]
impl<M: crate::core::model::Model> Scheduler<M> {
    #[allow(dead_code)] // used by prefill_admitted_inner SINK; allow kept so future cfg-gated changes that drop the call site stay clippy-clean
    pub(crate) fn cloned_active_row_p5h_trace_and_root(
        &self,
    ) -> anyhow::Result<
        Option<(
            crate::core::p5h::P5hTraceContext,
            crate::core::p5h::SpanHandle,
        )>,
    > {
        let active: Vec<&RequestState> = self.slots.iter().filter_map(|s| s.as_ref()).collect();
        anyhow::ensure!(
            active.len() == 1,
            "p5h-profile invariant: expected exactly 1 active row, found {} (--b-max 1 required)",
            active.len(),
        );
        let state = active[0];
        match (state.p5h_trace.clone(), state.p5h_root_span.clone()) {
            (Some(ctx), Some(root_span)) => Ok(Some((ctx, root_span))),
            (None, None) => Ok(None),
            (Some(_), None) => anyhow::bail!(
                "p5h-profile invariant: active RequestState has p5h_trace but no p5h_root_span — \
                 mixed-state bug (only openai.rs handler sets either field, and it sets both)"
            ),
            (None, Some(_)) => anyhow::bail!(
                "p5h-profile invariant: active RequestState has p5h_root_span but no p5h_trace — \
                 mixed-state bug (only openai.rs handler sets either field, and it sets both)"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concrete scheduler type for unit tests — pinned to `Qwen35Model` so
    /// `Scheduler::new` calls don't need turbofish at every site.
    type TestScheduler = Scheduler<crate::models::qwen3_5::Qwen35Model>;

    /// Helper: build a minimal `GenerateRequest` for tests. Uses
    /// `Sampler::greedy()` and an arbitrary 4-token prompt unless overridden.
    fn mk_req(prompt_ids: Vec<u32>) -> GenerateRequest {
        GenerateRequest {
            prompt_ids,
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248056,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        }
    }

    #[test]
    fn scheduler_new_empty() {
        let s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        assert_eq!(s.b_max(), 4);
        assert_eq!(s.active_count(), 0);
        assert!(s.active().is_empty());
        assert!(s.occupied_rows().is_empty());
    }

    #[test]
    fn admit_happy_path() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1, 2, 3, 4])).expect("admit");
        assert_eq!(id, RequestId(0));
        assert_eq!(s.active_count(), 1);
        let state = s.get(id).expect("get");
        assert_eq!(state.row_idx, 0);
        assert_eq!(state.real_len, 4);
        assert_eq!(state.prompt_ids, vec![1, 2, 3, 4]);
        assert!(state.generated_tokens.is_empty());
        assert!(!state.finished);
        assert!(state.finish_reason.is_none());
    }

    #[test]
    fn admit_assigns_distinct_rows() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let ids: Vec<_> = (0..4)
            .map(|i| s.admit(mk_req(vec![i as u32])).expect("admit"))
            .collect();
        let rows: Vec<usize> = ids.iter().map(|id| s.get(*id).unwrap().row_idx).collect();
        assert_eq!(rows, vec![0, 1, 2, 3]);
        assert_eq!(s.active_count(), 4);
    }

    #[test]
    fn evict_releases_row() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.active_count(), 1);
        s.evict(id).expect("evict");
        assert_eq!(s.active_count(), 0);
        assert!(s.get(id).is_none());
    }

    #[test]
    fn admit_after_evict_reuses_row() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        assert_eq!(s.get(id_a).unwrap().row_idx, 0);
        s.evict(id_a).expect("evict a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");
        assert_eq!(s.get(id_b).unwrap().row_idx, 0); // same slot
        assert_ne!(id_a, id_b); // distinct id
    }

    #[test]
    fn admit_full_returns_err() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        s.admit(mk_req(vec![1])).expect("admit 0");
        s.admit(mk_req(vec![2])).expect("admit 1");
        let err = s.admit(mk_req(vec![3])).expect_err("admit full");
        let msg = format!("{err}");
        assert!(msg.contains("scheduler full"), "unexpected err: {msg}");
        assert!(msg.contains("b_max=2"), "missing b_max in err: {msg}");
    }

    #[test]
    fn evict_unknown_id_returns_err() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let err = s.evict(RequestId(42)).expect_err("evict unknown");
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn id_monotonic_after_evict() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        s.evict(id_a).expect("evict a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");
        assert!(
            id_b.0 > id_a.0,
            "next id should be > previous: {:?} vs {:?}",
            id_b,
            id_a
        );
    }

    #[test]
    fn sampler_cloned_per_request() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");

        // Distinct `RequestState`s must own distinct Sampler instances at distinct addresses.
        let p_a: *const Sampler = &s.get(id_a).unwrap().sampler;
        let p_b: *const Sampler = &s.get(id_b).unwrap().sampler;
        assert_ne!(p_a, p_b);
    }

    #[test]
    fn occupied_rows_reflects_state() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _id_0 = s.admit(mk_req(vec![1])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![2])).expect("admit 1");
        let _id_2 = s.admit(mk_req(vec![3])).expect("admit 2");
        assert_eq!(s.occupied_rows(), vec![0, 1, 2]);
        s.evict(id_1).expect("evict 1");
        assert_eq!(s.occupied_rows(), vec![0, 2]);
    }

    #[test]
    fn phase_starts_idle() {
        let s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        assert_eq!(s.phase(), Phase::Idle);
        // Verify cache starts unallocated (visible through manual Debug impl
        // which surfaces `cache_layers: None`).
        assert!(
            format!("{s:?}").contains("cache_layers: None"),
            "fresh scheduler should report cache_layers: None — got {s:?}"
        );
    }

    #[test]
    fn admit_transitions_idle_to_admitting() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn admit_stays_in_admitting() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _ = s.admit(mk_req(vec![1])).expect("admit 1");
        let _ = s.admit(mk_req(vec![2])).expect("admit 2");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn evict_last_admitted_returns_to_idle() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
        s.evict(id).expect("evict");
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn admit_in_decoding_ok_phase_stays_decoding() {
        // 3c-3: admit during Decoding is now legal (mid-batch admit).
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        s.force_phase(Phase::Decoding);
        let id = s
            .admit(mk_req(vec![1]))
            .expect("admit during Decoding must succeed");
        assert_eq!(s.phase(), Phase::Decoding, "phase must stay Decoding");
        assert!(s.get(id).is_some());
    }

    #[test]
    fn admit_in_finished_returns_err() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        s.force_phase(Phase::Finished);
        let err = s.admit(mk_req(vec![1])).expect_err("admit must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("Finished") && msg.contains("cannot admit"),
            "unexpected err message: {msg}"
        );
    }

    #[test]
    fn evict_in_decoding_ok_transitions_to_finished_when_last() {
        // 3c-3: evict during Decoding is now legal.
        // Evicting the last row transitions Decoding -> Finished.
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Decoding);
        s.evict(id).expect("evict during Decoding must succeed");
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.phase(), Phase::Finished);
    }

    #[test]
    fn evict_all_from_finished_resets_to_idle() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Finished);
        s.evict_all().expect("evict_all");
        assert_eq!(s.phase(), Phase::Idle);
        assert_eq!(s.active_count(), 0);
    }

    #[test]
    fn evict_all_in_idle_returns_err() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let err = s.evict_all().expect_err("evict_all from Idle must fail");
        assert!(format!("{err}").contains("Idle"), "unexpected err: {err}");
    }

    #[test]
    fn evict_all_in_admitting_returns_err() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        // phase is now Admitting; evict_all must reject
        let err = s
            .evict_all()
            .expect_err("evict_all from Admitting must fail");
        assert!(
            format!("{err}").contains("Admitting"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn force_poison_then_admit_returns_err() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        s.poisoned = true;
        let err = s
            .admit(mk_req(vec![1]))
            .expect_err("admit after poison must fail");
        assert!(
            format!("{err}").contains("poisoned"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn force_poison_then_evict_returns_err() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1])).expect("admit");
        s.poisoned = true;
        let err = s.evict(id).expect_err("evict after poison must fail");
        assert!(
            format!("{err}").contains("poisoned"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn evict_all_clears_poison() {
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Finished); // evict_all requires Decoding/Finished
        s.poisoned = true;
        s.evict_all()
            .expect("evict_all should succeed even when poisoned");
        assert!(!s.poisoned, "poisoned flag must be cleared after evict_all");
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn scheduler_admit_during_decoding_ok() {
        // Force phase to Decoding (test seam); admit should succeed and
        // Phase should stay Decoding (mid-batch admit semantics).
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        s.force_phase(Phase::Decoding);

        let id_b = s
            .admit(mk_req(vec![4, 5, 6, 7]))
            .expect("admit b during Decoding");
        assert_eq!(s.phase(), Phase::Decoding, "phase should remain Decoding");
        assert_eq!(s.active_count(), 2);
        // Both ids should be findable.
        assert!(s.get(id_a).is_some());
        assert!(s.get(id_b).is_some());
    }

    #[test]
    fn scheduler_evict_during_decoding_transitions_to_finished_when_last() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        s.force_phase(Phase::Decoding);

        // Evict during Decoding: legal now (was Err pre-3c-3).
        s.evict(id_a).expect("evict during Decoding");
        // active_count == 0 + was Decoding -> Finished
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.phase(), Phase::Finished);
    }

    #[test]
    fn scheduler_evict_during_decoding_not_last_stays_decoding() {
        // Evict one row mid-Decoding when other rows are still active:
        // Phase must stay Decoding (only last-evict transitions to Finished).
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        let _id_b = s.admit(mk_req(vec![2])).expect("admit b");
        s.force_phase(Phase::Decoding);

        s.evict(id_a).expect("evict id_a mid-Decoding");
        assert_eq!(s.active_count(), 1, "id_b should remain active");
        assert_eq!(
            s.phase(),
            Phase::Decoding,
            "phase must stay Decoding when other rows are still active"
        );
    }

    #[test]
    fn scheduler_gc_finished_rows_clears_slots_and_transitions() {
        use std::collections::HashMap;
        use tokio::sync::mpsc;

        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        let id_b = s.admit(mk_req(vec![4, 5, 6])).expect("admit b");
        s.force_phase(Phase::Decoding);

        // Mark both as finished (test seam: directly mutate state).
        s.get_mut(id_a).unwrap().finished = true;
        s.get_mut(id_a).unwrap().finish_reason = Some("length");
        s.get_mut(id_b).unwrap().finished = true;
        s.get_mut(id_b).unwrap().finish_reason = Some("stop");

        let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
        let (tx_a, _rx_a) = mpsc::unbounded_channel::<StepEvent>();
        let (tx_b, _rx_b) = mpsc::unbounded_channel::<StepEvent>();
        event_txs.insert(id_a, tx_a);
        event_txs.insert(id_b, tx_b);

        let evicted = s.gc_finished_rows(&mut event_txs);
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&id_a));
        assert!(evicted.contains(&id_b));
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.phase(), Phase::Finished);
        assert!(event_txs.is_empty(), "event_txs should be empty after gc");
    }

    #[test]
    fn scheduler_gc_finished_rows_partial_sweep_stays_decoding() {
        use std::collections::HashMap;
        use tokio::sync::mpsc;

        // 2 rows admitted; only row A finishes. gc should evict A only,
        // leave B alive, and Phase must stay Decoding (active_count==1).
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        let id_b = s.admit(mk_req(vec![4, 5, 6])).expect("admit b");
        s.force_phase(Phase::Decoding);

        s.get_mut(id_a).unwrap().finished = true;
        s.get_mut(id_a).unwrap().finish_reason = Some("length");
        // id_b stays unfinished.

        let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
        let (tx_a, _rx_a) = mpsc::unbounded_channel::<StepEvent>();
        let (tx_b, _rx_b) = mpsc::unbounded_channel::<StepEvent>();
        event_txs.insert(id_a, tx_a);
        event_txs.insert(id_b, tx_b);

        let evicted = s.gc_finished_rows(&mut event_txs);
        assert_eq!(evicted, vec![id_a], "only id_a should be evicted");
        assert_eq!(s.active_count(), 1, "id_b should remain active");
        assert_eq!(
            s.phase(),
            Phase::Decoding,
            "phase must stay Decoding when other rows continue"
        );
        assert!(
            event_txs.contains_key(&id_b),
            "id_b's event channel must remain"
        );
        assert!(
            !event_txs.contains_key(&id_a),
            "id_a's event channel must be dropped"
        );
    }

    #[test]
    fn scheduler_gc_finished_rows_noop_when_no_finished() {
        use std::collections::HashMap;
        use tokio::sync::mpsc;

        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        s.force_phase(Phase::Decoding);

        let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
        let (tx_a, _rx_a) = mpsc::unbounded_channel::<StepEvent>();
        event_txs.insert(id_a, tx_a);

        let evicted = s.gc_finished_rows(&mut event_txs);
        assert!(
            evicted.is_empty(),
            "no rows finished, evicted should be empty"
        );
        assert_eq!(s.active_count(), 1);
        assert_eq!(
            s.phase(),
            Phase::Decoding,
            "phase unchanged when no eviction"
        );
        assert!(event_txs.contains_key(&id_a), "event channel must persist");
    }

    #[test]
    fn admit_carries_vl_fields() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use crate::core::sampler::Sampler;
        use mlx::Dtype;

        let mut sched =
            TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
                .expect("scheduler startup");

        // Synthesize a dummy pixel_values array (shape doesn't matter for plumbing)
        let pv: Array = (&[0.0_f32; 4][..], &[1_i32, 4][..]).try_into().unwrap();
        let pv_bf16 = mlx::ops::astype(&pv, Dtype::Bfloat16).unwrap();
        let grids = vec![(1_i32, 4_i32, 4_i32)];

        let req = GenerateRequest {
            prompt_ids: vec![1, 2, 3, IMAGE_TOKEN_ID as u32, 4],
            max_new_tokens: 8,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: Some(pv_bf16),
            image_grid_thw: Some(grids.clone()),
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };

        let id = sched.admit(req).expect("admit");

        let slot = sched
            .slots
            .iter()
            .find_map(|s| s.as_ref().filter(|r| r.id == id))
            .expect("slot");

        assert!(slot.pixel_values.is_some(), "pixel_values carried");
        assert_eq!(slot.image_grid_thw.as_deref(), Some(&grids[..]));
        assert_eq!(slot.image_spatial_merge_size, 2);
        assert_eq!(slot.image_token_id, IMAGE_TOKEN_ID);
    }

    #[test]
    fn evict_all_drops_cache() {
        // B1-p2.3f: evict_all drops cache (replaces pre-3f offset reset) so
        // the next prefill_admitted lazy-allocates with the new batch's cap.
        let mut s = TestScheduler::new(4, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");

        let req = GenerateRequest {
            prompt_ids: vec![1, 2, 3],
            max_new_tokens: 8,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };
        let _id = s.admit(req).expect("admit");
        s.force_phase(Phase::Decoding);

        assert!(
            s.cache.is_none(),
            "pre-evict_all: cache should be None (no prefill)"
        );

        s.evict_all().expect("evict_all");

        assert!(
            s.cache.is_none(),
            "post-evict_all: cache must be None (3f drops)"
        );
    }

    #[test]
    fn admit_rejects_oversize_request() {
        // B1-p2.3f: admit cap gate. cap_max=1024; request with
        // prompt_len=1500 + max_new=600 = 2100 > 1024 must reject with
        // SchedulerError::RequestTooLarge.
        use crate::core::SchedulerError;

        let mut s = TestScheduler::new(1, 1024, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");

        let oversize_req = GenerateRequest {
            prompt_ids: vec![0; 1500],
            max_new_tokens: 600,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };

        let result = s.admit(oversize_req);
        let err = result.expect_err("admit should reject oversize");

        let sched_err = err
            .downcast_ref::<SchedulerError>()
            .expect("err should be downcast-able to SchedulerError");
        match sched_err {
            SchedulerError::RequestTooLarge { needed, max } => {
                assert_eq!(*needed, 2100, "needed cap should be prompt+max_new");
                assert_eq!(
                    *max, 1024,
                    "max should be effective_cap_max from Scheduler::new"
                );
            }
            other => panic!("expected RequestTooLarge, got {other:?}"),
        }

        let msg = format!("{err:#}");
        assert!(
            msg.contains("2100"),
            "msg should contain needed=2100, got: {msg}"
        );
        assert!(
            msg.contains("1024"),
            "msg should contain max=1024, got: {msg}"
        );
    }

    #[test]
    fn dynamic_cap_from_slots_bounded_by_cap_max_and_gpu_floor() {
        // B1-p2.3f: cap = max(min(slots_max, effective_cap_max), MIN_KV_CACHE_CAP_FOR_GPU_PERF).
        // The GPU-perf floor is applied by Scheduler before handing the cap
        // to `make_cache` (so callers `prefill_admitted_inner` and
        // `admit_mid_inner` consistently pass a kernel-friendly cap).
        let mut s = TestScheduler::new(4, 2048, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");

        let req = |prompt_len: usize, max_new: usize| GenerateRequest {
            prompt_ids: vec![0; prompt_len],
            max_new_tokens: max_new,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: vec![],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };

        // Case A: slots_max well above the floor; cap_max does not bind.
        // Admit 3 slots: cap_needed values [50+50=100, 700+100=800, 1300+200=1500].
        s.admit(req(50, 50)).expect("admit 1");
        s.admit(req(700, 100)).expect("admit 2");
        s.admit(req(1300, 200)).expect("admit 3");

        // cap = max(min(1500, 2048), 256) = max(1500, 256) = 1500.
        let cap = s.computed_cap_for_prefill();
        assert_eq!(
            cap, 1500,
            "cap should equal max(slot cap_needed); floor & cap_max don't bind"
        );

        // Case B: cap_max < floor; floor wins.
        let mut s3 = TestScheduler::new(4, 200, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        s3.admit(req(50, 50)).expect("admit (cap_needed=100 < 200)");
        s3.admit(req(150, 30))
            .expect("admit (cap_needed=180 < 200)");
        // cap = max(min(180, 200), 256) = max(180, 256) = 256. Floor exceeds cap_max.
        let cap3 = s3.computed_cap_for_prefill();
        assert_eq!(
            cap3, 256,
            "cap = max(180, 256) = 256; floor exceeds user cap_max (allowed — physical-buffer concern)"
        );

        // Case C: small slots_max, no cap_max binding. Floor binds.
        let mut s_floor =
            TestScheduler::new(4, 2048, crate::core::memory_budget::test_meta_qwen35())
                .expect("scheduler startup");
        s_floor.admit(req(50, 50)).expect("admit cap_needed=100");
        let cap_floor = s_floor.computed_cap_for_prefill();
        assert_eq!(
            cap_floor, 256,
            "cap = max(min(100, 2048), 256) = 256; floor binds"
        );

        // Case D: empty-slot fallback. slots_max defaults to 256
        // (defensive — not reachable in production).
        let s4 = TestScheduler::new(4, 1000, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        assert_eq!(
            s4.computed_cap_for_prefill(),
            256,
            "empty slots fallback = 256 (defensive default)"
        );
    }

    // ─── B1-p2.3c+ chunked admit_mid helper unit tests ──────────────────

    #[test]
    fn vl_image_pad_crosses_chunk_boundary_detects_run_across() {
        // image_token_id=42, run at positions 250..260, chunk_size=256.
        // Run crosses 256-boundary (positions 250-255 in chunk 0, 256-259 in chunk 1).
        let ids: Vec<u32> = (0..400_u32)
            .map(|i| {
                if (250..260).contains(&(i as i32)) {
                    42
                } else {
                    1
                }
            })
            .collect();
        assert!(
            super::vl_image_pad_crosses_chunk_boundary(&ids, 42, 256),
            "run [250..260] should cross 256-boundary at chunk_size=256"
        );
        // chunk_size=512 → entire run fits in chunk 0; no crossing.
        assert!(
            !super::vl_image_pad_crosses_chunk_boundary(&ids, 42, 512),
            "run [250..260] should NOT cross at chunk_size=512"
        );
    }

    #[test]
    fn vl_image_pad_no_pads_returns_false() {
        // Empty pad run set.
        let ids: Vec<u32> = (0..200_u32).collect();
        assert!(
            !super::vl_image_pad_crosses_chunk_boundary(&ids, 42, 64),
            "no image_pad tokens → no crossing possible"
        );
        // Also degenerate: empty prompt.
        assert!(
            !super::vl_image_pad_crosses_chunk_boundary(&[], 42, 64),
            "empty prompt → no crossing"
        );
        // Degenerate: image_token_id < 0 disables the check.
        let ids2: Vec<u32> = vec![5; 100];
        assert!(
            !super::vl_image_pad_crosses_chunk_boundary(&ids2, -1, 32),
            "image_token_id < 0 disables detection"
        );
    }

    #[test]
    fn vl_image_pad_run_within_single_chunk_returns_false() {
        // image_pad run [100..150], chunk_size=256 — all in chunk 0.
        let ids: Vec<u32> = (0..200_u32)
            .map(|i| {
                if (100..150).contains(&(i as i32)) {
                    42
                } else {
                    1
                }
            })
            .collect();
        assert!(
            !super::vl_image_pad_crosses_chunk_boundary(&ids, 42, 256),
            "run [100..150] within chunk 0 should NOT cross"
        );
        // Adjacent boundary case: run ends exactly at chunk boundary.
        // Run [200..256], chunk_size=256. Run start chunk = 200/256 = 0.
        // Run end-1 = 255, 255/256 = 0. Same chunk → no crossing.
        let ids2: Vec<u32> = (0..400_u32)
            .map(|i| {
                if (200..256).contains(&(i as i32)) {
                    42
                } else {
                    1
                }
            })
            .collect();
        assert!(
            !super::vl_image_pad_crosses_chunk_boundary(&ids2, 42, 256),
            "run [200..256] ends exactly at boundary — fits in chunk 0"
        );
    }
}
