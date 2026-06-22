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

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

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

use crate::core::cache::{
    MtpCache, PagedPrefixCacheConfig, PagedPrefixEntry, PagedPrefixEntryStats,
    PagedPrefixLoadStatus, PagedPrefixStore, PrefixLruCache, PrefixLruCacheConfig,
    PrefixLruInsertStatus, PrefixMtpLayerSpec, PrefixTensorSpec, TurboQuantKVBits,
};
use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_decode_position_ids,
    build_per_row_decode_mask, build_position_ids, build_position_ids_batched,
    build_position_ids_vl, build_position_ids_vl_batched, count_image_pad,
    extend_vl_chunk_end_for_image_pad, log_vl_chunk_composition, slice_logits_row,
    slice_pos_ids_axis2, slice_vision_embeds_rows, GenerateRequest,
};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::core::speculative::{
    add_elapsed_us, adjust_mtp_draft_budget, commit_mtp_cache_hidden_prefix,
    commit_mtp_cache_hidden_tail, resolve_speculative_tokens, restore_layer_cache,
    sample_logits_positions, slice_hidden_position, verify_input, zero_hidden_like_position,
    MtpDraftResult, MtpSpeculativeConfig, MtpSpeculativeModel, MtpSpeculativeStats,
};
use crate::nn::{
    enable_paged_kv_caches, enable_turboquant_kv_caches, prefix_entry_for_row,
    prefix_key_spec_for_caches, restore_prefix_entry_for_row, restore_prefix_entry_for_rows,
    LayerCache,
};

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
type PixelValuesSlice<'a> = Option<&'a [Array]>;
type PrefixLruCacheHandle = Arc<Mutex<PrefixLruCache>>;

struct SchedulerMtpRowState {
    mtp_cache: MtpCache,
    pending_tokens: VecDeque<u32>,
    last_hidden: Array,
    adaptive_draft_tokens: usize,
}

struct SchedulerMtpState {
    cfg: MtpSpeculativeConfig,
    rows: HashMap<usize, SchedulerMtpRowState>,
    stats: MtpSpeculativeStats,
}

/// Extension trait for VL-capable models, intentionally NOT part of `core::Model`
/// (per P5 spec §3.1 — VL methods stay inherent / extension-trait-only).
///
/// Implemented by Qwen3.5 variants that expose the scheduler-facing VL runtime
/// surface. `Scheduler<M>` methods that call VL code paths (vision tower +
/// cross-modal scatter + VL prefill) require `M: Model + DenseVlMethods`.
pub trait DenseVlMethods {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batched_prefill_vl(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&[mlx::Array]>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;

    fn compute_vision_embeds(
        &self,
        pixel_values: &[mlx::Array],
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

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_hidden(
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
        per_row_pixel_values: &[Option<&[mlx::Array]>],
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
        pixel_values: &[mlx::Array],
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

    fn forward_vl_hidden(
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
        crate::models::qwen3_5::Qwen35Model::forward_vl_hidden(
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

fn maybe_build_decode_mask(mask_row_lens: &[i32], max_real_len: i32) -> Result<Option<Array>> {
    if mask_row_lens.iter().all(|&len| len == max_real_len) {
        return Ok(None);
    }

    #[cfg(feature = "p5h-profile")]
    {
        crate::core::p5h::try_with_p5h_span_from_current_trace(
            "scheduler_decode_mask_build",
            crate::core::p5h::SpanFields::default,
            || build_per_row_decode_mask(mask_row_lens, max_real_len, Dtype::Bfloat16),
        )
        .map(Some)
    }
    #[cfg(not(feature = "p5h-profile"))]
    {
        build_per_row_decode_mask(mask_row_lens, max_real_len, Dtype::Bfloat16).map(Some)
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
/// [`Scheduler::admit_mid_finalize`]. `SchedulerActor::driver_loop` owns
/// this between calls and interleaves `Scheduler::step` between chunks so
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
    /// Per-chunk max token count. When `req.prefill_chunk_size == 0`, this
    /// equals `prompt_len` to preserve the "disable chunking" CLI semantic.
    /// Otherwise it equals `req.prefill_chunk_size.max(1)`. VL chunks may
    /// exceed it only when extending a boundary to keep one contiguous image
    /// token run intact.
    pub(crate) chunk_size: i32,
    pub(crate) decode_cadence_mid_chunk_cap: usize,
    pub(crate) chunk_start: i32,
    /// B=1 temp KV cache; `temp_cache.offsets[0]` advances from `0` to
    /// `prompt_len` across the chunk loop.
    pub(crate) temp_cache: Vec<crate::nn::LayerCache>,
    pub(crate) prefix_fingerprint: Option<String>,
    pub(crate) is_vl: bool,
    pub(crate) image_token_id: i32,
    /// Whether `position_ids_full` holds real full-prompt MRoPE ids. When
    /// false, it is a reusable placeholder for models that derive positions
    /// internally.
    pub(crate) position_ids_required: bool,
    /// Pre-computed `[3, 1, prompt_len]` MRoPE position ids for the full
    /// prompt when `position_ids_required` is true; sliced per chunk inside
    /// `admit_mid_chunk`. For VL it incorporates `image_spatial_merge_size`
    /// + `image_grid_thw`.
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
    /// Vision inputs in image order. `None` for text-only rows. `Array`
    /// clone is mlx reference-counted — cheap. Lives until evict.
    pub pixel_values: Option<Vec<Array>>,
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
    /// `GenerateRequest::prefill_chunk_size` at admit time, clamped to i32.
    /// `0` is preserved as "disable chunking" and expanded to prompt length
    /// when `admit_mid_begin` initialises `AdmitMidHandle::chunk_size`.
    pub prefill_chunk_size: i32,
    /// Request-level rolling mid-admit chunk cap selected from the runtime
    /// scheduler profile.
    pub decode_cadence_mid_chunk_cap: usize,
    /// Optional TurboQuant K/V bit-widths for full-attention KV cache reads.
    pub kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
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

/// Read pre-write per-row offsets from the first cache layer that tracks
/// per-row sequence positions (`Full` KV or `Mla` latent). Used by
/// [`Scheduler::step`] to construct the per-row decode mask before the forward.
///
/// All such layers advance their `offsets()` in lockstep across decode steps
/// (per-row offsets diverge across rows but NOT across layers for a given row).
/// Any such layer's offsets view is equivalent — picking the first is arbitrary
/// but consistent. `KVCache::offsets()` and `MlaLatentCache::offsets()` have
/// identical semantics. `Linear` (GatedDelta) caches do not expose per-row
/// offsets and are skipped.
fn first_full_layer_offsets(cache: &[LayerCache]) -> Result<&[i32]> {
    cache
        .iter()
        .find_map(|c| match c {
            LayerCache::Full(kv) => Some(kv.offsets()),
            LayerCache::Mla(mla) => Some(mla.offsets()),
            _ => None,
        })
        .ok_or_else(|| {
            anyhow!(
                "Scheduler::step: no offset-tracking (Full/Mla) layer in cache; per-row offsets unavailable"
            )
        })
}

fn cache_row_cached_len(cache: &[LayerCache], row: usize) -> Result<Option<i32>> {
    if cache.is_empty() {
        return Ok(None);
    }

    let mut cached_len = None;
    for (idx, layer) in cache.iter().enumerate() {
        let layer_offsets = match layer {
            LayerCache::Full(kv) => kv.offsets(),
            LayerCache::Linear(gd) => gd.offsets(),
            LayerCache::Mla(mla) => mla.offsets(),
        };
        let layer_cached_len = *layer_offsets.get(row).ok_or_else(|| {
            anyhow!(
                "cache_row_cached_len: row {} out of range for layer {}",
                row,
                idx
            )
        })?;
        if let Some(expected) = cached_len {
            if layer_cached_len != expected {
                anyhow::bail!(
                    "cache_row_cached_len: layer {idx} cached_len {layer_cached_len} != layer0 {expected}"
                );
            }
        } else {
            cached_len = Some(layer_cached_len);
        }
    }
    Ok(cached_len)
}

fn mtp_cache_row_cached_len(mtp_cache: &MtpCache, row: usize) -> Result<i32> {
    let mut cached_len = None;
    for idx in 0..mtp_cache.num_layers() {
        let layer = mtp_cache.layer(idx);
        let layer_cached_len = *layer.offsets().get(row).ok_or_else(|| {
            anyhow!(
                "mtp_cache_row_cached_len: row {} out of range for layer {}",
                row,
                idx
            )
        })?;
        if let Some(expected) = cached_len {
            if layer_cached_len != expected {
                anyhow::bail!(
                    "mtp_cache_row_cached_len: layer {idx} cached_len {layer_cached_len} != layer0 {expected}"
                );
            }
        } else {
            cached_len = Some(layer_cached_len);
        }
    }
    Ok(cached_len.unwrap_or(0))
}

fn cache_cap_and_dtype(cache: &[LayerCache]) -> Result<(i32, Dtype)> {
    let mut linear_cap = None;
    for layer in cache {
        match layer {
            LayerCache::Full(kv) => return Ok((kv.cap(), kv.dtype())),
            LayerCache::Mla(mla) => return Ok((mla.cap(), mla.dtype())),
            LayerCache::Linear(gd) => {
                linear_cap.get_or_insert(gd.cap());
            }
        };
    }
    linear_cap
        .map(|cap| (cap, Dtype::Bfloat16))
        .ok_or_else(|| anyhow!("cache_cap_and_dtype: cache has no layers"))
}

fn adopt_cache_row_layers(
    dst: &mut [LayerCache],
    src: &[LayerCache],
    dst_row: usize,
    src_row: usize,
    context: &str,
) -> Result<()> {
    if dst.len() != src.len() {
        return Err(anyhow!(
            "{context}: cache layer count mismatch ({} vs {})",
            dst.len(),
            src.len()
        ));
    }
    for (dst_layer, src_layer) in dst.iter_mut().zip(src.iter()) {
        match (dst_layer, src_layer) {
            (LayerCache::Full(dst_kv), LayerCache::Full(src_kv)) => {
                dst_kv.adopt_row_from(src_kv, dst_row, src_row)?;
            }
            (LayerCache::Linear(dst_gd), LayerCache::Linear(src_gd)) => {
                dst_gd.adopt_row_from(src_gd, dst_row, src_row)?;
            }
            (LayerCache::Mla(dst_mla), LayerCache::Mla(src_mla)) => {
                dst_mla.adopt_row_from(src_mla, dst_row, src_row)?;
            }
            _ => return Err(anyhow!("{context}: cache layer kind mismatch")),
        }
    }
    Ok(())
}

fn update_prefix_fingerprint_hash(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn update_prefix_fingerprint_i32(hash: &mut u64, value: i32) {
    update_prefix_fingerprint_hash(hash, &value.to_le_bytes());
}

fn update_prefix_fingerprint_usize(hash: &mut u64, value: usize) {
    update_prefix_fingerprint_hash(hash, &value.to_le_bytes());
}

fn update_prefix_fingerprint_str(hash: &mut u64, value: &str) {
    update_prefix_fingerprint_usize(hash, value.len());
    update_prefix_fingerprint_hash(hash, value.as_bytes());
}

fn generate_request_from_state(state: &RequestState) -> Result<GenerateRequest> {
    Ok(GenerateRequest {
        prompt_ids: state.prompt_ids.clone(),
        max_new_tokens: state.max_new_tokens,
        sampler: state.sampler,
        stop_token_ids: state.stop_token_ids.clone(),
        prefill_chunk_size: usize::try_from(state.prefill_chunk_size)
            .map_err(|_| anyhow!("generate_request_from_state: negative prefill_chunk_size"))?,
        decode_cadence_mid_chunk_cap: state.decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits: state.kv_cache_turboquant_bits,
        pixel_values: state.pixel_values.clone(),
        image_grid_thw: state.image_grid_thw.clone(),
        image_spatial_merge_size: state.image_spatial_merge_size,
        image_token_id: state.image_token_id,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: state.p5h_trace.clone(),
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: state.p5h_root_span.clone(),
    })
}

fn add_mtp_stats(dst: &mut MtpSpeculativeStats, src: MtpSpeculativeStats) {
    dst.windows = dst.windows.saturating_add(src.windows);
    dst.drafted_tokens = dst.drafted_tokens.saturating_add(src.drafted_tokens);
    dst.accepted_draft_tokens = dst
        .accepted_draft_tokens
        .saturating_add(src.accepted_draft_tokens);
    dst.rollback_count = dst.rollback_count.saturating_add(src.rollback_count);
    dst.mtp_cache_reuse_count = dst
        .mtp_cache_reuse_count
        .saturating_add(src.mtp_cache_reuse_count);
    dst.mtp_cache_reused_tokens = dst
        .mtp_cache_reused_tokens
        .saturating_add(src.mtp_cache_reused_tokens);
    dst.draft_budget_reductions = dst
        .draft_budget_reductions
        .saturating_add(src.draft_budget_reductions);
    dst.draft_budget_increases = dst
        .draft_budget_increases
        .saturating_add(src.draft_budget_increases);
    dst.draft_forward_us = dst.draft_forward_us.saturating_add(src.draft_forward_us);
    dst.verify_forward_us = dst.verify_forward_us.saturating_add(src.verify_forward_us);
    dst.projection_us = dst.projection_us.saturating_add(src.projection_us);
    dst.sampling_us = dst.sampling_us.saturating_add(src.sampling_us);
    dst.main_rollback_us = dst.main_rollback_us.saturating_add(src.main_rollback_us);
    dst.mtp_cache_commit_us = dst
        .mtp_cache_commit_us
        .saturating_add(src.mtp_cache_commit_us);
    dst.mtp_cache_restore_us = dst
        .mtp_cache_restore_us
        .saturating_add(src.mtp_cache_restore_us);
}

fn paged_prefix_fingerprint_for_request(
    pixel_values: Option<&[Array]>,
    image_grid_thw: Option<&[(i32, i32, i32)]>,
    image_token_id: i32,
    image_spatial_merge_size: i32,
) -> Result<Option<String>> {
    let Some(pixel_values) = pixel_values else {
        return Ok(None);
    };
    let grids = image_grid_thw.ok_or_else(|| {
        anyhow!("paged prefix cache VL fingerprint: pixel_values present but grid_thw is None")
    })?;

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    update_prefix_fingerprint_str(&mut hash, "ironmlx-vl-prefix-v1");
    update_prefix_fingerprint_i32(&mut hash, image_token_id);
    update_prefix_fingerprint_i32(&mut hash, image_spatial_merge_size);
    update_prefix_fingerprint_usize(&mut hash, grids.len());
    for &(t, h, w) in grids {
        update_prefix_fingerprint_i32(&mut hash, t);
        update_prefix_fingerprint_i32(&mut hash, h);
        update_prefix_fingerprint_i32(&mut hash, w);
    }

    update_prefix_fingerprint_usize(&mut hash, pixel_values.len());
    for pixel_array in pixel_values {
        update_prefix_fingerprint_str(&mut hash, &pixel_array.dtype().to_string());
        let shape = pixel_array.shape();
        update_prefix_fingerprint_usize(&mut hash, shape.as_slice().len());
        for &dim in shape.as_slice() {
            update_prefix_fingerprint_i32(&mut hash, dim);
        }
        let values = mlx::ops::astype(pixel_array, Dtype::Float32)?.to_vec::<f32>()?;
        update_prefix_fingerprint_usize(&mut hash, values.len());
        for value in values {
            update_prefix_fingerprint_hash(&mut hash, &value.to_bits().to_le_bytes());
        }
    }

    Ok(Some(format!("vl:{hash:016x}")))
}

fn try_restore_paged_prefix_for_prompt(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &mut [LayerCache],
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<i32>> {
    try_restore_paged_prefix_for_prompt_row(
        config,
        prefix_lru_cache,
        cache,
        0,
        prompt_ids,
        fingerprint,
    )
}

fn paged_prefix_restore_candidates(
    store: &PagedPrefixStore,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    prompt_len: usize,
) -> Result<Vec<(usize, i32)>> {
    if prompt_len <= 1 {
        return Ok(Vec::new());
    }
    let max_cached_len = i32::try_from(prompt_len - 1)
        .map_err(|_| anyhow!("paged prefix restore length exceeds i32"))?;
    let mut cached_lengths = Vec::new();
    if let Some(prefix_lru_cache) = prefix_lru_cache {
        cached_lengths.extend(
            lock_prefix_lru_cache(prefix_lru_cache)?
                .cached_lengths_descending(max_cached_len as usize),
        );
    }
    cached_lengths.extend(store.cached_lengths_descending(max_cached_len)?);
    cached_lengths.sort_unstable_by(|a, b| b.cmp(a));
    cached_lengths.dedup();
    let mut candidates = Vec::with_capacity(cached_lengths.len());
    for cached_len in cached_lengths {
        let restore_len = usize::try_from(cached_len)
            .map_err(|_| anyhow!("paged prefix cached length must be positive"))?;
        candidates.push((restore_len, cached_len));
    }
    Ok(candidates)
}

fn lock_prefix_lru_cache(
    prefix_lru_cache: &PrefixLruCacheHandle,
) -> Result<MutexGuard<'_, PrefixLruCache>> {
    prefix_lru_cache
        .lock()
        .map_err(|_| anyhow!("prefix LRU cache lock poisoned"))
}

fn try_load_prefix_lru_entry(
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    spec: &crate::core::cache::PagedPrefixKeySpec,
) -> Result<Option<(String, PagedPrefixEntry, PagedPrefixEntryStats, u128)>> {
    let Some(prefix_lru_cache) = prefix_lru_cache else {
        return Ok(None);
    };
    let load_start = Instant::now();
    let observed = lock_prefix_lru_cache(prefix_lru_cache)?.load_observed(spec)?;
    let load_us = load_start.elapsed().as_micros();
    if observed.status != PagedPrefixLoadStatus::Hit {
        tracing::trace!(
            "prefix LRU cache miss: key={} status={:?} load_us={}",
            observed.key,
            observed.status,
            load_us
        );
        return Ok(None);
    }
    let key = observed.key;
    let stats = observed
        .stats
        .unwrap_or_else(|| empty_prefix_stats(spec.cached_len));
    let entry = observed
        .entry
        .ok_or_else(|| anyhow!("prefix LRU observed hit without entry"))?;
    Ok(Some((key, entry, stats, load_us)))
}

fn try_insert_prefix_lru_entry(
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    spec: crate::core::cache::PagedPrefixKeySpec,
    entry: PagedPrefixEntry,
    main_row: usize,
    mtp_row: Option<usize>,
) -> Result<Option<String>> {
    let Some(prefix_lru_cache) = prefix_lru_cache else {
        return Ok(None);
    };
    let save_start = Instant::now();
    let result = lock_prefix_lru_cache(prefix_lru_cache)?.insert(spec, entry)?;
    let save_us = save_start.elapsed().as_micros();
    match result.status {
        PrefixLruInsertStatus::Stored | PrefixLruInsertStatus::Replaced => {
            log_prefix_lru_save(
                match result.status {
                    PrefixLruInsertStatus::Stored => "saved",
                    PrefixLruInsertStatus::Replaced => "updated",
                    PrefixLruInsertStatus::SkippedOversized => unreachable!(),
                },
                &result.key,
                main_row,
                mtp_row,
                result.stats,
                save_us,
            );
            Ok(Some(result.key))
        }
        PrefixLruInsertStatus::SkippedOversized => {
            tracing::trace!(
                "prefix LRU cache save skipped: row={} key={} status=oversized payload_bytes={} max_bytes={}",
                main_row,
                result.key,
                result.stats.payload_bytes,
                lock_prefix_lru_cache(prefix_lru_cache)?.max_bytes()
            );
            Ok(None)
        }
    }
}

fn try_restore_paged_prefix_for_prompt_row(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &mut [LayerCache],
    cache_row: usize,
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<i32>> {
    let Some(config) = config else {
        return Ok(None);
    };
    if prompt_ids.len() <= 1 {
        return Ok(None);
    }
    let store = config.store();
    for (restore_len, cached_len) in
        paged_prefix_restore_candidates(&store, prefix_lru_cache, prompt_ids.len())?
    {
        let Some(spec) = prefix_key_spec_for_caches(
            &config.model_id,
            &prompt_ids[..restore_len],
            cached_len,
            fingerprint,
            config.block_size,
            cache,
        )?
        else {
            return Ok(None);
        };
        if let Some((key, entry, stats, load_us)) =
            try_load_prefix_lru_entry(prefix_lru_cache, &spec)?
        {
            restore_prefix_entry_for_row(cache, &entry, cache_row, cached_len)?;
            log_prefix_lru_hit("hit", &key, cache_row, None, restore_len, stats, load_us);
            return Ok(Some(cached_len));
        }
        let load_start = Instant::now();
        let observed = store.load_observed(&spec)?;
        let load_us = load_start.elapsed().as_micros();
        if observed.status != PagedPrefixLoadStatus::Hit {
            tracing::trace!(
                "paged SSD prefix cache miss: row={} tokens={} key={} status={:?} load_us={}",
                cache_row,
                restore_len,
                observed.key,
                observed.status,
                load_us
            );
            continue;
        };
        let key = observed.key;
        let stats = observed
            .stats
            .unwrap_or_else(|| empty_prefix_stats(cached_len));
        let entry = observed
            .entry
            .ok_or_else(|| anyhow!("paged prefix observed hit without entry"))?;
        try_insert_prefix_lru_entry(prefix_lru_cache, spec, entry.clone(), cache_row, None)?;
        restore_prefix_entry_for_row(cache, &entry, cache_row, cached_len)?;
        log_paged_prefix_hit("hit", &key, cache_row, None, restore_len, stats, load_us);
        return Ok(Some(cached_len));
    }
    Ok(None)
}

fn try_restore_paged_prefix_for_prompt_rows(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &mut [LayerCache],
    cache_rows: &[usize],
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<i32>> {
    if cache_rows.is_empty() {
        return Ok(None);
    }
    if cache_rows.len() == 1 {
        return try_restore_paged_prefix_for_prompt_row(
            config,
            prefix_lru_cache,
            cache,
            cache_rows[0],
            prompt_ids,
            fingerprint,
        );
    }
    let Some(config) = config else {
        return Ok(None);
    };
    if prompt_ids.len() <= 1 {
        return Ok(None);
    }
    let store = config.store();
    for (restore_len, cached_len) in
        paged_prefix_restore_candidates(&store, prefix_lru_cache, prompt_ids.len())?
    {
        let Some(spec) = prefix_key_spec_for_caches(
            &config.model_id,
            &prompt_ids[..restore_len],
            cached_len,
            fingerprint,
            config.block_size,
            cache,
        )?
        else {
            return Ok(None);
        };
        if let Some((key, entry, stats, load_us)) =
            try_load_prefix_lru_entry(prefix_lru_cache, &spec)?
        {
            restore_prefix_entry_for_rows(cache, &entry, cache_rows, cached_len)?;
            for (idx, &cache_row) in cache_rows.iter().enumerate() {
                let row_load_us = if idx == 0 { load_us } else { 0 };
                log_prefix_lru_hit(
                    "hit",
                    &key,
                    cache_row,
                    None,
                    restore_len,
                    stats,
                    row_load_us,
                );
            }
            return Ok(Some(cached_len));
        }
        let load_start = Instant::now();
        let observed = store.load_observed(&spec)?;
        let load_us = load_start.elapsed().as_micros();
        if observed.status != PagedPrefixLoadStatus::Hit {
            tracing::trace!(
                "paged SSD prefix cache miss: rows={:?} tokens={} key={} status={:?} load_us={}",
                cache_rows,
                restore_len,
                observed.key,
                observed.status,
                load_us
            );
            continue;
        };
        let key = observed.key;
        let stats = observed
            .stats
            .unwrap_or_else(|| empty_prefix_stats(cached_len));
        let entry = observed
            .entry
            .ok_or_else(|| anyhow!("paged prefix observed hit without entry"))?;
        try_insert_prefix_lru_entry(prefix_lru_cache, spec, entry.clone(), cache_rows[0], None)?;
        restore_prefix_entry_for_rows(cache, &entry, cache_rows, cached_len)?;
        for (idx, &cache_row) in cache_rows.iter().enumerate() {
            let row_load_us = if idx == 0 { load_us } else { 0 };
            log_paged_prefix_hit(
                "hit",
                &key,
                cache_row,
                None,
                restore_len,
                stats,
                row_load_us,
            );
        }
        return Ok(Some(cached_len));
    }
    Ok(None)
}

fn try_save_paged_prefix_for_prompt(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &[LayerCache],
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<String>> {
    try_save_paged_prefix_for_prompt_row(
        config,
        prefix_lru_cache,
        cache,
        0,
        prompt_ids,
        fingerprint,
    )
}

fn try_save_paged_prefix_for_prompt_row(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &[LayerCache],
    cache_row: usize,
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<String>> {
    let Some(config) = config else {
        return Ok(None);
    };
    if prompt_ids.is_empty() {
        return Ok(None);
    }
    let Some(cached_len) = cache_row_cached_len(cache, cache_row)? else {
        return Ok(None);
    };
    if cached_len == 0 {
        return Ok(None);
    }
    if cached_len != prompt_ids.len() as i32 {
        anyhow::bail!(
            "try_save_paged_prefix_for_prompt: cache cached_len {cached_len} != token length {}",
            prompt_ids.len()
        );
    }
    let Some(spec) = prefix_key_spec_for_caches(
        &config.model_id,
        prompt_ids,
        cached_len,
        fingerprint,
        config.block_size,
        cache,
    )?
    else {
        return Ok(None);
    };
    let Some((entry, cached_len)) = prefix_entry_for_row(cache, cache_row)? else {
        return Ok(None);
    };
    if cached_len == 0 {
        return Ok(None);
    }
    if cached_len != prompt_ids.len() as i32 {
        anyhow::bail!(
            "try_save_paged_prefix_for_prompt: cache cached_len {cached_len} != token length {}",
            prompt_ids.len()
        );
    }
    let stats = entry.observability_stats(cached_len);
    try_insert_prefix_lru_entry(
        prefix_lru_cache,
        spec.clone(),
        entry.clone(),
        cache_row,
        None,
    )?;
    let store = config.store();
    if let Some(key) = store.matching_entry_key(&spec)? {
        tracing::trace!(
            "paged SSD prefix cache save skipped: row={} tokens={} key={} status=already_present",
            cache_row,
            prompt_ids.len(),
            key
        );
        return Ok(None);
    }
    let save_start = Instant::now();
    let (key, saved) = store.save_if_absent(&spec, &entry)?;
    let save_us = save_start.elapsed().as_micros();
    if !saved {
        tracing::trace!(
            "paged SSD prefix cache save skipped: row={} tokens={} key={} status=already_present",
            cache_row,
            prompt_ids.len(),
            key
        );
        return Ok(None);
    }
    log_paged_prefix_save("saved", &key, cache_row, None, stats, save_us);
    Ok(Some(key))
}

fn mtp_layer_specs_for_cache(mtp_cache: &MtpCache, cached_len: i32) -> Vec<PrefixMtpLayerSpec> {
    (0..mtp_cache.num_layers())
        .map(|idx| {
            let layer = mtp_cache.layer(idx);
            PrefixMtpLayerSpec {
                k: PrefixTensorSpec {
                    dtype: layer.dtype(),
                    shape: vec![1_i32, layer.n_kv_heads(), cached_len, layer.head_dim()],
                },
                v: PrefixTensorSpec {
                    dtype: layer.dtype(),
                    shape: vec![1_i32, layer.n_kv_heads(), cached_len, layer.v_head_dim()],
                },
            }
        })
        .collect()
}

fn mtp_last_hidden_spec(dtype: Dtype, hidden_size: i32) -> PrefixTensorSpec {
    PrefixTensorSpec {
        dtype,
        shape: vec![1_i32, 1_i32, hidden_size],
    }
}

#[allow(clippy::too_many_arguments)]
fn try_restore_paged_prefix_for_prompt_with_mtp(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    main_cache: &mut [LayerCache],
    mtp_cache: &mut MtpCache,
    prompt_ids: &[u32],
    hidden_size: i32,
    hidden_dtype: Dtype,
    fingerprint: Option<&str>,
) -> Result<Option<(i32, Array)>> {
    try_restore_paged_prefix_for_prompt_with_mtp_row(
        config,
        prefix_lru_cache,
        main_cache,
        0,
        mtp_cache,
        0,
        prompt_ids,
        hidden_size,
        hidden_dtype,
        fingerprint,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_restore_paged_prefix_for_prompt_with_mtp_row(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    main_cache: &mut [LayerCache],
    main_cache_row: usize,
    mtp_cache: &mut MtpCache,
    mtp_cache_row: usize,
    prompt_ids: &[u32],
    hidden_size: i32,
    hidden_dtype: Dtype,
    fingerprint: Option<&str>,
) -> Result<Option<(i32, Array)>> {
    let Some(config) = config else {
        return Ok(None);
    };
    if prompt_ids.len() <= 1 {
        return Ok(None);
    }
    let store = config.store();
    for (restore_len, cached_len) in
        paged_prefix_restore_candidates(&store, prefix_lru_cache, prompt_ids.len())?
    {
        let Some(mut spec) = prefix_key_spec_for_caches(
            &config.model_id,
            &prompt_ids[..restore_len],
            cached_len,
            fingerprint,
            config.block_size,
            main_cache,
        )?
        else {
            return Ok(None);
        };
        spec.mtp_layers = mtp_layer_specs_for_cache(mtp_cache, cached_len);
        spec.mtp_last_hidden = Some(mtp_last_hidden_spec(hidden_dtype, hidden_size));
        if let Some((key, entry, stats, load_us)) =
            try_load_prefix_lru_entry(prefix_lru_cache, &spec)?
        {
            restore_prefix_entry_for_row(main_cache, &entry, main_cache_row, cached_len)?;
            mtp_cache.restore_prefix_layers_for_row_on(
                &entry.mtp_layers,
                mtp_cache_row,
                cached_len,
                (),
            )?;
            let last_hidden = entry
                .mtp_last_hidden
                .ok_or_else(|| anyhow!("prefix LRU MTP hit missing last_hidden"))?;
            log_prefix_lru_hit(
                "MTP hit",
                &key,
                main_cache_row,
                Some(mtp_cache_row),
                restore_len,
                stats,
                load_us,
            );
            return Ok(Some((cached_len, last_hidden)));
        }
        let load_start = Instant::now();
        let observed = store.load_observed(&spec)?;
        let load_us = load_start.elapsed().as_micros();
        if observed.status != PagedPrefixLoadStatus::Hit {
            tracing::trace!(
                "paged SSD prefix cache MTP miss: main_row={} mtp_row={} tokens={} key={} status={:?} load_us={}",
                main_cache_row,
                mtp_cache_row,
                restore_len,
                observed.key,
                observed.status,
                load_us
            );
            continue;
        };
        let key = observed.key;
        let stats = observed
            .stats
            .unwrap_or_else(|| empty_prefix_stats(cached_len));
        let entry = observed
            .entry
            .ok_or_else(|| anyhow!("paged prefix observed MTP hit without entry"))?;
        try_insert_prefix_lru_entry(
            prefix_lru_cache,
            spec,
            entry.clone(),
            main_cache_row,
            Some(mtp_cache_row),
        )?;
        restore_prefix_entry_for_row(main_cache, &entry, main_cache_row, cached_len)?;
        mtp_cache.restore_prefix_layers_for_row_on(
            &entry.mtp_layers,
            mtp_cache_row,
            cached_len,
            (),
        )?;
        let last_hidden = entry
            .mtp_last_hidden
            .ok_or_else(|| anyhow!("paged prefix MTP hit missing last_hidden"))?;
        log_paged_prefix_hit(
            "MTP hit",
            &key,
            main_cache_row,
            Some(mtp_cache_row),
            restore_len,
            stats,
            load_us,
        );
        return Ok(Some((cached_len, last_hidden)));
    }
    Ok(None)
}

fn try_save_paged_prefix_for_prompt_with_mtp(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    main_cache: &[LayerCache],
    mtp_cache: &MtpCache,
    last_hidden: &Array,
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<String>> {
    try_save_paged_prefix_for_prompt_with_mtp_row(
        config,
        prefix_lru_cache,
        main_cache,
        0,
        mtp_cache,
        0,
        last_hidden,
        prompt_ids,
        fingerprint,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_save_paged_prefix_for_prompt_with_mtp_row(
    config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    main_cache: &[LayerCache],
    main_cache_row: usize,
    mtp_cache: &MtpCache,
    mtp_cache_row: usize,
    last_hidden: &Array,
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
) -> Result<Option<String>> {
    let Some(config) = config else {
        return Ok(None);
    };
    if prompt_ids.is_empty() {
        return Ok(None);
    }
    let Some(cached_len) = cache_row_cached_len(main_cache, main_cache_row)? else {
        return Ok(None);
    };
    if cached_len == 0 {
        return Ok(None);
    }
    if cached_len != prompt_ids.len() as i32 {
        anyhow::bail!(
            "try_save_paged_prefix_for_prompt_with_mtp: main row {main_cache_row} cached_len {cached_len} != token length {}",
            prompt_ids.len()
        );
    }
    let mtp_cached_len = mtp_cache_row_cached_len(mtp_cache, mtp_cache_row)?;
    if mtp_cached_len != cached_len {
        anyhow::bail!(
            "try_save_paged_prefix_for_prompt_with_mtp: MTP row {mtp_cache_row} cached_len {mtp_cached_len} != main {cached_len}"
        );
    }
    let Some(mut spec) = prefix_key_spec_for_caches(
        &config.model_id,
        prompt_ids,
        cached_len,
        fingerprint,
        config.block_size,
        main_cache,
    )?
    else {
        return Ok(None);
    };
    spec.mtp_layers = mtp_layer_specs_for_cache(mtp_cache, cached_len);
    spec.mtp_last_hidden = Some(PrefixTensorSpec::from_array(last_hidden));
    let Some((mut entry, cached_len)) = prefix_entry_for_row(main_cache, main_cache_row)? else {
        return Ok(None);
    };
    if cached_len == 0 {
        return Ok(None);
    }
    if cached_len != prompt_ids.len() as i32 {
        anyhow::bail!(
            "try_save_paged_prefix_for_prompt_with_mtp: main row {main_cache_row} cached_len {cached_len} != token length {}",
            prompt_ids.len()
        );
    }
    let (mtp_layers, mtp_cached_len) = mtp_cache.prefix_layers_for_row_on(mtp_cache_row, ())?;
    if mtp_cached_len != cached_len {
        anyhow::bail!(
            "try_save_paged_prefix_for_prompt_with_mtp: MTP row {mtp_cache_row} cached_len {mtp_cached_len} != main {cached_len}"
        );
    }
    entry.mtp_layers = mtp_layers;
    entry.mtp_last_hidden = Some(last_hidden.clone());

    spec.mtp_layers = entry.mtp_layer_specs();
    spec.mtp_last_hidden = entry.mtp_last_hidden_spec();
    let stats = entry.observability_stats(cached_len);
    try_insert_prefix_lru_entry(
        prefix_lru_cache,
        spec.clone(),
        entry.clone(),
        main_cache_row,
        Some(mtp_cache_row),
    )?;
    let store = config.store();
    if let Some(key) = store.matching_entry_key(&spec)? {
        tracing::trace!(
            "paged SSD prefix cache MTP save skipped: main_row={} mtp_row={} tokens={} key={} status=already_present",
            main_cache_row,
            mtp_cache_row,
            prompt_ids.len(),
            key
        );
        return Ok(None);
    }
    let save_start = Instant::now();
    let (key, saved) = store.save_if_absent(&spec, &entry)?;
    let save_us = save_start.elapsed().as_micros();
    if !saved {
        tracing::trace!(
            "paged SSD prefix cache MTP save skipped: main_row={} mtp_row={} tokens={} key={} status=already_present",
            main_cache_row,
            mtp_cache_row,
            prompt_ids.len(),
            key
        );
        return Ok(None);
    }
    log_paged_prefix_save(
        "MTP saved",
        &key,
        main_cache_row,
        Some(mtp_cache_row),
        stats,
        save_us,
    );
    Ok(Some(key))
}

fn empty_prefix_stats(cached_len: i32) -> PagedPrefixEntryStats {
    PagedPrefixEntryStats {
        cached_len,
        ..PagedPrefixEntryStats::default()
    }
}

fn log_paged_prefix_hit(
    label: &str,
    key: &str,
    main_row: usize,
    mtp_row: Option<usize>,
    restored_tokens: usize,
    stats: PagedPrefixEntryStats,
    load_us: u128,
) {
    match mtp_row {
        Some(mtp_row) => tracing::info!(
            "paged SSD prefix cache {label}: key={} main_row={} mtp_row={} tokens={} restored={} load_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            mtp_row,
            stats.cached_len,
            restored_tokens,
            load_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
        None => tracing::info!(
            "paged SSD prefix cache {label}: key={} row={} tokens={} restored={} load_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            stats.cached_len,
            restored_tokens,
            load_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
    }
}

fn log_paged_prefix_save(
    label: &str,
    key: &str,
    main_row: usize,
    mtp_row: Option<usize>,
    stats: PagedPrefixEntryStats,
    save_us: u128,
) {
    match mtp_row {
        Some(mtp_row) => tracing::info!(
            "paged SSD prefix cache {label}: key={} main_row={} mtp_row={} tokens={} save_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            mtp_row,
            stats.cached_len,
            save_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
        None => tracing::info!(
            "paged SSD prefix cache {label}: key={} row={} tokens={} save_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            stats.cached_len,
            save_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
    }
}

fn log_prefix_lru_hit(
    label: &str,
    key: &str,
    main_row: usize,
    mtp_row: Option<usize>,
    restored_tokens: usize,
    stats: PagedPrefixEntryStats,
    load_us: u128,
) {
    match mtp_row {
        Some(mtp_row) => tracing::info!(
            "prefix LRU cache {label}: key={} main_row={} mtp_row={} tokens={} restored={} load_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            mtp_row,
            stats.cached_len,
            restored_tokens,
            load_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
        None => tracing::info!(
            "prefix LRU cache {label}: key={} row={} tokens={} restored={} load_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            stats.cached_len,
            restored_tokens,
            load_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
    }
}

fn log_prefix_lru_save(
    label: &str,
    key: &str,
    main_row: usize,
    mtp_row: Option<usize>,
    stats: PagedPrefixEntryStats,
    save_us: u128,
) {
    match mtp_row {
        Some(mtp_row) => tracing::info!(
            "prefix LRU cache {label}: key={} main_row={} mtp_row={} tokens={} save_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            mtp_row,
            stats.cached_len,
            save_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
        None => tracing::info!(
            "prefix LRU cache {label}: key={} row={} tokens={} save_us={} payload_bytes={} tensors={} main_layers={} full_layers={} linear_layers={} mla_layers={} mtp_layers={} full_pages={}",
            key,
            main_row,
            stats.cached_len,
            save_us,
            stats.payload_bytes,
            stats.tensor_count,
            stats.main_layers,
            stats.full_paged_layers,
            stats.linear_layers,
            stats.mla_layers,
            stats.mtp_layers,
            stats.full_paged_pages,
        ),
    }
}

fn forward_single_text_suffix<M: Model>(
    model: &M,
    cache: &mut [LayerCache],
    prompt_ids: &[u32],
    start_pos: i32,
    end_pos: i32,
    dummy_position_ids: Option<&Array>,
) -> Result<Array> {
    if start_pos < 0 || start_pos >= end_pos {
        anyhow::bail!("forward_single_text_suffix: invalid range [{start_pos}, {end_pos})");
    }
    let last_pos = end_pos - 1;
    if start_pos < last_pos {
        let prefix_ids = &prompt_ids[start_pos as usize..last_pos as usize];
        let prefix_len = last_pos - start_pos;
        let prefix_arr: Array = (prefix_ids, &[1_i32, prefix_len][..]).try_into()?;
        let prefix_position_ids = if let Some(dummy) = dummy_position_ids {
            dummy.clone()
        } else {
            build_position_ids(start_pos, prefix_len)?
        };
        let prefix_hidden = model.forward_text_hidden(
            &prefix_arr,
            &prefix_position_ids,
            None,
            None,
            Some(&mut *cache),
            mlx::StreamOrDevice::default(),
        )?;
        mlx::transforms::eval(&[&prefix_hidden])?;
    }

    let last_ids = &prompt_ids[last_pos as usize..end_pos as usize];
    let last_arr: Array = (last_ids, &[1_i32, 1_i32][..]).try_into()?;
    let last_position_ids = if let Some(dummy) = dummy_position_ids {
        dummy.clone()
    } else {
        build_position_ids(last_pos, 1)?
    };
    model.forward_on(
        &last_arr,
        &last_position_ids,
        None,
        None,
        Some(cache),
        mlx::StreamOrDevice::default(),
    )
}

fn maybe_build_sparse_decode_mask(
    cache: &[LayerCache],
    per_row_lens: &[i32],
) -> Result<Option<Array>> {
    let Some(pre_offsets) = cache.iter().find_map(|c| match c {
        LayerCache::Full(kv) => Some(kv.offsets()),
        LayerCache::Mla(mla) => Some(mla.offsets()),
        _ => None,
    }) else {
        return Ok(None);
    };
    anyhow::ensure!(
        pre_offsets.len() == per_row_lens.len(),
        "maybe_build_sparse_decode_mask: offset rows {} != per_row_lens {}",
        pre_offsets.len(),
        per_row_lens.len()
    );
    let mask_row_lens: Vec<i32> = pre_offsets
        .iter()
        .zip(per_row_lens.iter())
        .map(|(&offset, &new_len)| {
            if offset == 0 && new_len == 0 {
                1
            } else {
                offset + new_len
            }
        })
        .collect();
    let max_real_len = mask_row_lens
        .iter()
        .copied()
        .max()
        .expect("per_row_lens is non-empty");
    maybe_build_decode_mask(&mask_row_lens, max_real_len)
}

fn concat_logits_rows(logit_rows: Vec<Array>) -> Result<Array> {
    let mut rows = Vec::with_capacity(logit_rows.len());
    for row in logit_rows {
        let shape = row.shape();
        let shape = shape.as_slice();
        anyhow::ensure!(
            shape.len() == 1,
            "concat_logits_rows: expected rank-1 row logits, got rank {}",
            shape.len()
        );
        rows.push(row.reshape(&[1_i32, 1_i32, shape[0]][..])?);
    }
    let refs: Vec<&Array> = rows.iter().collect();
    mlx::ops::shape::concatenate(&refs, 0).map_err(anyhow::Error::from)
}

fn prefix_restore_groups(
    prompt_ids: &[Vec<u32>],
    fingerprints: Option<&[Option<String>]>,
) -> Result<Vec<Vec<usize>>> {
    if let Some(fingerprints) = fingerprints {
        anyhow::ensure!(
            fingerprints.len() == prompt_ids.len(),
            "prefix_restore_groups: fingerprints rows {} != prompt rows {}",
            fingerprints.len(),
            prompt_ids.len()
        );
    }

    let mut groups = Vec::new();
    let mut grouped = vec![false; prompt_ids.len()];
    for row in 0..prompt_ids.len() {
        if grouped[row] {
            continue;
        }
        let row_fingerprint = fingerprints.and_then(|values| values[row].as_deref());
        let mut group = vec![row];
        grouped[row] = true;
        for other in row + 1..prompt_ids.len() {
            if grouped[other] || prompt_ids[other] != prompt_ids[row] {
                continue;
            }
            let other_fingerprint = fingerprints.and_then(|values| values[other].as_deref());
            if other_fingerprint != row_fingerprint {
                continue;
            }
            grouped[other] = true;
            group.push(other);
        }
        groups.push(group);
    }
    Ok(groups)
}

struct BatchedTextPrefixReplay<'a> {
    prompt_ids: &'a [Vec<u32>],
    dummy_position_ids: Option<&'a Array>,
    prefix_cache_config: Option<&'a PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&'a PrefixLruCacheHandle>,
}

fn batched_text_input_ids(
    prompt_ids: &[Vec<u32>],
    per_row_lens: &[i32],
    max_len: usize,
) -> Result<Array> {
    anyhow::ensure!(
        prompt_ids.len() == per_row_lens.len(),
        "batched_text_input_ids: prompt rows {} != lens {}",
        prompt_ids.len(),
        per_row_lens.len()
    );
    let b = prompt_ids.len();
    let mut flat = vec![0_i32; b * max_len];
    for (row, ids) in prompt_ids.iter().enumerate() {
        let len = usize::try_from(per_row_lens[row])
            .map_err(|_| anyhow!("batched_text_input_ids: negative row length"))?;
        anyhow::ensure!(
            len <= ids.len(),
            "batched_text_input_ids: row {row} len {len} exceeds prompt len {}",
            ids.len()
        );
        anyhow::ensure!(
            len <= max_len,
            "batched_text_input_ids: row {row} len {len} exceeds max_len {max_len}"
        );
        for col in 0..len {
            flat[row * max_len + col] = ids[col] as i32;
        }
    }
    (&flat[..], &[b as i32, max_len as i32][..])
        .try_into()
        .map_err(anyhow::Error::from)
}

fn try_save_batched_text_prefix_row(
    prefix_cache_config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &[LayerCache],
    row: usize,
    prompt_ids: &[u32],
    label: &str,
) {
    match try_save_paged_prefix_for_prompt_row(
        prefix_cache_config,
        prefix_lru_cache,
        cache,
        row,
        prompt_ids,
        None,
    ) {
        Ok(Some(key)) => {
            tracing::debug!(
                "paged SSD prefix cache saved batched text {label}: row={row} key={key}"
            );
        }
        Ok(None) => {}
        Err(err) => {
            tracing::warn!(
                "paged SSD prefix cache batched text {label} save skipped: row={row} {err:#}"
            );
        }
    }
}

fn forward_batched_text_cold_miss_with_paged_prefix<M: Model>(
    model: &M,
    cache: &mut [LayerCache],
    prompt_ids: &[Vec<u32>],
    prefix_cache_config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
) -> Result<Array> {
    let b = prompt_ids.len();
    anyhow::ensure!(
        b > 0,
        "forward_batched_text_cold_miss_with_paged_prefix: empty batch"
    );
    anyhow::ensure!(
        prompt_ids.iter().all(|ids| ids.len() > 1),
        "forward_batched_text_cold_miss_with_paged_prefix: prompt length must be > 1"
    );

    let mut prefix_lens = Vec::with_capacity(b);
    for ids in prompt_ids {
        prefix_lens.push(
            i32::try_from(ids.len() - 1)
                .map_err(|_| anyhow!("batched text prefix length exceeds i32"))?,
        );
    }
    let max_prefix_len = prefix_lens
        .iter()
        .copied()
        .max()
        .expect("batch is non-empty");
    let prefix_input_ids =
        batched_text_input_ids(prompt_ids, &prefix_lens, max_prefix_len as usize)?;
    let attention_mask = build_batch_attention_mask(&prefix_lens, max_prefix_len, Dtype::Bfloat16)?;
    let linear_attention_mask = build_batch_linear_mask(&prefix_lens, max_prefix_len)?;
    let position_ids = build_position_ids_batched(&prefix_lens, max_prefix_len)?;
    let prefix_logits = model.batched_prefill(
        &prefix_input_ids,
        &position_ids,
        &attention_mask,
        &linear_attention_mask,
        &prefix_lens,
        Some(&mut *cache),
        mlx::StreamOrDevice::default(),
    )?;
    mlx::transforms::eval(&[&prefix_logits])?;

    for (row, ids) in prompt_ids.iter().enumerate() {
        try_save_batched_text_prefix_row(
            prefix_cache_config,
            prefix_lru_cache,
            cache,
            row,
            &ids[..ids.len() - 1],
            "prefix",
        );
    }

    let last_tokens: Vec<i32> = prompt_ids
        .iter()
        .map(|ids| ids.last().copied().expect("prompt len > 1") as i32)
        .collect();
    let last_input_ids: Array = (&last_tokens[..], &[b as i32, 1_i32][..]).try_into()?;
    let per_row_lens = vec![1_i32; b];
    let position_ids = build_decode_position_ids(&prefix_lens)?;
    let decode_mask = maybe_build_sparse_decode_mask(cache, &per_row_lens)?;
    let logits = model.forward_on(
        &last_input_ids,
        &position_ids,
        Some(&per_row_lens),
        decode_mask.as_ref(),
        Some(&mut *cache),
        mlx::StreamOrDevice::default(),
    )?;
    mlx::transforms::eval(&[&logits])?;

    for (row, ids) in prompt_ids.iter().enumerate() {
        try_save_batched_text_prefix_row(
            prefix_cache_config,
            prefix_lru_cache,
            cache,
            row,
            ids,
            "prompt",
        );
    }

    Ok(logits)
}

fn forward_batched_text_with_paged_prefix<M: Model>(
    model: &M,
    cache: &mut [LayerCache],
    input: BatchedTextPrefixReplay<'_>,
) -> Result<Array> {
    let BatchedTextPrefixReplay {
        prompt_ids,
        dummy_position_ids,
        prefix_cache_config,
        prefix_lru_cache,
    } = input;
    anyhow::ensure!(
        !prompt_ids.is_empty(),
        "forward_batched_text_with_paged_prefix: empty batch"
    );
    let b = prompt_ids.len();
    let mut replay_from = vec![0_usize; b];
    for group in prefix_restore_groups(prompt_ids, None)? {
        let first_row = group[0];
        let restored = try_restore_paged_prefix_for_prompt_rows(
            prefix_cache_config,
            prefix_lru_cache,
            cache,
            &group,
            &prompt_ids[first_row],
            None,
        )?
        .unwrap_or(0) as usize;
        for row in group {
            replay_from[row] = restored;
        }
    }

    if dummy_position_ids.is_none()
        && replay_from.iter().all(|&pos| pos == 0)
        && prompt_ids.iter().all(|ids| ids.len() > 1)
    {
        return forward_batched_text_cold_miss_with_paged_prefix(
            model,
            cache,
            prompt_ids,
            prefix_cache_config,
            prefix_lru_cache,
        );
    }

    let max_len = prompt_ids.iter().map(Vec::len).max().unwrap_or(0);
    let mut final_logits: Vec<Option<Array>> = (0..b).map(|_| None).collect();

    for pos in 0..max_len {
        let mut token_flat = vec![0_i32; b];
        let mut per_row_lens = vec![0_i32; b];
        let mut per_row_pos = vec![0_i32; b];
        let mut has_active = false;

        for (row, ids) in prompt_ids.iter().enumerate() {
            if pos < replay_from[row] || pos >= ids.len() {
                continue;
            }
            token_flat[row] = ids[pos] as i32;
            per_row_lens[row] = 1;
            per_row_pos[row] = pos as i32;
            has_active = true;
        }
        if !has_active {
            continue;
        }

        let input_ids: Array = (&token_flat[..], &[b as i32, 1_i32][..]).try_into()?;
        let position_ids = if let Some(dummy) = dummy_position_ids {
            dummy.clone()
        } else {
            build_decode_position_ids(&per_row_pos)?
        };
        let decode_mask = maybe_build_sparse_decode_mask(cache, &per_row_lens)?;
        let logits = model.forward_on(
            &input_ids,
            &position_ids,
            Some(&per_row_lens),
            decode_mask.as_ref(),
            Some(&mut *cache),
            mlx::StreamOrDevice::default(),
        )?;
        mlx::transforms::eval(&[&logits])?;

        for (row, ids) in prompt_ids.iter().enumerate() {
            if per_row_lens[row] == 0 {
                continue;
            }
            if ids.len() > 1 && pos + 2 == ids.len() {
                match try_save_paged_prefix_for_prompt_row(
                    prefix_cache_config,
                    prefix_lru_cache,
                    cache,
                    row,
                    &ids[..ids.len() - 1],
                    None,
                ) {
                    Ok(Some(key)) => {
                        tracing::debug!(
                            "paged SSD prefix cache saved batched text prefix: row={row} key={key}"
                        );
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!(
                            "paged SSD prefix cache batched text prefix save skipped: row={row} {err:#}"
                        );
                    }
                }
            }
            if pos + 1 == ids.len() {
                match try_save_paged_prefix_for_prompt_row(
                    prefix_cache_config,
                    prefix_lru_cache,
                    cache,
                    row,
                    ids,
                    None,
                ) {
                    Ok(Some(key)) => {
                        tracing::debug!(
                            "paged SSD prefix cache saved batched text prompt: row={row} key={key}"
                        );
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!(
                            "paged SSD prefix cache batched text prompt save skipped: row={row} {err:#}"
                        );
                    }
                }
                final_logits[row] = Some(slice_logits_row(&logits, row)?);
            }
        }
    }

    let mut rows = Vec::with_capacity(b);
    for (row, logits) in final_logits.into_iter().enumerate() {
        rows.push(logits.ok_or_else(|| {
            anyhow!("forward_batched_text_with_paged_prefix: missing final logits for row {row}")
        })?);
    }
    concat_logits_rows(rows)
}

fn build_vl_position_stream(
    prompt_ids: &[i32],
    grid_thw: GridThwSlice<'_>,
    image_token_id: i32,
    image_spatial_merge_size: i32,
) -> Result<Vec<i32>> {
    let len = prompt_ids.len();
    anyhow::ensure!(!prompt_ids.is_empty(), "VL prefix replay: empty prompt row");
    match grid_thw {
        Some(grids) if !grids.is_empty() => {
            let position_ids =
                build_position_ids_vl(prompt_ids, grids, image_token_id, image_spatial_merge_size)?;
            position_ids
                .to_vec::<i32>()
                .map_err(|e| anyhow!("VL prefix replay position_ids to_vec failed: {e}"))
        }
        _ => {
            let mut flat = vec![0_i32; 3 * len];
            for col in 0..len {
                flat[col] = col as i32;
                flat[len + col] = col as i32;
                flat[2 * len + col] = col as i32;
            }
            Ok(flat)
        }
    }
}

struct BatchedVlPrefixReplay<'a> {
    prompt_ids: &'a [Vec<i32>],
    pixel_values: &'a [PixelValuesSlice<'a>],
    grid_thw: &'a [GridThwSlice<'a>],
    image_token_id: i32,
    image_spatial_merge_size: i32,
    dummy_position_ids: Option<&'a Array>,
    prefix_cache_config: Option<&'a PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&'a PrefixLruCacheHandle>,
}

fn batched_vl_input_ids(
    prompt_ids: &[Vec<i32>],
    per_row_lens: &[i32],
    max_len: usize,
) -> Result<Array> {
    anyhow::ensure!(
        prompt_ids.len() == per_row_lens.len(),
        "batched_vl_input_ids: prompt rows {} != lens {}",
        prompt_ids.len(),
        per_row_lens.len()
    );
    let b = prompt_ids.len();
    let mut flat = vec![0_i32; b * max_len];
    for (row, ids) in prompt_ids.iter().enumerate() {
        let len = usize::try_from(per_row_lens[row])
            .map_err(|_| anyhow!("batched_vl_input_ids: negative row length"))?;
        anyhow::ensure!(
            len <= ids.len(),
            "batched_vl_input_ids: row {row} len {len} exceeds prompt len {}",
            ids.len()
        );
        anyhow::ensure!(
            len <= max_len,
            "batched_vl_input_ids: row {row} len {len} exceeds max_len {max_len}"
        );
        for col in 0..len {
            flat[row * max_len + col] = ids[col];
        }
    }
    (&flat[..], &[b as i32, max_len as i32][..])
        .try_into()
        .map_err(anyhow::Error::from)
}

fn batched_vl_last_position_ids(
    prompt_ids: &[Vec<i32>],
    grid_thw: &[GridThwSlice<'_>],
    image_token_id: i32,
    image_spatial_merge_size: i32,
) -> Result<Array> {
    let b = prompt_ids.len();
    anyhow::ensure!(
        grid_thw.len() == b,
        "batched_vl_last_position_ids: grid rows {} != batch {b}",
        grid_thw.len()
    );
    let mut flat = vec![0_i32; 3 * b];
    for row in 0..b {
        let ids = &prompt_ids[row];
        anyhow::ensure!(
            !ids.is_empty(),
            "batched_vl_last_position_ids: row {row} has empty prompt"
        );
        let row_stream =
            build_vl_position_stream(ids, grid_thw[row], image_token_id, image_spatial_merge_size)?;
        let pos = ids.len() - 1;
        for stream in 0..3 {
            flat[stream * b + row] = row_stream[stream * ids.len() + pos];
        }
    }
    (&flat[..], &[3_i32, b as i32, 1_i32][..])
        .try_into()
        .map_err(anyhow::Error::from)
}

fn try_save_batched_vl_prefix_row(
    prefix_cache_config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    cache: &[LayerCache],
    row: usize,
    prompt_ids: &[u32],
    fingerprint: Option<&str>,
    label: &str,
) {
    match try_save_paged_prefix_for_prompt_row(
        prefix_cache_config,
        prefix_lru_cache,
        cache,
        row,
        prompt_ids,
        fingerprint,
    ) {
        Ok(Some(key)) => {
            tracing::debug!("paged SSD prefix cache saved batched VL {label}: row={row} key={key}");
        }
        Ok(None) => {}
        Err(err) => {
            tracing::warn!(
                "paged SSD prefix cache batched VL {label} save skipped: row={row} {err:#}"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_batched_vl_cold_miss_with_paged_prefix<M: Model + DenseVlMethods>(
    model: &M,
    cache: &mut [LayerCache],
    prompt_ids: &[Vec<i32>],
    prompt_ids_u32: &[Vec<u32>],
    pixel_values: &[PixelValuesSlice<'_>],
    grid_thw: &[GridThwSlice<'_>],
    image_token_id: i32,
    image_spatial_merge_size: i32,
    dummy_position_ids: Option<&Array>,
    prefix_cache_config: Option<&PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<&PrefixLruCacheHandle>,
    fingerprints: &[Option<String>],
) -> Result<Array> {
    let b = prompt_ids.len();
    anyhow::ensure!(
        b > 0,
        "forward_batched_vl_cold_miss_with_paged_prefix: empty batch"
    );
    anyhow::ensure!(
        prompt_ids.iter().all(|ids| ids.len() > 1),
        "forward_batched_vl_cold_miss_with_paged_prefix: prompt length must be > 1"
    );
    anyhow::ensure!(
        prompt_ids_u32.len() == b
            && pixel_values.len() == b
            && grid_thw.len() == b
            && fingerprints.len() == b,
        "forward_batched_vl_cold_miss_with_paged_prefix: per-row inputs must match batch"
    );

    let mut prefix_lens = Vec::with_capacity(b);
    for ids in prompt_ids {
        prefix_lens.push(
            i32::try_from(ids.len() - 1)
                .map_err(|_| anyhow!("batched VL prefix length exceeds i32"))?,
        );
    }
    let max_prefix_len = prefix_lens
        .iter()
        .copied()
        .max()
        .expect("batch is non-empty");
    let max_prefix_len_usize = usize::try_from(max_prefix_len)
        .map_err(|_| anyhow!("batched VL max prefix length is negative"))?;
    let prefix_input_ids = batched_vl_input_ids(prompt_ids, &prefix_lens, max_prefix_len_usize)?;
    let prefix_prompt_refs: Vec<&[i32]> = prompt_ids
        .iter()
        .zip(prefix_lens.iter())
        .map(|(ids, &len)| &ids[..len as usize])
        .collect();
    let prefix_position_ids = if let Some(dummy) = dummy_position_ids {
        dummy.clone()
    } else {
        build_position_ids_vl_batched(
            &prefix_prompt_refs,
            grid_thw,
            image_token_id,
            image_spatial_merge_size,
            max_prefix_len,
        )?
    };
    let attention_mask = build_batch_attention_mask(&prefix_lens, max_prefix_len, Dtype::Bfloat16)?;
    let linear_attention_mask = build_batch_linear_mask(&prefix_lens, max_prefix_len)?;
    let prefix_logits = model.batched_prefill_vl(
        &prefix_input_ids,
        &prefix_position_ids,
        &attention_mask,
        &linear_attention_mask,
        &prefix_lens,
        pixel_values,
        grid_thw,
        image_token_id,
        Some(&mut *cache),
        mlx::StreamOrDevice::default(),
    )?;
    mlx::transforms::eval(&[&prefix_logits])?;

    for row in 0..b {
        try_save_batched_vl_prefix_row(
            prefix_cache_config,
            prefix_lru_cache,
            cache,
            row,
            &prompt_ids_u32[row][..prompt_ids_u32[row].len() - 1],
            fingerprints[row].as_deref(),
            "prefix",
        );
    }

    let last_tokens: Vec<i32> = prompt_ids
        .iter()
        .map(|ids| ids.last().copied().expect("prompt len > 1"))
        .collect();
    let last_input_ids: Array = (&last_tokens[..], &[b as i32, 1_i32][..]).try_into()?;
    let per_row_lens = vec![1_i32; b];
    let position_ids = if let Some(dummy) = dummy_position_ids {
        dummy.clone()
    } else {
        batched_vl_last_position_ids(
            prompt_ids,
            grid_thw,
            image_token_id,
            image_spatial_merge_size,
        )?
    };
    let decode_mask = maybe_build_sparse_decode_mask(cache, &per_row_lens)?;
    let logits = model.forward_vl_chunk(
        &last_input_ids,
        &position_ids,
        Some(&per_row_lens),
        decode_mask.as_ref(),
        Some(&mut *cache),
        None,
        image_token_id,
        mlx::StreamOrDevice::default(),
    )?;
    mlx::transforms::eval(&[&logits])?;

    for row in 0..b {
        try_save_batched_vl_prefix_row(
            prefix_cache_config,
            prefix_lru_cache,
            cache,
            row,
            &prompt_ids_u32[row],
            fingerprints[row].as_deref(),
            "prompt",
        );
    }

    Ok(logits)
}

fn forward_batched_vl_with_paged_prefix<M: Model + DenseVlMethods>(
    model: &M,
    cache: &mut [LayerCache],
    input: BatchedVlPrefixReplay<'_>,
) -> Result<Array> {
    let BatchedVlPrefixReplay {
        prompt_ids,
        pixel_values,
        grid_thw,
        image_token_id,
        image_spatial_merge_size,
        dummy_position_ids,
        prefix_cache_config,
        prefix_lru_cache,
    } = input;
    let b = prompt_ids.len();
    anyhow::ensure!(b > 0, "forward_batched_vl_with_paged_prefix: empty batch");
    anyhow::ensure!(
        pixel_values.len() == b,
        "forward_batched_vl_with_paged_prefix: pixel_values rows {} != batch {b}",
        pixel_values.len()
    );
    anyhow::ensure!(
        grid_thw.len() == b,
        "forward_batched_vl_with_paged_prefix: grid_thw rows {} != batch {b}",
        grid_thw.len()
    );

    let prompt_ids_u32: Vec<Vec<u32>> = prompt_ids
        .iter()
        .map(|ids| ids.iter().map(|&tok| tok as u32).collect())
        .collect();
    let mut fingerprints = Vec::with_capacity(b);
    for row in 0..b {
        let fingerprint = if prefix_cache_config.is_some() {
            paged_prefix_fingerprint_for_request(
                pixel_values[row],
                grid_thw[row],
                image_token_id,
                image_spatial_merge_size,
            )?
        } else {
            None
        };
        fingerprints.push(fingerprint);
    }

    let mut replay_from = vec![0_usize; b];
    for group in prefix_restore_groups(&prompt_ids_u32, Some(&fingerprints))? {
        let first_row = group[0];
        let restored = try_restore_paged_prefix_for_prompt_rows(
            prefix_cache_config,
            prefix_lru_cache,
            cache,
            &group,
            &prompt_ids_u32[first_row],
            fingerprints[first_row].as_deref(),
        )?
        .unwrap_or(0) as usize;
        for row in group {
            replay_from[row] = restored;
        }
    }

    if replay_from.iter().all(|&pos| pos == 0)
        && prompt_ids.iter().all(|ids| ids.len() > 1)
        && prompt_ids
            .iter()
            .all(|ids| ids.last().copied() != Some(image_token_id))
    {
        return forward_batched_vl_cold_miss_with_paged_prefix(
            model,
            cache,
            prompt_ids,
            &prompt_ids_u32,
            pixel_values,
            grid_thw,
            image_token_id,
            image_spatial_merge_size,
            dummy_position_ids,
            prefix_cache_config,
            prefix_lru_cache,
            &fingerprints,
        );
    }

    let position_streams = if dummy_position_ids.is_some() {
        Vec::new()
    } else {
        let mut streams = Vec::with_capacity(b);
        for row in 0..b {
            streams.push(build_vl_position_stream(
                &prompt_ids[row],
                grid_thw[row],
                image_token_id,
                image_spatial_merge_size,
            )?);
        }
        streams
    };

    let mut vision_embeds_full = Vec::with_capacity(b);
    for row in 0..b {
        let embeds = match (pixel_values[row], grid_thw[row]) {
            (Some(pv), Some(grids)) if !grids.is_empty() => {
                Some(model.compute_vision_embeds(pv, grids, mlx::StreamOrDevice::default())?)
            }
            (Some(_), None) => {
                anyhow::bail!(
                    "forward_batched_vl_with_paged_prefix: row {row} has pixel_values but grid_thw is None"
                );
            }
            _ => None,
        };
        vision_embeds_full.push(embeds);
    }

    let max_len = prompt_ids.iter().map(Vec::len).max().unwrap_or(0);
    let mut final_logits: Vec<Option<Array>> = (0..b).map(|_| None).collect();

    for pos in 0..max_len {
        let mut token_flat = vec![0_i32; b];
        let mut per_row_lens = vec![0_i32; b];
        let mut has_active = false;

        for row in 0..b {
            let ids = &prompt_ids[row];
            if pos < replay_from[row] || pos >= ids.len() {
                continue;
            }
            token_flat[row] = ids[pos];
            per_row_lens[row] = 1;
            has_active = true;
        }
        if !has_active {
            continue;
        }

        let input_ids: Array = (&token_flat[..], &[b as i32, 1_i32][..]).try_into()?;
        let position_ids = if let Some(dummy) = dummy_position_ids {
            dummy.clone()
        } else {
            let mut flat = vec![0_i32; 3 * b];
            for row in 0..b {
                if per_row_lens[row] == 0 {
                    continue;
                }
                let row_len = prompt_ids[row].len();
                let row_stream = &position_streams[row];
                for stream in 0..3 {
                    flat[stream * b + row] = row_stream[stream * row_len + pos];
                }
            }
            (&flat[..], &[3_i32, b as i32, 1_i32][..]).try_into()?
        };

        let mut vision_slices = Vec::new();
        for row in 0..b {
            if per_row_lens[row] == 0 || token_flat[row] != image_token_id {
                continue;
            }
            let embeds = vision_embeds_full[row].as_ref().ok_or_else(|| {
                anyhow!(
                    "forward_batched_vl_with_paged_prefix: row {row} image token without vision embeds"
                )
            })?;
            let image_row = prompt_ids[row][..pos]
                .iter()
                .filter(|&&tok| tok == image_token_id)
                .count();
            vision_slices.push(slice_vision_embeds_rows(embeds, image_row, image_row + 1)?);
        }
        let vision_embeds_slice = match vision_slices.len() {
            0 => None,
            1 => vision_slices.pop(),
            _ => {
                let refs: Vec<&Array> = vision_slices.iter().collect();
                Some(
                    mlx::ops::shape::concatenate(&refs, 0)
                        .map_err(|e| anyhow!("VL prefix replay vision concatenate failed: {e}"))?,
                )
            }
        };

        let decode_mask = maybe_build_sparse_decode_mask(cache, &per_row_lens)?;
        let logits = model.forward_vl_chunk(
            &input_ids,
            &position_ids,
            Some(&per_row_lens),
            decode_mask.as_ref(),
            Some(&mut *cache),
            vision_embeds_slice.as_ref(),
            image_token_id,
            mlx::StreamOrDevice::default(),
        )?;
        mlx::transforms::eval(&[&logits])?;

        for row in 0..b {
            if per_row_lens[row] == 0 {
                continue;
            }
            let ids = &prompt_ids[row];
            let ids_u32 = &prompt_ids_u32[row];
            let fingerprint = fingerprints[row].as_deref();
            if ids.len() > 1 && pos + 2 == ids.len() {
                match try_save_paged_prefix_for_prompt_row(
                    prefix_cache_config,
                    prefix_lru_cache,
                    cache,
                    row,
                    &ids_u32[..ids_u32.len() - 1],
                    fingerprint,
                ) {
                    Ok(Some(key)) => {
                        tracing::debug!(
                            "paged SSD prefix cache saved batched VL prefix: row={row} key={key}"
                        );
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!(
                            "paged SSD prefix cache batched VL prefix save skipped: row={row} {err:#}"
                        );
                    }
                }
            }
            if pos + 1 == ids.len() {
                match try_save_paged_prefix_for_prompt_row(
                    prefix_cache_config,
                    prefix_lru_cache,
                    cache,
                    row,
                    ids_u32,
                    fingerprint,
                ) {
                    Ok(Some(key)) => {
                        tracing::debug!(
                            "paged SSD prefix cache saved batched VL prompt: row={row} key={key}"
                        );
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!(
                            "paged SSD prefix cache batched VL prompt save skipped: row={row} {err:#}"
                        );
                    }
                }
                final_logits[row] = Some(slice_logits_row(&logits, row)?);
            }
        }
    }

    let mut rows = Vec::with_capacity(b);
    for (row, logits) in final_logits.into_iter().enumerate() {
        rows.push(logits.ok_or_else(|| {
            anyhow!("forward_batched_vl_with_paged_prefix: missing final logits for row {row}")
        })?);
    }
    concat_logits_rows(rows)
}

fn count_image_pad_i32(tokens: &[i32], image_token_id: i32) -> usize {
    tokens.iter().filter(|&&tok| tok == image_token_id).count()
}

fn slice_vision_embeds_for_image_pads(
    vision_embeds_full: Option<&Array>,
    start: usize,
    count: usize,
) -> Result<Option<Array>> {
    if count == 0 {
        return Ok(None);
    }
    let ve_full = vision_embeds_full
        .ok_or_else(|| anyhow!("slice_vision_embeds_for_image_pads: image pads without embeds"))?;
    Ok(Some(slice_vision_embeds_rows(
        ve_full,
        start,
        start + count,
    )?))
}

struct SingleVlSuffixInput<'a> {
    prompt_ids: &'a [i32],
    start_pos: i32,
    end_pos: i32,
    position_ids_full: &'a Array,
    position_ids_is_dummy: bool,
    vision_embeds_full: Option<&'a Array>,
    image_token_id: i32,
}

fn forward_single_vl_suffix<M: Model + DenseVlMethods>(
    model: &M,
    cache: &mut [LayerCache],
    input: SingleVlSuffixInput<'_>,
) -> Result<Array> {
    let SingleVlSuffixInput {
        prompt_ids,
        start_pos,
        end_pos,
        position_ids_full,
        position_ids_is_dummy,
        vision_embeds_full,
        image_token_id,
    } = input;
    if start_pos < 0 || start_pos >= end_pos {
        anyhow::bail!("forward_single_vl_suffix: invalid range [{start_pos}, {end_pos})");
    }
    let last_pos = end_pos - 1;
    let image_pads_before = count_image_pad_i32(&prompt_ids[..start_pos as usize], image_token_id);
    let mut image_pad_cursor = image_pads_before;

    if start_pos < last_pos {
        let prefix_ids = &prompt_ids[start_pos as usize..last_pos as usize];
        let prefix_len = last_pos - start_pos;
        let prefix_arr: Array = (prefix_ids, &[1_i32, prefix_len][..]).try_into()?;
        let prefix_position_ids = if position_ids_is_dummy {
            position_ids_full.clone()
        } else {
            slice_pos_ids_axis2(position_ids_full, start_pos, last_pos)?
        };
        let prefix_image_pads = count_image_pad_i32(prefix_ids, image_token_id);
        let prefix_vision_embeds = slice_vision_embeds_for_image_pads(
            vision_embeds_full,
            image_pad_cursor,
            prefix_image_pads,
        )?;
        image_pad_cursor += prefix_image_pads;
        let prefix_hidden = model.forward_vl_hidden(
            &prefix_arr,
            &prefix_position_ids,
            None,
            None,
            Some(&mut *cache),
            prefix_vision_embeds.as_ref(),
            image_token_id,
            mlx::StreamOrDevice::default(),
        )?;
        mlx::transforms::eval(&[&prefix_hidden])?;
    }

    let last_ids = &prompt_ids[last_pos as usize..end_pos as usize];
    let last_arr: Array = (last_ids, &[1_i32, 1_i32][..]).try_into()?;
    let last_position_ids = if position_ids_is_dummy {
        position_ids_full.clone()
    } else {
        slice_pos_ids_axis2(position_ids_full, last_pos, end_pos)?
    };
    let last_image_pads = count_image_pad_i32(last_ids, image_token_id);
    let last_vision_embeds =
        slice_vision_embeds_for_image_pads(vision_embeds_full, image_pad_cursor, last_image_pads)?;

    model.forward_vl_chunk(
        &last_arr,
        &last_position_ids,
        None,
        None,
        Some(cache),
        last_vision_embeds.as_ref(),
        image_token_id,
        mlx::StreamOrDevice::default(),
    )
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
    /// Scheduler slot rows represented by `cache` batch rows. The cache is
    /// compact: `cache_rows[i]` is the scheduler slot stored at model batch
    /// row `i`.
    cache_rows: Vec<usize>,
    /// Reusable placeholder for models that derive positions internally and
    /// do not consume caller-built MRoPE position ids.
    dummy_position_ids: Option<Array>,
    /// Optional row-scoped MTP runtime state for active speculative rows.
    mtp_state: Option<SchedulerMtpState>,
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
    /// Optional paged SSD prefix cache. When set, full-attention KV caches use
    /// paged storage; text-only single-row prefill can restore/save prompt
    /// prefixes through the on-disk store.
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    /// Optional process-local hot prefix cache layered above the SSD store.
    prefix_lru_cache: Option<PrefixLruCacheHandle>,
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
            .field("cache_rows", &self.cache_rows)
            .field("has_dummy_position_ids", &self.dummy_position_ids.is_some())
            .field("has_mtp_state", &self.mtp_state.is_some())
            .field("has_prefix_lru_cache", &self.prefix_lru_cache.is_some())
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
            cache_rows: Vec::new(),
            dummy_position_ids: None,
            mtp_state: None,
            poisoned: false,
            effective_cap_max,
            prng_state,
            budget_state,
            meta,
            memory_budget_exceeded_count,
            paged_prefix_cache: None,
            prefix_lru_cache: None,
            _marker: PhantomData,
        })
    }

    pub fn enable_paged_prefix_cache(&mut self, config: PagedPrefixCacheConfig) -> Result<()> {
        config.validate()?;
        self.paged_prefix_cache = Some(config);
        Ok(())
    }

    pub fn enable_prefix_lru_cache(&mut self, config: PrefixLruCacheConfig) -> Result<()> {
        if self.paged_prefix_cache.is_none() {
            anyhow::bail!("prefix LRU cache requires paged prefix cache");
        }
        self.prefix_lru_cache = Some(Arc::new(Mutex::new(PrefixLruCache::new(config)?)));
        Ok(())
    }

    fn share_prefix_lru_cache(&mut self, prefix_lru_cache: PrefixLruCacheHandle) {
        self.prefix_lru_cache = Some(prefix_lru_cache);
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

    fn reusable_dummy_position_ids(&mut self) -> Result<Array> {
        if let Some(position_ids) = self.dummy_position_ids.as_ref() {
            return Ok(position_ids.clone());
        }
        let position_ids = build_position_ids(0, 1)?;
        self.dummy_position_ids = Some(position_ids.clone());
        Ok(position_ids)
    }

    fn kv_cache_turboquant_bits_for_rows(
        &self,
        rows: &[usize],
    ) -> Result<Option<TurboQuantKVBits>> {
        let mut bits = None;
        for &row in rows {
            let Some(state) = self.slots.get(row).and_then(Option::as_ref) else {
                continue;
            };
            if let Some(row_bits) = state.kv_cache_turboquant_bits {
                if let Some(existing) = bits {
                    anyhow::ensure!(
                        existing == row_bits,
                        "scheduler batch mixes TurboQuant KV configs: {existing} and {row_bits}"
                    );
                } else {
                    bits = Some(row_bits);
                }
            }
        }
        Ok(bits)
    }

    fn make_model_cache(
        &self,
        model: &M,
        batch: i32,
        cap: i32,
        dtype: Dtype,
        turboquant_bits: Option<TurboQuantKVBits>,
    ) -> Result<Vec<LayerCache>> {
        if self.paged_prefix_cache.is_some() && turboquant_bits.is_some() {
            anyhow::bail!("paged SSD prefix cache is mutually exclusive with TurboQuant KV cache");
        }
        let mut cache = model.make_cache(batch, cap, dtype)?;
        if let Some(config) = &self.paged_prefix_cache {
            enable_paged_kv_caches(&mut cache, config.block_size, config.max_pages)?;
        }
        if let Some(bits) = turboquant_bits {
            enable_turboquant_kv_caches(&mut cache, bits)?;
        }
        Ok(cache)
    }

    fn mtp_position_ids(&mut self, model: &M, start_pos: i32, len: i32) -> Result<Array> {
        if model.requires_position_ids() {
            build_position_ids(start_pos, len)
        } else {
            self.reusable_dummy_position_ids()
        }
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
            prefill_chunk_size: i32::try_from(req.prefill_chunk_size).unwrap_or(i32::MAX),
            decode_cadence_mid_chunk_cap: req.decode_cadence_mid_chunk_cap,
            kv_cache_turboquant_bits: req.kv_cache_turboquant_bits,
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
        // 3c-3: evict allowed in all phases. Slot is cleared; compact cache
        // rows are reconciled lazily on the next decode step or mid-admit
        // finalize.
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
            self.mtp_state = None;
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

    pub(crate) fn mtp_batch_active_greedy_eligible(&self) -> bool {
        let mut saw_active = false;
        for state in self.slots.iter().filter_map(|slot| slot.as_ref()) {
            saw_active = true;
            if state.finished
                || !state.generated_tokens.is_empty()
                || !state.sampler.is_pipelinable()
            {
                return false;
            }
        }
        saw_active
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

    fn prefill_cache_cap(&self) -> i32 {
        // B1-p2.3f: dynamic cap = max(prompt_len + max_new_tokens) over
        // admitted slots, bounded by effective_cap_max (defense-in-depth;
        // admit gate already rejects oversize).
        //
        // Logical cap is then floored at MIN_KV_CACHE_CAP_FOR_GPU_PERF to
        // avoid the MLX Metal kernel slow-path cliff for tight K/V buffer
        // widths. The floor is a physical-buffer concern; admit-gate
        // semantics still use the user-requested cap.
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

    fn compact_prng_state_for_rows(&self, rows: &[usize]) -> Result<Array> {
        let host: Vec<u32> = self.prng_state.to_vec()?;
        let mut compact = Vec::with_capacity(rows.len() * 2);
        for &row in rows {
            anyhow::ensure!(
                row < self.b_max,
                "compact_prng_state_for_rows: row={row} >= b_max={}",
                self.b_max
            );
            let start = row * 2;
            compact.extend_from_slice(&host[start..start + 2]);
        }
        (&compact[..], &[rows.len() as i32, 2_i32][..])
            .try_into()
            .map_err(|e| anyhow!("compact_prng_state_for_rows: Array build failed: {e:?}"))
    }

    fn scatter_prng_state_from_rows(&mut self, rows: &[usize], compact: &Array) -> Result<()> {
        let compact_host: Vec<u32> = compact.to_vec()?;
        anyhow::ensure!(
            compact_host.len() == rows.len() * 2,
            "scatter_prng_state_from_rows: compact len {} != rows*2 {}",
            compact_host.len(),
            rows.len() * 2
        );
        let mut host: Vec<u32> = self.prng_state.to_vec()?;
        for (compact_row, &slot_row) in rows.iter().enumerate() {
            anyhow::ensure!(
                slot_row < self.b_max,
                "scatter_prng_state_from_rows: slot_row={slot_row} >= b_max={}",
                self.b_max
            );
            let dst = slot_row * 2;
            let src = compact_row * 2;
            host[dst] = compact_host[src];
            host[dst + 1] = compact_host[src + 1];
        }
        self.prng_state = (&host[..], &[self.b_max as i32, 2_i32][..])
            .try_into()
            .map_err(|e| anyhow!("scatter_prng_state_from_rows: Array rebuild failed: {e:?}"))?;
        Ok(())
    }

    fn rebuild_cache_layout(&mut self, model: &M, target_rows: &[usize]) -> Result<()> {
        if self.cache_rows == target_rows {
            return Ok(());
        }
        if target_rows.is_empty() {
            self.cache = None;
            self.cache_rows.clear();
            return Ok(());
        }

        let old_cache = self
            .cache
            .take()
            .ok_or_else(|| anyhow!("rebuild_cache_layout: cache absent"))?;
        let old_rows = std::mem::take(&mut self.cache_rows);
        let (cap, dtype) = cache_cap_and_dtype(&old_cache)?;
        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(target_rows)?;
        let mut new_cache =
            self.make_model_cache(model, target_rows.len() as i32, cap, dtype, turboquant_bits)?;

        for (dst_row, &slot_row) in target_rows.iter().enumerate() {
            let src_row = old_rows
                .iter()
                .position(|&row| row == slot_row)
                .ok_or_else(|| {
                    anyhow!(
                        "rebuild_cache_layout: target slot row {slot_row} missing from old layout {:?}",
                        old_rows
                    )
                })?;
            adopt_cache_row_layers(
                &mut new_cache,
                &old_cache,
                dst_row,
                src_row,
                "rebuild_cache_layout",
            )?;
        }

        self.cache = Some(new_cache);
        self.cache_rows = target_rows.to_vec();
        Ok(())
    }

    fn install_cache_with_temp_row(
        &mut self,
        model: &M,
        temp_cache: &[LayerCache],
        temp_slot_row: usize,
    ) -> Result<()> {
        let old_cache = self
            .cache
            .take()
            .ok_or_else(|| anyhow!("install_cache_with_temp_row: main cache absent"))?;
        let old_rows = std::mem::take(&mut self.cache_rows);
        let (old_cap, dtype) = cache_cap_and_dtype(&old_cache)?;
        let (temp_cap, _) = cache_cap_and_dtype(temp_cache)?;
        let cap = old_cap.max(temp_cap);

        let mut target_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| {
                matches!(slot, Some(state) if !state.finished && !state.generated_tokens.is_empty())
                    .then_some(row)
            })
            .collect();
        if !target_rows.contains(&temp_slot_row) {
            target_rows.push(temp_slot_row);
        }
        target_rows.sort_unstable();

        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(&target_rows)?;
        let mut new_cache =
            self.make_model_cache(model, target_rows.len() as i32, cap, dtype, turboquant_bits)?;
        for (dst_row, &slot_row) in target_rows.iter().enumerate() {
            if slot_row == temp_slot_row {
                adopt_cache_row_layers(
                    &mut new_cache,
                    temp_cache,
                    dst_row,
                    0,
                    "install_cache_with_temp_row",
                )?;
            } else {
                let src_row = old_rows
                    .iter()
                    .position(|&row| row == slot_row)
                    .ok_or_else(|| {
                        anyhow!(
                            "install_cache_with_temp_row: slot row {slot_row} missing from old layout {:?}",
                            old_rows
                        )
                    })?;
                adopt_cache_row_layers(
                    &mut new_cache,
                    &old_cache,
                    dst_row,
                    src_row,
                    "install_cache_with_temp_row",
                )?;
            }
        }

        self.cache = Some(new_cache);
        self.cache_rows = target_rows;
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
        self.cache_rows.clear();
        self.mtp_state = None;
        self.phase = Phase::Idle;
        self.poisoned = false;
        Ok(())
    }

    pub fn mtp_stats(&self) -> Option<MtpSpeculativeStats> {
        self.mtp_state.as_ref().map(|state| state.stats)
    }

    pub fn prefill_admitted_mtp_single(
        &mut self,
        model: &M,
        mtp: &M::MtpHead,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel + DenseVlMethods,
    {
        self.ensure_not_poisoned()?;
        match self.prefill_admitted_mtp_single_inner(model, mtp, cfg) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                if !matches!(self.phase, Phase::Decoding | Phase::Finished) {
                    self.phase = Phase::Finished;
                }
                Err(e)
            }
        }
    }

    pub fn step_mtp_single(&mut self, model: &M, mtp: &M::MtpHead) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel,
    {
        self.ensure_not_poisoned()?;
        match self.step_mtp_single_inner(model, mtp) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn prefill_admitted_mtp_single_inner(
        &mut self,
        model: &M,
        mtp: &M::MtpHead,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel + DenseVlMethods,
    {
        if self.b_max != 1 {
            return Err(anyhow!(
                "prefill_admitted_mtp_single currently requires b_max 1"
            ));
        }
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "prefill_admitted_mtp_single illegal in {:?} phase: call evict_all first",
                    self.phase
                ));
            }
        }

        let active_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| slot.as_ref().map(|_| row))
            .collect();
        if active_rows.len() != 1 {
            return Err(anyhow!(
                "prefill_admitted_mtp_single requires exactly one admitted request, got {}",
                active_rows.len()
            ));
        }
        let row_idx = active_rows[0];
        let (
            id,
            prompt_ids,
            max_new_tokens,
            sampler,
            stop_token_ids,
            prefill_chunk_size,
            pixel_values,
            image_grid_thw,
            image_spatial_merge_size,
            image_token_id,
        ) = {
            let state = self.slots[row_idx]
                .as_ref()
                .expect("active row implies slot is Some");
            (
                state.id,
                state.prompt_ids.clone(),
                state.max_new_tokens,
                state.sampler,
                state.stop_token_ids.clone(),
                state.prefill_chunk_size,
                state.pixel_values.clone(),
                state.image_grid_thw.clone(),
                state.image_spatial_merge_size,
                state.image_token_id,
            )
        };
        if prompt_ids.is_empty() {
            return Err(anyhow!(
                "prefill_admitted_mtp_single: prompt_ids cannot be empty"
            ));
        }
        if pixel_values.is_none() && image_grid_thw.is_some() {
            return Err(anyhow!(
                "prefill_admitted_mtp_single: image_grid_thw present but pixel_values is None"
            ));
        }
        MtpSpeculativeConfig::new(cfg.max_draft_tokens, sampler)?;
        let is_vl = pixel_values.is_some();
        let prompt_ids_i32: Vec<i32> = if is_vl {
            prompt_ids.iter().map(|&t| t as i32).collect()
        } else {
            Vec::new()
        };
        let prefix_fingerprint = if self.paged_prefix_cache.is_some() {
            paged_prefix_fingerprint_for_request(
                pixel_values.as_deref(),
                image_grid_thw.as_deref(),
                image_token_id,
                image_spatial_merge_size,
            )?
        } else {
            None
        };
        let vl_position_ids_full = if is_vl {
            if model.requires_position_ids() {
                let grids = image_grid_thw.as_deref().ok_or_else(|| {
                    anyhow!(
                        "prefill_admitted_mtp_single: pixel_values present but grid_thw is None"
                    )
                })?;
                Some(build_position_ids_vl(
                    &prompt_ids_i32,
                    grids,
                    image_token_id,
                    image_spatial_merge_size,
                )?)
            } else {
                Some(self.reusable_dummy_position_ids()?)
            }
        } else {
            None
        };
        let vl_position_ids_is_dummy = is_vl && !model.requires_position_ids();
        let mut vl_vision_embeds_full: Option<Array> = None;

        let prompt_len = prompt_ids.len();
        let cap =
            (prompt_len.saturating_add(max_new_tokens) as i32).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        if self.cache.is_some() {
            return Err(anyhow!(
                "prefill_admitted_mtp_single: cache already allocated before prefill"
            ));
        }
        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(&[row_idx])?;
        self.cache = Some(self.make_model_cache(model, 1, cap, dtype, turboquant_bits)?);
        self.cache_rows = vec![row_idx];

        let mut mtp_cache = model.make_mtp_cache(mtp, 1, cap, dtype)?;
        let prompt_len_i32 = prompt_len as i32;
        let mut stats = MtpSpeculativeStats::default();
        let mut last_prompt_hidden = None;
        let mut mtp_prev_hidden: Option<Array> = None;
        let mut pos = 0_i32;
        if let Some((restore_len, restored_last_hidden)) =
            try_restore_paged_prefix_for_prompt_with_mtp(
                self.paged_prefix_cache.as_ref(),
                self.prefix_lru_cache.as_ref(),
                self.cache
                    .as_mut()
                    .ok_or_else(|| {
                        anyhow!("prefill_admitted_mtp_single: cache absent after allocate")
                    })?
                    .as_mut_slice(),
                &mut mtp_cache,
                &prompt_ids,
                model.mtp_hidden_size(mtp),
                model.mtp_hidden_dtype(mtp),
                prefix_fingerprint.as_deref(),
            )?
        {
            pos = restore_len;
            mtp_prev_hidden = Some(restored_last_hidden);
        }
        let mut image_pad_cursor = if is_vl {
            count_image_pad_i32(&prompt_ids_i32[..pos as usize], image_token_id)
        } else {
            0
        };
        while pos < prompt_len_i32 {
            let remaining = prompt_len_i32 - pos;
            let mut n = if prefill_chunk_size == 0 {
                remaining
            } else {
                remaining.min(prefill_chunk_size.max(1))
            };
            if self.paged_prefix_cache.is_some()
                && pos + n == prompt_len_i32
                && pos < prompt_len_i32 - 1
            {
                n = prompt_len_i32 - 1 - pos;
            }
            let chunk_ids = &prompt_ids[pos as usize..(pos as usize + n as usize)];
            let (hidden, chunk_pos_ids) = if is_vl {
                let chunk_ids_i32 = &prompt_ids_i32[pos as usize..(pos as usize + n as usize)];
                let chunk_arr: Array = (chunk_ids_i32, &[1_i32, n][..]).try_into()?;
                let position_ids_full = vl_position_ids_full.as_ref().ok_or_else(|| {
                    anyhow!("prefill_admitted_mtp_single: VL position ids absent")
                })?;
                let chunk_pos_ids = if vl_position_ids_is_dummy {
                    position_ids_full.clone()
                } else {
                    slice_pos_ids_axis2(position_ids_full, pos, pos + n)?
                };
                let chunk_image_pads = count_image_pad_i32(chunk_ids_i32, image_token_id);
                if chunk_image_pads > 0 && vl_vision_embeds_full.is_none() {
                    let pv = pixel_values.as_deref().ok_or_else(|| {
                        anyhow!("prefill_admitted_mtp_single: image pads without pixel_values")
                    })?;
                    let grids = image_grid_thw.as_deref().ok_or_else(|| {
                        anyhow!("prefill_admitted_mtp_single: image pads without grid_thw")
                    })?;
                    vl_vision_embeds_full = Some(model.compute_vision_embeds(
                        pv,
                        grids,
                        mlx::StreamOrDevice::default(),
                    )?);
                }
                let chunk_vision_embeds = slice_vision_embeds_for_image_pads(
                    vl_vision_embeds_full.as_ref(),
                    image_pad_cursor,
                    chunk_image_pads,
                )?;
                image_pad_cursor += chunk_image_pads;
                let hidden = {
                    let cache = self
                        .cache
                        .as_mut()
                        .ok_or_else(|| anyhow!("prefill_admitted_mtp_single: cache absent"))?;
                    model.forward_vl_hidden(
                        &chunk_arr,
                        &chunk_pos_ids,
                        None,
                        None,
                        Some(cache.as_mut_slice()),
                        chunk_vision_embeds.as_ref(),
                        image_token_id,
                        mlx::StreamOrDevice::default(),
                    )?
                };
                (hidden, chunk_pos_ids)
            } else {
                let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
                let chunk_pos_ids = self.mtp_position_ids(model, pos, n)?;
                let hidden = {
                    let cache = self
                        .cache
                        .as_mut()
                        .ok_or_else(|| anyhow!("prefill_admitted_mtp_single: cache absent"))?;
                    model.forward_text_hidden(
                        &chunk_arr,
                        &chunk_pos_ids,
                        None,
                        None,
                        Some(cache.as_mut_slice()),
                        mlx::StreamOrDevice::default(),
                    )?
                };
                (hidden, chunk_pos_ids)
            };
            let prev_hidden = match mtp_prev_hidden.as_ref() {
                Some(hidden) => hidden.clone(),
                None => zero_hidden_like_position(&hidden)?,
            };
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_prefix(
                model,
                mtp,
                &mut mtp_cache,
                &prev_hidden,
                chunk_ids,
                &hidden,
                &chunk_pos_ids,
                mlx::StreamOrDevice::default(),
            )?;
            add_elapsed_us(&mut stats.mtp_cache_commit_us, commit_start);
            let chunk_last_hidden = slice_hidden_position(&hidden, n - 1)?;
            mtp_prev_hidden = Some(chunk_last_hidden.clone());
            let new_pos = pos + n;
            if let Some(cache) = self.cache.as_ref() {
                match try_save_paged_prefix_for_prompt_with_mtp(
                    self.paged_prefix_cache.as_ref(),
                    self.prefix_lru_cache.as_ref(),
                    cache,
                    &mtp_cache,
                    &chunk_last_hidden,
                    &prompt_ids[..new_pos as usize],
                    prefix_fingerprint.as_deref(),
                ) {
                    Ok(Some(key)) => {
                        tracing::debug!("paged SSD prefix cache MTP saved: key={key}");
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!("paged SSD prefix cache MTP save skipped: {err:#}");
                    }
                }
            }
            if new_pos == prompt_len_i32 {
                last_prompt_hidden = Some(chunk_last_hidden);
            }
            pos = new_pos;
        }
        let last_prompt_hidden = last_prompt_hidden
            .ok_or_else(|| anyhow!("prefill_admitted_mtp_single produced no prompt hidden"))?;

        let projection_start = Instant::now();
        let first_logits =
            model.project_hidden_on(&last_prompt_hidden, mlx::StreamOrDevice::default())?;
        add_elapsed_us(&mut stats.projection_us, projection_start);
        let mut compact_prng = self.compact_prng_state_for_rows(&[row_idx])?;
        let sampling_start = Instant::now();
        let first_tokens =
            sample_logits_positions(&first_logits, sampler, &prompt_ids, &mut compact_prng)?;
        add_elapsed_us(&mut stats.sampling_us, sampling_start);
        self.scatter_prng_state_from_rows(&[row_idx], &compact_prng)?;
        let first_token = *first_tokens
            .first()
            .ok_or_else(|| anyhow!("prefill_admitted_mtp_single produced no first token"))?;

        let finish_reason = {
            let state = self.slots[row_idx]
                .as_mut()
                .expect("active row implies slot is Some");
            state.generated_tokens.push(first_token);
            state.real_len += 1;
            if stop_token_ids.contains(&first_token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }
            state.finish_reason
        };

        self.phase = if finish_reason.is_some() {
            Phase::Finished
        } else {
            Phase::Decoding
        };
        self.mtp_state = Some(SchedulerMtpState {
            cfg,
            rows: HashMap::from([(
                row_idx,
                SchedulerMtpRowState {
                    mtp_cache,
                    pending_tokens: VecDeque::new(),
                    last_hidden: last_prompt_hidden,
                    adaptive_draft_tokens: cfg.max_draft_tokens,
                },
            )]),
            stats,
        });

        if finish_reason.is_none() {
            self.fill_mtp_window_single(row_idx, model, mtp)?;
        }

        Ok(vec![StepEvent {
            id,
            token: first_token,
            finish_reason,
        }])
    }

    pub fn prefill_admitted_mtp_batch(
        &mut self,
        model: &M,
        mtp: &M::MtpHead,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel + DenseVlMethods,
    {
        self.ensure_not_poisoned()?;
        match self.prefill_admitted_mtp_batch_inner(model, mtp, cfg) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                if !matches!(self.phase, Phase::Decoding | Phase::Finished) {
                    self.phase = Phase::Finished;
                }
                Err(e)
            }
        }
    }

    fn prefill_admitted_mtp_batch_inner(
        &mut self,
        model: &M,
        mtp: &M::MtpHead,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel + DenseVlMethods,
    {
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "prefill_admitted_mtp_batch illegal in {:?} phase: call evict_all first",
                    self.phase
                ));
            }
        }
        if self.cache.is_some() {
            return Err(anyhow!(
                "prefill_admitted_mtp_batch: cache already allocated before prefill"
            ));
        }

        let active_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| slot.as_ref().map(|_| row))
            .collect();
        if active_rows.is_empty() {
            return Err(anyhow!(
                "prefill_admitted_mtp_batch requires at least one admitted request"
            ));
        }
        for &row_idx in &active_rows {
            let state = self.slots[row_idx]
                .as_ref()
                .expect("active row implies slot is Some");
            if state.finished || !state.generated_tokens.is_empty() {
                return Err(anyhow!(
                    "prefill_admitted_mtp_batch requires fresh admitted rows"
                ));
            }
            if state.prompt_ids.is_empty() {
                return Err(anyhow!(
                    "prefill_admitted_mtp_batch: prompt_ids cannot be empty"
                ));
            }
            MtpSpeculativeConfig::new(cfg.max_draft_tokens, state.sampler)?;
        }

        let mut temp_rows = Vec::with_capacity(active_rows.len());
        let mut row_states = HashMap::with_capacity(active_rows.len());
        let mut stats = MtpSpeculativeStats::default();
        let mut events = Vec::with_capacity(active_rows.len());
        let mut final_cap = MIN_KV_CACHE_CAP_FOR_GPU_PERF;
        let dtype = model.cache_dtype();

        for &row_idx in &active_rows {
            let mut temp = self.temp_mtp_scheduler_for_row(row_idx)?;
            let event = temp
                .prefill_admitted_mtp_single(model, mtp, cfg)
                .map_err(|err| anyhow!("prefill_admitted_mtp_batch row {row_idx}: {err:#}"))?;
            let temp_cache = temp.cache.take().ok_or_else(|| {
                anyhow!("prefill_admitted_mtp_batch row {row_idx}: temp cache absent")
            })?;
            let (cap, _) = cache_cap_and_dtype(&temp_cache)?;
            final_cap = final_cap.max(cap);
            let mut temp_mtp_state = temp.mtp_state.take().ok_or_else(|| {
                anyhow!("prefill_admitted_mtp_batch row {row_idx}: temp MTP state absent")
            })?;
            add_mtp_stats(&mut stats, temp_mtp_state.stats);
            let row_state = temp_mtp_state.rows.remove(&0).ok_or_else(|| {
                anyhow!("prefill_admitted_mtp_batch row {row_idx}: temp row state absent")
            })?;
            row_states.insert(row_idx, row_state);
            let temp_slot = temp.slots[0].as_ref().ok_or_else(|| {
                anyhow!("prefill_admitted_mtp_batch row {row_idx}: temp slot absent")
            })?;
            let slot = self.slots[row_idx]
                .as_mut()
                .expect("active row implies slot is Some");
            slot.generated_tokens = temp_slot.generated_tokens.clone();
            slot.real_len = temp_slot.real_len;
            slot.finished = temp_slot.finished;
            slot.finish_reason = temp_slot.finish_reason;
            temp_rows.push((row_idx, temp_cache));
            events.extend(event);
        }

        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(&active_rows)?;
        let mut final_cache = self.make_model_cache(
            model,
            active_rows.len() as i32,
            final_cap,
            dtype,
            turboquant_bits,
        )?;
        for (dst_row, &slot_row) in active_rows.iter().enumerate() {
            let (_, temp_cache) = temp_rows
                .iter()
                .find(|(row, _)| *row == slot_row)
                .ok_or_else(|| {
                    anyhow!(
                        "prefill_admitted_mtp_batch: missing temp cache for slot row {slot_row}"
                    )
                })?;
            adopt_cache_row_layers(
                &mut final_cache,
                temp_cache,
                dst_row,
                0,
                "prefill_admitted_mtp_batch",
            )?;
        }

        let all_finished = active_rows.iter().all(|&row| {
            self.slots[row]
                .as_ref()
                .expect("active row implies slot is Some")
                .finished
        });
        self.cache = Some(final_cache);
        self.cache_rows = active_rows;
        self.phase = if all_finished {
            Phase::Finished
        } else {
            Phase::Decoding
        };
        self.mtp_state = Some(SchedulerMtpState {
            cfg,
            rows: row_states,
            stats,
        });

        Ok(events)
    }

    fn temp_mtp_scheduler_for_row(&self, row_idx: usize) -> Result<Scheduler<M>> {
        let state = self.slots[row_idx]
            .as_ref()
            .ok_or_else(|| anyhow!("temp_mtp_scheduler_for_row: row slot absent"))?;
        let mut temp = Scheduler::<M>::new(1, self.effective_cap_max, self.meta)
            .map_err(anyhow::Error::from)?;
        if let Some(config) = self.paged_prefix_cache.as_ref() {
            temp.enable_paged_prefix_cache(config.clone())?;
        }
        if let Some(prefix_lru_cache) = self.prefix_lru_cache.as_ref() {
            temp.share_prefix_lru_cache(Arc::clone(prefix_lru_cache));
        }
        let temp_id = temp.admit(generate_request_from_state(state)?)?;
        {
            let temp_state = temp.slots[0]
                .as_mut()
                .ok_or_else(|| anyhow!("temp_mtp_scheduler_for_row: temp slot absent"))?;
            temp_state.id = state.id;
            temp_state.generated_tokens = state.generated_tokens.clone();
            temp_state.real_len = state.real_len;
            temp_state.finished = state.finished;
            temp_state.finish_reason = state.finish_reason;
        }
        anyhow::ensure!(
            temp_id == RequestId(0),
            "temp_mtp_scheduler_for_row: unexpected temp id {}",
            temp_id.0
        );
        Ok(temp)
    }

    fn install_temp_mtp_step_result(
        &mut self,
        model: &M,
        row_idx: usize,
        mut temp: Scheduler<M>,
    ) -> Result<(SchedulerMtpRowState, MtpSpeculativeStats)> {
        let temp_cache = temp
            .cache
            .take()
            .ok_or_else(|| anyhow!("install_temp_mtp_step_result: temp cache absent"))?;
        self.install_cache_with_temp_row(model, &temp_cache, row_idx)?;
        let temp_slot = temp.slots[0]
            .as_ref()
            .ok_or_else(|| anyhow!("install_temp_mtp_step_result: temp slot absent"))?;
        let slot = self.slots[row_idx]
            .as_mut()
            .ok_or_else(|| anyhow!("install_temp_mtp_step_result: row slot absent"))?;
        slot.generated_tokens = temp_slot.generated_tokens.clone();
        slot.real_len = temp_slot.real_len;
        slot.finished = temp_slot.finished;
        slot.finish_reason = temp_slot.finish_reason;

        let mut mtp_state = temp
            .mtp_state
            .take()
            .ok_or_else(|| anyhow!("install_temp_mtp_step_result: temp MTP state absent"))?;
        let row_state = mtp_state
            .rows
            .remove(&0)
            .ok_or_else(|| anyhow!("install_temp_mtp_step_result: temp row state absent"))?;
        Ok((row_state, mtp_state.stats))
    }

    fn build_temp_mtp_step_scheduler(
        &mut self,
        model: &M,
        row_idx: usize,
        cfg: MtpSpeculativeConfig,
        stats: MtpSpeculativeStats,
        row_state: SchedulerMtpRowState,
    ) -> Result<Scheduler<M>> {
        let mut temp = self.temp_mtp_scheduler_for_row(row_idx)?;
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| anyhow!("build_temp_mtp_step_scheduler: main cache absent"))?;
        let compact_row = self
            .cache_rows
            .iter()
            .position(|&row| row == row_idx)
            .ok_or_else(|| {
                anyhow!(
                    "build_temp_mtp_step_scheduler: slot row {row_idx} missing from layout {:?}",
                    self.cache_rows
                )
            })?;
        let (cap, dtype) = cache_cap_and_dtype(cache)?;
        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(&[row_idx])?;
        let mut temp_cache = self.make_model_cache(model, 1, cap, dtype, turboquant_bits)?;
        adopt_cache_row_layers(
            &mut temp_cache,
            cache,
            0,
            compact_row,
            "build_temp_mtp_step_scheduler",
        )?;
        temp.cache = Some(temp_cache);
        temp.cache_rows = vec![0];
        temp.phase = Phase::Decoding;
        temp.mtp_state = Some(SchedulerMtpState {
            cfg,
            rows: HashMap::from([(0, row_state)]),
            stats,
        });
        Ok(temp)
    }

    pub fn step_mtp_batch(&mut self, model: &M, mtp: &M::MtpHead) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel,
    {
        self.ensure_not_poisoned()?;
        match self.step_mtp_batch_inner(model, mtp) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn step_mtp_batch_inner(&mut self, model: &M, mtp: &M::MtpHead) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel,
    {
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "step_mtp_batch illegal in {:?} phase: call prefill_admitted_mtp_batch first",
                self.phase
            ));
        }
        let active_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| matches!(slot, Some(state) if !state.finished).then_some(row))
            .collect();
        if active_rows.is_empty() {
            self.phase = Phase::Finished;
            return Ok(Vec::new());
        }

        let mut mtp_state = self
            .mtp_state
            .take()
            .ok_or_else(|| anyhow!("step_mtp_batch: MTP state absent"))?;
        let cfg = mtp_state.cfg;
        let mut events = Vec::with_capacity(active_rows.len());

        for row_idx in active_rows {
            let row_state = mtp_state
                .rows
                .remove(&row_idx)
                .ok_or_else(|| anyhow!("step_mtp_batch: row {row_idx} MTP state absent"))?;
            let mut temp = self.build_temp_mtp_step_scheduler(
                model,
                row_idx,
                cfg,
                mtp_state.stats,
                row_state,
            )?;
            let row_events = temp
                .step_mtp_single(model, mtp)
                .map_err(|err| anyhow!("step_mtp_batch row {row_idx}: {err:#}"))?;
            let (row_state, stats) = self.install_temp_mtp_step_result(model, row_idx, temp)?;
            mtp_state.stats = stats;
            mtp_state.rows.insert(row_idx, row_state);
            events.extend(row_events);
        }

        let any_unfinished = self
            .slots
            .iter()
            .any(|slot| matches!(slot, Some(state) if !state.finished));
        self.phase = if any_unfinished {
            Phase::Decoding
        } else {
            Phase::Finished
        };
        self.mtp_state = Some(mtp_state);

        Ok(events)
    }

    fn step_mtp_single_inner(&mut self, model: &M, mtp: &M::MtpHead) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel,
    {
        if self.b_max != 1 {
            return Err(anyhow!("step_mtp_single currently requires b_max 1"));
        }
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "step_mtp_single illegal in {:?} phase: call prefill_admitted_mtp_single first",
                self.phase
            ));
        }
        let row_idx = self
            .slots
            .iter()
            .position(|slot| matches!(slot, Some(state) if !state.finished))
            .ok_or_else(|| anyhow!("step_mtp_single: no unfinished row"))?;

        if self
            .mtp_state
            .as_ref()
            .ok_or_else(|| anyhow!("step_mtp_single: MTP state absent"))?
            .rows
            .get(&row_idx)
            .ok_or_else(|| anyhow!("step_mtp_single: row MTP state absent"))?
            .pending_tokens
            .is_empty()
        {
            self.fill_mtp_window_single(row_idx, model, mtp)?;
        }

        let token = {
            let mtp_state = self
                .mtp_state
                .as_mut()
                .ok_or_else(|| anyhow!("step_mtp_single: MTP state absent"))?;
            mtp_state
                .rows
                .get_mut(&row_idx)
                .ok_or_else(|| anyhow!("step_mtp_single: row MTP state absent"))?
                .pending_tokens
                .pop_front()
                .ok_or_else(|| anyhow!("step_mtp_single: pending token queue is empty"))?
        };

        let (id, finish_reason) = {
            let state = self.slots[row_idx]
                .as_mut()
                .expect("unfinished row implies slot is Some");
            state.generated_tokens.push(token);
            state.real_len += 1;
            if state.stop_token_ids.contains(&token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }
            (state.id, state.finish_reason)
        };

        if finish_reason.is_some() {
            self.phase = Phase::Finished;
        } else if self
            .mtp_state
            .as_ref()
            .and_then(|state| state.rows.get(&row_idx))
            .is_some_and(|state| state.pending_tokens.is_empty())
        {
            self.fill_mtp_window_single(row_idx, model, mtp)?;
        }

        Ok(vec![StepEvent {
            id,
            token,
            finish_reason,
        }])
    }

    fn fill_mtp_window_single(&mut self, row_idx: usize, model: &M, mtp: &M::MtpHead) -> Result<()>
    where
        M: MtpSpeculativeModel,
    {
        let mut mtp_state = self
            .mtp_state
            .take()
            .ok_or_else(|| anyhow!("fill_mtp_window_single: MTP state absent"))?;
        let mut row_state = mtp_state
            .rows
            .remove(&row_idx)
            .ok_or_else(|| anyhow!("fill_mtp_window_single: row MTP state absent"))?;
        let result = self.fill_mtp_window_single_with_state(
            row_idx,
            mtp_state.cfg,
            &mut mtp_state.stats,
            &mut row_state,
            model,
            mtp,
        );
        mtp_state.rows.insert(row_idx, row_state);
        self.mtp_state = Some(mtp_state);
        result
    }

    fn fill_mtp_window_single_with_state(
        &mut self,
        row_idx: usize,
        cfg: MtpSpeculativeConfig,
        stats: &mut MtpSpeculativeStats,
        row_state: &mut SchedulerMtpRowState,
        model: &M,
        mtp: &M::MtpHead,
    ) -> Result<()>
    where
        M: MtpSpeculativeModel,
    {
        let (prompt_ids, generated_tokens, max_new_tokens, sampler, stop_token_ids) = {
            let state = self.slots[row_idx]
                .as_ref()
                .ok_or_else(|| anyhow!("fill_mtp_window_single: row slot absent"))?;
            (
                state.prompt_ids.clone(),
                state.generated_tokens.clone(),
                state.max_new_tokens,
                state.sampler,
                state.stop_token_ids.clone(),
            )
        };
        let emitted = generated_tokens.len();
        let remaining = max_new_tokens.saturating_sub(emitted);
        if remaining == 0 {
            return Ok(());
        }
        let current_token = *generated_tokens
            .last()
            .ok_or_else(|| anyhow!("fill_mtp_window_single: no current token"))?;
        let mut history = Vec::with_capacity(prompt_ids.len() + generated_tokens.len());
        history.extend_from_slice(&prompt_ids);
        history.extend_from_slice(&generated_tokens);

        let draft_budget = row_state
            .adaptive_draft_tokens
            .clamp(1, cfg.max_draft_tokens)
            .min(remaining);
        let mut compact_prng = self.compact_prng_state_for_rows(&[row_idx])?;
        let draft_result = self.draft_mtp_tokens_single(
            stats,
            row_state,
            model,
            mtp,
            current_token,
            draft_budget,
            &history,
            sampler,
            &mut compact_prng,
        )?;
        let draft_tokens = draft_result.tokens;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (prompt_ids.len() + generated_tokens.len() - 1) as i32;
        let verify_pos_ids =
            self.mtp_position_ids(model, verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;
        let pre_window_hidden = row_state.last_hidden.clone();

        let base_snapshot = {
            let cache = self
                .cache
                .as_ref()
                .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
            cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>()
        };
        let verify_forward_start = Instant::now();
        let verified_hidden = {
            let cache = self
                .cache
                .as_mut()
                .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
            model.forward_text_hidden(
                &verify_arr,
                &verify_pos_ids,
                None,
                None,
                Some(cache.as_mut_slice()),
                mlx::StreamOrDevice::default(),
            )?
        };
        add_elapsed_us(&mut stats.verify_forward_us, verify_forward_start);
        let projection_start = Instant::now();
        let verified_logits =
            model.project_hidden_on(&verified_hidden, mlx::StreamOrDevice::default())?;
        add_elapsed_us(&mut stats.projection_us, projection_start);
        let sampling_start = Instant::now();
        let verified_tokens =
            sample_logits_positions(&verified_logits, sampler, &history, &mut compact_prng)?;
        add_elapsed_us(&mut stats.sampling_us, sampling_start);
        self.scatter_prng_state_from_rows(&[row_idx], &compact_prng)?;

        let resolution = resolve_speculative_tokens(&draft_tokens, &verified_tokens)?;
        stats.windows += 1;
        stats.drafted_tokens += draft_tokens.len();
        stats.accepted_draft_tokens += resolution.accepted_draft_len;
        if resolution.needs_rollback {
            stats.rollback_count += 1;
        }
        adjust_mtp_draft_budget(
            cfg.max_draft_tokens,
            &mut row_state.adaptive_draft_tokens,
            draft_tokens.len(),
            resolution.accepted_draft_len,
            stats,
        );

        let (accepted_input, accepted_hidden, accepted_position_ids, accepted_last_hidden) =
            if resolution.needs_rollback {
                let rollback_start = Instant::now();
                {
                    let cache = self
                        .cache
                        .as_mut()
                        .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
                    restore_layer_cache(cache.as_mut_slice(), &base_snapshot)?;
                }
                add_elapsed_us(&mut stats.main_rollback_us, rollback_start);
                let replay_len = resolution.accepted_verify_input_len;
                let replay_input = &verify_input[..replay_len];
                let replay_arr: Array =
                    (replay_input, &[1_i32, replay_len as i32][..]).try_into()?;
                let replay_pos_ids =
                    self.mtp_position_ids(model, verify_start_pos, replay_len as i32)?;
                let replay_forward_start = Instant::now();
                let replay_hidden = {
                    let cache = self
                        .cache
                        .as_mut()
                        .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
                    model.forward_text_hidden(
                        &replay_arr,
                        &replay_pos_ids,
                        None,
                        None,
                        Some(cache.as_mut_slice()),
                        mlx::StreamOrDevice::default(),
                    )?
                };
                add_elapsed_us(&mut stats.verify_forward_us, replay_forward_start);
                let last_hidden = slice_hidden_position(&replay_hidden, replay_len as i32 - 1)?;
                (
                    replay_input.to_vec(),
                    replay_hidden,
                    replay_pos_ids,
                    last_hidden,
                )
            } else {
                (
                    verify_input[..resolution.accepted_verify_input_len].to_vec(),
                    verified_hidden.clone(),
                    verify_pos_ids.clone(),
                    slice_hidden_position(
                        &verified_hidden,
                        resolution.accepted_verify_input_len as i32 - 1,
                    )?,
                )
            };

        if resolution.needs_rollback {
            let restore_start = Instant::now();
            row_state.mtp_cache.restore(&draft_result.cache_snapshot)?;
            add_elapsed_us(&mut stats.mtp_cache_restore_us, restore_start);
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_prefix(
                model,
                mtp,
                &mut row_state.mtp_cache,
                &pre_window_hidden,
                &accepted_input,
                &accepted_hidden,
                &accepted_position_ids,
                mlx::StreamOrDevice::default(),
            )?;
            add_elapsed_us(&mut stats.mtp_cache_commit_us, commit_start);
        } else {
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_tail(
                model,
                mtp,
                &mut row_state.mtp_cache,
                &pre_window_hidden,
                &accepted_input,
                &accepted_hidden,
                &accepted_position_ids,
                mlx::StreamOrDevice::default(),
            )?;
            add_elapsed_us(&mut stats.mtp_cache_commit_us, commit_start);
            stats.mtp_cache_reuse_count = stats.mtp_cache_reuse_count.saturating_add(1);
            stats.mtp_cache_reused_tokens = stats
                .mtp_cache_reused_tokens
                .saturating_add(accepted_input.len().saturating_sub(1));
        }
        row_state.last_hidden = accepted_last_hidden;

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        row_state.pending_tokens.extend(tokens_to_append);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_mtp_tokens_single(
        &mut self,
        stats: &mut MtpSpeculativeStats,
        row_state: &mut SchedulerMtpRowState,
        model: &M,
        mtp: &M::MtpHead,
        current_token: u32,
        draft_budget: usize,
        history: &[u32],
        sampler: Sampler,
        prng_state: &mut Array,
    ) -> Result<MtpDraftResult>
    where
        M: MtpSpeculativeModel,
    {
        let mtp_snapshot = row_state.mtp_cache.snapshot();
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = history.to_vec();
        let mut input_hidden = row_state.last_hidden.clone();
        let mut input_token = current_token;
        let start_pos = (history.len() - 1) as i32;

        for offset in 0..draft_budget {
            let token_arr: Array = (&[input_token][..], &[1_i32, 1_i32][..]).try_into()?;
            let position_ids = self.mtp_position_ids(model, start_pos + offset as i32, 1)?;
            let draft_forward_start = Instant::now();
            let output = model.mtp_forward_on(
                mtp,
                &input_hidden,
                &token_arr,
                &position_ids,
                None,
                Some(&mut row_state.mtp_cache),
                mlx::StreamOrDevice::default(),
            )?;
            add_elapsed_us(&mut stats.draft_forward_us, draft_forward_start);
            let sampling_start = Instant::now();
            let sampled =
                sample_logits_positions(&output.logits, sampler, &draft_history, prng_state)?;
            add_elapsed_us(&mut stats.sampling_us, sampling_start);
            let next_token = *sampled
                .first()
                .ok_or_else(|| anyhow!("draft_mtp_tokens_single: MTP draft produced no token"))?;
            draft_tokens.push(next_token);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok(MtpDraftResult {
            tokens: draft_tokens,
            cache_snapshot: mtp_snapshot,
        })
    }

    /// Run batched prefill for every currently-admitted request. Only legal
    /// in `Idle`/`Admitting` phase with `active_count() >= 1`.
    ///
    /// Allocates the batched KV cache on first call using the compact occupied
    /// row count (`B_prefill`, not `b_max`). Capacity is
    /// `min(max(prompt_len + max_new_tokens) over slots, effective_cap_max)`,
    /// bf16. Subsequent calls after `evict_all` allocate fresh — `evict_all`
    /// drops the cache (3f) so the next batch's cap is sized to its slots, not
    /// inherited from the prior batch.
    ///
    /// Builds a right-padded model-facing batch over occupied rows only. Its
    /// tensors are `[B_prefill, T_max]` input_ids, `[3, B_prefill, T_max]`
    /// position_ids, `[B_prefill, 1, T_max, T_max]` attention mask, and
    /// `[B_prefill, T_max]` linear mask, then calls `M::batched_prefill` via
    /// the `Model` trait. The resulting cache remains compact; decode uses the
    /// same scheduler-row mapping rather than padding up to `b_max`.
    ///
    /// After prefill, samples the first token via a three-stage dispatch:
    /// Stage A collects per-row `sampler` refs + prompt histories in compact
    /// prefill order. Stage B reshapes `[B_prefill, 1, vocab]` →
    /// `[B_prefill, vocab]` and calls `sample_batch` once — coalescing
    /// all-greedy batches into a single GPU op rather than B serial kernel
    /// launches. Stage C distributes tokens back to their scheduler rows,
    /// checks EOS / `max_new_tokens`, and emits one [`StepEvent`] per occupied
    /// row. Transitions to `Decoding` (or `Finished` if every first token was
    /// EOS). See spec §4.5.
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
        let active_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| slot.as_ref().map(|_| row))
            .collect();
        if active_rows.is_empty() {
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

        let prefill_rows = active_rows;

        // Build the model-facing prefill batch in active slot order. When
        // there are empty scheduler slots, the model and cache see only
        // occupied rows.
        let prompt_lens: Vec<i32> = prefill_rows
            .iter()
            .map(|&row| {
                self.slots[row]
                    .as_ref()
                    .expect("prefill_rows contain only occupied slots")
                    .prompt_ids
                    .len() as i32
            })
            .collect();
        let max_len = prompt_lens.iter().copied().max().unwrap_or(0);
        if max_len <= 0 {
            return Err(anyhow!(
                "prefill_admitted: max prompt length is 0 — all admitted prompts are empty"
            ));
        }

        // Build [B_prefill, T_max] right-padded input_ids (pad value 0).
        let b = prefill_rows.len();
        let t = max_len as usize;
        let mut flat: Vec<i32> = vec![0; b * t];
        for (batch_row, &slot_row) in prefill_rows.iter().enumerate() {
            let state = self.slots[slot_row]
                .as_ref()
                .expect("prefill_rows contain only occupied slots");
            for (j, &tok) in state.prompt_ids.iter().enumerate() {
                flat[batch_row * t + j] = tok as i32;
            }
            // positions [state.prompt_ids.len() .. t] stay 0 (pad)
        }
        let input_ids: Array = (&flat[..], &[b as i32, max_len][..])
            .try_into()
            .map_err(|e| anyhow!("input_ids try_into Array failed: {e:?}"))?;

        // B1-p2.4: detect any VL row. Dispatch determines both position_ids
        // builder and prefill entry point.
        let any_vl = prefill_rows.iter().any(|&row| {
            self.slots[row]
                .as_ref()
                .is_some_and(|r| r.pixel_values.is_some())
        });

        // Allocate the compact cache exactly to the occupied scheduler rows.
        let cap = self.prefill_cache_cap();
        if self.cache.is_some() {
            return Err(anyhow!(
                "prefill_admitted: cache already allocated before prefill"
            ));
        }
        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(&prefill_rows)?;
        self.cache = Some(self.make_model_cache(
            model,
            b as i32,
            cap,
            model.cache_dtype(),
            turboquant_bits,
        )?);
        self.cache_rows = prefill_rows.clone();
        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(self.reusable_dummy_position_ids()?)
        };
        let paged_prefix_cache_config = self.paged_prefix_cache.clone();
        let prefix_lru_cache = self.prefix_lru_cache.clone();

        // Run prefill. Capture [B, 1, vocab] logits (sequence axis already
        // collapsed via slice_last_and_project) for first-token sampling.
        // Single-row batches use the single-stream model API so text and VL
        // requests avoid right-pad masks that have no semantic work at B=1.
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
                // Mla: matches on (p5h_trace, mpf_span) profiling tuple, not LayerCache — cache-kind-independent, correct for GLM.
                _ => None,
            };
            let prefill_cache = self
                .cache
                .as_mut()
                .ok_or_else(|| anyhow!("cache missing after allocation — internal bug"))?
                .as_mut_slice();
            let logits = if any_vl {
                // Collect per-row prompt ids, vision args, and tokenizer constants in compact cache order.
                let per_row_ids_i32: Vec<Vec<i32>> = prefill_rows
                    .iter()
                    .map(|&row| {
                        self.slots[row]
                            .as_ref()
                            .expect("prefill_rows contain only occupied slots")
                            .prompt_ids
                            .iter()
                            .map(|&t| t as i32)
                            .collect()
                    })
                    .collect();
                let per_row_ids_refs: Vec<&[i32]> =
                    per_row_ids_i32.iter().map(|v| v.as_slice()).collect();
                let per_row_grids_owned: Vec<Option<Vec<(i32, i32, i32)>>> = prefill_rows
                    .iter()
                    .map(|&row| {
                        self.slots[row]
                            .as_ref()
                            .expect("prefill_rows contain only occupied slots")
                            .image_grid_thw
                            .clone()
                    })
                    .collect();
                let per_row_grids: Vec<GridThwSlice<'_>> = per_row_grids_owned
                    .iter()
                    .map(|opt| opt.as_deref())
                    .collect();
                let per_row_pv: Vec<PixelValuesSlice<'_>> = prefill_rows
                    .iter()
                    .map(|&row| {
                        self.slots[row]
                            .as_ref()
                            .expect("prefill_rows contain only occupied slots")
                            .pixel_values
                            .as_deref()
                    })
                    .collect();

                // Tokenizer-defined constants from the first VL slot.
                let (img_token_id, merge_size) = prefill_rows
                    .iter()
                    .find_map(|&row| {
                        self.slots[row]
                            .as_ref()
                            .filter(|r| r.pixel_values.is_some())
                            .map(|r| (r.image_token_id, r.image_spatial_merge_size))
                    })
                    .expect("any_vl == true implies at least one VL slot");

                if b == 1 {
                    let grids = per_row_grids[0].ok_or_else(|| {
                        anyhow!("single-row VL prefill: pixel_values present but grid_thw is None")
                    })?;
                    let position_ids_full = if let Some(dummy) = dummy_position_ids.as_ref() {
                        dummy.clone()
                    } else if grids.is_empty() {
                        build_position_ids(0, max_len)?
                    } else {
                        build_position_ids_vl(&per_row_ids_i32[0], grids, img_token_id, merge_size)?
                    };
                    let vision_embeds_full = if grids.is_empty() {
                        None
                    } else {
                        let pv = per_row_pv[0].ok_or_else(|| {
                            anyhow!(
                                "single-row VL prefill: grid_thw present but pixel_values is None"
                            )
                        })?;
                        Some(model.compute_vision_embeds(
                            pv,
                            grids,
                            mlx::StreamOrDevice::default(),
                        )?)
                    };
                    let prompt_ids_u32: Vec<u32> =
                        per_row_ids_i32[0].iter().map(|&tok| tok as u32).collect();
                    let vl_prefix_fingerprint = if paged_prefix_cache_config.is_some() {
                        paged_prefix_fingerprint_for_request(
                            per_row_pv[0],
                            per_row_grids[0],
                            img_token_id,
                            merge_size,
                        )?
                    } else {
                        None
                    };
                    if let Some(start_pos) = try_restore_paged_prefix_for_prompt(
                        paged_prefix_cache_config.as_ref(),
                        prefix_lru_cache.as_ref(),
                        prefill_cache,
                        &prompt_ids_u32,
                        vl_prefix_fingerprint.as_deref(),
                    )? {
                        forward_single_vl_suffix(
                            model,
                            prefill_cache,
                            SingleVlSuffixInput {
                                prompt_ids: &per_row_ids_i32[0],
                                start_pos,
                                end_pos: max_len,
                                position_ids_full: &position_ids_full,
                                position_ids_is_dummy: dummy_position_ids.is_some(),
                                vision_embeds_full: vision_embeds_full.as_ref(),
                                image_token_id: img_token_id,
                            },
                        )?
                    } else if max_len > 1 {
                        let prefix_len = max_len - 1;
                        let prefix_input_ids = mlx::ops::indexing::slice_strided(
                            &input_ids,
                            &[0_i32, 0][..],
                            &[1_i32, prefix_len][..],
                            &[1_i32, 1][..],
                        )?;
                        let last_input_ids = mlx::ops::indexing::slice_strided(
                            &input_ids,
                            &[0_i32, prefix_len][..],
                            &[1_i32, max_len][..],
                            &[1_i32, 1][..],
                        )?;
                        let (prefix_position_ids, last_position_ids) =
                            if dummy_position_ids.is_some() {
                                (position_ids_full.clone(), position_ids_full.clone())
                            } else {
                                (
                                    slice_pos_ids_axis2(&position_ids_full, 0, prefix_len)?,
                                    slice_pos_ids_axis2(&position_ids_full, prefix_len, max_len)?,
                                )
                            };

                        let prompt_ids = &per_row_ids_i32[0];
                        let prefix_image_pads = prompt_ids[..prefix_len as usize]
                            .iter()
                            .filter(|&&tok| tok == img_token_id)
                            .count();
                        let last_image_pads = prompt_ids[prefix_len as usize..max_len as usize]
                            .iter()
                            .filter(|&&tok| tok == img_token_id)
                            .count();
                        let prefix_vision_embeds = match vision_embeds_full.as_ref() {
                            Some(ve) if prefix_image_pads > 0 => {
                                Some(slice_vision_embeds_rows(ve, 0, prefix_image_pads)?)
                            }
                            // Mla: matches on vision_embeds_full Option (VL-only path), not LayerCache — GLM is text-only, never reaches here.
                            _ => None,
                        };
                        let last_vision_embeds = match vision_embeds_full.as_ref() {
                            Some(ve) if last_image_pads > 0 => Some(slice_vision_embeds_rows(
                                ve,
                                prefix_image_pads,
                                prefix_image_pads + last_image_pads,
                            )?),
                            // Mla: matches on vision_embeds_full Option (VL-only path), not LayerCache — GLM is text-only, never reaches here.
                            _ => None,
                        };

                        let prefix_hidden = model.forward_vl_hidden(
                            &prefix_input_ids,
                            &prefix_position_ids,
                            None,
                            None,
                            Some(&mut *prefill_cache),
                            prefix_vision_embeds.as_ref(),
                            img_token_id,
                            mlx::StreamOrDevice::default(),
                        )?;
                        // Commit the prefix chunk's lazy KV-cache writes before the
                        // last-token chunk reads the SAME cache. Without this barrier the
                        // prefix-write → last-read chain fuses into one lazy graph whose
                        // read-after-write on the shared mutable cache buffer MLX
                        // intermittently miscomputes to all-NaN logits (→ argmax 0 = <pad>
                        // → empty output) on all-KVCache models such as Gemma4. Evaluating
                        // the prefix hidden transitively materializes the per-layer
                        // slice_update/concatenate cache writes it depends on via attention.
                        mlx::transforms::eval(&[&prefix_hidden])?;

                        match try_save_paged_prefix_for_prompt(
                            paged_prefix_cache_config.as_ref(),
                            prefix_lru_cache.as_ref(),
                            prefill_cache,
                            &prompt_ids_u32[..prefix_len as usize],
                            vl_prefix_fingerprint.as_deref(),
                        ) {
                            Ok(Some(key)) => {
                                tracing::debug!(
                                    "paged SSD prefix cache saved VL prefix: key={key}"
                                );
                            }
                            Ok(None) => {}
                            Err(err) => {
                                tracing::warn!(
                                    "paged SSD prefix cache VL prefix save skipped: {err:#}"
                                );
                            }
                        }

                        model.forward_vl_chunk(
                            &last_input_ids,
                            &last_position_ids,
                            None,
                            None,
                            Some(&mut *prefill_cache),
                            last_vision_embeds.as_ref(),
                            img_token_id,
                            mlx::StreamOrDevice::default(),
                        )?
                    } else {
                        model.forward_vl_chunk(
                            &input_ids,
                            &position_ids_full,
                            None,
                            None,
                            Some(prefill_cache),
                            vision_embeds_full.as_ref(),
                            img_token_id,
                            mlx::StreamOrDevice::default(),
                        )?
                    }
                } else if paged_prefix_cache_config.is_some() {
                    forward_batched_vl_with_paged_prefix(
                        model,
                        prefill_cache,
                        BatchedVlPrefixReplay {
                            prompt_ids: &per_row_ids_i32,
                            pixel_values: &per_row_pv,
                            grid_thw: &per_row_grids,
                            image_token_id: img_token_id,
                            image_spatial_merge_size: merge_size,
                            dummy_position_ids: dummy_position_ids.as_ref(),
                            prefix_cache_config: paged_prefix_cache_config.as_ref(),
                            prefix_lru_cache: prefix_lru_cache.as_ref(),
                        },
                    )?
                } else {
                    let attention_mask =
                        build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16)?;
                    let linear_attention_mask = build_batch_linear_mask(&prompt_lens, max_len)?;
                    let position_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                        dummy.clone()
                    } else {
                        build_position_ids_vl_batched(
                            &per_row_ids_refs,
                            &per_row_grids,
                            img_token_id,
                            merge_size,
                            max_len,
                        )?
                    };

                    model.batched_prefill_vl(
                        &input_ids,
                        &position_ids,
                        &attention_mask,
                        &linear_attention_mask,
                        &prompt_lens,
                        &per_row_pv,
                        &per_row_grids,
                        img_token_id,
                        Some(prefill_cache),
                        mlx::StreamOrDevice::default(),
                    )?
                }
            } else if b == 1 {
                let prompt_ids = &self.slots[prefill_rows[0]]
                    .as_ref()
                    .expect("prefill_rows contain only occupied slots")
                    .prompt_ids;
                if let Some(start_pos) = try_restore_paged_prefix_for_prompt(
                    paged_prefix_cache_config.as_ref(),
                    prefix_lru_cache.as_ref(),
                    prefill_cache,
                    prompt_ids,
                    None,
                )? {
                    forward_single_text_suffix(
                        model,
                        prefill_cache,
                        prompt_ids,
                        start_pos,
                        max_len,
                        dummy_position_ids.as_ref(),
                    )?
                } else if max_len > 1 {
                    let prefix_len = max_len - 1;
                    let prefix_input_ids = mlx::ops::indexing::slice_strided(
                        &input_ids,
                        &[0_i32, 0][..],
                        &[1_i32, prefix_len][..],
                        &[1_i32, 1][..],
                    )?;
                    let prefix_position_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                        dummy.clone()
                    } else {
                        build_position_ids(0, prefix_len)?
                    };
                    let prefix_hidden = model.forward_text_hidden(
                        &prefix_input_ids,
                        &prefix_position_ids,
                        None,
                        None,
                        Some(&mut *prefill_cache),
                        mlx::StreamOrDevice::default(),
                    )?;
                    // Commit the prefix chunk's lazy KV-cache writes before the last-token
                    // chunk reads the shared cache (see the VL split site above for the
                    // all-NaN read-after-write rationale; same b==1 split-prefill hazard).
                    mlx::transforms::eval(&[&prefix_hidden])?;

                    match try_save_paged_prefix_for_prompt(
                        paged_prefix_cache_config.as_ref(),
                        prefix_lru_cache.as_ref(),
                        prefill_cache,
                        &prompt_ids[..prefix_len as usize],
                        None,
                    ) {
                        Ok(Some(key)) => {
                            tracing::debug!("paged SSD prefix cache saved prefix: key={key}");
                        }
                        Ok(None) => {}
                        Err(err) => {
                            tracing::warn!("paged SSD prefix cache prefix save skipped: {err:#}");
                        }
                    }

                    let last_input_ids = mlx::ops::indexing::slice_strided(
                        &input_ids,
                        &[0_i32, prefix_len][..],
                        &[1_i32, max_len][..],
                        &[1_i32, 1][..],
                    )?;
                    let last_position_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                        dummy.clone()
                    } else {
                        build_position_ids(prefix_len, 1)?
                    };
                    model.forward_on(
                        &last_input_ids,
                        &last_position_ids,
                        None,
                        None,
                        Some(&mut *prefill_cache),
                        mlx::StreamOrDevice::default(),
                    )?
                } else {
                    let position_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                        dummy.clone()
                    } else {
                        build_position_ids(0, max_len)?
                    };
                    model.forward_on(
                        &input_ids,
                        &position_ids,
                        None,
                        None,
                        Some(prefill_cache),
                        mlx::StreamOrDevice::default(),
                    )?
                }
            } else if paged_prefix_cache_config.is_some() {
                let per_row_prompt_ids: Vec<Vec<u32>> = prefill_rows
                    .iter()
                    .map(|&row| {
                        self.slots[row]
                            .as_ref()
                            .expect("prefill_rows contain only occupied slots")
                            .prompt_ids
                            .clone()
                    })
                    .collect();
                forward_batched_text_with_paged_prefix(
                    model,
                    prefill_cache,
                    BatchedTextPrefixReplay {
                        prompt_ids: &per_row_prompt_ids,
                        dummy_position_ids: dummy_position_ids.as_ref(),
                        prefix_cache_config: paged_prefix_cache_config.as_ref(),
                        prefix_lru_cache: prefix_lru_cache.as_ref(),
                    },
                )?
            } else {
                let attention_mask =
                    build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16)?;
                let linear_attention_mask = build_batch_linear_mask(&prompt_lens, max_len)?;
                let position_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                    dummy.clone()
                } else {
                    build_position_ids_batched(&prompt_lens, max_len)?
                };
                model.batched_prefill(
                    &input_ids,
                    &position_ids,
                    &attention_mask,
                    &linear_attention_mask,
                    &prompt_lens,
                    Some(prefill_cache),
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
        if b == 1 {
            let (prompt_ids, prefix_fingerprint) = {
                let state = self.slots[prefill_rows[0]]
                    .as_ref()
                    .expect("prefill_rows contain only occupied slots");
                let prefix_fingerprint = if paged_prefix_cache_config.is_some() {
                    paged_prefix_fingerprint_for_request(
                        state.pixel_values.as_deref(),
                        state.image_grid_thw.as_deref(),
                        state.image_token_id,
                        state.image_spatial_merge_size,
                    )?
                } else {
                    None
                };
                (state.prompt_ids.clone(), prefix_fingerprint)
            };
            if let Some(cache) = self.cache.as_ref() {
                match try_save_paged_prefix_for_prompt(
                    paged_prefix_cache_config.as_ref(),
                    prefix_lru_cache.as_ref(),
                    cache,
                    &prompt_ids,
                    prefix_fingerprint.as_deref(),
                ) {
                    Ok(Some(key)) => {
                        tracing::debug!("paged SSD prefix cache saved: key={key}");
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::warn!("paged SSD prefix cache save skipped: {err:#}");
                    }
                }
            }
        }

        // After per-row prefill, row i's cache is filled up to position
        // prompt_lens[i] - 1. The first decode step must use position
        // prompt_lens[i] for that row.
        for (&slot_row, &plen) in prefill_rows.iter().zip(prompt_lens.iter()) {
            let state = self.slots[slot_row]
                .as_mut()
                .expect("prefill_rows contain only occupied slots");
            state.real_len = plen;
        }

        // Sample first token per occupied row from logits[:, 0, :].
        // batched_prefill returns [B, 1, vocab]. Reshape to [B, vocab] for
        // sample_batch, then dispatch once to coalesce all-greedy batches.
        //
        // P5h+1 T1: split the legacy `first_token_sampling` span into two
        // siblings under root so the wrapper-dominance gap is closed.
        //   * `first_token_sampling_prepare` — logits reshape + per-row
        //     sampler refs + per-row history construction (CPU-bound; no
        //     MLX materialization on its own).
        //   * `first_token_sampling_materialize_and_sample` — wraps
        //     `sample_batch(...)` which calls `.to_vec()` internally and
        //     therefore forces the full prefill graph to materialize. With
        //     the split, the materialize span's inclusive_us now reflects
        //     the real lazy-graph cost previously hidden inside one wrapper.
        // Explicit-API discipline: no `?` while a manual span is open;
        // close-on-error branches forward the original error after the
        // span has been closed.
        #[cfg(feature = "p5h-profile")]
        let mut prepare_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
            crate::core::p5h::open_p5h_span(ctx, Some(root_span), "first_token_sampling_prepare")
        });

        let logits_shape = logits.shape();
        let vocab = logits_shape.as_slice()[2];
        let logits_bv_result = logits.reshape(&[b as i32, vocab][..]).map_err(|e| {
            anyhow!("prefill_admitted: reshape logits [B,1,vocab]->[B,vocab] failed: {e:?}")
        });

        let logits_bv = match logits_bv_result {
            Ok(value) => value,
            Err(err) => {
                #[cfg(feature = "p5h-profile")]
                if let (Some((ctx, _)), Some(span)) = (p5h_trace.as_ref(), prepare_span.take()) {
                    crate::core::p5h::close_p5h_span(
                        ctx,
                        span,
                        crate::core::p5h::monotonic_ns_public(),
                        crate::core::p5h::SpanFields::default(),
                    );
                }
                return Err(err);
            }
        };

        // Stage A — collect per-row sampler refs + histories in compact prefill order.
        let mut row_samplers: Vec<&Sampler> = Vec::with_capacity(b);
        let mut row_histories: Vec<Vec<u32>> = Vec::with_capacity(b);
        for &slot_row in &prefill_rows {
            let state = self.slots[slot_row]
                .as_ref()
                .expect("prefill_rows contain only occupied slots");
            row_samplers.push(&state.sampler);
            row_histories.push(state.prompt_ids.clone());
        }

        // Close the prepare span BEFORE opening the materialize sibling so
        // both intervals stay strictly contained under root and disjoint
        // from each other (per § 2.5a interval containment).
        #[cfg(feature = "p5h-profile")]
        if let (Some((ctx, _)), Some(span)) = (p5h_trace.as_ref(), prepare_span.take()) {
            crate::core::p5h::close_p5h_span(
                ctx,
                span,
                crate::core::p5h::monotonic_ns_public(),
                crate::core::p5h::SpanFields::default(),
            );
        }

        #[cfg(feature = "p5h-profile")]
        let materialize_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
            crate::core::p5h::open_p5h_span(
                ctx,
                Some(root_span),
                "first_token_sampling_materialize_and_sample",
            )
        });

        // Stage B — dispatch sample_batch once over [B, vocab]. `sample_batch`
        // internally calls `.to_vec()` which forces the entire prefill graph
        // to materialize; this wrapper makes that cost attributable.
        let mut compact_prng = self.compact_prng_state_for_rows(&prefill_rows)?;
        let history_refs: Vec<&[u32]> = row_histories.iter().map(|h| h.as_slice()).collect();
        let sample_result = crate::core::sampler::sample_batch(
            &row_samplers,
            &logits_bv,
            &history_refs,
            &mut compact_prng,
        )
        .map_err(|e| anyhow!("prefill_admitted: sample_batch failed: {e:?}"));

        #[cfg(feature = "p5h-profile")]
        if let (Some((ctx, _)), Some(span)) = (p5h_trace.as_ref(), materialize_span) {
            crate::core::p5h::close_p5h_span(
                ctx,
                span,
                crate::core::p5h::monotonic_ns_public(),
                crate::core::p5h::SpanFields::default(),
            );
        }

        let tokens = sample_result?;
        drop(history_refs);
        drop(row_samplers);
        self.scatter_prng_state_from_rows(&prefill_rows, &compact_prng)?;

        // Stage C — distribute tokens + termination per occupied row.
        let mut events: Vec<StepEvent> = Vec::new();
        for (batch_row, &token) in tokens.iter().enumerate() {
            let slot_row = prefill_rows[batch_row];
            let state = self.slots[slot_row]
                .as_mut()
                .expect("prefill_rows contain only occupied slots");

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
    /// compact cache layout and a three-stage sample_batch dispatch rather
    /// than per-row sampler calls. Stage A collects compact active rows and
    /// builds per-row sampler refs + histories. Stage B packs
    /// `[B_active, 1]` input_ids, runs `forward_on`, reshapes
    /// `[B_active, 1, vocab]` → `[B_active, vocab]`, and calls
    /// `sample_batch` once. Stage C distributes tokens back to scheduler
    /// rows, advances `real_len`, checks EOS / `max_new_tokens`, and collects
    /// events.
    ///
    /// Finished, evicted, empty, and mid-admit-reserved rows are excluded
    /// from the model-facing decode batch. Transitions phase to `Finished`
    /// when all occupied rows are done.
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
        let active_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| {
                matches!(slot, Some(r) if !r.finished && !r.generated_tokens.is_empty())
                    .then_some(row)
            })
            .collect();
        if active_rows.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "p5h-profile")]
        crate::core::p5h::try_with_p5h_span_from_current_trace(
            "scheduler_decode_rebuild_cache_layout",
            crate::core::p5h::SpanFields::default,
            || self.rebuild_cache_layout(model, &active_rows),
        )?;
        #[cfg(not(feature = "p5h-profile"))]
        self.rebuild_cache_layout(model, &active_rows)?;
        let b = active_rows.len();

        // Build [B_active, 1] input_ids in compact cache order.
        let last_tokens: Vec<i32> = active_rows
            .iter()
            .map(|&slot_row| {
                let r = self.slots[slot_row]
                    .as_ref()
                    .expect("active row implies Some");
                *r.generated_tokens
                    .last()
                    .expect("active_rows guarantees generated_tokens is non-empty")
                    as i32
            })
            .collect();
        let input_ids: Array = (&last_tokens[..], &[b as i32, 1][..])
            .try_into()
            .map_err(|e| anyhow!("step: build input_ids Array failed: {e:?}"))?;

        // Build [3, B_active, 1] decode position ids only for models that consume
        // them. Gemma4 derives positions from KV offsets, so the hot path can
        // reuse a placeholder without changing model semantics.
        let position_ids = if model.requires_position_ids() {
            let per_row_pos: Vec<i32> = active_rows
                .iter()
                .map(|&slot_row| {
                    self.slots[slot_row]
                        .as_ref()
                        .expect("active row implies Some")
                        .real_len
                })
                .collect();
            build_decode_position_ids(&per_row_pos)?
        } else {
            self.reusable_dummy_position_ids()?
        };

        // Per-row lens for compact decode: every model-facing row writes
        // exactly one token.
        let per_row_lens: Vec<i32> = vec![1; b];

        let cache_ref = self
            .cache
            .as_mut()
            .ok_or_else(|| anyhow!("step: cache absent — was prefill_admitted called?"))?;

        // Build per-row decode mask BEFORE the forward — necessary so
        // SDPA correctly masks stale K/V cells for rows whose cache
        // offsets have diverged from max(offsets).
        let pre_offsets: Vec<i32> = first_full_layer_offsets(cache_ref)?.to_vec();
        anyhow::ensure!(
            pre_offsets.len() == b,
            "step: cache offset rows {} != active rows {}",
            pre_offsets.len(),
            b
        );
        let mask_row_lens: Vec<i32> = pre_offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .collect();
        let max_real_len = mask_row_lens
            .iter()
            .copied()
            .max()
            .expect("active_rows is non-empty");
        let decode_mask = maybe_build_decode_mask(&mask_row_lens, max_real_len)?;

        #[cfg(feature = "p5h-profile")]
        let logits = crate::core::p5h::try_with_p5h_span_from_current_trace(
            "model_decode_forward",
            crate::core::p5h::SpanFields::default,
            || {
                model.forward_on(
                    &input_ids,
                    &position_ids,
                    Some(&per_row_lens),
                    decode_mask.as_ref(),
                    Some(cache_ref),
                    mlx::StreamOrDevice::default(),
                )
            },
        )?;
        #[cfg(not(feature = "p5h-profile"))]
        let logits = model.forward_on(
            &input_ids,
            &position_ids,
            Some(&per_row_lens),
            decode_mask.as_ref(),
            Some(cache_ref),
            mlx::StreamOrDevice::default(),
        )?;

        // logits shape: [B, 1, vocab]. Reshape to [B, vocab] for sample_batch.
        let logits_shape = logits.shape();
        let vocab = logits_shape.as_slice()[2];
        let logits_bv = logits
            .reshape(&[b as i32, vocab][..])
            .map_err(|e| anyhow!("step: reshape logits [B,1,vocab]->[B,vocab] failed: {e:?}"))?;

        // Stage A — collect per-row sampler refs + histories in compact
        // active-row order.
        let mut row_samplers: Vec<&Sampler> = Vec::with_capacity(b);
        let mut row_histories: Vec<Vec<u32>> = Vec::with_capacity(b);
        for &slot_row in &active_rows {
            let state = self.slots[slot_row]
                .as_ref()
                .expect("active_rows guaranteed Some");
            row_samplers.push(&state.sampler);
            let mut hist: Vec<u32> =
                Vec::with_capacity(state.prompt_ids.len() + state.generated_tokens.len());
            hist.extend_from_slice(&state.prompt_ids);
            hist.extend_from_slice(&state.generated_tokens);
            row_histories.push(hist);
        }

        // Stage B — dispatch sample_batch once over [B, vocab].
        let mut compact_prng = self.compact_prng_state_for_rows(&active_rows)?;
        let history_refs: Vec<&[u32]> = row_histories.iter().map(|h| h.as_slice()).collect();
        #[cfg(feature = "p5h-profile")]
        let tokens = crate::core::p5h::try_with_p5h_span_from_current_trace(
            "decode_sampling_materialize_and_sample",
            crate::core::p5h::SpanFields::default,
            || {
                crate::core::sampler::sample_batch(
                    &row_samplers,
                    &logits_bv,
                    &history_refs,
                    &mut compact_prng,
                )
                .map_err(|e| anyhow!("step: sample_batch failed: {e:?}"))
            },
        )?;
        #[cfg(not(feature = "p5h-profile"))]
        let tokens = crate::core::sampler::sample_batch(
            &row_samplers,
            &logits_bv,
            &history_refs,
            &mut compact_prng,
        )
        .map_err(|e| anyhow!("step: sample_batch failed: {e:?}"))?;
        drop(history_refs);
        drop(row_samplers);
        self.scatter_prng_state_from_rows(&active_rows, &compact_prng)?;

        // Stage C — distribute tokens + termination per active row.
        let mut events: Vec<StepEvent> = Vec::new();
        for (batch_row, &slot_row) in active_rows.iter().enumerate() {
            let token = tokens[batch_row];
            let state = self.slots[slot_row]
                .as_mut()
                .expect("active_rows guaranteed Some");

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
    /// # VL chunk boundaries
    /// VL chunking keeps each contiguous `image_pad` run in one chunk by
    /// extending a fixed boundary when it lands inside an image span. This
    /// preserves chunked prefill semantics while avoiding extra text forwards
    /// for one logical image.
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
    /// full-prompt position ids, and compute full-prompt vision embeddings.
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
            decode_cadence_mid_chunk_cap,
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
                state.decode_cadence_mid_chunk_cap,
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

        // Dtype from the first cache layer (Full or Mla).
        let dtype = {
            let main_cache = self
                .cache
                .as_ref()
                .ok_or_else(|| anyhow!("admit_mid_begin: main cache absent"))?;
            main_cache
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(kv) => Some(kv.dtype()),
                    LayerCache::Mla(mla) => Some(mla.dtype()),
                    _ => None,
                })
                .unwrap_or(Dtype::Bfloat16)
        };

        let turboquant_bits = self.kv_cache_turboquant_bits_for_rows(&[row_idx])?;
        let mut temp_cache =
            self.make_model_cache(model, 1, cap_for_temp, dtype, turboquant_bits)?;

        let is_vl = pixel_values.is_some();
        let prefix_fingerprint = if self.paged_prefix_cache.is_some() {
            paged_prefix_fingerprint_for_request(
                pixel_values.as_deref(),
                image_grid_thw.as_deref(),
                image_token_id,
                image_spatial_merge_size,
            )?
        } else {
            None
        };
        let prefix_restored_start = try_restore_paged_prefix_for_prompt(
            self.paged_prefix_cache.as_ref(),
            self.prefix_lru_cache.as_ref(),
            &mut temp_cache,
            &prompt_ids,
            prefix_fingerprint.as_deref(),
        )?
        .unwrap_or(0);
        let position_ids_required = model.requires_position_ids();

        let chunk_size = if prefill_chunk_size == 0 {
            prompt_len.max(1)
        } else {
            prefill_chunk_size.max(1)
        };

        // Pre-build full-prompt MRoPE position ids only for models that
        // consume them. Others carry a reusable placeholder and skip slicing
        // in `admit_mid_chunk`.
        let position_ids_full = if !position_ids_required {
            self.reusable_dummy_position_ids()?
        } else if is_vl {
            let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
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
                .as_deref()
                .expect("is_vl implies pixel_values is Some");
            let grids = image_grid_thw
                .as_deref()
                .expect("is_vl implies image_grid_thw is Some");
            Some(model.compute_vision_embeds(pv, grids, mlx::StreamOrDevice::default())?)
        } else {
            None
        };
        let image_pad_consumed = if is_vl {
            count_image_pad(
                &prompt_ids[..prefix_restored_start as usize],
                image_token_id,
            )
        } else {
            0
        };

        Ok(AdmitMidHandle {
            request_id: id,
            row_idx,
            prompt_ids,
            prompt_len,
            chunk_size,
            decode_cadence_mid_chunk_cap,
            chunk_start: prefix_restored_start,
            temp_cache,
            prefix_fingerprint,
            is_vl,
            image_token_id,
            position_ids_required,
            position_ids_full,
            vision_embeds_full,
            image_pad_consumed,
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

        let base_chunk_end = handle
            .chunk_start
            .saturating_add(handle.chunk_size)
            .min(handle.prompt_len);
        let mut chunk_end = if handle.is_vl {
            extend_vl_chunk_end_for_image_pad(
                &handle.prompt_ids,
                handle.image_token_id,
                handle.chunk_start,
                base_chunk_end,
            )
        } else {
            base_chunk_end
        };
        if self.paged_prefix_cache.is_some()
            && chunk_end == handle.prompt_len
            && handle.chunk_start < handle.prompt_len - 1
        {
            chunk_end = handle.prompt_len - 1;
        }
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

        // Slice axis 2 only when real caller-built position ids are required.
        // Models that derive positions internally carry a reusable placeholder.
        let position_ids = if handle.position_ids_required {
            slice_pos_ids_axis2(&handle.position_ids_full, handle.chunk_start, chunk_end)?
        } else {
            handle.position_ids_full.clone()
        };

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
        // - Intermediate chunk: text and VL paths use hidden-only forwards
        //   (skips lm_head; we don't need logits). Either way the result is
        //   `eval`-d before return so the
        //   chunk's lazy graph materialises here rather than ballooning
        //   into the interleaved `Scheduler::step` call. (GenerationStream
        //   chunked prefill at core/generate.rs ~line 1053 uses the
        //   same `eval(hidden)` pattern for the same reason.)
        let result_logits: Option<Array> = if handle.is_vl {
            // VL chunk: slice the rows of `vision_embeds_full` that
            // correspond to this chunk's `image_pad` token count.
            let k_i = count_image_pad(chunk_ids_u32, handle.image_token_id);
            let image_rows_start = handle.image_pad_consumed;
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
            log_vl_chunk_composition(
                "scheduler",
                handle.chunk_start..chunk_end,
                is_last,
                chunk_ids_u32,
                handle.image_token_id,
                image_rows_start..image_rows_start + k_i,
            );

            if is_last {
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
                Some(logits)
            } else {
                let hidden = model.forward_vl_hidden(
                    &input_ids,
                    &position_ids,
                    None, // per_row_lens (B=1 path derives from input shape)
                    None, // decode_mask (prefill — model builds its own causal mask)
                    Some(&mut handle.temp_cache),
                    ve_slice.as_ref(),
                    handle.image_token_id,
                    mlx::StreamOrDevice::default(),
                )?;
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
                    || mlx::transforms::eval(&[&hidden]).map_err(anyhow::Error::from),
                )?;
                #[cfg(not(feature = "p5h-profile"))]
                mlx::transforms::eval(&[&hidden])?;
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
        } else {
            match try_save_paged_prefix_for_prompt(
                self.paged_prefix_cache.as_ref(),
                self.prefix_lru_cache.as_ref(),
                &handle.temp_cache,
                &handle.prompt_ids[..chunk_end as usize],
                handle.prefix_fingerprint.as_deref(),
            ) {
                Ok(Some(key)) => {
                    tracing::debug!(
                        "paged SSD prefix cache saved during mid-admit chunk: key={key}"
                    );
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!("paged SSD prefix cache mid-admit chunk save skipped: {err:#}");
                }
            }
        }
        handle.chunk_start = chunk_end;
        Ok(is_last)
    }

    /// Finalise a chunked mid-batch admit: rebuild the compact main-cache
    /// layout from still-live decode rows plus `temp_cache` row 0 at
    /// `handle.row_idx`, sample the new row's first token from
    /// `handle.last_logits`, then update the row's termination state.
    ///
    /// Returns `(request_id, first_event)`; caller routes the event
    /// to its `event_rx`.
    pub fn admit_mid_finalize(
        &mut self,
        handle: AdmitMidHandle,
        model: &M,
    ) -> Result<(RequestId, StepEvent)> {
        self.ensure_not_poisoned()?;
        let AdmitMidHandle {
            request_id: id,
            row_idx,
            temp_cache,
            prefix_fingerprint,
            last_logits,
            prompt_ids,
            ..
        } = handle;

        let logits = last_logits
            .ok_or_else(|| anyhow!("admit_mid_finalize: last_logits absent (no chunks ran?)"))?;

        match try_save_paged_prefix_for_prompt(
            self.paged_prefix_cache.as_ref(),
            self.prefix_lru_cache.as_ref(),
            &temp_cache,
            &prompt_ids,
            prefix_fingerprint.as_deref(),
        ) {
            Ok(Some(key)) => {
                tracing::debug!("paged SSD prefix cache saved during mid-admit: key={key}");
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!("paged SSD prefix cache mid-admit save skipped: {err:#}");
            }
        }

        self.install_cache_with_temp_row(model, &temp_cache, row_idx)?;

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
    /// and return the evicted IDs. The compact cache is reconciled lazily
    /// by the next decode step or mid-admit finalize.
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
        self.prefill_cache_cap()
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
        self.cloned_active_row_p5h_trace_and_root_with_multi_row_escape_hatch(
            crate::core::p5h::scheduler_decode_allow_multi_row(),
        )
    }

    fn cloned_active_row_p5h_trace_and_root_with_multi_row_escape_hatch(
        &self,
        allow_multi_row: bool,
    ) -> anyhow::Result<
        Option<(
            crate::core::p5h::P5hTraceContext,
            crate::core::p5h::SpanHandle,
        )>,
    > {
        let active: Vec<&RequestState> = self.slots.iter().filter_map(|s| s.as_ref()).collect();
        if active.len() > 1 {
            let has_request_root_profile = active
                .iter()
                .any(|state| state.p5h_trace.is_some() || state.p5h_root_span.is_some());
            if allow_multi_row || !has_request_root_profile {
                return Ok(None);
            }
        }
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
    use std::collections::VecDeque;

    use crate::core::cache::MtpCache;
    use crate::core::speculative::{MtpSpeculativeConfig, MtpSpeculativeModel};
    use crate::nn::MtpStepOutput;
    use serial_test::serial;

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
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
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

    /// Minimal fake model for P5h+2.c scheduler unit tests.  Implements
    /// `Model` + `DenseVlMethods` without requiring a real Qwen35 weight
    /// file.  `batched_prefill` returns synthetic logits that always argmax
    /// to token 3; all other forward paths are unreachable from unit tests.
    struct P5h2cFakeModel;

    impl crate::core::model::Model for P5h2cFakeModel {
        fn make_cache(
            &self,
            _batch: i32,
            _cap: i32,
            _dtype: mlx::Dtype,
        ) -> crate::Result<Vec<crate::nn::LayerCache>> {
            Ok(Vec::new())
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let b = input_ids.shape().as_slice()[0] as usize;
            let vocab = 8_usize;
            let mut flat = vec![0.0_f32; b * vocab];
            for row in 0..b {
                flat[row * vocab + 3] = 100.0;
            }
            let logits_bv: mlx::Array = (&flat[..], &[b as i32, vocab as i32][..])
                .try_into()
                .expect("fake logits [B,V]");
            logits_bv
                .reshape(&[b as i32, 1, vocab as i32][..])
                .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
        }

        fn batched_prefill(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let b = input_ids.shape().as_slice()[0] as usize;
            let vocab = 8_usize;
            let mut flat = vec![0.0_f32; b * vocab];
            for row in 0..b {
                flat[row * vocab + 3] = 100.0;
            }
            let logits_bv: mlx::Array = (&flat[..], &[b as i32, vocab as i32][..])
                .try_into()
                .expect("fake logits [B,V]");
            logits_bv
                .reshape(&[b as i32, 1, vocab as i32][..])
                .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake hidden failed: {e:?}"))
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl DenseVlMethods for P5h2cFakeModel {
        fn batched_prefill_vl(
            &self,
            _input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            _grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }

        fn forward_vl_chunk(
            &self,
            _input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }

        fn forward_vl_hidden(
            &self,
            _input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }
    }

    #[derive(Default)]
    struct RecordingPrefillModel {
        make_cache_batches: std::sync::Mutex<Vec<i32>>,
        text_forward_batches: std::sync::Mutex<Vec<i32>>,
        text_forward_seq_lens: std::sync::Mutex<Vec<i32>>,
        text_forward_masks: std::sync::Mutex<Vec<(bool, bool)>>,
        text_hidden_shapes: std::sync::Mutex<Vec<(i32, i32)>>,
        text_prefill_batches: std::sync::Mutex<Vec<i32>>,
        vl_chunk_batches: std::sync::Mutex<Vec<i32>>,
        vl_chunk_vision_present: std::sync::Mutex<Vec<bool>>,
        vl_hidden_shapes: std::sync::Mutex<Vec<(i32, i32)>>,
        vl_hidden_vision_present: std::sync::Mutex<Vec<bool>>,
        vision_grid_lens: std::sync::Mutex<Vec<usize>>,
        vl_prefill_batches: std::sync::Mutex<Vec<i32>>,
        vl_pixel_value_lens: std::sync::Mutex<Vec<usize>>,
    }

    impl RecordingPrefillModel {
        fn make_cache_batches(&self) -> Vec<i32> {
            self.make_cache_batches.lock().unwrap().clone()
        }

        fn text_forward_batches(&self) -> Vec<i32> {
            self.text_forward_batches.lock().unwrap().clone()
        }

        fn text_forward_seq_lens(&self) -> Vec<i32> {
            self.text_forward_seq_lens.lock().unwrap().clone()
        }

        fn text_forward_masks(&self) -> Vec<(bool, bool)> {
            self.text_forward_masks.lock().unwrap().clone()
        }

        fn text_hidden_shapes(&self) -> Vec<(i32, i32)> {
            self.text_hidden_shapes.lock().unwrap().clone()
        }

        fn text_prefill_batches(&self) -> Vec<i32> {
            self.text_prefill_batches.lock().unwrap().clone()
        }

        fn vl_chunk_batches(&self) -> Vec<i32> {
            self.vl_chunk_batches.lock().unwrap().clone()
        }

        fn vl_chunk_vision_present(&self) -> Vec<bool> {
            self.vl_chunk_vision_present.lock().unwrap().clone()
        }

        fn vl_hidden_shapes(&self) -> Vec<(i32, i32)> {
            self.vl_hidden_shapes.lock().unwrap().clone()
        }

        fn vl_hidden_vision_present(&self) -> Vec<bool> {
            self.vl_hidden_vision_present.lock().unwrap().clone()
        }

        fn vision_grid_lens(&self) -> Vec<usize> {
            self.vision_grid_lens.lock().unwrap().clone()
        }

        fn vl_prefill_batches(&self) -> Vec<i32> {
            self.vl_prefill_batches.lock().unwrap().clone()
        }

        // Only read by `prefill_admitted_compacts_vl_rows`, which is gated out
        // of the p5h-profile build (multi-row test incompatible with the
        // single-row invariant). Gate the accessor identically so the
        // p5h-profile build does not warn on a dead method.
        #[cfg(not(feature = "p5h-profile"))]
        fn vl_pixel_value_lens(&self) -> Vec<usize> {
            self.vl_pixel_value_lens.lock().unwrap().clone()
        }
    }

    fn fake_logits_for_batch(batch: i32) -> crate::Result<mlx::Array> {
        let batch_usize = batch as usize;
        let vocab = 16_usize;
        let mut flat = vec![0.0_f32; batch_usize * vocab];
        for row in 0..batch_usize {
            flat[row * vocab + row + 3] = 100.0;
        }
        let logits_bv: mlx::Array = (&flat[..], &[batch, vocab as i32][..])
            .try_into()
            .expect("fake logits [B,V]");
        logits_bv
            .reshape(&[batch, 1, vocab as i32][..])
            .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
    }

    impl crate::core::model::Model for RecordingPrefillModel {
        fn make_cache(
            &self,
            batch: i32,
            _cap: i32,
            _dtype: mlx::Dtype,
        ) -> crate::Result<Vec<crate::nn::LayerCache>> {
            self.make_cache_batches.lock().unwrap().push(batch);
            Ok(Vec::new())
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            let batch = dims[0];
            let seq = dims[1];
            self.text_forward_batches.lock().unwrap().push(batch);
            self.text_forward_seq_lens.lock().unwrap().push(seq);
            self.text_forward_masks
                .lock()
                .unwrap()
                .push((per_row_lens.is_some(), decode_mask.is_some()));
            fake_logits_for_batch(batch)
        }

        fn batched_prefill(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let batch = input_ids.shape().as_slice()[0];
            self.text_prefill_batches.lock().unwrap().push(batch);
            fake_logits_for_batch(batch)
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.text_hidden_shapes
                .lock()
                .unwrap()
                .push((dims[0], dims[1]));
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake hidden failed: {e:?}"))
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl DenseVlMethods for RecordingPrefillModel {
        fn batched_prefill_vl(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let batch = input_ids.shape().as_slice()[0];
            self.vl_prefill_batches.lock().unwrap().push(batch);
            self.vl_pixel_value_lens
                .lock()
                .unwrap()
                .push(per_row_pixel_values.len());
            fake_logits_for_batch(batch)
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            self.vision_grid_lens.lock().unwrap().push(grid_thw.len());
            let flat = vec![0.0_f32; grid_thw.len().max(1)];
            (&flat[..], &[grid_thw.len().max(1) as i32, 1_i32][..])
                .try_into()
                .map_err(|e| anyhow::anyhow!("fake vision embeds failed: {e:?}"))
        }

        fn forward_vl_chunk(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let batch = input_ids.shape().as_slice()[0];
            self.vl_chunk_batches.lock().unwrap().push(batch);
            self.vl_chunk_vision_present
                .lock()
                .unwrap()
                .push(vision_embeds_slice.is_some());
            fake_logits_for_batch(batch)
        }

        fn forward_vl_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.vl_hidden_shapes
                .lock()
                .unwrap()
                .push((dims[0], dims[1]));
            self.vl_hidden_vision_present
                .lock()
                .unwrap()
                .push(vision_embeds_slice.is_some());
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vl hidden failed: {e:?}"))
        }
    }

    #[derive(Default)]
    struct StepDecodeMaskModel {
        decode_lens_seen: std::sync::Mutex<Vec<Vec<i32>>>,
        decode_mask_seen: std::sync::Mutex<Vec<bool>>,
        forward_seq_lens: std::sync::Mutex<Vec<i32>>,
        hidden_seq_lens: std::sync::Mutex<Vec<i32>>,
        batched_prefill_batches: std::sync::Mutex<Vec<i32>>,
        batched_prefill_lens: std::sync::Mutex<Vec<Vec<i32>>>,
    }

    impl StepDecodeMaskModel {
        fn bump_first_full_cache(
            cache: Option<&mut [crate::nn::LayerCache]>,
            input_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
        ) -> crate::Result<()> {
            let Some(cache) = cache else {
                return Ok(());
            };
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            let batch = dims[0];
            let seq = dims[1];
            let lens_owned;
            let lens = if let Some(lens) = per_row_lens {
                lens
            } else {
                lens_owned = vec![seq; batch as usize];
                lens_owned.as_slice()
            };
            let k = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("fake k failed: {e:?}"))?;
            let v = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("fake v failed: {e:?}"))?;
            for layer in cache {
                if let crate::nn::LayerCache::Full(kv) = layer {
                    kv.update_and_fetch(&k, &v, lens)?;
                    break;
                }
            }
            Ok(())
        }

        fn decode_lens_seen(&self) -> Vec<Vec<i32>> {
            self.decode_lens_seen.lock().unwrap().clone()
        }

        fn forward_seq_lens(&self) -> Vec<i32> {
            self.forward_seq_lens.lock().unwrap().clone()
        }

        fn hidden_seq_lens(&self) -> Vec<i32> {
            self.hidden_seq_lens.lock().unwrap().clone()
        }

        fn batched_prefill_batches(&self) -> Vec<i32> {
            self.batched_prefill_batches.lock().unwrap().clone()
        }

        fn batched_prefill_lens(&self) -> Vec<Vec<i32>> {
            self.batched_prefill_lens.lock().unwrap().clone()
        }

        #[cfg(not(feature = "p5h-profile"))]
        fn decode_mask_seen(&self) -> Vec<bool> {
            self.decode_mask_seen.lock().unwrap().clone()
        }
    }

    impl crate::core::model::Model for StepDecodeMaskModel {
        fn make_cache(
            &self,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> crate::Result<Vec<crate::nn::LayerCache>> {
            Ok(vec![crate::nn::LayerCache::Full(
                crate::core::KVCache::new(batch, 1, 1, 1, dtype, cap),
            )])
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.forward_seq_lens.lock().unwrap().push(dims[1]);
            if let Some(lens) = per_row_lens {
                self.decode_lens_seen.lock().unwrap().push(lens.to_vec());
            }
            self.decode_mask_seen
                .lock()
                .unwrap()
                .push(decode_mask.is_some());
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            fake_logits_for_batch(dims[0])
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
        ) -> crate::Result<mlx::Array> {
            self.batched_prefill_batches
                .lock()
                .unwrap()
                .push(input_ids.shape().as_slice()[0]);
            self.batched_prefill_lens
                .lock()
                .unwrap()
                .push(per_row_lens.to_vec());
            Self::bump_first_full_cache(cache, input_ids, Some(per_row_lens))?;
            fake_logits_for_batch(input_ids.shape().as_slice()[0])
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.hidden_seq_lens.lock().unwrap().push(dims[1]);
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake hidden failed: {e:?}"))
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            1
        }
    }

    impl DenseVlMethods for StepDecodeMaskModel {
        fn batched_prefill_vl(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            self.batched_prefill_batches
                .lock()
                .unwrap()
                .push(input_ids.shape().as_slice()[0]);
            self.batched_prefill_lens
                .lock()
                .unwrap()
                .push(per_row_lens.to_vec());
            Self::bump_first_full_cache(cache, input_ids, Some(per_row_lens))?;
            fake_logits_for_batch(input_ids.shape().as_slice()[0])
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let rows = grid_thw.len().max(1) as i32;
            mlx::Array::zeros((rows, 1_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vision embeds failed: {e:?}"))
        }

        fn forward_vl_chunk(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.forward_seq_lens.lock().unwrap().push(dims[1]);
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            fake_logits_for_batch(dims[0])
        }

        fn forward_vl_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.hidden_seq_lens.lock().unwrap().push(dims[1]);
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vl hidden failed: {e:?}"))
        }
    }

    #[derive(Clone, Copy)]
    struct FakeMtpHead;

    #[derive(Default)]
    struct ScriptedMtpSchedulerModel {
        first_token: u32,
        first_token_calls_remaining: std::sync::Mutex<usize>,
        draft_tokens: std::sync::Mutex<VecDeque<u32>>,
        verify_sequences: std::sync::Mutex<VecDeque<Vec<u32>>>,
        mtp_hidden_seq_lens: std::sync::Mutex<Vec<i32>>,
        vl_hidden_seq_lens: std::sync::Mutex<Vec<i32>>,
        vl_hidden_vision_present: std::sync::Mutex<Vec<bool>>,
        vision_grid_lens: std::sync::Mutex<Vec<usize>>,
        project_calls: std::sync::Mutex<usize>,
        fail_mtp_cache: bool,
    }

    impl ScriptedMtpSchedulerModel {
        fn new(first_token: u32, draft_tokens: Vec<u32>, verify_sequences: Vec<Vec<u32>>) -> Self {
            Self::new_with_first_token_calls(first_token, 1, draft_tokens, verify_sequences)
        }

        fn new_with_first_token_calls(
            first_token: u32,
            first_token_calls: usize,
            draft_tokens: Vec<u32>,
            verify_sequences: Vec<Vec<u32>>,
        ) -> Self {
            Self {
                first_token,
                first_token_calls_remaining: std::sync::Mutex::new(first_token_calls),
                draft_tokens: std::sync::Mutex::new(draft_tokens.into()),
                verify_sequences: std::sync::Mutex::new(verify_sequences.into()),
                mtp_hidden_seq_lens: std::sync::Mutex::new(Vec::new()),
                vl_hidden_seq_lens: std::sync::Mutex::new(Vec::new()),
                vl_hidden_vision_present: std::sync::Mutex::new(Vec::new()),
                vision_grid_lens: std::sync::Mutex::new(Vec::new()),
                project_calls: std::sync::Mutex::new(0),
                fail_mtp_cache: false,
            }
        }

        fn with_mtp_cache_failure(first_token: u32) -> Self {
            Self {
                first_token,
                first_token_calls_remaining: std::sync::Mutex::new(1),
                draft_tokens: std::sync::Mutex::new(VecDeque::new()),
                verify_sequences: std::sync::Mutex::new(VecDeque::new()),
                mtp_hidden_seq_lens: std::sync::Mutex::new(Vec::new()),
                vl_hidden_seq_lens: std::sync::Mutex::new(Vec::new()),
                vl_hidden_vision_present: std::sync::Mutex::new(Vec::new()),
                vision_grid_lens: std::sync::Mutex::new(Vec::new()),
                project_calls: std::sync::Mutex::new(0),
                fail_mtp_cache: true,
            }
        }

        fn mtp_hidden_seq_lens(&self) -> Vec<i32> {
            self.mtp_hidden_seq_lens.lock().unwrap().clone()
        }

        fn vl_hidden_seq_lens(&self) -> Vec<i32> {
            self.vl_hidden_seq_lens.lock().unwrap().clone()
        }

        fn vl_hidden_vision_present(&self) -> Vec<bool> {
            self.vl_hidden_vision_present.lock().unwrap().clone()
        }

        fn vision_grid_lens(&self) -> Vec<usize> {
            self.vision_grid_lens.lock().unwrap().clone()
        }

        fn bump_first_full_cache(
            cache: Option<&mut [crate::nn::LayerCache]>,
            input_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
        ) -> crate::Result<()> {
            let Some(cache) = cache else {
                return Ok(());
            };
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            let batch = dims[0];
            let seq = dims[1];
            let lens_owned;
            let lens = if let Some(lens) = per_row_lens {
                lens
            } else {
                lens_owned = vec![seq; batch as usize];
                lens_owned.as_slice()
            };
            let k = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("fake k failed: {e:?}"))?;
            let v = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("fake v failed: {e:?}"))?;
            for layer in cache {
                if let crate::nn::LayerCache::Full(kv) = layer {
                    kv.update_and_fetch(&k, &v, lens)?;
                    break;
                }
            }
            Ok(())
        }
    }

    fn fake_logits_for_token_sequence(tokens: &[u32]) -> crate::Result<mlx::Array> {
        let seq = tokens.len();
        let vocab = 32_usize;
        let mut flat = vec![0.0_f32; seq * vocab];
        for (pos, &token) in tokens.iter().enumerate() {
            let token = token as usize;
            assert!(
                token < vocab,
                "fake token {token} must fit fake vocab {vocab}"
            );
            flat[pos * vocab + token] = 100.0;
        }
        let logits: mlx::Array = (&flat[..], &[1_i32, seq as i32, vocab as i32][..])
            .try_into()
            .expect("fake logits [1,S,V]");
        Ok(logits)
    }

    impl crate::core::model::Model for ScriptedMtpSchedulerModel {
        fn make_cache(
            &self,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> crate::Result<Vec<crate::nn::LayerCache>> {
            Ok(vec![crate::nn::LayerCache::Full(
                crate::core::KVCache::new(batch, 1, 1, 1, dtype, cap),
            )])
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            fake_logits_for_token_sequence(&[self.first_token])
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
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, Some(per_row_lens))?;
            fake_logits_for_token_sequence(&[self.first_token])
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake hidden failed: {e:?}"))
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            1
        }
    }

    impl DenseVlMethods for ScriptedMtpSchedulerModel {
        fn batched_prefill_vl(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, Some(per_row_lens))?;
            fake_logits_for_token_sequence(&[self.first_token])
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            self.vision_grid_lens.lock().unwrap().push(grid_thw.len());
            let rows: i32 = grid_thw
                .iter()
                .map(|&(t, h, w)| t * (h / 2).max(1) * (w / 2).max(1))
                .sum::<i32>()
                .max(1);
            mlx::Array::zeros((rows, 1_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vision embeds failed: {e:?}"))
        }

        fn forward_vl_chunk(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            fake_logits_for_token_sequence(&[self.first_token])
        }

        fn forward_vl_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<mlx::Array> {
            Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            self.vl_hidden_seq_lens.lock().unwrap().push(dims[1]);
            self.vl_hidden_vision_present
                .lock()
                .unwrap()
                .push(vision_embeds_slice.is_some());
            mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vl hidden failed: {e:?}"))
        }
    }

    impl MtpSpeculativeModel for ScriptedMtpSchedulerModel {
        type MtpHead = FakeMtpHead;

        fn load_mtp_head(&self, _loader: &crate::core::Loader) -> crate::Result<Self::MtpHead> {
            Ok(FakeMtpHead)
        }

        fn make_mtp_cache(
            &self,
            _mtp: &Self::MtpHead,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> crate::Result<MtpCache> {
            if self.fail_mtp_cache {
                anyhow::bail!("fake mtp cache failure");
            }
            MtpCache::new_with_cap(1, batch, 1, 1, 1, dtype, cap)
        }

        fn project_hidden_on(
            &self,
            hidden: &mlx::Array,
            _target: impl Into<mlx::StreamOrDevice>,
        ) -> crate::Result<mlx::Array> {
            let seq = hidden.shape().as_slice()[1] as usize;
            let mut calls = self.project_calls.lock().unwrap();
            let mut first_token_calls_remaining = self.first_token_calls_remaining.lock().unwrap();
            let tokens = if seq == 1 && *first_token_calls_remaining > 0 {
                *first_token_calls_remaining -= 1;
                vec![self.first_token]
            } else {
                self.verify_sequences
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("verify sequence available")
            };
            *calls += 1;
            assert_eq!(
                tokens.len(),
                seq,
                "project logits sequence length must match hidden seq"
            );
            fake_logits_for_token_sequence(&tokens)
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
        ) -> crate::Result<mlx::Array> {
            self.mtp_hidden_seq_lens
                .lock()
                .unwrap()
                .push(next_token_ids.shape().as_slice()[1]);
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
        ) -> crate::Result<MtpStepOutput> {
            let hidden_states = self.mtp_forward_hidden_on(
                mtp,
                hidden_states,
                next_token_ids,
                position_ids,
                mask,
                mtp_cache,
                target,
            )?;
            let token = self
                .draft_tokens
                .lock()
                .unwrap()
                .pop_front()
                .expect("draft token available");
            Ok(MtpStepOutput {
                hidden_states,
                logits: fake_logits_for_token_sequence(&[token])?,
            })
        }
    }

    fn mtp_req(prompt_ids: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
        let mut req = mk_req(prompt_ids);
        req.max_new_tokens = max_new_tokens;
        req.stop_token_ids = vec![31];
        req
    }

    fn mtp_vl_req(prompt_ids: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
        let mut req = mtp_req(prompt_ids, max_new_tokens);
        req.pixel_values = Some(vec![
            mlx::Array::zeros((1_i32, 1_i32), mlx::Dtype::Float32).unwrap()
        ]);
        req.image_grid_thw = Some(vec![(1, 2, 2)]);
        req
    }

    fn mtp_cache_offset(s: &Scheduler<ScriptedMtpSchedulerModel>) -> i32 {
        s.mtp_state
            .as_ref()
            .expect("scheduler MTP state")
            .rows
            .get(&0)
            .expect("row 0 MTP state")
            .mtp_cache
            .offset()
    }

    #[test]
    #[serial(mlx_metal)]
    fn mtp_prefill_uses_paged_ssd_prefix_cache_on_exact_hit() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-mtp-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "mtp-test", 2, 32)
            .expect("prefix config");

        let mut warm = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        warm.enable_paged_prefix_cache(config.clone())
            .expect("enable prefix cache");
        let warm_id = warm
            .admit(mtp_req(vec![1, 2, 3, 4], 1))
            .expect("admit warm MTP");
        let warm_model = ScriptedMtpSchedulerModel::new(5, Vec::new(), Vec::new());
        let warm_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let warm_events = warm
            .prefill_admitted_mtp_single(&warm_model, &FakeMtpHead, warm_cfg)
            .expect("warm MTP prefill");
        assert_eq!(
            warm_events,
            vec![StepEvent {
                id: warm_id,
                token: 5,
                finish_reason: Some("length")
            }]
        );
        assert_eq!(
            warm_model.mtp_hidden_seq_lens(),
            vec![3, 1],
            "warm MTP prefill should split N-1 prefix before the final token"
        );
        assert!(
            std::fs::read_dir(&root)
                .expect("prefix cache dir")
                .next()
                .is_some(),
            "warm MTP prefill should save prefix entries"
        );

        let mut hit = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        hit.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let hit_id = hit
            .admit(mtp_req(vec![1, 2, 3, 4], 1))
            .expect("admit hit MTP");
        let hit_model = ScriptedMtpSchedulerModel::new(5, Vec::new(), Vec::new());
        let hit_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let hit_events = hit
            .prefill_admitted_mtp_single(&hit_model, &FakeMtpHead, hit_cfg)
            .expect("hit MTP prefill");
        assert_eq!(
            hit_events,
            vec![StepEvent {
                id: hit_id,
                token: 5,
                finish_reason: Some("length")
            }]
        );
        assert_eq!(
            hit_model.mtp_hidden_seq_lens(),
            vec![1],
            "exact MTP prefix hit should restore N-1 main/MTP cache state and compute only the final token"
        );

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    #[serial(mlx_metal)]
    fn mtp_batch_prefill_uses_paged_ssd_prefix_cache_on_exact_hits() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-mtp-batch-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config =
            crate::core::cache::PagedPrefixCacheConfig::new(&root, "mtp-batch-test", 2, 64)
                .expect("prefix config");

        for prompt in [vec![1, 2, 3, 4], vec![9, 8, 7]] {
            let mut warm = Scheduler::<ScriptedMtpSchedulerModel>::new(
                1,
                32768,
                crate::core::memory_budget::test_meta_qwen35(),
            )
            .expect("scheduler startup");
            warm.enable_paged_prefix_cache(config.clone())
                .expect("enable prefix cache");
            warm.admit(mtp_req(prompt, 1)).expect("admit warm MTP");
            let warm_model = ScriptedMtpSchedulerModel::new(5, Vec::new(), Vec::new());
            let warm_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
            warm.prefill_admitted_mtp_single(&warm_model, &FakeMtpHead, warm_cfg)
                .expect("warm MTP prefill");
        }

        let mut hit = Scheduler::<ScriptedMtpSchedulerModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        hit.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let id0 = hit
            .admit(mtp_req(vec![1, 2, 3, 4], 1))
            .expect("admit first hit MTP");
        let id1 = hit
            .admit(mtp_req(vec![9, 8, 7], 1))
            .expect("admit second hit MTP");
        let hit_model =
            ScriptedMtpSchedulerModel::new_with_first_token_calls(5, 2, Vec::new(), Vec::new());
        let hit_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let hit_events = hit
            .prefill_admitted_mtp_batch(&hit_model, &FakeMtpHead, hit_cfg)
            .expect("hit MTP batch prefill");

        assert_eq!(
            hit_events,
            vec![
                StepEvent {
                    id: id0,
                    token: 5,
                    finish_reason: Some("length")
                },
                StepEvent {
                    id: id1,
                    token: 5,
                    finish_reason: Some("length")
                }
            ]
        );
        assert_eq!(
            hit_model.mtp_hidden_seq_lens(),
            vec![1, 1],
            "exact MTP prefix hits should restore each row's N-1 main/MTP cache state and compute only final tokens"
        );

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    fn mtp_batch_eligibility_accepts_single_text_greedy_request() {
        let mut scheduler = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            16,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler");
        scheduler
            .admit(mtp_req(vec![1, 2], 4))
            .expect("admit text greedy");

        assert!(scheduler.mtp_batch_active_greedy_eligible());
    }

    #[test]
    fn mtp_batch_eligibility_accepts_vl_and_rejects_non_greedy_requests() {
        let mut vl_scheduler = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            16,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler");
        let vl = mtp_vl_req(vec![1, 248056, 2], 4);
        vl_scheduler.admit(vl).expect("admit vl");
        assert!(vl_scheduler.mtp_batch_active_greedy_eligible());

        let mut sampling_scheduler = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            16,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler");
        let mut sampled = mtp_req(vec![1, 2], 4);
        sampled.sampler = Sampler::greedy().with_temperature(0.7);
        sampling_scheduler.admit(sampled).expect("admit sampled");
        assert!(!sampling_scheduler.mtp_batch_active_greedy_eligible());
    }

    #[test]
    fn mtp_prefill_rejects_bmax_gt_one() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.admit(mtp_req(vec![1, 2, 3], 4)).expect("admit");

        let model = ScriptedMtpSchedulerModel::new(3, vec![4], vec![vec![4, 5]]);
        let cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let err = s
            .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect_err("b_max > 1 must reject scheduler MTP");
        assert!(err.to_string().contains("b_max 1"), "unexpected err: {err}");
    }

    #[test]
    fn mtp_prefill_vl_uses_paged_ssd_prefix_cache_on_exact_hit() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-mtp-vl-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "mtp-vl-test", 2, 32)
            .expect("prefix config");
        let prompt = vec![1, 248056, 2, 3];

        let mut warm = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        warm.enable_paged_prefix_cache(config.clone())
            .expect("enable prefix cache");
        let warm_id = warm
            .admit(mtp_vl_req(prompt.clone(), 1))
            .expect("admit warm VL MTP");
        let warm_model = ScriptedMtpSchedulerModel::new(5, Vec::new(), Vec::new());
        let warm_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let warm_events = warm
            .prefill_admitted_mtp_single(&warm_model, &FakeMtpHead, warm_cfg)
            .expect("warm VL MTP prefill");
        assert_eq!(
            warm_events,
            vec![StepEvent {
                id: warm_id,
                token: 5,
                finish_reason: Some("length")
            }]
        );
        assert_eq!(warm_model.vision_grid_lens(), vec![1]);
        assert_eq!(
            warm_model.vl_hidden_seq_lens(),
            vec![3, 1],
            "warm VL MTP prefill should split N-1 prefix before the final token"
        );
        assert_eq!(
            warm_model.vl_hidden_vision_present(),
            vec![true, false],
            "only the prefix chunk should carry the image embedding slice"
        );
        assert_eq!(warm_model.mtp_hidden_seq_lens(), vec![3, 1]);

        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let hit_id = s.admit(mtp_vl_req(prompt, 1)).expect("admit hit VL MTP");
        let model = ScriptedMtpSchedulerModel::new(5, Vec::new(), Vec::new());
        let cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let hit_events = s
            .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect("hit VL MTP prefill");
        assert_eq!(
            hit_events,
            vec![StepEvent {
                id: hit_id,
                token: 5,
                finish_reason: Some("length")
            }]
        );
        assert_eq!(
            model.vl_hidden_seq_lens(),
            vec![1],
            "exact VL MTP prefix hit should restore N-1 main/MTP cache state and compute only the final token"
        );
        assert_eq!(model.vl_hidden_vision_present(), vec![false]);
        assert_eq!(model.mtp_hidden_seq_lens(), vec![1]);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    #[serial(mlx_metal)]
    fn mtp_batch_prefill_vl_uses_paged_ssd_prefix_cache_on_exact_hits() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-mtp-vl-batch-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config =
            crate::core::cache::PagedPrefixCacheConfig::new(&root, "mtp-vl-batch-test", 2, 64)
                .expect("prefix config");
        let prompts = [vec![1, 248056, 2, 3], vec![4, 248056, 5]];

        for prompt in prompts.iter().cloned() {
            let mut warm = Scheduler::<ScriptedMtpSchedulerModel>::new(
                1,
                32768,
                crate::core::memory_budget::test_meta_qwen35(),
            )
            .expect("scheduler startup");
            warm.enable_paged_prefix_cache(config.clone())
                .expect("enable prefix cache");
            warm.admit(mtp_vl_req(prompt, 1))
                .expect("admit warm VL MTP");
            let warm_model = ScriptedMtpSchedulerModel::new(5, Vec::new(), Vec::new());
            let warm_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
            warm.prefill_admitted_mtp_single(&warm_model, &FakeMtpHead, warm_cfg)
                .expect("warm VL MTP prefill");
        }

        let mut hit = Scheduler::<ScriptedMtpSchedulerModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        hit.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let id0 = hit
            .admit(mtp_vl_req(prompts[0].clone(), 1))
            .expect("admit first VL hit");
        let id1 = hit
            .admit(mtp_vl_req(prompts[1].clone(), 1))
            .expect("admit second VL hit");
        let hit_model =
            ScriptedMtpSchedulerModel::new_with_first_token_calls(5, 2, Vec::new(), Vec::new());
        let hit_cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
        let hit_events = hit
            .prefill_admitted_mtp_batch(&hit_model, &FakeMtpHead, hit_cfg)
            .expect("hit VL MTP batch prefill");

        assert_eq!(
            hit_events,
            vec![
                StepEvent {
                    id: id0,
                    token: 5,
                    finish_reason: Some("length")
                },
                StepEvent {
                    id: id1,
                    token: 5,
                    finish_reason: Some("length")
                }
            ]
        );
        assert_eq!(hit_model.vl_hidden_seq_lens(), vec![1, 1]);
        assert_eq!(hit_model.vl_hidden_vision_present(), vec![false, false]);
        assert_eq!(hit_model.mtp_hidden_seq_lens(), vec![1, 1]);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    fn mtp_prefill_error_allows_evict_all_recovery() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let _id = s.admit(mtp_req(vec![1, 2], 4)).expect("admit");
        let model = ScriptedMtpSchedulerModel::with_mtp_cache_failure(3);
        let cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");

        let err = s
            .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect_err("fake MTP cache failure must poison scheduler");
        assert!(err.to_string().contains("fake mtp cache failure"));

        s.evict_all()
            .expect("evict_all must recover after MTP prefill failure");
        assert_eq!(s.phase(), Phase::Idle);
        assert_eq!(s.active_count(), 0);
        assert!(s.mtp_stats().is_none());

        s.admit(mtp_req(vec![1, 2], 1))
            .expect("scheduler accepts new request after recovery");
    }

    #[test]
    fn mtp_step_emits_one_pending_token_per_call() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id = s.admit(mtp_req(vec![1, 2], 4)).expect("admit");
        let model = ScriptedMtpSchedulerModel::new(3, vec![4, 5], vec![vec![4, 5, 6]]);
        let cfg = MtpSpeculativeConfig::new(2, Sampler::greedy()).expect("mtp cfg");

        let first = s
            .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect("mtp prefill");
        assert_eq!(
            first,
            vec![StepEvent {
                id,
                token: 3,
                finish_reason: None
            }]
        );
        assert_eq!(s.get(id).unwrap().generated_tokens, vec![3]);
        assert_eq!(
            mtp_cache_offset(&s),
            5,
            "MTP cache should contain prompt tokens plus current token and accepted drafts"
        );

        let step_1 = s.step_mtp_single(&model, &FakeMtpHead).expect("step 1");
        assert_eq!(
            step_1,
            vec![StepEvent {
                id,
                token: 4,
                finish_reason: None
            }]
        );
        assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4]);

        let step_2 = s.step_mtp_single(&model, &FakeMtpHead).expect("step 2");
        assert_eq!(
            step_2,
            vec![StepEvent {
                id,
                token: 5,
                finish_reason: None
            }]
        );
        assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4, 5]);

        let step_3 = s.step_mtp_single(&model, &FakeMtpHead).expect("step 3");
        assert_eq!(
            step_3,
            vec![StepEvent {
                id,
                token: 6,
                finish_reason: Some("length")
            }]
        );
        assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4, 5, 6]);

        let stats = s.mtp_stats().expect("mtp stats");
        assert_eq!(stats.windows, 1);
        assert_eq!(stats.drafted_tokens, 2);
        assert_eq!(stats.accepted_draft_tokens, 2);
        assert_eq!(stats.rollback_count, 0);
    }

    #[test]
    fn mtp_full_accept_reuses_draft_cache_and_commits_only_tail() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.admit(mtp_req(vec![1, 2], 4)).expect("admit");
        let model = ScriptedMtpSchedulerModel::new(3, vec![4, 5], vec![vec![4, 5, 6]]);
        let cfg = MtpSpeculativeConfig::new(2, Sampler::greedy()).expect("mtp cfg");

        s.prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect("mtp prefill");

        assert_eq!(
            model.mtp_hidden_seq_lens(),
            vec![2, 1, 1, 1],
            "full-accept window should reuse the two draft cache steps and commit only the missing tail token"
        );
    }

    #[test]
    fn mtp_adaptive_budget_reduces_after_first_token_mismatch() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.admit(mtp_req(vec![1, 2], 8)).expect("admit");
        let model =
            ScriptedMtpSchedulerModel::new(3, vec![8, 9, 10, 11], vec![vec![4, 5, 6, 7, 8]]);
        let cfg = MtpSpeculativeConfig::new(4, Sampler::greedy()).expect("mtp cfg");

        s.prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect("mtp prefill");

        let mtp_state = s.mtp_state.as_ref().expect("scheduler MTP state");
        assert_eq!(
            mtp_state
                .rows
                .get(&0)
                .expect("row 0 MTP state")
                .adaptive_draft_tokens,
            1,
            "a first-token mismatch should reduce the next draft budget to one"
        );
        assert_eq!(mtp_state.stats.draft_budget_reductions, 1);
    }

    #[test]
    fn mtp_step_updates_stats_after_mismatch() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id = s.admit(mtp_req(vec![1, 2], 2)).expect("admit");
        let model = ScriptedMtpSchedulerModel::new(3, vec![8], vec![vec![4, 5]]);
        let cfg = MtpSpeculativeConfig::new(2, Sampler::greedy()).expect("mtp cfg");

        let first = s
            .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
            .expect("mtp prefill");
        assert_eq!(
            first,
            vec![StepEvent {
                id,
                token: 3,
                finish_reason: None
            }]
        );

        let stats = s.mtp_stats().expect("mtp stats after prefill window");
        assert_eq!(stats.windows, 1);
        assert_eq!(stats.drafted_tokens, 1);
        assert_eq!(stats.accepted_draft_tokens, 0);
        assert_eq!(stats.rollback_count, 1);
        assert_eq!(
            mtp_cache_offset(&s),
            3,
            "MTP cache should contain prompt tokens plus only the kept current token after mismatch"
        );

        let step = s
            .step_mtp_single(&model, &FakeMtpHead)
            .expect("corrected step");
        assert_eq!(
            step,
            vec![StepEvent {
                id,
                token: 4,
                finish_reason: Some("length")
            }]
        );
        assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn mtp_batch_step_emits_pending_tokens_for_each_unfinished_row() {
        let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id0 = s.admit(mtp_req(vec![1, 2], 3)).expect("admit row 0");
        let id1 = s.admit(mtp_req(vec![10, 11], 3)).expect("admit row 1");
        let model = ScriptedMtpSchedulerModel::new_with_first_token_calls(
            3,
            2,
            vec![4, 6],
            vec![vec![4, 5], vec![6, 7]],
        );
        let cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");

        let first = s
            .prefill_admitted_mtp_batch(&model, &FakeMtpHead, cfg)
            .expect("mtp batch prefill");
        assert_eq!(
            first,
            vec![
                StepEvent {
                    id: id0,
                    token: 3,
                    finish_reason: None
                },
                StepEvent {
                    id: id1,
                    token: 3,
                    finish_reason: None
                }
            ]
        );

        let step_1 = s
            .step_mtp_batch(&model, &FakeMtpHead)
            .expect("first batch MTP step");
        assert_eq!(
            step_1,
            vec![
                StepEvent {
                    id: id0,
                    token: 4,
                    finish_reason: None
                },
                StepEvent {
                    id: id1,
                    token: 6,
                    finish_reason: None
                }
            ]
        );

        let step_2 = s
            .step_mtp_batch(&model, &FakeMtpHead)
            .expect("second batch MTP step");
        assert_eq!(
            step_2,
            vec![
                StepEvent {
                    id: id0,
                    token: 5,
                    finish_reason: Some("length")
                },
                StepEvent {
                    id: id1,
                    token: 7,
                    finish_reason: Some("length")
                }
            ]
        );
        assert_eq!(s.get(id0).unwrap().generated_tokens, vec![3, 4, 5]);
        assert_eq!(s.get(id1).unwrap().generated_tokens, vec![3, 6, 7]);
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

    #[cfg(feature = "p5h-profile")]
    #[test]
    fn p5h_scheduler_decode_multi_row_escape_hatch_skips_legacy_request_root() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let id_0 = s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![4, 5, 6])).expect("admit 1");
        let ctx_0 = crate::core::p5h::P5hTraceContext {
            request_id: "p5h-test-0".to_string(),
            prompt_tokens: 3,
            routing_path: "scheduler",
        };
        let ctx_1 = crate::core::p5h::P5hTraceContext {
            request_id: "p5h-test-1".to_string(),
            prompt_tokens: 3,
            routing_path: "scheduler",
        };
        let root_0 = crate::core::p5h::open_p5h_span(&ctx_0, None, "request_root");
        let root_1 = crate::core::p5h::open_p5h_span(&ctx_1, None, "request_root");
        {
            let state = s.get_mut(id_0).expect("state 0");
            state.p5h_trace = Some(ctx_0.clone());
            state.p5h_root_span = Some(root_0.clone());
        }
        {
            let state = s.get_mut(id_1).expect("state 1");
            state.p5h_trace = Some(ctx_1.clone());
            state.p5h_root_span = Some(root_1.clone());
        }

        let err = s
            .cloned_active_row_p5h_trace_and_root_with_multi_row_escape_hatch(false)
            .expect_err("legacy request-root profiling still requires one row");
        assert!(err.to_string().contains("expected exactly 1 active row"));

        let skipped = s
            .cloned_active_row_p5h_trace_and_root_with_multi_row_escape_hatch(true)
            .expect("escape hatch should skip legacy request-root profiling");
        assert!(skipped.is_none());

        crate::core::p5h::close_p5h_span(
            &ctx_0,
            root_0,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
        crate::core::p5h::close_p5h_span(
            &ctx_1,
            root_1,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );
    }

    #[cfg(feature = "p5h-profile")]
    #[test]
    fn p5h_request_root_noops_for_multi_row_without_trace() {
        let mut s = TestScheduler::new(2, 32768, crate::core::memory_budget::test_meta_qwen35())
            .expect("scheduler startup");
        let _id_0 = s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        let _id_1 = s.admit(mk_req(vec![4, 5, 6])).expect("admit 1");

        let skipped = s
            .cloned_active_row_p5h_trace_and_root_with_multi_row_escape_hatch(false)
            .expect("multi-row requests without p5h trace should not open request-root spans");
        assert!(skipped.is_none());
    }

    // Multi-row prefill test: builds >1 active row, so it exercises the
    // batched compaction path. Under `p5h-profile` the scheduler enforces a
    // hard single-row invariant (`prefill_admitted_inner` →
    // `cloned_active_row_p5h_trace_and_root`: "expected exactly 1 active row,
    // --b-max 1 required"), so a multi-row build is meaningless / would fail
    // there. Gate it out of the p5h-profile build; it still runs under
    // default features.
    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn prefill_admitted_compacts_sparse_initial_rows() {
        let mut s = Scheduler::<RecordingPrefillModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id_0 = s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        let id_gap = s.admit(mk_req(vec![4, 5, 6])).expect("admit gap");
        let id_2 = s.admit(mk_req(vec![7, 8, 9])).expect("admit 2");
        s.evict(id_gap).expect("evict middle row");

        let model = RecordingPrefillModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.make_cache_batches(), vec![2]);
        assert_eq!(model.text_prefill_batches(), vec![2]);
        assert_eq!(events.len(), 2);
        assert_eq!((events[0].id, events[0].token), (id_0, 3));
        assert_eq!((events[1].id, events[1].token), (id_2, 4));
    }

    #[test]
    fn prefill_admitted_single_text_row_uses_forward_on_fast_path() {
        let mut s = Scheduler::<RecordingPrefillModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1, 2, 3])).expect("admit");

        let model = RecordingPrefillModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.make_cache_batches(), vec![1]);
        assert_eq!(model.text_hidden_shapes(), vec![(1, 2)]);
        assert_eq!(model.text_forward_batches(), vec![1]);
        assert_eq!(model.text_forward_seq_lens(), vec![1]);
        assert_eq!(model.text_forward_masks(), vec![(false, false)]);
        assert_eq!(model.text_prefill_batches(), Vec::<i32>::new());
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].id, events[0].token), (id, 3));
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_enables_turboquant_cache_from_request() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let mut req = mk_req(vec![1, 2, 3]);
        req.kv_cache_turboquant_bits = Some(crate::core::cache::TurboQuantKVBits::K3V4);
        let id = s.admit(req).expect("admit");

        let model = StepDecodeMaskModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, id);
        let cache = s.cache.as_ref().expect("scheduler cache");
        match &cache[0] {
            LayerCache::Full(kv) => {
                let tq = kv.turboquant().expect("turboquant cache");
                assert_eq!(tq.bits(), crate::core::cache::TurboQuantKVBits::K3V4);
                assert_eq!(tq.key_bits(), 3);
                assert_eq!(tq.value_bits(), 4);
            }
            _ => panic!("expected full-attention cache"),
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_uses_paged_ssd_prefix_cache_on_exact_hit() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-scheduler-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "step-test", 2, 32)
            .expect("prefix config");

        let mut warm = Scheduler::<StepDecodeMaskModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        warm.enable_paged_prefix_cache(config.clone())
            .expect("enable prefix cache");
        warm.admit(mk_req(vec![1, 2, 3, 4])).expect("admit warm");
        let warm_model = StepDecodeMaskModel::default();
        warm.prefill_admitted(&warm_model).expect("warm prefill");
        assert_eq!(warm_model.hidden_seq_lens(), vec![3]);
        assert_eq!(warm_model.forward_seq_lens(), vec![1]);
        assert!(
            std::fs::read_dir(&root)
                .expect("prefix cache dir")
                .next()
                .is_some(),
            "warm prefill should save a prefix cache entry"
        );

        let mut hit = Scheduler::<StepDecodeMaskModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        hit.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        hit.admit(mk_req(vec![1, 2, 3, 4])).expect("admit hit");
        let hit_model = StepDecodeMaskModel::default();
        hit.prefill_admitted(&hit_model).expect("hit prefill");

        assert_eq!(
            hit_model.hidden_seq_lens(),
            Vec::<i32>::new(),
            "exact cache hit should restore prompt prefix instead of recomputing it"
        );
        assert_eq!(hit_model.forward_seq_lens(), vec![1]);
        match &hit.cache.as_ref().expect("cache")[0] {
            LayerCache::Full(kv) => assert!(kv.paged().is_some()),
            _ => panic!("expected full-attention cache"),
        }

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_uses_prefix_lru_cache_when_ssd_entry_is_gone() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-prefix-lru-scheduler-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "step-test", 2, 32)
            .expect("prefix config");

        let mut scheduler = Scheduler::<StepDecodeMaskModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        scheduler
            .enable_paged_prefix_cache(config)
            .expect("enable paged prefix cache");
        scheduler
            .enable_prefix_lru_cache(
                crate::core::cache::PrefixLruCacheConfig::new(1024 * 1024).expect("L1 config"),
            )
            .expect("enable prefix LRU cache");

        scheduler
            .admit(mk_req(vec![1, 2, 3, 4]))
            .expect("admit warm");
        let warm_model = StepDecodeMaskModel::default();
        scheduler
            .prefill_admitted(&warm_model)
            .expect("warm prefill");
        assert_eq!(warm_model.hidden_seq_lens(), vec![3]);
        assert_eq!(warm_model.forward_seq_lens(), vec![1]);
        assert!(
            scheduler
                .prefix_lru_cache
                .as_ref()
                .expect("L1 cache")
                .lock()
                .expect("L1 lock")
                .len()
                > 0,
            "warm prefill should populate L1"
        );

        scheduler.evict_all().expect("clear warm request");
        std::fs::remove_dir_all(&root).expect("remove SSD cache");
        scheduler
            .admit(mk_req(vec![1, 2, 3, 4]))
            .expect("admit L1 hit");
        let hit_model = StepDecodeMaskModel::default();
        scheduler
            .prefill_admitted(&hit_model)
            .expect("L1 hit prefill");

        assert_eq!(
            hit_model.hidden_seq_lens(),
            Vec::<i32>::new(),
            "L1 exact hit should restore prompt prefix without recomputing it"
        );
        assert_eq!(hit_model.forward_seq_lens(), vec![1]);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_batched_text_uses_paged_ssd_prefix_cache_on_exact_hits() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-batch-text-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "step-test", 2, 32)
            .expect("prefix config");
        let prompts = [vec![1, 2, 3, 4], vec![5, 6, 7, 8]];

        for prompt in prompts.iter() {
            let mut warm = Scheduler::<StepDecodeMaskModel>::new(
                1,
                32768,
                crate::core::memory_budget::test_meta_qwen35(),
            )
            .expect("scheduler startup");
            warm.enable_paged_prefix_cache(config.clone())
                .expect("enable prefix cache");
            warm.admit(mk_req(prompt.clone())).expect("admit warm");
            let warm_model = StepDecodeMaskModel::default();
            warm.prefill_admitted(&warm_model).expect("warm prefill");
        }

        let mut hit = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        hit.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let id_0 = hit.admit(mk_req(prompts[0].clone())).expect("admit hit 0");
        let id_1 = hit.admit(mk_req(prompts[1].clone())).expect("admit hit 1");
        let hit_model = StepDecodeMaskModel::default();
        let events = hit.prefill_admitted(&hit_model).expect("hit prefill");

        assert_eq!(
            hit_model.batched_prefill_batches(),
            Vec::<i32>::new(),
            "fully hit batch should not recompute full prompts through batched_prefill"
        );
        assert_eq!(hit_model.batched_prefill_lens(), Vec::<Vec<i32>>::new());
        assert_eq!(
            hit_model.hidden_seq_lens(),
            Vec::<i32>::new(),
            "fully hit batch should restore prefixes instead of recomputing them"
        );
        assert_eq!(hit_model.forward_seq_lens(), vec![1]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, id_0);
        assert_eq!(events[1].id, id_1);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_batched_text_paged_prefix_cold_miss_uses_batched_prefill() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-batch-text-cold-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "step-test", 2, 32)
            .expect("prefix config");

        let mut scheduler = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        scheduler
            .enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let id_0 = scheduler.admit(mk_req(vec![1, 2, 3, 4])).expect("admit 0");
        let id_1 = scheduler.admit(mk_req(vec![5, 6, 7, 8])).expect("admit 1");
        let model = StepDecodeMaskModel::default();
        let events = scheduler.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.batched_prefill_batches(), vec![2]);
        assert_eq!(model.batched_prefill_lens(), vec![vec![3, 3]]);
        assert_eq!(
            model.forward_seq_lens(),
            vec![1],
            "cold paged-prefix batch should only decode the final token after batched prefix prefill"
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, id_0);
        assert_eq!(events[1].id, id_1);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    fn prefix_restore_groups_coalesce_identical_text_prompts() {
        let prompts = vec![
            vec![1_u32, 2, 3, 4],
            vec![9_u32, 8, 7],
            vec![1_u32, 2, 3, 4],
            vec![9_u32, 8, 7],
        ];

        assert_eq!(
            prefix_restore_groups(&prompts, None).expect("groups"),
            vec![vec![0, 2], vec![1, 3]]
        );
    }

    #[test]
    fn prefix_restore_groups_keep_different_vl_fingerprints_apart() {
        let prompts = vec![
            vec![1_u32, 2, 3],
            vec![1_u32, 2, 3],
            vec![1_u32, 2, 3],
            vec![4_u32, 5, 6],
        ];
        let fingerprints = vec![
            Some("vl:a".to_owned()),
            Some("vl:b".to_owned()),
            Some("vl:a".to_owned()),
            Some("vl:a".to_owned()),
        ];

        assert_eq!(
            prefix_restore_groups(&prompts, Some(&fingerprints)).expect("groups"),
            vec![vec![0, 2], vec![1], vec![3]]
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_batched_vl_uses_fingerprinted_paged_ssd_prefix_cache_on_exact_hits() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use mlx::Dtype;

        fn vl_req_with_pixel(value: f32) -> GenerateRequest {
            let pixel_values: mlx::Array = (&[value; 4][..], &[1_i32, 4][..])
                .try_into()
                .expect("pixel_values");
            let pixel_values = mlx::ops::astype(&pixel_values, Dtype::Bfloat16).unwrap();
            let mut vl_req = mk_req(vec![1, IMAGE_TOKEN_ID as u32, 2]);
            vl_req.pixel_values = Some(vec![pixel_values]);
            vl_req.image_grid_thw = Some(vec![(1_i32, 2_i32, 2_i32)]);
            vl_req.image_spatial_merge_size = 2;
            vl_req.image_token_id = IMAGE_TOKEN_ID;
            vl_req
        }

        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-batch-vl-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "vl-test", 2, 32)
            .expect("prefix config");

        for pixel in [0.0_f32, 1.0_f32] {
            let mut warm = Scheduler::<StepDecodeMaskModel>::new(
                1,
                32768,
                crate::core::memory_budget::test_meta_qwen35(),
            )
            .expect("scheduler startup");
            warm.enable_paged_prefix_cache(config.clone())
                .expect("enable prefix cache");
            warm.admit(vl_req_with_pixel(pixel)).expect("admit warm");
            let warm_model = StepDecodeMaskModel::default();
            warm.prefill_admitted(&warm_model).expect("warm prefill");
        }

        let mut hit = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        hit.enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        hit.admit(vl_req_with_pixel(0.0)).expect("admit hit 0");
        hit.admit(vl_req_with_pixel(1.0)).expect("admit hit 1");
        let hit_model = StepDecodeMaskModel::default();
        hit.prefill_admitted(&hit_model).expect("hit prefill");

        assert_eq!(
            hit_model.batched_prefill_batches(),
            Vec::<i32>::new(),
            "fully hit VL batch should not recompute full prompts through batched_prefill_vl"
        );
        assert_eq!(
            hit_model.hidden_seq_lens(),
            Vec::<i32>::new(),
            "fully hit VL batch should restore prefixes instead of recomputing them"
        );
        assert_eq!(hit_model.forward_seq_lens(), vec![1]);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefill_admitted_batched_vl_paged_prefix_cold_miss_uses_batched_prefill() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use mlx::Dtype;

        fn vl_req_with_pixel(value: f32) -> GenerateRequest {
            let pixel_values: mlx::Array = (&[value; 4][..], &[1_i32, 4][..])
                .try_into()
                .expect("pixel_values");
            let pixel_values = mlx::ops::astype(&pixel_values, Dtype::Bfloat16).unwrap();
            let mut vl_req = mk_req(vec![1, IMAGE_TOKEN_ID as u32, 2]);
            vl_req.pixel_values = Some(vec![pixel_values]);
            vl_req.image_grid_thw = Some(vec![(1_i32, 2_i32, 2_i32)]);
            vl_req.image_spatial_merge_size = 2;
            vl_req.image_token_id = IMAGE_TOKEN_ID;
            vl_req
        }

        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-batch-vl-cold-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "vl-test", 2, 32)
            .expect("prefix config");

        let mut scheduler = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        scheduler
            .enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        let id_0 = scheduler.admit(vl_req_with_pixel(0.0)).expect("admit vl 0");
        let id_1 = scheduler.admit(vl_req_with_pixel(1.0)).expect("admit vl 1");
        let model = StepDecodeMaskModel::default();
        let events = scheduler.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.batched_prefill_batches(), vec![2]);
        assert_eq!(model.batched_prefill_lens(), vec![vec![2, 2]]);
        assert_eq!(
            model.hidden_seq_lens(),
            Vec::<i32>::new(),
            "cold paged-prefix VL batch should use batched_prefill_vl, not per-row hidden replay"
        );
        assert_eq!(
            model.forward_seq_lens(),
            vec![1],
            "cold paged-prefix VL batch should only decode the final token after batched prefix prefill"
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, id_0);
        assert_eq!(events[1].id, id_1);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[test]
    #[serial(mlx_metal)]
    fn admit_mid_vl_saves_fingerprinted_paged_prefix_cache() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use mlx::Dtype;

        fn vl_req_with_pixel(value: f32) -> GenerateRequest {
            let pixel_values: mlx::Array = (&[value; 4][..], &[1_i32, 4][..])
                .try_into()
                .expect("pixel_values");
            let pixel_values = mlx::ops::astype(&pixel_values, Dtype::Bfloat16).unwrap();
            let mut vl_req = mk_req(vec![1, IMAGE_TOKEN_ID as u32, 2]);
            vl_req.pixel_values = Some(vec![pixel_values]);
            vl_req.image_grid_thw = Some(vec![(1_i32, 2_i32, 2_i32)]);
            vl_req.image_spatial_merge_size = 2;
            vl_req.image_token_id = IMAGE_TOKEN_ID;
            vl_req
        }

        let root = std::env::temp_dir().join(format!(
            "ironmlx-paged-prefix-vl-mid-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config = crate::core::cache::PagedPrefixCacheConfig::new(&root, "vl-test", 2, 32)
            .expect("prefix config");

        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.enable_paged_prefix_cache(config.clone())
            .expect("enable prefix cache");

        s.admit(mk_req(vec![1, 2, 3])).expect("admit active");
        let model = StepDecodeMaskModel::default();
        s.prefill_admitted(&model).expect("prefill active");
        let _ = std::fs::remove_dir_all(&root);

        let mut handle = s
            .admit_mid_begin(vl_req_with_pixel(0.0), &model)
            .expect("admit_mid_begin");
        assert!(!s
            .admit_mid_chunk(&mut handle, &model)
            .expect("VL prefix mid-admit chunk"));
        assert!(s
            .admit_mid_chunk(&mut handle, &model)
            .expect("VL final mid-admit chunk"));
        s.admit_mid_finalize(handle, &model)
            .expect("finalize VL mid-admit");

        let entry_count = std::fs::read_dir(&root)
            .map(|entries| entries.count())
            .unwrap_or(0);
        assert!(entry_count > 0, "VL mid-admit should write prefix entries");

        let mut same_image_hit = Scheduler::<StepDecodeMaskModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        same_image_hit
            .enable_paged_prefix_cache(config.clone())
            .expect("enable prefix cache");
        same_image_hit
            .admit(vl_req_with_pixel(0.0))
            .expect("admit same-image hit");
        let same_image_model = StepDecodeMaskModel::default();
        same_image_hit
            .prefill_admitted(&same_image_model)
            .expect("same-image prefill");
        assert_eq!(
            same_image_model.hidden_seq_lens(),
            Vec::<i32>::new(),
            "same VL image should restore the cached prefix"
        );
        assert_eq!(same_image_model.forward_seq_lens(), vec![1]);

        let mut different_image_miss = Scheduler::<StepDecodeMaskModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        different_image_miss
            .enable_paged_prefix_cache(config)
            .expect("enable prefix cache");
        different_image_miss
            .admit(vl_req_with_pixel(1.0))
            .expect("admit different-image miss");
        let different_image_model = StepDecodeMaskModel::default();
        different_image_miss
            .prefill_admitted(&different_image_model)
            .expect("different-image prefill");
        assert_eq!(
            different_image_model.hidden_seq_lens(),
            vec![2],
            "different VL image must miss despite identical token ids"
        );
        assert_eq!(different_image_model.forward_seq_lens(), vec![1]);

        std::fs::remove_dir_all(root).expect("cleanup prefix cache");
    }

    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn prefill_admitted_rejects_mixed_turboquant_kv_configs() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let mut req_k3v4 = mk_req(vec![1, 2, 3]);
        req_k3v4.kv_cache_turboquant_bits = Some(crate::core::cache::TurboQuantKVBits::K3V4);
        let mut req_k4v4 = mk_req(vec![4, 5, 6]);
        req_k4v4.kv_cache_turboquant_bits = Some(crate::core::cache::TurboQuantKVBits::K4V4);
        s.admit(req_k3v4).expect("admit K3V4");
        s.admit(req_k4v4).expect("admit K4V4");

        let model = StepDecodeMaskModel::default();
        let err = s
            .prefill_admitted(&model)
            .expect_err("mixed TurboQuant KV configs should fail");

        assert!(err
            .to_string()
            .contains("scheduler batch mixes TurboQuant KV configs"));
    }

    // Multi-row prefill test (2 active rows) — incompatible with the
    // p5h-profile single-row invariant (see
    // prefill_admitted_compacts_sparse_initial_rows). Default-feature only.
    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn prefill_admitted_uses_full_batch_when_all_rows_active() {
        let mut s = Scheduler::<RecordingPrefillModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id_0 = s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![4, 5, 6])).expect("admit 1");

        let model = RecordingPrefillModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.make_cache_batches(), vec![2]);
        assert_eq!(model.text_prefill_batches(), vec![2]);
        assert_eq!(events.len(), 2);
        assert_eq!((events[0].id, events[0].token), (id_0, 3));
        assert_eq!((events[1].id, events[1].token), (id_1, 4));
    }

    #[test]
    fn step_with_empty_scheduler_slot_decodes_only_compact_active_row() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1, 2, 3])).expect("admit");

        let model = StepDecodeMaskModel::default();
        let prefill_events = s.prefill_admitted(&model).expect("prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(prefill_events[0].id, id);
        assert_eq!(s.phase(), Phase::Decoding);

        let step_events = s
            .step(&model)
            .expect("step must tolerate empty non-active scheduler slots");
        assert_eq!(step_events.len(), 1);
        assert_eq!(step_events[0].id, id);
        assert_eq!(model.decode_lens_seen(), vec![vec![1]]);
    }

    #[test]
    fn step_single_active_row_with_larger_bmax_decodes_compact_b1() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let id = s.admit(mk_req(vec![1, 2, 3])).expect("admit");

        let model = StepDecodeMaskModel::default();
        let prefill_events = s.prefill_admitted(&model).expect("prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(prefill_events[0].id, id);
        assert_eq!(s.phase(), Phase::Decoding);

        let step_events = s.step(&model).expect("step");
        assert_eq!(step_events.len(), 1);
        assert_eq!(step_events[0].id, id);
        assert_eq!(model.decode_lens_seen(), vec![vec![1]]);
    }

    // Multi-row test: prefills 2 active rows (then exercises mid-admit stale
    // slot reuse). The initial `prefill_admitted` with 2 active rows trips the
    // p5h-profile single-row invariant (see
    // prefill_admitted_compacts_sparse_initial_rows). Default-feature only.
    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn admit_mid_finalize_replaces_stale_cache_row_when_slot_reused() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let _id_0 = s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![4, 5, 6])).expect("admit 1");

        let model = StepDecodeMaskModel::default();
        s.prefill_admitted(&model).expect("prefill");
        s.get_mut(id_1).expect("row 1").finished = true;

        let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
        let evicted = s.gc_finished_rows(&mut event_txs);
        assert_eq!(evicted, vec![id_1]);

        let mut mid_req = mk_req(vec![7, 8, 9]);
        mid_req.prefill_chunk_size = 512;
        let mut handle = s
            .admit_mid_begin(mid_req, &model)
            .expect("admit_mid_begin should reuse stale slot row");
        assert_eq!(handle.row_idx, 1);
        assert!(s
            .admit_mid_chunk(&mut handle, &model)
            .expect("single chunk"));
        let (_id, event) = s
            .admit_mid_finalize(handle, &model)
            .expect("finalize should replace stale cache row");
        assert_eq!(event.finish_reason, None);

        let step_events = s.step(&model).expect("step after stale-row reuse");
        assert_eq!(step_events.len(), 2);
        assert_eq!(model.decode_lens_seen(), vec![vec![1, 1]]);
    }

    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn step_uniform_decode_lengths_omits_all_zero_decode_mask() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        s.admit(mk_req(vec![4, 5, 6])).expect("admit 1");

        let model = StepDecodeMaskModel::default();
        s.prefill_admitted(&model).expect("prefill");

        let step_events = s.step(&model).expect("step");
        assert_eq!(step_events.len(), 2);
        assert_eq!(model.decode_lens_seen(), vec![vec![1, 1]]);
        assert_eq!(model.decode_mask_seen(), vec![false]);
    }

    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn step_ragged_decode_lengths_keeps_decode_mask() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");
        s.admit(mk_req(vec![4, 5, 6, 7])).expect("admit 1");

        let model = StepDecodeMaskModel::default();
        s.prefill_admitted(&model).expect("prefill");

        let step_events = s.step(&model).expect("step");
        assert_eq!(step_events.len(), 2);
        assert_eq!(model.decode_lens_seen(), vec![vec![1, 1]]);
        assert_eq!(model.decode_mask_seen(), vec![true]);
    }

    #[test]
    fn admit_mid_prefill_chunk_size_zero_disables_chunking() {
        let mut s = Scheduler::<StepDecodeMaskModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let _id_0 = s.admit(mk_req(vec![1, 2, 3])).expect("admit 0");

        let model = StepDecodeMaskModel::default();
        s.prefill_admitted(&model).expect("prefill");

        let mid_req = mk_req(vec![7, 8, 9, 10]);
        assert_eq!(mid_req.prefill_chunk_size, 0);
        let mut handle = s.admit_mid_begin(mid_req, &model).expect("admit_mid_begin");

        assert_eq!(handle.chunk_size, handle.prompt_len);
        assert!(s
            .admit_mid_chunk(&mut handle, &model)
            .expect("0 disables chunking, so first chunk is last"));
    }

    #[test]
    fn prefill_admitted_single_vl_row_splits_prefix_and_last_token() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use mlx::Dtype;

        let mut s = Scheduler::<RecordingPrefillModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let pixel_values: mlx::Array = (&[0.0_f32; 4][..], &[1_i32, 4][..])
            .try_into()
            .expect("pixel_values");
        let pixel_values = mlx::ops::astype(&pixel_values, Dtype::Bfloat16).unwrap();
        let req = GenerateRequest {
            prompt_ids: vec![1, IMAGE_TOKEN_ID as u32, 2],
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: Some(vec![pixel_values]),
            image_grid_thw: Some(vec![(1_i32, 2_i32, 2_i32)]),
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };
        let id = s.admit(req).expect("admit vl");

        let model = RecordingPrefillModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.make_cache_batches(), vec![1]);
        assert_eq!(model.vision_grid_lens(), vec![1]);
        assert_eq!(model.vl_hidden_shapes(), vec![(1, 2)]);
        assert_eq!(model.vl_hidden_vision_present(), vec![true]);
        assert_eq!(model.vl_chunk_batches(), vec![1]);
        assert_eq!(model.vl_chunk_vision_present(), vec![false]);
        assert_eq!(model.vl_prefill_batches(), Vec::<i32>::new());
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].id, events[0].token), (id, 3));
    }

    #[test]
    fn prefill_admitted_single_vl_row_preserves_multi_image_grids() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use mlx::Dtype;

        let mut s = Scheduler::<RecordingPrefillModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let pixel_values: mlx::Array = (&[0.0_f32; 8][..], &[2_i32, 4][..])
            .try_into()
            .expect("pixel_values");
        let pixel_values = mlx::ops::astype(&pixel_values, Dtype::Bfloat16).unwrap();
        let req = GenerateRequest {
            prompt_ids: vec![1, IMAGE_TOKEN_ID as u32, 2, IMAGE_TOKEN_ID as u32, 3],
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: Some(vec![pixel_values]),
            image_grid_thw: Some(vec![(1_i32, 2_i32, 2_i32), (1_i32, 2_i32, 2_i32)]),
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };
        let id = s.admit(req).expect("admit vl");

        let model = RecordingPrefillModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.make_cache_batches(), vec![1]);
        assert_eq!(model.vision_grid_lens(), vec![2]);
        assert_eq!(model.vl_hidden_shapes(), vec![(1, 4)]);
        assert_eq!(model.vl_hidden_vision_present(), vec![true]);
        assert_eq!(model.vl_chunk_batches(), vec![1]);
        assert_eq!(model.vl_chunk_vision_present(), vec![false]);
        assert_eq!(model.vl_prefill_batches(), Vec::<i32>::new());
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].id, events[0].token), (id, 3));
    }

    // Multi-row VL prefill test (2 active VL rows) — incompatible with the
    // p5h-profile single-row invariant (see
    // prefill_admitted_compacts_sparse_initial_rows). Default-feature only.
    #[cfg(not(feature = "p5h-profile"))]
    #[test]
    fn prefill_admitted_compacts_vl_rows() {
        use crate::core::generate::IMAGE_TOKEN_ID;
        use mlx::Dtype;

        let mut s = Scheduler::<RecordingPrefillModel>::new(
            3,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        let pixel_values: mlx::Array = (&[0.0_f32; 4][..], &[1_i32, 4][..])
            .try_into()
            .expect("pixel_values");
        let pixel_values = mlx::ops::astype(&pixel_values, Dtype::Bfloat16).unwrap();
        let mk_vl_req = |pixel_values: mlx::Array| GenerateRequest {
            prompt_ids: vec![1, IMAGE_TOKEN_ID as u32, 2],
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: Some(vec![pixel_values]),
            image_grid_thw: Some(vec![(1_i32, 2_i32, 2_i32)]),
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        };
        let id_0 = s
            .admit(mk_vl_req(pixel_values.clone()))
            .expect("admit vl 0");
        let id_1 = s.admit(mk_vl_req(pixel_values)).expect("admit vl 1");

        let model = RecordingPrefillModel::default();
        let events = s.prefill_admitted(&model).expect("prefill");

        assert_eq!(model.make_cache_batches(), vec![2]);
        assert_eq!(model.vl_prefill_batches(), vec![2]);
        assert_eq!(model.vl_pixel_value_lens(), vec![2]);
        assert_eq!(events.len(), 2);
        assert_eq!((events[0].id, events[0].token), (id_0, 3));
        assert_eq!((events[1].id, events[1].token), (id_1, 4));
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
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: Some(vec![pv_bf16]),
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
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
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
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
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
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
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

    /// P5h+2.c regression: `max_new_tokens=1` requests must transition
    /// scheduler to `Phase::Finished` after `prefill_admitted` (the first
    /// sampled token is also the last token → request finished → no
    /// `any_unfinished` → phase = Finished per scheduler.rs:1247-1250).
    ///
    /// This locks the spec invariant that the actor's
    /// `finalize_finished_batch_if_any` helper relies on.
    #[test]
    fn test_max_new_tokens_1_transitions_to_finished_after_prefill() {
        let model = P5h2cFakeModel;
        let mut s = Scheduler::<P5h2cFakeModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");

        let mut req = mk_req(vec![1, 2, 3, 4]);
        req.max_new_tokens = 1;
        req.stop_token_ids = vec![];

        let id = s.admit(req).expect("admit OK");
        assert!(matches!(s.phase(), Phase::Admitting | Phase::Idle));
        let events = s.prefill_admitted(&model).expect("prefill OK");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, id);
        assert_eq!(events[0].finish_reason, Some("length"));
        assert_eq!(
            s.phase(),
            Phase::Finished,
            "max_new_tokens=1 should transition to Finished after prefill"
        );
    }

    /// P5h+2.c regression: `step` MUST still raise an Err in
    /// `Phase::Finished` to preserve fail-fast discipline. The actor-side
    /// fix in P5h+2.c works AROUND this guard (via pre-event finalization)
    /// rather than relaxing it. If this test ever passes by returning Ok,
    /// the scheduler core semantics were silently changed.
    #[test]
    fn test_step_finished_phase_still_returns_err() {
        let model = P5h2cFakeModel;
        let mut s = Scheduler::<P5h2cFakeModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        // Force scheduler into Finished phase (use existing test seam).
        s.force_phase(Phase::Finished);
        let result = s.step(&model);
        assert!(result.is_err(), "step in Phase::Finished must return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("step illegal in Finished phase"),
            "expected `step illegal in Finished phase` in error, got: {err_msg}"
        );
    }

    #[test]
    fn test_step_only_mid_admit_reserved_rows_is_noop() {
        let model = P5h2cFakeModel;
        let mut s = Scheduler::<P5h2cFakeModel>::new(
            2,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");

        s.force_phase(Phase::Decoding);
        let id = s
            .admit(mk_req(vec![1, 2, 3, 4]))
            .expect("mid-admit reserve");

        let events = s.step(&model).expect("reserved-only step should noop");
        assert!(events.is_empty());
        assert_eq!(s.phase(), Phase::Decoding);
        assert!(
            s.get(id).unwrap().generated_tokens.is_empty(),
            "reserved row must not receive a generated token before admit_mid_finalize"
        );
    }
}
