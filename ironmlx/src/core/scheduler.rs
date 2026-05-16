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

use anyhow::{anyhow, Result};
use mlx::{Array, Dtype};
use thiserror::Error;
use tokio::sync::mpsc;

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
}

use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_decode_position_ids,
    build_per_row_decode_mask, build_position_ids_batched, build_position_ids_vl_batched,
    slice_logits_row, GenerateRequest,
};
use crate::core::sampler::Sampler;
use crate::models::qwen3_5::Qwen35Model;
use crate::nn::LayerCache;

/// Convenience alias — avoids `clippy::type_complexity` on Vec<Option<&[...]>> sites.
type GridThwSlice<'a> = Option<&'a [(i32, i32, i32)]>;

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
    /// each row owns independent sampler state (the `Cell` inside `Sampler`
    /// requires per-row independence — see `core/sampler.rs:43`).
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
pub struct Scheduler {
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
}

impl std::fmt::Debug for Scheduler {
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

impl Scheduler {
    /// Construct a scheduler with `b_max` pre-allocated slots, all `None`.
    /// `effective_cap_max` is the hard upper bound on per-request
    /// `prompt_len + max_new_tokens` — admit gates reject requests beyond
    /// this with [`SchedulerError::RequestTooLarge`] (HTTP 413 downstream).
    pub fn new(b_max: usize, effective_cap_max: usize) -> Self {
        let mut slots = Vec::with_capacity(b_max);
        for _ in 0..b_max {
            slots.push(None);
        }
        Self {
            b_max,
            slots,
            next_id: 0,
            phase: Phase::Idle,
            cache: None,
            poisoned: false,
            effective_cap_max,
        }
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
        };
        self.slots[row_idx] = Some(state);
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
        self.slots[row_idx] = None;
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
            *slot = None;
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
    /// Lazy-allocates the batched KV cache on first call (`b_max` rows,
    /// capacity 8192, bf16). On subsequent calls (after `evict_all`) the
    /// cache is reused — `evict_all` already reset every layer.
    ///
    /// Builds right-padded `[B, T_max]` input_ids + `[3, B, T_max]`
    /// position_ids + `[B, 1, T_max, T_max]` attention mask + `[B, T_max]`
    /// linear mask, then calls `Qwen35Model::batched_prefill`.
    ///
    /// Samples the first token per occupied row from the prefill logits
    /// (`batched_prefill` already collapses per-row to the last real position
    /// `prompt_lens[i] - 1`, returning `[B, 1, vocab]`). Emits a
    /// [`StepEvent`] per row, then transitions to `Decoding` (or `Finished`
    /// if every row's first token was EOS). This keeps the KV cache
    /// trajectory aligned with `GenerationStream`'s pipelined-mode which also
    /// uses the prefill argmax as `token_0`. See spec §4.5.
    pub fn prefill_admitted(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>> {
        self.ensure_not_poisoned()?;
        match self.prefill_admitted_inner(model) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn prefill_admitted_inner(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>> {
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

        // Lazy-allocate the cache (or reuse the existing one — Task 1's
        // evict_all already reset every layer to offset 0).
        // TODO: when a non-bf16 model lands, expose dtype via Qwen35Model
        // accessor and thread it here.
        if self.cache.is_none() {
            self.cache = Some(model.make_cache(b as i32, 8192, Dtype::Bfloat16)?);
        }
        let cache_ref = self
            .cache
            .as_mut()
            .ok_or_else(|| anyhow!("cache missing after lazy-alloc — internal bug"))?;

        // Run batched prefill. Capture [B, 1, vocab] logits (sequence axis
        // already collapsed via slice_last_and_project) for first-token
        // sampling.
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
                (),
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
                (),
            )?
        };

        // After per-row prefill, row i's cache is filled up to position
        // prompt_lens[i] - 1. The first decode step must use position
        // prompt_lens[i] for that row.
        for (slot, &plen) in self.slots.iter_mut().zip(prompt_lens.iter()) {
            if let Some(state) = slot.as_mut() {
                state.real_len = plen;
            }
        }

        // Sample first token per occupied row from logits[:, 0, :].
        // batched_prefill returns [B, 1, vocab] with the sequence axis already
        // collapsed internally. Use slice_logits_row (same helper as step_inner).
        let mut events: Vec<StepEvent> = Vec::new();
        for b_idx in 0..b {
            let was_active = self.slots[b_idx].is_some();
            if !was_active {
                continue;
            }
            let row_flat = slice_logits_row(&logits, b_idx).map_err(|e| {
                anyhow!("prefill_admitted: slice_logits_row(row {b_idx}) failed: {e:?}")
            })?;

            let state = self.slots[b_idx]
                .as_mut()
                .expect("was_active guaranteed Some");

            let history: Vec<u32> = state.prompt_ids.clone();
            let token = state.sampler.sample(&row_flat, &history)?;

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
    /// Packs `[B, 1]` input_ids (each row's last token; pad zero for
    /// already-finished rows and for empty slots), builds per-row decode
    /// position ids `[3, B, 1]`, calls `Qwen35Model::forward_on`, then
    /// loops over rows: slices `logits[b, 0, :]`, samples via
    /// `RequestState::sampler.sample`, pushes the token, advances
    /// `real_len`, and checks for EOS / `max_new_tokens` termination.
    ///
    /// Returns events **only** for rows that were not yet finished at the
    /// start of this step. Rows that transition to `finished` during this
    /// step appear once (with `finish_reason = Some(...)`); rows that were
    /// already finished are silently skipped.
    ///
    /// Transitions phase to `Finished` when every active row has
    /// `finished == true`.
    ///
    /// Note: already-finished rows are still padded into the forward
    /// (lockstep cost — see spec §7). Only active-at-start rows contribute
    /// to the returned event list.
    pub fn step(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>> {
        self.ensure_not_poisoned()?;
        match self.step_inner(model) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn step_inner(&mut self, model: &Qwen35Model) -> Result<Vec<StepEvent>> {
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "step illegal in {:?} phase: call prefill_admitted first",
                self.phase
            ));
        }

        let b = self.b_max;

        // Capture which rows were not-yet-finished at the start of this
        // step. Only these rows participate in sampling and in the event
        // list. Already-finished rows are still padded into the forward
        // (lockstep cost — see spec §7).
        let active_at_start: Vec<bool> = self
            .slots
            .iter()
            .map(|s| matches!(s, Some(r) if !r.finished))
            .collect();

        // Build [B, 1] input_ids in slot order.
        // - For active rows: last generated token (prefill_admitted always pushes
        //   ≥1 token before the first step call, so generated_tokens is non-empty).
        // - For already-finished rows or empty slots: pad 0.
        let last_tokens: Vec<i32> = self
            .slots
            .iter()
            .map(|slot| match slot {
                Some(r) if !r.finished => {
                    let tok = *r
                        .generated_tokens
                        .last()
                        .expect("prefill_admitted always pushes ≥ 1 token before step");
                    tok as i32
                }
                _ => 0,
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
            .map(|s| match s {
                Some(r) if !r.finished => r.real_len,
                _ => 0,
            })
            .collect();
        let position_ids = build_decode_position_ids(&per_row_pos)?;

        // Per-row lens for decode: each active row writes 1 token; pad
        // rows (finished or None slots) write 0 to skip the K/V write.
        let per_row_lens: Vec<i32> = self
            .slots
            .iter()
            .map(|s| match s {
                Some(r) if !r.finished => 1,
                _ => 0,
            })
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
            (),
        )?;

        // logits shape: [B, 1, vocab]
        let mut events: Vec<StepEvent> = Vec::new();
        for (b_idx, was_active) in active_at_start.iter().enumerate() {
            if !was_active {
                continue;
            }
            let row_flat = slice_logits_row(&logits, b_idx)
                .map_err(|e| anyhow!("step: slice_logits_row(row {b_idx}) failed: {e:?}"))?;

            let state = self.slots[b_idx]
                .as_mut()
                .expect("active_at_start guaranteed Some");

            // Per-row sampler invocation. The sampler.history is the union
            // of prompt_ids and generated_tokens so far (so repetition
            // penalty sees both).
            let mut history: Vec<u32> =
                Vec::with_capacity(state.prompt_ids.len() + state.generated_tokens.len());
            history.extend_from_slice(&state.prompt_ids);
            history.extend_from_slice(&state.generated_tokens);
            let token = state.sampler.sample(&row_flat, &history)?;

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

    /// Mid-batch admit + prefill. Caller is `SchedulerActor::driver_loop`
    /// after `cmd_rx` delivers an Admit during the rolling decode loop.
    ///
    /// Architecture: runs prefill in a temporary B=1 cache (the
    /// `GenerationStream`-equivalent path), then adopts the prefilled
    /// row into the main cache via per-layer `adopt_row_from` copies.
    /// This avoids wasted compute on a B=b_max sub-batch + variable-
    /// shape mask construction + GatedDeltaNet state corruption for
    /// other active rows.
    ///
    /// Synchronous: stalls active rows for ~L_new × B=1_prefill_per_
    /// token_time. Adoption cost is sub-microsecond. 3c+ chunked prefill
    /// reduces stall further.
    ///
    /// Returns `(RequestId, StepEvent)` — the assigned request ID and
    /// the first generated token's event. Caller registers the event
    /// channel using the returned `id`.
    pub fn admit_mid(
        &mut self,
        req: GenerateRequest,
        model: &Qwen35Model,
    ) -> Result<(RequestId, StepEvent)> {
        self.ensure_not_poisoned()?;
        // B1-p2.3f: mirror admit's cap gate. Mid-batch admits must also
        // respect the bound — otherwise the queue drain path could push an
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
                "admit_mid illegal in {:?} phase: only Decoding (use admit for Idle/Admitting)",
                self.phase
            ));
        }
        let row_idx =
            self.slots.iter().position(|s| s.is_none()).ok_or_else(|| {
                anyhow!("scheduler full: no row available (b_max={})", self.b_max)
            })?;

        // 1. Insert RequestState via the relaxed admit() path. Phase stays Decoding.
        let id = self.admit(req)?;

        // Steps 2-8: prefill into temp cache, adopt, sample, update.
        // If anything fails, roll back by evicting the orphan slot —
        // otherwise next step() would panic on empty generated_tokens.
        match self.admit_mid_inner(id, row_idx, model) {
            Ok(event) => Ok((id, event)),
            Err(e) => {
                // Rollback: evict the orphan slot. evict ignores poison
                // and works in any Phase including Decoding (per Task 3).
                let _ = self.evict(id);
                Err(e)
            }
        }
    }

    /// Inner body of admit_mid (steps 2-8 from the spec). Separated so
    /// `admit_mid` can roll back the inserted slot if any `?` fails.
    fn admit_mid_inner(
        &mut self,
        id: RequestId,
        row_idx: usize,
        model: &Qwen35Model,
    ) -> Result<StepEvent> {
        let (prompt_ids, prompt_len, max_new_tokens) = {
            let state = self.slots[row_idx].as_ref().expect("admit inserted");
            (
                state.prompt_ids.clone(),
                state.prompt_ids.len() as i32,
                state.max_new_tokens,
            )
        };

        // Saturating conversion: max_new_tokens is usize and may exceed
        // i32::MAX in pathological caller inputs. Saturate to i32::MAX so
        // cap_for_temp stays a valid i32 even at the API limit (the
        // actual cap is bounded by model + memory anyway).
        let max_new_i32 = i32::try_from(max_new_tokens).unwrap_or(i32::MAX);
        let cap_for_temp = prompt_len.saturating_add(max_new_i32).max(prompt_len);

        // 2. Capture KVCache dtype from main cache (first Full layer).
        let dtype = {
            let main_cache = self
                .cache
                .as_ref()
                .ok_or_else(|| anyhow!("admit_mid called before prefill_admitted: cache absent"))?;
            main_cache
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(kv) => Some(kv.dtype()),
                    _ => None,
                })
                .unwrap_or(Dtype::Bfloat16)
        };

        // 3. Allocate a fresh B=1 temp cache.
        let mut temp_cache = model.make_cache(1, cap_for_temp, dtype)?;

        // 4. Build B=1 prefill inputs (mirror GenerationStream prefill).
        let input_ids_data: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
        let input_ids: Array = (&input_ids_data[..], &[1_i32, prompt_len][..])
            .try_into()
            .map_err(|e| anyhow!("admit_mid: build input_ids Array failed: {e:?}"))?;

        // B1-p2.4: VL-aware position_ids — VL path uses build_position_ids_vl_batched
        // so MRoPE three-stream values match what forward_vl/batched_prefill_vl expect.
        let (state_pv, state_grids, state_img_token_id, state_merge_size) = {
            let state = self.slots[row_idx].as_ref().expect("admit inserted");
            (
                state.pixel_values.clone(),
                state.image_grid_thw.clone(),
                state.image_token_id,
                state.image_spatial_merge_size,
            )
        };
        let position_ids = if state_pv.is_some() {
            build_position_ids_vl_batched(
                &[&input_ids_data[..]],
                &[state_grids.as_deref()],
                state_img_token_id,
                state_merge_size,
                prompt_len,
            )?
        } else {
            build_position_ids_batched(&[prompt_len], prompt_len)?
        };
        let attention_mask = build_batch_attention_mask(&[prompt_len], prompt_len, dtype)?;
        let linear_attention_mask = build_batch_linear_mask(&[prompt_len], prompt_len)?;

        // 5. Run B=1 prefill into the temp cache. Returns logits [1, 1, vocab].
        let logits = if state_pv.is_some() {
            let per_row_pv: Vec<Option<&Array>> = vec![state_pv.as_ref()];
            let per_row_grids_inner: Vec<GridThwSlice<'_>> = vec![state_grids.as_deref()];
            model.batched_prefill_vl(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_attention_mask,
                &[prompt_len],
                &per_row_pv,
                &per_row_grids_inner,
                state_img_token_id,
                Some(&mut temp_cache),
                (),
            )?
        } else {
            model.batched_prefill(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_attention_mask,
                &[prompt_len],
                Some(&mut temp_cache),
                (),
            )?
        };

        // 6. Adopt the temp cache's row 0 into main_cache at row_idx.
        {
            let main_cache = self.cache.as_mut().expect("cache asserted Some above");
            if main_cache.len() != temp_cache.len() {
                return Err(anyhow!(
                    "admit_mid: cache layer count mismatch ({} vs {})",
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
                    _ => {
                        return Err(anyhow!(
                            "admit_mid: cache layer kind mismatch between main and temp"
                        ))
                    }
                }
            }
        }

        // 7. Sample first token from prefill logits (last position).
        //    Logits shape [1, 1, vocab] -- slice row 0.
        let row_logits = slice_logits_row(&logits, 0)?;
        let token = {
            let state = self.slots[row_idx].as_ref().expect("admit_mid slot");
            let history: Vec<u32> = prompt_ids.clone();
            state.sampler.sample(&row_logits, &history)?
        };

        // 8. Update state + check termination.
        let state = self.slots[row_idx].as_mut().expect("admit_mid slot");
        state.generated_tokens.push(token);
        state.real_len += 1;

        if state.stop_token_ids.contains(&token) {
            state.finished = true;
            state.finish_reason = Some("stop");
        } else if state.generated_tokens.len() >= state.max_new_tokens {
            state.finished = true;
            state.finish_reason = Some("length");
        }

        Ok(StepEvent {
            id,
            token,
            finish_reason: state.finish_reason,
        })
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
            if let Some(state) = slot.as_ref() {
                if state.finished {
                    let id = state.id;
                    event_txs.remove(&id);
                    evicted.push(id);
                    *slot = None;
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn scheduler_new_empty() {
        let s = Scheduler::new(4, 32768);
        assert_eq!(s.b_max(), 4);
        assert_eq!(s.active_count(), 0);
        assert!(s.active().is_empty());
        assert!(s.occupied_rows().is_empty());
    }

    #[test]
    fn admit_happy_path() {
        let mut s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(4, 32768);
        let ids: Vec<_> = (0..4)
            .map(|i| s.admit(mk_req(vec![i as u32])).expect("admit"))
            .collect();
        let rows: Vec<usize> = ids.iter().map(|id| s.get(*id).unwrap().row_idx).collect();
        assert_eq!(rows, vec![0, 1, 2, 3]);
        assert_eq!(s.active_count(), 4);
    }

    #[test]
    fn evict_releases_row() {
        let mut s = Scheduler::new(4, 32768);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.active_count(), 1);
        s.evict(id).expect("evict");
        assert_eq!(s.active_count(), 0);
        assert!(s.get(id).is_none());
    }

    #[test]
    fn admit_after_evict_reuses_row() {
        let mut s = Scheduler::new(4, 32768);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        assert_eq!(s.get(id_a).unwrap().row_idx, 0);
        s.evict(id_a).expect("evict a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");
        assert_eq!(s.get(id_b).unwrap().row_idx, 0); // same slot
        assert_ne!(id_a, id_b); // distinct id
    }

    #[test]
    fn admit_full_returns_err() {
        let mut s = Scheduler::new(2, 32768);
        s.admit(mk_req(vec![1])).expect("admit 0");
        s.admit(mk_req(vec![2])).expect("admit 1");
        let err = s.admit(mk_req(vec![3])).expect_err("admit full");
        let msg = format!("{err}");
        assert!(msg.contains("scheduler full"), "unexpected err: {msg}");
        assert!(msg.contains("b_max=2"), "missing b_max in err: {msg}");
    }

    #[test]
    fn evict_unknown_id_returns_err() {
        let mut s = Scheduler::new(2, 32768);
        let err = s.evict(RequestId(42)).expect_err("evict unknown");
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn id_monotonic_after_evict() {
        let mut s = Scheduler::new(2, 32768);
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
        let mut s = Scheduler::new(2, 32768);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");

        // Distinct `RequestState`s must own distinct Sampler instances at distinct addresses.
        let p_a: *const Sampler = &s.get(id_a).unwrap().sampler;
        let p_b: *const Sampler = &s.get(id_b).unwrap().sampler;
        assert_ne!(p_a, p_b);
    }

    #[test]
    fn occupied_rows_reflects_state() {
        let mut s = Scheduler::new(4, 32768);
        let _id_0 = s.admit(mk_req(vec![1])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![2])).expect("admit 1");
        let _id_2 = s.admit(mk_req(vec![3])).expect("admit 2");
        assert_eq!(s.occupied_rows(), vec![0, 1, 2]);
        s.evict(id_1).expect("evict 1");
        assert_eq!(s.occupied_rows(), vec![0, 2]);
    }

    #[test]
    fn phase_starts_idle() {
        let s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(4, 32768);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn admit_stays_in_admitting() {
        let mut s = Scheduler::new(4, 32768);
        let _ = s.admit(mk_req(vec![1])).expect("admit 1");
        let _ = s.admit(mk_req(vec![2])).expect("admit 2");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn evict_last_admitted_returns_to_idle() {
        let mut s = Scheduler::new(4, 32768);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
        s.evict(id).expect("evict");
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn admit_in_decoding_ok_phase_stays_decoding() {
        // 3c-3: admit during Decoding is now legal (mid-batch admit).
        let mut s = Scheduler::new(4, 32768);
        s.force_phase(Phase::Decoding);
        let id = s
            .admit(mk_req(vec![1]))
            .expect("admit during Decoding must succeed");
        assert_eq!(s.phase(), Phase::Decoding, "phase must stay Decoding");
        assert!(s.get(id).is_some());
    }

    #[test]
    fn admit_in_finished_returns_err() {
        let mut s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(4, 32768);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Decoding);
        s.evict(id).expect("evict during Decoding must succeed");
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.phase(), Phase::Finished);
    }

    #[test]
    fn evict_all_from_finished_resets_to_idle() {
        let mut s = Scheduler::new(4, 32768);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Finished);
        s.evict_all().expect("evict_all");
        assert_eq!(s.phase(), Phase::Idle);
        assert_eq!(s.active_count(), 0);
    }

    #[test]
    fn evict_all_in_idle_returns_err() {
        let mut s = Scheduler::new(4, 32768);
        let err = s.evict_all().expect_err("evict_all from Idle must fail");
        assert!(format!("{err}").contains("Idle"), "unexpected err: {err}");
    }

    #[test]
    fn evict_all_in_admitting_returns_err() {
        let mut s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(4, 32768);
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
        let mut s = Scheduler::new(2, 32768);
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
        let mut s = Scheduler::new(2, 32768);
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
        let mut s = Scheduler::new(4, 32768);
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

        let mut s = Scheduler::new(2, 32768);
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
        let mut s = Scheduler::new(2, 32768);
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

        let mut s = Scheduler::new(2, 32768);
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

        let mut sched = Scheduler::new(2, 32768);

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
        let mut s = Scheduler::new(4, 32768);

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

        let mut s = Scheduler::new(1, 1024);

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
}
