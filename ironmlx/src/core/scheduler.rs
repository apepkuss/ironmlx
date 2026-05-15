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

use anyhow::{anyhow, Result};
use mlx::{Array, Dtype};

use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_decode_position_ids,
    build_per_row_decode_mask, build_position_ids_batched, GenerateRequest,
};
use crate::core::sampler::Sampler;
use crate::models::qwen3_5::Qwen35Model;
use crate::nn::LayerCache;

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
/// - `admit()` from `Idle`/`Admitting` → `Admitting`
/// - `prefill_admitted()` from `Idle`/`Admitting` → `Decoding`
/// - `step()` from `Decoding`: stays `Decoding` while ≥1 row unfinished,
///   transitions to `Finished` when all active rows are `finished`.
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
    pub fn new(b_max: usize) -> Self {
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
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "scheduler in {:?} phase: cannot admit; call evict_all first",
                    self.phase
                ));
            }
        }
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
        };
        self.slots[row_idx] = Some(state);
        self.phase = Phase::Admitting;
        Ok(id)
    }

    /// Evict an in-flight request, freeing its slot for reuse. The slot
    /// index is freed but the [`RequestId`] is **never** reissued (the
    /// counter keeps incrementing).
    pub fn evict(&mut self, id: RequestId) -> Result<()> {
        self.ensure_not_poisoned()?;
        match self.phase {
            Phase::Decoding => {
                return Err(anyhow!(
                    "evict illegal in {:?} phase: call evict_all after the batch finishes",
                    self.phase
                ));
            }
            Phase::Idle | Phase::Admitting | Phase::Finished => {}
        }
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        self.slots[row_idx] = None;
        if self.phase == Phase::Admitting && self.active_count() == 0 {
            self.phase = Phase::Idle;
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
        if let Some(cache) = self.cache.as_mut() {
            for lc in cache.iter_mut() {
                lc.reset()?;
            }
        }
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

        // Build [3, B, T_max] position ids and [B, 1, T_max, T_max] attn
        // mask and [B, T_max] linear mask via existing public helpers.
        let position_ids = build_position_ids_batched(&prompt_lens, max_len)?;
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
        let logits = model.batched_prefill(
            &input_ids,
            &position_ids,
            &attention_mask,
            &linear_attention_mask,
            &prompt_lens,
            Some(cache_ref),
            (),
        )?;

        // After per-row prefill, row i's cache is filled up to position
        // prompt_lens[i] - 1. The first decode step must use position
        // prompt_lens[i] for that row.
        for (slot, &plen) in self.slots.iter_mut().zip(prompt_lens.iter()) {
            if let Some(state) = slot.as_mut() {
                state.real_len = plen;
            }
        }

        // logits shape: [B, 1, vocab] — batched_prefill already collapsed
        // the sequence axis via slice_last_and_project.
        let shape = logits.shape();
        let shape_slice = shape.as_slice();
        let vocab = shape_slice[2];

        // Sample first token per occupied row from logits[:, max_len-1, :].
        let mut events: Vec<StepEvent> = Vec::new();
        for b_idx in 0..b {
            let was_active = self.slots[b_idx].is_some();
            if !was_active {
                continue;
            }
            // batched_prefill returns [B, 1, vocab] — the per-row last-token
            // position is already collapsed internally (see
            // `tests/b1_p2_1_batched_prefill.rs:173`). Slice
            // `logits[b_idx, 0, :]` → [1, 1, vocab] then reshape to [vocab].
            let row = mlx::ops::indexing::slice(
                &logits,
                &[b_idx as i32, 0_i32, 0_i32][..],
                &[b_idx as i32 + 1, 1_i32, vocab][..],
            )
            .map_err(|e| anyhow!("prefill_admitted: slice logits row {b_idx} failed: {e:?}"))?;
            let row_flat = row.reshape(&[vocab][..]).map_err(|e| {
                anyhow!("prefill_admitted: reshape logits row {b_idx} failed: {e:?}")
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
        let shape = logits.shape();
        let shape_slice = shape.as_slice();
        let vocab = shape_slice[2];

        let mut events: Vec<StepEvent> = Vec::new();
        for (b_idx, was_active) in active_at_start.iter().enumerate() {
            if !was_active {
                continue;
            }
            // Slice logits[b_idx, 0, :] → [1, 1, vocab] then reshape to [vocab].
            // Using mlx::ops::indexing::slice (same pattern as b1_p2_2_batched_decode.rs).
            let row = mlx::ops::indexing::slice(
                &logits,
                &[b_idx as i32, 0_i32, 0_i32][..],
                &[b_idx as i32 + 1, 1_i32, vocab][..],
            )
            .map_err(|e| anyhow!("step: slice logits row {b_idx} failed: {e:?}"))?;
            let row_flat = row
                .reshape(&[vocab][..])
                .map_err(|e| anyhow!("step: reshape logits row {b_idx} failed: {e:?}"))?;

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
        let s = Scheduler::new(4);
        assert_eq!(s.b_max(), 4);
        assert_eq!(s.active_count(), 0);
        assert!(s.active().is_empty());
        assert!(s.occupied_rows().is_empty());
    }

    #[test]
    fn admit_happy_path() {
        let mut s = Scheduler::new(4);
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
        let mut s = Scheduler::new(4);
        let ids: Vec<_> = (0..4)
            .map(|i| s.admit(mk_req(vec![i as u32])).expect("admit"))
            .collect();
        let rows: Vec<usize> = ids.iter().map(|id| s.get(*id).unwrap().row_idx).collect();
        assert_eq!(rows, vec![0, 1, 2, 3]);
        assert_eq!(s.active_count(), 4);
    }

    #[test]
    fn evict_releases_row() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.active_count(), 1);
        s.evict(id).expect("evict");
        assert_eq!(s.active_count(), 0);
        assert!(s.get(id).is_none());
    }

    #[test]
    fn admit_after_evict_reuses_row() {
        let mut s = Scheduler::new(4);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        assert_eq!(s.get(id_a).unwrap().row_idx, 0);
        s.evict(id_a).expect("evict a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");
        assert_eq!(s.get(id_b).unwrap().row_idx, 0); // same slot
        assert_ne!(id_a, id_b); // distinct id
    }

    #[test]
    fn admit_full_returns_err() {
        let mut s = Scheduler::new(2);
        s.admit(mk_req(vec![1])).expect("admit 0");
        s.admit(mk_req(vec![2])).expect("admit 1");
        let err = s.admit(mk_req(vec![3])).expect_err("admit full");
        let msg = format!("{err}");
        assert!(msg.contains("scheduler full"), "unexpected err: {msg}");
        assert!(msg.contains("b_max=2"), "missing b_max in err: {msg}");
    }

    #[test]
    fn evict_unknown_id_returns_err() {
        let mut s = Scheduler::new(2);
        let err = s.evict(RequestId(42)).expect_err("evict unknown");
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn id_monotonic_after_evict() {
        let mut s = Scheduler::new(2);
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
        let mut s = Scheduler::new(2);
        let id_a = s.admit(mk_req(vec![1])).expect("admit a");
        let id_b = s.admit(mk_req(vec![2])).expect("admit b");

        // Distinct `RequestState`s must own distinct Sampler instances at distinct addresses.
        let p_a: *const Sampler = &s.get(id_a).unwrap().sampler;
        let p_b: *const Sampler = &s.get(id_b).unwrap().sampler;
        assert_ne!(p_a, p_b);
    }

    #[test]
    fn occupied_rows_reflects_state() {
        let mut s = Scheduler::new(4);
        let _id_0 = s.admit(mk_req(vec![1])).expect("admit 0");
        let id_1 = s.admit(mk_req(vec![2])).expect("admit 1");
        let _id_2 = s.admit(mk_req(vec![3])).expect("admit 2");
        assert_eq!(s.occupied_rows(), vec![0, 1, 2]);
        s.evict(id_1).expect("evict 1");
        assert_eq!(s.occupied_rows(), vec![0, 2]);
    }

    #[test]
    fn phase_starts_idle() {
        let s = Scheduler::new(4);
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
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn admit_stays_in_admitting() {
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit 1");
        let _ = s.admit(mk_req(vec![2])).expect("admit 2");
        assert_eq!(s.phase(), Phase::Admitting);
    }

    #[test]
    fn evict_last_admitted_returns_to_idle() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        assert_eq!(s.phase(), Phase::Admitting);
        s.evict(id).expect("evict");
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn admit_in_decoding_returns_err() {
        let mut s = Scheduler::new(4);
        s.force_phase(Phase::Decoding);
        let err = s.admit(mk_req(vec![1])).expect_err("admit must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("Decoding") && msg.contains("cannot admit"),
            "unexpected err message: {msg}"
        );
    }

    #[test]
    fn admit_in_finished_returns_err() {
        let mut s = Scheduler::new(4);
        s.force_phase(Phase::Finished);
        let err = s.admit(mk_req(vec![1])).expect_err("admit must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("Finished") && msg.contains("cannot admit"),
            "unexpected err message: {msg}"
        );
    }

    #[test]
    fn evict_in_decoding_returns_err() {
        let mut s = Scheduler::new(4);
        let id = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Decoding);
        let err = s.evict(id).expect_err("evict must fail");
        assert!(
            format!("{err}").contains("Decoding"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn evict_all_from_finished_resets_to_idle() {
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Finished);
        s.evict_all().expect("evict_all");
        assert_eq!(s.phase(), Phase::Idle);
        assert_eq!(s.active_count(), 0);
    }

    #[test]
    fn evict_all_in_idle_returns_err() {
        let mut s = Scheduler::new(4);
        let err = s.evict_all().expect_err("evict_all from Idle must fail");
        assert!(format!("{err}").contains("Idle"), "unexpected err: {err}");
    }

    #[test]
    fn evict_all_in_admitting_returns_err() {
        let mut s = Scheduler::new(4);
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
        let mut s = Scheduler::new(4);
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
        let mut s = Scheduler::new(4);
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
        let mut s = Scheduler::new(4);
        let _ = s.admit(mk_req(vec![1])).expect("admit");
        s.force_phase(Phase::Finished); // evict_all requires Decoding/Finished
        s.poisoned = true;
        s.evict_all()
            .expect("evict_all should succeed even when poisoned");
        assert!(!s.poisoned, "poisoned flag must be cleared after evict_all");
        assert_eq!(s.phase(), Phase::Idle);
    }
}
