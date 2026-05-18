# B1-p2.3c+ Chunked admit_mid Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Scheduler::admit_mid_inner`'s single-shot batched_prefill with a chunked prefill (chunk_size = `req.prefill_chunk_size`, default 512) interleaved 1:1 with `Scheduler::step` so active rows continue emitting tokens during mid-batch admit.

**Architecture:** Split `Scheduler::admit_mid_inner` into three Scheduler methods — `admit_mid_begin` (alloc temp cache + vision once for VL + reserve slot), `admit_mid_chunk` (one chunk of prefill into temp_cache; returns `is_last`), `admit_mid_finalize` (adopt temp → main + sample first token). `driver_loop`'s new `handle_admit_mid_chunked` orchestrates: loop `admit_mid_chunk` → `step` → repeat → `admit_mid_finalize`. Event routing stays in driver_loop. VL v1 forces single-chunk when `image_pad` straddles a chunk boundary; per-chunk vision slicing is deferred to v2.

**Tech Stack:** Rust, MLX (Metal kernels), tokio (SchedulerActor), Qwen3.5-4B-MLX-4bit fixture for integration tests.

**Spec ref:** `docs/superpowers/specs/2026-05-17-b1-p2-3c-plus-chunked-admit-mid-design.md` (commit `d0d56d4`).

**Branch target:** `ironmlx-b1-p2-3c-plus-chunked-admit-mid` (cut from current `ironmlx-b1-p2-3f-cache-cap` head after 3f close-out).

---

## Pre-flight

### Step 0: Cut feature branch

- [ ] **Step 0.1: Confirm 3f close-out merged.** Verify HEAD has the 3f close-out commit (final commit on `ironmlx-b1-p2-3f-cache-cap`).

```bash
git log -1 --format='%h %s'
```

Expected output line includes one of: "close-out", "regression sweep", or matches the latest commit known to fully pass 15-suite sweep.

- [ ] **Step 0.2: Cut new branch.**

```bash
git checkout -b ironmlx-b1-p2-3c-plus-chunked-admit-mid
```

- [ ] **Step 0.3: Hygiene gate baseline (PASS before any edit).**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

Expected: all three exit 0. If not, stop and ask Boss — base branch is broken.

---

## Task 1 + Task 2: Scheduler API refactor + driver_loop orchestrator (atomic, ~1.5d, sonnet)

**Why combined:** removing `Scheduler::admit_mid` breaks `scheduler_actor.rs::handle_admit_mid`'s call site. Boss's "no compat code" preference + "every commit fmt/clippy/build green" gate require these two changes to land in one commit. Subagent does T1 + T2 together; reviewer subagent verifies both halves.

### Original T1 steps below; original T2 steps spliced in at the end before Task 3

## Task 1 (subtask): Scheduler::admit_mid_begin/chunk/finalize + AdmitMidHandle

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (replace `admit_mid_inner`, add 3 new public methods + `AdmitMidHandle` struct)

**Goal:** Carve admit_mid_inner into three pieces with explicit shared state via `AdmitMidHandle`. Caller (driver_loop, T2) orchestrates the chunk loop.

### Step 1.1: Define AdmitMidHandle struct

- [ ] **Add `AdmitMidHandle` near `Scheduler` struct in scheduler.rs.**

```rust
/// State shared across the three `admit_mid_*` calls that make up a
/// chunked mid-batch admit. Built by `admit_mid_begin`, mutated by
/// `admit_mid_chunk` calls, consumed by `admit_mid_finalize`. The
/// caller (`driver_loop::handle_admit_mid_chunked`) owns this between
/// calls and interleaves `Scheduler::step` between chunks.
#[doc(hidden)]
pub struct AdmitMidHandle {
    /// Slot index this admit reserved at `admit_mid_begin`.
    pub request_id: RequestId,
    pub(crate) row_idx: usize,
    /// Full prompt for this request (cloned out of `RequestState` so we
    /// can index it across chunks without re-borrowing slot state).
    pub(crate) prompt_ids: Vec<u32>,
    /// Total prompt length (i32 for direct use in MLX shape args).
    pub(crate) prompt_len: i32,
    /// Per-chunk max token count. Equals `req.prefill_chunk_size.max(1)`
    /// at construction; may be overridden to `prompt_len` for VL
    /// requests where `image_pad` straddles a chunk boundary
    /// (forces single-chunk path; see spec §4.6 R6).
    pub(crate) chunk_size: i32,
    /// How far through `prompt_ids` we've prefilled.
    pub(crate) chunk_start: i32,
    /// B=1 temp KV cache holding this admit's prefill in progress.
    /// Allocated once in `admit_mid_begin` with cap = prompt_len + max_new_tokens.
    pub(crate) temp_cache: Vec<crate::nn::LayerCache>,
    /// VL routing flag + carried vision args (cloned from RequestState
    /// at begin so we don't re-borrow slot state per chunk).
    pub(crate) is_vl: bool,
    pub(crate) pixel_values: Option<mlx::Array>,
    pub(crate) image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    pub(crate) image_token_id: i32,
    pub(crate) image_spatial_merge_size: i32,
    /// Last chunk's [1, 1, vocab] logits — captured only on the final
    /// chunk for first-token sampling in `admit_mid_finalize`.
    /// Intermediate chunk logits are discarded.
    pub(crate) last_logits: Option<mlx::Array>,
}
```

- [ ] **Step 1.2: Run cargo build to confirm struct shape compiles.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -5
```

Expected: no errors.

- [ ] **Step 1.3: Write failing test for admit_mid_begin slot reservation.**

Add to scheduler.rs `#[cfg(test)] mod tests`:

```rust
#[test]
fn admit_mid_begin_reserves_slot_and_returns_handle() {
    // Test without a real model — use force_phase to put sched in Decoding,
    // then call admit_mid_begin with a dummy GenerateRequest. The call must
    // fail because we can't actually allocate temp_cache without a real model,
    // but the slot-reservation logic should run first.
    //
    // Skipped until 1.4 implements admit_mid_begin; this test goes into
    // U1 (admit_mid_chunk_loop_advances_temp_cache_offset) which is a
    // real-model integration test.
}
```

This is a placeholder; real unit tests for `admit_mid_*` need a real model (see Task 4 integration tests). Skip TDD-step-1 for this method — go directly to implementation.

- [ ] **Step 1.4: Implement admit_mid_begin.**

Replace `Scheduler::admit_mid` (the public entry) and `admit_mid_inner` (private) with:

```rust
impl Scheduler {
    /// Mid-batch admit: chunked entry. Caller is `driver_loop::handle_admit_mid_chunked`.
    ///
    /// Chunked architecture: caller invokes
    /// `admit_mid_begin` → loop {`admit_mid_chunk`; if !is_last, `step`} → `admit_mid_finalize`.
    /// Each chunk's GPU work runs under the model lock; between chunks the
    /// caller releases the lock to run one active-row decode step, so SSE
    /// consumers see decode events at chunk-boundary cadence rather than
    /// a single multi-second prefill stall.
    ///
    /// Same returned tuple as pre-3c+ admit_mid:
    /// `(RequestId, StepEvent)` describing the new row's first generated token.
    /// Caller routes the StepEvent and registers `event_rx` against `request_id`.
    ///
    /// # Errors
    /// - `SchedulerError::RequestTooLarge` if `prompt_len + max_new_tokens > effective_cap_max`.
    /// - Other admit failures (poison, phase) bubble up as anyhow Errs.
    pub fn admit_mid_begin(
        &mut self,
        req: GenerateRequest,
        model: &Qwen35Model,
    ) -> Result<AdmitMidHandle> {
        self.ensure_not_poisoned()?;
        // Mirror admit's cap gate from 3f.
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

        // Reserve the slot via the existing relaxed admit() path (phase stays Decoding).
        let id = self.admit(req)?;
        // Roll back the slot if temp_cache or vision allocation fails below.
        let rollback = |this: &mut Scheduler| {
            let _ = this.evict(id);
        };

        // Extract per-row state we need across chunks.
        let (prompt_ids, prompt_len_usz, max_new_tokens, pixel_values, image_grid_thw,
             image_token_id, image_spatial_merge_size) = {
            let state = self.slots[row_idx].as_ref().expect("admit inserted");
            (
                state.prompt_ids.clone(),
                state.prompt_ids.len(),
                state.max_new_tokens,
                state.pixel_values.clone(),
                state.image_grid_thw.clone(),
                state.image_token_id,
                state.image_spatial_merge_size,
            )
        };
        let prompt_len = prompt_len_usz as i32;
        let max_new_i32 = i32::try_from(max_new_tokens).unwrap_or(i32::MAX);
        let cap_for_temp = prompt_len.saturating_add(max_new_i32).max(prompt_len);

        // Capture KVCache dtype from main cache.
        let dtype = self
            .cache
            .as_ref()
            .ok_or_else(|| anyhow!("admit_mid_begin: main cache absent"))
            .map(|main| {
                main.iter()
                    .find_map(|c| match c {
                        LayerCache::Full(kv) => Some(kv.dtype()),
                        _ => None,
                    })
                    .unwrap_or(Dtype::Bfloat16)
            });
        let dtype = match dtype {
            Ok(d) => d,
            Err(e) => {
                rollback(self);
                return Err(e);
            }
        };

        // Alloc temp_cache.
        let temp_cache = match model.make_cache(1, cap_for_temp, dtype) {
            Ok(t) => t,
            Err(e) => {
                rollback(self);
                return Err(e);
            }
        };

        let is_vl = pixel_values.is_some();
        let chunk_size_req = i32::try_from(
            self.slots[row_idx]
                .as_ref()
                .expect("admit inserted")
                .prompt_ids // dummy to keep linter; real chunk_size from request
                .len() // overridden below
        ).unwrap_or(i32::MAX);
        // chunk_size comes from the request itself — fetched via prefill_chunk_size
        // field. But by the time we're here, the request has been consumed into the
        // RequestState. We need to either:
        //   (a) carry prefill_chunk_size into RequestState (preferred), or
        //   (b) pass chunk_size as a separate argument to admit_mid_begin.
        // Choose (a): see Step 1.5.
        let _ = chunk_size_req; // placeholder until Step 1.5 wires the field.
        let chunk_size = self.slots[row_idx]
            .as_ref()
            .expect("admit inserted")
            .prefill_chunk_size
            .max(1);

        // VL chunking edge case (R6): if image_pad token range crosses any chunk
        // boundary, force single-chunk (chunk_size = prompt_len) for v1.
        let chunk_size = if is_vl
            && pixel_values.is_some()
            && vl_image_pad_crosses_chunk_boundary(&prompt_ids, image_token_id, chunk_size)
        {
            tracing::warn!(
                "[admit_mid_begin] VL request with image_pad spanning chunk boundary; \
                 forcing single-chunk (chunk_size={}->{}) — v2 will support per-chunk vision slicing",
                chunk_size,
                prompt_len
            );
            prompt_len
        } else {
            chunk_size
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
            pixel_values,
            image_grid_thw,
            image_token_id,
            image_spatial_merge_size,
            last_logits: None,
        })
    }
}

// Helper, near admit_mid_begin:

/// Returns true if any image_pad token in `prompt_ids` would span a chunk
/// boundary at `chunk_size`. Used to decide v1 single-chunk fallback (R6).
fn vl_image_pad_crosses_chunk_boundary(
    prompt_ids: &[u32],
    image_token_id: i32,
    chunk_size: i32,
) -> bool {
    if image_token_id < 0 || chunk_size <= 0 {
        return false;
    }
    let pad = image_token_id as u32;
    let mut in_run = false;
    let mut run_start = 0usize;
    for (i, &t) in prompt_ids.iter().enumerate() {
        if t == pad {
            if !in_run {
                in_run = true;
                run_start = i;
            }
            // run continues
        } else if in_run {
            // run ended at i (exclusive)
            let run_end = i;
            if (run_start / chunk_size as usize) != ((run_end - 1) / chunk_size as usize) {
                return true;
            }
            in_run = false;
        }
    }
    if in_run {
        let run_end = prompt_ids.len();
        if (run_start / chunk_size as usize) != ((run_end - 1) / chunk_size as usize) {
            return true;
        }
    }
    false
}
```

- [ ] **Step 1.5: Carry `prefill_chunk_size` into `RequestState`.**

Modify `RequestState` struct to add a field:

```rust
pub struct RequestState {
    // ... existing fields ...
    /// Per-request chunk size for chunked prefill. Copied from
    /// `GenerateRequest::prefill_chunk_size` at admit time.
    pub prefill_chunk_size: i32,
}
```

Modify `admit()` to copy the field. Find the `RequestState` construction in `admit`:

```rust
let state = RequestState {
    id,
    row_idx,
    prompt_ids: req.prompt_ids,
    generated_tokens: Vec::new(),
    max_new_tokens: req.max_new_tokens,
    stop_token_ids: req.stop_token_ids,
    sampler: req.sampler.clone(),
    real_len: prompt_len_i32,
    finished: false,
    finish_reason: None,
    pixel_values: req.pixel_values,
    image_grid_thw: req.image_grid_thw,
    image_spatial_merge_size: req.image_spatial_merge_size,
    image_token_id: req.image_token_id,
    prefill_chunk_size: i32::try_from(req.prefill_chunk_size).unwrap_or(512).max(1),
};
```

`GenerateRequest::prefill_chunk_size` is `usize` per existing schema; saturate-clamp to i32 and floor at 1.

- [ ] **Step 1.6: Build to verify struct change compiles.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -10
```

Expected: PASS. If compile errors, fix call-sites then re-run.

### Step 1.7: Implement admit_mid_chunk

- [ ] **Add `Scheduler::admit_mid_chunk` method:**

```rust
impl Scheduler {
    /// Run one chunk of admit_mid prefill into `handle.temp_cache`. Caller
    /// invokes this in a loop until it returns `true` (last chunk). Between
    /// chunks, caller runs one `Scheduler::step` on the main cache so active
    /// rows continue emitting tokens.
    ///
    /// On the last chunk this method captures `[1, 1, vocab]` logits into
    /// `handle.last_logits` for `admit_mid_finalize`'s first-token sample.
    pub fn admit_mid_chunk(
        &mut self,
        handle: &mut AdmitMidHandle,
        model: &Qwen35Model,
    ) -> Result<bool /* is_last */> {
        self.ensure_not_poisoned()?;

        let chunk_end = (handle.chunk_start + handle.chunk_size).min(handle.prompt_len);
        let is_last = chunk_end == handle.prompt_len;
        let chunk_len = chunk_end - handle.chunk_start;
        let chunk_ids: Vec<i32> = handle.prompt_ids
            [handle.chunk_start as usize..chunk_end as usize]
            .iter()
            .map(|&t| t as i32)
            .collect();

        // Build chunk-local input_ids [1, chunk_len].
        let input_ids: Array =
            (&chunk_ids[..], &[1_i32, chunk_len][..])
                .try_into()
                .map_err(|e| anyhow!("admit_mid_chunk: input_ids try_into Array failed: {e:?}"))?;

        // Build chunk position_ids: starts at handle.chunk_start so MRoPE
        // sees the absolute position, not chunk-local 0..chunk_len.
        let chunk_ids_all_i32: Vec<i32> = handle.prompt_ids[0..chunk_end as usize]
            .iter()
            .map(|&t| t as i32)
            .collect();
        // Slice last chunk_len positions from the absolute-position array.
        let position_ids = if handle.is_vl {
            // VL: build full-prompt VL positions then slice chunk.
            let full_pos = build_position_ids_vl_batched(
                &[&chunk_ids_all_i32[..]],
                &[handle.image_grid_thw.as_deref()],
                handle.image_token_id,
                handle.image_spatial_merge_size,
                chunk_end,
            )?;
            // Slice axis 2 [chunk_start..chunk_end]
            mlx::ops::indexing::slice_strided_on(
                &full_pos,
                [0_i32, 0, handle.chunk_start],
                [3_i32, 1, chunk_end],
                [1_i32, 1, 1],
                (),
            )?
        } else {
            let full_pos = build_position_ids_batched(&[chunk_end], chunk_end)?;
            mlx::ops::indexing::slice_strided_on(
                &full_pos,
                [0_i32, 0, handle.chunk_start],
                [3_i32, 1, chunk_end],
                [1_i32, 1, 1],
                (),
            )?
        };

        // Build attention masks. Width = chunk_start + chunk_len (cross-chunk
        // attention to earlier KV positions + within-chunk causal).
        let dtype = match handle.temp_cache.iter().find_map(|c| match c {
            LayerCache::Full(kv) => Some(kv.dtype()),
            _ => None,
        }) {
            Some(d) => d,
            None => Dtype::Bfloat16,
        };
        let attention_mask =
            build_chunked_prefill_attention_mask(handle.chunk_start, chunk_len, dtype)?;
        let linear_attention_mask =
            build_chunked_prefill_linear_mask(handle.chunk_start, chunk_len)?;

        // Run chunk prefill. temp_cache.offsets[0] advances chunk_start → chunk_end.
        let logits = if handle.is_vl {
            // First chunk carries vision args; later chunks pass empty vision args
            // because chunked v1 enforces image_pad fully within a single chunk.
            let pv_for_chunk: Vec<Option<&Array>> = if handle.chunk_start == 0 {
                vec![handle.pixel_values.as_ref()]
            } else {
                vec![None]
            };
            let grids_for_chunk: Vec<Option<&[(i32, i32, i32)]>> = if handle.chunk_start == 0 {
                vec![handle.image_grid_thw.as_deref()]
            } else {
                vec![None]
            };
            model.batched_prefill_vl(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_attention_mask,
                &[chunk_len],
                &pv_for_chunk,
                &grids_for_chunk,
                handle.image_token_id,
                Some(&mut handle.temp_cache),
                (),
            )?
        } else {
            model.batched_prefill(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_attention_mask,
                &[chunk_len],
                Some(&mut handle.temp_cache),
                (),
            )?
        };

        if is_last {
            handle.last_logits = Some(logits);
        }
        handle.chunk_start = chunk_end;
        Ok(is_last)
    }
}
```

- [ ] **Step 1.8: Add `build_chunked_prefill_attention_mask` + `build_chunked_prefill_linear_mask` helpers in `core/generate.rs`** (if they don't already exist — they're likely already there from GenerationStream's chunked prefill).

Search:

```bash
grep -n "build_chunked_prefill" ironmlx/src/core/generate.rs
```

If absent: derive from `build_batch_attention_mask` + `build_batch_linear_mask`. The chunked variant differs in width: `chunk_start + chunk_len` columns instead of `T_max`. Provide signatures:

```rust
pub fn build_chunked_prefill_attention_mask(
    chunk_start: i32,
    chunk_len: i32,
    dtype: Dtype,
) -> Result<Array> { ... }

pub fn build_chunked_prefill_linear_mask(
    chunk_start: i32,
    chunk_len: i32,
) -> Result<Array> { ... }
```

If GenerationStream already has equivalent helpers, reuse them. (Check `chunked_prefill` in generate.rs / GenerationStream.)

- [ ] **Step 1.9: Build to verify chunk method compiles.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -10
```

### Step 1.10: Implement admit_mid_finalize

- [ ] **Add `Scheduler::admit_mid_finalize`:**

```rust
impl Scheduler {
    /// Final phase of chunked admit_mid: adopt temp_cache row 0 into main
    /// cache at the reserved row_idx, then sample first generated token
    /// from the last chunk's logits.
    ///
    /// Returns `(request_id, first_event)`. Caller routes `first_event`
    /// to its event_rx, marking the slot as fully active.
    pub fn admit_mid_finalize(
        &mut self,
        handle: AdmitMidHandle,
        _model: &Qwen35Model,
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

        // Grow main cache cap if needed (3f Option C).
        let cap_for_temp = temp_cache
            .iter()
            .find_map(|c| match c {
                LayerCache::Full(kv) => Some(kv.cap()),
                _ => None,
            })
            .unwrap_or(0);
        self.grow_main_cache_to(cap_for_temp)?;

        // Adopt temp → main.
        {
            let main_cache = self.cache.as_mut().expect("cache asserted Some by Decoding phase");
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
                    _ => {
                        return Err(anyhow!(
                            "admit_mid_finalize: cache layer kind mismatch"
                        ))
                    }
                }
            }
        }

        // Sample first generated token (same code as pre-3c+).
        let row_logits = slice_logits_row(&logits, 0)?;
        let token = {
            let state = self.slots[row_idx]
                .as_ref()
                .expect("admit_mid_begin reserved the slot");
            let history: Vec<u32> = prompt_ids.clone();
            state.sampler.sample(&row_logits, &history)?
        };

        // Update state + termination.
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

        Ok((id, StepEvent {
            id,
            token,
            finish_reason,
        }))
    }
}
```

- [ ] **Step 1.11: Remove the old `admit_mid` + `admit_mid_inner` methods.**

The pre-3c+ public `Scheduler::admit_mid(req, model) -> Result<(RequestId, StepEvent)>` and private `admit_mid_inner` are replaced by the three new methods. Delete them outright per Boss "no compat code" preference. Update any non-test caller (there shouldn't be any beyond `scheduler_actor.rs::handle_admit_mid` — T2 rewrites that caller).

- [ ] **Step 1.12: Build to confirm all internal references resolve.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -10
```

Expect compile errors only at `handle_admit_mid` call sites in scheduler_actor.rs — those are intentional and get fixed in T2.

If errors elsewhere (tests, etc.), fix immediately or BLOCKED report to controller.

### Step 1.13: Unit tests U1, U2 (handle-state, single-chunk fallback)

- [ ] **Add U1 test (cfg(test) seam: temp cache observable):**

Add `pub(crate) fn temp_cache_offsets(handle: &AdmitMidHandle) -> Vec<i32>` cfg-test accessor near AdmitMidHandle:

```rust
#[cfg(test)]
pub(crate) fn admit_mid_handle_first_full_layer_offsets(
    handle: &AdmitMidHandle,
) -> Vec<i32> {
    handle.temp_cache
        .iter()
        .find_map(|c| match c {
            LayerCache::Full(kv) => Some(kv.offsets().to_vec()),
            _ => None,
        })
        .unwrap_or_default()
}
```

The U1, U2, U3, U5, U6 spec tests all require a real model — they're integration tests in T4. T1 unit tests are limited to compile-time + helper coverage:

```rust
#[test]
fn vl_image_pad_crosses_chunk_boundary_detects_run_across() {
    // image_token_id=42, run from idx 250..260, chunk_size=256 → straddles 256-boundary
    let ids: Vec<u32> = (0..400u32)
        .map(|i| if (250..260).contains(&(i as i32)) { 42 } else { 1 })
        .collect();
    assert!(vl_image_pad_crosses_chunk_boundary(&ids, 42, 256));
    // chunk_size=512 → 250..260 all in chunk 0; no crossing
    assert!(!vl_image_pad_crosses_chunk_boundary(&ids, 42, 512));
}

#[test]
fn vl_image_pad_no_pads_returns_false() {
    let ids: Vec<u32> = (0..200u32).collect();
    assert!(!vl_image_pad_crosses_chunk_boundary(&ids, 42, 64));
}

#[test]
fn vl_image_pad_run_within_single_chunk_returns_false() {
    // image_pad run at 100..150, chunk_size=256 → all in chunk 0
    let ids: Vec<u32> = (0..200u32)
        .map(|i| if (100..150).contains(&(i as i32)) { 42 } else { 1 })
        .collect();
    assert!(!vl_image_pad_crosses_chunk_boundary(&ids, 42, 256));
}
```

- [ ] **Step 1.14: Run lib tests.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::scheduler 2>&1 | tail -15
```

Expected: existing tests still pass + 3 new VL-helper tests pass. If any existing test fails, it's likely from removed `admit_mid` API — refactor the test to use the new chunked path or remove if obsolete.

- [ ] **Step 1.15: DO NOT commit yet.** This subtask intentionally breaks `scheduler_actor.rs::handle_admit_mid`'s call site. The combined T1+T2 commit happens at the end of Task 2 (subtask) below. Proceed directly to Task 2 (subtask).

---

## Task 2 (subtask of combined T1+T2): handle_admit_mid_chunked in scheduler_actor.rs

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (replace `handle_admit_mid` with `handle_admit_mid_chunked`)

**Goal:** Orchestrate the T1 three-phase chunked admit from `driver_loop`. Maintain identical external contract (single `AdmitReply` reply followed by `StepEvent` stream).

### Step 2.1: Write the new handler

- [ ] **Replace `handle_admit_mid` with `handle_admit_mid_chunked`:**

```rust
/// Mid-batch admit handler — chunked. Acquires the model lock, calls
/// `Scheduler::admit_mid_begin` to reserve a slot + alloc temp cache,
/// then runs `admit_mid_chunk` ↔ `Scheduler::step` in a 1:1 interleave
/// loop until the last chunk lands. Finally calls `admit_mid_finalize`
/// to adopt the temp cache into main and sample the first generated
/// token. Lock is reacquired per phase (begin / per-chunk / per-step /
/// finalize) so the actor's own thread can yield between GPU calls —
/// the `step` call's events route immediately to active rows' event_rx.
fn handle_admit_mid_chunked(
    cmd: SchedulerCommand,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    let mut handle = {
        let m = model.blocking_lock();
        match sched.admit_mid_begin(request, &m) {
            Ok(h) => h,
            Err(e) => {
                let _ = reply_tx.send(Err(e));
                return;
            }
        }
    };
    let id = handle.request_id;
    event_txs.insert(id, event_tx);
    if reply_tx
        .send(Ok(AdmitReply { request_id: id, event_rx }))
        .is_err()
    {
        // Caller bailed before any work happened. Drop and exit.
        let _ = sched.evict(id);
        event_txs.remove(&id);
        return;
    }

    // Chunk loop: chunk → step (skipped on last chunk) → repeat.
    loop {
        let is_last = {
            let m = model.blocking_lock();
            match sched.admit_mid_chunk(&mut handle, &m) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!("[SchedulerActor] admit_mid_chunk error: {e:?}");
                    let _ = sched.evict(id);
                    event_txs.remove(&id);
                    return;
                }
            }
        };

        if is_last {
            // Adopt + sample first token + route event.
            let m = model.blocking_lock();
            match sched.admit_mid_finalize(handle, &m) {
                Ok((_id, first_event)) => {
                    admit_count.fetch_add(1, Ordering::Relaxed);
                    route_event(first_event, event_txs);
                }
                Err(e) => {
                    tracing::error!("[SchedulerActor] admit_mid_finalize error: {e:?}");
                    let _ = sched.evict(id);
                    event_txs.remove(&id);
                }
            }
            return;
        }

        // Interleave one active-row decode step.
        let step_result = {
            let m = model.blocking_lock();
            sched.step(&m)
        };
        match step_result {
            Ok(events) => {
                for ev in events {
                    route_event(ev, event_txs);
                }
                sched.gc_finished_rows(event_txs);
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] step error inside chunked admit_mid: {e:?}");
                let _ = sched.evict(id);
                event_txs.remove(&id);
                return;
            }
        }
    }
}
```

### Step 2.2: Replace call sites

- [ ] **Update driver_loop's `RollingEvent::Admit` arm:**

Change `handle_admit_mid(cmd, ...)` → `handle_admit_mid_chunked(cmd, ...)` at the call site (around scheduler_actor.rs:285 per current HEAD line).

- [ ] **Update `drain_admission_queue`:** Same swap inside the queue drain loop body.

### Step 2.3: Delete pre-3c+ handle_admit_mid

- [ ] **Remove the old `handle_admit_mid` fn outright.** No compat shim.

### Step 2.4: Build + hygiene + commit combined T1+T2

- [ ] **Build:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -5
```

Expected: clean build (T1's broken state is now fixed by T2's caller swap).

- [ ] **Hygiene:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

- [ ] **Lib tests:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -10
```

- [ ] **Commit (combined T1+T2 atomic):**

```bash
git add ironmlx/src/core/scheduler.rs ironmlx/src/core/generate.rs ironmlx/src/core/server/scheduler_actor.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3c+-t1+t2): chunked admit_mid Scheduler API + driver_loop orchestrator

Replaces Scheduler::admit_mid (single-shot batched_prefill) with three
chunked entry points carrying state via AdmitMidHandle:
- admit_mid_begin: reserve slot, alloc temp cache, capture vision args.
- admit_mid_chunk: one chunk into temp_cache; returns is_last.
- admit_mid_finalize: adopt temp → main + sample first token.

Driver_loop's new handle_admit_mid_chunked loops admit_mid_chunk ↔
Scheduler::step ↔ ... ↔ admit_mid_finalize. Model lock reacquired per
phase so active-row SSE consumers see token events at chunk-boundary
cadence rather than a single multi-second prefill stall.

Adds RequestState::prefill_chunk_size carry-through, build_chunked_prefill_*
mask helpers (or reuses GS chunked-prefill helpers), and 3 unit tests
covering the VL image_pad boundary-crossing detection helper.

T1 + T2 are atomic per Boss "no compat code" preference — single
deletion + replacement of pre-3c+ admit_mid call path.

Spec ref: §4.2-4.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: VL chunk-vision-arg slicing v1 + R6 fallback verify (~1d, sonnet)

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (refine `admit_mid_chunk` VL branch + R6 fallback)
- Add: 1-2 integration test scenarios in `ironmlx/tests/b1_p2_3c_plus_chunked_admit_mid.rs` (NEW)

**Goal:** Validate that VL admit_mid with image_pad fully within first chunk works correctly; verify R6 single-chunk fallback for VL spanning chunk boundary.

### Step 3.1: Verify R6 fallback path numerically

- [ ] **Real-model integration test scenario `vl_image_pad_within_first_chunk_chunked`:**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn vl_image_pad_within_first_chunk_chunked() {
    // Boot SchedulerActor, b_max=2, chunk_size=256.
    // Admit a VL request whose image_pad tokens are all in positions 0..200.
    // Expected: 1+ chunks via admit_mid_chunk; chunk 0 uses batched_prefill_vl
    // (pixel_values + grid set), later chunks use batched_prefill (text path).
    // Verify generated text matches single-shot baseline.
    todo!("compose VL prompt + grid then send via SchedulerActor; assert token bytes match")
}
```

Implementation: synthesize a 600-token VL prompt where the first ~200 are image_pad tokens. Baseline: capture tokens generated by the same prompt admitted in admit-flow before queue saturates (single-shot path is no longer reachable — we removed it). Alternative: golden-test approach where token bytes are pre-captured from a known-good earlier build.

Pragmatic alternative: since we removed admit_mid_inner, golden values come from `GenerationStream` running the same prompt outside the scheduler entirely. Capture once, freeze.

- [ ] **Integration test `vl_image_pad_spans_chunk_forces_single_chunk`:**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn vl_image_pad_spans_chunk_forces_single_chunk() {
    // VL request with image_pad spanning positions 240..280, chunk_size=256.
    // admit_mid_begin must detect the boundary and override chunk_size = prompt_len.
    // Verify behavior: single chunk runs; same generated text as a hypothetical
    // single-shot baseline.
    todo!("verify R6 fallback logs a warning + produces same tokens as GS baseline")
}
```

### Step 3.2: Run tests

- [ ] **Run new VL scenarios:**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  MLX_DIR=$HOME/.local/mlx \
  cargo +stable test --release --test b1_p2_3c_plus_chunked_admit_mid -- --ignored --test-threads=1
```

### Step 3.3: Commit T3

- [ ] **Commit:**

```bash
git add ironmlx/src/core/scheduler.rs ironmlx/tests/b1_p2_3c_plus_chunked_admit_mid.rs
git commit -m "$(cat <<'EOF'
test(b1-p2.3c+-t3): VL chunk-boundary path + R6 single-chunk fallback

Two integration scenarios exercising admit_mid VL path:
- vl_image_pad_within_first_chunk_chunked: image_pad fully in chunk 0;
  later chunks use text-only batched_prefill; tokens match GS baseline.
- vl_image_pad_spans_chunk_forces_single_chunk: R6 fallback to
  single-chunk path; same tokens as baseline.

Spec ref: §4.6 NG7, §4.7 R6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integration scenarios I1, I2 + perf gate + close-out (~1d, sonnet)

**Files:**
- Add: `ironmlx/tests/b1_p2_3c_plus_chunked_admit_mid.rs` (extend with I1, I2)
- Add: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_plus_closeout/report.md` (NEW)

### Step 4.1: Stall-delta integration test I1

- [ ] **Add I1:**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_3c_plus_chunked_admit_stall_delta() {
    // Spawn SchedulerActor, b_max=4, chunk_size=256.
    // Step A: admit 3 short-prompt requests; wait until each has emitted ≥ 3 tokens.
    // Step B: admit 1 long-prompt request (prompt_len ≈ 1024) via queue drain
    //         (so it routes through chunked admit_mid_chunk loop).
    // Step C: collect per-row token-emission timestamps from event_rx (4 rows total).
    // Step D: assert: max inter-token gap across active rows ≤ 600 ms (≈ 2× chunk forward time + interleave overhead).
    // Pre-3c+ baseline (if we still had it) would be ~1.5 s.
    todo!()
}
```

### Step 4.2: VL stall-delta integration test I2

- [ ] **Add I2:**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_3c_plus_chunked_admit_vl_stall_delta() {
    // Same as I1 but mixed text + VL.
    todo!()
}
```

### Step 4.3: Run new integration scenarios

- [ ] **Run:**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  MLX_DIR=$HOME/.local/mlx \
  cargo +stable test --release --test b1_p2_3c_plus_chunked_admit_mid -- --ignored --test-threads=1
```

### Step 4.4: 16-suite regression sweep

- [ ] **Run sweep:** Reuse `/tmp/3f_regression_sweep.sh` pattern; add `b1_p2_3c_plus_chunked_admit_mid` to suite list.

```bash
bash /tmp/3c_plus_regression_sweep.sh 2>&1 | tee /tmp/3c_plus_sweep.log
```

Expected: 16/16 PASS.

**If any pre-existing test fails:** BLOCKED report to controller. Don't silently rewrite expectations — pre-existing tests assert pre-3c+ semantics; failures may be legitimate regressions.

### Step 4.5: Perf gate via iron-bench v2

- [ ] **Run perf gate.** Boot server with `--prefill-chunk-size 256 --b-max 4`, run `iron-bench v2 c=8 PP=512 max_new=64 -m chat`. Compare against the 3c-3 baseline report.

Acceptance:
- p99 inter-token gap (within a single SSE stream) ≤ 400 ms.
- Aggregate throughput ≥ 95% of pre-3c+ throughput.

### Step 4.6: Close-out report

- [ ] **Write close-out at `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_plus_closeout/report.md`:**

Structure:
- Goal recap
- 4-task commit log
- Acceptance results: U1-U6 PASS, I1/I2 numbers, 16-suite sweep, perf gate p99 numbers
- Risk retro: which R1-R8 fired, how mitigated
- Follow-ups: NG7 v2 (per-chunk vision slicing), NG3 adaptive chunk sizing

### Step 4.7: Final commit + tag

- [ ] **Commit + push:**

```bash
git add ironmlx/tests/b1_p2_3c_plus_chunked_admit_mid.rs \
        ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_plus_closeout/report.md
git commit -m "$(cat <<'EOF'
test+docs(b1-p2.3c+-t4): integration scenarios + 16-suite regression + close-out

I1 stall-delta scenario (text-only, c=4 PP=1024) confirms max inter-row
token gap drops from ~1.5 s baseline to ≤ 600 ms with chunk_size=256.
I2 mirrors with mixed text + VL.

16-suite regression sweep PASS. iron-bench v2 c=8 PP=512 perf gate:
p99 inter-token gap ≤ 400 ms; aggregate throughput within 95%
of pre-3c+ baseline.

Close-out report at
tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_plus_closeout/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (controller, post-implementation)

Before declaring 3c+ done, controller must verify:

1. **Spec §2 goals covered:** G1-G7 each maps to specific code change.
2. **Type consistency:** AdmitMidHandle fields used in T1/T2/T3 match. RequestState.prefill_chunk_size carries i32.
3. **No placeholders left:** Every `todo!()` resolved with real implementation by T4.
4. **No compat code:** old `admit_mid` / `admit_mid_inner` deleted, no shims.
5. **Frequent commits:** 4+ commits, each with fmt/clippy/build green.
6. **CLAUDE.md hygiene gate:** `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`, `cargo +stable build --release` (without `--tests` flag) — all 3 pass at every commit.
