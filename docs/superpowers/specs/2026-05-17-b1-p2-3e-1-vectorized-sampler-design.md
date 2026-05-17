# B1-p2.3e.1 Vectorized Per-Row Sampler — Design

**Status:** Draft (brainstormed 2026-05-17, autonomous-loop)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Branch target:** `ironmlx-b1-p2-3e1-vectorized-sampler` (cut from 3c+ close-out head)

## 0. Program context

| Sub-spec | Status |
| --- | --- |
| B1-p2.1 batched prefill | ✅ DONE |
| B1-p2.2 batched decode | ✅ DONE |
| B1-p2.3a/b1..4/c1..3 continuous batching | ✅ DONE |
| 3d admission queue + config exposure | ✅ DONE |
| 3e.3 typed SchedulerError | ✅ DONE |
| 3f dynamic cap + bounded | ✅ DONE |
| 3c+ chunked admit_mid prefill | 🚧 In progress |
| **3e.1 vectorized per-row sampler** | **This spec** |
| B1-p2.4 batched VL | ✅ DONE |
| B1-p2.5 production hardening | Future |

## 1. Motivation

`Scheduler::step` currently invokes `Sampler::sample` once per active row inside a sequential for-loop (`core/scheduler.rs:805+`). For each row this is:

1. **Slice logits** `[B, 1, vocab]` → `[vocab]` (`slice_logits_row`).
2. **Apply penalties / scaling** on `Array` (small CPU+GPU work).
3. **Argmax (greedy)** or **categorical sample** (GPU dispatch + `.item::<u32>()` sync read).
4. **Push token + termination check** in `RequestState`.

The `.item()` call synchronously evaluates the lazy MLX graph and pulls one u32 back to the host. For B=4 active rows, that's **4 sequential GPU↔CPU sync points per decode step**. Each sync point is ~1-3 ms on a 4B model (the sync itself, not the compute). Total: ~4-12 ms of sampler-driven serialization per step. As b_max grows (3p2.5 will push to 8+), this fraction grows roughly linearly.

**Goal:** replace the per-row Python-style loop with a single vectorized GPU operation that processes all B rows in one dispatch. Default-config (greedy) is the hot path and benefits the most. Configured-sampler (temperature, top-p, repetition penalty, top-k) needs careful per-row variant handling but most of it can also be vectorized.

## 1.1 Brainstorm-time analysis correction (Boss-approved 2026-05-17)

Boss originally suggested "tokio/rayon CPU parallel sampler" as 3e.1a — running each row's `Sampler::sample` on a separate thread. Deep analysis (autonomous-loop, 2026-05-17) found this is **not viable** under current `Sampler` design:

- `Sampler` holds `Cell<Option<Array>>` for the PRNG key (`core/sampler.rs:43`). `Cell` is `!Send`, blocking `tokio::spawn` and `rayon::spawn`.
- Making `Sampler` `Send` (replacing `Cell` with `Mutex`) trades the lock-free property for synchronization overhead per call — Mutex acquire + release on every sample defeats the CPU parallelism win.
- Even with `Send` `Sampler`, the actual hot path is GPU dispatch + `.item()` sync, not CPU math. CPU parallelism wins are limited to penalty multiplier construction (≤ 0.5 ms per row).
- Further: MLX dispatches to a **single GPU stream**, so multi-threaded `.item()` calls queue serially on the GPU. True CPU parallelism only buys the ~0.5 ms/row penalty math (≤ 2 ms across B=4), not the ~1-3 ms/row GPU sync time (B-serialised). Realistic speedup with CPU parallel: ~2× (vs vectorize greedy's ~3-4× from coalesced [B, vocab] argmax).

**Revised approach (Boss-approved):** skip CPU parallelism entirely and go directly to vectorized GPU sampling. The "incremental" staging Boss chose (3e.1a then 3e.1b) is preserved — only the *technical content* of 3e.1a changes:

| | Original | Revised (current) |
| --- | --- | --- |
| 3e.1a (~< 1 d) | tokio/rayon CPU parallel sampler (~2× greedy + all configured) | **Vectorize greedy argmax** (~3-4× greedy only; configured falls back to per-row) |
| 3e.1b (~2-3 d) | Vectorize temperature + top-p + repetition penalty (top-k stays CPU) | (unchanged) |
| Top-k handling | CPU per-row (future Metal kernel) | (unchanged) |
| Total | 3-4 d | 3-4 d |

**Foundation-for-3e.1b property:** vectorize-greedy 3e.1a builds the `sample_batch` infrastructure (function signature, Scheduler::step integration, all-greedy fast path, mixed-batch routing, per-row PRNG key handling pattern) that 3e.1b extends in place — no rework. CPU-parallel 3e.1a would have built `Sampler::Send` refactor + thread spawning code, all of which 3e.1b vectorize would discard. The revised path is therefore strictly cleaner toward the final goal "all-config batched vectorized sampler".

## 2. Goals

### 2.1 Phase 3e.1a — vectorize greedy argmax (≤ 1 d)

- **G1a.** Replace the per-row loop in `Scheduler::step` (greedy path) with a single `argmax(logits, axis=-1)` over `[B, vocab]` that returns `[B]` token ids in one GPU dispatch + one `.to_vec()` host transfer.
- **G2a.** Greedy detection: a per-row sampler is greedy when `is_pipelinable()` returns true (temperature ≤ 0 + no penalty). If **all active rows** are greedy, take the vectorized path. Mixed batches (some greedy, some configured) fall back to per-row for now — Phase 3e.1b handles the configured path uniformly.
- **G3a.** No behavior change for non-greedy rows. Same generated tokens as pre-3e.1a per-row sampler.
- **G4a.** Per-row history advancement + termination logic remains per-row (it's CPU-only and fast). Only the token-sampling step is vectorized.

### 2.2 Phase 3e.1b — vectorize configurable sampler (2-3 d)

- **G1b.** Vectorize temperature scaling, top-p (nucleus), and repetition penalty over `[B, vocab]`.
  - **Temperature:** per-row scalar `logits[b] /= temperature[b]` → element-wise broadcast.
  - **Top-p:** per-row scalar threshold p[b]; current implementation already does this on a single row, generalizable to batched via `cumsum` + per-row mask.
  - **Repetition penalty:** per-row penalty scalar p[b] + per-row history. Vectorize via per-row scatter-divide / scatter-multiply. Approach: pad histories to a fixed max length (max over active rows), build `[B, vocab]` multiplier mask via scatter (each `(b, history_token_id)` set to penalty value), apply.
- **G2b.** Vectorize categorical sample: `random::categorical(logits, axis=-1)` over `[B, vocab]` returns `[B]` token ids.
- **G3b.** Top-k stays per-row. Top-k requires partial sort which is non-trivial on GPU (would need a custom Metal kernel à la P8a stage 9). Rows configured with top_k take the existing per-row CPU path. If both greedy and configured rows are mixed in a batch, run vectorized for the vectorizable rows and per-row CPU for the top-k rows.
- **G4b.** Frequency / presence penalty: same as repetition penalty, vectorize via batched scatter.
- **G5b.** PRNG key handling: each row owns its own seed + key state (existing per-row Sampler clone semantics). Vectorize key splitting: maintain `[B]` of keys, split each into `(consumed, next)` in parallel.

### 2.3 Cross-cutting goals

- **G5.** Numerical regression: greedy-path token output is bit-identical pre vs post 3e.1a (argmax on the same logits gives the same token id). Non-greedy paths within ±1 token under fp32 reductions (acceptable due to reduction ordering).
- **G6.** Perf: B=4 4B-decode-step `sample` block goes from ~8-12 ms (4× sequential `.item()`) to ≤ 2 ms (single vectorized + 1 sync). B=8 saves proportionally more.
- **G7.** Iron-bench v2 c=4 / c=8 throughput PASS at ≥ 1.05× pre-3e.1a baseline.

## 3. Non-goals

- **NG1.** Vectorize top-k. Out of scope — requires a custom Metal partial-sort kernel. Tracked as a separate future task ("3e.1c top-k Metal kernel").
- **NG2.** Multi-thread CPU sampler. Analyzed in §1.1 and rejected.
- **NG3.** GPU-side history token-id buffer (so the penalty mask can be built entirely on GPU). 3e.1b's first cut accepts a host-side scatter prep + GPU broadcast. A future task may move the prep to GPU using `take_along_axis`.
- **NG4.** Change to the per-row `Sampler` struct API. Public `Sampler::sample(logits, history) -> u32` stays. The vectorized path is a NEW free function `sample_batch(samplers: &[Sampler], logits: &Array, histories: &[&[u32]]) -> Result<Vec<u32>>` that callers (Scheduler::step) prefer when shape and config permit.
- **NG5.** Cache or precompile the penalty mask. Each step builds a fresh mask from active histories — cheap relative to the forward.

## 4. Architecture

### 4.1 New free function

```rust
// In core/sampler.rs

/// Vectorized sample: process `B = samplers.len()` rows in one GPU
/// dispatch where possible.
///
/// `logits`: `[B, vocab]` lazy Array (caller slices the `[B, 1, vocab]`
/// step output to drop the seq=1 dim before calling).
/// `histories`: per-row history slices. Repetition / frequency /
/// presence penalty implementations consume these.
///
/// Returns `[B]` `Vec<u32>` of sampled token ids, one per row.
///
/// # Routing
/// - If every `samplers[b].top_k` is `None` AND
///   (`samplers[b]` is fully greedy OR `samplers[b]` is fully configured
///   without top_k): take vectorized path.
/// - Otherwise (any row uses top_k): fall back to per-row CPU loop
///   using `Sampler::sample`.
pub fn sample_batch(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
) -> Result<Vec<u32>> { ... }
```

### 4.2 Vectorized greedy (3e.1a, simplest case)

```rust
fn vectorized_greedy(logits: &Array /* [B, vocab] */) -> Result<Vec<u32>> {
    // argmax over axis -1 → [B] of u32 token ids in one dispatch.
    let ids = reduction::argmax(logits, /* axis */ -1, /* keepdims */ false)?;
    // One sync host transfer for the whole batch.
    let tokens: Vec<u32> = ids.to_vec()?;
    Ok(tokens)
}
```

This single function delivers G1a. Caller in `Scheduler::step` checks "all active samplers greedy" and dispatches here.

### 4.3 Vectorized configurable (3e.1b)

```rust
fn vectorized_configured(
    samplers: &[&Sampler],
    logits: &Array, // [B, vocab]
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    let b = samplers.len() as i32;
    let mut logits = logits.clone();

    // 1. Repetition penalty: build [B, vocab] multiplier mask via scatter.
    //    Where samplers[b].repetition_penalty is None, multiplier row b is all 1.0.
    if any_row_has_repetition_penalty(samplers) {
        let mul_mask = build_batched_repetition_penalty_mask(samplers, histories /* B histories */)?;
        logits = &logits * &mul_mask;
    }

    // 2. Frequency / presence penalty: similar scatter-based [B, vocab] additive mask.
    if any_row_has_freq_or_presence(samplers) {
        let add_mask = build_batched_freq_presence_mask(samplers, histories)?;
        logits = &logits + &add_mask;
    }

    // 3. Temperature: per-row scalar division. Broadcast a [B, 1] inv_t array.
    //    Greedy rows (temperature <= 0) take argmax instead; handle via
    //    boolean mask: `tokens[b] = greedy_mask[b] ? argmax_row[b] : sampled_row[b]`.
    let inv_t = build_per_row_inv_temperature(samplers)?; // [B, 1]
    let scaled = &logits / &inv_t; // broadcasts to [B, vocab]

    // 4. Top-p (nucleus). Per-row scalar threshold. Vectorized via sort along axis -1
    //    + cumsum + mask, then unsort with argsort indices. Already O(vocab log vocab)
    //    but batched in a single dispatch.
    let mut after_top_p = scaled;
    if any_row_has_top_p(samplers) {
        after_top_p = apply_batched_top_p(&after_top_p, samplers)?;
    }

    // 5. min_p (relative to top-1 prob). Per-row vectorize same pattern as top_p.
    if any_row_has_min_p(samplers) {
        after_top_p = apply_batched_min_p(&after_top_p, samplers)?;
    }

    // 6. Categorical sample (per-row PRNG key). [B] tokens in one dispatch.
    let tokens = sample_categorical_per_row(&after_top_p, samplers)?;

    // 7. For greedy rows (temperature <= 0), override with argmax instead of sampled.
    let greedy_mask = build_greedy_mask(samplers)?; // [B] of bool
    let argmaxed = reduction::argmax(&logits /* pre-temperature */, -1, false)?;
    let final_ids = where_(greedy_mask, argmaxed, tokens)?;

    final_ids.to_vec()
}
```

### 4.4 Scheduler::step integration

```rust
// In core/scheduler.rs: Scheduler::step

// Collect [B, vocab] logits row by row → keep as [B, 1, vocab] from forward, then
// slice to [B, vocab].
let logits_2d = reshape_or_squeeze_logits(&logits)?; // [B, 1, vocab] -> [B, vocab]

// Collect active-row samplers + histories in slot order.
let active_samplers: Vec<&Sampler> = self.slots.iter().filter_map(|s| ...).collect();
let active_histories: Vec<Vec<u32>> = self.slots.iter().filter_map(...).collect();

// Vectorized sample for active rows; pad rows skipped.
let tokens = sample_batch(&active_samplers, &logits_2d, &active_history_refs)?;

// Per-row update + termination check (cheap CPU loop).
for (slot_idx, &token) in active_indices.iter().zip(tokens.iter()) {
    let state = self.slots[slot_idx].as_mut().expect("active");
    state.generated_tokens.push(token);
    state.real_len += 1;
    // termination check as before
    ...
    events.push(StepEvent { id: state.id, token, finish_reason: state.finish_reason });
}
```

### 4.5 PRNG key state management

Each row's Sampler currently owns a `Cell<Option<Array>>` that splits on each call. Vectorized path:
- Collect each row's current key (`split` once host-side or batch the splits).
- One vectorized `random::categorical` call with `keys = [B] of [key_b]`.
- Update each row's key Cell with the next-key half.

`random::categorical` already supports a batched key (per the MLX random API — verify in implementation).

### 4.6 Top-k row routing

If ANY active row has `top_k` set, the implementation:

a) **Option A (simpler):** Take the per-row loop path for the entire batch. Acceptable when top-k is rare.

b) **Option B (faster, more complex):** Split active rows into "vectorizable" and "top_k" subsets. Vectorize the subset that has no top_k. Loop per-row for the top_k subset.

Adopt **Option A** for 3e.1b v1. Option B is a follow-up if profiling shows top-k rows commonly mixed with greedy.

## 5. Acceptance

### 5.1 Unit tests

- **U1.** `sample_batch_greedy_matches_per_row` — B=4 random logits; per-row argmax vs vectorized `sample_batch` → identical token ids.
- **U2.** `sample_batch_greedy_handles_b1_correctly` — B=1 single-row → matches `Sampler::sample` greedy output.
- **U3.** `sample_batch_temperature_only` — B=4 with different temperatures, no top-k / penalties; verify per-row temperature scaling matches per-row sampler output (within fp tolerance).
- **U4.** `sample_batch_repetition_penalty_vs_per_row` — B=4 with per-row distinct histories + same penalty; vectorized batched penalty mask vs per-row penalty produces same logits within fp tolerance.
- **U5.** `sample_batch_top_k_fallback_to_per_row` — B=4 with one row top_k set; assert vectorized path is skipped (top_k routing fallback hit).
- **U6.** `sample_batch_mixed_greedy_and_configured` — B=4 some greedy some not; verify `greedy_mask` select picks correct path per row.

### 5.2 Integration tests

- **I1.** `b1_p2_3e_1a_greedy_decode_speedup` (real-model, `#[ignore]`):
  - Spawn SchedulerActor, b_max=4, all greedy samplers.
  - Send 4 chat completions concurrently; measure per-step latency p50/p99.
  - Assert p50 step latency ≤ 0.95× of pre-3e.1a baseline.
- **I2.** `b1_p2_3e_1b_configured_decode_correctness` — B=4 with temperature=0.7 + top_p=0.9 + seed=fixed; assert deterministic token output across runs and identical to per-row sampler reference output for the same seed.

### 5.3 Regression sweep

17-suite sweep (15 existing + b1_p2_3c_plus_chunked_admit_mid + b1_p2_3e_1_vectorized_sampler). All PASS.

### 5.4 Perf gate

iron-bench v2 c=8 PP=512 max_new=64:
- Aggregate throughput ≥ 1.05× pre-3e.1a baseline (G7).

## 6. Risks

- **R1.** Numerical drift in vectorized reductions (argmax / softmax / cumsum) vs per-row reductions due to reduction ordering. Mitigation: U1-U6 with tolerances; if a row has a tie at the top of logits, argmax order can differ. Document `±1 token` tolerance for non-greedy / non-deterministic paths.
- **R2.** PRNG key vectorization doesn't match per-row sequential splitting. The categorical RNG sequence differs from "split-each-then-sample-sequential". Boss-acceptable since per-row sampler clones independently anyway — the seed semantics are "deterministic given a fixed seed", not "deterministic given a fixed batch order".
- **R3.** `random::categorical` may not support `[B, vocab]` directly. Verify in implementation; fallback is a manual `multinomial`-style implementation with `cumsum` + `argmin` on uniform draws.
- **R4.** Repetition penalty scatter mask grows as `B × max_history_len × scatter_writes`. For 4B model with B=8 and 2K-token histories, that's ~16K scatter writes per step. Acceptable on Apple Silicon GPU (memory-bandwidth dominated, not compute). Mitigation: profile in I1.
- **R5.** Top-k row fallback (Option A) tanks perf when one row uses top-k. Mitigation: Option B follow-up gated by profile data.
- **R6.** `sample_batch` is a new free function, breaks no API. But `Scheduler::step` is refactored — its tests need to keep passing. Pre-3e.1a tests still admit + step + assert tokens; same tokens should come out under greedy default, so existing tests should pass without changes.

## 7. Implementation plan handoff

5 tasks (~3-4 d total):

| Task | Scope | Est. | Model |
| --- | --- | --- | --- |
| 3e.1a-T1 | `sample_batch` greedy path + Scheduler::step integration + U1, U2 + I1 | 0.5 d | sonnet |
| 3e.1b-T1 | Vectorize temperature scaling + categorical sample. U3, U6 | 1 d | sonnet |
| 3e.1b-T2 | Vectorize repetition / freq / presence penalty. U4 | 1 d | sonnet |
| 3e.1b-T3 | Vectorize top-p + min-p. Top-k routing fallback (Option A). U5 | 0.5 d | sonnet |
| 3e.1b-T4 | I2 + 17-suite regression + perf gate + close-out | 1 d | sonnet |

Spec → plan → subagent-driven implementation.

Status update (2026-05-17): Boss reviewed §1.1 and confirmed the
"skip CPU parallelism, vectorize greedy in 3e.1a" direction. Plan
authoring starts immediately on branch
`ironmlx-b1-p2-3e1a-vectorize-greedy` (cut from
`ironmlx-b1-p2-3c-plus-chunked-admit-mid` HEAD).
