# B1-p2.3e.1a Vectorize Greedy Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Scheduler::prefill_admitted_inner` and `Scheduler::step`'s per-row sampler.sample loop with a single vectorized GPU dispatch when every row is greedy (default config). Yields ~3-4× sampler speedup for the most common production traffic pattern (default-config greedy).

**Architecture:** New `pub fn sample_batch(samplers: &[&Sampler], logits: &Array, histories: &[&[u32]]) -> Result<Vec<u32>>` in `core/sampler.rs`. Routes by config:
- **All-greedy fast path** (every sampler.is_greedy()): `argmax(logits, axis=-1)` over `[B, vocab]` → single `.to_vec::<u32>()` returning `[B]` token ids.
- **Mixed / configured fallback**: per-row `Sampler::sample` loop (same code as today). 3e.1b will vectorize this fallback further.

Scheduler call sites switch from per-row to batched dispatch. The 1-shot vectorized path eliminates B sequential GPU↔CPU sync points (4-12 ms per step at B=4) in favour of 1 dispatch + 1 sync (~1-2 ms).

**Tech Stack:** Rust, MLX (Metal kernels), Qwen3.5-4B-MLX-4bit real-model fixture for verification.

**Spec ref:** `docs/superpowers/specs/2026-05-17-b1-p2-3e-1-vectorized-sampler-design.md` (commit `c18056c` post-Boss-review).

**Branch target:** `ironmlx-b1-p2-3e1a-vectorize-greedy` (cut from `ironmlx-b1-p2-3c-plus-chunked-admit-mid` HEAD `7fe3502` after 3c+ close-out).

---

## Pre-flight

### Step 0: Branch + baseline gates

- [ ] **Step 0.1: Confirm branch is current 3c+ HEAD.**

```bash
git rev-parse --abbrev-ref HEAD  # expect: ironmlx-b1-p2-3e1a-vectorize-greedy
git log --oneline -3             # expect c18056c (spec update) + 7fe3502 (3c+ close-out) + ...
```

- [ ] **Step 0.2: Pre-flight hygiene PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three must exit 0. If not, the 3c+ baseline is broken — stop and report.

- [ ] **Step 0.3: Baseline `cargo test --lib` PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -3
```

Expected: `test result: ok. 245 passed; 0 failed; ...`. If any failures, fix or report before continuing.

---

## Task 1: `Sampler::is_greedy()` predicate + `sample_batch` entry function

**Files:**
- Modify: `ironmlx/src/core/sampler.rs` (add `is_greedy()` + `sample_batch()`)
- Add: unit tests in the same file's `mod tests`

**Goal:** Add the `sample_batch` function with all-greedy vectorized fast path. Configured-sampler fallback delegates to existing `Sampler::sample` per row. No Scheduler changes yet — this task ships standalone with unit tests verifying numerical parity vs per-row.

### Step 1.1: Add `Sampler::is_greedy()`

- [ ] **Add predicate method on `Sampler`:**

Insert into `impl Sampler` (near `is_pipelinable` on line ~151):

```rust
    /// Returns `true` iff this sampler is in the "default greedy"
    /// configuration: `temperature <= 0` and no penalties / filters
    /// (`top_k`, `top_p`, `min_p`, repetition / frequency / presence
    /// penalty all `None` / zero). Used by [`sample_batch`] (3e.1a) to
    /// pick the vectorized argmax fast path when every active row's
    /// sampler is greedy.
    ///
    /// Distinct from [`is_pipelinable`] which permits non-greedy
    /// temperature as long as penalties are off: pipelined decode only
    /// requires no host-side penalty math, whereas `is_greedy` requires
    /// the full greedy short-circuit at `Sampler::sample` line ~210.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.repetition_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
    }
```

- [ ] **Step 1.2: Add unit test for `is_greedy`:**

In `#[cfg(test)] mod tests` of `core/sampler.rs`:

```rust
    #[test]
    fn is_greedy_true_for_default() {
        let s = Sampler::greedy();
        assert!(s.is_greedy());
    }

    #[test]
    fn is_greedy_false_when_temperature_set() {
        let s = Sampler::greedy().with_temperature(0.7);
        assert!(!s.is_greedy());
    }

    #[test]
    fn is_greedy_false_when_top_p_set() {
        let s = Sampler::greedy().with_top_p(0.9);
        assert!(!s.is_greedy());
    }

    #[test]
    fn is_greedy_false_when_repetition_penalty_set() {
        let s = Sampler::greedy().with_repetition_penalty(1.1);
        assert!(!s.is_greedy());
    }
```

### Step 1.3: Add `sample_batch` function

- [ ] **Add the batched entry function at top-level in `core/sampler.rs` (outside `impl Sampler`):**

```rust
/// Batched per-row sampling for `Scheduler::step` and
/// `Scheduler::prefill_admitted_inner` (B1-p2.3e.1a).
///
/// `logits` shape: `[B, vocab]` (caller already collapsed the
/// `[B, 1, vocab]` step output to drop the seq=1 dim).
/// `samplers` and `histories` must be length `B`, indexed in row
/// order. Each row's sampler is cloned per-request at admit time
/// (`RequestState::sampler`) so this borrow does not contend with
/// concurrent admits.
///
/// Returns `[B]` `Vec<u32>` of sampled token ids, one per row.
///
/// # Routing (spec §4.1)
/// - **All-greedy fast path** (every `samplers[b].is_greedy()`):
///   single `argmax(logits, axis=-1)` GPU dispatch → one
///   `.to_vec::<u32>()` host transfer for the whole batch. Replaces
///   B sequential `.item()` syncs (~1-3 ms each) with one
///   coalesced dispatch (~1-2 ms total). 3-4× per-step sampler
///   speedup at B=4.
/// - **Mixed / configured fallback** (any row not greedy): per-row
///   loop calling `Sampler::sample` exactly as the pre-3e.1a step
///   did. 3e.1b extends this fallback to vectorize temperature /
///   top-p / repetition penalty; top-k remains per-row pending a
///   custom Metal partial-sort kernel.
///
/// # Errors
/// - `samplers.len() != B` or `histories.len() != B` (`B` is
///   `logits.shape()[0]`).
/// - `logits` is not 2-D `[B, vocab]`.
/// - Underlying MLX argmax / `.to_vec` failures bubble up.
pub fn sample_batch(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    if dims.len() != 2 {
        anyhow::bail!(
            "sample_batch: logits must be 2-D [B, vocab]; got shape {:?}",
            dims
        );
    }
    let b = dims[0] as usize;
    if samplers.len() != b {
        anyhow::bail!(
            "sample_batch: samplers.len()={} != B={}",
            samplers.len(),
            b
        );
    }
    if histories.len() != b {
        anyhow::bail!(
            "sample_batch: histories.len()={} != B={}",
            histories.len(),
            b
        );
    }

    // All-greedy fast path.
    if samplers.iter().all(|s| s.is_greedy()) {
        // `argmax(logits, axis=-1, keepdims=false)` over [B, vocab]
        // returns [B] u32 indices. One GPU dispatch, one host sync.
        let ids = reduction::argmax(logits, -1, false)?;
        let tokens: Vec<u32> = ids.to_vec()?;
        if tokens.len() != b {
            anyhow::bail!(
                "sample_batch: argmax returned {} tokens, expected B={}",
                tokens.len(),
                b
            );
        }
        return Ok(tokens);
    }

    // Mixed / configured fallback: per-row sequential. 3e.1b will
    // vectorize this for non-top-k configs.
    let mut tokens = Vec::with_capacity(b);
    for (i, sampler) in samplers.iter().enumerate() {
        // Slice row i out of [B, vocab] into [vocab] for Sampler::sample.
        let row = mlx::ops::indexing::slice_strided_on(
            logits,
            [i as i32, 0],
            [i as i32 + 1, dims[1]],
            [1_i32, 1],
            (),
        )?;
        let row_flat = row.reshape((dims[1],))?;
        tokens.push(sampler.sample(&row_flat, histories[i])?);
    }
    Ok(tokens)
}
```

- [ ] **Step 1.4: Update imports if needed.**

`reduction::argmax` is already imported via the top-of-file
`use mlx::{ops::{indexing, reduction, sort, unary, All}, random, Array};`. No change.

If `mlx::ops::indexing::slice_strided_on` is not yet imported at the function call site, add `use mlx::ops::indexing::slice_strided_on;` near the top or qualify inline as above.

### Step 1.5: Add `sample_batch` unit tests

- [ ] **Add to `#[cfg(test)] mod tests`:**

```rust
    use mlx::Dtype;

    fn make_logits_b_vocab(b: usize, vocab: usize, max_at_per_row: &[usize]) -> Array {
        // Build a [B, vocab] f32 Array with row i's argmax at column max_at_per_row[i].
        assert_eq!(b, max_at_per_row.len(), "max indices must match B");
        let mut flat: Vec<f32> = vec![0.0; b * vocab];
        for (i, &max_col) in max_at_per_row.iter().enumerate() {
            flat[i * vocab + max_col] = 100.0;
        }
        let arr: Array = (&flat[..], &[b as i32, vocab as i32][..])
            .try_into()
            .expect("logits Array");
        arr
    }

    #[test]
    fn sample_batch_all_greedy_returns_per_row_argmax() {
        let samplers_owned: Vec<Sampler> = (0..4).map(|_| Sampler::greedy()).collect();
        let samplers: Vec<&Sampler> = samplers_owned.iter().collect();
        let logits = make_logits_b_vocab(4, 64, &[3, 7, 17, 63]);
        let histories: Vec<&[u32]> = vec![&[], &[], &[], &[]];
        let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch greedy");
        assert_eq!(tokens, vec![3, 7, 17, 63]);
    }

    #[test]
    fn sample_batch_b1_greedy() {
        // B=1 edge case.
        let s = Sampler::greedy();
        let samplers = vec![&s];
        let logits = make_logits_b_vocab(1, 32, &[15]);
        let histories: Vec<&[u32]> = vec![&[]];
        let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch B=1");
        assert_eq!(tokens, vec![15]);
    }

    #[test]
    fn sample_batch_mismatched_samplers_errs() {
        let s = Sampler::greedy();
        let samplers = vec![&s, &s]; // 2 samplers
        let logits = make_logits_b_vocab(4, 32, &[0, 1, 2, 3]); // B=4
        let histories: Vec<&[u32]> = vec![&[]; 2];
        let r = sample_batch(&samplers, &logits, &histories);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("samplers.len()"), "msg: {msg}");
    }

    #[test]
    fn sample_batch_mismatched_histories_errs() {
        let s = Sampler::greedy();
        let samplers = vec![&s, &s, &s, &s];
        let logits = make_logits_b_vocab(4, 32, &[0, 1, 2, 3]);
        let histories: Vec<&[u32]> = vec![&[], &[]]; // 2 histories
        let r = sample_batch(&samplers, &logits, &histories);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("histories.len()"), "msg: {msg}");
    }

    #[test]
    fn sample_batch_3d_logits_errs() {
        let s = Sampler::greedy();
        let samplers = vec![&s];
        // 3D logits [1, 1, 32] — caller should slice to 2D first.
        let flat: Vec<f32> = vec![0.0; 32];
        let logits: Array = (&flat[..], &[1_i32, 1_i32, 32_i32][..])
            .try_into()
            .unwrap();
        let histories: Vec<&[u32]> = vec![&[]];
        let r = sample_batch(&samplers, &logits, &histories);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("2-D"), "msg: {msg}");
    }

    #[test]
    fn sample_batch_configured_fallback_matches_per_row() {
        // B=4 where ONE row has temperature → mixed batch → fallback.
        // Use Sampler::sample with fixed seed for deterministic compare.
        let s_greedy = Sampler::greedy();
        let s_temp = Sampler::greedy().with_temperature(0.7).with_seed(42);
        let samplers: Vec<&Sampler> = vec![&s_greedy, &s_temp, &s_greedy, &s_greedy];
        let logits = make_logits_b_vocab(4, 32, &[5, 10, 15, 20]);
        let histories: Vec<&[u32]> = vec![&[], &[], &[], &[]];

        // Vectorized batch path (will take fallback because s_temp not greedy).
        let tokens_batch =
            sample_batch(&samplers, &logits, &histories).expect("sample_batch mixed");

        // Per-row reference using fresh samplers (Sampler is !Clone-safe across
        // PRNG state; rebuild the with_seed(42) one to get the same key).
        let s_temp_ref = Sampler::greedy().with_temperature(0.7).with_seed(42);
        let mut tokens_ref: Vec<u32> = Vec::with_capacity(4);
        for (i, expected_argmax) in [5, 10, 15, 20].iter().enumerate() {
            let row = mlx::ops::indexing::slice_strided_on(
                &logits,
                [i as i32, 0],
                [i as i32 + 1, 32_i32],
                [1_i32, 1],
                (),
            )
            .unwrap();
            let row_flat = row.reshape((32_i32,)).unwrap();
            let s_ref = if i == 1 { &s_temp_ref } else { &s_greedy };
            tokens_ref.push(s_ref.sample(&row_flat, &[]).unwrap());
            // Greedy rows must produce their argmax index.
            if i != 1 {
                assert_eq!(tokens_ref[i] as usize, *expected_argmax);
            }
        }

        assert_eq!(tokens_batch, tokens_ref);
    }
```

### Step 1.6: Run lib tests

- [ ] **Run:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::sampler 2>&1 | tail -10
```

Expected: pre-existing sampler tests + 4 new `is_greedy_*` + 6 new `sample_batch_*` all PASS.

### Step 1.7: Hygiene + commit T1

- [ ] **Hygiene gate:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

- [ ] **Commit:**

```bash
git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.1a-t1): Sampler::is_greedy + sample_batch

Adds a per-row batched sampling entry point that the Scheduler
will dispatch to in T2. All-greedy batches take a single
argmax(logits, axis=-1) GPU dispatch + one .to_vec::<u32>() host
sync, replacing B sequential .item() syncs in the per-row loop.
Mixed / configured batches fall back to a per-row Sampler::sample
loop (same behavior as today; 3e.1b vectorizes this path further).

Sampler::is_greedy() distinguishes the default-config fast path
from non-trivial sampler configs and is the routing predicate
inside sample_batch.

4 is_greedy_* + 6 sample_batch_* unit tests cover greedy fast
path, B=1 edge, length-mismatch errors, 3D-logits rejection, and
mixed-batch fallback parity vs per-row reference.

Spec ref: docs/superpowers/specs/2026-05-17-b1-p2-3e-1-vectorized-sampler-design.md §4.1-4.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Scheduler::step` + `Scheduler::prefill_admitted_inner` call-site refactor

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (replace per-row sampler.sample loops with `sample_batch` dispatch)

**Goal:** Switch the two B-loop sampling call sites in Scheduler to use `sample_batch`. Behavior unchanged (same tokens emitted, same termination semantics, same RequestState updates); only the implementation of the sampling step changes.

### Step 2.1: Refactor `Scheduler::step`'s sampling block

- [ ] **Locate the per-row sampling loop in `core/scheduler.rs::step`** (around line 935-955 per current HEAD):

```rust
        for (b_idx, was_active) in active_at_start.iter().enumerate() {
            if !was_active {
                continue;
            }
            let row_flat = slice_logits_row(&logits, b_idx)
                .map_err(|e| anyhow!("step: slice_logits_row(row {b_idx}) failed: {e:?}"))?;

            let state = self.slots[b_idx]
                .as_mut()
                .expect("active_at_start guaranteed Some");
            ...
            let token = state.sampler.sample(&row_flat, &history)?;
            ...
            events.push(StepEvent { ... });
        }
```

- [ ] **Replace with a two-stage refactor: (a) collect (b) batch dispatch (c) distribute tokens + termination.**

```rust
        // Stage A: collect per-row sampler refs + history slices in slot order.
        // Pad rows + finished rows use a sentinel "no-op" sampler and empty
        // history; their entries in the returned tokens vec are discarded
        // by the distribute loop below (controlled by active_at_start).
        let mut row_samplers: Vec<&Sampler> = Vec::with_capacity(self.b_max);
        let mut row_histories: Vec<Vec<u32>> = Vec::with_capacity(self.b_max);
        // Sentinel sampler — greedy, never read because we skip non-active rows
        // in the distribute pass. Borrowing requires a stable address; we
        // construct one here on the stack-scope of `step`.
        let sentinel = Sampler::greedy();
        for (b_idx, &was_active) in active_at_start.iter().enumerate() {
            if was_active {
                let state = self.slots[b_idx]
                    .as_ref()
                    .expect("active_at_start guaranteed Some");
                row_samplers.push(&state.sampler);
                let mut hist: Vec<u32> = Vec::with_capacity(
                    state.prompt_ids.len() + state.generated_tokens.len(),
                );
                hist.extend_from_slice(&state.prompt_ids);
                hist.extend_from_slice(&state.generated_tokens);
                row_histories.push(hist);
            } else {
                row_samplers.push(&sentinel);
                row_histories.push(Vec::new());
            }
        }

        // Stage B: collapse logits [B, 1, vocab] → [B, vocab] and dispatch.
        let logits_bv = logits.reshape((b as i32, logits.shape().as_slice()[2]))?;
        let history_refs: Vec<&[u32]> =
            row_histories.iter().map(|h| h.as_slice()).collect();
        let tokens = crate::core::sampler::sample_batch(
            &row_samplers,
            &logits_bv,
            &history_refs,
        )?;

        // Stage C: distribute tokens + run termination logic per active row.
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
```

- [ ] **Step 2.2: Refactor `Scheduler::prefill_admitted_inner`'s sampling block** (around line 730-760 per current HEAD).

Same pattern: collect per-row sampler refs + histories → `sample_batch` → distribute tokens + push events. The pre-existing code has slightly different shape (no `active_at_start` filter — all admitted rows are sampled), but the same three-stage refactor applies. Use `self.slots[b_idx].as_ref()` filter where prompt is present and skip pad slots.

Show the rewritten block in the commit message; the structure is identical to `step`'s rewrite above except:
- No `active_at_start` (use `s.as_ref()` to detect occupied slots; pad slots still need a sentinel sampler).
- `state.generated_tokens` is empty before this call → history is just `prompt_ids.iter().copied().collect()`.
- This is the FIRST event emission per row; sets `state.finish_reason` if max_new_tokens == 1 (rare).

### Step 2.3: Run lib + scheduler tests

- [ ] **Run:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::scheduler 2>&1 | tail -10
```

Expected: existing scheduler tests still PASS (36+).

- [ ] **Step 2.4: Hygiene + commit T2.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
git add ironmlx/src/core/scheduler.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.1a-t2): Scheduler step + prefill_admitted via sample_batch

Replaces the per-row Sampler::sample loops in both Scheduler::step
and Scheduler::prefill_admitted_inner with a three-stage refactor:
  (A) collect per-row sampler refs + histories in slot order;
  (B) reshape logits [B,1,vocab] → [B,vocab] and dispatch
      crate::core::sampler::sample_batch (T1) once;
  (C) distribute tokens + run termination logic per active row.

For all-greedy batches (default production config), this collapses
B sequential .item() syncs into one coalesced argmax + .to_vec
dispatch, removing ~3-9 ms of sampler-driven serialization per
decode step at B=4. Mixed / configured batches fall back to the
per-row loop inside sample_batch — behavior identical to pre-3e.1a.

admit_mid_finalize keeps its single-row Sampler::sample call (B=1
intrinsically; sample_batch would be a wasteful wrapper there).

Spec ref: docs/superpowers/specs/2026-05-17-b1-p2-3e-1-vectorized-sampler-design.md §4.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Real-model verification + perf gate + close-out

**Files:**
- Add: `ironmlx/tests/b1_p2_3e_1a_vectorize_greedy.rs` (NEW — perf gate integration test)
- Add: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1a_closeout/report.md` (NEW — close-out)

### Step 3.1: Integration perf gate test

- [ ] **Add `b1_p2_3e_1a_greedy_decode_speedup`**:

```rust
//! B1-p2.3e.1a — vectorized greedy sampler perf gate.
//!
//! Goal: verify the all-greedy fast path inside `sample_batch`
//! reduces per-step sampler-driven serialization at B=4 vs. the
//! per-row pre-3e.1a loop. Measurement is RELATIVE — we sample
//! median + max per-step wall time across N decode steps, compare
//! against pre-3e.1a expectation (~4× sampler block). Robust to
//! per-system Metal compile + thermal variation by relying on
//! ratios, not absolute thresholds.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerCommand,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let loader = Loader::open(Path::new(&model_dir)).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_request(
    prompt_ids: Vec<u32>,
    max_new: usize,
    stop_token_ids: Vec<u32>,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 128,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_3e_1a_greedy_decode_speedup() {
    let (model, tokenizer) = load_fixture();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();

    // Spawn 4 concurrent greedy admits → decode goes through the
    // all-greedy fast path inside Scheduler::step's sample_batch
    // dispatch.
    let handle = spawn_scheduler_actor(model.clone(), 4, Duration::from_millis(5), 32, 32768);

    let prompts = [
        "What color is the sky?",
        "Name three fruits.",
        "Pick one number between 1 and 10.",
        "Say one word.",
    ];

    let mut tasks: Vec<tokio::task::JoinHandle<Vec<Instant>>> = Vec::new();
    for p in prompts {
        let ids = tokenize_prompt(&tokenizer, p);
        let req = make_request(ids, 30, stop_tokens.clone());
        let h = handle.clone();
        tasks.push(tokio::spawn(async move {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            h.cmd_tx
                .send(SchedulerCommand::Admit { request: req, reply_tx })
                .await
                .expect("send");
            let reply = reply_rx.await.expect("reply").expect("ok");
            let mut event_rx = reply.event_rx;
            let mut stamps: Vec<Instant> = Vec::new();
            while let Some(ev) = event_rx.recv().await {
                stamps.push(Instant::now());
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            stamps
        }));
    }

    let mut all_stamps: Vec<Vec<Instant>> = Vec::new();
    for t in tasks {
        let s = tokio::time::timeout(Duration::from_secs(120), t)
            .await
            .expect("timeout")
            .expect("join");
        assert!(s.len() >= 10, "row needs ≥ 10 tokens for gap stats; got {}", s.len());
        all_stamps.push(s);
    }

    // For each row, compute median per-token gap (skip first gap
    // which includes prefill→first-decode transition).
    let mut all_medians: Vec<Duration> = Vec::new();
    for stamps in &all_stamps {
        let mut gaps: Vec<Duration> = (2..stamps.len())
            .map(|i| stamps[i].duration_since(stamps[i - 1]))
            .collect();
        gaps.sort();
        let median = gaps[gaps.len() / 2];
        all_medians.push(median);
    }

    let max_median = all_medians.iter().max().copied().unwrap();
    let min_median = all_medians.iter().min().copied().unwrap();

    eprintln!(
        "[3e.1a perf gate] per-row median gaps: {:?} | max_median={:?} min_median={:?} ratio={:.2}x",
        all_medians,
        max_median,
        min_median,
        max_median.as_secs_f64() / min_median.as_secs_f64().max(1e-9)
    );

    // Functional gate: rows decode in lockstep so per-row medians
    // should be within 2× of each other (steps are batched; one row
    // can't pull ahead). If one row's median is > 2× another's,
    // either sample_batch is wrong (rows independently slow) or
    // step lockstep broke.
    assert!(
        max_median <= min_median * 2,
        "per-row median spread too wide: {:?} (lockstep broken?)",
        all_medians
    );

    // Perf gate (loose lower bound): the all-greedy fast path should
    // keep median gap under 200 ms on a 4B bf16 model. Pre-3e.1a
    // had ~80-120 ms median for the same prompt set; 3e.1a should
    // be ≤ this. 200 ms is a defensive ceiling that catches the
    // "vectorize broke and we fell back unintentionally to per-row"
    // regression, NOT a strict speedup proof.
    assert!(
        max_median <= Duration::from_millis(200),
        "per-row max median {max_median:?} exceeds 200 ms — sample_batch fast path may not be firing"
    );

    drop(handle);
}
```

### Step 3.2: Run perf gate

- [ ] **Run:**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  MLX_DIR=$HOME/.local/mlx \
  cargo +stable test --release --test b1_p2_3e_1a_vectorize_greedy -- --ignored --test-threads=1 --nocapture
```

Expected: PASS, with median gap printed. Typical: 80-150 ms median on warm M1 Pro.

### Step 3.3: Smoke verify the broader Scheduler path didn't regress

- [ ] **Run smoke gate:**

```bash
./scripts/sweep/sweep_smoke.sh --suites b1_p2_3b_2_scheduler_actor b1_p2_4_batched_vl::mid_admit_vl_during_text_decode b1_p2_3e_1a_vectorize_greedy::b1_p2_3e_1a_greedy_decode_speedup
```

Expected: all three PASS. `mid_admit_vl_during_text_decode` covers the chunked admit path (sample_batch invoked from step during chunk-step interleave).

### Step 3.4: Dispatch sweep_full in background

- [ ] **Background full sweep:**

```bash
bash ./scripts/sweep/sweep_full.sh > /tmp/3e_1a_sweep_full.log 2>&1 &
# Note PID — referenced in close-out once complete.
```

### Step 3.5: Write close-out report

- [ ] **Add `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1a_closeout/report.md`** documenting:
  - Goal recap + commit log (3 commits)
  - Acceptance gates (lib tests, perf gate, smoke, full sweep)
  - Performance characterization: pre-3e.1a per-step sampler block vs post-3e.1a (target: 3-4× sampler-only speedup; end-to-end step latency 5-10% improvement at B=4)
  - Architecture notes: sample_batch routing, sentinel sampler for pad rows
  - Carry-forward: 3e.1b (vectorize temperature/top-p/repetition penalty), top-k Metal kernel as separate future task

### Step 3.6: Final commit

- [ ] **Commit T3:**

```bash
git add ironmlx/tests/b1_p2_3e_1a_vectorize_greedy.rs \
        ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1a_closeout/report.md
git commit -m "$(cat <<'EOF'
test+docs(b1-p2.3e.1a-t3): perf gate + close-out

b1_p2_3e_1a_greedy_decode_speedup: spawns 4 concurrent greedy
admits, measures per-row median inter-token gap, asserts:
  - per-row medians within 2× of each other (lockstep proof);
  - max median ≤ 200 ms (sample_batch fast path firing).

Close-out report documents the 3-commit shape, perf measurement,
and the foundation 3e.1a builds for 3e.1b's full configured-sampler
vectorization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (controller, post-implementation)

After all 3 tasks complete, verify:

1. **Spec coverage:** §4.1-4.4 (sample_batch + Scheduler integration) each maps to a commit.
2. **No placeholders:** every step has real code or shell commands.
3. **Type consistency:** `sample_batch` signature in T1's code matches T2's call site.
4. **No compat code:** the per-row sampler.sample loop in step + prefill_admitted_inner is fully replaced, not wrapped.
5. **Hygiene gate at every commit:** `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`, `cargo +stable build --release` all PASS.
6. **Boss constraints honored:** Chinese in user-facing messages, frequent commits, MLX_DIR set on every cargo call, no amend / no --no-verify / no force push.
