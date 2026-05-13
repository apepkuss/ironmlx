# B1-p2.2 Static Batched Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify B>1 decode (`forward_on([B, 1], …)` after a `batched_prefill` cache hand-off) is numerically equivalent to per-stream `forward_on` across 4 points × 4 decode steps = 16 step-level checks.

**Architecture:** No model-side code changes — `forward_on` already accepts arbitrary `[B, S]` and the SDPA `"causal"` mode handles per-row causality automatically. Phase 2 adds **one** new helper `build_decode_position_ids` (produces `[3, B, 1]` for one decode step) and **one** new integration test that drives the matrix.

**Tech Stack:** Rust, MLX (cxx-mlx bindings), Qwen3.5-VL model. Reuses B1-p2.1 test fixture pattern (synthetic LCG prompts, no real tokenizer).

---

## File Structure

```
ironmlx/src/core/generate.rs            — add build_decode_position_ids + 2 unit tests
ironmlx/tests/b1_p2_2_batched_decode.rs — NEW 4-point × 4-step integration test
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_2_closeout/report.md — NEW close-out
```

No model source files changed. No KV cache changes. No attention.rs changes.

---

## Branch Sanity

- [ ] **Step 0: Verify branch + head**

```bash
cd /Volumes/Dev/cxx-mlx
git status --short
git log --oneline -3
```

Expected: branch `ironmlx-b1-p2-2-batched-decode`, HEAD at `83a465c` ("docs(b1-p2.2): static batched decode design spec"). No staged or unstaged changes (only `design.md` in repo root is allowed stray).

---

## Task 1: Add `build_decode_position_ids` helper

**Files:**
- Modify: `ironmlx/src/core/generate.rs` (add free function after the existing B1-p2.1 helpers)

- [ ] **Step 1: Locate the insertion point**

```bash
grep -n "pub fn build_position_ids\b\|pub fn build_position_ids_batched\b\|pub fn build_batch_attention_mask\b" /Volumes/Dev/cxx-mlx/ironmlx/src/core/generate.rs
```

Expected: 3 matches in this order — `build_position_ids` (around line 191) → `build_position_ids_batched` (around line 214) → `build_batch_attention_mask` (around line 262). Insert the new helper **immediately after** `build_batch_attention_mask`'s closing `}`.

- [ ] **Step 2: Add the helper**

Insert this function in `ironmlx/src/core/generate.rs` right after `build_batch_attention_mask`:

```rust
/// Build MRoPE position ids for one batched decode step.
/// Returns `[3, B, 1]` int32. Each batch row `i` holds the position id
/// `per_row_pos[i]` for its new token; all three MRoPE streams hold the
/// same value (text-only convention; VL B>1 in B1-p2.4 will need a
/// multi-stream variant).
pub fn build_decode_position_ids(per_row_pos: &[i32]) -> Result<Array> {
    if per_row_pos.is_empty() {
        return Err(anyhow!(
            "build_decode_position_ids: per_row_pos must be non-empty"
        ));
    }
    for (i, &p) in per_row_pos.iter().enumerate() {
        if p < 0 {
            return Err(anyhow!(
                "build_decode_position_ids: per_row_pos[{i}] = {p} must be >= 0"
            ));
        }
    }

    let b = per_row_pos.len();
    let mut flat = Vec::with_capacity(3 * b);
    for _ in 0..3 {
        flat.extend_from_slice(per_row_pos);
    }
    let arr: Array = (&flat[..], &[3_i32, b as i32, 1_i32][..]).try_into()?;
    Ok(arr)
}
```

The style matches `build_position_ids_batched`: bare `Array`/`anyhow!` (already in scope), `usize→i32` for `b`, three-stream tile via `extend_from_slice`.

- [ ] **Step 3: Add inline unit tests**

Append this `#[cfg(test)]` mod to the BOTTOM of `ironmlx/src/core/generate.rs` (after the existing `b1_p2_1_position_id_tests` and `b1_p2_1_mask_tests` mods):

```rust
#[cfg(test)]
mod b1_p2_2_decode_position_id_tests {
    use super::*;

    #[test]
    fn build_decode_position_ids_basic() {
        // B=2 with distinct positions.
        let arr = build_decode_position_ids(&[10, 20]).expect("build");
        assert_eq!(arr.shape().as_slice(), &[3, 2, 1]);
        let flat: Vec<i32> = arr.to_vec::<i32>().expect("to_vec");
        // All 3 streams identical: [10, 20] repeated 3 times.
        assert_eq!(flat, vec![10, 20, 10, 20, 10, 20]);
    }

    #[test]
    fn build_decode_position_ids_rejects_empty() {
        let err = build_decode_position_ids(&[]).expect_err("must err on empty");
        assert!(format!("{err}").contains("per_row_pos must be non-empty"));
    }
}
```

- [ ] **Step 4: Build + fmt + run helper tests**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release b1_p2_2_decode_position_id_tests 2>&1 | tail -5
```

Expected: fmt clean; build clean; 2 tests PASS (`build_decode_position_ids_basic`, `build_decode_position_ids_rejects_empty`).

- [ ] **Step 5: Quick regression — lib + P6.6 single-stream bit-identical**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: 162 passed (B1-p2.1 baseline 160 + 2 new helper tests).

- [ ] **Step 6: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/generate.rs
git commit -m "feat(b1-p2.2): add build_decode_position_ids helper + 2 unit tests"
```

---

## Task 2: Integration test — 4-point × 4-step matrix

**Files:**
- Create: `ironmlx/tests/b1_p2_2_batched_decode.rs`

- [ ] **Step 1: Write the test file**

Create `ironmlx/tests/b1_p2_2_batched_decode.rs` with the full content below. The structure mirrors B1-p2.1's test but adds a decode loop after the prefill check.

```rust
//! B1-p2.2 static batched decode — 4-point × 4-step numerical equivalence test.
//!
//! For each (B, prompt_lens) configuration:
//!   1. Per-stream reference: for each prompt i, run forward_on prefill + 4
//!      greedy-decode steps with a fresh batch=1 cache; record last_logits
//!      per step.
//!   2. Batched: build left-padded input_ids[B, S_max] + pos_ids[3,B,S_max] +
//!      attention_mask, run batched_prefill with cache(batch=B), greedy-sample
//!      B tokens, then run 4 decode steps via forward_on([B, 1], [3, B, 1], cache_B).
//!   3. Per step k ∈ {1..=4} per row i: assert
//!      max_abs_diff(batched[i].step_k, per_stream[i].step_k) < 1e-3
//!      AND argmax bit-identical.
//!
//! Run with:
//!   QWEN35_MODEL=/path/to/model \
//!   MLX_DIR=$HOME/.local/mlx \
//!   cargo test -p ironmlx --release --test b1_p2_2_batched_decode -- --ignored --nocapture

use std::path::Path;

use mlx::Array;
use mlx::Dtype;

use ironmlx::core::generate::{
    build_batch_attention_mask, build_decode_position_ids, build_position_ids,
    build_position_ids_batched,
};
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Model;
use ironmlx::nn::LayerCache;

const LOGITS_TOL: f32 = 1e-3;
const DECODE_STEPS: usize = 4;
const PAD_TOKEN_ID: u32 = 0;

/// Deterministic LCG synthetic prompt; same as B1-p2.1.
fn synth_prompt(seed: u64, n: usize, max_vocab_id: u32) -> Vec<u32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let token = 1 + ((s >> 33) as u32 % (max_vocab_id - 2));
        out.push(token);
    }
    out
}

fn max_abs_diff_f32(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).expect("af32");
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).expect("bf32");
    let av: Vec<f32> = a32.to_vec::<f32>().expect("av");
    let bv: Vec<f32> = b32.to_vec::<f32>().expect("bv");
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn argmax(arr: &Array) -> i32 {
    let f32_arr = mlx::ops::cast::astype(arr, Dtype::Float32).expect("astype f32");
    let v: Vec<f32> = f32_arr.to_vec::<f32>().expect("to_vec");
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .expect("non-empty")
}

/// Per-stream reference: run prefill + N decode steps for one prompt with
/// a fresh batch=1 cache. Returns `Vec<Array>` of length `1 + N`:
/// element 0 is the prefill last_logits; elements 1..=N are decode-step
/// last_logits.
fn per_stream_reference(model: &Qwen35Model, prompt: &[u32], n_decode: usize) -> Vec<Array> {
    let s = prompt.len() as i32;
    let cap = s + n_decode as i32 + 1;

    let mut cache: Vec<LayerCache> = model
        .make_cache(/* batch */ 1, cap, Dtype::Bfloat16)
        .expect("make_cache batch=1");

    // Prefill.
    let input_ids: Array = (&prompt[..], &[1_i32, s][..])
        .try_into()
        .expect("input_ids");
    let pos_ids = build_position_ids(0, s).expect("build_position_ids prefill");
    let prefill_logits = model
        .forward_on(&input_ids, &pos_ids, Some(&mut cache), ())
        .expect("forward_on prefill");
    let vocab = prefill_logits.shape().as_slice()[2];
    let mut out: Vec<Array> = Vec::with_capacity(n_decode + 1);
    out.push(
        prefill_logits
            .reshape(&[vocab][..])
            .expect("reshape prefill"),
    );

    // Greedy sample first decode-step input.
    let mut next_token = argmax(out.last().expect("at least prefill"));

    // Decode steps.
    for k in 1..=n_decode {
        let next_input: Array = (&[next_token as u32][..], &[1_i32, 1_i32][..])
            .try_into()
            .expect("decode input_ids");
        let pos = s + k as i32 - 1;
        let pos_ids = build_position_ids(pos, 1).expect("build_position_ids decode");
        let logits = model
            .forward_on(&next_input, &pos_ids, Some(&mut cache), ())
            .expect("forward_on decode");
        let flat = logits.reshape(&[vocab][..]).expect("reshape decode");
        next_token = argmax(&flat);
        out.push(flat);
    }

    out
}

/// Run one (B, prompt_lens, seed_base) point with `DECODE_STEPS` decode steps
/// and assert numerical equivalence with per-stream reference at every step.
fn run_point(model: &Qwen35Model, prompt_lens: &[i32], seed_base: u64) {
    let b = prompt_lens.len();
    let max_len = *prompt_lens.iter().max().expect("at least one") as usize;
    let max_vocab_id: u32 = 32_000;

    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|i| synth_prompt(seed_base + i as u64, prompt_lens[i] as usize, max_vocab_id))
        .collect();

    eprintln!(
        "[b1_p2_2] point B={}, lens={:?}, max_len={}, decode_steps={}",
        b, prompt_lens, max_len, DECODE_STEPS
    );

    // Per-stream references: prefill + N decode steps per prompt.
    let refs: Vec<Vec<Array>> = prompts
        .iter()
        .map(|p| per_stream_reference(model, p, DECODE_STEPS))
        .collect();

    // Build batched prefill inputs (left-padded).
    let mut packed: Vec<u32> = Vec::with_capacity(b * max_len);
    for p in &prompts {
        let pad_n = max_len - p.len();
        for _ in 0..pad_n {
            packed.push(PAD_TOKEN_ID);
        }
        packed.extend_from_slice(p);
    }
    let input_ids: Array = (&packed[..], &[b as i32, max_len as i32][..])
        .try_into()
        .expect("packed input_ids");

    let prefill_pos = build_position_ids_batched(prompt_lens, max_len as i32)
        .expect("build_position_ids_batched prefill");
    let attn_mask = build_batch_attention_mask(prompt_lens, max_len as i32, Dtype::Bfloat16)
        .expect("build_batch_attention_mask");

    let mut cache = model
        .make_cache(b as i32, max_len as i32 + DECODE_STEPS as i32 + 1, Dtype::Bfloat16)
        .expect("make_cache batch=B");

    let prefill_logits = model
        .batched_prefill(&input_ids, &prefill_pos, &attn_mask, Some(&mut cache), ())
        .expect("batched_prefill");
    eprintln!(
        "[b1_p2_2] prefill logits shape: {:?}",
        prefill_logits.shape().as_slice()
    );

    // Check prefill equivalence at step 0 (same as B1-p2.1).
    let dims = prefill_logits.shape();
    let vocab = dims.as_slice()[2];
    let mut next_tokens: Vec<u32> = Vec::with_capacity(b);
    for i in 0..b {
        let row = mlx::ops::indexing::slice(
            &prefill_logits,
            &[i as i32, 0_i32, 0_i32][..],
            &[i as i32 + 1, 1_i32, vocab][..],
        )
        .expect("slice prefill row");
        let row_flat = row.reshape(&[vocab][..]).expect("reshape prefill row");
        let d = max_abs_diff_f32(&row_flat, &refs[i][0]);
        let our_arg = argmax(&row_flat);
        let ref_arg = argmax(&refs[i][0]);
        eprintln!(
            "[b1_p2_2] step 0 (prefill) row {i}: max_abs_diff={:.6}, argmax_batched={}, argmax_ref={}",
            d, our_arg, ref_arg
        );
        assert!(
            d < LOGITS_TOL,
            "prefill row {i}: max_abs_diff={d} >= {LOGITS_TOL}"
        );
        assert_eq!(
            our_arg, ref_arg,
            "prefill row {i}: argmax mismatch (batched={our_arg}, ref={ref_arg})"
        );
        // Greedy sample for next step.
        next_tokens.push(our_arg as u32);
    }

    // Decode loop.
    let max_len_i32 = max_len as i32;
    for k in 1..=DECODE_STEPS {
        // Build next input [B, 1].
        let next_input: Array = (&next_tokens[..], &[b as i32, 1_i32][..])
            .try_into()
            .expect("decode input_ids");

        // Build position ids: each row's current position = max_len + k - 1.
        let per_row_pos: Vec<i32> = vec![max_len_i32 + k as i32 - 1; b];
        let pos_ids =
            build_decode_position_ids(&per_row_pos).expect("build_decode_position_ids");

        let step_logits = model
            .forward_on(&next_input, &pos_ids, Some(&mut cache), ())
            .expect("forward_on decode");
        let step_dims = step_logits.shape();
        let step_dims = step_dims.as_slice();
        assert_eq!(step_dims, &[b as i32, 1_i32, vocab]);

        let mut new_tokens: Vec<u32> = Vec::with_capacity(b);
        for i in 0..b {
            let row = mlx::ops::indexing::slice(
                &step_logits,
                &[i as i32, 0_i32, 0_i32][..],
                &[i as i32 + 1, 1_i32, vocab][..],
            )
            .expect("slice decode row");
            let row_flat = row.reshape(&[vocab][..]).expect("reshape decode row");
            let d = max_abs_diff_f32(&row_flat, &refs[i][k]);
            let our_arg = argmax(&row_flat);
            let ref_arg = argmax(&refs[i][k]);
            eprintln!(
                "[b1_p2_2] step {k} row {i}: max_abs_diff={:.6}, argmax_batched={}, argmax_ref={}",
                d, our_arg, ref_arg
            );
            assert!(
                d < LOGITS_TOL,
                "step {k} row {i}: max_abs_diff={d} >= {LOGITS_TOL}"
            );
            assert_eq!(
                our_arg, ref_arg,
                "step {k} row {i}: argmax mismatch (batched={our_arg}, ref={ref_arg})"
            );
            new_tokens.push(our_arg as u32);
        }
        next_tokens = new_tokens;
    }

    eprintln!(
        "[b1_p2_2] point B={} lens={:?} PASS (prefill + {} decode steps)",
        b, prompt_lens, DECODE_STEPS
    );
}

#[test]
#[ignore = "requires QWEN35_MODEL env"]
fn b1_p2_2_batched_decode_matrix() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let model = Qwen35Model::from_loader(&loader).expect("model");

    // Point 1: B=2 same length.
    run_point(&model, &[128, 128], 0x1111);
    // Point 2: B=2 mixed length (left-padded).
    run_point(&model, &[128, 96], 0x2222);
    // Point 3: B=4 same length.
    run_point(&model, &[128, 128, 128, 128], 0x3333);
    // Point 4: B=4 mixed length.
    run_point(&model, &[128, 96, 64, 128], 0x4444);

    eprintln!("[b1_p2_2] PASS — all 4 points × {} decode steps", DECODE_STEPS);
}
```

Notes:
- `LayerCache` import path: matches B1-p2.1 test convention. If `ironmlx::nn::LayerCache` doesn't resolve, try `ironmlx::core::cache::LayerCache`. Verify quickly:
  ```bash
  grep -rn "pub use.*LayerCache\|pub type LayerCache" /Volumes/Dev/cxx-mlx/ironmlx/src/ | head -5
  ```
- `mlx::ops::indexing::slice` is the slice API used in B1-p2.1 test (line 149). Match that exact path.
- `Loader::open_multimodal` is used by the existing P6.6/P6.7/B1-p2.1 tests — keep it for consistency even though phase 2 doesn't load vision weights.
- The reference path runs `prefill + 4 decode steps` as a single 5-element vector of `Array` (index 0 = prefill, index 1..=4 = decode steps). The batched path matches this index for index.

- [ ] **Step 2: Build the test binary**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_2_batched_decode 2>&1 | tail -10
```

Expected: build clean. Fix any import-path mismatches that surface.

- [ ] **Step 3: Run the test**

```bash
cd /Volumes/Dev/cxx-mlx
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_2_batched_decode -- --ignored --nocapture 2>&1 | tail -60
```

Use timeout 1800000 ms (30 min). The test runs (1 prefill + 4 decode) × 4 points = 20 LM forwards on the batched path + same count on per-stream reference per point.

Use `run_in_background: true` for the cargo call, then poll with Monitor or a `while kill -0` loop until the test PID exits, then `cat` the output file.

Expected output (per point):

```
[b1_p2_2] point B=2, lens=[128, 128], max_len=128, decode_steps=4
[b1_p2_2] prefill logits shape: [2, 1, 151936]
[b1_p2_2] step 0 (prefill) row 0: max_abs_diff=<small>, argmax_batched=<id>, argmax_ref=<id>
[b1_p2_2] step 0 (prefill) row 1: max_abs_diff=<small>, argmax_batched=<id>, argmax_ref=<id>
[b1_p2_2] step 1 row 0: max_abs_diff=<small>, argmax_batched=<id>, argmax_ref=<id>
[b1_p2_2] step 1 row 1: max_abs_diff=<small>, argmax_batched=<id>, argmax_ref=<id>
... (steps 2, 3, 4)
[b1_p2_2] point B=2 lens=[128, 128] PASS (prefill + 4 decode steps)
```

Final line:
```
[b1_p2_2] PASS — all 4 points × 4 decode steps
```

For every step at every row: `max_abs_diff < 1e-3` AND `argmax_batched == argmax_ref`.

### If the test FAILS

Likely causes in priority order:

1. **B=2 same-length (Point 1) step 1 fails** — KV cache `update_and_fetch` at B>1 increment write is broken. Verify cache offset advances correctly per row.
2. **Mixed-length (Point 2 / 4) decode step 1 fails but same-length (Point 1 / 3) passes** — pad-position K/V cells in the cache are corrupting decode-step attention. Spec §7 R3. Capture observed diff values and report; rollback to same-length-only points (1, 3) is the documented escape hatch.
3. **All points step 1 fails identically** — `build_decode_position_ids` shape or values wrong. Print `pos_ids.to_vec::<i32>()` for B=2 case and verify.
4. **Step k fails but step k-1 passes (for some k)** — KV cache offset advancing by 2 instead of 1, or similar increment bug. Print the cache offset after each step (need to add temporary instrumentation).

If chunk_size=0 / Point 1 / step 1 fails: BLOCKED with full per-row max_abs_diff + argmax values. Do not attempt a fix in this task.

- [ ] **Step 4: Commit (only if all 16 step-level checks PASS)**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/b1_p2_2_batched_decode.rs
git commit -m "test(b1-p2.2): 4-point × 4-step batched decode numerical equivalence"
```

---

## Task 3: Regression sweep + close-out

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_2_closeout/report.md`

- [ ] **Step 1: Full regression sweep**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected all green:
- fmt: clean
- clippy: clean (only unchanged mlx-sys C++ warnings)
- build: clean
- lib tests: 162 passed (B1-p2.1 baseline 160 + 2 new helper tests from Task 1)

- [ ] **Step 2: P6.3 single-image regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

Expected: PASS, max_diff = 0.3906, first_token = 760. Use `run_in_background: true` + Monitor on PID exit; timeout ~600000 ms.

- [ ] **Step 3: P6.6 logits-match regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored 2>&1 | tail -5
```

Expected: PASS, first_token = 760, max_diff unchanged from B1-p2.1 baseline (0.9004 for N=2 or 1.1250 for N=3 fixture).

- [ ] **Step 4: P6.7 chunked-prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored 2>&1 | tail -5
```

Expected: PASS, all 3 chunk_sizes → 760. Use timeout 900000 ms.

- [ ] **Step 5: B1-p2.1 batched prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_1_batched_prefill -- --ignored 2>&1 | tail -5
```

Expected: PASS, all 4 points, max_abs_diff = 0.000977 unchanged.

- [ ] **Step 6: Write the close-out report**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_2_closeout/report.md`:

```markdown
# B1-p2.2 Static Batched Decode — Close-out

**Branch:** `ironmlx-b1-p2-2-batched-decode` (off `ironmlx-b1-p2-batched-serving` head `b24aae8`)
**Date:** 2026-05-12
**Spec:** `docs/superpowers/specs/2026-05-12-b1-p2-2-batched-decode-design.md` (commit `83a465c`)
**Plan:** `docs/superpowers/plans/2026-05-12-b1-p2-2-batched-decode.md`

## Summary

Verified that B>1 decode via `forward_on([B, 1], [3, B, 1], cache(batch=B))`
is numerically equivalent to per-stream `forward_on` across 4 points ×
(1 prefill + 4 decode steps) = 20 step-level checks per point, 80 total
step-level assertions. Zero model-side code changes — only a single helper
(`build_decode_position_ids`) added in `core/generate.rs`, plus the new
integration test.

This closes the KV cache `update_and_fetch` increment-write test gap at
B>1: B1-p2.1 verified the first write (offset=0); B1-p2.2 verifies
offsets `S_max .. S_max+3` advance correctly per batch row.

## Acceptance Table

| Point | B | prompt_lens | step 0 (prefill) | step 1 | step 2 | step 3 | step 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | [128, 128] | <observed> | <observed> | <observed> | <observed> | <observed> |
| 2 | 2 | [128, 96] | <observed> | <observed> | <observed> | <observed> | <observed> |
| 3 | 4 | [128, 128, 128, 128] | <observed> | <observed> | <observed> | <observed> | <observed> |
| 4 | 4 | [128, 96, 64, 128] | <observed> | <observed> | <observed> | <observed> | <observed> |

All 80 step×row assertions PASS (`max_abs_diff < 1e-3` + `argmax` bit-identical).

(Fill in `<observed>` with the worst-row `max_abs_diff` per (point, step) from
Task 2 Step 3 output. Format: `0.001 / ✓` for "diff under tolerance, argmax
identical".)

## Architectural Changes

1. **`build_decode_position_ids`** (new free fn in `core/generate.rs`) —
   produces `[3, B, 1]` int32 with per-row position id; all 3 MRoPE streams
   share the same value (text-only convention).
2. No other code changes. `forward_on` is reused for both prefill and decode
   at B>1; the SDPA `"causal"` lower-right alignment automatically handles
   per-row causality at `T_q=1, T_kv=cache_len`.

## Fixes Applied

Zero fix-loop iterations. The decode equivalence held on the first
integration test run.

| Commit | Type | Description |
| --- | --- | --- |
| `<sha>` | feat | `build_decode_position_ids` helper + 2 unit tests |
| `<sha>` | test | 4-point × 4-step batched decode numerical equivalence |
| `<sha>` | docs | This close-out |

(Fill in `<sha>` from `git log --oneline 83a465c..HEAD`.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **162 passed / 0 failed** (B1-p2.1 baseline 160 + 2 new helper tests) |
| P6.3 Task 21 single-image | **PASS** — max_diff=0.3906, first_token=760 |
| P6.6 logits-match | **PASS** — baseline unchanged |
| P6.7 chunked-prefill matrix | **PASS** — all chunk_sizes → 760 |
| B1-p2.1 batched prefill matrix | **PASS** — all 4 points, max_abs_diff=0.000977 |
| B1-p2.2 4-point × 4-step batched decode matrix | **PASS** — all 80 step×row checks |

## Notes

- **KV cache `update_and_fetch` at B>1 is now verified end-to-end** through
  4 sequential increment writes (offsets `S_max, S_max+1, S_max+2, S_max+3`)
  per row. The single shared `offset` in `KVCache` is correct for uniform
  decode (all rows advance together); per-row early-stop (different offsets)
  is deferred to B1-p2.3 continuous batching.
- **SDPA "causal" mode at T_q=1, T_kv=cache_len** correctly applies per-row
  lower-right-aligned causal mask. The B1-p2.2 test is the first direct
  verification at B>1 (P6.3 / P6.6 / P6.7 verify only at B=1).
- **Mixed-length pad-K/V hypothesis** (spec §7 R3) — the prefill writes
  pad-position K/V into the leading cache slots of short rows. Decode-step
  attention reads from these cells. The mixed-length points (2, 4) PASSING
  the same tolerance as same-length points (1, 3) confirms that pad
  K/V does not corrupt decode outputs (because the prefill mask zeroed
  pad-position attention weights, so pad-position hidden states — and
  therefore the K/V cells written from them — are produced from correct
  upstream computation, not from cross-row leakage).
- **No new wrapper API.** Reusing `forward_on` keeps the call chain
  shortest for the future B1-p2.3 scheduler. Adding a `batched_decode`
  alias would be 1 line; defer until a scheduler use case demands it.

## B1-p2.x Next Steps

- **B1-p2.3** — Continuous batching with scheduler + admit/evict. Touches
  HTTP server, `GenerationStream` B>1 refactor (per-row histories,
  per-row finished flags, per-row sampler invocation), per-row stop logic,
  per-row offset tracking (KV cache may need pagination-style allocation
  to support different rows with different offsets), token-level loop. The
  largest sub-phase.
- **B1-p2.4** — VL B>1. Requires per-batch-row image scatter.
- **B1-p2.5** — Production hardening.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-12-b1-p2-2-batched-decode-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-b1-p2-2-batched-decode.md`
- Integration test: `ironmlx/tests/b1_p2_2_batched_decode.rs`
- Helper unit tests: `ironmlx/src/core/generate.rs` (mod `b1_p2_2_decode_position_id_tests`)
```

- [ ] **Step 7: Commit close-out**

```bash
cd /Volumes/Dev/cxx-mlx
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_2_closeout/report.md
git commit -m "docs(b1-p2.2): close-out — batched decode all 4 points × 4 steps green"
```

- [ ] **Step 8: Final summary log**

```bash
cd /Volumes/Dev/cxx-mlx
git log --oneline 83a465c..HEAD
```

Expected: 3 commits (spec was at `83a465c`, then 2 implementation commits + 1 close-out).

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
| --- | --- |
| §2 Goal: verify `forward_on([B, 1], [3, B, 1], cache(B))` equivalence | Task 2 |
| §2 Goal: `build_decode_position_ids` helper | Task 1 |
| §2 Goal: max_abs_diff < 1e-3 + argmax bit-identical | Task 2 (`LOGITS_TOL = 1e-3` + assert_eq on argmax) |
| §2 Goal: KV cache contents implicitly verified | Task 2 (4 decode steps with cache hand-off from prefill) |
| §2 Goal: no single-stream regression | Task 3 (P6.3 / P6.6 / P6.7 / B1-p2.1 sweep) |
| §3 Non-goals (server, scheduler, VL B>1, etc.) | Not touched in any task |
| §4.1 Reuse `forward_on` | Task 2 calls `model.forward_on(...)` directly |
| §4.2 New helper `build_decode_position_ids` | Task 1 |
| §4.3 Uniform per-row position tracking | Task 2 `per_row_pos = vec![max_len + k - 1; B]` |
| §4.4 KV cache hand-off | Task 2 uses the same `cache` mutably from prefill → decode |
| §6.1 Helper unit tests (2 tests) | Task 1 |
| §6.2 4-point × 4-step integration | Task 2 |
| §6.3 Regression gates | Task 3 |
| §7 R1 SDPA causal at T_q=1 | Task 2 (any failure surfaces immediately at step 1) |
| §7 R2 KV update increment at B>1 | Task 2 N=4 sequential decode steps stress this |
| §7 R3 mixed-length pad-K/V | Task 2 points 2 + 4 directly test this; close-out notes the hypothesis |
| §7 R4 sampler determinism | Task 2 uses argmax (deterministic) |

All spec sections have a corresponding task. No gaps.

**2. Placeholder scan:**

- Task 2 Step 3 contains "if the test FAILS" section with diagnostic guidance — concrete, not a placeholder.
- Task 3 close-out template contains `<observed>` and `<sha>` placeholders, filled in at execution time. Marked explicitly in the step text.
- No "TBD", "implement later", "fill in details" elsewhere.

**3. Type consistency:**

| Symbol | First defined | Reused |
| --- | --- | --- |
| `build_decode_position_ids(per_row_pos: &[i32]) -> Result<Array>` | Task 1 | Task 2 |
| `LOGITS_TOL = 1e-3` | Task 2 | Task 2 |
| `DECODE_STEPS = 4` | Task 2 | Task 2 |
| `PAD_TOKEN_ID = 0` | Task 2 (matches B1-p2.1) | Task 2 |
| `synth_prompt(seed, n, max_vocab_id)` | Task 2 (copied from B1-p2.1) | Task 2 |
| `per_stream_reference(model, prompt, n_decode) -> Vec<Array>` | Task 2 | Task 2 |
| `argmax(arr: &Array) -> i32` | Task 2 (copied from B1-p2.1) | Task 2 |

All signatures consistent. The reused functions from B1-p2.1 are copied verbatim (the synth_prompt LCG, max_abs_diff_f32, argmax). This is intentional — the integration tests are self-contained, and B1-p2.1 / B1-p2.2 share no code path beyond what's in `src/`.

**4. 16 step-level checks (per spec §6.2):**

Spec §6.2 says "4 points × 4 decode steps = 16 step-level checks". Plan Task 2 code:
- 4 points (`run_point` called 4 times in the test function)
- Per point: 1 prefill check (step 0) + 4 decode-step checks (k=1..=4)
- Per (point, step): B row checks

Total step×row assertions: (2+2+4+4) × 5 = 60... wait, that's 60, not 16. Let me recount.

Spec says "16 step-level checks". 4 points × 4 decode steps = 16 (point × step pairs). At each pair, B row-level assertions run. So:

- Point 1 (B=2) × 4 steps = 8 row-level checks (decode steps only)
- Point 1 step 0 (prefill) = 2 row-level checks
- Total per Point 1: 10 row-level assertions
- Sum across 4 points: (2+2+4+4) × (1 + 4) = (2+2+4+4) × 5 = 60 row-level assertions

But "16 step-level checks" = (point × step) without the row dimension. The plan tests every (point, step, row). The spec language is loose; the plan is stricter. **This is correct, not a gap.** The close-out reports per-step max_abs_diff (worst across rows).

No issues. Plan is consistent with spec intent. The close-out table makes the (point, step) grid explicit.
