# B1-p2.3e.2 PRNG Key Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `Sampler.key: Cell<Option<Array>>` field and move per-row PRNG state to `Scheduler.prng_state: Array` ([b_max, 2] u32). `Sampler` becomes pure config POD with auto-derived `Clone + Copy + Send + Sync`. `sample_row_cpu` / `configured_pipeline` / `sample_batch` signatures gain `prng_state` parameter (`&mut Array`).

**Architecture:** Scheduler holds centralized `prng_state: Array`. `Scheduler::new` zero-inits; `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner` call `self.init_row_prng(row_idx, sampler.seed)` to mint row's key from `mlx::random::key(seed)` and write into `prng_state[row_idx, :]` via `mlx::ops::indexing::slice_update`. `Scheduler::step` / `prefill_admitted_inner` pass `&mut self.prng_state` to `sample_batch`. Per-row sample step slices `prng_state[row]`, splits + advances + samples, writes back via `slice_update`.

**Tech Stack:** Rust, mlx Rust binding (`/Volumes/Dev/cxx-mlx/mlx`), Qwen3.5-4B-MLX-4bit fixture.

**Spec ref:** [`docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md`](../specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md) (commit `d4c0fc3`).

**Branch target:** `ironmlx-b1-p2-3e2-prng-centralization` (cut from `ironmlx-b1-p2-3e1b-vectorize-configured` HEAD `d146d60`; branch HEAD now `d4c0fc3` after spec rewrite).

---

## Pre-flight

### Step 0: Branch + baseline gates

- [ ] **Step 0.1: Confirm branch.**

```bash
cd /Volumes/Dev/cxx-mlx
git rev-parse --abbrev-ref HEAD  # expect: ironmlx-b1-p2-3e2-prng-centralization
git log --oneline -3             # expect d4c0fc3 (3e.2 spec rewrite) + d146d60 (3e.1b close-out addendum) + ...
```

- [ ] **Step 0.2: Pre-flight hygiene PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

- [ ] **Step 0.3: Baseline `cargo test --lib` PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -3
```

Expected: `test result: ok. 267 passed; 0 failed; ...` (post-3e.1b baseline).

---

## Task 0: `mlx::ops::indexing::slice_update` API verify + micro-bench

**Files:**
- Modify (test-only): `ironmlx/src/core/sampler.rs::mod tests` (add 2 probe tests; may be removed in T1 if confirmed standard)

**Goal:** Verify `slice_update` API surface for writing into `prng_state[row, :]` + measure per-call cost. R3 risk pre-check.

### Step 0.1: Verify API signature

- [ ] **Step 0.1.1: Read source.**

```bash
sed -n '195,220p' /Volumes/Dev/cxx-mlx/mlx/src/ops/indexing.rs
```

Expected: `pub fn slice_update<S1: IntoShape, S2: IntoShape, S3: IntoShape>(...) -> Result<Array>`. Args: source array, update array, start, stop. Verify signature matches spec §4.2 usage.

### Step 0.2: Probe test — round-trip

- [ ] **Step 0.2.1: Add probe to `core/sampler.rs::mod tests`.**

```rust
#[test]
fn probe_slice_update_per_row_round_trip() {
    use mlx::ops::indexing::{slice, slice_update};
    let b_max = 4_usize;
    let zeros: Array = mlx::ops::constructors::zeros(
        &[b_max as i32, 2_i32][..],
        mlx::Dtype::Uint32,
        (),
    )
    .expect("zeros");
    // Write key [42, 43] into row 1.
    let key_row1: Array = (&[42_u32, 43_u32][..], &[1_i32, 2_i32][..])
        .try_into()
        .expect("key_row1");
    let after_write = slice_update(
        &zeros,
        &key_row1,
        &[1_i32, 0_i32][..],
        &[2_i32, 2_i32][..],
    )
    .expect("slice_update");
    // Read row 1 back.
    let read_back = slice(
        &after_write,
        &[1_i32, 0_i32][..],
        &[2_i32, 2_i32][..],
    )
    .expect("slice");
    let read_flat = read_back
        .reshape(&[2_i32][..])
        .expect("reshape");
    let v: Vec<u32> = read_flat.to_vec().expect("to_vec");
    assert_eq!(v, vec![42, 43], "round-trip slice_update + slice");
    // Row 0 should still be zeros.
    let row0 = slice(
        &after_write,
        &[0_i32, 0_i32][..],
        &[1_i32, 2_i32][..],
    )
    .expect("slice row0");
    let row0_flat: Vec<u32> = row0
        .reshape(&[2_i32][..])
        .expect("reshape row0")
        .to_vec()
        .expect("to_vec row0");
    assert_eq!(row0_flat, vec![0, 0], "row 0 unmodified");
}
```

- [ ] **Step 0.2.2: Run probe.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- probe_slice_update_per_row_round_trip --nocapture 2>&1 | tail -10
```

Expected: PASS.

### Step 0.3: Micro-bench slice_update cost

- [ ] **Step 0.3.1: Add bench probe.**

```rust
#[test]
#[ignore]  // bench-mode
fn probe_slice_update_per_row_bench() {
    use mlx::ops::indexing::slice_update;
    use std::time::Instant;
    let b_max = 4_usize;
    let mut prng_state: Array = mlx::ops::constructors::zeros(
        &[b_max as i32, 2_i32][..],
        mlx::Dtype::Uint32,
        (),
    )
    .expect("zeros");
    prng_state.eval().expect("eval");

    let key_new: Array = (&[1_u32, 2_u32][..], &[1_i32, 2_i32][..])
        .try_into()
        .expect("key");
    key_new.eval().expect("eval key");

    // Warm-up (mlx Metal JIT)
    for _ in 0..3 {
        prng_state = slice_update(
            &prng_state,
            &key_new,
            &[0_i32, 0_i32][..],
            &[1_i32, 2_i32][..],
        )
        .expect("warm");
        prng_state.eval().expect("eval");
    }

    // Bench: 100 round trips of writing row 0 + eval
    let t0 = Instant::now();
    for _ in 0..100 {
        prng_state = slice_update(
            &prng_state,
            &key_new,
            &[0_i32, 0_i32][..],
            &[1_i32, 2_i32][..],
        )
        .expect("bench iter");
        prng_state.eval().expect("eval iter");
    }
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_secs_f64() * 1e6 / 100.0;
    eprintln!("[T0 bench] slice_update [b_max=4, write row 0]: {per_call_us:.2} µs/call");
    // No assertion — diagnostic only.
}
```

- [ ] **Step 0.3.2: Run bench.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- probe_slice_update_per_row_bench --ignored --nocapture 2>&1 | tail -10
```

**Decision rule (record in T0 commit message):**
- If `< 100 µs/call` → use `slice_update` path in `configured_pipeline` per-row (spec §4.4 default)
- If `≥ 100 µs/call` → batch-end stack alternative (collect updated row keys into `Vec<Array>`, stack at end, single assignment to `prng_state`)

At b=4, 100µs/call × 4 = 400µs overhead — acceptable. Fallback only if > 500µs.

### Step 0.4: Hygiene + commit T0

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
chore(b1-p2.3e.2-t0): mlx slice_update probe + bench

Two probe tests:
  - probe_slice_update_per_row_round_trip: writes key to row 1,
    verifies row 1 read back + row 0 unmodified
  - probe_slice_update_per_row_bench (#[ignore]): measures
    100-iteration cost on [b_max=4, 2] u32, decides whether
    spec §4.4 in-pipeline slice_update path is viable vs
    batch-end stack alternative (R3 mitigation pre-check)

Both pass / measured. Decision: <slice_update OR batch-end-stack>
based on per-call cost <X> µs.

Spec §4.2 + R3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: `Sampler` struct shrink + auto-derive Copy

**Files:**
- Modify: `ironmlx/src/core/sampler.rs` (remove `key: Cell<Option<Array>>` field, manual `Clone` impl, `ensure_key`/`store_key`/`sample_async_greedy` fns referencing key; add `#[derive(Clone, Copy)]`; static_assertions for Send+Sync; remove `use std::cell::Cell`)

**Goal:** Sampler becomes pure config struct. All key-related code removed. Sampler::sample (currently uses key) signature may need adjustment (postponed to T2 — for now mark `Sampler::sample` `#[deprecated]` placeholder or stub bail!). T1 should compile + lib tests where possible should PASS (those not using sample). Tests using `Sampler::sample` will be touched in T2.

### Step 1.1: Remove `key` field + manual Clone

- [ ] **Step 1.1.1: Edit `Sampler` struct (line 26-44).**

Before:
```rust
pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: u64,
    key: Cell<Option<Array>>,
}
```

After:
```rust
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: u64,
}
```

- [ ] **Step 1.1.2: Delete manual `Debug` impl (line 46-58, replaced by derive).**

- [ ] **Step 1.1.3: Delete manual `Clone` impl (line 61-77, replaced by derive).**

- [ ] **Step 1.1.4: Delete `use std::cell::Cell;` (line 14).**

- [ ] **Step 1.1.5: Remove `key: Cell::new(None)` field init in `Sampler::greedy()` (line 92).**

### Step 1.2: Stub-out key-using fns (full removal in T2 after Scheduler plumbing)

- [ ] **Step 1.2.1: Replace `Sampler::sample_async_greedy` body** (line 185-209) with `unimplemented!("removed in 3e.2; use sample_batch (greedy fast path)")` OR delete the fn entirely.

Recommend: **delete `sample_async_greedy`** — grep shows zero callers in production (sample_batch handles greedy):
```bash
grep -rn "sample_async_greedy" /Volumes/Dev/cxx-mlx/ --include="*.rs" | grep -v "fn sample_async_greedy"
```
Expected: 0 hits.

- [ ] **Step 1.2.2: Stub-out `Sampler::sample(&self, logits, history)` (line 211-263).**

Replace body with:
```rust
pub fn sample(&self, _logits: &Array, _history: &[u32]) -> Result<u32> {
    anyhow::bail!("Sampler::sample: API requires &mut Array prng_state in 3e.2; \
                   use sample_batch or sample_row_cpu via Scheduler")
}
```

Will be re-implemented in T2 with new signature.

- [ ] **Step 1.2.3: Remove `ensure_key` and `store_key` methods** (line 196-209).

### Step 1.3: Add static_assertions for trait bounds

- [ ] **Step 1.3.1: Add at top of sampler.rs after imports:**

```rust
// Compile-time verification that Sampler is a pure config POD.
const _: () = {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_clone<T: Clone>() {}
    fn assert_copy<T: Copy>() {}
    let _ = assert_send::<Sampler>;
    let _ = assert_sync::<Sampler>;
    let _ = assert_clone::<Sampler>;
    let _ = assert_copy::<Sampler>;
};
```

### Step 1.4: Adjust callers within sampler.rs

- [ ] **Step 1.4.1: `sample_row_cpu` body uses `sampler.ensure_key()` (line 594).** Replace with bail!:
```rust
fn sample_row_cpu(probs: &[f32], top_p: f32, min_p: f32, _sampler: &Sampler) -> Result<u32> {
    anyhow::bail!("sample_row_cpu: API requires &mut Array prng_state in 3e.2 T2 (Scheduler plumbing)")
}
```

(Real impl in T2 with new signature.)

- [ ] **Step 1.4.2: `configured_pipeline` body still calls `sample_row_cpu` (line 524). It'll bail! until T2.**

### Step 1.5: Compile + tests pass (most should — sample_batch tests will bail!)

- [ ] **Step 1.5.1: cargo build.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -15
```

Expected: clean compile (no errors). Possibly some `unused_variables` warnings — fix by prefixing with `_`.

- [ ] **Step 1.5.2: cargo test --lib (filtered).**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::sampler::is_greedy 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::sampler::sample_batch_validates 2>&1 | tail -10
```

Expected: is_greedy tests PASS (don't touch Sampler::sample). Tests that exercise sample_batch may FAIL with bail! — this is expected for T1 intermediate state.

### Step 1.6: Hygiene + commit T1

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

T1 intermediate state: build PASSES, clippy clean. Some sampler tests will FAIL because Sampler::sample / sample_row_cpu bail!. **This is expected and T2 wires the new path.**

If lib tests pre-T1 baseline was N passing, T1 may have ~M failing (those using sample_batch's configured path or Sampler::sample). Record M in commit message.

```bash
git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
refactor(b1-p2.3e.2-t1): Sampler struct shrink — remove Cell, derive Copy

Sampler becomes pure config POD:
  - Remove `key: Cell<Option<Array>>` field
  - Remove `use std::cell::Cell`
  - Remove manual `Debug` impl (auto-derive)
  - Remove manual `Clone` impl (auto-derive Clone + Copy)
  - Remove `Sampler::ensure_key` + `Sampler::store_key`
  - Delete `Sampler::sample_async_greedy` (zero production callers;
    sample_batch handles greedy via 3e.1a fast path)
  - Stub `Sampler::sample` with bail! (T2 reimplements with
    `prng_state: &mut Array` parameter)
  - Stub `sample_row_cpu` with bail! (T2 reimplements with
    `prng_state_row: &mut Array` parameter replacing `&Sampler`)

Static assertions verify Sampler: Send + Sync + Clone + Copy.

INTERMEDIATE STATE: lib tests using sample_batch's configured path
or Sampler::sample will bail! at runtime. T2 wires Scheduler
prng_state plumbing and restores functionality.

Failing tests count post-T1: <M>/267 (recorded for T2 baseline).

Spec §4.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Scheduler.prng_state` field + plumbing + sample fn rewires

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (add `prng_state: Array` field to `Scheduler` struct; init in `Scheduler::new`; add `init_row_prng`; call in `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner`; pass `&mut self.prng_state` to `sample_batch` calls; update `admit_mid_finalize` to pass single-row slice)
- Modify: `ironmlx/src/core/sampler.rs` (re-implement `sample_row_cpu` with new signature; re-implement `Sampler::sample` with new signature; update `configured_pipeline` to slice prng_state per-row + write back; update `sample_batch` signature)

**Goal:** Restore functionality with centralized PRNG state. After T2, all callers updated + lib tests should PASS except those needing Sampler::with_seed semantic update (T3).

### Step 2.1: `Scheduler.prng_state` field + init

- [ ] **Step 2.1.1: Read `Scheduler` struct location.**

```bash
grep -n "pub struct Scheduler\|pub(crate) struct Scheduler\|impl Scheduler" /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs | head
```

- [ ] **Step 2.1.2: Add field to `Scheduler` struct.**

Add after existing fields:
```rust
    /// Per-row PRNG state. Shape `[b_max, 2]` u32. Row `i` 持 row i 的 mlx
    /// random key. Init from `Sampler.seed` 在 admit 时 (via `init_row_prng`),
    /// advance 在每次 configured_pipeline sample step.
    pub(crate) prng_state: Array,
```

- [ ] **Step 2.1.3: Init in `Scheduler::new`.**

In `Scheduler::new`, after `b_max` is determined:
```rust
    let prng_state = mlx::ops::constructors::zeros(
        &[b_max as i32, 2_i32][..],
        mlx::Dtype::Uint32,
        (),
    )?;
```

Add `prng_state` to struct literal.

- [ ] **Step 2.1.4: Add `Scheduler::init_row_prng` method.**

```rust
    /// Initialize row `row_idx`'s PRNG state from `seed`. Called by
    /// admit paths when a new request occupies the slot.
    fn init_row_prng(&mut self, row_idx: usize, seed: u64) -> Result<()> {
        let key = mlx::random::key(seed)?; // [2] u32
        let key_2d = key.reshape(&[1_i32, 2_i32][..])?; // [1, 2]
        self.prng_state = mlx::ops::indexing::slice_update(
            &self.prng_state,
            &key_2d,
            &[row_idx as i32, 0][..],
            &[row_idx as i32 + 1, 2][..],
        )?;
        Ok(())
    }
```

### Step 2.2: Call `init_row_prng` in admit paths

- [ ] **Step 2.2.1: `admit_inner` (B>0 admit).** After `self.slots[row_idx] = Some(state)`, add:
```rust
    self.init_row_prng(row_idx, req.sampler.seed)?;
```

- [ ] **Step 2.2.2: `admit_mid_inner` / `admit_mid_begin` (mid-batch admit).** Same pattern at row insertion point.

- [ ] **Step 2.2.3: `prefill_admitted_inner` if it does new admissions.** Verify with `grep -n "fn admit\|fn prefill_admitted" /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs`; add call where new slot is occupied.

### Step 2.3: `sample_row_cpu` signature + impl in sampler.rs

- [ ] **Step 2.3.1: Replace stub from T1 Step 1.4.1 with full impl.**

```rust
fn sample_row_cpu(probs: &[f32], top_p: f32, min_p: f32, prng_state_row: &mut Array) -> Result<u32> {
    let vocab = probs.len();
    // 1. Sort (prob, idx) descending — unchanged from 3e.1b
    let mut indexed: Vec<(f32, u32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i as u32))
        .collect();
    indexed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // 2. Top-p nucleus — unchanged
    let mut keep_count = vocab;
    if top_p < 1.0 {
        let mut cum = 0.0_f32;
        for (k, &(p, _)) in indexed.iter().enumerate() {
            if cum >= top_p {
                keep_count = k;
                break;
            }
            cum += p;
        }
    }
    let nucleus = &indexed[..keep_count];

    // 3. Min-p — unchanged
    let max_prob = nucleus.first().map(|&(p, _)| p).unwrap_or(1.0);
    let min_p_thresh = min_p * max_prob;

    // 4. Collect eligible + renormalize — unchanged
    let mut eligible: Vec<(f32, u32)> = nucleus
        .iter()
        .filter(|&&(p, _)| p >= min_p_thresh)
        .copied()
        .collect();
    if eligible.is_empty() {
        return Ok(indexed[0].1);
    }
    let total: f32 = eligible.iter().map(|&(p, _)| p).sum();
    let inv_total = if total > 0.0 { 1.0 / total } else { 1.0 };
    for (p, _) in eligible.iter_mut() {
        *p *= inv_total;
    }

    // 5. CDF sampling — PRNG via prng_state_row (NEW in 3e.2)
    let (next_key, sample_key) = mlx::random::split(prng_state_row)?;
    let u_arr = mlx::random::uniform()
        .shape(1_i32)
        .dtype(mlx::Dtype::Float32)
        .key(&sample_key)
        .sample()?;
    let u: f32 = u_arr.item()?;
    *prng_state_row = next_key;

    let mut cum = 0.0_f32;
    for &(p, idx) in &eligible {
        cum += p;
        if cum > u {
            return Ok(idx);
        }
    }
    // Floating-point rounding fallback
    Ok(eligible.last().unwrap().1)
}
```

### Step 2.4: `configured_pipeline` signature + slice prng_state per row

- [ ] **Step 2.4.1: Add `prng_state: &mut Array` param.**

```rust
fn configured_pipeline(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
    prng_state: &mut Array,  // NEW: [B, 2] u32
) -> Result<Vec<u32>> {
    let dims_owned = logits.shape();
    let dims = dims_owned.as_slice();
    let b = dims[0] as usize;
    let vocab_i32 = dims[1];
    let vocab = vocab_i32 as usize;

    let configs = collect_per_row_configs(samplers, vocab_i32)?;
    let history_count = if configs.need_history {
        Some(build_history_count(histories, vocab)?)
    } else {
        None
    };
    let logits = apply_penalties(logits, history_count.as_ref(), &configs)?;
    let logits = apply_temperature(&logits, &configs.temp)?;
    let logits = apply_top_k_batched(&logits, &configs.top_k)?;
    let probs_gpu = apply_softmax(&logits)?;
    let probs_flat: Vec<f32> = probs_gpu.to_vec()?;
    let top_p_host: Vec<f32> = configs.top_p.to_vec()?;
    let min_p_host: Vec<f32> = configs.min_p.to_vec()?;

    let mut tokens = Vec::with_capacity(b);
    for row in 0..b {
        let row_probs = &probs_flat[row * vocab..(row + 1) * vocab];
        // Slice row's PRNG state.
        let mut row_key = mlx::ops::indexing::slice(
            prng_state,
            &[row as i32, 0][..],
            &[row as i32 + 1, 2][..],
        )?
        .reshape(&[2_i32][..])?;
        let token = sample_row_cpu(row_probs, top_p_host[row], min_p_host[row], &mut row_key)?;
        // Write advanced key back.
        let row_key_2d = row_key.reshape(&[1_i32, 2_i32][..])?;
        *prng_state = mlx::ops::indexing::slice_update(
            prng_state,
            &row_key_2d,
            &[row as i32, 0][..],
            &[row as i32 + 1, 2][..],
        )?;
        tokens.push(token);
    }
    Ok(tokens)
}
```

### Step 2.5: `sample_batch` signature change

- [ ] **Step 2.5.1: Add `prng_state` param.**

```rust
pub fn sample_batch(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
    prng_state: &mut Array,  // NEW
) -> Result<Vec<u32>> {
    // ... validation unchanged ...

    if samplers.iter().all(|s| s.is_greedy()) {
        // all-greedy fast path UNCHANGED (no PRNG needed)
        let ids = reduction::argmax(logits, -1, false)?;
        return Ok(ids.to_vec()?);
    }
    configured_pipeline(samplers, logits, histories, prng_state)
}
```

### Step 2.6: `Sampler::sample` re-implement

- [ ] **Step 2.6.1: Replace stub from T1 Step 1.2.2.**

```rust
pub fn sample(&self, logits: &Array, history: &[u32], prng_state_row: &mut Array) -> Result<u32> {
    // ... original sample logic from 3e.1a era ...
    // ... but replace `self.ensure_key()` with `prng_state_row` usage ...
    // ... applies same per-row logic as sample_row_cpu but for the single-row
    //     Sampler::sample API (used by admit_mid_finalize) ...
    
    // Simplified: call sample_row_cpu's GPU-free path if non-greedy,
    // or argmax if greedy. Re-use sample_row_cpu after converting logits
    // to Vec<f32> (since admit_mid_finalize is B=1, overhead is minor).
    
    // Greedy short-circuit:
    if self.is_greedy() {
        let idx = reduction::argmax(logits, All, false)?;
        return Ok(idx.item::<u32>()?);
    }
    
    // Configured: apply penalties + temp + top_k via existing scalar paths,
    // then call sample_row_cpu equivalent on host.
    // Re-use 3e.1a-era apply_repetition_penalty, apply_freq_presence_penalty,
    // apply_top_k (scalar version line 563), apply_top_p (line 574), apply_min_p (line 583)
    // ... unchanged from 3e.1a era except sample step uses prng_state_row ...
    
    let mut logits = logits.clone();
    if let Some(p) = self.repetition_penalty {
        if !history.is_empty() && (p - 1.0).abs() > f32::EPSILON {
            logits = apply_repetition_penalty(&logits, history, p)?;
        }
    }
    if self.frequency_penalty.unwrap_or(0.0).abs() > f32::EPSILON
        || self.presence_penalty.unwrap_or(0.0).abs() > f32::EPSILON
    {
        let f = self.frequency_penalty.unwrap_or(0.0);
        let pp = self.presence_penalty.unwrap_or(0.0);
        logits = apply_freq_presence_penalty(&logits, history, f, pp)?;
    }
    let inv_t = 1.0_f32 / self.temperature;
    let mut logits = &logits * inv_t;
    if let Some(k) = self.top_k {
        logits = apply_top_k(&logits, k)?;
    }
    if let Some(p) = self.min_p {
        logits = apply_min_p(&logits, p)?;
    }
    if let Some(p) = self.top_p {
        if p < 1.0 {
            logits = apply_top_p(&logits, p)?;
        }
    }

    // categorical sample via prng_state_row
    let (next_key, sample_key) = mlx::random::split(prng_state_row)?;
    let sample = random::categorical(&logits).num_samples(1).key(&sample_key).sample()?;
    *prng_state_row = next_key;
    Ok(sample.item::<u32>()?)
}
```

### Step 2.7: `Scheduler::step` / `prefill_admitted_inner` / `admit_mid_finalize` call site updates

- [ ] **Step 2.7.1: `Scheduler::step` `sample_batch` call.** Add `&mut self.prng_state`:

```rust
let tokens = sample_batch(&samplers, &logits_2d, &histories, &mut self.prng_state)?;
```

- [ ] **Step 2.7.2: `Scheduler::prefill_admitted_inner` `sample_batch` call.** Same.

- [ ] **Step 2.7.3: `Scheduler::admit_mid_finalize` (B=1) sample step.** Slice prng_state for the row:

```rust
let mut row_key = mlx::ops::indexing::slice(
    &self.prng_state,
    &[row_idx as i32, 0][..],
    &[row_idx as i32 + 1, 2][..],
)?
.reshape(&[2_i32][..])?;
let token = state.sampler.sample(&logits_1d, &history, &mut row_key)?;
let row_key_2d = row_key.reshape(&[1_i32, 2_i32][..])?;
self.prng_state = mlx::ops::indexing::slice_update(
    &self.prng_state,
    &row_key_2d,
    &[row_idx as i32, 0][..],
    &[row_idx as i32 + 1, 2][..],
)?;
```

### Step 2.8: Build + run lib tests

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -10
```

Expected: build clean; lib tests should mostly PASS (any test that doesn't depend on bit-exact PRNG token output). Some tests using `Sampler::with_seed(N)` + token equality may FAIL — fix in T3.

### Step 2.9: Hygiene + commit T2

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/src/core/sampler.rs ironmlx/src/core/scheduler.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.2-t2): Scheduler.prng_state + sample_batch &mut Array plumbing

Scheduler now centralizes PRNG state:
  - `Scheduler.prng_state: Array` shape [b_max, 2] u32 (init zeros)
  - `Scheduler::new` zero-inits
  - `init_row_prng(row_idx, seed)` mints mlx::random::key + writes
    via slice_update
  - `admit_inner` / `admit_mid_inner` / `prefill_admitted_inner`
    call init_row_prng on new admission

Sample fn rewires:
  - sample_batch signature gains `prng_state: &mut Array`
  - configured_pipeline gains same; per-row slice + sample_row_cpu
    + slice_update write-back
  - sample_row_cpu signature: (probs, top_p, min_p, prng_state_row: &mut Array)
    replacing &Sampler — splits + samples + advances in-place
  - Sampler::sample signature: gains prng_state_row: &mut Array;
    body uses prng_state_row instead of self.key.ensure_key

admit_mid_finalize (B=1) slices prng_state[row_idx] for single-row
sample, writes back via slice_update.

Lib tests post-T2: <N>/267 PASS (vs T1 baseline <M>/267); failing
tests use Sampler::with_seed(N) + token equality, fixed in T3.

Spec §4.2-4.5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Test updates for PRNG init-time drift

**Files:**
- Modify: `ironmlx/src/core/sampler.rs::mod tests` (update expected tokens or change to statistical assertions for tests using `Sampler::with_seed(N)` + token comparisons)
- Modify: `ironmlx/tests/*.rs` (any integration tests using `Sampler::with_seed` + bit-exact assertions)
- Remove T0 probe tests if cluttering (optional)

**Goal:** Restore 100% test passing. Tests that fail post-T2 due to PRNG init-time drift (mint at admit vs lazy ensure_key) are adjusted.

### Step 3.1: Identify failing tests

- [ ] **Step 3.1.1: Run full lib tests to find failures.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -50
```

List failures. Likely candidates:
- `sample_batch_configured_fallback_no_panic_in_range` — already uses in-range, should pass
- `sample_batch_no_op_default_configured_pipeline_in_range` — already in-range, should pass
- Any test with `Sampler::with_seed(N)` + `assert_eq!(token, X)`

### Step 3.2: For each failing test

- [ ] **Step 3.2.1: Decide strategy per failure**:
  - If test asserts a property (in-range / nucleus contains / no-panic) — leave alone (likely already passing)
  - If test asserts bit-exact token from PRNG-driven sample — choose:
    - **Update expected value** to new PRNG-init-mint output (run test, capture, paste in)
    - **OR convert to statistical** (1000 sample loop + freq histogram + tolerance check)

- [ ] **Step 3.2.2: For each test, apply fix + verify**:

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- <test_name> --nocapture 2>&1 | tail -10
```

### Step 3.3: Verify full lib tests PASS

- [ ] **Step 3.3.1:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -5
```

Expected: `test result: ok. 267+ passed; 0 failed; ...`.

### Step 3.4: Hygiene + commit T3

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add -A
git commit -m "$(cat <<'EOF'
fix(b1-p2.3e.2-t3): test updates for PRNG init-time drift

Post-T2 PRNG init path changed from lazy ensure_key (mint on first
sample call) to admit-time init_row_prng (mint when Scheduler accepts
the admission). For same Sampler.seed, the first sample's PRNG state
differs by how many split-advances have occurred. Tests using
Sampler::with_seed(N) + bit-exact token comparison adjusted:

  - <list each test name + fix strategy (value update or statistical)>

All lib tests PASS post-fix: <N>/267.

Spec NG1 accepts PRNG init-time drift.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Perf gate + sweep + close-out + auto-push

**Files:**
- (existing) `ironmlx/tests/b1_p2_3e_1b_configured_sampler.rs` (perf gate test reused)
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_2_closeout/report.md`

### Step 4.1: Run perf gate (3e.1b's, reused)

- [ ] **Step 4.1.1:**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  MLX_DIR=$HOME/.local/mlx \
  cargo +stable test --release --test b1_p2_3e_1b_configured_sampler -- --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/3e_2_perf.log | tail -20
```

Expected: PASS (max median ≤ 250ms, ratio ≤ 2×). 3e.2 should match 3e.1b perf (~82ms median).

### Step 4.2: Sweep smoke

```bash
./scripts/sweep/sweep_smoke.sh \
  --suites b1_p2_3b_2_scheduler_actor \
           b1_p2_4_batched_vl::mid_admit_vl_during_text_decode \
           b1_p2_3e_1a_vectorize_greedy::b1_p2_3e_1a_greedy_decode_speedup \
           b1_p2_3e_1b_configured_sampler::b1_p2_3e_1b_configured_decode_speedup \
  2>&1 | tee /tmp/3e_2_smoke.log | tail -20
```

Expected: 4 PASS.

### Step 4.3: Sweep_full background

```bash
bash ./scripts/sweep/sweep_full.sh > /tmp/3e_2_sweep_full.log 2>&1 &
echo "PID: $!" > /tmp/3e_2_sweep_full.pid
```

### Step 4.4: Write close-out report

- [ ] **Step 4.4.1: Create:**

Path: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_2_closeout/report.md`

Template:
```markdown
# B1-p2.3e.2 PRNG Key Centralization — Close-out

**Branch:** `ironmlx-b1-p2-3e2-prng-centralization`
**Base:** `ironmlx-b1-p2-3e1b-vectorize-configured` HEAD `d146d60`
**Spec:** `docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md` (`d4c0fc3`)
**Plan:** `docs/superpowers/plans/2026-05-18-b1-p2-3e-2-prng-key-centralization.md`

## Goal Recap

Remove `Sampler.key: Cell<Option<Array>>` field. Centralize per-row
PRNG state in `Scheduler.prng_state: Array` ([b_max, 2] u32). Auto-
derive `Sampler: Clone + Copy + Send + Sync`.

## Commits

- `<T0 SHA>` chore(b1-p2.3e.2-t0): mlx slice_update probe + bench
- `<T1 SHA>` refactor(b1-p2.3e.2-t1): Sampler struct shrink — remove Cell, derive Copy
- `<T2 SHA>` feat(b1-p2.3e.2-t2): Scheduler.prng_state + sample_batch &mut Array plumbing
- `<T3 SHA>` fix(b1-p2.3e.2-t3): test updates for PRNG init-time drift
- `<T4 close-out SHA>` docs(b1-p2.3e.2-t4): close-out report

## Acceptance Gates

| Gate | Result | Notes |
| --- | --- | --- |
| Sampler: Send + Sync + Clone + Copy | ✅ static_assertions PASS | T1 |
| cargo test --lib | ✅ 267 PASS | post-T3 |
| Hygiene (fmt / clippy / build) | ✅ all green | every commit |
| 3e.1b perf gate (b1_p2_3e_1b_configured_decode_speedup) | ✅ PASS | <实际数字> |
| sweep_smoke (4 suites) | ✅ PASS | |
| sweep_full (17 suites) | <in progress / N/N> | Appendix |

## Performance Characterization

- 3e.1b perf gate: 82.57ms median
- 3e.2 perf gate: <实测> ms median (expected ~82ms ±10%; centralization
  removes per-row Cell.take/set ~5µs × B = 20µs at B=4 ≈ 0.02% gain)

## Architecture Notes

### Sampler struct (post-3e.2)
Pure config POD. No interior mutability. Auto-derived Clone + Copy +
Send + Sync. PRNG state moved to Scheduler.

### Scheduler.prng_state
Shape `[b_max, 2]` u32 zero-init in `Scheduler::new`. Row `i` populated
on admit via `init_row_prng(i, sampler.seed) → mlx::random::key(seed)`
+ `slice_update`. Eviction does not clear; next admit overwrites.

### sample_row_cpu signature
(probs, top_p, min_p, prng_state_row: &mut Array) — no longer takes
&Sampler. PRNG advance: `mlx::random::split(prng_state_row) → (next, sample_key)`,
write next back to *prng_state_row, use sample_key for uniform draw.

### configured_pipeline per-row plumbing
Slice prng_state[row, :] → call sample_row_cpu → slice_update write back.
Per-call overhead from slice/slice_update measured at <X> µs (T0 bench).

## Carry-Forward

- **Qwen3.5 MoE** — main path forward; sampler vectorization series complete
- (Optional) GPU sample re-enable if mlx ships scatter_along_axis + optimized
  categorical Metal kernel cache
- (Optional) Sweep_full hygiene (cooldown / shard for parallel runs)
```

- [ ] **Step 4.4.2: Commit close-out.**

```bash
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_2_closeout/report.md
git commit -m "$(cat <<'EOF'
docs(b1-p2.3e.2-t4): close-out report

Documents 3e.2 4-commit shape, acceptance gates, perf
characterization (expected unchanged from 3e.1b), and
architecture (Sampler shrink, Scheduler.prng_state, per-row
slice plumbing).

sweep_full running in background; controller appends result.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 4.5: Wait for sweep_full + addendum (don't push yet — T5 still pending)

- [ ] **Step 4.5.1: Wait for sweep_full to complete.**

Monitor `/tmp/3e_2_sweep_full.log` for "full sweep done" line.

- [ ] **Step 4.5.2: Append sweep_full result to close-out + commit addendum.**

(Push happens after T5 — keep T5 work on same branch.)

---

## Task 5: `put_along_axis` exposure in cxx-mlx + apply_top_p_batched refactor

**Files:**
- Modify: `cxx-mlx/mlx-sys/` (FFI bridge — exact path determined by inspecting `mlx-sys/src/lib.rs` or `cxx_bridge!` macro location)
- Modify: `cxx-mlx/mlx/src/ops/indexing.rs` (safe Rust wrapper + unit test)
- Modify: `cxx-mlx/ironmlx/src/core/sampler.rs` (refactor `apply_top_p_batched` `#[cfg(test)]` impl to use `put_along_axis`)

**Goal:** Expose `mlx::put_along_axis` (Apple C++ MLX `ops.h:1089`) through cxx-mlx Rust binding. Refactor 3e.1b's `apply_top_p_batched` (GPU impl, `#[cfg(test)]`) to use `put_along_axis` instead of inverse-permutation workaround. Production CPU path unchanged. Sets up 80% prep for future 3e.3 GPU re-enable.

**Note:** This task touches the cxx-mlx workspace member `mlx-sys` and `mlx` crates (not just `ironmlx`). All three are workspace members so single `cargo build --release` covers them.

### Step 5.1: Inspect mlx-sys FFI bridge layout

- [ ] **Step 5.1.1:**

```bash
ls -la /Volumes/Dev/cxx-mlx/mlx-sys/
find /Volumes/Dev/cxx-mlx/mlx-sys -name "*.rs" -o -name "*.hpp" -o -name "*.cpp" 2>/dev/null | head -20
grep -rn "take_along_axis" /Volumes/Dev/cxx-mlx/mlx-sys/ 2>/dev/null | head -10
```

Identify:
- Location of cxx bridge declarations (likely `mlx-sys/src/lib.rs` or `mlx-sys/src/ops.rs`)
- Location of C++ wrapper implementations (likely `mlx-sys/src/*.cc` or `mlx-sys/include/*.hpp`)
- The existing `take_along_axis` FFI is the template to mirror

### Step 5.2: Add `put_along_axis` to FFI bridge

- [ ] **Step 5.2.1: Locate the `take_along_axis` declaration** (mirror it):

```bash
grep -B 2 -A 10 "fn take_along_axis" /Volumes/Dev/cxx-mlx/mlx-sys/src/*.rs
```

- [ ] **Step 5.2.2: Add `put_along_axis` declaration in cxx bridge** (mirror take_along_axis):

The bridge typically looks like:
```rust
#[cxx::bridge]
mod ffi {
    extern "C++" {
        // ... existing ...
        unsafe fn take_along_axis(...) -> Result<UniquePtr<MlxArray>>;
        // ADD:
        unsafe fn put_along_axis(
            a: &MlxArray,
            indices: &MlxArray,
            values: &MlxArray,
            axis: i32,
            has_target: bool,
            device_only: bool,
            device_type: i32,
            stream_idx: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
```

(Exact signature depends on existing patterns in the FFI bridge — copy `take_along_axis` style precisely.)

- [ ] **Step 5.2.3: Add C++ wrapper impl** in `mlx-sys/src/*.cc` or header:

```cpp
extern "C" std::unique_ptr<MlxArray> put_along_axis(
    const MlxArray& a,
    const MlxArray& indices,
    const MlxArray& values,
    int32_t axis,
    bool has_target,
    bool device_only,
    int32_t device_type,
    int32_t stream_idx
) {
    mlx::core::StreamOrDevice s = make_stream_or_device(has_target, device_only, device_type, stream_idx);
    auto result = mlx::core::put_along_axis(a.inner, indices.inner, values.inner, axis, s);
    return std::make_unique<MlxArray>(std::move(result));
}
```

(Mirror the existing `take_along_axis` C++ wrapper exactly.)

### Step 5.3: Add safe Rust wrapper in `cxx-mlx/mlx/src/ops/indexing.rs`

- [ ] **Step 5.3.1: Locate existing `take_along_axis` wrapper** (~line 79):

```rust
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    // ... existing ...
}
```

- [ ] **Step 5.3.2: Add `put_along_axis` after it:**

```rust
/// Put `values` into `a` along `axis` at positions given by `indices`.
/// Returns a new array (a is not mutated).
///
/// Equivalent to numpy's `np.put_along_axis(a, indices, values, axis)`.
/// Inverse of `take_along_axis`.
///
/// Wraps `mlx::core::put_along_axis` (Apple C++ MLX `ops.h:1089`).
pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
    put_along_axis_on(a, indices, values, axis, ())
}

pub fn put_along_axis_on(
    a: &Array,
    indices: &Array,
    values: &Array,
    axis: i32,
    target: impl Into<crate::StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = unsafe {
        mlx_sys::ops::ffi::put_along_axis(
            a.as_inner(),
            indices.as_inner(),
            values.as_inner(),
            axis,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
```

(Match the exact patterns from `take_along_axis_on` — same target encoding, same Result type.)

### Step 5.4: Add unit test in `cxx-mlx/mlx/src/ops/indexing.rs::mod tests` (or wherever test module is)

```rust
#[test]
fn put_along_axis_round_trip() {
    use crate::Array;
    // [2, 3] base, scatter into via [2, 3] u32 indices
    let base: Array = (&[0_f32; 6][..], &[2_i32, 3_i32][..]).try_into().unwrap();
    // indices identity: each pos writes to itself
    let idx: Array = (&[0_u32, 1, 2, 0, 1, 2][..], &[2_i32, 3_i32][..]).try_into().unwrap();
    let values: Array = (&[1_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2_i32, 3_i32][..])
        .try_into()
        .unwrap();
    let out = put_along_axis(&base, &idx, &values, -1).expect("put");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn put_along_axis_inverse_of_take() {
    use crate::Array;
    let probs: Array = (&[0.5_f32, 0.05, 0.15, 0.3, 0.2, 0.15, 0.1, 0.05][..], &[2_i32, 4_i32][..])
        .try_into()
        .unwrap();
    // argsort gives ascending order; take_along_axis(probs, argsort) gives sorted
    let sort_idx = crate::ops::sort::argsort(&probs, -1).expect("argsort");
    let sorted = crate::ops::indexing::take_along_axis(&probs, &sort_idx, -1).expect("take");
    // put_along_axis with same sort_idx should restore
    let zeros: Array = (&[0_f32; 8][..], &[2_i32, 4_i32][..]).try_into().unwrap();
    let restored = put_along_axis(&zeros, &sort_idx, &sorted, -1).expect("put");
    let v: Vec<f32> = restored.to_vec().expect("to_vec");
    let orig: Vec<f32> = probs.to_vec().expect("to_vec orig");
    for (g, o) in v.iter().zip(orig.iter()) {
        assert!((g - o).abs() < 1e-5, "got {g}, orig {o}");
    }
}
```

### Step 5.5: Run mlx crate's own tests + build

- [ ] **Step 5.5.1:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo +stable test --release -p mlx -- put_along_axis 2>&1 | tail -10
```

Expected: build clean; 2 new put_along_axis tests PASS.

### Step 5.6: Refactor `apply_top_p_batched` in sampler.rs to use put_along_axis

- [ ] **Step 5.6.1: Locate `apply_top_p_batched`** (line ~672 in sampler.rs, `#[cfg(test)]` block since 3e.1b T4).

Read existing impl. The scatter-back is currently:
```rust
let inv_perm = argsort(&sort_idx_desc, -1)?;
take_along_axis(&sorted_masked, &inv_perm, -1)
```

- [ ] **Step 5.6.2: Replace with `put_along_axis`:**

```rust
// Scatter sorted_masked back to vocab order via put_along_axis.
// Equivalent to: out[sort_idx[i, j]] = sorted_masked[i, j]
let zeros = mlx::ops::constructors::zeros_like(probs)?;
mlx::ops::indexing::put_along_axis(&zeros, &sort_idx_desc, &sorted_masked, -1)
```

(Verify `zeros_like` exists or use `(&[0_f32; B*vocab][..], &[B, vocab][..]).try_into()?` equivalent.)

- [ ] **Step 5.6.3: Run apply_top_p_batched unit test verify still PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- apply_top_p_batched 2>&1 | tail -10
```

Expected: PASS (functional output identical to inverse-permutation workaround).

### Step 5.7: Hygiene + commit T5

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add cxx-mlx/mlx-sys cxx-mlx/mlx/src/ops/indexing.rs ironmlx/src/core/sampler.rs
# (adjust paths above to match actual file locations)
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.2-t5): expose put_along_axis + refactor apply_top_p_batched

Apple C++ MLX has `mlx::put_along_axis` (ops.h:1089) since long ago,
but cxx-mlx Rust binding didn't expose it. Adds:
  - mlx-sys: cxx FFI bridge (mirror take_along_axis pattern)
  - mlx: safe Rust wrapper `put_along_axis(a, indices, values, axis)`
    + `put_along_axis_on` (stream variant), in src/ops/indexing.rs
  - 2 unit tests in mlx crate (identity round-trip + inverse of take)

Refactors ironmlx apply_top_p_batched (#[cfg(test)] GPU impl) to use
put_along_axis for scatter-back, replacing the inverse-permutation
workaround `argsort(sort_idx) → take_along_axis` (which T0 §0.2
verified mathematically but was clunky). apply_top_p_batched unit
test still PASS (functional output identical).

This is Boss-approved Option C scope expansion in 3e.2: prep 80% of
the work to enable future 3e.3 GPU sample re-enable, but
DO NOT switch production path back to GPU (configured_pipeline
stays CPU). Future 3e.3 sub-tasks:
  - Re-bench GPU apply_top_p_batched isolated (with put_along_axis)
  - Investigate mlx::random::categorical Metal kernel cache cost
  - If both wins → re-enable GPU production path

Spec ref: docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md §8.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 5.8: Push 3e.2 branch (auto-push per Boss Q4)

After T5 commit + close-out has been finalized:

```bash
git push -u origin ironmlx-b1-p2-3e2-prng-centralization
```

---

## Self-Review Checklist (controller, post-implementation)

After all 4 tasks complete:

1. **Spec coverage:**
   - Spec §4.1 Sampler shrink → T1
   - Spec §4.2 Scheduler.prng_state + init_row_prng → T2
   - Spec §4.3 sample_row_cpu signature → T2
   - Spec §4.4 configured_pipeline plumbing → T2
   - Spec §4.5 sample_batch signature → T2
   - Spec §4.6 admit_mid_finalize → T2
   - Spec §6 acceptance → T3 + T4
   - Spec §7 R1-R6 → R1 T3, R2 noted in init_row_prng overwrite, R3 T0 verified, R4 admit_mid_finalize handled, R5 noted, R6 static_assertions

2. **No placeholders:** every step has real code.

3. **Type consistency:**
   - `prng_state: &mut Array` signature consistent across `sample_batch` / `configured_pipeline` / `sample_row_cpu` / `Sampler::sample`
   - `Sampler` field set consistent between T1 struct def and T2 callers

4. **No compat code:** Cell completely removed; no `Option<Cell>` wrapper; no parallel "old path".

5. **Hygiene gate at every commit:** explicit in each Task §N.

6. **Boss constraints:** Chinese in user-facing messages, frequent commits, MLX_DIR set, no amend / no --no-verify / no force push, auto-push after sweep_full PASS (Q4).
