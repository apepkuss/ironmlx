# B1-p2.3e.1b Vectorize Configured Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `sample_batch` 的 mixed/configured fallback path (现 per-row `Sampler::sample` 循环) with a fully batched `[B, vocab]` pipeline: 7 logits/probability ops (rep+freq+pres fused / temp / top_k via partition / softmax / top_p via argsort+inverse perm / min_p / renormalize) + batched `mlx::random::categorical` sample step. All-greedy fast path retained (3e.1a). Mixed batch performance: B sequential dispatches → 7-8 batched dispatches + 1 sync; per-step sampler block 16-32ms → 3-6ms at B=4.

**Architecture:** New `configured_pipeline` free function in `core/sampler.rs`. `sample_batch` routes by `is_greedy()`: all-greedy → 3e.1a fast path unchanged; otherwise → `configured_pipeline`. Per-row config 通过 broadcasted no-op default value 控制启用 (temp=1 / top_p=1 / top_k=vocab / min_p=0 / rep_pen=1 / freq_pen=0 / pres_pen=0 全 identity). Per-row history → CPU bincount → upload [B, vocab] u32. Sampler struct unchanged (3e.2 will centralize PRNG).

**Tech Stack:** Rust, mlx Rust binding (`/Volumes/Dev/cxx-mlx/mlx`), Qwen3.5-4B-MLX-4bit fixture.

**Spec ref:** [`docs/superpowers/specs/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md`](../specs/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md) (commit `92b5b50`).

**Branch target:** `ironmlx-b1-p2-3e1b-vectorize-configured` cut from `ironmlx-b1-p2-3e1a-vectorize-greedy` HEAD (commit `461240a` plan or later; verify with `git rev-parse HEAD`). Note: 3e.1a branch has docs ancestors (`ab4c839` / `92b5b50` / `978d288` / `461240a`) carried into 3e.1b — `git push` of 3e.1b will publish them as well; this is intentional.

---

## Pre-flight

### Step 0: Branch + baseline gates

- [ ] **Step 0.1: Cut branch.**

```bash
cd /Volumes/Dev/cxx-mlx
git switch ironmlx-b1-p2-3e1a-vectorize-greedy
git rev-parse HEAD  # expect 978d288
git switch -c ironmlx-b1-p2-3e1b-vectorize-configured
```

- [ ] **Step 0.2: Pre-flight hygiene gate PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three exit 0. If not, 3e.1a baseline is broken — stop and report.

- [ ] **Step 0.3: Baseline `cargo test --lib` PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -3
```

Expected: `test result: ok. 249 passed; 0 failed; 6 ignored; ...` (post-3e.1a baseline).

---

## Task 0: mlx API verification + design choices

**Files:**
- Read-only: `/Volumes/Dev/cxx-mlx/mlx/src/random/{categorical.rs,state.rs}`, `/Volumes/Dev/cxx-mlx/mlx/src/ops/{sort.rs,indexing.rs,cumulative.rs,shape.rs}`
- Output: comment block in `core/sampler.rs` documenting verified API + chosen path for top_p scatter and top_k impl

**Goal:** Bench / verify the 3 risk points (R1 categorical batched key, R2 partition perf, R3 scatter equivalent) before writing real code. Pin design decisions in a comment block so T1-T3 implementers have unambiguous reference.

### Step 0.0: Verify `mlx::random::categorical` batched behavior

- [ ] **Step 0.0.1: Read categorical builder source.**

```bash
sed -n '60,120p' /Volumes/Dev/cxx-mlx/mlx/src/random/categorical.rs
```

Confirm: builder has `.key(&Array)`, `.axis(i32)`, `.num_samples(n)`, `.sample()`. Default output for `[B, vocab]` logits + `axis=-1` is `[B]`.

- [ ] **Step 0.0.2: Write 1-shot probe binary or unit test.**

Add to `ironmlx/src/core/sampler.rs::mod tests`:

```rust
#[test]
fn probe_categorical_batched_single_key_independent_rows() {
    use mlx::random;
    // Build [B=4, vocab=8] logits where each row has its argmax at a different col.
    let mut data: Vec<f32> = vec![0.0; 32];
    for i in 0..4 { data[i * 8 + i] = 100.0; }  // row i argmax at col i
    let logits: Array = (&data[..], &[4_i32, 8_i32][..]).try_into().expect("logits");
    let key = random::key(42).expect("key");
    let tokens = random::categorical(&logits).key(&key).sample().expect("sample");
    assert_eq!(tokens.shape().as_slice(), &[4], "categorical([B,vocab]) → [B]");
    let v: Vec<u32> = tokens.to_vec().expect("to_vec");
    // Each row's argmax dominates → categorical concentrates on that col.
    assert_eq!(v, vec![0, 1, 2, 3], "row i should sample col i (skewed logits)");
}
```

- [ ] **Step 0.0.3: Run probe.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- probe_categorical_batched_single_key_independent_rows --nocapture 2>&1 | tail -10
```

Expected: PASS. Confirms R1 — categorical accepts single key + auto row-independent.

### Step 0.1: Verify `mlx::ops::partition` API + perf vs `sort`

- [ ] **Step 0.1.1: Read partition signature.**

```bash
sed -n '20,50p' /Volumes/Dev/cxx-mlx/mlx/src/ops/sort.rs
```

Confirm: `pub fn partition(a: &Array, kth: i32, axis: i32) -> Result<Array>`.

- [ ] **Step 0.1.2: Add benchmark probe test (only at vocab=151936 if feasible).**

```rust
#[test]
#[ignore]  // bench-mode, run on demand
fn probe_sort_vs_partition_vocab_151k() {
    use mlx::ops::sort;
    use std::time::Instant;
    let b = 4usize;
    let vocab = 151936usize;
    let data: Vec<f32> = (0..b * vocab).map(|i| (i as f32).sin()).collect();
    let arr: Array = (&data[..], &[b as i32, vocab as i32][..]).try_into().unwrap();
    arr.eval().unwrap();

    let t0 = Instant::now();
    let sorted = sort::sort(&arr, -1).unwrap(); sorted.eval().unwrap();
    let dt_sort = t0.elapsed();

    let t1 = Instant::now();
    let parted = sort::partition(&arr, (vocab - 50) as i32, -1).unwrap(); parted.eval().unwrap();
    let dt_part = t1.elapsed();

    eprintln!("[T0 bench] sort=[B=4,vocab=151936] {dt_sort:?} | partition(kth=151886) {dt_part:?}");
}
```

- [ ] **Step 0.1.3: Run bench.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- probe_sort_vs_partition_vocab_151k --ignored --nocapture 2>&1 | tail -10
```

**Decision rule:**
- If `sort` < 3ms at B=4 vocab=151k → top_k uses `sort + take_along_axis(threshold)` (R2 cleared)
- If `sort` ≥ 3ms → top_k uses `partition(kth=vocab-top_k_max)` (R2 mitigation)

Document chosen path in §0.3 comment block.

### Step 0.2: Verify scatter equivalent (argsort inverse permutation + take_along_axis)

- [ ] **Step 0.2.1: Probe inverse permutation identity.**

```rust
#[test]
fn probe_argsort_inverse_permutation_identity() {
    use mlx::ops::{sort, indexing};
    let b = 2usize;
    let vocab = 8usize;
    // probs row 0: descending order is reversed of natural; row 1: random.
    let data: Vec<f32> = vec![
        0.1, 0.05, 0.2, 0.3, 0.05, 0.1, 0.1, 0.1, // row 0
        0.2, 0.15, 0.1, 0.05, 0.1, 0.15, 0.15, 0.1, // row 1
    ];
    let probs: Array = (&data[..], &[b as i32, vocab as i32][..]).try_into().unwrap();
    // argsort gives ascending order; for descending, negate first or argsort then reverse.
    // Verify: idx = argsort(probs); inv = argsort(idx); take_along_axis(idx, inv) == arange(vocab)
    let idx = sort::argsort(&probs, -1).unwrap();
    let inv = sort::argsort(&idx, -1).unwrap();
    let arange: Array = (&(0..vocab as i32).collect::<Vec<_>>()[..], &[vocab as i32][..])
        .try_into().unwrap();
    // For each row: take_along_axis(idx, inv, -1) should produce broadcast(arange, [B, vocab])
    let got = indexing::take_along_axis(&idx, &inv, -1).unwrap();
    let got_row0: Vec<i32> = got.to_vec().unwrap();
    let expected: Vec<i32> = (0..vocab as i32).chain(0..vocab as i32).collect();
    assert_eq!(got_row0, expected, "inverse permutation identity failed");
}
```

- [ ] **Step 0.2.2: Run probe.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- probe_argsort_inverse_permutation_identity --nocapture 2>&1 | tail -10
```

Expected: PASS. R3 cleared — scatter back via inverse permutation works.

### Step 0.3: Pin design decisions

- [ ] **Step 0.3.1: Add comment block at top of `core/sampler.rs::configured_pipeline` (define stub function first, body added in T1-T3).**

```rust
/// Configured-sampler vectorized pipeline. Called by [`sample_batch`]
/// when not all rows are greedy. See spec
/// `docs/superpowers/specs/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md`.
///
/// **mlx API verification (T0, plan §Step 0):**
/// - `mlx::random::categorical(logits=[B,vocab]).key(&single_key).sample() → [B]`
///   — single key + automatic row-independent batching. Per-row PRNG
///   reproducibility (each Sampler having its own seed) is NOT preserved
///   by the batched op; spec NG6 accepts this drift.
/// - `mlx::ops::sort::partition(kth, axis)` and `sort(axis)` both exist;
///   plan T0 §0.1 chose: <SORT_OR_PARTITION>.  (filled by T0 implementer)
/// - `scatter_along_axis` not exposed in mlx Rust binding; top_p scatter
///   back uses `argsort(sort_idx) = inverse permutation` then
///   `take_along_axis(sorted_masked, inv_perm, -1)`. Verified in
///   `probe_argsort_inverse_permutation_identity`.
fn configured_pipeline(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    anyhow::bail!("configured_pipeline: not yet implemented (3e.1b T1-T3)")
}
```

- [ ] **Step 0.3.2: Hygiene + commit T0.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
chore(b1-p2.3e.1b-t0): mlx API verification + design pins

Three probe tests added (kept in mod tests):
  - probe_categorical_batched_single_key_independent_rows (R1)
  - probe_sort_vs_partition_vocab_151k (#[ignore], R2 bench)
  - probe_argsort_inverse_permutation_identity (R3)

All pass. configured_pipeline stub added with verified-API
comment block + chosen top_k path (sort vs partition decided by
§0.1 bench).

Spec ref: §8 Open Questions resolved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Per-row configs + history bincount + apply_penalties (fused rep+freq+pres)

**Files:**
- Modify: `ironmlx/src/core/sampler.rs` (add `PerRowConfigs` struct + `collect_per_row_configs` + `build_history_count` + `apply_penalties` + 4 unit tests)

**Goal:** Implement the "input plumbing + first op" of `configured_pipeline`. Stand-alone testable; T1 does NOT call `configured_pipeline` from `sample_batch` yet (that integration is T3).

### Step 1.1: Add `PerRowConfigs` struct + `collect_per_row_configs`

- [ ] **Step 1.1.1: Insert before `configured_pipeline` stub:**

```rust
/// Per-row config tensors used by [`configured_pipeline`]. Each
/// field is shape `[B]`; no-op defaults make the corresponding op
/// behave as identity for rows that don't need it.
struct PerRowConfigs {
    /// `[B] f32`. None → 1.0 (identity divisor).
    temp: Array,
    /// `[B] i32`. None → `vocab_size` (no clip).
    top_k: Array,
    /// `[B] f32`. None → 1.0 (no nucleus cut).
    top_p: Array,
    /// `[B] f32`. None → 0.0 (no min_p floor).
    min_p: Array,
    /// `[B] f32`. None → 1.0 (no repetition penalty).
    rep_pen: Array,
    /// `[B] f32`. None → 0.0 (no frequency penalty).
    freq_pen: Array,
    /// `[B] f32`. None → 0.0 (no presence penalty).
    pres_pen: Array,
    /// True if any row has rep_pen / freq_pen / pres_pen set —
    /// drives the history-bincount short-circuit.
    need_history: bool,
}

fn collect_per_row_configs(samplers: &[&Sampler], vocab: i32) -> Result<PerRowConfigs> {
    let b = samplers.len();
    let mut temp = Vec::with_capacity(b);
    let mut top_k = Vec::with_capacity(b);
    let mut top_p = Vec::with_capacity(b);
    let mut min_p = Vec::with_capacity(b);
    let mut rep_pen = Vec::with_capacity(b);
    let mut freq_pen = Vec::with_capacity(b);
    let mut pres_pen = Vec::with_capacity(b);
    let mut need_history = false;
    for s in samplers {
        // temperature: <=0 means greedy in per-row Sampler::sample, but
        // configured_pipeline is only entered when batch is mixed.
        // Greedy rows in a mixed batch use temp=1.0 (no-op).
        temp.push(if s.temperature > 0.0 { s.temperature } else { 1.0 });
        top_k.push(s.top_k.unwrap_or(vocab));
        top_p.push(s.top_p.unwrap_or(1.0));
        min_p.push(s.min_p.unwrap_or(0.0));
        rep_pen.push(s.repetition_penalty.unwrap_or(1.0));
        freq_pen.push(s.frequency_penalty.unwrap_or(0.0));
        pres_pen.push(s.presence_penalty.unwrap_or(0.0));
        if s.repetition_penalty.is_some()
            || s.frequency_penalty.is_some()
            || s.presence_penalty.is_some()
        {
            need_history = true;
        }
    }
    let dim = &[b as i32][..];
    Ok(PerRowConfigs {
        temp: (&temp[..], dim).try_into()?,
        top_k: (&top_k[..], dim).try_into()?,
        top_p: (&top_p[..], dim).try_into()?,
        min_p: (&min_p[..], dim).try_into()?,
        rep_pen: (&rep_pen[..], dim).try_into()?,
        freq_pen: (&freq_pen[..], dim).try_into()?,
        pres_pen: (&pres_pen[..], dim).try_into()?,
        need_history,
    })
}
```

- [ ] **Step 1.1.2: Add unit test.**

```rust
#[test]
fn collect_per_row_configs_defaults_and_overrides() {
    let s1 = Sampler::greedy().with_temperature(0.7);
    let s2 = Sampler::greedy().with_top_p(0.9).with_repetition_penalty(1.1);
    let s3 = Sampler::greedy();
    let samplers: Vec<&Sampler> = vec![&s1, &s2, &s3];
    let cfg = collect_per_row_configs(&samplers, 32000).expect("collect");
    let temp: Vec<f32> = cfg.temp.to_vec().expect("temp vec");
    assert_eq!(temp, vec![0.7, 1.0, 1.0]); // s2/s3 default to 1.0
    let top_p: Vec<f32> = cfg.top_p.to_vec().expect("top_p vec");
    assert_eq!(top_p, vec![1.0, 0.9, 1.0]);
    let rep: Vec<f32> = cfg.rep_pen.to_vec().expect("rep vec");
    assert_eq!(rep, vec![1.0, 1.1, 1.0]);
    let top_k: Vec<i32> = cfg.top_k.to_vec().expect("top_k vec");
    assert_eq!(top_k, vec![32000, 32000, 32000]); // all None → vocab_size
    assert!(cfg.need_history); // s2 has rep_pen
}
```

- [ ] **Step 1.1.3: Run unit test to verify PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- collect_per_row_configs_defaults_and_overrides --nocapture 2>&1 | tail -10
```

Expected: PASS.

### Step 1.2: Add `build_history_count`

- [ ] **Step 1.2.1: Insert after `collect_per_row_configs`:**

```rust
/// Build `[B, vocab] u32` count tensor from per-row histories. CPU
/// bincount → device upload. Cost dominated by upload at large vocab
/// (vocab=151k × B=4 = 2.4 MB ~80µs on Apple Silicon UMA).
fn build_history_count(histories: &[&[u32]], vocab: usize) -> Result<Array> {
    let b = histories.len();
    let mut flat = vec![0_u32; b * vocab];
    for (row, hist) in histories.iter().enumerate() {
        let offset = row * vocab;
        for &tok in *hist {
            let idx = tok as usize;
            if idx < vocab {
                flat[offset + idx] = flat[offset + idx].saturating_add(1);
            }
        }
    }
    let arr: Array = (&flat[..], &[b as i32, vocab as i32][..]).try_into()?;
    Ok(arr)
}
```

- [ ] **Step 1.2.2: Add unit test.**

```rust
#[test]
fn build_history_count_per_row_bincount() {
    let h0: &[u32] = &[3, 3, 5];
    let h1: &[u32] = &[7];
    let h2: &[u32] = &[];
    let histories: Vec<&[u32]> = vec![h0, h1, h2];
    let counts = build_history_count(&histories, 8).expect("counts");
    let v: Vec<u32> = counts.to_vec().expect("to_vec");
    // row 0: [0,0,0,2,0,1,0,0]; row 1: [0,0,0,0,0,0,0,1]; row 2: zeros
    assert_eq!(&v[0..8], &[0, 0, 0, 2, 0, 1, 0, 0]);
    assert_eq!(&v[8..16], &[0, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(&v[16..24], &[0; 8]);
}
```

- [ ] **Step 1.2.3: Run unit test.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- build_history_count_per_row_bincount 2>&1 | tail -5
```

Expected: PASS.

### Step 1.3: Add `apply_penalties` (fused rep + freq + pres)

- [ ] **Step 1.3.1: Insert after `build_history_count`:**

```rust
/// Apply repetition + frequency + presence penalties as a single
/// fused op over `[B, vocab]` logits. Returns updated logits (or
/// `logits.clone()` if `history_count.is_none()`).
fn apply_penalties(
    logits: &Array,
    history_count: Option<&Array>,
    configs: &PerRowConfigs,
) -> Result<Array> {
    use mlx::ops::{binary, cast, indexing::where_};
    let Some(history_count) = history_count else {
        // Short-circuit: no row needs history-based penalties.
        return Ok(logits.clone());
    };
    let history_count_f32 = history_count.cast(mlx::Dtype::Float32)?;
    let zero_u32: Array = 0_u32.try_into()?;
    let history_mask_bool = history_count.gt(&zero_u32)?; // [B, vocab] bool
    let history_mask_f32 = history_mask_bool.cast(mlx::Dtype::Float32)?;

    // Broadcast helpers — reshape [B] → [B, 1]
    let b = logits.shape().as_slice()[0];
    let rep_pen_bv = configs.rep_pen.reshape(&[b, 1][..])?;
    let freq_pen_bv = configs.freq_pen.reshape(&[b, 1][..])?;
    let pres_pen_bv = configs.pres_pen.reshape(&[b, 1][..])?;

    // Repetition: where(logit > 0, logit / rep_pen, logit * rep_pen) for seen tokens
    let one_f32: Array = 1.0_f32.try_into()?;
    let rep_inv_bv = (&one_f32).div(&rep_pen_bv)?; // [B, 1]
    let zero_f32: Array = 0.0_f32.try_into()?;
    let positive_logit_mask = logits.gt(&zero_f32)?; // [B, vocab] bool
    // rep_factor[i, v] = rep_inv[i] if logits[i,v]>0 else rep_pen[i]
    let rep_factor = where_(&positive_logit_mask, &rep_inv_bv, &rep_pen_bv)?;
    let logits_rep_full = logits.mul(&rep_factor)?;
    let logits_rep = where_(&history_mask_bool, &logits_rep_full, logits)?;

    // Frequency: logit -= freq_pen * count
    let freq_term = freq_pen_bv.mul(&history_count_f32)?;
    let logits_freq = logits_rep.sub(&freq_term)?;

    // Presence: logit -= pres_pen * (history_mask as f32)
    let pres_term = pres_pen_bv.mul(&history_mask_f32)?;
    let logits_pres = logits_freq.sub(&pres_term)?;

    let _ = binary::dummy_link(); // keep import even if compiler drops above ops
    Ok(logits_pres)
}
```

**Note for implementer:** Above pseudocode uses fluent `.mul/.sub/.div/.gt` methods on `Array`. If mlx Rust binding doesn't expose these as methods, use `mlx::ops::binary::{multiply, subtract, divide, greater}` free functions. Verify via `grep -n "fn mul\|impl Mul" /Volumes/Dev/cxx-mlx/mlx/src/array.rs` and adjust. The `dummy_link` line is a placeholder — replace with whatever import-anchor pattern the codebase uses (or just drop `use mlx::ops::binary` if not needed).

- [ ] **Step 1.3.2: Verify mlx binding API before continuing.**

```bash
grep -n "pub fn add\|pub fn sub\|pub fn mul\|pub fn div\|pub fn gt\|pub fn ge\|pub fn lt\|pub fn le" /Volumes/Dev/cxx-mlx/mlx/src/array.rs | head -20
grep -n "pub fn multiply\|pub fn subtract\|pub fn divide\|pub fn greater\|pub fn less" /Volumes/Dev/cxx-mlx/mlx/src/ops/binary.rs | head -10
```

Adjust the `apply_penalties` body to match actual API names. If `Array::mul` etc. exist as methods, use them; otherwise import free functions. Document in commit msg which path used.

- [ ] **Step 1.3.3: Add 3 unit tests (rep / freq / pres each isolated).**

```rust
fn make_logits(b: usize, vocab: usize, fill: f32) -> Array {
    let v: Vec<f32> = vec![fill; b * vocab];
    (&v[..], &[b as i32, vocab as i32][..]).try_into().expect("logits")
}

#[test]
fn apply_penalties_repetition_divides_seen_when_positive() {
    let logits = make_logits(1, 8, 2.0); // all positive
    let h0: &[u32] = &[5];
    let s = Sampler::greedy().with_repetition_penalty(2.0);
    let samplers: Vec<&Sampler> = vec![&s];
    let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
    let history_count = build_history_count(&[h0], 8).expect("hc");
    let out = apply_penalties(&logits, Some(&history_count), &cfg).expect("out");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert!((v[5] - 1.0).abs() < 1e-5, "row 0 token 5 should be 2.0/2.0=1.0; got {}", v[5]);
    assert!((v[0] - 2.0).abs() < 1e-5, "row 0 token 0 unseen → unchanged");
}

#[test]
fn apply_penalties_frequency_subtracts_count_times_penalty() {
    let logits = make_logits(1, 8, 5.0);
    let h0: &[u32] = &[3, 3, 3]; // count(3) = 3
    let s = Sampler::greedy().with_frequency_penalty(1.5);
    let samplers: Vec<&Sampler> = vec![&s];
    let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
    let history_count = build_history_count(&[h0], 8).expect("hc");
    let out = apply_penalties(&logits, Some(&history_count), &cfg).expect("out");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    // token 3: 5.0 - 1.5*3 = 0.5
    assert!((v[3] - 0.5).abs() < 1e-4, "row 0 token 3 should be 5.0-4.5=0.5; got {}", v[3]);
    assert!((v[0] - 5.0).abs() < 1e-5, "row 0 token 0 unchanged");
}

#[test]
fn apply_penalties_presence_subtracts_once_per_token() {
    let logits = make_logits(1, 8, 5.0);
    let h0: &[u32] = &[3, 3, 3]; // presence is binary (not 3-times)
    let s = Sampler::greedy().with_presence_penalty(1.5);
    let samplers: Vec<&Sampler> = vec![&s];
    let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
    let history_count = build_history_count(&[h0], 8).expect("hc");
    let out = apply_penalties(&logits, Some(&history_count), &cfg).expect("out");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    // token 3: 5.0 - 1.5*1 = 3.5
    assert!((v[3] - 3.5).abs() < 1e-4, "row 0 token 3 should be 5.0-1.5=3.5; got {}", v[3]);
    assert!((v[0] - 5.0).abs() < 1e-5, "row 0 token 0 unchanged");
}

#[test]
fn apply_penalties_short_circuit_when_no_history_needed() {
    let logits = make_logits(2, 8, 5.0);
    let s1 = Sampler::greedy().with_temperature(0.7);
    let s2 = Sampler::greedy().with_top_p(0.9);
    let samplers: Vec<&Sampler> = vec![&s1, &s2];
    let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
    assert!(!cfg.need_history);
    // history_count = None → short-circuit
    let out = apply_penalties(&logits, None, &cfg).expect("out");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert!(v.iter().all(|&x| (x - 5.0).abs() < 1e-5), "all logits unchanged");
}
```

- [ ] **Step 1.3.4: Run all T1 unit tests.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::sampler 2>&1 | tail -20
```

Expected: 22 (3e.1a) + 4 (T1 new) + 3 (T0 probes) = 29 PASS, 0 FAIL.

### Step 1.4: Hygiene + commit T1

- [ ] **Step 1.4.1:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.1b-t1): per-row configs + history bincount + apply_penalties

Foundation for configured_pipeline:
  - PerRowConfigs struct (7 [B] tensors + need_history flag)
  - collect_per_row_configs (no-op defaults for None fields)
  - build_history_count (CPU bincount → upload [B, vocab] u32)
  - apply_penalties (rep + freq + pres fused in 1 batched pass,
    short-circuit when none of rep/freq/pres set on any row)

4 unit tests: collect defaults, bincount per-row, repetition /
frequency / presence isolated, short-circuit identity.

Spec §4.3-4.5.1. configured_pipeline still stub; T3 wires it in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: temperature + top_k + softmax + top_p + min_p + renormalize

**Files:**
- Modify: `ironmlx/src/core/sampler.rs` (add 6 op functions + 5 unit tests)

**Goal:** Implement the remaining 6 ops of `configured_pipeline`. Each op is a stand-alone helper that takes `[B, vocab]` array + relevant per-row config → returns `[B, vocab]`. No integration into `sample_batch` yet (T3).

### Step 2.1: `apply_temperature`

- [ ] **Step 2.1.1: Insert after `apply_penalties`:**

```rust
/// Scale logits by per-row temperature: `logits / temp[:, None]`.
/// No-op when `temp == 1.0`.
fn apply_temperature(logits: &Array, temp_per_row: &Array) -> Result<Array> {
    let b = logits.shape().as_slice()[0];
    let temp_bv = temp_per_row.reshape(&[b, 1][..])?;
    logits.div(&temp_bv)
}
```

(Use `mlx::ops::binary::divide(logits, &temp_bv)` if `Array::div` not available — apply T1 §1.3.2 lookup adjustment.)

- [ ] **Step 2.1.2: Add unit test.**

```rust
#[test]
fn apply_temperature_scales_per_row() {
    // logits = [[2.0, 4.0], [3.0, 6.0]]; temp = [2.0, 3.0]
    // → [[1.0, 2.0], [1.0, 2.0]]
    let data: Vec<f32> = vec![2.0, 4.0, 3.0, 6.0];
    let logits: Array = (&data[..], &[2_i32, 2_i32][..]).try_into().unwrap();
    let temp: Array = (&[2.0_f32, 3.0][..], &[2_i32][..]).try_into().unwrap();
    let out = apply_temperature(&logits, &temp).expect("scaled");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert!((v[0] - 1.0).abs() < 1e-5);
    assert!((v[1] - 2.0).abs() < 1e-5);
    assert!((v[2] - 1.0).abs() < 1e-5);
    assert!((v[3] - 2.0).abs() < 1e-5);
}
```

### Step 2.2: `apply_top_k`

- [ ] **Step 2.2.1: Insert after `apply_temperature`. Implementation depends on T0 §0.1 decision (sort vs partition). Default: use `sort` (full sort, simpler).**

```rust
/// Mask logits below per-row top-k threshold with NEG_INFINITY.
/// `top_k_per_row[i]` = `vocab_size` → no-op (mask passes everything).
fn apply_top_k(logits: &Array, top_k_per_row: &Array) -> Result<Array> {
    use mlx::ops::{indexing::{take_along_axis, where_}, sort::sort};
    let dims = logits.shape();
    let dims = dims.as_slice();
    let b = dims[0];
    let vocab = dims[1];

    // Ascending sort. sorted[i, j] is the j-th smallest in row i.
    // The k-th largest is sorted[i, vocab - top_k[i]].
    let sorted = sort(logits, -1)?; // [B, vocab] ascending
    // threshold index per row = vocab - top_k
    let vocab_arr: Array = vocab.try_into()?;
    let thresh_idx = (&vocab_arr).sub(top_k_per_row)?; // [B] i32
    // Reshape to [B, 1] for gather along axis -1
    let thresh_idx_bv = thresh_idx.reshape(&[b, 1][..])?;
    let threshold = take_along_axis(&sorted, &thresh_idx_bv, -1)?; // [B, 1]
    // Mask: logits >= threshold are kept
    let mask = logits.ge(&threshold)?; // [B, vocab] bool, broadcasted
    let neg_inf: Array = f32::NEG_INFINITY.try_into()?;
    // where(mask, logits, -inf)
    where_(&mask, logits, &neg_inf)
}
```

**Implementer:** if T0 §0.1 measured `sort` slow at vocab=151k, swap to `partition`:
```rust
// Use partition(kth=vocab - max(top_k_per_row)) for speed
let max_top_k = top_k_per_row.to_vec::<i32>()?.into_iter().max().unwrap_or(vocab);
let kth = (vocab - max_top_k).max(0);
let parted = mlx::ops::sort::partition(logits, kth, -1)?; // [B, vocab] kth element in correct position
let threshold = take_along_axis(&parted, &thresh_idx_bv, -1)?;
// rest is same
```

- [ ] **Step 2.2.2: Add unit test.**

```rust
#[test]
fn apply_top_k_keeps_top_k_per_row() {
    // row 0: top_k=2 → keep top 2; row 1: top_k=4 (full size) → identity
    let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0];
    let logits: Array = (&data[..], &[2_i32, 4_i32][..]).try_into().unwrap();
    let topk: Array = (&[2_i32, 4_i32][..], &[2_i32][..]).try_into().unwrap();
    let out = apply_top_k(&logits, &topk).expect("top_k");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    // row 0: top 2 are 5.0, 3.0 (at cols 1, 2); cols 0 (1.0) and 3 (2.0) → -inf
    assert_eq!(v[0], f32::NEG_INFINITY);
    assert_eq!(v[1], 5.0);
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], f32::NEG_INFINITY);
    // row 1: identity
    assert_eq!(&v[4..8], &[1.0, 2.0, 3.0, 4.0]);
}
```

### Step 2.3: `softmax`

- [ ] **Step 2.3.1: Insert after `apply_top_k`:**

```rust
/// Numerically stable softmax over axis=-1.
fn apply_softmax(logits: &Array) -> Result<Array> {
    use mlx::ops::activation::softmax; // OR mlx::nn::softmax — verify path
    softmax(logits, &[-1], None)
}
```

- [ ] **Step 2.3.2: Verify softmax import path.**

```bash
grep -rn "pub fn softmax" /Volumes/Dev/cxx-mlx/mlx/src/ 2>/dev/null
```

Adjust import to match (may be `mlx::ops::activation::softmax`, `mlx::ops::reduction::softmax`, or in `mlx::nn::*`).

- [ ] **Step 2.3.3: No new unit test — `apply_top_p` and `apply_min_p` tests below implicitly cover softmax correctness.**

### Step 2.4: `apply_top_p`

- [ ] **Step 2.4.1: Insert after `apply_softmax`:**

```rust
/// Nucleus filter: zero out probs that fall outside the smallest
/// set summing to `top_p[i]`. The first token whose inclusion crosses
/// `top_p` is RETAINED (matches HF semantics).
fn apply_top_p(probs: &Array, top_p_per_row: &Array) -> Result<Array> {
    use mlx::ops::{indexing::{take_along_axis, where_}, cumulative::cumsum, sort::argsort};
    let dims = probs.shape();
    let dims = dims.as_slice();
    let b = dims[0];

    // Negate to make argsort give descending order indices.
    let neg_one: Array = (-1.0_f32).try_into()?;
    let neg_probs = probs.mul(&neg_one)?; // [B, vocab]
    let sort_idx_desc = argsort(&neg_probs, -1)?; // [B, vocab] i32

    // Gather probs in descending order.
    let sorted_probs = take_along_axis(probs, &sort_idx_desc, -1)?;
    let csum = cumsum(&sorted_probs, -1, false, false)?; // exclusive=false, reverse=false
    // mask_sorted[i, j] = (csum[i, j] - sorted[i, j]) < top_p[i]
    //   (keep first token whose inclusion crosses threshold)
    let csum_excl = csum.sub(&sorted_probs)?;
    let top_p_bv = top_p_per_row.reshape(&[b, 1][..])?;
    let mask_sorted = csum_excl.lt(&top_p_bv)?; // [B, vocab] bool
    let zero_f32: Array = 0.0_f32.try_into()?;
    let sorted_masked = where_(&mask_sorted, &sorted_probs, &zero_f32)?;

    // Scatter back to vocab order using inverse permutation.
    // inv_perm = argsort(sort_idx_desc) — see T0 probe_argsort_inverse_permutation_identity.
    let inv_perm = argsort(&sort_idx_desc, -1)?;
    take_along_axis(&sorted_masked, &inv_perm, -1)
}
```

- [ ] **Step 2.4.2: Add unit test (post-softmax probs).**

```rust
#[test]
fn apply_top_p_keeps_nucleus_first_crossing_retained() {
    // probs row 0 (sorted desc by hand): 0.5, 0.3, 0.15, 0.05
    // top_p=0.6 → keep 0.5 + 0.3 (0.8 > 0.6, first crossing at 0.3 retained)
    // → mask out 0.15, 0.05
    let probs_row: Vec<f32> = vec![0.5, 0.05, 0.15, 0.3]; // intentionally scrambled
    let probs: Array = (&probs_row[..], &[1_i32, 4_i32][..]).try_into().unwrap();
    let tp: Array = (&[0.6_f32][..], &[1_i32][..]).try_into().unwrap();
    let out = apply_top_p(&probs, &tp).expect("top_p");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert!((v[0] - 0.5).abs() < 1e-5, "col 0 (0.5) kept");
    assert_eq!(v[1], 0.0, "col 1 (0.05) outside nucleus");
    assert_eq!(v[2], 0.0, "col 2 (0.15) outside nucleus");
    assert!((v[3] - 0.3).abs() < 1e-5, "col 3 (0.3, first crossing) kept");
}
```

### Step 2.5: `apply_min_p`

- [ ] **Step 2.5.1: Insert after `apply_top_p`:**

```rust
/// min_p floor: keep probs >= min_p[i] * max_prob[i]. Sets others to 0.
fn apply_min_p(probs: &Array, min_p_per_row: &Array) -> Result<Array> {
    use mlx::ops::{reduction::max, indexing::where_};
    let dims = probs.shape();
    let dims = dims.as_slice();
    let b = dims[0];
    let max_per_row = max(probs, &[-1], true)?; // [B, 1] keepdims
    let min_p_bv = min_p_per_row.reshape(&[b, 1][..])?;
    let threshold = min_p_bv.mul(&max_per_row)?;
    let mask = probs.ge(&threshold)?;
    let zero_f32: Array = 0.0_f32.try_into()?;
    where_(&mask, probs, &zero_f32)
}
```

(Verify `mlx::ops::reduction::max` path with `grep -n "pub fn max" /Volumes/Dev/cxx-mlx/mlx/src/ops/reduction.rs`.)

- [ ] **Step 2.5.2: Add unit test.**

```rust
#[test]
fn apply_min_p_filters_below_threshold() {
    // probs: [0.5, 0.3, 0.15, 0.05]; min_p=0.4 → threshold = 0.4 * 0.5 = 0.2
    // keep 0.5, 0.3; drop 0.15, 0.05
    let probs: Array = (&[0.5_f32, 0.3, 0.15, 0.05][..], &[1_i32, 4_i32][..])
        .try_into().unwrap();
    let mp: Array = (&[0.4_f32][..], &[1_i32][..]).try_into().unwrap();
    let out = apply_min_p(&probs, &mp).expect("min_p");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert!((v[0] - 0.5).abs() < 1e-5);
    assert!((v[1] - 0.3).abs() < 1e-5);
    assert_eq!(v[2], 0.0);
    assert_eq!(v[3], 0.0);
}
```

### Step 2.6: `renormalize`

- [ ] **Step 2.6.1: Insert after `apply_min_p`:**

```rust
/// Renormalize per-row probs so each row sums to 1.0. Used after
/// top_p / min_p possibly zero out tokens.
fn renormalize(probs: &Array) -> Result<Array> {
    use mlx::ops::reduction::sum;
    let row_sum = sum(probs, &[-1], true)?; // [B, 1] keepdims
    probs.div(&row_sum)
}
```

- [ ] **Step 2.6.2: Add unit test.**

```rust
#[test]
fn renormalize_rows_sum_to_one() {
    // After top_p might leave: [[0.5, 0.0, 0.0, 0.3], [0.2, 0.1, 0.0, 0.0]]
    let data: Vec<f32> = vec![0.5, 0.0, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0];
    let probs: Array = (&data[..], &[2_i32, 4_i32][..]).try_into().unwrap();
    let out = renormalize(&probs).expect("renorm");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let row0_sum = v[0] + v[1] + v[2] + v[3];
    let row1_sum = v[4] + v[5] + v[6] + v[7];
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum={row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum={row1_sum}");
    assert!((v[0] - 0.625).abs() < 1e-5, "0.5 / 0.8 = 0.625");
}
```

### Step 2.7: Run all T2 unit tests + hygiene + commit

- [ ] **Step 2.7.1: Run sampler tests.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::sampler 2>&1 | tail -20
```

Expected: 29 (T0+T1) + 5 (T2: temp / top_k / top_p / min_p / renorm) = 34 PASS, 0 FAIL.

- [ ] **Step 2.7.2: Hygiene + commit.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.1b-t2): batched temp / top_k / softmax / top_p / min_p / renorm

Six additional batched ops over [B, vocab]:
  - apply_temperature (broadcast divide)
  - apply_top_k (sort + take_along_axis threshold + where)
  - apply_softmax (mlx::ops::activation::softmax)
  - apply_top_p (argsort desc + cumsum + nucleus mask + inverse
    permutation scatter-back via argsort(sort_idx) + take_along_axis)
  - apply_min_p (max + broadcast multiply threshold + where)
  - renormalize (sum + divide)

5 unit tests covering each op's identity (no-op) and active path.

configured_pipeline still stub — T3 wires the full chain
including batched categorical sample step.

Spec §4.5.2-4.5.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: configured_pipeline integration + batched categorical + PRNG handling

**Files:**
- Modify: `ironmlx/src/core/sampler.rs` (fill in `configured_pipeline` body + add `sample_batched_categorical` helper + 3 integration unit tests)

**Goal:** Wire T1+T2 ops into the complete `configured_pipeline`, add batched categorical sample step, integrate into `sample_batch` routing. After T3, `sample_batch` mixed-batch fallback runs through the new vectorized pipeline.

### Step 3.1: Implement `sample_batched_categorical`

- [ ] **Step 3.1.1: Insert after `renormalize`:**

```rust
/// Batched categorical sample over `[B, vocab]` probs. Uses a single
/// PRNG key derived from `samplers[0].seed` (or fresh time-based
/// fallback); mlx::random::categorical auto-derives row-independent
/// samples from the single key. Per-row reproducibility is NOT
/// preserved (spec NG6 accepts this drift; 3e.2 will centralize PRNG
/// state).
fn sample_batched_categorical(samplers: &[&Sampler], probs: &Array) -> Result<Vec<u32>> {
    use mlx::random;
    // Derive a single batch key. Use the first sampler's PRNG state
    // (via its `ensure_key` accessor) so that resampling with the
    // same Sampler chain produces deterministic outputs.
    let key = samplers[0].ensure_key()?;
    // Advance the key for the first sampler so the next batch step
    // uses a fresh PRNG state.
    let (new_key, _used_key) = random::split(&key)?;
    samplers[0].store_key(new_key.clone())?;

    let tokens = random::categorical(probs)
        .axis(-1)
        .key(&key)
        .sample()?;
    Ok(tokens.to_vec::<u32>()?)
}
```

- [ ] **Step 3.1.2: Add `Sampler::store_key` accessor (mirroring `ensure_key`).**

Locate `impl Sampler` block (around line 196 for `ensure_key`). Insert:

```rust
    /// Replace the cached PRNG key. Used by [`sample_batched_categorical`]
    /// to advance the key after a batch sample.
    fn store_key(&self, k: Array) -> Result<()> {
        self.key.set(Some(k));
        Ok(())
    }
```

(`store_key` is module-private — `fn` not `pub fn`.)

### Step 3.2: Fill in `configured_pipeline` body

- [ ] **Step 3.2.1: Replace the stub body:**

```rust
fn configured_pipeline(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    let vocab = dims[1];

    let configs = collect_per_row_configs(samplers, vocab)?;

    let history_count = if configs.need_history {
        Some(build_history_count(histories, vocab as usize)?)
    } else {
        None
    };

    let logits = apply_penalties(logits, history_count.as_ref(), &configs)?;
    let logits = apply_temperature(&logits, &configs.temp)?;
    let logits = apply_top_k(&logits, &configs.top_k)?;
    let probs = apply_softmax(&logits)?;
    let probs = apply_top_p(&probs, &configs.top_p)?;
    let probs = apply_min_p(&probs, &configs.min_p)?;
    let probs = renormalize(&probs)?;

    sample_batched_categorical(samplers, &probs)
}
```

### Step 3.3: Wire `configured_pipeline` into `sample_batch`

- [ ] **Step 3.3.1: Replace the per-row fallback loop in `sample_batch` (currently at line ~337-353).**

Current:

```rust
    // Mixed / configured fallback: per-row sequential. 3e.1b will
    // vectorize this for non-top-k configs.
    let mut tokens = Vec::with_capacity(b);
    for (i, sampler) in samplers.iter().enumerate() {
        let row = indexing::slice_strided_on(
            logits,
            &[i as i32, 0_i32][..],
            &[i as i32 + 1, dims[1]][..],
            &[1_i32, 1_i32][..],
            (),
        )?;
        let row_flat = row.reshape(&[dims[1]][..])?;
        tokens.push(sampler.sample(&row_flat, histories[i])?);
    }
    Ok(tokens)
```

Replace with:

```rust
    // Mixed / configured pipeline (3e.1b).
    configured_pipeline(samplers, logits, histories)
```

- [ ] **Step 3.3.2: Update doc comment on `sample_batch`.**

Locate doc comment at lines ~278-294 (3e.1a era). Replace the "# Routing (spec §4.1)" section with:

```rust
/// # Routing (spec §4.1 + 3e.1b §4.1)
/// - **All-greedy fast path** (every `samplers[b].is_greedy()`):
///   single `argmax(logits, axis=-1)` GPU dispatch → one
///   `.to_vec::<u32>()` host transfer for the whole batch.
/// - **Mixed / configured pipeline** (3e.1b): batched per-row
///   penalty/temp/top-k/softmax/top-p/min-p/renorm + batched
///   categorical sample. See `configured_pipeline` for details.
```

### Step 3.4: Mixed-batch integration unit test

- [ ] **Step 3.4.1: Add to `mod tests`:**

```rust
#[test]
fn sample_batch_mixed_batch_uses_configured_pipeline_no_panic() {
    // B=4 mixed: row 0 greedy, row 1 temp=0.7, row 2 top_p=0.9, row 3 +rep_pen
    let s0 = Sampler::greedy();
    let s1 = Sampler::greedy().with_temperature(0.7).with_seed(11);
    let s2 = Sampler::greedy().with_temperature(0.8).with_top_p(0.9).with_seed(22);
    let s3 = Sampler::greedy()
        .with_temperature(0.5).with_repetition_penalty(1.2).with_seed(33);
    let samplers: Vec<&Sampler> = vec![&s0, &s1, &s2, &s3];

    // Skewed logits: each row has a different max col.
    let vocab = 16usize;
    let mut data = vec![0.0_f32; 4 * vocab];
    for i in 0..4 { data[i * vocab + i] = 10.0; }
    let logits: Array = (&data[..], &[4_i32, vocab as i32][..]).try_into().unwrap();
    let h0: &[u32] = &[];
    let h1: &[u32] = &[];
    let h2: &[u32] = &[];
    let h3: &[u32] = &[3, 3]; // exercises rep_pen on row 3
    let histories: Vec<&[u32]> = vec![h0, h1, h2, h3];

    let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch");
    assert_eq!(tokens.len(), 4);
    // With highly skewed logits + moderate temp/top_p, each row should
    // almost surely sample its argmax column. Don't assert exact (PRNG)
    // but verify in-range.
    for (i, &t) in tokens.iter().enumerate() {
        assert!((t as usize) < vocab, "row {i} token {t} out of range");
    }
}
```

### Step 3.5: No-op identity parity test

- [ ] **Step 3.5.1: Add to `mod tests`:**

```rust
#[test]
fn sample_batch_no_op_default_identity_matches_argmax() {
    // ALL rows non-greedy via temperature only, but with very low temperature
    // (no-op for argmax under softmax), all other configs default. Expect
    // outputs to match argmax per row.
    let s = Sampler::greedy().with_temperature(0.01).with_seed(7); // very peaked
    let samplers: Vec<&Sampler> = vec![&s, &s, &s, &s];
    let vocab = 8usize;
    let mut data = vec![0.0_f32; 4 * vocab];
    for i in 0..4 { data[i * vocab + (i + 2) % vocab] = 100.0; }
    let logits: Array = (&data[..], &[4_i32, vocab as i32][..]).try_into().unwrap();
    let h: &[u32] = &[];
    let histories: Vec<&[u32]> = vec![h, h, h, h];
    let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch");
    let expected: Vec<u32> = (0..4).map(|i| ((i + 2) % vocab) as u32).collect();
    assert_eq!(tokens, expected, "very peaked logits should produce argmax per row");
}
```

### Step 3.6: Greedy short-circuit test (regression for 3e.1a path)

- [ ] **Step 3.6.1: Add to `mod tests`:**

```rust
#[test]
fn sample_batch_all_greedy_still_uses_fast_path() {
    // All rows greedy → must NOT enter configured_pipeline (which
    // would also work but bypasses 3e.1a fast path). We verify by
    // expecting deterministic argmax output (no PRNG involvement).
    let s = Sampler::greedy();
    let samplers: Vec<&Sampler> = vec![&s, &s];
    let vocab = 4usize;
    let data: Vec<f32> = vec![1.0, 5.0, 2.0, 0.0, 9.0, 1.0, 0.0, 0.0];
    let logits: Array = (&data[..], &[2_i32, vocab as i32][..]).try_into().unwrap();
    let h: &[u32] = &[];
    let histories: Vec<&[u32]> = vec![h, h];
    let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch");
    assert_eq!(tokens, vec![1, 0], "argmax: row 0→col 1 (5.0), row 1→col 0 (9.0)");
}
```

### Step 3.7: Run all sampler tests + hygiene + commit

- [ ] **Step 3.7.1: Run.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::sampler 2>&1 | tail -20
```

Expected: 34 (T0+T1+T2) + 3 (T3) = 37 PASS, 0 FAIL.

- [ ] **Step 3.7.2: Run full scheduler tests (regression).**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::scheduler 2>&1 | tail -10
```

Expected: 36 PASS (post-3e.1a baseline), 0 FAIL.

- [ ] **Step 3.7.3: Hygiene + commit.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3e.1b-t3): configured_pipeline + batched categorical wired

sample_batch routing post-3e.1b:
  - all-greedy → 3e.1a argmax fast path (unchanged)
  - mixed batch → new configured_pipeline:
      apply_penalties → apply_temperature → apply_top_k →
      apply_softmax → apply_top_p → apply_min_p → renormalize →
      sample_batched_categorical
  - sample_batched_categorical uses single key from samplers[0]
    (mlx::random::categorical auto-batches row-indep samples;
    per-row PRNG reproducibility dropped per spec NG6)
  - Sampler::store_key accessor (module-private) for advancing
    the PRNG key after each batched sample

3 integration unit tests:
  - mixed batch (4 rows w/ different configs) no panic + in-range
  - no-op identity: very peaked logits + only-temperature config →
    matches argmax per row
  - greedy short-circuit regression: still hits 3e.1a fast path

Per-row Sampler::sample fallback is fully removed from sample_batch
(remains used only by admit_mid_finalize for B=1).

Spec §4.2 + §4.6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Real-model perf gate + sweep_smoke + sweep_full + close-out

**Files:**
- Create: `ironmlx/tests/b1_p2_3e_1b_configured_sampler.rs` (perf gate integration test)
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1b_closeout/report.md`

### Step 4.1: Perf gate integration test

- [ ] **Step 4.1.1: Create new test file:**

Path: `ironmlx/tests/b1_p2_3e_1b_configured_sampler.rs`

```rust
//! B1-p2.3e.1b — vectorized configured sampler perf gate.
//!
//! Builds a 4-row concurrent admit batch with all rows using
//! `temperature=0.7, top_p=0.9, repetition_penalty=1.1` — guaranteed
//! to hit configured_pipeline (not the 3e.1a fast path). Measures
//! per-row median inter-token gap, asserts:
//!   - per-row medians within 2× (batched-step lockstep)
//!   - max median ≤ 250 ms (configured pipeline budget; 3e.1a fast
//!     path was 64.7 ms argmax; configured pipeline adds ~50-100 ms
//!     for the 7 ops + categorical)

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
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
    let msgs = vec![Message { role: "user".into(), content: text.into() }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_configured_request(prompt_ids: Vec<u32>, max_new: usize, stop: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_repetition_penalty(1.1)
            .with_seed(42),
        stop_token_ids: stop,
        prefill_chunk_size: 128,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_3e_1b_configured_decode_speedup() {
    let (model, tokenizer) = load_fixture();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();
    let handle = spawn_scheduler_actor(model.clone(), 4, Duration::from_millis(5), 32, 32768);

    let prompts = [
        "Write a short essay on the history of Italian cuisine.",
        "Explain the principles of quantum entanglement in simple terms.",
        "Describe the most important inventions of the 20th century.",
        "Tell a creative short story about a robot who learns to paint.",
    ];

    let mut tasks: Vec<tokio::task::JoinHandle<Vec<Instant>>> = Vec::new();
    for p in prompts {
        let ids = tokenize_prompt(&tokenizer, p);
        let req = make_configured_request(ids, 50, stop_tokens.clone());
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
                if ev.finish_reason.is_some() { break; }
            }
            stamps
        }));
    }

    let mut all_stamps: Vec<Vec<Instant>> = Vec::new();
    for t in tasks {
        let s = tokio::time::timeout(Duration::from_secs(240), t)
            .await
            .expect("timeout")
            .expect("join");
        assert!(s.len() >= 10, "row needs ≥ 10 tokens; got {}", s.len());
        all_stamps.push(s);
    }

    let mut all_medians: Vec<Duration> = Vec::new();
    for stamps in &all_stamps {
        let mut gaps: Vec<Duration> = (2..stamps.len())
            .map(|i| stamps[i].duration_since(stamps[i - 1]))
            .collect();
        gaps.sort();
        all_medians.push(gaps[gaps.len() / 2]);
    }

    let max_median = all_medians.iter().max().copied().unwrap();
    let min_median = all_medians.iter().min().copied().unwrap();

    eprintln!(
        "[3e.1b perf gate] per-row medians: {:?} | max={:?} min={:?} ratio={:.2}x",
        all_medians, max_median, min_median,
        max_median.as_secs_f64() / min_median.as_secs_f64().max(1e-9)
    );

    assert!(
        max_median <= min_median * 2,
        "per-row median spread > 2×: {:?} (lockstep broken?)",
        all_medians
    );

    assert!(
        max_median <= Duration::from_millis(250),
        "max median {max_median:?} exceeds 250 ms — configured_pipeline regression?"
    );

    drop(handle);
}
```

- [ ] **Step 4.1.2: Run perf gate.**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  MLX_DIR=$HOME/.local/mlx \
  cargo +stable test --release --test b1_p2_3e_1b_configured_sampler -- --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/3e_1b_perf.log | tail -30
```

Expected: PASS. Typical 100-180 ms median per row on M1 Pro.

If FAIL:
- If `max_median > 250 ms`: investigate which op is slow via `eprintln` in `configured_pipeline` after each `apply_*` (use `Instant::now()` checkpoints). Root-cause before relaxing threshold.
- If `lockstep ratio > 2`: indicates per-row work imbalance — likely a bug in batched ops not actually using `[B, vocab]`. Investigate.

### Step 4.2: Commit perf gate test

- [ ] **Step 4.2.1:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add ironmlx/tests/b1_p2_3e_1b_configured_sampler.rs
git commit -m "$(cat <<'EOF'
test(b1-p2.3e.1b-t4): real-model perf gate (#[ignore])

b1_p2_3e_1b_configured_decode_speedup: 4 concurrent admits with
temperature=0.7 + top_p=0.9 + repetition_penalty=1.1 — 100%
configured-pipeline traffic (NOT 3e.1a fast path). Asserts:
  - per-row median spread ≤ 2× (batched-step lockstep)
  - max median ≤ 250 ms (configured pipeline budget;
    3e.1a fast path was 64.7 ms argmax)

Spec §6.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 4.3: Sweep smoke gate

- [ ] **Step 4.3.1:**

```bash
./scripts/sweep/sweep_smoke.sh \
  --suites b1_p2_3b_2_scheduler_actor \
           b1_p2_4_batched_vl::mid_admit_vl_during_text_decode \
           b1_p2_3e_1a_vectorize_greedy::b1_p2_3e_1a_greedy_decode_speedup \
           b1_p2_3e_1b_configured_sampler::b1_p2_3e_1b_configured_decode_speedup \
  2>&1 | tee /tmp/3e_1b_smoke.log | tail -20
```

Expected: 4 PASS. If any FAIL — root-cause before continuing.

### Step 4.4: Sweep_full background

- [ ] **Step 4.4.1: Launch background.**

```bash
bash ./scripts/sweep/sweep_full.sh > /tmp/3e_1b_sweep_full.log 2>&1 &
echo "PID: $!"
```

Note PID. Sweep takes ~60-70 min on M1 Pro. Continue with close-out write — sweep result will be appended.

### Step 4.5: Write close-out report

- [ ] **Step 4.5.1: Create:**

Path: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1b_closeout/report.md`

```markdown
# B1-p2.3e.1b Vectorize Configured Sampler — Close-out

**Branch:** `ironmlx-b1-p2-3e1b-vectorize-configured`
**Base:** `ironmlx-b1-p2-3e1a-vectorize-greedy` HEAD `978d288`
**Spec:** `docs/superpowers/specs/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md` (`92b5b50`)
**Plan:** `docs/superpowers/plans/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler.md`

## Goal Recap

Replace `sample_batch`'s mixed/configured fallback (was per-row `Sampler::sample` loop) with a fully batched `[B, vocab]` pipeline: 7 logits/probability ops + batched `mlx::random::categorical`. All-greedy fast path (3e.1a) retained.

## Commits

- `<T0 SHA>` chore(b1-p2.3e.1b-t0): mlx API verification + design pins
- `<T1 SHA>` feat(b1-p2.3e.1b-t1): per-row configs + history bincount + apply_penalties
- `<T2 SHA>` feat(b1-p2.3e.1b-t2): batched temp / top_k / softmax / top_p / min_p / renorm
- `<T3 SHA>` feat(b1-p2.3e.1b-t3): configured_pipeline + batched categorical wired
- `<T4 perf SHA>` test(b1-p2.3e.1b-t4): real-model perf gate (#[ignore])
- `<T4 close-out SHA>` docs(b1-p2.3e.1b-t4): close-out report

## Acceptance Gates

| Gate | Result | Notes |
| --- | --- | --- |
| cargo test core::sampler (lib) | ✅ 37 PASS | 22 (3e.1a) + 3 (T0 probes) + 4 (T1) + 5 (T2) + 3 (T3) |
| cargo test core::scheduler (lib) | ✅ 36 PASS | No regression |
| Hygiene (fmt / clippy / build) | ✅ all green | Every commit |
| Real-model perf gate `b1_p2_3e_1b_configured_decode_speedup` | ✅ PASS | <填实际 median + ratio> |
| sweep_smoke (4 suites) | ✅ PASS | b1_p2_3b_2 + b1_p2_4 mid_admit + 3e.1a + 3e.1b |
| sweep_full (17 suites) | 🟡 in progress (PID: <pid>) | <update when done> |

## Performance Characterization

- **Pre-3e.1b mixed-batch fallback** (per-row loop): B × (6 GPU op + 1 .item() sync) ≈ B × 4-8 ms = 16-32 ms at B=4
- **Post-3e.1b configured_pipeline** (batched): 8 GPU op + 1 sync (`to_vec`) ≈ 3-6 ms at B=4
- **Measured (perf gate, M1 Pro 4B bf16)**: per-row median <实际值> ms (lockstep ratio <值>×)

## Architecture Notes

### `configured_pipeline`
Mixed/configured batch (any row non-greedy) routes through:
```
collect_per_row_configs (no-op defaults) →
  build_history_count (CPU bincount → upload [B, vocab] u32, short-circuit when no row needs history) →
  apply_penalties (rep + freq + pres fused) →
  apply_temperature (broadcast divide) →
  apply_top_k (sort + take_along_axis threshold + where) →
  apply_softmax →
  apply_top_p (argsort desc + cumsum + nucleus mask + inverse permutation scatter-back) →
  apply_min_p (broadcast threshold + where) →
  renormalize (sum + divide) →
  sample_batched_categorical (mlx::random::categorical with single key from samplers[0])
```

### Per-row PRNG drift
`mlx::random::categorical(logits=[B,vocab]).key(&single_key).sample()` auto-derives row-independent samples from a single PRNG key. Per-row reproducibility (e.g., row 0 seed=42 producing the same token across runs) is **not preserved** when entering configured_pipeline. Spec NG6 accepted this drift. 3e.2 will centralize PRNG state in Scheduler.

### Scatter-back via inverse permutation
mlx Rust binding doesn't expose `scatter_along_axis`. top_p scatter (sorted-order mask → vocab-order mask) implemented as:
```
inv_perm = argsort(sort_idx, axis=-1)
out      = take_along_axis(sorted_masked, inv_perm, -1)
```
Verified by T0 probe `probe_argsort_inverse_permutation_identity`.

### Sampler struct unchanged
3e.1b keeps `Sampler.key: Cell<Option<Array>>` (backwards compat preserved until 3e.2). `sample_batched_categorical` uses `samplers[0].ensure_key()` + `store_key` to thread the single batch PRNG key.

## Carry-Forward

- **3e.2** PRNG state centralization — remove `Cell<Option<Array>>` from Sampler, move `[2] u32` key to Scheduler. Spec outline at `docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md`.
- **Future** custom Metal partial-sort kernel for top_k (if `mlx::ops::sort::partition` proves insufficient at larger vocab).
- **Future** fused-op custom kernel for apply_penalties (if graph fusion ever underperforms).
```

- [ ] **Step 4.5.2: Commit close-out.**

```bash
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1b_closeout/report.md
git commit -m "$(cat <<'EOF'
docs(b1-p2.3e.1b-t4): close-out report

Documents 3e.1b 6-commit shape, acceptance gates, perf
characterization, architecture (configured_pipeline routing,
inverse-permutation scatter, PRNG drift trade-off), and
carry-forward to 3e.2 PRNG centralization.

sweep_full running in background; controller updates close-out
addendum after completion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 4.6: Wait for sweep_full + update close-out with result

- [ ] **Step 4.6.1: Poll sweep_full log until done.**

```bash
tail -f /tmp/3e_1b_sweep_full.log
# wait for "=== full sweep done ..." line
```

- [ ] **Step 4.6.2: Append sweep_full result to close-out.**

If `17/17 PASS`, append:

```markdown
## Sweep_full Result (post-close-out addendum)

`sweep_full.sh` (17 suites incl new `b1_p2_3e_1b_configured_sampler`) — **17/17 PASS** in <X>m <Y>s.

| Suite | Result |
| --- | --- |
| (16 pre-existing suites) | ✅ PASS |
| b1_p2_3e_1b_configured_sampler | ✅ PASS |

**Conclusion**: 3e.1b introduces zero regression. Configured-sampler vectorization layered cleanly atop 3e.1a all-greedy fast path.
```

If any FAIL, BLOCKED state — analyze root cause before close-out final.

- [ ] **Step 4.6.3: Commit addendum.**

```bash
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3e_1b_closeout/report.md
git commit -m "$(cat <<'EOF'
docs(b1-p2.3e.1b): close-out — sweep_full 17/17 PASS

Sweep_full launched in T4 background completed with all 17 suites
green (16 pre-existing + new b1_p2_3e_1b_configured_sampler).

Confirms configured_pipeline introduces zero regression.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 4.7: Push 3e.1b branch (controller-authorized)

- [ ] **Step 4.7.1: Push.**

```bash
git push -u origin ironmlx-b1-p2-3e1b-vectorize-configured
```

---

## Self-Review Checklist (controller, post-implementation)

After all 4 tasks complete:

1. **Spec coverage:**
   - Spec §4.1 routing → T3
   - Spec §4.2 configured_pipeline structure → T3
   - Spec §4.3 collect_per_row_configs → T1
   - Spec §4.4 build_history_count → T1
   - Spec §4.5.1 apply_penalties → T1
   - Spec §4.5.2-4.5.7 (6 ops) → T2
   - Spec §4.6 batched categorical + PRNG → T3 (single-key simplification noted in commit)
   - Spec §4.7 Sampler unchanged → preserved
   - Spec §4.8 mixed-batch semantics → T3 integration test
   - Spec §5 R1-R8 → R1/R2/R3 verified in T0
   - Spec §6 acceptance → T4

2. **No placeholders:** every step has real code.

3. **Type consistency:**
   - `PerRowConfigs` struct field names consistent across T1 (define) and T3 (use)
   - `sample_batched_categorical` signature consistent T3 (define) and T3 (use)

4. **No compat code:** original per-row `sample_batch` fallback fully replaced by `configured_pipeline` in T3 Step 3.3.

5. **Hygiene gate at every commit:** explicit in each Task §N.4 / N.7.

6. **Boss constraints:** Chinese in user-facing messages, frequent commits, MLX_DIR set, no amend / no --no-verify / no force push, auto-push only after sweep_full PASS.
