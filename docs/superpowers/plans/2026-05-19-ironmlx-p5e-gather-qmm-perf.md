# P5e SparseMoeBlock gather_qmm Perf Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the three `gather_quantized_matmul_on` calls (gate/up/down) in `SparseMoeBlock::forward_on` — T0 profile identified 64.8% of `Qwen3.5-35B-A3B-4bit` PP=2048 prefill wall-clock on M5 Max. Validate via ironmlx self before/after with no decode/sweep/numerical regression.

**Architecture:** Two-stage serial. Stage 1 (Approach A) gates 3 independent experiments behind Cargo features (`p5e-stream-parallel`, `p5e-compile`, `p5e-shape-elim`) for isolated measurement. Stage 2 (Approach B.1) introduces sorted routing (`sorted_indices=true`) gated by a PP threshold. Each stage and sub-experiment is measured against a captured baseline; winners promoted to default (cfg removed), non-winners deleted or kept behind feature flag if benign.

**Tech Stack:** Rust 1.94 / mlx (cxx-mlx wrapper) / Apple Silicon Metal (M5 Max 128 GB) / quantized 4-bit MLX weights.

**Spec reference:** [docs/superpowers/specs/2026-05-19-ironmlx-p5e-gather-qmm-perf-design.md](../specs/2026-05-19-ironmlx-p5e-gather-qmm-perf-design.md)
**T0 profile reference:** [reports/p5e-t0-profile.md](../../reports/p5e-t0-profile.md)

---

## Pre-flight

### Step 0.1: Confirm branch + clean state

- [ ] On `ironmlx-p5e-perf`

Run: `git -C /Users/xin/workspace/ironmlx-backend branch --show-current`
Expected: `ironmlx-p5e-perf`

- [ ] Working tree clean

Run: `git -C /Users/xin/workspace/ironmlx-backend status --short`
Expected: empty output

### Step 0.2: Confirm spec + T0 profile committed

- [ ] Spec and profile in branch history

Run: `git -C /Users/xin/workspace/ironmlx-backend log --oneline -3`
Expected: at least see `c4b9c27 docs(p5e): SparseMoeBlock gather_qmm perf optimization design spec` and `f47d471 research(p5e-t0): MoE prefill hot path profile`.

### Step 0.3: Baseline build verifies

- [ ] Release build is green

Run: `MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx`
Expected: `Finished \`release\` profile [optimized] target(s)`, zero Rust warnings (mlx-sys C++ warnings ok).

### Step 0.4: Confirm 35B-A3B-4bit snapshot present

- [ ] Snapshot path exists

Run:
```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1
```
Expected: outputs a snapshot SHA like `1e20fd8d42056f870933bf98ca6211024744f7ec`.

Capture for use throughout the plan:
```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
```

### Step 0.5: Confirm 4B snapshot for sweep_full present

- [ ] 4B snapshot path exists

Run:
```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1
```
Expected: outputs `32f3e8ecf65426fc3306969496342d504bfa13f3` or similar.

Capture for sweep:
```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
```

---

## Task 0: P5e Wall-Clock Baseline

**Goal:** Capture pristine wall-clock measurements for `Model::forward_on` at PP=128/512/2048 on the current `ironmlx-p5e-perf` branch (= P5 close-out state). All later experiments compare to these numbers.

**Files:**
- Create: `ironmlx/tests/p5e_baseline.rs`
- Create: `reports/p5e-baseline.md`

### Step 0.1: Write the baseline measurement test

Create `ironmlx/tests/p5e_baseline.rs`:

```rust
//! P5e baseline / measurement infrastructure: wall-clock `Model::forward_on`
//! at PP=128/512/2048 with 1 warmup + 3 measured runs per length. Output is
//! median wall-clock per length, printed via eprintln! for harvest by
//! reports/p5e-*.md.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test p5e_baseline \
//!       -- --ignored --nocapture --test-threads=1
//!
//! Identical test body is reused for each Stage 1 feature experiment (T1-T3)
//! by toggling Cargo features (no test code changes).

use mlx::Dtype;
use std::time::Instant;

use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Model};
use ironmlx::models::Qwen35MoeModel;

const PROMPT_LENGTHS: [i32; 3] = [128, 512, 2048];
const RUNS: usize = 3;
const WARMUP: usize = 1;

fn locate_snapshot() -> String {
    if let Ok(p) = std::env::var("IRONMLX_MOE_MODEL_DIR") {
        return p;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let glob = format!(
        "{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots"
    );
    let entries = std::fs::read_dir(&glob).expect("snapshots dir");
    let first = entries
        .filter_map(|e| e.ok())
        .next()
        .expect("at least one snapshot");
    first.path().to_string_lossy().into_owned()
}

fn synth_token_ids(len: i32) -> Vec<i32> {
    // Deterministic pseudo-prompt: id = (10000 + i % 100). Stays within vocab
    // (vocab_size 248320) and produces well-defined embeddings for measurement.
    (0..len).map(|i| 10_000 + (i % 100)).collect()
}

fn run_once(model: &Qwen35MoeModel, prompt_len: i32) -> std::time::Duration {
    let ids: Vec<i32> = synth_token_ids(prompt_len);
    let input_ids: mlx::Array = (&ids[..], &[1_i32, prompt_len][..])
        .try_into()
        .expect("input_ids try_into");
    let pos = build_position_ids(0, prompt_len).expect("build_position_ids");

    let cap = prompt_len.max(ironmlx::models::qwen3_5_moe::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache = Model::make_cache(model, 1, cap, Dtype::Bfloat16).expect("make_cache");

    // Force GPU sync before timing (drain any pending lazy ops from prior calls).
    let _ = Model::forward_on(
        model,
        &input_ids,
        &pos,
        None,
        None,
        Some(&mut cache),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on warmup");

    // Re-make cache so prefill runs from empty state every measurement.
    let mut cache = Model::make_cache(model, 1, cap, Dtype::Bfloat16).expect("make_cache");

    let start = Instant::now();
    let logits = Model::forward_on(
        model,
        &input_ids,
        &pos,
        None,
        None,
        Some(&mut cache),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on");
    // Force eval to materialize all lazy ops before stopping the timer.
    mlx::transforms::eval(&[&logits]).expect("eval");
    start.elapsed()
}

#[test]
#[ignore]
fn p5e_prefill_wallclock_pp_sweep() {
    let dir = locate_snapshot();
    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");

    eprintln!("[p5e_baseline] model loaded; running PP={:?}", PROMPT_LENGTHS);

    for &pp in &PROMPT_LENGTHS {
        // Warmup runs (not measured)
        for _ in 0..WARMUP {
            let _ = run_once(&model, pp);
        }

        // Measured runs
        let mut samples: Vec<f64> = (0..RUNS).map(|_| run_once(&model, pp).as_secs_f64() * 1000.0).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = samples[samples.len() / 2];
        let tok_per_s = (pp as f64) / (median_ms / 1000.0);

        eprintln!(
            "[p5e_baseline] PP={pp} runs={samples:?} median_ms={median_ms:.2} tok/s={tok_per_s:.1}",
        );
    }
}
```

- [ ] **Step 0.1 actions**: create file with the above content.

### Step 0.2: Compile the new test binary

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5e_baseline --no-run 2>&1 | tail -3
```
Expected: `Finished` with no errors.

### Step 0.3: Execute baseline measurement

Run (with `IRONMLX_MOE_MODEL_DIR` exported from Pre-flight Step 0.4):
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5e_baseline \
  -- --ignored --nocapture --test-threads=1 2>&1 | tail -20
```
Expected (approximate, will vary on hardware):
```
[p5e_baseline] PP=128 runs=[<ms>, <ms>, <ms>] median_ms=<m> tok/s=<t>
[p5e_baseline] PP=512 runs=[<ms>, <ms>, <ms>] median_ms=<m> tok/s=<t>
[p5e_baseline] PP=2048 runs=[<ms>, <ms>, <ms>] median_ms=<m> tok/s=<t>
```

Capture the three median values. These are the P5e baseline numbers.

### Step 0.4: Write baseline report

Create `reports/p5e-baseline.md`:

```markdown
# P5e Baseline (P5 close-out state, ironmlx-p5e-perf branch)

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Branch HEAD | c4b9c27 (P5e spec commit) |
| Hardware | M5 Max 128GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Method | tests/p5e_baseline.rs Model::forward_on direct call, 1 warmup + 3 measured runs, median |
| Token IDs | Deterministic synth (10000 + i % 100) |

## Wall-clock medians (ms) and throughput (tok/s)

| PP | Runs (ms) | Median (ms) | tok/s |
|---|---|---|---|
| 128 | <fill 3 samples> | <fill median> | <fill> |
| 512 | <fill 3 samples> | <fill median> | <fill> |
| 2048 | <fill 3 samples> | <fill median> | <fill> |

## Cross-check vs P5d T2 + P5e T0

- P5d T2 (via iron-bench HTTP on M1 Pro): PP=2048 ≈ 996 tok/s.
- P5e T0 (Model::forward_on with eval barriers on M5 Max): PP=2048 ≈ 921 tok/s.
- This baseline (Model::forward_on no eval barriers on M5 Max): PP=2048 = <fill>.

Discrepancies: eval barrier overhead (~96 ms at PP=2048 in T0), hardware delta vs M1 Pro,
HTTP path overhead in P5d.
```

Replace each `<fill>` with the actual numbers from Step 0.3 output.

### Step 0.5: Commit T0

Run:
```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/tests/p5e_baseline.rs \
  reports/p5e-baseline.md
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
test(p5e-t0): wall-clock prefill baseline at PP=128/512/2048

Adds tests/p5e_baseline.rs — Model::forward_on direct measurement
(no HTTP, no eval barriers) with 1 warmup + 3 measured runs per
length, median reported. Captured P5 close-out state baseline for
ironmlx-p5e-perf branch in reports/p5e-baseline.md; all subsequent
Stage 1 / Stage 2 experiments compare wall-clock to these medians.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Stage 1 A.1 — Stream parallelism for gate/up

**Goal:** Hypothesis: dispatching `gate` and `up` projections on independent MLX `Stream`s lets the Metal command scheduler run their kernels in parallel (or at least overlap dispatch).

**Files:**
- Modify: `ironmlx/Cargo.toml` (add `p5e-stream-parallel` feature)
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` (insert cfg-gated branch)

### Step 1.1: Add the Cargo feature

Read `ironmlx/Cargo.toml` for the current `[features]` block:
```toml
[features]
default = []
vision-dump = []
```

Edit to add:
```toml
[features]
default = []
vision-dump = []
p5e-stream-parallel = []
```

- [ ] **Step 1.1 actions**: apply the Edit. Verify by:
```bash
grep '^\[features\]' -A4 /Users/xin/workspace/ironmlx-backend/ironmlx/Cargo.toml
```
Expected: prints the 4 lines including `p5e-stream-parallel = []`.

### Step 1.2: Insert cfg-gated stream-parallel branch in SparseMoeBlock::forward_on

Read `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` lines 230–265 to find the existing pair of `gather_quantized_matmul_on` calls for gate and up.

The existing structure (around lines 241–272):
```rust
let gate_out = mlx::quantization::gather_quantized_matmul_on(
    &x_in,
    &self.routed.gate_weight,
    &self.routed.gate_scales,
    self.routed.gate_biases.as_ref(),
    None,
    Some(&topk_idx_u32),
    true,
    Some(self.routed.group_size),
    Some(self.routed.bits),
    "affine",
    false,
    target,
)?;
let up_out = mlx::quantization::gather_quantized_matmul_on(
    &x_in,
    &self.routed.up_weight,
    /* ...same args... */
    target,
)?;
```

Replace with a cfg-gated branch that picks separate streams under feature, default unchanged:
```rust
// gate / up gather_qmm. Default: single `target` stream. Under
// `p5e-stream-parallel`: dispatch gate on `stream_gate`, up on
// `stream_up`, both forks of the gpu device; the eval at the end of
// forward_on (or any downstream op consuming both) synchronizes.
#[cfg(feature = "p5e-stream-parallel")]
let (stream_gate, stream_up) = {
    let dev = mlx::Device::gpu(0);
    (
        mlx::StreamOrDevice::Stream(mlx::Stream::new(dev)?),
        mlx::StreamOrDevice::Stream(mlx::Stream::new(dev)?),
    )
};
#[cfg(not(feature = "p5e-stream-parallel"))]
let (stream_gate, stream_up) = (target, target);

let gate_out = mlx::quantization::gather_quantized_matmul_on(
    &x_in,
    &self.routed.gate_weight,
    &self.routed.gate_scales,
    self.routed.gate_biases.as_ref(),
    None,
    Some(&topk_idx_u32),
    true,
    Some(self.routed.group_size),
    Some(self.routed.bits),
    "affine",
    false,
    stream_gate,
)?;
let up_out = mlx::quantization::gather_quantized_matmul_on(
    &x_in,
    &self.routed.up_weight,
    &self.routed.up_scales,
    self.routed.up_biases.as_ref(),
    None,
    Some(&topk_idx_u32),
    true,
    Some(self.routed.group_size),
    Some(self.routed.bits),
    "affine",
    false,
    stream_up,
)?;
```

Notes:
- If `mlx::StreamOrDevice` is an opaque type with a constructor like `mlx::StreamOrDevice::default()` and does not have a public `Stream(Stream)` variant, replace with `mlx::StreamOrDevice::new(stream)` or whichever the public API exposes. Check `mlx/src/lib.rs` or `mlx/src/array.rs` for the actual signature before editing.
- If `mlx::Stream::new(dev)` is not the public constructor, check `mlx/src/stream.rs` for the right API.

- [ ] **Step 1.2 actions**: apply the Edit to sparse_moe.rs.

### Step 1.3: Build with the feature

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5e-stream-parallel -p ironmlx 2>&1 | tail -3
```
Expected: `Finished` with no errors.

If `Stream` / `StreamOrDevice` API doesn't match: fix the call sites per the actual API surface and re-run. Do NOT add new API surface to the `mlx` crate (that's out-of-scope for P5e).

### Step 1.4: Build test binary with feature

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-stream-parallel --test p5e_baseline --no-run 2>&1 | tail -3
```
Expected: `Finished`.

### Step 1.5: Run A.1 measurement

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-stream-parallel \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture the 3 PP medians. Compare to T0 baseline.

### Step 1.6: Numerical precision verification

Confirm A.1 doesn't break behavior:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-stream-parallel \
  --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: `2 passed` (smoke + sentinel argmax=11).

### Step 1.7: Decision + commit A.1

Compute per-PP improvement: `(baseline_median - a1_median) / baseline_median * 100%`.

- If any of PP=128/512/2048 has > 5% wall-clock improvement → keep `p5e-stream-parallel` feature (leave cfg-gated, do NOT remove); document numbers.
- If improvement ≤ 5% on all PP → keep the cfg-gated code for record; will be removed in T4 close-out.
- If any numerical / sentinel regression → revert the sparse_moe.rs change immediately; commit only the Cargo.toml feature line as a dead flag (T4 will clean up).

Commit:
```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/Cargo.toml \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
test(p5e-t1): A.1 stream parallelism experiment for gate/up gather_qmm

Adds `p5e-stream-parallel` Cargo feature, default off. When enabled,
SparseMoeBlock::forward_on dispatches the gate and up gather_qmm
projections on independent MLX streams (forks of gpu device) under
the hypothesis that Metal command scheduler can overlap their
execution.

Measurements (PP=128/512/2048 median wall-clock ms):
  - baseline (T0):     <fill from Task 0>
  - p5e-stream-parallel: <fill from Step 1.5>
  - per-PP improvement: <fill %>

Numerical: p5_qwen35_moe_smoke regression sentinel still PASS
(argmax=11).

Decision: <"keep feature for stage 1 close-out evaluation" or
"feature 0-impact, deprecate in T4">.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Fill in the actual numbers (no placeholders in the commit body — these are concrete measurements).

---

## Task 2: Stage 1 A.2 — `mlx::compile` wrap

**Goal:** Hypothesis: wrapping the SparseMoeBlock forward in `mlx::compile(.., ShapeMode::Shapeless)` lets MLX fuse SwiGLU activation epilogue, eliminate redundant kernel launches, and amortize compile cost across forward calls.

**Files:**
- Modify: `ironmlx/Cargo.toml` (add `p5e-compile` feature)
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`

### Step 2.1: Add Cargo feature

Edit `ironmlx/Cargo.toml`:
```toml
[features]
default = []
vision-dump = []
p5e-stream-parallel = []
p5e-compile = []
```

- [ ] **Step 2.1 actions**: apply the Edit.

### Step 2.2: Research mlx::compile public API

Before writing the wrap, confirm what's exposed:
```bash
grep -rn "pub fn compile\|pub fn compile_\|pub enum ShapeMode\|ShapeMode::Shapeless" /Users/xin/workspace/ironmlx-backend/mlx/src/ 2>/dev/null | head -10
```

Expected: at least one `pub fn compile(...)` symbol. If `mlx::compile` is not exposed in the safe wrapper, skip A.2 and proceed to T3 (document the skip in Step 2.7's commit message). Continue here only if compile API is available.

### Step 2.3: Add cfg-gated compile wrap

The exact compile API call shape will depend on Step 2.2 output. The pattern is:

```rust
// At the top of SparseMoeBlock impl, near other helper imports:
#[cfg(feature = "p5e-compile")]
use std::sync::OnceLock;

#[cfg(feature = "p5e-compile")]
type CompiledMoeForward = /* closure type returned by mlx::compile, depends on safe wrapper signature */;

impl SparseMoeBlock {
    #[cfg(feature = "p5e-compile")]
    fn compiled_forward(&self) -> &CompiledMoeForward {
        static CELL: OnceLock<CompiledMoeForward> = OnceLock::new();
        CELL.get_or_init(|| {
            mlx::compile::compile(
                /* closure with body identical to forward_on inner */,
                mlx::compile::ShapeMode::Shapeless,
            )
        })
    }

    pub fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        #[cfg(feature = "p5e-compile")]
        {
            return self.compiled_forward()(x, target);
        }
        #[cfg(not(feature = "p5e-compile"))]
        {
            // ... existing forward_on body unchanged ...
        }
    }
}
```

**Important practical adaptation**:
- `mlx::compile` returns a closure-like value. If the safe wrapper's `compile()` requires closures with specific signatures (e.g., `Fn(&[Array]) -> Result<Vec<Array>>`), rewrite the SparseMoeBlock forward into that shape — likely by hoisting the routed-expert weights into closure captures and accepting `x` as the single input.
- If the safe wrapper's signature is incompatible with closures that read `&self`, you may need to construct a free function `compute_sparse_moe_forward(weights, x, target) -> Result<Array>` and compile that.
- If reading and conforming to the safe wrapper takes more than ~30 minutes, abort A.2: leave the feature defined but a no-op (`#[cfg(feature = "p5e-compile")] { /* fallback to default body */ }`), measure it as a 0-impact gate, and commit + move on.

- [ ] **Step 2.3 actions**: apply the Edit to sparse_moe.rs. If compile API is too constrained, implement the no-op fallback and continue.

### Step 2.4: Build with feature

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5e-compile -p ironmlx 2>&1 | tail -3
```
Expected: `Finished` with no errors.

### Step 2.5: Run A.2 measurement

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-compile \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture the 3 PP medians.

### Step 2.6: Numerical precision verification

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-compile \
  --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 2 passed (smoke + sentinel argmax=11).

If sentinel fails (compile may reorder reductions causing > ULP drift in some cases), record the failure and revert to no-op A.2 in Step 2.3, then re-run.

### Step 2.7: Decision + commit A.2

```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/Cargo.toml \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
test(p5e-t2): A.2 mlx::compile wrap experiment for SparseMoeBlock

Adds `p5e-compile` Cargo feature, default off. When enabled, wraps
SparseMoeBlock::forward_on body in mlx::compile with ShapeMode::Shapeless
so prefill (variable PP) and decode (PP=1) share a single compiled
graph and SwiGLU activation epilogue can be fused into preceding
gather_qmm kernels.

Measurements (PP=128/512/2048 median wall-clock ms):
  - baseline (T0):     <fill from Task 0>
  - p5e-compile:       <fill from Step 2.5>
  - per-PP improvement: <fill %>

Numerical: <"sentinel argmax=11 PASS" or "sentinel drift > ULP, A.2 reverted to no-op">.

Decision: <one-liner: "keep" / "0-impact, deprecate in T4" / "API limit, gate is no-op">.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Stage 1 A.3 — Shape elimination

**Goal:** Eliminate the `squeeze(-2)` after the down projection by carrying rank-4 tensors through to the weighted sum, then collapsing two reduction axes in a single op.

**Files:**
- Modify: `ironmlx/Cargo.toml` (add `p5e-shape-elim` feature)
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`

### Step 3.1: Add Cargo feature

Edit `ironmlx/Cargo.toml`:
```toml
[features]
default = []
vision-dump = []
p5e-stream-parallel = []
p5e-compile = []
p5e-shape-elim = []
```

- [ ] **Step 3.1 actions**: apply the Edit.

### Step 3.2: Implement shape-elim branch

Read `sparse_moe.rs` lines 280–310 to find the current `down_out_4d` → `squeeze` → `weighted_sum` chain.

Current structure (approx lines 282–310):
```rust
let down_out_4d = mlx::quantization::gather_quantized_matmul_on(
    &act,
    &self.routed.down_weight,
    /* ... */
    target,
)?;
let down_out = mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32][..], target)
    .context("SparseMoeBlock: squeeze down_proj dim -2")?; // [BS, k, H]

let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1_i32, target)
    .context("SparseMoeBlock: expand_dims scores → [BS, k, 1]")?;
let weighted = (&down_out * &scores_unsq)
    .context("SparseMoeBlock: weighted = down_out * scores_unsq")?;
let routed_y = mlx::ops::sum_on(&weighted, &[-2_i32][..], false, target)
    .context("SparseMoeBlock: sum weighted across k axis")?;
```

Replace with a cfg-gated branch:
```rust
let down_out_4d = mlx::quantization::gather_quantized_matmul_on(
    &act,
    &self.routed.down_weight,
    &self.routed.down_scales,
    self.routed.down_biases.as_ref(),
    None,
    Some(&topk_idx_u32),
    true,
    Some(self.routed.group_size),
    Some(self.routed.bits),
    "affine",
    false,
    target,
)?;

#[cfg(not(feature = "p5e-shape-elim"))]
let routed_y = {
    let down_out = mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32][..], target)
        .context("SparseMoeBlock: squeeze down_proj dim -2")?; // [BS, k, H]
    let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1_i32, target)
        .context("SparseMoeBlock: expand_dims scores → [BS, k, 1]")?;
    let weighted = (&down_out * &scores_unsq)
        .context("SparseMoeBlock: weighted = down_out * scores_unsq")?;
    mlx::ops::sum_on(&weighted, &[-2_i32][..], false, target)
        .context("SparseMoeBlock: sum weighted across k axis")?
};

#[cfg(feature = "p5e-shape-elim")]
let routed_y = {
    // Keep rank-4 [BS, k, 1, H] all the way through weighted sum, then
    // reduce both the k axis (-3) and the singleton axis (-2) in one
    // sum_on call. Saves one squeeze kernel dispatch + one expand_dims.
    let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, &[-1_i32, -2_i32][..], target)
        .context("SparseMoeBlock: expand_dims scores → [BS, k, 1, 1]")?;
    let weighted = (&down_out_4d * &scores_unsq)
        .context("SparseMoeBlock: weighted = down_out_4d * scores_unsq [BS,k,1,H]")?;
    // Sum on (-3, -2) reduces [BS, k, 1, H] → [BS, H] in one op.
    mlx::ops::sum_on(&weighted, &[-3_i32, -2_i32][..], false, target)
        .context("SparseMoeBlock: sum weighted across k and singleton axis")?
};
```

Note: verify that `mlx::ops::shape::expand_dims_on` accepts a multi-axis slice (some MLX wrappers only accept a single axis). If single-axis only, call it twice in sequence:
```rust
let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1_i32, target)?;
let scores_unsq = mlx::ops::shape::expand_dims_on(&scores_unsq, -1_i32, target)?;
```
Same caveat for `sum_on` if it accepts only one axis at a time:
```rust
let summed_k = mlx::ops::sum_on(&weighted, &[-3_i32][..], false, target)?; // → [BS, 1, H]
let routed_y = mlx::ops::shape::squeeze_on(&summed_k, &[-2_i32][..], target)?; // → [BS, H]
```

This still saves one squeeze (we had two before: down_out + sum keepdim implicit; now we have just one squeeze) and one element-wise multiply on a smaller tensor.

- [ ] **Step 3.2 actions**: apply the Edit. Adapt to actual `expand_dims_on` / `sum_on` arity per inspection.

### Step 3.3: Build with feature

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5e-shape-elim -p ironmlx 2>&1 | tail -3
```
Expected: `Finished` with no errors.

### Step 3.4: Run A.3 measurement

Run:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-shape-elim \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture the 3 PP medians.

### Step 3.5: Numerical precision verification

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5e-shape-elim \
  --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 2 passed.

Reduction-order changes can cause ULP-level drift; the sentinel uses argmax which is robust to that. If sentinel fails, the shape rewrite reordered reductions in a way that crossed an argmax tie — that's a behavior change. Document and revert.

### Step 3.6: Decision + commit A.3

```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/Cargo.toml \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
test(p5e-t3): A.3 shape elimination experiment for down_proj path

Adds `p5e-shape-elim` Cargo feature, default off. When enabled,
SparseMoeBlock::forward_on keeps the down_proj output at rank 4
[BS, k, 1, H] through the weighted-sum step, eliminating the
squeeze(-2) kernel dispatch and folding both reduction axes (k +
singleton) into a single sum_on call.

Measurements (PP=128/512/2048 median wall-clock ms):
  - baseline (T0):       <fill from Task 0>
  - p5e-shape-elim:      <fill from Step 3.4>
  - per-PP improvement:  <fill %>

Numerical: <"sentinel argmax=11 PASS" or "sentinel drift, A.3 reverted">.

Decision: <one-liner: "keep" / "0-impact" / "regression revert">.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Stage 1 Close-Out

**Goal:** Promote winning A.x features (improvement > 5%) to default, remove cfg gates, run sweep_full, write Stage 1 report.

**Files:**
- Modify: `ironmlx/Cargo.toml` (remove winning feature flags)
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` (remove cfg gates for winners; remove dead-code arms for non-winners)
- Create: `reports/p5e-stage1-results.md`

### Step 4.1: Aggregate measurements from T0–T3

Collect all the median ms numbers from T0 baseline + T1 A.1 + T2 A.2 + T3 A.3 commit messages.

Compute a 4-way combined run (all 3 features enabled simultaneously):
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release \
  --features p5e-stream-parallel --features p5e-compile --features p5e-shape-elim \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture the 3 PP medians. Some optimizations may interact (positively or negatively).

### Step 4.2: Decide what to promote

For each A.x feature:
- If single-feature improvement > 5% AND combined run shows the gain is preserved (not lost to interaction) → **promote to default**: remove `#[cfg(feature = "...")]` gates, make the code path the only path; remove the corresponding line from `[features]` in Cargo.toml.
- Otherwise (≤ 5% improvement, regression, or canceled by interaction) → **delete the experimental arm**: remove the `#[cfg(feature = "...")]` block; remove the feature from Cargo.toml.

In either case the Cargo feature flag must disappear by end of T4 (no half-state cfg gates linger).

- [ ] **Step 4.2 actions**: apply Edits to remove cfg gates and Cargo.toml feature lines.

### Step 4.3: Verify post-cleanup build

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | grep -E "^(warning|error):" | grep -v "mlx-sys@" | head
cargo +nightly fmt --all -- --check
```
Expected: all clean; no warnings.

### Step 4.4: Run post-cleanup baseline (this becomes the Stage 1 final number)

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture 3 PP medians. These are the Stage 1 ship-state numbers.

### Step 4.5: Numerical precision verification on full integration set

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke \
  -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched \
  -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke \
  -- --ignored --test-threads=1 2>&1 | tail -5
```
Expected: all PASS.

### Step 4.6: sweep_full regression gate

Per [feedback_regression_sweep_at_closeout]:
```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh 2>&1 | tail -5
```
Expected: `19/19 PASS in ~2-3 minutes`.

### Step 4.7: Write Stage 1 results report

Create `reports/p5e-stage1-results.md`:

```markdown
# P5e Stage 1 Results (Approach A: MLX op rearrangement)

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Hardware | M5 Max 128GB |
| Branch HEAD post-Stage-1 | <fill from `git rev-parse HEAD`> |

## Per-experiment measurements

| PP | T0 baseline | A.1 stream | A.2 compile | A.3 shape-elim | All 3 combined |
|---|---|---|---|---|---|
| 128 | <ms> | <ms> | <ms> | <ms> | <ms> |
| 512 | <ms> | <ms> | <ms> | <ms> | <ms> |
| 2048 | <ms> | <ms> | <ms> | <ms> | <ms> |

## Promotion decisions

- A.1 stream parallelism: <"promoted" or "discarded — reason">
- A.2 mlx::compile wrap: <"promoted" or "discarded — reason">
- A.3 shape elimination: <"promoted" or "discarded — reason">

## Stage 1 final wall-clock (post-cleanup)

| PP | Stage 1 final (ms) | tok/s | Δ vs T0 baseline |
|---|---|---|---|
| 128 | <fill> | <fill> | <fill> % |
| 512 | <fill> | <fill> | <fill> % |
| 2048 | <fill> | <fill> | <fill> % |

## Validation gates passed

- p5_qwen35_moe_smoke regression sentinel argmax=11: PASS
- p5_qwen35_moe_batched (B=2 vs B=1 per-row): PASS
- p5_qwen35_moe_http_smoke chat completion: PASS
- sweep_full: 19/19 in <fill> seconds
- clippy: 0 warnings
- fmt: clean

## Notes for Stage 2

<observations, e.g. "down_proj still dominant after Stage 1; sorted routing likely to
yield more improvement at long PP">
```

Fill all placeholders with real numbers.

### Step 4.8: Commit Stage 1 close-out

```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/Cargo.toml \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs \
  reports/p5e-stage1-results.md
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
chore(p5e-t4): Stage 1 close-out — promote winning A.x experiments

Stage 1 (Approach A) close-out. Each experiment measured in T1-T3;
this commit promotes those with > 5% wall-clock improvement to
default (removes cfg gates), drops non-winners (removes both the
cfg block and Cargo feature line).

Promoted: <list>
Discarded: <list>

Stage 1 final wall-clock vs T0 baseline (PP=128/512/2048):
  <fill 3 lines: PP=X T0=<ms> stage1=<ms> Δ=<%>>

Validation: 4 MoE integration tests PASS, sweep_full 19/19,
clippy + fmt clean. Stage 2 (sorted routing) can proceed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Stage 2 B.1 — Sorted Routing

**Goal:** Sort tokens by expert index before invoking gather_qmm, set `sorted_indices=true`, and scatter back. Hypothesis: MLX's sorted-path gather_qmm has better memory access patterns; sort overhead is amortized across the 3 gather_qmm calls plus SwiGLU. Add a PP threshold so short prompts skip the sort overhead.

**Files:**
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`

### Step 5.1: Inspect MLX `gather_qmm`'s sorted_indices behavior

Confirm MLX honors `sorted_indices=true` (no-op otherwise). Quick check:
```bash
grep -rn "sorted_indices\|sorted_indices_" /Users/xin/workspace/iron-rivals/mlx/mlx/ 2>/dev/null | head -20
```
Expected: at least one path where MLX takes a different code branch when `sorted_indices=true`. If grep returns no matches → MLX may not have a sorted-fast-path; abort Stage 2 in Step 5.6 commit message and explain.

Capture findings as a 1-sentence note in the commit message later.

### Step 5.2: Implement sorted routing branch

In `sparse_moe.rs`, between the topk computation (~line 220) and the gate gather_qmm call (~line 241), insert:

```rust
// B.1: PP threshold check. For BS * k < 128 (effectively very short prompts),
// argsort/scatter overhead is likely larger than gather_qmm savings.
const P5E_SORT_THRESHOLD: i32 = 128;
let bs_k = b * s * self.num_experts_per_tok;
let use_sorted = bs_k >= P5E_SORT_THRESHOLD;

let (x_in_routed, topk_for_qmm, sort_perm_opt) = if use_sorted {
    // Flatten topk_idx [BS, k] → [BS*k]
    let flat_topk = mlx::ops::shape::reshape(&topk_idx_u32, &[bs_k][..])
        .context("B.1: reshape topk_idx to flat")?;
    // Permutation that sorts by expert id
    let sort_perm = mlx::ops::sort::argsort_on(&flat_topk, -1_i32, target)
        .context("B.1: argsort flat_topk for permutation")?;

    // Sort the topk indices themselves
    let sorted_topk = mlx::ops::indexing::take_along_axis_on(&flat_topk, &sort_perm, -1_i32, target)
        .context("B.1: take_along_axis topk by sort_perm")?;

    // For each of [BS*k] slots, we need to know which token index in [BS] it
    // came from. token_idx = slot_idx / k (integer division). Construct that
    // index array and gather x accordingly.
    //
    // Easiest construction: build [BS*k] array of token indices, then
    // take_along_axis with sort_perm.
    let mut token_idx_vec: Vec<u32> = Vec::with_capacity(bs_k as usize);
    for token in 0..(b * s) {
        for _ in 0..self.num_experts_per_tok {
            token_idx_vec.push(token as u32);
        }
    }
    let token_idx: mlx::Array = (token_idx_vec.as_slice(), &[bs_k][..])
        .try_into()
        .context("B.1: token_idx try_into")?;
    let sorted_token_idx = mlx::ops::indexing::take_along_axis_on(&token_idx, &sort_perm, -1_i32, target)
        .context("B.1: take_along_axis token_idx by sort_perm")?;

    // Gather x (rank-3 [B, S, H]) → flat [BS, H] → take by sorted_token_idx
    let flat_x_3d = mlx::ops::shape::reshape(&flat_x, &[b * s, h][..])
        .context("B.1: reshape flat_x to [BS, H]")?;
    let sorted_x_2d = mlx::ops::indexing::take_on(&flat_x_3d, &sorted_token_idx, 0_i32, target)
        .context("B.1: take flat_x by sorted_token_idx")?;
    // Reshape to [BS*k, 1, 1, H] for gather_qmm rank-4 input contract
    let sorted_x_in = mlx::ops::shape::reshape(&sorted_x_2d, &[bs_k, 1, 1, h][..])
        .context("B.1: reshape sorted_x to [BS*k,1,1,H]")?;
    // sorted_topk is currently [BS*k]; gather_qmm expects [BS*k, 1] for
    // rhs_indices when input is rank-4. Reshape:
    let sorted_topk_2d = mlx::ops::shape::reshape(&sorted_topk, &[bs_k, 1][..])
        .context("B.1: reshape sorted_topk to [BS*k, 1]")?;

    (sorted_x_in, sorted_topk_2d, Some(sort_perm))
} else {
    // Fallback: same shape as before B.1.
    (x_in.clone(), topk_idx_u32.clone(), None)
};
```

Replace the 3 gather_qmm calls below this point with calls using `x_in_routed` and `topk_for_qmm`, set `sorted_indices` to `use_sorted`:

```rust
let gate_out = mlx::quantization::gather_quantized_matmul_on(
    &x_in_routed,
    &self.routed.gate_weight,
    &self.routed.gate_scales,
    self.routed.gate_biases.as_ref(),
    None,
    Some(&topk_for_qmm),
    true,
    Some(self.routed.group_size),
    Some(self.routed.bits),
    "affine",
    use_sorted, // <-- key change: was `false`
    target,
)?;
// up / down same pattern, all use `use_sorted` for the sorted_indices argument.
```

After the down_proj gather_qmm (lines ~285), if `use_sorted` is true, scatter back to original token order before the weighted sum:

```rust
let down_out_3d = if use_sorted {
    let sort_perm = sort_perm_opt.expect("use_sorted ⇒ sort_perm");
    // Inverse permutation: argsort(sort_perm) gives the index map back to original positions
    let inv_perm = mlx::ops::sort::argsort_on(&sort_perm, -1_i32, target)
        .context("B.1: argsort sort_perm for inv_perm")?;

    // down_out_4d is currently [BS*k, 1, 1, H] in sorted order; reshape to [BS*k, H]
    let down_out_2d = mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32, -3_i32][..], target)
        .context("B.1: squeeze down_out_4d (sorted) → [BS*k, H]")
        .or_else(|_| -> Result<_> {
            // Fallback if squeeze_on doesn't accept multi-axis
            let s1 = mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32][..], target)?;
            mlx::ops::shape::squeeze_on(&s1, &[-2_i32][..], target)
        })?;
    let unsorted_2d = mlx::ops::indexing::take_on(&down_out_2d, &inv_perm, 0_i32, target)
        .context("B.1: take down_out by inv_perm to restore order")?;
    // Reshape [BS*k, H] → [BS, k, H]
    mlx::ops::shape::reshape(&unsorted_2d, &[b * s, self.num_experts_per_tok, h][..])
        .context("B.1: reshape unsorted_2d → [BS, k, H]")?
} else {
    // Original path (Stage 1 final): squeeze to [BS, k, H]
    mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32][..], target)
        .context("SparseMoeBlock: squeeze down_proj dim -2")?
};
```

Then the weighted_sum + shared_expert path continues as in Stage 1 final (using `down_out_3d` in place of `down_out`).

Notes:
- The `take_on` API signature must be verified — `mlx::ops::indexing::take_on(arr, indices, axis, stream)` is the expected shape; if the safe wrapper uses `take_along_axis_on` or similar, adapt.
- If `mlx::ops::sort::argsort_on` is not exposed, check `mlx::ops::sort::*` for the actual function name. If `argsort` only exists without `_on`, use it without the stream argument.
- All work must use the same `target` stream so the partial ordering is preserved.

- [ ] **Step 5.2 actions**: apply the Edit to sparse_moe.rs.

### Step 5.3: Build

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```
Expected: `Finished` with no errors. Fix compile errors per the actual MLX API surface (Step 5.2 notes).

### Step 5.4: Run B.1 measurement

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture 3 PP medians. Note: PP=128 with k=4 gives BS*k = 512, well above threshold 128, so all 3 test lengths exercise the sorted path. (The threshold matters more for very small decode-step BS=1 cases.)

### Step 5.5: Numerical precision verification

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke \
  -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched \
  -- --ignored --test-threads=1 2>&1 | tail -5
```
Expected: both PASS.

If sentinel argmax shifts: the sort/scatter reordered floating-point summations in a way that crossed the argmax tie. Investigate logit margin around the top-1 — if margin > threshold for the sentinel prompt, the change should be safe; if shifted, accept and update the sentinel value (record that ironmlx's deterministic output legitimately changed in P5e), OR roll back B.1.

### Step 5.6: Decision + commit B.1

If the change is shipped:
```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
feat(p5e-t5): B.1 sorted routing for gather_qmm

SparseMoeBlock::forward_on now (when BS*k >= 128) sorts tokens by
expert index before the 3 gather_qmm calls and passes sorted_indices
= true. After down_proj, an inverse permutation restores original
token order before the weighted sum + shared_expert path. For very
short BS*k (decode steps), falls back to the unsorted path to avoid
argsort/scatter overhead.

Hypothesis: MLX gather_qmm sorted-path has better cache locality for
quantized weight access (each expert's rows accessed contiguously).

Measurements (PP=128/512/2048 median wall-clock ms):
  - Stage 1 final:    <fill from T4 step 4.4>
  - Stage 2 B.1:      <fill from Step 5.4>
  - per-PP improvement: <fill %>

Numerical: p5_qwen35_moe_smoke sentinel + p5_qwen35_moe_batched both PASS.
MLX sorted_indices fast path confirmed: <one-sentence from Step 5.1 grep>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If MLX has no sorted fast path (Step 5.1) and measurement shows no improvement:
- Revert the sparse_moe.rs change to Stage 1 final state.
- Commit a one-line note describing the negative finding:

```bash
git -C /Users/xin/workspace/ironmlx-backend add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
chore(p5e-t5): Stage 2 B.1 sorted routing reverted — no MLX fast path

Implemented sorted routing per Stage 2 design. Verified MLX's
gather_qmm does not have a distinct sorted-fast-path code branch
(grep for sorted_indices in mlx/mlx/ shows it as parameter only,
no kernel-level dispatch). Measurement confirmed ≤ 1% improvement
across PP=128/512/2048 — within noise. Net effect after sort
overhead included would have been negative for short PP.

Reverted sparse_moe.rs to Stage 1 final state. P5e Stage 2 yields
zero increment beyond Stage 1; close-out (T6) will report Stage 1
gains only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Stage 2 + P5e Final Close-Out

**Goal:** Run full validation, write P5e final report, commit.

**Files:**
- Create: `reports/p5e-final-results.md`

### Step 6.1: Run final hygiene chain

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | grep -E "^(warning|error):" | grep -v "mlx-sys@" | head
MLX_DIR=$HOME/.local/mlx cargo build --release
```
Expected: all clean.

### Step 6.2: Final integration test set

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke \
  -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched \
  -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke \
  -- --ignored --test-threads=1 2>&1 | tail -5
```
Expected: all PASS.

### Step 6.3: Final wall-clock measurement

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release \
  --test p5e_baseline -- --ignored --nocapture --test-threads=1 2>&1 | tail -10
```
Capture the 3 PP medians. These are the P5e final ship-state numbers.

### Step 6.4: sweep_full regression gate

```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh 2>&1 | tail -5
```
Expected: 19/19 PASS.

### Step 6.5: Write P5e final report

Create `reports/p5e-final-results.md`:

```markdown
# P5e Final — SparseMoeBlock gather_qmm Perf Optimization Close-Out

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Branch | ironmlx-p5e-perf |
| Spec | docs/superpowers/specs/2026-05-19-ironmlx-p5e-gather-qmm-perf-design.md |
| Plan | docs/superpowers/plans/2026-05-19-ironmlx-p5e-gather-qmm-perf.md |
| Hardware | M5 Max 128GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |

## P5e wall-clock summary

| PP | T0 baseline (ms) | Stage 1 final (ms) | Stage 2 final (ms) | Δ vs T0 | Δ tok/s |
|---|---|---|---|---|---|
| 128 | <fill> | <fill> | <fill> | <%> | <baseline tok/s → final tok/s> |
| 512 | <fill> | <fill> | <fill> | <%> | <baseline tok/s → final tok/s> |
| 2048 | <fill> | <fill> | <fill> | <%> | <baseline tok/s → final tok/s> |

All numbers are 3-run median of `Model::forward_on` direct call (no HTTP, no
eval barriers), 1 warmup pass discarded.

## Promotions (Stage 1)

- A.1 stream parallelism: <"promoted" or "discarded — reason">
- A.2 mlx::compile wrap: <"promoted" or "discarded — reason">
- A.3 shape elimination: <"promoted" or "discarded — reason">

## Stage 2 outcome

<"B.1 sorted routing shipped — Δ% improvement" or "B.1 reverted — no MLX fast path">

## Validation gates (post-Stage-2)

- p5_qwen35_moe_smoke regression sentinel argmax=11: PASS
- p5_qwen35_moe_batched (B=2 vs B=1 per-row): PASS
- p5_qwen35_moe_http_smoke chat completion: PASS
- sweep_full: 19/19 in <fill> seconds (M5 Max)
- clippy --all-features --workspace -D warnings: 0 warnings
- fmt --check: clean
- release build: PASS

## Comparison to T0 profile expectations

T0 profile expected hot path:
- 3× gather_qmm = 64.8% of PP=2048 prefill
- Per-call: down (28.9%) + gate (18.4%) + up (17.5%)

P5e changes targeted these directly. Observed wall-clock change at PP=2048:
<fill brief commentary, e.g. "Stage 1 A.X promotion shaved Y ms; Stage 2 B.1
shifted memory access pattern but net <Z%>...">

## Known debt / follow-ups (not P5e scope)

- **B.2 grouped matmul per expert** — if P5e net gain insufficient, future
  P5e+1 / P5f could group tokens by expert and replace gather_qmm with
  per-expert quantized_matmul. Requires MLX grouped matmul API exposure check.
- **GatedDeltaNet 20.6% of PP=2048 wall-clock** — second-largest hot path,
  unchanged by P5e. Candidate for future linear-attention optimization phase.
- **GatedAttention 6.5% (10 layers, O(S²))** — smaller share but
  super-linear; matters at long context.
- **mrope.rs / vision/ mlx-vlm framing cleanup** — P5 reframe legacy still
  pending future cleanup pass.

## Cross-reference: omlx (observation only)

omlx serve on the same snapshot with its default-on optimizations (body-
replacement patches + paged cache, not opt-out-able) achieves prefill ~4214
tok/s at PP=2048 (per P5d T2). The P5e final number is recorded here as
ironmlx self-improvement vs T0; per memory[no_spec_from_competitors] no
target alignment to omlx is implied.
```

Fill all `<fill>` placeholders.

### Step 6.6: P5e final close-out commit

```bash
git -C /Users/xin/workspace/ironmlx-backend add \
  reports/p5e-final-results.md
git -C /Users/xin/workspace/ironmlx-backend commit -m "$(cat <<'EOF'
chore(p5e-t6): P5e close-out — SparseMoeBlock gather_qmm perf complete

P5e (SparseMoeBlock gather_qmm optimization) complete. Branch
ironmlx-p5e-perf ready for merge consideration.

ironmlx self-improvement (M5 Max, Model::forward_on direct, 3-run median):
  - PP=128:  baseline <ms> → final <ms> (Δ <%>)
  - PP=512:  baseline <ms> → final <ms> (Δ <%>)
  - PP=2048: baseline <ms> → final <ms> (Δ <%>)

Stage 1 (Approach A) promotions: <list>
Stage 2 (Approach B.1 sorted routing): <"shipped — Δ%" or "reverted">

Validation:
  - 4 MoE integration tests PASS (smoke + batched + http_smoke + logits_dump)
  - sweep_full 19/19 PASS on M5 Max
  - clippy + fmt + release build clean

Known debt (out of P5e scope): B.2 grouped matmul, GatedDeltaNet opt,
GatedAttention long-context opt, mlx-vlm framing cleanup in legacy code.

Per memory[no_spec_from_competitors]: no competitor-alignment target
was used; only ironmlx self before/after comparison drives P5e closure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 6.7: Verify branch state

```bash
git -C /Users/xin/workspace/ironmlx-backend log --oneline c4b9c27..HEAD
git -C /Users/xin/workspace/ironmlx-backend status --short
```
Expected: 7+ P5e commits (T0 through T6, plus any sub-commits during T1-T5 if any extra cleanups happened); clean working tree.

---

## P5e Final Acceptance

All of the following must be true at HEAD:

- [ ] `cargo build --release -p ironmlx`: PASS
- [ ] `cargo test -p ironmlx --lib --release -- --test-threads=1`: 291+ lib tests PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored`: PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored`: PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored`: PASS
- [ ] `QWEN35_MODEL=<snap> ./scripts/sweep/sweep_full.sh`: 19/19 PASS
- [ ] `cargo +nightly clippy --all-features --workspace --release -- -D warnings`: 0 warnings
- [ ] `cargo +nightly fmt --all -- --check`: clean
- [ ] `reports/p5e-baseline.md`, `reports/p5e-stage1-results.md`, `reports/p5e-final-results.md` all written with real numbers
- [ ] PP=2048 final tok/s > PP=2048 baseline tok/s (some net improvement); OR documented why net is 0% (Stage 2 B.1 reverted + Stage 1 yielded < 5%) — in which case P5e is also considered closed (negative result also valid; record it and don't gate further phases on it)

After acceptance, the P5e branch is ready for Boss decision on merge timing (immediate vs roll into a larger merge after P6 VL).
