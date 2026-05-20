# P5g GatedDeltaNet Deep Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile GatedDeltaNet (`ironmlx/src/nn/gated_delta_net.rs`) per § 3.2 3-layer protocol via HTTP-path harness, then promote ≥ 1 profile-driven op-level optimization via § 7.3 ship metrics (long-prompt geomean prefill +5% AND per-PP regression < 2% AND decode TG regression < 2%) without changing GatedDeltaNet public API or touching dense / MTP / GatedAttention / SparseMoeBlock paths.

**Architecture:** Profile-first. T0 instruments GatedDeltaNet (prefix-parsed layer_idx, mode-gated by `IRONMLX_P5G_PROFILE_MODE`, `OnceLock<ProfileMode>` cached) and runs 4-phase HTTP harness (Phase A whole-prefill baseline + B Layer 1 boundary-isolated + C Layer 2 per-step breakdown + D Layer 3 shape-preserving cost ablation). T0.d locks § 7.2 target. T1-T3 implement op-level candidates by T0.c ranking, each independently ship-or-revert by § 7.3 metrics relative to its starting HEAD. T4 closes-out with 4-way bench + sweep_full + report + P5h scope quantification. Scope gate: if T0/T1 requires Metal kernel rewrite → pause for Boss decision.

**Tech Stack:** Rust 1.94 / cxx-mlx Rust/C++ FFI / Apple Silicon Metal (M5 Max 128 GB) / Qwen3.5-35B-A3B-4bit MoE. iron-bench Rust HTTP harness for ship validation.

**Spec reference:** [docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md](../specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md) (HEAD `d864e6e`, 8-commit iteration through 7 ChatGPT review rounds)

---

## Pre-flight

### Shell preamble (applies to ALL bash blocks below)

**Required at the top of every bash block in this plan:**

```bash
set -o pipefail
```

Reason: many validation blocks use `<cmd> 2>&1 | tail -N` to surface only the last few output lines. Without `pipefail`, a failing `<cmd>` whose stderr piped into `tail` returns 0 — silently masking build / clippy / test failures. With `pipefail`, the pipeline returns the first non-zero exit code, so a cargo failure surfaces correctly.

If a step explicitly does NOT want `pipefail` (none in this plan), say so inline.

### Repo + rival paths (capture at session start)

```bash
set -o pipefail
REPO="$(git -C "$(pwd)" rev-parse --show-toplevel)"
cd "$REPO"
RIVALS_DIR="${RIVALS_DIR:-/Users/xin/workspace/iron-rivals}"
ls "$RIVALS_DIR/omlx/pyproject.toml" > /dev/null && echo "[preflight] RIVALS_DIR=$RIVALS_DIR ok"
```

Expected:
- `pwd` shows the ironmlx-backend repo root.
- `$REPO` is captured for subsequent steps (relative paths preferred; if a step uses `$REPO/...` form, the value must be defined).
- `$RIVALS_DIR` defaults to `/Users/xin/workspace/iron-rivals` (the Boss-environment path); override via env if Boss/agent runs from a different machine layout.

### Step P-1: Confirm branch + clean state

- [ ] On `ironmlx-p5g-perf`

Run: `git -C "$REPO" branch --show-current`
Expected: `ironmlx-p5g-perf`

- [ ] Working tree clean

Run: `git -C "$REPO" status --short`
Expected: empty

### Step P-2: Confirm spec history present

Run: `git -C "$REPO" log --oneline -10`
Expected: includes `d864e6e docs(p5g): seventh-round review-driven spec polish` and 6 earlier `docs(p5g):` commits all the way to `eacf8b6 docs(p5g): GatedDeltaNet deep refactor design spec`.

### Step P-3: Baseline build verifies

- [ ] Release build green

Run:
```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tee /tmp/p5g-preflight-build.log
tail -3 /tmp/p5g-preflight-build.log
```
Expected: `Finished release profile [optimized] target(s)`, 0 Rust warnings (mlx-sys C++ warnings ok). With `pipefail` set, a cargo build failure causes the whole block to exit non-zero.

### Step P-4: Confirm 35B MoE snapshot present

Run:
```bash
set -o pipefail
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1
```
Expected: outputs `1e20fd8d42056f870933bf98ca6211024744f7ec`.

Capture for the rest of the plan:
```bash
set -o pipefail
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
```

### Step P-5: Confirm 4B snapshot for sweep_full

Run:
```bash
set -o pipefail
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1
```
Expected: outputs `32f3e8ecf65426fc3306969496342d504bfa13f3` or similar.

Capture:
```bash
set -o pipefail
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
```

### Step P-6: Confirm mlx-lm bench venv (for T4 4-way bench)

Run: `ls "$REPO/scripts/bench-venvs/mlx-lm/.venv/bin/mlx_lm.server" 2>&1`
Expected: prints path.

### Step P-7: Confirm omlx repo (for T4 4-way bench)

Run: `ls "$RIVALS_DIR/omlx/pyproject.toml"`
Expected: prints path.

---

## Task 0: GatedDeltaNet 3-Layer Profile Infrastructure + Phase A-D Run

**Goal:** Build the profile hook in `gated_delta_net.rs` (compile-time feature `p5g-profile` + runtime `IRONMLX_P5G_PROFILE_MODE` env var, OnceLock cached, prefix-parsed layer_idx, `tracing::info!` after timer.stop only) + HTTP-path harness `tests/p5g_t0_gated_delta_profile.rs` (4-phase server spawn + iron-bench HTTP requests + server log parse) + report `reports/p5g-t0-gated-delta-profile.md` + lock § 7.2 target.

**Files:**
- Modify: `ironmlx/Cargo.toml` (add feature)
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (instrument + prefix parse + struct field)
- Create: `ironmlx/tests/p5g_t0_gated_delta_profile.rs` (HTTP-path harness)
- Create: `reports/p5g-t0-gated-delta-profile.md`
- Modify: `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md` (§ 7.2 lock target)

### Step 0.1: Add `p5g-profile` Cargo feature

Read `ironmlx/Cargo.toml` `[features]` block. Expected current:
```toml
[features]
default = []
vision-dump = []
```

Edit to append `p5g-profile = []`:
```toml
[features]
default = []
vision-dump = []
p5g-profile = []
```

- [ ] Apply Edit. Verify:
```bash
set -o pipefail
grep -A4 "^\[features\]" "$REPO/ironmlx/Cargo.toml"
```
Expected output includes `p5g-profile = []` line.

### Step 0.2: Add ProfileMode enum + OnceLock + parser at top of gated_delta_net.rs

Read first ~25 lines of `ironmlx/src/nn/gated_delta_net.rs` to find existing top-of-file imports + module setup. Insert (after the existing `use` block and before the first `pub struct`) the following code block:

```rust
// P5g T0: profile mode (compile-time gated by `p5g-profile` feature).
//
// Runtime mode selected by `IRONMLX_P5G_PROFILE_MODE` env var, cached once
// via OnceLock to avoid per-forward env lookup. Disabled (Mode::Off) path
// must produce zero measurable overhead beyond a single cached-flag check.
//
// `as_str()` is the single source of truth for mode names — env parser
// matches the same strings the log emits, so `IRONMLX_P5G_PROFILE_MODE=layer1`
// round-trips identically to the log line `mode=layer1`. Never log
// `{mode:?}` (would emit Debug names `Layer1` / `AblateComputeG` and
// break Phase B/C/D aggregation by env-string match).
#[cfg(feature = "p5g-profile")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileMode {
    Off,
    Layer1,
    Layer2,
    AblateComputeG,
    AblateConv,
    AblateTArr,
}

#[cfg(feature = "p5g-profile")]
impl ProfileMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ProfileMode::Off => "off",
            ProfileMode::Layer1 => "layer1",
            ProfileMode::Layer2 => "layer2",
            ProfileMode::AblateComputeG => "ablate-compute-g",
            ProfileMode::AblateConv => "ablate-conv",
            ProfileMode::AblateTArr => "ablate-t-arr",
        }
    }
}

#[cfg(feature = "p5g-profile")]
static PROFILE_MODE: std::sync::OnceLock<ProfileMode> = std::sync::OnceLock::new();

#[cfg(feature = "p5g-profile")]
pub(crate) fn profile_mode() -> ProfileMode {
    *PROFILE_MODE.get_or_init(|| match std::env::var("IRONMLX_P5G_PROFILE_MODE").as_deref() {
        Ok(s) if s == ProfileMode::Layer1.as_str() => ProfileMode::Layer1,
        Ok(s) if s == ProfileMode::Layer2.as_str() => ProfileMode::Layer2,
        Ok(s) if s == ProfileMode::AblateComputeG.as_str() => ProfileMode::AblateComputeG,
        Ok(s) if s == ProfileMode::AblateConv.as_str() => ProfileMode::AblateConv,
        Ok(s) if s == ProfileMode::AblateTArr.as_str() => ProfileMode::AblateTArr,
        _ => ProfileMode::Off,
    })
}

#[cfg(feature = "p5g-profile")]
fn parse_layer_idx_from_prefix(prefix: &str) -> Option<i32> {
    // Expects `model.layers.{N}.linear_attn` shape. Returns Some(N) on
    // parse success, None on naming drift (treated as "unknown layer").
    prefix
        .split('.')
        .nth(2)
        .and_then(|s| s.parse::<i32>().ok())
}
```

- [ ] Apply Edit.

### Step 0.3: Add `profile_layer_idx` field to GatedDeltaNet struct

Read the `pub struct GatedDeltaNet { ... }` block (line ~71-87). Add a new field at the end of the struct, gated by `#[cfg(feature = "p5g-profile")]`:

```rust
pub struct GatedDeltaNet {
    in_proj_qkvz: Linear,
    in_proj_ba: Linear,
    conv1d: Conv1d,
    norm: RmsNormGated,
    out_proj: Linear,
    a_log: Array,
    dt_bias: Array,
    cfg: GatedDeltaNetConfig,
    kernel_no_mask: OnceLock<MetalKernel>,
    kernel_masked: OnceLock<MetalKernel>,
    /// Layer index for profile log. Some(N) if parsed from `model.layers.{N}.linear_attn`
    /// prefix at `from_loader`; None for `from_components` (unit-test path) or prefix
    /// parse failure. Profile-only field (zero footprint without `p5g-profile`).
    #[cfg(feature = "p5g-profile")]
    profile_layer_idx: Option<i32>,
}
```

- [ ] Apply Edit. The field is only present when `p5g-profile` is enabled — no impact on normal release builds.

### Step 0.4: Initialize `profile_layer_idx` in `from_loader`

Read the `from_loader` method signature around line 92, then find where `Self { ... }` is finally constructed at the end of `from_loader`. Edit that constructor to add the field (gated):

```rust
Ok(Self {
    in_proj_qkvz,
    in_proj_ba,
    conv1d,
    norm,
    out_proj,
    a_log,
    dt_bias,
    cfg,
    kernel_no_mask: OnceLock::new(),
    kernel_masked: OnceLock::new(),
    #[cfg(feature = "p5g-profile")]
    profile_layer_idx: parse_layer_idx_from_prefix(prefix),
})
```

- [ ] Apply Edit.

### Step 0.5: Initialize `profile_layer_idx` in `from_components`

Find `from_components` (around line 249). Locate the `Self { ... }` it constructs (around line 259) and add:

```rust
Self {
    in_proj_qkvz,
    in_proj_ba,
    conv1d,
    norm,
    out_proj,
    a_log,
    dt_bias,
    cfg,
    kernel_no_mask: OnceLock::new(),
    kernel_masked: OnceLock::new(),
    #[cfg(feature = "p5g-profile")]
    profile_layer_idx: None,
}
```

- [ ] Apply Edit.

### Step 0.6: Build with feature on to verify struct invariant

Run:
```bash
set -o pipefail
cd "$REPO" && MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tail -5
```
Expected: `Finished release profile`. If "missing field profile_layer_idx" error → re-check Steps 0.4-0.5.

Also confirm default build (no feature) still green:
```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```
Expected: Finished.

### Step 0.7: Instrument GatedDeltaNet::forward_on with mode-gated barriers

Read `forward_on` method body. Verified structure (HEAD `ac077d0`):

- Signature (line ~349): `pub fn forward_on(&self, x: &Array, mask: Option<&Array>, per_row_lens: Option<&[i32]>, mut cache: Option<&mut GatedDeltaCache>, target: impl Into<StreamOrDevice>) -> Result<Array>` — `cache` is `Option<&mut GatedDeltaCache>` with `mut` binding.
- Step 7e cache update (line ~667-679): currently `if let Some(c) = cache { c.update_recurrent(...); c.advance(lens_ref)?; }` — **this moves `cache` out of the Option**, so it cannot be reused in the exit block. We must change to `cache.as_deref_mut()`.
- Tail return (line ~688): currently `self.out_proj.forward_on(&normed_flat, target)` — **direct tail-return expression, no `out` variable**. We must refactor to `let out = self.out_proj.forward_on(&normed_flat, target)?;` + exit barrier + `Ok(out)`.
- Cache offset accessor: `GatedDeltaCache::offsets(&self) -> &[i32]` (file `ironmlx/src/core/cache/gated_delta.rs:67`). Returns per-row offsets of length `batch`. **No** `offset()` accessor. B=1 HTTP-path harness — `offsets().first()` is the canonical scalar. Do NOT fallback to `0` when cache exists; only when `cache.is_none()`.

Apply three Edits to `forward_on`:

#### Edit 1: Entry barrier (insert after pre-flight validation, just before "Step 1: in_proj_qkvz")

```rust
// P5g T0 Layer 1 entry: materialize input + cache states before timer starts.
// Drains prior lazy ops so they're not attributed to GatedDeltaNet's forward
// cost; also forces cache.conv_state + cache.recurrent_state to be tangible
// so Step 2c/7e cache updates produced inside this forward are the only
// cache-related materialization captured by the exit barrier.
#[cfg(feature = "p5g-profile")]
let _p5g_timer_start = {
    let mode = profile_mode();
    if mode != ProfileMode::Off {
        let mut eval_set: Vec<&Array> = vec![x];
        if let Some(c) = cache.as_deref() {
            eval_set.push(c.conv_state());
            eval_set.push(c.recurrent_state());
        }
        if let Some(m) = mask {
            eval_set.push(m);
        }
        mlx::transforms::eval(&eval_set[..])?;
        Some((mode, std::time::Instant::now()))
    } else {
        None
    }
};

// Capture offset_before BEFORE any cache-modifying op in Steps 2c / 7e.
// HTTP-path B=1 invariant: offsets() always has at least one element when
// cache exists. unwrap_or(0) only fires for the cache.is_none() branch.
#[cfg(feature = "p5g-profile")]
let _p5g_offset_before: i32 = cache
    .as_deref()
    .and_then(|c| c.offsets().first().copied())
    .unwrap_or(0);
```

#### Edit 2: Step 7e cache update at line ~667 — switch from move to mutable borrow

Before:

```rust
if let Some(c) = cache {
    c.update_recurrent(new_state);
    let lens_owned: Vec<i32>;
    let lens_ref: &[i32] = match per_row_lens {
        Some(l) => l,
        None => {
            lens_owned = vec![seq; batch as usize];
            &lens_owned
        }
    };
    c.advance(lens_ref)?;
}
```

After (single-keyword change — `cache` → `cache.as_deref_mut()`):

```rust
if let Some(c) = cache.as_deref_mut() {
    c.update_recurrent(new_state);
    let lens_owned: Vec<i32>;
    let lens_ref: &[i32] = match per_row_lens {
        Some(l) => l,
        None => {
            lens_owned = vec![seq; batch as usize];
            &lens_owned
        }
    };
    c.advance(lens_ref)?;
}
```

Functionally identical inside the block; preserves the `cache` Option binding for the exit barrier in Edit 3.

#### Edit 3: Refactor tail return at line ~688 — bind `out` + exit barrier + explicit `Ok(out)`

Before (lines ~686-688):

```rust
let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
let normed_flat = normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
self.out_proj.forward_on(&normed_flat, target)
```

After:

```rust
let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
let normed_flat = normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
let out = self.out_proj.forward_on(&normed_flat, target)?;

#[cfg(feature = "p5g-profile")]
{
    if let Some((mode, start)) = _p5g_timer_start {
        // Materialize all GDN produced outputs INCLUDING updated cache states.
        let mut eval_out: Vec<&Array> = vec![&out];
        if let Some(c) = cache.as_deref() {
            eval_out.push(c.conv_state());
            eval_out.push(c.recurrent_state());
        }
        mlx::transforms::eval(&eval_out[..])?;
        let elapsed_us = start.elapsed().as_micros() as u64;

        // offset_after read AFTER cache.advance() in Step 7e has executed.
        let offset_after: i32 = cache
            .as_deref()
            .and_then(|c| c.offsets().first().copied())
            .unwrap_or(0);
        let offset_before = _p5g_offset_before;

        // tracing::info! placed strictly AFTER timer-related work — no log
        // calls inside the measured window. Uses mode.as_str() for
        // env-name / log-name consistency (defined in Step 0.2).
        let layer = self.profile_layer_idx.unwrap_or(-1);
        let dims = x.shape();
        let dvec = dims.as_slice();
        let batch_dim = if !dvec.is_empty() { dvec[0] } else { 0 };
        let seq_dim = if dvec.len() >= 2 { dvec[1] } else { 0 };
        tracing::info!(
            "[p5g-profile] mode={} layer={} batch={} seq={} \
             offset_before={} offset_after={} elapsed_us={}",
            mode.as_str(), layer, batch_dim, seq_dim,
            offset_before, offset_after, elapsed_us
        );
    }
}

Ok(out)
```

- [ ] Apply the three Edits. Build with feature (pipefail set to ensure cargo failures surface):

```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tee /tmp/p5g-build-step07.log
tail -10 /tmp/p5g-build-step07.log
```

Expected: `Finished release profile`. On compile error, re-verify line numbers — `forward_on` may have drifted from this plan's snapshot (`git log -p -- ironmlx/src/nn/gated_delta_net.rs` shows recent changes).

### Step 0.8: Hygiene chain with feature

Run:
```bash
set -o pipefail
cd "$REPO"
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo build --release 2>&1 | tail -3
```
Expected: fmt clean, clippy 0 Rust warnings, release build PASS.

### Step 0.9: Sentinel + batched + http_smoke (default build, no profile)

This confirms profile feature is truly gated — default build behavior unchanged.

```bash
set -o pipefail
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: smoke 2/2 PASS (argmax=11), batched 1/1 PASS, http_smoke 1/1 PASS.

### Step 0.10: Commit instrumentation infrastructure

```bash
set -o pipefail
cd "$REPO"
git add ironmlx/Cargo.toml ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<'EOF'
test(p5g-t0): add gated_delta_net profile instrumentation

Adds `p5g-profile` Cargo feature, default off. When enabled and
IRONMLX_P5G_PROFILE_MODE env var is set, instruments
GatedDeltaNet::forward_on:

  - ProfileMode enum cached via OnceLock (no per-forward env lookup)
  - profile_layer_idx field parsed from `model.layers.{N}.linear_attn`
    prefix; from_components path stores None (unit-test)
  - Layer 1 boundary timing: eval(input set) -> timer.start ->
    forward -> eval(output set incl. cache.conv_state +
    cache.recurrent_state) -> timer.stop -> tracing::info! AFTER stop
  - Log schema: [p5g-profile] mode=<m> layer=<i32> batch=<i32>
    seq=<i32> offset_before=<i32> offset_after=<i32> elapsed_us=<u64>

Modes (extensible, layer2 + ablate-* land in Step 0.x as
profile-driven optimizations T1-T3 decide which to keep):
  - off (default; zero overhead, single cached-flag check only)
  - layer1: boundary-isolated timing
  - layer2: per-step barrier (Step 0.x adds breakdown)
  - ablate-compute-g / ablate-conv / ablate-t-arr (Step 0.x adds)

Public API unchanged. `from_loader` keeps its existing signature
— PIVOT from typed propagation: GatedDeltaNet is also called from
common nn/decoder_layer.rs (dense + MTP); prefix-parsing the layer
index avoids cascading the change to 6 files and dense/MTP scope
contamination.

Validation: default build + sentinel + batched + http_smoke PASS;
feature build clean (fmt + clippy --all-features).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -1
```

### Step 0.11: Add Layer 2 per-step barrier instrumentation

Read the 8-step body of `forward_on` (Steps 1a/1b/2a/2b/2c/3/4/5/6/7a-d/8 per spec § 1.3). For Layer 2 mode, insert per-step `mlx::transforms::eval(&[&intermediate])?;` + `Instant::elapsed()` capture between each step. Buffer per-step elapsed in a `Vec<u64>` on the stack; emit batched as `step_breakdown=...` in the log line at function exit.

The pattern at each step:

```rust
#[cfg(feature = "p5g-profile")]
let _p5g_step_start = if matches!(profile_mode(), ProfileMode::Layer2) {
    Some(std::time::Instant::now())
} else {
    None
};

// ... existing step code ...
let stepN_out = ...;

#[cfg(feature = "p5g-profile")]
let _ = if let Some(start) = _p5g_step_start {
    mlx::transforms::eval(&[&stepN_out])?;
    _p5g_step_elapsed.push((start.elapsed().as_micros() as u64));
};
```

Where `_p5g_step_elapsed` is `Vec<u64>` declared once at the top of `forward_on`:

```rust
#[cfg(feature = "p5g-profile")]
let mut _p5g_step_elapsed: Vec<u64> = if matches!(profile_mode(), ProfileMode::Layer2) {
    Vec::with_capacity(12)
} else {
    Vec::new()
};
```

At the exit log emit (Step 0.7's exit block), if `mode == Layer2`, also format `step_breakdown=` from `_p5g_step_elapsed`. The same `mode.as_str()` rule applies (no `{mode:?}` — must round-trip with env var):

```rust
#[cfg(feature = "p5g-profile")]
if matches!(mode, ProfileMode::Layer2) && !_p5g_step_elapsed.is_empty() {
    let breakdown: Vec<String> = _p5g_step_elapsed.iter().map(|us| us.to_string()).collect();
    tracing::info!(
        "[p5g-profile] mode={} layer={} step_breakdown={}",
        mode.as_str(), layer, breakdown.join(",")
    );
}
```

Preferred: extend Step 0.7's single-line log to include `step_breakdown=` as an additional whitespace-separated `key=value` field when `mode == Layer2`. This keeps the parser logic (Step 0.14 harness) uniform — one record per forward call, all fields on one line — instead of needing to associate paired log lines per layer.

- [ ] Apply Edits. Identify 8 step boundaries in the existing forward_on body and insert the timer captures. Build:
```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tail -3
```
Expected: Finished.

### Step 0.12: Add ablate-compute-g / ablate-conv / ablate-t-arr shape-preserving substitutes

For each Layer 3 candidate, the substitute must produce a same-shape / same-dtype output (downstream consumers unchanged) but skip the heavy compute:

- **ablate-compute-g** (Step 5): instead of `exp(-exp(A_log) * softplus(a + dt_bias))`, return `zeros_like(a)` cast to the right dtype. Output shape `[BS, num_v_heads]` (or whatever Step 5's normal output is).
- **ablate-conv** (Step 2a-c): instead of `concatenate(conv_state, qkv) → conv1d → silu` chain, return `qkv` directly (shape-preserving — qkv already matches the output shape of conv1d in this path).
- **ablate-t-arr** (Step 7c): instead of constructing `t_arr` from `(seq,).try_into()`, use a pre-allocated cached const Array (e.g., a `OnceLock<Array>` keyed by chunk_size).

Each gated by `matches!(profile_mode(), ProfileMode::AblateX)` branches in the appropriate step.

- [ ] Apply Edits. For each candidate, add an `if matches!(profile_mode(), ProfileMode::AblateX)` branch in the matching step that returns the substitute. Build to verify:
```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tail -3
```
Expected: Finished.

### Step 0.13: Commit Layer 2 + ablation instrumentation

```bash
set -o pipefail
cd "$REPO"
git add ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<'EOF'
test(p5g-t0): add Layer 2 per-step barriers + Layer 3 ablation substitutes

Extends p5g-profile instrumentation with:

  - Layer 2 mode: per-step Instant captures between 8 GDN steps,
    buffered to forward end, emitted as step_breakdown=us1,us2,...
    in the single tracing::info! line (NO log calls inside measured
    timer window).
  - ablate-compute-g: replace Step 5 compute_g chain with zeros_like
    cast to correct dtype (shape-preserving upper bound).
  - ablate-conv: replace Step 2a-c chain with qkv passthrough
    (shape-preserving; conv kernel skipped).
  - ablate-t-arr: bypass per-call t_arr construction with cached
    OnceLock<Array> keyed by chunk_size.

Each ablation is shape-preserving (downstream consumers see the
same Array shape/dtype), giving Layer 3 an upper bound on per-step
end-to-end wall-time reduction WITHOUT requiring the ablation
substitute to be mathematically valid. Layer 3 ablation results
are DIAGNOSTIC only — promote decisions per § 7.3 use real
optimization end-to-end iron-bench.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -2
```

### Step 0.14: Create T0 profile harness (complete — no placeholders, no stubs)

Create `ironmlx/tests/p5g_t0_gated_delta_profile.rs` as a COMPLETE 4-phase harness — no `phase_a.insert(pp, 0.0)` placeholders, no Phase C/D stubs. The Phase D set is the three pre-defined ablation modes from Step 0.12 (`ablate-compute-g`, `ablate-conv`, `ablate-t-arr`); if Phase C reveals a top step not covered by these three, Step 0.17 may add a new variant + re-run only that variant's Phase D leg.

Dependencies already available: `serde_json = "1"` in `ironmlx/Cargo.toml` (regular dep — accessible from integration tests). No Cargo.toml change needed.

```rust
//! P5g T0 — GatedDeltaNet 4-phase HTTP-path profile harness.
//!
//! Phase A: whole-prefill baseline (server NO profile mode, iron-bench sweep, median pp_tps)
//! Phase B: Layer 1 boundary-isolated (server mode=layer1, iron-bench sweep, parse [p5g-profile] log)
//! Phase C: Layer 2 per-step breakdown (server mode=layer2, iron-bench sweep, parse step_breakdown)
//! Phase D: Layer 3 ablation across 3 pre-defined modes from Step 0.12
//!          (ablate-compute-g, ablate-conv, ablate-t-arr); per-mode pp_tps median
//!          + delta vs Phase A
//!
//! Run:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5g-profile \
//!       --test p5g_t0_gated_delta_profile \
//!       -- --ignored --test-threads=1 --nocapture
//!
//! Output:
//!   /tmp/p5g-t0-phases.json — full parsed phase data for Step 0.18 report writing.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

const PP_LIST: [i32; 4] = [2048, 4096, 8192, 16384];
const WARMUP: usize = 1;
const RUNS: usize = 3;
const PROFILE_PORT: u16 = 18080;

const ABLATION_MODES: [&str; 3] = ["ablate-compute-g", "ablate-conv", "ablate-t-arr"];

fn snapshot_dir() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR env var")
}

fn output_path() -> PathBuf {
    PathBuf::from("/tmp/p5g-t0-phases.json")
}

/// Median of f64. Returns None on empty input. Panics if NaN present (iron-bench
/// shouldn't emit NaN; failure-fast surfaces upstream measurement bugs).
fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("pp_tps contained NaN"));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Spawn `cargo run -p iron-bench` (cross-package, can't use env!("CARGO_BIN_EXE_iron-bench")).
fn iron_bench_run(port: u16, model_dir: &str, prompt_len: i32) -> std::process::Output {
    Command::new("cargo")
        .args([
            "run", "-p", "iron-bench", "--release", "--",
            "--target", &format!("p5g_profile=http://127.0.0.1:{port}"),
            "--model", "qwen3.5-moe",
            "--model-dir", model_dir,
            "--prompt-len", &prompt_len.to_string(),
            "--max-tokens", "32",
            "--runs", &RUNS.to_string(),
            "--warmup", &WARMUP.to_string(),
            "--format", "json",
        ])
        .output()
        .expect("iron-bench spawn")
}

/// Parse iron-bench `--format json` stdout — extract `raw_runs[].pp_tps` values
/// (one per measured run). Panics with full stdout context on parse failure.
fn parse_pp_tps_from_bench(stdout_bytes: &[u8]) -> Vec<f64> {
    let s = String::from_utf8_lossy(stdout_bytes);
    let v: Value = serde_json::from_str(&s).unwrap_or_else(|e| {
        let preview: String = s.chars().take(400).collect();
        panic!("iron-bench JSON parse failed: {e}; raw stdout (first 400): {preview}")
    });
    let mut tps = Vec::new();
    if let Some(arr) = v.get("raw_runs").and_then(|x| x.as_array()) {
        for r in arr {
            if let Some(p) = r.get("pp_tps").and_then(|x| x.as_f64()) {
                tps.push(p);
            }
        }
    }
    tps
}

fn spawn_server(profile_mode: Option<&str>, model_dir: &str, port: u16) -> Child {
    let bin = env!("CARGO_BIN_EXE_ironmlx");
    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--model", model_dir,
        "--port", &port.to_string(),
        "--host", "127.0.0.1",
    ]);
    if let Some(mode) = profile_mode {
        cmd.env("IRONMLX_P5G_PROFILE_MODE", mode);
    }
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    cmd.stderr(Stdio::piped());
    cmd.spawn().expect("ironmlx serve spawn")
}

fn wait_for_ready(port: u16, max_seconds: u64) {
    let url = format!("http://127.0.0.1:{port}/healthz");
    let deadline = std::time::Instant::now() + Duration::from_secs(max_seconds);
    loop {
        if let Ok(out) = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
        {
            if String::from_utf8_lossy(&out.stdout).trim() == "200" {
                return;
            }
        }
        if std::time::Instant::now() > deadline {
            panic!("ironmlx serve did not become ready within {max_seconds}s");
        }
        std::thread::sleep(Duration::from_secs(3));
    }
}

/// Parse `[p5g-profile]` log lines from server stderr.
/// Each line example (Layer 1):
///   [p5g-profile] mode=layer1 layer=12 batch=1 seq=2048 offset_before=4096 offset_after=6144 elapsed_us=15301
/// Layer 2 additionally has: step_breakdown=us1,us2,us3,...
/// Returns one record (k=v map) per matched line. `mode` value is the env-name
/// (`layer1` / `layer2`), NOT the Debug form — ProfileMode::as_str() guarantees.
fn parse_profile_log(stderr_bytes: &[u8]) -> Vec<BTreeMap<String, String>> {
    let mut records = Vec::new();
    for line in BufReader::new(stderr_bytes).lines().filter_map(|l| l.ok()) {
        if let Some(rest) = line.split_once("[p5g-profile] ").map(|(_, r)| r) {
            let mut rec: BTreeMap<String, String> = BTreeMap::new();
            for kv in rest.split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    rec.insert(k.to_string(), v.to_string());
                }
            }
            records.push(rec);
        }
    }
    records
}

/// Sweep across PP_LIST against an already-spawned server, return per-PP median pp_tps.
fn run_pp_sweep(model_dir: &str, port: u16) -> BTreeMap<i32, f64> {
    let mut result: BTreeMap<i32, f64> = BTreeMap::new();
    for &pp in &PP_LIST {
        let out = iron_bench_run(port, model_dir, pp);
        if !out.status.success() {
            eprintln!("[p5g-t0] iron-bench failed at PP={pp}: exit={}", out.status);
            eprintln!("stderr: {}", String::from_utf8_lossy(&out.stderr));
            panic!("iron-bench failed at PP={pp}");
        }
        let tps_list = parse_pp_tps_from_bench(&out.stdout);
        let med = median(tps_list).expect("no pp_tps in iron-bench output");
        eprintln!("[p5g-t0] PP={pp}: pp_tps median = {:.2}", med);
        result.insert(pp, med);
    }
    result
}

/// Drain server.stderr into a shared buffer while the sweep runs; parse on shutdown.
fn run_sweep_capturing_stderr(
    model_dir: &str,
    port: u16,
    server: &mut Child,
) -> (BTreeMap<i32, f64>, Vec<BTreeMap<String, String>>) {
    let stderr_buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let stderr_handle = server.stderr.take().expect("server stderr");
    let buf_clone = std::sync::Arc::clone(&stderr_buf);
    let stderr_thread = std::thread::spawn(move || {
        let mut rdr = BufReader::new(stderr_handle);
        let mut local = Vec::new();
        let _ = rdr.read_to_end(&mut local);
        buf_clone.lock().unwrap().extend_from_slice(&local);
    });
    let tps_map = run_pp_sweep(model_dir, port);
    // Sweep done. Server stays alive until shutdown_server() is called by caller.
    let captured_so_far = stderr_buf.lock().unwrap().clone();
    let records = parse_profile_log(&captured_so_far);
    // stderr_thread continues to drain after this point until the server is killed,
    // but we already captured everything emitted DURING the sweep.
    drop(stderr_thread); // detach; final lines lost on shutdown but were after sweep ended
    (tps_map, records)
}

fn shutdown_server(server: &mut Child) {
    let _ = server.kill();
    let _ = server.wait();
    std::thread::sleep(Duration::from_secs(3));
}

#[test]
#[ignore]
fn p5g_t0_gated_delta_profile_4phase() {
    let model_dir = snapshot_dir();
    eprintln!("[p5g-t0] starting 4-phase harness; model={model_dir}");

    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    out.insert("pp_list".into(), json!(PP_LIST));
    out.insert("warmup".into(), json!(WARMUP));
    out.insert("runs".into(), json!(RUNS));
    out.insert("model_dir".into(), json!(model_dir));

    // ===== Phase A =====
    eprintln!("[p5g-t0] Phase A: ironmlx serve (NO profile mode) — whole-prefill baseline");
    let mut server = spawn_server(None, &model_dir, PROFILE_PORT);
    wait_for_ready(PROFILE_PORT, 300);
    let phase_a = run_pp_sweep(&model_dir, PROFILE_PORT);
    shutdown_server(&mut server);
    out.insert("phase_a_pp_tps".into(), json!(phase_a.iter()
        .map(|(k, v)| (k.to_string(), *v)).collect::<BTreeMap<_, _>>()));

    // ===== Phase B =====
    eprintln!("[p5g-t0] Phase B: IRONMLX_P5G_PROFILE_MODE=layer1 — boundary-isolated GDN");
    let mut server = spawn_server(Some("layer1"), &model_dir, PROFILE_PORT);
    wait_for_ready(PROFILE_PORT, 300);
    let (phase_b_tps, phase_b_records) =
        run_sweep_capturing_stderr(&model_dir, PROFILE_PORT, &mut server);
    shutdown_server(&mut server);
    out.insert("phase_b_pp_tps".into(), json!(phase_b_tps.iter()
        .map(|(k, v)| (k.to_string(), *v)).collect::<BTreeMap<_, _>>()));
    out.insert("phase_b_records".into(), json!(phase_b_records));
    eprintln!("[p5g-t0] Phase B captured {} profile records", phase_b_records.len());

    // ===== Phase C =====
    eprintln!("[p5g-t0] Phase C: IRONMLX_P5G_PROFILE_MODE=layer2 — per-step breakdown");
    let mut server = spawn_server(Some("layer2"), &model_dir, PROFILE_PORT);
    wait_for_ready(PROFILE_PORT, 300);
    let (phase_c_tps, phase_c_records) =
        run_sweep_capturing_stderr(&model_dir, PROFILE_PORT, &mut server);
    shutdown_server(&mut server);
    out.insert("phase_c_pp_tps".into(), json!(phase_c_tps.iter()
        .map(|(k, v)| (k.to_string(), *v)).collect::<BTreeMap<_, _>>()));
    out.insert("phase_c_records".into(), json!(phase_c_records));
    eprintln!(
        "[p5g-t0] Phase C captured {} profile records (step_breakdown included)",
        phase_c_records.len()
    );

    // ===== Phase D =====
    let mut phase_d_results: BTreeMap<String, BTreeMap<i32, f64>> = BTreeMap::new();
    for &abl_mode in &ABLATION_MODES {
        eprintln!("[p5g-t0] Phase D[{abl_mode}]: IRONMLX_P5G_PROFILE_MODE={abl_mode}");
        let mut server = spawn_server(Some(abl_mode), &model_dir, PROFILE_PORT);
        wait_for_ready(PROFILE_PORT, 300);
        let tps_map = run_pp_sweep(&model_dir, PROFILE_PORT);
        shutdown_server(&mut server);
        phase_d_results.insert(abl_mode.to_string(), tps_map);
    }
    out.insert(
        "phase_d_pp_tps".into(),
        json!(phase_d_results.iter()
            .map(|(mode, tps)| (
                mode.clone(),
                tps.iter().map(|(k, v)| (k.to_string(), *v)).collect::<BTreeMap<_, _>>()
            ))
            .collect::<BTreeMap<_, _>>()),
    );

    // ===== Write output =====
    let v = serde_json::to_value(&out).expect("serialize phases");
    let json_str = serde_json::to_string_pretty(&v).expect("pretty print");
    let mut f = std::fs::File::create(output_path())
        .unwrap_or_else(|e| panic!("create {}: {e}", output_path().display()));
    f.write_all(json_str.as_bytes()).unwrap();

    eprintln!(
        "[p5g-t0] complete. {} bytes written to {}",
        json_str.len(),
        output_path().display()
    );
    eprintln!("[p5g-t0] Phase A baselines: {phase_a:?}");
    eprintln!("[p5g-t0] Phase B Layer 1 baselines: {phase_b_tps:?}");
    eprintln!("[p5g-t0] Phase C Layer 2 baselines: {phase_c_tps:?}");
    eprintln!("[p5g-t0] Phase D ablation results: {phase_d_results:?}");
}
```

- [ ] Create the file with this content. **Hard check**: grep for placeholders / stubs before committing.

```bash
set -o pipefail
grep -nE "phase_._.*insert\(pp, 0\.0\)|Phase [CD] require manual|TODO|FIXME|placeholder|stub" \
  "$REPO/ironmlx/tests/p5g_t0_gated_delta_profile.rs" && {
    echo "[error] harness contains forbidden placeholder/stub markers — fix before commit" >&2
    exit 1
} || echo "[ok] no placeholder/stub markers in harness"
```

- [ ] Compile test binary:

```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5g-profile --test p5g_t0_gated_delta_profile --no-run 2>&1 | tee /tmp/p5g-build-step014.log
tail -5 /tmp/p5g-build-step014.log
```

Expected: `Finished release profile [optimized] target(s)`. No `--no-run` failure.

### Step 0.15: Run T0 profile (Phase A + B)

Run the harness:

```bash
set -o pipefail
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5g-profile \
  --test p5g_t0_gated_delta_profile \
  -- --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/p5g-t0-phases.log
```

Expected duration: 4 PP × 4 runs × 2 phases × ~10-30s per run = ~20 min. PP=16384 Phase B will take longer due to profile barriers (estimated 2-3× slowdown).

Capture: stderr `[p5g-profile]` lines per layer per PP (Phase B), iron-bench wall-time output per PP (Phase A).

- [ ] Extract Phase A medians (PP=2048/4096/8192/16384 prefill tok/s) into a notebook for the report.
- [ ] Aggregate Phase B layer-elapsed records: group by `layer` field, sum 30 layers per PP, divide by 30 for per-layer median; total GDN time per PP = sum of all 30 layer times per PP.

### Step 0.16: Extend harness for Phase C + D, run Phase C

Extend the test function to:

1. After Phase B records aggregated, identify which steps dominate (need Phase C to see per-step breakdown).
2. Spawn `IRONMLX_P5G_PROFILE_MODE=layer2`, run same sweep, parse `step_breakdown=us1,us2,...`, aggregate per-step times across 30 layers.
3. Rank the 12-ish steps by total time; identify top 3.

```bash
set -o pipefail
# Re-compile + re-run (Phase C addition)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5g-profile \
  --test p5g_t0_gated_delta_profile \
  -- --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/p5g-t0-phases.log
```

- [ ] Run Phase C. Top 3 steps identified.

### Step 0.17: Extend harness for Phase D (per top-3 ablation)

For each of top-3 steps identified in Phase C, the harness:

1. Spawns server with appropriate `IRONMLX_P5G_PROFILE_MODE=ablate-<X>` (per Step 0.12 — extend Step 0.12 with more `AblateX` variants if Phase C's top-3 includes a step not yet covered).
2. Runs iron-bench sweep, parses Phase A baseline output (whole-prefill tok/s with the ablation).
3. Computes wall-time delta vs Phase A baseline = **upper bound** of that candidate's reachable optimization.

- [ ] Run Phase D. Each top-3 candidate has an upper-bound cut %.

### Step 0.18: Write reports/p5g-t0-gated-delta-profile.md

Create the report:

```markdown
# P5g T0 — GatedDeltaNet Independent Profile

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Branch | ironmlx-p5g-perf |
| HEAD | <fill from `git rev-parse HEAD` after T0 commit> |
| Methodology | 3-layer protocol per spec § 3.2; HTTP-path via instrumented server + iron-bench (same path as P5f HTTP baseline 1844 tok/s); 1 warmup + 3 measured median per PP |

## §1 Phase A — whole-prefill baseline (no profile mode)

| PP | iron-bench prefill (tok/s median) | wall-time per request (ms) |
|---:|---:|---:|
| 2048 | <fill> | <fill> |
| 4096 | <fill> | <fill> |
| 8192 | <fill> | <fill> |
| 16384 | <fill> | <fill> |

## §2 Phase B — Layer 1 boundary-isolated GDN estimate

Per-layer GatedDeltaNet wall-time (boundary-isolated, includes Layer 1 instrumentation overhead). 30 GDN layers per forward; aggregate sum below.

| PP | Per-layer GDN median (ms) | 30-layer total (ms) | Whole-prefill total (ms, from Phase A) | GDN occupancy estimate (%) |
|---:|---:|---:|---:|---:|
| 2048 | <fill> | <fill> | <fill> | <fill>% |
| 4096 | <fill> | <fill> | <fill> | <fill>% |
| 8192 | <fill> | <fill> | <fill> | <fill>% |
| 16384 | <fill> | <fill> | <fill> | <fill>% |

**Annotation**: Layer 1 is boundary-isolated estimate (含 entry/exit eval barrier
引入的轻 instrumentation overhead), 不等于完全无 instrumentation 下 GatedDeltaNet
的 ground-truth 占比 — 后者在 lazy MLX 下不可直接测；端到端 ROI 仍以 T1-T3
实施后 iron-bench benchmark 为准。

## §3 Phase C — Layer 2 per-step breakdown

| Step | Mean elapsed_us (across 30 layers × 4 PP) | % of GDN total (Layer 2) |
|---|---:|---:|
| 1a in_proj_qkvz | <fill> | <fill>% |
| 1b in_proj_ba | <fill> | <fill>% |
| 2a concatenate(conv_state, qkv) | <fill> | <fill>% |
| 2b conv1d + silu | <fill> | <fill>% |
| 2c update conv_state | <fill> | <fill>% |
| 3 split per-head | <fill> | <fill>% |
| 4 q/k rms_norm + scale | <fill> | <fill>% |
| 5 compute_g | <fill> | <fill>% |
| 6 beta = sigmoid | <fill> | <fill>% |
| 7a-d kernel dispatch | <fill> | <fill>% |
| 8 RmsNormGated + out_proj | <fill> | <fill>% |

**Layer 2 / Layer 1 slowdown ratio**: <fill>× (e.g. 1.4-1.8× typical for per-op
barrier instrumentation per P5e T0 precedent).

**Top 3 ranked candidates**: <fill from this table>. Per-step relative % is
**indicative only** (barrier 改变 fusion 边界); promote decisions use end-to-end
benchmark (§ 7.3).

## §4 Phase D — Layer 3 shape-preserving cost ablation (upper bound)

For each top-3 candidate identified in §3, the harness runs ironmlx with
`IRONMLX_P5G_PROFILE_MODE=ablate-<candidate>` (shape-preserving substitute
maintaining output shape/dtype/downstream consumption). Wall-time delta vs
Phase A = upper bound of reachable optimization.

| Candidate | Ablation substitute | PP=2048 prefill delta vs Phase A | PP=16384 delta | Upper bound cut (geomean %) |
|---|---|---:|---:|---:|
| <top1> | <substitute description> | <fill> | <fill> | <fill>% |
| <top2> | <substitute description> | <fill> | <fill> | <fill>% |
| <top3> | <substitute description> | <fill> | <fill> | <fill>% |

**Note**: upper bound 仅作 T0/T1 优化值得性诊断, 不作 ship 依据 (per § 3.2 + § 4.2).
端到端 ROI 仍以 T1-T3 实现后 iron-bench benchmark 为准 (§ 7.3 ship metrics).

## §5 P5g ceiling 推导 (using §1-§4 data)

Apply § 7.1 数学结构: PP=2048 baseline 1.111s × GDN 占比 (§2) × 候选 upper-bound
cut (§4) = realistic P5g target.

| PP | Phase A baseline (tok/s) | GDN occupancy | Best candidate cut | Realistic P5g target | omlx+10% target (§1.1 spec) | Gap remaining |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | <fill> | <fill>% | <fill>% | <fill> | 4653 | <fill>% |
| 4096 | <fill> | <fill>% | <fill>% | <fill> | 4855 | <fill>% |
| 8192 | <fill> | <fill>% | <fill>% | <fill> | 4782 | <fill>% |
| 16384 | <fill> | <fill>% | <fill>% | <fill> | 4252 | <fill>% |

## §6 P5g T1-T3 candidate ranking + decision

Based on §3 (per-step) + §4 (upper bound):

| Rank | Candidate (spec § 4.1) | Layer 2 weight | Layer 3 upper bound | Implementation complexity | Scope gate |
|---|---|---:|---:|---|---|
| T1 | <fill: C1/C2/C3/C4> | <fill>% | <fill>% | <op-level / kernel-level> | <yes/no Boss gate> |
| T2 | <fill> | <fill>% | <fill>% | <fill> | <fill> |
| T3 | <fill> | <fill>% | <fill>% | <fill> | <fill> |

**Sanity gate (§ 3.4)**: Layer 1 GDN occupancy ≥ 10% (P5f baseline 1.111s →
110 ms). Result: <PASS/FAIL>. If FAIL → Boss decision needed.

**Layer 3 sanity gate (§ 3.4)**: at least 1 candidate upper-bound > 5%. Result:
<PASS/FAIL>. If FAIL → P5g scope insufficient,回 Boss 决策.

**P5g scope gate per § 4.1**: any candidate requires Metal kernel rewrite?
<list any>. If yes → pause for Boss decision before T1.

## §7 Lock § 7.2 final target

Write back to `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md`
§ 7.2 with concrete numbers from §5 above. Provisional table replaced with
actual ceiling.
```

- [ ] Create the file. Fill all `<fill>` placeholders from harness output. No `<fill>` may remain.

### Step 0.19: Lock spec § 7.2 final target

Read current spec § 7.2 (which states "TBD by T0.a"). Replace with the locked target table using §5 from the report. Provisional language replaced by concrete numbers (e.g. "PP=2048 target ≥ 2050 tok/s based on 22% Layer 1 occupancy × 45% best ablation cut").

- [ ] Apply Edit to spec § 7.2. Verify no "TBD by T0.a" or "<fill>" remains in spec § 7.2.

### Step 0.20: Hygiene chain + final integration tests with profile feature off

```bash
set -o pipefail
cd "$REPO"
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo build --release 2>&1 | tail -3

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -5
```

Expected: all clean + PASS. Confirms profile feature is fully gated off in default build.

### Step 0.21: Commit T0

```bash
set -o pipefail
cd "$REPO"
git add ironmlx/tests/p5g_t0_gated_delta_profile.rs \
        reports/p5g-t0-gated-delta-profile.md \
        docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md \
        ironmlx/src/nn/gated_delta_net.rs
git status --short
git commit -m "$(cat <<'EOF'
test(p5g-t0): 4-phase GatedDeltaNet profile harness + report + spec § 7.2 lock

Adds tests/p5g_t0_gated_delta_profile.rs — HTTP-path harness that:
  Phase A: spawns ironmlx serve (no profile mode), runs iron-bench
           PP=2048/4096/8192/16384 (1 warmup + 3 measured median),
           captures whole-prefill baseline tok/s
  Phase B: spawns server with IRONMLX_P5G_PROFILE_MODE=layer1,
           parses [p5g-profile] log lines, aggregates 30-layer GDN
           per-PP total + per-layer median
  Phase C: spawns server with IRONMLX_P5G_PROFILE_MODE=layer2,
           parses step_breakdown=, aggregates per-step time across
           all GDN forward calls, computes Layer 2/Layer 1 slowdown
  Phase D: per top-3 candidates from C, spawns server with
           IRONMLX_P5G_PROFILE_MODE=ablate-<X>, records wall-time
           delta vs Phase A baseline (= upper bound)

server binary: env!("CARGO_BIN_EXE_ironmlx") (same package).
iron-bench: std::process::Command "cargo run -p iron-bench"
(cross-package; CARGO_BIN_EXE_iron-bench NOT injected in ironmlx tests).

Captured data:
  - Phase A whole-prefill baseline: PP=2048 <X> tok/s, ...
  - Phase B Layer 1 GDN occupancy estimate: PP=2048 <X>%, ...
  - Phase C top-3 step ranking: <X>, <Y>, <Z>
  - Phase D upper bounds: <X>% / <Y>% / <Z>%

Sanity gates per § 3.4:
  - Layer 1 occupancy ≥ 10%: <PASS/FAIL>
  - At least 1 ablation > 5% upper bound: <PASS/FAIL>

Scope gate per § 4.1: <none / list candidates needing kernel rewrite>.

Spec § 7.2 final target locked based on §5 ceiling derivation
(reports/p5g-t0-gated-delta-profile.md).

P5g T1-T3 will pursue: T1=<X>, T2=<Y>, T3=<Z> in ranked order.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -3
```

Fill `<X>` placeholders in commit message with actual measured numbers from the report.

---

## Task 1: First Profile-Driven Optimization (highest T0.c ranked)

**Goal:** Implement the highest single-ROI optimization candidate identified by T0.c. Promote per § 7.3 ship metrics or revert per § 4.2 Tn template.

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (specific edit determined by T0.c ranking — likely Step 5 compute_g cache, Step 2 conv path rewrite, Step 7c t_arr cache, or similar)

### Step 1.0: Capture T1-start baseline (REQUIRED before implementation)

T1 promote/revert decision (Step 1.9) compares measured T1 against **T1-start baseline** — not P5f baseline, not T0 phase data. Baseline must be captured against the same harness configuration that Steps 1.6-1.8 will use, on the SAME HEAD that T1 implementation starts from. This avoids attributing day-to-day GPU variance, model-load noise, or PATH/env drift to T1.

```bash
set -o pipefail
cd "$REPO"
T1_START_SHA=$(git rev-parse HEAD)
echo "[t1-baseline] T1-start HEAD = $T1_START_SHA" | tee /tmp/p5g-t1-start.txt

# Verify the working tree is clean (no T1 code yet).
if [ -n "$(git status --short)" ]; then
  echo "[error] working tree not clean; commit or stash before capturing T1-start baseline" >&2
  exit 1
fi

# Port guard.
if lsof -ti :8080 > /dev/null 2>&1; then
  echo "[guard] port 8080 in use; abort." >&2
  exit 1
fi

export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 \
  2> /tmp/p5g-t1-start-server.log &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

# Same sweep config used in Steps 1.6 + 1.7 + 1.8 — runs=3, warmup=1, max-tokens=32.
# Short PP (Step 1.6 mirror):
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target p5g_t1_start=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512 --max-tokens 32 --runs 3 --warmup 1 \
  --format json > /tmp/p5g-t1-start-short.json

# Long PP (Step 1.7 mirror):
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target p5g_t1_start=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 2048,4096,8192,16384 --max-tokens 32 --runs 3 --warmup 1 \
  --format json > /tmp/p5g-t1-start-long.json

# Decode TG (Step 1.8 mirror):
MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target p5g_t1_start=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,2048,16384 --max-tokens 32 --runs 3 --warmup 1 \
  --format json > /tmp/p5g-t1-start-decode.json

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done

# Aggregate baseline medians (machine-readable for Step 1.9 table).
python3 <<EOF > /tmp/p5g-t1-start-medians.json
import json, statistics
def med(xs): return statistics.median(xs) if xs else None
def extract(path, field):
    with open(path) as f: d = json.load(f)
    out = {}
    for r in d.get("raw_runs", []):
        pp = r["pp_target"]
        out.setdefault(pp, []).append(r.get(field))
    return {pp: med([v for v in vs if v is not None]) for pp, vs in out.items()}
print(json.dumps({
    "t1_start_sha": "$T1_START_SHA",
    "short_pp_tps": extract("/tmp/p5g-t1-start-short.json", "pp_tps"),
    "long_pp_tps":  extract("/tmp/p5g-t1-start-long.json",  "pp_tps"),
    "decode_tg_tps": extract("/tmp/p5g-t1-start-decode.json", "tg_tps"),
}, indent=2))
EOF
cat /tmp/p5g-t1-start-medians.json
```

- [ ] Confirm `/tmp/p5g-t1-start-medians.json` exists with non-null values for all 6 PP × 3 metrics. This file is the ONLY authoritative T1-start reference for Step 1.9.

### Step 1.1: Identify candidate from T0 report

Read `reports/p5g-t0-gated-delta-profile.md` § 6 ranking. T1 implements the rank-1 (highest single-ROI) candidate.

- [ ] Confirm T1 candidate name (e.g. C1 compute_g chain cache, C4 t_arr cache, etc.). Read the corresponding source line range in `ironmlx/src/nn/gated_delta_net.rs` per spec § 4.1 (line refs).
- [ ] Confirm Scope gate per § 4.1: candidate is op-level (Rust/MLX op rearrangement / `mlx::fast` reuse / constant cache / graph shape). If T1 requires new Metal kernel → **STOP and report DONE_WITH_CONCERNS** with the specific scope gate trigger; do not proceed without Boss decision.

### Step 1.2: Implement T1 optimization

Apply the specific Edit to `gated_delta_net.rs` per the T1 candidate. For example, if T1 = "C4 t_arr cache":

```rust
// At top of gated_delta_net.rs (module level):
static T_ARR_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<i32, Array>>> =
    std::sync::OnceLock::new();

// In forward_on, replace `let t_arr: Array = ((seq,), ()).try_into()?;` with:
let t_arr = {
    let cache = T_ARR_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(arr) = guard.get(&seq) {
        arr.clone()
    } else {
        let arr: Array = ((seq,), ()).try_into()?;
        guard.insert(seq, arr.clone());
        arr
    }
};
```

(Adapt to actual code structure. Mutex is fine here — only contended on first call per chunk-size.)

If T1 = "C1 compute_g chain cache":

Add a `compute_g_const` precomputed field to `GatedDeltaNet` (gated by `#[cfg(not(...))]` or stored unconditionally since it's a static computation result):

```rust
// In GatedDeltaNet struct:
neg_exp_a_log_f32: Array, // precomputed: -exp(a_log.astype(f32)), constant across forwards

// In from_loader / from_components, compute once:
let a_log_f32 = mlx::ops::cast::astype(&a_log, Dtype::Float32)?;
let exp_alog = a_log_f32.exp()?;
let neg_exp_a_log_f32 = mlx::ops::binary::negative(&exp_alog)?;
mlx::transforms::eval(&[&neg_exp_a_log_f32])?; // freeze it

// In forward_on Step 5, replace the per-call compute with the cached version:
let inner = &self.neg_exp_a_log_f32 * &sp;
let g = inner.exp()?;
```

(Spec § 4.1 C1 specifies this pattern.)

- [ ] Apply the Edit. Build verifies the change compiles.

### Step 1.3: Build + hygiene chain

```bash
set -o pipefail
cd "$REPO"
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
```
Expected: Finished + clean + 0 warnings.

### Step 1.4: Sentinel (argmax=11)

```bash
set -o pipefail
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 2/2 PASS, argmax=11.

If argmax shifts: Investigate logit margin. If logit margin large → accept new sentinel value and document in commit; if small → revert T1 per spec § 4.3.

### Step 1.5: Batched + http_smoke

```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 1/1 + 1/1 PASS.

### Step 1.6: iron-bench short-PP prefill smoke

```bash
set -o pipefail
# Verify port 8080 free before starting (port-targeted, not name-pattern;
# never auto-kill someone else's process on a shared host).
if lsof -ti :8080 > /dev/null 2>&1; then
  echo "[guard] port 8080 already bound by:" >&2
  lsof -i :8080 >&2
  echo "[guard] free it before re-running Step 1.6; refusing to auto-kill." >&2
  exit 1
fi

MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target p5g_t1=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512 --max-tokens 32 --runs 3 --warmup 1 \
  --format markdown 2>&1 | tail -20

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
# Confirm port 8080 is released (port-targeted, not name-pattern based).
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done
```

Capture PP=128 + PP=512 prefill medians. Compare to Tn-start baseline (committed HEAD before T1) — should be within ±2% (no regression).

### Step 1.7: iron-bench long-PP prefill sweep

```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target p5g_t1=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 2048,4096,8192,16384 --max-tokens 32 --runs 3 --warmup 1 \
  --format markdown 2>&1 | tail -20

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
# Confirm port 8080 is released (port-targeted, not name-pattern based).
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done
```

Capture PP=2048/4096/8192/16384 prefill medians. Compute **geometric mean** of these 4 numbers vs T1-start baseline geomean.

### Step 1.8: iron-bench decode TG smoke

```bash
set -o pipefail
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target p5g_t1=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,2048,16384 --max-tokens 32 --runs 3 --warmup 1 \
  --format markdown 2>&1 | tail -20

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
# Confirm port 8080 is released (port-targeted, not name-pattern based).
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done
```

Capture PP=128/2048/16384 decode TG (tg_tps) medians. Compare to T1-start baseline:
- PP=128/2048: regression < 2% required
- PP=16384: must keep +10.3% over omlx (P5f shipped advantage); regression < 2% from T1-start required

### Step 1.9: Promote / revert decision

Compile all measurements per § 7.3. **T1-start baseline column populated from `/tmp/p5g-t1-start-medians.json` (Step 1.0).** T1 measured column from Steps 1.6/1.7/1.8 medians.

| Metric | T1-start baseline | T1 measured | Delta | Threshold | Status |
|---|---|---|---|---|---|
| Long-PP prefill geomean | `<long_pp_tps geomean of 2048/4096/8192/16384 from /tmp/p5g-t1-start-medians.json>` | `<Step 1.7 geomean>` | <%>  | > +5% | <PASS/FAIL> |
| PP=2048 single | `<long_pp_tps[2048]>` | `<Step 1.7 PP=2048>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=4096 single | `<long_pp_tps[4096]>` | `<Step 1.7 PP=4096>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=8192 single | `<long_pp_tps[8192]>` | `<Step 1.7 PP=8192>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=16384 single | `<long_pp_tps[16384]>` | `<Step 1.7 PP=16384>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=128 prefill | `<short_pp_tps[128]>` | `<Step 1.6 PP=128>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=512 prefill | `<short_pp_tps[512]>` | `<Step 1.6 PP=512>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=128 decode TG | `<decode_tg_tps[128]>` | `<Step 1.8 PP=128>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=2048 decode TG | `<decode_tg_tps[2048]>` | `<Step 1.8 PP=2048>` | <%> | < 2% regression | <PASS/FAIL> |
| PP=16384 decode TG | `<decode_tg_tps[16384]>` | `<Step 1.8 PP=16384>` | <%> | < 2% regression | <PASS/FAIL> |
| sentinel + batched + http_smoke | PASS | <PASS/FAIL> | | ALL PASS | <PASS/FAIL> |

If ALL **promote** rows PASS:
- T1 promotes. Commit per Step 1.10.

If ANY row FAILS:
- T1 reverts. Revert `gated_delta_net.rs` to pre-T1 state via Edit (NOT `git checkout --` — preserve T0 instrument). Commit revert with negative ROI documentation.

### Step 1.10: Commit T1 (promote) or T1-revert

For promote:
```bash
set -o pipefail
cd "$REPO"
git add ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<'EOF'
feat(p5g-t1): <T1 candidate name> profile-driven optimization

T0 ranking (reports/p5g-t0-gated-delta-profile.md § 6): T1 = <candidate>.

Implementation: <2-3 line description of the actual code change>.

End-to-end iron-bench validation (relative to T1-start HEAD <prior commit SHA>):
  long-PP prefill geomean: <X> -> <Y> (+<%>)
  PP=128/512 prefill:      no regression (<X>%/<X>%)
  PP=2048 decode TG:       no regression (<X>%)
  PP=16384 decode TG:      no regression (<X>%), maintains +10.3% over omlx

Correctness gates (§ 5 sentinel suite):
  p5_qwen35_moe_smoke (argmax=11):     PASS
  p5_qwen35_moe_batched (B=2 row-eq):  PASS
  p5_qwen35_moe_http_smoke:            PASS

Hygiene: cargo build --release + cargo +nightly fmt --check +
  cargo +nightly clippy --all-features --workspace -- -D warnings PASS.

Per spec § 4.2 + § 7.3: T1 promote threshold met (long-PP geomean > 5%
+ all per-PP regressions < 2% + decode TG regressions < 2% + correctness
gates PASS). Promotes to default code path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -2
```

For revert (T1 failed promote):
```bash
set -o pipefail
# Apply Edit to revert gated_delta_net.rs to pre-T1 state (KEEP T0 instrument).
# Then:
git add ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<'EOF'
chore(p5g-t1): <T1 candidate> reverted — failed § 7.3 ship metrics

T1 candidate from T0 ranking: <candidate>.

Measured (relative to T1-start HEAD <prior commit SHA>):
  long-PP prefill geomean: <X> -> <Y> (<%> — <below threshold> / regression)
  PP=<N> regression: <%> (threshold 2%; <FAIL>)

[other failing rows here]

Code change reverted; T0 instrument retained.

T2 will pursue the next ranked candidate. T0 report § 6 remains
the authoritative ranking; this commit records the negative result.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -2
```

---

## Task 2: Second Profile-Driven Optimization (T0.c rank 2)

**Goal:** Same template as Task 1, applied to T0 ranking #2 candidate.

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (specific edit by T0.c ranking #2)

### Step 2.0: Capture T2-start baseline (REQUIRED before implementation)

Same template as Step 1.0, but baselines saved to `/tmp/p5g-t2-start-*.json` / `/tmp/p5g-t2-start-medians.json`, and `$T2_START_SHA=$(git rev-parse HEAD)` captured. T2-start HEAD is the branch HEAD AFTER T1 promote/revert lands (Step 1.10) — NOT a re-use of Step 1.0 data. Same sweep config (port 8080, PP same lists, runs=3, warmup=1, max-tokens=32, same harness JSON output).

```bash
set -o pipefail
cd "$REPO"
T2_START_SHA=$(git rev-parse HEAD)
echo "[t2-baseline] T2-start HEAD = $T2_START_SHA" | tee /tmp/p5g-t2-start.txt

if [ -n "$(git status --short)" ]; then
  echo "[error] working tree not clean; commit/stash before capturing T2-start baseline" >&2
  exit 1
fi
if lsof -ti :8080 > /dev/null 2>&1; then
  echo "[guard] port 8080 in use; abort." >&2
  exit 1
fi

export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 \
  2> /tmp/p5g-t2-start-server.log &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

for sweep in "short:128,512:pp_tps" "long:2048,4096,8192,16384:pp_tps" "decode:128,2048,16384:tg_tps"; do
  label=$(echo "$sweep" | cut -d: -f1)
  pps=$(echo "$sweep" | cut -d: -f2)
  MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
    --target p5g_t2_start=http://127.0.0.1:8080 \
    --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
    --prompt-len "$pps" --max-tokens 32 --runs 3 --warmup 1 \
    --format json > "/tmp/p5g-t2-start-$label.json"
done

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done

python3 <<EOF > /tmp/p5g-t2-start-medians.json
import json, statistics
def med(xs): return statistics.median(xs) if xs else None
def extract(path, field):
    with open(path) as f: d = json.load(f)
    out = {}
    for r in d.get("raw_runs", []):
        pp = r["pp_target"]
        out.setdefault(pp, []).append(r.get(field))
    return {pp: med([v for v in vs if v is not None]) for pp, vs in out.items()}
print(json.dumps({
    "t2_start_sha": "$T2_START_SHA",
    "short_pp_tps": extract("/tmp/p5g-t2-start-short.json", "pp_tps"),
    "long_pp_tps":  extract("/tmp/p5g-t2-start-long.json",  "pp_tps"),
    "decode_tg_tps": extract("/tmp/p5g-t2-start-decode.json", "tg_tps"),
}, indent=2))
EOF
cat /tmp/p5g-t2-start-medians.json
```

- [ ] Confirm `/tmp/p5g-t2-start-medians.json` exists with non-null values for all 6 PP × 3 metrics. Step 2.5 promote/revert table reads ONLY from this file.

### Step 2.1: Identify candidate

Read `reports/p5g-t0-gated-delta-profile.md` § 6 ranking. T2 implements the rank-2 candidate. **T2-start HEAD** = `$T2_START_SHA` from Step 2.0. All T2 measurements compare to T2-start HEAD, not P5f baseline, not T1-start.

- [ ] Confirm T2 candidate. Confirm scope gate (op-level). If kernel rewrite → STOP for Boss decision.

### Step 2.2: Implement T2 optimization

Apply the Edit per the T2 candidate (e.g., if T2 = "C2 stateful causal conv" — but if this requires kernel rewrite, scope gate triggers; pick a different candidate or pause for Boss).

Likely T2 candidates (depending on T0 ranking and what T1 took):
- If T1 was C1 compute_g: T2 might be C4 t_arr cache (also op-level)
- If T1 was C4 t_arr: T2 might be C1 compute_g (also op-level)
- C3 conv1d+silu fusion: only if `mlx::fast` already exposes a fused op; else scope gate

- [ ] Apply Edit. Build verify.

### Step 2.3: Build + hygiene + tests (same template as Task 1 Steps 1.3-1.8)

```bash
set -o pipefail
cd "$REPO"
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5

export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -5
```
Expected: all green, argmax=11.

### Step 2.4: iron-bench validation (same as Task 1 Step 1.6-1.8)

Run short-PP prefill smoke (PP=128/512), long-PP prefill sweep (PP=2048/4096/8192/16384), decode TG smoke (PP=128/2048/16384). Compare measurements to **T2-start HEAD baseline** (current branch HEAD at start of T2, not P5f baseline).

Repeat server-spawn / iron-bench commands from Step 1.6-1.8 (substituting `--target p5g_t2`).

### Step 2.5: Promote / revert decision per § 7.3 (same template as Step 1.9)

Compile the same metric table as Step 1.9, but baseline column reads from `/tmp/p5g-t2-start-medians.json` (NOT t1-start, NOT P5f). Threshold identical: long-PP geomean > +5% vs T2-start, per-PP regression < 2%, decode TG regression < 2%.

### Step 2.6: Commit T2 (promote or revert)

Use the same template as Step 1.10 with `feat(p5g-t2):` or `chore(p5g-t2):` prefix.

---

## Task 3: Third Profile-Driven Optimization (T0.c rank 3)

**Goal:** Same template as Task 1/2, applied to T0 ranking #3 candidate.

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (specific edit by T0.c ranking #3)

### Step 3.0: Capture T3-start baseline (REQUIRED before implementation)

Same template as Step 2.0, with baselines saved to `/tmp/p5g-t3-start-*.json` / `/tmp/p5g-t3-start-medians.json`, and `$T3_START_SHA=$(git rev-parse HEAD)` captured. T3-start HEAD is the branch HEAD AFTER T2 promote/revert lands. Identical sweep configuration (PP=128,512 / 2048,4096,8192,16384 / 128,2048,16384; runs=3; warmup=1; max-tokens=32). Substitute `t2` → `t3` throughout Step 2.0 shell block.

```bash
set -o pipefail
cd "$REPO"
T3_START_SHA=$(git rev-parse HEAD)
echo "[t3-baseline] T3-start HEAD = $T3_START_SHA" | tee /tmp/p5g-t3-start.txt

if [ -n "$(git status --short)" ]; then
  echo "[error] working tree not clean; commit/stash before capturing T3-start baseline" >&2
  exit 1
fi
if lsof -ti :8080 > /dev/null 2>&1; then
  echo "[guard] port 8080 in use; abort." >&2
  exit 1
fi

export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 \
  2> /tmp/p5g-t3-start-server.log &
SERVER_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

for sweep in "short:128,512:pp_tps" "long:2048,4096,8192,16384:pp_tps" "decode:128,2048,16384:tg_tps"; do
  label=$(echo "$sweep" | cut -d: -f1)
  pps=$(echo "$sweep" | cut -d: -f2)
  MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
    --target p5g_t3_start=http://127.0.0.1:8080 \
    --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
    --prompt-len "$pps" --max-tokens 32 --runs 3 --warmup 1 \
    --format json > "/tmp/p5g-t3-start-$label.json"
done

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done

python3 <<EOF > /tmp/p5g-t3-start-medians.json
import json, statistics
def med(xs): return statistics.median(xs) if xs else None
def extract(path, field):
    with open(path) as f: d = json.load(f)
    out = {}
    for r in d.get("raw_runs", []):
        pp = r["pp_target"]
        out.setdefault(pp, []).append(r.get(field))
    return {pp: med([v for v in vs if v is not None]) for pp, vs in out.items()}
print(json.dumps({
    "t3_start_sha": "$T3_START_SHA",
    "short_pp_tps": extract("/tmp/p5g-t3-start-short.json", "pp_tps"),
    "long_pp_tps":  extract("/tmp/p5g-t3-start-long.json",  "pp_tps"),
    "decode_tg_tps": extract("/tmp/p5g-t3-start-decode.json", "tg_tps"),
}, indent=2))
EOF
cat /tmp/p5g-t3-start-medians.json
```

- [ ] Confirm `/tmp/p5g-t3-start-medians.json` populated.

### Step 3.1-3.6: Same template as Task 2 Steps 2.1-2.6

Substitute T3 candidate (T0 ranking #3), `feat(p5g-t3):` / `chore(p5g-t3):` commit prefix, `/tmp/p5g-t3-start-medians.json` as the baseline-column source for Step 3.5 promote/revert table.

---

## Task 4: P5g Close-Out

**Goal:** Run full validation (sweep_full 19/19 + 4-way bench + clippy + fmt + integration tests), write self-contained `reports/p5g-final-results.md`, quantify P5h scope drivers, commit.

**Files:**
- Create: `reports/p5g-final-results.md`

### Step 4.1: Hygiene chain (sanity)

```bash
set -o pipefail
cd "$REPO"
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo build --release 2>&1 | tail -3
```
Expected: all clean.

### Step 4.2: Full integration tests

```bash
set -o pipefail
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: all PASS, argmax=11.

### Step 4.3: sweep_full regression gate

```bash
set -o pipefail
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh 2>&1 | tail -10
```
Expected: 19/19 PASS in ~140-160s. If a single transient flake (e.g. `b1_p2_3c_plus_chunked_admit_mid`), retry once. If second run also fails → STOP, report BLOCKED.

### Step 4.4: 4-way bench (ironmlx / mlx-lm / omlx, strict serial per [feedback_serial_perf_experiments])

This is the same procedure as P5f T3 close-out. 3 separate sweeps, one server up at a time.

**4.4.a — ironmlx sweep**:

```bash
set -o pipefail
# Pre-flight port guard (port-targeted, not name-pattern). Refuse to auto-kill
# someone else's process on the same host.
for port in 8080 8081 8082; do
  if lsof -ti :$port > /dev/null 2>&1; then
    echo "[guard] port $port already bound:" >&2
    lsof -i :$port >&2
    echo "[guard] free it before re-running Step 4.4; refusing to auto-kill." >&2
    exit 1
  fi
done

MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$IRONMLX_MOE_MODEL_DIR" --port 8080 --host 127.0.0.1 2> /tmp/p5g-ironmlx-server.log &
IRONMLX_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/healthz 2>/dev/null | grep -q "^200$"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:8080 \
  --model qwen3.5-moe --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1 \
  --format json > /tmp/p5g-ironmlx.json 2> /tmp/p5g-ironmlx.log
echo "[p5g-ironmlx sweep done]"; tail -3 /tmp/p5g-ironmlx.log

kill $IRONMLX_PID 2>/dev/null
wait $IRONMLX_PID 2>/dev/null || true
for _retry in 1 2 3 4 5; do
  lsof -ti :8080 > /dev/null 2>&1 || break
  sleep 2
done
```

**4.4.b — omlx sweep**:

```bash
set -o pipefail
SNAP_SHA=$(basename "$IRONMLX_MOE_MODEL_DIR")
( cd "$RIVALS_DIR/omlx" && \
  uv run omlx serve --model-dir "$IRONMLX_MOE_MODEL_DIR" --host 127.0.0.1 --port 8081 ) &
OMLX_PID=$!
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -qE "^(200|404)$"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target omlx=http://127.0.0.1:8081 \
  --model "$SNAP_SHA" --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1 \
  --format json > /tmp/p5g-omlx.json 2> /tmp/p5g-omlx.log

kill $OMLX_PID 2>/dev/null
wait $OMLX_PID 2>/dev/null || true
for _retry in 1 2 3 4 5; do
  lsof -ti :8081 > /dev/null 2>&1 || break
  sleep 2
done
```

**4.4.c — mlx-lm sweep**:

```bash
set -o pipefail
( cd "$REPO/scripts/bench-venvs/mlx-lm" && \
  uv run mlx_lm.server --model "$IRONMLX_MOE_MODEL_DIR" --host 127.0.0.1 --port 8082 --log-level INFO ) &
MLXLM_PID=$!
until curl -s -X POST http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}' 2>/dev/null | grep -q "choices"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target mlx_lm=http://127.0.0.1:8082 \
  --model default_model --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1 \
  --format json > /tmp/p5g-mlx_lm.json 2> /tmp/p5g-mlx_lm.log

kill $MLXLM_PID 2>/dev/null
wait $MLXLM_PID 2>/dev/null || true
for _retry in 1 2 3 4 5; do
  lsof -ti :8082 > /dev/null 2>&1 || break
  sleep 2
done
```

### Step 4.5: Aggregate medians + p95

```bash
set -o pipefail
cd "$REPO" && python3 <<'EOF' > /tmp/p5g-aggregate.md
import json, statistics

def load(path):
    with open(path) as f: return json.load(f)

PP_LIST = [128, 512, 2048, 4096, 8192, 16384]
TARGETS = [("ironmlx", "/tmp/p5g-ironmlx.json"),
           ("mlx_lm",  "/tmp/p5g-mlx_lm.json"),
           ("omlx",    "/tmp/p5g-omlx.json")]
METRICS = ["ttft_ms", "pp_tps", "tg_tps", "tpot_ms", "e2e_s"]

data = {}
for name, path in TARGETS:
    d = load(path)
    data[name] = {pp: {m: [] for m in METRICS} for pp in PP_LIST}
    for r in d["raw_runs"]:
        pp = r["pp_target"]
        if pp not in PP_LIST: continue
        for m in METRICS:
            v = r.get(m)
            if v is not None: data[name][pp][m].append(v)

def med(vs): return statistics.median(vs) if vs else None
def p95(vs):
    if not vs: return None
    s = sorted(vs); k = int(0.95 * (len(s) - 1)); return s[k]

print("# P5g median + p95\n")
for m in METRICS:
    print(f"## {m} median")
    print("| PP | ironmlx | mlx_lm | omlx |")
    print("|---:|---:|---:|---:|")
    for pp in PP_LIST:
        row = [str(pp)]
        for name, _ in TARGETS:
            v = med(data[name][pp][m])
            row.append(f"{v:.2f}" if v is not None else "n/a")
        print("| " + " | ".join(row) + " |")
    print()
EOF
echo "[aggregate written]"; wc -l /tmp/p5g-aggregate.md
```

### Step 4.6: Write reports/p5g-final-results.md

Create the report (template):

```markdown
# P5g Final — GatedDeltaNet Deep Refactor Close-Out

> **Self-contained for offline code-level analysis.** Embeds all
> bench data, T0 profile + ablation findings, T1-T3 outcomes, and
> P5h scope drivers.

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Branch | ironmlx-p5g-perf |
| HEAD | <fill via git log after this commit lands> |
| Spec | docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md |
| Plan | docs/superpowers/plans/2026-05-20-ironmlx-p5g-gated-delta-net.md |
| Harness | iron-bench (Rust HTTP) |
| Sweep | --prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1, strict serial |

## §1 P5g ship summary

- **T0 profile** (commit <fill T0 commit SHA>): 4-phase HTTP-path
  harness; per § 2-§ 5 of reports/p5g-t0-gated-delta-profile.md.
  Layer 1 GDN occupancy <X>% at PP=2048; top-3 candidates ranked.
- **T1 <PROMOTED/REVERTED>** (commit <fill>): <T1 candidate>; <%>
  geomean improvement OR <%> negative ROI.
- **T2 <PROMOTED/REVERTED>** (commit <fill>): <T2 candidate>; <%>.
- **T3 <PROMOTED/REVERTED>** (commit <fill>): <T3 candidate>; <%>.

## §2 P5g vs P5f baseline measurement (post-T1-T3)

| PP | P5f baseline | P5g final | delta | omlx | omlx+10% target | P5g vs target |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 953 | <fill> | <%> | <fill> | <fill> | <%> |
| 512 | 1577 | <fill> | <%> | <fill> | <fill> | <%> |
| 2048 | 1844 | <fill> | <%> | <fill> | <fill> | <%> |
| 4096 | 1827 | <fill> | <%> | <fill> | <fill> | <%> |
| 8192 | 1723 | <fill> | <%> | <fill> | <fill> | <%> |
| 16384 | 1598 | <fill> | <%> | <fill> | <fill> | <%> |

P5f baseline = `reports/p5f-final-results.md` § 2 ironmlx column. omlx
in this row = P5g T4 re-measurement (today).

## §3 Decode TG sweep (full-PP)

| PP | ironmlx TG | omlx TG | omlx+10% target |
|---:|---:|---:|---:|
| 128 | <fill> | <fill> | <fill> |
| 512 | <fill> | <fill> | <fill> |
| 2048 | <fill> | <fill> | <fill> |
| 4096 | <fill> | <fill> | <fill> |
| 8192 | <fill> | <fill> | <fill> |
| 16384 | <fill> | <fill> | <fill> |

PP=16384 must keep +10.3% over omlx per P5f close-out advantage.
P5g status: <fill — preserved / regressed>.

## §4 P5g overall success bar

Per § 7.3:
  - At least 1 T1/T2/T3 promoted: <fill — yes/no>
  - sweep_full 19/19 PASS in <X>s: <fill>
  - clippy --all-features --workspace -D warnings: 0 warnings
  - fmt --check: clean
  - sentinel + batched + http_smoke: ALL PASS

## §5 P5h scope drivers (P5g close-out per § 7.4)

Residual gap to omlx+10% target per PP (after P5g):

| PP | P5g shipped | omlx+10% target | residual gap | likely attribution |
|---:|---:|---:|---:|---|
| 128 | <fill> | <fill> | <%> | <HTTP/scheduler residue / other> |
| 512 | <fill> | <fill> | <%> | <fill> |
| 2048 | <fill> | <fill> | <%> | **GatedAttention** (full attn, 10/40 layers, super-linear long PP) / chunk-size tuning / Scheduler admit |
| 4096 | <fill> | <fill> | <%> | same as 2048; chunk count starts |
| 8192 | <fill> | <fill> | <%> | GatedAttention O(S²) dominant |
| 16384 | <fill> | <fill> | <%> | GatedAttention long-context + chunk count |

## §6 P5h candidate ranking (post-P5g)

1. **GatedAttention optimization** (full attn, T0 profile 6.5%
   at PP=2048; PP=16384 likely 30%+; SDPA dispatch tuning, KV layout)
2. **Long-prompt chunk-size sweep** (`prefill_chunk_size=512..4096`
   exploration at PP=4096-16384)
3. **Router bypass conditional** (if Scheduler admission > 50ms
   measured by P5g instrumentation)
4. **Multi-request batching (P5h/P6+ deferred)** per Boss 2026-05-19
   directive; --b-max N > 1 functional, awaits multi-user scenario
5. **Metal kernel rewrite for GatedDeltaNet** — only if P5g op-level
   was insufficient (P5g scope gate verdict: <triggered/not triggered>)

## §7 Out of scope (still deferred)

- GatedAttention (P5h)
- Long-prompt chunk-size sweep (P5h)
- Router bypass (P5h conditional)
- Multi-request batching default change (P5h/P6+ per Boss directive)
- omlx PagedCache style port (out per [feedback_design_philosophy])
- mlx::compile wrap (still blocked by 4 safe-wrapper API gaps from P5e T2)
```

Fill all `<fill>` placeholders with actual numbers from §5 aggregate (`/tmp/p5g-aggregate.md`) + T1-T3 commit logs.

- [ ] Create + fill the file. Verify no `<fill>` remaining:
```bash
set -o pipefail
grep "<fill" "$REPO/reports/p5g-final-results.md"
```
Expected: empty.

### Step 4.7: Commit T4 close-out

```bash
set -o pipefail
cd "$REPO"
git add reports/p5g-final-results.md
git commit -m "$(cat <<'EOF'
chore(p5g-t4): P5g close-out — GatedDeltaNet refactor + P5h scope

P5g (GatedDeltaNet deep refactor) complete. Branch ironmlx-p5g-perf
ready for merge consideration.

T0: 4-phase HTTP-path profile harness; Layer 1 GDN occupancy <X>%
    at PP=2048. Top-3 candidates ranked by Layer 2 + Layer 3.

T1: <PROMOTED/REVERTED> — <candidate>; <%> geomean prefill change.
T2: <PROMOTED/REVERTED> — <candidate>; <%>.
T3: <PROMOTED/REVERTED> — <candidate>; <%>.

P5g vs P5f baseline (M5 Max, 6 PP iron-bench median 5 runs):
  PP=128:  953  -> <fill> tok/s (<%>)
  PP=512:  1577 -> <fill> tok/s (<%>)
  PP=2048: 1844 -> <fill> tok/s (<%>)
  PP=4096: 1827 -> <fill> tok/s (<%>)
  PP=8192: 1723 -> <fill> tok/s (<%>)
  PP=16384: 1598 -> <fill> tok/s (<%>)

PP=16384 decode TG preserved at +<%>x over omlx (P5f shipped advantage).

Validation:
  - 4 MoE integration tests (smoke + sentinel argmax=11 + batched +
    http_smoke + p5g harness profile-disabled) PASS
  - sweep_full 19/19 PASS in <X>s
  - 4-way bench (ironmlx / mlx-lm / omlx) per
    reports/p5g-final-results.md
  - clippy + fmt + release build: clean

P5h scope quantified in reports/p5g-final-results.md § 5-§ 6:
  1. GatedAttention optimization (primary; full attn O(S²))
  2. Long-prompt chunk-size sweep (PP=4096-16384)
  3. Router bypass (conditional on Scheduler admit overhead)
  4. Multi-request batching (deferred per Boss directive)
  5. Metal kernel rewrite (only if P5g op-level insufficient;
     scope gate verdict: <triggered/not triggered>)

Per memory[no-spec-from-competitors]: omlx remains observation
only; P5h design path independent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -5
```

Fill `<X>` / `<%>` / `<fill>` with actual measured numbers.

### Step 4.8: Verify branch state

```bash
set -o pipefail
git -C "$REPO" log --oneline d864e6e..HEAD
git -C "$REPO" status --short
```
Expected: 5-10 P5g commits (T0 instrument + harness + Layer 2 + ablation + report = 3-5 commits, T1 + T2 + T3 = 1-2 commits each promote/revert, T4 close-out); clean working tree.

---

## P5g Final Acceptance

All of the following must be true at HEAD:

- [ ] `MLX_DIR=$HOME/.local/mlx cargo build --release`: PASS
- [ ] `cargo +nightly fmt --all -- --check`: clean
- [ ] `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings`: 0 Rust warnings
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored`: PASS, argmax=11
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored`: PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored`: PASS
- [ ] `QWEN35_MODEL=<snap> ./scripts/sweep/sweep_full.sh`: 19/19 PASS
- [ ] `reports/p5g-t0-gated-delta-profile.md`, `reports/p5g-final-results.md` written with real numbers (no `<fill>` placeholders)
- [ ] Spec § 7.2 final target locked (no "TBD by T0.a" or `<fill>` left)
- [ ] At least 1 of T1/T2/T3 promoted per § 7.3 ship metrics (OR P5g closes with documented "no promote — Layer 3 upper bounds insufficient; P5h scope refresh")
- [ ] Multi-request batching capability preserved: `--b-max N > 1` boots functional server with explicit flag (unchanged from P5f)
- [ ] Profile feature truly gated: default `cargo build --release` and `cargo test` (no `--features p5g-profile`) produce zero `[p5g-profile]` log lines

After acceptance, branch is ready for Boss decision on merge timing + P5h spec writing.
