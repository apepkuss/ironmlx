# P5g GatedDeltaNet Deep Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile GatedDeltaNet (`ironmlx/src/nn/gated_delta_net.rs`) per § 3.2 3-layer protocol via HTTP-path harness, then promote ≥ 1 profile-driven op-level optimization via § 7.3 ship metrics (long-prompt geomean prefill +5% AND per-PP regression < 2% AND decode TG regression < 2%) without changing GatedDeltaNet public API or touching dense / MTP / GatedAttention / SparseMoeBlock paths.

**Architecture:** Profile-first. T0 instruments GatedDeltaNet (prefix-parsed layer_idx, mode-gated by `IRONMLX_P5G_PROFILE_MODE`, `OnceLock<ProfileMode>` cached) and runs 4-phase HTTP harness (Phase A whole-prefill baseline + B Layer 1 boundary-isolated + C Layer 2 per-step breakdown + D Layer 3 shape-preserving cost ablation). T0.d locks § 7.2 target. T1-T3 implement op-level candidates by T0.c ranking, each independently ship-or-revert by § 7.3 metrics relative to its starting HEAD. T4 closes-out with 4-way bench + sweep_full + report + P5h scope quantification. Scope gate: if T0/T1 requires Metal kernel rewrite → pause for Boss decision.

**Tech Stack:** Rust 1.94 / cxx-mlx Rust/C++ FFI / Apple Silicon Metal (M5 Max 128 GB) / Qwen3.5-35B-A3B-4bit MoE. iron-bench Rust HTTP harness for ship validation.

**Spec reference:** [docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md](../specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md) (HEAD `d864e6e`, 8-commit iteration through 7 ChatGPT review rounds)

---

## Pre-flight

### Step P-1: Confirm branch + clean state

- [ ] On `ironmlx-p5g-perf`

Run: `git -C /Users/xin/workspace/ironmlx-backend branch --show-current`
Expected: `ironmlx-p5g-perf`

- [ ] Working tree clean

Run: `git -C /Users/xin/workspace/ironmlx-backend status --short`
Expected: empty

### Step P-2: Confirm spec history present

Run: `git -C /Users/xin/workspace/ironmlx-backend log --oneline -10`
Expected: includes `d864e6e docs(p5g): seventh-round review-driven spec polish` and 6 earlier `docs(p5g):` commits all the way to `eacf8b6 docs(p5g): GatedDeltaNet deep refactor design spec`.

### Step P-3: Baseline build verifies

- [ ] Release build green

Run: `MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx`
Expected: `Finished release profile [optimized] target(s)`, 0 Rust warnings (mlx-sys C++ warnings ok).

### Step P-4: Confirm 35B MoE snapshot present

Run:
```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1
```
Expected: outputs `1e20fd8d42056f870933bf98ca6211024744f7ec`.

Capture for the rest of the plan:
```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
```

### Step P-5: Confirm 4B snapshot for sweep_full

Run:
```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1
```
Expected: outputs `32f3e8ecf65426fc3306969496342d504bfa13f3` or similar.

Capture:
```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
```

### Step P-6: Confirm mlx-lm bench venv (for T4 4-way bench)

Run: `ls /Users/xin/workspace/ironmlx-backend/scripts/bench-venvs/mlx-lm/.venv/bin/mlx_lm.server 2>&1`
Expected: prints path.

### Step P-7: Confirm omlx repo (for T4 4-way bench)

Run: `ls /Users/xin/workspace/iron-rivals/omlx/pyproject.toml`
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
grep -A4 "^\[features\]" /Users/xin/workspace/ironmlx-backend/ironmlx/Cargo.toml
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
static PROFILE_MODE: std::sync::OnceLock<ProfileMode> = std::sync::OnceLock::new();

#[cfg(feature = "p5g-profile")]
pub(crate) fn profile_mode() -> ProfileMode {
    *PROFILE_MODE.get_or_init(|| match std::env::var("IRONMLX_P5G_PROFILE_MODE").as_deref() {
        Ok("layer1") => ProfileMode::Layer1,
        Ok("layer2") => ProfileMode::Layer2,
        Ok("ablate-compute-g") => ProfileMode::AblateComputeG,
        Ok("ablate-conv") => ProfileMode::AblateConv,
        Ok("ablate-t-arr") => ProfileMode::AblateTArr,
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
cd /Users/xin/workspace/ironmlx-backend && MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tail -5
```
Expected: `Finished release profile`. If "missing field profile_layer_idx" error → re-check Steps 0.4-0.5.

Also confirm default build (no feature) still green:
```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
```
Expected: Finished.

### Step 0.7: Instrument GatedDeltaNet::forward_on with mode-gated barriers

Read `forward_on` method body (around line 299+). Locate the function signature, then plan the instrumentation: at function entry materialize input + cache states, start timer, run forward as before, before function return materialize output + updated cache states, stop timer, emit log line **after** timer.stop().

Add code right after the function entry (after the existing prelude that establishes `x` and cache references):

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
```

Right before the function returns (locate the final `Ok(out)` or final expression), add the exit barrier + log emit:

```rust
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
        // tracing::info! placed strictly AFTER timer-related work — no log
        // calls inside the measured window.
        let layer = self.profile_layer_idx.unwrap_or(-1);
        let dims = x.shape();
        let dvec = dims.as_slice();
        let batch = if dvec.len() >= 1 { dvec[0] } else { 0 };
        let seq = if dvec.len() >= 2 { dvec[1] } else { 0 };
        let (off_before, off_after) = cache
            .as_deref()
            .map(|c| (c.offset() - seq, c.offset()))
            .unwrap_or((0, 0));
        tracing::info!(
            "[p5g-profile] mode={mode:?} layer={layer} batch={batch} seq={seq} \
             offset_before={off_before} offset_after={off_after} elapsed_us={elapsed_us}"
        );
    }
}
```

Note: variable names (`out`, `cache`, `mask`) must match the actual `forward_on` signature; adjust per the actual function. `cache.offset()` may or may not exist as a public accessor on `GatedDeltaCache` — if not, add a public accessor in `gated_delta.rs` or skip the offset_before/offset_after fields and log `offset_before=0 offset_after=0`.

- [ ] Apply the Edit. Build with feature:
```bash
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tail -5
```
Expected: Finished. Fix any compile errors per actual API surface.

### Step 0.8: Hygiene chain with feature

Run:
```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo build --release 2>&1 | tail -3
```
Expected: fmt clean, clippy 0 Rust warnings, release build PASS.

### Step 0.9: Sentinel + batched + http_smoke (default build, no profile)

This confirms profile feature is truly gated — default build behavior unchanged.

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: smoke 2/2 PASS (argmax=11), batched 1/1 PASS, http_smoke 1/1 PASS.

### Step 0.10: Commit instrumentation infrastructure

```bash
cd /Users/xin/workspace/ironmlx-backend
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

At the exit log emit (Step 0.7's exit block), if `mode == Layer2`, also format `step_breakdown=` from `_p5g_step_elapsed`:

```rust
#[cfg(feature = "p5g-profile")]
if matches!(mode, ProfileMode::Layer2) && !_p5g_step_elapsed.is_empty() {
    let breakdown: Vec<String> = _p5g_step_elapsed.iter().map(|us| us.to_string()).collect();
    tracing::info!("[p5g-profile] mode={mode:?} layer={layer} step_breakdown={}", breakdown.join(","));
}
```

(Or extend the single-line log format to include `step_breakdown=...` conditionally — adapt to actual log line shape from Step 0.7.)

- [ ] Apply Edits. Identify 8 step boundaries in the existing forward_on body and insert the timer captures. Build:
```bash
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
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5g-profile -p ironmlx 2>&1 | tail -3
```
Expected: Finished.

### Step 0.13: Commit Layer 2 + ablation instrumentation

```bash
cd /Users/xin/workspace/ironmlx-backend
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

### Step 0.14: Create T0 profile harness skeleton

Create `ironmlx/tests/p5g_t0_gated_delta_profile.rs`:

```rust
//! P5g T0 — GatedDeltaNet 4-phase HTTP-path profile harness.
//!
//! Phase A: whole-prefill baseline (server NO profile mode, iron-bench sweep)
//! Phase B: Layer 1 boundary-isolated (server mode=layer1, iron-bench sweep, parse log)
//! Phase C: Layer 2 per-step breakdown (server mode=layer2, iron-bench sweep, parse log)
//! Phase D: Layer 3 ablation per top-3 (server mode=ablate-X, iron-bench sweep, record delta vs A)
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --features p5g-profile \
//!       --test p5g_t0_gated_delta_profile \
//!       -- --ignored --test-threads=1 --nocapture

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const PP_LIST: [i32; 4] = [2048, 4096, 8192, 16384];
const WARMUP: usize = 1;
const RUNS: usize = 3;

fn snapshot_dir() -> String {
    std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR env var")
}

fn ironmlx_bin() -> &'static str {
    env!("CARGO_BIN_EXE_ironmlx")
}

fn iron_bench_run(port: u16, model_dir: &str, prompt_len: i32, runs: usize, warmup: usize) -> std::process::Output {
    // Use `cargo run -p iron-bench --release --` since CARGO_BIN_EXE_iron-bench is
    // NOT injected in ironmlx integration tests (cross-package).
    Command::new("cargo")
        .args([
            "run", "-p", "iron-bench", "--release", "--",
            "--target", &format!("p5g_profile=http://127.0.0.1:{port}"),
            "--model", "qwen3.5-moe",
            "--model-dir", model_dir,
            "--prompt-len", &prompt_len.to_string(),
            "--max-tokens", "32",
            "--runs", &runs.to_string(),
            "--warmup", &warmup.to_string(),
            "--format", "json",
        ])
        .output()
        .expect("iron-bench spawn")
}

fn spawn_server(profile_mode: Option<&str>, model_dir: &str, port: u16) -> Child {
    let mut cmd = Command::new(ironmlx_bin());
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
        if let Ok(out) = Command::new("curl").args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &url]).output() {
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

fn parse_profile_log(stderr_bytes: &[u8]) -> Vec<HashMap<String, String>> {
    // Parse lines like:
    //   [p5g-profile] mode=Layer1 layer=12 batch=1 seq=2048 offset_before=4096 offset_after=6144 elapsed_us=15301
    let mut records = Vec::new();
    for line in BufReader::new(stderr_bytes).lines().filter_map(|l| l.ok()) {
        if let Some(rest) = line.split_once("[p5g-profile] ").map(|(_, r)| r) {
            let mut rec: HashMap<String, String> = HashMap::new();
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

#[test]
#[ignore]
fn p5g_t0_gated_delta_profile_4phase() {
    let model_dir = snapshot_dir();
    eprintln!("[p5g-t0] starting 4-phase harness; model={model_dir}");

    // --- Phase A: whole-prefill baseline (no profile mode) ---
    eprintln!("[p5g-t0] Phase A: spawning ironmlx serve (no profile mode)");
    let mut server = spawn_server(None, &model_dir, 18080);
    wait_for_ready(18080, 300);
    let mut phase_a: HashMap<i32, f64> = HashMap::new();
    for &pp in &PP_LIST {
        let out = iron_bench_run(18080, &model_dir, pp, RUNS, WARMUP);
        let stdout = String::from_utf8_lossy(&out.stdout);
        eprintln!("[p5g-t0] Phase A PP={pp} bench output (first 400 chars):\n{}", &stdout.chars().take(400).collect::<String>());
        // Parse iron-bench JSON for median pp_tps. (Real impl: serde_json::from_str.)
        phase_a.insert(pp, 0.0); // placeholder; harness must parse iron-bench JSON
    }
    let _ = server.kill();
    server.wait().ok();
    std::thread::sleep(Duration::from_secs(3));

    // --- Phase B: Layer 1 boundary-isolated ---
    eprintln!("[p5g-t0] Phase B: spawning ironmlx serve with IRONMLX_P5G_PROFILE_MODE=layer1");
    let mut server = spawn_server(Some("layer1"), &model_dir, 18080);
    wait_for_ready(18080, 300);
    let stderr_buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let stderr_handle = server.stderr.take().expect("server stderr");
    let buf_clone = std::sync::Arc::clone(&stderr_buf);
    let stderr_thread = std::thread::spawn(move || {
        let mut rdr = BufReader::new(stderr_handle);
        let mut local = Vec::new();
        let _ = rdr.read_to_end(&mut local);
        buf_clone.lock().unwrap().extend_from_slice(&local);
    });
    for &pp in &PP_LIST {
        let _ = iron_bench_run(18080, &model_dir, pp, RUNS, WARMUP);
    }
    let _ = server.kill();
    server.wait().ok();
    stderr_thread.join().ok();
    let layer1_records = parse_profile_log(&stderr_buf.lock().unwrap());
    eprintln!("[p5g-t0] Phase B captured {} profile records", layer1_records.len());
    std::thread::sleep(Duration::from_secs(3));

    // --- Phase C: Layer 2 per-step breakdown ---
    // (Same pattern as Phase B but with mode=layer2)

    // --- Phase D: Layer 3 ablation (per candidate from C ranking) ---
    // (For each candidate, spawn server with mode=ablate-<X>, run sweep, record wall-time vs Phase A)

    eprintln!("[p5g-t0] harness complete. Phase A baselines + Phase B Layer 1 records collected; \
               Phase C/D require manual extension based on Phase B/C ranking.");
}
```

- [ ] Create the file with the above content. This is a skeleton — Phase C/D are skipped in the auto test; implementer expands them after Phase B layer ranking is known (or fills in iron-bench JSON parsing + serde aggregation).

Note on simplifications: this harness skeleton does NOT fully parse iron-bench JSON (placeholder `phase_a.insert(pp, 0.0)`); implementer must add `serde_json` + median-extraction logic. Also Phase C/D loops are stubbed pending T0.b ranking; implementer extends after running Phase B once.

- [ ] Compile test binary:
```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --features p5g-profile --test p5g_t0_gated_delta_profile --no-run 2>&1 | tail -3
```
Expected: Finished.

### Step 0.15: Run T0 profile (Phase A + B)

Run the harness:

```bash
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
cd /Users/xin/workspace/ironmlx-backend
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
cd /Users/xin/workspace/ironmlx-backend
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
cd /Users/xin/workspace/ironmlx-backend
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
```
Expected: Finished + clean + 0 warnings.

### Step 1.4: Sentinel (argmax=11)

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 2/2 PASS, argmax=11.

If argmax shifts: Investigate logit margin. If logit margin large → accept new sentinel value and document in commit; if small → revert T1 per spec § 4.3.

### Step 1.5: Batched + http_smoke

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: 1/1 + 1/1 PASS.

### Step 1.6: iron-bench short-PP prefill smoke

```bash
# Free ports
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
sleep 3

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
sleep 3
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```

Capture PP=128 + PP=512 prefill medians. Compare to Tn-start baseline (committed HEAD before T1) — should be within ±2% (no regression).

### Step 1.7: iron-bench long-PP prefill sweep

```bash
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
sleep 3
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```

Capture PP=2048/4096/8192/16384 prefill medians. Compute **geometric mean** of these 4 numbers vs T1-start baseline geomean.

### Step 1.8: iron-bench decode TG smoke

```bash
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
sleep 3
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```

Capture PP=128/2048/16384 decode TG (tg_tps) medians. Compare to T1-start baseline:
- PP=128/2048: regression < 2% required
- PP=16384: must keep +10.3% over omlx (P5f shipped advantage); regression < 2% from T1-start required

### Step 1.9: Promote / revert decision

Compile all measurements per § 7.3:

| Metric | T1-start baseline | T1 measured | Delta | Threshold | Status |
|---|---|---|---|---|---|
| Long-PP prefill geomean | <X> | <Y> | <%>  | > +5% | <PASS/FAIL> |
| PP=2048 single | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=4096 single | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=8192 single | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=16384 single | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=128 prefill | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=512 prefill | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=128 decode TG | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=2048 decode TG | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| PP=16384 decode TG | <X> | <Y> | <%> | < 2% regression | <PASS/FAIL> |
| sentinel + batched + http_smoke | PASS | <PASS/FAIL> | | ALL PASS | <PASS/FAIL> |

If ALL **promote** rows PASS:
- T1 promotes. Commit per Step 1.10.

If ANY row FAILS:
- T1 reverts. Revert `gated_delta_net.rs` to pre-T1 state via Edit (NOT `git checkout --` — preserve T0 instrument). Commit revert with negative ROI documentation.

### Step 1.10: Commit T1 (promote) or T1-revert

For promote:
```bash
cd /Users/xin/workspace/ironmlx-backend
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

### Step 2.1: Identify candidate

Read `reports/p5g-t0-gated-delta-profile.md` § 6 ranking. T2 implements the rank-2 candidate. **T2-start HEAD** = current branch HEAD (after T1 promote OR T1 revert). All T2 measurements compare to T2-start HEAD, not P5f baseline.

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
cd /Users/xin/workspace/ironmlx-backend
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

Compile the same metric table as Step 1.9. Threshold: long-PP geomean > +5% vs T2-start, per-PP regression < 2%, decode TG regression < 2%.

### Step 2.6: Commit T2 (promote or revert)

Use the same template as Step 1.10 with `feat(p5g-t2):` or `chore(p5g-t2):` prefix.

---

## Task 3: Third Profile-Driven Optimization (T0.c rank 3)

**Goal:** Same template as Task 1/2, applied to T0 ranking #3 candidate.

**Files:**
- Modify: `ironmlx/src/nn/gated_delta_net.rs` (specific edit by T0.c ranking #3)

### Step 3.1-3.6: Same template as Task 2

Substitute T3 candidate, `feat(p5g-t3):` / `chore(p5g-t3):` prefix, T3-start HEAD baseline for comparison.

---

## Task 4: P5g Close-Out

**Goal:** Run full validation (sweep_full 19/19 + 4-way bench + clippy + fmt + integration tests), write self-contained `reports/p5g-final-results.md`, quantify P5h scope drivers, commit.

**Files:**
- Create: `reports/p5g-final-results.md`

### Step 4.1: Hygiene chain (sanity)

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --release -- -D warnings 2>&1 | tail -5
MLX_DIR=$HOME/.local/mlx cargo build --release 2>&1 | tail -3
```
Expected: all clean.

### Step 4.2: Full integration tests

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1 2>&1 | tail -10
```
Expected: all PASS, argmax=11.

### Step 4.3: sweep_full regression gate

```bash
export QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh 2>&1 | tail -10
```
Expected: 19/19 PASS in ~140-160s. If a single transient flake (e.g. `b1_p2_3c_plus_chunked_admit_mid`), retry once. If second run also fails → STOP, report BLOCKED.

### Step 4.4: 4-way bench (ironmlx / mlx-lm / omlx, strict serial per [feedback_serial_perf_experiments])

This is the same procedure as P5f T3 close-out. 3 separate sweeps, one server up at a time.

**4.4.a — ironmlx sweep**:

```bash
pkill -f "ironmlx serve\|omlx serve\|mlx_lm.server" 2>/dev/null
sleep 3
lsof -i :8080 -i :8081 -i :8082 2>/dev/null | head
# Expected: empty.

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
sleep 3
pkill -f "ironmlx serve.*--port 8080" 2>/dev/null
```

**4.4.b — omlx sweep**:

```bash
SNAP_SHA=$(basename "$IRONMLX_MOE_MODEL_DIR")
cd /Users/xin/workspace/iron-rivals/omlx && \
  uv run omlx serve --model-dir "$IRONMLX_MOE_MODEL_DIR" --host 127.0.0.1 --port 8081 &
OMLX_PID=$!
cd /Users/xin/workspace/ironmlx-backend
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -qE "^(200|404)$"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target omlx=http://127.0.0.1:8081 \
  --model "$SNAP_SHA" --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1 \
  --format json > /tmp/p5g-omlx.json 2> /tmp/p5g-omlx.log

kill $OMLX_PID 2>/dev/null
sleep 3
pkill -f "omlx serve.*--port 8081" 2>/dev/null
```

**4.4.c — mlx-lm sweep**:

```bash
cd /Users/xin/workspace/ironmlx-backend/scripts/bench-venvs/mlx-lm && \
  uv run mlx_lm.server --model "$IRONMLX_MOE_MODEL_DIR" --host 127.0.0.1 --port 8082 --log-level INFO &
MLXLM_PID=$!
cd /Users/xin/workspace/ironmlx-backend
until curl -s -X POST http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}' 2>/dev/null | grep -q "choices"; do sleep 5; done

MLX_DIR=$HOME/.local/mlx cargo run --release -p iron-bench -- \
  --target mlx_lm=http://127.0.0.1:8082 \
  --model default_model --model-dir "$IRONMLX_MOE_MODEL_DIR" \
  --prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1 \
  --format json > /tmp/p5g-mlx_lm.json 2> /tmp/p5g-mlx_lm.log

kill $MLXLM_PID 2>/dev/null
sleep 3
pkill -f "mlx_lm.server.*--port 8082" 2>/dev/null
```

### Step 4.5: Aggregate medians + p95

```bash
cd /Users/xin/workspace/ironmlx-backend && uv run --no-project --with statistics python3 <<'EOF' > /tmp/p5g-aggregate.md
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
grep "<fill" /Users/xin/workspace/ironmlx-backend/reports/p5g-final-results.md
```
Expected: empty.

### Step 4.7: Commit T4 close-out

```bash
cd /Users/xin/workspace/ironmlx-backend
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
git -C /Users/xin/workspace/ironmlx-backend log --oneline d864e6e..HEAD
git -C /Users/xin/workspace/ironmlx-backend status --short
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
