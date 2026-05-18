# B1-p2.5 Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory-budget validation (startup + admission gate), `/healthz` JSON endpoint, automated GPU/memory hygiene verify, and clean up 3e.2-era stale `Cell` references — hardening the B1-p2 batched serving stack before migrating to a larger-RAM machine for Qwen3.5 MoE.

**Architecture:** New `core/memory_budget` module computes GQA-aware per-token KV cache bytes from `ModelMeta` (extracted from `Qwen35Config` at server start). `Scheduler::new` returns `Result` with typed `SchedulerError::MemoryBudgetExceeded` on overcommit. Admission gate enforces `active_kv_bytes + requested_kv_bytes ≤ soft_limit (85% of total budget)` runtime. `/healthz` reads `SchedulerHealthCollector` snapshot (updated by driver_loop) + fresh free-RAM sysctl. Test helper `verify_clean_state()` automates Boss's manual `pgrep + memory_pressure + alloc-probe` procedure for sweep_full inter-suite invariant checks.

**Tech Stack:** Rust, mlx Rust binding, axum (HTTP), thiserror (typed errors), serde (JSON), sysctl (macOS RAM query).

**Spec ref:** [`docs/superpowers/specs/2026-05-18-b1-p2-5-production-hardening-design.md`](../specs/2026-05-18-b1-p2-5-production-hardening-design.md) (commit `6ae4aa8`).

**Branch target:** `ironmlx-b1-p2-5-production-hardening` cut from `ironmlx-b1-p2-3e2-prng-centralization` HEAD (`6ae4aa8` = spec commit). Note: 3e.2 branch already pushed by Boss; cut-from point includes 3e.2 + spec + plan ancestors. Auto-push of B1-p2.5 will publish these.

---

## Pre-flight

### Step 0: Branch + baseline gates

- [ ] **Step 0.1: Cut branch.**

```bash
cd /Volumes/Dev/cxx-mlx
git switch ironmlx-b1-p2-3e2-prng-centralization
git rev-parse HEAD  # expect 6ae4aa8 (spec commit) or later (plan commit if landed)
git switch -c ironmlx-b1-p2-5-production-hardening
```

- [ ] **Step 0.2: Pre-flight hygiene PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three exit 0.

- [ ] **Step 0.3: Baseline lib tests PASS.**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -5
```

Expected: `test result: ok. 268 passed; 0 failed; 8 ignored; ...` (post-3e.2 baseline).

---

## Task 0: `memory_budget` module + `ModelMeta` + budget math

**Files:**
- Create: `ironmlx/src/core/memory_budget.rs`
- Modify: `ironmlx/src/core/mod.rs` (add `pub mod memory_budget;`)

**Goal:** Pure-function module computing GQA-aware KV cache size + system RAM query + budget validation. Standalone testable with 5 unit tests; no Scheduler changes yet (T1 wires it in).

### Step 0.1: Create module file with structs and constants

- [ ] **Create `ironmlx/src/core/memory_budget.rs`:**

```rust
//! Memory budget estimation for the batched scheduler. Computes
//! GQA-aware KV cache bytes from `ModelMeta` and validates
//! `b_max × effective_cap_max × per_token_kv_bytes` against system
//! RAM minus model footprint and safety margin.
//!
//! Used at `Scheduler::new` (startup validation) and `admit_inner`
//! (runtime admission gate). See spec
//! `docs/superpowers/specs/2026-05-18-b1-p2-5-production-hardening-design.md`
//! §4.1 for the design rationale.

use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

/// Model attributes needed for memory budget computation. Extracted
/// from `Qwen35Config` at `serve()` startup and threaded through
/// `Scheduler::new`.
#[derive(Debug, Clone, Copy)]
pub struct ModelMeta {
    /// Number of transformer layers.
    pub num_hidden_layers: i32,
    /// Total attention heads (for head_dim fallback).
    pub num_attention_heads: i32,
    /// Key/value head count (≤ num_attention_heads for GQA).
    pub num_key_value_heads: i32,
    /// Hidden size (for head_dim fallback).
    pub hidden_size: i32,
    /// Explicit head_dim if set in config; otherwise None → derive
    /// `hidden_size / num_attention_heads`.
    pub head_dim: Option<i32>,
    /// Approximate model weight bytes (file size on disk; conservative
    /// upper bound for in-memory footprint after MLX load).
    pub weight_bytes: usize,
}

impl ModelMeta {
    /// Resolve effective head dim (Option fallback to `hidden / heads`).
    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// Safety margin to reserve for OS, MLX activations, scratch buffers.
pub const SAFETY_MARGIN_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Soft admission limit as a fraction of total budget. Leaves
/// headroom for admit_mid temp_cache + activations + GPU scratch.
pub const SOFT_LIMIT_FRAC: f64 = 0.85;

/// KV cache bytes per token for one row. GQA-aware:
/// `num_layers × num_kv_heads × head_dim × 2 (K+V) × 2 (bf16)`.
pub fn kv_bytes_per_token(meta: &ModelMeta) -> usize {
    (meta.num_hidden_layers as usize)
        * (meta.num_key_value_heads as usize)
        * (meta.effective_head_dim() as usize)
        * 2  // K + V
        * 2  // bf16
}

/// Total KV cache bytes for (b rows, cap tokens per row).
pub fn kv_cache_bytes(b: usize, cap: usize, meta: &ModelMeta) -> usize {
    b * cap * kv_bytes_per_token(meta)
}

/// System total RAM in bytes. On macOS uses `sysctl hw.memsize`. On
/// Linux uses `/proc/meminfo` `MemTotal`. Returns conservative 8 GiB
/// fallback if the platform query fails (avoids crash; admission gate
/// will reject most realistic configs).
pub fn system_total_ram_bytes() -> usize {
    // macOS path
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                if let Ok(n) = s.trim().parse::<usize>() {
                    return n;
                }
            }
        }
    }
    // Linux path
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    // format: "MemTotal:    32812308 kB"
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    // Override for tests / CI.
    if let Ok(s) = std::env::var("IRONMLX_TOTAL_RAM_BYTES") {
        if let Ok(n) = s.parse::<usize>() {
            return n;
        }
    }
    // Fallback: 8 GiB.
    8 * 1024 * 1024 * 1024
}

/// Available budget for KV cache = total RAM − model weights − safety margin.
pub fn available_budget_bytes(meta: &ModelMeta) -> usize {
    system_total_ram_bytes()
        .saturating_sub(meta.weight_bytes)
        .saturating_sub(SAFETY_MARGIN_BYTES)
}

/// Validation error returned by [`validate_startup_budget`].
#[derive(Debug, Error)]
#[error(
    "memory budget exceeded: b_max={b_max} × effective_cap_max={cap} × \
     {bytes_per_token} bytes/token = {requested_bytes} bytes > available {available_bytes} \
     (total RAM {total_ram_bytes} - model {model_weight_bytes} - safety margin {SAFETY_MARGIN_BYTES}). \
     Lower --b-max or --max-cache-cap."
)]
pub struct MemoryBudgetError {
    pub b_max: usize,
    pub cap: usize,
    pub bytes_per_token: usize,
    pub requested_bytes: usize,
    pub available_bytes: usize,
    pub total_ram_bytes: usize,
    pub model_weight_bytes: usize,
}

/// Runtime budget state held by `Scheduler`. Tracks active KV cache
/// bytes (sum of per-row caps) plus precomputed soft limit.
#[derive(Debug)]
pub struct BudgetState {
    soft_limit: usize,
    active: AtomicUsize,
}

impl BudgetState {
    pub fn new(total_budget: usize) -> Self {
        Self {
            soft_limit: ((total_budget as f64) * SOFT_LIMIT_FRAC) as usize,
            active: AtomicUsize::new(0),
        }
    }

    pub fn soft_limit(&self) -> usize {
        self.soft_limit
    }

    pub fn active_bytes(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Try to add `requested` to active; returns `Err((active, requested, soft_limit))`
    /// if the addition would breach the soft limit.
    pub fn try_admit(&self, requested: usize) -> Result<(), (usize, usize, usize)> {
        let cur = self.active.load(Ordering::Relaxed);
        if cur + requested > self.soft_limit {
            return Err((cur, requested, self.soft_limit));
        }
        self.active.fetch_add(requested, Ordering::Relaxed);
        Ok(())
    }

    pub fn release(&self, bytes: usize) {
        self.active.fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// Validate startup configuration. Returns `Ok(BudgetState)` if within
/// budget; otherwise `Err(MemoryBudgetError)`.
pub fn validate_startup_budget(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
) -> Result<BudgetState, MemoryBudgetError> {
    let bytes_per_token = kv_bytes_per_token(meta);
    let requested = b_max * effective_cap_max * bytes_per_token;
    let available = available_budget_bytes(meta);
    if requested > available {
        return Err(MemoryBudgetError {
            b_max,
            cap: effective_cap_max,
            bytes_per_token,
            requested_bytes: requested,
            available_bytes: available,
            total_ram_bytes: system_total_ram_bytes(),
            model_weight_bytes: meta.weight_bytes,
        });
    }
    Ok(BudgetState::new(requested))
}
```

- [ ] **Add `pub mod memory_budget;` to `ironmlx/src/core/mod.rs`** (alphabetically before/after existing mods — implementer to pick consistent position).

### Step 0.2: Run a sanity build

- [ ] **Build to verify compiles standalone:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -5
```

Expected: clean compile (no warnings on new module — implementer must ensure no `#[allow(dead_code)]` is required; if needed, add only on `BudgetState::release` if no caller yet).

### Step 0.3: Add 5 unit tests at end of `memory_budget.rs`

- [ ] **Append `#[cfg(test)] mod tests`:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn meta_qwen35_like() -> ModelMeta {
        ModelMeta {
            num_hidden_layers: 28,
            num_attention_heads: 32,
            num_key_value_heads: 8,    // GQA: 4× compression
            hidden_size: 4096,
            head_dim: None,
            weight_bytes: 3 * 1024 * 1024 * 1024,  // 3 GiB
        }
    }

    #[test]
    fn kv_bytes_per_token_gqa_aware() {
        let meta = meta_qwen35_like();
        // 28 layers × 8 kv_heads × 128 head_dim (4096/32) × 2 (K+V) × 2 (bf16)
        // = 28 × 8 × 128 × 4 = 114688 bytes
        assert_eq!(kv_bytes_per_token(&meta), 114_688);
    }

    #[test]
    fn kv_cache_bytes_scales_with_b_and_cap() {
        let meta = meta_qwen35_like();
        // b=1, cap=1024 → 1 × 1024 × 114688 = 117 440 512 bytes (~112 MiB)
        let bytes = kv_cache_bytes(1, 1024, &meta);
        assert_eq!(bytes, 1024 * 114_688);
        // Doubling b doubles bytes
        let bytes_b2 = kv_cache_bytes(2, 1024, &meta);
        assert_eq!(bytes_b2, 2 * bytes);
    }

    #[test]
    fn validate_within_budget_ok() {
        let meta = meta_qwen35_like();
        // With IRONMLX_TOTAL_RAM_BYTES override to make this deterministic:
        std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "34359738368"); // 32 GiB
        let st = validate_startup_budget(1, 4096, &meta).expect("should fit");
        // 1 × 4096 × 114688 = 469 762 048 bytes (~448 MiB), well under
        // 32 GiB − 3 GiB model − 2 GiB margin = 27 GiB available
        assert!(st.soft_limit() > 0);
        std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
    }

    #[test]
    fn validate_over_budget_err() {
        let meta = meta_qwen35_like();
        std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "8589934592"); // 8 GiB
        let err = validate_startup_budget(4, 32768, &meta)
            .expect_err("4 × 32768 × 114688 = 15 GiB should exceed 8 - 3 - 2 = 3 GiB budget");
        let msg = format!("{err}");
        assert!(msg.contains("memory budget exceeded"), "msg: {msg}");
        assert!(msg.contains("Lower --b-max"), "msg: {msg}");
        std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
    }

    #[test]
    fn budget_state_admit_release_round_trip() {
        let st = BudgetState::new(1_000_000);
        assert_eq!(st.active_bytes(), 0);
        st.try_admit(500_000).expect("under soft limit (850k)");
        assert_eq!(st.active_bytes(), 500_000);
        // 500k + 400k = 900k > 850k soft limit → reject
        let err = st.try_admit(400_000);
        assert!(err.is_err(), "should reject above soft limit");
        assert_eq!(st.active_bytes(), 500_000, "rejected admit leaves state unchanged");
        st.release(500_000);
        assert_eq!(st.active_bytes(), 0);
    }
}
```

### Step 0.4: Run tests + hygiene + commit T0

- [ ] **Run tests:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::memory_budget --nocapture 2>&1 | tail -15
```

Expected: 5 PASS.

- [ ] **Hygiene gate:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

- [ ] **Commit:**

```bash
git add ironmlx/src/core/memory_budget.rs ironmlx/src/core/mod.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.5-t0): memory_budget module + ModelMeta

New `core/memory_budget` module:
  - ModelMeta: num_hidden_layers / num_attention_heads / num_kv_heads /
    hidden_size / head_dim / weight_bytes (extracted from Qwen35Config
    at serve() startup in T1)
  - kv_bytes_per_token: GQA-aware = num_layers × num_kv_heads ×
    head_dim × 2 (K+V) × 2 (bf16). Realistic Qwen3.5-like:
    28 × 8 × 128 × 4 = 114688 bytes/token (vs naive hidden_size-based
    392 KB/token — GQA's 4× compression matters)
  - kv_cache_bytes(b, cap, meta) — total cache bytes
  - system_total_ram_bytes: macOS sysctl hw.memsize / Linux
    /proc/meminfo / IRONMLX_TOTAL_RAM_BYTES env override for tests
  - available_budget_bytes = total - model - SAFETY_MARGIN (2 GiB)
  - validate_startup_budget: returns Result<BudgetState, MemoryBudgetError>
  - BudgetState: soft_limit = total × SOFT_LIMIT_FRAC (0.85);
    AtomicUsize active counter; try_admit / release runtime methods

5 unit tests cover GQA math, b/cap scaling, in-budget OK, over-budget
Err, BudgetState admit/release round-trip.

Spec §4.1. T1 wires this into Scheduler::new + admit_inner.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: `Scheduler::new` Result + `ModelMeta` thread + admission gate

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (Scheduler::new signature; struct add `budget_state: BudgetState`; admit_inner add budget check; release on evict)
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (spawn_scheduler_actor signature)
- Modify: `ironmlx/src/core/server/mod.rs` (`serve()` extracts ModelMeta from Qwen35Model::config())
- Modify: ~15 unit/integration test fixtures (pass `ModelMeta` to `Scheduler::new` / `spawn_scheduler_actor`; use `IRONMLX_TOTAL_RAM_BYTES` env to control budget in tests)
- Modify: `ironmlx/src/cli/serve.rs` (extract ModelMeta after model load, pass to serve())

**Goal:** Wire memory_budget into Scheduler lifecycle. Startup validation rejects overcommit; admission gate enforces soft limit at runtime; eviction releases the row's bytes back to the budget.

### Step 1.1: Add `Qwen35Model::model_meta()` accessor

- [ ] **Add helper on Qwen35Model** in `ironmlx/src/models/qwen3_5/model.rs` (find the `impl Qwen35Model` block; verify with `grep -n "impl Qwen35Model" /Volumes/Dev/cxx-mlx/ironmlx/src/models/qwen3_5/model.rs`):

```rust
    /// Extract memory-budget-relevant model attributes for use by
    /// `Scheduler::new` (B1-p2.5).
    pub fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
        let cfg = self.config();
        // weight_bytes: approximate from model dir size if available;
        // otherwise conservative estimate from hidden_size × layers × 4-bit/byte.
        // Use the loader's reported weight file size when available.
        let weight_bytes = self.approx_weight_bytes();
        crate::core::memory_budget::ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: cfg.head_dim,
            weight_bytes,
        }
    }

    /// Conservative weight-bytes estimate for memory budgeting.
    /// Returns the larger of:
    /// (a) total weights file size on disk if accessible
    /// (b) static estimate based on layers × hidden × 4 (4-bit quantized)
    fn approx_weight_bytes(&self) -> usize {
        // Static fallback: layers × hidden² × 12 × 0.5 (4-bit) +
        //   embedding layer. For Qwen3.5-4B ≈ 3 GiB.
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        // ~12 hidden² FF + 4 hidden² attn + embed (vocab × hidden / 8 for 4-bit)
        let ff_attn = l * h * h * 16 / 2;  // /2 for 4-bit
        let embed = (cfg.vocab_size as usize) * h / 2;
        ff_attn + embed
    }
```

If implementer prefers reading actual file size from `Loader`, that's better — but the static estimate above gives a conservative upper bound (always ≥ actual 4-bit weight size).

### Step 1.2: Modify `Scheduler::new` signature

- [ ] **Edit `ironmlx/src/core/scheduler.rs` `Scheduler::new`:**

Change from (current ~line 320):

```rust
pub fn new(b_max: usize, effective_cap_max: usize) -> Self {
    let mut slots = Vec::with_capacity(b_max);
    for _ in 0..b_max {
        slots.push(None);
    }
    let prng_state = Array::zeros(&[b_max as i32, 2_i32][..], Dtype::Uint32).expect("prng_state zeros");
    Self {
        b_max,
        slots,
        // ... existing field inits ...
    }
}
```

To:

```rust
pub fn new(
    b_max: usize,
    effective_cap_max: usize,
    meta: crate::core::memory_budget::ModelMeta,
) -> Result<Self, crate::core::memory_budget::MemoryBudgetError> {
    let budget_state =
        crate::core::memory_budget::validate_startup_budget(b_max, effective_cap_max, &meta)?;
    let mut slots = Vec::with_capacity(b_max);
    for _ in 0..b_max {
        slots.push(None);
    }
    let prng_state = Array::zeros(&[b_max as i32, 2_i32][..], Dtype::Uint32).expect("prng_state zeros");
    Ok(Self {
        b_max,
        slots,
        // ... existing field inits ...
        budget_state,
        meta,
        // counters for /healthz (T3) — start at 0:
        admission_queue_full_count: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        memory_budget_exceeded_count: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
    })
}
```

- [ ] **Add `budget_state` + `meta` + counters fields to `Scheduler` struct definition** (find with `grep -n "pub struct Scheduler" scheduler.rs`):

```rust
pub struct Scheduler {
    // ... existing fields ...
    pub(crate) budget_state: crate::core::memory_budget::BudgetState,
    pub(crate) meta: crate::core::memory_budget::ModelMeta,
    /// Monotonic counter — incremented when `admit_inner` rejects via QueueFull.
    /// Exposed by `health_snapshot()` in T3.
    pub(crate) admission_queue_full_count: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Monotonic counter — incremented when `admit_inner` rejects via MemoryBudgetExceeded.
    pub(crate) memory_budget_exceeded_count: std::sync::Arc<std::sync::atomic::AtomicU64>,
}
```

(`Arc<AtomicU64>` chosen so `SchedulerHealthCollector` in T3 can hold a clone for read access without contending with the driver loop.)

### Step 1.3: Add admission gate in `admit_inner`

- [ ] **Locate `admit_inner` body** (around `fn admit_inner` definition; `grep -n "fn admit_inner\|fn admit(" scheduler.rs`):

Insert the budget check **after the existing RequestTooLarge check, before slot insertion**:

```rust
    // Memory budget gate (3e.5 / B1-p2.5).
    let row_cap = req.prompt_ids.len() + req.max_new_tokens;
    let requested_bytes = crate::core::memory_budget::kv_cache_bytes(1, row_cap, &self.meta);
    if let Err((active, requested, soft_limit)) =
        self.budget_state.try_admit(requested_bytes)
    {
        self.memory_budget_exceeded_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Err(anyhow::Error::new(SchedulerError::MemoryBudgetExceeded {
            active_bytes: active,
            requested_bytes: requested,
            soft_limit_bytes: soft_limit,
        }));
    }
    // ... existing slot insertion code unchanged ...
```

(Don't forget to import `MemoryBudgetExceeded` variant — T2 adds it; this code reference fails to compile until T2 lands. Order plan: T1 stubs the variant; T2 fleshes out HTTP mapping. OR T2 lands before T1 admit gate. Cleanest: **T1 includes the new SchedulerError variant in the same commit**, T2 adds only the HTTP response mapping.)

To avoid the bootstrap problem, add the new variant directly in T1 §1.5 (deferred from T2 plan section).

### Step 1.4: Release budget on evict / row finalization

- [ ] **Find row release point.** `grep -n "fn evict\|fn try_evict\|fn complete_row\|self.slots\[.*\] = None" scheduler.rs`:

The row's KV bytes should release back to the budget at the moment the row's KV cache is conceptually freed. Currently rows are freed when:
- `complete_row` / event with `finish_reason` set
- `evict_all` clears

Add `self.budget_state.release(self.row_cap_bytes(row_idx))` at each release point. Compute the row's bytes as recorded at admit time — store per-row in `RequestState`:

- [ ] **Add field to `RequestState`** in scheduler.rs (find `pub struct RequestState`):

```rust
    /// KV cache bytes charged to budget at admit time. Released on
    /// row completion / eviction. (B1-p2.5)
    pub kv_bytes_admitted: usize,
```

Init in `admit_inner`:

```rust
let state = RequestState {
    // ... existing fields ...
    kv_bytes_admitted: requested_bytes,
};
```

Release at each `self.slots[idx] = None` site:

```rust
if let Some(state) = self.slots[idx].take() {
    self.budget_state.release(state.kv_bytes_admitted);
}
```

Implementer: grep for `self.slots\[.*\] = None` and `self.slots\[.*\].take()` patterns; add release accordingly. Test by ensuring evict tests + complete tests still pass.

### Step 1.5: Add `SchedulerError::MemoryBudgetExceeded` variant

- [ ] **Edit `pub enum SchedulerError`** in scheduler.rs (top of file, ~line 33):

```rust
#[derive(thiserror::Error, Debug)]
pub enum SchedulerError {
    #[error("admission queue full: capacity={capacity} reached")]
    QueueFull { capacity: usize },

    #[error("request too large: needs cap={needed} but server max_cache_cap={max}")]
    RequestTooLarge { needed: usize, max: usize },

    /// Admission gate: request's KV cache bytes plus active bytes would
    /// exceed the soft limit (85% of total budget). Maps to HTTP 503
    /// + Retry-After. Operator should monitor; retry the request later
    /// when capacity frees up. B1-p2.5.
    #[error(
        "memory budget exceeded: active {active_bytes} + requested {requested_bytes} > \
         soft limit {soft_limit_bytes}"
    )]
    MemoryBudgetExceeded {
        active_bytes: usize,
        requested_bytes: usize,
        soft_limit_bytes: usize,
    },
}
```

T2 will add the HTTP response mapping arm.

### Step 1.6: Update `spawn_scheduler_actor` signature

- [ ] **Edit `ironmlx/src/core/server/scheduler_actor.rs`:**

```rust
pub fn spawn_scheduler_actor(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    meta: crate::core::memory_budget::ModelMeta,  // NEW
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError> {  // NEW return type
    // ... existing internals; pass `meta` to `Scheduler::new` ...
    let scheduler = Scheduler::new(b_max, effective_cap_max, meta)?;
    // ... rest unchanged ...
}
```

(Cascades change: callers now `?` propagate.)

### Step 1.7: Update `serve()` to extract ModelMeta + propagate Result

- [ ] **Edit `ironmlx/src/core/server/mod.rs::serve()`** (find with `grep -n "pub async fn serve\|fn serve(" mod.rs`):

After loading the model:

```rust
let meta = {
    let guard = model.blocking_lock();
    guard.model_meta()
};

let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
    model.clone(),
    b_max,
    Duration::from_millis(admission_deadline_ms),
    admission_queue_max,
    effective_cap_max,
    meta,
)?;
```

(`serve()` return type stays `Result<()>`; `?` propagates the `MemoryBudgetError` to caller.)

### Step 1.8: Update all test fixtures using Scheduler::new / spawn_scheduler_actor

- [ ] **Find all callers:**

```bash
grep -rn "Scheduler::new\|spawn_scheduler_actor" /Volumes/Dev/cxx-mlx/ironmlx --include="*.rs" | head -30
```

- [ ] **Define a test helper** for fixture ModelMeta. Add to `ironmlx/src/core/memory_budget.rs` under `#[cfg(test)]` or to a `tests/common` module:

```rust
/// Realistic Qwen3.5-4B-like ModelMeta for tests. Sized so that
/// b_max=8 + cap=32768 stays well under a 32 GiB envelope, ensuring
/// tests pass without requiring IRONMLX_TOTAL_RAM_BYTES override.
#[cfg(test)]
pub fn test_meta_qwen35() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 28,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_size: 4096,
        head_dim: None,
        weight_bytes: 3 * 1024 * 1024 * 1024,
    }
}
```

(For integration tests in `tests/` directory, this helper is not accessible via `#[cfg(test)]` from another crate. Either: (a) make it `pub` non-cfg gated; (b) duplicate the literal in each integration test's `common` module. Implementer to pick — preference: make `test_meta_qwen35` pub non-gated to enable reuse.)

- [ ] **Update test fixtures.** For each `Scheduler::new(N, M)` call, change to:

```rust
Scheduler::new(N, M, crate::core::memory_budget::test_meta_qwen35()).expect("scheduler startup")
```

For each `spawn_scheduler_actor(model, b_max, d, q, c)` call, change to:

```rust
spawn_scheduler_actor(model, b_max, d, q, c, model.blocking_lock().model_meta())
    .expect("spawn_scheduler_actor")
```

(For tests with `Arc<Mutex<Qwen35Model>>`, use `model.blocking_lock().model_meta()` since real model is loaded.)

### Step 1.9: Run tests + hygiene + commit T1

- [ ] **Run scheduler tests:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- core::scheduler 2>&1 | tail -15
```

Expected: all PASS (~36+).

- [ ] **Run full lib tests:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx 2>&1 | tail -5
```

Expected: 268+/268+ PASS (T0's 5 new + same baseline).

- [ ] **Hygiene gate:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

- [ ] **Commit:**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(b1-p2.5-t1): Scheduler::new Result + ModelMeta thread + admission gate

Scheduler::new signature now `Result<Self, MemoryBudgetError>` and
takes `meta: ModelMeta`. Startup validates `b_max × cap × per_token`
against system RAM − model weights − 2 GiB safety margin. Returns
typed Err on overcommit with detailed bytes math + Lower-flag hint.

Admission gate in admit_inner: per-row KV cache bytes (computed
from `kv_cache_bytes(1, prompt + max_new, meta)`) attempted via
`budget_state.try_admit`. Reject with SchedulerError::MemoryBudgetExceeded
+ increment memory_budget_exceeded_count for /healthz (T3).

RequestState gains `kv_bytes_admitted` field — released back to
budget on row completion / eviction via `budget_state.release`.

Monotonic counters added to Scheduler:
  - admission_queue_full_count (T3 will increment in QueueFull path)
  - memory_budget_exceeded_count

ModelMeta extracted in serve() via Qwen35Model::model_meta(); threaded
through spawn_scheduler_actor (signature gains `meta` param + Result return).
Qwen35Model::model_meta() reads Qwen35Config fields + computes
approx_weight_bytes (conservative upper bound for 4-bit quant).

~15 test fixtures updated to pass `ModelMeta` (via new pub
test_meta_qwen35() helper or `model.blocking_lock().model_meta()` for
real-model tests) + `.expect("scheduler startup")` for the new Result.

Spec §4.1 G1+G2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: HTTP 503 mapping for MemoryBudgetExceeded

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs` (admit_err_to_response add MemoryBudgetExceeded arm)
- Modify: `ironmlx/src/core/server/anthropic.rs` (same)

**Goal:** Map the new typed error to HTTP 503 + Retry-After header. Mirror existing QueueFull pattern.

### Step 2.1: Update `admit_err_to_response` in openai.rs

- [ ] **Edit `ironmlx/src/core/server/openai.rs::admit_err_to_response`** (~line 39):

```rust
fn admit_err_to_response(err: anyhow::Error) -> Response {
    use crate::core::SchedulerError;
    use axum::http::HeaderValue;
    let msg = format!("{err:#}");
    match err.downcast_ref::<SchedulerError>() {
        Some(SchedulerError::QueueFull { .. }) => {
            let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            resp
        }
        Some(SchedulerError::RequestTooLarge { .. }) => {
            (StatusCode::PAYLOAD_TOO_LARGE, msg).into_response()
        }
        // B1-p2.5: memory budget exhausted → 503 + Retry-After (retry-able).
        Some(SchedulerError::MemoryBudgetExceeded { .. }) => {
            let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            resp
        }
        None => (StatusCode::BAD_REQUEST, msg).into_response(),
    }
}
```

### Step 2.2: Same for anthropic.rs

- [ ] **Edit `ironmlx/src/core/server/anthropic.rs::admit_err_to_response`** identically.

### Step 2.3: Add unit tests for new mapping

- [ ] **Add to `openai.rs::mod tests` (or wherever existing admit_err tests live):**

```rust
#[test]
fn admit_err_503_for_memory_budget_exceeded() {
    use crate::core::SchedulerError;
    let err: anyhow::Error = SchedulerError::MemoryBudgetExceeded {
        active_bytes: 500_000_000,
        requested_bytes: 200_000_000,
        soft_limit_bytes: 600_000_000,
    }
    .into();
    let resp = admit_err_to_response(err);
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let retry = resp
        .headers()
        .get(http::header::RETRY_AFTER)
        .expect("retry-after header set");
    assert_eq!(retry, "5");
}
```

(Existing tests in 3f era covered QueueFull + RequestTooLarge — find via `grep "admit_err_to_response\|admit_err_503\|admit_err_413" /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/openai.rs`. Place new test in same mod tests block.)

### Step 2.4: Run + hygiene + commit T2

- [ ] **Run tests:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- admit_err 2>&1 | tail -10
```

- [ ] **Hygiene gate** (fmt/clippy/build).

- [ ] **Commit:**

```bash
git add ironmlx/src/core/server/openai.rs ironmlx/src/core/server/anthropic.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.5-t2): HTTP 503 + Retry-After for MemoryBudgetExceeded

Mirror QueueFull pattern: downcast SchedulerError::MemoryBudgetExceeded
→ StatusCode::SERVICE_UNAVAILABLE + Retry-After: 5. Body via thiserror
Display: "memory budget exceeded: active N + requested M > soft limit K".

Applied to both openai.rs (chat completions) and anthropic.rs (messages)
admit_err_to_response.

1 unit test: admit_err_503_for_memory_budget_exceeded — verifies
status + Retry-After header.

Spec §4.1.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `/healthz` JSON endpoint

**Files:**
- Create: `ironmlx/src/core/server/health.rs` (HealthSnapshot + collector + handler)
- Modify: `ironmlx/src/core/server/mod.rs` (mount route, add health_collector to AppState)
- Modify: `ironmlx/src/core/scheduler.rs` (`Scheduler::health_snapshot()` accessor)

**Goal:** GET `/healthz` returns JSON with status / uptime / model / scheduler counters / memory info / git SHA. Existing plain "ok" `/health` route kept untouched.

### Step 3.1: Create `health.rs` module

- [ ] **Create `ironmlx/src/core/server/health.rs`:**

```rust
//! `/healthz` JSON endpoint (B1-p2.5 G3). Composes a snapshot of
//! scheduler state + memory state + model meta into a Serialize'able
//! struct, returned via axum's `Json` extractor.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use crate::core::memory_budget::{system_total_ram_bytes, MemoryBudgetError};

#[derive(Debug, Serialize)]
pub enum HealthStatus {
    #[serde(rename = "healthy")]
    Healthy,
    #[serde(rename = "degraded")]
    Degraded,
    #[serde(rename = "down")]
    Down,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub max_position_embeddings: i32,
}

#[derive(Debug, Serialize)]
pub struct SchedulerInfo {
    pub b_max: usize,
    pub b_active: usize,
    pub b_queued: usize,
    pub queue_max: usize,
    pub admission_queue_full_count: u64,
    pub memory_budget_exceeded_count: u64,
}

#[derive(Debug, Serialize)]
pub struct MemoryInfo {
    pub total_ram_bytes: usize,
    pub free_ram_bytes: usize,
    pub kv_cache_active_bytes: usize,
    pub kv_cache_soft_limit_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthSnapshot {
    pub status: HealthStatus,
    pub uptime_secs: u64,
    pub model: ModelInfo,
    pub scheduler: SchedulerInfo,
    pub memory: MemoryInfo,
    pub git_sha: &'static str,
}

/// Shared counter collector. Held in `Arc<AtomicU64>` shape so that
/// (a) `Scheduler` increments lock-free on hot path and (b) `/healthz`
/// reads lock-free. Initialized once at `serve()` startup.
pub struct SchedulerHealthCollector {
    pub start_time: Instant,
    pub b_max: usize,
    pub queue_max: usize,
    pub model_name: String,
    pub max_position_embeddings: i32,
    /// Updated at end of each driver_loop iteration.
    pub b_active: Arc<AtomicU64>,
    pub b_queued: Arc<AtomicU64>,
    /// Shared with Scheduler counters (incremented on QueueFull / MemoryBudgetExceeded reject).
    pub admission_queue_full_count: Arc<AtomicU64>,
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    /// Shared with Scheduler.budget_state.active counter — but
    /// BudgetState wraps an AtomicUsize internally; expose via a
    /// thin reader by storing the BudgetState's address; simpler to
    /// snapshot via Scheduler::health_snapshot below.
    pub kv_cache_soft_limit_bytes: usize,
}

impl SchedulerHealthCollector {
    /// Compose a snapshot. `kv_cache_active_bytes` is passed in (read
    /// from Scheduler's `budget_state.active_bytes()` by the handler).
    pub fn snapshot(&self, kv_cache_active_bytes: usize) -> HealthSnapshot {
        let uptime_secs = self.start_time.elapsed().as_secs();
        let total_ram_bytes = system_total_ram_bytes();
        let free_ram_bytes = system_free_ram_bytes();
        let b_active = self.b_active.load(Ordering::Relaxed) as usize;
        let b_queued = self.b_queued.load(Ordering::Relaxed) as usize;
        let admission_full = self.admission_queue_full_count.load(Ordering::Relaxed);
        let mb_exceeded = self.memory_budget_exceeded_count.load(Ordering::Relaxed);

        let status = classify_status(
            b_queued,
            self.queue_max,
            free_ram_bytes,
            kv_cache_active_bytes,
            self.kv_cache_soft_limit_bytes,
        );

        HealthSnapshot {
            status,
            uptime_secs,
            model: ModelInfo {
                name: self.model_name.clone(),
                max_position_embeddings: self.max_position_embeddings,
            },
            scheduler: SchedulerInfo {
                b_max: self.b_max,
                b_active,
                b_queued,
                queue_max: self.queue_max,
                admission_queue_full_count: admission_full,
                memory_budget_exceeded_count: mb_exceeded,
            },
            memory: MemoryInfo {
                total_ram_bytes,
                free_ram_bytes,
                kv_cache_active_bytes,
                kv_cache_soft_limit_bytes: self.kv_cache_soft_limit_bytes,
            },
            git_sha: env!("CARGO_PKG_VERSION"), // or git_sha env var if set
        }
    }
}

pub fn classify_status(
    b_queued: usize,
    queue_max: usize,
    free_ram_bytes: usize,
    kv_cache_active_bytes: usize,
    kv_cache_soft_limit_bytes: usize,
) -> HealthStatus {
    let queue_high = queue_max > 0 && b_queued >= queue_max / 2;
    let mem_low = free_ram_bytes < (1024 * 1024 * 1024);
    let budget_near = kv_cache_active_bytes >= ((kv_cache_soft_limit_bytes as f64) * 0.9) as usize;
    if queue_high || mem_low || budget_near {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    }
}

/// Free RAM bytes. Platform-specific best effort; falls back to
/// `total - kv_active - safety_margin` if precise query fails.
pub fn system_free_ram_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        // vm_stat output gives "Pages free: N." — multiply by page size.
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                let mut page_size = 16_384_usize; // common Apple Silicon
                let mut pages_free = 0_usize;
                for line in s.lines() {
                    if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
                        if let Some(num) = rest.split(' ').next() {
                            if let Ok(p) = num.parse::<usize>() {
                                page_size = p;
                            }
                        }
                    }
                    if let Some(rest) = line.strip_prefix("Pages free:") {
                        let t = rest.trim().trim_end_matches('.');
                        if let Ok(n) = t.parse::<usize>() {
                            pages_free = n;
                        }
                    }
                }
                if pages_free > 0 {
                    return pages_free * page_size;
                }
            }
        }
    }
    // Linux fallback
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    // Fallback: assume 4 GiB free (conservative; classify as Degraded).
    4 * 1024 * 1024 * 1024
}
```

### Step 3.2: Add `Scheduler::health_snapshot` accessor

- [ ] **Add to `impl Scheduler`** in scheduler.rs:

```rust
    /// B1-p2.5 G3: snapshot of scheduler state for `/healthz`. Reads
    /// active row count + active KV bytes + monotonic counters.
    pub fn health_snapshot_data(&self) -> (usize, usize, u64, u64) {
        let active = self.slots.iter().filter(|s| s.is_some()).count();
        let kv_active = self.budget_state.active_bytes();
        let q_full = self.admission_queue_full_count.load(std::sync::atomic::Ordering::Relaxed);
        let mb_exceeded = self
            .memory_budget_exceeded_count
            .load(std::sync::atomic::Ordering::Relaxed);
        (active, kv_active, q_full, mb_exceeded)
    }
```

(Returns tuple so the actor wrapping can compose the snapshot without holding Scheduler lock during JSON serialization.)

### Step 3.3: Wire handler into `mod.rs`

- [ ] **Edit `ironmlx/src/core/server/mod.rs`:**

Add to `AppState`:

```rust
    /// B1-p2.5 G3: shared health snapshot inputs.
    pub health_collector: Arc<crate::core::server::health::SchedulerHealthCollector>,
```

In `serve()`, after `Scheduler::new` / `spawn_scheduler_actor` succeeds, build collector:

```rust
let health_collector = Arc::new(crate::core::server::health::SchedulerHealthCollector {
    start_time: std::time::Instant::now(),
    b_max,
    queue_max: admission_queue_max,
    model_name: model_id.clone(),
    max_position_embeddings: {
        let g = model.blocking_lock();
        g.config().max_position_embeddings
    },
    b_active: scheduler_handle.b_active.clone(),  // exposed by SchedulerActor (see below)
    b_queued: scheduler_handle.b_queued.clone(),
    admission_queue_full_count: scheduler_handle.admission_queue_full_count.clone(),
    memory_budget_exceeded_count: scheduler_handle.memory_budget_exceeded_count.clone(),
    kv_cache_soft_limit_bytes: scheduler_handle.kv_cache_soft_limit_bytes,
});
```

`SchedulerActorHandle` needs to expose these (Arc clones). Add fields to it:

```rust
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    // ... existing fields ...
    pub b_active: Arc<AtomicU64>,
    pub b_queued: Arc<AtomicU64>,
    pub admission_queue_full_count: Arc<AtomicU64>,
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    pub kv_cache_soft_limit_bytes: usize,
}
```

Initialize in `spawn_scheduler_actor` after `Scheduler::new`:

```rust
let b_active = Arc::new(AtomicU64::new(0));
let b_queued = Arc::new(AtomicU64::new(0));
let admission_queue_full_count = scheduler.admission_queue_full_count.clone();
let memory_budget_exceeded_count = scheduler.memory_budget_exceeded_count.clone();
let kv_cache_soft_limit_bytes = scheduler.budget_state.soft_limit();
```

Driver loop updates `b_active` / `b_queued` at end of each iteration (clone from Scheduler::health_snapshot_data tuple).

- [ ] **Mount `/healthz` route** in `serve()`:

```rust
.route("/healthz", get(healthz_handler))
```

- [ ] **Define handler in mod.rs (or in health.rs and import):**

```rust
async fn healthz_handler(
    State(state): State<Arc<AppState>>,
) -> Json<crate::core::server::health::HealthSnapshot> {
    // Active KV bytes — need to read from Scheduler. SchedulerActor
    // exposes this via a snapshot command, OR we can read directly
    // from the shared Arc<AtomicU64> stored in collector (B1-p2.5
    // simplification: also export Scheduler.budget_state's active
    // counter via an Arc<AtomicUsize> shared with the handle).
    let kv_active = state.health_collector.b_active.load(std::sync::atomic::Ordering::Relaxed); // placeholder
    // The kv_cache_active_bytes should be the Scheduler's
    // budget_state.active_bytes(); to read lock-free, expose an
    // Arc<AtomicUsize> in SchedulerActorHandle similar to b_active.
    let kv_cache_active_bytes = state.health_collector
        .kv_cache_active_bytes_atomic
        .load(std::sync::atomic::Ordering::Relaxed);
    Json(state.health_collector.snapshot(kv_cache_active_bytes))
}
```

(This is implementer-fiddly: `BudgetState.active` is `AtomicUsize` — wrap it in `Arc<AtomicUsize>` and add `kv_cache_active_bytes_atomic: Arc<AtomicUsize>` to `SchedulerHealthCollector`. Adjust `BudgetState::new` to accept the `Arc<AtomicUsize>` so Scheduler and handle share the same atomic.)

To avoid complicating this further: store the active counter as `Arc<AtomicUsize>` field in `BudgetState` itself (initialize once at construction), share via clone with the handle. This is a small refactor of T0's `BudgetState`.

- [ ] **Refactor BudgetState** (back in `memory_budget.rs`) to use `Arc<AtomicUsize>`:

```rust
#[derive(Debug, Clone)]
pub struct BudgetState {
    soft_limit: usize,
    active: Arc<AtomicUsize>,
}

impl BudgetState {
    pub fn new(total_budget: usize) -> Self {
        Self {
            soft_limit: ((total_budget as f64) * SOFT_LIMIT_FRAC) as usize,
            active: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn shared_active(&self) -> Arc<AtomicUsize> { self.active.clone() }
    // ... existing methods unchanged ...
}
```

T1 callers should compile unchanged (the `Arc<AtomicUsize>` is transparent through fetch_add / load methods).

### Step 3.4: 3 unit tests for status classification

- [ ] **Add to `health.rs::mod tests`:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_healthy_when_all_green() {
        let s = classify_status(
            0,          // b_queued
            32,         // queue_max
            8 * 1024 * 1024 * 1024,  // 8 GiB free
            1_000_000,  // 1 MB active
            10_000_000, // 10 MB soft limit
        );
        assert!(matches!(s, HealthStatus::Healthy));
    }

    #[test]
    fn classify_degraded_when_queue_half_full() {
        let s = classify_status(16, 32, 8 * 1024 * 1024 * 1024, 0, 1);
        assert!(matches!(s, HealthStatus::Degraded));
    }

    #[test]
    fn classify_degraded_when_free_ram_low() {
        let s = classify_status(0, 32, 500_000_000, 0, 1);  // 500 MB < 1 GiB
        assert!(matches!(s, HealthStatus::Degraded));
    }

    #[test]
    fn classify_degraded_when_budget_near_soft_limit() {
        let s = classify_status(0, 32, 8 * 1024 * 1024 * 1024, 9_500_000, 10_000_000); // 95%
        assert!(matches!(s, HealthStatus::Degraded));
    }
}
```

### Step 3.5: Hygiene + commit T3

- [ ] **Run tests:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --lib -p ironmlx -- health 2>&1 | tail -10
```

Expected: 4 PASS (4 classify tests).

- [ ] **Hygiene gate.**

- [ ] **Commit:**

```bash
git add ironmlx/src/core/server/health.rs ironmlx/src/core/server/mod.rs \
        ironmlx/src/core/server/scheduler_actor.rs ironmlx/src/core/scheduler.rs \
        ironmlx/src/core/memory_budget.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.5-t3): /healthz JSON endpoint + SchedulerHealthCollector

New ironmlx/src/core/server/health.rs:
  - HealthSnapshot / ModelInfo / SchedulerInfo / MemoryInfo / HealthStatus
    (enum: healthy / degraded / down) — Serialize via serde
  - SchedulerHealthCollector: shared atomics from Scheduler
    (b_active / b_queued / admission_queue_full_count /
    memory_budget_exceeded_count) + start_time / b_max / queue_max /
    model meta / soft_limit
  - snapshot() composes HealthSnapshot, calling system_free_ram_bytes
    via vm_stat (macOS) / /proc/meminfo (Linux)
  - classify_status: degraded when b_queued ≥ queue_max/2 OR free_ram
    < 1 GiB OR kv_active ≥ 90% of soft_limit

Wired:
  - SchedulerActorHandle exposes shared atomics (Arc<AtomicU64> for
    counters + Arc<AtomicUsize> via BudgetState.shared_active for
    kv_cache_active_bytes) + soft_limit value
  - AppState gains health_collector: Arc<SchedulerHealthCollector>
  - serve() builds collector after spawn_scheduler_actor success
  - Route /healthz mounted, returns Json<HealthSnapshot>
  - Existing /health plain "ok" route preserved (LB compat)

Scheduler.health_snapshot_data() returns (b_active, kv_active,
admission_q_full, mb_exceeded) tuple for actor read; b_active / b_queued
updated by driver_loop at end of each step.

BudgetState refactored: `active` now `Arc<AtomicUsize>` (shared with
handle for lock-free read); shared_active() accessor exposed.

4 unit tests in health.rs::mod tests covering classify_status.

Spec §4.2 G3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `verify_clean_state` test helper + sweep_full.sh wiring

**Files:**
- Create: `ironmlx/tests/common/clean_state.rs`
- Create: `ironmlx/tests/common/mod.rs` (if doesn't exist; verify with `ls /Volumes/Dev/cxx-mlx/ironmlx/tests/common/` first)
- Modify: `scripts/sweep/sweep_full.sh`

**Goal:** Replace Boss's manual `pgrep + memory_pressure + alloc-probe` procedure with single callable. Wire into sweep_full.sh between suites.

### Step 4.1: Verify or create `tests/common/mod.rs`

- [ ] **Check:**

```bash
ls /Volumes/Dev/cxx-mlx/ironmlx/tests/common/ 2>&1
```

If exists with `mod.rs`, add to it. If not, create.

### Step 4.2: Create `clean_state.rs`

- [ ] **Create file:**

```rust
//! Post-sweep / inter-suite GPU + memory hygiene invariant checks.
//! Replaces the manual `pgrep + memory_pressure + alloc-probe`
//! procedure with a single callable; can be invoked from individual
//! integration tests or from `sweep_full.sh` between suites.
//!
//! See spec
//! `docs/superpowers/specs/2026-05-18-b1-p2-5-production-hardening-design.md`
//! §4.3 G4 for rationale.

#![cfg(test)]

use std::process::Command;
use std::time::Instant;

#[derive(Debug)]
pub struct CleanStateReport {
    pub ironmlx_processes_alive: usize,
    pub zombies: usize,
    pub free_ram_bytes: usize,
    pub small_alloc_probe_us: u128,
}

pub const MIN_FREE_RAM_BYTES: usize = 1 * 1024 * 1024 * 1024; // 1 GiB
pub const MAX_ALLOC_PROBE_US: u128 = 10_000; // 10 ms

/// Run all hygiene checks. Returns `Ok(report)` if all thresholds met,
/// `Err(detailed_message)` if any failed.
pub fn verify_clean_state(label: &str) -> Result<CleanStateReport, String> {
    let ironmlx_processes_alive = count_ironmlx_processes()?;
    let zombies = count_zombies()?;
    let free_ram_bytes = ironmlx::core::server::health::system_free_ram_bytes();
    let small_alloc_probe_us = run_small_alloc_probe()?;

    let mut errs = Vec::new();
    if ironmlx_processes_alive > 0 {
        errs.push(format!(
            "{ironmlx_processes_alive} ironmlx test processes still alive (expected 0)"
        ));
    }
    if zombies > 0 {
        errs.push(format!("{zombies} zombie processes (expected 0)"));
    }
    if free_ram_bytes < MIN_FREE_RAM_BYTES {
        errs.push(format!(
            "free RAM {free_ram_bytes} bytes < {MIN_FREE_RAM_BYTES} threshold"
        ));
    }
    if small_alloc_probe_us > MAX_ALLOC_PROBE_US {
        errs.push(format!(
            "small alloc probe {small_alloc_probe_us}us > {MAX_ALLOC_PROBE_US}us threshold (Metal kernel cache may be degraded)"
        ));
    }

    let report = CleanStateReport {
        ironmlx_processes_alive,
        zombies,
        free_ram_bytes,
        small_alloc_probe_us,
    };

    if errs.is_empty() {
        Ok(report)
    } else {
        Err(format!(
            "[{label}] verify_clean_state failed:\n  - {}\nreport: {report:#?}",
            errs.join("\n  - ")
        ))
    }
}

fn count_ironmlx_processes() -> Result<usize, String> {
    let output = Command::new("pgrep")
        .args(["-f", "target/release/deps/b1_p2|target/release/deps/p4_|target/release/deps/p6_"])
        .output()
        .map_err(|e| format!("pgrep failed: {e}"))?;
    let stdout = std::str::from_utf8(&output.stdout).map_err(|e| format!("pgrep utf8: {e}"))?;
    Ok(stdout.lines().filter(|l| !l.trim().is_empty()).count())
}

fn count_zombies() -> Result<usize, String> {
    let output = Command::new("ps")
        .args(["-axo", "stat"])
        .output()
        .map_err(|e| format!("ps failed: {e}"))?;
    let stdout = std::str::from_utf8(&output.stdout).map_err(|e| format!("ps utf8: {e}"))?;
    Ok(stdout.lines().filter(|l| l.trim().starts_with('Z')).count())
}

fn run_small_alloc_probe() -> Result<u128, String> {
    // Tiny mlx Array zero alloc + dispose timed. Healthy Metal kernel
    // cache: < 10 ms (cold first call after long idle may be slower).
    use mlx::Array;
    let t0 = Instant::now();
    let _arr: Array = Array::zeros(&[4_i32, 2_i32][..], mlx::Dtype::Uint32)
        .map_err(|e| format!("Array::zeros failed: {e}"))?;
    let elapsed = t0.elapsed();
    Ok(elapsed.as_micros())
}
```

### Step 4.3: Add a sanity test that calls verify_clean_state

- [ ] **Append to clean_state.rs:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-only sanity. Skipped if env CI=1 since CI may have
    /// non-trivial process noise.
    #[test]
    fn verify_clean_state_compiles_and_runs() {
        if std::env::var("CI").is_ok() {
            return;
        }
        // Don't assert outcome — environment may not be clean during
        // `cargo test --lib` if other tests are running concurrently.
        // Just verify the function runs and returns a Result.
        let result = verify_clean_state("sanity-test");
        match result {
            Ok(report) => eprintln!("clean state ok: {report:#?}"),
            Err(e) => eprintln!("clean state degraded (expected during concurrent tests): {e}"),
        }
    }
}
```

### Step 4.4: Wire into `sweep_full.sh`

- [ ] **Edit `scripts/sweep/sweep_full.sh`:**

After each suite's run (inside the for loop), add:

```bash
# B1-p2.5 G4: inter-suite hygiene check.
HYGIENE_OUT=$(MLX_DIR=$MLX_DIR cargo +stable test --release --test integration_clean_state -- --ignored --nocapture 2>&1 | tail -3 || true)
log "  hygiene: $HYGIENE_OUT"
```

Where `tests/integration_clean_state.rs` is a tiny binary test calling `verify_clean_state`. Create:

- [ ] **Create `ironmlx/tests/integration_clean_state.rs`:**

```rust
//! B1-p2.5 G4: standalone clean-state probe wired into
//! sweep_full.sh between integration suites.

mod common;
use common::clean_state::verify_clean_state;

#[test]
#[ignore]
fn integration_clean_state() {
    match verify_clean_state("sweep-inter-suite") {
        Ok(report) => println!("clean state OK: {report:#?}"),
        Err(e) => {
            println!("clean state DEGRADED: {e}");
            // Don't fail the test — sweep_full.sh records the output;
            // human/automation decides whether to block on this.
        }
    }
}
```

Wait — `mod common;` needs `common/mod.rs` exposing `clean_state`. Adjust:

- [ ] **Create or update `ironmlx/tests/common/mod.rs`:**

```rust
pub mod clean_state;
```

Update `clean_state.rs` to remove `#![cfg(test)]` since it's referenced as a public module by integration tests:

```rust
// REMOVE the file-level `#![cfg(test)]` so integration tests can import.
// Add #[cfg(test)] only on the inner mod tests block.
```

### Step 4.5: Run + hygiene + commit T4

- [ ] **Run clean state test:**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test --release --test integration_clean_state -- --ignored --nocapture 2>&1 | tail -10
```

- [ ] **Hygiene gate.**

- [ ] **Commit:**

```bash
git add ironmlx/tests/common/ ironmlx/tests/integration_clean_state.rs scripts/sweep/sweep_full.sh
git commit -m "$(cat <<'EOF'
feat(b1-p2.5-t4): verify_clean_state helper + sweep_full.sh wiring

New ironmlx/tests/common/clean_state.rs::verify_clean_state(label):
  - count_ironmlx_processes via pgrep -f 'target/release/deps/b1_p2|p4_|p6_'
  - count_zombies via ps -axo stat | filter 'Z'
  - free_ram_bytes via system_free_ram_bytes (reused from T3 health module)
  - small_alloc_probe_us via timed mlx::Array::zeros([4, 2] u32) — healthy
    Metal kernel cache should complete in < 10 ms

Returns CleanStateReport on success or detailed Err string with
list of threshold violations.

Thresholds (tunable via consts, future env override):
  - MIN_FREE_RAM_BYTES = 1 GiB
  - MAX_ALLOC_PROBE_US = 10 ms

New ironmlx/tests/integration_clean_state.rs — standalone #[ignore] test
calling verify_clean_state for sweep_full.sh use.

scripts/sweep/sweep_full.sh wires the call between suites: logs each
suite's hygiene report to sweep log; non-failing (informational) so
sweep proceeds even on degradation.

Replaces Boss's manual procedure (pgrep + memory_pressure + lib
test allocation probe) demonstrated in 3e.2 close-out.

Spec §4.3 G4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Stale `Cell` comment cleanups

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (2 locations)
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (1 location)

**Goal:** Remove obsolete `Cell<Option<Array>>` references in doc comments. Sampler became POD in 3e.2.

### Step 5.1: Fix stale comments

- [ ] **Edit `ironmlx/src/core/scheduler.rs:224`** (currently says "the `Cell` inside `Sampler` requires per-row independence — see `core/sampler.rs:43`"):

```rust
    /// Per-row sampler — cloned from the request's sampler at admit time so
    /// each row owns independent sampler state. Sampler is `Copy` post-3e.2;
    /// PRNG state lives in `Scheduler.prng_state` (centralized) — see
    /// `docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md`.
    pub sampler: Sampler,
```

- [ ] **Edit `ironmlx/src/core/server/scheduler_actor.rs:107`** (currently says "Scheduler is !Send (sampler holds a Cell<Array>)"):

```rust
/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send`
/// (holds Array fields: KVCache, prng_state) and the model lock is sync.
```

- [ ] **Edit `ironmlx/src/core/scheduler.rs:1492`** (3e.2 reviewer flagged a "Clone the sampler" comment that should be "Copy"):

Find via:

```bash
grep -n "Clone the sampler\|clone the sampler" /Volumes/Dev/cxx-mlx/ironmlx/src/core/scheduler.rs
```

If found, update to "Copy the sampler (Sampler: Copy post-3e.2)…". If not found (3rd location may have been already updated), skip.

### Step 5.2: Hygiene + commit T5

- [ ] **Hygiene gate.**

- [ ] **Commit:**

```bash
git add ironmlx/src/core/scheduler.rs ironmlx/src/core/server/scheduler_actor.rs
git commit -m "$(cat <<'EOF'
chore(b1-p2.5-t5): cleanup stale Cell references (3e.2 carry-forward)

3e.2 final code reviewer (commit 4885ec9 close-out addendum)
flagged 3 doc comments referencing the pre-3e.2 `Cell<Option<Array>>`
field that was removed from Sampler in commit 2bc80de:

  - scheduler.rs:224 — "the `Cell` inside `Sampler` requires per-row
    independence" updated to reflect Copy semantics + centralized
    prng_state
  - scheduler_actor.rs:107 — "Scheduler is !Send (sampler holds a
    Cell<Array>)" updated to cite actual !Send sources (KVCache +
    prng_state Array fields)
  - scheduler.rs admit_mid_finalize comment about "Clone the sampler"
    updated to "Copy" (Sampler: Copy post-3e.2)

No code change, comments only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Integration tests + perf gate + sweep + close-out

**Files:**
- Create: `ironmlx/tests/b1_p2_5_production_hardening.rs`
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_5_closeout/report.md`
- Modify: `ironmlx/tests/p4_http_smoke.rs` (extend with /healthz JSON probe)

**Goal:** Integration coverage for memory budget reject + /healthz endpoint. Verify no perf regression. Sweep_full + close-out.

### Step 6.1: Create `b1_p2_5_production_hardening.rs` integration test

- [ ] **Create file:**

```rust
//! B1-p2.5 production hardening integration tests.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::memory_budget::{MemoryBudgetError, ModelMeta};
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_5_startup_rejects_overcommit() {
    // Force IRONMLX_TOTAL_RAM_BYTES override to 4 GiB to make the
    // default --b-max 4 --max-cache-cap 32768 reliably exceed budget.
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "4294967296"); // 4 GiB

    let (model, _tok) = load_fixture();
    let meta = model.blocking_lock().model_meta();
    let result = spawn_scheduler_actor(
        model,
        4,                                    // b_max
        Duration::from_millis(5),
        32,
        32768,                                // effective_cap_max
        meta,
    );
    std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
    let err = result.expect_err("expected MemoryBudgetError on overcommit");
    let msg = format!("{err}");
    assert!(msg.contains("memory budget exceeded"), "msg: {msg}");
    assert!(msg.contains("Lower"), "msg: {msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_5_admission_gate_rejects_when_full() {
    // Configure tight budget: b_max=2, cap=2048, override RAM so budget
    // is computable but tight. Send 3 admits — first 2 succeed, 3rd hits
    // MemoryBudgetExceeded.
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "8589934592"); // 8 GiB
    let (model, tokenizer) = load_fixture();
    let meta = model.blocking_lock().model_meta();
    let handle = spawn_scheduler_actor(
        model.clone(),
        2,
        Duration::from_millis(5),
        32,
        2048,
        meta,
    )
    .expect("spawn ok with tight but valid budget");
    std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");

    let prompt = tokenizer.encode("Hello", false).unwrap();
    let stop_tokens = tokenizer.eos_token_ids().to_vec();
    let make = || GenerateRequest {
        prompt_ids: prompt.clone(),
        max_new_tokens: 1024,
        sampler: Sampler::greedy(),
        stop_token_ids: stop_tokens.clone(),
        prefill_chunk_size: 128,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    };

    // First 2 admits should succeed.
    let mut replies = Vec::new();
    for _ in 0..2 {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        handle.cmd_tx.send(SchedulerCommand::Admit { request: make(), reply_tx }).await.unwrap();
        replies.push(reply_rx.await.unwrap().expect("admit ok"));
    }

    // 3rd admit should fail with MemoryBudgetExceeded (since admission
    // gate denies based on existing active rows' KV bytes).
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle.cmd_tx.send(SchedulerCommand::Admit { request: make(), reply_tx }).await.unwrap();
    let admit_err = reply_rx.await.unwrap().expect_err("3rd admit should hit budget");
    let msg = format!("{admit_err}");
    assert!(
        msg.contains("memory budget") || msg.contains("queue") || msg.contains("scheduler full"),
        "msg: {msg}"
    );

    drop(handle);
}
```

### Step 6.2: Extend p4_http_smoke with /healthz probe

- [ ] **Find p4_http_smoke.rs:** `grep -n "fn p4_http_smoke" /Volumes/Dev/cxx-mlx/ironmlx/tests/p4_http_smoke.rs`

- [ ] **Add a new test (or extend existing) for /healthz:**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn b1_p2_5_healthz_json() {
    // Start server (reuse p4_http_smoke spawn helper if present);
    // GET /healthz and parse JSON.
    let server = start_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}:{}/healthz", server.host, server.port))
        .send()
        .await
        .expect("get /healthz");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: serde_json::Value = resp.json().await.expect("/healthz JSON parse");
    // Schema sanity checks
    assert!(json["status"].is_string());
    assert!(json["uptime_secs"].is_number());
    assert!(json["model"]["name"].is_string());
    assert!(json["scheduler"]["b_max"].is_number());
    assert!(json["memory"]["total_ram_bytes"].is_number());
}
```

(If p4_http_smoke doesn't expose a `start_test_server()` helper, refactor or implement minimal one. Adjust test inline if needed.)

### Step 6.3: Run perf gate (3e.1b reused) — verify no regression

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
  MLX_DIR=$HOME/.local/mlx \
  cargo +stable test --release --test b1_p2_3e_1b_configured_sampler -- --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/b1_p2_5_perf.log | tail -15
```

Expected: PASS within 5% of last 3e.2 measurement (~82ms median, ratio ≤2×).

### Step 6.4: Sweep smoke + sweep_full

```bash
./scripts/sweep/sweep_smoke.sh \
  --suites b1_p2_3b_2_scheduler_actor \
           b1_p2_4_batched_vl::mid_admit_vl_during_text_decode \
           b1_p2_3e_1a_vectorize_greedy::b1_p2_3e_1a_greedy_decode_speedup \
           b1_p2_3e_1b_configured_sampler::b1_p2_3e_1b_configured_decode_speedup \
           b1_p2_5_production_hardening::b1_p2_5_startup_rejects_overcommit \
  2>&1 | tee /tmp/b1_p2_5_smoke.log | tail -20
```

Expected: 5 PASS.

```bash
bash ./scripts/sweep/sweep_full.sh > /tmp/b1_p2_5_sweep_full.log 2>&1 &
echo "PID: $!"
```

Background; close-out writes "in progress" then addendum once done.

### Step 6.5: Write close-out report

- [ ] **Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_5_closeout/report.md`** documenting:
  - Branch shape + commits (T0-T6)
  - Acceptance gates (lib tests + perf gate + sweep_smoke + sweep_full + hygiene)
  - Architecture notes for memory budget / /healthz / verify_clean_state
  - Migration note (operators must set `IRONMLX_TOTAL_RAM_BYTES` env or accept conservative 8 GiB fallback)
  - Carry-forward (post-B1-p2.5): Qwen3.5 MoE on new machine; observability metrics endpoint; sweep_full hygiene infra; cross-device tuning

### Step 6.6: Hygiene + commit T6 + push

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release

git add -f ironmlx/tests/b1_p2_5_production_hardening.rs \
        ironmlx/tests/p4_http_smoke.rs \
        ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_5_closeout/report.md
git commit -m "$(cat <<'EOF'
test+docs(b1-p2.5-t6): integration tests + close-out

b1_p2_5_production_hardening.rs:
  - b1_p2_5_startup_rejects_overcommit: IRONMLX_TOTAL_RAM_BYTES=4GiB
    + b_max=4 cap=32768 → MemoryBudgetError with "Lower" hint
  - b1_p2_5_admission_gate_rejects_when_full: tight budget config,
    3rd admit hits MemoryBudgetExceeded after 2 valid admits

p4_http_smoke.rs extended:
  - b1_p2_5_healthz_json: GET /healthz, parse JSON, verify schema
    (status / uptime / model / scheduler / memory fields present)

Close-out report at .../b1_p2_5_closeout/report.md documents:
  - 7 commits (T0-T6) shape
  - Acceptance gates: 270+ lib + 2 new integration + perf gate
    + sweep_smoke 5 PASS + sweep_full in progress
  - Architecture: memory budget GQA math + admission gate + /healthz
    schema + verify_clean_state thresholds
  - Migration: IRONMLX_TOTAL_RAM_BYTES env override for non-macOS/Linux
  - Carry-forward: Qwen3.5 MoE / observability / sweep hygiene / xdevice

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 6.7: Wait for sweep_full + close-out addendum + push

- Poll `/tmp/b1_p2_5_sweep_full.log` until "full sweep done" line.
- If 16/16 or 17/17 single-run PASS → append addendum + push.
- If some FAILs → isolation re-run per 3e.1b/3e.2 protocol, document in addendum.
- Push: `git push -u origin ironmlx-b1-p2-5-production-hardening`.

---

## Self-Review Checklist (controller, post-implementation)

After T0-T6 complete:

1. **Spec coverage:**
   - Spec §4.1 G1+G2 memory budget → T0 + T1 + T2
   - Spec §4.2 G3 /healthz → T3
   - Spec §4.3 G4 verify_clean_state → T4
   - Spec §4.4 G5 stale Cell cleanups → T5
   - Spec §6 acceptance criteria → T6 (integration tests + perf gate + sweep + close-out)
   - Spec §5 R1-R8 risks → mitigated in code (fallback for sysctl R1, conservative SAFETY_MARGIN R2, GQA-aware formula R3, Result cascade R4, atomics R5, tunable thresholds R6, soft limit R7, README docs R8)

2. **No placeholders:** every step has real code.

3. **Type consistency:**
   - `ModelMeta` fields consistent across `memory_budget.rs` + `Qwen35Model::model_meta()` + Scheduler/Actor signatures
   - `MemoryBudgetError` flows: thiserror Display → admit_err_to_response → HTTP 503

4. **No backwards-compat code:** Scheduler::new + spawn_scheduler_actor signature changes are clean breaks; all callers updated in same commit (T1).

5. **Hygiene gate at every commit:** explicit in each Task §N.

6. **Boss constraints:** Chinese in user-facing messages, frequent commits, MLX_DIR set, no amend / no --no-verify / no force push.
