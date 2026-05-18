# B1-p2.5 Production Hardening — Design

**Status:** Draft (brainstormed 2026-05-18, post-3e.2 ship)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Predecessor:** [B1-p2.3e.2 PRNG centralization](2026-05-17-b1-p2-3e-2-prng-key-batching-design.md) (shipped 2026-05-18)
**Branch target:** `ironmlx-b1-p2-5-production-hardening` (cut from 3e.2 push HEAD)

## 0. Program context

| Sub-spec | Status |
| --- | --- |
| 3e.1a / 3e.1b / 3e.2 sampler vectorization series | ✅ DONE |
| **B1-p2.5 production hardening** | **This spec** — series terminal |
| Cross-device tuning (M3+) | Future, post-MoE |
| Qwen3.5 MoE | Next main path (after this) |

## 1. Motivation

Three B1-p2 sub-feature ships (3e.1a / 3e.1b / 3e.2) have surfaced production-grade gaps that block confidence in extended-load / multi-tenant scenarios:

- **No memory budget validation**: `Scheduler::new(b_max=N, --max-cache-cap M)` accepts any combo. Operator can configure `4 × 32768 × Qwen3.5-4B` ≈ 51 GB nominal — silently swap-spirals on a 32 GB machine. 3e.2 sweep_full's 16-second single-step stall is highly correlated with swap pressure during chunked admit_mid's KV cache alloc burst.
- **No health endpoint**: production load balancers / monitoring / oncall can't query server state. `/v1/chat/completions` works but no `/healthz`.
- **GPU/memory cleanup verification is manual**: Boss had to manually run `pgrep + memory_pressure + small lib alloc latency` after each sweep_full to confirm clean state. Should be test-harness automation.
- **Stale `Cell` comments** in `scheduler.rs:224` + `scheduler_actor.rs:107` + `scheduler.rs:1492` reference the pre-3e.2 `Sampler.key: Cell<Option<Array>>` which no longer exists.

After B1-p2.5 ships, the codebase migrates to a new (larger-RAM) machine to attempt Qwen3.5 MoE. Hardening done on the smaller M1 Pro is stress-test-driven and will transfer.

## 2. Goals

- **G1.** Memory budget validation at `Scheduler::new` — reject configurations whose computed KV cache budget exceeds available system RAM minus model + safety margin.
- **G2.** Memory budget admission gate — runtime reject new admissions if active KV cache total would breach a soft budget. Return typed `SchedulerError::MemoryBudgetExceeded` → HTTP 503 (retry-able).
- **G3.** Health endpoint `/healthz` — GET returns JSON with status, scheduler counters, memory metrics, model meta, git SHA.
- **G4.** GPU/Memory hygiene auto-verify — helper `verify_clean_state()` callable from tests + `sweep_full.sh` inter-suite, replacing Boss's manual checks.
- **G5.** Cleanup 3 stale `Cell` references from 3e.2 final review.
- **G6.** Test coverage: 2-3 unit tests per goal + 1-2 integration tests (memory budget rejection in real-model context, `/healthz` HTTP smoke).
- **G7.** No regression in 3e.1a/1b/2 perf gates + sweep_full 17/17 PASS.

## 3. Non-goals

- **NG1.** Sweep_full hygiene infrastructure (cooldown between suites, parallel sharding) — separate dev-infra effort, doesn't block production.
- **NG2.** Observability metrics endpoint (Prometheus / OTLP) — independent sub-feature; requires protocol decision (Prometheus vs OTLP vs JSON-poll). File as B1-p3 candidate.
- **NG3.** Circuit breaker — no trigger data yet; collect metrics first.
- **NG4.** Process-level recovery (auto-restart on panic) — OS / launchctl / systemd / Kubernetes responsibility.
- **NG5.** Cross-device tuning (M3+ tile selection / nax-aware kernels) — defer to post-MoE stage.
- **NG6.** Backwards-compat shims for `--b-max`/`--max-cache-cap`: G1 might reject previously-accepted CLI combos. No grace period. Document migration in changelog.
- **NG7.** Dynamic b_max auto-tuning at runtime — too risky; operator still sets explicit values.

## 4. Design

### 4.1 Memory budget validation (G1 + G2)

#### 4.1.1 Estimation functions (`ironmlx/src/core/memory_budget.rs` new module)

```rust
//! Memory budget estimation for the batched scheduler. Used at
//! `Scheduler::new` startup and on each admission to ensure
//! `b_max × effective_cap_max × per_token_kv_bytes` stays within
//! system RAM minus model footprint and safety margin.

/// Bytes per token for one row of KV cache, derived from model meta.
///
/// `kv_bytes_per_token = num_hidden_layers × hidden_size × 2 (K+V) × dtype_size`
///   - num_hidden_layers and hidden_size from `Qwen35Config` / model meta
///   - dtype_size = 2 (bf16) since `make_cache` uses Dtype::Bfloat16
pub fn kv_bytes_per_token(meta: &ModelMeta) -> usize {
    (meta.num_hidden_layers as usize)
        * (meta.hidden_size as usize)
        * 2  // K + V
        * 2  // bf16
}

/// Total bytes for KV cache at given (b, cap).
pub fn kv_cache_bytes(b: usize, cap: usize, meta: &ModelMeta) -> usize {
    b * cap * kv_bytes_per_token(meta)
}

/// Estimate of model weight bytes. For Qwen3.5-4B-MLX-4bit this is
/// dominated by quantized weights (~3 GB).
pub fn model_weight_bytes(meta: &ModelMeta) -> usize {
    meta.weight_bytes
}

/// System total RAM. On macOS via `sysctl hw.memsize`. On Linux via
/// `/proc/meminfo`. Wraps the platform-specific call.
pub fn system_total_ram_bytes() -> usize {
    // platform-specific via sysctl / procfs
    // (impl detail in module)
}

/// Safety margin to reserve for OS + activations + scratch buffers.
pub const SAFETY_MARGIN_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GB

/// Compute the budget the scheduler can claim for KV cache.
pub fn available_budget_bytes(meta: &ModelMeta) -> usize {
    system_total_ram_bytes()
        .saturating_sub(model_weight_bytes(meta))
        .saturating_sub(SAFETY_MARGIN_BYTES)
}

/// Validate startup configuration. Returns `Err` if `b_max × cap × per_token`
/// exceeds available budget; otherwise returns the computed budget for
/// later use in admission gate.
pub fn validate_startup_budget(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
) -> Result<BudgetState, MemoryBudgetError>;
```

`ModelMeta` is a small struct (`num_hidden_layers`, `hidden_size`, `weight_bytes`, etc.) — extract from existing `Qwen35Model::config()` at `Scheduler::new` call site, pass into `Scheduler`.

#### 4.1.2 `Scheduler` integration

`Scheduler::new` signature:

```rust
pub fn new(
    b_max: usize,
    effective_cap_max: usize,
    meta: ModelMeta,
) -> Result<Self, MemoryBudgetError>
```

(Note: changes return type to `Result`; existing callers must `?` propagate. `meta` is new param threaded from `serve()` / `spawn_scheduler_actor()`.)

Inside `Scheduler::new`:

```rust
let budget = memory_budget::validate_startup_budget(b_max, effective_cap_max, &meta)?;
// ... existing construction ...
self.budget_state = budget;  // store for admission gate
```

#### 4.1.3 Admission gate

In `admit_inner` (after row_idx allocation, before slot insertion):

```rust
let prompt_kv = memory_budget::kv_cache_bytes(
    1,
    request.prompt_ids.len() + request.max_new_tokens,
    &self.meta,
);
let active_kv = self.budget_state.active_kv_bytes();
if active_kv + prompt_kv > self.budget_state.soft_limit() {
    return Err(SchedulerError::MemoryBudgetExceeded {
        active_bytes: active_kv,
        requested_bytes: prompt_kv,
        soft_limit_bytes: self.budget_state.soft_limit(),
    }.into());
}
```

Soft limit = `total_budget × 0.85` (reserves 15% for admit_mid temp_cache + activations + headroom).

#### 4.1.4 HTTP error mapping

New variant in `SchedulerError`:

```rust
#[derive(thiserror::Error, Debug)]
pub enum SchedulerError {
    // ... existing variants (QueueFull, RequestTooLarge) ...
    #[error("memory budget exceeded: active {active_bytes} + requested {requested_bytes} > soft limit {soft_limit_bytes}")]
    MemoryBudgetExceeded {
        active_bytes: usize,
        requested_bytes: usize,
        soft_limit_bytes: usize,
    },
}
```

`admit_err_to_response` (openai.rs + anthropic.rs) downcast:
- `MemoryBudgetExceeded` → **HTTP 503** Service Unavailable + retry-after header (e.g., 5s) + JSON body with bytes detail.

### 4.2 Health endpoint `/healthz` (G3)

#### 4.2.1 Route

New HTTP handler `GET /healthz` in `ironmlx/src/core/server/mod.rs`:

```rust
.route("/healthz", get(healthz_handler))
```

Handler returns:

```rust
async fn healthz_handler(State(state): State<Arc<AppState>>) -> Json<HealthSnapshot> {
    Json(state.scheduler_health.snapshot())
}
```

#### 4.2.2 HealthSnapshot struct

```rust
#[derive(Serialize)]
pub struct HealthSnapshot {
    pub status: HealthStatus,            // "healthy" | "degraded" | "down"
    pub uptime_secs: u64,
    pub model: ModelInfo,
    pub scheduler: SchedulerInfo,
    pub memory: MemoryInfo,
    pub git_sha: &'static str,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub max_position_embeddings: i32,
}

#[derive(Serialize)]
pub struct SchedulerInfo {
    pub b_max: usize,
    pub b_active: usize,        // currently-decoding rows
    pub b_queued: usize,        // in admission queue
    pub queue_max: usize,
    pub admission_queue_full_count: u64,    // monotonic counter
    pub memory_budget_exceeded_count: u64,  // monotonic counter
}

#[derive(Serialize)]
pub struct MemoryInfo {
    pub total_ram_bytes: usize,
    pub free_ram_bytes: usize,           // queried fresh each /healthz call
    pub kv_cache_active_bytes: usize,    // from Scheduler.budget_state
    pub kv_cache_soft_limit_bytes: usize,
}

pub enum HealthStatus { Healthy, Degraded, Down }
```

Status rules:

- `Down`: server not yet finished startup OR shutting down (rarely reached — process would have exited)
- `Degraded`: any of:
  - `b_queued >= queue_max × 0.5`
  - `free_ram_bytes < 1 GB`
  - `kv_cache_active_bytes >= kv_cache_soft_limit_bytes × 0.9` (within 10% of soft cap)
- `Healthy`: otherwise

`admission_queue_full_count` and `memory_budget_exceeded_count` are **monotonic counters since process start** (not rolling); they're informational metrics for monitoring trend, not direct status inputs. Monitoring tools can derive rate-of-change externally if needed (avoids implementing in-server windowing logic).

#### 4.2.3 Data flow

- `Scheduler` exposes `health_snapshot() -> SchedulerInfo` (snapshot under internal RwLock)
- `AppState` holds `Arc<SchedulerHealthCollector>` updated by `driver_loop` at end of each step
- Counters: atomic u64 incremented on each `MemoryBudgetExceeded` / `QueueFull` reject
- `/healthz` handler reads collector snapshot + queries fresh `free_ram_bytes` via sysctl

#### 4.2.4 No new dependencies

Uses existing axum router + serde + thiserror. No prometheus/OTLP libs.

### 4.3 GPU/Memory hygiene auto-verify (G4)

#### 4.3.1 Helper `verify_clean_state()` in test-only module

`ironmlx/tests/common/clean_state.rs` (new):

```rust
//! Post-sweep / inter-suite invariant checks for GPU + memory hygiene.
//! Replaces Boss's manual `pgrep + memory_pressure + small alloc probe`
//! procedure with a single callable.

#[derive(Debug)]
pub struct CleanStateReport {
    pub ironmlx_processes_alive: usize,
    pub zombies: usize,
    pub free_ram_bytes: usize,
    pub small_alloc_probe_us: u128,
}

pub fn verify_clean_state(label: &str) -> Result<CleanStateReport, String> {
    // 1. pgrep equivalent for cargo / target/release/deps test binaries.
    //    Use `std::process::Command::new("pgrep")` — macOS / Linux portable.
    // 2. Zombie scan via `ps -o stat` filter.
    // 3. Free RAM via sysctl (memory_budget::system_free_ram_bytes()).
    // 4. Small allocation latency probe: time an mlx::ops::indexing::slice
    //    round-trip on a [4, 2] u32 array — should complete < 10ms on
    //    healthy Metal kernel cache.
    // Return Err with detailed diagnostics if any check fails thresholds.
}

pub const MIN_FREE_RAM_BYTES: usize = 1 * 1024 * 1024 * 1024;        // 1 GB
pub const MAX_ALLOC_PROBE_US: u128 = 10_000;                          // 10 ms
```

#### 4.3.2 Wiring into sweep_full.sh

Modify `scripts/sweep/sweep_full.sh` to invoke between suites:

```bash
for s in "${SUITES[@]}"; do
    # ... existing run ...
    cargo test --release --lib -p ironmlx -- verify_clean_state --nocapture 2>&1 | tail -3 | tee -a "$REPORT"
done
```

Or wire as a single post-sweep call at script end.

#### 4.3.3 Wiring into integration tests (optional)

Larger tests can `verify_clean_state("post-test")` at end of `#[tokio::test]` to catch leaks per-test.

### 4.4 Trivial cleanups (G5)

| Location | Old | New |
|---|---|---|
| `ironmlx/src/core/scheduler.rs:224` (stale Cell rationale) | "the `Cell` inside `Sampler` requires per-row independence — see `core/sampler.rs:43`" | "each row's `Sampler` is config-only (3e.2); per-row PRNG state lives in `Scheduler.prng_state`" |
| `ironmlx/src/core/server/scheduler_actor.rs:107` ("Scheduler is !Send (sampler holds a Cell<Array>)") | "Scheduler is !Send (sampler holds a Cell<Array>)" | "Scheduler is !Send (holds Array fields: KVCache, prng_state)" |
| `ironmlx/src/core/scheduler.rs:1492` ("Clone the sampler so we release the borrow") | "Clone the sampler so we release the borrow" | "Copy the sampler (Sampler: Copy post-3e.2) to release the borrow" |

## 5. Risks

| R# | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | `system_total_ram_bytes` platform call fails (e.g., sandbox restrictions) | Low | Fallback to env var override `IRONMLX_TOTAL_RAM_BYTES` for CI / test |
| **R2** | `SAFETY_MARGIN_BYTES=2 GB` too conservative for 64 GB+ machines (wastes budget) | Low | Future tweak per-arch; 2 GB acceptable now |
| **R3** | `kv_bytes_per_token` formula wrong for non-Qwen3.5 models | Medium | Document Qwen3.5 assumption in code; extend `ModelMeta` for variants later |
| **R4** | `Scheduler::new` returning `Result` cascades into many test fixtures' constructors | Medium | One-shot grep-and-replace; test fixtures use `.expect("scheduler startup")` |
| **R5** | `/healthz` snapshot rwlock contention on high-QPS server | Low | Use `ArcSwap` or atomic counters; sample at end of each driver step (no per-request lock) |
| **R6** | `verify_clean_state` thresholds machine-dependent (1 GB free / 10 ms probe) | Medium | Tunable via env var `IRONMLX_CLEAN_FREE_RAM_BYTES`; document defaults |
| **R7** | Admission gate may reject legitimately small requests near boundary (false positives) | Low | Soft limit 0.85 leaves headroom; tunable via `--memory-soft-frac` CLI arg later |
| **R8** | macOS Activity Monitor reports memory differently from `sysctl hw.memsize` / `vm_stat`. Operator confusion. | Low | Document the metric source in `/healthz` JSON + README |

## 6. Acceptance criteria

### 6.1 Unit tests

- `memory_budget`: `kv_bytes_per_token`, `kv_cache_bytes`, `validate_startup_budget` (under/at/over budget cases) — 3-4 tests
- `memory_budget::system_total_ram_bytes` — sanity (returns > 0)
- `SchedulerError::MemoryBudgetExceeded` variant Display format
- `HealthSnapshot::status` thresholds — 3 tests (healthy / degraded / down conditions)
- `verify_clean_state` helper compiles + returns reasonable report on idle system

### 6.2 Integration tests

- `b1_p2_5_memory_budget_admit_reject`: real-model test — set tiny artificial budget via env var override, attempt large admit, expect `MemoryBudgetExceeded` → HTTP 503
- `b1_p2_5_healthz_endpoint`: extend `p4_http_smoke` — GET `/healthz`, parse JSON, verify schema + reasonable values

### 6.3 Hygiene

- `cargo +nightly fmt --all -- --check`
- `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`
- `cargo +stable build --release`
- All 3 must PASS every commit.

### 6.4 Regression gates

- `cargo test --lib -p ironmlx` — full lib tests PASS (no regression, ~270+)
- Existing perf gates (3e.1a / 3e.1b reused): isolated PASS within 5% of last measured (~64ms / ~82ms median)
- `sweep_smoke.sh` 4 suites PASS
- `sweep_full.sh` 17 suites — target 16/16 or 17/17 single run (env env recovery should help; isolation re-run allowed if env outlier)
- `verify_clean_state` post-sweep PASSES on this final 3e.2 + 5 codebase

### 6.5 No backwards-compat

- `Scheduler::new` returning `Result` cascades cleanly through all callers (no `.unwrap_or_default()` shims)
- `SchedulerError::MemoryBudgetExceeded` is typed, not stringly-matched
- No `--legacy-no-budget-check` flag; if a configuration fails, it fails

## 7. Implementation plan decomposition (preview)

Plan file: `docs/superpowers/plans/2026-05-18-b1-p2-5-production-hardening.md`

Tasks:

- **T0** (~0.5d) — `memory_budget` module + `ModelMeta` struct + 3-4 unit tests
- **T1** (~1d) — `Scheduler::new` signature change to `Result<Self, MemoryBudgetError>` + thread `ModelMeta` through callers (serve + spawn_scheduler_actor + ~15 test fixtures) + admission gate in `admit_inner`
- **T2** (~0.5d) — `SchedulerError::MemoryBudgetExceeded` typed variant + `admit_err_to_response` HTTP 503 mapping (openai.rs + anthropic.rs) + 2 unit tests
- **T3** (~0.5d) — `/healthz` route + `HealthSnapshot` + `SchedulerHealthCollector` + 3 unit tests (status rules)
- **T4** (~0.5d) — `tests/common/clean_state.rs` `verify_clean_state` helper + sweep_full.sh wiring
- **T5** (~0.1d) — 3 stale `Cell` comment cleanups
- **T6** (~0.5d) — 2 integration tests (memory budget reject + healthz endpoint) + perf gate + sweep_smoke + sweep_full + close-out

Total: ~3-3.5 days.

## 8. Open questions (resolved inline above; left here for explicit acknowledgment)

- **SAFETY_MARGIN_BYTES default**: 2 GB — agreed; tunable later if needed.
- **Soft limit fraction**: 0.85 — agreed; tunable via env var if false positives observed.
- **Status thresholds for `/healthz`**: codified in §4.2.2 — straightforward; revisit after first production feedback.
- **`verify_clean_state` thresholds**: documented in §4.3.1 — environment-tunable.

## 9. Carry-forward (post-B1-p2.5)

- **Qwen3.5 MoE** — main path on new larger-RAM machine
- **Observability metrics endpoint** (Prometheus / OTLP) — independent sub-feature, requires protocol decision
- **Sweep_full hygiene** (cooldown / shard for parallel) — dev infra
- **Cross-device tuning** (M3+ tile / nax kernel) — post-MoE
- **Circuit breaker** — depends on metrics data collection first
- **Dynamic b_max auto-tuning** — needs production traffic data + careful design (out of B1-p2.5)

---

**Document history:**

- 2026-05-18 — Initial draft (brainstormed post-3e.2 ship, Boss-approved scope: memory budget + healthz + GPU verify automation + trivial cleanups, ~3-3.5d total)
