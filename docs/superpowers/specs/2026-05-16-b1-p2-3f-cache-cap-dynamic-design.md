# B1-p2.3f Cache Cap Dynamic + Bounded — Design

**Status:** Draft (brainstormed 2026-05-16, expanded to scope Z)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Branch target:** `ironmlx-b1-p2-3f-cache-cap` (cut from `ironmlx-b1-p2-3e3-typed-err` head `5d09693` post 3f spec)

## 0. Program context

B1-p2 backlog after B1-p2.4 + 3d + 3e.3:

| Sub-spec | Scope | Status |
| --- | --- | --- |
| B1-p2.3e.3 | typed SchedulerError + p4 fix | ✅ DONE (`8125537`) |
| **B1-p2.3f** | **Cache cap dynamic + bounded** | **This spec** |
| B1-p2.3c+ | Chunked admit_mid prefill | Next (after 3f) |
| B1-p2.3e.1 | Async per-row sampler | Backlog |
| B1-p2.3e.2 | HTTP cancellation propagation | Backlog |

Per Boss decision 2026-05-16 (option Z): 3f scope expanded from "dynamic cap only" to **"dynamic cap + model-context awareness + per-server cap upper bound + admit-time rejection"** — production-grade alternative to "fix it later in B1-p2.5".

## 1. Motivation

Three layered concerns about KV cache cap, all needing fix before 3c+ chunked admit_mid is meaningful:

### 1.1 Hardcoded 8192 cap blocks agent long-prompt

`Scheduler::prefill_admitted_inner` ([scheduler.rs:480](../../../ironmlx/src/core/scheduler.rs#L480)):
```rust
self.cache = Some(model.make_cache(b as i32, 8192, Dtype::Bfloat16)?);
```
Boss-typical agent prompts of 10-20K tokens exceed cap. `batched_prefill` writes K/V beyond cap → fails or truncates silently.

### 1.2 Cache survives evict_all but cap is locked

`evict_all` resets offsets but keeps allocation ([scheduler.rs:377](../../../ironmlx/src/core/scheduler.rs#L377)). After a fix to 1.1, the first batch's cap would still be "locked" for the actor's lifetime — later batches with longer prompts would inherit the smaller cap.

### 1.3 No model-context awareness; no cap upper bound

`ironmlx` does not read `config.json["text_config"]["max_position_embeddings"]` (Qwen3.5: **262144**). The Scheduler accepts arbitrary `prompt_len + max_new_tokens` requests:
- A `max_tokens: 999999` typo → `cap = 1M` → `make_cache(b=4, cap=1M, bf16, layers=36) ≈ 700 GB allocation` → server hang / OOM crash
- Without a CLI cap upper bound (`--max-cache-cap`), production cannot bound memory
- Without model-context awareness, requests beyond the model's training context (262K for Qwen3.5) silently produce garbage / NaN (MRoPE out-of-distribution)

3f addresses all three with a clean three-tier cap model.

## 2. Goals

- **G1.** `Qwen35Config` reads `max_position_embeddings` from `config.json["text_config"]`. Exposed via existing `model.config().max_position_embeddings` accessor.
- **G2.** `ServeArgs` adds `--max-cache-cap` flag (default `32768`). `AppState` propagates to `Scheduler` via `spawn_scheduler_actor`.
- **G3.** `Scheduler` holds `effective_cap_max = min(cli_max_cache_cap, model.config().max_position_embeddings)`. Computed once at actor boot.
- **G4.** `Scheduler::admit` (both first-batch and `admit_mid`) rejects requests where `prompt_len + max_new_tokens > effective_cap_max` with `SchedulerError::RequestTooLarge { needed, max }`.
- **G5.** `prefill_admitted_inner` lazy-allocates with `cap = min(max(prompt_len + max_new_tokens for slot), effective_cap_max)`.
- **G6.** `Scheduler::evict_all` drops cache (`self.cache = None`) instead of resetting offsets.
- **G7.** HTTP handlers map `SchedulerError::RequestTooLarge` → HTTP 413 Payload Too Large + body indicating `needed` and `max`.
- **G8.** Long-prompt acceptance: `PP=10240 max_new=2048` admit + decode-to-completion PASS; `PP=40000 max_new=2048` (exceeds default cap_max=32768) returns HTTP 413.

## 3. Non-goals

- **NG1.** Per-request cap override (e.g., a request specifying its own cap).
- **NG2.** Cache cap shrinking when batch composition reduces — once allocated for a batch, cache persists until evict_all.
- **NG3.** Memory pool / reuse across outer batches.
- **NG4.** Reading `max_position_embeddings` from `rope_scaling`'s post-scaling effective context (some HF models extend via YaRN). 3f uses raw `max_position_embeddings` only.
- **NG5.** Multi-tenant per-user cap limits.

## 4. Architecture

### 4.1 Three-tier cap model

```
cap_needed (per-request)  = prompt_len + max_new_tokens
                  ↓ check 1: SchedulerError::RequestTooLarge → HTTP 413
effective_cap_max          = min(cli_max_cache_cap, model.config.max_position_embeddings)
                  ↓ derived from
cli_max_cache_cap          = ServeArgs --max-cache-cap (default 32768)
model_max_context          = Qwen35Config.max_position_embeddings (Qwen3.5: 262144)
```

Effective cap_max bound for the actor's lifetime — computed at `spawn_scheduler_actor`. Production operator picks `--max-cache-cap` based on available memory; model_max_context is the safety net preventing requests beyond training distribution.

### 4.2 Code changes

#### 4.2.1 `Qwen35Config` adds field ([models/qwen3_5/config.rs:49](../../../ironmlx/src/models/qwen3_5/config.rs#L49))

```rust
pub struct Qwen35Config {
    // ... existing fields ...
    /// Maximum sequence length the model was trained / supports
    /// (`config.json["text_config"]["max_position_embeddings"]`).
    /// Qwen3.5-4B: 262144. Used as a hard upper bound on per-request
    /// `prompt_len + max_new_tokens` (B1-p2.3f).
    pub max_position_embeddings: i32,
}
```

Serde reads it from `text_config`. Existing `from_loader` works unchanged.

#### 4.2.2 `SchedulerError::RequestTooLarge` variant ([core/scheduler.rs](../../../ironmlx/src/core/scheduler.rs))

```rust
#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("admission queue full: capacity={capacity} reached")]
    QueueFull { capacity: usize },

    /// Request's `prompt_len + max_new_tokens` exceeds the server's
    /// effective cap_max. Maps to HTTP 413 Payload Too Large.
    #[error("request too large: needs cap={needed} but server max_cache_cap={max}")]
    RequestTooLarge { needed: usize, max: usize },
}
```

#### 4.2.3 `Scheduler::new` + admit gates

```rust
pub struct Scheduler {
    // ... existing fields ...
    effective_cap_max: usize,  // min(cli_max_cache_cap, model_max_context)
}

impl Scheduler {
    pub fn new(b_max: usize, effective_cap_max: usize) -> Self {
        // existing init + store effective_cap_max
    }

    pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
        // ... existing Phase checks ...

        // 3f: cap check before slot allocation
        let cap_needed = req.prompt_ids.len() + req.max_new_tokens;
        if cap_needed > self.effective_cap_max {
            return Err(anyhow::Error::new(SchedulerError::RequestTooLarge {
                needed: cap_needed,
                max: self.effective_cap_max,
            }));
        }
        // ... existing admit body ...
    }

    pub fn admit_mid(&mut self, req: GenerateRequest, model: &Qwen35Model)
        -> Result<(RequestId, StepEvent)> {
        // Same cap check at entry (before admit_mid_inner).
        let cap_needed = req.prompt_ids.len() + req.max_new_tokens;
        if cap_needed > self.effective_cap_max {
            return Err(anyhow::Error::new(SchedulerError::RequestTooLarge {
                needed: cap_needed,
                max: self.effective_cap_max,
            }));
        }
        // ... existing body ...
    }
}
```

#### 4.2.4 `evict_all` drops cache ([core/scheduler.rs:364](../../../ironmlx/src/core/scheduler.rs#L364))

```rust
pub fn evict_all(&mut self) -> Result<()> {
    // ... existing Phase check + slot clear ...
    self.cache = None;  // 3f: drop instead of reset offsets
    self.phase = Phase::Idle;
    self.poisoned = false;
    Ok(())
}
```

#### 4.2.5 `prefill_admitted_inner` dynamic cap ([core/scheduler.rs:479](../../../ironmlx/src/core/scheduler.rs#L479))

```rust
if self.cache.is_none() {
    let cap = self
        .slots
        .iter()
        .filter_map(|s| s.as_ref())
        .map(|r| {
            let max_new_i32 = i32::try_from(r.max_new_tokens).unwrap_or(i32::MAX);
            (r.prompt_ids.len() as i32).saturating_add(max_new_i32)
        })
        .max()
        .unwrap_or(256);
    // Bound by effective_cap_max (also enforced at admit gate; this is a
    // defense-in-depth assertion that no slot's cap exceeds the bound).
    let cap = cap.min(self.effective_cap_max as i32);
    self.cache = Some(model.make_cache(b as i32, cap, Dtype::Bfloat16)?);
}
```

#### 4.2.6 `spawn_scheduler_actor` signature extension

```rust
pub fn spawn_scheduler_actor(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,  // 3f new param
) -> SchedulerActorHandle
```

Internally `Scheduler::new(b_max, effective_cap_max)`.

#### 4.2.7 `serve()` + `ServeArgs` + `AppState`

```rust
// cli/serve.rs
pub struct ServeArgs {
    // ... existing ...
    #[arg(long, default_value_t = 32768)]
    pub max_cache_cap: usize,
}

// core/server/mod.rs
pub struct AppState {
    // ... existing ...
    pub effective_cap_max: usize,
}

pub async fn serve(
    // ... existing ...
    max_cache_cap: usize,
) -> Result<()> {
    let model = Arc::new(Mutex::new(model));
    // 3f: compute effective_cap_max = min(cli, model_max_context).
    let model_max_context = {
        let m = model.blocking_lock();
        m.config().max_position_embeddings as usize
    };
    let effective_cap_max = max_cache_cap.min(model_max_context);
    if max_cache_cap > model_max_context {
        tracing::warn!(
            "max_cache_cap ({}) > model_max_context ({}); capping at {}",
            max_cache_cap, model_max_context, model_max_context
        );
    }
    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(), b_max, admission_deadline, admission_queue_max,
        effective_cap_max,
    );
    // ... rest of serve ...
}
```

#### 4.2.8 HTTP 413 mapping ([core/server/openai.rs:43](../../../ironmlx/src/core/server/openai.rs#L43) + anthropic.rs)

```rust
fn admit_err_to_response(err: anyhow::Error) -> Response {
    use crate::core::SchedulerError;
    use axum::http::HeaderValue;
    let msg = format!("{err:#}");
    match err.downcast_ref::<SchedulerError>() {
        Some(SchedulerError::QueueFull { .. }) => {
            let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            resp.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            resp
        }
        Some(SchedulerError::RequestTooLarge { .. }) => {
            (StatusCode::PAYLOAD_TOO_LARGE, msg).into_response()
        }
        None => (StatusCode::BAD_REQUEST, msg).into_response(),
    }
}
```

### 4.3 Acceptance

**1 integration test** `tests/b1_p2_3f_cache_cap.rs::admit_long_prompt_pp10k`:
- Real-model prompt with `prompt_len ≈ 10240` tokens, `max_new_tokens = 2048` (default cap_max=32768 → 12288 < 32768 → fits)
- admit + drain to completion → finish_reason = "length" at exactly max_new_tokens
- `#[ignore]` real-model heavy

**3 unit tests** in `core::scheduler::tests`:
- `evict_all_drops_cache` — admit + force_phase(Decoding) + evict_all → assert `sched.cache.is_none()`
- `admit_rejects_oversize_request` — Scheduler::new with effective_cap_max=1024; admit GenerateRequest with prompt+max_new=2000 → `SchedulerError::RequestTooLarge`
- `dynamic_cap_from_slots_bounded_by_cap_max` — admit slots with various prompt_len+max_new; verify computed cap = min(slots_max, effective_cap_max)

**2 unit tests** in `core::server::openai::tests`:
- `admit_err_413_for_request_too_large` — typed Err `RequestTooLarge { needed: 50000, max: 32768 }` → HTTP 413 + body contains both numbers
- `admit_err_400_falls_through` — random anyhow Err → 400 (verify the pattern match covers None case)

**14-suite regression sweep** with default config (`b_max=4 / deadline=5ms / queue_max=32 / max_cache_cap=32768`).

### 4.4 Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| **R1: max_position_embeddings field missing in some checkpoints** | Low | `#[serde(default = "default_max_pos")]` with sensible fallback (32768). Qwen3.5 family all declare it. |
| **R2: GatedDeltaCache cap behavior** | Medium | Existing `make_cache` plumbs cap to both Full and Linear layers. 3c-1/3c-2 tests cover Linear path. |
| **R3: User confusion when cli max_cache_cap > model_max_context** | Low | Log a `tracing::warn` at startup; document in --help. Effective behavior is "cap to model_max_context" (silent correctness, loud warning). |
| **R4: i32 overflow for very large prompt_len + max_new_tokens** | Low | `saturating_add` + `i32::try_from(usize).unwrap_or(i32::MAX)` already used in admit_mid_inner. Apply same in admit. |
| **R5: Tests asserting old "scheduler full" or specific cap=8192 behavior** | Medium | Grep audit at T5. Update any direct cap=8192 assertions to be cap-agnostic. |
| **R6: ~10ms cache alloc overhead per outer batch** | Low | Negligible vs prefill GPU time. |
| **R7: HTTP 413 client behavior** | Low | Standard; clients respect 413 with body explaining the issue. |
| **R8: model_max_context for VL or other Qwen3 family variants** | Low | All Qwen3.5-VL variants share max_position_embeddings=262144. New variants would surface during 3f integration testing. |

### 4.5 Plan decomposition

4 tasks (~1.5 day total):

1. **T1** (~3h): `Qwen35Config.max_position_embeddings` field + `SchedulerError::RequestTooLarge` variant + `Scheduler::new` signature + admit cap check + evict_all cache drop. 2 unit tests.
2. **T2** (~3h): `prefill_admitted_inner` dynamic cap (bounded by effective_cap_max) + spawn_scheduler_actor signature + ServeArgs/AppState/serve plumbing + 1 unit test for cap computation.
3. **T3** (~2h): HTTP 413 mapping in openai.rs + anthropic.rs + 2 unit tests.
4. **T4** (~3h): 1 long-prompt integration test + 14-suite regression sweep + close-out report.

Total ~1.5d. T1+T2 can collapse to 1 task if subagent-driven feels too granular.

## 5. Migration / compat

- **CLI compatibility preserved**: old invocations without `--max-cache-cap` get default 32768 (sensible production limit). Behavior change: requests > 32768 now reject with 413 instead of allocating excessive memory.
- **No model file format change**: `max_position_embeddings` is already in every Qwen3.5 config.json.
- **`spawn_scheduler_actor` signature breaks** — internal API change; all callers (server/mod.rs + 5 test files) updated in T2 (pattern from 3d).

## 6. Linked artifacts

- [B1-p2.3d close-out](../../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/report.md) — 3d's queue/cap discussion + spec §4.6 atomic counters pattern
- [3e.3 typed SchedulerError](../../../docs/superpowers/specs/2026-05-16-b1-p2-3d-admission-queue-design.md) — TypedErr enum pattern that 3f extends
- [3c-3 perf baseline](../../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_perf_baseline/report.md)
- [B1-p2.4 close-out](../../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_4_closeout/report.md)
