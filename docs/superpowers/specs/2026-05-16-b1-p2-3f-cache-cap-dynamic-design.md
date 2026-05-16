# B1-p2.3f Cache Cap Dynamic Allocation — Design

**Status:** Draft (brainstormed 2026-05-16)
**Owner:** ironmlx
**Parent program:** B1-p2 batched serving (see [B1-p2.1 §0](2026-05-12-b1-p2-1-batched-prefill-design.md))
**Branch target:** `ironmlx-b1-p2-3f-cache-cap` (cut from `ironmlx-b1-p2-3e3-typed-err` head `8125537` post 3e.3 land)

## 0. Program context

B1-p2 backlog after B1-p2.4 + 3d:

| Sub-spec | Status |
| --- | --- |
| B1-p2.3e.3 | typed SchedulerError + p4 fix | ✅ DONE (`8125537`) |
| **B1-p2.3f** | **Cache cap dynamic alloc** | **This spec** |
| B1-p2.3c+ | Chunked admit_mid prefill | Next (after 3f) |
| B1-p2.3e.1 | Async per-row sampler | Backlog |
| B1-p2.3e.2 | HTTP cancellation propagation | Backlog |

Per Boss decision 2026-05-16: 3f inserted before 3c+ because `make_cache(b, 8192, dtype)` hardcoded cap blocks agent long-prompt (10-20K tokens) use case from reaching `admit_mid` path at all — 3c+ effort is pointless if first-batch and main-cache admission can't fit the prompt.

## 1. Motivation

The main KV cache is allocated once in `Scheduler::prefill_admitted_inner` ([scheduler.rs:480](../../../ironmlx/src/core/scheduler.rs#L480)):

```rust
if self.cache.is_none() {
    self.cache = Some(model.make_cache(b as i32, 8192, Dtype::Bfloat16)?);
}
```

**Problem 1 — cap=8192 hardcoded**: agent prompts of 10-20K tokens exceed cap. `batched_prefill` writes K/V beyond cap → fails or truncates silently.

**Problem 2 — cache survives evict_all**: `evict_all` resets offsets but keeps allocation ([scheduler.rs:377](../../../ironmlx/src/core/scheduler.rs#L377)). After 3f's dynamic cap fix, the first batch's cap would be "locked" for the actor's lifetime — later batches with longer prompts would still hit the old cap.

3f changes both:
1. `evict_all` drops cache (`self.cache = None`) instead of resetting offsets
2. `prefill_admitted_inner` lazy-allocates with `cap = max(prompt_len + max_new_tokens for slot in slots)`

`admit_mid` path is already dynamic: `cap_for_temp = prompt_len + max_new` ([scheduler.rs:902](../../../ironmlx/src/core/scheduler.rs#L902)). `adopt_row_from`'s `grow_to` already extends the main cache when mid-batch admits arrive with longer prompts than the current cap — no change needed there.

`GenerationStream` already uses `cap = prompt_len + max_new_tokens` ([generate.rs:849](../../../ironmlx/src/core/generate.rs#L849)) — no change needed.

## 2. Goals

- **G1.** Replace hardcoded `8192` in `prefill_admitted_inner` with `max(slot.prompt_ids.len() + slot.max_new_tokens for slot in self.slots if slot.is_some())`, with a safe `min_cap = 256` fallback for None-slot edge cases.
- **G2.** `Scheduler::evict_all` drops cache (`self.cache = None`) instead of `lc.reset()`.
- **G3.** No CLI/AppState changes — fully internal scheduler refactor. (DoS protection / cap_max upper bound deferred to B1-p2.5.)
- **G4.** Long-prompt acceptance: `PP=10240 max_new=2048` admit + decode-to-completion PASS.
- **G5.** No regression in 14-suite test sweep (existing pattern from 3d).

## 3. Non-goals

- **NG1.** CLI `--max-cache-cap` flag — DoS protection deferred to B1-p2.5.
- **NG2.** Per-request cap override.
- **NG3.** Cache cap shrinking when batch composition reduces — once allocated, cache persists for the batch lifetime (dropped at evict_all).
- **NG4.** Memory pool / reuse across outer batches — each evict_all+prefill_admitted is a fresh allocation.
- **NG5.** Test fixture additions for cap stress (PP=32K, PP=128K) — out of typical agent use; future scope.

## 4. Architecture

### 4.1 Code changes

#### 4.1.1 `Scheduler::evict_all` ([scheduler.rs:364](../../../ironmlx/src/core/scheduler.rs#L364))

**Before:**

```rust
pub fn evict_all(&mut self) -> Result<()> {
    match self.phase {
        Phase::Decoding | Phase::Finished => {}
        Phase::Idle | Phase::Admitting => {
            return Err(anyhow!(
                "evict_all illegal in {:?} phase: only Decoding/Finished are valid",
                self.phase
            ));
        }
    }
    for slot in self.slots.iter_mut() {
        *slot = None;
    }
    if let Some(cache) = self.cache.as_mut() {
        for lc in cache.iter_mut() {
            lc.reset()?;
        }
    }
    self.phase = Phase::Idle;
    self.poisoned = false;
    Ok(())
}
```

**After:**

```rust
pub fn evict_all(&mut self) -> Result<()> {
    match self.phase {
        Phase::Decoding | Phase::Finished => {}
        Phase::Idle | Phase::Admitting => {
            return Err(anyhow!(
                "evict_all illegal in {:?} phase: only Decoding/Finished are valid",
                self.phase
            ));
        }
    }
    for slot in self.slots.iter_mut() {
        *slot = None;
    }
    // 3f: drop cache instead of resetting offsets — next prefill_admitted
    // lazy-allocates with cap matching the new batch's prompt+max_new
    // requirements. ~10ms allocation overhead per outer batch is negligible
    // vs the prefill GPU time (100s of ms to seconds).
    self.cache = None;
    self.phase = Phase::Idle;
    self.poisoned = false;
    Ok(())
}
```

#### 4.1.2 `Scheduler::prefill_admitted_inner` cap calculation ([scheduler.rs:479-481](../../../ironmlx/src/core/scheduler.rs#L479))

**Before:**

```rust
if self.cache.is_none() {
    self.cache = Some(model.make_cache(b as i32, 8192, Dtype::Bfloat16)?);
}
```

**After:**

```rust
if self.cache.is_none() {
    // 3f: dynamic cap = max(prompt_len + max_new_tokens) across slots.
    // min_cap = 256 fallback if all slots None (defensive; not reachable in
    // production since prefill_admitted asserts active_count() >= 1).
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
    self.cache = Some(model.make_cache(b as i32, cap, Dtype::Bfloat16)?);
}
```

#### 4.1.3 Doc comment updates ([scheduler.rs:390-393](../../../ironmlx/src/core/scheduler.rs#L390))

Replace:
> Lazy-allocates the batched KV cache on first call (`b_max` rows, capacity 8192, bf16). On subsequent calls (after `evict_all`) the cache is reused — `evict_all` already reset every layer.

With:
> Lazy-allocates the batched KV cache on first call (`b_max` rows; capacity = `max(prompt_len + max_new_tokens)` over admitted slots, bf16). Subsequent calls after `evict_all` allocate fresh — `evict_all` drops the cache so the next batch's cap is sized to its slots, not inherited from the prior batch (B1-p2.3f).

### 4.2 Invariants preserved

- **Single-flight per outer batch**: cache is allocated by `prefill_admitted_inner` and lives until `evict_all`. Mid-batch admit (`admit_mid` path) does NOT reallocate — uses `adopt_row_from`'s existing `grow_to` to extend the main cache if the new row needs more cap.
- **admit_mid temp cache cap**: untouched. Already `prompt_len + max_new`.
- **GenerationStream cache cap**: untouched. Already `prompt_len + max_new`.

### 4.3 Acceptance

Per Boss approval — minimal + regression scope (option β):

**1 integration test** `tests/b1_p2_3f_cache_cap.rs::admit_long_prompt_pp10k`:
- Construct a real-model prompt with `prompt_len ≈ 10240` tokens
- `admit + drain + decode` to completion with `max_new_tokens = 2048`
- Assert: at least 1 decode event emitted (proves cache cap correctly sized);
  finish_reason = "length" at exactly `max_new_tokens`
- `#[ignore]` (real-model heavy, ~60-90s)

**2 unit tests** in `core::scheduler::tests`:
- `evict_all_drops_cache`: admit + force_phase(Decoding) + evict_all → assert `sched.cache.is_none()`
- `compute_dynamic_cap_from_slots`: directly observe cap via the public `prefill_admitted` path with a mock — assert cap = `max(prompt_len + max_new_tokens)` across slots

(Note: 2nd unit test may require minor instrumentation — a `cfg(test)`-only `pub(crate) fn cache_cap(&self) -> Option<i32>` accessor on Scheduler. Acceptable for test-only code.)

**14-suite regression sweep**: same matrix as 3d (P6.* + B1-p2.1/2/3a/3b-1..4/3c-1..3/4 + new b1_p2_3f). Default config preserved (3d's `b_max=4 / deadline=5ms / queue_max=32`).

### 4.4 Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| **R1: GatedDeltaCache `cap` semantics differ from KVCache** | Medium | `make_cache` already plumbs cap to both Full and Linear cache layers. Verify GatedDeltaCache::reset behavior unchanged by 3f. Existing 3c-1/3c-2 tests cover Linear path. |
| **R2: `lc.reset()` removal — were any tests relying on offset-reset-but-cache-kept invariant?** | Low | grep tests for `reset` direct calls: only internal uses in cache module's own unit tests; integration tests rely on observable behavior (admit/step/evict_all) which `cache = None + re-alloc` preserves. |
| **R3: `min_cap = 256` fallback hit in practice** | Low | `prefill_admitted` already errors on `active_count() == 0` ([scheduler.rs:407-408](../../../ironmlx/src/core/scheduler.rs#L407)). The defensive fallback is reachable only via test fixtures bypassing `prefill_admitted`'s precondition check; not a production path. |
| **R4: i32 overflow when prompt_len + max_new is very large** | Low | `saturating_add` returns `i32::MAX` instead of wrapping. `i32::try_from(max_new_tokens)` uses `unwrap_or(i32::MAX)`. Pattern matches admit_mid_inner's existing safeguard. |
| **R5: Memory growth across batches with long-then-longer prompts** | Low | `evict_all` drops cache, so each batch's cap is exactly what it needs. No monotonic growth. |
| **R6: ~10ms alloc overhead per outer batch** | Low | Negligible vs prefill GPU time (100s of ms to seconds). Confirmed by mlx Array alloc benchmarks elsewhere in codebase. |
| **R7: Doc comment claims "capacity 8192" elsewhere** | Low | Sole reference is scheduler.rs:391 — update in §4.1.3 covers it. |

### 4.5 Plan decomposition

3 tasks (~1 day total — small enough to consider single-task but kept as 3 for clean review):

1. **T1**: `evict_all` cache drop + `prefill_admitted_inner` dynamic cap + doc update + 1 unit test (`evict_all_drops_cache`). ~3h.
2. **T2**: 1 integration test (`admit_long_prompt_pp10k`) + cache_cap test accessor + 2nd unit test (`compute_dynamic_cap_from_slots`). ~3h.
3. **T3**: 14-suite regression sweep + close-out report. ~2h.

Total ~1d. Could also collapse to 2 tasks (T1+T2 merged) if subagent-driven feels too granular for the scope.

## 5. Linked artifacts

- [B1-p2.3d close-out](../../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/report.md) — 3d's queue / cap discussion
- [3c-3 perf baseline](../../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_perf_baseline/report.md)
- [B1-p2.4 close-out](../../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_4_closeout/report.md)
