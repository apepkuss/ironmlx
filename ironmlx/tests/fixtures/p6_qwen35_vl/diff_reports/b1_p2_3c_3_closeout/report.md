# B1-p2.3c-3 Continuous batching — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3c-2 head `d27aced`)
**Date:** 2026-05-15
**Spec:** `docs/superpowers/specs/2026-05-14-b1-p2-3c-3-continuous-batching-design.md` (commit `dc19170`)
**Plan:** `docs/superpowers/plans/2026-05-14-b1-p2-3c-3-continuous-batching.md` (commit `aefda2c`)

## Summary

`SchedulerActor::driver_loop` is now a rolling decode loop: biased
`tokio::select!` between `cmd_rx.recv()` and an always-ready step branch
per iteration. Mid-batch admits route through new
`Scheduler::admit_mid` which runs prefill in a standalone B=1 temp
cache (GenerationStream-equivalent path) and adopts the prefilled
row into the main b_max cache via per-layer `KVCache::adopt_row_from` /
`GatedDeltaCache::adopt_row_from`. Finished rows are reclaimed by
`gc_finished_rows` called after every step.

`Phase` enum (Idle / Admitting / Decoding / Finished) is unchanged;
the transitions are relaxed: admit/evict legal in Decoding;
Decoding → Finished triggers in 3 places (idempotent): `step`'s
end-of-loop check (preserved per Boss A decision), `gc_finished_rows`
(slot reclaim path), `evict` (single-row last-evict path).

`admit_mid` uses a standalone B=1 temp cache + per-layer adoption,
trading 3-8× faster admit performance (vs. B=b_max sub-batch with
mask) for the temp cache allocation cost (sub-millisecond). Adoption
is sub-microsecond on Apple Silicon unified memory. Stall during
admit_mid is L_new × B=1_prefill_per_token_time; 3c+ chunked prefill
will reduce this further.

## Acceptance

| Test | Result |
| --- | --- |
| `kvcache_adopt_row_from_basic` + `_shape_mismatch_err` + `_out_of_bounds_err` + `_dtype_mismatch_err` (T1) | PASS |
| `gdcache_adopt_row_from_state_and_offset` + `_out_of_bounds_err` + `_shape_mismatch_err` (T2) | PASS |
| `scheduler_admit_during_decoding_ok` + `_evict_during_decoding_*` + `_gc_finished_rows_*` (T3, 5 tests) | PASS |
| `continuous_batching_mid_decode_admit` (T5 central gate) | PASS — bit-id A=1.0000 / B=1.0000 / C=1.0000 |
| `continuous_batching_full_reject` (T5) | PASS |
| `continuous_batching_drains_to_empty` (T5) | PASS |

Observed `continuous_batching_mid_decode_admit` output:
```
[continuous_batching] tokens_a=[9419, 0, 2500] bit-id=1.0000
[continuous_batching] tokens_b=[9419, 0, 561, 1814, 369, 264, 12401, 321] bit-id=1.0000
[continuous_batching] tokens_c=[14773, 27450, 0, 1049, 557] bit-id=1.0000
```

## Architectural Changes (per spec §4.9 file map)

- `core/cache/kv_cache.rs` (T1): +`dtype()` accessor, +`adopt_row_from` (~70 lines), +4 unit tests
- `core/cache/gated_delta.rs` (T2): +`adopt_row_from` (~70 lines), +3 unit tests; added `slice_strided_on` / `slice_update_on` imports
- `core/generate.rs` (T4): +`slice_logits_row` helper (~25 lines)
- `core/scheduler.rs` (T3+T4): admit/evict Phase guards relaxed (T3); step Phase transition PRESERVED per Boss A (T3); +`gc_finished_rows` (T3); +`admit_mid` (T4) with rollback on inner failure (T4 followup); step_inner + prefill_admitted_inner refactored to use slice_logits_row (T4 + T4 followup)
- `core/server/scheduler_actor.rs` (T4): replaced by rolling decode loop; +`RollingEvent` enum; +`handle_admit_mid` helper; `run_batch_once` removed; end-of-iteration `evict_all` Phase-guarded (T4 followup)
- New integration test `tests/b1_p2_3c_3_continuous_batching.rs` (T5): 3 scenarios, ~320 LOC

No changes to: `nn/*`, `models/*`, `core/server/{openai,anthropic}.rs`, `core/generate.rs::GenerationStream`.

## Compat sunset notes

3c-3 inherits all 5 sunset markers from 3b series + 3c-1:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI VL → GS | B1-p2.4 batched VL |
| OpenAI long-prompt → GS | 3c+ chunked-prefill |
| Anthropic long-prompt → GS | 3c+ chunked-prefill |
| Anthropic image-content → 400 | Future Anthropic VL phase |
| `ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config |

3c-3 closes one pre-existing limitation:
- **Pre-3c-3:** "batch boundary at evict_all" (3a/3b convention). Removed.

3c-3 documents two new limitations:
- **Prefill stall:** Synchronous B=1 prefill in `admit_mid` stalls active rows for `~L_new × B=1_prefill_per_token_time`. Sunset: **3c+** chunked prefill.
- **`b_max`-full reject:** `admit_mid` returns `Err("scheduler full")` when all slots are occupied. Sunset: **3d** admission queue + fair scheduling.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `dc19170` | docs | Design spec |
| `aefda2c` | docs | Implementation plan |
| `c38911b` | feat | T1: KVCache::adopt_row_from + dtype() accessor |
| `c0d93e9` | test | T1 review followup: K/V data verification + complete OOB + dtype mismatch |
| `cbfe8ba` | feat | T2: GatedDeltaCache::adopt_row_from |
| `bfde11f` | test | T2 review followup: complete OOB + shape mismatch coverage |
| `1ba6729` | feat | T3: Scheduler API mid-batch foundation (admit/evict relaxation, gc_finished_rows, Boss A) |
| `90d6e65` | docs+test | T3 review followup: Phase doc + edge-case test coverage |
| `9c1e9e5` | feat | T4: admit_mid + rolling decode loop |
| `584cba4` | fix | T4 review followup: admit_mid rollback + overflow + Phase guard |
| `ceff1c6` | test+docs | T5: continuous batching scenarios + close-out |

## Regression Status

All commands run with `--test-threads=1` against
`QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)`.

| Check | Result | Time |
| --- | --- | --- |
| `cargo +nightly fmt --all -- --check` | clean | - |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean | - |
| `cargo build --release -p ironmlx` | clean | - |
| `cargo test -p ironmlx --lib --release` | 218 passed / 0 failed / 2 ignored | 4.47s |
| P6.3 single-image (`p6_qwen35_vl_logits_match`) | 1 passed / 0 failed | 119.23s |
| P6.6 logits-match | 1 passed / 0 failed | 2.91s |
| P6.7 chunked-prefill matrix | 1 passed / 0 failed | 8.45s |
| B1-p2.1 batched prefill | 1 passed / 0 failed | 10.98s |
| B1-p2.2 batched decode | 1 passed / 0 failed | 634.81s |
| B1-p2.3b-1 scheduler scenarios (3) | 3 passed / 0 failed | 18.53s |
| B1-p2.3b-2 scheduler_actor scenarios (3) | 3 passed / 0 failed | 4.67s |
| B1-p2.3b-3 admission_window scenarios (4) | 4 passed / 0 failed | 271.45s |
| B1-p2.3b-4 anthropic_actor scenarios (3) | 3 passed / 0 failed | 195.34s |
| B1-p2.3c-1 per_row_offset scenarios (5) | 5 passed / 0 failed | 427.42s |
| B1-p2.3c-2 scheduler_decode_mask scenarios (1) | 1 passed / 0 failed | 151.35s |
| B1-p2.3c-3 continuous_batching scenarios (3) | 3 passed / 0 failed | 140.82s |

Exit code: `0`. **No regressions.**

## Notes

- **Continuous batching is live.** `iron-bench v2` (multi-concurrent-request performance comparison) is unblocked. Concurrent requests on b_max=N share a single rolling batch; finished rows yield slots immediately to admission queue head (or rejected if b_max saturated in 3c-3; admission queue lands in 3d).
- **Mid-batch admit correctness verified.** `continuous_batching_mid_decode_admit` (central scenario) shows row C's tokens matching B=1 GenerationStream baseline at bit-id `1.0000` despite C being admitted into row A's vacated slot mid-decode. K/V and SSM state are adopted from the temp cache cleanly.
- **No GatedDeltaNet state corruption.** The standalone B=1 temp cache approach (vs. B=b_max sub-batch + variable mask) avoids touching other active rows' SSM state during admit_mid. Other rows' recurrent_state continues evolving normally through their own `step` invocations.
- **`gc_finished_rows` runs after every step.** Slots are reclaimed within one decode step's latency of a row finishing. Drop of `event_tx` for finished rows means the HTTP handler sees EOF on its event_rx — clean SSE close.
- **Boss A decision rationale:** original plan had `step` delete its Phase→Finished transition (delegating to `gc_finished_rows`). Boss chose to preserve step's transition to keep `b1_p2_3b_1_scheduler_step` integration tests (which call step directly in a `while phase==Decoding` loop) bit-id-unchanged. The two transition sites (step + gc_finished_rows) are idempotent in the rolling decode loop.
- **admit_mid rollback semantics:** if any step after `admit()` fails (e.g., OOM, dtype mismatch, batched_prefill error), the inserted slot is rolled back via `evict(id)` so the next `step()` doesn't panic on an empty `generated_tokens`. Critical for production stability.

## Plan-correction deviations

- **Boss A decision (Task 3):** Original plan §4.5 + Task 3 Step 5 said to remove `step_inner`'s Phase→Finished transition. Boss chose zero-regression path: preserve transition AND add `gc_finished_rows`. Idempotent dual-transition; documented in Phase enum doc.
- **Task 1 followup:** review-driven expansion of OOB test to 3 cases + adding dtype mismatch test + expanding basic test to verify K/V data via probe write.
- **Task 2 followup:** review-driven mirror of Task 1's coverage gap fix (OOB 3-case + shape mismatch tests).
- **Task 3 followup:** Phase enum doc updated to list 3c-3's new transitions; 3 new edge-case tests for evict-not-last + gc partial sweep + gc noop.
- **Task 4 Critical fix:** admit_mid orphan-slot rollback (refactored into outer + admit_mid_inner with evict on Err). Also: cap_for_temp i32 overflow guard, end-of-outer evict_all Phase-guard skip in Idle, prefill_admitted_inner consistency refactor to slice_logits_row.

## B1-p2.3x Next Steps

- **B1-p2.3c+** — Chunked batched prefill: interleave prefill chunks with decode steps in `admit_mid` to bound prefill stall. Also removes long-prompt GS fallback in OpenAI/Anthropic handlers.
- **B1-p2.3d** — Admission queue + preemption: replaces the "Err scheduler full" behavior with a fair admission queue. Exposes `ADMISSION_DEADLINE` + `b_max` via `AppConfig` + CLI flags.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL GS fallback.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-14-b1-p2-3c-3-continuous-batching-design.md`
- Plan: `docs/superpowers/plans/2026-05-14-b1-p2-3c-3-continuous-batching.md`
- Predecessor close-out: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_2_closeout/report.md`
- Scheduler API: `ironmlx/src/core/scheduler.rs`
- driver_loop: `ironmlx/src/core/server/scheduler_actor.rs`
- Cache adopt_row_from primitives: `ironmlx/src/core/cache/{kv_cache,gated_delta}.rs`
- slice_logits_row helper: `ironmlx/src/core/generate.rs`
- New integration test: `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs`
