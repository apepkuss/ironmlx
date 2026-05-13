# B1-p2.3a Scheduler Skeleton — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off `ironmlx-b1-p2-2-batched-decode` head `1ed51dc`)
**Date:** 2026-05-13
**Spec:** `docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md` (commit `3db4b7e`)
**Plan:** `docs/superpowers/plans/2026-05-13-b1-p2-3a-scheduler-skeleton.md`

## Summary

Added the scheduler data foundation for B1-p2.3 continuous batching. New `core::scheduler` module with `Scheduler`, `RequestState`, and `RequestId` types plus `admit`/`evict`/`active`/`get`/`get_mut`/`occupied_rows` API, covered by 10 unit tests + 1 integration test. Zero touches to model/server/cache/generate — purely additive.

Subsequent sub-phases extend this skeleton: 3b adds `Scheduler::step()` + HTTP refactor; 3c per-row KV cache offsets; 3d admission queue / preemption; 3e per-row sampler invocation.

## Acceptance

| Test | Result |
| --- | --- |
| `scheduler::tests::scheduler_new_empty` | ✅ |
| `scheduler::tests::admit_happy_path` | ✅ |
| `scheduler::tests::admit_assigns_distinct_rows` | ✅ |
| `scheduler::tests::evict_releases_row` | ✅ |
| `scheduler::tests::admit_after_evict_reuses_row` | ✅ |
| `scheduler::tests::admit_full_returns_err` | ✅ |
| `scheduler::tests::evict_unknown_id_returns_err` | ✅ |
| `scheduler::tests::id_monotonic_after_evict` | ✅ |
| `scheduler::tests::sampler_cloned_per_request` | ✅ |
| `scheduler::tests::occupied_rows_reflects_state` | ✅ |
| `b1_p2_3a_admit_evict_sequence` (integration) | ✅ |

## Architectural Changes

1. **New module `ironmlx/src/core/scheduler.rs`** (~314 lines including tests)
   - `RequestId(u64)` — opaque, monotonic, never reused.
   - `RequestState` — per-request state owned by the scheduler. Fields: id, row_idx, prompt_ids, generated_tokens, max_new_tokens, stop_token_ids, sampler (moved at admit time), real_len, finished, finish_reason. VL fields deferred to B1-p2.4.
   - `Scheduler` — fixed-capacity `Vec<Option<RequestState>>` of length `b_max`. `admit` linear-scans for first `None`; `evict` linear-scans by id. ID counter advances on admit only.
2. **`ironmlx/src/core/mod.rs`** — added `pub mod scheduler;` and re-exports of `RequestId`, `RequestState`, `Scheduler`.

No changes to: `models/`, `core/server/`, `core/generate.rs`, `core/cache/`, `core/sampler.rs`, `core/tokenizer.rs`, `nn/`.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `ae5db40` | feat | Scheduler skeleton + 10 unit tests |
| `8a41ce3` | test | Integration test for admit/evict sequence |
| `<TASK3_SHA>` | docs | This close-out |

(Fill in `<TASK3_SHA>` after Step 3.8 commit.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **174 passed / 0 failed** |
| P6.3 single-image | PASS, max_diff=0.3906, first_token=760 |
| P6.6 logits-match | PASS, max_diff=0.9004, first_token=760 |
| P6.7 chunked-prefill matrix | PASS, all chunk_sizes (0, 256, 64) → first_token=760 |
| B1-p2.1 batched prefill | PASS, 10/12 argmax bit-id, max_abs_diff=0.1875 |
| B1-p2.2 batched decode | PASS, 57/60 argmax bit-id, decode max_abs_diff=1.6191 |

## Notes

- **Pre-allocation choice**: `Vec<Option<RequestState>>` with fixed length `b_max` avoids reallocation churn and lets `row_idx` stay stable across the request's lifetime. Subsequent sub-phases rely on the row_idx being a fixed slot index into a batched KV cache.
- **Monotonic ID, no reuse**: prevents stale-id bugs after a slot is reused immediately after eviction. The cost is a 64-bit counter (practically infinite headroom).
- **Single-threaded**: Scheduler is not `Send + Sync`. 3b will choose between running it on the main runtime thread or in `spawn_blocking`.
- **Sampler ownership**: `RequestState` moves the Sampler from `GenerateRequest` (no clone). The `sampler_cloned_per_request` test verifies the resulting `Sampler` instances live in distinct memory addresses, satisfying the per-row independence requirement that 3e relies on.
- **Plan typo caught at integration test**: The plan's Step 2.1 expected `final_ids = vec![0, 2, 4, 5]`, but the actual `active()` returns rows in slot order, so the correct value is `vec![0, 4, 2, 5]`. The test file was committed with the corrected value and an explanatory comment.

## B1-p2.3x Next Steps

- **B1-p2.3b** — Add `Scheduler::step(model: &Qwen35Model, cache: &mut [LayerCache], target) -> Result<Vec<StepEvent>>` packing all `active()` rows' input tokens into a `[B, 1]` tensor, calling `model.forward_on`, sampling per row, updating each `RequestState`, returning per-row events. HTTP server (OpenAI handler) refactored to drive the scheduler instead of `GenerationStream`.
- **B1-p2.3c** — Per-row KV cache offset tracking + per-row decode mask.
- **B1-p2.3d** — Admission queue + preemption when `b_max` is full.
- **B1-p2.3e** — Per-row sampler invocation (temperature, top_k, penalties per row).

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3a-scheduler-skeleton-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3a-scheduler-skeleton.md`
- New module: `ironmlx/src/core/scheduler.rs`
- Integration test: `ironmlx/tests/b1_p2_3a_scheduler_skeleton.rs`
