# B1-p2.3c-1 Per-row KV cache offset — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-4 head `170c50b`)
**Date:** 2026-05-14
**Spec:** `docs/superpowers/specs/2026-05-14-b1-p2-3c-1-per-row-offset-design.md` (commit `e7e57bd`)
**Plan:** `docs/superpowers/plans/2026-05-14-b1-p2-3c-1-per-row-offset.md` (commit `16906ad`)

## Summary

`KVCache` and `GatedDeltaCache` now track per-row offsets via `Vec<i32>`
(length == batch). `update_and_fetch_on` / `advance` take a
`per_row_lens: &[i32]` argument specifying how many tokens row `i`
writes in this call; rows with `per_row_lens[i] == 0` skip entirely.
Internal write uses Strategy A — a B-loop of `slice_update_on` calls.

Per-row `per_row_lens: Option<&[i32]>` threads from `Qwen35Model::{forward_on,
batched_prefill}` down through `text_model` → `DecoderLayer` →
`Attention` / `GatedAttention` / `GatedDeltaNet` → cache write.

The prefill data flow was migrated from **left-padding** (pre-3c-1
convention: real tokens at `[pad_start..max_len]`) to **right-padding**
(real tokens at `[0..L_i]`). Required for per-row offset semantics —
`KVCache::write_per_row` writes K's leading-n columns, which only
matches real K under right-padding. Plus a `take_along_axis` rewrite
of `GatedDeltaNet::forward_on` conv_state per-row slice (originally a
per-row `slice_strided_on + concatenate_on` loop introduced 3.45x
slowdown; the single-op rewrite eliminates the graph-build overhead).

New `core::generate::build_per_row_decode_mask` helper produces
`[B, 1, 1, max_len]` additive bf16 mask for the ragged decode path
3c-2 will activate.

`Scheduler::prefill_admitted` now passes `&prompt_lens` to
`batched_prefill` and updates per-slot `real_len = prompt_lens[i]`
(was `max_len`, a latent left-pad-slot bug). `Scheduler::step` passes
`per_row_lens = [1 active / 0 finished-or-empty]`.

Scheduler state-machine semantics unchanged. SchedulerActor / openai.rs
/ anthropic.rs untouched. Lib build green at every commit.

## Acceptance

| Test | Result |
| --- | --- |
| `kv_cache::tests` (13 total: 9 new per-row + 3 retained + 1 fix-followup) | PASS |
| `gated_delta::tests` (9 total: 5 new per-row + 2 construction + 2 split error-arg) | PASS |
| `per_row_decode_mask_tests` (4 total: uniform + ragged + invalid_args + bfloat16_dtype) | PASS |
| `per_row_slice_tests` (3 total: uniform_pick + ragged_pick + invalid_args) | PASS |
| `per_row_offset_uniform_lens_matches_lockstep_baseline` (Scenario 1) | PASS — bit-id row 0: 1.0000, row 1: 1.0000 |
| `per_row_offset_ragged_lens_offsets_diverge` (Scenario 2) | PASS |
| `per_row_offset_zero_len_skips_row` (Scenario 3) | PASS |
| `per_row_offset_decode_with_ragged_offsets` (Scenario 4) | PASS — bit-id row 0 (len 14): 1.0000, row 1 (len 24): 1.0000 |
| `per_row_offset_invalid_args_return_err` (Scenario 5) | PASS |

## Architectural Changes

Per spec §4.8 file map + Boss-approved right-pad expansion:

- `core/cache/kv_cache.rs` (T1): offsets: Vec<i32>; Strategy A per-row write loop; all-zero fast path; 4 Err paths
- `core/cache/gated_delta.rs` (T2): offsets: Vec<i32>; advance(&[i32]); 3 Err paths
- `core/cache/mtp_cache.rs` (T1 in-scope expansion): offset() → offsets().iter().max() (lockstep-uniform invariant)
- `core/generate.rs` (T3 + T4 right-pad): build_per_row_decode_mask helper + right-pad helpers (position_ids / attn_mask / linear_mask docs + math)
- `models/qwen3_5/model.rs` (T4): per_row_lens threaded; slice_last_and_project gets Option<last_positions> for per-row slice; per_row_slice_last free function extracted for unit-testing
- `models/qwen3_5/text_model.rs` (T4): per_row_lens threaded through forward_on / forward_post_embedding_on
- `nn/{attention, gated_attention, gated_delta_net}.rs` (T4): per_row_lens threaded; temp uniform-vec fallbacks removed
- `nn/gated_delta_net.rs` (T4 follow-up): conv_state per-row extraction via take_along_axis (single op)
- `nn/decoder_layer.rs` (T4): per_row_lens threaded to attn dispatch
- `core/scheduler.rs` (T4): prefill_admitted passes &prompt_lens + sets real_len = prompt_lens[i]; step passes [1 active / 0 pad]; right-pad input fill
- `tests/p2_kv_cache.rs` (T4): cache.offset() → cache.offsets()[0]; 3-arg update_and_fetch
- `tests/b1_p2_1_batched_prefill.rs` (T4): right-pad input + per_row_lens
- `tests/b1_p2_2_batched_decode.rs` (T4): right-pad input + per_row_lens
- `tests/p4_qwen35_logits_match.rs` / `p6_6_logits_match.rs` / `p6_qwen35_vl_logits_match.rs` (T4): API signature updates
- `tests/b1_p2_3c_1_per_row_offset.rs` (T5+T6): 5 new scenarios

## Regression Status

All commands run with `--test-threads=1` against the QWEN35_MODEL env var pointing to Qwen3.5-4B-MLX-4bit (mlx-community snapshot `32f3e8e`).

| Check | Result | Time |
| --- | --- | --- |
| `cargo +nightly fmt --all -- --check` | clean | - |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean | - |
| `cargo build --release -p ironmlx` | clean | - |
| `cargo test -p ironmlx --lib --release` | 202 passed / 0 failed / 2 ignored | 7.19s |
| P6.3 single-image (`p6_qwen35_vl_logits_match`) | 1 passed / 0 failed | 60.72s |
| P6.6 logits-match | 1 passed / 0 failed | 68.07s |
| P6.7 chunked-prefill matrix | 1 passed / 0 failed | 29.11s |
| B1-p2.1 batched prefill | 1 passed / 0 failed | 254.64s |
| B1-p2.2 batched decode | 1 passed / 0 failed | 13.61s |
| B1-p2.3b-1 scheduler scenarios | 3 passed / 0 failed | 100.81s |
| B1-p2.3b-2 scheduler_actor scenarios | 3 passed / 0 failed | 46.42s |
| B1-p2.3b-3 admission_window scenarios | 4 passed / 0 failed | 54.40s |
| B1-p2.3b-4 anthropic_actor scenarios | 3 passed / 0 failed | 33.79s |
| B1-p2.3c-1 per_row_offset scenarios | 5 passed / 0 failed | 66.96s |

Exit code: `0`. No regressions.

## Plan-Correction Deviations

1. **Boss-approved scope expansion: right-pad migration.** Plan originally assumed left-padding stayed in place. During Task 4 execution, the implementer flagged that `KVCache::write_per_row` writes K's leading-n columns (incompatible with left-pad's real-K-in-tail-n layout) — `b1_p2_1` mixed-length regression confirmed the conflict (max_abs_diff jumped from ~0.6 to 1.97). Boss approved migrating prefill to right-padding in the same Task 4 commit. The migration touched 7 sites (position_ids / attn_mask / linear_mask / slice_last_and_project / Scheduler input fill / test inputs / + the unplanned GatedDeltaNet conv_state per-row slice).

2. **GatedDeltaNet conv_state per-row slice rewrite.** Initial Task 4 used `slice_strided_on + concatenate_on` per row per layer for conv_state extraction. This added B+1 graph nodes per layer per step that the JIT couldn't fuse with downstream kernels — produced a 3.45x slowdown on `b1_p2_2_batched_decode_matrix` (211s → 729s). Task 4 review followup rewrote it as a single `take_along_axis_on` op against a broadcast index tensor; result was a 50x speedup vs the slow state (~13s) and 16x speedup vs the pre-3c-1 baseline (~211s).

3. **Scenario 3 deviation.** Plan proposed `per_row_lens = [0, 12]` via `batched_prefill`. This doesn't work end-to-end because `batched_prefill` builds `last_positions = per_row_lens[i] - 1`, which yields `-1` for the zero entry and fails `per_row_slice_last`'s bounds check. Scenario 3 instead exercises the same invariant at the cache layer: direct `KVCache::update_and_fetch` + `GatedDeltaCache::advance` with `[0, 12]` to verify `offsets() == [0, 12]`. Cache-layer is the correct verification scope for Task 1+2's row-skip invariants.

4. **mtp_cache.rs unplanned patch (Task 1).** `MtpCache::offset()` delegated to `KVCache::offset()`, which was hard-deleted. The followup `c.offsets().iter().max()` patch was minimum scope to keep `cargo build` green.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `e7e57bd` | docs | Spec |
| `16906ad` | docs | Plan (6 tasks) |
| `acf9fa6` | feat | T1: KVCache per-row offsets + Strategy A |
| `0dfde46` | fix | T1 review followup: all-zero per_row_lens panic |
| `ab85668` | feat | T2: GatedDeltaCache per-row offsets + advance(&[i32]) |
| `28955d5` | fix | T2 review followup: drop redundant b field + idiom polish + test splits |
| `65a6711` | feat | T3: build_per_row_decode_mask helper |
| `dd33383` | fix | T3 review followup: reject zero-len rows + add bfloat16 test |
| `3c204cc` | feat | T4: thread per_row_lens + migrate prefill to right-padding |
| `6191ef5` | fix | T4 review followup: take_along_axis perf rewrite (729s → 13s) + per_row_slice_last unit tests + doc fixes |
| `190c816` | test | T5: per-row offset scenarios 1+2+3 |
| `a21e9b6` | test | T5 review followup: full_seen guards + Linear-layer assertion + doc polish |
| `5fcdcf6` | test | T6: scenarios 4+5 |
| `<this>` | docs | This close-out |

## Notes

- **Right-padding is now the established prefill convention.** Pre-3c-1 left-padding (logits[:, max_len-1, :] picks last column uniformly) was a B1-p2.1 convenience that breaks under per-row offsets. Right-padding aligns with vLLM/SGLang internals and unblocks 3c-2 mid-batch admit/evict.
- **Performance impact: net win on batched decode.** B1-p2.2 wall time went from 211s (pre-3c-1) → 729s (3c-1 broken state) → ~13s (3c-1 final, after take_along_axis rewrite). The take_along_axis fusion eliminated graph nodes that previously blocked downstream-kernel fusion.
- **Latent bug fix: real_len.** Pre-3c-1 `Scheduler::prefill_admitted` set `real_len = max_len` for every row, which (under left-pad) skipped pad_start positions in decode. With right-pad + per-row offsets, `real_len = prompt_lens[i]` is correct. Verified by `b1_p2_3b_1_scheduler_step::mixed_finish` bit-id 1.0000 (was 0.875 before this fix).
- **3c-1 ready for 3c-2.** The cache + model + helper machinery is in place; per-row offsets work; right-pad data flow is correct. 3c-2 (Scheduler state-machine relaxation) can now rewrite `Scheduler::step` to issue per-row finished/active patterns + admit mid-batch + use `build_per_row_decode_mask` for ragged decode.

## B1-p2.3x Next Steps

- **B1-p2.3c-2** — `Scheduler` state machine relaxation; lifts "all rows finish together" constraint. Activates per-row decode mask.
- **B1-p2.3c-3** — `SchedulerActor::driver_loop` admission window during active Decoding phase. Real continuous batching.
- **B1-p2.3c+** — Chunked batched prefill; removes long-prompt GS fallback in both OpenAI and Anthropic handlers.
- **B1-p2.3d** — Admission queue + preemption; exposes `ADMISSION_DEADLINE` via AppConfig + CLI.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-14-b1-p2-3c-1-per-row-offset-design.md`
- Plan: `docs/superpowers/plans/2026-05-14-b1-p2-3c-1-per-row-offset.md`
- Predecessor close-out: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_4_closeout/report.md`
- KVCache: `ironmlx/src/core/cache/kv_cache.rs`
- GatedDeltaCache: `ironmlx/src/core/cache/gated_delta.rs`
- New mask helper: `ironmlx/src/core/generate.rs`
- Model API: `ironmlx/src/models/qwen3_5/model.rs`
- Integration test: `ironmlx/tests/b1_p2_3c_1_per_row_offset.rs`
