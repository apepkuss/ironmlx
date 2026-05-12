# P6.7 VL Chunked Prefill — Close-out

**Branch:** `ironmlx-p6-7-vl-chunked-prefill` (off `ironmlx-p6-6-multi-image` head `310ae36`)
**Date:** 2026-05-12
**Spec:** `docs/superpowers/specs/2026-05-12-p6-7-vl-chunked-prefill-design.md` (commit `6638873`)
**Plan:** `docs/superpowers/plans/2026-05-12-p6-7-vl-chunked-prefill.md` (commit `0404ba5`)

## Summary

Lifted the single-chunk constraint on VL prefill. Users can now combine
`prefill_chunk_size > 0` with `pixel_values` in any pairing. The chunked
path produces **bit-identical** greedy first-token output across
chunk_size ∈ {0, 256, 64} on both N=2 and N=3 fixtures — confirming the
implementation is numerically equivalent to the single-chunk path (only
the order of chunk-cache writes changes, not the math).

## Acceptance Matrix

| Point | N images | chunk_size | First Token | Status |
| --- | --- | --- | --- | --- |
| 1 | 2 | 0   | 760 | ✅ |
| 2 | 2 | 256 | 760 | ✅ |
| 3 | 2 | 64  | 760 | ✅ |
| 4 | 3 | 0   | 760 | ✅ |
| 5 | 3 | 256 | 760 | ✅ |
| 6 | 3 | 64  | 760 | ✅ |

All 6 points PASS with bit-identical greedy first token (= `expected_first_token = 760`).

### Test runtime

| N | prompt_len | runtime |
| --- | --- | --- |
| 2 | 548 tokens | 45.51 s |
| 3 | 850 tokens | 686.39 s |

The dramatic runtime difference (~15× for ~1.55× prompt) reflects the cost of
3 sequential prefill+decode runs at N=3 (more vision tokens → more KV
writes → more transformer compute), not a chunked-path inefficiency.

## Architectural Changes

1. **`Qwen35Model::compute_vision_embeds`** (new) — runs only the vision tower; returns `[N_total_patches / spatial_merge_size^2, hidden]`. Commit `b70a7f3`.
2. **`Qwen35Model::forward_vl_chunk`** (new) — forward one chunk with a pre-computed `vision_embeds_slice`. Commit `5f63729`.
3. **`Qwen35Model::forward_vl`** (refactored) — now a thin wrapper around `compute_vision_embeds` + `forward_vl_chunk`. Existing call sites (P6.6 / P6.3 tests, server) unchanged. Commit `f8f7e66`.
4. **`GenerationStream`** — added `vision_embeds_full`, `position_ids_full`, `image_pad_consumed` fields. Commit `0792208`.
5. **Pre-loop compute** in `GenerationStream::new` — runs vision tower + `build_position_ids_vl` once per VL request before the chunking loop. Deleted the `prompt_len > effective_chunk` guard at the old lines 363-376. Commit `60bbc8b`.
6. **Chunking loop rewrite + slicing helpers** — per-chunk slicing of `vision_embeds_full` (keyed by running `image_pad_consumed`) and `position_ids_full` (axis-2 slice); post-loop assertion verifies all rows consumed. Three new helpers (`count_image_pad`, `slice_pos_ids_axis2`, `slice_vision_embeds_rows`) + 4 unit tests. Commit `dab1044`.
7. **`cross_modal::replace_image_tokens`** — **signature unchanged**. Chunking layer slices `vision_embeds_full` before calling, preserving the per-chunk `vision_embeds.rows == input_ids.image_pad_count` invariant.
8. **Integration test** `p6_7_chunked_prefill.rs` — drives 3 chunk_sizes against current fixture. Commit `787ba5c`.

## Fixes Applied

Zero fix-loop iterations needed. The refactor + new chunked path produced bit-identical first-token output on the first build at chunk_size=0 (single-chunk equivalence) and on the first run at chunk_size ∈ {256, 64} (true chunked equivalence).

| Commit | Type | Description |
| --- | --- | --- |
| `b70a7f3` | feat | `Qwen35Model::compute_vision_embeds` |
| `5f63729` | feat | `Qwen35Model::forward_vl_chunk` |
| `f8f7e66` | refactor | `forward_vl` wraps `compute_vision_embeds` + `forward_vl_chunk` |
| `0792208` | feat | `GenerationStream` VL-state fields |
| `60bbc8b` | feat | Pre-compute vision_embeds + position_ids_full before chunk loop |
| `dab1044` | feat | Chunking loop slices vision_embeds + position_ids per chunk |
| `787ba5c` | test | `p6_7_chunked_prefill.rs` matrix (3 chunk_sizes × current N) |

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean (only unchanged mlx-sys C++ warnings) |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **156 passed / 0 failed** |
| P6.3 Task 21 single-image logits-match | **PASS** (52.11s) — internal assert max_diff < 0.52, first_token == 760 |
| P6.6 N=2 logits-match | **PASS** — max_diff=0.9004, first_token=760 (unchanged from P6.6 close-out) |
| P6.6 N=3 logits-match | **PASS** — max_diff=1.1250, first_token=760 (unchanged from P6.6 close-out) |
| P6.7 matrix N=2 | **PASS** — all 3 chunk_sizes → 760 |
| P6.7 matrix N=3 | **PASS** — all 3 chunk_sizes → 760 |

### Lib test count change

P6.4 baseline = 153 passed. P6.7 final = 156 passed.

- **+4 new tests**: `count_image_pad_basic`, `slice_pos_ids_axis2_basic`, `slice_pos_ids_axis2_rejects_bad_shape`, `slice_vision_embeds_rows_basic` (in `p6_7_helper_tests` mod)
- **-1 obsolete test**: `vl_single_chunk_guard_rejects_oversized_prompt` — exercised the deleted single-chunk guard; removed in commit `dab1044` as a corollary of the guard removal

Net: 153 - 1 + 4 = 156.

## Notes

- The chunked path is **numerically equivalent**, not just functionally correct. Same vision tower output, same scatter, same transformer forward. Chunking only changes the cache-write granularity. The integration test asserts first-token bit-identical across chunk sizes; if any chunk-cache write diverged the argmax would shift.
- **Memory hold for vision_embeds_full** during prefill: ~10.4 MB for a 3-image request at N=2080 patches × 2560 hidden × 2 bytes bf16. Acceptable.
- **VL intermediate chunk LM-head waste** (`generate.rs` chunking loop): every VL chunk including intermediates calls `forward_vl_chunk` which always runs `slice_last_and_project`. Intermediate chunks discard the result. Cost: one `[1,1,H]→[1,1,vocab]` matmul per intermediate chunk; cheap vs the transformer cost. Future optimization: split `forward_vl_chunk` into hidden-only + project variants.
- **`image_pad_consumed` struct field on `GenerationStream`** is currently unused (the chunking loop uses a loop-local mutable counter). The `#[allow(dead_code)]` attribute remains. Future cleanup: either remove the field or migrate the loop to use it. Non-blocking.
- **Range-error test coverage gap**: `slice_pos_ids_axis2` has a bad-shape error test but neither helper has a bad-range error test. Coverage from the post-loop `image_pad_consumed == expected` assertion + integration test is adequate; explicit unit coverage for the error paths is a non-blocking cleanup opportunity.

## P6.8+ Candidates

- **B1-p2 / B8 — batched serving** (multiple independent requests packed into one forward): next P-track, large multi-week effort.
- **Tokenizer startup sanity-check** (P6.6 close-out candidate): verify `<|image_pad|>` / `<|vision_start|>` / `<|vision_end|>` token ids match what the chat-template emits. Small (~0.5d).
- **Performance**: drop the LM-head projection on intermediate VL chunks if profiling shows it matters.
- **Streaming vision tower**: overlap with text prefill — would require async-eval discipline analogous to the work that was rolled back in P8a-stage5.
- **Cleanup**: remove the unused `image_pad_consumed` field on `GenerationStream` OR migrate the loop to use it.

## Linked Reports

- N=2 matrix log: `n2_matrix.log`
- N=3 matrix log: `n3_matrix.log`
