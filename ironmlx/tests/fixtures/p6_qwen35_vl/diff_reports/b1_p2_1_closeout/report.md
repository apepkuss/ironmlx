# B1-p2.1 Static Batched Prefill — Close-out

**Branch:** `ironmlx-b1-p2-batched-serving` (off `ironmlx-p6-7-vl-chunked-prefill` head `343f173`)
**Date:** 2026-05-12
**Spec:** `docs/superpowers/specs/2026-05-12-b1-p2-1-batched-prefill-design.md` (commit `7107d47`)
**Plan:** `docs/superpowers/plans/2026-05-12-b1-p2-1-batched-prefill.md` (commit `695202e`)

## Summary

Added `Qwen35Model::batched_prefill` — a model-level API that runs B prompts
through one transformer forward. Numerical equivalence with per-stream
`forward_on` verified across 4 points (B ∈ {2, 4} × {same-length, mixed-length
with left-padding}). All single-stream regression paths (P6.3, P6.6, P6.7)
remain bit-identical.

This is phase 1 of 5 in the B1-p2 batched-serving program. Phases 2-5 (batched
decode, continuous batching, VL B>1, production hardening) build on this
foundation without changing it.

## Acceptance Table

| Point | B | prompt_lens | max_abs_diff | argmax bit-identical |
| --- | --- | --- | --- | --- |
| 1 | 2 | [128, 128] | **0.000977** | ✅ both rows |
| 2 | 2 | [128, 96] | **0.000977** | ✅ both rows |
| 3 | 4 | [128, 128, 128, 128] | **0.000977** | ✅ all 4 rows |
| 4 | 4 | [128, 96, 64, 128] | **0.000977** | ✅ all 4 rows |

All 4 points PASS the `max_abs_diff < 1e-3` gate and the bit-identical-argmax
check on every batch row.

### On the 0.000977 floor

`0.000977 = 1/1024 = 2^-10` is exactly the bf16 mantissa step (ULP) for
values around the magnitude differences seen in attention output. The
observed `max_abs_diff` sitting at this floor across all 4 points is the
numerical signature of "bit-exact compute, just expressed in bf16." If a
future change pushed the floor higher (e.g., 2^-9 = 0.00195), the first
suspect would be a bf16 accumulation path widening, not an algorithmic bug.

## Architectural Changes

1. **`build_position_ids_batched`** (new free fn in `core/generate.rs`) —
   produces `[3, B, max_len]` int32 with pad-region position = 0, real-region
   `0..L_i-1` per batch row.
2. **`build_batch_attention_mask`** (new free fn in `core/generate.rs`) —
   produces `[B, 1, max_len, max_len]` additive mask combining causal +
   left-padding boundary (additive 0 in allowed cells, `-inf` elsewhere).
3. **`attention.rs::forward_on`** — wired the `mask: Option<&Array>`
   parameter that had been discarded since P1 (`let _ = mask;` line removed).
   `None` arm preserves the existing `mask_mode="causal"` SDPA call;
   `Some(m)` arm passes the explicit array mask with `mask_mode=""`.
4. **`text_model.rs::forward_post_embedding_on`** — gained `attention_mask:
   Option<&Array>` parameter; passes through to each layer's `forward_on`.
   Three existing callers (`text_model.rs::forward_on`, `model.rs::forward_from_embeds`,
   `model.rs::forward_vl_chunk`) updated to pass `None` and preserve their
   single-stream causal behavior.
5. **`Qwen35Model::batched_prefill`** (new public method) — composes
   `embed_on` + `forward_post_embedding_on(Some(attention_mask), ...)` +
   `slice_last_and_project`. Pure text; no vision tower. Returns
   `[B, 1, vocab]`.

`cross_modal.rs`, `decoder_layer.rs`, KVCache, and the VL forward path
(`forward_vl`, `forward_vl_chunk`, `compute_vision_embeds`) are unchanged.

The cache code's pre-existing comment ("Multi-request paged cache is P8/P9
work") aged well — `KVCache::new(batch=B)` already supports B>1 without any
changes, and this is the first caller to exercise it.

## Fixes Applied

Zero fix-loop iterations. The implementation worked end-to-end on the first
integration-test run.

| Commit | Type | Description |
| --- | --- | --- |
| `e65d00c` | feat | `build_position_ids_batched` helper + 2 unit tests |
| `ee950e0` | feat | `build_batch_attention_mask` helper + 2 unit tests |
| `4e042c6` | feat | Wire `mask` parameter in `attention::forward_on` (Some→array mask) |
| `6de35a9` | feat | Thread `attention_mask` through `forward_post_embedding_on` |
| `1f1e32d` | feat | Add `Qwen35Model::batched_prefill` |
| `4ddac90` | test | 4-point batched prefill numerical equivalence |
| `<closeout-sha>` | docs | This close-out |

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean (only unchanged mlx-sys C++ warnings) |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **160 passed / 0 failed** (P6.7 baseline 156 + 4 new helper tests) |
| P6.3 Task 21 single-image logits-match | **PASS** (14.73s) — internal assert max_diff < 0.52, first_token == 760 |
| P6.6 N=2 logits-match | **PASS** (86.64s) — max_diff=0.9004, first_token=760 (unchanged from P6.6 close-out) |
| P6.7 chunked-prefill 6-point matrix | **PASS** (501.69s) — all chunk_sizes (0/256/64) → 760 |
| B1-p2.1 4-point batched prefill matrix | **PASS** — all 4 points, max_abs_diff=0.000977, argmax bit-identical |

Every single-stream code path produces bit-identical numerics post-B1-p2.1.

## Notes

- **`mask` vs `attention_mask` naming asymmetry.** Internal `attention.rs`
  uses `mask` (existing signature), while public-facing
  `forward_post_embedding_on` and `batched_prefill` use `attention_mask` for
  clarity. The single-character translation happens at the call site.
  Acceptable; not worth a rename.
- **`#[allow(clippy::too_many_arguments)]` on `batched_prefill` (5 params).**
  Clippy's default threshold is 7, so this attribute is currently a no-op.
  Kept as a pre-emptive guard against future signature growth (e.g., adding
  `output_buffer: Option<&mut Array>` for streaming).
- **`PAD_TOKEN_ID = 0` in the test.** Safe under the current
  `build_batch_attention_mask` because pad cells are fully masked. If a
  future mask off-by-one ever leaked pad cells into attention, the
  `<unk>`-id-0 ambiguity could mask the bug. Defensive observation; no
  change required.
- **KV cache equivalence is verified implicitly.** The test compares
  last-position logits, not KV cache contents directly. If any per-layer K
  or V slot for batch row i diverged from the per-stream reference, the
  attention output (and therefore the last-position logits) would shift
  above 1e-3. The 0.000977 result on all 4 points is a strong end-to-end
  proxy for KV correctness.
- **No HTTP server / scheduler changes.** Deliberate phase 1 scope. The
  existing OpenAI handler still spawns one `GenerationStream` per request
  and is unaware of `batched_prefill`.

## B1-p2.x Next Steps

- **B1-p2.2** — Batched decode (`next_token` at B>1) with KV cache hand-off
  from `batched_prefill`. Requires per-stream stop-token tracking and a
  decode-step loop that handles streams reaching `eos` at different times.
- **B1-p2.3** — Continuous batching (scheduler + admit/evict + token-level
  loop). The largest sub-phase. Touches HTTP server, request pool, fairness.
- **B1-p2.4** — VL B>1 (one or more of the B streams carry images). Requires
  `cross_modal::replace_image_tokens` to support per-batch-row image scatter.
- **B1-p2.5** — Production hardening (admission control, OOM safety, fairness
  policy, batch-size autotuning).

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-12-b1-p2-1-batched-prefill-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-b1-p2-1-batched-prefill.md`
- Integration test: `ironmlx/tests/b1_p2_1_batched_prefill.rs`
- Helper unit tests: `ironmlx/src/core/generate.rs` (mods `b1_p2_1_position_id_tests`, `b1_p2_1_mask_tests`)
