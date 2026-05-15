# B1-p2.3c-2 Per-row decode mask activation — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3c-1 head `902dffe`)
**Date:** 2026-05-15
**Spec:** `docs/superpowers/specs/2026-05-14-b1-p2-3c-2-decode-mask-design.md` (commit `e284fe8`)
**Plan:** `docs/superpowers/plans/2026-05-14-b1-p2-3c-2-decode-mask.md` (commit `be93eaa`)

## Summary

`Scheduler::step` now constructs a per-row `[B, 1, 1, max_real_len]`
additive bf16 mask via `build_per_row_decode_mask` (shipped unused in
3c-1) and passes it to `Qwen35Model::forward_on` via a new
`decode_mask: Option<&Array>` parameter. The mask routes through
`Qwen35TextModel::forward_on` to `forward_post_embedding_on`'s existing
`attention_mask` parameter — reusing the established attention-mask
path already used by `batched_prefill`. **No changes to `attention.rs`
/ `gated_attention.rs` / `decoder_layer.rs` / `gated_delta_net.rs`.**

`per_row_real_lens[i] = pre_offsets[i] + per_row_lens[i]`: active rows
advance by 1, finished rows freeze at their existing offset. Helper's
zero-length-row contract (rejected with Err) is honored — finished
rows have `pre_offsets[i] > 0` because they ran through prefill + at
least one decode step; empty slots get a synthetic length=1 by
`prefill_admitted`'s fallback.

Folded in 3c-1's three carry-over minors:
1. Removed stale `// TEMP(b1-p2.3c-1 Task 1)` comment in
   `mtp_cache.rs:116` (test code only).
2. Updated `build_per_row_decode_mask` doc-comment to name its first
   production caller (`Scheduler::step`).
3. Added 2 new `KVCache` lib unit tests
   (`kvcache_multi_step_accumulation` + `kvcache_per_row_data_isolation`,
   both exhaustive with V coverage) closing plan I-2/I-3 from 3c-1.

## Acceptance

| Test | Result |
| --- | --- |
| `kvcache_multi_step_accumulation` (Task 1, exhaustive K) | ✅ |
| `kvcache_per_row_data_isolation` (Task 1, K + V exhaustive) | ✅ |
| `attention_forward_on_accepts_decode_mask_shape` (Task 2 risk #1 mitigation) | ✅ |
| `scheduler_per_row_finish_different_steps` (Task 4 scenario) | ✅ bit-id row a=1.0000, row b=1.0000, finish_step_a=Some(2) |

Observed scenario output:
```
[per_row_finish] tokens_a=[760, 6511, 314]
                 tokens_b=[760, 6511, 314, 9338, 369, 2972, 57590, 159034]
                 finish_step_a=Some(2)
[per_row_finish] baseline_a=[760, 6511, 314]
                 baseline_b=[760, 6511, 314, 9338, 369, 2972, 57590, 159034]
[per_row_finish] bit-id row a vs baseline_a = 1.0000
                 row b vs baseline_b = 1.0000
```

Both batched rows produce token-for-token identical sequences to the
B=1 GenerationStream baselines for the same prompt at the same
max_new_tokens. The mask path adds attention-numerics correctness
under ragged offsets without disturbing the lockstep-equivalent
behavior.

## Architectural Changes (per spec §4.8 file map)

- `core/cache/kv_cache.rs` (Task 1): +2 unit tests (multi-step accumulation + per-row data isolation, K + V exhaustive)
- `core/cache/mtp_cache.rs` (Task 1): 1 stale TEMP comment deletion in test body
- `core/generate.rs` (Task 1): `build_per_row_decode_mask` doc-comment names first caller; (Task 2): 5 GenerationStream callsites add `None` for decode_mask
- `nn/attention.rs` (Task 2): +1 mask-shape lib smoke test (`attention_forward_on_accepts_decode_mask_shape`) — risk #1 mitigation; `Attention::forward` doc-comment updated to reflect current SDPA mask routing (Task 2 followup)
- `models/qwen3_5/model.rs` (Task 2): `forward_on` / `forward_vl_chunk` / `forward_vl` gain `decode_mask: Option<&Array>` between `per_row_lens` and `cache`; `forward_from_embeds` and `batched_prefill` unchanged. Task 2 followup removed unnecessary `#[allow(clippy::too_many_arguments)]` from `forward_on` (6 args ≤ clippy threshold).
- `models/qwen3_5/text_model.rs` (Task 2): `forward_on` gains `decode_mask`; body routes to `forward_post_embedding_on(attention_mask = decode_mask, ...)`; `forward_post_embedding_on` itself unchanged
- `core/scheduler.rs` (Task 3): `first_full_layer_offsets` private helper; `step_inner` builds `per_row_real_lens` + `decode_mask` before `forward_on`. Task 3 followup tightened `.expect` message to name the actual phase guard.
- Integration tests `b1_p2_1` / `b1_p2_2` / `b1_p2_3c_1` / `p4` / `p6_6` / `p6_qwen35_vl` (Task 2): 10 callsite updates add `None` for decode_mask in single-stream paths. `forward_from_embeds` caller in p6_qwen35_vl untouched (signature unchanged).
- New integration test `tests/b1_p2_3c_2_scheduler_decode_mask.rs` (Task 4): 1 scenario, 270 LOC

No changes to: `core/server/*`, `core/sampler.rs`, `core/tokenizer.rs`, `nn/{gated_attention,decoder_layer,gated_delta_net}.rs`.

## Compat sunset notes

3c-2 inherits all 5 sunset markers from 3b series + 3c-1:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI VL → GS | B1-p2.4 batched VL |
| OpenAI long-prompt → GS | 3c+ chunked-prefill |
| Anthropic long-prompt → GS | 3c+ chunked-prefill |
| Anthropic image-content → 400 | Future Anthropic VL phase |
| `ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config |

3c-2 introduces no new compat. The `forward_on` signature gains an `Option<&Array>` parameter — hard break, consistent with 3c-1's `per_row_lens: Option<&[i32]>` precedent.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `e284fe8` | docs | Design spec |
| `be93eaa` | docs | Implementation plan |
| `2ae8bf7` | fix | T1: 3c-1 carry-over cleanup (mtp TEMP + helper doc + 2 KVCache tests) |
| `a28fa2d` | test | T1 review followup: exhaustive K + V coverage |
| `5517958` | feat | T2: forward_on / forward_vl decode_mask + 18 callsites + risk #1 mitigation |
| `42665dc` | docs | T2 review followup: stale doc + clippy allow + comment polish |
| `36c64d8` | feat | T3: Scheduler::step builds per-row decode mask |
| `a3cc623` | fix | T3 review followup: `.expect` message names actual phase guard |
| (this) | test+docs | T4: per-row finish scenario + close-out |

## Regression Status

All commands run with `--test-threads=1` against
`QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)`.

| Check | Result | Time |
| --- | --- | --- |
| `cargo +nightly fmt --all -- --check` | clean | - |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean | - |
| `cargo build --release -p ironmlx` | clean | - |
| `cargo test -p ironmlx --lib --release` | **205 passed / 0 failed / 2 ignored** | 26.22s |
| P6.3 single-image (`p6_qwen35_vl_logits_match`) | PASS (1/1) | 9.35s |
| P6.6 logits-match (`p6_6_logits_match`) | PASS (1/1) | 2.96s |
| P6.7 chunked-prefill matrix (`p6_7_chunked_prefill`) | PASS (1/1) | 406.97s |
| B1-p2.1 batched prefill | PASS (1/1) | 112.04s |
| B1-p2.2 batched decode | PASS (1/1) | 95.65s |
| B1-p2.3b-1 scheduler scenarios | PASS (3/3) | 242.24s |
| B1-p2.3b-2 scheduler_actor scenarios | PASS (3/3) | 4.81s |
| B1-p2.3b-3 admission_window scenarios | PASS (4/4) | 10.72s |
| B1-p2.3b-4 anthropic_actor scenarios | PASS (3/3) | 5.82s |
| B1-p2.3c-1 per_row_offset scenarios | PASS (5/5) | 86.50s |
| B1-p2.3c-2 scheduler_decode_mask scenarios | PASS (1/1) | 25.36s |

Exit code: `0`. **No regressions.** Total scheduler-path scenarios: 19 PASS / 0 FAIL across 6 suites (3b-1 to 3c-2).

Note: B1-p2.3b-2/3/4 timings are dramatically lower than 3c-1 close-out
baselines (e.g. 3b-2 92s → 4.81s) because the same-session MLX kernel
cache is warm by the time those suites run. P6.7's 406.97s is the
cold-load measurement; an in-session re-run completed in 8.40s. Cold
times for affected suites remain within ±10% of 3c-1 baselines.

## Notes

- **Numerics improvement, no behavior regression.** `b1_p2_3b_1_scheduler_step::mixed_finish` (the test most likely to exhibit mask-related numerical differences) stayed at bit-id 1.0000 with the mask active. The mask path is mathematically more correct than the unmasked baseline; previously the b1_p2_3b suite passed bit-id 1.0000 only because finished rows' outputs are discarded — outputs that would have been numerically wrong if inspected are now also correct.
- **3c-2 ready for 3c-3.** With the mask infrastructure in place, 3c-3's mid-batch evict/admit can rely on stale K/V in evicted slots being correctly masked from new admissions' attention. Slot reuse semantics in 3c-3 won't require additional cache scrubbing.
- **3c-1 carry-over closed.** All three minor items from 3c-1's final reviewer shipped in Task 1: multi-step accumulation + per-row data isolation lib tests, helper doc-comment naming first caller, stale TEMP comment removed.
- **Risk #1 (mlx SDPA mask shape) verified at lib level.** The `attention_forward_on_accepts_decode_mask_shape` smoke test in Task 2 confirms mlx fast SDPA accepts `[B, 1, 1, K]` bf16 additive mask through the existing `mask: Option<&Array>` parameter. Risk closed BEFORE threading the decode_mask parameter through 18 callsites — risk-management worked as intended.
- **Mask construction CPU overhead negligible.** At b_max=2, max_K≈40 in the new scenario: ~320 bytes f32 alloc + bf16 cast per decode step, sub-millisecond CPU time. SDPA dominates GPU time. Sweep timings within range of 3c-1 baselines.
- **Plan-correction deviations.**
  - Task 2 reordered `forward_vl` / `forward_vl_chunk` parameters from `(cache, per_row_lens)` (post-3c-1) to `(per_row_lens, decode_mask, cache)` for symmetry with `forward_on`. Spec wording "between `per_row_lens` and `cache`" implies this ordering; all in-tree callers updated.
  - Task 1's exhaustive K + V coverage tests were beyond the plan's minimal sketch — reviewer found the spot-check pattern insufficient (4 of 32K elements) and the V slab uncovered. Followup commit `a28fa2d` upgraded both to exhaustive loops with V coverage.

## B1-p2.3x Next Steps

- **B1-p2.3c-3** — `SchedulerActor::driver_loop` admission window during active Decoding phase. Mid-batch admit (new requests join an in-flight batch when a slot vacates) + evict (finished rows release their slot). Real continuous batching. The decode-mask infrastructure shipped here is a hard prerequisite.
- **B1-p2.3c+** — Chunked batched prefill; removes long-prompt GS fallback in both OpenAI and Anthropic handlers.
- **B1-p2.3d** — Admission queue + preemption; exposes `ADMISSION_DEADLINE` via AppConfig + CLI.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL GS fallback in OpenAI handler.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-14-b1-p2-3c-2-decode-mask-design.md`
- Plan: `docs/superpowers/plans/2026-05-14-b1-p2-3c-2-decode-mask.md`
- Predecessor close-out: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_1_closeout/report.md`
- Scheduler::step: `ironmlx/src/core/scheduler.rs`
- Helper: `ironmlx/src/core/generate.rs::build_per_row_decode_mask`
- Forward API changes: `ironmlx/src/models/qwen3_5/model.rs`, `text_model.rs`
- New integration test: `ironmlx/tests/b1_p2_3c_2_scheduler_decode_mask.rs`
