# B1-p2.4 VL B>1 Batched Serving — Close-out

**Branch:** `ironmlx-b1-p2-4-batched-vl` (off `ironmlx-b1-p2-3-continuous-batching` HEAD d999d62)
**Date:** 2026-05-16
**Status:** COMPLETE

## Summary

Removes the OpenAI/Anthropic `has_images -> GenerationStream` fallback by
extending `Scheduler` + `Qwen35Model` + `cross_modal` + `core::generate`
to handle mixed text+VL batches end-to-end. VL requests flow through
SchedulerActor + continuous batching.

Decode path (`step_inner` / `build_per_row_decode_mask` /
`build_decode_position_ids`) UNCHANGED -- VL row K/V is bit-equivalent
to text row K/V at the cache abstraction (verified by 13-suite regression
sweep).

## Acceptance

Per-row sampled tokens from the SchedulerActor are compared against B=1
GenerationStream baselines using `argmax_bit_id_ratio >= 0.95` over an
8-token window. Matches 3c-3 / b1_p2_1 established convention for
B>1 batched paths.

| Gate | Result |
| --- | --- |
| S1: B=2 full-VL ratio >= 0.95 | PASS (row A: 1.0000, row B: 1.0000) |
| S2: B=2 mixed text+VL ratio >= 0.95 | PASS (text: 1.0000, VL: 1.0000) |
| S3: mid-admit VL during text decode ratio >= 0.95 | PASS (A: 1.000, B: 1.000, VL: 1.000) |
| S4: multi-image per row in batched VL ratio >= 0.95 | PASS (row 0: 1.000, row 1: 1.000) |
| Unit: cross_modal 3 B>1 tests | PASS |
| Unit: build_position_ids_vl_batched 2 tests | PASS |
| Unit: batched_prefill_vl 2 equivalence tests (#[ignore]) | PASS |
| Unit: admit_carries_vl_fields | PASS |
| fmt --check / clippy -D warnings / build --release | ALL CLEAN |

## Architectural changes per spec section 4

| Item | File | Change |
| --- | --- | --- |
| 4.3.1 RequestState +4 vision fields | `core/scheduler.rs:79` | Added |
| 4.3.2 batched_prefill_vl | `models/qwen3_5/model.rs` | Added |
| 4.3.3 replace_image_tokens B>1 | `models/qwen3_5/cross_modal.rs:36` | B=1 guard removed |
| 4.3.4 build_position_ids_vl_batched | `core/generate.rs` | Added |
| 4.3.5 admit_mid_inner VL dispatch | `core/scheduler.rs:759` | Dispatch added |
| 4.3.6 prefill_admitted_inner VL dispatch | `core/scheduler.rs:373` | Dispatch added |
| 4.3.7 HTTP fallback removal | `core/server/openai.rs` | `has_images` dropped from `use_scheduler` |
| 4.2 step_inner unchanged invariant | `core/scheduler.rs:547` | Verified by regression sweep |

## Plan-correction deviations

- **Spec 4.5 acceptance gate** said "bit-identical" -- strict equality is
  not achievable at B>1 due to documented bf16 numerical drift in GPU
  kernel reduction (`b1_p2_1`: "GPU kernel reduction scheduling can flip
  argmax on near-tied logits"; `3c-3`: `ARGMAX_BITID_GATE = 0.95`).
  Adopted ratio metric: `argmax_bit_id_ratio >= 0.95` over 8-token window.
  Observed: all 4 scenarios achieved 1.0000 (bit-identical) in practice.
- **Test prompt construction** uses `apply_chat_template` with
  `enable_thinking: false` (matches 3c-3 pattern). VL hand-built prompts
  append `<think>\n\n</think>\n\n` after assistant opener -- Qwen3's thinking
  mode otherwise emits a canned opener identical across prompts, making
  per-row distinguishable baselines impossible. This was the root cause of
  the BLOCKED state: prompts without thinking disabled collapsed to
  identical `<think>` preambles, ratio 0.05 false alarm.
- **T4 4.3.6 prefill_admitted_inner** introduced a `GridThwSlice<'a>`
  type alias to fix `clippy::type_complexity` on
  `Vec<Option<&[(i32,i32,i32)]>>`.
- **T3 unit tests** marked `#[ignore]` (real-model heavy ~168s combined)
  matching `forward_vl_text_only_matches_forward_on` convention. Run via
  `cargo test ... -- --ignored`.

## Commits (T1-T4 pre-existing, T5 this commit)

- T1: `890ec00` cross_modal B>1 + 3 unit tests
- T2: `2b99acd` build_position_ids_vl_batched + `95f8323` doc fix
- T3: `1a4c535` batched_prefill_vl + `ed63cff` ignore/doc fix
- T4: `3c025d2` Scheduler dispatch + HTTP fallback removal
- T5: integration scenarios + 13-suite regression + close-out

## Regression Status

| Suite | Result | Time |
| --- | --- | --- |
| p6_qwen35_vl_logits_match | PASS | 545s |
| p6_6_logits_match | PASS | 444s |
| p6_7_chunked_prefill | PASS | 30s |
| b1_p2_1_batched_prefill | PASS | 2381s |
| b1_p2_2_batched_decode | PASS | 1544s |
| b1_p2_3a_scheduler_skeleton (default mode) | PASS | <1s |
| b1_p2_3b_1_scheduler_step | PASS | 295s |
| b1_p2_3b_2_scheduler_actor | PASS | 54s |
| b1_p2_3b_3_admission_window | PASS | 450s |
| b1_p2_3b_4_anthropic_actor | PASS | 74s |
| b1_p2_3c_1_per_row_offset | PASS | 524s |
| b1_p2_3c_2_scheduler_decode_mask | PASS | 283s |
| b1_p2_3c_3_continuous_batching | PASS | 945s |
| **B1-p2.4 batched VL (4 scenarios)** | **PASS** | **892s** |

Total wall time: ~2.0 hours for the 13-suite sweep.

## Compat sunset

| Removed | Replaced with |
| --- | --- |
| OpenAI `has_images -> GS` fallback | VL requests route through SchedulerActor + batched_prefill_vl |
| `cross_modal::replace_image_tokens` B=1 guard | B>1 supported |

## Notes / known limitations carrying forward to backlog

- Vision encoder per-row sequential -- spec NG1. Concat-pv ViT future micro-optimization.
- Long VL prompt fallback -- falls back to GS when `prefill_chunk_size`. Sunsets in 3c+.
- VL admit_mid stall -- synchronous vision encoder. Sunsets in 3c+.
- `b_max=4` hardcoded -- `core/server/mod.rs:54`. Sunsets in 3d.
- bf16 batched drift -- documented in b1_p2_1; argmax bit-id ratio is the canonical acceptance metric for B>1 paths.

## B1-p2 Next Steps

| Sub-spec | Scope | Status |
| --- | --- | --- |
| B1-p2.3c+ | Chunked admit_mid prefill + decode-interleave | Backlog |
| B1-p2.3d | Admission queue + b_max config exposure | Backlog |
| B1-p2.3e | Per-row async sampler tuning | Backlog |
| B1-p2.4 | Batched VL serving | DONE (this report) |
| B1-p2.5 | Production hardening (OOM safety, fairness) | Future |

After B1-p2.4: Qwen3.5 VL multi-request serving complete. Next major program: Qwen3.5 MoE.
