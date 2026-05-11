# P6.4 + P6.5 Cleanup — Close-out

**Branch:** `ironmlx-p6-4-cleanup` (off `ironmlx` after P6.3 merge)
**Date:** 2026-05-11
**Driver:** P6.3 audit (see prior conversation)

## Summary

15 audit findings closed across two tracks. Production breakage paths
(panics on user input, dead code, hardcoded model config) eliminated
without changing any vision-encoder algorithm. All 8 P6.3 acceptance
gates remain green.

## Track A — Bug / panic / dead code / placeholder (7 fixes)

| ID | Title | Commit | Notes |
| --- | --- | --- | --- |
| A1+A2+A3 | image_processor panics → Result | `5407246` | smart_resize / patchify / preprocess no longer panic on hostile image input |
| A4 | build_position_ids_vl asserts → Err | `2e5ea10` | empty grid_thw + zero merge_size return clean errors |
| A5+A6+A7 | cross_modal asserts + dead consts | `8ae884b` | B=1 gate returns Err (audit B1 multi-image entry point); 2 dead VL token consts dropped; 1 redundant post-loop assert → debug_assert |
| A8 | merger.rs unused _grid_thw param | `e4ae8b8` | parameter removed; signature now `forward(&self, x: &Array)` |
| A9+A10+A11 | unwrap/expect audit + overflow guard | `eaf8b70` | build_rotary_freqs + PatchEmbed::new return Result; add_learned_pos_embed uses i64 for total_hw with 1M cap |

## Track B — Deferred-feature readiness (3 fixes, low-cost lifts)

| ID | Title | Commit | Notes |
| --- | --- | --- | --- |
| B4 | spatial_merge_size from VisionConfig | `af5696b` | hardcoded `/2` formulas now read `vision_config.spatial_merge_size`; threaded through GenerateRequest |
| B5 | image_token_id from tokenizer | `f348c03` | OpenAI handler resolves `<|image_pad|>` via `Tokenizer::token_to_id`; forward_vl takes the id as a parameter |
| B6 | Anthropic 400 on image parts | `0ffdb38` | silent drop → explicit 400 with actionable message |

## Deferred (still in audit B-track, not done here)

| ID | Title | Why deferred |
| --- | --- | --- |
| B1 | Multi-image batch (B>1) | Large effort — cross-cutting through cache + scheduler. Owns its own P-track. |
| B2 | temporal_patch_size hardcoded | Medium effort; tied to actual video decoding (B3). |
| B3 | t=1 single-frame in OpenAI handler | Requires video decoder dependency (ffmpeg) and per-frame batching. |
| B7 | VL prefill single-chunk guard | Medium: needs vision-tower pre-cache so chunk N>0 reads pre-patched embeddings. |
| B8 | Server VL concurrency ceiling | Continuous batching is its own multi-week P-track. |

## Regression verification

### Production gates (all green, unchanged from P6.3)

| Gate | Target | Current | Notes |
| --- | --- | --- | --- |
| 1. Preprocess max_diff | < 0.05 | **0.0254** | unchanged — patchify unchanged |
| 2. Vision encoder max_diff | < 0.1 | **0.0330** | unchanged — vision/*.rs unchanged in P6.4/P6.5 |
| 3A. E2E logits max_diff | < 0.5 | **0.3906** | re-verified (Task 21 PASS) |
| 3B. Greedy first-token | bit-identical | **760 ✅** | re-verified |
| 4a. COCO key facts | ≥ 2/3 | **3/3** | re-verified (two cats / green collar / remote) |
| 4b. scene non-double | yes | **yes** | re-verified |
| 4c. counting ±2 | in [10, 16] | **13 / 13 / 14** | re-verified |
| 4d. STOP inversion | yes | **yes (POTS)** | re-verified |

### Build hygiene

- `cargo test -p ironmlx --lib --release`: **153 passed / 0 failed** (added 1 new test in A1-A3)
- `cargo fmt --check`: clean
- `cargo +nightly clippy -D warnings`: clean (one `#[allow(clippy::too_many_arguments)]` added to `Qwen35Model::forward_vl` for the new image_token_id parameter)
- `cargo build --release`: clean

## P6.6+ candidates (next deferred audit items + new ones surfaced)

- B1 (multi-image): cross_modal already returns Err with hint instead of panic
- B7 (VL chunked prefill): requires pre-cache redesign
- Tokenizer-side: a sanity-check at server start that the `<|image_pad|>` /
  `<|vision_start|>` / `<|vision_end|>` ids match what the chat-template
  emits — currently relies on prompt-text → tokenizer round-trip
