# P6.6 Multi-Image-Per-Request Verification — Design

**Status:** Approved (brainstormed 2026-05-11)
**Owner:** ironmlx
**Parent:** P6.4 + P6.5 cleanup (commit `fcde351` on branch `ironmlx-p6-4-cleanup`)
**Branch target:** `ironmlx-p6-6-multi-image` (already cut from `ironmlx-p6-4-cleanup`)

## 1. Motivation

P6 / P6.1 / P6.2 / P6.3 validated single-image-per-request VL inference against mlx-vlm. P6.5 already plumbs `spatial_merge_size` + `image_token_id` from model config / tokenizer (audit B4 + B5). But **every test path through these phases used exactly one image per request**:

- Task 22 HTTP smoke: 1 image (COCO sample, two cats)
- Task 21 logits-match: 1 image
- P6.3 Item 3 semantic: 4 separate requests, 1 image each
- P6.3a/3b/3c diff pipeline: 1 image fixture

The data structures support multi-image already:
- `GenerateRequest.image_grid_thw: Option<Vec<(i32, i32, i32)>>` — list, not scalar
- `GenerateRequest.pixel_values: Option<Array>` — already `concat`'d along axis 0 in `expand_image_parts_in_messages`
- `VisionTower::forward` iterates `grid_thw` (line 124, 181, 378 of `vision/mod.rs`)
- `add_learned_pos_embed` / `compute_rotary_pos_emb` / `cu_seqlens` build all iterate per image
- `build_position_ids_vl` iterates `grid_thw` (line 263 of `generate.rs`)
- `replace_image_tokens` walks `input_ids` matching image-token positions to vision_embeds rows in order

But this is unverified. The audit (P6.4 prep) flagged this as **Track B item B1, phase 1** (the larger phase 2 is batched serving B>1).

P6.6's goal: **verify multi-image-per-request actually works against mlx-vlm reference**, and fix anything that surfaces.

## 2. Goals

Close all four gates against a 2-image fixture. Thresholds are **set after the first diagnose-only run** (mirrors P6.1's methodology) — anchoring to P6.3 single-image numbers as a reference but allowing modest headroom for the multi-image accumulation:

| Gate | What | Threshold determination |
| --- | --- | --- |
| 1 | Preprocess byte-diff against mlx-vlm processor (2 images, each compared independently) | Set after baseline; expect ≈ P6.3 single-image (0.05) |
| 2 | Vision encoder max_diff at `29_merger_out` (concat of 2 images' merger outputs) | Set after baseline; expect modest growth from P6.3 0.0330 |
| 3A | E2E logits max_diff on a 1-chat-request-2-image-parts prompt | Set after baseline; expect ≈ P6.3 0.39–0.5 |
| 3B | Greedy first-token bit-identical (1-chat-request) | Hard: must match mlx-vlm exactly. Not relaxable. |
| 4 | 2-image semantic check: 1 prompt "describe both images" → output contains ≥ 2 key facts per image | Set after baseline; per-image criterion mirrors Item 3 (≥ 2/3) |

If a gate's "diagnose-first" baseline turns out far worse than P6.3 single-image expectation (say ≥ 5× the P6.3 number), that signals a real multi-image bug — proceed to fix loop. If baselines are close to P6.3, set the threshold at the next-cleaner round number above the observed value (e.g. 0.06 if Gate 1 observes 0.054).

## 3. Non-goals

- **Batched serving (B>1)**: multiple independent chat requests packed into one forward. Own P-track (B8/B1-phase-2). P6.6 stays at B=1 per request.
- **Video input** (temporal_patch_size > 2 from real video frames): audit B2 + B3.
- **Anthropic multi-image**: P6.5 (B6) already 400-rejects image content on `/v1/messages`; will not address here.
- **N > 2 images**: YAGNI. If N=2 passes, the loop-based code paths trivially extend to N=3+. P6.6 only validates N=2 explicitly.

## 4. Architecture

Three sub-phases mirror P6.3:

```
┌──────────────────────────────┐
│ P6.6a Fixture + diagnose run │  → set thresholds, identify rupture
├──────────────────────────────┤
│ P6.6b Diagnose-fix loop      │  → iterate per-finding fixes
├──────────────────────────────┤
│ P6.6c Semantic + close-out   │  → Gate 4 + final report
└──────────────────────────────┘
```

### P6.6a — Fixture + diagnostic baseline

1. Download 2 new COCO val2017 images, topics deliberately different (e.g. dog scene + train scene). Save to `tests/fixtures/p6_qwen35_vl/multi_image/`.
2. Generate mlx-vlm reference fixture: pixel_values per image (saved as `image_0_pv.safetensors`, `image_1_pv.safetensors`), concatenated vision_embeds output (`vision_embeds.safetensors`), input_ids for the 2-image prompt, last-position logits, expected first-token.
3. Run ironmlx side: load same images via `image_processor::preprocess`, dump preprocess outputs, run `VisionTower::forward` on the concatenated pixel_values + multi-grid, dump intermediate + final tensors.
4. Diff against mlx-vlm fixture: per-image preprocess diff (Gate 1), per-block tensor diff for the vision encoder run on concat input (Gate 2-style, reusing P6.1 op-level diff format).
5. Run e2e: ironmlx HTTP server with 1 chat request containing 2 `image_url` parts; mlx-vlm full forward on same; compare last-position logits (Gate 3A) + greedy first-token (Gate 3B).
6. Compile baseline numbers; set 4 thresholds for the fix loop.

### P6.6b — Diagnose-fix loop

For each gate that doesn't pass the diagnose-set threshold:

| Likely failure surface | Candidate fix file |
| --- | --- |
| Gate 1 diff per-image differs from P6.3 single-image | nothing — preprocess is per-image so each image should match P6.3 |
| Gate 2 `29_merger_out` jumps from 0.0330 to e.g. 0.5+ | `vision/mod.rs::add_learned_pos_embed` (multi-grid path), `vision/mod.rs::compute_rotary_pos_emb` (multi-image), `vision/mod.rs::forward` (cu_seqlens construction) |
| Gate 2 first jumps at a particular block | op-level diff (P6.3b infrastructure) shows which sub-op |
| Gate 3A fails at e2e but Gate 2 passes | `cross_modal::replace_image_tokens` scatter loop with multiple image-token spans |
| Gate 3B fails | hard error — must investigate which token differs; usually points at one of the above |
| Gate 4 fails but 1-3 pass | model can't follow 2-image prompts; not a code bug |

Iterate per-fix commits (`fix(p6.6): …`). Cap to 3 hypothesis-test iterations per gate; escalate if stuck.

### P6.6c — Semantic + close-out

After Gates 1-3 pass:

1. Run Gate 4 via extended `p6_6_semantic_check.py`.
2. Write close-out report at `tests/fixtures/p6_qwen35_vl/diff_reports/p6_6_closeout/report.md` with the same gate-comparison table format as P6.3.

## 5. File Structure

New tools (none replace existing P6.1/P6.3 single-image tools — zero regression risk):

```
ironmlx/tests/fixtures/p6_qwen35_vl/
├── multi_image/
│   ├── image_0.jpg                              # new COCO val2017 download
│   ├── image_1.jpg                              # new COCO val2017 download
│   └── .gitignore                                # gitignore expected_*.npy/safetensors
├── run_p6_6_dump.py                              # NEW — mlx-vlm side dump driver
├── run_p6_6_diff.sh                              # NEW — top-level orchestrator
├── diff_preprocess_multi.py                      # NEW — multi-image preprocess diff
├── diff_pipeline_multi.py                        # NEW — multi-image op-level diff
└── p6_6_semantic_check.py                        # NEW — Gate 4 driver

ironmlx/tests/
├── p6_6_multi_image_dump.rs                      # NEW — feature-gated Rust dump
└── p6_6_logits_match.rs                          # NEW — e2e Gate 3 test

ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
├── p6_6a-<stamp>/                                # NEW — Gate 1+2 baseline reports
├── p6_6b-<stamp>/                                # NEW — fix iteration reports
├── p6_6c-<stamp>/                                # NEW — Gate 4 semantic report
└── p6_6_closeout/                                # NEW — final acceptance table
```

Source files **may** need to change:
- `ironmlx/src/models/qwen3_5/vision/mod.rs` (if Gate 2 reveals multi-grid bug in pos_embed / rotary)
- `ironmlx/src/models/qwen3_5/cross_modal.rs` (if Gate 3A reveals multi-span scatter bug)
- (mlx-vlm fork already has the dump hooks from P6.3; no new hooks needed)

## 6. Acceptance Report Format

`tests/fixtures/p6_qwen35_vl/diff_reports/p6_6_closeout/report.md` table:

```
| Gate | Target | Baseline (diagnose) | Final | Status |
| 1A. image_0 preprocess max_diff | < <tbd> | <observed> | <observed> | ?/✅ |
| 1B. image_1 preprocess max_diff | < <tbd> | <observed> | <observed> | ?/✅ |
| 2. Vision encoder concat max_diff | < <tbd> | <observed> | <observed> | ?/✅ |
| 3A. E2E logits max_diff | < <tbd> | <observed> | <observed> | ?/✅ |
| 3B. Greedy first-token | bit-identical | <observed> | <observed> | ?/✅ |
| 4a. image_0 key facts | ≥ 2/3 | <observed> | <observed> | ?/✅ |
| 4b. image_1 key facts | ≥ 2/3 | <observed> | <observed> | ?/✅ |
```

All 7 cells green → P6.6 done.

## 7. Risk + Rollback

**Biggest risk**: `cross_modal::replace_image_tokens` has only been tested with one contiguous image-token span. With 2 spans, the host-side scatter loop (lines 80-89) walks the input_ids once and increments `k` per match — algorithmically correct, but maps positions to `vision_embeds[k, :]` assuming vision_embeds rows are in the same order as image-token spans appear in input_ids. If the OpenAI handler's `pixel_values = concat(all_pixel_values)` interleaves multi-image patches in a different order than the chat-template's `<|image_pad|>` insertion order, output is silently wrong.

Mitigation: Gate 3B (greedy first-token bit-identical) is precisely the catch for this — even a single mis-routed patch would shift the LM head's argmax.

**Rollback strategy**: each fix commit is independent. If Gate 2 doesn't close after 3 iterations, capture the rupture point in a P6.6 close-out marked "PARTIAL" and either re-spec or escalate. Do **not** silently widen thresholds.

## 8. Estimated Effort

| Phase | Work | Estimate |
| --- | --- | --- |
| P6.6a | fixture download + 6 new tools + diagnose run + threshold-set | 3–4h |
| P6.6b | hypothesis-driven fixes (unknown count) | 2–6h |
| P6.6c | semantic check + close-out doc | 1–2h |
| **Total** | | **~6–12h** (1–1.5 working days) |

## 9. Out of Scope / Deferred

- N > 2 images stress (delayed until N=2 passes)
- Batched serving (B>1) — own P-track
- Video — own P-track
- Anthropic multi-image — currently 400 reject; future spec if customer demand
- Performance optimization of multi-image hot path — first make it work
