# P6.3 Vision Functional Correctness — Design

**Status:** Approved (brainstormed 2026-05-11)
**Owner:** ironmlx
**Parent:** P6.2 (commit `f2dfda3` on branch `ironmlx-p6-1-vision-diff`)
**Branch target:** new `ironmlx-p6-3-vision-correctness` branched from `ironmlx-p6-1-vision-diff`

## 1. Motivation

P6 (Qwen3.5 VL inference) reached numerical acceptance:
- Task 21 logits-match: max_diff = 0.5039, greedy first-token bit-identical
- Task 22 HTTP smoke test: server identifies a cat in the COCO sample

P6.2 fixed the P6.1 diff test driver's reshape bug, dropping `29_merger_out` max_diff from 8.08 to 0.90 when the same mlx-vlm pixel_values are fed to both vision towers.

However, **real-inference functional correctness is still poor**. Item 3 verification (3 distinct test images run through ironmlx vs mlx-vlm) showed:

- 5/5 test images misclassified by ironmlx as "side-by-side composite / stereoscopic 3D pair"
- Counting tasks off by 2× (network court group: ironmlx reports "20-25 per panel" vs mlx-vlm "13+1 = 14", ground truth ~12)
- Special-attribute recognition: ironmlx missed an upside-down STOP sign entirely; mlx-vlm correctly answered "rotated 180 degrees... reads POTS"
- COCO sample: ironmlx says "a tabby cat" (singular); mlx-vlm correctly says "two tabby cats"

P6.3's goal: drive ironmlx's real-inference quality to functional parity with mlx-vlm on the four test scenarios, while NOT pursuing performance optimization yet (that comes later).

## 2. Goals

Achieve all four acceptance gates below. Each gate is binary pass/fail with a specific quantitative threshold.

### Gate 1 — Preprocess consistency

Compare `ironmlx::models::qwen3_5::image_processor::preprocess(image)` output against `mlx-vlm processor(image)` output for the same input image.

- **Metric**: `max_abs_diff(pv_ironmlx, pv_mlxvlm)` after both are normalized to bf16 and reshaped to the same `[N, 1536]` C-major layout.
- **Threshold**: < 0.05 (bf16 quantization edge).
- **Why this gate**: production HTTP path uses ironmlx's own preprocess, so divergence here pollutes every downstream stage. P6.1/P6.2 diff numbers do NOT measure this — they feed mlx-vlm pixel_values into both towers.

### Gate 2 — Vision encoder consistency

Re-run P6.1 diff pipeline (feeds same mlx-vlm pixel_values into both vision towers, dumps 29 intermediate tensors).

- **Metric**: `29_merger_out` max_abs_diff from the latest diff report.
- **Threshold**: < 0.1 (P6.2 post-fix baseline is 0.9023; need a 9× drop).
- **Why this gate**: even with byte-identical input, blocks 15-23 still produce up to 1193 max_diff at block_23 (P6.2 report). LayerNorm at merger absorbs most of it but 0.90 remains. This implies ≥ 1 op-level implementation difference inside the ViT chain.

### Gate 3 — End-to-end logits consistency (production path)

Run ironmlx HTTP/integration path end-to-end with its own preprocess (NOT mlx-vlm's), forward_vl, and compare last-position logits against mlx-vlm completion on the same image + prompt + greedy temperature=0.

- **Metric A**: `max_abs_diff(logits_ironmlx, logits_mlxvlm)` on the last-position logit vector.
- **Threshold A**: < 0.5 (preserves Task 21 acceptance gate; current measurement = 0.5039).
- **Metric B**: greedy first-token equality.
- **Threshold B**: bit-identical (current: 760 == 760 ✅).

### Gate 4 — Semantic functional correctness

Four test images (already gathered):
- `coco_sample.jpg` (two tabby cats + remote controls on a pink sofa)
- `/tmp/p6vl_test_imgs/scene.jpg` (yellow-green living room with one person, dining table, kitchen)
- `/tmp/p6vl_test_imgs/counting.jpg` (group of ~12 children on a tennis court)
- `/tmp/p6vl_test_imgs/text.jpg` (upside-down STOP sign)

Run each through the ironmlx HTTP server with prompt "Describe this image in detail. If there are multiple people or objects, count them." and `temperature=0`, `enable_thinking=false`, `max_tokens=400`.

| # | Image | Key facts (mlx-vlm baseline) | ironmlx pass criteria |
|---|---|---|---|
| 4a | COCO 39769 | "two cats" / "green collar" / "remote" | ≥ 2/3 key facts present |
| 4b | scene | single continuous room (NOT "side-by-side", "composite", "stereoscopic", "duplicated") | no double-image misclassification |
| 4c | counting | "13" or "14" people | reported count in [10, 16] (±2 of ground truth ~12) |
| 4d | text | "upside down" or "rotated" or "POTS" | ≥ 1 inversion-related keyword in output |

- **Threshold**: 3/4 sub-gates pass.

## 3. Acceptance Report (P6.3 close-out artifact)

P6.3 is complete when this table is filled in with all gates green:

```
| Gate | Target | Before P6.3 | After P6.3 | Status |
| 1. Preprocess max_diff | < 0.05 | <tbd>    | <tbd> | ?/✅ |
| 2. Vision encoder max_diff | < 0.1 | 0.9023 | <tbd> | ?/✅ |
| 3A. E2E logits max_diff | < 0.5 | 0.5039 | <tbd> | ?/✅ |
| 3B. Greedy first-token | bit-identical | 760 ✅ | <tbd> | ?/✅ |
| 4a. COCO key facts | ≥ 2/3 | 1/3 (a cat) | <tbd> | ?/✅ |
| 4b. scene non-double | yes | no | <tbd> | ?/✅ |
| 4c. counting ±2 | in [10, 16] | 20-25 | <tbd> | ?/✅ |
| 4d. STOP inversion ≥1 keyword | yes | no | <tbd> | ?/✅ |
```

If 4 of 4 numerical gates green AND 3/4 semantic sub-gates green → P6.3 done.

If any gate stuck after reasonable effort, re-spec or escalate (do not lower thresholds without re-brainstorming).

## 4. Approach (3 sequenced sub-phases)

Each sub-phase is its own implementation effort with intermediate gates. The full plan will be written in a separate writing-plans pass after this spec is approved.

### P6.3a — Preprocess byte-level diff + alignment

Diagnostic-then-fix loop targeting Gate 1.

1. Extend P6.1 diff pipeline to support a "preprocess-only" mode: dump ironmlx `preprocess(image)` output as a 30th tensor (`-1_ironmlx_pv.safetensors`), pair against mlx-vlm's `00_pixel_values.safetensors`, report diff.
2. Locate root cause of any byte-level divergence: image decoder (jpeg-decoder vs PIL), resampling algorithm (Lanczos3 implementation differences), normalize formula application order, patchify byte layout.
3. Fix until Gate 1 is green.

### P6.3b — Op-level vision encoder diff (block 15-23 focus)

Diagnostic-then-fix loop targeting Gate 2.

1. Extend P6.1 dump points from 30 → ~120 by adding 4 intermediate dumps inside each `VitBlock`: `after_norm1`, `after_attn`, `after_norm2`, `after_mlp`. Mirror in mlx-vlm fork.
2. Run pipeline. Identify which sub-op inside blocks 15-23 first hits max_diff > 0.5 (current is 1193 at block_23).
3. Likely candidates (in order of suspicion):
   - SDPA rank-4 expand_dims/squeeze packaging artifacts
   - `apply_rotary_vision` split-half vs concat-half ordering differences
   - LayerNorm accumulator precision in `mlx::fast::layer_norm_on`
   - Float reduction order in matmul tiles (less likely but possible)
4. Fix per-op until Gate 2 is green.

### P6.3c — Semantic verification + Gate 3 (production path)

After Gate 1 + 2 are green:

1. Run Item 3 four-image test through the ironmlx HTTP server.
2. Run Gate 3 (end-to-end logits diff) — this validates that the production path improvements (Gate 1) compose with the vision encoder improvements (Gate 2).
3. Fill in P6.3 acceptance report table.
4. If any sub-gate fails, regress to P6.3a or P6.3b investigation.

## 5. Non-Goals

- No performance optimization (no self_qmm vision tower, no independent Metal kernels, no GPU-side preprocess). Those are deferred to P7+.
- No changes to mlx-vlm fork beyond extending dump hooks for op-level granularity.
- No changes to LM path (already verified within bounds per Task 21).
- No changes to checkpoint loading.
- No changes to chat template / tokenizer / cross-modal token routing logic.
- No alternative model targets (P6.3 is Qwen3.5-4B-MLX-4bit only).

## 6. Out of Scope (deferred to later P6.x or P7+)

- Multi-image batched VL inference.
- VL decoding optimization (currently slow, but functional first).
- Vision tower quantization (bf16 dense path stays — P7 candidate).

## 7. Risk + Rollback

**Risk**: Gate 2 might require deep numerical alignment work (LayerNorm accumulator, SDPA reduction order). If those are gated by mlx Rust crate limitations rather than ironmlx code, P6.3b could escalate to "upstream mlx Rust improvements" — outside ironmlx scope.

**Rollback**: Each sub-phase fix is its own commit. If P6.3b stalls, P6.3a + P6.3c can still ship — they give a partial improvement that may already lift semantic gates 4a-4d to passing without touching block-23 numerics.

## 8. Estimated effort

- P6.3a (preprocess diff + fix): 4-8 hours (depends on root cause depth)
- P6.3b (op-level diff + fix): 8-16 hours (most uncertain; 24 blocks × 4 sub-ops = 96 candidate sites, but likely 1-2 actual divergence sources)
- P6.3c (verification + close-out): 2 hours

Total: ~14-26 hours, 2-3 working days.

## 9. Files (anticipated touchpoints)

P6.3a:
- `ironmlx/tests/fixtures/p6_qwen35_vl/run_python_dump.py` (extend to also dump ironmlx-side preprocess output for comparison)
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_pipeline.py` (add preprocess pair handling)
- `ironmlx/src/models/qwen3_5/image_processor.rs` (likely fix candidates)

P6.3b:
- `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py` (extend hooks to 4 intra-block sites)
- `ironmlx/tests/fixtures/p6_qwen35_vl/mlx_vlm_patches/02_op_level_hooks.patch` (archive new patch)
- `ironmlx/src/models/qwen3_5/vision/mod.rs` (add 96 op-level dump_tensor calls)
- `ironmlx/src/models/qwen3_5/vision/block.rs` (likely fix candidates)

P6.3c:
- `ironmlx/tests/fixtures/p6_qwen35_vl/item3_semantic_check.sh` (new — drives 4-image curl test against running server)
- `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/p6_3_closeout/report.md` (new — close-out artifact)
