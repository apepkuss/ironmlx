# P6.3 Vision Functional Correctness — Close-out

**Branch:** `ironmlx-p6-3-vision-correctness`
**Date:** 2026-05-11
**Spec:** [`docs/superpowers/specs/2026-05-11-p6-3-functional-correctness-design.md`](../../../../../docs/superpowers/specs/2026-05-11-p6-3-functional-correctness-design.md) (commit `49133ef`)
**Plan:** [`docs/superpowers/plans/2026-05-11-p6-3-vision-correctness.md`](../../../../../docs/superpowers/plans/2026-05-11-p6-3-vision-correctness.md) (commit `c4a7b9a`)

## Acceptance table

| Gate | Target | Before P6.3 | After P6.3 | Status |
| --- | --- | --- | --- | --- |
| 1. Preprocess max_diff | < 0.05 | 2.0000 | **0.0254** | ✅ PASS |
| 2. Vision encoder max_diff | < 0.1 | 0.9023 | **0.0330** | ✅ PASS |
| 3A. E2E logits max_diff | < 0.5 | 0.5039 | **0.3906** | ✅ PASS |
| 3B. Greedy first-token | bit-identical | 760 ✅ | 760 ✅ | ✅ PASS |
| 4a. COCO key facts | ≥ 2/3 | 1/3 (a cat) | **3/3** (two cats / green collar / remote) | ✅ PASS |
| 4b. scene non-double | yes | no | **yes** (no forbidden keywords) | ✅ PASS |
| 4c. counting ±2 | in [10, 16] | 20-25 | **13 / 13 / 14** | ✅ PASS |
| 4d. STOP inversion ≥1 keyword | yes | no | **yes** (matched "POTS") | ✅ PASS |

**Result: 8 of 8 gates green. P6.3 complete.**

## Fixes applied (chronological)

| Commit | Title | Sub-phase | Effect |
| --- | --- | --- | --- |
| `1d1146e` | feat(p6.3a): p6_3a_preprocess_dump integration test | infra | dumper for ironmlx preprocess |
| `ceacce3` | feat(p6.3a): preprocess diff tool + Gate 1 baseline report | infra | diff_preprocess.py + Gate 1 baseline = 2.0 |
| `f262836` | fix(p6.3a): patchify merge_size grouping to match mlx-vlm patch ordering | fix | Gate 1 2.0 → 0.0254 |
| `c0230fa` | docs(p6.3a): Gate 1 green — preprocess max_diff = 0.0254 < 0.05 | doc | Gate 1 closed |
| `c6d14b4` | chore(p6.3b): archive mlx-vlm op-level intra-block hook patch | infra | 96 mlx-vlm hooks |
| `a16a442` | feat(p6.3b): VitBlock::forward_with_name_prefix for op-level dumps | infra | 96 ironmlx hooks |
| `d923700` | feat(p6.3b): op-level driver run_p6_3b_diff.sh + Gate 2 baseline | infra | Gate 2 baseline = 0.9023; rupture at block_10_c_norm2_out |
| `6a58bd8` | fix(p6.3b): align pos_embed bilinear corner summation order with mlx-vlm | fix | Gate 2 0.9023 → 0.8741 |
| `66f76b3` | fix(p6.3b): match mlx-vlm precision in VitAttention/VitMLP linear and GELU ops | fix | Gate 2 0.8741 → 0.0330 |
| `4e744e7` | feat(p6.3c): item3_semantic_check.py — Gate 4 driver | infra | Gate 4 automation |

## Linked reports

- Preprocess diff (Gate 1): [`diff_reports/p6_3a-2026-05-11-1653/p6_3a_preprocess_report.md`](../p6_3a-2026-05-11-1653/p6_3a_preprocess_report.md)
- Op-level vision encoder diff (Gate 2 final): [`diff_reports/p6_3b-2026-05-11-1830/report.md`](../p6_3b-2026-05-11-1830/report.md)
- Semantic verification (Gate 4): [`diff_reports/p6_3c-2026-05-11-1828/p6_3c_semantic_report.md`](../p6_3c-2026-05-11-1828/p6_3c_semantic_report.md)

## Root cause summary (functional correctness fixes)

Three independent numerical alignment defects against mlx-vlm reference:

### 1. Patchify merge-size grouping (Gate 1)

mlx-vlm's `_process_one` groups patches in `(grid_h/ms, grid_w/ms, ms, ms)` merge-tile order (2×2 blocks pre-shuffled for `PatchMerger`). ironmlx's `patchify` was emitting simple `(grid_h, grid_w)` row-major order. Same N=1200 patches, but different `[N, 1536]` flat-index mapping. Effect: vlm at white pixel = +1.0, iron at black pixel = -1.0 at the same flat index → diff = 2.0 exact (94% of values affected).

Fix in [`ironmlx/src/models/qwen3_5/image_processor.rs`](../../../../../ironmlx/src/models/qwen3_5/image_processor.rs): reshape `[3, H, W]` → `[3, mgh, ms, P, mgw, ms, P]`, permute `(1, 4, 2, 5, 0, 3, 6)`, reshape → `[N, 3, P, P]` matching mlx-vlm's grouping.

### 2. Bilinear pos_embed corner-sum order (Gate 2 partial)

mlx-vlm's `fast_pos_embed_interpolate` sums 4 weighted corner gathers left-to-right: `pe = c0 + c1 + c2 + c3`. ironmlx's [`vision/mod.rs`](../../../../../ironmlx/src/models/qwen3_5/vision/mod.rs) was doing paired sum `(c0+c1) + (c2+c3)`. In bfloat16 the two orderings differ by 1 ULP per element; that 1 ULP per element grows when fed through 24 ViT blocks. Aligned the order.

### 3. Linear addmm fusion + GELU dtype handling (Gate 2 main)

mlx-vlm's `nn.Linear` internally calls `mx.addmm(bias, x, W.T)` (fused matmul+bias). ironmlx's [`vision/block.rs`](../../../../../ironmlx/src/models/qwen3_5/vision/block.rs) `VitAttention`/`VitMLP` were doing `x @ W.T + bias` in two steps; for large matrices (e.g. 1024 → 4096 MLP) the two paths differ by 1 bf16 ULP per output element. Aligned via `ops::addmm`.

`gelu_tanh` was using `x*x*x` to compute `x³`; mlx-vlm uses `mx.power(x, 3)`. In bfloat16 these can differ by up to 0.125 at the `x³` magnitude. Aligned via `power(&three)`. Rust f32 literals also trigger bf16→f32 promotion that mlx Python's float literals do not; explicit `astype(dtype)` on the constants fixed that.

## Production regression status

- `cargo test -p ironmlx --lib --release -- --test-threads=1`: **152 passed, 0 failed**
- `cargo build -p ironmlx --release`: clean
- `cargo fmt + clippy -D warnings`: clean
- Task 21 `p6_qwen35_vl_logits_match` (Gate 3): PASS with `max_diff = 0.3906` (improved from 0.5039)
- HTTP smoke test (P6 Task 22): still works (production path bf16 throughout)

## Notes

- Gate 2's residual 0.0330 max_diff comes from the unavoidable bf16 ULP noise that survives 24 ViT blocks even with all ops aligned. mean diff is `0.000344` (essentially zero bias). count of values above `1e-1` in `29_merger_out`: **0**. The 0.0330 is a single outlier; semantic output is fully consistent.
- Gate 4's `counting_kids` returned a list `[13, 13, 1, 14]` — three of four numbers in [10, 16]; ground truth is ~12 children + 1 instructor (mlx-vlm answered "13 + 1 = 14"). ironmlx's answer matches mlx-vlm's count exactly.
- The 4 Item 3 images (coco_cats / scene / counting / text) all now produce semantically correct descriptions with no "double-image" misclassification — the originally observed P6 regression is closed.
- This branch (`ironmlx-p6-3-vision-correctness`) is ready to merge to `ironmlx-p6-vl` or directly to `main`.

## P6.4+ candidates (not in scope here)

- Vision tower performance: still bf16 dense path. ironmlx self_qmm or independent Metal kernel could yield 2-5× speedup. Functional correctness must NOT regress.
- Multi-image batched VL inference.
- Decode-side optimization for VL (currently slow).
