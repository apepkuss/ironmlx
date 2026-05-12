# P6.6 Multi-Image-Per-Request — Close-out

**Branch:** `ironmlx-p6-6-multi-image` (off `ironmlx-p6-4-cleanup` commit `fcde351`)
**Date:** 2026-05-11
**Spec:** `docs/superpowers/specs/2026-05-11-p6-6-multi-image-design.md` (commit `ee3aef7`)
**Plan:** `docs/superpowers/plans/2026-05-11-p6-6-multi-image.md` (commit `e66fd33`)

## Summary

Verified that 1 chat request carrying 2 image_url parts produces a numerically
faithful and semantically correct response against the mlx-vlm reference.
**All 7 acceptance cells green.** Zero ironmlx source changes were required —
the multi-image data structures already in place (since P6) carried through
end-to-end on the first run. The only Gate 1 elevation traced to a deliberate
algorithm-choice differential (ironmlx Lanczos3 vs mlx-vlm PIL BICUBIC),
which is by design and absorbed by the vision encoder before reaching the LM.

## Acceptance Table

| Gate | Target | Baseline (diagnose) | Final | Status |
| --- | --- | --- | --- | --- |
| 1A. image_0 preprocess max_diff | < 0.20 | 0.1113 | 0.1113 | ✅ |
| 1B. image_1 preprocess max_diff | < 0.20 | 0.1963 | 0.1963 | ✅ |
| 2. Vision encoder concat max_diff | < 0.10 | 0.0625 | 0.0625 | ✅ |
| 3A. E2E logits max_diff | < 0.95 | 0.9004 | 0.9004 | ✅ |
| 3B. Greedy first-token | bit-identical | 760 == 760 | 760 == 760 | ✅ |
| 4a. image_0 key facts | ≥ 2/3 | 3/3 | 3/3 | ✅ |
| 4b. image_1 key facts | ≥ 2/3 | 3/3 | 3/3 | ✅ |

## Thresholds vs P6.3 anchor

| Gate | P6.3 single-image | P6.6 baseline | Multiplier | Threshold | Rationale |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0254 | 0.1113 / 0.1963 | 4.38× / 7.73× | 0.20 | Lanczos3/BICUBIC algorithm differential — Boss-approved override (see §Gate 1 decision) |
| 2 | 0.0330 | 0.0625 | 1.89× | 0.10 | ceil(0.0625/0.05)·0.05 per spec §2 rule |
| 3A | 0.3906 | 0.9004 | 2.30× | 0.95 | ceil(0.9004/0.05)·0.05 per spec §2 rule |
| 3B | bit-id | bit-id | — | bit-id | hard gate (unchanged) |

## Gate 1 decision — Lanczos3 vs BICUBIC (Boss-approved 2026-05-11)

Diagnostic finding (Task 8.1): mlx-vlm uses `PIL.Image.resize(resample=Image.BICUBIC)`
(`mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py:73`); ironmlx uses
`image::imageops::FilterType::Lanczos3` (`ironmlx/src/models/qwen3_5/image_processor.rs:157`).
The diff distribution (37–38% pixels > 1e-3) exactly matches
PIL-BICUBIC-vs-PIL-LANCZOS on the same fixtures. P6.3's 0.0254 baseline was on
COCO 397133 whose source dims equal the smart_resize target — ironmlx hit the
identity-skip path and never invoked Lanczos3, leaving only JPEG-decoder
variance (~1.4%). The 1.5×/5× anchor rule in spec §2 was calibrated against
that low-noise baseline; it does not apply to fixtures that actually resize.

**Decision:** keep Lanczos3 (higher reconstruction quality, aligns with
ironmlx design philosophy of independent best-design rather than competitor
parity). Widen Gate 1 threshold from the default per-baseline `ceil/0.05`
(which would give 0.15/0.20) to a single shared **0.20** ceiling documenting
the algorithm frontier — explicitly **not** a code bug.

Gate 2 (0.0625 < 0.10) confirms the vision encoder absorbs the resize-algorithm
differential, and Gate 3B (bit-identical first token) confirms the LM is
unaffected.

## Fixes Applied

**Zero source changes.** All commits are tooling / fixture / documentation:

| Commit | Type | Description |
| --- | --- | --- |
| `49b7aef` | fixture | 2 COCO val2017 images (image_0 kitchen, image_1 NYC street) + .gitignore |
| `a71d41e` | tool | `run_p6_6_dump.py` — mlx-vlm 2-image fixture driver |
| `fd0571d` | tool | `tests/p6_6_multi_image_dump.rs` — ironmlx vision-dump integration test |
| `0bf2506` | tool | `diff_preprocess_multi.py` + `diff_pipeline_multi.py` |
| `50784f4` | test | `tests/p6_6_logits_match.rs` — e2e Gate 3 integration test |
| `69d6b5a` | tool | `run_p6_6_diff.sh` orchestrator + baseline reports + thresholds.md |
| `e860a0f` | docs | Gate 1 root-cause investigation + threshold-widen decision |
| `9636f3d` | test | `p6_6_semantic_check.py` Gate 4 verification |

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean (only unchanged mlx-sys C++ warnings) |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release -- --test-threads=1` | **153 passed / 0 failed** (unchanged from P6.4 baseline) |
| P6.3 Task 21 single-image logits-match | **PASS** — max_diff=0.3906, first_token=760 (identical to P6.3 close-out) |

## Notes

- The Gate 3A 2.30× elevation (0.9004 vs P6.3 0.3906) almost certainly stems
  from the same Lanczos3/BICUBIC differential propagating through the LM
  decoder. Gate 3B's bit-identical token confirms argmax is unaffected.
- `count > 1e-3` percentages on Gate 1 (37–38%) are pixel-count statistics,
  not failure signals — `mean` of 0.0045/0.0061 indicates the bulk of pixels
  are essentially identical and only a long tail of high-frequency-content
  pixels (image_1 NYC street's construction scaffolding) drives `max`.
- N=2 is the verified scope (per spec §3 non-goals). The loop-based code in
  `VisionTower::forward`, `build_position_ids_vl`, and
  `cross_modal::replace_image_tokens` walks `grid_thw` and image-token spans
  in order — extends to N=3+ without code change but is not explicitly
  validated here.

## P6.7+ candidates

- **N > 2 image stress** — trivially extend `run_p6_6_dump.py` to N=3,4
- **Batched serving (B>1)** — separate P-track (audit B1 phase 2)
- **Video temporal_patch_size > 2** — audit B2 + B3
- **Anthropic multi-image** — currently 400-rejected (P6.5 B6); brainstorm
  on demand
- **Optional: align resize to BICUBIC** — measurable Gate 1 closure to 0.05
  if cross-implementation parity ever outweighs quality preference

## Linked Reports

- Preprocess (Gate 1): `diff_reports/p6_6-2026-05-11-2249/p6_6_preprocess_report.md`
- Vision encoder (Gate 2): `diff_reports/p6_6-2026-05-11-2249/vision/report.md`
- E2E logits (Gate 3): `diff_reports/p6_6-2026-05-11-2249/p6_6_logits_match.log`
- Thresholds + decision: `diff_reports/p6_6-2026-05-11-2249/thresholds.md`
- Semantic (Gate 4): `diff_reports/p6_6c-2026-05-11-2319/p6_6_semantic_report.md`

---

## Addendum: N=3 stress (2026-05-12)

Verified that the loop-based multi-image code paths in `VisionTower::forward`,
`build_position_ids_vl`, and `cross_modal::replace_image_tokens` extend
correctly from N=2 to N=3 with no source changes. Driven by parameterised
versions of the existing P6.6 tooling (`run_p6_6_dump.py --images`,
`diff_preprocess_multi.py --n-images`, orchestrator `N_IMAGES=3`,
`p6_6_semantic_check.py --n-images 3`).

## Fixture

Third image: COCO val2017 60347 — man seated on a wooden bench in a sunlit
forest. Deliberately picks a third semantic quadrant (vs. image_0 kitchen
interior, image_1 NYC urban street). Source dims 480×640 → smart_resize
target matches → identity-resize path; Lanczos3 not invoked.

## Acceptance Table (N=3)

| Gate | Threshold | N=3 Final | N=2 Final | Status |
| --- | --- | --- | --- | --- |
| 1A. image_0 preprocess max_diff | < 0.20 | 0.1113 | 0.1113 | ✅ |
| 1B. image_1 preprocess max_diff | < 0.20 | 0.1963 | 0.1963 | ✅ |
| 1C. image_2 preprocess max_diff | < 0.20 | **0.0239** | — | ✅ |
| 2. Vision encoder concat max_diff | < 0.10 | **0.0625** | 0.0625 | ✅ |
| 3A. E2E logits max_diff | < **1.20** | **1.1250** | 0.9004 | ✅ |
| 3B. Greedy first-token | bit-identical | 760 == 760 | 760 == 760 | ✅ |
| 4a. image_0 key facts | ≥ 2/3 | 3/3 | 3/3 | ✅ |
| 4b. image_1 key facts | ≥ 2/3 | 3/3 | 3/3 | ✅ |
| 4c. image_2 key facts | ≥ 2/3 | 3/3 | — | ✅ |

## Threshold update — Gate 3A widened 0.95 → 1.20

Boss-approved 2026-05-12. Rule: ceil(1.125/0.05)·0.05 = 1.15, plus safety
margin → 1.20. Covers N ∈ {2, 3} on this fixture. The N=3 result superseded
the N=2 threshold of 0.95 because N=3 is the larger validated configuration
and the elevation is bf16-numerical, not logical (see §Gate 3A scaling
investigation below).

## Two structural confirmations

### Gate 1C — reverse-confirms the Gate 1 root cause

image_2 (480×640) does NOT need resize after smart_resize, so ironmlx skips
Lanczos3 entirely. Result: max_diff = 0.0239 — within the JPEG-decoder
variance floor (P6.3 single-image was 0.0254). This independently confirms
the Task 8 root-cause finding: Gate 1A/1B elevation is the Lanczos3-vs-BICUBIC
resize differential, not a multi-image code path bug.

### Gate 2 — multi-image vision encoder is O(1) in N

Gate 2 max_diff is **0.0625 for both N=2 and N=3** — bit-identical. ViT
multi-grid path (`cu_seqlens` construction, per-image rotary, per-image
`add_learned_pos_embed`) is verified stable: no per-N drift. The 24-layer
encoder absorbs upstream preprocess differential to a fixed point.

## Gate 3A scaling investigation

Gate 3A grew from 0.39 (P6.3 N=1) → 0.90 (N=2) → 1.125 (N=3). Sub-linear
in vision-token count (1040 / 2080 / 3280 patches).

### Signed-diff distribution (N=3, last-position logits, 248320 vocab)

| Stat | Value |
| --- | --- |
| max_abs_diff | 1.1250 |
| signed mean | -0.0856 |
| signed median | -0.0820 |
| abs(diff) > 0.5 count | 8863 / 248320 (3.57%) |
| abs(diff) > 1.0 count | 16 / 248320 (0.0064%) |
| residual_max (after mean subtraction) | 1.0851 |

### Top-5 outliers (N=3)

| logit idx | ironmlx | mlx-vlm | diff |
| --- | --- | --- | --- |
| 18257 | -2.6719 | -1.5469 | -1.1250 |
| 60025 | 1.8594 | 2.9844 | -1.1250 |
| 96445 | -5.4062 | -4.2812 | -1.1250 |
| 9825 | -2.9688 | -1.8672 | -1.1016 |
| 110112 | -2.4688 | -1.3672 | -1.1016 |

### Verdict — bf16 numerical, not a code bug

Evidence:

1. `residual_max (1.085) ≈ max_diff (1.125)` — the small mean offset
   contributes almost nothing; this is true scatter, not systematic shift.
2. Top-5 outliers cluster at diff = -1.1250 across 3 unrelated token ids =
   bf16 mantissa-grid step (0.0625 × 18). Consistent with LM-head quant
   path landing on the same quantisation bin.
3. Gate 3B (greedy first-token bit-identical) and Gate 4 (9/9 keys with
   detailed accurate descriptions of all three images) confirm output
   correctness end-to-end. If this were a multi-image data-flow bug it
   would surface as either an argmax flip or a degraded semantic response.
4. Scaling is sub-linear (vision-token count: 2.0×→3.15× from N=2 to N=3;
   max_diff: 1.25×). Linear-cumulative numerical drift through 28 LM layers
   processing more vision tokens is the expected signature.

No source change attempted; tightening Gate 3A would require an LM-side
dtype audit (cross_modal scatter, LM-head fp32 vs bf16 path) for diminishing
return on output correctness. P6.7+ candidate if customer demand surfaces.

## Regression Status (after N=3 addendum)

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo test -p ironmlx --lib --release -- --test-threads=1` | **153 passed / 0 failed** |
| P6.3 Task 21 single-image logits-match | **PASS** — max_diff=0.3906, first_token=760 |
| P6.6 N=2 (re-run not required; no source touched) | unchanged |

## Linked Reports (N=3)

- Preprocess (Gate 1, N=3): `diff_reports/p6_6_n3-2026-05-12-1034/p6_6_preprocess_report.md`
- Vision encoder (Gate 2, N=3): `diff_reports/p6_6_n3-2026-05-12-1034/vision/report.md`
- E2E logits + distribution (Gate 3, N=3): `diff_reports/p6_6_n3-2026-05-12-1034/p6_6_logits_match.log`
- Semantic (Gate 4, N=3): `diff_reports/p6_6c_n3-2026-05-12-1053/p6_6_semantic_report.md`
