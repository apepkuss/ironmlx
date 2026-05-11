# P6.6 baseline + threshold decisions

Anchor: P6.3 single-image (Gate 1 0.0254, Gate 2 0.0330, Gate 3A 0.3906, Gate 3B bit-identical)

| Gate | Baseline | Multiplier (vs P6.3) | Threshold | Notes |
| --- | --- | --- | --- | --- |
| 1A. image_0 preprocess max_diff | 0.111328 | 4.38× | 0.20 | resize-algorithm differential (see §Gate 1 root cause) |
| 1B. image_1 preprocess max_diff | 0.196289 | 7.73× | 0.20 | resize-algorithm differential (see §Gate 1 root cause) |
| 2. Vision encoder concat max_diff | 0.062500 | 1.89× | 0.10 | 1.5×<1.89×<5× → ceil(0.0625/0.05)*0.05=0.10; PASS |
| 3A. E2E logits max_diff | 0.900400 | 2.30× | 0.95 | 1.5×<2.30×<5× → ceil(0.9004/0.05)*0.05=0.95; PASS |
| 3B. Greedy first-token | 760 == 760 | — | bit-identical | hard gate; PASS |

## Verdict

All 5 gates PASS at the chosen thresholds. **No code fix needed.**

## Gate 1 root cause — resize-algorithm differential (not a code bug)

Investigation (Task 8, hypothesis-test 1) traced Gate 1A/1B to a deliberate
algorithm choice mismatch:

- mlx-vlm: `PIL.Image.resize(..., resample=Image.BICUBIC)`
  (`mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py:73`)
- ironmlx: `image::imageops::FilterType::Lanczos3`
  (`ironmlx/src/models/qwen3_5/image_processor.rs:157`)

Evidence:

| Comparison | max uint8 diff | count > 1e-3 |
| --- | --- | --- |
| PIL BICUBIC vs PIL LANCZOS, image_0 427→416 | 14 | 36.7% |
| PIL BICUBIC vs PIL LANCZOS, image_1 428→416 | 25 | 37.6% |
| **Gate 1A actual (image_0)** | — | **37.2%** |
| **Gate 1B actual (image_1)** | — | **38.1%** |
| PIL LANCZOS vs Rust Lanczos3 | 3 | 1.7% |
| COCO P6.3 (identity resize, decode-only) | 3 | 1.4% |

The distribution lines up with the BICUBIC-vs-LANCZOS distribution exactly.
Image-content sensitivity (image_1 high-frequency NYC street content >
image_0 kitchen) explains image_1's larger diff (0.196 vs 0.111).

P6.3 single-image (COCO 397133) coincidentally had source == target dims, so
ironmlx hit the identity-skip path and never invoked Lanczos3, leaving only
JPEG-decoder variance (~1.4%). The 1.5×/5× anchor rule in spec §2 was
calibrated against that low-noise baseline and does not apply to multi-image
fixtures that actually resize.

## Design decision (Boss approval, 2026-05-11)

**Keep Lanczos3.** Aligns with ironmlx design philosophy ("不对齐任何竞品" —
parity with competing implementations is not a goal; quality is). Lanczos3
produces sharper resampling than BICUBIC. Gate 2 (0.0625 < 0.10) confirms
the vision encoder absorbs the resize-algorithm differential, and Gate 3B
(bit-identical first token) confirms the LM head is unaffected.

Gate 1 threshold widened from the default `ceil(baseline/0.05)*0.05` rule
(which would give 0.15/0.20 per image) to a single shared **0.20** ceiling
documenting the algorithm-choice frontier — not a code bug.

Gate 1B's 7.73× multiplier in the original ≥5× "BUG" rule is **overridden**
by this explicit decision; the override is recorded here and in the
close-out report.

## Re-run guidance

Since no source was modified, the baseline numbers above are the final
numbers. No re-run is required before Task 9 (Gate 4 semantic check).
