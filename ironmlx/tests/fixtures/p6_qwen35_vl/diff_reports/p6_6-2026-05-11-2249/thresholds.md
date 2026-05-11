# P6.6 baseline + threshold decisions

Anchor: P6.3 single-image (Gate 1 0.0254, Gate 2 0.0330, Gate 3A 0.3906, Gate 3B bit-identical)

| Gate | Baseline | Multiplier (vs P6.3) | Threshold | Notes |
| --- | --- | --- | --- | --- |
| 1A. image_0 preprocess max_diff | 0.111328 | 4.38× | 0.15 | 1.5×<4.38×<5× → ceil(0.111328/0.05)*0.05=0.15; FAIL at 0.05 gate but within 5× anchor |
| 1B. image_1 preprocess max_diff | 0.196289 | 7.73× | BUG | ≥5× P6.3 anchor → flag as bug; do NOT widen; defer to Task 8 |
| 2. Vision encoder concat max_diff | 0.062500 | 1.89× | 0.10 | 1.5×<1.89×<5× → ceil(0.0625/0.05)*0.05=0.10; already PASS at 0.1 gate |
| 3A. E2E logits max_diff | 0.900400 | 2.30× | 0.95 | 1.5×<2.30×<5× → ceil(0.9004/0.05)*0.05=0.95 |
| 3B. Greedy first-token | PASS | — | bit-identical | hard gate; 760==760 ✓ |

## Verdict
- Gates passing baseline: 2, 3A, 3B
- Gates failing baseline (need Task 8 fix loop): 1A (4.38× — borderline, 0.111328 > 0.05 original gate), 1B (7.73× — exceeds 5× threshold rule, flagged as bug)

## Notes on Gate 1 failures

Gate 1A/1B measure per-image preprocess (pixel_values before vision encoder).
The multi-image path uses the same preprocess code as single-image P6.3, but max_diff is significantly
higher (4.38× and 7.73× respectively). Root cause investigation needed in Task 8.
The vision encoder (Gate 2) absorbs the preprocess noise and produces correct embeddings (0.0625 < 0.10),
and the end-to-end logits (Gate 3A 0.9004) and greedy token (Gate 3B PASS) are still correct.
This suggests the preprocess diff is a known float32 vs bfloat16 representation issue
that gets normalized in subsequent layers, but must still be investigated.
