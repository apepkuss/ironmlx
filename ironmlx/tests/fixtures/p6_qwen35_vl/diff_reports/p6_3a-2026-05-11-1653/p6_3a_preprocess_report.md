# P6.3a Preprocess Diff (Gate 1)

- Tensor shape: [1200, 1536]
- Gate 1 threshold: < 0.05
- Observed max_diff: **0.0254**
- Gate 1 verdict: **PASS**

## Stats

- max: 0.025391
- mean: 0.000125
- p99: 0.007812
- count > 1e-3: 25186 / 1843200
- count > 1e-2: 4042 / 1843200

## Top 5 outliers

| flat_idx | vlm | iron | abs_diff |
| --- | --- | --- | --- |
| 1726054 | 0.4980 | 0.5234 | 0.0254 |
| 1726310 | 0.4980 | 0.5234 | 0.0254 |
| 415781 | 0.2715 | 0.2471 | 0.0244 |
| 416037 | 0.2715 | 0.2471 | 0.0244 |
| 480438 | 0.0510 | 0.0747 | 0.0237 |
