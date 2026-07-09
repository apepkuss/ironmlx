# Quant Validation Matrix

- Run directory: `docs/benchmarks/quant-validation/2026-07-09-170226`
- Overall status: `passed`
- Models: 4

## Matrix Summary

| model | category | PP | TG | C | requests | TTFT p50 ms | TTFT p95 ms | E2E p95 s | tok/s | ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MiniCPM5-1B-8bit | http_e2e | 128 | 16 | 1 | 2 | 29.162 | 29.644 | 0.098 | 235.350 | true |
| MiniCPM5-1B-8bit | http_e2e | 512 | 16 | 1 | 2 | 40.076 | 40.619 | 0.107 | 241.778 | true |
| MiniCPM5-1B-8bit | long_context | 8192 | 16 | 1 | 2 | 672.961 | 673.916 | 0.727 | 282.104 | true |
| MiniCPM5-1B-8bit | long_context | 32768 | 16 | 1 | 2 | 6176.875 | 6187.441 | 6.266 | 190.813 | true |
| MiniCPM5-1B-8bit | concurrent | 8192 | 16 | 4 | 43 | 2876.737 | 2904.680 | 2.958 | 22.100 | true |
| MiniCPM5-1B-8bit | concurrent | 32768 | 16 | 4 | 8 | 25937.136 | 26672.457 | 26.759 | 4.067 | true |
| MiniCPM5-1B-8bit | concurrent | 8192 | 16 | 8 | 43 | 6573.387 | 6644.482 | 6.702 | 21.933 | true |
| MiniCPM5-1B-8bit | concurrent | 32768 | 16 | 8 | 12 | 50968.310 | 58477.998 | 58.557 | 6.167 | true |
| MiniCPM5-1B-8bit | multi_turn |  |  | 1 | 3 |  |  | 0.091 |  | true |
| MiniCPM5-1B-8bit | stability |  |  | 1 | 5 |  |  | 0.084 |  | true |
| Qwen3.5-4B-MLX-8bit | http_e2e | 128 | 16 | 1 | 2 | 72.445 | 73.276 | 0.222 | 69.182 | true |
| Qwen3.5-4B-MLX-8bit | http_e2e | 512 | 16 | 1 | 2 | 147.986 | 149.163 | 0.354 | 77.620 | true |
| Qwen3.5-4B-MLX-8bit | long_context | 8192 | 16 | 1 | 2 | 2624.025 | 2624.217 | 2.797 | 92.594 | true |
| Qwen3.5-4B-MLX-8bit | long_context | 32768 | 16 | 1 | 2 | 13095.234 | 13116.120 | 13.255 | 74.737 | true |
| Qwen3.5-4B-MLX-8bit | concurrent | 8192 | 16 | 4 | 14 | 11574.552 | 11861.310 | 12.036 | 6.867 | true |
| Qwen3.5-4B-MLX-8bit | concurrent | 32768 | 16 | 4 | 6 | 58472.615 | 59179.688 | 59.314 | 2.000 | true |
| Qwen3.5-4B-MLX-8bit | concurrent | 8192 | 16 | 8 | 17 | 25900.196 | 26150.359 | 26.325 | 8.867 | true |
| Qwen3.5-4B-MLX-8bit | concurrent | 32768 | 16 | 8 | 9 | 76226.550 | 122792.211 | 122.928 | 3.000 | true |
| Qwen3.5-4B-MLX-8bit | multi_turn |  |  | 1 | 3 |  |  | 0.280 |  | true |
| Qwen3.5-4B-MLX-8bit | stability |  |  | 1 | 5 |  |  | 0.192 |  | true |
| gemma-4-e2b-it-bf16 | http_e2e | 128 | 16 | 1 | 2 | 116.093 | 116.227 | 0.858 | 21.587 | true |
| gemma-4-e2b-it-bf16 | http_e2e | 512 | 16 | 1 | 2 | 291.226 | 316.648 | 1.194 | 18.676 | true |
| gemma-4-e2b-it-bf16 | long_context | 8192 | 16 | 1 | 2 | 3681.442 | 3688.544 | 4.357 | 23.927 | true |
| gemma-4-e2b-it-bf16 | long_context | 32768 | 16 | 1 | 2 | 20582.658 | 20591.123 | 21.294 | 22.964 | true |
| gemma-4-e2b-it-bf16 | concurrent | 8192 | 16 | 4 | 10 | 16796.705 | 17221.761 | 17.890 | 5.333 | true |
| gemma-4-e2b-it-bf16 | concurrent | 32768 | 16 | 4 | 5 | 62829.694 | 84070.793 | 84.759 | 2.667 | true |
| gemma-4-e2b-it-bf16 | concurrent | 8192 | 16 | 8 | 14 | 34266.105 | 34331.021 | 34.996 | 7.467 | true |
| gemma-4-e2b-it-bf16 | concurrent | 32768 | 16 | 8 | 9 | 104253.761 | 167496.624 | 168.185 | 4.800 | true |
| gemma-4-e2b-it-bf16 | multi_turn |  |  | 1 | 3 |  |  | 1.005 |  | true |
| gemma-4-e2b-it-bf16 | stability |  |  | 1 | 5 |  |  | 0.624 |  | true |
| gemma-4-e4b-it-OptiQ-4bit | http_e2e | 128 | 16 | 1 | 2 | 83.906 | 84.661 | 0.325 | 66.753 | true |
| gemma-4-e4b-it-OptiQ-4bit | http_e2e | 512 | 16 | 1 | 2 | 242.557 | 249.613 | 0.524 | 58.643 | true |
| gemma-4-e4b-it-OptiQ-4bit | long_context | 8192 | 16 | 1 | 2 | 4590.985 | 4600.230 | 4.819 | 72.314 | true |
| gemma-4-e4b-it-OptiQ-4bit | long_context | 32768 | 16 | 1 | 2 | 24601.482 | 24843.685 | 25.081 | 46.323 | true |
| gemma-4-e4b-it-OptiQ-4bit | concurrent | 8192 | 16 | 4 | 10 | 19097.661 | 19263.738 | 19.480 | 5.333 | true |
| gemma-4-e4b-it-OptiQ-4bit | concurrent | 32768 | 16 | 4 | 5 | 74969.988 | 100079.964 | 100.323 | 1.833 | true |
| gemma-4-e4b-it-OptiQ-4bit | concurrent | 8192 | 16 | 8 | 14 | 38506.932 | 38798.910 | 39.007 | 7.467 | true |
| gemma-4-e4b-it-OptiQ-4bit | concurrent | 32768 | 16 | 8 | 9 | 125078.337 | 200296.770 | 200.538 | 3.300 | true |
| gemma-4-e4b-it-OptiQ-4bit | multi_turn |  |  | 1 | 3 |  |  | 0.332 |  | true |
| gemma-4-e4b-it-OptiQ-4bit | stability |  |  | 1 | 5 |  |  | 0.306 |  | true |

## Artifacts

- Manifest: `docs/benchmarks/quant-validation/2026-07-09-170226/manifest.json`
- CSV summary: `docs/benchmarks/quant-validation/2026-07-09-170226/summary.csv`
- `MiniCPM5-1B-8bit`: `docs/benchmarks/quant-validation/2026-07-09-170226/MiniCPM5-1B-8bit`
- `Qwen3.5-4B-MLX-8bit`: `docs/benchmarks/quant-validation/2026-07-09-170226/Qwen3.5-4B-MLX-8bit`
- `gemma-4-e2b-it-bf16`: `docs/benchmarks/quant-validation/2026-07-09-170226/gemma-4-e2b-it-bf16`
- `gemma-4-e4b-it-OptiQ-4bit`: `docs/benchmarks/quant-validation/2026-07-09-170226/gemma-4-e4b-it-OptiQ-4bit`
