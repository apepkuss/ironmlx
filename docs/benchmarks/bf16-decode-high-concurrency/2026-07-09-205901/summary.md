# bf16 Decode / High-Concurrency Matrix

- Run directory: `docs/benchmarks/bf16-decode-high-concurrency/2026-07-09-205901`
- Models: Qwen3.5-4B-MLX-4bit / 8bit / bf16
- Server: `--prefill-chunk-size 2048 --max-cache-cap 65536`; sequential uses `--max-sequences 1`; concurrent uses `--max-sequences C`.
- Sequential cells: `runs=2`, `warmup=1`; concurrent cells: `duration=30s`, `warmup-duration=5s`.

## Key Cells

| cell | model | n | TTFT p50/med ms | TTFT p95 ms | E2E p95 s | decode/agg tok/s | ITL/TPOT ms | finish |
|---|---|---:|---:|---:|---:|---:|---:|---|
| seq_pp2048_tg512_c1 | Qwen3.5-4B-MLX-4bit | 2 | 460.317 | 461.213 | 2.852 | 142.428 | 7.045 | stop×2 |
| seq_pp2048_tg512_c1 | Qwen3.5-4B-MLX-8bit | 2 | 975.652 | 988.031 | 3.363 | 89.357 | 11.260 | stop×2 |
| seq_pp2048_tg512_c1 | Qwen3.5-4B-MLX-bf16 | 2 | 720.775 | 891.890 | 5.116 | 51.403 | 19.547 | stop×2 |
| seq_pp8192_tg512_c1 | Qwen3.5-4B-MLX-4bit | 2 | 1865.946 | 1868.537 | 3.138 | 136.086 | 7.422 | stop×2 |
| seq_pp8192_tg512_c1 | Qwen3.5-4B-MLX-8bit | 2 | 3122.118 | 3221.368 | 6.229 | 85.626 | 11.726 | stop×2 |
| seq_pp8192_tg512_c1 | Qwen3.5-4B-MLX-bf16 | 2 | 2503.175 | 2616.979 | 4.565 | 54.173 | 19.569 | stop×2 |
| seq_pp32768_tg128_c1 | Qwen3.5-4B-MLX-4bit | 2 | 10137.051 | 10289.501 | 11.421 | 112.933 | 8.925 | length×2 |
| seq_pp32768_tg128_c1 | Qwen3.5-4B-MLX-8bit | 2 | 13586.170 | 13612.331 | 13.742 | 77.542 | 14.329 | stop×2 |
| seq_pp32768_tg128_c1 | Qwen3.5-4B-MLX-bf16 | 2 | 11934.596 | 11953.046 | 12.146 | 51.630 | 21.521 | stop×2 |
| seq_pp32768_tg512_c1 | Qwen3.5-4B-MLX-4bit | 2 | 10337.544 | 10461.056 | 13.219 | 111.373 | 9.007 | stop×2 |
| seq_pp32768_tg512_c1 | Qwen3.5-4B-MLX-8bit | 2 | 13158.462 | 13174.413 | 13.302 | 77.894 | 14.265 | stop×2 |
| seq_pp32768_tg512_c1 | Qwen3.5-4B-MLX-bf16 | 2 | 11935.132 | 12011.571 | 12.203 | 52.154 | 21.305 | stop×2 |
| conc_pp2048_tg512_c8 | Qwen3.5-4B-MLX-4bit | 17 | 21107.673 | 24574.494 | 25.943 | 173.400 | 7.516 | stop=17 |
| conc_pp2048_tg512_c8 | Qwen3.5-4B-MLX-8bit | 18 | 19433.179 | 20871.677 | 22.829 | 98.500 | 11.263 | stop=18 |
| conc_pp2048_tg512_c8 | Qwen3.5-4B-MLX-bf16 | 14 | 28387.780 | 32752.397 | 36.160 | 80.900 | 19.575 | stop=14 |
| conc_pp8192_tg512_c8 | Qwen3.5-4B-MLX-4bit | 15 | 30779.822 | 32571.956 | 33.760 | 83.667 | 7.707 | stop=15 |
| conc_pp8192_tg512_c8 | Qwen3.5-4B-MLX-8bit | 15 | 29031.406 | 31183.102 | 32.239 | 39.667 | 11.501 | stop=15 |
| conc_pp8192_tg512_c8 | Qwen3.5-4B-MLX-bf16 | 13 | 32920.865 | 40446.762 | 42.234 | 55.200 | 19.695 | stop=13 |
| conc_pp32768_tg128_c8 | Qwen3.5-4B-MLX-4bit | 10 | 86095.032 | 115948.151 | 117.148 | 40.600 | 9.428 | length=8,stop=2 |
| conc_pp32768_tg128_c8 | Qwen3.5-4B-MLX-8bit | 10 | 81738.155 | 108993.826 | 109.124 | 3.333 | 14.458 | stop=10 |
| conc_pp32768_tg128_c8 | Qwen3.5-4B-MLX-bf16 | 10 | 70455.035 | 93999.489 | 94.191 | 3.333 | 21.245 | stop=10 |
| conc_pp32768_tg512_c8 | Qwen3.5-4B-MLX-4bit | 9 | 81586.799 | 132553.043 | 136.000 | 94.967 | 10.108 | length=1,stop=8 |
| conc_pp32768_tg512_c8 | Qwen3.5-4B-MLX-8bit | 10 | 82037.718 | 109564.660 | 109.695 | 3.333 | 14.460 | stop=10 |
| conc_pp32768_tg512_c8 | Qwen3.5-4B-MLX-bf16 | 10 | 71193.844 | 94900.948 | 95.091 | 3.333 | 21.348 | stop=10 |

## Initial Interpretation

- `bf16` decode cost is consistently higher: sequential TPOT is about 19.5-21.5 ms, compared with 4bit about 7.0-9.0 ms and 8bit about 11.3-14.3 ms.
- At `PP=2048, TG=512, C=8`, bf16 is clearly worse for decode-heavy concurrency: E2E p95 36.16s vs 25.94s for 4bit, aggregate 80.9 tok/s vs 173.4 tok/s for 4bit.
- At `PP=32768, C=8`, bf16 has lower E2E p95 than 4bit/8bit in this run, but finish summaries are mostly `stop`, so these cells are dominated by long-prompt admission/TTFT and early stopping rather than full 512-token decode.
- The current concurrent JSON summary does not retain per-request raw completion counts; use CSV or extend JSON before making strict claims about 512-token completion length distribution.
