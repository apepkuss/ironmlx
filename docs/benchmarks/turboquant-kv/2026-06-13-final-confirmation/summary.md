# TurboQuant KV Long Context Validation

- root: `/Users/xin/workspace/ironmlx-backend/docs/benchmarks/turboquant-kv/2026-06-13-final-confirmation`
- kv matrix: `none,turbo3,turbo4,k3v4`

## Final Confirmation

- branch: `codex/scheduler-autotune-v2`
- commit: `a7c632d8837d98988a0ab89b34f4d2cc5ebdeeae`
- command shape: `STAMP=2026-06-13-final-confirmation KV_QUANTS=none,turbo3,turbo4,k3v4 MAX_TOKENS=32 BENCH_MAX_TOKENS=32 RUNS=1 WARMUP_RUNS=0`
- coverage: 4k, 8k, 16k, and 32k prompts with logits replay plus core generation benchmark.
- result: all measured KV modes generated the expected checksum prefix for every context length.
- `turbo4` (K4V4): logits replay matched the full 32-token prefix at 8k, 16k, and 32k; 4k diverged at token 24 while still generating the expected checksum.
- `k3v4`: logits replay matched the full 32-token prefix at 4k and 8k, then diverged at token 30 for 16k and 32k while still generating the expected checksum.
- `turbo3` (K3V3 smoke): logits replay matched the full 32-token prefix across all contexts in this run, but has lower cosine similarity than `turbo4` and should remain a lower-confidence option.
- 32k measured generation throughput: `none` 103.73 tok/s, `turbo3` 54.63 tok/s, `turbo4` 56.19 tok/s, `k3v4` 57.72 tok/s.
- Compared with `2026-06-10-packed-kv-long-context`, 32k TurboQuant generation throughput is higher in this single-run gate: `turbo4` 47.19 -> 56.19 tok/s, `k3v4` 47.25 -> 57.72 tok/s.

## Logits Replay

| context | prompt tokens | kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ctx-4k | 4735 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-4k | 4735 | turbo3 | 32 | none | 32/32 | 2.536381 | 0.403560 | 0.501337 | 0.962342000 |
| ctx-4k | 4735 | turbo4 | 24 | 24 | 31/32 | 1.471050 | 0.211518 | 0.265665 | 0.989537000 |
| ctx-4k | 4735 | k3v4 | 32 | none | 32/32 | 2.087173 | 0.314053 | 0.392127 | 0.965126160 |
| ctx-8k | 9399 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-8k | 9399 | turbo3 | 32 | none | 32/32 | 2.333649 | 0.348072 | 0.435215 | 0.943033930 |
| ctx-8k | 9399 | turbo4 | 32 | none | 32/32 | 1.218620 | 0.181959 | 0.228423 | 0.985397500 |
| ctx-8k | 9399 | k3v4 | 32 | none | 32/32 | 1.800476 | 0.288042 | 0.355240 | 0.961921500 |
| ctx-16k | 18727 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-16k | 18727 | turbo3 | 32 | none | 32/32 | 2.329102 | 0.358908 | 0.448574 | 0.940926600 |
| ctx-16k | 18727 | turbo4 | 32 | none | 32/32 | 1.214224 | 0.182251 | 0.229105 | 0.987300460 |
| ctx-16k | 18727 | k3v4 | 30 | 30 | 31/32 | 1.742615 | 0.259866 | 0.326450 | 0.956303660 |
| ctx-32k | 37383 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-32k | 37383 | turbo3 | 32 | none | 32/32 | 2.372650 | 0.351006 | 0.439646 | 0.944042400 |
| ctx-32k | 37383 | turbo4 | 32 | none | 32/32 | 1.275455 | 0.189946 | 0.238245 | 0.984870800 |
| ctx-32k | 37383 | k3v4 | 30 | 30 | 31/32 | 1.684570 | 0.252093 | 0.316290 | 0.974615300 |

## Core Generation

| context | prompt tokens | kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none | generated text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ctx-4k | 4735 | none | 32 | 1078.67 | 1294.41 | 143.69 | 2.457 | 5.759 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | turbo3 | 32 | 1061.01 | 1316.01 | 121.57 | 2.461 | 5.743 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | turbo4 | 32 | 1082.56 | 1338.93 | 120.92 | 2.445 | 5.751 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | k3v4 | 32 | 1059.41 | 1313.81 | 121.85 | 2.460 | 5.746 | 24 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> Thinking Process: 1. ** |
| ctx-8k | 9399 | none | 32 | 2269.31 | 2499.77 | 134.51 | 2.457 | 7.357 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | turbo3 | 32 | 2376.21 | 2677.11 | 103.02 | 2.461 | 7.327 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | turbo4 | 32 | 2454.42 | 2750.04 | 104.87 | 2.461 | 7.344 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | k3v4 | 32 | 2480.05 | 2784.11 | 101.95 | 2.447 | 7.334 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | none | 32 | 5297.02 | 5555.41 | 119.97 | 2.461 | 11.377 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | turbo3 | 32 | 5493.70 | 5888.77 | 78.47 | 2.450 | 11.513 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | turbo4 | 32 | 5467.00 | 5850.03 | 80.94 | 2.451 | 11.555 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | k3v4 | 32 | 5522.25 | 5887.82 | 84.80 | 2.452 | 11.533 | 30 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate the KV |
| ctx-32k | 37383 | none | 32 | 14380.90 | 14679.74 | 103.73 | 2.472 | 29.196 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | turbo3 | 32 | 14979.83 | 15547.30 | 54.63 | 2.462 | 30.204 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | turbo4 | 32 | 14764.36 | 15316.02 | 56.19 | 2.477 | 30.411 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | k3v4 | 32 | 14594.17 | 15131.22 | 57.72 | 2.477 | 30.293 | 30 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate the KV |

## Notes

- `prompt tokens` are measured by the ironmlx tokenizer at runtime.
- `exact-prefix vs none` compares the first measured core-bench record for each KV setting against `none`.
- `peak footprint GiB` comes from macOS `/usr/bin/time -l` and is more sensitive to MLX memory pressure than `maximum resident set size` on this workload.
- The benchmark uses one measured run by default; use higher `RUNS` for stable latency percentiles.
