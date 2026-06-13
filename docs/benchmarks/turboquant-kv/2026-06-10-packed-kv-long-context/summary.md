# TurboQuant KV Long Context Validation

- root: `/Users/xin/workspace/ironmlx-backend-turboquant-packed-kv/docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context`
- kv matrix: `none,turbo4,k3v4`

## Logits Replay

| context | prompt tokens | kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ctx-4k | 4735 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-4k | 4735 | turbo4 | 24 | 24 | 31/32 | 1.509163 | 0.205742 | 0.259366 | 0.984687270 |
| ctx-4k | 4735 | k3v4 | 32 | none | 32/32 | 2.025585 | 0.301722 | 0.377176 | 0.967960830 |
| ctx-8k | 9399 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-8k | 9399 | turbo4 | 32 | none | 32/32 | 1.243233 | 0.193541 | 0.240586 | 0.985789300 |
| ctx-8k | 9399 | k3v4 | 32 | none | 32/32 | 1.801361 | 0.281403 | 0.349155 | 0.961803140 |
| ctx-16k | 18727 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-16k | 18727 | turbo4 | 32 | none | 32/32 | 1.836769 | 0.234906 | 0.298463 | 0.936246200 |
| ctx-16k | 18727 | k3v4 | 30 | 30 | 31/32 | 1.687241 | 0.248596 | 0.312586 | 0.963063200 |
| ctx-32k | 37383 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-32k | 37383 | turbo4 | 32 | none | 32/32 | 1.285767 | 0.189694 | 0.238520 | 0.982761600 |
| ctx-32k | 37383 | k3v4 | 30 | 30 | 31/32 | 1.724365 | 0.249579 | 0.313427 | 0.973182500 |

## Core Generation

| context | prompt tokens | kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none | generated text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ctx-4k | 4735 | none | 32 | 1045.66 | 1258.48 | 145.66 | 2.441 | 5.758 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | turbo4 | 32 | 1052.55 | 1322.25 | 114.94 | 2.456 | 6.075 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | k3v4 | 32 | 1052.22 | 1322.04 | 114.89 | 2.442 | 6.069 | 24 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> Thinking Process: 1. ** |
| ctx-8k | 9399 | none | 32 | 2173.14 | 2397.55 | 138.14 | 2.442 | 7.356 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | turbo4 | 32 | 2239.02 | 2564.88 | 95.13 | 2.443 | 8.022 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | k3v4 | 32 | 2289.19 | 2612.18 | 95.98 | 2.443 | 8.012 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | none | 32 | 5201.97 | 5445.32 | 127.39 | 2.447 | 11.377 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | turbo4 | 32 | 5323.95 | 5757.16 | 71.56 | 2.448 | 12.440 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | k3v4 | 32 | 5281.26 | 5718.69 | 70.87 | 2.448 | 12.422 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | none | 32 | 13445.38 | 13738.60 | 105.72 | 2.459 | 29.196 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | turbo4 | 32 | 13667.01 | 14323.92 | 47.19 | 2.459 | 32.166 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | k3v4 | 32 | 13747.83 | 14403.96 | 47.25 | 2.460 | 32.071 | 30 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate the KV |

## Notes

- `prompt tokens` are measured by the ironmlx tokenizer at runtime.
- `exact-prefix vs none` compares the first measured core-bench record for each KV setting against `none`.
- `peak footprint GiB` comes from macOS `/usr/bin/time -l` and is more sensitive to MLX memory pressure than `maximum resident set size` on this workload.
- The benchmark uses one measured run by default; use higher `RUNS` for stable latency percentiles.
