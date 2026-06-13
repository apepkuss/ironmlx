# TurboQuant KV Long Context Validation

- root: `/Users/xin/workspace/ironmlx-backend-turboquant-kv/docs/benchmarks/turboquant-kv/2026-06-10-long-context`
- kv matrix: `none,turbo4,k3v4`

## Logits Replay

| context | prompt tokens | kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ctx-4k | 4735 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-4k | 4735 | turbo4 | 32 | none | 32/32 | 1.503525 | 0.210268 | 0.265584 | 0.987734560 |
| ctx-4k | 4735 | k3v4 | 32 | none | 32/32 | 2.057101 | 0.308833 | 0.385300 | 0.968668340 |
| ctx-8k | 9399 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-8k | 9399 | turbo4 | 24 | 24 | 31/32 | 1.307678 | 0.191432 | 0.240678 | 0.974690900 |
| ctx-8k | 9399 | k3v4 | 32 | none | 32/32 | 1.852743 | 0.282299 | 0.350597 | 0.945408640 |
| ctx-16k | 18727 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-16k | 18727 | turbo4 | 32 | none | 32/32 | 1.307396 | 0.191492 | 0.240915 | 0.971678900 |
| ctx-16k | 18727 | k3v4 | 30 | 30 | 31/32 | 1.656540 | 0.252955 | 0.317293 | 0.967671800 |
| ctx-32k | 37383 | none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 |
| ctx-32k | 37383 | turbo4 | 32 | none | 32/32 | 1.264408 | 0.183222 | 0.230557 | 0.985120700 |
| ctx-32k | 37383 | k3v4 | 30 | 30 | 31/32 | 1.679054 | 0.249005 | 0.312447 | 0.967162550 |

## Core Generation

| context | prompt tokens | kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none | generated text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ctx-4k | 4735 | none | 32 | 1048.94 | 1263.01 | 144.81 | 2.452 | 5.754 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | turbo4 | 32 | 1064.66 | 1487.77 | 73.27 | 2.456 | 6.613 | 32 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context |
| ctx-4k | 4735 | k3v4 | 32 | 1064.56 | 1479.19 | 74.77 | 2.453 | 6.604 | 24 | CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> Thinking Process: 1. ** |
| ctx-8k | 9399 | none | 32 | 2150.71 | 2376.48 | 137.31 | 2.457 | 7.357 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | turbo4 | 32 | 2274.17 | 2898.28 | 49.67 | 2.458 | 9.193 | 32 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context |
| ctx-8k | 9399 | k3v4 | 32 | 2340.51 | 2977.71 | 48.65 | 2.458 | 9.168 | 30 | CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate the KV |
| ctx-16k | 18727 | none | 32 | 5248.18 | 5496.15 | 125.01 | 2.462 | 11.378 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | turbo4 | 32 | 5502.66 | 6528.13 | 30.23 | 2.462 | 14.565 | 32 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context |
| ctx-16k | 18727 | k3v4 | 32 | 5442.60 | 6436.61 | 31.19 | 2.459 | 14.477 | 30 | CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate the KV |
| ctx-32k | 37383 | none | 32 | 14140.96 | 14447.83 | 101.02 | 2.473 | 29.197 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | turbo4 | 32 | 14901.36 | 16814.57 | 16.20 | 2.469 | 37.518 | 32 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context |
| ctx-32k | 37383 | k3v4 | 32 | 15017.24 | 16813.10 | 17.26 | 2.474 | 36.942 | 30 | CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate the KV |

## Notes

- `prompt tokens` are measured by the ironmlx tokenizer at runtime.
- `exact-prefix vs none` compares the first measured core-bench record for each KV setting against `none`.
- `peak footprint GiB` comes from macOS `/usr/bin/time -l` and is more sensitive to MLX memory pressure than `maximum resident set size` on this workload.
- The benchmark uses one measured run by default; use higher `RUNS` for stable latency percentiles.
