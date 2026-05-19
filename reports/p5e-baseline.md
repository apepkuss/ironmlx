# P5e Baseline (P5 close-out state, ironmlx-p5e-perf branch)

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Branch HEAD | c4b9c27 (P5 close-out state; measurements harvested by 96dce61 which only added the measurement infra without changing inference code) |
| Hardware | M5 Max 128GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Method | tests/p5e_baseline.rs Model::forward_on direct call, 1 warmup + 3 measured runs, median |
| Token IDs | Deterministic synth (10000 + i % 100) |

## Wall-clock medians (ms) and throughput (tok/s)

| PP | Runs (ms) | Median (ms) | tok/s |
|---|---|---|---|
| 128 | [127.47, 127.66, 128.08] | 127.66 | 1002.7 |
| 512 | [488.22, 488.30, 488.41] | 488.30 | 1048.5 |
| 2048 | [2036.56, 2067.45, 2112.15] | 2067.45 | 990.6 |

## Cross-check vs P5d T2 + P5e T0

- P5d T2 (via iron-bench HTTP on M1 Pro): PP=2048 ≈ 996 tok/s.
- P5e T0 (Model::forward_on with per-layer eval barriers on M5 Max): PP=2048 ≈ 921 tok/s.
- This baseline (Model::forward_on with single end-of-forward eval on M5 Max): PP=2048 = 990.6 tok/s.

Discrepancies:

- vs T0 profile (≈921 tok/s on M5 Max): T0 inserts `eval` barriers after every
  layer to attribute time per-call; that bookkeeping adds ~155 ms wall-clock at
  PP=2048. This baseline uses a single end-of-forward `eval`.
- vs P5d T2 (≈996 tok/s on M1 Pro): different hardware (M5 Max here vs M1 Pro
  there) plus HTTP path overhead in P5d.
