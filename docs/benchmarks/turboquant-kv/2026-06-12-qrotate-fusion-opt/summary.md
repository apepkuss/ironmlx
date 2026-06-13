# TurboQuant Q-Rotate Fusion Exploration

Date: 2026-06-12

Branch: `codex/turboquant-qrotate-fusion-opt`

Base: `960ccd4 perf: hoist turboquant qk norm multiply`

## Goal

Investigate whether the remaining TurboQuant decode `q_rotate` cost can be reduced by changing the standalone query-rotation kernel shape. The previous pass showed local Hadamard shuffle changes did not materially improve `q_rotate`, so this pass tested a more structural simdgroup-per-query implementation.

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quantization: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 64 --runs 1 --warmup-runs 1`
- Observed attention shape in this run: `batch=1`, `q_heads=16`, `kv_heads=4`, `head_dim=256`, measured decode `seq_len=37446`

Profile statistics below use only the measured half of the profile output, excluding the warmup half.

## Experiment

The rejected candidate replaced the existing `HEAD_DIM`-thread, threadgroup-memory query rotate kernel with a simdgroup-per-query variant:

- each simdgroup owned one full query vector;
- each lane held `HEAD_DIM / 32` values in registers;
- widths `1..16` used `simd_shuffle_xor`;
- remaining Hadamard stages were completed across the per-lane register array;
- `Q_ROT_SIMDGROUPS_PER_THREADGROUP` was swept across `1`, `2`, `4`, and `8`.

Correctness was checked with the dense-materialized TurboQuant parallel decode reference test before profiling the candidates.

## Result

Do not retain the simdgroup q-rotate kernel.

The best q-rotate stage reduction was only about `0.8%`, and the end-to-end profiled decode path was flat or worse. The added kernel complexity is not justified by the measured gain.

| Run | Profile TPS | Decode mean (ms) | q_rotate total (ms) | q_rotate mean (us) | q_rotate p95 (us) | q_rotate max (us) | qk total (ms) | weighted_v_chunk total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline qknorm | 41.365400 | 1523.012000 | 461.693 | 916.1 | 966 | 5751 | 371.619 | 405.644 |
| simdgroup1 | 41.358215 | 1523.276583 | 458.106 | 908.9 | 963 | 1118 | 372.734 | 403.897 |
| simdgroup2 | 40.925540 | 1539.381042 | 460.186 | 913.1 | 969 | 1907 | 373.775 | 404.911 |
| simdgroup4 | 41.190673 | 1529.472459 | 457.974 | 908.7 | 966 | 1043 | 374.565 | 406.512 |
| simdgroup8 | 39.716702 | 1586.234417 | 480.270 | 952.9 | 1142 | 1450 | 391.612 | 424.329 |

## Interpretation

This result supports the earlier hypothesis that the standalone `q_rotate` stage is dominated by dispatch/synchronization and low-granularity pipeline cost, not by the local Hadamard arithmetic alone. A meaningful improvement likely requires removing the standalone q-rotate dispatch entirely, for example by folding the rotate into query projection weights or by designing a broader attention/query projection fusion. That is a larger model-path design change and was not attempted in this pass.

## Code State

No production code change is retained from this experiment. The worktree source is back to the `960ccd4` qk norm-hoist baseline.

## Artifacts

- Baseline profile: `ctx-32k/profile/baseline-qknorm-profile-1x64.json`, `ctx-32k/profile/baseline-qknorm-profile-1x64.stderr.txt`
- Rejected simdgroup profiles:
  - `ctx-32k/profile/simdgroup1-qrotate-profile-1x64.json`, `ctx-32k/profile/simdgroup1-qrotate-profile-1x64.stderr.txt`
  - `ctx-32k/profile/simdgroup2-qrotate-profile-1x64.json`, `ctx-32k/profile/simdgroup2-qrotate-profile-1x64.stderr.txt`
  - `ctx-32k/profile/simdgroup-qrotate-profile-1x64.json`, `ctx-32k/profile/simdgroup-qrotate-profile-1x64.stderr.txt`
  - `ctx-32k/profile/simdgroup8-qrotate-profile-1x64.json`, `ctx-32k/profile/simdgroup8-qrotate-profile-1x64.stderr.txt`
