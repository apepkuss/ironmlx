# TurboQuant Score-V Decode Fusion Spike - 32K K3V4

Date: 2026-06-13

Worktree: `/Users/xin/workspace/ironmlx-backend-turboquant-mrope-qrotate-fusion`

Branch: `codex/turboquant-mrope-qrotate-fusion`

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Candidate: keep retained parallel QK, replace `softmax -> weighted_v_chunk -> weighted_v_reduce` with score-driven `score_chunk_stats -> score_weighted_v_chunk -> score_weighted_v_reduce`
- Control run: `--max-tokens 64 --runs 3 --warmup-runs 1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 16 --runs 1 --warmup-runs 1`
- Retention gate: candidate p50 generation TPS must be at least `67.7396`.

## Control Results

| Variant | Runs | Valid | Decode p50 ms | Generation p50 TPS | Generation mean TPS | TTFT p50 ms | E2E p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline qksg4 | 3 | 3 | 957.934 | 65.767 | 65.340 | 12032.438 | 12989.783 |
| Score-V candidate | 3 | 3 | 957.969 | 65.764 | 65.808 | 12477.661 | 13435.630 |

Threshold p50 TPS: `67.7396`.

Result: `65.7642 TPS`, below the retention gate and `-0.004%` versus baseline p50 TPS.

## Attention Profile

| Variant | Stage | N | Mean us | P50 us | P95 us | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline qksg4 | qk | 240 | 798.996 | 722.0 | 906.0 | 191.759 |
| Baseline qksg4 | softmax | 240 | 184.813 | 186.0 | 227.0 | 44.355 |
| Baseline qksg4 | weighted_v_chunk | 240 | 677.633 | 663.0 | 790.0 | 162.632 |
| Baseline qksg4 | weighted_v_reduce | 240 | 191.142 | 187.0 | 237.0 | 45.874 |
| Score-V candidate | qk | 240 | 737.575 | 733.5 | 820.1 | 177.018 |
| Score-V candidate | score_chunk_stats | 240 | 203.408 | 199.0 | 292.0 | 48.818 |
| Score-V candidate | score_weighted_v_chunk | 240 | 661.942 | 656.5 | 727.0 | 158.866 |
| Score-V candidate | score_weighted_v_reduce | 240 | 199.075 | 199.0 | 234.0 | 47.778 |

## Decision

Do not retain the score-driven V candidate.

The candidate improves profiled QK and V chunk time, but adds a `score_chunk_stats` kernel and does not improve the no-profile control run. Since the p50 generation TPS misses the retention gate, the implementation was reverted and this directory keeps benchmark evidence only.
