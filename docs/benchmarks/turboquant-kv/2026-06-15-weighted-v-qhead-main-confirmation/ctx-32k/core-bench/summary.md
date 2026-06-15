# TurboQuant Weighted-V Q-Head Main Confirmation

Date: 2026-06-15

Branch: `codex/scheduler-autotune-v2`

Commit: `7cc702a perf: group turboquant weighted v query heads`

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Command shape: `--max-tokens 64 --runs 3 --warmup-runs 1`

## Result

| variant | valid | decode p50 ms | decode mean ms | generation p50 tps | generation mean tps |
| --- | ---: | ---: | ---: | ---: | ---: |
| post-QK baseline | 3/3 | 908.927 | 907.363 | 69.313 | 69.439 |
| q-head group 4 candidate | 3/3 | 876.178 | 875.133 | 71.903 | 71.991 |
| main confirmation | 3/3 | 865.687 | 868.605 | 72.775 | 72.536 |

The merged mainline result improves p50 generation TPS by about 5.0% versus the
post-QK baseline and about 1.2% versus the retained candidate run. Decode p50
time improves by about 4.8% versus the post-QK baseline.

## Decision

The squash-merged `weighted_v_chunk` q-head grouping optimization is confirmed
on the current main branch.

## Artifacts

- `main-k3v4-3x64.json`
