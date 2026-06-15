# TurboQuant Weighted-V Q-Head Grouping

Date: 2026-06-15

Branch: `codex/turboquant-weighted-v-chunk-next`

Base: `codex/scheduler-autotune-v2` / `a482d11 perf: optimize turboquant qk block decode`

## Decision

Retain `V_Q_HEADS_PER_THREADGROUP = 4`.

The current GQA workload has seven query heads per KV head. The previous
`weighted_v_chunk` kernel decoded the same packed V vector once per query head.
This candidate groups query heads inside the same KV head, so one V unpack feeds
up to four query-head accumulators while keeping the existing `v_partial` layout
and `weighted_v_reduce` kernel unchanged.

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Control run: `--max-tokens 64 --runs 3 --warmup-runs 1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 64 --runs 1 --warmup-runs 0`
- Baseline: post-QK mainline attribution from `2026-06-14-weighted-v-word-reuse`

## No-Profile Control Results

| variant | valid | decode p50 ms | decode mean ms | generation p50 tps | generation mean tps | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| post-QK baseline | 3/3 | 908.927 | 907.363 | 69.313 | 69.439 | baseline |
| q-head group 2 | 3/3 | 894.196 | 892.756 | 70.454 | 70.579 | positive, not best |
| q-head group 3 | 3/3 | 1161.153 | 1146.720 | 54.256 | 55.004 | reject |
| q-head group 4 | 3/3 | 876.178 | 875.133 | 71.903 | 71.991 | keep |

`q-head group 4` improves p50 generation TPS by about 3.74% and reduces p50
decode time by about 3.60% versus the post-QK baseline.

## Attention Profile

Profile rows are 32k K3V4, 64 decode tokens.

| variant | stage | count | total ms | mean us | p50 us | p95 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| post-QK baseline | qk | 504 | 328.000 | 650.8 | 638.0 | 893.8 |
| post-QK baseline | softmax | 504 | 93.300 | 185.1 | 186.5 | 231.0 |
| post-QK baseline | weighted_v_chunk | 504 | 360.900 | 716.1 | 684.5 | 934.3 |
| post-QK baseline | weighted_v_reduce | 504 | 94.100 | 186.8 | 186.0 | 228.8 |
| q-head group 4 | qk | 504 | 316.520 | 628.0 | 636.0 | 846.0 |
| q-head group 4 | softmax | 504 | 94.713 | 187.9 | 189.0 | 232.0 |
| q-head group 4 | weighted_v_chunk | 504 | 280.093 | 555.7 | 539.0 | 691.0 |
| q-head group 4 | weighted_v_reduce | 504 | 94.347 | 187.2 | 189.0 | 228.0 |

The retained candidate reduces `weighted_v_chunk` mean time by about 22.4% and
p50 time by about 21.3%. `weighted_v_reduce` is effectively unchanged, which
matches the intended scope: reduce repeated V unpacking without changing the
partial-output contract.

## Artifacts

- `ctx-32k/core-bench/qhead2-k3v4-3x64.json`
- `ctx-32k/core-bench/qhead3-k3v4-3x64.json`
- `ctx-32k/core-bench/qhead4-k3v4-3x64.json`
- `ctx-32k/profile/qhead4-k3v4-profile-1x64.json`
- `ctx-32k/profile/qhead4-k3v4-profile-1x64.stderr.txt`
