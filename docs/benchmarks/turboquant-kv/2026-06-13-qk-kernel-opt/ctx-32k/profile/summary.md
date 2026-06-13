# TurboQuant QK Kernel Opt - 32K K3V4

Date: 2026-06-13

Worktree: `/Users/xin/workspace/ironmlx-backend-turboquant-mrope-qrotate-fusion`

Branch: `codex/turboquant-mrope-qrotate-fusion`

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Retained precondition: `V_CHUNK_DIMS_PER_THREADGROUP = 16`
- Control run: `--max-tokens 64 --runs 3 --warmup-runs 1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 16 --runs 1 --warmup-runs 1`
- Retention gate: keep the code change only when no-profile p50 generation TPS improves by at least 3%.

## Control Results

| Variant | QK simdgroups per threadgroup | Runs | Valid | Decode p50 ms | Generation p50 TPS | Generation mean TPS | TTFT p50 ms | E2E p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 4 | 3 | 3 | 957.934 | 65.767 | 65.340 | 12032.438 | 12989.783 |
| qksg8 | 8 | 3 | 3 | 966.759 | 65.166 | 64.520 | 12728.569 | 13695.327 |

Threshold p50 TPS: `65.7666 * 1.03 = 67.7396`.

Result: `65.1662 TPS`, `-0.91%` versus baseline. Decode p50 regressed by `+0.92%`.

## Attention Profile

| Variant | Stage | N | Mean us | P50 us | P95 us | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | qk | 240 | 798.996 | 722 | 906 | 191.759 |
| Baseline | softmax | 240 | 184.813 | 186 | 227 | 44.355 |
| Baseline | weighted_v_chunk | 240 | 677.633 | 663 | 790 | 162.632 |
| Baseline | weighted_v_reduce | 240 | 191.142 | 187 | 237 | 45.874 |
| qksg8 | qk | 240 | 738.921 | 719 | 904 | 177.341 |
| qksg8 | softmax | 240 | 188.075 | 187 | 240 | 45.138 |
| qksg8 | weighted_v_chunk | 240 | 681.442 | 667 | 792 | 163.546 |
| qksg8 | weighted_v_reduce | 240 | 186.825 | 187 | 229 | 44.838 |

## Decision

Do not retain `QK_SIMDGROUPS_PER_THREADGROUP = 8`.

The candidate reduces profiled QK mean (`798.996us -> 738.921us`, `-7.52%`), but the no-profile control run misses the retention gate and regresses p50 generation throughput. The working tree was reverted to the baseline value `QK_SIMDGROUPS_PER_THREADGROUP = 4`; this directory keeps the benchmark evidence only.
