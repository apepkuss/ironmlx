# TurboQuant Packed Attention Kernel Opt - 32K K3V4

Date: 2026-06-13

Worktree: `/Users/xin/workspace/ironmlx-backend-turboquant-mrope-qrotate-fusion`

Branch: `codex/turboquant-mrope-qrotate-fusion`

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Control run: `--max-tokens 64 --runs 3 --warmup-runs 1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 16 --runs 1 --warmup-runs 1`
- Retention gate: keep the code change only when no-profile p50 generation TPS improves by at least 3%.

## Control Results

| Variant | V chunk dims per group | Runs | Valid | Decode p50 ms | Generation p50 TPS | Generation mean TPS | TTFT p50 ms | E2E p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8 | 3 | 3 | 1055.762 | 59.673 | 58.439 | 13303.834 | 14436.461 |
| vchunk-vdim16 | 16 | 3 | 3 | 968.719 | 65.034 | 63.836 | 12648.214 | 13616.933 |

Threshold p50 TPS: `59.6725 * 1.03 = 61.4627`.

Result: `65.0343 TPS`, `+8.99%` over baseline. Decode p50 improved by `-8.24%`.

## Attention Profile

| Variant | Stage | N | Mean us | P50 us | P95 us | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | qk | 240 | 765.038 | 749 | 922 | 183.609 |
| Baseline | softmax | 240 | 186.329 | 187 | 224 | 44.719 |
| Baseline | weighted_v_chunk | 240 | 830.204 | 807 | 973 | 199.249 |
| Baseline | weighted_v_reduce | 240 | 187.638 | 188 | 232 | 45.033 |
| vchunk-vdim16 | qk | 240 | 745.313 | 740 | 833 | 178.875 |
| vchunk-vdim16 | softmax | 240 | 186.988 | 186 | 231 | 44.877 |
| vchunk-vdim16 | weighted_v_chunk | 240 | 687.779 | 685 | 766 | 165.067 |
| vchunk-vdim16 | weighted_v_reduce | 240 | 191.229 | 191 | 230 | 45.895 |

## Decision

Retain `V_CHUNK_DIMS_PER_THREADGROUP = 16`.

The retained change passes the no-profile retention gate and directly reduces the targeted `weighted_v_chunk` stage (`830.204us -> 687.779us`, `-17.16%` mean). The small `weighted_v_reduce` increase is outweighed by the chunk-stage and end-to-end decode gains.
