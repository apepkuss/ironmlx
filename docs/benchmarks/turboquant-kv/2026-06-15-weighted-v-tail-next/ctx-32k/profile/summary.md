# TurboQuant weighted-v tail branchless candidate

Date: 2026-06-15

Branch: `codex/turboquant-weighted-v-tail-next`

Base: `codex/turboquant-packed-attn-tail-next` at `2a8a55e`

## Candidate

Add a branchless weighted-v chunk kernel for the dominant Qwen3.5 K3V4 decode shape:

- `q_per_kv == V_Q_HEADS_PER_THREADGROUP == 4`
- `head_dim % V_CHUNK_DIMS_PER_THREADGROUP == 0`

The generic weighted-v chunk kernel remains the fallback for other shapes. The branchless kernel removes the per-q and per-dim boundary checks and dispatches one q-group per KV head for this exact shape.

## No-profile core benchmark

Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`

Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`

Command shape: `--mode gs-text --max-tokens 64 --runs 3 --warmup-runs 1 --kv-quant k3v4`

| Variant | Decode p50 ms | Decode mean ms | TPS p50 | TPS mean |
| --- | ---: | ---: | ---: | ---: |
| qkpos4 base | 813.897 | 812.814 | 77.405 | 77.529 |
| branchless weighted-v chunk | 799.448 | 801.741 | 78.804 | 78.584 |
| Delta | -1.78% | -1.36% | +1.81% | +1.36% |

## Attention stage profile

Profile command shape: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 64 --runs 1 --warmup-runs 0 --kv-quant k3v4`

Baseline uses post-qkpos4 stable r2+r3. Candidate uses branchless r1+r2.

| Stage | Base mean us | Candidate mean us | Base p50 us | Candidate p50 us | Base p95 us | Candidate p95 us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_v_chunk | 561.340 | 541.009 | 539 | 517 | 715 | 678 |
| qk | 551.122 | 527.355 | 538 | 518 | 778 | 706 |
| weighted_v_reduce | 187.529 | 184.141 | 190 | 185 | 227 | 224 |
| softmax | 182.079 | 177.640 | 183 | 174 | 226 | 224 |

## Decision

Retain the candidate. The no-profile 32k K3V4 benchmark improves both p50 and mean decode/TPS, and the focused attention profile shows `weighted_v_chunk` mean/p50/p95 moving down versus the post-qkpos4 stable baseline.

## Artifacts

- `docs/benchmarks/turboquant-kv/2026-06-15-weighted-v-tail-next/ctx-32k/core-bench/branchless-vchunk-k3v4-3x64.json`
- `docs/benchmarks/turboquant-kv/2026-06-15-weighted-v-tail-next/ctx-32k/profile/branchless-vchunk-k3v4-profile-1x64.json`
- `docs/benchmarks/turboquant-kv/2026-06-15-weighted-v-tail-next/ctx-32k/profile/branchless-vchunk-k3v4-profile-1x64.stderr.txt`
- `docs/benchmarks/turboquant-kv/2026-06-15-weighted-v-tail-next/ctx-32k/profile/branchless-vchunk-k3v4-profile-r2-1x64.json`
- `docs/benchmarks/turboquant-kv/2026-06-15-weighted-v-tail-next/ctx-32k/profile/branchless-vchunk-k3v4-profile-r2-1x64.stderr.txt`
