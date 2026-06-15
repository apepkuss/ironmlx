# TurboQuant Fused Decode Attention Spike

Date: 2026-06-13

Worktree: `/Users/xin/workspace/ironmlx-backend-turboquant-mrope-qrotate-fusion`

Branch: `codex/turboquant-mrope-qrotate-fusion`

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Candidate: experimental fused chunk kernel combining QK score, chunk softmax stats, and weighted V accumulation for `head_dim=256`.

## Core Result

| Variant | Runs | Max tokens | p50 generation TPS | mean generation TPS | p50 decode ms | p50 TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| Baseline qksg4 K3V4 | 3 | 64 | 65.7666 | 65.3404 | 957.9335 | 12032.4378 |
| Fused candidate K3V4 | 3 | 64 | 21.5486 | 21.5496 | 2923.6218 | 12521.6041 |

Retention threshold: baseline p50 TPS * 1.03 = 67.7396 TPS.

Decision: reject and remove the fused candidate from the implementation. The candidate was 67.23% slower than the retained baseline by p50 generation TPS.

## Profile Summary

Profile command used `IRONMLX_TURBOQUANT_ATTN_PROFILE=1`, `--runs 1`, `--warmup-runs 1`, and `--max-tokens 16`.

| Stage | Count | mean us | p50 us | max us |
|---|---:|---:|---:|---:|
| fused_qk_softmax_v_chunk | 240 | 5415.6 | 5144.5 | 8900 |
| fused_v_reduce | 240 | 232.5 | 226.0 | 439 |

The fused chunk stage is the bottleneck. It scans each 256-token chunk twice and reduces QK inside a single threadgroup per chunk, which loses the retained path's broader parallelism across `(batch, q_head, position)` for QK and across V dimension groups for weighted V.

## Artifacts

- Core benchmark: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/core-bench/fused-k3v4-3x64.json`
- Profile JSON: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/fused-k3v4-profile-1x16.json`
- Profile JSONL stderr: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/fused-k3v4-profile-1x16.stderr.txt`
