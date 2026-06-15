# TurboQuant Packed Attention QK Position Block Result

## Setup

- Branch: `codex/turboquant-packed-attn-qk-opt`
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Prompt: `ctx-32k.txt`
- KV quantization: `k3v4`
- Candidate: `QK_POSITIONS_PER_SIMDGROUP = 2`
- Keep threshold: candidate no-profile p50 TPS >= baseline p50 TPS * 1.03

## No-profile core bench

| Variant | Runs | Valid runs | p50 TPS | mean TPS | p50 decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 3 | 63.7182 | 63.1872 | 988.7282 |
| qkpos2 | 3 | 3 | 70.0019 | 67.7215 | 899.9757 |
| main-squash | 3 | 3 | 67.2761 | 67.2433 | 936.4390 |

QK position blocking improved p50 TPS by 9.86%, mean TPS by 7.18%, and reduced p50 decode time by 8.98% in the candidate worktree. After squash-merging the candidate onto `codex/scheduler-autotune-v2`, the final main-branch confirmation run still improved p50 TPS by 5.58% over the original baseline.

## Attention profile

| Stage | baseline mean us | qkpos2 mean us | baseline p50 us | qkpos2 p50 us |
| --- | ---: | ---: | ---: | ---: |
| qk | 781.96 | 606.57 | 743.5 | 629.5 |
| softmax | 185.35 | 187.12 | 183.0 | 187.0 |
| weighted_v_chunk | 699.33 | 684.77 | 668.5 | 671.0 |
| weighted_v_reduce | 188.27 | 193.43 | 183.5 | 191.0 |

The candidate reduced QK mean time by 22.43% and QK p50 time by 15.33%. The small softmax and reduce changes are within the expected profile noise for this workload, while the end-to-end no-profile result clears the retention threshold.

## Decision

Keep qkpos2.

The candidate improves the primary no-profile throughput metric and passes the packed-attention correctness controls. It also matches the root-cause hypothesis: each simdgroup now computes two adjacent sequence positions for the same `(batch, q_head)` block, reusing each lane's rotated query value across two K vectors.
