# TurboQuant Pre-Rotated QK Cost Probe

## Scope

Separate the fused TurboQuant decode query transform from the profiled `qk`
attention stage on the 32k long-context K3V4 workload.

## Benchmark

- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quantization: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`
- Decode tokens: `64`

## Attention Stage Profile

The profile run was executed with `IRONMLX_TURBOQUANT_ATTN_PROFILE=1`. The new
`turboquant_mrope_stage` event forces and records the fused MRoPE + TurboQuant
query transform before attention, so the following `qk` event no longer absorbs
that lazy upstream work.

| Event | Stage | Count | P50 us | P95 us | Mean us |
| --- | --- | ---: | ---: | ---: | ---: |
| `turboquant_mrope_stage` | `decode_query_turbo_rotation` | 1008 | 929 | 969 | 911.377 |
| `turboquant_attn_stage` | `qk` | 1008 | 736 | 830 | 750.299 |
| `turboquant_attn_stage` | `softmax` | 1008 | 188 | 228 | 185.579 |
| `turboquant_attn_stage` | `weighted_v_chunk` | 1008 | 801 | 887 | 813.803 |
| `turboquant_attn_stage` | `weighted_v_reduce` | 1008 | 190 | 230 | 189.510 |

Compared with the previous fused profile, `qk` p50 drops from `1473us` to
`736us` once the fused transform is materialized separately. This matches the
older q-rotate baseline `qk` range, so the earlier high fused `qk` measurement
was profile attribution noise rather than a qk kernel regression.

## End-to-End Runs

The profile run includes forced synchronization for attribution and should not
be compared directly with production no-profile throughput. The no-profile
runs keep the normal lazy execution path.

| Run | Decode ms mean | Generation TPS mean | Valid runs | Notes |
| --- | ---: | ---: | ---: | --- |
| `pre-rotated-qk-cost-profile-1x64` | 1535.107 | 41.039 | 1/1 | Attribution run with profile sync |
| `pre-rotated-qk-cost-noprofile-1x64` | 1014.331 | 62.110 | 1/1 | Normal execution |
| `pre-rotated-qk-cost-noprofile-3x64` | 1030.825 | 61.121 | 3/3 | Normal execution |

All runs reported `valid=true` and `finish_reason=length`. The generated answer
prefix remained:

```text
CHECKSUM record=00704 alpha=07 beta=08 gamma=16
```

## Decision

Do not add a production qk kernel change in this step. The real qk cost is back
near the previous baseline once the fused transform is measured separately.
Keeping the profile-only attribution event is useful because it prevents future
TurboQuant decode profiles from mislabeling fused MRoPE + WHT materialization as
qk time.
