# TurboQuant Decode Query Transform Attribution

## Scope

Split the fused TurboQuant decode query transform profile into:

- upstream input materialization for `q`, `k`, `cos`, `sin`, and `query_signs`
- the fused MRoPE + TurboQuant query WHT kernel output materialization

The goal was to decide whether the next optimization should target the fused
Metal kernel or the work that produces its inputs.

## Benchmark

- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quantization: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`
- Decode tokens: `64`

## Attribution Profile

The attribution run was executed with `IRONMLX_TURBOQUANT_ATTN_PROFILE=1`.

| Event | Stage | Count | P50 us | P95 us | Mean us |
| --- | --- | ---: | ---: | ---: | ---: |
| `turboquant_mrope_stage` | `decode_query_turbo_inputs` | 1008 | 922 | 960 | 914.081 |
| `turboquant_mrope_stage` | `decode_query_turbo_rotation` | 1008 | 181 | 222 | 184.977 |
| `turboquant_attn_stage` | `qk` | 1008 | 723 | 820 | 746.196 |
| `turboquant_attn_stage` | `softmax` | 1008 | 188 | 217 | 186.675 |
| `turboquant_attn_stage` | `weighted_v_chunk` | 1008 | 798 | 897 | 812.945 |
| `turboquant_attn_stage` | `weighted_v_reduce` | 1008 | 191 | 231 | 191.453 |

## End-to-End Runs

The profile run includes forced synchronization and should only be used for
attribution. The no-profile run keeps the normal lazy execution path.

| Run | Decode ms mean | Generation TPS mean | Valid runs | Notes |
| --- | ---: | ---: | ---: | --- |
| `decode-query-transform-attribution-profile-1x64` | 1623.111 | 38.814 | 1/1 | Attribution run with input and output sync |
| `decode-query-transform-attribution-noprofile-3x64` | 1031.159 | 61.107 | 3/3 | Normal execution |

All runs reported `valid=true` and `finish_reason=length`. The generated answer
prefix remained:

```text
CHECKSUM record=00704 alpha=07 beta=08 gamma=16
```

## Decision

Do not run a fused-kernel optimization experiment in this step. The pure fused
MRoPE + WHT kernel is only `181us` p50, while input materialization is `922us`
p50. The earlier broad `decode_query_turbo_rotation` cost was dominated by
upstream lazy Q/K/cos/sin materialization, not by the fused Metal kernel.

The next meaningful optimization target is upstream decode attention input
production and scheduling, especially the lazy Q/K projection path feeding the
MRoPE+TurboQuant transform.
