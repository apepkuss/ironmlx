# TurboQuant MRoPE Q-Rotate Fusion

## Scope

Validate the fused decode-only MRoPE plus TurboQuant query rotation path on the
32k long-context K3V4 benchmark.

## Benchmark

- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quantization: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`
- Decode tokens: `64`

## Results

| Run | Decode ms mean | Generation TPS mean | Notes |
| --- | ---: | ---: | --- |
| Baseline `simdgroup-qrotate-profile-1x64` | 1529.472 | 41.191 | Existing q-rotate optimized path |
| Fused `fused-mrope-qrotate-profile-1x64` | 1442.786 | 43.666 | Single measured run |
| Fused `fused-mrope-qrotate-profile-3x64` | 1445.954 | 43.571 | Three measured runs |

Compared with the existing baseline, the 3-run fused profile improved decode
time by 5.46% and generation TPS by 5.78%.

## Attention Stage Profile

The fused profile removes the standalone TurboQuant `q_rotate` stage. The
3-run fused profile emitted no `q_rotate` events.

| Stage | Count | P50 us | P95 us | Mean us |
| --- | ---: | ---: | ---: | ---: |
| qk | 2016 | 1473 | 1552 | 1470.273 |
| softmax | 2016 | 204 | 245 | 205.186 |
| weighted_v_chunk | 2016 | 799 | 884 | 810.110 |
| weighted_v_reduce | 2016 | 189 | 227 | 188.389 |

## Quality Check

All three measured fused runs reported `valid=true` and `finish_reason=length`.
The generated checksum prefix remained:

```text
CHECKSUM record=00704 alpha=07 beta=08 gamma=16
```

## Decision

Keep the fused path. It eliminates the decode-time q-rotate dispatch and gives a
repeatable decode throughput improvement on the K3V4 long-context workload.
