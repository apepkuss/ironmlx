# Dense bf16 Linear Layout A/B - Qwen3.5-4B-bf16

Date: 2026-07-09

## Scope

This benchmark isolates dense bf16 `Linear` matmul only:

- Current path: `weight.transpose_on(target)` then `x.matmul_on(&wt, target)`.
- Candidate path: one-time `weight.transpose_on(target).contiguous_on(false, target)` materialization, then repeated `x.matmul_on(&row_major_wt, target)`.

The measurement excludes HTTP, scheduler, attention, KV cache, tokenizer, sampler, and full model forward costs.

## Command

```bash
MLX_DIR=/Users/xin/.local/mlx cargo run --release -p ironmlx --bin ironmlx-bf16-linear-layout-ab -- \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-bf16/snapshots/475632ded9a95863da4e4b235ab9ccbc5d3cc6bf \
  --out docs/benchmarks/bf16-linear-layout-ab/2026-07-09-qwen35-4b-bf16-gate-proj/result.json \
  --runs 50 \
  --warmup-runs 10

MLX_DIR=/Users/xin/.local/mlx cargo run --release -p ironmlx --bin ironmlx-bf16-linear-layout-ab -- \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-bf16/snapshots/475632ded9a95863da4e4b235ab9ccbc5d3cc6bf \
  --weight-key language_model.model.layers.0.mlp.down_proj.weight \
  --out docs/benchmarks/bf16-linear-layout-ab/2026-07-09-qwen35-4b-bf16-down-proj/result.json \
  --runs 50 \
  --warmup-runs 10
```

## Results

Raw JSON:

- `docs/benchmarks/bf16-linear-layout-ab/2026-07-09-qwen35-4b-bf16-gate-proj/result.json`
- `docs/benchmarks/bf16-linear-layout-ab/2026-07-09-qwen35-4b-bf16-down-proj/result.json`

### `mlp.gate_proj.weight` `[9216, 2560]`

One-time row-major transpose materialization: 6.767 ms.

| case | current lazy p50 / p95 / mean ms | row-major p50 / p95 / mean ms | mean delta |
| --- | ---: | ---: | ---: |
| decode-c1 `[1,1,2560]` | 0.502 / 1.352 / 0.724 | 0.487 / 1.316 / 0.623 | -13.9% |
| decode-c8 `[8,1,2560]` | 0.338 / 0.414 / 0.341 | 0.286 / 0.397 / 0.296 | -13.1% |
| prefill-2048 `[1,2048,2560]` | 1.795 / 1.895 / 1.804 | 1.788 / 1.884 / 1.793 | -0.6% |

### `mlp.down_proj.weight` `[2560, 9216]`

One-time row-major transpose materialization: 7.280 ms.

| case | current lazy p50 / p95 / mean ms | row-major p50 / p95 / mean ms | mean delta |
| --- | ---: | ---: | ---: |
| decode-c1 `[1,1,9216]` | 0.507 / 1.384 / 0.736 | 0.834 / 1.874 / 1.020 | +38.6% |
| decode-c8 `[8,1,9216]` | 0.322 / 0.428 / 0.337 | 0.276 / 0.343 / 0.287 | -14.8% |
| prefill-2048 `[1,2048,9216]` | 2.068 / 2.197 / 2.084 | 2.075 / 2.319 / 2.092 | +0.4% |

## Conclusion

The row-major pretranspose candidate is not a reliable production optimization for dense bf16 `Linear`:

- It helps some decode batch shapes, especially `decode-c8`, by about 13-15% mean in these two MLP projections.
- It regresses `down_proj` `decode-c1` by about 39% mean.
- It is effectively neutral for `prefill-2048`, where the long-prompt prefill shape dominates.
- It adds about 6.8-7.3 ms one-time materialization per projection. That is acceptable only if done at load time and only if the original layout can be dropped; duplicating both layouts across all dense bf16 weights is not acceptable.

Decision: do not change production `Linear::Fp` to forced row-major pretranspose based on this result. The bf16 decode/high-concurrency gap is more likely from dense bf16 arithmetic volume, MLX kernel choice by shape/batch, and full decode pipeline behavior than from the current lazy transpose layout alone.
