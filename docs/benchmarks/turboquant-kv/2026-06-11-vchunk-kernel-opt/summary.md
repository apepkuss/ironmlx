# TurboQuant weighted_v_chunk kernel optimization

日期: 2026-06-11

分支: `codex/turboquant-vchunk-kernel-opt`

基线: `codex/turboquant-qk-kernel-opt` / `7dd0247 feat: optimize turboquant qk decode kernel`

## 结论

本阶段保留 `vdim2+simd` 方案。它在 QK kernel 优化后的基础上继续优化 decode parallel path 中的 `weighted_v_chunk`:

- 使用 simdgroup reduction 代替 256-lane threadgroup tree reduction。
- 单个 threadgroup 同时计算 2 个相邻 V head dim，复用同一 position 的 softmax weight 和 V norm 读取。
- `v_partial` layout 保持 `[batch, q_heads, v_chunks, head_dim]` 不变，后续 `weighted_v_reduce` 无需改动。

在 32k context / K3V4 / 32 decode tokens / warm run 下:

- 端到端 generation TPS: `31.403639 -> 47.565883`，提升 `51.47%`。
- `weighted_v_chunk` profile total: `152.825 ms -> 82.145 ms`，下降 `46.25%`。
- 相比 simd-only V chunk 实验，TPS 继续提升 `34.96%`，`weighted_v_chunk` 继续下降 `38.23%`。

## 环境

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quant: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`

## End-to-end results

| Variant | Tokens | Run | TPS | Decode ms | Generated token check |
| --- | ---: | --- | ---: | ---: | --- |
| QK baseline | 32 | warm | 31.403639 | 987.146750 | reference |
| V chunk simd-only | 32 | cold | 17.201931 | 1802.123250 | 32-token prefix equals baseline |
| V chunk simd-only | 32 | warm | 35.243962 | 879.583291 | first difference at token 31 |
| V chunk vdim2+simd | 32 | cold | 25.468441 | 1217.192667 | first difference at token 31 |
| V chunk vdim2+simd | 32 | warm | 47.565883 | 651.727625 | 32-token prefix equals baseline |

The first difference at token 31 is within the previously observed floating-point tie sensitivity range. The retained `vdim2+simd` warm run exactly matched the QK baseline token sequence for all 32 generated tokens.

## Stage profile

Profile runs used 8 decode tokens with `IRONMLX_TURBOQUANT_ATTN_PROFILE=1`.

| Variant | Run | Stage | Total ms | p50 us | p95 us | Max us | >=10ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| QK baseline | warm | weighted_v_chunk | 152.825 | 2710 | 2815 | 3016 | 0 |
| QK baseline | warm | q_rotate | 52.898 | 959 | 996 | 1205 | 0 |
| QK baseline | warm | qk | 44.264 | 790 | 880 | 942 | 0 |
| QK baseline | warm | weighted_v_reduce | 11.251 | 196 | 231 | 336 | 0 |
| QK baseline | warm | softmax | 11.095 | 194 | 238 | 357 | 0 |
| V chunk simd-only | warm | weighted_v_chunk | 132.975 | 2364 | 2481 | 2598 | 0 |
| V chunk simd-only | warm | q_rotate | 53.055 | 964 | 1004 | 1168 | 0 |
| V chunk simd-only | warm | qk | 42.908 | 754 | 874 | 949 | 0 |
| V chunk simd-only | warm | weighted_v_reduce | 10.558 | 187 | 214 | 328 | 0 |
| V chunk simd-only | warm | softmax | 9.977 | 176 | 220 | 258 | 0 |
| V chunk vdim2+simd | warm | weighted_v_chunk | 82.145 | 1440 | 1601 | 1639 | 0 |
| V chunk vdim2+simd | warm | q_rotate | 52.166 | 935 | 1009 | 1183 | 0 |
| V chunk vdim2+simd | warm | qk | 44.751 | 799 | 887 | 940 | 0 |
| V chunk vdim2+simd | warm | weighted_v_reduce | 11.110 | 192 | 245 | 343 | 0 |
| V chunk vdim2+simd | warm | softmax | 10.743 | 190 | 234 | 305 | 0 |

## Experiment decision

| Candidate | Decision | Reason |
| --- | --- | --- |
| simd-only V chunk | Rejected as final | Correct and faster than QK baseline, but still launches one threadgroup per output dim and reloads the same weight/norm for each dim. |
| vdim2+simd V chunk | Retained | Cuts V chunk threadgroups by 2, reuses weight/norm per two adjacent dims, and preserves the existing partial output contract. |

## Artifacts

- `ctx-32k/core-bench/vchunk-simd-k3v4.json`
- `ctx-32k/core-bench/vchunk-simd-k3v4-rerun.json`
- `ctx-32k/core-bench/vchunk-vdim2-simd-k3v4.json`
- `ctx-32k/core-bench/vchunk-vdim2-simd-k3v4-rerun.json`
- `ctx-32k/profile/vchunk-simd-k3v4-8tok.stderr.txt`
- `ctx-32k/profile/vchunk-simd-k3v4-8tok-rerun.stderr.txt`
- `ctx-32k/profile/vchunk-vdim2-simd-k3v4-8tok.stderr.txt`
- `ctx-32k/profile/vchunk-vdim2-simd-k3v4-8tok-rerun.stderr.txt`

## Next validation target

Run the required Rust checks and then use the retained `vdim2+simd` kernel as the new candidate for any longer multi-run quality/performance validation.
