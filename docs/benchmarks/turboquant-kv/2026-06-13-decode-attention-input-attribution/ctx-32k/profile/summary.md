# TurboQuant decode attention input attribution

## Setup

- Branch: `codex/turboquant-mrope-qrotate-fusion`
- Binary: `target/release/ironmlx-core-bench`
- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Prompt tokens: 37,383
- Decode tokens: 64
- KV quant: `k3v4`
- Profile env: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1`

## Profile Result

The attribution run inserted decode-only eval probes before attention projection,
after QKV projection, after query split/norm/reshape, inside TurboQuant query
rotation, and inside packed attention. Counts are 1,008 stage samples
(16 layers x 63 measured decode steps).

| stage | count | mean us | p50 us | p95 us | p99 us | min us | max us |
|---|---:|---:|---:|---:|---:|---:|---:|
| `decode_attention_input` | 1008 | 865.6 | 886.0 | 917 | 952 | 667 | 1109 |
| `decode_qkv_proj` | 1008 | 204.9 | 202.0 | 239 | 257 | 157 | 300 |
| `decode_q_split_norm_reshape` | 1008 | 164.7 | 162.0 | 198 | 213 | 115 | 241 |
| `decode_query_turbo_inputs` | 1008 | 25.0 | 0.0 | 199 | 217 | 0 | 316 |
| `decode_query_turbo_rotation` | 1008 | 164.3 | 163.5 | 192 | 215 | 107 | 335 |
| `qk` | 1008 | 723.0 | 709.0 | 820 | 882 | 646 | 1210 |
| `softmax` | 1008 | 179.1 | 182.0 | 209 | 231 | 129 | 416 |
| `weighted_v_chunk` | 1008 | 792.8 | 781.0 | 871 | 935 | 708 | 1259 |
| `weighted_v_reduce` | 1008 | 186.2 | 187.0 | 219 | 239 | 130 | 294 |

## End-to-end Control

No-profile control uses the normal execution path with the same model, prompt,
decode length, and `k3v4`.

| run set | runs | valid | decode p50 ms | decode mean ms | generation p50 tps | generation mean tps |
|---|---:|---:|---:|---:|---:|---:|
| no-profile 3x64 | 3 | 3 | 1044.145 | 1035.803 | 60.336 | 60.830 |
| profile 1x64 | 1 | 1 | 1766.561 | 1766.561 | 35.663 | 35.663 |

## Conclusion

The previous `decode_query_turbo_inputs` p50 around 922 us was not intrinsic to
the TurboQuant query input construction. Once `x` is explicitly materialized at
attention entry, `decode_query_turbo_inputs` falls to p50 0 us and mean 25 us.
The main newly exposed upstream cost is `decode_attention_input` at p50 886 us,
which means the next optimization target should move one level earlier into the
decoder block input path that produces the attention `x`.

The packed attention kernels remain the largest local attention cost:
`weighted_v_chunk` p50 781 us and `qk` p50 709 us. However, for the specific
question raised by the previous attribution, the false hotspot has been
disambiguated: the 900 us class cost is upstream lazy materialization before
attention, not MRoPE/TurboQuant query rotation itself.

