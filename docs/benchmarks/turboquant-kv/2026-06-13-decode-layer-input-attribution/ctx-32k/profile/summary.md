# TurboQuant decode layer input attribution

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

The run adds decode-only eval probes around dense `DecoderLayer` stages, while
keeping the existing GatedAttention, MRoPE, and packed-attention probes. The
trace includes one warmup and one measured profile pass. Full-attention stage
counts are 1,008 samples: 8 full-attention layers x 63 decode steps x 2 passes.
Linear-attention stage counts are 3,024 samples: 24 linear layers x 63 decode
steps x 2 passes.

### Full Decoder Layers

| stage | count | mean us | p50 us | p95 us | p99 us | min us | max us | total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `decode_layer_input` | 1008 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.0 |
| `decode_input_norm` | 1008 | 163.7 | 161.0 | 196 | 252 | 118 | 443 | 165.0 |
| `decode_attention_path` | 1008 | 179.2 | 174.0 | 213 | 264 | 142 | 370 | 180.6 |
| `decode_attention_residual` | 1008 | 153.9 | 147.0 | 193 | 221 | 115 | 317 | 155.1 |
| `decode_post_attention_norm` | 1008 | 161.4 | 155.0 | 195 | 215 | 118 | 289 | 162.7 |
| `decode_mlp_path` | 1008 | 279.3 | 274.0 | 314 | 459 | 228 | 622 | 281.5 |
| `decode_layer_output` | 1008 | 153.0 | 147.0 | 196 | 215 | 118 | 285 | 154.3 |

### Full-Attention Inner Path

| stage | count | mean us | p50 us | p95 us | p99 us | min us | max us | total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `decode_attention_input` | 1008 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.0 |
| `decode_qkv_proj` | 1008 | 189.1 | 185.0 | 220 | 257 | 153 | 383 | 190.6 |
| `decode_q_split_norm_reshape` | 1008 | 161.7 | 158.0 | 191 | 225 | 126 | 312 | 163.0 |
| `decode_query_turbo_inputs` | 1008 | 25.2 | 0.0 | 201 | 230 | 0 | 260 | 25.4 |
| `decode_query_turbo_rotation` | 1008 | 164.2 | 162.0 | 194 | 234 | 128 | 307 | 165.6 |
| `qk` | 1008 | 760.5 | 722.0 | 893 | 1377 | 651 | 2683 | 766.5 |
| `softmax` | 1008 | 183.5 | 185.0 | 222 | 249 | 135 | 318 | 184.9 |
| `weighted_v_chunk` | 1008 | 824.4 | 784.5 | 949 | 1481 | 709 | 2866 | 831.0 |
| `weighted_v_reduce` | 1008 | 186.9 | 188.0 | 225 | 245 | 136 | 422 | 188.4 |

### Linear Decoder Layers

| stage | count | mean us | p50 us | p95 us | p99 us | min us | max us | total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `decode_layer_input` | 3024 | 7.2 | 0.0 | 0 | 182 | 0 | 380 | 21.7 |
| `decode_input_norm` | 3024 | 161.8 | 160.0 | 193 | 238 | 98 | 318 | 489.3 |
| `decode_attention_path` | 3024 | 280.1 | 271.0 | 319 | 405 | 223 | 612 | 847.1 |
| `decode_attention_residual` | 3024 | 156.6 | 150.0 | 194 | 226 | 111 | 444 | 473.5 |
| `decode_post_attention_norm` | 3024 | 158.4 | 151.0 | 197 | 217 | 121 | 423 | 479.0 |
| `decode_mlp_path` | 3024 | 250.0 | 247.0 | 282 | 362 | 195 | 624 | 756.0 |
| `decode_layer_output` | 3024 | 163.4 | 164.0 | 200 | 243 | 119 | 447 | 494.2 |

## End-to-end Control

No-profile control uses the normal execution path with the same model, prompt,
decode length, and `k3v4`.

| run set | runs | valid | decode p50 ms | decode mean ms | generation p50 tps | generation mean tps |
|---|---:|---:|---:|---:|---:|---:|
| no-profile 3x64 | 3 | 3 | 1040.523 | 1034.952 | 60.546 | 60.877 |
| profile 1x64 | 1 | 1 | 3773.699 | 3773.699 | 16.694 | 16.694 |

## Conclusion

The previous `decode_attention_input` hotspot at p50 about 886 us is not a
standalone TurboQuant input construction cost. With decoder-layer boundary
probes enabled, `decode_attention_input` falls to p50 0 us and total 0 ms across
1,008 full-attention samples. The cost was upstream lazy materialization being
charged to the first eval inside attention.

The newly exposed full-layer stages are all in the low-hundreds of microseconds:
`decode_input_norm` p50 161 us, `decode_attention_path` p50 174 us,
`decode_post_attention_norm` p50 155 us, and `decode_mlp_path` p50 274 us. There
is no remaining single decoder-layer input stage in the previous 900 us class.

The largest local full-attention costs remain the packed kernels:
`weighted_v_chunk` p50 784.5 us and `qk` p50 722 us. The next optimization pass
should therefore return to the packed attention kernels and only treat decoder
layer stage probes as attribution tooling, not as normal execution-path code.

## Artifacts

- Profile trace: `decode-layer-input-attribution-profile-1x64.stderr.txt`
- Profile benchmark JSON: `decode-layer-input-attribution-profile-1x64.json`
- No-profile control JSON: `decode-layer-input-attribution-noprofile-3x64.json`
