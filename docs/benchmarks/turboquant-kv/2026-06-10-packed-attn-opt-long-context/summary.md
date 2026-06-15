# TurboQuant Packed Decode Attention Optimization

- branch: `codex/turboquant-packed-attn-opt`
- base: `26f45ab` (`codex/turboquant-packed-attn`)
- model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- prompt: `ctx-32k.txt` (`37383` prompt tokens)
- max tokens: `32`

## Conclusion

The optimized solution 2 remains memory-favorable and is now meaningfully faster than the original packed-attention attempt. It is still slower than solution 1, but the decode bottleneck is no longer a dead end.

| kv | solution 1 peak GiB | optimized solution 2 peak GiB | peak delta GiB | solution 1 tps | optimized solution 2 tps | original solution 2 tps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| turbo4 | 32.167 | 30.411 | -1.756 | 48.89 | 15.31 | 1.58 |
| k3v4 | 32.071 | 30.293 | -1.778 | 48.89 | 22.30 | 1.61 |

`k3v4` is still the better mixed mode in this test: it has lower peak memory than `turbo4`, higher decode TPS, and preserves the same 32-token exact prefix as baseline.

## Implementation Notes

The original solution 2 decode kernel serialized too much work inside one query/head threadgroup. The optimized path keeps K/V packed and splits decode attention into smaller kernels:

- rotate query once into WHT/sign space
- compute packed QK scores in parallel over sequence positions
- apply MLX softmax over the score tensor
- compute weighted V in sequence chunks and reduce chunk partials before inverse WHT/sign recovery

This adds only small temporary f32 tensors for scores, weights, and V partials. It does not materialize dense K/V cache.

## ctx-32k Core Bench

| kv | generated tokens | ttft p50 ms | e2e p50 ms | decode p50 ms | tps p50 | max RSS GiB | peak footprint GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| turbo4 | 32 | 11462.27 | 13486.77 | 2024.50 | 15.31 | 2.478 | 30.411 |
| k3v4 | 32 | 11383.60 | 12773.44 | 1389.84 | 22.30 | 2.477 | 30.293 |

## Quality Check

Baseline is `kv_quant=none`; all rows use the same ctx-32k prompt and 32 generated tokens.

| kv | exact prefix vs none | first mismatch | argmax agreement | max abs logit diff | mean abs logit diff avg | rms logit diff avg | min top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| none | 32/32 | none | 1.000 | 0.000000 | 0.000000 | 0.000000 | 5 |
| turbo4 | 32/32 | none | 1.000 | 1.960938 | 0.191014 | 0.239918 | 4 |
| k3v4 | 32/32 | none | 1.000 | 3.250000 | 0.249006 | 0.312368 | 4 |

Generated text prefix:

- `none`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context

## Artifacts

- core bench JSON/time files: `ctx-32k/core-bench/`
- logits validation: `ctx-32k/logits.json`

`peak footprint GiB` comes from macOS `/usr/bin/time -l`.
