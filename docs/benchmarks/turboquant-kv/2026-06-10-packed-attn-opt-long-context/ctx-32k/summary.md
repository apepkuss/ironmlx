# ctx-32k Packed Decode Attention Optimization

- branch: `codex/turboquant-packed-attn-opt`
- prompt tokens: `37383`
- max tokens: `32`
- measured runs: `1`

## Result

Chunked packed V accumulation improves the optimized solution 2 path again while keeping the same memory advantage.

| kv | generated tokens | ttft p50 ms | e2e p50 ms | decode p50 ms | tps p50 | max RSS GiB | peak footprint GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| turbo4 | 32 | 11462.27 | 13486.77 | 2024.50 | 15.31 | 2.478 | 30.411 |
| k3v4 | 32 | 11383.60 | 12773.44 | 1389.84 | 22.30 | 2.477 | 30.293 |

## Comparison

| kv | original solution 2 tps | optimized solution 2 tps | speedup | original solution 2 peak GiB | optimized solution 2 peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| turbo4 | 1.58 | 15.31 | 9.69x | 30.388 | 30.411 |
| k3v4 | 1.61 | 22.30 | 13.85x | 30.274 | 30.293 |

| kv | solution 1 tps | optimized solution 2 tps | solution 1 peak GiB | optimized solution 2 peak GiB | peak delta GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| turbo4 | 48.89 | 15.31 | 32.167 | 30.411 | -1.756 |
| k3v4 | 48.89 | 22.30 | 32.071 | 30.293 | -1.778 |

## Quality

| kv | exact prefix vs none | first mismatch | argmax agreement | max abs logit diff | mean abs logit diff avg | rms logit diff avg |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| turbo4 | 32/32 | none | 1.000 | 1.960938 | 0.191014 | 0.239918 |
| k3v4 | 32/32 | none | 1.000 | 3.250000 | 0.249006 | 0.312368 |

Generated text prefix:

- `turbo4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context
