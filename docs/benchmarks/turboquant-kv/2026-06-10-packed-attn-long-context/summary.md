# TurboQuant Packed Attention Long Context Validation

- root: `/Users/xin/workspace/ironmlx-backend-turboquant-packed-attn/docs/benchmarks/turboquant-kv/2026-06-10-packed-attn-long-context`
- branch: `codex/turboquant-packed-attn`
- scheme: solution 2, packed K/V cache plus packed decode attention
- baseline rerun: solution 1 at `/Users/xin/workspace/ironmlx-backend-turboquant-packed-kv`, commit `6433c3e`
- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: reused from `../2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`

## Result

Solution 2 proves that direct packed decode attention can reduce ctx-32k peak footprint, but the current single-kernel implementation is much slower than solution 1.

| kv | solution 1 peak GiB | solution 2 peak GiB | delta GiB | solution 1 tps | solution 2 tps | recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| turbo4 | 32.167 | 30.388 | -1.779 | 48.89 | 1.58 | prefer solution 1 for now |
| k3v4 | 32.071 | 30.274 | -1.797 | 48.89 | 1.61 | prefer solution 1 for now |

## ctx-32k Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.240753 | 0.186021 | 0.233575 | 0.985301500 | 4.72 |
| k3v4 | 30 | 30 | 31/32 | 1.680603 | 0.243220 | 0.306149 | 0.972530200 | 4.53 |

## ctx-32k Core Generation

| scheme | kv | generated tokens | ttft p50 ms | e2e p50 ms | decode p50 ms | tps p50 | max RSS GiB | peak footprint GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| solution 1 rerun | turbo4 | 32 | 11301.01 | 11935.13 | 634.12 | 48.89 | 2.460 | 32.167 |
| solution 1 rerun | k3v4 | 32 | 11309.57 | 11943.67 | 634.10 | 48.89 | 2.460 | 32.071 |
| solution 2 | none | 32 | 13517.20 | 13810.96 | 293.76 | 105.53 | 2.454 | 29.192 |
| solution 2 | turbo4 | 32 | 13461.66 | 33032.95 | 19571.29 | 1.58 | 2.472 | 30.388 |
| solution 2 | k3v4 | 32 | 11617.16 | 30851.63 | 19234.47 | 1.61 | 2.475 | 30.274 |

## Generated Text

- `solution 2 none`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context
- `solution 2 turbo4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate long-context
- `solution 2 k3v4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 `<think>` The user wants me to validate the KV

## Notes

- `peak footprint GiB` comes from macOS `/usr/bin/time -l`.
- Solution 2 only changes the decode attention path. Prefill still uses the existing attention path and may materialize packed cache when attention needs dense K/V.
- The current packed decode Metal kernel computes a complete query/head largely in one thread. It avoids dense K/V materialization, but it does not yet exploit enough parallelism across sequence length and head dimension, which explains the large decode latency regression.
