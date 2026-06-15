# TurboQuant KV Long Context: ctx-16k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-kv/docs/benchmarks/turboquant-kv/2026-06-10-long-context/prompts/ctx-16k.txt`
- prompt_tokens: `18727`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00352 alpha=12 beta=04 gamma=08`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.307396 | 0.191492 | 0.240915 | 0.971678900 | 4.66 |
| k3v4 | 30 | 30 | 31/32 | 1.656540 | 0.252955 | 0.317293 | 0.967671800 | 4.53 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 5248.18 | 5496.15 | 125.01 | 2.462 | 11.378 | 32 |
| turbo4 | 32 | 5502.66 | 6528.13 | 30.23 | 2.462 | 14.565 | 32 |
| k3v4 | 32 | 5442.60 | 6436.61 | 31.19 | 2.459 | 14.477 | 30 |

## Generated Text

- `none`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate the KV
