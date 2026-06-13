# TurboQuant KV Long Context: ctx-8k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend/docs/benchmarks/turboquant-kv/2026-06-13-final-confirmation/prompts/ctx-8k.txt`
- prompt_tokens: `9399`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00176 alpha=06 beta=02 gamma=04`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo3 | 32 | none | 32/32 | 2.333649 | 0.348072 | 0.435215 | 0.943033930 | 4.34 |
| turbo4 | 32 | none | 32/32 | 1.218620 | 0.181959 | 0.228423 | 0.985397500 | 4.69 |
| k3v4 | 32 | none | 32/32 | 1.800476 | 0.288042 | 0.355240 | 0.961921500 | 4.53 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 2269.31 | 2499.77 | 134.51 | 2.457 | 7.357 | 32 |
| turbo3 | 32 | 2376.21 | 2677.11 | 103.02 | 2.461 | 7.327 | 32 |
| turbo4 | 32 | 2454.42 | 2750.04 | 104.87 | 2.461 | 7.344 | 32 |
| k3v4 | 32 | 2480.05 | 2784.11 | 101.95 | 2.447 | 7.334 | 32 |

## Generated Text

- `none`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `turbo3`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
