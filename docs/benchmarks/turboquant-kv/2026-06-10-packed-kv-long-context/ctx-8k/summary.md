# TurboQuant KV Long Context: ctx-8k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-packed-kv/docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-8k.txt`
- prompt_tokens: `9399`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00176 alpha=06 beta=02 gamma=04`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.243233 | 0.193541 | 0.240586 | 0.985789300 | 4.62 |
| k3v4 | 32 | none | 32/32 | 1.801361 | 0.281403 | 0.349155 | 0.961803140 | 4.59 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 2173.14 | 2397.55 | 138.14 | 2.442 | 7.356 | 32 |
| turbo4 | 32 | 2239.02 | 2564.88 | 95.13 | 2.443 | 8.022 | 32 |
| k3v4 | 32 | 2289.19 | 2612.18 | 95.98 | 2.443 | 8.012 | 32 |

## Generated Text

- `none`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
