# TurboQuant KV Long Context: ctx-4k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-kv/docs/benchmarks/turboquant-kv/2026-06-10-long-context/prompts/ctx-4k.txt`
- prompt_tokens: `4735`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00088 alpha=03 beta=01 gamma=02`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.503525 | 0.210268 | 0.265584 | 0.987734560 | 4.47 |
| k3v4 | 32 | none | 32/32 | 2.057101 | 0.308833 | 0.385300 | 0.968668340 | 4.44 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 1048.94 | 1263.01 | 144.81 | 2.452 | 5.754 | 32 |
| turbo4 | 32 | 1064.66 | 1487.77 | 73.27 | 2.456 | 6.613 | 32 |
| k3v4 | 32 | 1064.56 | 1479.19 | 74.77 | 2.453 | 6.604 | 24 |

## Generated Text

- `none`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> Thinking Process: 1. **
