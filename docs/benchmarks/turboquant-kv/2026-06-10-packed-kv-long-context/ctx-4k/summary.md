# TurboQuant KV Long Context: ctx-4k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-packed-kv/docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-4k.txt`
- prompt_tokens: `4735`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00088 alpha=03 beta=01 gamma=02`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 24 | 24 | 31/32 | 1.509163 | 0.205742 | 0.259366 | 0.984687270 | 4.41 |
| k3v4 | 32 | none | 32/32 | 2.025585 | 0.301722 | 0.377176 | 0.967960830 | 4.41 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 1045.66 | 1258.48 | 145.66 | 2.441 | 5.758 | 32 |
| turbo4 | 32 | 1052.55 | 1322.25 | 114.94 | 2.456 | 6.075 | 32 |
| k3v4 | 32 | 1052.22 | 1322.04 | 114.89 | 2.442 | 6.069 | 24 |

## Generated Text

- `none`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> Thinking Process: 1. **
