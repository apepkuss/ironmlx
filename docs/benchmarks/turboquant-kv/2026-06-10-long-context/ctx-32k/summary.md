# TurboQuant KV Long Context: ctx-32k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-kv/docs/benchmarks/turboquant-kv/2026-06-10-long-context/prompts/ctx-32k.txt`
- prompt_tokens: `37383`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00704 alpha=07 beta=08 gamma=16`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.264408 | 0.183222 | 0.230557 | 0.985120700 | 4.66 |
| k3v4 | 30 | 30 | 31/32 | 1.679054 | 0.249005 | 0.312447 | 0.967162550 | 4.53 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 14140.96 | 14447.83 | 101.02 | 2.473 | 29.197 | 32 |
| turbo4 | 32 | 14901.36 | 16814.57 | 16.20 | 2.469 | 37.518 | 32 |
| k3v4 | 32 | 15017.24 | 16813.10 | 17.26 | 2.474 | 36.942 | 30 |

## Generated Text

- `none`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate the KV
