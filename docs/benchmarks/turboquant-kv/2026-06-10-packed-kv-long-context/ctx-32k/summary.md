# TurboQuant KV Long Context: ctx-32k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-packed-kv/docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- prompt_tokens: `37383`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00704 alpha=07 beta=08 gamma=16`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.285767 | 0.189694 | 0.238520 | 0.982761600 | 4.66 |
| k3v4 | 30 | 30 | 31/32 | 1.724365 | 0.249579 | 0.313427 | 0.973182500 | 4.47 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 13445.38 | 13738.60 | 105.72 | 2.459 | 29.196 | 32 |
| turbo4 | 32 | 13667.01 | 14323.92 | 47.19 | 2.459 | 32.166 | 32 |
| k3v4 | 32 | 13747.83 | 14403.96 | 47.25 | 2.460 | 32.071 | 30 |

## Generated Text

- `none`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate the KV
