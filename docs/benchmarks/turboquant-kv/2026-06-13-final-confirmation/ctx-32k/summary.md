# TurboQuant KV Long Context: ctx-32k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend/docs/benchmarks/turboquant-kv/2026-06-13-final-confirmation/prompts/ctx-32k.txt`
- prompt_tokens: `37383`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00704 alpha=07 beta=08 gamma=16`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo3 | 32 | none | 32/32 | 2.372650 | 0.351006 | 0.439646 | 0.944042400 | 4.38 |
| turbo4 | 32 | none | 32/32 | 1.275455 | 0.189946 | 0.238245 | 0.984870800 | 4.59 |
| k3v4 | 30 | 30 | 31/32 | 1.684570 | 0.252093 | 0.316290 | 0.974615300 | 4.50 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 14380.90 | 14679.74 | 103.73 | 2.472 | 29.196 | 32 |
| turbo3 | 32 | 14979.83 | 15547.30 | 54.63 | 2.462 | 30.204 | 32 |
| turbo4 | 32 | 14764.36 | 15316.02 | 56.19 | 2.477 | 30.411 | 32 |
| k3v4 | 32 | 14594.17 | 15131.22 | 57.72 | 2.477 | 30.293 | 30 |

## Generated Text

- `none`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `turbo3`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00704 alpha=07 beta=08 gamma=16 <think> The user wants me to validate the KV
