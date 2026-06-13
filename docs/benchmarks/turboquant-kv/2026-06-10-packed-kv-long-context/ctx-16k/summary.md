# TurboQuant KV Long Context: ctx-16k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-packed-kv/docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-16k.txt`
- prompt_tokens: `18727`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00352 alpha=12 beta=04 gamma=08`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.836769 | 0.234906 | 0.298463 | 0.936246200 | 4.44 |
| k3v4 | 30 | 30 | 31/32 | 1.687241 | 0.248596 | 0.312586 | 0.963063200 | 4.47 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 5201.97 | 5445.32 | 127.39 | 2.447 | 11.377 | 32 |
| turbo4 | 32 | 5323.95 | 5757.16 | 71.56 | 2.448 | 12.440 | 32 |
| k3v4 | 32 | 5281.26 | 5718.69 | 70.87 | 2.448 | 12.422 | 32 |

## Generated Text

- `none`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
