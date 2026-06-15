# TurboQuant KV Long Context: ctx-8k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-kv/docs/benchmarks/turboquant-kv/2026-06-10-long-context/prompts/ctx-8k.txt`
- prompt_tokens: `9399`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00176 alpha=06 beta=02 gamma=04`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 24 | 24 | 31/32 | 1.307678 | 0.191432 | 0.240678 | 0.974690900 | 4.69 |
| k3v4 | 32 | none | 32/32 | 1.852743 | 0.282299 | 0.350597 | 0.945408640 | 4.41 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 2150.71 | 2376.48 | 137.31 | 2.457 | 7.357 | 32 |
| turbo4 | 32 | 2274.17 | 2898.28 | 49.67 | 2.458 | 9.193 | 32 |
| k3v4 | 32 | 2340.51 | 2977.71 | 48.65 | 2.458 | 9.168 | 30 |

## Generated Text

- `none`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00176 alpha=06 beta=02 gamma=04 <think> The user wants me to validate the KV
