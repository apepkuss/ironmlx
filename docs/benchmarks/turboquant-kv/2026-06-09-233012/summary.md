# TurboQuant KV Validation

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend-turboquant-kv/docs/benchmarks/turboquant-kv/2026-06-09-233012/prompts/technical_summary.txt`
- prompt_tokens: `36`
- logits_max_tokens: `16`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 16 | none | 16/16 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo3 | 6 | 6 | 15/16 | 1.633057 | 0.222057 | 0.279114 | 0.988286500 | 4.56 |
| turbo4 | 16 | none | 16/16 | 0.905365 | 0.128348 | 0.161345 | 0.996525000 | 4.81 |
| k3v4 | 16 | none | 16/16 | 1.162445 | 0.154378 | 0.195189 | 0.993040400 | 4.50 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 31.05 | 234.20 | 152.56 | 2.454 | 32 |
| turbo3 | 32 | 31.58 | 242.66 | 146.89 | 2.455 | 2 |
| turbo4 | 32 | 31.74 | 243.35 | 146.44 | 2.452 | 32 |
| k3v4 | 32 | 31.60 | 243.51 | 146.29 | 2.455 | 2 |

## Generated Text

- `none`: 1. Logits drift: Compare the quantized logits to the full-precision logits. 2. Generation quality: Evaluate the generated text. 3.
- `turbo3`: 1. Compare logits drift using a reference model. 2. Compare generation quality using a metric. 3. Compare latency and memory usage. <think>
- `turbo4`: 1. Logits drift: Compare the quantized logits to the full-precision logits. 2. Generation quality: Evaluate the generated text. 3.
- `k3v4`: 1. Compare logits drift using the L2 norm of the difference between quantized and full-precision logits. 2. Compare generation quality by comparing the perplex

## Raw Artifacts

- logits replay: `logits.json`
- benchmark JSON: `core-bench/<kv>.json`
- benchmark RSS/time: `core-bench/<kv>.time.txt`
