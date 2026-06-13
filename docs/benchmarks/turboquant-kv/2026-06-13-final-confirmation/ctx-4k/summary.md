# TurboQuant KV Long Context: ctx-4k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend/docs/benchmarks/turboquant-kv/2026-06-13-final-confirmation/prompts/ctx-4k.txt`
- prompt_tokens: `4735`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00088 alpha=03 beta=01 gamma=02`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo3 | 32 | none | 32/32 | 2.536381 | 0.403560 | 0.501337 | 0.962342000 | 4.31 |
| turbo4 | 24 | 24 | 31/32 | 1.471050 | 0.211518 | 0.265665 | 0.989537000 | 4.38 |
| k3v4 | 32 | none | 32/32 | 2.087173 | 0.314053 | 0.392127 | 0.965126160 | 4.38 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 1078.67 | 1294.41 | 143.69 | 2.457 | 5.759 | 32 |
| turbo3 | 32 | 1061.01 | 1316.01 | 121.57 | 2.461 | 5.743 | 32 |
| turbo4 | 32 | 1082.56 | 1338.93 | 120.92 | 2.445 | 5.751 | 32 |
| k3v4 | 32 | 1059.41 | 1313.81 | 121.85 | 2.460 | 5.746 | 24 |

## Generated Text

- `none`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `turbo3`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00088 alpha=03 beta=01 gamma=02 <think> Thinking Process: 1. **
