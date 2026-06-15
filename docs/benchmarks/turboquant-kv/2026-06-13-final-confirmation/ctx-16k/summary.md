# TurboQuant KV Long Context: ctx-16k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `/Users/xin/workspace/ironmlx-backend/docs/benchmarks/turboquant-kv/2026-06-13-final-confirmation/prompts/ctx-16k.txt`
- prompt_tokens: `18727`
- logits_max_tokens: `32`
- expected: `CHECKSUM record=00352 alpha=12 beta=04 gamma=08`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo3 | 32 | none | 32/32 | 2.329102 | 0.358908 | 0.448574 | 0.940926600 | 4.19 |
| turbo4 | 32 | none | 32/32 | 1.214224 | 0.182251 | 0.229105 | 0.987300460 | 4.62 |
| k3v4 | 30 | 30 | 31/32 | 1.742615 | 0.259866 | 0.326450 | 0.956303660 | 4.56 |

## Core Generation

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | peak footprint GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | 5297.02 | 5555.41 | 119.97 | 2.461 | 11.377 | 32 |
| turbo3 | 32 | 5493.70 | 5888.77 | 78.47 | 2.450 | 11.513 | 32 |
| turbo4 | 32 | 5467.00 | 5850.03 | 80.94 | 2.451 | 11.555 | 32 |
| k3v4 | 32 | 5522.25 | 5887.82 | 84.80 | 2.452 | 11.533 | 30 |

## Generated Text

- `none`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `turbo3`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `turbo4`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate long-context
- `k3v4`: CHECKSUM record=00352 alpha=12 beta=04 gamma=08 <think> The user wants me to validate the KV
