# TurboQuant Packed Attention: ctx-32k

- model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- prompt_tokens: `37383`
- generated tokens: `32`
- expected: `CHECKSUM record=00704 alpha=07 beta=08 gamma=16`

## Logits Replay

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 32 | none | 32/32 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| turbo4 | 32 | none | 32/32 | 1.240753 | 0.186021 | 0.233575 | 0.985301500 | 4.72 |
| k3v4 | 30 | 30 | 31/32 | 1.680603 | 0.243220 | 0.306149 | 0.972530200 | 4.53 |

## Core Generation

| scheme | kv | ttft p50 ms | e2e p50 ms | decode p50 ms | tps p50 | max RSS GiB | peak footprint GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| solution 1 rerun | turbo4 | 11301.01 | 11935.13 | 634.12 | 48.89 | 2.460 | 32.167 |
| solution 1 rerun | k3v4 | 11309.57 | 11943.67 | 634.10 | 48.89 | 2.460 | 32.071 |
| solution 2 | none | 13517.20 | 13810.96 | 293.76 | 105.53 | 2.454 | 29.192 |
| solution 2 | turbo4 | 13461.66 | 33032.95 | 19571.29 | 1.58 | 2.472 | 30.388 |
| solution 2 | k3v4 | 11617.16 | 30851.63 | 19234.47 | 1.61 | 2.475 | 30.274 |

## Conclusion

Packed decode attention reduces peak footprint by about `1.8 GiB` compared with the packed-KV-only branch on this ctx-32k workload, but the current kernel regresses decode throughput by about `30x`.
