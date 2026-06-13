# TurboQuant V Chunk Dim Sweep

Date: 2026-06-11

Branch: `codex/turboquant-vchunk-dim-sweep`

Base: `9772556 perf: optimize turboquant v chunk decode`

## Goal

Sweep `V_CHUNK_DIMS_PER_THREADGROUP` for the TurboQuant SDPA decode V chunk kernel. The previous optimization introduced grouped V-dim computation with `V_CHUNK_DIMS_PER_THREADGROUP = 2`; this pass checks whether larger V-dim groups improve long-context K3V4 decode without reintroducing long-tail kernel latency.

## Setup

- Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`
- Prompt: `docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt`
- Mode: `gs-text`
- KV quantization: `k3v4`
- Prefill chunk size: `2048`
- Batch max: `1`
- Stability run: `--max-tokens 32 --runs 5 --warmup-runs 1`
- Profile run: `IRONMLX_TURBOQUANT_ATTN_PROFILE=1 --max-tokens 8 --runs 3 --warmup-runs 1`

## Result

Keep `V_CHUNK_DIMS_PER_THREADGROUP = 8`.

`vdim8` is the best end-to-end choice in the same 5-run stability setup. `vdim16` slightly reduces isolated `weighted_v_chunk` time, but it increases `q_rotate` and `qk` enough to lose overall decode throughput.

Compared with the previous `vdim2` baseline:

- Generation TPS: `46.935441` -> `60.597760` (+29.1%)
- Decode time: `660.696275 ms` -> `511.859666 ms` (-22.5%)
- Profiled `weighted_v_chunk` mean: `1476.3 us` -> `864.9 us` (-41.4%)
- No profiled stage had events >= 10 ms.

Compared with `vdim16` in the same 5-run stability setup:

- Generation TPS: `56.948455` -> `60.597760` (+6.4%)
- `weighted_v_chunk` mean is 3.2% higher, but `q_rotate` total is 12.0% lower and `qk` total is 12.2% lower.

## Core Bench

| Candidate | Runs | Max tokens | TPS mean | Decode mean (ms) | TPS CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdim2 | 5 | 32 | 46.935441 | 660.696275 | 2.007% |
| vdim4 | 1 | 32 | 55.558327 | 557.972167 | n/a |
| vdim8 | 1 | 32 | 62.151385 | 498.782125 | n/a |
| vdim8 | 5 | 32 | 60.597760 | 511.859666 | 2.647% |
| vdim16 | 1 | 32 | 66.163722 | 468.534709 | n/a |
| vdim16 | 5 | 32 | 56.948455 | 544.452167 | 1.520% |

## Profile

Three-run profile, 8 generated tokens per run.

| Candidate | Stage | Total (ms) | Mean (us) | Median (us) | p95 (us) | Max (us) | >=10 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdim2 | weighted_v_chunk | 330.698 | 1476.3 | 1465 | 1566 | 1741 | 0 |
| vdim2 | q_rotate | 214.419 | 957.2 | 964 | 1021 | 4693 | 0 |
| vdim2 | qk | 179.460 | 801.2 | 798 | 900 | 1047 | 0 |
| vdim8 | weighted_v_chunk | 193.727 | 864.9 | 861 | 957 | 1190 | 0 |
| vdim8 | q_rotate | 224.046 | 1000.2 | 1030 | 1140 | 1283 | 0 |
| vdim8 | qk | 185.688 | 829.0 | 824 | 919 | 1113 | 0 |
| vdim16 | weighted_v_chunk | 187.712 | 838.0 | 831 | 949 | 1107 | 0 |
| vdim16 | q_rotate | 254.488 | 1136.1 | 1124 | 1404 | 1594 | 0 |
| vdim16 | qk | 211.388 | 943.7 | 934 | 1060 | 1240 | 0 |

## Notes

- The last generated token varied between two known tie-sensitive continuations at token 31. The prefix before that point matched, and this pattern was already observed in the earlier V chunk optimization runs.
- Single-run warm results favored `vdim16`, but the 5-run stability data did not. The retained value is based on the repeated long-context K3V4 decode benchmark.
- Larger V-dim groups reduce the number of V chunk threadgroups, but excessive grouping appears to increase pressure on adjacent decode stages.

## Artifacts

- Core bench JSON/time files: `ctx-32k/core-bench/`
- Profile JSON/stderr files: `ctx-32k/profile/`
