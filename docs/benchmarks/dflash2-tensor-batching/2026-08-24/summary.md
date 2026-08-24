# DFlash2 Tensor Batching B=N Performance Archive

Date: 2026-08-24

Status: passed. Wide DFlash2 tensor batching preserves exact output records and
improves sustained Greedy aggregate throughput. The certified default maximum
width remains 4.

## Baseline

| Component | Revision or configuration |
|---|---|
| IronMLX | `feat/dflash2-execution-path@9bbee41061afe611a102deedd6ec388d4990d302` |
| B2 kernel qualification | `331b0d05cc745204a079d6c5f39b36b5f4db9822` |
| Pairwise tensor batching | `fc28af6a6568ae7bccb60874550ad47c84aa9045` |
| B=N tensor batching | `9bbee41061afe611a102deedd6ec388d4990d302` |
| MLX | local NAX/QMM baseline `73ad5df20cb30be4192e5c4d0ae8130674773427` |
| macOS | `26.4` (`25E246`) |
| Hardware | MacBook Pro `Mac17,6`, Apple M5 Max, 18 CPU cores, 128 GB unified memory |
| Target | `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

All servers used DFlash2 block size 4, draft 4-bit, `max-sequences=8` and
`max-cache-cap=131072`. The lower cache cap is required by the startup memory
admission gate for C8 on this machine; `max-cache-cap=262144` was correctly
rejected before model startup.

This archive extends the earlier [P3 final validation](../../dflash2-final-validation/2026-08-23/summary.md).
The P3 record remains the ordinary/MTP/DFlash2 B1 functional baseline; this
record qualifies the later cross-request tensor execution path. The preceding
[Pairwise/B2 historical record](../2026-08-24-pairwise-b2/summary.md) preserves
the fixed-width implementation and its request-rotation baseline.

## Short-Prompt Measurement Contract

- One fixed short Rust task; raw prompt text is not retained in this archive.
- 128 forced output tokens with EOS ignored.
- Greedy: `temperature=0`, `top_p=1`.
- Sampled: `temperature=0.7`, `top_p=1`, checkpoint default `top_k=20`.
- Fixed base seed `20260824`; each worker uses `base_seed + worker`.
- Concurrency C1/C2/C4/C8 and configured width limits B1/B2/B4/B8.
- One warmup closed batch, followed by three measured closed batches per cell.
- Aggregate TPS is total completion tokens divided by whole closed-batch wall
  time, so requests crossing an arbitrary duration boundary cannot inflate it.
- Width 1 disables cross-request tensor batching while preserving actor-level
  request concurrency.

The sanitized command shape was:

```text
ironmlx serve ... --max-sequences 8 --max-cache-cap 131072 \
  --dflash2-block-size 4 --dflash2-draft-bits 4 \
  --dflash2-tensor-batch-max-width <1|2|4|8>
python3 scripts/benchmark_dflash2_tensor_batching_gate.py \
  --concurrency 1,2,4,8 --modes greedy,sampled \
  --max-tokens 128 --warmup-batches 1 --measured-batches 3
```

## Sustained Aggregate Throughput

### Greedy

| Width limit | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| B1 | 38.78 | 40.39 | 40.29 | 36.98 |
| B2 | 39.41 | 44.58 | 44.48 | 40.72 |
| B4 | 38.40 | 43.80 | 46.04 | 42.59 |
| B8 | 37.59 | 43.15 | 45.65 | 43.18 |

### Sampled (`top_k=20`)

| Width limit | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| B1 | 32.27 | 31.73 | 31.75 | 31.17 |
| B2 | 31.96 | 31.75 | 31.91 | 31.64 |
| B4 | 31.99 | 31.61 | 32.24 | 32.24 |
| B8 | 32.38 | 32.11 | 32.52 | 32.32 |

At C8, B4 improves Greedy throughput by 15.2% and Sampled throughput by 3.4%
relative to B1. B8 improves over B4 by only 1.4% Greedy and 0.3% Sampled,
which is not large enough to establish a stable advantage over run-state
variation.

## Memory and Scheduler Evidence

| Width limit | Maximum observed width | Peak memory | Tensor windows | Groups created | Divergent splits |
|---:|---:|---:|---:|---:|---:|
| B1 | 0 | 19.48 GiB | 0 | 0 | 0 |
| B2 | 2 | 20.39 GiB | 1692 | 150 | 198 |
| B4 | 4 | 21.72 GiB | 1120 | 132 | 192 |
| B8 | 8 | 23.60 GiB | 940 | 124 | 188 |

B8 consumes approximately 1.88 GiB more peak memory than B4 while providing
only a marginal C8 throughput change. This supports the certified default
maximum width of 4. A user-supplied `--dflash2-tensor-batch-max-width` remains an
advanced safety cap, and the actual group width is also bounded by
`max_sequences` and the number of ready compatible requests.

## Long-Context Control

`iron-bench` used a synthetic 4,096-token prompt, 64 forced Greedy output
tokens, C8, 10 seconds configured warmup and 45 seconds configured measurement.
The runner waits for in-flight requests at phase boundaries, so each cell
completed eight full requests.

| Width limit | Requests | P50 TTFT | P95 TTFT | P50 ITL | P95 first-8 ITL | Aggregate tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| B1 | 8 | 60,522.8 ms | 60,549.4 ms | 340.06 ms | 175.47 ms | 11.4 |
| B4 | 8 | 60,085.7 ms | 60,091.6 ms | 356.67 ms | 161.20 ms | 11.4 |

The approximately 60-second C8 prefill dominates this scenario. Tensor
batching optimizes draft/target verification, so it neither materially regresses
nor resolves the long-context prefill bottleneck.

## Correctness

- Every short-prompt cell produced exactly 128 output tokens per row.
- B1/B2/B4/B8 Greedy and fixed-seed Sampled output SHA-256 records matched
  exactly across width configurations.
- The flattened output-token/hash record digest was
  `bab46e5071d3d2f0d8f40075d893155ee11185233451c79a6163856139eed6c4`
  for every width report.
- The previously validated heterogeneous-row, JSON-schema, forced-tool and
  streaming-cancellation cases remain the functional correctness gates; this
  archive is the performance qualification layer.

## Run-State Bias Exclusion

The first cold B1 screening run was excluded from the sustained matrix. Its C1
median was 59.01 tok/s, while its three C8 Greedy batches declined from 50.19 to
46.68 and 43.21 tok/s. After sustained load, B1 C1 re-measured at 38.78 tok/s,
consistent with the later-width C1 range of 37.59-39.41 tok/s.

This is evidence of order and sustained-load state bias, not evidence that width
2 regressed the single-request path. The accepted matrix therefore uses the hot
B1 repeat and the subsequent B2/B4/B8 runs. No public claim should combine the
excluded cold screening numbers with the sustained matrix.

## Evidence Boundaries

- Results apply only to the pinned machine, revisions, checkpoints and request
  parameters above.
- Aggregate TPS is not a universal per-request TPS promise.
- Sampled results retain the model default `top_k=20`; they are not comparable
  with earlier no-top-k engineering experiments.
- Prefix-cache state was disabled for the short tensor matrix. Prefix-cache
  performance is archived separately.
- CSV scheduler counters, peak memory and last-request acceptance are
  cumulative or end-of-run health snapshots for each width run. They are
  repeated across that width's rows for joinability and are not per-cell
  counters.
- The data supports a default maximum width of 4 on this hardware. It does not
  establish a universal optimum for all Apple Silicon generations.
- No absolute local paths, raw prompts or generated response bodies are part of
  this candidate archive.

Detailed sanitized rows are available in [`summary.csv`](summary.csv).
