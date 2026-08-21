# Qwen/Gemma4 MTP Policy Split 32K Performance Gate

Date: 2026-08-21

Environment: Apple M5 Max, 128 GB unified memory. The candidate is the
uncommitted policy-split tree based on `3632460e`. Qwen is compared with the
pre-Gemma4 baseline `f8ff3f8`; Gemma4 is compared directly with `3632460e`.

## Scope and method

- Fixed 32,768-token prompt and 64 generated tokens.
- Greedy decoding with EOS ignored, one warmup, and one measured run per
  paired command.
- Qwen used the checkpoint chat template. Gemma4 used the historical rendered
  raw-prompt repetition so that the `3632460e` comparison retained the same
  input-token semantics.
- Throughput is aggregate output tokens per second from `scheduler-text`.
- Correctness requires exact generated-token equality between ordinary decode
  and MTP/drafter output.
- Qwen B1 values are single paired directional measurements. Gemma4 MTP was
  measured twice in reverse commit order; the two-sample median is the
  midpoint of the two measurements.

## Checkpoints

| Architecture | Base | Draft model |
|---|---|---|
| Qwen3.5 Dense | `mlx-community/Qwen3.5-4B-MLX-4bit` | `mlx-community/Qwen3.5-4B-MTP-4bit` |
| Qwen3.6 Dense | `mlx-community/Qwen3.6-27B-4bit` | `mlx-community/Qwen3.6-27B-MTP-4bit` |
| Qwen3.6 MoE | `mlx-community/Qwen3.6-35B-A3B-4bit` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` |
| Qwen3.8 Dense | `mlx-community/Qwen3.8-27B-4bit` | `mlx-community/Qwen3.8-27B-MTP-4bit` |
| Gemma4 Dense | `mlx-community/gemma-4-12B-it-4bit` | `mlx-community/gemma-4-12B-it-assistant-4bit` |
| Gemma4 MoE | `mlx-community/gemma-4-26b-a4b-it-8bit` | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` |

## Qwen B1 results

| Architecture | `f8ff3f8` MTP | Candidate ordinary | Candidate MTP | Candidate MTP delta | MTP vs `f8ff3f8` | Correctness |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5 Dense | 63.767 | 99.612 | 99.351 | -0.26% | +55.80% | Exact |
| Qwen3.6 Dense | 23.436 | 24.077 | 25.003 | +3.85% | +6.68% | Exact |
| Qwen3.6 MoE | 68.944 | 92.658 | 94.569 | +2.06% | +37.17% | Exact |
| Qwen3.8 Dense | 23.125 | 23.357 | 25.274 | +8.20% | +9.29% | Exact |

The candidate removes all measured negative-MTP cases from `f8ff3f8`.
Qwen3.5 B1 is throughput-neutral against its faster candidate ordinary path,
while remaining substantially faster than the historical MTP path.

## Qwen B2 results

| Architecture | Candidate ordinary | Candidate MTP | Delta | Correctness | `f8ff3f8` result |
|---|---:|---:|---:|---|---|
| Qwen3.5 Dense | 130.834 | 138.416 | +5.80% | Exact | Memory admission rejected |
| Qwen3.6 Dense | 34.189 | 35.321 | +3.31% | Exact | Memory admission rejected |
| Qwen3.6 MoE | 127.828 | 136.501 | +6.79% | Exact | Memory admission rejected |
| Qwen3.8 Dense | 34.886 | 35.286 | +1.14% | Exact | Memory admission rejected |

The candidate's row-isolated prefill path makes all four 32K B2 combinations
admissible without disabling the process-memory guard.

## Gemma4 cross-commit B1 results

| Architecture | `3632460e` run 1 | `3632460e` run 2 | `3632460e` median | Candidate run 1 | Candidate run 2 | Candidate median | Median delta | Correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 37.487 | 37.892 | 37.690 | 37.617 | 39.517 | 38.567 | +2.33% | Exact |
| MoE | 35.830 | 37.556 | 36.693 | 36.690 | 36.940 | 36.815 | +0.33% | Exact |

The first candidate pair also measured ordinary decode at 32.260 tok/s for
Dense and 30.464 tok/s for MoE. Candidate drafter throughput was respectively
37.617 tok/s (+16.61%) and 36.690 tok/s (+20.43%). The MoE cross-commit margin
is within normal run-to-run noise and is evidence of no measurable regression,
not a statistically significant speedup.

Gemma4 Dense and MoE B2 were rejected by the same 103,903,852,953-byte prefill
peak target on both `3632460e` and the candidate. This is an unchanged safety
capacity boundary, not a candidate regression.

## Command shape

```sh
target/release/ironmlx-core-bench \
  --model <base-checkpoint> \
  --mtp-model-dir <draft-checkpoint> \
  --prompt-file docs/benchmarks/mtp-phase3-performance/2026-06-07-141108/fixed_prompt.txt \
  --prompt-target-tokens 32768 \
  --ignore-eos \
  --mode scheduler-text \
  --max-tokens 64 \
  --warmup-runs 1 \
  --runs 1 \
  --b-max 1 \
  --scheduler-baseline-out <baseline.json> \
  --out <mtp.json>
```

Qwen runs add `--chat`. B2 repeats `--prompt-file` and sets `--b-max 2`.

## Evidence boundaries

- Raw JSON files are intentionally not committed; this document records the
  durable aggregate evidence, exact command shape, checkpoint identities, and
  correctness outcome.
- The gate covers 32K only, as selected for the final regression pass. It does
  not replace broader short/8K/64K release characterization.
- Qwen results are single paired directional measurements. Gemma4 uses two MTP
  measurements per commit, which is sufficient to detect the observed
  regression class but not to claim small statistically significant gains.
- Results apply to the listed checkpoints, Apple M5 Max, and the local MLX
  runtime used on 2026-08-21.
