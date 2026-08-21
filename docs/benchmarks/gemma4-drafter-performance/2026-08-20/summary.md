# Gemma4 Drafter Performance Matrix

Date: 2026-08-20

Environment: Apple M5 Max, 128 GB unified memory. Results compare ordinary
greedy decode with the matching Gemma4 assistant drafter and report aggregate
output throughput.

## Checkpoints

| Architecture | Base | Assistant |
|---|---|---|
| Dense | `mlx-community/gemma-4-12B-it-4bit` | `mlx-community/gemma-4-12B-it-assistant-4bit` |
| MoE | `mlx-community/gemma-4-26b-a4b-it-8bit` | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` |

## Results

| Context | Architecture | Batch | KV mode | Baseline tok/s | Drafter tok/s | Delta | Correctness |
|---:|---|---:|---|---:|---:|---:|---|
| Short | Dense | B1 | Default | 48.656 | 49.044 | +0.80% | Exact |
| Short | Dense | B2 | Default | 85.274 | 85.216 | -0.07% | Exact |
| Short | MoE | B1 | Default | 48.887 | 49.280 | +0.80% | Exact |
| Short | MoE | B2 | Default | 85.422 | 84.680 | -0.87% | Exact |
| 8K | Dense | B1 | Default | 34.490 | 38.889 | +12.76% | Exact |
| 8K | Dense | B2 | Default | 52.626 | 56.924 | +8.17% | Exact |
| 8K | MoE | B1 | Default | 34.660 | 75.502 | +117.84% | Exact |
| 8K | MoE | B2 | Default | 54.717 | 101.355 | +85.23% | Exact |
| 32K | Dense | B1 | Default | 27.583 | 39.129 | +41.86% | Exact |
| 32K | Dense | B2 | Default | 28.542 | 33.943 | +18.92% | Exact |
| 32K | MoE | B1 | Default | 23.652 | 40.970 | +73.22% | Exact |
| 32K | MoE | B2 | Default | 35.587 | 51.536 | +44.82% | Exact |
| 64K | Dense | B1 | Default | 11.763 | 40.487 | +244.21% | Exact |
| 64K | Dense | B2 | K3V4 | N/A | N/A | N/A | Rejected by memory admission |
| 64K | MoE | B1 | K3V4 | 4.171 | 7.229 | +73.32% | Exact |
| 64K | MoE | B2 | Default | 3.396 | 10.887 | +220.53% | Exact; diagnostic only |

## Evidence Boundaries

- Short-context rows used six measured runs after two warmups. Low acceptance
  caused the cost-aware policy to remain close to ordinary decode.
- Long-context rows are single paired runs and provide directional evidence,
  not a stable release benchmark.
- The 64K MoE B2 row was a high-memory diagnostic run performed before the
  final admission guard. It must not be used as a production capacity claim.
- Dense 64K B2 exceeds the safe memory budget on this 128 GB machine and now
  fails admission without executing the unsafe workload.
- A final real-checkpoint K3V4 gate verified exact Dense 8K output for B1 and
  B2 with multi-token verification.
- The unexpected machine restart removed temporary raw JSON artifacts. This
  report preserves the recovered aggregate results and their evidence level.
