# MTP Phase 2.5 llama.cpp Parity Benchmark

Run directory: `docs/benchmarks/mtp-phase2-llamacpp-parity/2026-06-07-122931`

This run uses the same fixed prompt, `--mode scheduler-text`, `--max-tokens 64`, `--runs 5`, `--warmup-runs 1`, `--prefill-chunk-size 2048`, and `--b-max 1` as the Phase 2 performance baseline. Speedup is relative to each model's non-MTP scheduler baseline in this run. `Phase2 tok/s` is copied from the previous baseline run for direct comparison.

## Result

| model | config | p50 tok/s | speedup | Phase2 tok/s | delta | p50 TTFT ms | p50 E2E ms | MTP accept | rollbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | baseline | 147.3553 | 1.0000 | 147.3648 | -0.01% | 46.4306 | 474.0090 |  |  |
| Qwen3.5-4B | mtp_d1 | 165.1942 | 1.1211 | 163.4467 | +1.07% | 52.2803 | 433.6573 | 0.7778 | 40 |
| Qwen3.5-4B | mtp_d2 | 159.5483 | 1.0827 | 122.7158 | +30.01% | 55.0725 | 449.9434 | 0.6545 | 55 |
| Qwen3.5-4B | mtp_d4 | 122.8600 | 0.8338 | 98.3304 | +24.95% | 60.7169 | 573.3394 | 0.4348 | 85 |
| Qwen3.6-27B | baseline | 31.5902 | 1.0000 | 31.5612 | +0.09% | 190.0536 | 2184.3403 |  |  |
| Qwen3.6-27B | mtp_d1 | 44.7365 | 1.4161 | 36.4494 | +22.74% | 202.0456 | 1610.3338 | 0.8529 | 25 |
| Qwen3.6-27B | mtp_d2 | 45.6837 | 1.4461 | 29.1171 | +56.90% | 224.6145 | 1603.6608 | 0.7959 | 35 |
| Qwen3.6-27B | mtp_d4 | 27.8195 | 0.8806 | 22.1093 | +25.83% | 261.8818 | 2526.8848 | 0.4941 | 85 |
| Qwen3.6-35B-A3B | baseline | 125.8773 | 1.0000 | 125.9443 | -0.05% | 71.0501 | 571.5375 |  |  |
| Qwen3.6-35B-A3B | mtp_d1 | 133.1567 | 1.0578 | 134.3595 | -0.90% | 79.5975 | 553.2375 | 0.7778 | 40 |
| Qwen3.6-35B-A3B | mtp_d2 | 130.1464 | 1.0339 | 108.1649 | +20.32% | 82.2769 | 566.3471 | 0.6667 | 65 |
| Qwen3.6-35B-A3B | mtp_d4 | 93.9428 | 0.7463 | 79.1616 | +18.67% | 88.5883 | 759.6486 | 0.4105 | 105 |

## Timing Breakdown

Values are per measured run, averaged over the five valid runs.

| model | config | draft fwd ms/run | verify fwd ms/run | sampling ms/run | mtp commit ms/run | restore us/run |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | mtp_d1 | 0.937 | 21.062 | 344.441 | 62.037 | 1.2 |
| Qwen3.5-4B | mtp_d2 | 2.071 | 19.054 | 349.405 | 74.591 | 1.2 |
| Qwen3.5-4B | mtp_d4 | 3.564 | 19.262 | 438.150 | 108.208 | 4.6 |
| Qwen3.6-27B | mtp_d1 | 1.065 | 41.059 | 1363.528 | 196.545 | 3.4 |
| Qwen3.6-27B | mtp_d2 | 1.770 | 32.787 | 1238.498 | 324.439 | 5.0 |
| Qwen3.6-27B | mtp_d4 | 3.232 | 40.402 | 1757.725 | 716.361 | 2.0 |
| Qwen3.6-35B-A3B | mtp_d1 | 1.404 | 39.814 | 409.799 | 96.624 | 4.2 |
| Qwen3.6-35B-A3B | mtp_d2 | 2.228 | 34.678 | 377.473 | 146.904 | 2.6 |
| Qwen3.6-35B-A3B | mtp_d4 | 4.319 | 40.832 | 492.885 | 215.652 | 2.8 |

## Takeaways

- The llama.cpp-aligned MTP cache commit semantics materially improve Phase 2 behavior. The largest win is Qwen3.6-27B `mtp_d2`, which moved from `0.9226x` to `1.4461x` versus baseline.
- `mtp_d2` is now the best setting for Qwen3.6-27B in this benchmark. `mtp_d1` remains the best setting for Qwen3.5-4B, and Qwen3.6-35B-A3B only shows a small positive gain with `mtp_d1`/`mtp_d2`.
- `mtp_d4` is still not viable in this path. Acceptance drops below 0.50 and MTP cache commit plus sampling overhead grows enough to erase the speculative benefit.
- The new timing counters show that rollback/restore overhead is not the bottleneck. The next useful optimization target is reducing per-token sampling overhead and making accepted-prefix MTP cache commit cheaper, especially for larger draft windows.
