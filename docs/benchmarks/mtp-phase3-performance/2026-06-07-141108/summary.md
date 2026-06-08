# MTP Phase 3 Performance Benchmark

- Date: 2026-06-07
- Branch: `codex/mtp-phase3-performance`
- Base: Phase 2.5 commit `0c99b6b` (`codex/mtp-phase2-llamacpp-parity`)
- Binary: `target/release/ironmlx-core-bench`
- Mode: `scheduler-text`
- Prompt tokens: 67
- Generated tokens: 64
- Runs: 5 measured + 1 warmup
- Sampler: greedy
- Fixed prompt: `fixed_prompt.txt`

## Summary

| Model | Config | TPS | Speedup vs Phase3 baseline | Phase2.5 TPS | Delta vs Phase2.5 | Accept | Rollbacks | Cache reuse windows | Reused MTP tokens | Budget reductions | Budget increases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | baseline | 147.356 | 1.000x | 147.355 | +0.00% |  |  |  |  |  |  |
| Qwen3.5-4B | mtp_d1 | 167.410 | 1.136x | 165.194 | +1.34% | 0.778 | 8.0 | 28.0 | 28.0 | 0.0 | 0.0 |
| Qwen3.5-4B | mtp_d2 | 159.519 | 1.083x | 159.548 | -0.02% | 0.647 | 11.0 | 20.0 | 33.0 | 7.0 | 7.0 |
| Qwen3.5-4B | mtp_d4 | 146.362 | 0.993x | 122.860 | +19.13% | 0.548 | 14.0 | 15.0 | 27.0 | 8.0 | 14.0 |
| Qwen3.6-27B | baseline | 31.585 | 1.000x | 31.590 | -0.02% |  |  |  |  |  |  |
| Qwen3.6-27B | mtp_d1 | 44.890 | 1.421x | 44.736 | +0.34% | 0.853 | 5.0 | 29.0 | 29.0 | 0.0 | 0.0 |
| Qwen3.6-27B | mtp_d2 | 50.245 | 1.591x | 45.684 | +9.98% | 0.867 | 5.0 | 19.0 | 37.0 | 1.0 | 1.0 |
| Qwen3.6-27B | mtp_d4 | 33.663 | 1.066x | 27.820 | +21.01% | 0.619 | 13.0 | 12.0 | 25.0 | 6.0 | 11.0 |
| Qwen3.6-35B-A3B | baseline | 125.159 | 1.000x | 125.877 | -0.57% |  |  |  |  |  |  |
| Qwen3.6-35B-A3B | mtp_d1 | 135.838 | 1.085x | 133.157 | +2.01% | 0.778 | 8.0 | 28.0 | 28.0 | 0.0 | 0.0 |
| Qwen3.6-35B-A3B | mtp_d2 | 137.424 | 1.098x | 130.146 | +5.59% | 0.720 | 11.0 | 17.0 | 30.0 | 3.0 | 3.0 |
| Qwen3.6-35B-A3B | mtp_d4 | 132.390 | 1.058x | 93.943 | +40.93% | 0.619 | 12.0 | 13.0 | 29.0 | 10.0 | 12.0 |

## Best Configs

| Model | Best Phase3 config | Phase3 speedup | Phase2.5 same-config speedup | TPS delta vs Phase2.5 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | mtp_d1 | 1.136x | 1.121x | +1.34% |
| Qwen3.6-27B | mtp_d2 | 1.591x | 1.446x | +9.98% |
| Qwen3.6-35B-A3B | mtp_d2 | 1.098x | 1.034x | +5.59% |

## Interpretation

- Phase 3 makes `mtp_d1` strongest for Qwen3.5-4B, and `mtp_d2` strongest for both Qwen3.6 models in this fixed-prompt greedy benchmark.
- The largest observed gain is Qwen3.6-27B `mtp_d2`: 50.245 tok/s, 1.591x over the Phase 3 non-MTP baseline, and +9.98% over the Phase 2.5 `mtp_d2` result.
- Draft-cache reuse is active on full-accept windows. For example, Qwen3.6-27B `mtp_d2` reused about 37 MTP tokens per measured run and avoided MTP cache restore work on those accepted windows.
- Adaptive draft budgeting reduces the damage from poor high-depth windows. `mtp_d4` improved materially versus Phase 2.5, but remains slower than `mtp_d2` for all three models in this prompt.
- Greedy vectorized draft sampling removes the large per-position sampler loop from the MTP path; `sampling_us_mean` is still dominated by verification/synchronization boundaries but no longer scales through repeated sampler calls for greedy MTP.

## Raw Artifacts

- `summary.csv`
- `benchmark.log`
- `qwen35_4b_baseline.json`, `qwen35_4b_mtp_d1.json`, `qwen35_4b_mtp_d2.json`, `qwen35_4b_mtp_d4.json`
- `qwen36_27b_baseline.json`, `qwen36_27b_mtp_d1.json`, `qwen36_27b_mtp_d2.json`, `qwen36_27b_mtp_d4.json`
- `qwen36_35b_a3b_baseline.json`, `qwen36_35b_a3b_mtp_d1.json`, `qwen36_35b_a3b_mtp_d2.json`, `qwen36_35b_a3b_mtp_d4.json`
