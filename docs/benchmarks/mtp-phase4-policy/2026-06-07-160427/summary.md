# MTP Phase 4 Policy Benchmark

> Historical benchmark snapshot. This document records the model-aware
> omitted-argument policy used by that build. Current IronMLX defaults to one
> MTP draft token unless a different value is explicitly configured.

- Date: 2026-06-07
- Branch: `codex/mtp-phase4-policy`
- Base: Phase 3 commit `82982a6` (`codex/mtp-phase3-performance`)
- Binary: `target/release/ironmlx-core-bench`
- Mode: `scheduler-text`
- Prompt tokens: 67
- Generated tokens: 64
- Runs: 5 measured + 1 warmup
- Sampler: greedy
- Policy check: MTP runs intentionally omitted `--mtp-draft-tokens`

## Summary

| Model | Policy d | Baseline TPS | Policy TPS | Speedup vs Phase4 baseline | Phase3 best | Delta vs Phase3 best | Accept | Rollbacks | Cache reuse windows | Reused MTP tokens | Budget reductions | Budget increases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 1 | 138.673 | 157.278 | 1.134x | mtp_d1 167.410 | -6.05% | 0.778 | 8.0 | 28.0 | 28.0 | 0.0 | 0.0 |
| Qwen3.6-27B | 2 | 27.947 | 48.018 | 1.718x | mtp_d2 50.245 | -4.43% | 0.867 | 5.0 | 19.0 | 37.0 | 1.0 | 1.0 |
| Qwen3.6-35B-A3B | 2 | 119.203 | 127.871 | 1.073x | mtp_d2 137.424 | -6.95% | 0.720 | 11.0 | 17.0 | 30.0 | 3.0 | 3.0 |

## Interpretation

- The omitted-argument policy selected the intended Phase 3 best draft depths: Qwen3.5-4B uses `d=1`, while Qwen3.6-27B and Qwen3.6-35B-A3B use `d=2`.
- The policy MTP path remains faster than non-MTP baseline on all three fixed-prompt greedy runs: 1.134x, 1.718x, and 1.073x respectively.
- The Phase 4 TPS is slightly lower than the earlier Phase 3 best run for each model in this fresh measurement, but it lands on the same best-depth policy and preserves the expected acceptance/cache-reuse behavior.

## Raw Artifacts

- `summary.csv`
- `fixed_prompt.txt`
- `qwen35_4b_baseline.json`
- `qwen35_4b_mtp_policy.json`
- `qwen36_27b_baseline.json`
- `qwen36_27b_mtp_policy.json`
- `qwen36_35b_a3b_baseline.json`
- `qwen36_35b_a3b_mtp_policy.json`
