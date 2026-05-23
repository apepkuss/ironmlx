# P5h+2.a PP=512 Protocol (Path A — Outcome a achieved)

**Status:** Validated protocol; supersedes the implicit RUNS=7 protocol for PP=512 after T4 validation commit.
**Date:** 2026-05-23
**Branch:** ironmlx-p5h+2-a-pp512-measurement
**Spec ref:** docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md § 3.1 Outcome (a)

## Selected protocol

- **RUNS**: 15 (bumped from prior implicit canonical RUNS=7)
- **cooldown**: current (~3s inter-PP; no change from existing iron-bench default)
- **preheat**: 300s thermal saturation per spec § 7.5 (use `iron-bench --runs 800 --warmup 0 --format csv` invocation to reach ≥4-5min wall on M5 Max; the default `--runs 20` reaches only ~7s wall which is INSUFFICIENT)
- **independent repeat count**: minimum 3 for between-sweep half-range validation per spec § 3.1 Outcome (a)

## Empirical SE achieved (T1 measurement)

| candidate | within-sweep CI95 max | between-sweep half-range | final uncertainty envelope |
|---|---|---|---|
| RUNS=7 (FAILED) | 6.85% (per-repeat 6.85% / 3.74% / 3.80%) | 2.58% | 6.85% |
| **RUNS=15 (PASS)** | 1.94% (per-repeat 0.68% / 1.94% / 0.90%) | 1.91% | **1.94%** |

RUNS=15 envelope 1.94% meets but is tight against ±2% target. See "Caveat" below.

Per-repeat medians (RUNS=15): 1366.18 / 1349.34 / 1401.82 pp_tps (mean 1372.45).

## Comparison scope (CRITICAL — read before using protocol)

This protocol is validated for: **(i) ironmlx-only pre/post regression decisions** — only ironmlx repeats were collected in T1.

Downstream decisions claiming ironmlx-vs-omlx +X% external target are NOT supported by this protocol as currently validated. To upgrade to (ii) ironmlx-vs-omlx target decisions:
1. Run ≥3 omlx PP=512 repeats with the same protocol (RUNS=15, cooldown=current, preheat=300s, fresh spawn per repeat)
2. Compute omlx final uncertainty envelope via same bootstrap + between-sweep methodology
3. T4 may extend with omlx repeats during validation; if it does, document the upgrade explicitly

If only (i) is validated: aggregator emits CI for ironmlx but caller must NOT compare against omlx point estimates without omlx CI envelope.

## Rationale for selection

Per spec § 5.2 (cost = `wall_time × final_uncertainty`):
- RUNS=7: failed (6.85% > 2.0% target); REJECTED
- RUNS=15: 1.94% < 2.0% target; PASS; lowest wall meeting target
- RUNS=21/30: not measured between-sweep (skipped per "lowest wall meeting target" rule); see "Caveat" for escalation option

Per T0 RUNS=30 single-warm-sweep finding (within-sweep CI 0.276%) + T1 fresh-spawn RUNS=15 envelope 1.94%, the variance source breakdown:
- Pure within-sweep thermal noise (warm sweep): ~0.3-0.5% (per T0)
- Fresh-spawn JIT/cache fill-in (first 7 runs of each spawn): ~1.5-6% additional variance (per T1 RUNS=7 vs RUNS=15)
- Between-sweep absolute median spread (cross-spawn): ~2% per T1 RUNS=15 repeats

Original P5i.a T4 ±5-10% SE was a COMPOUND artifact:
1. **Insufficient preheat** (T4 used short preheats; default `--runs 20` = 7s wall on M5 Max — far less than the 5-min target)
2. **Insufficient RUNS=7 sample size** (RUNS=7 doesn't amortize fresh-spawn JIT variance; RUNS=15 does)

Both fixes needed — neither alone is sufficient. The new protocol enforces both.

## Reproducibility

```bash
# 1. Spawn ironmlx serve
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx serve \
  --model "$SNAP" --port 18099 --host 127.0.0.1 > /tmp/serve.log 2>&1 &
IRONMLX_PID=$!

# 2. Wait healthz (up to 5min for model load)
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:18099/healthz 2>/dev/null | grep -q ok && break
  sleep 5
done

# 3. 5-min preheat (CRITICAL — default --runs 20 is insufficient on M5 Max)
cargo run --release -p iron-bench -- \
  --target ironmlx_preheat=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 --runs 800 --warmup 0 --format csv > /tmp/preheat.log 2>&1
# Verify preheat wall ≥300s before proceeding

# 4a. PP=128 measurement sweep (RUNS=7, warmup=1)
# Note: iron-bench --runs accepts only a single value (not per-PP), so PP=128 and PP=512
# are run as two sequential invocations within the same server spawn (preheat amortized).
cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 128 --max-tokens 1 --runs 7 --warmup 1 --format csv > pp128-sweep.csv

# 4b. PP=512 measurement sweep (RUNS=15, warmup=1, NO --capture-server-request-id)
cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 --runs 15 --warmup 1 --format csv > pp512-sweep.csv

# 4c. Concatenate for aggregator (skip pp512 header; aggregator requires both PPs in one CSV)
cat pp128-sweep.csv > sweep-combined.csv
tail -n +2 pp512-sweep.csv >> sweep-combined.csv

# 5. Kill server + cooldown ~3s before next sweep
kill $IRONMLX_PID; wait $IRONMLX_PID 2>/dev/null; sleep 3

# 6. Repeat steps 1-5 for ≥3 independent spawns for between-sweep validation
# 7. Aggregate via tools/p5i_a_baseline_aggregate.py — emits per-PP 95% CI in summary.json
# 8. Verify final uncertainty envelope = MAX(within-sweep CI max, between-sweep half-range) ≤ ±2%
```

## Caveat — tight margin against ±2% ceiling

RUNS=15 between-sweep half-range 1.91% is uncomfortably close to ±2% ceiling. Future hardware (M5 Max → M6/M7) / build (ironmlx version) / MLX-version changes could push over ceiling. If P5h+3 finds the envelope drifting toward / above 2%:

- **Mitigation A**: bump to RUNS=21 (T0 within-sweep CI at RUNS=21 was 0.405%; estimated between-sweep envelope ~1.2-1.5% with comfortable margin; cost +40% wall per sweep)
- **Mitigation B**: investigate longer preheat (10min) to further reduce within-sweep variance
- **Mitigation C**: use trimmed-mean or IQR instead of median for more robust point estimate (P5h+3 iron-bench tool enhancement)

These mitigations are P5h+3 follow-up if/when needed; T1 + T4 validation found RUNS=15 sufficient today.

## References

- Spec § 4.1-4.6 for the task-by-task derivation
- reports/p5h+2-a-bench-log.md T0/T1 sections for raw data per candidate
- tools/p5h_2a_se_analysis.py for bootstrap-resample methodology
- T0 commit 3a929d2 + T1 bench log entries (gitignored)
