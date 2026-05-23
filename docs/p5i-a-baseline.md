# P5i.a Baseline — ironmlx vs omlx (PP=128 + PP=512)

**Status:** T0 baseline measurement complete.
**Date:** 2026-05-23
**Branch:** ironmlx-p5i-a-gather-qmm-feasibility (HEAD at T0 commit)
**Sweep wall:** preheat 5min × 2 + measurement (~30s × 4 PP-runs) ≈ ~12min total.

## Setup

- Model: `mlx-community/Qwen3.5-35B-A3B-4bit` (snapshot `1e20fd8d42056f870933bf98ca6211024744f7ec`).
- Machine: M5 Max 128GB (per `[reference_current_machine]`).
- MLX: `MLX_DIR=$HOME/.local/mlx`.
- ironmlx: built `cargo build --release -p ironmlx` (flag-OFF; no `p5h-profile`); served on port 18099.
- omlx: `uv run --with-editable /Users/xin/workspace/iron-rivals/omlx omlx serve --model-dir $HOME/.ironmlx/models --port 18100` (per `[feedback_omlx_cli_default]`).
- iron-bench: `--prompt-len {128,512} --max-tokens 1 --runs 7 --warmup 1 --format csv`.
- Serial execution (one server alive at a time per `[feedback_serial_perf_experiments]`).
- 5-min thermal preheat at each backend sweep entry (per P5h T0b H1 binding); preheat data discarded.

## Per-PP result

| PP | ironmlx pp_tps median | omlx pp_tps median | delta_pct | +10% target |
|---|---|---|---|---|
| 128 | 919.68 | 1060.90 | -13.31% | MISS |
| 512 | 1562.84 | 2513.74 | -37.83% | MISS |

Raw data:
- `/tmp/p5i-a-baseline-ironmlx.csv` (15 rows: 1 header + 7×PP=128 + 7×PP=512)
- `/tmp/p5i-a-baseline-omlx.csv` (same shape)
- `/tmp/p5i-a-baseline-summary.json` (machine-parseable; per-PP medians + delta_pct + passes_plus10_target)

## Interpretation

- P5h+1 ranking used ironmlx flag-OFF root as denominator (NOT external omlx); this T0 measurement is the canonical "ironmlx vs omlx" gap.
- delta_pct >= +10% means ironmlx already beats omlx by ≥10% on that PP (Full PASS condition satisfied for that PP from baseline; P5i.a only needs to not regress).
- delta_pct < +10% means optimization required to reach Full PASS; gap = (+10% - delta_pct).

## Next

T1/T2/T3 (+ T4 if conditional triggers) all benchmark against this baseline. After each landed optimization, re-run ironmlx sweep against the same omlx baseline (omlx baseline stays canonical; do NOT re-measure omlx between iterations unless model/snapshot changes per spec § 7.3).
