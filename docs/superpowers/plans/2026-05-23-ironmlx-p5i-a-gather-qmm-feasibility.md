# P5i.a — gather_qmm Feasibility + Short-PP Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish ironmlx vs omlx external pp_tps gap at PP=128/512, land low-risk wrapper/fusion optimizations toward the +24% PP=128 target, quantify the residual PP=512 gap, and produce a 立项/否决 verdict for a multi-sprint self-quant gather Metal kernel rewrite (P5i.b).

**Architecture:** 6-task exploration/convergence phase. T0 produces controlled iron-bench baseline (ironmlx flag-OFF vs omlx CLI; warmup=1 RUNS=7 serial; same model + same prompt). T1 simplifies sparse_moe.rs MoE input shaping (Level b wrapper-layer). T2 PoC fuses gate_proj+up_proj `gather_qmm` calls (Level c). T3 writes a self-quant gather Metal kernel design memo + ROI estimate (Level a feasibility ONLY; no kernel impl). T4 (conditional) tunes `gda_step_1a_in_proj_qkvz` tile params for M5 Max (Level d). T5 close-out with Full PASS / Feasibility PASS / Blocked verdict. **No Metal kernel implementation in P5i.a** — that's gated to P5i.b with Boss Scope-gate approval.

**Tech Stack:** Rust workspace toolchain (`rust-version = 1.94`; nightly used for fmt/clippy checks per CLAUDE.md), MLX-rs lazy graph + mlx::quantization::gather_qmm, Python 3.13 + uv + ruff for analysis scripts, iron-bench HTTP sweep harness from this repo's Cargo workspace (`/Users/xin/workspace/ironmlx-backend/iron-bench`), omlx source-CLI from `/Users/xin/workspace/iron-rivals/omlx` (per `[feedback_omlx_cli_default]`).

---

## File structure

**New branch:** `ironmlx-p5i-a-gather-qmm-feasibility` (fork from `ironmlx-p5h-perf` HEAD `6579633` which has the P5i.a spec).

**Rust src (T1/T2/T4 modify; T3/T0/T5 no src changes):**
- `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` (T1 simplifications around lines 217-331 input shaping + 357/481 gather_qmm calls; T2 fuses gate+up into one call)
- `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs::RoutedExperts::from_loader` (T2 if combined gate_up storage is needed)
- `ironmlx/src/nn/gated_delta_net.rs` (T4 conditional — gda_step_1a_in_proj_qkvz tile param adjustments)
- `ironmlx/src/nn/self_qmm/lookup.rs` (T4 conditional — M5 Max tile lookup entry)

**Python tooling (T0 only):**
- `tools/p5i_a_baseline_aggregate.py` (new; produces `/tmp/p5i-a-baseline-summary.json` from raw CSVs)

**Docs (T0/T3/T5 commits; reports/ gitignored per `[feedback_no_reports_commit]`):**
- `docs/p5i-a-baseline.md` (T0 committed concise summary)
- `docs/p5i-a-gather-kernel-feasibility.md` (T3 committed design memo + ROI estimate + 立项/否决 recommendation)
- `docs/p5i-a-close-out.md` (T5 committed close-out)
- `reports/p5i-a-baseline-detail.md` (T0 gitignored full detail if needed)
- `reports/p5i-a-bench-log.md` (T1/T2/T4 gitignored per-experiment iron-bench raw bench notes)

**Spec (T5 conditional update):**
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2 (T5 update IFF Full PASS on PP=128 +10%)

**Memory (T5; outside repo, not in commits):**
- `~/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md` (T5 extend with P5i.a closure section)

---

## Task 1: T0 — Controlled iron-bench baseline (omlx vs ironmlx)

**Files:**
- Create: `tools/p5i_a_baseline_aggregate.py` (~80 lines)
- Create: `docs/p5i-a-baseline.md`
- Output (gitignored): `/tmp/p5i-a-baseline-ironmlx.csv`, `/tmp/p5i-a-baseline-omlx.csv`, `/tmp/p5i-a-baseline-summary.json`, `/tmp/p5i-a-ironmlx.log`, `/tmp/p5i-a-omlx.log`

### Step 1.1: Branch + spec verification

- [ ] Create + checkout the new branch:

```bash
cd /Users/xin/workspace/ironmlx-backend
git fetch
git checkout -b ironmlx-p5i-a-gather-qmm-feasibility ironmlx-p5h-perf
git log --oneline -3
```

Expected: HEAD `6579633` (P5i.a spec commit) at top.

- [ ] Verify spec + plan are committed on this branch:

```bash
ls docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md
ls docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md
```

Expected: both files present.

### Step 1.2: Identify model snapshot dir + omlx CLI availability

- [ ] Find Qwen3.5-35B-A3B-4bit snapshot:

```bash
ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/
```

Expected: at least one snapshot SHA dir (e.g. `1e20fd8d42056f870933bf98ca6211024744f7ec`).

- [ ] Verify omlx CLI is invokable (per `[feedback_omlx_cli_default]` + `[reference_iron_rivals_baselines]`):

```bash
uv run --with-editable /Users/xin/workspace/iron-rivals/omlx omlx --help 2>&1 | head -20
```

Expected: omlx CLI help output mentioning `serve` subcommand.

- [ ] Verify omlx serve --help shows `--model-dir`, `--port`, `--host`:

```bash
uv run --with-editable /Users/xin/workspace/iron-rivals/omlx omlx serve --help 2>&1 | head -30
```

Expected: serve subcommand args including `--model-dir`, `--port`, `--host`.

### Step 1.3: Pre-sweep cleanup + port availability

- [ ] Clear stale outputs:

```bash
rm -f /tmp/p5i-a-baseline-ironmlx.csv /tmp/p5i-a-baseline-omlx.csv /tmp/p5i-a-baseline-summary.json /tmp/p5i-a-ironmlx.log /tmp/p5i-a-omlx.log /tmp/p5i-a-preheat.log /tmp/p5i-a-env.sh
```

- [ ] Verify ports free (ironmlx 18099, omlx 18100):

```bash
lsof -i :18099 2>&1 || echo "PORT_18099_FREE"
lsof -i :18100 2>&1 || echo "PORT_18100_FREE"
```

Expected: both PORT_*_FREE.

### Step 1.4: Spawn ironmlx serve (background) + wait for ready

- [ ] Build release binary first:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
```

Expected: `Finished release` clean.

- [ ] Spawn ironmlx serve in background (no `--features p5h-profile`; production path):

```bash
SNAP=$(ls -d ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/*/ | head -1)
echo "snap=$SNAP"
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx serve \
  --model "$SNAP" \
  --port 18099 \
  --host 127.0.0.1 > /tmp/p5i-a-ironmlx-serve.log 2>&1 &
IRONMLX_PID=$!
echo "ironmlx_pid=$IRONMLX_PID"
{
  printf 'export SNAP=%q\n' "$SNAP"
  printf 'export IRONMLX_PID=%q\n' "$IRONMLX_PID"
} > /tmp/p5i-a-env.sh
```

This writes `/tmp/p5i-a-env.sh`; every later shell step that needs `SNAP` or a PID must run `source /tmp/p5i-a-env.sh` first. This avoids relying on shell-local variables surviving across agent/tool steps.

- [ ] Wait for ironmlx ready (poll healthz; up to 5min for model load):

```bash
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:18099/healthz 2>/dev/null | grep -q ok; then
    echo "ready_after=${i}*5s"
    break
  fi
  sleep 5
done
curl -sf http://127.0.0.1:18099/healthz || (echo "ironmlx not ready after 5min"; exit 1)
```

Expected: `ready_after=Nx5s` with N ≤ 60 (5 min max).

### Step 1.5: ironmlx 5-min thermal preheat (per P5h T0b H1 binding)

- [ ] Preheat with extended iron-bench warmup-only sweep at PP=128/512 (~5min wall):

```bash
source /tmp/p5i-a-env.sh
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --target ironmlx_preheat=http://127.0.0.1:18099 \
  --model qwen3.5-moe \
  --model-dir "$SNAP" \
  --prompt-len 512 \
  --max-tokens 1 \
  --runs 20 \
  --warmup 0 \
  --format csv > /tmp/p5i-a-preheat.log 2>&1
echo "preheat_exit=$?"
```

Expected: `preheat_exit=0`. ~5min wall (20 runs × ~15s per request at PP=512 ≈ 5min).

Note: preheat data discarded; this step exists only for thermal saturation.

### Step 1.6: ironmlx measurement sweep (PP=128/512 warmup=1 RUNS=7)

- [ ] Run measurement sweep against ironmlx:

```bash
source /tmp/p5i-a-env.sh
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe \
  --model-dir "$SNAP" \
  --prompt-len 128 \
  --max-tokens 1 \
  --runs 7 \
  --warmup 1 \
  --format csv > /tmp/p5i-a-baseline-ironmlx-pp128.csv 2>>/tmp/p5i-a-ironmlx.log
echo "pp128_exit=$?"

cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe \
  --model-dir "$SNAP" \
  --prompt-len 512 \
  --max-tokens 1 \
  --runs 7 \
  --warmup 1 \
  --format csv > /tmp/p5i-a-baseline-ironmlx-pp512.csv 2>>/tmp/p5i-a-ironmlx.log
echo "pp512_exit=$?"

# Concat into single ironmlx baseline CSV (header from PP=128; skip header on PP=512):
head -1 /tmp/p5i-a-baseline-ironmlx-pp128.csv > /tmp/p5i-a-baseline-ironmlx.csv
tail -n +2 /tmp/p5i-a-baseline-ironmlx-pp128.csv | grep -v '^$' >> /tmp/p5i-a-baseline-ironmlx.csv
tail -n +2 /tmp/p5i-a-baseline-ironmlx-pp512.csv | grep -v '^$' >> /tmp/p5i-a-baseline-ironmlx.csv
wc -l /tmp/p5i-a-baseline-ironmlx.csv
```

Expected: both `pp*_exit=0`; final concat ~15 lines (1 header + 7 PP=128 + 7 PP=512).

### Step 1.7: Stop ironmlx + verify port free (per `[feedback_serial_perf_experiments]`)

- [ ] Kill ironmlx + verify:

```bash
source /tmp/p5i-a-env.sh
kill $IRONMLX_PID 2>/dev/null
wait $IRONMLX_PID 2>/dev/null
sleep 2
lsof -i :18099 2>&1 || echo "PORT_18099_FREE_AFTER_KILL"
```

Expected: `PORT_18099_FREE_AFTER_KILL`.

### Step 1.8: Spawn omlx serve (background) + wait for ready

- [ ] Identify omlx model-dir + model name:

omlx `--model-dir` points at the PARENT containing model subdirs. Use the snapshot's parent (`~/.ironmlx/models/`):

```bash
OMLX_MODEL_DIR=$HOME/.ironmlx/models
source /tmp/p5i-a-env.sh
# Verify the Qwen snapshot is visible from this parent:
ls "$OMLX_MODEL_DIR/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/" | head -3
printf 'export OMLX_MODEL_DIR=%q\n' "$OMLX_MODEL_DIR" >> /tmp/p5i-a-env.sh
```

Expected: snapshot SHA dir listed.

- [ ] Spawn omlx serve in background on port 18100:

```bash
source /tmp/p5i-a-env.sh
uv run --with-editable /Users/xin/workspace/iron-rivals/omlx omlx serve \
  --model-dir "$OMLX_MODEL_DIR" \
  --port 18100 \
  --host 127.0.0.1 > /tmp/p5i-a-omlx-serve.log 2>&1 &
OMLX_PID=$!
echo "omlx_pid=$OMLX_PID"
printf 'export OMLX_PID=%q\n' "$OMLX_PID" >> /tmp/p5i-a-env.sh
```

- [ ] Wait for omlx ready:

```bash
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:18100/v1/models 2>/dev/null | grep -q '"data"'; then
    echo "omlx_ready_after=${i}*5s"
    break
  fi
  sleep 5
done
curl -sf http://127.0.0.1:18100/v1/models | python3 -m json.tool 2>&1 | head -20
```

Expected: `/v1/models` returns JSON with `data` array listing discoverable models including a Qwen3.5-35B-A3B-4bit entry. Note the exact model `id` string omlx assigns (e.g. `models--mlx-community--Qwen3.5-35B-A3B-4bit` or similar) — this is what iron-bench passes via `--model` for omlx.

- [ ] Capture omlx model id into variable:

```bash
OMLX_MODEL_ID=$(curl -sf http://127.0.0.1:18100/v1/models | python3 -c "import sys, json; d = json.load(sys.stdin); print([m['id'] for m in d['data'] if 'Qwen3.5' in m['id'] and 'A3B' in m['id'] and '4bit' in m['id']][0])")
echo "omlx_model_id=$OMLX_MODEL_ID"
printf 'export OMLX_MODEL_ID=%q\n' "$OMLX_MODEL_ID" >> /tmp/p5i-a-env.sh
```

Expected: a non-empty model id matching Qwen3.5-35B-A3B-4bit.

### Step 1.9: omlx 5-min thermal preheat

- [ ] Same preheat protocol as Step 1.5 but targeting omlx:

```bash
source /tmp/p5i-a-env.sh
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --target omlx_preheat=http://127.0.0.1:18100 \
  --model "$OMLX_MODEL_ID" \
  --model-dir "$SNAP" \
  --prompt-len 512 \
  --max-tokens 1 \
  --runs 20 \
  --warmup 0 \
  --format csv > /tmp/p5i-a-omlx-preheat.log 2>&1
echo "omlx_preheat_exit=$?"
```

Expected: `omlx_preheat_exit=0`. ~5min wall.

### Step 1.10: omlx measurement sweep (PP=128/512 warmup=1 RUNS=7)

- [ ] Run measurement sweep against omlx:

```bash
source /tmp/p5i-a-env.sh
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --target omlx=http://127.0.0.1:18100 \
  --model "$OMLX_MODEL_ID" \
  --model-dir "$SNAP" \
  --prompt-len 128 \
  --max-tokens 1 \
  --runs 7 \
  --warmup 1 \
  --format csv > /tmp/p5i-a-baseline-omlx-pp128.csv 2>>/tmp/p5i-a-omlx.log
echo "omlx_pp128_exit=$?"

cargo run --release -p iron-bench -- \
  --target omlx=http://127.0.0.1:18100 \
  --model "$OMLX_MODEL_ID" \
  --model-dir "$SNAP" \
  --prompt-len 512 \
  --max-tokens 1 \
  --runs 7 \
  --warmup 1 \
  --format csv > /tmp/p5i-a-baseline-omlx-pp512.csv 2>>/tmp/p5i-a-omlx.log
echo "omlx_pp512_exit=$?"

head -1 /tmp/p5i-a-baseline-omlx-pp128.csv > /tmp/p5i-a-baseline-omlx.csv
tail -n +2 /tmp/p5i-a-baseline-omlx-pp128.csv | grep -v '^$' >> /tmp/p5i-a-baseline-omlx.csv
tail -n +2 /tmp/p5i-a-baseline-omlx-pp512.csv | grep -v '^$' >> /tmp/p5i-a-baseline-omlx.csv
wc -l /tmp/p5i-a-baseline-omlx.csv
```

Expected: both `omlx_pp*_exit=0`; final concat ~15 lines.

### Step 1.11: Stop omlx

- [ ] Kill omlx + verify:

```bash
source /tmp/p5i-a-env.sh
kill $OMLX_PID 2>/dev/null
wait $OMLX_PID 2>/dev/null
sleep 2
lsof -i :18100 2>&1 || echo "PORT_18100_FREE_AFTER_KILL"
```

Expected: `PORT_18100_FREE_AFTER_KILL`.

### Step 1.12: Aggregate baseline to JSON

- [ ] Create `tools/p5i_a_baseline_aggregate.py`:

```python
"""P5i.a baseline aggregator: read ironmlx + omlx iron-bench CSVs, produce per-PP
summary JSON with medians + delta_pct + +10%-target threshold.

CSV header (from iron-bench --format csv; P5i.a does not use request-id join):
target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,
prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_PPS = (128, 512)
EXPECTED_RUNS_PER_PP = 7


def load_pp_tps_by_pp(csv_path: Path) -> dict[int, list[float]]:
    by_pp: dict[int, list[float]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"target", "pp_target", "run_idx", "pp_tps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path}: missing required CSV columns: {sorted(missing)}")
        for row in reader:
            try:
                pp = int(row["pp_target"])
                pp_tps = float(row["pp_tps"])
            except (KeyError, ValueError):
                continue
            if pp not in EXPECTED_PPS:
                raise SystemExit(f"{csv_path}: unexpected pp_target={pp}; expected {EXPECTED_PPS}")
            if pp_tps <= 0:
                raise SystemExit(f"{csv_path}: non-positive pp_tps={pp_tps} for pp={pp}")
            by_pp[pp].append(pp_tps)
    for pp in EXPECTED_PPS:
        got = len(by_pp.get(pp, []))
        if got != EXPECTED_RUNS_PER_PP:
            raise SystemExit(
                f"{csv_path}: expected {EXPECTED_RUNS_PER_PP} measured rows for PP={pp}, got {got}"
            )
    return by_pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ironmlx-csv", required=True, type=Path)
    ap.add_argument("--omlx-csv", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    ironmlx = load_pp_tps_by_pp(args.ironmlx_csv)
    omlx = load_pp_tps_by_pp(args.omlx_csv)

    summary = {"per_pp": {}}
    for pp in EXPECTED_PPS:
        i_tps = ironmlx[pp]
        o_tps = omlx[pp]
        i_med = statistics.median(i_tps)
        o_med = statistics.median(o_tps)
        delta_pct = (i_med - o_med) / o_med * 100.0
        passes_plus10 = (delta_pct is not None) and (delta_pct >= 10.0)
        summary["per_pp"][str(pp)] = {
            "ironmlx_runs": len(i_tps),
            "omlx_runs": len(o_tps),
            "ironmlx_pp_tps_median": i_med,
            "omlx_pp_tps_median": o_med,
            "delta_pct": delta_pct,
            "passes_plus10_target": passes_plus10,
        }

    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"OK: wrote {args.out_json}")
    for pp, row in summary["per_pp"].items():
        i, o, d = row["ironmlx_pp_tps_median"], row["omlx_pp_tps_median"], row["delta_pct"]
        flag = "PASS" if row["passes_plus10_target"] else "MISS"
        print(f"  PP={pp}: ironmlx={i:.2f} omlx={o:.2f} delta={d:+.2f}% {flag}")


if __name__ == "__main__":
    main()
```

- [ ] Run aggregator:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run python tools/p5i_a_baseline_aggregate.py \
  --ironmlx-csv /tmp/p5i-a-baseline-ironmlx.csv \
  --omlx-csv /tmp/p5i-a-baseline-omlx.csv \
  --out-json /tmp/p5i-a-baseline-summary.json
cat /tmp/p5i-a-baseline-summary.json
```

Expected: JSON with 2 PPs (128 + 512); each has ironmlx_pp_tps_median, omlx_pp_tps_median, delta_pct, passes_plus10_target.

### Step 1.13: Write `docs/p5i-a-baseline.md`

- [ ] Create the doc with the per-PP table populated from the JSON. Template:

```markdown
# P5i.a Baseline — ironmlx vs omlx (PP=128 + PP=512)

**Status:** T0 baseline measurement complete.
**Date:** YYYY-MM-DD (fill from `date +%Y-%m-%d`)
**Branch:** ironmlx-p5i-a-gather-qmm-feasibility (HEAD at T0 commit)
**Sweep wall:** preheat 5min × 2 + measurement (~30s × 4 PP-runs) ≈ ~12min total.

## Setup

- Model: `mlx-community/Qwen3.5-35B-A3B-4bit` (snapshot `<SNAPSHOT_SHA>`).
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
| 128 | <FILL> | <FILL> | <FILL>% | <FILL> ✓/✗ |
| 512 | <FILL> | <FILL> | <FILL>% | <FILL> ✓/✗ |

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
```

Run `date +%Y-%m-%d` and fill the date placeholder. Fill the table from the JSON output. Fill `<SNAPSHOT_SHA>` from `source /tmp/p5i-a-env.sh && basename "$SNAP"`.

### Step 1.14: Commit T0

- [ ] Hygiene + commit:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5i_a_baseline_aggregate.py
uv run --with ruff ruff format --check tools/p5i_a_baseline_aggregate.py

git add tools/p5i_a_baseline_aggregate.py docs/p5i-a-baseline.md
git commit -m "$(cat <<'COMMIT'
docs(p5i-a-t0): controlled iron-bench baseline (ironmlx flag-OFF vs omlx CLI; PP=128/512)

Per docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md § 4.1
and docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md T0.

Establishes the canonical "ironmlx vs omlx" external gap at PP=128 + PP=512
(P5h+1 ranking used ironmlx flag-OFF root as its own denominator; that's NOT
a direct external comparison). All subsequent P5i.a optimizations measure
against this omlx baseline.

Protocol:
* Serial spawn (ironmlx 18099 → kill → omlx 18100)
* 5min thermal preheat per backend (P5h T0b H1 binding)
* iron-bench --prompt-len {128,512} --runs 7 --warmup 1 --format csv
* RUNS=7 measured per PP per backend; warmup=1 unmeasured per PP per backend
* Fixed PP order 128 then 512; abort-then-rerun on partial failure

Per-PP result table populated in docs/p5i-a-baseline.md. Aggregator
tools/p5i_a_baseline_aggregate.py reads raw CSVs + emits
/tmp/p5i-a-baseline-summary.json with per-PP medians + delta_pct +
+10%-target threshold.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
```

---

## Task 2: T1 — sparse_moe.rs shape/dispatch 检视 + Level (b) wrapper optimization

**Files:**
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` (lines 217-331 input shaping region; lines 357 + 481 gather_qmm call sites)
- Reference: `/Users/xin/workspace/iron-rivals/mlx/mlx/backend/metal/quantized.cpp:1484` (gather_qmm_rhs fast-path entry conditions)
- Output (gitignored): `reports/p5i-a-bench-log.md` (per-experiment notes)

### Step 2.1: Identify simplification candidates

- [ ] Read sparse_moe.rs input shaping region:

```bash
sed -n '217,331p' /Users/xin/workspace/ironmlx-backend/ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
```

Enumerate every `expand_dims`, `reshape`, `as_type`, rank promotion, broadcast that touches `gather_qmm` inputs. For each, record:
- Line number
- What it does
- Why it was added (comment or commit context)
- Whether it's required by MLX gather_qmm_rhs fast-path entry conditions

- [ ] Read MLX gather_qmm_rhs fast-path conditions:

```bash
sed -n '1480,1560p' /Users/xin/workspace/iron-rivals/mlx/mlx/backend/metal/quantized.cpp
```

Document the fast-path entry condition checks (typically rank requirements, batch dim alignment, indices dtype, etc.).

- [ ] Produce a candidate list in `reports/p5i-a-bench-log.md` (gitignored), one entry per simplification candidate:

```markdown
# P5i.a T1 sparse_moe.rs simplification candidates

## Candidate C1: <description>
- Site: sparse_moe.rs:<line>
- Current shape transform: <what it does>
- Hypothesis: <why simplification might work without losing fast-path>
- Fast-path requirement check: <which condition this might violate or preserve>
- Plan: <impl outline + bench>
- Status: PENDING / LANDED / REJECTED (after experiment)

## Candidate C2: ...
```

### Step 2.2: Per-candidate experiment loop

For each candidate Cn from Step 2.1, repeat these sub-steps (commit per landed candidate; document negative results in the bench log):

- [ ] Implement the simplification in sparse_moe.rs (one focused edit).

- [ ] Smoke test (production parity per spec § 7.1):

```bash
cd /Users/xin/workspace/ironmlx-backend
source /tmp/p5i-a-env.sh
MLX_DIR=$HOME/.local/mlx IRONMLX_MOE_MODEL_DIR="$SNAP" \
  cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

Expected: 2 passed (argmax sentinel + shape/finite). If argmax fails, REJECT this candidate; revert edit; document in bench log.

- [ ] Rebuild + ironmlx iron-bench (same T0 protocol; reuse T0 omlx baseline):

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
# Spawn ironmlx serve, preheat, sweep, kill — same as Step 1.4-1.7 but only ironmlx side.
# Save to /tmp/p5i-a-cn-ironmlx.csv where n = candidate number.
```

- [ ] Aggregate vs T0 baseline:

```bash
uv run python tools/p5i_a_baseline_aggregate.py \
  --ironmlx-csv /tmp/p5i-a-cn-ironmlx.csv \
  --omlx-csv /tmp/p5i-a-baseline-omlx.csv \
  --out-json /tmp/p5i-a-cn-summary.json
diff <(jq '.per_pp."128".ironmlx_pp_tps_median' /tmp/p5i-a-baseline-summary.json) \
     <(jq '.per_pp."128".ironmlx_pp_tps_median' /tmp/p5i-a-cn-summary.json)
diff <(jq '.per_pp."512".ironmlx_pp_tps_median' /tmp/p5i-a-baseline-summary.json) \
     <(jq '.per_pp."512".ironmlx_pp_tps_median' /tmp/p5i-a-cn-summary.json)
```

Compute pp_tps improvement vs baseline per PP. Land condition: **repeatable ≥1% pp_tps improvement at PP=128 OR PP=512, with no regression beyond ±2% noise band on the other PP** (per spec § 7.2 noise tolerance).

Repeatability protocol:
- If the first sweep does not meet the land condition, reject the candidate without a confirm sweep.
- If the first sweep meets the land condition, run one independent confirm sweep with a fresh ironmlx spawn + 5-min preheat + PP=128/512 measurement. Save it to `/tmp/p5i-a-cn-confirm-ironmlx.csv` and aggregate to `/tmp/p5i-a-cn-confirm-summary.json`.
- Land only if both the first and confirm sweeps meet the same land condition. If the confirm sweep falls below threshold, reject and document as noise/unstable.

- [ ] **Land or reject decision**:
  - LAND: commit the change. Document in bench log with measured pp_tps improvement.
  - REJECT: revert the edit (`git checkout -- ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`); document in bench log with negative finding rationale (which fast-path condition violated OR which perf concern).

Per spec § 4.2: no simplification is allowed to land only to satisfy task closure. Negative results are first-class evidence for Feasibility PASS gate.

- [ ] If landing, commit (one commit per landed simplification):

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release

git add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
git commit -m "$(cat <<COMMIT
perf(p5i-a-t1-cn): <description of simplification>

Per P5i.a spec § 4.2 Level (b) wrapper optimization.

Simplification: <what was removed/changed>.
Site: sparse_moe.rs:<line>.

Bench result:
* PP=128 ironmlx pp_tps median: <baseline> -> <new> (<+N>%)
* PP=512 ironmlx pp_tps median: <baseline> -> <new> (<+M>%)
* MLX gather_qmm_rhs fast-path: still entered (verified via <how>).
* Smoke test (p5_qwen35_moe_smoke argmax sentinel): PASS.

Cumulative pp_tps progression noted in reports/p5i-a-bench-log.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
```

### Step 2.3: T1 close

- [ ] When every candidate from Step 2.1 has been either LANDED (with measurement) or REJECTED (with documented negative finding), close T1.

- [ ] Final cumulative state captured in `reports/p5i-a-bench-log.md`: list of all candidates + their disposition + cumulative ironmlx pp_tps at PP=128/512 vs T0 baseline.

If zero candidates land with ≥1% improvement → T1 still closes (negative-result task; contributes to Feasibility PASS gate per spec § 4.2).

---

## Task 3: T2 — gate+up fusion 可行性验证 (Level c)

**Files:**
- Modify (PoC): `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` (around line 357 — gate_proj + up_proj gather_qmm calls)
- Modify (if PoC lands): weight loader site for combined gate_up_proj stacking
- Temporary during PoC only: dual-path assertion helper inside `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` to compare fused vs legacy gate/up outputs before deleting the legacy path
- Reference: `mlx_lm/models/qwen3_5_moe.py:43-49` (sanitize step showing upstream weights ARE combined `gate_up_proj` but stored split as `gate_proj.weight` + `up_proj.weight`)
- Output (gitignored): `reports/p5i-a-bench-log.md` (T2 section)

### Step 3.1: Check MLX gather_qmm API for combined gate_up weight

- [ ] Read `mlx::quantization::gather_qmm` Rust wrapper:

```bash
sed -n '300,400p' /Users/xin/workspace/ironmlx-backend/mlx/src/quantization.rs
```

Identify whether the API accepts a combined weight matrix with intermediate dim = gate + up (i.e. stacked along intermediate axis at load time). If yes, the fusion is a load-time + single-call change. If no, fusion requires either MLX C++ API extension (out of P5i.a scope) or different shape strategy.

- [ ] Read MLX upstream gather_qmm C++ signature:

```bash
grep -nE 'array gather_qmm\(' /Users/xin/workspace/iron-rivals/mlx/mlx/backend/metal/quantized.cpp
```

Confirm the weight shape contract.

### Step 3.2: Check ironmlx weight loader

- [ ] Locate where ironmlx loads `gate_proj.weight` + `up_proj.weight` for MoE experts:

```bash
grep -rn 'gate_proj\.weight\|gate_proj_weight\|"gate_proj"' /Users/xin/workspace/ironmlx-backend/ironmlx/src/models/qwen3_5_moe/ | head -10
```

Determine whether weights are loaded as 2 separate quantized arrays (with separate scales + affine biases) or already combined.

### Step 3.3: PoC — stack gate + up weights at load time + single gather_qmm call

- [ ] PoC implementation:
  - At weight load time: `mx::concatenate([gate_proj_weight, up_proj_weight], axis=intermediate_dim)`. Also stack matching `scales` + `affine_biases` along the same axis. Preserve `bits`, `group_size`, `mode`, expert axis, and packed-weight layout exactly (per spec § 4.3 correctness gate).
  - At forward time (sparse_moe.rs:357 area): single `gather_qmm` call producing combined `[BS*k, 1, 1, 2*I]` output (sorted branch) or `[BS, k, 1, 2*I]` (default branch). Slice into `(gate_out, up_out)` along intermediate axis.

- [ ] **Correctness gate (per spec § 4.3 — MUST PASS BEFORE perf bench)**:
  - Before removing the legacy two-call code, keep a temporary dual-path helper in `sparse_moe.rs` that computes both:
    - legacy `gate_proj gather_qmm` + `up_proj gather_qmm`; and
    - fused `gate_up gather_qmm` + slice.
  - Run for BOTH sorted and default branches:
    - default branch: use a small prompt where `BS*k < SORTED_ROUTING_MIN_BS_K` (e.g. prompt length 16 with k=8 → 128);
    - sorted branch: use PP=128 or PP=512 where `BS*k >= SORTED_ROUTING_MIN_BS_K`.
  - Compare fused output `(gate_out, up_out)` against the existing two-call path's output for the same inputs.
  - Verify shape contracts: sorted `[BS*k, 1, 1, I]`, default `[BS, k, 1, I]`.
  - Compute max_abs and max_rel error per output. Must stay within the existing MoE smoke sentinel tolerance for 4-bit affine paths (check `tests/p5_qwen35_moe_smoke.rs` for the actual tolerance values; typically 1e-2 to 1e-1 for 4-bit affine quant).
  - If `gate_biases` or `up_biases` is absent (None case), the fused path must prove equivalent handling — either both branches stay None, or both have the same combined-biases shape. If unequal handling → REJECT fusion.
  - Record the exact default/sorted branch inputs, max_abs, max_rel, and shape observations in `reports/p5i-a-bench-log.md`.
  - Remove the temporary dual-path assertion helper before the final LAND commit; the production code must contain only the selected fused path (if landed) or the original two-call path (if rejected).

If correctness gate fails on either branch → REJECT; revert; document in bench log; T2 closes with negative finding.

### Step 3.4: Perf bench (only if correctness gate passed)

- [ ] Smoke test (production parity):

```bash
source /tmp/p5i-a-env.sh
MLX_DIR=$HOME/.local/mlx IRONMLX_MOE_MODEL_DIR="$SNAP" \
  cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

Expected: 2 passed.

- [ ] iron-bench sweep (same T0 protocol; one ironmlx spawn + preheat + sweep + kill):

```bash
# Spawn ironmlx, preheat, sweep PP=128/512, kill — same pattern as Steps 1.4-1.7
# Save to /tmp/p5i-a-t2-ironmlx.csv
```

- [ ] Aggregate vs T0 baseline:

```bash
uv run python tools/p5i_a_baseline_aggregate.py \
  --ironmlx-csv /tmp/p5i-a-t2-ironmlx.csv \
  --omlx-csv /tmp/p5i-a-baseline-omlx.csv \
  --out-json /tmp/p5i-a-t2-summary.json
```

Per spec § 4.3: land if **repeatable ≥1% pp_tps improvement at PP=128 OR PP=512 with no regression beyond ±2% noise on the other PP**.

Repeatability protocol:
- If the first T2 sweep does not meet the land condition, reject fusion without a confirm sweep.
- If the first T2 sweep meets the land condition, run one independent confirm sweep with a fresh ironmlx spawn + 5-min preheat + PP=128/512 measurement. Save it to `/tmp/p5i-a-t2-confirm-ironmlx.csv` and aggregate to `/tmp/p5i-a-t2-confirm-summary.json`.
- Land only if both sweeps meet the same land condition. If the confirm sweep falls below threshold, reject and document as unstable/noise.

### Step 3.5: T2 land or reject + commit

- [ ] If LAND:
  - Hygiene cycle + commit:

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release

git add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs <weight loader file if changed>
git commit -m "$(cat <<COMMIT
perf(p5i-a-t2): fuse gate_proj + up_proj gather_qmm into single call

Per P5i.a spec § 4.3 Level (c) gate+up fusion.

Approach: stack gate_proj.weight + up_proj.weight along intermediate axis
at weight load time; matching scales + affine biases stacked the same way;
preserve bits/group_size/mode/expert axis/packed-weight layout. Single
gather_qmm call replaces the prior two-call (gate then up). Output sliced
into (gate_out, up_out) before SwiGLU.

Correctness gate:
* sorted branch: max_abs <X> / max_rel <Y> (within smoke tolerance)
* default branch: max_abs <X> / max_rel <Y> (within smoke tolerance)
* gate/up biases handling: <None case verified / equivalent>

Perf bench (vs T0 baseline /tmp/p5i-a-baseline-summary.json):
* PP=128 ironmlx pp_tps median: <baseline> -> <new> (<+N>%)
* PP=512 ironmlx pp_tps median: <baseline> -> <new> (<+M>%)
* Smoke (p5_qwen35_moe_smoke): PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
```

- [ ] If REJECT: revert; document negative finding in bench log (correctness failure detail OR perf < 1% on both PPs). T2 closes with documented 否决.

---

## Task 4: T3 — Self-quant gather Metal kernel 立项设计 (Level a feasibility ONLY)

**Files:**
- Read-only: `ironmlx/src/nn/self_qmm/` (415 LoC: mod.rs + kernel.rs + lookup.rs + metal/qmm_t.metal.in)
- Reference: `[project_p8a_stage9_findings]` memory for self_qmm +35% over MLX baseline evidence
- Create: `docs/p5i-a-gather-kernel-feasibility.md` (committed design memo + ROI + 立项/否决 recommendation)

**No Metal source code written in P5i.a** — that's P5i.b if 立项.

### Step 4.1: Read + summarize self_qmm pattern

- [ ] Read all 4 files:

```bash
wc -l /Users/xin/workspace/ironmlx-backend/ironmlx/src/nn/self_qmm/{mod,kernel,lookup}.rs
cat /Users/xin/workspace/ironmlx-backend/ironmlx/src/nn/self_qmm/metal/qmm_t.metal.in | head -80
```

- [ ] Summarize in design memo draft:
  - What the dense Linear self-quant pattern does (template structure + tile lookup + dispatch contract)
  - Where +35% over MLX baseline came from per `[project_p8a_stage9_findings]` (reuse existing evidence; do NOT re-bench unless stale per spec § 4.4)
  - Which design choices are reusable for gather pattern vs which need adaptation

### Step 4.2: Design analogous gather_qmm.metal.in template

- [ ] Pseudocode the gather extensions to the dense self_qmm pattern:
  - expert_indices indirection (per-thread or per-tile lookup of which expert's weights to read)
  - per-expert weight gather (or per-token-expert combined gather)
  - scatter result back to output buffer (per-token-expert position)
  - per-tile thread group layout (mirror self_qmm tile design adapted for gather; specifically: how does tile size interact with the gather indirection memory pattern?)

### Step 4.3: ROI estimate with confidence interval

- [ ] Compute per spec § 4.4 estimate format:
  - Upper bound: gain analogous to self_qmm +35% → gather_qmm wall reduction ~35% × 35% root share ≈ 12% root_inclusive at PP=128/512
  - Lower bound: gather indirection overhead reduces gain to 10-15% kernel wall reduction → ~3-5% root_inclusive
  - Explicit confidence range — what's the source of uncertainty (gather indirection cost, expert dispatch granularity, scatter contention)

### Step 4.4: Cost estimate

- [ ] Per spec § 4.4:
  - 2-4 weeks Metal kernel impl + correctness validation + sweep + integration
  - Per `[feedback_task_breakdown_bounded]` would need P5i.b decomposition: e.g. P5i.b.1 kernel impl + microbench, P5i.b.2 correctness oracle + smoke, P5i.b.3 integration + sweep, P5i.b.4 close-out.

### Step 4.5: Write `docs/p5i-a-gather-kernel-feasibility.md`

- [ ] Compose the design memo with these sections:
  - § 1 Summary (1 paragraph + 立项/否决 verdict)
  - § 2 self_qmm precedent (cite +35% evidence; reuse don't re-bench)
  - § 3 gather pattern design (pseudocode + dispatch contract + thread group layout)
  - § 4 Bench plan (microbench harness shape; correctness oracle vs MLX upstream)
  - § 5 ROI estimate (range + confidence)
  - § 6 Cost estimate (P5i.b task decomposition outline)
  - § 7 Scope gate hook — explicit "P5i.b requires Boss approval before kernel impl begins" per spec § 5
  - § 8 Recommendation (立项 OR 否决 with rationale)

### Step 4.6: Commit T3

- [ ] Commit:

```bash
cd /Users/xin/workspace/ironmlx-backend
git add docs/p5i-a-gather-kernel-feasibility.md
git commit -m "$(cat <<COMMIT
docs(p5i-a-t3): self-quant gather Metal kernel feasibility memo (Level a; design only)

Per P5i.a spec § 4.4 Level (a) feasibility-only — bench plan + minimal
prototype design + ROI estimate. NO Metal kernel implementation in
P5i.a (that's P5i.b if 立项, requiring Boss Scope gate approval).

Memo sections:
* self_qmm precedent (reuse [project_p8a_stage9_findings] +35% evidence; no re-bench)
* gather pattern design (expert_indices indirection + per-expert weight
  gather + scatter result + thread group layout)
* Bench plan (microbench harness + correctness oracle vs MLX upstream)
* ROI estimate: upper bound ~12% root_inclusive at PP=128/512 (analogous
  to self_qmm +35%); lower bound ~3-5% (gather indirection overhead).
* Cost estimate: 2-4 weeks Metal impl + correctness + sweep + integration;
  P5i.b would decompose into 4 sub-phases per [feedback_task_breakdown_bounded]
* Recommendation: <立项 / 否决> with rationale

Per spec § 5: Level (a) Metal kernel rewrite requires explicit Boss
approval before P5i.b work begins.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
```

---

## Task 5: T4 — Level (d) backup: gda_step_1a_in_proj_qkvz op-level tuning (CONDITIONAL)

**Files:**
- Modify (if landed): `ironmlx/src/nn/gated_delta_net.rs` (step_1a site)
- Modify (if tile change): `ironmlx/src/nn/self_qmm/lookup.rs` (M5 Max tile entry)
- Output (gitignored): `reports/p5i-a-bench-log.md` (T4 section)

### Step 5.1: Trigger check (per spec § 4.5)

- [ ] Read latest cumulative state from `reports/p5i-a-bench-log.md` + `/tmp/p5i-a-baseline-summary.json` (post-T1+T2 effective).

- [ ] Evaluate triggers (ANY ONE triggers T4 execution; ALL must be false to skip T4):
  - PP=128 still below omlx+10% after T1+T2 landed work, OR
  - PP=512 still misses omlx+10% target by >5% (i.e. delta_pct < +5%), OR
  - T3 recommends 否决/延期 for Level (a) Metal kernel rewrite (so Level d is the remaining gain source)

- [ ] If none triggered (Full PASS achieved by T1+T2 + T3 立项 ready) → SKIP T4; document in bench log; jump to Task 6 (T5 close-out).

- [ ] If any triggered → continue with Step 5.2.

### Step 5.2: M5 Max tile sweep

- [ ] Read existing `gda_step_1a_in_proj_qkvz` site:

```bash
grep -nE 'gda_step_1a_in_proj_qkvz|in_proj_qkvz' /Users/xin/workspace/ironmlx-backend/ironmlx/src/nn/gated_delta_net.rs | head -5
sed -n '<line-5>,<line+25>p' /Users/xin/workspace/ironmlx-backend/ironmlx/src/nn/gated_delta_net.rs
```

(Fill `<line>` from grep output.)

- [ ] Read self_qmm tile lookup table:

```bash
cat /Users/xin/workspace/ironmlx-backend/ironmlx/src/nn/self_qmm/lookup.rs
```

Identify the current M5 Max entry (or M1 Pro fallback if M5 Max not in table).

- [ ] Design tile sweep: enumerate plausible tile params for the in_proj_qkvz shape (rows × cols × bits = 4 of the relevant Linear). Typically: thread group size {32, 64, 128} × tile height {8, 16, 32, 64} × tile width {64, 128, 256}.

- [ ] Microbench each tile candidate using the existing `ironmlx-bench-kernel` binary in this workspace; do not add a new benchmark dependency. Identify best M5 Max tile params.

### Step 5.3: Integrate best tile param into lookup table

- [ ] If best tile differs from current entry: update `lookup.rs` with M5 Max-specific entry (per `[feedback_device_aware_tile]` ironmlx Metal kernel should be device-aware).

- [ ] Smoke test (production parity):

```bash
source /tmp/p5i-a-env.sh
MLX_DIR=$HOME/.local/mlx IRONMLX_MOE_MODEL_DIR="$SNAP" \
  cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

Expected: 2 passed.

- [ ] iron-bench sweep (same T0 protocol; one ironmlx spawn cycle):

```bash
# Spawn + preheat + sweep + kill; save to /tmp/p5i-a-t4-ironmlx.csv
uv run python tools/p5i_a_baseline_aggregate.py \
  --ironmlx-csv /tmp/p5i-a-t4-ironmlx.csv \
  --omlx-csv /tmp/p5i-a-baseline-omlx.csv \
  --out-json /tmp/p5i-a-t4-summary.json
```

### Step 5.4: T4 land or reject + commit

- [ ] Land if ≥1% pp_tps improvement at PP=128 OR PP=512 (no regression beyond ±2% on the other).
- [ ] If the first T4 sweep meets the land condition, run one independent confirm sweep with a fresh ironmlx spawn + 5-min preheat + PP=128/512 measurement. Save it to `/tmp/p5i-a-t4-confirm-ironmlx.csv` and aggregate to `/tmp/p5i-a-t4-confirm-summary.json`. Land only if both sweeps meet the condition.
- [ ] Reject if no improvement; revert lookup change; document in bench log.

- [ ] If LAND, hygiene + commit:

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release

git add ironmlx/src/nn/self_qmm/lookup.rs ironmlx/src/nn/gated_delta_net.rs
git commit -m "$(cat <<COMMIT
perf(p5i-a-t4): gda_step_1a_in_proj_qkvz M5 Max tile tuning (Level d backup)

Per P5i.a spec § 4.5 Level (d) op-level tuning.

Triggered by: <Full PASS not achieved by T1+T2 / PP=512 gap >5% / T3 否决>

self_qmm tile param updated for M5 Max:
* Old tile: <prev params>
* New tile: <best params from sweep>
* Sweep range: <enumeration>

Bench result (vs T0 baseline):
* PP=128 ironmlx pp_tps median: <baseline> -> <new> (<+N>%)
* PP=512 ironmlx pp_tps median: <baseline> -> <new> (<+M>%)
* Smoke (p5_qwen35_moe_smoke): PASS

Device-aware lookup structure preserved (per [feedback_device_aware_tile]);
M1/M2/M3/M4 entries unchanged (cross-device validation deferred per
[project_cross_device_tuning_deferred]).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
```

---

## Task 6: T5 — Close-out

**Files:**
- Create: `docs/p5i-a-close-out.md` (committed)
- Modify (conditional): `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2 (IFF Full PASS on PP=128 +10%)
- Modify (outside repo): `~/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md` (extend with P5i.a closure section)

### Step 6.1: Aggregate final state

- [ ] Run aggregator one more time with the final cumulative ironmlx state:

```bash
# Final ironmlx sweep (post-T1+T2+T4 landed work; same T0 protocol)
# Save to /tmp/p5i-a-final-ironmlx.csv
uv run python tools/p5i_a_baseline_aggregate.py \
  --ironmlx-csv /tmp/p5i-a-final-ironmlx.csv \
  --omlx-csv /tmp/p5i-a-baseline-omlx.csv \
  --out-json /tmp/p5i-a-final-summary.json
cat /tmp/p5i-a-final-summary.json
```

### Step 6.1a: Optional Dense diagnostic control (non-blocking; outside gate)

- [ ] Evaluate whether Dense control is needed. Run it only if MoE T0/final numbers are surprising or the residual gap is hard to attribute. Do **not** add Dense results to the P5i.a Close Gate and do **not** start Dense-specific optimization work in this phase.

- [ ] If needed, run a lightweight Qwen3.5 Dense PP=128/512 ironmlx-vs-omlx comparison with the same T0 protocol (`--runs 7 --warmup 1`, serial backend sweeps, fixed PP order, 5-min preheat). Save raw data to `/tmp/p5i-a-dense-ironmlx.csv` + `/tmp/p5i-a-dense-omlx.csv` and summarize in `reports/p5i-a-bench-log.md`.

- [ ] Interpret only as a diagnostic:
  - Dense passes while MoE misses → continue to attribute the residual gap to MoE-specific routing/gather_qmm/expert dispatch.
  - Dense also misses → document a follow-up for common text-only prefill pipeline investigation outside P5i.a.

### Step 6.2: Determine close-out status (per spec § 3.2 vocab)

- [ ] Classify per spec § 3.2:
  - **Full PASS**: 4 deliverables done AND PP=128 final delta_pct ≥ +10%
  - **Feasibility PASS**: 4 deliverables done AND PP=128/512 still miss target AND all in-scope Level b/c/d candidates exhausted (each either landed or written negative finding) AND remaining path tied to follow-up
  - **Blocked**: baseline invalid OR T3 verdict missing OR an in-scope candidate with plausible ≥1% gain remains untested

### Step 6.3: Write `docs/p5i-a-close-out.md`

- [ ] Compose with these sections:
  - Status (Full PASS / Feasibility PASS / Blocked) + date + branch + commit chain
  - § 1 Close Gate 4-condition result (cite per-PP measurements)
  - § 2 Per-PP final state table: baseline ironmlx vs omlx delta → final ironmlx vs omlx delta. PP=128 +X% verdict; PP=512 -Y% gap remaining
  - § 3 What landed (per-task summary: T1 simplifications landed + rejected; T2 fusion verdict; T4 tile tuning if triggered)
  - § 4 Self-quant gather kernel verdict (T3 outcome — 立项 → recommend P5i.b spec; OR 否决 → document why + alternative for PP=512)
  - § 5 P5i+ follow-up:
    - If T3 立项: P5i.b spec writing as next phase
    - If T3 否决: P5i.c new candidate discovery (e.g., scheduler overhead investigation, KV cache layout) — out of P5i.a scope
    - P5h+2 items carried (validate_chunk_ancestry cycle, P5hChunkContextGuard.active dead field, roi_ranking.py stale literal, GA kv_mask_update duplicate-eval, emit cost reduction, T0b H4 same-mode control, T4.2 mid-admit ctx plumbing, spec § 1.2 PP=2048 partition — all unchanged from P5h+1 § 7.2.1.5)
  - Optional Dense diagnostic control result, if Step 6.1a ran; explicitly state it is outside the Close Gate.
  - § 6 Memory update — link to `[project_p5h_findings]` extension
  - § 7 References

### Step 6.4: Update memory (outside repo)

- [ ] Edit `~/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md`. Append section:

```markdown
## P5i.a closure update (YYYY-MM-DD, same-day after P5h+1)

P5i.a closed as **<Full PASS / Feasibility PASS / Blocked>**. Branch
`ironmlx-p5i-a-gather-qmm-feasibility` (forked from ironmlx-p5h-perf
6579633).

T0 baseline (canonical external comparison; NOT P5h+1 ironmlx-only ranking):
- PP=128 ironmlx <X> pp_tps vs omlx <Y> pp_tps; delta_pct = <+Z%>
- PP=512 ironmlx <X> pp_tps vs omlx <Y> pp_tps; delta_pct = <+Z%>

Post-P5i.a cumulative result:
- PP=128 ironmlx <X> pp_tps; delta vs omlx = <+Z%>; +10% target: ✓ / ✗
- PP=512 ironmlx <X> pp_tps; delta vs omlx = <+Z%>; gap remaining: <N%>

Landed in P5i.a:
- T1: <N> simplifications landed, <M> rejected
- T2 gate+up fusion: <立项 → landed / 否决>
- T4 gda_step_1a tile tuning: <triggered + landed / skipped>

T3 verdict: <立项 → P5i.b recommended (Scope gate Boss approval required) / 否决 → alternative is P5i.c new candidate discovery>

P5h+2 follow-up list unchanged (carried from P5h+1 § 7.2.1.5).
```

Also update the description field in the file frontmatter if Full PASS achieves PP=128 +10%.

### Step 6.5: Update spec § 7.2 (conditional)

- [ ] If status = Full PASS AND PP=128 ironmlx ≥ omlx+10%:
  - Edit `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2.1 (add P5i.a closure subsection or update existing P5h+1 cross-reference):

```markdown
### 7.2.1.6 P5i.a close-out cross-reference (YYYY-MM-DD)

P5i.a closed Full PASS. PP=128 ironmlx pp_tps now ≥ omlx+10% on Qwen3.5-35B-A3B-4bit baseline (T0 measurement). PP=512 gap remaining: <N%>. T3 verdict: <立项 / 否决>. See `docs/p5i-a-close-out.md`.
```

- [ ] If status = Feasibility PASS OR Blocked: spec § 7.2 stays UNCHANGED from P5h+1 post-state; close-out doc cites the unchanged state explicitly.

### Step 6.6: Final hygiene + commit

- [ ] Hygiene check on all touched files:

```bash
cd /Users/xin/workspace/ironmlx-backend
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
uv run --with ruff ruff check tools/p5i_a_baseline_aggregate.py
uv run --with ruff ruff format --check tools/p5i_a_baseline_aggregate.py
```

All should be clean.

- [ ] Commit T5:

```bash
git add docs/p5i-a-close-out.md
if [ -n "$(git diff --name-only docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md)" ]; then
  git add docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md
fi
git commit -m "$(cat <<COMMIT
docs(p5i-a-t5): close-out — <Full PASS / Feasibility PASS / Blocked>

Per docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md § 4.6
and docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md T5.

P5i.a closes as <status>:
* PP=128 ironmlx pp_tps: T0 baseline <X> -> post-P5i.a <Y> (delta vs omlx <+Z%>; +10% target <met / missed by N%>)
* PP=512 ironmlx pp_tps: T0 baseline <X> -> post-P5i.a <Y> (delta vs omlx <+Z%>; gap remaining <N%>)

What landed:
* T1: <N> Level (b) simplifications, <M> rejected (per-candidate detail in
  reports/p5i-a-bench-log.md; rejected candidates documented as evidence
  for Feasibility PASS gate)
* T2 gate+up fusion (Level c): <landed / 否决 + rationale>
* T4 gda_step_1a tile tuning (Level d): <landed / skipped (triggers not met) / 否决>

T3 self-quant gather kernel verdict: <立项 → P5i.b spec recommended; OR
否决 → PP=512 gap-closing alternative is P5i.c new candidate discovery>.
Per spec § 5: P5i.b kernel implementation requires explicit Boss Scope
gate approval before work begins.

P5h+2 follow-up list carried unchanged from P5h+1 § 7.2.1.5.

Memory project_p5h_findings.md extended with P5i.a closure section.

<IF Full PASS: spec § 7.2.1.6 added documenting P5i.a Full PASS; otherwise
spec § 7.2 unchanged from P5h+1 post-state>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
```

---

## Self-Review

### Spec coverage (against spec § 1 + § 3 + § 4 + § 5 + § 6 + § 7 + § 8)

- Spec § 1 Goal — Plan tasks 1-6 cover all 4 goals (baseline + low-risk landings + PP=512 gap quantification + 立项/否决 verdict).
- Spec § 3.1 Program target T-Target-B — Plan retains as overall P5i program goal; first-phase gate (§ 3.2) is what P5i.a tasks close.
- Spec § 3.2 Close Gate 4 conditions — Task 1 (T0) covers condition #1; Tasks 2-5 (T1-T4) cover condition #2; Task 6 step 6.2 + 6.3 explicit on Full PASS / Feasibility PASS / Blocked vocab (condition #3 + #4).
- Spec § 4.1-4.6 tasks T0-T5 — Plan tasks 1-6 mirror 1:1.
- Spec § 5 Scope gate — Plan Task 4 (T3) explicitly restricts Level (a) to design memo only; commit message + memo content include the Scope gate hook for P5i.b.
- Spec § 6 Out of scope — Plan does not include any deferred implementation items (P5i.b impl / P5j / cross-device M1-M4 / P5h+2 follow-ups / algorithmic exploration). Optional Dense control in Step 6.1a is diagnostic-only, outside the Close Gate, and cannot start Dense-specific optimization work.
- Spec § 7 Validation gates 1-7 — Plan tasks include production parity smoke + cumulative pp_tps noise band + omlx baseline integrity + serial execution + 5-min preheat + iron-bench --prompt-len exact + Rust hygiene cycle (per CLAUDE.md).
- Spec § 8 Sequencing — Plan T5 commit message includes "P5i.b kernel implementation requires explicit Boss Scope gate approval" hook.

### Placeholder scan

- `<line>` / `<line+25>` in Task 5 Step 5.2 — implementer reads grep output to fill. Reasonable runtime substitution; not a TBD.
- `<line>` in Task 2 Step 2.1 (commit message template `Site: sparse_moe.rs:<line>`) — implementer fills from actual edit site.
- `<N>` / `<M>` / `<X>` / `<Y>` / `<Z>` in commit message templates — implementer fills from measurement output. Reasonable per-experiment substitution.
- `<status>` / `<立项 / 否决>` / `<landed / 否决 / skipped>` in T5 commit message — implementer fills from determined close-out state.
- `<FILL>` markers in `docs/p5i-a-baseline.md` template (Step 1.13) — implementer fills from `/tmp/p5i-a-baseline-summary.json` output.

All placeholders are runtime substitutions tied to measurement data, not "TBD / implement later" markers. Per `writing-plans` skill rule "every step must contain the actual content an engineer needs", these are acceptable because the surrounding step text shows exactly where to source each value.

### Type consistency

- Aggregator JSON schema (`per_pp.<pp>.{ironmlx_pp_tps_median, omlx_pp_tps_median, delta_pct, passes_plus10_target}`) referenced consistently in Task 1 Step 1.12 (creation), Task 2 Step 2.2 (consumption), Task 3 Step 3.4 (consumption), Task 5 Step 5.3 (consumption), Task 6 Step 6.1 (final aggregation).
- Aggregator input contract is now strict: CSV must contain PP=128 and PP=512, exactly 7 measured rows per PP per backend, positive `pp_tps`, and the non-request-id iron-bench CSV schema. P5i.a uses `--warmup 1` and therefore does not enable server request-id capture.
- File paths consistent: `/tmp/p5i-a-baseline-ironmlx.csv` + `/tmp/p5i-a-baseline-omlx.csv` + `/tmp/p5i-a-baseline-summary.json` used throughout; per-iteration `/tmp/p5i-a-{cn,t2,t4,final}-*` follow predictable pattern.
- Acceptance criteria consistent: ≥1% pp_tps improvement at PP=128 OR PP=512 with no regression beyond ±2% noise band on the other PP — same wording in Task 2 Step 2.2, Task 3 Step 3.4, Task 5 Step 5.4.
- Repeatability is explicit for landed perf changes: T1/T2/T4 require an independent confirm sweep before LAND when the first sweep meets threshold.
- Rust hygiene is complete for Rust-modifying tasks: `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.
- Close-out vocab consistent: "Full PASS / Feasibility PASS / Blocked" used in Task 6 Step 6.2 + 6.3 + 6.6 commit message.

Post-Codex-review fixes applied inline: local iron-bench path, request-id/warmup conflict, env/PID persistence, strict baseline aggregation, MLX C++ path, repeatability protocol, T2 branch correctness gate, Rust build hygiene, uv ruff invocation, and optional Dense diagnostic control. No remaining issues found in this self-review.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Fresh subagent per task + two-stage review (spec compliance + code quality) after each. Established pattern from P5h+1 T1-T5.

2. **Inline Execution** — Execute tasks in this session with executing-plans skill; batch execution with checkpoints.

Which approach?
