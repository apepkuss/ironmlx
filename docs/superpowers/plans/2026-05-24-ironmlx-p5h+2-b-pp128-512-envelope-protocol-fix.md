# P5h+2.b — PP=128/512 Production Envelope Protocol Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close P5i.c Phase 0 § 7 #4 hard gate by achieving ironmlx production `pp_tps` envelope ≤ ±2% per PP on ≥3 fresh-spawn repeats at PP=128 + PP=512 under a final protocol, with explicit per-PP mechanism statement (explained or eliminated).

**Architecture:** T0 offline analyzes existing Phase 0 r1-r4 data to localize outlier source (client-side vs server-side). T1/T2 run sequential protocol-state and logging-perturbation experiments via an extended capture harness + new iron-bench `--capture-run-timestamps` CLI surface + new Python protocol-experiment driver; T3 piggybacks powermetrics sidecars on those runs and performs overlay analysis only. T4 codifies the final protocol and runs an acceptance sweep gated on envelope ≤ ±2%. T5 PASS close-out backfills Phase 0; T5F is the FAIL/DEFERRED close-out if the 15hr cap escalates.

**Tech Stack:** Rust (`cargo test --features p5h-profile`), `iron-bench` workspace member (Rust + clap CLI), Python 3 (stdlib + pytest + ruff), `tools/p5h_aggregator/multi_repeat.py`, `tools/p5i_c_pp_tps_envelope.py`, `ironmlx/tests/p5i_c_phase_0_capture.rs`, P5h+2.a monolithic ≥300s preheat methodology, M5 Max `--runs 1100` calibration, macOS `powermetrics` (observational; sudo or sysadmin perms may be required).

**Spec ref:** `docs/superpowers/specs/2026-05-24-ironmlx-p5h+2-b-pp128-512-envelope-protocol-fix-design.md` (commit `aabf21f`).

---

## File structure (created / modified by this plan)

**Create:**
- `tools/p5h_2b_t0_outlier_source.py` — T0 offline analyzer joining bench.csv + server.log root spans per PP per cell; emits per-PP verdict + decomposition table
- `tools/p5h_2b_protocol_experiment.py` — T1/T2/T4 driver: iterates experiment matrix rows, invokes capture harness with appropriate env vars, runs envelope analysis per experiment
- `tools/p5h_2b_thermal_overlay.py` — T3 powermetrics JSON parser + iron-bench timestamp joiner; emits per-experiment thermal alignment summary
- `tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py` — pytest for warmup-aware ordinal join + verdict logic
- `tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py` — pytest for powermetrics → iron-bench timestamp join
- `docs/p5h+2-b-protocol.md` — final protocol doc (T5 PASS only) OR rejected-candidate doc (T5F)
- `docs/p5h+2-b-close-out.md` — T5 PASS or T5F FAIL/DEFERRED close-out
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2b_findings.md` — memory entry (outside repo)
- `reports/p5h+2-b-bench-log.md` — bench log (gitignored)
- `reports/p5h+2-b-t0-outlier-source.md` — T0 analyzer output (gitignored)
- `reports/p5h+2-b-t4-predeclared-exclusions.md` — T4.1 locked exclusion rules (gitignored draft; folded into committed protocol doc at T5)

**Modify:**
- `ironmlx/tests/p5i_c_phase_0_capture.rs` — env var extensions (`P5I_C_SERVER_LIFECYCLE`, `P5I_C_PP_ORDER`, `P5I_C_LOGGING_MODE`); `meta.json` schema additions (warmup_count, server lifecycle timestamps, server spawn/health/preheat/measurement/kill Unix timestamps)
- `iron-bench/src/main.rs` — new `--capture-run-timestamps` clap arg; pass through to runner
- `iron-bench/src/runner.rs` — capture `run_start_unix_ns` + `run_end_unix_ns` per outcome
- `iron-bench/src/report.rs` — conditionally emit two new columns when `--capture-run-timestamps` is on; composable with `--capture-server-request-id`
- `docs/p5i-c-phase-0-close-out.md` — T5 backfill criterion #4 status + envelope numbers (PASS path) or add failed-attempt note (T5F path)
- `docs/p5i-c-phase-0-ranking-snapshot.md` — T5 backfill envelope section
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md` — add `project-p5h-2b-findings` index entry

**Do NOT modify:**
- Production `ironmlx serve` admission/scheduling logic
- MLX backend or model code
- Fan curve / system performance mode
- `tools/p5i_c_pp_tps_envelope.py` (acceptance gate tool; reused as-is)
- `tools/p5h_aggregator/multi_repeat.py` (reused as-is)

---

## Task 1: T0 — Offline outlier-source localization (~1.5 hr; no GPU)

**Files:**
- Create: `tools/p5h_2b_t0_outlier_source.py`
- Create: `tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py`
- Output: `reports/p5h+2-b-t0-outlier-source.md` (gitignored)

- [ ] **Step 1.1: Inspect existing Phase 0 data shape**

```bash
cd /Users/xin/workspace/ironmlx-backend
ls /tmp/p5i-c-phase-0-r1-pp128-probe/   # expect: bench.csv, server.log, meta.json
head -2 /tmp/p5i-c-phase-0-r1-pp128-probe/bench.csv  # confirm columns + request_id presence
cat /tmp/p5i-c-phase-0-r1-pp128-probe/meta.json     # confirm warmup field absent (legacy)
head -2 /tmp/p5i-c-phase-0-r1-pp128-production/bench.csv  # confirm production CSV has NO request_id
```

Expected: probe CSV has `request_id` column; production CSV does not. Legacy `meta.json` from Phase 0 does NOT have a `warmup_count` field (per spec § 5.1 inference rule).

- [ ] **Step 1.2: Create `tools/p5h_2b_t0_outlier_source.py` skeleton**

```python
"""P5h+2.b T0 — offline outlier-source localization.

Per spec § 5.1: joins per-cell bench.csv with server.log root spans to
decompose each run's wall time into client_overhead + server_root_inclusive_us.
Verdict per PP: client_side / server_side / cross / inconclusive.

Probe-mode cells: join via `request_id` column.
Production-mode cells: warmup-aware ordinal join. Legacy Phase 0 cells lack
`warmup_count` in meta.json; infer warmup=1 for production, warmup=0 for probe,
and mark `legacy_warmup_inferred=True`.

CLI:
    python tools/p5h_2b_t0_outlier_source.py \\
        --cells-glob '/tmp/p5i-c-phase-0-r1-pp*-*' \\
        --cells-glob '/tmp/p5i-c-phase-0-r2-pp*-*' \\
        --cells-glob '/tmp/p5i-c-phase-0-r3-pp*-*' \\
        --cells-glob '/tmp/p5i-c-phase-0-r4-pp*-*' \\
        --out-md reports/p5h+2-b-t0-outlier-source.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# parse_line lives in p5h_aggregator package — import via path insertion since
# tools/ is the package root.
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
from p5h_aggregator.schema_validator import parse_line  # noqa: E402

DEFAULT_PROBE_WARMUP = 0
DEFAULT_PRODUCTION_WARMUP = 1
OUTLIER_THRESHOLD_PCT = 10.0  # run flagged outlier if pp_tps deviates >10% from cell median


@dataclass
class RunDecomp:
    run_idx: int
    pp_tps: float
    ttft_ms: float
    server_root_inclusive_us: float | None
    client_overhead_us: float | None
    is_outlier: bool


@dataclass
class CellVerdict:
    cell_dir: str
    pp: int
    mode: str
    warmup_count: int
    legacy_warmup_inferred: bool
    runs: list[RunDecomp] = field(default_factory=list)
    verdict: str = "inconclusive"  # client_side / server_side / cross / inconclusive
    note: str = ""
```

- [ ] **Step 1.3: Add cell parsing logic with warmup-aware ordinal join**

Append to `tools/p5h_2b_t0_outlier_source.py`:

```python
def _parse_meta(cell_dir: Path) -> tuple[int, bool, str]:
    """Return (warmup_count, legacy_warmup_inferred, mode)."""
    meta_path = cell_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{cell_dir}: meta.json missing")
    meta = json.loads(meta_path.read_text())
    mode = meta.get("mode", "")
    if "warmup_count" in meta:
        return meta["warmup_count"], False, mode
    # Legacy Phase 0 inference per spec § 5.1
    if mode == "production":
        return DEFAULT_PRODUCTION_WARMUP, True, mode
    if mode == "probe":
        return DEFAULT_PROBE_WARMUP, True, mode
    raise SystemExit(f"{cell_dir}: unknown mode {mode!r}; cannot infer warmup")


def _parse_bench(cell_dir: Path) -> tuple[list[dict], bool]:
    """Parse bench.csv via DictReader. Ignore empty / malformed trailing rows."""
    bench_path = cell_dir / "bench.csv"
    with bench_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [
            row for row in reader
            if (row.get("run_idx") or "").strip() and (row.get("pp_tps") or "").strip()
        ]
        has_rid = "request_id" in fieldnames
    return rows, has_rid


def _expected_prompt_tokens(row: dict, pp: int) -> int:
    """Expected server-side prompt token count for this mono-PP cell."""
    server = (row.get("prompt_tokens_server") or "").strip()
    if server:
        return int(server)
    local = (row.get("prompt_tokens_local") or "").strip()
    if local:
        # Qwen chat-template overhead observed in Phase 0 synthetic prompts.
        return int(local) + 12
    return pp + 12


def _parse_roots(cell_dir: Path) -> list:
    """Parse server.log; return root spans (parent_span_id is None) in
    emission order. Filters None from parse_line + non-root spans."""
    log_path = cell_dir / "server.log"
    roots = []
    with log_path.open() as f:
        for line in f:
            span = parse_line(line)
            if span is None:
                continue
            if span.parent_span_id is None:
                roots.append(span)
    return roots


def decompose_cell(cell_dir: Path) -> CellVerdict:
    pp = int(re.search(r"-pp(\d+)-", str(cell_dir)).group(1))
    warmup, legacy, mode = _parse_meta(cell_dir)
    bench_rows, has_rid = _parse_bench(cell_dir)
    roots = _parse_roots(cell_dir)
    verdict = CellVerdict(
        cell_dir=str(cell_dir),
        pp=pp,
        mode=mode,
        warmup_count=warmup,
        legacy_warmup_inferred=legacy,
    )

    if not bench_rows:
        verdict.verdict = "inconclusive"
        verdict.note = "empty bench.csv"
        return verdict

    # Probe: join by request_id. Production: warmup-aware ordinal.
    if mode == "probe" and has_rid:
        root_by_rid = {s.request_id: s for s in roots if s.request_id}
        join = []
        missing = []
        for row in bench_rows:
            rid = row.get("request_id", "")
            root = root_by_rid.get(rid)
            if root is None:
                missing.append(rid)
            join.append((row, root))
        if missing:
            verdict.verdict = "inconclusive"
            verdict.note = f"probe request_id join missing {len(missing)} roots: {missing[:3]}"
            return verdict
    else:
        # Production-mode warmup-aware ordinal per spec § 5.1
        expected = warmup + len(bench_rows)
        if len(roots) != expected:
            verdict.verdict = "inconclusive"
            verdict.note = f"server root count {len(roots)} != expected (warmup={warmup} + measured={len(bench_rows)})"
            return verdict
        measured_roots = roots[warmup:]
        join = [(row, root) for row, root in zip(bench_rows, measured_roots)]

    prompt_mismatches = []
    for row, root in join:
        expected_prompt = _expected_prompt_tokens(row, pp)
        if root is None or root.prompt_tokens != expected_prompt:
            prompt_mismatches.append(
                (row.get("run_idx", "?"), expected_prompt, None if root is None else root.prompt_tokens)
            )
    if prompt_mismatches:
        verdict.verdict = "inconclusive"
        verdict.note = f"prompt_tokens mismatch in joined roots: {prompt_mismatches[:3]}"
        return verdict

    pp_tps_list = [float(r[0]["pp_tps"]) for r in join]
    cell_median = median(pp_tps_list)
    for row, root in join:
        root_us = root.inclusive_us if root is not None else None
        pp_tps = float(row["pp_tps"])
        ttft_ms = float(row["ttft_ms"])
        deviation_pct = abs(pp_tps - cell_median) / cell_median * 100 if cell_median > 0 else 0
        is_outlier = deviation_pct > OUTLIER_THRESHOLD_PCT
        client_overhead = (ttft_ms * 1000) - root_us if root_us is not None else None
        verdict.runs.append(RunDecomp(
            run_idx=int(row["run_idx"]),
            pp_tps=pp_tps,
            ttft_ms=ttft_ms,
            server_root_inclusive_us=root_us,
            client_overhead_us=client_overhead,
            is_outlier=is_outlier,
        ))

    # Verdict: examine outliers' decomposition
    outliers = [r for r in verdict.runs if r.is_outlier]
    if not outliers:
        verdict.verdict = "inconclusive"
        verdict.note = "no pp_tps outliers above threshold; no source classification needed"
        return verdict
    # Among outliers, fraction where server_root is also abnormally slow
    server_slow_count = 0
    for o in outliers:
        if o.server_root_inclusive_us is None:
            continue
        median_root = median(r.server_root_inclusive_us for r in verdict.runs if r.server_root_inclusive_us is not None)
        if o.server_root_inclusive_us > median_root * 1.1:
            server_slow_count += 1
    if server_slow_count == len(outliers):
        verdict.verdict = "server_side"
    elif server_slow_count == 0:
        verdict.verdict = "client_side"
    else:
        verdict.verdict = "cross"
    return verdict
```

- [ ] **Step 1.4: Add CLI + markdown report rendering**

Append:

```python
def render_md(verdicts: list[CellVerdict]) -> str:
    lines = [
        "# P5h+2.b T0 — Outlier-Source Localization",
        "",
        "Joins existing Phase 0 r1-r4 per-cell bench.csv with server.log root spans.",
        "Outlier threshold: ±10% deviation from cell median.",
        "",
        "## Per-cell verdict",
        "",
        "| Cell | PP | Mode | warmup | inferred? | runs | outliers | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        n_outliers = sum(1 for r in v.runs if r.is_outlier)
        lines.append(
            f"| {Path(v.cell_dir).name} | {v.pp} | {v.mode} | {v.warmup_count} | "
            f"{'Y' if v.legacy_warmup_inferred else 'N'} | {len(v.runs)} | {n_outliers} | "
            f"`{v.verdict}` |"
        )
    lines.append("")
    lines.append("## Per-PP per-run decomposition")
    lines.append("")
    for v in verdicts:
        if not v.runs:
            continue
        lines.append(f"### {Path(v.cell_dir).name}")
        lines.append("")
        lines.append("| run_idx | pp_tps | ttft_ms | server_root_us | client_overhead_us | outlier? |")
        lines.append("|---|---|---|---|---|---|")
        for r in v.runs:
            ot = "★" if r.is_outlier else ""
            root_str = f"{r.server_root_inclusive_us:.0f}" if r.server_root_inclusive_us is not None else "N/A"
            client_str = f"{r.client_overhead_us:.0f}" if r.client_overhead_us is not None else "N/A"
            lines.append(f"| {r.run_idx} | {r.pp_tps:.2f} | {r.ttft_ms:.2f} | {root_str} | {client_str} | {ot} |")
        lines.append("")
        if v.note:
            lines.append(f"_Note: {v.note}_")
            lines.append("")
    noted = [v for v in verdicts if v.note and not v.runs]
    if noted:
        lines.append("## Notes")
        lines.append("")
        lines.append("| Cell | Note |")
        lines.append("|---|---|")
        for v in noted:
            lines.append(f"| {Path(v.cell_dir).name} | {v.note} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells-glob", required=True, action="append",
                   help="glob for cell dirs (repeat for multiple patterns)")
    p.add_argument("--out-md", type=Path, required=True)
    args = p.parse_args()
    cell_dirs: list[Path] = []
    for pattern in args.cells_glob:
        cell_dirs.extend(sorted(Path(p) for p in glob.glob(pattern)))
    if not cell_dirs:
        raise SystemExit(f"no cells matched glob(s) {args.cells_glob}")
    verdicts = [decompose_cell(d) for d in cell_dirs]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(verdicts))
    print(f"Wrote {args.out_md}")
    # Also print per-PP cross-cell verdict summary to stdout
    by_pp: dict[int, list[str]] = {}
    for v in verdicts:
        by_pp.setdefault(v.pp, []).append(v.verdict)
    for pp in sorted(by_pp):
        print(f"PP={pp}: verdicts across cells = {by_pp[pp]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.5: Create pytest `tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py`**

```python
"""Pytest for tools/p5h_2b_t0_outlier_source.py (P5h+2.b T0)."""

from __future__ import annotations
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5h_2b_t0_outlier_source import (  # noqa: E402
    DEFAULT_PRODUCTION_WARMUP,
    decompose_cell,
)


def _write_cell(tmp_path: Path, pp: int, mode: str,
                bench_csv: str, server_log: str,
                meta: dict | None = None) -> Path:
    cell = tmp_path / f"r1-pp{pp}-{mode}"
    cell.mkdir()
    (cell / "bench.csv").write_text(bench_csv)
    (cell / "server.log").write_text(server_log)
    if meta is None:
        meta = {"mode": mode, "pp": pp}
    (cell / "meta.json").write_text(json.dumps(meta))
    return cell


def test_decompose_probe_request_id_join(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,request_id\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-a\n"
        "x,128,1,1,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-b\n"
        "x,128,1,2,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-c\n"
        "x,128,1,3,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-d\n"
        "x,128,1,4,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-e\n"
        "x,128,1,5,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-f\n"
        "x,128,1,6,100.0,1.0,1.0,500.0,1.0,128,140,1,0,length,req-g\n"
    )
    # Server log: 7 root spans matching req-a..req-g; req-g abnormally slow
    log_lines = []
    for i, (rid, dur_us) in enumerate([
        ("req-a", 90000.0),
        ("req-b", 90000.0),
        ("req-c", 90000.0),
        ("req-d", 90000.0),
        ("req-e", 90000.0),
        ("req-f", 90000.0),
        ("req-g", 180000.0),  # server-side slowdown
    ]):
        start_ns = 1000000000 + i * 200000000
        end_ns = int(start_ns + dur_us * 1000)
        log_lines.append(
            f"[p5h-profile] request_id={rid} routing_path=scheduler prompt_tokens=140 "
            f"seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
            f"span_name=root parent_span=null start_ns={start_ns} end_ns={end_ns} "
            f"mode=off span_kind=tree\n"
        )
    cell = _write_cell(tmp_path, 128, "probe", bench, "".join(log_lines),
                       meta={"mode": "probe", "warmup_count": 0})
    v = decompose_cell(cell)
    assert v.verdict == "server_side"
    # req-g outlier; its root_us should be ~180k (server-slow) not the median ~90k
    outliers = [r for r in v.runs if r.is_outlier]
    assert len(outliers) == 1


def test_decompose_production_warmup_aware_ordinal(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
        "x,128,1,1,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
    )
    # 1 warmup + 2 measured = 3 server roots
    log_lines = []
    for i in range(3):
        start_ns = 1000000000 + i * 200000000
        end_ns = start_ns + 100_000_000  # 100ms inclusive
        log_lines.append(
            f"[p5h-profile] request_id= routing_path=scheduler prompt_tokens=140 "
            f"seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
            f"span_name=root parent_span=null start_ns={start_ns} end_ns={end_ns} "
            f"mode=off span_kind=tree\n"
        )
    cell = _write_cell(tmp_path, 128, "production", bench, "".join(log_lines))
    v = decompose_cell(cell)
    assert v.warmup_count == DEFAULT_PRODUCTION_WARMUP
    assert v.legacy_warmup_inferred is True
    assert len(v.runs) == 2  # measured rows only after dropping warmup


def test_decompose_inconclusive_on_root_count_mismatch(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
        "x,128,1,1,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
    )
    # 1 warmup + 2 measured = 3 expected; provide only 2 → inconclusive
    log_lines = []
    for i in range(2):
        log_lines.append(
            f"[p5h-profile] request_id= routing_path=scheduler prompt_tokens=140 "
            f"seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
            f"span_name=root parent_span=null start_ns=1000 end_ns=101000 "
            f"mode=off span_kind=tree\n"
        )
    cell = _write_cell(tmp_path, 128, "production", bench, "".join(log_lines))
    v = decompose_cell(cell)
    assert v.verdict == "inconclusive"
    assert "root count" in v.note


def test_decompose_inconclusive_on_prompt_token_mismatch(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
    )
    log = (
        "[p5h-profile] request_id= routing_path=scheduler prompt_tokens=140 "
        "seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
        "span_name=root parent_span=null start_ns=1000 end_ns=101000 "
        "mode=off span_kind=tree\n"
        "[p5h-profile] request_id= routing_path=scheduler prompt_tokens=524 "
        "seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
        "span_name=root parent_span=null start_ns=2000 end_ns=102000 "
        "mode=off span_kind=tree\n"
    )
    cell = _write_cell(tmp_path, 128, "production", bench, log)
    v = decompose_cell(cell)
    assert v.verdict == "inconclusive"
    assert "prompt_tokens mismatch" in v.note
```

- [ ] **Step 1.6: Hygiene + pytest**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2b_t0_outlier_source.py tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py
uv run --with ruff ruff format --check tools/p5h_2b_t0_outlier_source.py tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py -v
```

Expected: all PASS.

- [ ] **Step 1.7: Run T0 analyzer on existing Phase 0 data**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run python tools/p5h_2b_t0_outlier_source.py \
  --cells-glob '/tmp/p5i-c-phase-0-r1-pp*-*' \
  --cells-glob '/tmp/p5i-c-phase-0-r2-pp*-*' \
  --cells-glob '/tmp/p5i-c-phase-0-r3-pp*-*' \
  --cells-glob '/tmp/p5i-c-phase-0-r4-pp*-*' \
  --out-md reports/p5h+2-b-t0-outlier-source.md
```

Expected: stdout shows per-PP verdict list across cells; file written.

- [ ] **Step 1.8: Manually review T0 verdict + record hypothesis ranking**

Read `reports/p5h+2-b-t0-outlier-source.md`. Interpret per-PP verdict:
- If PP=128 verdict is `client_side` → trailing outliers are iron-bench / HTTP / network jitter; T2 logging perturbation is the right next investigation.
- If PP=128 verdict is `server_side` → trailing outliers are ironmlx / MLX state; T2 likely won't help; consider expanding T1 matrix.
- If PP=512 verdict is `cross` or bimodal across cells → state machine hypothesis (T1) most likely.
- If verdicts conflict between r1+r4 (fast cluster) vs r2+r3 (slow cluster) → spawn-state hypothesis confirmed.

Append a "T0 hypothesis-ranking" section to `reports/p5h+2-b-bench-log.md` (gitignored; create file with `mkdir -p reports && ...`) summarizing the verdict per PP + which T1+T2 experiments are highest priority.

- [ ] **Step 1.9: T0 checkpoint (no commit yet)**

```bash
git diff -- tools/p5h_2b_t0_outlier_source.py tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py
git status --short
```

Expected: T0 analyzer + pytest files are present and validated, but do not commit here. Per design § 5.1/§ 5.6, reusable tooling is committed at T5 PASS or T5F FAIL/DEFERRED after the protocol outcome is known.

Note: the analyzer output `reports/p5h+2-b-t0-outlier-source.md` is gitignored and never committed.

---

## Task 2: Build harness extensions + iron-bench `--capture-run-timestamps` + protocol driver (~2-3 hr)

**Files:**
- Modify: `ironmlx/tests/p5i_c_phase_0_capture.rs` (env var extensions + meta.json schema additions)
- Modify: `iron-bench/src/main.rs` + `iron-bench/src/runner.rs` + `iron-bench/src/report.rs` (timestamps flag)
- Create: `tools/p5h_2b_protocol_experiment.py` (driver)

- [ ] **Step 2.1: Read existing harness + iron-bench code paths**

```bash
cd /Users/xin/workspace/ironmlx-backend
grep -nE "P5I_C_|env_or|parse_runs_per_pp" ironmlx/tests/p5i_c_phase_0_capture.rs | head -20
grep -nE "capture_server_request_id|render_csv|RequestResult" iron-bench/src/main.rs iron-bench/src/runner.rs iron-bench/src/report.rs | head -20
```

- [ ] **Step 2.2: Extend harness with `P5I_C_SERVER_LIFECYCLE`, `P5I_C_PP_ORDER`, `P5I_C_LOGGING_MODE` env vars**

Edit `ironmlx/tests/p5i_c_phase_0_capture.rs`. Replace the old `P5I_C_PP_LIST` doc/default/parser with `P5I_C_PP_ORDER`; do not keep a compatibility parser because clippy `-D warnings` will flag the old `parse_pp_list` as dead code after the refactor. Add the lifecycle/logging helpers at top of file:

Also update the imports to include `remove_dir_all`:

```rust
use std::fs::{create_dir_all, remove_dir_all, File, OpenOptions};
```

```rust
const DEFAULT_SERVER_LIFECYCLE: &str = "phase0_current";
const DEFAULT_PP_ORDER: &str = "128,512";
const DEFAULT_LOGGING_MODE: &str = "default_profile";

#[derive(Debug, Clone, Copy, PartialEq)]
enum ServerLifecycle {
    Phase0Current,         // dedicated preheat spawn + fresh measurement spawn per PP
    SameSpawnCrossPp,      // single spawn for preheat + both PP measurements
    SameSpawnPerPp,        // single spawn per PP for preheat + that PP measurement
}

fn parse_server_lifecycle() -> ServerLifecycle {
    match env_or("P5I_C_SERVER_LIFECYCLE", DEFAULT_SERVER_LIFECYCLE).as_str() {
        "phase0_current" => ServerLifecycle::Phase0Current,
        "same_spawn_cross_pp" => ServerLifecycle::SameSpawnCrossPp,
        "same_spawn_per_pp" => ServerLifecycle::SameSpawnPerPp,
        other => panic!("P5I_C_SERVER_LIFECYCLE invalid: {other:?}"),
    }
}

fn parse_pp_order() -> Vec<i32> {
    let out = env_or("P5I_C_PP_ORDER", DEFAULT_PP_ORDER)
        .split(',')
        .map(|s| s.trim().parse::<i32>().expect("P5I_C_PP_ORDER entries must be i32"))
        .collect::<Vec<_>>();
    if out.is_empty() {
        panic!("P5I_C_PP_ORDER must contain at least one PP");
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LoggingMode {
    DefaultProfile,    // RUST_LOG default; full [p5h-profile] info emission
    QuietAcceptance,   // RUST_LOG=error; no root-log decomposition possible
    BufferedProfile,   // info emission via buffered sink; decomposition after flush
}

fn parse_logging_mode() -> LoggingMode {
    match env_or("P5I_C_LOGGING_MODE", DEFAULT_LOGGING_MODE).as_str() {
        "default_profile" => LoggingMode::DefaultProfile,
        "quiet_acceptance" => LoggingMode::QuietAcceptance,
        "buffered_profile" => LoggingMode::BufferedProfile,
        other => panic!("P5I_C_LOGGING_MODE invalid: {other:?}"),
    }
}

fn now_unix_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
```

Then update `spawn_server_to_log` to honor `LoggingMode`:

```rust
struct ServerProcess {
    child: Child,
    stderr_drainer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}

fn spawn_server_to_log(
    model_dir: &str,
    mode: &str,
    log_path: &str,
    logging_mode: LoggingMode,
) -> std::io::Result<ServerProcess> {
    let bin = env!("CARGO_BIN_EXE_ironmlx");
    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--model",
        model_dir,
        "--port",
        &PORT.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    if mode == "probe" {
        cmd.arg("--p5h-measurement-eval-probes");
    }
    cmd.env_remove("IRONMLX_P5G_PROFILE_MODE");
    cmd.env("MLX_DIR", std::env::var("MLX_DIR").unwrap_or_default());
    match logging_mode {
        LoggingMode::QuietAcceptance => {
            cmd.env("RUST_LOG", "error");
            let log_file = OpenOptions::new().create(true).append(true).open(log_path)?;
            cmd.stderr(Stdio::from(log_file));
        }
        LoggingMode::BufferedProfile => {
            // T2 buffered sink: server stderr -> pipe; harness drainer buffers writes
            // to server.log. This preserves info-level root logs after flush while
            // removing per-line direct file writes from the server process.
            cmd.stderr(Stdio::piped());
        }
        LoggingMode::DefaultProfile => {
            let log_file = OpenOptions::new().create(true).append(true).open(log_path)?;
            cmd.stderr(Stdio::from(log_file));
        }
    }
    let mut child = cmd.spawn()?;
    let stderr_drainer = if logging_mode == LoggingMode::BufferedProfile {
        let stderr = child.stderr.take().expect("buffered_profile stderr pipe");
        let path = log_path.to_string();
        Some(std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stderr);
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            let mut writer = std::io::BufWriter::new(file);
            std::io::copy(&mut reader, &mut writer)?;
            writer.flush()?;
            Ok(())
        }))
    } else {
        None
    };
    Ok(ServerProcess { child, stderr_drainer })
}

fn kill_and_wait(mut server: ServerProcess) {
    let _ = server.child.kill();
    let _ = server.child.wait();
    if let Some(handle) = server.stderr_drainer.take() {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("[p5h+2-b WARN] buffered stderr drainer failed: {e}"),
            Err(_) => eprintln!("[p5h+2-b WARN] buffered stderr drainer panicked"),
        }
    }
}
```

- [ ] **Step 2.3: Extend `meta.json` schema with lifecycle + timestamps + logging mode**

Edit `capture_one_cell` to record richer meta. Replace existing `meta` string construction with:

```rust
fn write_cell_meta(
    meta_path: &str,
    repeat: u32,
    pp: i32,
    runs: usize,
    mode: &str,
    warmup_count: usize,
    preheat_wall_s: u64,
    server_lifecycle: ServerLifecycle,
    pp_order: &[i32],
    logging_mode: LoggingMode,
    server_spawn_unix_ns: u64,
    server_healthy_unix_ns: u64,
    preheat_start_unix_ns: u64,
    preheat_end_unix_ns: u64,
    measurement_start_unix_ns: u64,
    measurement_end_unix_ns: u64,
    server_kill_unix_ns: u64,
) -> std::io::Result<()> {
    let lifecycle_s = match server_lifecycle {
        ServerLifecycle::Phase0Current => "phase0_current",
        ServerLifecycle::SameSpawnCrossPp => "same_spawn_cross_pp",
        ServerLifecycle::SameSpawnPerPp => "same_spawn_per_pp",
    };
    let logging_s = match logging_mode {
        LoggingMode::DefaultProfile => "default_profile",
        LoggingMode::QuietAcceptance => "quiet_acceptance",
        LoggingMode::BufferedProfile => "buffered_profile",
    };
    let pp_order_s = pp_order
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\n  \"repeat\": {repeat},\n  \"pp\": {pp},\n  \"runs\": {runs},\n  \"mode\": \"{mode}\",\n  \"warmup_count\": {warmup_count},\n  \"preheat_wall_s\": {preheat_wall_s},\n  \"server_lifecycle\": \"{lifecycle_s}\",\n  \"pp_order\": \"{pp_order_s}\",\n  \"logging_mode\": \"{logging_s}\",\n  \"server_spawn_unix_ns\": {server_spawn_unix_ns},\n  \"server_healthy_unix_ns\": {server_healthy_unix_ns},\n  \"preheat_start_unix_ns\": {preheat_start_unix_ns},\n  \"preheat_end_unix_ns\": {preheat_end_unix_ns},\n  \"measurement_start_unix_ns\": {measurement_start_unix_ns},\n  \"measurement_end_unix_ns\": {measurement_end_unix_ns},\n  \"server_kill_unix_ns\": {server_kill_unix_ns},\n  \"port\": {PORT}\n}}\n"
    );
    let mut f = File::create(meta_path)?;
    f.write_all(json.as_bytes())?;
    f.sync_all()?;
    Ok(())
}
```

Then refactor `p5i_c_phase_0_capture_one_repeat` to track lifecycle + propagate timestamps. Three lifecycle modes:

- **`phase0_current`**: keep existing flow (one preheat spawn + per-cell spawn). Each cell gets its own server with its own timestamps.
- **`same_spawn_cross_pp`**: ONE server spawn for the entire repeat. Preheat in that spawn. Then iterate PPs in `pp_order` running iron-bench against the same server. Kill at end.
- **`same_spawn_per_pp`**: ONE server spawn per PP for the entire repeat. Preheat in that spawn. Measure PP. Kill. Move to next PP.

The refactored entry test must handle all three modes. Skeleton:

```rust
#[test]
#[ignore = "p5h+2-b — single-repeat capture with configurable server lifecycle (~6-15 min GPU); invoke explicitly per env vars"]
fn p5i_c_phase_0_capture_one_repeat() -> anyhow::Result<()> {
    let repeat = parse_repeat_index();
    let mode = parse_mode();
    let model_dir = ironmlx_model_dir();
    let pp_order = parse_pp_order();
    let runs_map = parse_runs_per_pp();
    let preheat_seconds: u64 = env_or("P5I_C_PREHEAT_SECONDS", DEFAULT_PREHEAT_SECONDS)
        .parse()
        .expect("P5I_C_PREHEAT_SECONDS must be u64");
    let lifecycle = parse_server_lifecycle();
    let logging_mode = parse_logging_mode();
    let warmup = if mode == "probe" { 0_usize } else { 1_usize };

    eprintln!(
        "[p5h+2-b] repeat={repeat} mode={mode} lifecycle={lifecycle:?} \
         pp_order={pp_order:?} runs={runs_map:?} logging={logging_mode:?} \
         preheat_target_s={preheat_seconds}"
    );

    match lifecycle {
        ServerLifecycle::Phase0Current => {
            run_phase0_current(repeat, mode, &model_dir, &pp_order, &runs_map,
                               preheat_seconds, lifecycle, logging_mode, warmup)
        }
        ServerLifecycle::SameSpawnCrossPp => {
            run_same_spawn_cross_pp(repeat, mode, &model_dir, &pp_order, &runs_map,
                                    preheat_seconds, lifecycle, logging_mode, warmup)
        }
        ServerLifecycle::SameSpawnPerPp => {
            run_same_spawn_per_pp(repeat, mode, &model_dir, &pp_order, &runs_map,
                                  preheat_seconds, lifecycle, logging_mode, warmup)
        }
    }
}
```

Implement the three `run_*` functions. Each must first remove any existing output directory for the cell/preheat/shared-log path before writing (`remove_dir_all` if it exists, then `create_dir_all`) so append-mode `server.log` never mixes stale records from a failed or previous run. Then:
1. Record `server_spawn_unix_ns` immediately before `spawn_server_to_log`
2. Record `server_healthy_unix_ns` after `wait_for_healthz` succeeds
3. Record `preheat_start_unix_ns` / `preheat_end_unix_ns` around the `monolithic_preheat` invocation
4. Record `measurement_start_unix_ns` / `measurement_end_unix_ns` around the iron-bench invocation per cell
5. Record `server_kill_unix_ns` immediately before `kill_and_wait`
6. Call `write_cell_meta` for each cell with all timestamps

For `same_spawn_cross_pp`, the preheat happens once but is recorded into BOTH cells' meta.json (same preheat_start/end timestamps).
Use one shared server log path for the repeat while the server is alive, then after `kill_and_wait` copy or symlink that log to each per-PP cell directory as `server.log` so every cell keeps the `{bench.csv, server.log, meta.json}` contract.

For `same_spawn_per_pp`, each PP has its own dedicated spawn+preheat+kill cycle.

- [ ] **Step 2.4: Verify Rust build + gate**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo build --release -p ironmlx --features p5h-profile --tests 2>&1 | tail -5
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -3
cargo build --release 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 2.5: Extend iron-bench with `--capture-run-timestamps`**

Edit `iron-bench/src/main.rs`. Add to the `Args` struct after `capture_server_request_id`:

```rust
    /// Append `run_start_unix_ns` and `run_end_unix_ns` columns to CSV output.
    /// When off, CSV is byte-identical to current output. When combined with
    /// `--capture-server-request-id`, both column families appear; downstream
    /// parsers MUST use header names (csv::DictReader) not fixed positions.
    #[arg(long, default_value_t = false)]
    pub capture_run_timestamps: bool,
```

Then reject concurrent mode when timestamp capture is requested, matching the existing request-id capture guard because concurrent CSV uses a different schema:

```rust
if args.capture_run_timestamps && args.concurrent.is_some() {
    anyhow::bail!(
        "--capture-run-timestamps is incompatible with --concurrent: \
         run_start_unix_ns/run_end_unix_ns are defined only for v1 sequential CSV rows."
    );
}
```

- [ ] **Step 2.6: Capture timestamps in iron-bench runner**

Edit `iron-bench/src/runner.rs`. The sequential CSV path uses `RunOutcome` (not concurrent-mode `RequestOutcome`). Add two optional fields there:

```rust
pub struct RunOutcome {
    pub run_idx: usize,
    pub prompt_tokens_local: usize,
    pub result: RequestResult,
    /// Wall-clock timestamp at request-send start (when --capture-run-timestamps is on).
    pub run_start_unix_ns: Option<u64>,
    /// Wall-clock timestamp at response-complete (when --capture-run-timestamps is on).
    pub run_end_unix_ns: Option<u64>,
}
```

Extend `run_cell(...)` with a `capture_run_timestamps: bool` argument immediately after `capture_request_id`, and pass it through from `main.rs`:

```rust
let cell = runner::run_cell(
    &client,
    target_name,
    target_url,
    &args.model,
    *pp,
    args.max_tokens,
    args.warmup,
    args.runs,
    args.capture_server_request_id,
    args.capture_run_timestamps,
    &tokenizer,
)
.await?;
```

In the sequential per-run loop, wrap the actual request issue:

```rust
let run_start_unix_ns = if capture_run_timestamps {
    Some(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0))
} else {
    None
};
// ... existing request execution ...
let run_end_unix_ns = if capture_run_timestamps {
    Some(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0))
} else {
    None
};
outcomes.push(RunOutcome {
    run_idx: i,
    prompt_tokens_local,
    result,
    run_start_unix_ns,
    run_end_unix_ns,
});
```

Do not add timestamp fields to `RequestOutcome`; that struct is for concurrent mode and is intentionally outside P5h+2.b.

- [ ] **Step 2.7: Emit timestamps in report.rs CSV**

Edit `iron-bench/src/report.rs`. Modify `render_csv`:

```rust
pub fn render_csv(
    cells: &[CellResult],
    capture_request_id: bool,
    capture_run_timestamps: bool,
) -> String {
    let mut out = String::new();
    let base = "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason";
    let mut header = String::from(base);
    if capture_request_id {
        header.push_str(",request_id");
    }
    if capture_run_timestamps {
        header.push_str(",run_start_unix_ns,run_end_unix_ns");
    }
    header.push('\n');
    out.push_str(&header);
    for c in cells {
        for outcome in &c.runs {
            out.push_str(&csv_row(c, outcome));
            if capture_request_id {
                out.push(',');
                out.push_str(outcome.result.request_id.as_deref().unwrap_or(""));
            }
            if capture_run_timestamps {
                out.push(',');
                out.push_str(&outcome.run_start_unix_ns.map(|v| v.to_string()).unwrap_or_default());
                out.push(',');
                out.push_str(&outcome.run_end_unix_ns.map(|v| v.to_string()).unwrap_or_default());
            }
            out.push('\n');
        }
    }
    out
}
```

Update the single call site in `main.rs`:

```rust
OutputFormat::Csv => report::render_csv(
    &seq_cells,
    args.capture_server_request_id,
    args.capture_run_timestamps,
),
```

Update existing `iron-bench/src/report.rs` unit tests that call `render_csv(&[cell], ...)` so the new third argument is explicit (`false` unless the test is checking timestamp columns). Add one focused timestamp-column test:

```rust
#[test]
fn csv_includes_run_timestamps_when_enabled() {
    let mut cell = CellResult {
        target_name: "ironmlx".into(),
        target_url: "http://localhost:8080".into(),
        pp_target: 128,
        tg_target: 64,
        runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
    };
    cell.runs[0].run_start_unix_ns = Some(1_000);
    cell.runs[0].run_end_unix_ns = Some(2_000);
    let csv = render_csv(&[cell], false, true);
    assert!(csv.lines().next().unwrap().ends_with(",run_start_unix_ns,run_end_unix_ns"));
    assert!(csv.contains(",1000,2000\n"));
}
```

Also update `fake_outcome(...)` in the same test module to initialize `run_start_unix_ns: None` and `run_end_unix_ns: None`.

- [ ] **Step 2.8: Update harness iron-bench invocation to pass `--capture-run-timestamps`**

In `ironmlx/tests/p5i_c_phase_0_capture.rs` `capture_one_cell` (and per-lifecycle equivalents), add `"--capture-run-timestamps".to_string()` to the iron_args vector unconditionally for all P5h+2.b cells (production AND probe). Per spec § 6 compose constraint.

- [ ] **Step 2.9: Rust gate after iron-bench + harness changes**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -3
cargo build --release
cargo test --release -p iron-bench 2>&1 | tail -5  # existing iron-bench tests still pass
```

Expected: all PASS; existing iron-bench `tests/concurrent_smoke.rs` still compiles + passes.

- [ ] **Step 2.10: Create `tools/p5h_2b_protocol_experiment.py` driver**

```python
"""P5h+2.b T1/T2/T4 protocol experiment driver.

Per spec § 5.2 + § 5.3: given an experiment matrix row (exp_id +
P5I_C_SERVER_LIFECYCLE + P5I_C_PP_ORDER + P5I_C_LOGGING_MODE + PPs +
repeats), invoke the extended capture harness for each repeat, gather
per-cell artifacts into `/tmp/p5h+2-b-t{phase}-{exp_id}-r{R}-pp{PP}/`,
then optionally run pp_tps envelope analysis per PP and write per-experiment
envelope JSON. Diagnostic captures with fewer than 3 repeats must use
--skip-envelope because tools/p5i_c_pp_tps_envelope.py requires >=3 repeat CSVs.

Each cell directory matches the harness output schema: bench.csv + server.log
+ meta.json. For lifecycles that share one server across PPs
(same_spawn_cross_pp), each per-cell directory still gets its own bench.csv
+ meta.json (per spec § 5.2), and server.log is replicated (the harness
writes the same server log to both cell dirs, OR symlinks — implementer's
choice).

CLI:
    python tools/p5h_2b_protocol_experiment.py \\
        --phase t1 --exp-id phase0_current \\
        --server-lifecycle phase0_current \\
        --pp-order 128,512 \\
        --logging-mode default_profile \\
        --mode production \\
        --repeats 3 \\
        --pps 128,512 \\
        --runs-per-pp '128:7,512:15' \\
        --preheat-seconds 300 \\
        --preheat-runs 1100 \\
        --model-dir $SNAP \\
        --mlx-dir $HOME/.local/mlx \\
        --out-base /tmp/p5h+2-b-t1
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent


def run_one_repeat(args, repeat: int) -> dict:
    """Invoke harness for one repeat; return paths to per-PP cell dirs."""
    env = os.environ.copy()
    env.update({
        "P5I_C_REPEAT_INDEX": str(repeat),
        "P5I_C_MODE": args.mode,
        "P5I_C_MODEL_DIR": args.model_dir,
        "MLX_DIR": args.mlx_dir,
        "P5I_C_SERVER_LIFECYCLE": args.server_lifecycle,
        "P5I_C_PP_ORDER": args.pp_order,
        "P5I_C_LOGGING_MODE": args.logging_mode,
        "P5I_C_RUNS_PER_PP": args.runs_per_pp,
        "P5I_C_PREHEAT_SECONDS": str(args.preheat_seconds),
        "P5I_C_PREHEAT_RUNS": str(args.preheat_runs),
    })
    cmd = [
        "cargo", "test", "--release", "-p", "ironmlx",
        "--features", "p5h-profile",
        "--test", "p5i_c_phase_0_capture",
        "--", "--ignored", "--test-threads=1", "--nocapture",
    ]
    log_path = Path(f"/tmp/p5h+2-b-{args.phase}-{args.exp_id}-r{repeat}.log")
    with log_path.open("w") as logf:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"harness failed for repeat={repeat}; see {log_path}")
    # Harness writes to /tmp/p5i-c-phase-0-r{R}-pp{PP}-{mode}; relocate to
    # /tmp/p5h+2-b-{phase}-{exp_id}-r{R}-pp{PP}/
    cell_map = {}
    for pp in args.pps.split(","):
        src = Path(f"/tmp/p5i-c-phase-0-r{repeat}-pp{pp}-{args.mode}")
        dst = Path(f"{args.out_base}-{args.exp_id}-r{repeat}-pp{pp}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        cell_map[pp] = dst
    return cell_map


def run_envelope(args, cell_map_per_repeat: list[dict]) -> None:
    """Per PP: collect bench.csv across repeats; run envelope script."""
    if args.skip_envelope:
        print("  envelope skipped (--skip-envelope)")
        return
    if args.repeats < 3:
        raise SystemExit("--repeats must be >=3 unless --skip-envelope is set")
    for pp in args.pps.split(","):
        repeat_csvs = [str(cell_map_per_repeat[r - 1][pp] / "bench.csv")
                       for r in range(1, args.repeats + 1)]
        out_json = Path(f"{args.out_base}-{args.exp_id}-pp{pp}-envelope.json")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "p5i_c_pp_tps_envelope.py"),
            "--pp", pp,
            "--out-json", str(out_json),
        ]
        for c in repeat_csvs:
            cmd.extend(["--repeat-csv", c])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"envelope failed for PP={pp}: {result.stderr}")
        print(f"  PP={pp} envelope → {out_json}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["t1", "t2", "t4"], required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--server-lifecycle", required=True,
                   choices=["phase0_current", "same_spawn_cross_pp", "same_spawn_per_pp"])
    p.add_argument("--pp-order", required=True)
    p.add_argument("--logging-mode", required=True,
                   choices=["default_profile", "quiet_acceptance", "buffered_profile"])
    p.add_argument("--mode", choices=["probe", "production"], required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pps", required=True, help="comma-separated, e.g. 128,512")
    p.add_argument("--runs-per-pp", required=True, help="e.g. 128:7,512:15")
    p.add_argument("--preheat-seconds", type=int, default=300)
    p.add_argument("--preheat-runs", type=int, default=1100)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mlx-dir", required=True)
    p.add_argument("--out-base", required=True, help="e.g. /tmp/p5h+2-b-t1")
    p.add_argument("--skip-envelope", action="store_true",
                   help="skip envelope computation for diagnostic captures with <3 repeats")
    args = p.parse_args()

    print(f"=== {args.phase} {args.exp_id} ({args.server_lifecycle} / {args.logging_mode}) ===")
    cell_map_per_repeat: list[dict] = []
    for r in range(1, args.repeats + 1):
        print(f"  repeat {r}...")
        cell_map = run_one_repeat(args, r)
        cell_map_per_repeat.append(cell_map)
    run_envelope(args, cell_map_per_repeat)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.11: Python hygiene + pytest existing**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2b_protocol_experiment.py
uv run --with ruff ruff format --check tools/p5h_2b_protocol_experiment.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v 2>&1 | tail -5  # ensure no regression
```

Expected: ruff clean + 132+ pytests PASS (no regressions).

- [ ] **Step 2.12: Task 2 checkpoint (no commit yet)**

```bash
git diff -- ironmlx/tests/p5i_c_phase_0_capture.rs iron-bench/src/main.rs iron-bench/src/runner.rs iron-bench/src/report.rs tools/p5h_2b_protocol_experiment.py
git status --short
```

Expected: Task 2 code is present and all gates in Steps 2.9/2.11 passed, but do not commit here. Per design § 5.6/§ 5.7, validated reusable infra is committed at T5 PASS or T5F FAIL/DEFERRED.

---

## Task 3: T1 — Protocol state matrix sweep (~5 hr GPU; serial)

**Files:**
- Outputs: `/tmp/p5h+2-b-t1-{exp_id}-r{1..3}-pp{128,512}/{bench.csv,server.log,meta.json}` + `/tmp/p5h+2-b-t1-{exp_id}-pp{128,512}-envelope.json` per experiment
- Append: `reports/p5h+2-b-bench-log.md`

- [ ] **Step 3.1: Prepare env file (reuse Phase 0 SNAP)**

```bash
set -o pipefail
source /tmp/p5i-c-env.sh
echo "SNAP=$SNAP"
echo "MLX_DIR=$MLX_DIR"
test -n "$SNAP" && test -f "$SNAP/tokenizer.json"
```

- [ ] **Step 3.2: Run T1 experiment matrix (4 experiments × 3 repeats production-only)**

Per spec § 5.2 Table:

```bash
source /tmp/p5i-c-env.sh
P5H_2B_POWERMETRICS=0
if sudo -n powermetrics --samplers smc -i 1000 --format json -n 1 >/tmp/p5h+2-b-powermetrics-precheck.json 2>/tmp/p5h+2-b-powermetrics-precheck.err; then
  P5H_2B_POWERMETRICS=1
  echo "powermetrics sidecar enabled for T1/T2"
else
  echo "powermetrics unavailable; T3 will be marked observational/unavailable"
fi
EXPS=(
  "phase0_current|phase0_current|128,512|default_profile"
  "same_spawn_cross_pp|same_spawn_cross_pp|128,512|default_profile"
  "order_swap_same_spawn|same_spawn_cross_pp|512,128|default_profile"
  "same_spawn_per_pp|same_spawn_per_pp|128,512|default_profile"
)
for ENTRY in "${EXPS[@]}"; do
  IFS='|' read -r EXP_ID LIFECYCLE PP_ORDER LOG_MODE <<< "$ENTRY"
  echo "=== T1 $EXP_ID start $(date +%H:%M:%S) ==="
  PM_PID=
  if [ "$P5H_2B_POWERMETRICS" = "1" ]; then
    sudo powermetrics --samplers smc,gpu_power,thermal -i 1000 --format json > "/tmp/p5h+2-b-t3-thermal-t1-${EXP_ID}.json" 2>&1 &
    PM_PID=$!
  fi
  uv run python tools/p5h_2b_protocol_experiment.py \
    --phase t1 --exp-id "$EXP_ID" \
    --server-lifecycle "$LIFECYCLE" \
    --pp-order "$PP_ORDER" \
    --logging-mode "$LOG_MODE" \
    --mode production \
    --repeats 3 \
    --pps 128,512 \
    --runs-per-pp '128:7,512:15' \
    --preheat-seconds 300 \
    --preheat-runs 1100 \
    --model-dir "$SNAP" \
    --mlx-dir "$MLX_DIR" \
    --out-base /tmp/p5h+2-b-t1 2>&1 | tee /tmp/p5h+2-b-t1-${EXP_ID}.log | tail -10
  if [ -n "$PM_PID" ]; then
    sudo kill "$PM_PID" 2>/dev/null || true
    wait "$PM_PID" 2>/dev/null || true
  fi
  echo "=== T1 $EXP_ID end $(date +%H:%M:%S) ==="
done
echo "ALL T1 DONE $(date +%H:%M:%S)"
```

Wall: ~5 hr (each repeat ~10 min × 3 × 4 exps).

- [ ] **Step 3.3: Verify all 4 experiments produced envelope JSONs**

```bash
for EXP in phase0_current same_spawn_cross_pp order_swap_same_spawn same_spawn_per_pp; do
  for PP in 128 512; do
    JSON=/tmp/p5h+2-b-t1-${EXP}-pp${PP}-envelope.json
    test -f "$JSON" || echo "MISSING: $JSON"
    [ -f "$JSON" ] && python3 -c "
import json
d = json.load(open('$JSON'))
env = d['ironmlx']['final_uncertainty_envelope_pct']
within = d['ironmlx']['within_sweep_ci95_max_pct']
between = d['ironmlx']['between_sweep_half_range_pct']
print(f'$EXP PP=$PP: envelope={env:.2f}% (within={within:.2f}, between={between:.2f}) → {d[\"ironmlx\"][\"verdict\"]}')"
  done
done
```

- [ ] **Step 3.4: Append T1 verdict + ranking to bench log**

```bash
mkdir -p reports
cat >> reports/p5h+2-b-bench-log.md << 'BENCHLOG_T1'

# P5h+2.b T1 — Protocol state matrix sweep

(per spec § 5.2; matrix: phase0_current vs same_spawn_cross_pp vs order_swap_same_spawn vs same_spawn_per_pp)

(Per-experiment envelope verdict copied from /tmp/p5h+2-b-t1-{EXP}-pp{PP}-envelope.json)

| exp_id | PP=128 envelope | PP=512 envelope | PP=128 verdict | PP=512 verdict |
|---|---|---|---|---|
(Implementer fills from Step 3.3 output)

## T1 analysis

Which experiment(s) eliminated or shifted PP=512 bimodal pattern? (e.g. did
same_spawn_cross_pp pass envelope while phase0_current still fails? Did
order swap matter?)

(Implementer writes 1-2 paragraph analysis comparing experiments)
BENCHLOG_T1
```

T1 has no commit. Data is gitignored and Task 2 code remains uncommitted until T5 PASS or T5F FAIL/DEFERRED.

---

## Task 4: T2 — Logging-perturbation control experiment (~2-3 hr GPU; serial)

**Files:**
- Outputs: `/tmp/p5h+2-b-t2-{exp_id}-r{1..3}-pp128/` + envelope JSON per experiment
- Append: `reports/p5h+2-b-bench-log.md`

- [ ] **Step 4.1: Determine T2 base lifecycle from T1 verdict**

If T1 verdict at PP=128 favors `same_spawn_cross_pp` or `same_spawn_per_pp` (i.e. that experiment showed cleaner envelope at PP=128), use that lifecycle for T2's logging perturbation experiments. If `phase0_current` is still the best PP=128 baseline, use `phase0_current`.

Record this decision in `reports/p5h+2-b-bench-log.md` before running T2.

- [ ] **Step 4.2: Run T2 logging experiments (3 experiments × 3 repeats × PP=128 only)**

Substitute `<T2_LIFECYCLE>` with the lifecycle decided in 4.1:

```bash
set -o pipefail
source /tmp/p5i-c-env.sh
T2_LIFECYCLE=<set per 4.1>  # e.g. "phase0_current" or "same_spawn_per_pp"
P5H_2B_POWERMETRICS=${P5H_2B_POWERMETRICS:-0}
if [ "$P5H_2B_POWERMETRICS" = "0" ] && sudo -n powermetrics --samplers smc -i 1000 --format json -n 1 >/tmp/p5h+2-b-powermetrics-precheck.json 2>/tmp/p5h+2-b-powermetrics-precheck.err; then
  P5H_2B_POWERMETRICS=1
  echo "powermetrics sidecar enabled for T2"
fi
EXPS=(
  "log_default|default_profile"
  "log_quiet|quiet_acceptance"
  "log_buffered|buffered_profile"
)
for ENTRY in "${EXPS[@]}"; do
  IFS='|' read -r EXP_ID LOG_MODE <<< "$ENTRY"
  echo "=== T2 $EXP_ID start $(date +%H:%M:%S) ==="
  PM_PID=
  if [ "$P5H_2B_POWERMETRICS" = "1" ]; then
    sudo powermetrics --samplers smc,gpu_power,thermal -i 1000 --format json > "/tmp/p5h+2-b-t3-thermal-t2-${EXP_ID}.json" 2>&1 &
    PM_PID=$!
  fi
  uv run python tools/p5h_2b_protocol_experiment.py \
    --phase t2 --exp-id "$EXP_ID" \
    --server-lifecycle "$T2_LIFECYCLE" \
    --pp-order 128 \
    --logging-mode "$LOG_MODE" \
    --mode production \
    --repeats 3 \
    --pps 128 \
    --runs-per-pp '128:7' \
    --preheat-seconds 300 \
    --preheat-runs 1100 \
    --model-dir "$SNAP" \
    --mlx-dir "$MLX_DIR" \
    --out-base /tmp/p5h+2-b-t2 2>&1 | tee /tmp/p5h+2-b-t2-${EXP_ID}.log | tail -10
  if [ -n "$PM_PID" ]; then
    sudo kill "$PM_PID" 2>/dev/null || true
    wait "$PM_PID" 2>/dev/null || true
  fi
  echo "=== T2 $EXP_ID end $(date +%H:%M:%S) ==="
done
echo "ALL T2 DONE $(date +%H:%M:%S)"
```

Wall: ~2-3 hr.

- [ ] **Step 4.3: Verify T2 outputs + record verdict**

```bash
for EXP in log_default log_quiet log_buffered; do
  JSON=/tmp/p5h+2-b-t2-${EXP}-pp128-envelope.json
  test -f "$JSON" || echo "MISSING: $JSON"
  [ -f "$JSON" ] && python3 -c "
import json
d = json.load(open('$JSON'))
env = d['ironmlx']['final_uncertainty_envelope_pct']
within = d['ironmlx']['within_sweep_ci95_max_pct']
between = d['ironmlx']['between_sweep_half_range_pct']
print(f'$EXP PP=128: envelope={env:.2f}% (within={within:.2f}, between={between:.2f}) → {d[\"ironmlx\"][\"verdict\"]}')"
done
```

- [ ] **Step 4.4: Append T2 verdict to bench log**

```bash
cat >> reports/p5h+2-b-bench-log.md << 'BENCHLOG_T2'

# P5h+2.b T2 — Logging perturbation control (PP=128)

Base lifecycle: <T2_LIFECYCLE per 4.1>

| exp_id | logging mode | PP=128 envelope | verdict |
|---|---|---|---|
(Implementer fills from Step 4.3 output)

## T2 analysis

Did any logging variant eliminate PP=128 trailing-outlier pattern?
If yes — which one + by how much?
If quiet_acceptance is the only PASS, note per spec § 5.3 caveat: mechanism
decomposition requires paired default_profile or buffered_profile diagnostic sweep.
BENCHLOG_T2
```

No commit at T2.

---

## Task 5: T3 — Thermal monitoring overlay (~2 hr; piggyback + analysis)

**Files:**
- Create: `tools/p5h_2b_thermal_overlay.py`
- Create: `tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py`
- Outputs: `/tmp/p5h+2-b-t3-thermal-t{1,2}-{exp}.json` (powermetrics raw sidecar from T1/T2) + `/tmp/p5h+2-b-t3-overlay-t{1,2}-{exp}-r{R}-pp{PP}.json` (joined)
- Append: `reports/p5h+2-b-bench-log.md`

- [ ] **Step 5.1: Verify powermetrics sidecar artifacts from T1/T2**

```bash
ls -la /tmp/p5h+2-b-t3-thermal-t1-*.json /tmp/p5h+2-b-t3-thermal-t2-*.json 2>/dev/null || true
```

If no thermal JSON exists, do not rerun GPU experiments. Continue with Steps 5.2-5.4 so the reusable parser is still validated, then document T3 as `unavailable` in Step 5.7. Per design § 5.4, T3 piggybacks on T1/T2 and must not add standalone GPU work.

- [ ] **Step 5.2: Create `tools/p5h_2b_thermal_overlay.py`**

```python
"""P5h+2.b T3 — powermetrics thermal overlay joiner.

Per spec § 5.4: parses powermetrics JSON output (--samplers smc,gpu_power,thermal
--format json) and joins to iron-bench per-run timestamps (run_start_unix_ns /
run_end_unix_ns columns from --capture-run-timestamps). Outputs per-run
thermal alignment + outlier-correlation summary.

CLI:
    python tools/p5h_2b_thermal_overlay.py \\
        --powermetrics-json /tmp/p5h+2-b-t3-thermal-t1-{exp}.json \\
        --cell-dir /tmp/p5h+2-b-t1-{exp}-r1-pp128 \\
        --out-json /tmp/p5h+2-b-t3-overlay-t1-{exp}-r1-pp128.json
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import median

OUTLIER_THRESHOLD_PCT = 10.0  # match T0 threshold


def parse_powermetrics_samples(json_path: Path) -> list[dict]:
    """Parse powermetrics JSON output. Each sample has timestamp_ms and thermal/gpu/fan fields.
    Powermetrics emits one JSON object per sample with timestamp in ms-since-epoch
    (or a separate `timestamp` field — implementer verifies actual schema at runtime
    by checking the first few samples)."""
    samples = []
    text = json_path.read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if samples:
        return samples
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            samples.append(obj)
        elif isinstance(obj, list):
            samples.extend(x for x in obj if isinstance(x, dict))
        idx = end
    return samples


def join_overlay(powermetrics_samples: list[dict], cell_dir: Path) -> dict:
    bench_path = cell_dir / "bench.csv"
    meta = json.loads((cell_dir / "meta.json").read_text())
    with bench_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows or "run_start_unix_ns" not in rows[0]:
        raise SystemExit(
            f"{cell_dir}: bench.csv missing run_start_unix_ns column "
            "(requires --capture-run-timestamps from iron-bench)"
        )
    # Find cell median for outlier threshold
    pp_tps_list = [float(r["pp_tps"]) for r in rows]
    cell_median = median(pp_tps_list)
    overlay = []
    for row in rows:
        start_ns = int(row["run_start_unix_ns"])
        end_ns = int(row["run_end_unix_ns"])
        # Find samples in [start_ns, end_ns]
        start_ms = start_ns // 1_000_000
        end_ms = end_ns // 1_000_000
        # powermetrics timestamp field name is implementer-verified;
        # typical: 'timestamp' (Unix ms) or 'sample_time_ms'.
        # Fallback: iterate samples and infer field name from first sample.
        ts_field = _infer_timestamp_field(powermetrics_samples)
        in_window = [s for s in powermetrics_samples
                     if start_ms <= int(s.get(ts_field, 0)) <= end_ms]
        pp_tps = float(row["pp_tps"])
        deviation_pct = abs(pp_tps - cell_median) / cell_median * 100 if cell_median > 0 else 0
        is_outlier = deviation_pct > OUTLIER_THRESHOLD_PCT
        thermal_summary = _summarize_thermal(in_window) if in_window else None
        overlay.append({
            "run_idx": int(row["run_idx"]),
            "pp_tps": pp_tps,
            "is_outlier": is_outlier,
            "thermal_samples_in_window": len(in_window),
            "thermal_summary": thermal_summary,
        })
    # Correlation: do outlier runs coincide with thermal spikes?
    outlier_thermal_max = [o["thermal_summary"]["max_gpu_die_c"]
                           for o in overlay if o["is_outlier"]
                           and o["thermal_summary"] is not None
                           and o["thermal_summary"]["max_gpu_die_c"] is not None]
    nonoutlier_thermal_max = [o["thermal_summary"]["max_gpu_die_c"]
                              for o in overlay if not o["is_outlier"]
                              and o["thermal_summary"] is not None
                              and o["thermal_summary"]["max_gpu_die_c"] is not None]
    correlation = "unknown"
    if outlier_thermal_max and nonoutlier_thermal_max:
        avg_out = sum(outlier_thermal_max) / len(outlier_thermal_max)
        avg_norm = sum(nonoutlier_thermal_max) / len(nonoutlier_thermal_max)
        if avg_out > avg_norm * 1.05:
            correlation = "outliers_run_hot"
        elif avg_out < avg_norm * 0.95:
            correlation = "outliers_run_cool"
        else:
            correlation = "no_thermal_correlation"
    return {
        "cell": str(cell_dir),
        "server_lifecycle": meta.get("server_lifecycle"),
        "logging_mode": meta.get("logging_mode"),
        "n_overlay_runs": len(overlay),
        "correlation": correlation,
        "overlay": overlay,
    }


def _infer_timestamp_field(samples: list[dict]) -> str:
    if not samples:
        return "timestamp"
    candidates = ["timestamp", "sample_time_ms", "timestamp_ms", "time_ms"]
    for c in candidates:
        if c in samples[0]:
            return c
    raise SystemExit(f"cannot infer powermetrics timestamp field; first sample keys: {list(samples[0].keys())}")


def _summarize_thermal(samples: list[dict]) -> dict:
    # Powermetrics schema: thermal data inside e.g. samples[i]['thermal_pressure'] or
    # samples[i]['gpu']['die_temperature_c']. Implementer verifies at runtime.
    # Conservative fallback: extract any numeric field with 'temp' or 'die' in name.
    gpu_die_temps = []
    for s in samples:
        # Walk dict for any *temp* or *die* numeric value
        def _walk(d, out):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)) and any(
                        kw in k.lower() for kw in ("temp", "die")
                    ):
                        out.append(v)
                    elif isinstance(v, (dict, list)):
                        _walk(v, out)
            elif isinstance(d, list):
                for item in d:
                    _walk(item, out)
        _walk(s, gpu_die_temps)
    if not gpu_die_temps:
        return {"max_gpu_die_c": None, "n_temp_samples": 0}
    return {
        "max_gpu_die_c": max(gpu_die_temps),
        "min_gpu_die_c": min(gpu_die_temps),
        "n_temp_samples": len(gpu_die_temps),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--powermetrics-json", type=Path, required=True)
    p.add_argument("--cell-dir", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()
    samples = parse_powermetrics_samples(args.powermetrics_json)
    if not samples:
        result = {
            "cell": str(args.cell_dir),
            "correlation": "unavailable",
            "note": "powermetrics JSON parsed 0 samples",
        }
    else:
        result = join_overlay(samples, args.cell_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out_json}")
    print(f"  correlation: {result.get('correlation')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.3: Create pytest for thermal overlay**

```python
"""Pytest for tools/p5h_2b_thermal_overlay.py."""
from __future__ import annotations
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5h_2b_thermal_overlay import (  # noqa: E402
    _infer_timestamp_field,
    join_overlay,
    parse_powermetrics_samples,
)


def test_parse_powermetrics_jsonl(tmp_path):
    p = tmp_path / "thermal.json"
    p.write_text('{"timestamp": 1000, "gpu_die_temp_c": 60}\n'
                 '{"timestamp": 2000, "gpu_die_temp_c": 65}\n')
    samples = parse_powermetrics_samples(p)
    assert len(samples) == 2
    assert _infer_timestamp_field(samples) == "timestamp"


def test_join_outlier_runs_hot(tmp_path):
    # Create cell with 3 runs; run 2 is outlier (slow) and runs hot.
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "bench.csv").write_text(
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,run_start_unix_ns,run_end_unix_ns\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,1000000000,1100000000\n"
        "x,128,1,1,100.0,1.0,1.0,500.0,1.0,128,140,1,0,length,1200000000,1500000000\n"
        "x,128,1,2,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,1600000000,1700000000\n"
    )
    (cell / "meta.json").write_text(json.dumps({"mode": "production", "pp": 128}))
    samples = [
        {"timestamp": 1000, "gpu_die_temp_c": 60},
        {"timestamp": 1050, "gpu_die_temp_c": 61},
        {"timestamp": 1100, "gpu_die_temp_c": 62},
        {"timestamp": 1250, "gpu_die_temp_c": 80},  # in outlier window
        {"timestamp": 1400, "gpu_die_temp_c": 82},  # in outlier window
        {"timestamp": 1650, "gpu_die_temp_c": 63},
    ]
    result = join_overlay(samples, cell)
    assert result["correlation"] == "outliers_run_hot"
    outlier_runs = [o for o in result["overlay"] if o["is_outlier"]]
    assert len(outlier_runs) == 1
    assert outlier_runs[0]["run_idx"] == 1
```

- [ ] **Step 5.4: Hygiene + pytest**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2b_thermal_overlay.py tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py
uv run --with ruff ruff format --check tools/p5h_2b_thermal_overlay.py tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py -v
```

Expected: all PASS.

- [ ] **Step 5.5: Select existing thermal sidecar inputs from T1/T2**

Pick up to two existing T1/T2 powermetrics JSON files captured by the sidecar in Steps 3.2 and 4.2. Do not start any new harness run here.

```bash
ls /tmp/p5h+2-b-t3-thermal-t1-*.json /tmp/p5h+2-b-t3-thermal-t2-*.json 2>/dev/null | head -2
```

Record the selected thermal JSON paths in `reports/p5h+2-b-bench-log.md`. If the list is empty, T3 verdict is `unavailable`.

- [ ] **Step 5.6: Run thermal overlay analysis on each thermal-instrumented cell**

```bash
# Example: analyze one selected T1 experiment and one selected T2 experiment.
analyze_t1_exp() {
  local thermal_json=$1
  local exp_id=$2
  test -f "$thermal_json" || { echo "T3 unavailable: missing $thermal_json"; return 0; }
  for R in 1 2 3; do
    for PP in 128 512; do
      CELL=/tmp/p5h+2-b-t1-${exp_id}-r${R}-pp${PP}
      test -d "$CELL" || { echo "MISSING $CELL"; continue; }
      uv run python tools/p5h_2b_thermal_overlay.py \
        --powermetrics-json "$thermal_json" \
        --cell-dir "$CELL" \
        --out-json /tmp/p5h+2-b-t3-overlay-t1-${exp_id}-r${R}-pp${PP}.json
    done
  done
}

analyze_t2_exp() {
  local thermal_json=$1
  local exp_id=$2
  test -f "$thermal_json" || { echo "T3 unavailable: missing $thermal_json"; return 0; }
  for R in 1 2 3; do
    CELL=/tmp/p5h+2-b-t2-${exp_id}-r${R}-pp128
    test -d "$CELL" || { echo "MISSING $CELL"; continue; }
    uv run python tools/p5h_2b_thermal_overlay.py \
      --powermetrics-json "$thermal_json" \
      --cell-dir "$CELL" \
      --out-json /tmp/p5h+2-b-t3-overlay-t2-${exp_id}-r${R}-pp128.json
  done
}

analyze_t1_exp /tmp/p5h+2-b-t3-thermal-t1-phase0_current.json phase0_current
analyze_t2_exp /tmp/p5h+2-b-t3-thermal-t2-log_default.json log_default
```

- [ ] **Step 5.7: Append T3 verdict to bench log**

```bash
cat >> reports/p5h+2-b-bench-log.md << 'BENCHLOG_T3'

# P5h+2.b T3 — Thermal monitoring overlay (observational)

(per spec § 5.4 + § 3.3; informational only)

| exp_id | PP | correlation | n_overlay_runs |
|---|---|---|---|
(Implementer fills from /tmp/p5h+2-b-t3-overlay-* JSONs)

## T3 analysis

Did outlier runs coincide with thermal spikes? Was thermal a probable
contributing factor or independent of outlier pattern? (Per spec § 3.3
thermal evidence is observational; cannot be acceptance dependency.)
BENCHLOG_T3
```

- [ ] **Step 5.8: T3 checkpoint (no commit yet)**

```bash
git diff -- tools/p5h_2b_thermal_overlay.py tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py
git status --short
```

Expected: T3 parser + pytest files are present and validated, but do not commit here. They are committed at T5 PASS or T5F FAIL/DEFERRED with the rest of the validated reusable infra.

---

## Task 6: T4 — Final protocol draft + acceptance sweep (~3-4 hr GPU; possible T4R)

**Files:**
- Create: `docs/p5h+2-b-protocol.md` (draft; not committed in T4 unless PASS)
- Create: `reports/p5h+2-b-t4-predeclared-exclusions.md` (gitignored draft)
- Outputs: `/tmp/p5h+2-b-t4-acceptance-r{1..3}-pp{128,512}/` + envelope JSONs
- Append: `reports/p5h+2-b-bench-log.md`

- [ ] **Step 6.1: T4.1 Predeclared exclusion rules (BEFORE any acceptance sweep)**

Per spec § 9: write any outlier-exclusion rules BEFORE T4.3 sweep data is observed. Save to a draft file:

```bash
mkdir -p reports
cat > reports/p5h+2-b-t4-predeclared-exclusions.md << 'PREDECLARED'
# P5h+2.b T4.1 — Predeclared outlier exclusion rules

**Locked at**: <date + time when writing>
**Source evidence**: T0 verdicts + T1 + T2 + T3 summaries above

(Implementer writes 0+ explicit rules per spec § 9 format:
- threshold
- scope
- justification

Examples:
- "Exclude PP=128 runs where pp_tps < 80% of remaining-runs median; justified
  by T0 finding that trailing outliers correlate with server-side state."
- "Exclude first sweep in each spawn (warmup-equivalent) under
  same_spawn_cross_pp lifecycle; justified by T1 finding that first-after-
  preheat sweep shows cold-cache pattern."
)

If T0-T3 evidence does NOT motivate any exclusion rule, write:
"No exclusion rules predeclared; final protocol relies on lifecycle/logging
choice alone to eliminate outlier pattern."
PREDECLARED

# Open in editor / write rules now; do NOT proceed to 6.2 before this is filled in
$EDITOR reports/p5h+2-b-t4-predeclared-exclusions.md 2>/dev/null || true
echo "Predeclared exclusion rules locked. Proceeding to T4.2."
```

- [ ] **Step 6.2: T4.2 Final protocol draft**

Synthesize T1+T2 verdicts (lifecycle winner + logging-mode winner) + T3 observation. Write `docs/p5h+2-b-protocol.md` skeleton (NOT committed yet):

```bash
cat > docs/p5h+2-b-protocol.md << 'PROTOCOL_DRAFT'
# P5h+2.b — PP=128/512 Production Envelope Protocol (draft)

**Status:** Draft for T4.3 acceptance sweep validation. NOT yet validated.
**Date:** <date>
**Spec ref:** docs/superpowers/specs/2026-05-24-ironmlx-p5h+2-b-pp128-512-envelope-protocol-fix-design.md (commit aabf21f)

## Selected configuration (per T1+T2 evidence)

- **Server lifecycle**: <winner from T1>
- **PP order** (if applicable): <winner from T1>
- **Logging mode**: <winner from T2 if PP=128 envelope improved>
- **Preheat**: monolithic `iron-bench --runs 1100 --warmup 0` ≈ 395s wall on M5 Max
- **RUNS per PP**: PP=128 → 7 (or adjusted per predeclared exclusion); PP=512 → 15
- **Repeats**: ≥3 fresh server-lifecycle repeats per PP

## Per-PP mechanism statement (spec § 3.2 hard binding)

- **PP=128 trailing outliers**: [explained: <root cause from T2/T3>] OR
  [eliminated by protocol: <which lifecycle/logging choice removes the pattern>] OR
  [both]
- **PP=512 bimodal medians**: [explained: <root cause from T1>] OR
  [eliminated by protocol: <which lifecycle choice removes the pattern>] OR
  [both]

## Predeclared exclusion rules (frozen at T4.1)

(Copied verbatim from reports/p5h+2-b-t4-predeclared-exclusions.md;
this section locks in T4.1 wording into the committed doc)

## Acceptance sweep results (filled at T4.3)

(T4.3 sweep numbers will be added after acceptance run)
PROTOCOL_DRAFT
```

- [ ] **Step 6.3: T4.3 Acceptance sweep — ≥3 fresh-spawn repeats × PP=128 + PP=512**

Per the selected lifecycle from T4.2:

```bash
set -o pipefail
source /tmp/p5i-c-env.sh
WINNER_LIFECYCLE=<from T4.2 e.g. same_spawn_cross_pp>
WINNER_PP_ORDER=<from T4.2 e.g. 128,512>
WINNER_LOG_MODE=<from T4.2 e.g. default_profile>

uv run python tools/p5h_2b_protocol_experiment.py \
  --phase t4 --exp-id acceptance \
  --server-lifecycle "$WINNER_LIFECYCLE" \
  --pp-order "$WINNER_PP_ORDER" \
  --logging-mode "$WINNER_LOG_MODE" \
  --mode production \
  --repeats 3 \
  --pps 128,512 \
  --runs-per-pp '128:7,512:15' \
  --preheat-seconds 300 \
  --preheat-runs 1100 \
  --model-dir "$SNAP" \
  --mlx-dir "$MLX_DIR" \
  --out-base /tmp/p5h+2-b-t4 2>&1 | tee /tmp/p5h+2-b-t4-acceptance.log | tail -10
```

- [ ] **Step 6.4: T4.4 Envelope verification + apply predeclared exclusions**

```bash
python3 - <<'PY'
import json
from pathlib import Path
for pp in (128, 512):
    d = json.load(open(f'/tmp/p5h+2-b-t4-acceptance-pp{pp}-envelope.json'))['ironmlx']
    env = d['final_uncertainty_envelope_pct']
    within = d['within_sweep_ci95_max_pct']
    between = d['between_sweep_half_range_pct']
    print(f'PP={pp}: envelope={env:.3f}% (within={within:.3f}, between={between:.3f}) → {d["verdict"]}')
    print(f'  per-repeat medians: {[round(m,2) for m in d["medians"]]}')
PY
```

If predeclared exclusions apply, re-compute envelope with excluded rows. Document the exclusion application clearly in `docs/p5h+2-b-protocol.md` § Acceptance sweep results.

- [ ] **Step 6.5: T4.5 / T4.6 / T4.7 decision tree**

- **T4.5 PASS**: both PPs envelope ≤ ±2% under final protocol → continue to Task 7 T5.
- **T4.6 FAIL within budget**: one bounded retry `T4R` with a SINGLE predeclared protocol adjustment (e.g. tighten preheat seconds, switch lifecycle to the runner-up from T1). Re-run Step 6.3+6.4 once. Document T4R adjustment in protocol doc.
- **T4.7 FAIL beyond cap OR after T4R**: skip to Task 7 T5F (FAIL/DEFERRED close-out) per spec § 5.7. Do NOT silently extend GPU work.

Wall budget check:

```bash
# Compute cumulative T1+T2+T3+T4 wall from .log files
ls -la /tmp/p5h+2-b-{t1,t2,t3,t4}-*.log 2>/dev/null
# Implementer estimates cumulative; if approaching 15hr cap, lean toward T4.7
```

- [ ] **Step 6.6: Append T4 results to bench log**

```bash
cat >> reports/p5h+2-b-bench-log.md << 'BENCHLOG_T4'

# P5h+2.b T4 — Final protocol acceptance sweep

**Final protocol selected**: <lifecycle + PP order + logging mode>
**Predeclared exclusions**: <inline summary; full text in protocol doc>

| PP | within CI95 max | between half-range | final envelope | verdict |
|---|---|---|---|---|
| 128 | <fill> | <fill> | <fill> | <PASS/FAIL> |
| 512 | <fill> | <fill> | <fill> | <PASS/FAIL> |

(If T4R ran, document the adjustment + second sweep numbers here.)

## T4 outcome

(PASS → proceed to T5; FAIL → T5F)
BENCHLOG_T4
```

T4 has no commit (protocol doc + close-out doc commit at T5/T5F).

---

## Task 7: T5 (PASS) OR T5F (FAIL/DEFERRED) close-out + Phase 0 backfill + commit (~1-1.5 hr)

Branch on T4 outcome.

### § 7.A T5 PASS path

- [ ] **Step 7.A.1: Finalize `docs/p5h+2-b-protocol.md`**

Replace draft section markers with actual T1+T2 verdicts + final lifecycle + logging mode + mechanism statement per spec § 3.2 + T4.3 acceptance sweep numbers. The doc becomes committable.

- [ ] **Step 7.A.2: Write `docs/p5h+2-b-close-out.md`**

```markdown
# P5h+2.b — PP=128/512 Production Envelope Protocol Fix: Close-out

**Status:** PASS — final protocol achieves envelope ≤ ±2% per PP on ≥3 fresh-spawn repeats.
**Date:** <date>
**Branch:** ironmlx-p5h+2-a-pp512-measurement
**Commit chain on this branch (P5h+2.b):**
- aabf21f spec
- <plan commit>
- this commit (validated reusable infra + T5 close-out + final protocol)

**Sources:**
- Spec: docs/superpowers/specs/2026-05-24-ironmlx-p5h+2-b-pp128-512-envelope-protocol-fix-design.md (commit aabf21f)
- Plan: docs/superpowers/plans/2026-05-24-ironmlx-p5h+2-b-pp128-512-envelope-protocol-fix.md (commit <plan_sha>)
- Final protocol: docs/p5h+2-b-protocol.md (this commit)
- Bench log + T0 + acceptance review (gitignored): reports/p5h+2-b-{bench-log,t0-outlier-source,t4-predeclared-exclusions}.md
- Predecessor: docs/p5i-c-phase-0-close-out.md (γ-lite; now backfilled to PASS)

## § 1 Acceptance per spec § 7.1 — ALL PASS

| # | Criterion | Verdict |
|---|---|---|
| 1 | Envelope ≤ ±2% per PP on ≥3 fresh-spawn repeats | ✓ PASS (PP=128 <X>%; PP=512 <Y>%) |
| 2 | Per-PP mechanism statement | ✓ See docs/p5h+2-b-protocol.md § Mechanism statement |
| 3 | Predeclared exclusion rules locked at T4.1 | ✓ See `reports/p5h+2-b-t4-predeclared-exclusions.md` + protocol § Predeclared exclusion |
| 4 | Independent investigation tracks per PP | ✓ T0/T1/T2/T3 docs separate PP=128 + PP=512 evidence |
| 5 | Phase 0 backfill complete | ✓ docs/p5i-c-phase-0-{close-out,ranking-snapshot}.md updated |
| 6 | Reusable infra emitted | ✓ tools/p5h_2b_{t0_outlier_source,protocol_experiment,thermal_overlay}.py committed |
| 7 | No production-path regression | ✓ smoke test `p5_qwen35_moe_smoke` pp_tps within ±2% |
| 8 | Rust/Python gates | ✓ cargo fmt/clippy/build + ruff + pytest all clean |

## § 2 Final protocol

(Summary; full detail in docs/p5h+2-b-protocol.md)

- Server lifecycle: <winner>
- Logging mode: <winner>
- RUNS per PP: PP=128 → 7; PP=512 → 15
- Predeclared exclusions: <inline summary>

## § 3 Mechanism statement (per spec § 3.2 hard binding)

- **PP=128 trailing outliers**: <explained / eliminated / both> — details per protocol doc § Mechanism statement
- **PP=512 bimodal medians**: <explained / eliminated / both> — details per protocol doc

## § 4 Phase 0 backfill summary

- docs/p5i-c-phase-0-close-out.md § 1 #4: FAIL/DEFERRED → ✓ PASS (envelope numbers updated)
- docs/p5i-c-phase-0-ranking-snapshot.md preamble + envelope table: updated with post-P5h+2.b numbers

## § 5 Phase 1 unblock

Phase 1 implementation acceptance is now unblocked. Phase 1 brainstorm (already
allowed parallel per Codex round-2 γ-lite Q9) may now proceed to spec/plan/implementation.

## § 6 References

(per spec § 13 + this commit's chain)
```

- [ ] **Step 7.A.3: Backfill `docs/p5i-c-phase-0-close-out.md`**

Edit § 1 row 4 from `✗ FAIL/DEFERRED` to `✓ PASS` with new envelope numbers; append note: "Resolved by P5h+2.b commit `<sha>`; see docs/p5h+2-b-close-out.md". Update § 3 evidence table with post-P5h+2.b row; § 4 vs-omlx delta if comparator re-run (only if comparator data is rerun or recomputed with preserved caveat); § 6 P5h+2.b hard bindings as `RESOLVED` with cross-link.

- [ ] **Step 7.A.4: Backfill `docs/p5i-c-phase-0-ranking-snapshot.md`**

Replace preamble status line from "production envelope FAIL/DEFERRED" → "production envelope PASS post-P5h+2.b commit `<sha>`". Update vs-omlx delta table with new ironmlx envelope + tighter delta CI (if comparator re-run).

- [ ] **Step 7.A.5: Memory file + MEMORY.md update**

```bash
cat > /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2b_findings.md << 'MEM'
---
name: project-p5h-2b-findings
description: P5h+2.b PP=128/512 production envelope protocol fix — closed PASS <date>; final protocol <lifecycle/logging>; PP=128 trailing outliers + PP=512 bimodal medians <explained|eliminated>; Phase 0 § 7 #4 backfilled to PASS
metadata:
  type: project
---

P5h+2.b closed <date> as PASS.

**Final protocol** (docs/p5h+2-b-protocol.md):
- Server lifecycle: <winner from T1>
- Logging mode: <winner from T2>
- RUNS per PP: PP=128 → 7; PP=512 → 15
- Predeclared exclusions: <inline summary>

**Acceptance envelope**:
- PP=128: <X>% (P5h+2.a target ≤ ±2% — PASS)
- PP=512: <Y>% (PASS)

**Mechanism per PP**:
- PP=128 trailing outliers: <explained: cause / OR eliminated by: protocol change>
- PP=512 bimodal medians: <explained: cause / OR eliminated by: protocol change>

**Reusable infra**:
- tools/p5h_2b_t0_outlier_source.py — offline bench+server-log decomposer
- tools/p5h_2b_protocol_experiment.py — protocol experiment driver
- tools/p5h_2b_thermal_overlay.py — observational thermal correlation
- ironmlx/tests/p5i_c_phase_0_capture.rs — extended with 3 server lifecycles + logging modes + lifecycle Unix timestamps
- iron-bench --capture-run-timestamps CLI flag — composable with --capture-server-request-id

**Phase 0 backfill complete**: docs/p5i-c-phase-0-{close-out,ranking-snapshot}.md updated to PASS.

**Phase 1 unblocked**: Phase 1 implementation acceptance can now proceed against gather_qmm_gate_up (R1 candidate from Phase 0).

Links: [[project-p5i-c-phase-0-findings]] (Phase 0 measure-only close); [[project-p5h-2a-findings]] (P5h+2.a envelope methodology); [[project-p5h-findings]] (P5h+1 ranking baseline).
MEM
```

Update MEMORY.md:

```bash
# Find P5i.c entry; insert after it
grep -n "project_p5i_c_phase_0_findings" /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md
# Manually edit MEMORY.md to add:
# - [P5h+2.b PP=128/512 envelope fix](project_p5h_2b_findings.md) — final protocol <lifecycle/logging> envelope <X>%/<Y>%; Phase 0 § 7 #4 backfilled PASS; PP=128 outliers <explained|eliminated>; PP=512 bimodal <explained|eliminated>
```

- [ ] **Step 7.A.6: Final hygiene check + production-parity smoke (spec § 7.1 #7 + #8)**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -3
cargo build --release
uv run --with ruff ruff check tools/p5h_2b_*.py tools/p5h_aggregator/tests/test_p5h_2b_*.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v 2>&1 | tail -5
# Production parity smoke (P5h+1 binding)
IRONMLX_MOE_MODEL_DIR=$SNAP cargo test --release -p ironmlx --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --nocapture 2>&1 | tail -10
```

Expected: all gates PASS; smoke pp_tps within ±2% of baseline.

- [ ] **Step 7.A.7: Append T5 PASS section to bench log + final commit**

```bash
cat >> reports/p5h+2-b-bench-log.md << 'BENCHLOG_T5'

# P5h+2.b T5 — PASS close-out

**Final protocol commit**: <SHA>
**Phase 0 backfill complete**: docs/p5i-c-phase-0-{close-out,ranking-snapshot}.md updated

**Wall summary**: T0 + T1 + T2 + T3 + T4 + T5 = <total> hr (cap 15 hr per Codex Q7 D)

P5h+2.b closed PASS. Phase 0 § 7 #4 hard gate now PASS. Phase 1 implementation
acceptance unblocked.
BENCHLOG_T5

git add \
  ironmlx/tests/p5i_c_phase_0_capture.rs \
  iron-bench/src/main.rs iron-bench/src/runner.rs iron-bench/src/report.rs \
  tools/p5h_2b_t0_outlier_source.py \
  tools/p5h_2b_protocol_experiment.py \
  tools/p5h_2b_thermal_overlay.py \
  tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py \
  tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py \
  docs/p5h+2-b-protocol.md docs/p5h+2-b-close-out.md \
  docs/p5i-c-phase-0-close-out.md docs/p5i-c-phase-0-ranking-snapshot.md
git commit -m "$(cat <<'EOF'
feat(p5h+2-b-t5): close PASS — production envelope ≤ ±2% per PP; Phase 0 §7 #4 backfilled

Final protocol (lifecycle <winner>, logging <winner>, predeclared exclusions
<summary>) achieves ironmlx production pp_tps envelope ≤ ±2% per PP on ≥3
fresh-spawn repeats (PP=128 <X>%; PP=512 <Y>%). Per-PP mechanism statement
per spec § 3.2 hard binding: PP=128 trailing outliers <explained|eliminated>;
PP=512 bimodal medians <explained|eliminated>.

Reusable infra emitted in this commit: T0 outlier-source analyzer, protocol
experiment driver, thermal overlay, capture harness lifecycle/logging modes,
and iron-bench --capture-run-timestamps.

Phase 0 backfill complete: docs/p5i-c-phase-0-close-out.md § 1 #4 PASS;
docs/p5i-c-phase-0-ranking-snapshot.md envelope updated. Phase 1
implementation acceptance unblocked.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -5
T5_SHA=$(git rev-parse --short HEAD)
echo "T5 SHA: $T5_SHA"
# Backfill memory file <date> + <T5_SHA> placeholders
sed -i.bak "s/<date>/$(date +%Y-%m-%d)/g; s/<sha>/$T5_SHA/g" /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2b_findings.md
rm /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2b_findings.md.bak
```

### § 7.B T5F FAIL/DEFERRED path

- [ ] **Step 7.B.1: Write `docs/p5h+2-b-close-out.md` as FAIL/DEFERRED**

```markdown
# P5h+2.b — PP=128/512 Production Envelope Protocol Fix: Close-out

**Status:** FAIL/DEFERRED per spec § 5.7. T4 + T4R failed to achieve envelope ≤ ±2% within 15hr cap.
**Date:** <date>
**Branch:** ironmlx-p5h+2-a-pp512-measurement
**Commit chain on this branch (P5h+2.b)**:
- aabf21f spec
- <plan commit>
- this commit (validated reusable infra + T5F FAIL/DEFERRED close-out + evidence)

## § 1 Acceptance per spec § 7.2 — FAIL/DEFERRED

| # | Criterion | Verdict |
|---|---|---|
| 1 | T4/T4R failure documented with raw envelope JSON paths + per-run preserved | ✓ |
| 2 | Status FAIL/DEFERRED (not PASS) | ✓ |
| 3 | Phase 0 criterion #4 remains FAIL/DEFERRED (no PASS backfill) | ✓ |
| 4 | Next design questions explicit for new Boss + Codex round | ✓ § 4 below |
| 5 | Any committed Rust/Python tooling passes the same gates as § 7.1 #8 | ✓ |

## § 2 Failure summary

Final attempted protocol: <lifecycle / logging / exclusion summary>.
T4 acceptance sweep PP=128 envelope <X>%; PP=512 envelope <Y>%. Both > ±2%.
T4R adjusted to <T4R protocol> with envelope <X'>% / <Y'>%. Still FAIL.

## § 3 Evidence preserved

(Paths to all raw artifacts; failed protocol candidates documented in `docs/p5h+2-b-protocol.md` if committed, otherwise inlined here)

## § 4 Next design questions

(Implementer writes ≥3 explicit questions for new Boss + Codex round.
Examples:
- Is the M5 Max hardware itself the limiting factor; do we need a cross-machine retest?
- Should we abandon the ±2% envelope target and accept a wider band as honest
  measurement reality?
- Are there ironmlx server-side changes (out of P5h+2.b scope) that would
  stabilize the protocol — e.g. explicit eval barriers at request boundaries?)

## § 5 Phase 0 backfill (limited)

- docs/p5i-c-phase-0-close-out.md: P5h+2.b failed-attempt note added; criterion #4 remains FAIL/DEFERRED
- docs/p5i-c-phase-0-ranking-snapshot.md: failed-attempt envelope evidence appended; no PASS wording
```

- [ ] **Step 7.B.2: Backfill Phase 0 docs minimally (per spec § 5.7)**

Edit `docs/p5i-c-phase-0-close-out.md`: § 1 row 4 stays `✗ FAIL/DEFERRED`; append note: "P5h+2.b attempted resolution FAIL — see docs/p5h+2-b-close-out.md".

Edit `docs/p5i-c-phase-0-ranking-snapshot.md`: append "P5h+2.b attempted resolution FAILED; envelope still deferred" to preamble.

- [ ] **Step 7.B.3: Memory file + MEMORY.md update (failure case)**

```bash
cat > /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2b_findings.md << 'MEM'
---
name: project-p5h-2b-findings
description: P5h+2.b PP=128/512 production envelope protocol fix — closed FAIL/DEFERRED <date>; T4/T4R could not achieve envelope ≤ ±2% within 15hr cap; design re-think dep recorded; Phase 0 § 7 #4 remains FAIL/DEFERRED
metadata:
  type: project
---

P5h+2.b closed <date> as FAIL/DEFERRED per spec § 5.7.

**Final attempted protocol**: <lifecycle/logging/exclusions>.
**Failure**: PP=128 envelope <X>%; PP=512 envelope <Y>%; both > ±2% under 15hr cap.

**Hypothesis verdicts**:
- PP=128: T0 verdict <client/server/cross>; T2 logging mode <winner|none-passes>
- PP=512: T1 lifecycle exploration <which experiments shifted bimodal pattern by how much>; root cause <if identified>

**Next design questions** (see docs/p5h+2-b-close-out.md § 4):
- <Q1>
- <Q2>
- <Q3>

**Reusable infra emitted** (committed regardless of outcome):
- tools/p5h_2b_t0_outlier_source.py + tools/p5h_2b_protocol_experiment.py + tools/p5h_2b_thermal_overlay.py
- ironmlx/tests/p5i_c_phase_0_capture.rs lifecycle/logging env vars
- iron-bench --capture-run-timestamps flag

**Phase 0 § 7 #4 status**: remains FAIL/DEFERRED. Phase 1 implementation acceptance STILL BLOCKED.

Links: [[project-p5i-c-phase-0-findings]]; [[project-p5h-2a-findings]]; [[project-p5h-findings]].
MEM
```

Update MEMORY.md analogously to 7.A.5 with FAIL wording.

- [ ] **Step 7.B.4: Final hygiene + commit**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -3
cargo build --release
uv run --with ruff ruff check tools/p5h_2b_*.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v 2>&1 | tail -5

cat >> reports/p5h+2-b-bench-log.md << 'BENCHLOG_T5F'

# P5h+2.b T5F — FAIL/DEFERRED close-out

P5h+2.b closed FAIL/DEFERRED. Phase 0 § 7 #4 remains FAIL/DEFERRED.
Design re-think escalated to Boss + Codex per spec § 5.7.

**Wall summary**: T0+T1+T2+T3+T4 = <total> hr (cap 15 hr per Codex Q7 D)
BENCHLOG_T5F

git add \
  ironmlx/tests/p5i_c_phase_0_capture.rs \
  iron-bench/src/main.rs iron-bench/src/runner.rs iron-bench/src/report.rs \
  tools/p5h_2b_t0_outlier_source.py \
  tools/p5h_2b_protocol_experiment.py \
  tools/p5h_2b_thermal_overlay.py \
  tools/p5h_aggregator/tests/test_p5h_2b_t0_outlier_source.py \
  tools/p5h_aggregator/tests/test_p5h_2b_thermal_overlay.py \
  docs/p5h+2-b-close-out.md docs/p5i-c-phase-0-close-out.md docs/p5i-c-phase-0-ranking-snapshot.md
# Only commit docs/p5h+2-b-protocol.md if it documents a rejected candidate as non-final (per spec § 5.7)
git status --short
git commit -m "$(cat <<'EOF'
feat(p5h+2-b-t5f): close FAIL/DEFERRED — envelope ≤ ±2% not achieved within 15hr cap

T4 + T4R failed to achieve ironmlx production envelope ≤ ±2% per PP under
15hr wall cap. Best-effort protocol candidates: <inline list>. Phase 0 § 7 #4
status remains FAIL/DEFERRED; Phase 1 implementation acceptance still blocked.

Next design questions (≥3) recorded in docs/p5h+2-b-close-out.md § 4 for
Boss + Codex round. Reusable infra (T0 analyzer + protocol driver + thermal
overlay + harness lifecycle/logging env vars + iron-bench --capture-run-timestamps)
committed regardless of outcome.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -5
```

---

## Self-Review checklist (run inline before handoff per `[feedback-self-review-before-handoff]`)

1. **Spec coverage:**
   - Spec § 2 goals: Goals 1+4 → Task 6+7; Goal 2+3 → Task 6.2; Goal 5 → Tasks 1/2/5 emit reusable tools
   - Spec § 3.1 strict ±2% gate → Task 6.4 envelope verification + 6.5 decision tree
   - Spec § 3.2 independent per-PP investigation → Tasks 1/3/4 separate PP=128 + PP=512 paths
   - Spec § 3.3 no production-path changes → § File structure "Do NOT modify" + Task 2 changes scope
   - Spec § 4.2.1-4.2.5 components → Task 1 (T0 analyzer), Task 2 (harness ext + iron-bench), Task 5 (thermal overlay), Task 6 (protocol doc), Task 7 (close-out)
   - Spec § 5.1-5.7 tasks → Tasks 1-7 1:1 mapping with T5/T5F split into § 7.A/§ 7.B
   - Spec § 6 measurement protocol → Task 2 harness lifecycle modes + Task 6.3 acceptance sweep
   - Spec § 7.1/§ 7.2 acceptance → Task 7 § 7.A/§ 7.B close-out structures
   - Spec § 8 Codex priority order → task order 1→3→4→5→6 matches T0→T1→T2→T3→T4
   - Spec § 9 predeclared exclusions → Task 6.1 (BEFORE 6.3 sweep) + Step 7.A.1 finalize
   - Spec § 10 Phase 0 backfill → Step 7.A.3 + 7.A.4 + 7.B.2
   - Spec § 11 risks → Task 6.5 budget check + 7.B (FAIL path); thermal observational handled in Task 5 (powermetrics sidecar artifact check)
   - Spec § 12 wall budget → Task 6.5 cumulative wall check

2. **Placeholder scan:** "<fill>" / "<X>" / "<Y>" / "<winner>" placeholders in close-out doc templates are intentional substitution markers — implementer fills from `/tmp/p5h+2-b-t4-acceptance-pp{128,512}-envelope.json` and T1/T2 verdicts at runtime. NOT plan-level placeholders (no "TBD / TODO" pattern in step content).

3. **Type consistency:**
   - `ServerLifecycle` enum in Task 2 Step 2.2 (Phase0Current / SameSpawnCrossPp / SameSpawnPerPp) matches CLI string values (`phase0_current` / `same_spawn_cross_pp` / `same_spawn_per_pp`) used in Task 3 Step 3.2 + Task 6.3 + driver argparse
   - `LoggingMode` enum same convention (DefaultProfile / QuietAcceptance / BufferedProfile → CLI strings)
   - `--capture-run-timestamps` consistent across Task 2 Steps 2.5-2.8
   - `tools/p5h_2b_*` filename convention consistent
   - meta.json schema keys consistent across Task 2 (write_cell_meta) + Task 1 (decompose_cell parses warmup_count) + Task 5 (thermal_overlay reads)
   - `bench.csv` column names + DictReader pattern consistent across Tasks 1 + 5

No inline fixes required. Plan ready for Boss review before commit per `[feedback-review-spec-before-commit]`.
