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
OUTLIER_THRESHOLD_PCT = (
    10.0  # run flagged outlier if pp_tps deviates >10% from cell median
)


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
            row
            for row in reader
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
    m = re.search(r"-pp(\d+)-", str(cell_dir))
    if m is None:
        raise SystemExit(
            f"{cell_dir}: cannot extract PP from dir name (expected -ppN- pattern)"
        )
    pp = int(m.group(1))
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
            verdict.note = (
                f"probe request_id join missing {len(missing)} roots: {missing[:3]}"
            )
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
                (
                    row.get("run_idx", "?"),
                    expected_prompt,
                    None if root is None else root.prompt_tokens,
                )
            )
    if prompt_mismatches:
        verdict.verdict = "inconclusive"
        verdict.note = (
            f"prompt_tokens mismatch in joined roots: {prompt_mismatches[:3]}"
        )
        return verdict

    pp_tps_list = [float(r[0]["pp_tps"]) for r in join]
    cell_median = median(pp_tps_list)
    for row, root in join:
        root_us = root.inclusive_us if root is not None else None
        pp_tps = float(row["pp_tps"])
        ttft_ms = float(row["ttft_ms"])
        deviation_pct = (
            abs(pp_tps - cell_median) / cell_median * 100 if cell_median > 0 else 0
        )
        is_outlier = deviation_pct > OUTLIER_THRESHOLD_PCT
        client_overhead = (ttft_ms * 1000) - root_us if root_us is not None else None
        verdict.runs.append(
            RunDecomp(
                run_idx=int(row["run_idx"]),
                pp_tps=pp_tps,
                ttft_ms=ttft_ms,
                server_root_inclusive_us=root_us,
                client_overhead_us=client_overhead,
                is_outlier=is_outlier,
            )
        )

    # Verdict: examine outliers' decomposition
    outliers = [r for r in verdict.runs if r.is_outlier]
    if not outliers:
        verdict.verdict = "inconclusive"
        verdict.note = (
            "no pp_tps outliers above threshold; no source classification needed"
        )
        return verdict
    # Among outliers, fraction where server_root is also abnormally slow
    # Hoist median_root once; explicit invariant for all-None edge case
    root_us_values = [
        r.server_root_inclusive_us
        for r in verdict.runs
        if r.server_root_inclusive_us is not None
    ]
    if not root_us_values:
        verdict.verdict = "inconclusive"
        verdict.note = "all outlier roots missing server_root_inclusive_us"
        return verdict
    median_root = median(root_us_values)
    server_slow_count = 0
    for o in outliers:
        if o.server_root_inclusive_us is None:
            continue  # should not occur post-prompt-mismatch guard, but safe
        if o.server_root_inclusive_us > median_root * 1.1:
            server_slow_count += 1
    if server_slow_count == len(outliers):
        verdict.verdict = "server_side"
    elif server_slow_count == 0:
        verdict.verdict = "client_side"
    else:
        verdict.verdict = "cross"
    return verdict


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
        lines.append(
            "| run_idx | pp_tps | ttft_ms | server_root_us | client_overhead_us | outlier? |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in v.runs:
            ot = "★" if r.is_outlier else ""
            root_str = (
                f"{r.server_root_inclusive_us:.0f}"
                if r.server_root_inclusive_us is not None
                else "N/A"
            )
            client_str = (
                f"{r.client_overhead_us:.0f}"
                if r.client_overhead_us is not None
                else "N/A"
            )
            lines.append(
                f"| {r.run_idx} | {r.pp_tps:.2f} | {r.ttft_ms:.2f} | {root_str} | {client_str} | {ot} |"
            )
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
    p.add_argument(
        "--cells-glob",
        required=True,
        action="append",
        help="glob for cell dirs (repeat for multiple patterns)",
    )
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
