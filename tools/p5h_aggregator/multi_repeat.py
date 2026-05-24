"""P5i.c Phase 0 — multi-repeat substep aggregator wrapper + production root extraction.

Per spec § 4.2.2 + plan Step 5.2: delegates per-repeat per-substep aggregation to
`tools/p5h_aggregator/aggregator.py` (probe mode only — production lacks
request-id join), collects per-substep medians across repeats, emits bootstrap
CI95 per substep. Also extracts production_root_us from flag-OFF server.log
root spans (no aggregator subprocess needed; direct parse_line on root rows).

CLI:
    python tools/p5h_aggregator/multi_repeat.py \\
        --repeat-dir /tmp/p5i-c-phase-0-r1-pp128-probe \\
        --repeat-dir /tmp/p5i-c-phase-0-r2-pp128-probe \\
        --repeat-dir /tmp/p5i-c-phase-0-r3-pp128-probe \\
        [--production-repeat-dir /tmp/p5i-c-phase-0-r1-pp128-production ...] \\
        --pp 128 \\
        --out-json /tmp/p5i-c-phase-0-pp128-multirepeat.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))
from p5h_2a_se_analysis import bootstrap_median_ci  # noqa: E402

# Import schema_validator's parse_line for production-mode server.log parsing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_validator import parse_line  # noqa: E402


def run_aggregator_one_probe_cell(repeat_dir: Path, tmp_dir: Path) -> Path:
    """Invoke tools/p5h_aggregator/aggregator.py on a probe-mode cell.

    Production cells lack request-id join and MUST NOT be passed here; use
    extract_production_root_us() for production-mode root_us instead.

    Returns the attribution CSV path. Hard-fails if aggregator returns non-zero
    or required files are missing.
    """
    server_log = repeat_dir / "server.log"
    bench_csv = repeat_dir / "bench.csv"
    if not server_log.exists() or not bench_csv.exists():
        raise SystemExit(f"{repeat_dir}: missing server.log or bench.csv")
    attribution_csv = tmp_dir / f"{repeat_dir.name}-attribution.csv"
    summary_csv = tmp_dir / f"{repeat_dir.name}-attribution.summary.csv"
    cmd = [
        sys.executable,
        "-m",
        "p5h_aggregator.aggregator",
        "--server-log",
        str(server_log),
        "--bench-csv",
        str(bench_csv),
        "--out",
        str(attribution_csv),
        "--summary-out",
        str(summary_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TOOLS_DIR))
    if result.returncode != 0:
        raise SystemExit(
            f"{repeat_dir}: aggregator failed (exit={result.returncode}):\n"
            f"  stdout={result.stdout}\n  stderr={result.stderr}"
        )
    return attribution_csv


def parse_attribution_csv(path: Path) -> dict[int, dict[str, float]]:
    """Extract per-PP per-substep share = per_request_sum_median / root_inclusive_us_median.

    Per-request sum first (P5h+1 Fix A binding per aggregator.py multi-emit
    correction): many spans (gather_qmm_*, gda_step_*, decoder per-layer
    spans) emit multiple times per request — once per MoE/decoder layer.
    The per-emission median undercounts by emit-count factor (~28x at
    Qwen3.5-35B-A3B). Aggregator.py sums per-request first then takes
    median across requests; parse_attribution_csv must mirror that.

    Root row identified by empty `parent_span_id` + span_kind=tree (per
    aggregator output contract; see aggregator.py write_attribution). Root
    span itself is EXCLUDED from substep shares (denominator only).

    Per spec § 4.2.4 P5h+1 § 6.5 denominator discipline: probe-mode root is
    used here for relative ranking; production_root_us is the production-mode
    denominator (extracted separately).
    """
    if not path.exists():
        raise SystemExit(f"{path}: attribution CSV not found")
    # Per-request per-span sum accumulator: (pp, request_id, span_name) → sum
    per_req_totals: dict[tuple[int, str, str], float] = defaultdict(float)
    # Per-request root: (pp, request_id) → inclusive_us
    per_req_root_us: dict[tuple[int, str], float] = {}
    root_span_names: dict[int, set[str]] = defaultdict(set)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pp = int(row["pp"])
            except (KeyError, ValueError):
                continue
            request_id = row.get("request_id", "")
            span_name = row.get("span_name", "")
            span_kind = row.get("span_kind", "")
            parent = row.get("parent_span_id", "")
            inclusive_raw = row.get("inclusive_us", "")
            exclusive_raw = row.get("exclusive_us", "")
            # Root row: parent_span_id is empty + tree kind. Each request has
            # exactly one root (per spec § 2.5a).
            if parent == "" and span_kind == "tree":
                try:
                    per_req_root_us[(pp, request_id)] = float(inclusive_raw)
                    root_span_names[pp].add(span_name)
                except ValueError:
                    pass
                continue
            # Skip diagnostic rows (no exclusive_us per spec § 2.5a)
            if span_kind == "diagnostic":
                continue
            if exclusive_raw == "":
                continue
            try:
                ex_us = float(exclusive_raw)
            except ValueError:
                continue
            per_req_totals[(pp, request_id, span_name)] += ex_us

    # Per-PP: collect per-request roots + per-request per-span totals
    root_us_by_pp: dict[int, list[float]] = defaultdict(list)
    for (pp, _rid), root_us in per_req_root_us.items():
        root_us_by_pp[pp].append(root_us)
    spans_by_pp: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (pp, _rid, span_name), total in per_req_totals.items():
        spans_by_pp[pp][span_name].append(total)

    result: dict[int, dict[str, float]] = {}
    for pp, spans in spans_by_pp.items():
        if not root_us_by_pp[pp]:
            raise SystemExit(f"{path}: PP={pp} has no root span row")
        root_med = median(root_us_by_pp[pp])
        if root_med <= 0:
            raise SystemExit(f"{path}: PP={pp} root_median <= 0")
        result[pp] = {
            name: median(totals) / root_med
            for name, totals in spans.items()
            if name not in root_span_names[pp] and totals
        }
    return result


def aggregate_multi_repeat(repeat_dirs: list[Path], pp: int) -> dict:
    """Run aggregator per probe-mode repeat, collect per-substep shares, bootstrap CI95.

    `between_sweep_half_range_pct` is a percentage-point half range:
    `(max(shares_pct) - min(shares_pct)) / 2`. Per plan Step 5.2 wording.
    """
    if len(repeat_dirs) < 3:
        raise SystemExit("need >=3 repeat-dir inputs")
    per_repeat_shares: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for rd in repeat_dirs:
            attribution_csv = run_aggregator_one_probe_cell(rd, tmp_dir)
            per_pp_data = parse_attribution_csv(attribution_csv)
            if pp not in per_pp_data:
                raise SystemExit(f"{rd}: PP={pp} not found in attribution")
            per_repeat_shares.append(per_pp_data[pp])

    all_substeps: set[str] = set()
    for d in per_repeat_shares:
        all_substeps.update(d.keys())

    per_substep: dict[str, dict] = {}
    for substep in all_substeps:
        shares_pct = [d.get(substep, 0.0) * 100 for d in per_repeat_shares]
        med = median(shares_pct)
        rng = random.Random(42)
        ci = bootstrap_median_ci(
            shares_pct, subset_size=len(shares_pct), iterations=1000, rng=rng
        )
        between_hr_pct = (max(shares_pct) - min(shares_pct)) / 2
        per_substep[substep] = {
            "median_pct": med,
            "ci95_low_pct": ci["ci95_low"],
            "ci95_high_pct": ci["ci95_high"],
            "ci95_half_width_pct": ci["ci95_half_width_pct"],
            "between_sweep_half_range_pct": between_hr_pct,
            "per_repeat_shares_pct": shares_pct,
        }

    return {
        "pp": pp,
        "n_repeats": len(repeat_dirs),
        "per_substep": per_substep,
    }


def extract_production_root_us(repeat_dirs: list[Path], pp: int) -> dict:
    """Parse production server.log root spans; report median root_inclusive_us per PP.

    Production mode does NOT use --capture-server-request-id, so the aggregator
    cannot run. We parse the [p5h-profile] log lines directly with
    schema_validator.parse_line and select root spans (parent_span_id is None).

    Per `[project_p5h_t1_findings]` Qwen3 ChatML overhead = 12 tokens, so the
    server emits prompt_tokens = pp + 12 (or similar) for an iron-bench
    `--prompt-len pp` request. The per-cell harness writes one PP's requests
    per server.log (each cell has its own server spawn + kill), so we accept
    ANY root span in the log — no prompt_tokens filter required.
    """
    if len(repeat_dirs) < 3:
        raise SystemExit("need >=3 production-repeat-dir inputs")
    per_repeat_root_medians: list[float] = []
    for rd in repeat_dirs:
        log_path = rd / "server.log"
        if not log_path.exists():
            raise SystemExit(f"{rd}: server.log missing")
        per_request_root_us: list[float] = []
        with log_path.open() as f:
            for line in f:
                span = parse_line(line)
                if span is None:
                    continue
                if span.parent_span_id is not None:
                    continue
                # No prompt_tokens filter: per-cell log is mono-PP by harness
                # construction.
                per_request_root_us.append(span.inclusive_us)
        if not per_request_root_us:
            raise SystemExit(f"{rd}: no production root spans found")
        per_repeat_root_medians.append(median(per_request_root_us))

    overall_median = median(per_repeat_root_medians)
    return {
        "pp": pp,
        "n_repeats": len(repeat_dirs),
        "per_repeat_root_us_median": per_repeat_root_medians,
        "production_root_us_median": overall_median,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repeat-dir",
        type=Path,
        action="append",
        required=True,
        help="probe-mode cell directories (server.log + bench.csv)",
    )
    p.add_argument(
        "--production-repeat-dir",
        type=Path,
        action="append",
        default=None,
        help="production-mode cell directories (server.log root spans only)",
    )
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()
    result: dict = aggregate_multi_repeat(args.repeat_dir, args.pp)
    if args.production_repeat_dir:
        result["production_root"] = extract_production_root_us(
            args.production_repeat_dir, args.pp
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
