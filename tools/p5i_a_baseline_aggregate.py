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
            raise SystemExit(
                f"{csv_path}: missing required CSV columns: {sorted(missing)}"
            )
        for row in reader:
            try:
                pp = int(row["pp_target"])
                pp_tps = float(row["pp_tps"])
            except (KeyError, ValueError):
                continue
            if pp not in EXPECTED_PPS:
                raise SystemExit(
                    f"{csv_path}: unexpected pp_target={pp}; expected {EXPECTED_PPS}"
                )
            if pp_tps <= 0:
                raise SystemExit(
                    f"{csv_path}: non-positive pp_tps={pp_tps} for pp={pp}"
                )
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
        i, o, d = (
            row["ironmlx_pp_tps_median"],
            row["omlx_pp_tps_median"],
            row["delta_pct"],
        )
        flag = "PASS" if row["passes_plus10_target"] else "MISS"
        print(f"  PP={pp}: ironmlx={i:.2f} omlx={o:.2f} delta={d:+.2f}% {flag}")


if __name__ == "__main__":
    main()
