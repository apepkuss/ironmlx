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
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_PPS = (128, 512)
EXPECTED_RUNS_PER_PP = 7
# Per-PP run_idx set must exactly match this. iron-bench emits one row per
# (pp_target, run_idx) with run_idx in [0, runs); spec § 7 sweeps run with
# --runs 7 so the expected set is {0..6}.
EXPECTED_RUN_IDX_SET = frozenset(range(EXPECTED_RUNS_PER_PP))


def load_pp_tps_by_pp(csv_path: Path) -> dict[int, list[float]]:
    """Parse iron-bench CSV at `csv_path` and return {pp_target: [pp_tps,...]}.

    Per Codex P5i.a P2 finding #4, this function fails HARD on any malformed
    row (missing column, unparseable int/float, non-positive pp_tps, unknown
    pp_target) and validates that each PP has exactly the expected run_idx
    set with no duplicates and no gaps. Silent skip-on-error is intentionally
    NOT used: that behavior could mask missing or duplicated measurement runs
    and produce a summary that misreports the underlying sweep.
    """
    by_pp_runs: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"target", "pp_target", "run_idx", "pp_tps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{csv_path}: missing required CSV columns: {sorted(missing)}"
            )
        # csv.DictReader starts data rows at line 2 (line 1 = header).
        for line_no, row in enumerate(reader, start=2):
            try:
                pp_raw = row["pp_target"]
                run_idx_raw = row["run_idx"]
                pp_tps_raw = row["pp_tps"]
            except KeyError as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: missing column {exc}; row={row!r}"
                ) from exc
            try:
                pp = int(pp_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: pp_target={pp_raw!r} not int; row={row!r}"
                ) from exc
            try:
                run_idx = int(run_idx_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: run_idx={run_idx_raw!r} not int; row={row!r}"
                ) from exc
            try:
                pp_tps = float(pp_tps_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: pp_tps={pp_tps_raw!r} not float; row={row!r}"
                ) from exc
            if pp not in EXPECTED_PPS:
                raise SystemExit(
                    f"{csv_path}: line {line_no}: unexpected pp_target={pp}; expected {EXPECTED_PPS}"
                )
            if pp_tps <= 0:
                raise SystemExit(
                    f"{csv_path}: line {line_no}: non-positive pp_tps={pp_tps} for pp={pp}"
                )
            by_pp_runs[pp].append((run_idx, pp_tps))
    # Per-PP validation: exact row count + run_idx set must match EXPECTED_RUN_IDX_SET
    # (no duplicates, no gaps, no extras).
    by_pp: dict[int, list[float]] = {}
    for pp in EXPECTED_PPS:
        rows = by_pp_runs.get(pp, [])
        got = len(rows)
        if got != EXPECTED_RUNS_PER_PP:
            raise SystemExit(
                f"{csv_path}: expected {EXPECTED_RUNS_PER_PP} measured rows for PP={pp}, got {got}"
            )
        run_idxs = [r for r, _ in rows]
        observed_set = set(run_idxs)
        if len(observed_set) != len(run_idxs):
            # Duplicate run_idx detected.
            dupes = sorted(idx for idx, count in Counter(run_idxs).items() if count > 1)
            raise SystemExit(
                f"{csv_path}: PP={pp} has duplicate run_idx values: {dupes}; observed={sorted(run_idxs)}"
            )
        if observed_set != EXPECTED_RUN_IDX_SET:
            raise SystemExit(
                f"{csv_path}: PP={pp} run_idx set was {sorted(observed_set)} "
                f"but expected {sorted(EXPECTED_RUN_IDX_SET)}"
            )
        by_pp[pp] = [tps for _, tps in rows]
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
