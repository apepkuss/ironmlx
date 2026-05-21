"""P5h T5 aggregator entry point.

Reads `[p5h-profile]` log lines from server stderr + iron-bench CSV
(with `request_id` column), joins on request_id, validates per request,
and emits per-PP attribution table.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from .schema_validator import parse_line, validate_request, group_by_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True, type=Path, help="server stderr capture (with [p5h-profile] lines)")
    ap.add_argument("--bench-csv", required=True, type=Path, help="iron-bench CSV (with request_id column)")
    ap.add_argument("--out", required=True, type=Path, help="output attribution table (CSV)")
    args = ap.parse_args()

    spans = []
    with args.server_log.open() as f:
        for line in f:
            s = parse_line(line)
            if s is not None:
                spans.append(s)

    if not spans:
        print("ERROR: no [p5h-profile] spans parsed from server log", file=sys.stderr)
        sys.exit(2)

    # Join iron-bench CSV to attach pp/run_idx
    bench_by_req: dict[str, dict] = {}
    with args.bench_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("request_id", "").strip()
            if rid:
                bench_by_req[rid] = row

    grouped = group_by_request(spans)

    # Per Codex plan review v1 P1 #3 + § 2.5a Join key:
    # iron-bench↔server request_id join MUST be 100%. Any orphan = broken
    # header propagation = hard-fail before any downstream computation.
    server_req_ids = set(grouped.keys())
    bench_req_ids = set(bench_by_req.keys())
    server_orphans = server_req_ids - bench_req_ids  # server log has spans for a request bench CSV doesn't know
    bench_orphans = bench_req_ids - server_req_ids   # bench CSV has a request server log has no spans for

    if server_orphans or bench_orphans:
        print("JOIN HARD-FAIL: per § 2.5a Join key, request_id join rate must = 100% (orphan rate = 0%)", file=sys.stderr)
        if server_orphans:
            print(f"  server log has {len(server_orphans)} request_id(s) absent from iron-bench CSV:", file=sys.stderr)
            for r in sorted(server_orphans)[:10]:
                print(f"    {r}", file=sys.stderr)
            if len(server_orphans) > 10:
                print(f"    ... +{len(server_orphans) - 10} more", file=sys.stderr)
        if bench_orphans:
            print(f"  iron-bench CSV has {len(bench_orphans)} request_id(s) absent from server log:", file=sys.stderr)
            for r in sorted(bench_orphans)[:10]:
                print(f"    {r}", file=sys.stderr)
            if len(bench_orphans) > 10:
                print(f"    ... +{len(bench_orphans) - 10} more", file=sys.stderr)
        print("Likely causes: server not built with --features p5h-profile; iron-bench --capture-server-request-id flag off; header propagation bug.", file=sys.stderr)
        sys.exit(4)

    # Per-PP join rate breakdown (informational; total join rate already
    # validated above as 100%).
    pp_join_rates: dict[str, tuple[int, int]] = {}
    for rid in server_req_ids:
        pp = bench_by_req.get(rid, {}).get("pp_target", "?")
        matched, total = pp_join_rates.get(pp, (0, 0))
        pp_join_rates[pp] = (matched + 1, total + 1)
    for pp in sorted(pp_join_rates, key=lambda x: int(x) if x.isdigit() else -1):
        matched, total = pp_join_rates[pp]
        print(f"  PP={pp}: join_rate={matched}/{total} (100.0%)", file=sys.stderr)

    failures = []
    for req_id, request_spans in grouped.items():
        rep = validate_request(request_spans)
        if not rep.ok:
            for fail in rep.failures:
                failures.append(f"{req_id}: {fail}")

    if failures:
        print("VALIDATION FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(3)

    # Compute per-PP per-span exclusive_us (placeholder — full T5 work below)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "pp", "span_name", "inclusive_us"])
        for req_id, request_spans in grouped.items():
            pp = bench_by_req.get(req_id, {}).get("pp_target", "")
            for s in request_spans:
                w.writerow([req_id, pp, s.span_name, f"{s.inclusive_us:.2f}"])

    print(f"OK: {len(grouped)} requests, {len(spans)} spans, join rate 100%, written to {args.out}")


if __name__ == "__main__":
    main()
