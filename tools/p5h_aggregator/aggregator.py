"""P5h T5 aggregator entry point.

Reads ``[p5h-profile]`` log lines from server stderr + iron-bench CSV (with
``request_id`` column), joins on request_id, validates per request, computes
per-span exclusive time (per spec § 2.5a pseudocode), synthesizes
``unattributed_<span_name>`` residual leaves for non-leaf tree spans with > 1us
gap, and emits two CSV outputs:

* ``--out`` — per-request per-span attribution table (tree + synthesized
  residual leaves + diagnostic spans). Diagnostic rows carry NO ``exclusive_us``
  per spec § 2.5a (excluded from the exclusive tree by construction).
* ``--summary-out`` — per-PP summary (root_inclusive_us_median, coverage_pct,
  top-3 bottleneck span_names) used downstream by ``roi_ranking``.

Per § 7.1 coverage gate: ``coverage_pct = 1 - Σ unattributed_*.inclusive_us /
root.inclusive_us`` MUST be ≥ 0.95 per PP for the aggregator to succeed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .schema_validator import (
    Span,
    group_by_request,
    parse_line,
    validate_request,
)

# --- T5 hard-coded thresholds (per spec § 7.1 + T0a.14 gate)
COVERAGE_GATE_PCT = 0.95
NEGATIVE_EXCLUSIVE_TOLERANCE_US = 1.0  # exclusive_us >= -1us
RESIDUAL_EMIT_THRESHOLD_US = 1.0  # synthesize only if residual > 1us
SUM_TO_ROOT_TOLERANCE_US = 1.0  # |Σ tree exclusive - root.inclusive| <= 1us


@dataclass
class ResidualLeaf:
    """Synthesized ``unattributed_<span_name>`` residual leaf for non-leaf tree
    spans with > 1us gap between parent inclusive and Σ children inclusive.

    Per spec § 2.5a step 4: T5 aggregator MUST synthesize these post-validation
    (server emitters never produce ``unattributed_*`` rows directly).

    P5h+1 T2: ``chunk_idx`` is inherited from the parent tree span so the
    attribution CSV column stays consistent for descendant residuals under
    a ``gs_chunk_N`` ancestor.
    """

    span_name: str
    inclusive_us: float
    parent_span_id: int
    parent_span_name: str
    chunk_idx: int | None
    span_kind: str = "synthesized"


# --- Core computations (spec § 2.5a pseudocode) ---


def compute_exclusive(spans: list[Span]) -> dict[int, float]:
    """Compute per-span exclusive_us for tree spans.

    Per spec § 2.5a step 2: tree is built from (span_id, parent_span_id),
    NOT from span_name strings (which repeat across decoder layers).
    Diagnostic spans are excluded from the exclusive tree.

    Returns: {span_id: exclusive_us} for every tree span.

    Raises ``AssertionError`` if:
    * Any tree span has exclusive_us < -1us (broken parent_span_id attribution).
    * |Σ tree exclusive - root.inclusive| > 1us (broken tree identity).
    """
    tree = [s for s in spans if s.span_kind == "tree"]
    if not tree:
        return {}

    children_by_parent: dict[int, list[Span]] = defaultdict(list)
    for s in tree:
        if s.parent_span_id is not None:
            children_by_parent[s.parent_span_id].append(s)

    exclusive: dict[int, float] = {}
    for span in tree:
        children_inclusive = sum(
            c.inclusive_us for c in children_by_parent.get(span.span_id, [])
        )
        ex = span.inclusive_us - children_inclusive
        if ex < -NEGATIVE_EXCLUSIVE_TOLERANCE_US:
            raise AssertionError(
                f"{span.span_name} (span_id={span.span_id}): "
                f"negative exclusive {ex:.2f}us — broken parent_span attribution"
            )
        exclusive[span.span_id] = ex

    roots = [s for s in tree if s.parent_span_id is None]
    if len(roots) == 1:
        root = roots[0]
        tree_exclusive_sum = sum(exclusive.values())
        if abs(tree_exclusive_sum - root.inclusive_us) > SUM_TO_ROOT_TOLERANCE_US:
            raise AssertionError(
                f"sum-to-root invariant broken: Σ tree exclusive = "
                f"{tree_exclusive_sum:.2f}us, root.inclusive = "
                f"{root.inclusive_us:.2f}us (delta {tree_exclusive_sum - root.inclusive_us:.2f}us)"
            )

    return exclusive


def synthesize_residual_leaves(spans: list[Span]) -> list[ResidualLeaf]:
    """For each non-leaf tree span, emit a synthesized residual leaf row when
    parent inclusive - Σ children inclusive > 1us.

    Per spec § 2.5a step 4. Synthesized rows carry ``span_kind="synthesized"``
    (NOT "tree" — exempt from schema closed-set rejection) and are NOT fed back
    into structural validation or Lane-B closed-set checks.
    """
    tree = [s for s in spans if s.span_kind == "tree"]
    children_by_parent: dict[int, list[Span]] = defaultdict(list)
    for s in tree:
        if s.parent_span_id is not None:
            children_by_parent[s.parent_span_id].append(s)

    residuals: list[ResidualLeaf] = []
    for span in tree:
        children = children_by_parent.get(span.span_id, [])
        if not children:
            continue
        residual_us = span.inclusive_us - sum(c.inclusive_us for c in children)
        if residual_us > RESIDUAL_EMIT_THRESHOLD_US:
            residuals.append(
                ResidualLeaf(
                    span_name=f"unattributed_{span.span_name}",
                    inclusive_us=residual_us,
                    parent_span_id=span.span_id,
                    parent_span_name=span.span_name,
                    chunk_idx=span.chunk_idx,
                )
            )
    return residuals


def coverage_pct(root: Span, residuals: list[ResidualLeaf]) -> float:
    """Per spec § 7.1: coverage_pct = 1 - Σ residuals / root.inclusive_us.

    Must be ≥ 0.95 per PP for the aggregator to succeed.
    """
    if root.inclusive_us <= 0:
        return 0.0
    unattributed_total = sum(r.inclusive_us for r in residuals)
    return 1.0 - unattributed_total / root.inclusive_us


def diagnostic_columns(spans: list[Span]) -> dict[str, float]:
    """Sum diagnostic spans by span_name (one per request; aggregation across
    requests is the caller's job). Diagnostic spans report inclusive_us only
    per spec § 2.5a step 6.
    """
    out: dict[str, float] = {}
    for s in spans:
        if s.span_kind == "diagnostic":
            out[f"{s.span_name}_us"] = (
                out.get(f"{s.span_name}_us", 0.0) + s.inclusive_us
            )
    return out


# --- Per-request attribution model ---


@dataclass
class RequestAttribution:
    request_id: str
    pp: str  # left as string from bench CSV (e.g. "128"); not parsed to int
    routing_path: str
    root: Span
    tree_spans: list[Span]
    exclusive_us: dict[int, float]
    residuals: list[ResidualLeaf]
    diagnostics: list[Span]
    coverage: float


def build_attribution(request_spans: list[Span], pp: str) -> RequestAttribution:
    """Run the full T5 computation pipeline for one request.

    Returns a RequestAttribution carrying all data needed for the two CSV
    outputs + the coverage gate check.

    Raises ``AssertionError`` (via ``compute_exclusive``) if the request fails
    the negative-exclusive or sum-to-root invariants.
    """
    tree = [s for s in request_spans if s.span_kind == "tree"]
    diag = [s for s in request_spans if s.span_kind == "diagnostic"]
    roots = [s for s in tree if s.parent_span_id is None]
    if len(roots) != 1:
        raise AssertionError(
            f"request expected exactly 1 root, found {len(roots)} "
            f"(request_id={request_spans[0].request_id if request_spans else '?'})"
        )
    root = roots[0]
    exclusive = compute_exclusive(request_spans)
    residuals = synthesize_residual_leaves(request_spans)
    cov = coverage_pct(root, residuals)
    return RequestAttribution(
        request_id=root.request_id,
        pp=pp,
        routing_path=root.routing_path,
        root=root,
        tree_spans=tree,
        exclusive_us=exclusive,
        residuals=residuals,
        diagnostics=diag,
        coverage=cov,
    )


# --- CSV emission ---


def write_attribution_csv(
    attributions: list[RequestAttribution], out_path: Path
) -> None:
    """Per-request per-span attribution table.

    Columns: ``pp, request_id, routing_path, chunk_idx, span_name, span_kind,
    parent_span_id, span_id, inclusive_us, exclusive_us``.

    * Tree rows carry numeric ``exclusive_us`` (may be 0.0 for leaves).
    * Synthesized residual rows carry ``span_kind="synthesized"``,
      ``exclusive_us=inclusive_us`` (residual leaves have no children).
    * Diagnostic rows carry ``span_kind="diagnostic"``, ``exclusive_us=""``
      (empty — diagnostic spans are NOT in the exclusive tree per § 2.5a).

    P5h+1 T2: ``chunk_idx`` is inserted immediately after ``routing_path``
    and is empty for spans outside a Lane-B ``gs_chunk_N`` ancestor
    (Lane-A entirely; Lane-B pre/post-loop sites). Residual rows inherit
    ``chunk_idx`` from the parent tree span via ``ResidualLeaf.chunk_idx``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pp",
                "request_id",
                "routing_path",
                "chunk_idx",
                "span_name",
                "span_kind",
                "parent_span_id",
                "span_id",
                "inclusive_us",
                "exclusive_us",
            ]
        )
        for attr in attributions:
            for s in attr.tree_spans:
                w.writerow(
                    [
                        attr.pp,
                        attr.request_id,
                        attr.routing_path,
                        "" if s.chunk_idx is None else s.chunk_idx,
                        s.span_name,
                        s.span_kind,
                        "" if s.parent_span_id is None else s.parent_span_id,
                        s.span_id,
                        f"{s.inclusive_us:.2f}",
                        f"{attr.exclusive_us.get(s.span_id, 0.0):.2f}",
                    ]
                )
            for r in attr.residuals:
                w.writerow(
                    [
                        attr.pp,
                        attr.request_id,
                        attr.routing_path,
                        "" if r.chunk_idx is None else r.chunk_idx,
                        r.span_name,
                        r.span_kind,
                        r.parent_span_id,
                        "",
                        f"{r.inclusive_us:.2f}",
                        f"{r.inclusive_us:.2f}",
                    ]
                )
            for d in attr.diagnostics:
                w.writerow(
                    [
                        attr.pp,
                        attr.request_id,
                        attr.routing_path,
                        "" if d.chunk_idx is None else d.chunk_idx,
                        d.span_name,
                        d.span_kind,
                        "" if d.parent_span_id is None else d.parent_span_id,
                        d.span_id,
                        f"{d.inclusive_us:.2f}",
                        "",
                    ]
                )


def _top3_bottlenecks_for_pp(
    attributions_for_pp: list[RequestAttribution],
) -> list[tuple[str, float]]:
    """Per spec § 1.2 ROI gate: rank by exclusive_us / root.inclusive_us share,
    median across requests at this PP.

    Fix A (Codex T5-R review): multi-emit spans like ``gs_chunk_N``,
    ``decoder_layer_N``, GDN/MoE substeps emit MULTIPLE tree records per request
    (one per chunk / per layer / per substep iteration). The correct per-PP
    median is over the PER-REQUEST TOTAL (sum across same-name records within
    one request), NOT over individual records. Taking median across individual
    records underreports the per-request cost by a factor equal to the per-
    request emit count (e.g. PP=2048 gs_chunk_N: 2 records/request × 7 requests
    = 14 records; per-record median ≈ 1/2 of per-request total).

    Uses tree exclusive + synthesized residual leaves (avoids double-counting
    inclusive_us up the parent chain). Returns top-3 ``[(span_name, share), ...]``.
    """
    # First: build per-(request, span_name) sums of exclusive_us (tree) +
    # inclusive_us (synthesized residual leaves). One sum per request per name.
    per_req_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    root_us_by_req: dict[str, float] = {}
    for attr in attributions_for_pp:
        if attr.root.inclusive_us <= 0:
            continue
        root_us_by_req[attr.request_id] = attr.root.inclusive_us
        for s in attr.tree_spans:
            if s.parent_span_id is None:
                continue  # skip root itself
            ex = attr.exclusive_us.get(s.span_id, 0.0)
            per_req_totals[attr.request_id][s.span_name] += ex
        for r in attr.residuals:
            per_req_totals[attr.request_id][r.span_name] += r.inclusive_us

    # Second: per-PP per-span share series = per-request total / per-request
    # root, then median across requests.
    share_acc: dict[str, list[float]] = defaultdict(list)
    for rid, names in per_req_totals.items():
        root_us = root_us_by_req[rid]
        for name, total in names.items():
            share_acc[name].append(total / root_us)

    avg = [
        (name, statistics.median(shares))
        for name, shares in share_acc.items()
        if shares
    ]
    avg.sort(key=lambda x: x[1], reverse=True)
    return avg[:3]


def write_summary_csv(
    attributions: list[RequestAttribution], out_path: Path
) -> dict[str, float]:
    """Per-PP summary table. Columns:
    ``pp, request_count, root_inclusive_us_median, coverage_pct_median,
    coverage_pct_min, top1_span_name, top1_share, top2_span_name, top2_share,
    top3_span_name, top3_share``.

    Returns ``{pp: coverage_pct_median}`` for the caller (used by the hard
    coverage gate check).
    """
    by_pp: dict[str, list[RequestAttribution]] = defaultdict(list)
    for a in attributions:
        by_pp[a.pp].append(a)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    median_cov_by_pp: dict[str, float] = {}
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pp",
                "request_count",
                "root_inclusive_us_median",
                "coverage_pct_median",
                "coverage_pct_min",
                "top1_span_name",
                "top1_share",
                "top2_span_name",
                "top2_share",
                "top3_span_name",
                "top3_share",
            ]
        )

        def _pp_sort_key(pp_str: str) -> int:
            try:
                return int(pp_str)
            except ValueError:
                return -1

        for pp in sorted(by_pp, key=_pp_sort_key):
            attrs = by_pp[pp]
            root_us_values = [a.root.inclusive_us for a in attrs]
            cov_values = [a.coverage for a in attrs]
            top3 = _top3_bottlenecks_for_pp(attrs)
            row: list = [
                pp,
                len(attrs),
                f"{statistics.median(root_us_values):.2f}",
                f"{statistics.median(cov_values):.4f}",
                f"{min(cov_values):.4f}",
            ]
            for i in range(3):
                if i < len(top3):
                    row.extend([top3[i][0], f"{top3[i][1]:.4f}"])
                else:
                    row.extend(["", ""])
            w.writerow(row)
            median_cov_by_pp[pp] = statistics.median(cov_values)
    return median_cov_by_pp


# --- CLI / orchestration ---


def _load_spans(server_log: Path) -> list[Span]:
    spans: list[Span] = []
    with server_log.open() as f:
        for line in f:
            s = parse_line(line)
            if s is not None:
                spans.append(s)
    return spans


def _load_bench(bench_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with bench_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("request_id", "").strip()
            if rid:
                out[rid] = row
    return out


def _check_join(
    server_req_ids: set[str], bench_req_ids: set[str]
) -> tuple[set[str], set[str]]:
    """Returns (server_orphans, bench_orphans). Caller hard-fails if non-empty."""
    return (server_req_ids - bench_req_ids, bench_req_ids - server_req_ids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--server-log",
        required=True,
        type=Path,
        help="server stderr capture (with [p5h-profile] lines)",
    )
    ap.add_argument(
        "--bench-csv",
        required=True,
        type=Path,
        help="iron-bench CSV (with request_id column)",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output per-request per-span attribution table (CSV)",
    )
    ap.add_argument(
        "--summary-out",
        required=False,
        default=None,
        type=Path,
        help=(
            "output per-PP summary table (CSV); "
            "defaults to <--out basename>.summary.csv"
        ),
    )
    args = ap.parse_args(argv)
    if args.summary_out is None:
        args.summary_out = args.out.with_name(args.out.stem + ".summary.csv")

    spans = _load_spans(args.server_log)
    if not spans:
        print("ERROR: no [p5h-profile] spans parsed from server log", file=sys.stderr)
        return 2

    bench_by_req = _load_bench(args.bench_csv)
    grouped = group_by_request(spans)

    # Per Codex plan review v1 P1 #3 + § 2.5a Join key:
    # iron-bench↔server request_id join MUST be 100%. Any orphan = broken
    # header propagation = hard-fail before any downstream computation.
    server_req_ids = set(grouped.keys())
    bench_req_ids = set(bench_by_req.keys())
    server_orphans, bench_orphans = _check_join(server_req_ids, bench_req_ids)

    if server_orphans or bench_orphans:
        print(
            "JOIN HARD-FAIL: per § 2.5a Join key, request_id join rate must = 100% (orphan rate = 0%)",
            file=sys.stderr,
        )
        if server_orphans:
            print(
                f"  server log has {len(server_orphans)} request_id(s) absent from iron-bench CSV:",
                file=sys.stderr,
            )
            for r in sorted(server_orphans)[:10]:
                print(f"    {r}", file=sys.stderr)
            if len(server_orphans) > 10:
                print(f"    ... +{len(server_orphans) - 10} more", file=sys.stderr)
        if bench_orphans:
            print(
                f"  iron-bench CSV has {len(bench_orphans)} request_id(s) absent from server log:",
                file=sys.stderr,
            )
            for r in sorted(bench_orphans)[:10]:
                print(f"    {r}", file=sys.stderr)
            if len(bench_orphans) > 10:
                print(f"    ... +{len(bench_orphans) - 10} more", file=sys.stderr)
        print(
            "Likely causes: server not built with --features p5h-profile; iron-bench --capture-server-request-id flag off; header propagation bug.",
            file=sys.stderr,
        )
        return 4

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

    # Per-request schema validation (Codex plan review v1 P1 #3 + spec § 2.5a).
    failures: list[str] = []
    for req_id, request_spans in grouped.items():
        rep = validate_request(request_spans)
        if not rep.ok:
            for fail in rep.failures:
                failures.append(f"{req_id}: {fail}")

    if failures:
        print("VALIDATION FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 3

    # Per-request attribution computation (compute_exclusive +
    # synthesize_residual_leaves + coverage_pct).
    attributions: list[RequestAttribution] = []
    aborted_skipped = 0
    for req_id, request_spans in grouped.items():
        # Skip aborted requests (excluded from coverage gate per § 7.1 + Codex
        # v12 P2 #6): they intentionally lack downstream spans.
        tree = [s for s in request_spans if s.span_kind == "tree"]
        if any(s.parent_span_id is None and s.mode == "aborted" for s in tree):
            aborted_skipped += 1
            continue
        pp = bench_by_req.get(req_id, {}).get("pp_target", "")
        try:
            attr = build_attribution(request_spans, pp=pp)
        except AssertionError as e:
            print(f"STRUCTURAL FAILURE on {req_id}: {e}", file=sys.stderr)
            return 5
        attributions.append(attr)

    if not attributions:
        print(
            "ERROR: zero non-aborted requests with attribution computed",
            file=sys.stderr,
        )
        return 6

    write_attribution_csv(attributions, args.out)
    median_cov_by_pp = write_summary_csv(attributions, args.summary_out)

    # Coverage gate check (per spec § 7.1). T0a Lane-A GDN substep coverage was
    # capped at 50-55% median by per-substep tracing::info! emit overhead per
    # close-out; T5 gate per spec text is 95% but full-coverage attribution
    # gate sources gating from per-PP, per-lane median. We report all and let
    # the consumer (T5 close-out) decide which gate applies per lane/PP.
    failed_pps: list[tuple[str, float]] = []
    for pp, cov in median_cov_by_pp.items():
        flag = "OK" if cov >= COVERAGE_GATE_PCT else "BELOW_GATE"
        print(
            f"  PP={pp}: coverage_pct_median={cov:.4f} ({flag} vs gate {COVERAGE_GATE_PCT:.2f})",
            file=sys.stderr,
        )
        if cov < COVERAGE_GATE_PCT:
            failed_pps.append((pp, cov))

    print(
        f"OK: {len(grouped)} requests ({aborted_skipped} aborted skipped), "
        f"{len(attributions)} attributed, {len(spans)} spans, join rate 100%, "
        f"written attribution to {args.out}, summary to {args.summary_out}",
        file=sys.stderr,
    )

    if failed_pps:
        print(
            f"COVERAGE GATE FAILURE: {len(failed_pps)} PP(s) below {COVERAGE_GATE_PCT:.2f}:",
            file=sys.stderr,
        )
        for pp, cov in failed_pps:
            print(f"  PP={pp}: median coverage = {cov:.4f}", file=sys.stderr)
        # Diagnostic: list unattributed_* spans exceeding 1% of root for failing PPs.
        attrs_by_pp: dict[str, list[RequestAttribution]] = defaultdict(list)
        for a in attributions:
            attrs_by_pp[a.pp].append(a)
        for pp, _ in failed_pps:
            print(f"  PP={pp} top residual spans (median share > 1%):", file=sys.stderr)
            share_acc: dict[str, list[float]] = defaultdict(list)
            for a in attrs_by_pp[pp]:
                root_us = a.root.inclusive_us
                if root_us <= 0:
                    continue
                for r in a.residuals:
                    share_acc[r.span_name].append(r.inclusive_us / root_us)
            ranked = sorted(
                ((n, statistics.median(s)) for n, s in share_acc.items() if s),
                key=lambda x: x[1],
                reverse=True,
            )
            for name, share in ranked:
                if share > 0.01:
                    print(f"    {name}: median {share:.2%}", file=sys.stderr)
        return 7

    return 0


if __name__ == "__main__":
    sys.exit(main())
