"""ROI ranking: identify top-3 bottleneck per PP + P5i (short PP) + P5j (long
PP) candidates from the T5 aggregator attribution table.

Inputs (both CSV files produced by ``aggregator.py``):

* ``--attribution-csv`` — per-request per-span table (one row per tree /
  synthesized / diagnostic span).
* ``--summary-csv`` — per-PP summary (root_inclusive_us_median, coverage_pct,
  top-3 bottleneck names — used for cross-check only).

Outputs:

* ``--out-ranking`` — CSV of all ROI candidates with score + scope_gate flag.
* ``--out-verdict`` — JSON of per-PP 4-tier feasibility verdict.

Scoring (per Codex Q-T5-3 / Q-T5-4 bindings in T5 plan):

* ``max_gain_pct = candidate_us / root_us`` (span → 0 upper bound).
* ``realistic_*`` reflects typical op-level (30-50%) or kernel rewrite
  (50-70%) reduction.
* ``gap_weight = max(0, target - current_gain) / target`` — urgency toward
  spec § 1.2 PP target gain.
* ``score = max_gain_pct * gap_weight`` (kernel rewrites surface at their
  real ROI cost; Scope gate decision is Boss's call per Codex Q-T5-3, not
  hidden as a discount).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

# --- Spec § 1.2 PP target gains (canonical source: design memo) ---
PP_TARGET_GAINS: dict[int, float] = {
    128: 0.24,
    512: 0.74,
    2048: 1.10,
    4096: 1.15,
    8192: 1.24,
    16384: 1.26,
}

LANE_A_PP_SET: set[int] = {128, 512, 2048}
LANE_B_PP_SET: set[int] = {4096, 8192, 16384}

P5I_TARGET_PP_SET: set[int] = {128, 512}
P5J_TARGET_PP_SET: set[int] = {2048, 4096, 8192, 16384}

# Estimated optimization gain percentages (per Codex Q-T5-4)
OP_LEVEL_REALISTIC_LOW = 0.30
OP_LEVEL_REALISTIC_HIGH = 0.50
KERNEL_REWRITE_REALISTIC_LOW = 0.50
KERNEL_REWRITE_REALISTIC_HIGH = 0.70

# Kernel-bound spans (per T0b H4 + T2.4 + T3.4 bindings — kernel rewrite is
# scope gate trigger per Codex Q-T5-3).
KERNEL_BOUND_SPANS: set[str] = {
    # GDN kernel-bound (T0b H4)
    "gda_step_7_kernel_and_cache_update",
    "gda_step_8_out_proj",
    "gda_step_8_norm_proj",  # T0a emitter name (norm_proj == out_proj per emitter rename)
    # GatedAttention kernel-bound (T2.4)
    "kv_mask_update",
    "fused_sdpa",
    # MoE kernel-bound (T3.4)
    "routing_sort_pack",
    "gather_qmm_gate_up",
    "gather_qmm_down",
    "routing_unsort_weighted_reduce",
}


class FeasibilityVerdict(str, Enum):
    YES = "yes"
    YES_WITH_SCOPE_GATE = "yes_with_scope_gate"
    NO_UNDER_MEASURED_CAP = "no_under_measured_cap"
    DATA_INSUFFICIENT = "data_insufficient"


@dataclass
class Candidate:
    span_name: str
    pp: int
    measured_exclusive_us: float
    root_inclusive_us: float
    max_gain_pct: float
    realistic_low_gain_pct: float
    realistic_high_gain_pct: float
    gap_weight: float
    score: float
    scope_gate_trigger: bool
    lane: str  # "A" (PP <= 2048 deep) or "B" (PP > 2048 top-level only)
    notes: str = ""


@dataclass
class PerPpAggregate:
    """Median per-PP per-span_name attribution, source for ROI candidates."""

    pp: int
    root_inclusive_us_median: float
    by_span: dict[str, float] = field(
        default_factory=dict
    )  # span_name -> median exclusive_us share
    by_span_exclusive_us: dict[str, float] = field(default_factory=dict)


# --- Helpers ---


def is_kernel_bound(span_name: str) -> bool:
    """Per T0b H4 + T2.4 + T3.4: kernel-bound spans trigger Scope gate."""
    return span_name in KERNEL_BOUND_SPANS


def compute_gap_weight(pp: int, current_gain: float = 0.0) -> float:
    """Urgency weight: how far this PP is from its target gain.

    Spec § 1.2 gap table sources targets. Returns 0 if pp is unknown or
    target is non-positive.
    """
    target = PP_TARGET_GAINS.get(pp, 0.0)
    if target <= 0:
        return 0.0
    return max(0.0, target - current_gain) / target


def _lane_for_pp(pp: int) -> str:
    if pp in LANE_A_PP_SET:
        return "A"
    if pp in LANE_B_PP_SET:
        return "B"
    return "?"


def _candidate_from_span(
    span_name: str,
    pp: int,
    measured_exclusive_us: float,
    root_inclusive_us: float,
    current_gain: float = 0.0,
) -> Candidate:
    """Build a Candidate row from one (PP, span_name) measurement.

    Scope gate trigger flag determines which realistic range applies but does
    NOT discount the score (per Codex Q-T5-3).
    """
    if root_inclusive_us <= 0:
        max_gain = 0.0
    else:
        max_gain = measured_exclusive_us / root_inclusive_us
    scope_gate = is_kernel_bound(span_name)
    if scope_gate:
        rlow = max_gain * KERNEL_REWRITE_REALISTIC_LOW
        rhigh = max_gain * KERNEL_REWRITE_REALISTIC_HIGH
    else:
        rlow = max_gain * OP_LEVEL_REALISTIC_LOW
        rhigh = max_gain * OP_LEVEL_REALISTIC_HIGH
    gap = compute_gap_weight(pp, current_gain=current_gain)
    score = max_gain * gap
    lane = _lane_for_pp(pp)
    notes_parts = []
    if scope_gate:
        notes_parts.append("scope_gate_trigger=kernel_rewrite")
    if lane == "B":
        notes_parts.append("lane_b_top_level_only")
    return Candidate(
        span_name=span_name,
        pp=pp,
        measured_exclusive_us=measured_exclusive_us,
        root_inclusive_us=root_inclusive_us,
        max_gain_pct=max_gain,
        realistic_low_gain_pct=rlow,
        realistic_high_gain_pct=rhigh,
        gap_weight=gap,
        score=score,
        scope_gate_trigger=scope_gate,
        lane=lane,
        notes="; ".join(notes_parts),
    )


# --- Attribution loading + per-PP aggregation ---


def load_attribution_csv(path: Path) -> list[dict]:
    """Read attribution CSV; returns list of dicts (one per row)."""
    with path.open() as f:
        return list(csv.DictReader(f))


def aggregate_per_pp(rows: list[dict]) -> dict[int, PerPpAggregate]:
    """Aggregate attribution rows to per-PP per-span median exclusive_us.

    For each (PP, span_name):
    * Collect exclusive_us across all requests at this PP.
    * Compute median exclusive_us; share = median / root_inclusive_us_median.

    Tree rows + synthesized residual rows both contribute (residuals carry
    exclusive_us == inclusive_us per aggregator emission). Diagnostic rows
    are EXCLUDED (no exclusive_us; not in exclusive tree per § 2.5a).
    """
    # First pass: group by (pp, request_id) → root inclusive (the unique
    # tree row with parent_span_id == "").
    root_us_by_req: dict[tuple[str, str], float] = {}
    for r in rows:
        if (
            r["span_kind"] == "tree"
            and r["parent_span_id"] == ""
            and r["span_name"] == "server_request_recv_to_first_content_sse_write"
        ):
            try:
                root_us_by_req[(r["pp"], r["request_id"])] = float(r["inclusive_us"])
            except (ValueError, TypeError):
                continue

    # Second pass: collect exclusive_us per (pp, span_name) across requests.
    excl_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["span_kind"] == "diagnostic":
            continue
        if r["span_kind"] not in ("tree", "synthesized"):
            continue
        if r["exclusive_us"] == "":
            continue
        try:
            ex = float(r["exclusive_us"])
        except (ValueError, TypeError):
            continue
        # Skip root itself — it's the denominator, not a candidate.
        if r["span_kind"] == "tree" and r["parent_span_id"] == "":
            continue
        excl_acc[(r["pp"], r["span_name"])].append(ex)

    # Compute per-PP root_inclusive_us_median + per-span medians.
    root_us_acc: dict[str, list[float]] = defaultdict(list)
    for (pp, _), v in root_us_by_req.items():
        root_us_acc[pp].append(v)

    out: dict[int, PerPpAggregate] = {}
    for pp_str, root_values in root_us_acc.items():
        try:
            pp_int = int(pp_str)
        except ValueError:
            continue
        root_median = statistics.median(root_values)
        agg = PerPpAggregate(pp=pp_int, root_inclusive_us_median=root_median)
        for (pp2, span_name), exs in excl_acc.items():
            if pp2 != pp_str or not exs:
                continue
            median_ex = statistics.median(exs)
            agg.by_span_exclusive_us[span_name] = median_ex
            agg.by_span[span_name] = median_ex / root_median if root_median > 0 else 0.0
        out[pp_int] = agg
    return out


# --- Rankings ---


def rank_top3_bottlenecks(
    per_pp: dict[int, PerPpAggregate],
) -> dict[int, list[Candidate]]:
    """Per-PP top-3 spans by exclusive_us / root.inclusive_us share."""
    out: dict[int, list[Candidate]] = {}
    for pp, agg in per_pp.items():
        cands = [
            _candidate_from_span(
                span_name=name,
                pp=pp,
                measured_exclusive_us=agg.by_span_exclusive_us[name],
                root_inclusive_us=agg.root_inclusive_us_median,
            )
            for name in agg.by_span_exclusive_us
        ]
        cands.sort(key=lambda c: c.max_gain_pct, reverse=True)
        out[pp] = cands[:3]
    return out


def _rank_for_pp_set(
    per_pp: dict[int, PerPpAggregate], pp_set: set[int]
) -> list[Candidate]:
    """Generate candidates for the given PP set, sorted by score desc."""
    cands: list[Candidate] = []
    for pp in pp_set:
        agg = per_pp.get(pp)
        if agg is None:
            continue
        for name, ex_us in agg.by_span_exclusive_us.items():
            cands.append(
                _candidate_from_span(
                    span_name=name,
                    pp=pp,
                    measured_exclusive_us=ex_us,
                    root_inclusive_us=agg.root_inclusive_us_median,
                )
            )
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def rank_p5i(per_pp: dict[int, PerPpAggregate]) -> list[Candidate]:
    """P5i candidates: span_name × pp combos for PP ∈ {128, 512}."""
    return _rank_for_pp_set(per_pp, P5I_TARGET_PP_SET)


def rank_p5j(per_pp: dict[int, PerPpAggregate]) -> list[Candidate]:
    """P5j candidates: span_name × pp combos for PP ∈ {2048, 4096, 8192, 16384}.

    PP=2048 measurements come from Lane A (full deep); PP=4096+ from Lane B
    (top-level only). The 'lane_b_top_level_only' caveat is set on Lane-B
    candidates by ``_candidate_from_span`` via the ``lane`` field + notes.
    """
    return _rank_for_pp_set(per_pp, P5J_TARGET_PP_SET)


def feasibility_verdict(
    p5i_candidates: list[Candidate],
    p5j_candidates: list[Candidate],
    pp: int,
) -> FeasibilityVerdict:
    """4-tier verdict per Codex Q-T5-4.

    Sums ``realistic_high_gain_pct`` across candidates relevant to this PP:
    * If op-only sum ≥ target → YES.
    * Else if op + kernel sum ≥ target → YES_WITH_SCOPE_GATE.
    * Else → NO_UNDER_MEASURED_CAP.
    * If no candidates relevant to this PP → DATA_INSUFFICIENT.
    """
    target = PP_TARGET_GAINS.get(pp, 0.0)
    relevant = [c for c in (p5i_candidates + p5j_candidates) if c.pp == pp]
    if not relevant:
        return FeasibilityVerdict.DATA_INSUFFICIENT
    if target <= 0:
        return FeasibilityVerdict.DATA_INSUFFICIENT

    sum_op_only = sum(
        c.realistic_high_gain_pct for c in relevant if not c.scope_gate_trigger
    )
    sum_with_kernel = sum(c.realistic_high_gain_pct for c in relevant)

    if sum_op_only >= target:
        return FeasibilityVerdict.YES
    if sum_with_kernel >= target:
        return FeasibilityVerdict.YES_WITH_SCOPE_GATE
    return FeasibilityVerdict.NO_UNDER_MEASURED_CAP


# --- CSV / JSON emission ---


def write_ranking_csv(
    top3: dict[int, list[Candidate]],
    p5i: list[Candidate],
    p5j: list[Candidate],
    out_path: Path,
) -> None:
    """One CSV with all candidates tagged by category."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "category",
                "rank_in_category",
                "pp",
                "lane",
                "span_name",
                "measured_exclusive_us",
                "root_inclusive_us",
                "max_gain_pct",
                "realistic_low_gain_pct",
                "realistic_high_gain_pct",
                "gap_weight",
                "score",
                "scope_gate_trigger",
                "notes",
            ]
        )
        for pp in sorted(top3):
            for i, c in enumerate(top3[pp], start=1):
                w.writerow(_candidate_row("top3", i, c))
        for i, c in enumerate(p5i, start=1):
            w.writerow(_candidate_row("p5i", i, c))
        for i, c in enumerate(p5j, start=1):
            w.writerow(_candidate_row("p5j", i, c))


def _candidate_row(category: str, rank: int, c: Candidate) -> list:
    return [
        category,
        rank,
        c.pp,
        c.lane,
        c.span_name,
        f"{c.measured_exclusive_us:.2f}",
        f"{c.root_inclusive_us:.2f}",
        f"{c.max_gain_pct:.4f}",
        f"{c.realistic_low_gain_pct:.4f}",
        f"{c.realistic_high_gain_pct:.4f}",
        f"{c.gap_weight:.4f}",
        f"{c.score:.4f}",
        c.scope_gate_trigger,
        c.notes,
    ]


def write_verdict_json(
    per_pp: dict[int, PerPpAggregate],
    p5i: list[Candidate],
    p5j: list[Candidate],
    out_path: Path,
) -> dict:
    """Per-PP feasibility verdict + top-3 summary for each measured PP."""
    out: dict[str, dict] = {}
    for pp in sorted(per_pp):
        verdict = feasibility_verdict(p5i, p5j, pp)
        relevant = [c for c in (p5i + p5j) if c.pp == pp]
        out[str(pp)] = {
            "target_gain_pct": PP_TARGET_GAINS.get(pp, 0.0),
            "lane": _lane_for_pp(pp),
            "verdict": verdict.value,
            "candidate_count": len(relevant),
            "sum_realistic_high_op_only": sum(
                c.realistic_high_gain_pct for c in relevant if not c.scope_gate_trigger
            ),
            "sum_realistic_high_with_kernel": sum(
                c.realistic_high_gain_pct for c in relevant
            ),
            "top_candidates": [
                {
                    "span_name": c.span_name,
                    "max_gain_pct": c.max_gain_pct,
                    "realistic_low_gain_pct": c.realistic_low_gain_pct,
                    "realistic_high_gain_pct": c.realistic_high_gain_pct,
                    "scope_gate_trigger": c.scope_gate_trigger,
                    "score": c.score,
                }
                for c in sorted(relevant, key=lambda x: x.score, reverse=True)[:5]
            ],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            out,
            f,
            indent=2,
            default=lambda o: (
                asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)
            ),
        )
    return out


# --- CLI ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--attribution-csv",
        required=True,
        type=Path,
        help="per-request per-span attribution table from aggregator --out",
    )
    ap.add_argument(
        "--summary-csv",
        required=True,
        type=Path,
        help="per-PP summary table from aggregator --summary-out (cross-check)",
    )
    ap.add_argument(
        "--out-ranking",
        required=True,
        type=Path,
        help="output ranked candidates CSV (top3 + p5i + p5j)",
    )
    ap.add_argument(
        "--out-verdict",
        required=True,
        type=Path,
        help="output per-PP feasibility verdict JSON",
    )
    args = ap.parse_args(argv)

    rows = load_attribution_csv(args.attribution_csv)
    if not rows:
        print("ERROR: attribution CSV is empty", file=sys.stderr)
        return 2

    per_pp = aggregate_per_pp(rows)
    if not per_pp:
        print("ERROR: no PPs aggregated from attribution rows", file=sys.stderr)
        return 3

    top3 = rank_top3_bottlenecks(per_pp)
    p5i = rank_p5i(per_pp)
    p5j = rank_p5j(per_pp)

    write_ranking_csv(top3, p5i, p5j, args.out_ranking)
    verdict = write_verdict_json(per_pp, p5i, p5j, args.out_verdict)

    print(f"OK: {len(per_pp)} PPs ranked", file=sys.stderr)
    for pp_str, info in verdict.items():
        print(
            f"  PP={pp_str} (Lane {info['lane']}): verdict={info['verdict']} "
            f"(target={info['target_gain_pct']:.2f}, "
            f"op_only_sum={info['sum_realistic_high_op_only']:.2f}, "
            f"with_kernel_sum={info['sum_realistic_high_with_kernel']:.2f})",
            file=sys.stderr,
        )
    print(
        f"Written ranking to {args.out_ranking}, verdict to {args.out_verdict}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
