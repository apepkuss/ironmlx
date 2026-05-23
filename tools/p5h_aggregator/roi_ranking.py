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

# Spec § 1.2 target-gain partition (used for P5i/P5j classification only — i.e.
# which PP belongs to the short-PP vs long-PP optimization phase). NOT used to
# label OBSERVED routing lane: at runtime, chat-template overhead can push
# prompt_tokens > prefill_chunk_size and route a spec-Lane-A PP through Lane B
# (gs_chunked). Use ``observed_lane_for_pp`` for measurement-granularity lane.
SPEC_LANE_A_PP_SET: set[int] = {128, 512, 2048}
SPEC_LANE_B_PP_SET: set[int] = {4096, 8192, 16384}

# Backward-compat aliases (kept for any importer; semantics = SPEC partition).
LANE_A_PP_SET: set[int] = SPEC_LANE_A_PP_SET
LANE_B_PP_SET: set[int] = SPEC_LANE_B_PP_SET

P5I_TARGET_PP_SET: set[int] = {128, 512}
P5J_TARGET_PP_SET: set[int] = {2048, 4096, 8192, 16384}

# Fix D threshold: wrapper spans dominating > 50% of root_inclusive flag a PP
# as data_insufficient (the dominant cost lives inside an un-instrumented
# wrapper — ranking can't progress until deeper attribution lands).
WRAPPER_DOMINANCE_THRESHOLD: float = 0.50

# Spans that are wrappers around un-instrumented compute (per spec § 3 T0a +
# T0b H4): when these dominate a PP, verdict = data_insufficient with a pointer
# at the P5h+1 follow-up that needs to land the deeper attribution.
LANE_B_WRAPPER_SPAN: str = "gs_chunk_N"
LANE_A_WRAPPER_SPAN: str = "first_token_sampling"

# Estimated optimization gain percentages (per Codex Q-T5-4)
OP_LEVEL_REALISTIC_LOW = 0.30
OP_LEVEL_REALISTIC_HIGH = 0.50
KERNEL_REWRITE_REALISTIC_LOW = 0.50
KERNEL_REWRITE_REALISTIC_HIGH = 0.70

# Kernel-bound spans (per T0b H4 + T2.4 + T3.4 bindings — kernel rewrite is
# scope gate trigger per Codex Q-T5-3).
KERNEL_BOUND_SPANS: set[str] = {
    # GDN kernel-bound (T0b H4).
    # P5h+1 T1.5 (Codex B-lite): `gda_step_7_kernel_and_cache_update` is
    # NOT in this set — after the T1.5 sub-span split it is a thin wrapper
    # (parent of `gda_step_7_kernel_dispatch_and_materialize` +
    # `cache_state_update`) and the kernel work it formerly owned now lives
    # in the new sub-span. The wrapper itself is not a kernel-rewrite
    # target; flagging it would mis-trigger Scope gate at the aggregator.
    "gda_step_7_kernel_dispatch_and_materialize",
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
    # Lane reflects OBSERVED routing (from routing_path field) per Fix B —
    # not the spec target-gain partition. Values: "A" (scheduler), "B"
    # (gs_chunked), "mixed", or "?" if unknown.
    lane: str
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


def _spec_lane_for_pp(pp: int) -> str:
    """Spec § 1.2 target-gain partition lane (for cross-check only)."""
    if pp in SPEC_LANE_A_PP_SET:
        return "A"
    if pp in SPEC_LANE_B_PP_SET:
        return "B"
    return "?"


def observed_lane_for_pp(rows: list[dict]) -> dict[int, str]:
    """Per Fix B (Codex T5-R review): derive OBSERVED lane per PP by reading
    ``routing_path`` from attribution rows.

    Returns ``{pp: lane}`` where lane is:
    * ``"A"`` — all rows for this PP have ``routing_path == "scheduler"``.
    * ``"B"`` — all rows for this PP have ``routing_path == "gs_chunked"``.
    * ``"mixed"`` — some of each (e.g. straddling prefill_chunk_size boundary).
    * ``"?"`` — no usable routing_path observed (shouldn't happen in practice).

    A PP that maps to a different observed lane than its spec partition
    (``SPEC_LANE_A_PP_SET`` / ``SPEC_LANE_B_PP_SET``) is reported via stderr by
    the caller (e.g. chat-template overhead at PP=2048 pushing prompt to 2060
    → gs_chunked → observed Lane B even though spec says Lane A).
    """
    by_pp: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        pp_str = r.get("pp", "")
        try:
            pp_int = int(pp_str)
        except ValueError:
            continue
        rp = r.get("routing_path", "").strip()
        if rp:
            by_pp[pp_int].add(rp)
    out: dict[int, str] = {}
    for pp, paths in by_pp.items():
        if paths == {"scheduler"}:
            out[pp] = "A"
        elif paths == {"gs_chunked"}:
            out[pp] = "B"
        elif paths:
            out[pp] = "mixed"
        else:
            out[pp] = "?"
    return out


def warn_lane_divergence(observed: dict[int, str], stream=sys.stderr) -> None:
    """Emit a stderr warning per PP where observed lane != spec partition."""
    for pp, obs in sorted(observed.items()):
        spec = _spec_lane_for_pp(pp)
        if spec == "?" or obs in ("?", "mixed"):
            continue
        if spec != obs:
            print(
                f"WARN: PP={pp} observed_lane={obs} but spec partition lane={spec} "
                f"(likely chat-template overhead pushing prompt across "
                f"prefill_chunk_size boundary)",
                file=stream,
            )


def _candidate_from_span(
    span_name: str,
    pp: int,
    measured_exclusive_us: float,
    root_inclusive_us: float,
    lane: str,
    current_gain: float = 0.0,
) -> Candidate:
    """Build a Candidate row from one (PP, span_name) measurement.

    Scope gate trigger flag determines which realistic range applies but does
    NOT discount the score (per Codex Q-T5-3). ``lane`` reflects OBSERVED
    routing per Fix B (caller is responsible for passing observed lane).
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

    Per Fix A (Codex T5-R review): the correct per-PP per-span median is over
    the PER-REQUEST TOTAL exclusive_us (summed across all records sharing the
    same span_name within one request), NOT over individual records. Multi-emit
    spans (gs_chunk_N, decoder_layer_N, GDN/MoE substeps) otherwise report
    per-record median which underrepresents per-request cost by N (e.g.
    PP=8192 gs_chunk_N emits ~5/req — median over 7×5 records ≈ 1/5 the truth).

    Per Fix C (Codex T5-R review): ``unattributed_<span_name>`` synthesized
    residual rows are EXCLUDED from the ROI candidate pool. They represent
    unattributed compute (no actionable optimization target) and double-count
    against the parent's exclusive_us (which already accounts for that
    residual; see compute_exclusive: ``exclusive = inclusive - Σ child
    inclusive == residual_under_parent``). Synthesized rows remain in the
    attribution CSV + summary CSV for diagnostic completeness.

    Tree rows contribute exclusive_us. Diagnostic + synthesized rows are
    EXCLUDED.
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

    # Second pass: sum exclusive_us per (pp, request_id, span_name) — Fix A:
    # per-request total, NOT per-record value.
    per_req_totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for r in rows:
        # Fix C: exclude diagnostic + synthesized rows from the ROI candidate
        # pool (synthesized = unattributed residual; not actionable + double
        # counts parent's exclusive_us).
        if r["span_kind"] != "tree":
            continue
        # Skip root itself — it's the denominator, not a candidate.
        if r["parent_span_id"] == "":
            continue
        if r["exclusive_us"] == "":
            continue
        try:
            ex = float(r["exclusive_us"])
        except (ValueError, TypeError):
            continue
        per_req_totals[(r["pp"], r["request_id"], r["span_name"])] += ex

    # Re-shape: group per-request totals by (pp, span_name) for median.
    excl_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (pp, _rid, span_name), total in per_req_totals.items():
        excl_acc[(pp, span_name)].append(total)

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
        for (pp2, span_name), totals in excl_acc.items():
            if pp2 != pp_str or not totals:
                continue
            median_ex = statistics.median(totals)
            agg.by_span_exclusive_us[span_name] = median_ex
            agg.by_span[span_name] = median_ex / root_median if root_median > 0 else 0.0
        out[pp_int] = agg
    return out


# --- Rankings ---


def rank_top3_bottlenecks(
    per_pp: dict[int, PerPpAggregate],
    observed_lane: dict[int, str] | None = None,
) -> dict[int, list[Candidate]]:
    """Per-PP top-3 spans by exclusive_us / root.inclusive_us share.

    ``observed_lane`` per Fix B: lane labels on each Candidate reflect OBSERVED
    routing_path (not spec partition). Falls back to "?" when not supplied.
    """
    out: dict[int, list[Candidate]] = {}
    obs = observed_lane or {}
    for pp, agg in per_pp.items():
        lane = obs.get(pp, "?")
        cands = [
            _candidate_from_span(
                span_name=name,
                pp=pp,
                measured_exclusive_us=agg.by_span_exclusive_us[name],
                root_inclusive_us=agg.root_inclusive_us_median,
                lane=lane,
            )
            for name in agg.by_span_exclusive_us
        ]
        cands.sort(key=lambda c: c.max_gain_pct, reverse=True)
        out[pp] = cands[:3]
    return out


def _rank_for_pp_set(
    per_pp: dict[int, PerPpAggregate],
    pp_set: set[int],
    observed_lane: dict[int, str] | None = None,
) -> list[Candidate]:
    """Generate candidates for the given PP set, sorted by score desc."""
    cands: list[Candidate] = []
    obs = observed_lane or {}
    for pp in pp_set:
        agg = per_pp.get(pp)
        if agg is None:
            continue
        lane = obs.get(pp, "?")
        for name, ex_us in agg.by_span_exclusive_us.items():
            cands.append(
                _candidate_from_span(
                    span_name=name,
                    pp=pp,
                    measured_exclusive_us=ex_us,
                    root_inclusive_us=agg.root_inclusive_us_median,
                    lane=lane,
                )
            )
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def rank_p5i(
    per_pp: dict[int, PerPpAggregate],
    observed_lane: dict[int, str] | None = None,
) -> list[Candidate]:
    """P5i candidates: span_name × pp combos for PP ∈ {128, 512}."""
    return _rank_for_pp_set(per_pp, P5I_TARGET_PP_SET, observed_lane=observed_lane)


def rank_p5j(
    per_pp: dict[int, PerPpAggregate],
    observed_lane: dict[int, str] | None = None,
) -> list[Candidate]:
    """P5j candidates: span_name × pp combos for PP ∈ {2048, 4096, 8192, 16384}.

    PP=2048 spec partition is Lane A, but observed routing may be Lane B (chat
    template overhead). The ``lane`` field + 'lane_b_top_level_only' note on
    each Candidate reflects OBSERVED routing per Fix B.
    """
    return _rank_for_pp_set(per_pp, P5J_TARGET_PP_SET, observed_lane=observed_lane)


def wrapper_dominated_verdict_explanation(
    pp: int,
    lane: str,
    per_pp_agg: PerPpAggregate | None,
) -> str | None:
    """Per Fix D (Codex T5-R review): when an un-instrumented wrapper span
    dominates a PP's root_inclusive (> WRAPPER_DOMINANCE_THRESHOLD), the real
    cost lives inside the wrapper and no ranking is actionable until deeper
    attribution lands. Returns an explanation string suitable for the verdict
    JSON, or None if no wrapper dominance is detected.

    Lane B (``gs_chunked``): ``gs_chunk_N`` is the wrapper (deep substeps
    deferred per spec § 3 T0a).
    Lane A (``scheduler``): ``first_token_sampling`` is the MLX lazy
    materialization wrapper (not a true leaf; deferred to P5h+1).
    """
    if per_pp_agg is None or per_pp_agg.root_inclusive_us_median <= 0:
        return None
    if lane == "B":
        wrapper = LANE_B_WRAPPER_SPAN
        followup = (
            f"{wrapper} wrapper dominates "
            f"({{share:.2%}} of root_inclusive); deep substep instrumentation "
            "deferred per spec § 3 T0a; ranking requires Lane B deep "
            "instrumentation P5h+1 follow-up"
        )
    elif lane == "A":
        wrapper = LANE_A_WRAPPER_SPAN
        followup = (
            f"{wrapper} is MLX lazy materialization wrapper "
            f"({{share:.2%}} of root_inclusive); ranking requires P5h+1 Lane A "
            "lazy materialization boundary attribution"
        )
    else:
        return None
    share = per_pp_agg.by_span.get(wrapper, 0.0)
    if share <= WRAPPER_DOMINANCE_THRESHOLD:
        return None
    return followup.format(share=share)


def feasibility_verdict(
    p5i_candidates: list[Candidate],
    p5j_candidates: list[Candidate],
    pp: int,
    per_pp_agg: PerPpAggregate | None = None,
    observed_lane: str | None = None,
) -> FeasibilityVerdict:
    """4-tier verdict per Codex Q-T5-4 + Fix D wrapper-dominance refinement.

    Sums ``realistic_high_gain_pct`` across candidates relevant to this PP:
    * If a wrapper span (gs_chunk_N for Lane B, first_token_sampling for Lane
      A) dominates > 50% of root → DATA_INSUFFICIENT (Fix D: the dominant cost
      lives inside an un-instrumented wrapper; ranking can't progress).
    * Else if op-only sum ≥ target → YES.
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

    # Fix D: wrapper-dominance gate — short-circuits to DATA_INSUFFICIENT.
    lane = observed_lane if observed_lane is not None else "?"
    if wrapper_dominated_verdict_explanation(pp, lane, per_pp_agg) is not None:
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
    observed_lane: dict[int, str] | None = None,
) -> dict:
    """Per-PP feasibility verdict + top-3 summary for each measured PP.

    Per Fix B: ``observed_lane`` (from ``observed_lane_for_pp``) supplies the
    measurement-granularity lane label and the spec partition is reported as
    ``spec_lane`` for cross-check.
    Per Fix D: when a wrapper span dominates a PP's root_inclusive,
    ``verdict_explanation`` is populated pointing at the P5h+1 follow-up.
    """
    obs = observed_lane or {}
    out: dict[str, dict] = {}
    for pp in sorted(per_pp):
        agg = per_pp[pp]
        observed = obs.get(pp, "?")
        verdict = feasibility_verdict(
            p5i, p5j, pp, per_pp_agg=agg, observed_lane=observed
        )
        relevant = [c for c in (p5i + p5j) if c.pp == pp]
        explanation = wrapper_dominated_verdict_explanation(pp, observed, agg)
        entry: dict = {
            "target_gain_pct": PP_TARGET_GAINS.get(pp, 0.0),
            "spec_lane": _spec_lane_for_pp(pp),
            "observed_lane": observed,
            "lane": observed,  # backward-compat alias; observed is canonical
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
        if explanation is not None:
            entry["verdict_explanation"] = explanation
        out[str(pp)] = entry
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

    observed = observed_lane_for_pp(rows)
    warn_lane_divergence(observed)

    top3 = rank_top3_bottlenecks(per_pp, observed_lane=observed)
    p5i = rank_p5i(per_pp, observed_lane=observed)
    p5j = rank_p5j(per_pp, observed_lane=observed)

    write_ranking_csv(top3, p5i, p5j, args.out_ranking)
    verdict = write_verdict_json(
        per_pp, p5i, p5j, args.out_verdict, observed_lane=observed
    )

    print(f"OK: {len(per_pp)} PPs ranked", file=sys.stderr)
    for pp_str, info in verdict.items():
        spec_lane = info.get("spec_lane", "?")
        obs_lane = info.get("observed_lane", "?")
        lane_str = (
            obs_lane if obs_lane == spec_lane else f"{obs_lane} (spec {spec_lane})"
        )
        line = (
            f"  PP={pp_str} (Lane {lane_str}): verdict={info['verdict']} "
            f"(target={info['target_gain_pct']:.2f}, "
            f"op_only_sum={info['sum_realistic_high_op_only']:.2f}, "
            f"with_kernel_sum={info['sum_realistic_high_with_kernel']:.2f})"
        )
        if "verdict_explanation" in info:
            line += f"\n    explanation: {info['verdict_explanation']}"
        print(line, file=sys.stderr)
    print(
        f"Written ranking to {args.out_ranking}, verdict to {args.out_verdict}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
