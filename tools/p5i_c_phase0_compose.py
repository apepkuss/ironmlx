"""P5i.c Phase 0 — final ranking JSON composer + docs generator.

Per plan Step 5.4 + Task 7 Step 7.1/7.2: loads multi-repeat substep JSON +
production root medians + ironmlx/omlx envelope JSON + T0 audit JSON; computes
production_share_pct ranking + tied tiers + 4-category coverage + Phase 1
default rule + Dense diagnostic trigger; writes ranking.json and (optionally)
human-readable snapshot/close-out/memory markdown files.

CLI (T3 invocation):
    python tools/p5i_c_phase0_compose.py \\
        --audit-json reports/p5i-c-phase-0-audit.json \\
        --pp128-multirepeat /tmp/p5i-c-phase-0-pp128-multirepeat.json \\
        --pp512-multirepeat /tmp/p5i-c-phase-0-pp512-multirepeat.json \\
        --pp128-envelope /tmp/p5i-c-phase-0-pp128-ironmlx-vs-omlx-envelope.json \\
        --pp512-envelope /tmp/p5i-c-phase-0-pp512-ironmlx-vs-omlx-envelope.json \\
        --out-json /tmp/p5i-c-phase-0-ranking.json \\
        --summary-md reports/p5i-c-phase-0-ranking-summary.md

CLI (T4 re-compose with Dense data):
    same as T3 + --dense-pp128-json + --dense-pp512-json

CLI (T5 docs generation):
    same as T3 + --close-out-md docs/p5i-c-phase-0-close-out.md \\
                + --memory-md /path/to/memory/project_p5i_c_phase_0_findings.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5h_aggregator.roi_ranking import (  # noqa: E402
    emit_category_coverage,
    emit_phase_1_default_rule,
    evaluate_dense_diagnostic_trigger,
    identify_tied_tiers,
)

MIN_SHARE_PCT = 1.0  # candidates below 1% omitted from ranking output


def _rank_per_pp(
    multirepeat: dict, production_root_us: float
) -> tuple[list[tuple[str, float]], dict[str, tuple[float, float]], dict[str, float]]:
    """Convert multirepeat per_substep JSON to ranking + ci95 + medians.

    production_share_pct = probe_exclusive_us_median / production_root_us * 100
    where probe_exclusive_us_median = probe_share (median_pct/100) * probe_root_us

    The aggregator emits per-substep median_pct as probe-share-of-probe-root in
    percent. We rescale to production-root denominator via:
        production_share_pct = probe_share_pct * (probe_root_us / production_root_us)
    But since we don't have probe_root_us per substep here (the aggregator only
    gives us shares against probe root), we report probe_share_pct directly as
    the ranking score AND report a separate `denominator_ratio` field at
    compose time. Per spec § 6.5: substep shares are accurate at probe
    denominator; production_root is the FEASIBILITY denominator only.

    For Phase 0 ranking decision (which span dominates), probe_share is the
    right comparison. production_root_us shows up in the ranking output as
    metadata so Phase 1 acceptance can rescale.
    """
    per_substep = multirepeat["per_substep"]
    ranking = sorted(
        ((name, s["median_pct"]) for name, s in per_substep.items()),
        key=lambda x: -x[1],
    )
    ranking = [(n, m) for n, m in ranking if m >= MIN_SHARE_PCT]
    ci95: dict[str, tuple[float, float]] = {
        n: (per_substep[n]["ci95_low_pct"], per_substep[n]["ci95_high_pct"])
        for n, _ in ranking
    }
    medians: dict[str, float] = {n: m for n, m in ranking}
    return ranking, ci95, medians


def _load_audit(audit_json: Path) -> dict[str, str]:
    data = json.loads(audit_json.read_text())
    return {cat: row["status"] for cat, row in data["categories"].items()}


def compose(
    audit_json: Path,
    multirepeat_paths: dict[int, Path],
    envelope_paths: dict[int, Path],
    dense_paths: dict[int, Path] | None = None,
) -> dict:
    audit = _load_audit(audit_json)
    ranking_per_pp: dict[int, list[tuple[str, float]]] = {}
    tiers_per_pp: dict[int, list[list[str]]] = {}
    ci95_per_pp: dict[int, dict[str, tuple[float, float]]] = {}
    medians_per_pp: dict[int, dict[str, float]] = {}
    production_root_per_pp: dict[int, float | None] = {}
    measured_spans: set[str] = set()

    for pp, mr_path in multirepeat_paths.items():
        mr = json.loads(mr_path.read_text())
        prod_root = (
            mr.get("production_root", {}).get("production_root_us_median")
            if mr.get("production_root")
            else None
        )
        production_root_per_pp[pp] = prod_root
        ranking, ci95, medians = _rank_per_pp(mr, prod_root or 0.0)
        ranking_per_pp[pp] = ranking
        ci95_per_pp[pp] = ci95
        medians_per_pp[pp] = medians
        tiers_per_pp[pp] = identify_tied_tiers(ranking, ci95)
        for name, _ in ranking:
            measured_spans.add(name)

    if any(v is None for v in production_root_per_pp.values()):
        missing = [pp for pp, v in production_root_per_pp.items() if v is None]
        return {
            "verdict": "data_insufficient_for_production_share",
            "missing_production_root_pps": missing,
            "ranking_per_pp": {
                str(pp): [{"span_name": n, "probe_share_pct": m} for n, m in r]
                for pp, r in ranking_per_pp.items()
            },
        }

    coverage = emit_category_coverage(audit, measured_spans)
    phase_1_rule = emit_phase_1_default_rule(ranking_per_pp, tiers_per_pp, coverage)
    dense_eval = evaluate_dense_diagnostic_trigger(tiers_per_pp, medians_per_pp)

    envelopes_per_pp: dict[str, dict] = {}
    for pp, env_path in envelope_paths.items():
        envelopes_per_pp[str(pp)] = json.loads(env_path.read_text())

    out: dict = {
        "ranking_per_pp": {
            str(pp): [
                {
                    "span_name": n,
                    "probe_share_pct": m,
                    "ci95_low_pct": ci95_per_pp[pp].get(n, (0.0, 0.0))[0],
                    "ci95_high_pct": ci95_per_pp[pp].get(n, (0.0, 0.0))[1],
                }
                for n, m in r
            ]
            for pp, r in ranking_per_pp.items()
        },
        "tiers_per_pp": {str(pp): t for pp, t in tiers_per_pp.items()},
        "ci95_per_pp": {
            str(pp): {n: list(ci) for n, ci in v.items()}
            for pp, v in ci95_per_pp.items()
        },
        "production_root_us_per_pp": {
            str(pp): v for pp, v in production_root_per_pp.items()
        },
        "category_coverage": coverage,
        "phase_1_default_rule": phase_1_rule,
        "dense_diagnostic_triggered": dense_eval["triggered"],
        "dense_diagnostic_reason": dense_eval["reason"],
        "envelopes_per_pp": envelopes_per_pp,
    }

    if dense_paths:
        out["dense_diagnostic"] = {}
        for pp, dp in dense_paths.items():
            out["dense_diagnostic"][str(pp)] = json.loads(dp.read_text())

    return out


def _render_summary_md(data: dict, audit_json: Path) -> str:
    lines: list[str] = [
        "# P5i.c Phase 0 Ranking Snapshot",
        "",
        f"**Date:** {date.today().isoformat()}",
        "",
        f"**Audit ref:** `{audit_json}` (gitignored; T0 output)",
        "",
        "## Phase 1 default rule",
        "",
        f"**Triggered:** `{data['phase_1_default_rule']['triggered_rule']}`",
        "",
        f"**Suggested candidates:** {data['phase_1_default_rule']['suggested_phase_1_candidates']}",
        "",
        f"**Rationale:** {data['phase_1_default_rule']['rationale']}",
        "",
        "## 4-category coverage status",
        "",
        "| Category | Status |",
        "|---|---|",
    ]
    for cat in ("scheduler", "kv_cache", "attention", "moe"):
        lines.append(f"| {cat} | `{data['category_coverage'].get(cat, 'unknown')}` |")
    lines.extend(["", "## Per-PP top-N ranking with CI95 + tier", ""])
    for pp_str in sorted(data["ranking_per_pp"].keys(), key=int):
        lines.extend(
            [
                f"### PP={pp_str}",
                "",
                "| Tier | Candidates (probe-share + CI95 half-width) |",
                "|---|---|",
            ]
        )
        tiers = data["tiers_per_pp"][pp_str]
        ranking = {row["span_name"]: row for row in data["ranking_per_pp"][pp_str]}
        for ti, tier in enumerate(tiers, start=1):
            members = []
            for name in tier:
                row = ranking.get(name)
                if row is None:
                    members.append(name)
                    continue
                hw = (row["ci95_high_pct"] - row["ci95_low_pct"]) / 2
                members.append(f"{name} ({row['probe_share_pct']:.2f}%, ±{hw:.2f}%)")
            lines.append(f"| tier-{ti} | {', '.join(members)} |")
        lines.append("")
    lines.extend(["## Dense diagnostic", ""])
    if data["dense_diagnostic_triggered"]:
        lines.append(f"**Triggered:** YES — {data['dense_diagnostic_reason']}")
    else:
        lines.append(f"**Skipped:** {data['dense_diagnostic_reason']}")
    lines.append("")
    lines.extend(
        [
            "## vs-omlx delta (P5h+2.a scope ii baseline)",
            "",
            "| PP | ironmlx_median | omlx_median | delta_pct | ironmlx_envelope | omlx_envelope |",
            "|---|---|---|---|---|---|",
        ]
    )
    for pp_str in sorted(data["envelopes_per_pp"].keys(), key=int):
        env = data["envelopes_per_pp"][pp_str]
        iron = env.get("ironmlx", {})
        comp = env.get("comparator", {})
        delta = env.get("delta_vs_comparator", {})
        lines.append(
            f"| {pp_str} | {iron.get('mean_median', 0):.2f} | "
            f"{comp.get('mean_median', 0):.2f} | "
            f"{delta.get('delta_pct_median', 0):+.2f}% | "
            f"±{iron.get('final_uncertainty_envelope_pct', 0):.2f}% | "
            f"±{comp.get('final_uncertainty_envelope_pct', 0):.2f}% |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_close_out_md(data: dict, summary_md_text: str) -> str:
    return (
        "# P5i.c Phase 0 — Close-out\n\n"
        "**Status:** Phase 0 closed.\n\n"
        f"**Phase 1 default rule:** `{data['phase_1_default_rule']['triggered_rule']}` — "
        f"{data['phase_1_default_rule']['rationale']}\n\n"
        "## Ranking snapshot summary\n\n" + summary_md_text + "\n## Next phase\n\n"
        "Boss reviews ranking snapshot → brainstorms Phase 1 form via "
        "spec § 9 default rule starting point.\n"
    )


def _render_memory_md(data: dict) -> str:
    pps = sorted(data["ranking_per_pp"].keys(), key=int)
    tier1_summary = "; ".join(
        f"PP={p} tier-1={data['tiers_per_pp'][p][0] if data['tiers_per_pp'][p] else []}"
        for p in pps
    )
    return (
        "---\n"
        "name: project-p5i-c-phase-0-findings\n"
        f"description: P5i.c Phase 0 closed {date.today().isoformat()}; "
        f"Phase 1 default rule {data['phase_1_default_rule']['triggered_rule']}; "
        f"suggested candidates {data['phase_1_default_rule']['suggested_phase_1_candidates']}\n"
        "metadata:\n"
        "  type: project\n"
        "---\n\n"
        f"P5i.c Phase 0 closed {date.today().isoformat()} (commit pending).\n\n"
        f"**Tier-1 per PP:** {tier1_summary}\n\n"
        f"**Phase 1 default rule:** {data['phase_1_default_rule']['triggered_rule']} — "
        f"{data['phase_1_default_rule']['rationale']}\n\n"
        f"**Category coverage:** {data['category_coverage']}\n\n"
        f"**Dense diagnostic:** "
        f"{'triggered' if data['dense_diagnostic_triggered'] else 'skipped'} — "
        f"{data['dense_diagnostic_reason']}\n\n"
        "**Reusable infra:**\n"
        "- `ironmlx/tests/p5i_c_phase_0_capture.rs` — env-var driven dual-mode capture harness\n"
        "- `tools/p5h_aggregator/multi_repeat.py` — per-substep CI + production root extraction\n"
        "- `tools/p5i_c_pp_tps_envelope.py` — per-PP envelope MAX(within, between)\n"
        "- `tools/p5i_c_phase0_compose.py` — ranking JSON + docs generator\n"
        "- `tools/p5h_aggregator/roi_ranking.py` extensions (tied tier, coverage, R1/R2/R3, dense trigger)\n\n"
        "Links: [[project-p5h-findings]] (P5h+1 ranking predecessor); "
        "[[project-p5h-2a-findings]] (RUNS=15 protocol used here); "
        "[[project-p5i-a-findings]] (T1+T2 LANDED state measured).\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--pp128-multirepeat", type=Path, required=True)
    p.add_argument("--pp512-multirepeat", type=Path, required=True)
    p.add_argument("--pp128-envelope", type=Path, required=True)
    p.add_argument("--pp512-envelope", type=Path, required=True)
    p.add_argument("--dense-pp128-json", type=Path, default=None)
    p.add_argument("--dense-pp512-json", type=Path, default=None)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--summary-md", type=Path, default=None)
    p.add_argument("--close-out-md", type=Path, default=None)
    p.add_argument("--memory-md", type=Path, default=None)
    args = p.parse_args()

    multirepeat_paths = {128: args.pp128_multirepeat, 512: args.pp512_multirepeat}
    envelope_paths = {128: args.pp128_envelope, 512: args.pp512_envelope}
    dense_paths = None
    if args.dense_pp128_json and args.dense_pp512_json:
        dense_paths = {128: args.dense_pp128_json, 512: args.dense_pp512_json}

    data = compose(args.audit_json, multirepeat_paths, envelope_paths, dense_paths)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.out_json}")

    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text(_render_summary_md(data, args.audit_json))
        print(f"Wrote {args.summary_md}")

    if args.close_out_md:
        summary_text = _render_summary_md(data, args.audit_json)
        args.close_out_md.parent.mkdir(parents=True, exist_ok=True)
        args.close_out_md.write_text(_render_close_out_md(data, summary_text))
        print(f"Wrote {args.close_out_md}")

    if args.memory_md:
        args.memory_md.parent.mkdir(parents=True, exist_ok=True)
        args.memory_md.write_text(_render_memory_md(data))
        print(f"Wrote {args.memory_md}")


if __name__ == "__main__":
    main()
