"""P5h+2.d Stage 1 thermal cooldown orchestrator + Mechanism gate analyzer.

Per spec § 1.1 + § 2 + § 2.4.

CLI:
    python tools/p5h_2d_thermal_experiment.py sweep \\
        --cooldown-levels 0,60,120 \\
        --pps 128,512 \\
        --repeats 3 \\
        --runs-per-pp '128:15,512:15' \\
        --model-dir $SNAP \\
        --mlx-dir $HOME/.local/mlx \\
        --out-base /tmp/p5h+2-d-stage1

    python tools/p5h_2d_thermal_experiment.py gate \\
        --cooldown-levels 0,60,120 \\
        --pps 128,512 \\
        --envelope-glob '/tmp/p5h+2-d-stage1-cd{cooldown}-pp{pp}-envelope.json' \\
        --out-json /tmp/p5h+2-d-stage1-mechanism-gate.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
PROTOCOL_DRIVER = TOOLS_DIR / "p5h_2b_protocol_experiment.py"

REDUCTION_THRESHOLD_PCT = 50.0
RESIDUAL_MAX_PCT = 10.0
BASELINE_CLEAN_THRESHOLD_PCT = 10.0


def _abs_trailing(per_repeat: list[dict]) -> float:
    """Median across repeats of |trailing_slowdown_pct| clipped to non-negative."""
    vals = [
        max(0.0, -r["trailing_slowdown_pct"])
        for r in per_repeat
        if r["trailing_slowdown_pct"] is not None
    ]
    return median(vals) if vals else 0.0


def _pos_fast_start(per_repeat: list[dict]) -> float:
    """Median across repeats of fast_start_drop_pct clipped to non-negative."""
    vals = [
        max(0.0, r["fast_start_drop_pct"])
        for r in per_repeat
        if r["fast_start_drop_pct"] is not None
    ]
    return median(vals) if vals else 0.0


def _pp_residual(per_repeat: list[dict], pp: str) -> float:
    """Per-PP dominant residual: PP=128 -> trailing; PP=512 -> fast-start."""
    if pp == "128":
        return _abs_trailing(per_repeat)
    if pp == "512":
        return _pos_fast_start(per_repeat)
    raise ValueError(f"unknown PP={pp}")


def _pick_best_cooldown(matrix: dict, pp: str) -> tuple[str, float]:
    """Return (best_cooldown_str, residual_at_best) among non-baseline cooldowns.

    `0s` is WORST/baseline for control comparisons, not a valid BEST
    mechanism candidate.
    """
    candidates = []
    for cooldown in matrix:
        if cooldown == "0s":
            continue
        residual = _pp_residual(matrix[cooldown][pp], pp)
        candidates.append((cooldown, residual))
    if not candidates:
        raise SystemExit("matrix has no non-baseline cooldown candidates")
    # Sort by residual ASC, then cooldown integer ASC (tie-breaker).
    candidates.sort(key=lambda x: (x[1], int(x[0].rstrip("s"))))
    return candidates[0]


def compute_mechanism_gate(matrix: dict) -> dict:
    """Mechanism gate logic per spec § 2.4.

    matrix shape: {cooldown_str: {pp_str: per_repeat_list_of_diagnostic_dicts}}
    Returns: {"verdict": "strong_yes"|"weak_yes"|"no", "best_cooldown_per_pp": {...},
              "reason": str, "details": {...}}
    """
    pps = ["128", "512"]
    if not all("0s" in matrix and pp in matrix["0s"] for pp in pps):
        raise SystemExit("matrix missing 0s baseline for one or both PPs")
    baseline_residuals = {pp: _pp_residual(matrix["0s"][pp], pp) for pp in pps}

    # Spec § 2.4 last clause: if 0s baseline already clean -> mechanism not demonstrated.
    if all(baseline_residuals[pp] <= BASELINE_CLEAN_THRESHOLD_PCT for pp in pps):
        return {
            "verdict": "no",
            "best_cooldown_per_pp": {pp: "0s" for pp in pps},
            "reason": "baseline_already_clean: 0s residual <= 10% for both PPs",
            "details": {"baseline_residuals": baseline_residuals},
        }

    # Per PP: BEST cooldown + reduction vs baseline.
    best_per_pp: dict[str, str] = {}
    pp_full_pass: dict[str, bool] = {}
    pp_reduced_by_50: dict[str, bool] = {}
    pp_details: dict[str, dict] = {}
    for pp in pps:
        best_cd, best_res = _pick_best_cooldown(matrix, pp)
        best_per_pp[pp] = best_cd
        base = baseline_residuals[pp]
        reduction_pct = (
            (base - best_res) / base * 100.0 if base > 0 else 0.0
        )
        reduced_by_50 = reduction_pct >= REDUCTION_THRESHOLD_PCT
        residual_ok = best_res <= RESIDUAL_MAX_PCT
        pp_reduced_by_50[pp] = reduced_by_50
        pp_full_pass[pp] = reduced_by_50 and residual_ok
        pp_details[pp] = {
            "baseline_residual_pct": base,
            "best_cooldown": best_cd,
            "best_residual_pct": best_res,
            "reduction_pct": reduction_pct,
            "reduced_by_50pct": reduced_by_50,
            "residual_le_10pct": residual_ok,
            "passes_50pct_and_residual_le_10pct": pp_full_pass[pp],
        }

    n_pass = sum(pp_full_pass.values())
    if n_pass == 2:
        verdict = "strong_yes"
        reason = "both PPs >=50% reduction AND BEST residual <=10%"
    elif n_pass == 1 or all(pp_reduced_by_50.values()):
        verdict = "weak_yes"
        if n_pass == 1:
            reason = (
                f"only PP={'128' if pp_full_pass['128'] else '512'} passes full gate; "
                "other PP not demonstrated"
            )
        else:
            reason = "both PPs reduce >=50%, but at least one BEST residual remains >10%"
    else:
        verdict = "no"
        reason = "neither PP shows >=50% reduction with BEST residual <=10%"

    return {
        "verdict": verdict,
        "best_cooldown_per_pp": best_per_pp,
        "reason": reason,
        "details": pp_details,
    }


def _cooldown_label(n: int) -> str:
    return f"{n}s"


def _parse_runs_map(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in s.split(","):
        pp_s, n_s = entry.split(":")
        out[pp_s.strip()] = int(n_s.strip())
    return out


def _run_envelope_for_cooldown(args: argparse.Namespace, label: str) -> None:
    """Run envelope once per PP from exact P5h+2.d Stage 1 cell dirs."""
    pps = [pp.strip() for pp in args.pps.split(",")]
    runs_map = _parse_runs_map(args.runs_per_pp)
    for pp in pps:
        out_json = Path(f"{args.out_base}-cd{label}-pp{pp}-envelope.json")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "p5i_c_pp_tps_envelope.py"),
            "--pp", pp,
            "--out-json", str(out_json),
            "--expected-runs", str(runs_map[pp]),
        ]
        for repeat in range(1, args.repeats + 1):
            cell_dir = Path(f"{args.out_base}-r{repeat}-pp{pp}-cd{label}")
            cmd.extend(["--repeat-csv", str(cell_dir / "bench.csv")])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f"envelope failed cooldown={label} PP={pp}: {result.stderr}")
        print(f"  cooldown={label} PP={pp} envelope -> {out_json}")


def run_sweep(args: argparse.Namespace) -> None:
    """Run cooldown matrix: for each cooldown level, invoke p5h_2b driver."""
    cooldown_levels = [int(s) for s in args.cooldown_levels.split(",")]
    pps = [pp.strip() for pp in args.pps.split(",")]
    for cooldown in cooldown_levels:
        label = _cooldown_label(cooldown)
        exp_id = f"stage1-cd{label}"
        driver_out_base = f"{args.out_base}-driver-cd{label}"
        cmd = [
            sys.executable,
            str(PROTOCOL_DRIVER),
            "--phase", "t4",  # reuse t4 phase label for production sweeps
            "--exp-id", exp_id,
            "--server-lifecycle", "same_spawn_per_pp",
            "--pp-order", args.pps,
            "--logging-mode", "quiet_acceptance",
            "--mode", "production",
            "--repeats", str(args.repeats),
            "--pps", args.pps,
            "--runs-per-pp", args.runs_per_pp,
            "--preheat-seconds", str(args.preheat_seconds),
            "--preheat-runs", str(args.preheat_runs),
            "--model-dir", args.model_dir,
            "--mlx-dir", args.mlx_dir,
            "--out-base", driver_out_base,
            "--inter-run-cooldown-secs", str(cooldown),
            "--skip-envelope",
        ]
        print(f"=== Stage 1 cooldown={label} ===")
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            raise SystemExit(f"Stage 1 cooldown={label} sweep failed (exit {r.returncode})")
        for repeat in range(1, args.repeats + 1):
            for pp in pps:
                src = Path(f"{driver_out_base}-{exp_id}-r{repeat}-pp{pp}")
                dst = Path(f"{args.out_base}-r{repeat}-pp{pp}-cd{label}")
                if not src.exists():
                    raise SystemExit(f"expected cell dir missing: {src}")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
        _run_envelope_for_cooldown(args, label)


def run_gate(args: argparse.Namespace) -> None:
    """Read per-cooldown per-PP envelope JSONs; compute Mechanism gate; write
    one consolidated JSON."""
    cooldown_levels = [int(s) for s in args.cooldown_levels.split(",")]
    pps = args.pps.split(",")
    matrix: dict = {}
    for cooldown in cooldown_levels:
        label = _cooldown_label(cooldown)
        matrix[label] = {}
        for pp in pps:
            env_path = Path(args.envelope_glob.format(cooldown=label, pp=pp))
            if not env_path.exists():
                raise SystemExit(f"missing envelope JSON: {env_path}")
            env_json = json.loads(env_path.read_text())
            matrix[label][pp] = env_json["ironmlx"]["per_repeat"]
    verdict = compute_mechanism_gate(matrix)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sweep = sub.add_parser("sweep", help="Run Stage 1 cooldown matrix sweep")
    sweep.add_argument("--cooldown-levels", default="0,60,120")
    sweep.add_argument("--pps", default="128,512")
    sweep.add_argument("--repeats", type=int, default=3)
    sweep.add_argument("--runs-per-pp", default="128:15,512:15")
    sweep.add_argument("--preheat-seconds", type=int, default=300)
    sweep.add_argument("--preheat-runs", type=int, default=1100)
    sweep.add_argument("--model-dir", required=True)
    sweep.add_argument("--mlx-dir", required=True)
    sweep.add_argument("--out-base", default="/tmp/p5h+2-d-stage1")

    gate = sub.add_parser("gate", help="Compute Mechanism gate from envelope JSONs")
    gate.add_argument("--cooldown-levels", default="0,60,120")
    gate.add_argument("--pps", default="128,512")
    gate.add_argument(
        "--envelope-glob",
        default="/tmp/p5h+2-d-stage1-cd{cooldown}-pp{pp}-envelope.json",
        help="Format string with {cooldown} (e.g. '60s') and {pp} (e.g. '128')",
    )
    gate.add_argument(
        "--out-json", default="/tmp/p5h+2-d-stage1-mechanism-gate.json"
    )

    args = p.parse_args()
    if args.cmd == "sweep":
        run_sweep(args)
    elif args.cmd == "gate":
        run_gate(args)


if __name__ == "__main__":
    main()
