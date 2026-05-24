"""P5h+2.b T1/T2/T4 protocol experiment driver.

Per spec § 5.2 + § 5.3: given an experiment matrix row (exp_id +
P5I_C_SERVER_LIFECYCLE + P5I_C_PP_ORDER + P5I_C_LOGGING_MODE + PPs +
repeats), invoke the extended capture harness for each repeat, gather
per-cell artifacts into `/tmp/p5h+2-b-{phase}-{exp_id}-r{R}-pp{PP}/`,
then optionally run pp_tps envelope analysis per PP and write per-experiment
envelope JSON. Diagnostic captures with fewer than 3 repeats must use
--skip-envelope because tools/p5i_c_pp_tps_envelope.py requires >=3 repeat CSVs.

Each cell directory matches the harness output schema: bench.csv + server.log
+ meta.json. For lifecycles that share one server across PPs
(same_spawn_cross_pp), each per-cell directory still gets its own bench.csv
+ meta.json (per spec § 5.2); the harness copies the shared server.log into
each cell directory so the per-cell contract holds uniformly.

CLI:
    python tools/p5h_2b_protocol_experiment.py \\
        --phase t1 --exp-id phase0_current \\
        --server-lifecycle phase0_current \\
        --pp-order 128,512 \\
        --logging-mode default_profile \\
        --mode production \\
        --repeats 3 \\
        --pps 128,512 \\
        --runs-per-pp '128:7,512:15' \\
        --preheat-seconds 300 \\
        --preheat-runs 1100 \\
        --model-dir $SNAP \\
        --mlx-dir $HOME/.local/mlx \\
        --out-base /tmp/p5h+2-b-t1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent


def check_no_scheduler_errors(server_log_path: Path, allow_server_errors: bool) -> None:
    """Acceptance precondition per Codex round-3 design question #3.

    Inspects server.log for `step illegal in <phase> phase` ERROR lines
    (production scheduler phase-guard violations). Default-deny: any
    such ERROR aborts the sweep + preserves the artifact directory.
    Diagnostic experiments wanting to allow these ERRORs explicitly set
    --allow-server-errors.
    """
    if allow_server_errors:
        return
    if not server_log_path.exists():
        return  # missing log handled by caller's downstream check
    count = 0
    with server_log_path.open() as f:
        for line in f:
            if "step illegal in" in line and "phase" in line:
                count += 1
    if count > 0:
        raise SystemExit(
            f"{server_log_path}: {count} `step illegal in <phase>` ERROR lines detected. "
            "Acceptance precondition VIOLATED (Codex round-3 design question #3). "
            "Re-run with --allow-server-errors to override for diagnostic experiments."
        )


def run_one_repeat(args: argparse.Namespace, repeat: int) -> dict[str, Path]:
    """Invoke harness for one repeat; return mapping {pp -> per-cell out dir}."""
    env = os.environ.copy()
    env.update(
        {
            "P5I_C_REPEAT_INDEX": str(repeat),
            "P5I_C_MODE": args.mode,
            "P5I_C_MODEL_DIR": args.model_dir,
            "MLX_DIR": args.mlx_dir,
            "P5I_C_SERVER_LIFECYCLE": args.server_lifecycle,
            "P5I_C_PP_ORDER": args.pp_order,
            "P5I_C_LOGGING_MODE": args.logging_mode,
            "P5I_C_RUNS_PER_PP": args.runs_per_pp,
            "P5I_C_PREHEAT_SECONDS": str(args.preheat_seconds),
            "P5I_C_PREHEAT_RUNS": str(args.preheat_runs),
        }
    )
    cmd = [
        "cargo",
        "test",
        "--release",
        "-p",
        "ironmlx",
        "--features",
        "p5h-profile",
        "--test",
        "p5i_c_phase_0_capture",
        "--",
        "--ignored",
        "--test-threads=1",
        "--nocapture",
    ]
    log_path = Path(f"/tmp/p5h+2-b-{args.phase}-{args.exp_id}-r{repeat}.log")
    with log_path.open("w") as logf:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(f"harness failed for repeat={repeat}; see {log_path}")
    # Harness writes to /tmp/p5i-c-phase-0-r{R}-pp{PP}-{mode}; relocate to
    # /tmp/p5h+2-b-{phase}-{exp_id}-r{R}-pp{PP}/ so multiple experiments
    # don't clobber each other.
    cell_map: dict[str, Path] = {}
    for pp in args.pps.split(","):
        src = Path(f"/tmp/p5i-c-phase-0-r{repeat}-pp{pp}-{args.mode}")
        dst = Path(f"{args.out_base}-{args.exp_id}-r{repeat}-pp{pp}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        check_no_scheduler_errors(dst / "server.log", args.allow_server_errors)
        cell_map[pp] = dst
    return cell_map


def run_envelope(
    args: argparse.Namespace, cell_map_per_repeat: list[dict[str, Path]]
) -> None:
    """Per PP: collect bench.csv across repeats; run envelope script."""
    if args.skip_envelope:
        print("  envelope skipped (--skip-envelope)")
        return
    if args.repeats < 3:
        raise SystemExit("--repeats must be >=3 unless --skip-envelope is set")
    for pp in args.pps.split(","):
        repeat_csvs = [
            str(cell_map_per_repeat[r - 1][pp] / "bench.csv")
            for r in range(1, args.repeats + 1)
        ]
        out_json = Path(f"{args.out_base}-{args.exp_id}-pp{pp}-envelope.json")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "p5i_c_pp_tps_envelope.py"),
            "--pp",
            pp,
            "--out-json",
            str(out_json),
        ]
        for csv_path in repeat_csvs:
            cmd.extend(["--repeat-csv", csv_path])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f"envelope failed for PP={pp}: {result.stderr}")
        print(f"  PP={pp} envelope -> {out_json}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["t1", "t2", "t4"], required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument(
        "--server-lifecycle",
        required=True,
        choices=["phase0_current", "same_spawn_cross_pp", "same_spawn_per_pp"],
    )
    p.add_argument("--pp-order", required=True)
    p.add_argument(
        "--logging-mode",
        required=True,
        choices=["default_profile", "quiet_acceptance", "buffered_profile"],
    )
    p.add_argument("--mode", choices=["probe", "production"], required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pps", required=True, help="comma-separated, e.g. 128,512")
    p.add_argument("--runs-per-pp", required=True, help="e.g. 128:7,512:15")
    p.add_argument("--preheat-seconds", type=int, default=300)
    p.add_argument("--preheat-runs", type=int, default=1100)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mlx-dir", required=True)
    p.add_argument("--out-base", required=True, help="e.g. /tmp/p5h+2-b-t1")
    p.add_argument(
        "--skip-envelope",
        action="store_true",
        help="skip envelope computation for diagnostic captures with <3 repeats",
    )
    p.add_argument(
        "--allow-server-errors",
        action="store_true",
        default=False,
        help="Allow `step illegal in <phase>` server ERROR lines (default: abort sweep). "
        "Use for diagnostic experiments where scheduler errors are expected.",
    )
    args = p.parse_args()

    print(
        f"=== {args.phase} {args.exp_id} ({args.server_lifecycle} / {args.logging_mode}) ==="
    )
    cell_map_per_repeat: list[dict[str, Path]] = []
    for r in range(1, args.repeats + 1):
        print(f"  repeat {r}...")
        cell_map = run_one_repeat(args, r)
        cell_map_per_repeat.append(cell_map)
    run_envelope(args, cell_map_per_repeat)
    print("Done.")


if __name__ == "__main__":
    main()
