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
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent


# Allow-list per P5h+2.d spec § 7.1. The substrings reflect *anticipated*
# benign WARN classes; the ironmlx server's current `warn!()` callers (scheduler
# evict/step, metal capture, self_qmm tile fallback, preheat wall underrun) do
# NOT yet match any of these substrings. T2 implementer must surface and
# triage the non-allow-listed WARN lines (written to server_log_scan.json) and
# extend this list with substrings of the actual benign WARNs observed before
# treating them as expected.
ALLOWLISTED_WARN_SUBSTRINGS = (
    "[tracing]",
    "KVCache",
    "buffer-resize",
    "mlx::eval",
)


def _classify_level(line: str) -> str | None:
    """Extract the tracing-format level token from a server.log line.

    Tracing default format: `<ISO timestamp> <SPACES> <LEVEL> <target>: <message>`
    where LEVEL is one of TRACE / DEBUG / INFO / WARN / ERROR.

    Returns the level token if it is one of the known levels; otherwise None.
    """
    parts = line.split(maxsplit=3)
    if len(parts) < 2:
        return None
    candidate = parts[1]
    if candidate in ("TRACE", "DEBUG", "INFO", "WARN", "ERROR"):
        return candidate
    return None


def scan_server_log(server_log_path: Path, allow_server_errors: bool) -> dict:
    """P5h+2.d Rule D server-log scan.

    Any ERROR line hard-fails by default. WARN lines are split into
    allow-listed vs non-allow-listed; non-allow-listed WARNs are surfaced for
    human review but do not auto-drop the cell.
    """
    scan = {
        "path": str(server_log_path),
        "error_count": 0,
        "error_lines": [],
        "allowlisted_warn_count": 0,
        "non_allowlisted_warn_count": 0,
        "non_allowlisted_warn_lines": [],
    }
    if not server_log_path.exists():
        scan["missing"] = True
        return scan
    with server_log_path.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            level = _classify_level(line)
            if level == "ERROR":
                scan["error_count"] += 1
                scan["error_lines"].append(f"{line_no}: {line}")
            elif level == "WARN":
                if any(token in line for token in ALLOWLISTED_WARN_SUBSTRINGS):
                    scan["allowlisted_warn_count"] += 1
                else:
                    scan["non_allowlisted_warn_count"] += 1
                    scan["non_allowlisted_warn_lines"].append(f"{line_no}: {line}")
    if scan["error_count"] > 0 and not allow_server_errors:
        preview = "\n".join(scan["error_lines"][:5])
        raise SystemExit(
            f"{server_log_path}: {scan['error_count']} ERROR lines detected. "
            f"Rule D hard-fail. First lines:\n{preview}"
        )
    return scan


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
    if args.inter_run_cooldown_secs > 0:
        env["P5I_C_INTER_RUN_COOLDOWN_SECS"] = str(args.inter_run_cooldown_secs)
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
        scan = scan_server_log(dst / "server.log", args.allow_server_errors)
        (dst / "server_log_scan.json").write_text(json.dumps(scan, indent=2))
        if scan["non_allowlisted_warn_count"] > 0:
            print(
                f"  WARN review required for {dst}: "
                f"{scan['non_allowlisted_warn_count']} non-allow-listed WARN lines"
            )
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
    # Parse runs-per-pp string ("128:15,512:15" -> {128:15, 512:15})
    runs_map: dict[int, int] = {}
    for entry in args.runs_per_pp.split(","):
        if ":" not in entry:
            raise SystemExit(
                f"--runs-per-pp entry {entry!r} is missing ':' separator "
                "(expected format: '128:15,512:15')"
            )
        pp_s, n_s = entry.split(":", 1)
        try:
            runs_map[int(pp_s.strip())] = int(n_s.strip())
        except ValueError as exc:
            raise SystemExit(
                f"--runs-per-pp entry {entry!r} has non-integer field: {exc} "
                "(expected format: '128:15,512:15')"
            ) from exc

    for pp in args.pps.split(","):
        repeat_csvs = [
            str(cell_map_per_repeat[r - 1][pp] / "bench.csv")
            for r in range(1, args.repeats + 1)
        ]
        out_json = Path(f"{args.out_base}-{args.exp_id}-pp{pp}-envelope.json")
        try:
            expected_runs_for_pp = runs_map[int(pp)]
        except KeyError as exc:
            raise SystemExit(
                f"PP={pp} appears in --pps but not in --runs-per-pp {args.runs_per_pp!r} "
                "(every PP must have an explicit runs count)"
            ) from exc
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "p5i_c_pp_tps_envelope.py"),
            "--pp",
            pp,
            "--out-json",
            str(out_json),
            "--expected-runs",
            str(expected_runs_for_pp),
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
        help="Allow server ERROR lines (default: abort sweep). "
        "Use for diagnostic experiments where server errors are expected.",
    )
    p.add_argument(
        "--inter-run-cooldown-secs",
        type=int,
        default=0,
        help="iron-bench --inter-run-cooldown-secs N for measured cells only "
        "(NOT applied to preheat). Default 0. Per P5h+2.d spec § 3.1.",
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
