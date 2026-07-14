#!/usr/bin/env python3
"""Run repeatable HTTP prefill measurements for affine 4/5/6-bit checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import quant_validation_matrix as qvm


DEFAULT_PROMPT_LENS = "2048,8192,32768"
DEFAULT_OUT_ROOT = "reports/affine56-prefill"
SERVER_B_MAX = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=qvm.parse_model,
        required=True,
        help="Model under test as LABEL=LOCAL_SNAPSHOT_PATH. Repeat as needed.",
    )
    parser.add_argument(
        "--prompt-lens",
        type=qvm.parse_int_list,
        default=qvm.parse_int_list(DEFAULT_PROMPT_LENS),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--inter-run-cooldown-secs", type=int, default=1)
    parser.add_argument("--nonce-seed", type=int, default=5606)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--port-base", type=int, default=19140)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--serve-prefill-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--serve-admission-deadline-ms",
        type=int,
        default=qvm.DEFAULT_SERVE_ADMISSION_DEADLINE_MS,
    )
    parser.add_argument(
        "--serve-admission-queue-max",
        type=int,
        default=qvm.DEFAULT_SERVE_ADMISSION_QUEUE_MAX,
    )
    parser.add_argument("--serve-max-cache-cap", type=int, default=65536)
    parser.add_argument(
        "--serve-decode-cadence-mid-chunk-cap",
        type=int,
        default=qvm.DEFAULT_SERVE_DECODE_CADENCE_MID_CHUNK_CAP,
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "port_base",
        "runs",
        "nonce_seed",
        "startup_timeout",
        "request_timeout",
        "serve_prefill_chunk_size",
        "serve_admission_deadline_ms",
        "serve_admission_queue_max",
        "serve_max_cache_cap",
        "serve_decode_cadence_mid_chunk_cap",
    ):
        if getattr(args, name) <= 0:
            raise qvm.MatrixError(f"--{name.replace('_', '-')} must be > 0")
    if args.warmup < 0 or args.inter_run_cooldown_secs < 0:
        raise qvm.MatrixError("warmup and cooldown must be >= 0")
    if not args.prompt_lens or any(prompt <= 0 for prompt in args.prompt_lens):
        raise qvm.MatrixError("--prompt-lens must contain positive values")
    if len(set(args.prompt_lens)) != len(args.prompt_lens):
        raise qvm.MatrixError("--prompt-lens must not contain duplicates")
    qvm.validate_models(args.model)


def validate_payload(
    payload: dict[str, Any], prompt_lens: list[int], runs: int
) -> dict[str, Any]:
    stats = payload.get("stats") or []
    raw_runs = payload.get("raw_runs") or []
    metadata = payload.get("metadata") or {}
    errors: list[str] = []

    if metadata.get("runs_measured") != runs:
        errors.append(
            f"metadata.runs_measured={metadata.get('runs_measured')!r}, expected {runs}"
        )

    cells: dict[int, dict[str, float]] = {}
    for row in stats:
        prompt = row.get("pp_target")
        median = row.get("ttft_ms_median")
        p95 = row.get("ttft_ms_p95")
        if not isinstance(prompt, int) or prompt in cells:
            errors.append(f"invalid or duplicate prefill stats cell: {prompt!r}")
            continue
        if row.get("tg_target") != 1 or row.get("n_runs") != runs:
            errors.append(f"invalid TG/run count for PP={prompt}")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in (median, p95)
        ):
            errors.append(f"invalid TTFT metrics for PP={prompt}")
            continue
        cells[prompt] = {
            "ttft_ms_median": float(median),
            "ttft_ms_p95": float(p95),
        }

    if set(cells) != set(prompt_lens):
        errors.append(
            f"prompt coverage={sorted(cells)}, expected={sorted(prompt_lens)}"
        )

    counts = {prompt: 0 for prompt in prompt_lens}
    failed_requests = 0
    for index, row in enumerate(raw_runs):
        prompt = row.get("pp_target")
        request_errors = []
        if prompt not in counts:
            request_errors.append(f"unexpected pp_target={prompt!r}")
        else:
            counts[prompt] += 1
        if row.get("tg_target") != 1:
            request_errors.append(f"tg_target={row.get('tg_target')!r}")
        if row.get("finish_reason") != "length":
            request_errors.append(f"finish_reason={row.get('finish_reason')!r}")
        if row.get("completion_tokens_server") != 1:
            request_errors.append(
                f"completion_tokens_server={row.get('completion_tokens_server')!r}"
            )
        ttft = row.get("ttft_ms")
        if not isinstance(ttft, (int, float)) or not math.isfinite(ttft) or ttft <= 0:
            request_errors.append(f"ttft_ms={ttft!r}")
        if request_errors:
            failed_requests += 1
            errors.append(f"request {index}: {', '.join(request_errors)}")

    for prompt, count in counts.items():
        if count != runs:
            errors.append(f"PP={prompt} request count={count}, expected={runs}")
    expected_requests = len(prompt_lens) * runs
    if len(raw_runs) != expected_requests:
        errors.append(
            f"raw request count={len(raw_runs)}, expected={expected_requests}"
        )

    return {
        "ok": not errors,
        "completed_requests": len(raw_runs),
        "failed_requests": failed_requests,
        "cells": cells,
        "errors": errors,
    }


def run_benchmark(
    repo: Path,
    model: qvm.ModelSpec,
    model_dir: Path,
    port: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_json = model_dir / "strict_prefill.json"
    err_log = model_dir / "strict_prefill.stderr.log"
    command = [
        str(repo / "target/release/iron-bench"),
        "--target",
        f"ironmlx=http://127.0.0.1:{port}",
        "--model-dir",
        str(model.path),
        "--model",
        model.label,
        "--prompt-len",
        ",".join(str(prompt) for prompt in args.prompt_lens),
        "--max-tokens",
        "1",
        "--ignore-eos",
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--inter-run-cooldown-secs",
        str(args.inter_run_cooldown_secs),
        "--nonce-seed",
        str(args.nonce_seed),
        "--format",
        "json",
        "--timeout",
        str(args.request_timeout),
    ]
    result = qvm.run_command(
        name="strict_prefill",
        command=command,
        cwd=repo,
        stdout_path=out_json,
        stderr_path=err_log,
        env=os.environ.copy(),
    )
    validation = {
        "ok": False,
        "completed_requests": 0,
        "failed_requests": 1,
        "cells": {},
        "errors": [f"iron-bench exited with status {result.exit_code}"],
    }
    if result.ok:
        try:
            validation = validate_payload(
                qvm.load_json(out_json), args.prompt_lens, args.runs
            )
        except Exception as exc:  # noqa: BLE001
            validation["errors"] = [f"failed to parse benchmark JSON: {exc}"]
    return {
        "ok": result.ok and validation["ok"],
        "command": qvm.command_to_string(command),
        "benchmark_json": str(out_json),
        "stderr_log": str(err_log),
        "exit_code": result.exit_code,
        "elapsed_s": result.elapsed_s,
        "validation": validation,
    }


def run_model(
    repo: Path,
    run_dir: Path,
    model: qvm.ModelSpec,
    port: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_dir = run_dir / model.label
    model_dir.mkdir(parents=True, exist_ok=False)
    base_url = f"http://127.0.0.1:{port}"
    process = None
    record: dict[str, Any] = {
        "label": model.label,
        "path": str(model.path),
        "checkpoint": qvm.checkpoint_identity(model),
        "artifact_dir": str(model_dir),
        "port": port,
        "ok": False,
    }
    try:
        process = qvm.start_server(
            repo,
            model,
            model_dir,
            port,
            max_sequences=SERVER_B_MAX,
            args=args,
        )
        qvm.wait_for_health(
            process,
            f"{base_url}/health",
            model_dir / "server.log",
            args.startup_timeout,
        )
        record["health_before"] = qvm.maybe_capture_healthz(
            base_url, model_dir / "health_before.json"
        )
        record["benchmark"] = run_benchmark(repo, model, model_dir, port, args)
        record["health_after"] = qvm.maybe_capture_healthz(
            base_url, model_dir / "health_after.json"
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = str(exc)
    finally:
        if process is not None:
            qvm.stop_server(process)

    health_ok = record.get("health_before", {}).get("ok") and record.get(
        "health_after", {}
    ).get("ok")
    record["ok"] = bool(health_ok) and record.get("benchmark", {}).get("ok", False)
    return record


def write_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Affine 5-bit and 6-bit Clean Prefill",
        "",
        f"- Status: `{manifest['overall_status']}`",
        f"- Runs per cell: `{manifest['args']['runs']}`",
        f"- Warmup per cell: `{manifest['args']['warmup']}`",
        f"- Inter-run cooldown: `{manifest['args']['inter_run_cooldown_secs']}s`",
        "",
        "| model | PP | runs | TTFT median ms | TTFT p95 ms | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model in manifest["models"]:
        validation = (model.get("benchmark") or {}).get("validation") or {}
        for prompt, cell in sorted((validation.get("cells") or {}).items()):
            lines.append(
                f"| {model['label']} | {prompt} | {manifest['args']['runs']} | "
                f"{cell['ttft_ms_median']:.3f} | {cell['ttft_ms_p95']:.3f} | "
                f"{'passed' if model['ok'] else 'failed'} |"
            )
    lines.extend(["", f"- Manifest: `{run_dir / 'manifest.json'}`", ""])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo = qvm.repo_root()
    run_dir = qvm.make_run_dir((repo / args.out_root).resolve())
    manifest: dict[str, Any] = {
        "started_at_unix": int(time.time()),
        "repo": str(repo),
        "args": {
            "prompt_lens": args.prompt_lens,
            "max_tokens": 1,
            "ignore_eos": True,
            "runs": args.runs,
            "warmup": args.warmup,
            "inter_run_cooldown_secs": args.inter_run_cooldown_secs,
            "nonce_seed": args.nonce_seed,
            "serve_prefill_chunk_size": args.serve_prefill_chunk_size,
            "serve_admission_deadline_ms": args.serve_admission_deadline_ms,
            "serve_admission_queue_max": args.serve_admission_queue_max,
            "serve_max_cache_cap": args.serve_max_cache_cap,
            "serve_decode_cadence_mid_chunk_cap": args.serve_decode_cadence_mid_chunk_cap,
        },
        "scheduler_config": qvm.scheduler_config_from_args(args, SERVER_B_MAX),
        "build": None,
        "models": [],
        "overall_status": "failed",
    }

    if args.skip_build:
        manifest["build"] = {"ok": True, "skipped": True}
    else:
        build = qvm.build_binaries(repo, run_dir)
        manifest["build"] = {
            "ok": build.ok,
            "command": qvm.command_to_string(build.command),
            "exit_code": build.exit_code,
            "elapsed_s": build.elapsed_s,
            "stdout": build.stdout,
            "stderr": build.stderr,
        }
        if not build.ok:
            qvm.write_json(run_dir / "manifest.json", manifest)
            write_summary(run_dir, manifest)
            return 1

    for index, model in enumerate(args.model):
        print(f"[affine56-prefill] running {model.label}", flush=True)
        record = run_model(repo, run_dir, model, args.port_base + index, args)
        manifest["models"].append(record)
        qvm.write_json(run_dir / "manifest.json", manifest)

    manifest["finished_at_unix"] = int(time.time())
    manifest["overall_status"] = (
        "passed"
        if manifest["models"] and all(model["ok"] for model in manifest["models"])
        else "failed"
    )
    qvm.write_json(run_dir / "manifest.json", manifest)
    write_summary(run_dir, manifest)
    print(run_dir)
    return 0 if manifest["overall_status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise
