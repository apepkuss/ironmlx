#!/usr/bin/env python3
"""Run fixed-prompt, full-length HTTP decode validation for MXFP checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import quant_validation_matrix as qvm


DEFAULT_PROMPT = "scripts/fixtures/mxfp_strict_decode_prompt.txt"
DEFAULT_OUT_ROOT = "reports/mxfp-strict-decode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate full-length decode through the external HTTP API."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=qvm.parse_model,
        required=True,
        help="Model under test as LABEL=LOCAL_SNAPSHOT_PATH. Repeat as needed.",
    )
    parser.add_argument("--prompt-file", type=Path, default=Path(DEFAULT_PROMPT))
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--port-base", type=int, default=18740)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--concurrent", type=qvm.parse_int_list, default=qvm.parse_int_list("1,8")
    )
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--warmup-duration", type=int, default=5)
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


def validate_args(args: argparse.Namespace, repo: Path) -> Path:
    for name in (
        "port_base",
        "max_tokens",
        "duration",
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
    if args.warmup_duration < 0:
        raise qvm.MatrixError("--warmup-duration must be >= 0")
    prompt_path = args.prompt_file
    if not prompt_path.is_absolute():
        prompt_path = repo / prompt_path
    prompt_path = prompt_path.resolve()
    if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
        raise qvm.MatrixError(f"strict decode prompt is missing or empty: {prompt_path}")
    qvm.validate_models(args.model)
    return prompt_path


def validate_payload(
    payload: dict[str, Any], max_tokens: int, concurrent: int
) -> dict[str, Any]:
    cells = payload.get("cells") or []
    raw_runs = payload.get("raw_runs") or []
    errors: list[str] = []
    failed_request_indices: set[int] = set()
    if len(cells) != 1:
        errors.append(f"expected one benchmark cell, found {len(cells)}")
    if not raw_runs:
        errors.append("benchmark completed no requests")

    for index, row in enumerate(raw_runs):
        reason = row.get("finish_reason")
        completion_tokens = row.get("completion_tokens")
        if reason != "length":
            failed_request_indices.add(index)
            errors.append(
                f"request {index} finish_reason={reason!r}, expected 'length'"
            )
        if completion_tokens != max_tokens:
            failed_request_indices.add(index)
            errors.append(
                f"request {index} completion_tokens={completion_tokens!r}, "
                f"expected {max_tokens}"
            )

    workers = {row.get("worker_id") for row in raw_runs}
    expected_workers = set(range(concurrent))
    if workers != expected_workers:
        errors.append(
            f"worker coverage={sorted(worker for worker in workers if worker is not None)}, "
            f"expected={sorted(expected_workers)}"
        )

    cell = cells[0] if cells else {}
    if cell.get("n_requests") != len(raw_runs):
        errors.append(
            f"cell request count={cell.get('n_requests')!r}, raw request count={len(raw_runs)}"
        )
    return {
        "ok": not errors,
        "completed_requests": len(raw_runs),
        "failed_requests": len(failed_request_indices),
        "prompt_tokens": cell.get("pp_target"),
        "itl_ms_p95": (cell.get("itl_ms") or {}).get("p95"),
        "e2e_s_p95": cell.get("e2e_s_p95"),
        "aggregate_tokens_per_sec": (cell.get("aggregate") or {}).get(
            "tokens_per_sec"
        ),
        "finish_reason_summary": cell.get("finish_reason_summary"),
        "errors": errors,
    }


def run_benchmark(
    repo: Path,
    model: qvm.ModelSpec,
    model_dir: Path,
    prompt_path: Path,
    port: int,
    concurrent: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_json = model_dir / f"strict_decode_c{concurrent}.json"
    err_log = model_dir / f"strict_decode_c{concurrent}.stderr.log"
    command = [
        str(repo / "target/release/iron-bench"),
        "--target",
        f"ironmlx=http://127.0.0.1:{port}",
        "--model-dir",
        str(model.path),
        "--model",
        model.label,
        "--fixed-prompt-file",
        str(prompt_path),
        "--max-tokens",
        str(args.max_tokens),
        "--concurrent",
        str(concurrent),
        "--duration",
        str(args.duration),
        "--warmup-duration",
        str(args.warmup_duration),
        "--format",
        "json",
        "--timeout",
        str(args.request_timeout),
    ]
    result = qvm.run_command(
        name=f"strict_decode_c{concurrent}",
        command=command,
        cwd=repo,
        stdout_path=out_json,
        stderr_path=err_log,
        env=os.environ.copy(),
    )
    payload: dict[str, Any] = {}
    validation = {
        "ok": False,
        "completed_requests": 0,
        "failed_requests": 1,
        "errors": [f"iron-bench exited with status {result.exit_code}"],
    }
    if result.ok:
        try:
            payload = qvm.load_json(out_json)
            validation = validate_payload(payload, args.max_tokens, concurrent)
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
    prompt_path: Path,
    port: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_dir = run_dir / model.label
    model_dir.mkdir(parents=True, exist_ok=False)
    base_url = f"http://127.0.0.1:{port}"
    process: subprocess.Popen[Any] | None = None
    record: dict[str, Any] = {
        "label": model.label,
        "path": str(model.path),
        "checkpoint": qvm.checkpoint_identity(model),
        "artifact_dir": str(model_dir),
        "port": port,
        "benchmarks": {},
        "ok": False,
    }
    try:
        process = qvm.start_server(
            repo,
            model,
            model_dir,
            port,
            max_sequences=max(args.concurrent),
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
        for concurrent in args.concurrent:
            record["benchmarks"][f"c{concurrent}"] = run_benchmark(
                repo, model, model_dir, prompt_path, port, concurrent, args
            )
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
    record["ok"] = bool(health_ok) and all(
        benchmark.get("ok") for benchmark in record["benchmarks"].values()
    ) and len(record["benchmarks"]) == len(args.concurrent)
    return record


def write_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# MXFP Strict Decode Validation",
        "",
        f"- Status: `{manifest['overall_status']}`",
        f"- Prompt SHA-256: `{manifest['prompt_sha256']}`",
        f"- Max completion tokens: `{manifest['args']['max_tokens']}`",
        f"- Concurrency levels: `{manifest['args']['concurrent']}`",
        "",
        "| model | C | requests | prompt tokens | ITL p95 ms | E2E p95 s | tok/s | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in manifest["models"]:
        for name, benchmark in model["benchmarks"].items():
            validation = benchmark["validation"]
            lines.append(
                "| {model} | {concurrent} | {requests} | {prompt_tokens} | {itl} | {e2e} | {tps} | {status} |".format(
                    model=model["label"],
                    concurrent=name.removeprefix("c"),
                    requests=validation.get("completed_requests", 0),
                    prompt_tokens=validation.get("prompt_tokens", ""),
                    itl=qvm.format_float(validation.get("itl_ms_p95")),
                    e2e=qvm.format_float(validation.get("e2e_s_p95")),
                    tps=qvm.format_float(validation.get("aggregate_tokens_per_sec")),
                    status="passed" if benchmark["ok"] else "failed",
                )
            )
    lines.extend(["", f"- Manifest: `{run_dir / 'manifest.json'}`", ""])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = qvm.repo_root()
    prompt_path = validate_args(args, repo)
    run_dir = qvm.make_run_dir((repo / args.out_root).resolve())
    prompt_bytes = prompt_path.read_bytes()
    manifest: dict[str, Any] = {
        "started_at_unix": int(time.time()),
        "repo": str(repo),
        "prompt_file": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "args": {
            "max_tokens": args.max_tokens,
            "concurrent": args.concurrent,
            "duration": args.duration,
            "warmup_duration": args.warmup_duration,
            "serve_prefill_chunk_size": args.serve_prefill_chunk_size,
            "serve_admission_deadline_ms": args.serve_admission_deadline_ms,
            "serve_admission_queue_max": args.serve_admission_queue_max,
            "serve_max_cache_cap": args.serve_max_cache_cap,
            "serve_decode_cadence_mid_chunk_cap": args.serve_decode_cadence_mid_chunk_cap,
        },
        "scheduler_config": qvm.scheduler_config_from_args(args, max(args.concurrent)),
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
        print(f"[mxfp-strict-decode] running {model.label}", flush=True)
        record = run_model(
            repo, run_dir, model, prompt_path, args.port_base + index, args
        )
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
