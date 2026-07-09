#!/usr/bin/env python3
"""Run HTTP production-readiness validation for quantized ironmlx models.

The matrix intentionally uses the external OpenAI-compatible HTTP boundary:

* ironmlx serve for the candidate model
* iron-bench sequential HTTP E2E
* iron-bench long-context HTTP E2E
* iron-bench concurrent HTTP E2E
* direct multi-turn /v1/chat/completions checks
* direct repeated-request stability checks

No third-party Python dependencies are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUT_ROOT = "docs/benchmarks/quant-validation"
DEFAULT_SEQUENTIAL_PROMPT_LENS = "128,512"
DEFAULT_LONG_PROMPT_LENS = "4096"
DEFAULT_CONCURRENT_PROMPT_LENS = DEFAULT_SEQUENTIAL_PROMPT_LENS
DEFAULT_MAX_TOKENS = 16
DEFAULT_SEQUENTIAL_RUNS = 2
DEFAULT_SEQUENTIAL_WARMUP = 1
DEFAULT_CONCURRENT = "2"
DEFAULT_DURATION = 10
DEFAULT_WARMUP_DURATION = 2
DEFAULT_STABILITY_RUNS = 5
DEFAULT_MULTI_TURN_TURNS = 3
DEFAULT_PORT_BASE = 18600


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


@dataclass
class CommandResult:
    name: str
    command: list[str]
    stdout: str | None
    stderr: str | None
    exit_code: int
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class MatrixError(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("prompt lengths must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one prompt length is required")
    return values


def parse_model(raw: str) -> ModelSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("model path cannot be empty")
    return ModelSpec(label=label, path=Path(path).expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ironmlx HTTP validation matrix for quantized models."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Model under test, as LABEL=LOCAL_SNAPSHOT_PATH. Repeat for multiple models.",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUT_ROOT}",
    )
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument(
        "--sequential-prompt-lens",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_SEQUENTIAL_PROMPT_LENS),
        help=f"Comma-separated short-context prompt lengths. Default: {DEFAULT_SEQUENTIAL_PROMPT_LENS}",
    )
    parser.add_argument(
        "--long-prompt-lens",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_LONG_PROMPT_LENS),
        help=f"Comma-separated long-context prompt lengths. Default: {DEFAULT_LONG_PROMPT_LENS}",
    )
    parser.add_argument(
        "--concurrent-prompt-lens",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_CONCURRENT_PROMPT_LENS),
        help=(
            "Comma-separated prompt lengths for concurrent HTTP E2E. "
            f"Default: {DEFAULT_CONCURRENT_PROMPT_LENS}"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sequential-runs", type=int, default=DEFAULT_SEQUENTIAL_RUNS)
    parser.add_argument("--sequential-warmup", type=int, default=DEFAULT_SEQUENTIAL_WARMUP)
    parser.add_argument(
        "--concurrent",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_CONCURRENT),
        help=f"Comma-separated concurrent worker levels. Default: {DEFAULT_CONCURRENT}",
    )
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--warmup-duration", type=int, default=DEFAULT_WARMUP_DURATION)
    parser.add_argument("--stability-runs", type=int, default=DEFAULT_STABILITY_RUNS)
    parser.add_argument("--multi-turn-turns", type=int, default=DEFAULT_MULTI_TURN_TURNS)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--serve-prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--serve-max-cache-cap", type=int, default=32768)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def ensure_positive_args(args: argparse.Namespace) -> None:
    fields = [
        "port_base",
        "max_tokens",
        "sequential_runs",
        "duration",
        "startup_timeout",
        "request_timeout",
        "serve_prefill_chunk_size",
        "serve_max_cache_cap",
    ]
    for field in fields:
        if getattr(args, field) <= 0:
            raise MatrixError(f"--{field.replace('_', '-')} must be > 0")
    if args.sequential_warmup < 0:
        raise MatrixError("--sequential-warmup must be >= 0")
    if args.warmup_duration < 0:
        raise MatrixError("--warmup-duration must be >= 0")
    for concurrent in args.concurrent:
        if concurrent <= 0:
            raise MatrixError("--concurrent entries must be > 0")
    if args.stability_runs <= 0:
        raise MatrixError("--stability-runs must be > 0")
    if args.multi_turn_turns <= 0:
        raise MatrixError("--multi-turn-turns must be > 0")


def validate_models(models: list[ModelSpec]) -> None:
    for model in models:
        if not model.path.is_dir():
            raise MatrixError(f"{model.label}: model path does not exist: {model.path}")
        tokenizer = model.path / "tokenizer.json"
        if not tokenizer.is_file():
            raise MatrixError(f"{model.label}: missing tokenizer.json at {tokenizer}")
        config = model.path / "config.json"
        if not config.is_file():
            raise MatrixError(f"{model.label}: missing config.json at {config}")


def make_run_dir(root: Path) -> Path:
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    run_dir = root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (root / "latest.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    return run_dir


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def run_command(
    name: str,
    command: list[str],
    cwd: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    start = time.monotonic()
    stdout_file = stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.PIPE
    stderr_file = stderr_path.open("w", encoding="utf-8") if stderr_path else subprocess.PIPE
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )
        stdout_text = None if stdout_path else completed.stdout
        stderr_text = None if stderr_path else completed.stderr
        return CommandResult(
            name=name,
            command=command,
            stdout=str(stdout_path) if stdout_path else stdout_text,
            stderr=str(stderr_path) if stderr_path else stderr_text,
            exit_code=completed.returncode,
            elapsed_s=time.monotonic() - start,
        )
    finally:
        if stdout_path and hasattr(stdout_file, "close"):
            stdout_file.close()
        if stderr_path and hasattr(stderr_file, "close"):
            stderr_file.close()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def http_get_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="replace")


def http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def tail(path: Path, max_bytes: int = 8192) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def start_server(
    repo: Path,
    model: ModelSpec,
    model_dir: Path,
    port: int,
    max_sequences: int,
    args: argparse.Namespace,
) -> subprocess.Popen[Any]:
    log_path = model_dir / "server.log"
    command = [
        str(repo / "target/release/ironmlx"),
        "serve",
        "--model",
        str(model.path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-sequences",
        str(max_sequences),
        "--max-cache-cap",
        str(args.serve_max_cache_cap),
        "--prefill-chunk-size",
        str(args.serve_prefill_chunk_size),
    ]
    server_env = os.environ.copy()
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=server_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._ironmlx_log_file = log_file  # type: ignore[attr-defined]
    return process


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    log_file = getattr(process, "_ironmlx_log_file", None)
    if log_file is not None:
        log_file.close()


def wait_for_health(
    process: subprocess.Popen[Any],
    health_url: str,
    log_path: Path,
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise MatrixError(
                f"server exited before health check passed; exit={process.returncode}; "
                f"log tail:\n{tail(log_path)}"
            )
        try:
            body = http_get_text(health_url, timeout_s=2)
            if body.strip() == "ok":
                return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise MatrixError(
        f"timed out waiting for {health_url}: {last_error}; log tail:\n{tail(log_path)}"
    )


def run_iron_bench(
    repo: Path,
    model: ModelSpec,
    model_dir: Path,
    port: int,
    name: str,
    prompt_lens: list[int],
    args: argparse.Namespace,
    concurrent_level: int | None,
) -> dict[str, Any]:
    out_json = model_dir / f"{name}.json"
    err_log = model_dir / f"{name}.stderr.log"
    command = [
        str(repo / "target/release/iron-bench"),
        "--target",
        f"ironmlx=http://127.0.0.1:{port}",
        "--model-dir",
        str(model.path),
        "--model",
        model.label,
        "--prompt-len",
        ",".join(str(n) for n in prompt_lens),
        "--max-tokens",
        str(args.max_tokens),
        "--format",
        "json",
        "--timeout",
        str(args.request_timeout),
    ]
    if concurrent_level is not None:
        command.extend(
            [
                "--concurrent",
                str(concurrent_level),
                "--duration",
                str(args.duration),
                "--warmup-duration",
                str(args.warmup_duration),
            ]
        )
    else:
        command.extend(
            [
                "--runs",
                str(args.sequential_runs),
                "--warmup",
                str(args.sequential_warmup),
            ]
        )
    result = run_command(
        name=name,
        command=command,
        cwd=repo,
        stdout_path=out_json,
        stderr_path=err_log,
        env=os.environ.copy(),
    )
    payload: dict[str, Any] | None = None
    error: str | None = None
    if result.ok:
        try:
            payload = load_json(out_json)
        except Exception as exc:  # noqa: BLE001
            error = f"failed to parse {out_json}: {exc}"
    else:
        error = tail(err_log)
    return {
        "name": name,
        "ok": result.ok and error is None,
        "command": command_to_string(command),
        "stdout_json": str(out_json),
        "stderr_log": str(err_log),
        "exit_code": result.exit_code,
        "elapsed_s": result.elapsed_s,
        "error": error,
        "payload": payload,
    }


def chat_completion(
    base_url: str,
    model_label: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model_label,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.monotonic()
    response = http_post_json(f"{base_url}/v1/chat/completions", payload, timeout_s)
    elapsed_s = time.monotonic() - start
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    usage = response.get("usage", {})
    return {
        "ok": isinstance(content, str) and len(content) > 0,
        "elapsed_s": elapsed_s,
        "content": content,
        "content_chars": len(content),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
    }


def run_multi_turn(
    model: ModelSpec,
    base_url: str,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Reply with one short sentence: what does a reproducible benchmark verify?",
        }
    ]
    turns: list[dict[str, Any]] = []
    error: str | None = None
    for turn_idx in range(args.multi_turn_turns):
        try:
            result = chat_completion(
                base_url,
                model.label,
                messages,
                max_tokens=args.max_tokens,
                timeout_s=args.request_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            turns.append({"turn": turn_idx, "ok": False, "error": error})
            break
        turns.append(
            {
                "turn": turn_idx,
                "ok": result["ok"],
                "elapsed_s": result["elapsed_s"],
                "content_chars": result["content_chars"],
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
            }
        )
        messages.append({"role": "assistant", "content": result["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"Turn {turn_idx + 2}: answer in one short sentence and mention token counting.",
            }
        )
    out_path = model_dir / "multi_turn.json"
    payload = {
        "name": "multi_turn",
        "ok": error is None and all(turn.get("ok") for turn in turns),
        "turns": turns,
        "error": error,
    }
    write_json(out_path, payload)
    payload["path"] = str(out_path)
    return payload


def run_stability(
    model: ModelSpec,
    base_url: str,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    error: str | None = None
    for run_idx in range(args.stability_runs):
        messages = [
            {
                "role": "user",
                "content": f"Stability probe {run_idx}: answer with exactly one short sentence.",
            }
        ]
        try:
            result = chat_completion(
                base_url,
                model.label,
                messages,
                max_tokens=args.max_tokens,
                timeout_s=args.request_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            runs.append({"run": run_idx, "ok": False, "error": error})
            break
        runs.append(
            {
                "run": run_idx,
                "ok": result["ok"],
                "elapsed_s": result["elapsed_s"],
                "content_chars": result["content_chars"],
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
            }
        )
    out_path = model_dir / "stability.json"
    payload = {
        "name": "stability",
        "ok": error is None and all(run.get("ok") for run in runs),
        "runs": runs,
        "error": error,
    }
    write_json(out_path, payload)
    payload["path"] = str(out_path)
    return payload


def maybe_capture_healthz(base_url: str, path: Path) -> dict[str, Any]:
    try:
        body = http_get_text(f"{base_url}/healthz", timeout_s=10)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body}
        write_json(path, payload)
        return {"ok": True, "path": str(path)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def summarize_sequential(model: str, category: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stat in payload.get("stats", []):
        rows.append(
            {
                "model": model,
                "category": category,
                "pp_target": stat.get("pp_target"),
                "tg_target": stat.get("tg_target"),
                "concurrency": 1,
                "requests": stat.get("n_runs"),
                "ttft_ms_p50": stat.get("ttft_ms_median"),
                "ttft_ms_p95": stat.get("ttft_ms_p95"),
                "e2e_s_p95": stat.get("e2e_s_p95"),
                "tokens_per_sec": stat.get("tg_tps_median"),
                "finish_reason_summary": stat.get("finish_reason_summary"),
                "ok": True,
            }
        )
    return rows


def summarize_concurrent(model: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in payload.get("cells", []):
        rows.append(
            {
                "model": model,
                "category": "concurrent",
                "pp_target": cell.get("pp_target"),
                "tg_target": cell.get("tg_target"),
                "concurrency": cell.get("concurrent"),
                "requests": cell.get("n_requests"),
                "ttft_ms_p50": cell.get("ttft_ms", {}).get("p50"),
                "ttft_ms_p95": cell.get("ttft_ms", {}).get("p95"),
                "e2e_s_p95": cell.get("e2e_s_p95"),
                "tokens_per_sec": cell.get("aggregate", {}).get("tokens_per_sec"),
                "finish_reason_summary": cell.get("finish_reason_summary"),
                "ok": True,
            }
        )
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def summarize_direct(model: str, category: str, payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("turns") or payload.get("runs") or []
    elapsed = [
        float(item["elapsed_s"])
        for item in items
        if item.get("ok") and isinstance(item.get("elapsed_s"), (int, float))
    ]
    return {
        "model": model,
        "category": category,
        "pp_target": None,
        "tg_target": None,
        "concurrency": 1,
        "requests": len(items),
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "e2e_s_p95": percentile(elapsed, 0.95),
        "tokens_per_sec": None,
        "finish_reason_summary": None,
        "ok": payload.get("ok", False),
    }


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(run_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    csv_path = run_dir / "summary.csv"
    columns = [
        "model",
        "category",
        "pp_target",
        "tg_target",
        "concurrency",
        "requests",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "e2e_s_p95",
        "tokens_per_sec",
        "ok",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    lines = [
        "# Quant Validation Matrix",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Models: {len(manifest['models'])}",
        "",
        "## Matrix Summary",
        "",
        "| model | category | PP | TG | C | requests | TTFT p50 ms | TTFT p95 ms | E2E p95 s | tok/s | ok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {category} | {pp} | {tg} | {concurrency} | {requests} | {ttft50} | {ttft95} | {e2e95} | {tps} | {ok} |".format(
                model=row.get("model", ""),
                category=row.get("category", ""),
                pp=row.get("pp_target") if row.get("pp_target") is not None else "",
                tg=row.get("tg_target") if row.get("tg_target") is not None else "",
                concurrency=row.get("concurrency") if row.get("concurrency") is not None else "",
                requests=row.get("requests") if row.get("requests") is not None else "",
                ttft50=format_float(row.get("ttft_ms_p50")),
                ttft95=format_float(row.get("ttft_ms_p95")),
                e2e95=format_float(row.get("e2e_s_p95")),
                tps=format_float(row.get("tokens_per_sec")),
                ok=str(row.get("ok", False)).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{run_dir / 'manifest.json'}`",
            f"- CSV summary: `{csv_path}`",
        ]
    )
    for model in manifest["models"]:
        lines.append(f"- `{model['label']}`: `{model['artifact_dir']}`")
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_binaries(repo: Path, run_dir: Path) -> CommandResult:
    return run_command(
        name="build",
        command=["cargo", "build", "--release", "-p", "ironmlx", "-p", "iron-bench"],
        cwd=repo,
        stdout_path=run_dir / "cargo-build.stdout.log",
        stderr_path=run_dir / "cargo-build.stderr.log",
        env=os.environ.copy(),
    )


def run_model(
    repo: Path,
    run_dir: Path,
    model: ModelSpec,
    port: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_dir = run_dir / model.label
    model_dir.mkdir(parents=True, exist_ok=False)
    base_url = f"http://127.0.0.1:{port}"
    process: subprocess.Popen[Any] | None = None
    rows: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "label": model.label,
        "path": str(model.path),
        "port": port,
        "artifact_dir": str(model_dir),
        "ok": False,
        "checks": {},
    }
    try:
        process = start_server(
            repo,
            model,
            model_dir,
            port,
            max_sequences=max(args.concurrent),
            args=args,
        )
        wait_for_health(
            process,
            f"{base_url}/health",
            model_dir / "server.log",
            args.startup_timeout,
        )
        record["checks"]["health_before"] = maybe_capture_healthz(
            base_url, model_dir / "health_before.json"
        )

        sequential = run_iron_bench(
            repo,
            model,
            model_dir,
            port,
            "http_e2e_sequential",
            args.sequential_prompt_lens,
            args,
            concurrent_level=None,
        )
        record["checks"]["http_e2e_sequential"] = without_payload(sequential)
        if sequential["ok"] and sequential["payload"] is not None:
            rows.extend(summarize_sequential(model.label, "http_e2e", sequential["payload"]))

        long_context = run_iron_bench(
            repo,
            model,
            model_dir,
            port,
            "long_context",
            args.long_prompt_lens,
            args,
            concurrent_level=None,
        )
        record["checks"]["long_context"] = without_payload(long_context)
        if long_context["ok"] and long_context["payload"] is not None:
            rows.extend(summarize_sequential(model.label, "long_context", long_context["payload"]))

        for concurrent_level in args.concurrent:
            check_name = f"concurrent_c{concurrent_level}"
            concurrent = run_iron_bench(
                repo,
                model,
                model_dir,
                port,
                check_name,
                args.concurrent_prompt_lens,
                args,
                concurrent_level=concurrent_level,
            )
            record["checks"][check_name] = without_payload(concurrent)
            if concurrent["ok"] and concurrent["payload"] is not None:
                rows.extend(summarize_concurrent(model.label, concurrent["payload"]))

        multi_turn = run_multi_turn(model, base_url, model_dir, args)
        record["checks"]["multi_turn"] = without_large_fields(multi_turn)
        rows.append(summarize_direct(model.label, "multi_turn", multi_turn))

        stability = run_stability(model, base_url, model_dir, args)
        record["checks"]["stability"] = without_large_fields(stability)
        rows.append(summarize_direct(model.label, "stability", stability))

        record["checks"]["health_after"] = maybe_capture_healthz(
            base_url, model_dir / "health_after.json"
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = str(exc)
    finally:
        if process is not None:
            stop_server(process)

    required_checks = [
        "http_e2e_sequential",
        "long_context",
        *(f"concurrent_c{concurrent}" for concurrent in args.concurrent),
        "multi_turn",
        "stability",
    ]
    record["ok"] = all(record["checks"].get(name, {}).get("ok") for name in required_checks)
    return record, rows


def without_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {key: val for key, val in value.items() if key != "payload"}


def without_large_fields(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for item in result.get("turns", []):
        item.pop("content", None)
    for item in result.get("runs", []):
        item.pop("content", None)
    return result


def main() -> int:
    args = parse_args()
    ensure_positive_args(args)
    repo = repo_root()
    models: list[ModelSpec] = args.model
    validate_models(models)

    out_root = (repo / args.out_root).resolve()
    run_dir = make_run_dir(out_root)
    manifest: dict[str, Any] = {
        "started_at_unix": int(time.time()),
        "repo": str(repo),
        "args": {
            "sequential_prompt_lens": args.sequential_prompt_lens,
            "long_prompt_lens": args.long_prompt_lens,
            "concurrent_prompt_lens": args.concurrent_prompt_lens,
            "max_tokens": args.max_tokens,
            "sequential_runs": args.sequential_runs,
            "sequential_warmup": args.sequential_warmup,
            "concurrent": args.concurrent,
            "duration": args.duration,
            "warmup_duration": args.warmup_duration,
            "stability_runs": args.stability_runs,
            "multi_turn_turns": args.multi_turn_turns,
            "serve_prefill_chunk_size": args.serve_prefill_chunk_size,
            "serve_max_cache_cap": args.serve_max_cache_cap,
        },
        "build": None,
        "models": [],
        "overall_status": "failed",
    }
    rows: list[dict[str, Any]] = []

    if not args.skip_build:
        build = build_binaries(repo, run_dir)
        manifest["build"] = {
            "ok": build.ok,
            "command": command_to_string(build.command),
            "stdout": build.stdout,
            "stderr": build.stderr,
            "exit_code": build.exit_code,
            "elapsed_s": build.elapsed_s,
        }
        write_json(run_dir / "manifest.json", manifest)
        if not build.ok:
            write_summary(run_dir, rows, manifest)
            print(f"build failed; see {run_dir / 'manifest.json'}", file=sys.stderr)
            return 1
    else:
        manifest["build"] = {"ok": True, "skipped": True}

    for idx, model in enumerate(models):
        print(f"[quant-validation] running {model.label}", flush=True)
        record, model_rows = run_model(
            repo=repo,
            run_dir=run_dir,
            model=model,
            port=args.port_base + idx,
            args=args,
        )
        manifest["models"].append(record)
        rows.extend(model_rows)
        write_json(run_dir / "manifest.json", manifest)
        write_summary(run_dir, rows, manifest)

    manifest["finished_at_unix"] = int(time.time())
    manifest["overall_status"] = (
        "passed" if manifest["models"] and all(m["ok"] for m in manifest["models"]) else "failed"
    )
    write_json(run_dir / "manifest.json", manifest)
    write_summary(run_dir, rows, manifest)
    print(run_dir)
    return 0 if manifest["overall_status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise
