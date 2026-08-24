#!/usr/bin/env python3
"""Measure DFlash2 request-level concurrency before tensor batching work.

Unlike iron-bench's duration-based concurrent mode, this runner measures whole
closed batches. Requests that start together are all included in the batch wall
time, so a request crossing a duration deadline cannot inflate aggregate TPS.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    worker: int
    elapsed_s: float
    completion_tokens: int
    content_sha256: str
    finish_reason: str | None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[index]


def fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def post_chat(
    endpoint: str,
    model: str,
    prompt: str,
    mode: str,
    max_tokens: int,
    seed: int,
    worker: int,
    barrier: threading.Barrier,
    timeout_s: float,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": False,
        "seed": seed + worker,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if mode == "greedy":
        payload.update({"temperature": 0.0, "top_p": 1.0})
    elif mode == "sampled":
        # The OpenAI Chat DTO intentionally does not expose top_k. The model's
        # checkpoint default (20 for this Qwen3.8 target) remains effective.
        payload.update({"temperature": 0.7, "top_p": 1.0})
    else:
        raise ValueError(f"unsupported mode: {mode}")

    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    barrier.wait()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    elapsed_s = time.perf_counter() - started
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""
    return RequestResult(
        worker=worker,
        elapsed_s=elapsed_s,
        completion_tokens=int(body["usage"]["completion_tokens"]),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        finish_reason=choice.get("finish_reason"),
    )


def run_batch(args: argparse.Namespace, prompt: str, mode: str, concurrency: int) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_chat,
                args.endpoint,
                args.model,
                prompt,
                mode,
                args.max_tokens,
                args.seed,
                worker,
                barrier,
                args.timeout,
            )
            for worker in range(concurrency)
        ]
        barrier.wait()
        started = time.perf_counter()
        results = [future.result() for future in futures]
        wall_s = time.perf_counter() - started

    completion_tokens = sum(result.completion_tokens for result in results)
    if any(result.completion_tokens != args.max_tokens for result in results):
        raise RuntimeError(
            f"{mode} C{concurrency} did not produce exactly {args.max_tokens} tokens per row"
        )
    return {
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "aggregate_tps": completion_tokens / wall_s,
        "request_p50_s": percentile([result.elapsed_s for result in results], 0.50),
        "request_p95_s": percentile([result.elapsed_s for result in results], 0.95),
        "requests": [asdict(result) for result in results],
    }


def summarize_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = [float(batch["aggregate_tps"]) for batch in batches]
    wall = [float(batch["wall_s"]) for batch in batches]
    return {
        "measured_batches": len(batches),
        "aggregate_tps_median": statistics.median(aggregate),
        "aggregate_tps_p05": percentile(aggregate, 0.05),
        "aggregate_tps_p95": percentile(aggregate, 0.95),
        "batch_wall_s_median": statistics.median(wall),
        "raw_batches": batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--modes", default="greedy,sampled")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measured-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.concurrency = [int(value) for value in args.concurrency.split(",")]
    args.modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not args.concurrency or any(value <= 0 for value in args.concurrency):
        parser.error("--concurrency must contain positive integers")
    if any(mode not in {"greedy", "sampled"} for mode in args.modes):
        parser.error("--modes accepts only greedy,sampled")
    if args.max_tokens <= 0 or args.warmup_batches < 0 or args.measured_batches <= 0:
        parser.error("token and batch counts are invalid")
    return args


def main() -> int:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise RuntimeError("prompt file is empty")
    health_before = fetch_json(f"{args.endpoint.rstrip('/')}/healthz", args.timeout)
    results: list[dict[str, Any]] = []
    for mode in args.modes:
        for concurrency in args.concurrency:
            for _ in range(args.warmup_batches):
                run_batch(args, prompt, mode, concurrency)
            batches = [
                run_batch(args, prompt, mode, concurrency)
                for _ in range(args.measured_batches)
            ]
            results.append(
                {
                    "mode": mode,
                    "concurrency": concurrency,
                    **summarize_batches(batches),
                }
            )
            summary = results[-1]
            print(
                f"{mode} C{concurrency}: "
                f"median={summary['aggregate_tps_median']:.3f} tok/s "
                f"wall={summary['batch_wall_s_median']:.3f}s",
                flush=True,
            )
    health_after = fetch_json(f"{args.endpoint.rstrip('/')}/healthz", args.timeout)
    report = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt_file": str(args.prompt_file),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "warmup_batches": args.warmup_batches,
        "measured_batches": args.measured_batches,
        "results": results,
        "health_before": health_before,
        "health_after": health_after,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
