#!/usr/bin/env python3
"""Gemma4 drafter + Active KV heavy regression runner.

The runner intentionally keeps iron-bench engine-neutral. It owns the
ironmlx-specific lifecycle: App daemon startup, dynamic model load, health
assertions, and benchmark artifact collation.
"""

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MAX_CACHE_CAP = 262144
DEFAULT_MTP_DRAFT_TOKENS = 2
DEFAULT_PROMPT_LENS = (2048, 24576)
BYTES_PER_MIB = 1024 * 1024
ROLLING_ACTIVE_GT_ONE_RE = re.compile(r"active_(?:before|after)=[2-9]\d*")
ROLLING_DECODE_ERROR_PATTERNS = (
    "event=decode_step_error",
    "[SchedulerActor] step error",
)


@dataclass(frozen=True)
class RegressionVariant:
    name: str
    model_label: str
    model_dir: Path
    drafter_dir: Path
    b_max: int
    concurrent: int
    max_cache_cap: int = DEFAULT_MAX_CACHE_CAP
    mtp_draft_tokens: int = DEFAULT_MTP_DRAFT_TOKENS
    kv_quant: str = "k3v4"


@dataclass
class RegressionConfig:
    root: Path
    out_root: Path
    serve_bin: Path
    iron_bench_bin: Path
    e4b_model_dir: Path
    e4b_drafter_dir: Path
    gemma12b_model_dir: Path
    gemma12b_drafter_dir: Path
    prompt_lens: Tuple[int, ...] = DEFAULT_PROMPT_LENS
    max_tokens: int = 32
    duration_secs: int = 20
    warmup_duration_secs: int = 5
    startup_timeout_secs: int = 180
    request_timeout_secs: int = 900
    host: str = "127.0.0.1"
    port_base: int = 19180
    prefix_cache_block_size: int = 16
    prefix_lru_cache_max_bytes: Optional[int] = None
    ssd_prefix_cache_max_gb: Optional[float] = None
    mlx_dir: Path = Path.home() / ".local" / "mlx"
    rust_log: str = "info"
    build: bool = False
    dry_run: bool = False
    allow_failures: bool = False
    prefix_cache_probe: bool = False
    variant_names: Tuple[str, ...] = field(default_factory=tuple)
    extra_serve_args: Tuple[str, ...] = field(default_factory=tuple)
    extra_bench_args: Tuple[str, ...] = field(default_factory=tuple)


def build_default_variants(config: RegressionConfig) -> List[RegressionVariant]:
    return [
        RegressionVariant(
            name="e4b_b2",
            model_label="gemma4-e4b-b2",
            model_dir=config.e4b_model_dir,
            drafter_dir=config.e4b_drafter_dir,
            b_max=2,
            concurrent=2,
        ),
        RegressionVariant(
            name="e4b_b4",
            model_label="gemma4-e4b-b4",
            model_dir=config.e4b_model_dir,
            drafter_dir=config.e4b_drafter_dir,
            b_max=4,
            concurrent=4,
        ),
        RegressionVariant(
            name="12b_b2",
            model_label="gemma4-12b-b2",
            model_dir=config.gemma12b_model_dir,
            drafter_dir=config.gemma12b_drafter_dir,
            b_max=2,
            concurrent=2,
        ),
    ]


def build_serve_command(
    config: RegressionConfig,
    variant: RegressionVariant,
    port: int,
    variant_dir: Path,
) -> List[str]:
    cmd = [
        str(config.serve_bin),
        "serve",
        "--host",
        config.host,
        "--port",
        str(port),
        "--max-sequences",
        str(variant.b_max),
        "--max-cache-cap",
        str(variant.max_cache_cap),
        "--kv-quant",
        variant.kv_quant,
        "--paged-prefix-cache-dir",
        str(variant_dir / "prefix-cache"),
        "--paged-prefix-cache-block-size",
        str(config.prefix_cache_block_size),
        "--active-kv-offload",
        "--active-kv-offload-dir",
        str(variant_dir / "active-kv"),
    ]
    if config.ssd_prefix_cache_max_gb is not None:
        cmd.extend(
            [
                "--ssd-prefix-cache-max-gb",
                format_gib_arg(config.ssd_prefix_cache_max_gb),
            ]
        )
    if config.prefix_lru_cache_max_bytes is not None:
        cmd.extend(["--prefix-lru-cache-max-bytes", str(config.prefix_lru_cache_max_bytes)])
    cmd.extend(config.extra_serve_args)
    return cmd


def build_load_payload(config: RegressionConfig, variant: RegressionVariant) -> Dict[str, Any]:
    del config
    return {
        "model": variant.model_label,
        "model_dir": str(variant.model_dir),
        "mtp_model_dir": str(variant.drafter_dir),
        "mtp_draft_tokens": variant.mtp_draft_tokens,
        "max_cache_cap": variant.max_cache_cap,
        "set_default": True,
        "pinned": True,
    }


def build_bench_command(
    config: RegressionConfig,
    variant: RegressionVariant,
    port: int,
    prompt_len: int,
) -> List[str]:
    cmd = [
        str(config.iron_bench_bin),
        "--target",
        "ironmlx=http://{}:{}".format(config.host, port),
        "--model-dir",
        str(variant.model_dir),
        "--model",
        variant.model_label,
        "--prompt-len",
        str(prompt_len),
        "--max-tokens",
        str(config.max_tokens),
        "--concurrent",
        str(variant.concurrent),
        "--duration",
        str(config.duration_secs),
        "--warmup-duration",
        str(config.warmup_duration_secs),
        "--timeout",
        str(config.request_timeout_secs),
        "--format",
        "json",
    ]
    if config.prefix_cache_probe:
        cmd.append("--prefix-cache-probe")
    cmd.extend(config.extra_bench_args)
    return cmd


def build_run_plan(config: RegressionConfig) -> List[Dict[str, Any]]:
    variants = build_default_variants(config)
    if config.variant_names:
        requested = set(config.variant_names)
        known = {variant.name for variant in variants}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError("unknown variant(s): {}".format(", ".join(unknown)))
        variants = [variant for variant in variants if variant.name in requested]

    plan = []
    for index, variant in enumerate(variants):
        port = config.port_base + index
        variant_dir = config.out_root / variant.name
        plan.append(
            {
                "variant": variant,
                "port": port,
                "variant_dir": variant_dir,
                "serve_cmd": build_serve_command(config, variant, port, variant_dir),
                "load_payload": build_load_payload(config, variant),
                "bench_cmds": [
                    build_bench_command(config, variant, port, prompt_len)
                    for prompt_len in config.prompt_lens
                ],
            }
        )
    return plan


def assert_health_delta(
    variant: RegressionVariant,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> None:
    errors = validate_health_delta(variant, before, after)
    if errors:
        raise AssertionError("; ".join(errors))


def assert_rolling_mid_admit_profile(log_text: str) -> None:
    for pattern in ROLLING_DECODE_ERROR_PATTERNS:
        if pattern in log_text:
            raise AssertionError("Gemma4 drafter adaptive run hit decode step errors")
    if "event=mid_begin" not in log_text:
        raise AssertionError("Gemma4 drafter adaptive run did not start rolling mid-admit")
    if "event=mid_finalize" not in log_text:
        raise AssertionError("Gemma4 drafter adaptive run did not finalize rolling mid-admit")
    if not ROLLING_ACTIVE_GT_ONE_RE.search(log_text):
        raise AssertionError("Gemma4 drafter adaptive run never exceeded active_count=1")


def validate_health_delta(
    variant: RegressionVariant,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> List[str]:
    errors = []
    before_budget = int_value(before, "scheduler.memory_budget_exceeded_count")
    after_budget = int_value(after, "scheduler.memory_budget_exceeded_count")
    if after_budget > before_budget:
        errors.append(
            "memory_budget_exceeded_count increased from {} to {}".format(
                before_budget, after_budget
            )
        )

    if int_value(after, "scheduler.b_max") not in (0, variant.b_max):
        errors.append(
            "scheduler.b_max expected {} got {}".format(
                variant.b_max, int_value(after, "scheduler.b_max")
            )
        )

    policy = str_value(after, "memory.kv_cache_budget_policy")
    if policy != "active_kv_offload":
        errors.append("memory.kv_cache_budget_policy expected active_kv_offload got {}".format(policy))

    logical_cap = int_value(after, "memory.kv_cache_logical_cap_tokens")
    resident_cap = int_value(after, "memory.kv_cache_resident_cap_tokens")
    expected_logical_cap = expected_effective_logical_cap(variant, after)
    if logical_cap != expected_logical_cap:
        errors.append(
            "kv_cache_logical_cap_tokens expected {} got {}".format(
                expected_logical_cap, logical_cap
            )
        )
    if resident_cap <= 0 or resident_cap >= logical_cap:
        errors.append(
            "kv_cache_resident_cap_tokens must be >0 and < logical cap, got {} vs {}".format(
                resident_cap, logical_cap
            )
        )

    if not bool_value(after, "active_kv_offload.enabled"):
        errors.append("active_kv_offload.enabled is false")
    if bool_value(after, "active_kv_offload.degraded"):
        errors.append("active_kv_offload.degraded is true")
    swap_error_count = int_value(after, "active_kv_offload.swap_error_count")
    if swap_error_count != 0:
        errors.append("active_kv_offload.swap_error_count expected 0 got {}".format(swap_error_count))

    if not bool_value(after, "mtp.enabled"):
        errors.append("mtp.enabled is false")
    draft_tokens = int_value(after, "mtp.draft_tokens")
    if draft_tokens != variant.mtp_draft_tokens:
        errors.append(
            "mtp.draft_tokens expected {} got {}".format(
                variant.mtp_draft_tokens, draft_tokens
            )
        )
    before_mtp = int_value(before, "mtp.prefill_count") + int_value(before, "mtp.step_count")
    after_mtp = int_value(after, "mtp.prefill_count") + int_value(after, "mtp.step_count")
    if after_mtp <= before_mtp:
        errors.append("mtp counters did not increase after greedy traffic")

    return errors


def summarize_bench_payload(
    variant: RegressionVariant,
    payload: Dict[str, Any],
    health_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for cell in payload.get("cells") or []:
        row = {
            "variant": variant.name,
            "model": variant.model_label,
            "prompt_len": int(cell.get("pp_target") or 0),
            "max_tokens": int(cell.get("tg_target") or 0),
            "b_max": variant.b_max,
            "concurrent": int(cell.get("concurrent") or variant.concurrent),
            "n_requests": int(cell.get("n_requests") or 0),
            "ttft_ms_p50": nested_float(cell, "ttft_ms.p50"),
            "ttft_ms_p95": nested_float(cell, "ttft_ms.p95"),
            "itl_ms_p50": nested_float(cell, "itl_ms.p50"),
            "itl_ms_p95": nested_float(cell, "itl_ms.p95"),
            "early_itl_ms_p50": nested_float(cell, "early_itl_ms.p50"),
            "early_itl_ms_p95": nested_float(cell, "early_itl_ms.p95"),
            "tokens_per_sec": nested_float(cell, "aggregate.tokens_per_sec"),
            "req_per_sec": nested_float(cell, "aggregate.req_per_sec"),
            "finish_reason_summary": str(cell.get("finish_reason_summary") or ""),
            "memory_budget_exceeded_count": int_value(
                health_payload, "scheduler.memory_budget_exceeded_count"
            ),
            "kv_cache_budget_policy": str_value(
                health_payload, "memory.kv_cache_budget_policy"
            ),
            "kv_cache_logical_cap_tokens": int_value(
                health_payload, "memory.kv_cache_logical_cap_tokens"
            ),
            "kv_cache_resident_cap_tokens": int_value(
                health_payload, "memory.kv_cache_resident_cap_tokens"
            ),
            "active_kv_degraded": bool_value(health_payload, "active_kv_offload.degraded"),
            "active_kv_swap_error_count": int_value(
                health_payload, "active_kv_offload.swap_error_count"
            ),
            "active_kv_swap_out_count": int_value(
                health_payload, "active_kv_offload.swap_out_count"
            ),
            "active_kv_swap_in_count": int_value(
                health_payload, "active_kv_offload.swap_in_count"
            ),
            "mtp_prefill_count": int_value(health_payload, "mtp.prefill_count"),
            "mtp_step_count": int_value(health_payload, "mtp.step_count"),
            "mtp_drafted_tokens": int_value(health_payload, "mtp.drafted_tokens"),
            "mtp_accepted_draft_tokens": int_value(
                health_payload, "mtp.accepted_draft_tokens"
            ),
            "mlx_peak_mib": bytes_to_mib(int_value(health_payload, "memory.mlx_peak_bytes")),
            "status": "ok",
            "notes": "",
        }
        if row["n_requests"] <= 0:
            row["status"] = "error"
            row["notes"] = "iron-bench completed zero requests"
        rows.append(row)
    if not rows:
        rows.append(error_row(variant, 0, "iron-bench JSON contained no cells"))
    return rows


def write_summary_files(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=False), encoding="utf-8"
    )
    fieldnames = [
        "variant",
        "model",
        "prompt_len",
        "max_tokens",
        "b_max",
        "concurrent",
        "n_requests",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "itl_ms_p50",
        "itl_ms_p95",
        "early_itl_ms_p50",
        "early_itl_ms_p95",
        "tokens_per_sec",
        "req_per_sec",
        "finish_reason_summary",
        "memory_budget_exceeded_count",
        "kv_cache_budget_policy",
        "kv_cache_logical_cap_tokens",
        "kv_cache_resident_cap_tokens",
        "active_kv_degraded",
        "active_kv_swap_error_count",
        "active_kv_swap_out_count",
        "active_kv_swap_in_count",
        "mtp_prefill_count",
        "mtp_step_count",
        "mtp_drafted_tokens",
        "mtp_accepted_draft_tokens",
        "mlx_peak_mib",
        "status",
        "notes",
    ]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (out_root / "summary.md").write_text(render_markdown(rows), encoding="utf-8")


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Gemma4 Drafter Active KV Regression",
        "",
        "| variant | pp | b_max | conc | req | ttft p50 ms | itl p95 ms | tok/s | budget | resident/logical | active kv | mtp steps | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {pp} | {bmax} | {conc} | {req} | {ttft} | {itl} | {tps} | {budget} | {resident}/{logical} | {active} | {mtp} | {status} |".format(
                variant=row.get("variant", ""),
                pp=row.get("prompt_len", ""),
                bmax=row.get("b_max", ""),
                conc=row.get("concurrent", ""),
                req=row.get("n_requests", ""),
                ttft=fmt(row.get("ttft_ms_p50")),
                itl=fmt(row.get("itl_ms_p95")),
                tps=fmt(row.get("tokens_per_sec")),
                budget=fmt(row.get("memory_budget_exceeded_count"), digits=0),
                resident=fmt(row.get("kv_cache_resident_cap_tokens"), digits=0),
                logical=fmt(row.get("kv_cache_logical_cap_tokens"), digits=0),
                active=active_kv_status(row.get("active_kv_degraded")),
                mtp=fmt(row.get("mtp_step_count"), digits=0),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- This is an opt-in heavy local regression. It requires local Gemma4 checkpoints.",
            "- `resident/logical` must show Active KV charging less resident KV than the logical MAX TOKENS cap.",
            "- Any `active kv` value other than `ok` is a production regression signal.",
            "",
        ]
    )
    return "\n".join(lines)


def run_regression(config: RegressionConfig) -> int:
    config.out_root.mkdir(parents=True, exist_ok=True)
    plan = build_run_plan(config)
    write_run_commands(config, plan)
    write_metadata(config, plan)

    if config.dry_run:
        write_summary_files(config.out_root, planned_rows(plan, config))
        print("dry-run wrote {}".format(config.out_root))
        return 0

    if config.build:
        build_binaries(config)

    rows: List[Dict[str, Any]] = []
    had_error = False
    for entry in plan:
        variant: RegressionVariant = entry["variant"]
        variant_dir: Path = entry["variant_dir"]
        port: int = entry["port"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        reset_runtime_dirs(variant_dir)
        server = None
        log_handle = None
        completed_prompt_lens = set()
        try:
            validate_variant_paths(variant)
            log_path = variant_dir / "server.log"
            log_handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("MLX_DIR", str(config.mlx_dir))
            env.setdefault("RUST_LOG", config.rust_log)
            server = subprocess.Popen(
                entry["serve_cmd"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
            wait_ready(config.host, port, config.startup_timeout_secs, log_path, server)
            load_response = post_json(
                config.host,
                port,
                "/admin/api/models/load",
                entry["load_payload"],
                config.startup_timeout_secs,
            )
            (variant_dir / "load-response.json").write_text(
                json.dumps(load_response, indent=2), encoding="utf-8"
            )
            if not load_response.get("success"):
                raise RuntimeError("model load failed: {}".format(load_response))

            before_health = fetch_json(config.host, port, "/healthz", config.request_timeout_secs)
            (variant_dir / "healthz-before.json").write_text(
                json.dumps(before_health, indent=2), encoding="utf-8"
            )
            for prompt_len, bench_cmd in zip(config.prompt_lens, entry["bench_cmds"]):
                bench_json = variant_dir / "bench-pp{}.json".format(prompt_len)
                bench_stderr = variant_dir / "bench-pp{}.stderr.log".format(prompt_len)
                run_bench_command(config, bench_cmd, bench_json, bench_stderr)
                payload = json.loads(bench_json.read_text(encoding="utf-8"))
                after_health = fetch_json(config.host, port, "/healthz", config.request_timeout_secs)
                (variant_dir / "healthz-pp{}.json".format(prompt_len)).write_text(
                    json.dumps(after_health, indent=2), encoding="utf-8"
                )
                assert_health_delta(variant, before_health, after_health)
                rows.extend(summarize_bench_payload(variant, payload, after_health))
                before_health = after_health
                completed_prompt_lens.add(prompt_len)
        except Exception as exc:
            had_error = True
            remaining = [
                prompt_len
                for prompt_len in config.prompt_lens
                if prompt_len not in completed_prompt_lens
            ]
            rows.extend(error_rows(variant, remaining, exc))
            (variant_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            print("ERROR [{}]: {}".format(variant.name, exc), file=sys.stderr)
        finally:
            if server is not None:
                terminate_process(server)
            if log_handle is not None:
                log_handle.close()

    write_summary_files(config.out_root, rows)
    print("wrote {}".format(config.out_root))
    if had_error and not config.allow_failures:
        return 1
    return 0


def write_run_commands(config: RegressionConfig, plan: List[Dict[str, Any]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for entry in plan:
        variant: RegressionVariant = entry["variant"]
        port = entry["port"]
        lines.append("# {}".format(variant.name))
        lines.append(shlex.join(entry["serve_cmd"]))
        load_url = "http://{}:{}/admin/api/models/load".format(config.host, port)
        lines.append(
            shlex.join(
                [
                    "curl",
                    "-fsS",
                    "-X",
                    "POST",
                    load_url,
                    "-H",
                    "Content-Type: application/json",
                    "--data",
                    json.dumps(entry["load_payload"], sort_keys=True),
                ]
            )
        )
        for cmd in entry["bench_cmds"]:
            lines.append(shlex.join(cmd))
        lines.append("")
    path = config.out_root / "run_commands.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_metadata(config: RegressionConfig, plan: List[Dict[str, Any]]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(config.root),
        "variants": [entry["variant"].name for entry in plan],
        "prompt_lens": list(config.prompt_lens),
        "max_tokens": config.max_tokens,
        "duration_secs": config.duration_secs,
        "warmup_duration_secs": config.warmup_duration_secs,
        "prefix_cache_probe": config.prefix_cache_probe,
    }
    (config.out_root / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def planned_rows(plan: List[Dict[str, Any]], config: RegressionConfig) -> List[Dict[str, Any]]:
    rows = []
    for entry in plan:
        variant: RegressionVariant = entry["variant"]
        for prompt_len in config.prompt_lens:
            row = error_row(variant, prompt_len, "dry-run planned only")
            row["max_tokens"] = config.max_tokens
            row["status"] = "planned"
            row["notes"] = "dry-run"
            rows.append(row)
    return rows


def error_rows(
    variant: RegressionVariant,
    prompt_lens: Sequence[int],
    exc: Exception,
) -> List[Dict[str, Any]]:
    return [error_row(variant, prompt_len, str(exc)) for prompt_len in prompt_lens]


def error_row(variant: RegressionVariant, prompt_len: int, notes: str) -> Dict[str, Any]:
    return {
        "variant": variant.name,
        "model": variant.model_label,
        "prompt_len": prompt_len,
        "max_tokens": 0,
        "b_max": variant.b_max,
        "concurrent": variant.concurrent,
        "n_requests": 0,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "itl_ms_p50": None,
        "itl_ms_p95": None,
        "early_itl_ms_p50": None,
        "early_itl_ms_p95": None,
        "tokens_per_sec": None,
        "req_per_sec": None,
        "finish_reason_summary": "",
        "memory_budget_exceeded_count": None,
        "kv_cache_budget_policy": "",
        "kv_cache_logical_cap_tokens": None,
        "kv_cache_resident_cap_tokens": None,
        "active_kv_degraded": None,
        "active_kv_swap_error_count": None,
        "active_kv_swap_out_count": None,
        "active_kv_swap_in_count": None,
        "mtp_prefill_count": None,
        "mtp_step_count": None,
        "mtp_drafted_tokens": None,
        "mtp_accepted_draft_tokens": None,
        "mlx_peak_mib": None,
        "status": "error",
        "notes": notes,
    }


def build_binaries(config: RegressionConfig) -> None:
    env = os.environ.copy()
    env.setdefault("MLX_DIR", str(config.mlx_dir))
    subprocess.run(
        ["cargo", "build", "--release", "-p", "ironmlx", "-p", "iron-bench"],
        cwd=str(config.root),
        env=env,
        check=True,
    )


def run_bench_command(
    config: RegressionConfig,
    cmd: Sequence[str],
    bench_json: Path,
    bench_stderr: Path,
) -> None:
    with bench_stderr.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            timeout=config.request_timeout_secs,
            check=False,
        )
    bench_json.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "iron-bench failed with exit code {} for {}".format(
                completed.returncode, shlex.join(cmd)
            )
        )


def wait_ready(
    host: str,
    port: int,
    timeout_secs: int,
    log_path: Path,
    process: subprocess.Popen,
) -> None:
    deadline = time.time() + timeout_secs
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "server exited before readiness with code {}. Log tail:\n{}".format(
                    process.returncode, tail_text(log_path)
                )
            )
        try:
            with urllib.request.urlopen(
                "http://{}:{}/health".format(host, port), timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(
        "server did not become ready on {}:{} within {}s: {}".format(
            host, port, timeout_secs, last_error
        )
    )


def fetch_json(host: str, port: int, path: str, timeout_secs: int) -> Dict[str, Any]:
    with urllib.request.urlopen(
        "http://{}:{}{}".format(host, port, path), timeout=timeout_secs
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(
    host: str,
    port: int,
    path: str,
    payload: Dict[str, Any],
    timeout_secs: int,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "http://{}:{}{}".format(host, port, path),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} for {}: {}".format(exc.code, path, body)) from exc


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.send_signal(signal.SIGKILL)
    except OSError:
        return
    process.wait(timeout=10)


def reset_runtime_dirs(variant_dir: Path) -> None:
    for child in ("prefix-cache", "active-kv"):
        path = variant_dir / child
        if path.exists():
            shutil.rmtree(str(path))
        path.mkdir(parents=True, exist_ok=True)


def validate_variant_paths(variant: RegressionVariant) -> None:
    if not variant.model_dir.is_dir():
        raise FileNotFoundError("model directory does not exist: {}".format(variant.model_dir))
    if not (variant.model_dir / "config.json").is_file():
        raise FileNotFoundError("model config.json missing: {}".format(variant.model_dir))
    if not (variant.model_dir / "tokenizer.json").is_file():
        raise FileNotFoundError("model tokenizer.json missing: {}".format(variant.model_dir))
    if not variant.drafter_dir.is_dir():
        raise FileNotFoundError("drafter directory does not exist: {}".format(variant.drafter_dir))
    if not (variant.drafter_dir / "config.json").is_file():
        raise FileNotFoundError("drafter config.json missing: {}".format(variant.drafter_dir))


def parse_args(argv: Optional[Sequence[str]]) -> RegressionConfig:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run Gemma4 drafter + Active KV heavy regression"
    )
    parser.add_argument("--out-root", type=Path, default=default_out_root(root))
    parser.add_argument("--serve-bin", type=Path, default=root / "target/release/ironmlx")
    parser.add_argument("--iron-bench-bin", type=Path, default=root / "target/release/iron-bench")
    parser.add_argument("--e4b-model-dir", type=Path, default=default_e4b_model_dir())
    parser.add_argument("--e4b-drafter-dir", type=Path, default=default_e4b_drafter_dir())
    parser.add_argument("--12b-model-dir", dest="gemma12b_model_dir", type=Path, default=default_12b_model_dir())
    parser.add_argument("--12b-drafter-dir", dest="gemma12b_drafter_dir", type=Path, default=default_12b_drafter_dir())
    parser.add_argument("--prompt-lens", default=",".join(str(v) for v in DEFAULT_PROMPT_LENS))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--duration", dest="duration_secs", type=int, default=20)
    parser.add_argument("--warmup-duration", dest="warmup_duration_secs", type=int, default=5)
    parser.add_argument("--startup-timeout", dest="startup_timeout_secs", type=int, default=180)
    parser.add_argument("--request-timeout", dest="request_timeout_secs", type=int, default=900)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=19180)
    parser.add_argument("--prefix-cache-block-size", type=int, default=16)
    parser.add_argument("--prefix-lru-cache-max-bytes", type=int)
    parser.add_argument("--ssd-prefix-cache-max-gb", type=float)
    parser.add_argument("--mlx-dir", type=Path, default=Path.home() / ".local" / "mlx")
    parser.add_argument("--rust-log", default="info")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--prefix-cache-probe", action="store_true")
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument("--build", dest="build", action="store_true")
    build_group.add_argument("--no-build", dest="build", action="store_false")
    parser.set_defaults(build=False)
    args = parser.parse_args(argv)

    return RegressionConfig(
        root=root,
        out_root=args.out_root,
        serve_bin=args.serve_bin,
        iron_bench_bin=args.iron_bench_bin,
        e4b_model_dir=args.e4b_model_dir,
        e4b_drafter_dir=args.e4b_drafter_dir,
        gemma12b_model_dir=args.gemma12b_model_dir,
        gemma12b_drafter_dir=args.gemma12b_drafter_dir,
        prompt_lens=parse_int_csv(args.prompt_lens),
        max_tokens=args.max_tokens,
        duration_secs=args.duration_secs,
        warmup_duration_secs=args.warmup_duration_secs,
        startup_timeout_secs=args.startup_timeout_secs,
        request_timeout_secs=args.request_timeout_secs,
        host=args.host,
        port_base=args.port_base,
        prefix_cache_block_size=args.prefix_cache_block_size,
        prefix_lru_cache_max_bytes=args.prefix_lru_cache_max_bytes,
        ssd_prefix_cache_max_gb=args.ssd_prefix_cache_max_gb,
        mlx_dir=args.mlx_dir,
        rust_log=args.rust_log,
        build=args.build,
        dry_run=args.dry_run,
        allow_failures=args.allow_failures,
        prefix_cache_probe=args.prefix_cache_probe,
        variant_names=tuple(args.variant),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = parse_args(argv)
        return run_regression(config)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


def parse_int_csv(raw: str) -> Tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("prompt lengths must be > 0")
        values.append(value)
    if not values:
        raise ValueError("at least one prompt length is required")
    return tuple(values)


def default_out_root(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return root / "docs/benchmarks/gemma4-drafter-active-kv-regression" / stamp


def default_e4b_model_dir() -> Path:
    return env_or_latest_snapshot(
        "IRONMLX_GEMMA4_E4B_MODEL_DIR",
        [
            hf_cache_repo("mlx-community", "gemma-4-E4B-it-qat-4bit"),
            hf_cache_repo("mlx-community", "gemma-4-E4B-it-4bit"),
        ],
    )


def default_e4b_drafter_dir() -> Path:
    return env_or_latest_snapshot(
        "IRONMLX_GEMMA4_E4B_DRAFTER_DIR",
        [hf_cache_repo("mlx-community", "gemma-4-E4B-it-qat-assistant-4bit")],
    )


def default_12b_model_dir() -> Path:
    return env_or_latest_snapshot(
        "IRONMLX_GEMMA4_12B_MODEL_DIR",
        [hf_cache_repo("mlx-community", "gemma-4-12B-it-4bit")],
    )


def default_12b_drafter_dir() -> Path:
    return env_or_latest_snapshot(
        "IRONMLX_GEMMA4_12B_DRAFTER_DIR",
        [hf_cache_repo("mlx-community", "gemma-4-12B-it-assistant-4bit")],
    )


def env_or_latest_snapshot(env_name: str, repo_roots: Sequence[Path]) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser()
    for root in repo_roots:
        snapshot = latest_snapshot(root)
        if snapshot is not None:
            return snapshot
    return repo_roots[0]


def hf_cache_repo(namespace: str, repo: str) -> Path:
    return Path.home() / ".ironmlx/models" / "models--{}--{}".format(namespace, repo) / "snapshots"


def latest_snapshot(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    snapshots = sorted(path for path in root.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return snapshots[-1]


def int_value(payload: Dict[str, Any], path: str) -> int:
    value = nested(payload, path)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def bool_value(payload: Dict[str, Any], path: str) -> bool:
    return bool(nested(payload, path))


def str_value(payload: Dict[str, Any], path: str) -> str:
    value = nested(payload, path)
    return "" if value is None else str(value)


def nested_float(payload: Dict[str, Any], path: str) -> Optional[float]:
    value = nested(payload, path)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / BYTES_PER_MIB


def expected_effective_logical_cap(
    variant: RegressionVariant, health_payload: Dict[str, Any]
) -> int:
    model_context = int_value(health_payload, "model.max_position_embeddings")
    if model_context > 0:
        return min(variant.max_cache_cap, model_context)
    return variant.max_cache_cap


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "{:.{}f}".format(value, digits)
    return str(value)


def active_kv_status(value: Any) -> str:
    if value is None:
        return "n/a"
    return "degraded" if value else "ok"


def format_gib_arg(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return data[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())
