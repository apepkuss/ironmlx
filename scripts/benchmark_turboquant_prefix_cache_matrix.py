#!/usr/bin/env python3
"""Run the TurboQuant x Prefix Cache benchmark matrix.

This runner keeps iron-bench engine-neutral. It owns the ironmlx-specific
matrix: baseline, TurboQuant only, Prefix Cache only, and the combined packed
TurboQuant Prefix Cache path.
"""

import argparse
import csv
import json
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BYTES_PER_MIB = 1024 * 1024
DEFAULT_MODEL_ROOT = (
    Path.home()
    / ".ironmlx"
    / "models"
    / "models--mlx-community--Qwen3.5-4B-MLX-4bit"
    / "snapshots"
)


@dataclass(frozen=True)
class MatrixVariant:
    name: str
    label: str
    kv_quant: Optional[str]
    prefix_cache: bool
    description: str


@dataclass
class MatrixConfig:
    root: Path
    model_dir: Path
    out_root: Path
    serve_bin: Path
    iron_bench_bin: Path
    prompt_lens: Tuple[int, ...] = (2048, 8192)
    max_tokens: int = 16
    runs: int = 3
    warmup: int = 0
    model_name: str = "qwen3.5-4b"
    host: str = "127.0.0.1"
    port_base: int = 19080
    max_sequences: int = 1
    max_cache_cap: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    scheduler_profile: Optional[Path] = None
    kv_quant: str = "k3v4"
    prefix_cache_block_size: int = 16
    ssd_prefix_cache_max_gb: Optional[float] = 10.0
    prefix_lru_cache_max_bytes: Optional[int] = None
    timeout_secs: int = 900
    startup_timeout_secs: int = 180
    mlx_dir: Path = Path.home() / ".local" / "mlx"
    rust_log: str = "info"
    build: bool = False
    dry_run: bool = False
    allow_failures: bool = False
    extra_serve_args: Tuple[str, ...] = field(default_factory=tuple)
    extra_bench_args: Tuple[str, ...] = field(default_factory=tuple)
    variant_names: Tuple[str, ...] = field(default_factory=tuple)


def build_matrix(kv_quant: str) -> List[MatrixVariant]:
    return [
        MatrixVariant(
            name="baseline_dense",
            label="Dense baseline",
            kv_quant=None,
            prefix_cache=False,
            description="No TurboQuant and no Prefix Cache.",
        ),
        MatrixVariant(
            name="turboquant_only",
            label="TurboQuant only",
            kv_quant=kv_quant,
            prefix_cache=False,
            description="Runtime KV cache uses TurboQuant, Prefix Cache disabled.",
        ),
        MatrixVariant(
            name="prefix_cache_only",
            label="Prefix Cache only",
            kv_quant=None,
            prefix_cache=True,
            description="Paged SSD Prefix Cache enabled with dense runtime KV.",
        ),
        MatrixVariant(
            name="turboquant_prefix_cache",
            label="TurboQuant + Prefix Cache",
            kv_quant=kv_quant,
            prefix_cache=True,
            description="Runtime KV uses TurboQuant and Prefix Cache persists packed tensors.",
        ),
    ]


def build_serve_command(
    config: MatrixConfig,
    variant: MatrixVariant,
    port: int,
    variant_dir: Path,
) -> List[str]:
    cmd = [
        str(config.serve_bin),
        "serve",
        "--model",
        str(config.model_dir),
        "--host",
        config.host,
        "--port",
        str(port),
        "--max-sequences",
        str(config.max_sequences),
    ]
    if config.max_cache_cap is not None:
        cmd.extend(["--max-cache-cap", str(config.max_cache_cap)])
    if config.prefill_chunk_size is not None:
        cmd.extend(["--prefill-chunk-size", str(config.prefill_chunk_size)])
    if config.scheduler_profile is not None:
        cmd.extend(["--scheduler-profile", str(config.scheduler_profile)])
    if variant.kv_quant is not None:
        cmd.extend(["--kv-quant", variant.kv_quant])
    if variant.prefix_cache:
        cache_dir = variant_dir / "prefix-cache"
        cmd.extend(
            [
                "--paged-prefix-cache-dir",
                str(cache_dir),
                "--paged-prefix-cache-block-size",
                str(config.prefix_cache_block_size),
            ]
        )
        if config.ssd_prefix_cache_max_gb is not None:
            cmd.extend(
                [
                    "--ssd-prefix-cache-max-gb",
                    format_gib_arg(config.ssd_prefix_cache_max_gb),
                ]
            )
        if config.prefix_lru_cache_max_bytes is not None:
            cmd.extend(
                [
                    "--prefix-lru-cache-max-bytes",
                    str(config.prefix_lru_cache_max_bytes),
                ]
            )
    cmd.extend(config.extra_serve_args)
    return cmd


def build_bench_command(config: MatrixConfig, port: int, prompt_len: int) -> List[str]:
    cmd = [
        str(config.iron_bench_bin),
        "--target",
        "ironmlx=http://{}:{}".format(config.host, port),
        "--model-dir",
        str(config.model_dir),
        "--model",
        config.model_name,
        "--prompt-len",
        str(prompt_len),
        "--max-tokens",
        str(config.max_tokens),
        "--runs",
        str(config.runs),
        "--warmup",
        str(config.warmup),
        "--timeout",
        str(config.timeout_secs),
        "--prefix-cache-probe",
        "--format",
        "json",
    ]
    cmd.extend(config.extra_bench_args)
    return cmd


def summarize_bench_payload(
    variant: MatrixVariant,
    prompt_len: int,
    payload: Dict[str, Any],
    cache_bytes: int,
    health_payload: Optional[Dict[str, Any]],
    cache_dir: Path,
) -> List[Dict[str, Any]]:
    raw_runs = payload.get("raw_runs") or []
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for run in raw_runs:
        pp = int(run.get("pp_target") or prompt_len)
        tg = int(run.get("tg_target") or 0)
        grouped.setdefault((pp, tg), []).append(run)

    rows = []
    for (pp, tg), runs in sorted(grouped.items()):
        cold = [
            run
            for run in runs
            if run.get("prefix_cache_probe_phase") == "cold_or_miss_candidate"
        ]
        warm = [
            run
            for run in runs
            if run.get("prefix_cache_probe_phase") == "warm_hit_candidate"
        ]
        if not cold and runs:
            cold = [min(runs, key=lambda item: int(item.get("run_idx") or 0))]
        if not warm:
            warm = [
                run
                for run in runs
                if int(run.get("run_idx") or 0) > int(cold[0].get("run_idx") or 0)
            ]
        sample_runs = warm or runs
        cold_ttft = cold[0].get("ttft_ms") if cold else None
        warm_ttft = median_number(run.get("ttft_ms") for run in warm)
        row = {
            "variant": variant.name,
            "variant_label": variant.label,
            "prompt_len": pp,
            "max_tokens": tg,
            "kv_quant": variant.kv_quant or "none",
            "prefix_cache": variant.prefix_cache,
            "cold_ttft_ms": as_float(cold_ttft),
            "warm_ttft_ms_median": warm_ttft,
            "warm_speedup_vs_cold": ratio(as_float(cold_ttft), warm_ttft),
            "warm_tg_tps_median": median_number(run.get("tg_tps") for run in warm),
            "warm_tpot_ms_median": median_number(run.get("tpot_ms") for run in warm),
            "warm_pp_tps_median": median_number(run.get("pp_tps") for run in warm),
            "warm_e2e_s_median": median_number(run.get("e2e_s") for run in warm),
            "warm_cached_tokens_median": median_number(
                run.get("cached_tokens") for run in warm
            ),
            "prompt_tokens_local_median": median_number(
                run.get("prompt_tokens_local") for run in sample_runs
            ),
            "prompt_tokens_server_median": median_number(
                run.get("prompt_tokens_server") for run in sample_runs
            ),
            "completion_tokens_server_median": median_number(
                run.get("completion_tokens_server") for run in sample_runs
            ),
            "cache_bytes": cache_bytes if variant.prefix_cache else 0,
            "cache_dir": str(cache_dir) if variant.prefix_cache else "",
            "memory_peak_mb": health_memory_peak_mb(health_payload),
            "status": "ok",
            "notes": "",
        }
        rows.append(row)
    return rows


def write_summary_files(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=False), encoding="utf-8"
    )

    fieldnames = [
        "variant",
        "variant_label",
        "prompt_len",
        "max_tokens",
        "kv_quant",
        "prefix_cache",
        "cold_ttft_ms",
        "warm_ttft_ms_median",
        "warm_speedup_vs_cold",
        "warm_tg_tps_median",
        "warm_tpot_ms_median",
        "warm_pp_tps_median",
        "warm_e2e_s_median",
        "warm_cached_tokens_median",
        "prompt_tokens_local_median",
        "prompt_tokens_server_median",
        "completion_tokens_server_median",
        "cache_bytes",
        "cache_dir",
        "memory_peak_mb",
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
        "# TurboQuant x Prefix Cache Benchmark Matrix",
        "",
        "| variant | pp | tg | kv | prefix | cold ttft ms | warm ttft ms | warm decode tps | warm e2e s | cached tokens | cache MiB | peak MLX MiB | status |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {prompt_len} | {max_tokens} | {kv_quant} | {prefix_cache} | {cold} | {warm} | {decode} | {e2e} | {cached} | {cache_mib} | {peak} | {status} |".format(
                variant=row.get("variant", ""),
                prompt_len=row.get("prompt_len", ""),
                max_tokens=row.get("max_tokens", ""),
                kv_quant=row.get("kv_quant", ""),
                prefix_cache="yes" if row.get("prefix_cache") else "no",
                cold=fmt(row.get("cold_ttft_ms")),
                warm=fmt(row.get("warm_ttft_ms_median")),
                decode=fmt(row.get("warm_tg_tps_median")),
                e2e=fmt(row.get("warm_e2e_s_median"), digits=4),
                cached=fmt(row.get("warm_cached_tokens_median")),
                cache_mib=fmt(bytes_to_mib(row.get("cache_bytes"))),
                peak=fmt(row.get("memory_peak_mb")),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `cold` is the first `--prefix-cache-probe` measured run in a cell.",
            "- `warm` is the median of subsequent measured runs in the same cell.",
            "- Prefix Cache variants use `--paged-prefix-cache-dir` with a fresh cache directory per variant.",
            "- TurboQuant + Prefix Cache validates the packed native persistence path.",
            "",
        ]
    )
    return "\n".join(lines)


def build_run_plan(config: MatrixConfig) -> List[Dict[str, Any]]:
    plan = []
    variants = build_matrix(config.kv_quant)
    if config.variant_names:
        requested = set(config.variant_names)
        known = {variant.name for variant in variants}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError("unknown variant(s): {}".format(", ".join(unknown)))
        variants = [variant for variant in variants if variant.name in requested]
    for index, variant in enumerate(variants):
        port = config.port_base + index
        variant_dir = config.out_root / variant.name
        plan.append(
            {
                "variant": variant,
                "port": port,
                "variant_dir": variant_dir,
                "serve_cmd": build_serve_command(config, variant, port, variant_dir),
                "bench_cmds": [
                    build_bench_command(config, port, prompt_len)
                    for prompt_len in config.prompt_lens
                ],
            }
        )
    return plan


def write_run_commands(out_root: Path, plan: List[Dict[str, Any]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for entry in plan:
        variant = entry["variant"]
        lines.append("# {}".format(variant.name))
        lines.append(shlex.join(entry["serve_cmd"]))
        for cmd in entry["bench_cmds"]:
            lines.append(shlex.join(cmd))
        lines.append("")
    path = out_root / "run_commands.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def run_matrix(config: MatrixConfig) -> int:
    config.out_root.mkdir(parents=True, exist_ok=True)
    plan = build_run_plan(config)
    write_run_commands(config.out_root, plan)
    write_metadata(config, plan)

    if config.dry_run:
        print("dry-run wrote {}".format(config.out_root))
        return 0

    if config.build:
        build_binaries(config)

    rows: List[Dict[str, Any]] = []
    had_error = False
    for entry in plan:
        variant = entry["variant"]
        port = entry["port"]
        variant_dir = entry["variant_dir"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = variant_dir / "prefix-cache"
        if cache_dir.exists():
            shutil.rmtree(str(cache_dir))
        if variant.prefix_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)

        log_path = variant_dir / "server.log"
        server = None
        log_handle = None
        completed_prompt_lens = set()
        try:
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
            initial_health = fetch_json(config.host, port, "/healthz")
            (variant_dir / "initial-healthz.json").write_text(
                json.dumps(initial_health, indent=2), encoding="utf-8"
            )

            for prompt_len, bench_cmd in zip(config.prompt_lens, entry["bench_cmds"]):
                bench_json = variant_dir / "bench-pp{}.json".format(prompt_len)
                bench_stderr = variant_dir / "bench-pp{}.stderr.log".format(prompt_len)
                run_bench_command(config, bench_cmd, bench_json, bench_stderr)
                payload = json.loads(bench_json.read_text(encoding="utf-8"))
                health = fetch_json(config.host, port, "/healthz")
                (variant_dir / "healthz-pp{}.json".format(prompt_len)).write_text(
                    json.dumps(health, indent=2), encoding="utf-8"
                )
                rows.extend(
                    summarize_bench_payload(
                        variant=variant,
                        prompt_len=prompt_len,
                        payload=payload,
                        cache_bytes=directory_size(cache_dir) if variant.prefix_cache else 0,
                        health_payload=health,
                        cache_dir=cache_dir,
                    )
                )
                completed_prompt_lens.add(prompt_len)
        except Exception as exc:
            had_error = True
            remaining_prompt_lens = tuple(
                prompt_len
                for prompt_len in config.prompt_lens
                if prompt_len not in completed_prompt_lens
            )
            rows.extend(error_rows(config, variant, exc, remaining_prompt_lens))
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


def run_bench_command(
    config: MatrixConfig,
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
            timeout=config.timeout_secs,
            check=False,
        )
    bench_json.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "iron-bench failed with exit code {} for {}".format(
                completed.returncode, shlex.join(cmd)
            )
        )


def build_binaries(config: MatrixConfig) -> None:
    env = os.environ.copy()
    env.setdefault("MLX_DIR", str(config.mlx_dir))
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "ironmlx",
            "-p",
            "iron-bench",
        ],
        cwd=str(config.root),
        env=env,
        check=True,
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
            tail = tail_text(log_path, max_chars=4000)
            raise RuntimeError(
                "server exited before readiness with code {}. Log tail:\n{}".format(
                    process.returncode, tail
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


def fetch_json(host: str, port: int, path: str) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(
            "http://{}:{}{}".format(host, port, path), timeout=10
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


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


def error_rows(
    config: MatrixConfig,
    variant: MatrixVariant,
    exc: Exception,
    prompt_lens: Sequence[int],
) -> List[Dict[str, Any]]:
    return [
        {
            "variant": variant.name,
            "variant_label": variant.label,
            "prompt_len": prompt_len,
            "max_tokens": config.max_tokens,
            "kv_quant": variant.kv_quant or "none",
            "prefix_cache": variant.prefix_cache,
            "cold_ttft_ms": None,
            "warm_ttft_ms_median": None,
            "warm_speedup_vs_cold": None,
            "warm_tg_tps_median": None,
            "warm_tpot_ms_median": None,
            "warm_pp_tps_median": None,
            "warm_e2e_s_median": None,
            "warm_cached_tokens_median": None,
            "prompt_tokens_local_median": None,
            "prompt_tokens_server_median": None,
            "completion_tokens_server_median": None,
            "cache_bytes": 0,
            "cache_dir": "",
            "memory_peak_mb": None,
            "status": "error",
            "notes": str(exc),
        }
        for prompt_len in prompt_lens
    ]


def write_metadata(config: MatrixConfig, plan: List[Dict[str, Any]]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(config.model_dir),
        "model_name": config.model_name,
        "prompt_lens": list(config.prompt_lens),
        "max_tokens": config.max_tokens,
        "runs": config.runs,
        "warmup": config.warmup,
        "kv_quant": config.kv_quant,
        "variants": [
            {
                "name": entry["variant"].name,
                "label": entry["variant"].label,
                "kv_quant": entry["variant"].kv_quant or "none",
                "prefix_cache": entry["variant"].prefix_cache,
                "port": entry["port"],
                "description": entry["variant"].description,
            }
            for entry in plan
        ],
    }
    (config.out_root / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> MatrixConfig:
    root = Path(__file__).resolve().parent.parent
    default_out = (
        root
        / "docs"
        / "benchmarks"
        / "turboquant-prefix-cache-matrix"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description="Run the TurboQuant x Prefix Cache benchmark matrix."
    )
    parser.add_argument(
        "--model-dir", type=Path, default=default_model_dir() or Path("<model-dir>")
    )
    parser.add_argument("--model-name", default="qwen3.5-4b")
    parser.add_argument("--out-root", type=Path, default=default_out)
    parser.add_argument("--serve-bin", type=Path, default=root / "target/release/ironmlx")
    parser.add_argument(
        "--iron-bench-bin", type=Path, default=root / "target/release/iron-bench"
    )
    parser.add_argument("--prompt-len", "--prompt-lens", default="2048,8192")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=19080)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-cache-cap", type=int)
    parser.add_argument("--prefill-chunk-size", type=int)
    parser.add_argument("--scheduler-profile", type=Path)
    parser.add_argument("--kv-quant", default="k3v4")
    parser.add_argument("--prefix-cache-block-size", type=int, default=16)
    parser.add_argument("--ssd-prefix-cache-max-gb", type=int, default=10)
    parser.add_argument("--prefix-lru-cache-max-bytes", type=int)
    parser.add_argument("--timeout-secs", type=int, default=900)
    parser.add_argument("--startup-timeout-secs", type=int, default=180)
    parser.add_argument(
        "--mlx-dir",
        type=Path,
        default=Path(os.environ.get("MLX_DIR", str(Path.home() / ".local" / "mlx"))),
    )
    parser.add_argument("--rust-log", default=os.environ.get("RUST_LOG", "info"))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.name for variant in build_matrix("k3v4")],
        default=[],
        help="Run only the selected matrix variant. Repeat to select multiple variants.",
    )
    parser.add_argument("--extra-serve-arg", action="append", default=[])
    parser.add_argument("--extra-bench-arg", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.dry_run and not args.model_dir.exists():
        parser.error("model directory does not exist: {}".format(args.model_dir))
    if args.runs < 2:
        parser.error("--runs must be at least 2 so cold and warm probe runs exist")
    if args.max_sequences < 1:
        parser.error("--max-sequences must be >= 1")

    return MatrixConfig(
        root=root,
        model_dir=args.model_dir,
        out_root=args.out_root,
        serve_bin=args.serve_bin,
        iron_bench_bin=args.iron_bench_bin,
        prompt_lens=parse_prompt_lens(args.prompt_len),
        max_tokens=args.max_tokens,
        runs=args.runs,
        warmup=args.warmup,
        model_name=args.model_name,
        host=args.host,
        port_base=args.port_base,
        max_sequences=args.max_sequences,
        max_cache_cap=args.max_cache_cap,
        prefill_chunk_size=args.prefill_chunk_size,
        scheduler_profile=args.scheduler_profile,
        kv_quant=args.kv_quant,
        prefix_cache_block_size=args.prefix_cache_block_size,
        ssd_prefix_cache_max_gb=args.ssd_prefix_cache_max_gb,
        prefix_lru_cache_max_bytes=args.prefix_lru_cache_max_bytes,
        timeout_secs=args.timeout_secs,
        startup_timeout_secs=args.startup_timeout_secs,
        mlx_dir=args.mlx_dir,
        rust_log=args.rust_log,
        build=args.build,
        dry_run=args.dry_run,
        allow_failures=args.allow_failures,
        extra_serve_args=tuple(args.extra_serve_arg),
        extra_bench_args=tuple(args.extra_bench_arg),
        variant_names=tuple(args.variant),
    )


def default_model_dir() -> Optional[Path]:
    if "MODEL" in os.environ:
        return Path(os.environ["MODEL"])
    if not DEFAULT_MODEL_ROOT.exists():
        return None
    snapshots = sorted(
        item for item in DEFAULT_MODEL_ROOT.iterdir() if item.is_dir()
    )
    if not snapshots:
        return None
    return snapshots[-1]


def parse_prompt_lens(value: str) -> Tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one prompt length is required")
    return result


def format_gib_arg(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def median_number(values: Iterable[Any]) -> Optional[float]:
    numbers = [as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return float(statistics.median(numbers))


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def health_memory_peak_mb(health_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not health_payload:
        return None
    memory = health_payload.get("memory") or {}
    if "mlx_peak_mb" in memory:
        return as_float(memory.get("mlx_peak_mb"))
    peak_bytes = as_float(memory.get("mlx_peak_bytes"))
    if peak_bytes is None:
        return None
    return peak_bytes / BYTES_PER_MIB


def bytes_to_mib(value: Any) -> Optional[float]:
    number = as_float(value)
    if number is None:
        return None
    return number / BYTES_PER_MIB


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return "{:.{}f}".format(number, digits)


def tail_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    return run_matrix(config)


if __name__ == "__main__":
    raise SystemExit(main())
