#!/usr/bin/env python3
"""Run fixed local release artifacts sequentially; only terminate owned processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("/Users/xin/workspace/b1-rival-benchmark/artifacts")
TARGET = Path(
    "/Users/xin/.ironmlx/models/huggingface/mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
)
DRAFT = Path(
    "/Users/xin/.ironmlx/models/huggingface/z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
)
REPORTS = ROOT / "reports/b1-api-performance"
TF = ARTIFACTS / "tensorfold"
PYTHON = TF / "venv/bin/python"
ORDER = [
    ("ironmlx", "omlx", "splash", "tensorfold"),
    ("omlx", "tensorfold", "ironmlx", "splash"),
    ("splash", "ironmlx", "tensorfold", "omlx"),
    ("tensorfold", "splash", "omlx", "ironmlx"),
]


def file_identity(path):
    path = Path(path)
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    )


def initialize_omlx_config():
    runtime = REPORTS / "omlx-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    settings = runtime / "settings.json"
    if not settings.exists():
        settings.write_bytes(
            (ROOT / "scripts/fixtures/b1-omlx-settings.json").read_bytes()
        )
    models = runtime / "model_settings.json"
    if not models.exists():
        models.write_text(
            json.dumps(
                dict(
                    version=1,
                    models={
                        "qwen38-27b-4bit": dict(
                            dflash_enabled=True,
                            dflash_draft_model=str(DRAFT),
                            dflash_draft_quant_enabled=True,
                            dflash_draft_quant_weight_bits=4,
                            dflash_draft_quant_activation_bits=16,
                            dflash_draft_quant_group_size=64,
                            dflash_in_memory_cache=False,
                            dflash_ssd_cache=False,
                            dflash_verify_mode="adaptive",
                            mtp_enabled=False,
                        )
                    },
                ),
                indent=2,
            )
            + "\n"
        )


def effective_settings(app):
    if app != "omlx":
        return (
            None  # Other services' effective settings are command-line and log based.
        )
    runtime = REPORTS / "omlx-runtime"
    settings = json.loads((runtime / "settings.json").read_text())
    # Never copy authentication, credentials or unrelated application settings.
    allowed = ("server", "scheduler", "cache", "sampling", "memory")
    return dict(
        settings={k: settings.get(k) for k in allowed},
        models=json.loads((runtime / "model_settings.json").read_text()),
    )


def configuration(app):
    env = dict(os.environ)
    # Each artifact must use its own MLX library, never an inherited search path.
    for key in ("PYTHONPATH", "PYTHONHOME", "DYLD_LIBRARY_PATH", "MLX_METAL_PATH"):
        env.pop(key, None)
    model, omit = "benchmark", False
    if app == "ironmlx":
        command = [
            str(ROOT / "target/release/ironmlx"),
            "--mlx-metallib",
            "/Users/xin/.local/mlx/lib/mlx.metallib",
            "serve",
            "--model",
            str(TARGET),
            "--model-id",
            model,
            "--dflash2-model-dir",
            str(DRAFT),
            "--dflash2-block-size",
            "8",
            "--max-sequences",
            "1",
            "--admission-deadline-ms",
            "0",
            "--max-cache-cap",
            "8192",
            "--port",
            "18480",
        ]
        env["DYLD_LIBRARY_PATH"] = "/Users/xin/.local/mlx/lib"
        omit = True
    elif app == "omlx":
        command = [
            str(ARTIFACTS / "omlx/oMLX.app/Contents/MacOS/omlx-cli"),
            "serve",
            "--base-path",
            str(REPORTS / "omlx-runtime"),
            "--model-dir",
            "/Users/xin/workspace/b1-rival-benchmark/runtime/omlx/models",
            "--host",
            "127.0.0.1",
            "--port",
            "18480",
            "--no-cache",
            "--max-concurrent-requests",
            "1",
        ]
        model = "qwen38-27b-4bit"
    elif app == "tensorfold":
        env["PYTHONPATH"] = str(TF / "source-v0.3.6.2/src")
        command = [
            str(PYTHON),
            "-m",
            "tensorfold.cli",
            "serve",
            str(TARGET),
            "--drafter",
            str(DRAFT),
            "--drafter-bits",
            "4",
            "--lane-kernels",
            "on",
            "--parallel",
            "1",
            "--context",
            "8192",
            "--max-tokens",
            "4096",
            "--no-thinking",
            "--temperature",
            "0",
            "--name",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            "18480",
            "--snapshot-dir",
            "none",
            "--prompt-cache-gib",
            "0",
            "--max-snapshots",
            "0",
            "--no-update-check",
        ]
        omit = True
    elif app == "splash":
        runtime = ARTIFACTS / "splash/bottle/splash/1.1.0/libexec"
        prepared = Path(
            "/Users/xin/Library/Application Support/Splash/models/.resolved/0c766d93b06d3cbb2000e81376cd681455898449ec91cec5e9749d1ff0b1979d"
        )
        command = [
            str(runtime / "python/bin/python3"),
            "-u",
            str(runtime / "server/server.py"),
            str(prepared / "target"),
            str(prepared / "draft"),
            "--tokenizer",
            str(prepared / "tokenizer"),
            "--model",
            "mlx-community/Qwen3.8-27B-4bit",
            "--served-model-name",
            model,
            "--binary",
            str(runtime / "engine/splash"),
            "--host",
            "127.0.0.1",
            "--port",
            "18480",
            "--max-memory",
            "auto",
            "--max-context",
            "8192",
            "--default-reasoning-effort",
            "none",
            "--no-webui",
        ]
    else:
        raise ValueError(app)
    return command, env, model, omit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", default="ironmlx,omlx,splash,tensorfold")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--only", help="Development prompt ids only")
    parser.add_argument("--ironmlx-tree-nodes", type=int, default=0)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if "omlx" in args.apps.split(","):
        initialize_omlx_config()
    for session in range(args.start, args.start + args.sessions):
        for app in ORDER[session % 4]:
            if app not in args.apps.split(","):
                continue
            prefix = REPORTS / f"{args.label}-{app}-s{session}"
            if prefix.with_suffix(".json").exists():
                raise RuntimeError(f"Already exists: {prefix}")
            with socket.socket() as check:
                if check.connect_ex(("127.0.0.1", 18480)) == 0:
                    raise RuntimeError(
                        "Port 18480 already occupied; refusing to interfere"
                    )
            command, env, model, omit = configuration(app)
            if app == "ironmlx" and args.ironmlx_tree_nodes:
                command += ["--dflash2-tree-max-nodes", str(args.ironmlx_tree_nodes)]
            meta = dict(
                command=command,
                session=session,
                app=app,
                entrypoint=file_identity(command[0]),
                benchmark_client=file_identity(ROOT / "scripts/benchmark_b1_api.py"),
                benchmark_runner=file_identity(Path(__file__)),
                protocol=file_identity(ROOT / "docs/b1-api-performance-protocol.md"),
                platform=subprocess.getoutput("sw_vers"),
                hardware=subprocess.getoutput(
                    "system_profiler SPHardwareDataType -detailLevel mini"
                ).split("Serial Number")[0],
                power=subprocess.getoutput("pmset -g batt"),
                effective_settings=effective_settings(app),
                experimental_env={
                    k: v for k, v in env.items() if k.startswith("IRONMLX_EXPERIMENTAL")
                },
                ironmlx_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                dirty_diff=subprocess.check_output(
                    ["git", "diff", "--stat"], cwd=ROOT, text=True
                ),
            )
            prefix.with_suffix(".command.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
            with prefix.with_suffix(".server.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    ready = False
                    for _ in range(600):
                        if process.poll() is not None:
                            raise RuntimeError(f"{app} exited: {prefix}.server.log")
                        try:
                            with urllib.request.urlopen(
                                "http://127.0.0.1:18480/v1/models", timeout=1
                            ) as response:
                                if response.status == 200:
                                    ready = True
                                    break
                        except (OSError, TimeoutError):
                            pass
                        time.sleep(1)
                    if not ready:
                        raise TimeoutError(app + " startup")
                    client = [
                        str(PYTHON),
                        str(ROOT / "scripts/benchmark_b1_api.py"),
                        "--label",
                        args.label + "-" + app,
                        "--session",
                        str(session),
                        "--model",
                        model,
                        "--tokenizer",
                        str(TARGET),
                        "--output",
                        str(prefix.with_suffix(".json")),
                    ]
                    if omit:
                        client.append("--omit-reasoning-effort")
                    if args.only:
                        client.extend(["--only", args.only])
                    subprocess.run(client, cwd=ROOT, check=True)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            time.sleep(10)


if __name__ == "__main__":
    main()
