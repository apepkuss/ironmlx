#!/usr/bin/env python3
"""Archive the exact dirty candidate sources and sanitized reproduction identity."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from run_b1_api_sessions import ARTIFACTS, DRAFT, MLX_LIB, TARGET

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def identity(path):
    path = Path(path)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(path=str(path), size=path.stat().st_size, sha256=digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    changed = set(git("diff", "--name-only", "HEAD", "-z").decode().split("\0"))
    changed.update(
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    files = []
    for name in sorted(changed - {""}):
        source = ROOT / name
        if not source.is_file():
            files.append(dict(path=name, deleted=True))
            continue
        dest = args.output / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        files.append(dict(relative=name, **identity(source)))
    (args.output / "tracked.patch").write_bytes(git("diff", "--binary", "HEAD"))
    report = dict(
        head=git("rev-parse", "HEAD").decode().strip(),
        branch=git("branch", "--show-current").decode().strip(),
        files=files,
        executable=identity(ROOT / "target/release/ironmlx"),
        diagnostic=identity(ROOT / "target/release/dflash2-lane-diagnostic"),
        mlx_library=identity(MLX_LIB / "libmlx.a"),
        metallib=identity(MLX_LIB / "mlx.metallib"),
        rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
        cargo=subprocess.check_output(["cargo", "--version"], text=True).strip(),
        macos=subprocess.check_output(["sw_vers"], text=True),
        rival_sources={
            name: [
                identity(f)
                for f in sorted(directory.rglob("*"))
                if f.is_file()
                and f.suffix in (".py", ".metal")
                and "__pycache__" not in f.parts
            ]
            for name, directory in {
                "tensorfold": ARTIFACTS / "tensorfold/source-v0.3.6.2/src",
                "omlx": ARTIFACTS / "omlx/oMLX.app/Contents/Resources/omlx",
                "splash_server": ARTIFACTS
                / "splash/bottle/splash/1.1.0/libexec/server",
            }.items()
        },
        splash_engine=[
            identity(ARTIFACTS / "splash/bottle/splash/1.1.0/libexec/engine" / name)
            for name in ("splash", "splash.metallib")
        ],
        models={
            name: dict(
                path=str(directory),
                metadata=[identity(f) for f in sorted(directory.glob("*.json"))],
                shards=[
                    dict(path=str(f.resolve()), bytes=f.stat().st_size)
                    for f in sorted(directory.glob("*.safetensors"))
                ],
            )
            for name, directory in [("target", TARGET), ("drafter", DRAFT)]
        },
        note="Base commit plus tracked.patch plus sources reproduce the dirty candidate. Sources are archived without credentials or runtime model settings.",
    )
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
