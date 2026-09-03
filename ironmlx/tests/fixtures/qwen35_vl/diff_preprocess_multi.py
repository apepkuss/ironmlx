#!/usr/bin/env python
"""P6.6 per-image preprocess diff (Gate 1).

Runs the diff_preprocess routine once per image (auto-discovers N from
the number of `image_*_pv.safetensors` files in --py) and emits a combined
report with one verdict line per image (1A, 1B, 1C, ...).

Each per-image diff treats the ironmlx-side preprocess output (vlmlayout
[N_i, 1536] C-major) against mlx-vlm's pre-split slice
(image_{i}_pv.safetensors).

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_preprocess_multi.py \\
        --py /tmp/p6_diff_multi/python \\
        --iron /tmp/p6_diff_multi/ironmlx_pre \\
        --out /path/to/p6_6_preprocess_report.md \\
        --gate 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_tensor(path: Path) -> np.ndarray:
    arr = mx.load(str(path))
    t = arr["tensor"] if isinstance(arr, dict) else arr
    t = t.astype(mx.float32)
    mx.eval(t)
    return np.array(t)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "count_above_1e-3": int((d > 1e-3).sum()),
        "count_above_1e-2": int((d > 1e-2).sum()),
        "total": int(d.size),
    }


def diff_one(vlm: Path, iron: Path) -> dict:
    a = load_tensor(vlm)
    b = load_tensor(iron)
    if a.shape != b.shape:
        return {"error": f"shape mismatch: vlm {a.shape} vs iron {b.shape}"}
    return diff_stats(a, b)


def gate_letter(i: int) -> str:
    # 0 -> 'A', 1 -> 'B', 2 -> 'C', ...
    return chr(ord("A") + i)


def render(image_id: int, stats: dict, gate: float) -> list[str]:
    if "error" in stats:
        return [f"## image_{image_id}", "", f"**ERROR**: {stats['error']}", ""]
    pass_gate = stats["max"] < gate
    return [
        f"## image_{image_id}",
        "",
        f"- max: {stats['max']:.6f}",
        f"- mean: {stats['mean']:.6f}",
        f"- p99: {stats['p99']:.6f}",
        f"- count > 1e-3: {stats['count_above_1e-3']} / {stats['total']}",
        f"- Gate 1{gate_letter(image_id)} verdict: **{'PASS' if pass_gate else 'FAIL'}**",
        "",
    ]


def discover_n(py_dir: Path) -> int:
    found = sorted(py_dir.glob("image_*_pv.safetensors"))
    return len(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path,
                        help="dir with image_{i}_pv.safetensors (i = 0 .. N-1)")
    parser.add_argument("--iron", required=True, type=Path,
                        help="dir with image_{i}_pv_vlmlayout.safetensors (i = 0 .. N-1)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate", type=float, default=0.05)
    parser.add_argument("--n-images", type=int, default=0,
                        help="Number of images; 0 = auto-discover from --py")
    args = parser.parse_args()

    n_images = args.n_images if args.n_images > 0 else discover_n(args.py)
    if n_images < 1:
        print(f"ERROR: no image_*_pv.safetensors found in {args.py}", file=sys.stderr)
        return 1

    lines = ["# P6.6 Multi-Image Preprocess Diff (Gate 1)", "",
             f"- Gate 1 threshold: < {args.gate}",
             f"- N images: {n_images}", ""]
    overall_pass = True
    for i in range(n_images):
        vlm = args.py / f"image_{i}_pv.safetensors"
        iron = args.iron / f"image_{i}_pv_vlmlayout.safetensors"
        if not vlm.exists() or not iron.exists():
            lines.append(f"## image_{i} — missing input")
            lines.append(f"- vlm exists: {vlm.exists()}")
            lines.append(f"- iron exists: {iron.exists()}")
            lines.append("")
            overall_pass = False
            continue
        stats = diff_one(vlm, iron)
        lines.extend(render(i, stats, args.gate))
        if "error" in stats or stats.get("max", 99.0) >= args.gate:
            overall_pass = False

    lines.append(f"## Overall Gate 1: **{'PASS' if overall_pass else 'FAIL'}**")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[diff_preprocess_multi] overall {'PASS' if overall_pass else 'FAIL'}; report → {args.out}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
