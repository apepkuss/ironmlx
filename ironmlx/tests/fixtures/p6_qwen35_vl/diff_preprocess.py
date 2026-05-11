"""P6.3a: focused byte-level diff between ironmlx and mlx-vlm preprocess outputs.

Unlike diff_pipeline.py (which does full 29-tensor vision-tower diff), this
tool only compares the two `00_*pv*` files and emits a Gate-1 verdict line
that the close-out report can ingest.

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_preprocess.py \
        --vlm /tmp/p6_diff/python/00_pixel_values.safetensors \
        --iron /tmp/p6_diff/ironmlx_pre/00_ironmlx_pv_vlmlayout.safetensors \
        --out /path/to/p6_3a_preprocess_report.md
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True, type=Path)
    parser.add_argument("--iron", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate", type=float, default=0.05, help="Gate 1 threshold")
    args = parser.parse_args()

    a = load_tensor(args.vlm)
    b = load_tensor(args.iron)
    if a.shape != b.shape:
        msg = f"shape mismatch: vlm {a.shape} vs iron {b.shape}"
        print(f"ERROR: {msg}", file=sys.stderr)
        args.out.write_text(f"# P6.3a Preprocess Diff\n\n**ERROR**: {msg}\n")
        return 2

    d = np.abs(a - b)
    stats = {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "count_above_1e-3": int((d > 1e-3).sum()),
        "count_above_1e-2": int((d > 1e-2).sum()),
        "total": int(d.size),
    }

    # Top 5 outliers
    flat_diff = d.flatten()
    k = min(5, len(flat_diff))
    idxs = np.argpartition(flat_diff, -k)[-k:]
    idxs = idxs[np.argsort(flat_diff[idxs])[::-1]]
    outliers = [
        {"idx": int(i),
         "vlm": float(a.flatten()[i]),
         "iron": float(b.flatten()[i]),
         "diff": float(flat_diff[i])}
        for i in idxs
    ]

    pass_gate = stats["max"] < args.gate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P6.3a Preprocess Diff (Gate 1)",
        "",
        f"- Tensor shape: {list(a.shape)}",
        f"- Gate 1 threshold: < {args.gate}",
        f"- Observed max_diff: **{stats['max']:.4f}**",
        f"- Gate 1 verdict: **{'PASS' if pass_gate else 'FAIL'}**",
        "",
        "## Stats",
        "",
        f"- max: {stats['max']:.6f}",
        f"- mean: {stats['mean']:.6f}",
        f"- p99: {stats['p99']:.6f}",
        f"- count > 1e-3: {stats['count_above_1e-3']} / {stats['total']}",
        f"- count > 1e-2: {stats['count_above_1e-2']} / {stats['total']}",
        "",
        "## Top 5 outliers",
        "",
        "| flat_idx | vlm | iron | abs_diff |",
        "| --- | --- | --- | --- |",
    ]
    for o in outliers:
        lines.append(f"| {o['idx']} | {o['vlm']:.4f} | {o['iron']:.4f} | {o['diff']:.4f} |")
    args.out.write_text("\n".join(lines) + "\n")

    print(f"[diff_preprocess] gate 1 max_diff = {stats['max']:.4f} ({'PASS' if pass_gate else 'FAIL'})")
    print(f"[diff_preprocess] report → {args.out}")
    return 0 if pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
