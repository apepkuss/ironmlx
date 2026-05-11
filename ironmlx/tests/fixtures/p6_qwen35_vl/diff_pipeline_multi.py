#!/usr/bin/env python
"""P6.6 multi-image vision-encoder diff (Gate 2 + op-level rupture).

Compares mlx-vlm and ironmlx vision-tower outputs for a 2-image input.
Reuses the per-tensor pairing pattern from diff_pipeline.py (P6.1) but
focuses on the final vision_embeds tensor for Gate 2; if op-level
intermediate tensors are also present in both dirs (29 module + 96
intra-block sites from P6.1+P6.3b hooks), include them in a per-tensor
table.

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_pipeline_multi.py \\
        --py /tmp/p6_diff_multi/python \\
        --rust /tmp/p6_diff_multi/rust \\
        --out /path/to/p6_6_vision_report
"""
from __future__ import annotations

import argparse
import json
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
        "count_above_1e-1": int((d > 1e-1).sum()),
        "total": int(d.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate2", type=float, default=0.1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Gate 2: final vision_embeds
    vlm_emb = args.py / "vision_embeds.safetensors"
    iron_emb = args.rust / "vision_embeds.safetensors"
    if not vlm_emb.exists() or not iron_emb.exists():
        msg = f"missing vision_embeds — vlm={vlm_emb.exists()}, iron={iron_emb.exists()}"
        print(f"ERROR: {msg}", file=sys.stderr)
        (args.out / "report.md").write_text(f"# P6.6 Vision Diff\n\n**ERROR**: {msg}\n")
        return 2

    a = load_tensor(vlm_emb)
    b = load_tensor(iron_emb)
    if a.shape != b.shape:
        msg = f"vision_embeds shape mismatch: vlm {a.shape} vs iron {b.shape}"
        print(f"ERROR: {msg}", file=sys.stderr)
        (args.out / "report.md").write_text(f"# P6.6 Vision Diff\n\n**ERROR**: {msg}\n")
        return 2

    final_stats = diff_stats(a, b)
    gate2_pass = final_stats["max"] < args.gate2

    # Op-level (optional) — only if the existing P6.1+P6.3b hook outputs are
    # also in py_dir AND rust_dir. Reuse the basename-pair pattern.
    op_rows = []
    py_files = {p.stem: p for p in sorted(args.py.glob("*.safetensors"))}
    rust_files = {p.stem: p for p in sorted(args.rust.glob("*.safetensors"))}
    common = sorted(set(py_files) & set(rust_files))
    common = [c for c in common if c != "vision_embeds"]  # already reported
    for name in common:
        try:
            a2 = load_tensor(py_files[name])
            b2 = load_tensor(rust_files[name])
            if a2.shape != b2.shape:
                op_rows.append({"name": name, "error": f"shape mismatch {a2.shape} vs {b2.shape}"})
                continue
            s = diff_stats(a2, b2)
            s["name"] = name
            s["shape"] = list(a2.shape)
            op_rows.append(s)
        except Exception as e:
            op_rows.append({"name": name, "error": str(e)})

    lines = [
        "# P6.6 Multi-Image Vision Diff (Gate 2)",
        "",
        f"- Gate 2 threshold: < {args.gate2}",
        f"- vision_embeds shape: {list(a.shape)}",
        f"- vision_embeds max_diff: **{final_stats['max']:.4f}**",
        f"- Gate 2 verdict: **{'PASS' if gate2_pass else 'FAIL'}**",
        "",
        "## Final vision_embeds stats",
        "",
        f"- max: {final_stats['max']:.6f}",
        f"- mean: {final_stats['mean']:.6f}",
        f"- p99: {final_stats['p99']:.6f}",
        f"- count > 1e-2: {final_stats['count_above_1e-2']} / {final_stats['total']}",
        f"- count > 1e-1: {final_stats['count_above_1e-1']} / {final_stats['total']}",
        "",
    ]
    if op_rows:
        lines.append("## Op-level intermediate tensors")
        lines.append("")
        lines.append("| tensor | shape | max | mean | >1e-2 | >1e-1 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in op_rows:
            if "error" in r:
                lines.append(f"| `{r['name']}` | — | error: {r['error']} | | | |")
                continue
            lines.append(
                f"| `{r['name']}` | {r['shape']} | {r['max']:.4f} | {r['mean']:.6f} | "
                f"{r['count_above_1e-2']}/{r['total']} | {r['count_above_1e-1']}/{r['total']} |"
            )
        lines.append("")

    (args.out / "report.md").write_text("\n".join(lines))
    (args.out / "summary.json").write_text(json.dumps({
        "gate2_max_diff": final_stats["max"],
        "gate2_pass": gate2_pass,
        "vision_embeds_shape": list(a.shape),
    }, indent=2))
    print(f"[diff_pipeline_multi] Gate 2 {'PASS' if gate2_pass else 'FAIL'}; report → {args.out}/report.md")
    return 0 if gate2_pass else 1


if __name__ == "__main__":
    sys.exit(main())
