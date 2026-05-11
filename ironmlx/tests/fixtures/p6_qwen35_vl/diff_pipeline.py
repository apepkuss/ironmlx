"""P6.1 vision encoder diff pipeline.

Reads ironmlx + mlx-vlm side-by-side `.safetensors` dumps, pairs them by
basename, computes per-tensor diff stats, identifies the first significant
divergence ("rupture"), and emits a markdown report + max_diff curve PNG.

Usage:
    ~/.venvs/mlxvlm-ref/bin/python diff_pipeline.py \
        --py /tmp/p6_diff/python \
        --rust /tmp/p6_diff/rust \
        --out tests/fixtures/p6_qwen35_vl/diff_reports/2026-05-11

See spec docs/superpowers/specs/2026-05-11-p6-1-vision-diff-pipeline-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_tensor(path: Path) -> np.ndarray:
    """Load a .safetensors file produced by either mlx-vlm or ironmlx and
    return it as a numpy float32 array (cast from whatever dtype was on disk).
    """
    arr = mx.load(str(path))
    if isinstance(arr, dict):
        t = arr["tensor"]
    else:
        t = arr
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


def top_outliers(a: np.ndarray, b: np.ndarray, n: int = 5) -> list[dict]:
    d = np.abs(a.flatten() - b.flatten())
    if len(d) == 0:
        return []
    k = min(n, len(d))
    idxs = np.argpartition(d, -k)[-k:]
    idxs = idxs[np.argsort(d[idxs])[::-1]]
    return [
        {
            "idx": int(i),
            "a_val": float(a.flatten()[i]),
            "b_val": float(b.flatten()[i]),
            "diff": float(d[i]),
        }
        for i in idxs
    ]


def pair_files(py_dir: Path, rust_dir: Path) -> tuple[list[tuple[str, Path, Path]], dict]:
    """Pair .safetensors files by basename. Returns (paired_list, unpaired_dict)."""
    py_files = {p.stem: p for p in sorted(py_dir.glob("*.safetensors"))}
    rust_files = {p.stem: p for p in sorted(rust_dir.glob("*.safetensors"))}
    common = sorted(set(py_files) & set(rust_files))
    pairs = [(name, py_files[name], rust_files[name]) for name in common]
    unpaired = {
        "py_only": sorted(set(py_files) - set(rust_files)),
        "rust_only": sorted(set(rust_files) - set(py_files)),
    }
    if unpaired["py_only"]:
        print(f"[diff_pipeline] py-only (skipped): {unpaired['py_only']}", file=sys.stderr)
    if unpaired["rust_only"]:
        print(f"[diff_pipeline] rust-only (skipped): {unpaired['rust_only']}", file=sys.stderr)
    return pairs, unpaired


def diff_pair(py_path: Path, rust_path: Path) -> dict:
    a = load_tensor(py_path)
    b = load_tensor(rust_path)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch for {py_path.stem}: python {a.shape} vs rust {b.shape}"
        )
    stats = diff_stats(a, b)
    stats["name"] = py_path.stem
    stats["shape"] = list(a.shape)
    stats["_a"] = a  # carried only when caller asks for outliers
    stats["_b"] = b
    return stats


def find_rupture(rows: list[dict], factor: float = 5.0) -> str | None:
    """Return the name of the first tensor whose max_diff jumps `factor`× over
    the previous max_diff. If no such jump, return the tensor with the largest
    max_diff overall."""
    prev = None
    for r in rows:
        if prev is not None and r["max"] > factor * max(prev, 1e-6):
            return r["name"]
        prev = r["max"]
    # Fallback: max
    return max(rows, key=lambda r: r["max"])["name"] if rows else None


def render_report(rows: list[dict], rupture: str | None, top_outliers: list[dict]) -> str:
    lines = ["# P6 VL Vision Encoder Diff Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total tensors compared: {len(rows)}")
    if rupture:
        lines.append(f"- First rupture point: **{rupture}**")
    if rows:
        last = rows[-1]
        lines.append(f"- Final tensor `{last['name']}`: max_diff = {last['max']:.4f}, mean = {last['mean']:.6f}")
    lines.append("")
    lines.append("## Per-tensor table")
    lines.append("")
    lines.append("| # | tensor | shape | max | mean | p99 | >1e-3 | >1e-2 | >1e-1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows):
        lines.append(
            f"| {i} | `{r['name']}` | {r['shape']} | {r['max']:.4f} | {r['mean']:.6f} | {r['p99']:.4f} | "
            f"{r['count_above_1e-3']}/{r['total']} | {r['count_above_1e-2']}/{r['total']} | {r['count_above_1e-1']}/{r['total']} |"
        )
    lines.append("")
    if top_outliers:
        lines.append("## Top outliers in final tensor")
        lines.append("")
        lines.append("| flat_idx | mlx-vlm | ironmlx | abs_diff |")
        lines.append("| --- | --- | --- | --- |")
        for o in top_outliers:
            lines.append(
                f"| {o['idx']} | {o['a_val']:.4f} | {o['b_val']:.4f} | {o['diff']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_curve(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(rows)))
    ys = [max(r["max"], 1e-9) for r in rows]
    labels = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(xs, ys, marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("max_abs_diff (log)")
    ax.set_title("P6 VL Vision Encoder: ironmlx vs mlx-vlm")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs, unpaired = pair_files(args.py, args.rust)
    if not pairs:
        print("ERROR: no paired tensors found", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for name, py_p, rust_p in pairs:
        try:
            r = diff_pair(py_p, rust_p)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        rows.append(r)

    rupture = find_rupture(rows)

    # Top outliers in the final tensor (merger output)
    final = rows[-1]
    final_top = top_outliers(final["_a"], final["_b"], n=5)

    # Strip the carried arrays before serializing
    rows_clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    report = render_report(rows_clean, rupture, final_top)
    (args.out / "report.md").write_text(report)
    plot_curve(rows_clean, args.out / "max_diff_curve.png")
    (args.out / "outliers.json").write_text(
        json.dumps({"final_tensor": final["name"], "top": final_top, "unpaired": unpaired}, indent=2)
    )

    print(f"[diff_pipeline] report → {args.out}/report.md")
    print(f"[diff_pipeline] curve  → {args.out}/max_diff_curve.png")
    print(f"[diff_pipeline] rupture: {rupture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
