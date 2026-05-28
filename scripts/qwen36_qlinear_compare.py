#!/usr/bin/env python3
"""Focused Qwen3.6 GDN quantized-linear benchmark under MLX Python.

This mirrors `ironmlx-qlinear-bench` for the qkvz and out_proj shapes that
showed Rust-side excess in the direct GDN stage breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_lm import load

from qwen36_gdn_path_compare import (
    build_fused_quantized,
    find_linear_attn,
    fused_quantized_matmul,
)


def pct(values: list[float], p: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "runs": len(values),
        "p50_ms": pct(values, 50),
        "p95_ms": pct(values, 95),
        "mean_ms": sum(values) / len(values) if values else None,
    }


def as_outputs(out: Any) -> tuple[Any, ...]:
    if isinstance(out, tuple):
        return out
    if isinstance(out, list):
        return tuple(out)
    return (out,)


def bench(label: str, fn: Callable[[], Any], warmup: int, runs: int) -> dict[str, Any]:
    warmups = []
    output_shapes: list[list[int]] = []
    for _ in range(warmup):
        start = time.perf_counter()
        outputs = as_outputs(fn())
        mx.eval(*outputs)
        mx.synchronize()
        warmups.append((time.perf_counter() - start) * 1000.0)
        output_shapes = [list(out.shape) for out in outputs]

    values = []
    for _ in range(runs):
        start = time.perf_counter()
        outputs = as_outputs(fn())
        mx.eval(*outputs)
        mx.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
        output_shapes = [list(out.shape) for out in outputs]

    return {
        "case": label,
        "output_shapes": output_shapes,
        "summary": summarize(values),
        "warmups": warmups,
        "values_ms": values,
    }


def projection_to_dict(linear: Any) -> dict[str, Any]:
    return {
        "weight": linear["weight"],
        "scales": linear["scales"],
        "biases": linear.get("biases"),
        "bias": linear.get("bias"),
        "group_size": linear.group_size,
        "bits": linear.bits,
        "mode": linear.mode,
    }


def quantized_matmul_with_bias(x: mx.array, projection: dict[str, Any]) -> mx.array:
    y = mx.quantized_matmul(
        x,
        projection["weight"],
        scales=projection["scales"],
        biases=projection["biases"],
        transpose=True,
        group_size=projection["group_size"],
        bits=projection["bits"],
        mode=projection["mode"],
    )
    if projection["bias"] is not None:
        y = y + projection["bias"]
    return y


def run_one(
    gdn: Any,
    fused_qkvz: dict[str, Any],
    out_proj: dict[str, Any],
    seq: int,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    batch = 1
    mx.random.seed(seed + seq)
    x_hidden = mx.random.normal((batch, seq, gdn.hidden_size), dtype=mx.bfloat16)
    mx.random.seed(seed + 100_000 + seq)
    x_value = mx.random.normal((batch, seq, gdn.num_v_heads * gdn.head_v_dim), dtype=mx.bfloat16)
    mx.random.seed(seed + 200_000 + seq)
    y_heads = mx.random.normal(
        (batch, seq, gdn.num_v_heads, gdn.head_v_dim),
        dtype=mx.bfloat16,
    )
    mx.random.seed(seed + 300_000 + seq)
    z_heads = mx.random.normal(
        (batch, seq, gdn.num_v_heads, gdn.head_v_dim),
        dtype=mx.bfloat16,
    )
    mx.eval(x_hidden, x_value, y_heads, z_heads)
    mx.synchronize()

    def qkvz_direct() -> mx.array:
        return fused_quantized_matmul(x_hidden, fused_qkvz)

    def qkvz_slice() -> tuple[mx.array, mx.array]:
        qkvz = fused_quantized_matmul(x_hidden, fused_qkvz)
        qkv = qkvz[..., : gdn.conv_dim]
        z = qkvz[..., gdn.conv_dim :].reshape(
            batch, seq, gdn.num_v_heads, gdn.head_v_dim
        )
        return qkv, z

    def out_direct() -> mx.array:
        return quantized_matmul_with_bias(x_value, out_proj)

    def out_module() -> mx.array:
        return gdn.out_proj(x_value)

    def norm_out_module() -> mx.array:
        normed = gdn.norm(y_heads, z_heads)
        return gdn.out_proj(normed.reshape(batch, seq, -1))

    cases = [
        bench("qkvz-direct-qmm", qkvz_direct, warmup, runs),
        bench("qkvz-linear-slice", qkvz_slice, warmup, runs),
        bench("out-direct-qmm", out_direct, warmup, runs),
        bench("out-linear", out_module, warmup, runs),
        bench("norm-out-linear", norm_out_module, warmup, runs),
    ]
    for case in cases:
        case["seq"] = seq
    return {"seq": seq, "records": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq", type=int, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model, _ = load(args.model_dir, lazy=False)
    mx.synchronize()
    gdn = find_linear_attn(model, args.layer)
    fused_qkvz = build_fused_quantized(gdn.in_proj_qkv, gdn.in_proj_z)
    out_proj = projection_to_dict(gdn.out_proj)

    seqs = args.seq or [521, 1]
    by_seq = [
        run_one(
            gdn=gdn,
            fused_qkvz=fused_qkvz,
            out_proj=out_proj,
            seq=seq,
            warmup=args.warmup,
            runs=args.runs,
            seed=args.seed,
        )
        for seq in seqs
    ]
    records = [record for group in by_seq for record in group["records"]]
    output = {
        "meta": {
            "backend": "mlx-python-qlinear",
            "model_dir": os.path.abspath(args.model_dir),
            "layer": args.layer,
            "seqs": seqs,
            "warmup": args.warmup,
            "runs": args.runs,
            "hidden_size": gdn.hidden_size,
            "conv_dim": gdn.conv_dim,
            "value_dim": gdn.num_v_heads * gdn.head_v_dim,
            "qkvz_out_dim": gdn.conv_dim + gdn.num_v_heads * gdn.head_v_dim,
            "num_v_heads": gdn.num_v_heads,
            "head_v_dim": gdn.head_v_dim,
        },
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
