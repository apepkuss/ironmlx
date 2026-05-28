#!/usr/bin/env python3
"""Stage-by-stage Qwen3.6 GatedDeltaNet benchmark under MLX Python.

This mirrors the ironmlx GDN stage boundaries closely enough to compare
forced-eval stage rankings against the Rust `p5g-profile` layer2 breakdown.
It intentionally uses the same fused qkvz / ba projection shape as ironmlx.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import gated_delta

from qwen36_gdn_path_compare import (
    build_fused_quantized,
    find_linear_attn,
    fused_quantized_matmul,
)


STAGE_NAMES = [
    "step_1a_in_proj_qkvz",
    "step_1b_in_proj_ba",
    "step_2a_prepend_conv_state",
    "step_2b_conv1d_silu",
    "step_2c_update_conv_state",
    "step_3_split_reshape_per_head",
    "step_4_qk_rmsnorm",
    "step_5_compute_g",
    "step_6_sigmoid_beta",
    "step_7_kernel_cache",
    "step_8_norm_proj",
]


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
        "count": len(values),
        "p50_us": pct(values, 50),
        "p95_us": pct(values, 95),
        "mean_us": sum(values) / len(values) if values else None,
    }


def timed(values_us: dict[str, list[float]], name: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    out = fn()
    if isinstance(out, (tuple, list)):
        mx.eval(*out)
    else:
        mx.eval(out)
    values_us[name].append((time.perf_counter() - start) * 1_000_000.0)
    return out


def run_once(gdn: Any, x: mx.array, fused_qkvz: dict[str, Any], fused_ba: dict[str, Any]) -> dict[str, list[float]]:
    values_us: dict[str, list[float]] = {name: [] for name in STAGE_NAMES}
    batch, seq, _ = x.shape

    def qkvz_project_slice() -> tuple[mx.array, mx.array]:
        qkvz = fused_quantized_matmul(x, fused_qkvz)
        qkv = qkvz[..., : gdn.conv_dim]
        z = qkvz[..., gdn.conv_dim :].reshape(
            batch, seq, gdn.num_v_heads, gdn.head_v_dim
        )
        return qkv, z

    qkv, z = timed(values_us, "step_1a_in_proj_qkvz", qkvz_project_slice)

    def ba_project_slice() -> tuple[mx.array, mx.array]:
        ba = fused_quantized_matmul(x, fused_ba)
        b = ba[..., : gdn.num_v_heads]
        a = ba[..., gdn.num_v_heads :]
        return b, a

    b, a = timed(values_us, "step_1b_in_proj_ba", ba_project_slice)

    conv_state = mx.zeros(
        (batch, gdn.conv_kernel_size - 1, gdn.conv_dim),
        dtype=x.dtype,
    )
    conv_input = timed(
        values_us,
        "step_2a_prepend_conv_state",
        lambda: mx.concatenate([conv_state, qkv], axis=1),
    )

    conv_out = timed(
        values_us,
        "step_2b_conv1d_silu",
        lambda: nn.silu(gdn.conv1d(conv_input)),
    )

    n_keep = gdn.conv_kernel_size - 1
    timed(
        values_us,
        "step_2c_update_conv_state",
        lambda: mx.contiguous(conv_input[:, -n_keep:, :]),
    )

    def split_reshape() -> tuple[mx.array, mx.array, mx.array]:
        parts = mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1)
        q, k, v = [
            part.reshape(batch, seq, heads, dim)
            for part, heads, dim in zip(
                parts,
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        return q, k, v

    q, k, v = timed(values_us, "step_3_split_reshape_per_head", split_reshape)

    inv_scale = k.shape[-1] ** -0.5
    q, k = timed(
        values_us,
        "step_4_qk_rmsnorm",
        lambda: (
            (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6),
            inv_scale * mx.fast.rms_norm(k, None, 1e-6),
        ),
    )

    g = timed(
        values_us,
        "step_5_compute_g",
        lambda: gated_delta.compute_g(gdn.A_log, a, gdn.dt_bias),
    )
    beta = timed(values_us, "step_6_sigmoid_beta", lambda: mx.sigmoid(b))

    state = mx.zeros(
        (batch, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
        dtype=mx.float32,
    )
    out, _state = timed(
        values_us,
        "step_7_kernel_cache",
        lambda: gated_delta.gated_delta_kernel(q, k, v, g, beta, state, None),
    )

    timed(
        values_us,
        "step_8_norm_proj",
        lambda: gdn.out_proj(gdn.norm(out, z).reshape(batch, seq, -1)),
    )

    return values_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq", type=int, default=521)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model, _ = load(args.model_dir, lazy=False)
    mx.synchronize()
    gdn = find_linear_attn(model, args.layer)
    fused_qkvz = build_fused_quantized(gdn.in_proj_qkv, gdn.in_proj_z)
    fused_ba = build_fused_quantized(gdn.in_proj_b, gdn.in_proj_a)

    mx.random.seed(args.seed + args.seq)
    x = mx.random.normal((1, args.seq, gdn.hidden_size), dtype=mx.bfloat16)
    mx.eval(x)

    for _ in range(args.warmup):
        run_once(gdn, x, fused_qkvz, fused_ba)

    values: dict[str, list[float]] = {name: [] for name in STAGE_NAMES}
    for _ in range(args.runs):
        current = run_once(gdn, x, fused_qkvz, fused_ba)
        for name, stage_values in current.items():
            values[name].extend(stage_values)

    stage_summary = {name: summarize(vals) for name, vals in values.items()}
    top_steps = sorted(
        (
            {"step": name, "p50_us": stats["p50_us"], "mean_us": stats["mean_us"]}
            for name, stats in stage_summary.items()
        ),
        key=lambda item: item["p50_us"] or 0.0,
        reverse=True,
    )
    output = {
        "meta": {
            "backend": "mlx-python-gdn-stage",
            "model_dir": os.path.abspath(args.model_dir),
            "layer": args.layer,
            "seq": args.seq,
            "warmup": args.warmup,
            "runs": args.runs,
            "stage_names": STAGE_NAMES,
        },
        "stages": stage_summary,
        "top_steps_by_p50_us": top_steps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
