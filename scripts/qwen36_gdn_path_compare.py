#!/usr/bin/env python3
"""Compare Qwen3.6 GatedDeltaNet execution shapes under MLX.

The Qwen3.6 MoE checkpoint routes linear-attention layers through mlx-lm's
qwen3_5.GatedDeltaNet, which uses four split input projections. ironmlx fuses
qkv+z and b+a into two quantized projections, then follows the same conv,
gated-delta, gated-RMSNorm, and output projection shape.

This script benchmarks both full paths with the same MLX weights so we can
separate "projection fusion / graph shape" from Rust binding and custom kernel
effects observed by p5g-profile.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import gated_delta, qwen3_5
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update


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


def bench(label: str, fn: Callable[[], Any], warmup: int, runs: int) -> dict[str, Any]:
    for _ in range(warmup):
        out = fn()
        if not isinstance(out, (tuple, list)):
            out = (out,)
        mx.eval(*out)
    values = []
    for _ in range(runs):
        start = time.perf_counter()
        out = fn()
        if not isinstance(out, (tuple, list)):
            out = (out,)
        mx.eval(*out)
        values.append((time.perf_counter() - start) * 1000.0)
    return {
        "label": label,
        "runs": runs,
        "p50_ms": pct(values, 50),
        "p95_ms": pct(values, 95),
        "mean_ms": sum(values) / len(values),
        "values_ms": values,
    }


def find_linear_attn(model: Any, layer: int) -> Any:
    candidates = [
        lambda: model.language_model.model.layers[layer].linear_attn,
        lambda: model.model.layers[layer].linear_attn,
        lambda: model.layers[layer].linear_attn,
    ]
    errors = []
    for get in candidates:
        try:
            gdn = get()
            required = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            if all(hasattr(gdn, name) for name in required):
                return gdn
            errors.append(f"{type(gdn).__name__} is not qwen3_5.GatedDeltaNet")
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError("could not locate linear_attn: " + "; ".join(errors))


def concat_biases(left: Any, right: Any) -> mx.array | None:
    left_biases = left.get("biases")
    right_biases = right.get("biases")
    if left_biases is None and right_biases is None:
        return None
    if left_biases is None or right_biases is None:
        raise RuntimeError("quantized biases presence mismatch")
    return mx.concatenate([left_biases, right_biases], axis=0)


def build_fused_quantized(left: Any, right: Any) -> dict[str, Any]:
    fused = {
        "weight": mx.concatenate([left["weight"], right["weight"]], axis=0),
        "scales": mx.concatenate([left["scales"], right["scales"]], axis=0),
        "biases": concat_biases(left, right),
        "group_size": left.group_size,
        "bits": left.bits,
        "mode": left.mode,
    }
    to_eval = [fused["weight"], fused["scales"]]
    if fused["biases"] is not None:
        to_eval.append(fused["biases"])
    mx.eval(*to_eval)
    return fused


def fused_quantized_matmul(x: mx.array, fused: dict[str, Any]) -> mx.array:
    return mx.quantized_matmul(
        x,
        fused["weight"],
        scales=fused["scales"],
        biases=fused["biases"],
        transpose=True,
        group_size=fused["group_size"],
        bits=fused["bits"],
        mode=fused["mode"],
    )


def make_cache(enabled: bool) -> ArraysCache | None:
    return ArraysCache(size=2) if enabled else None


def ironmlx_shape_gdn(
    gdn: Any,
    x: mx.array,
    fused_qkvz: dict[str, Any],
    fused_ba: dict[str, Any],
    use_cache: bool,
) -> mx.array:
    batch, seq, _ = x.shape
    cache = make_cache(use_cache)

    qkvz = fused_quantized_matmul(x, fused_qkvz)
    qkv = qkvz[..., : gdn.conv_dim]
    z = qkvz[..., gdn.conv_dim :].reshape(batch, seq, gdn.num_v_heads, gdn.head_v_dim)

    ba = fused_quantized_matmul(x, fused_ba)
    b = ba[..., : gdn.num_v_heads]
    a = ba[..., gdn.num_v_heads :]

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (batch, gdn.conv_kernel_size - 1, gdn.conv_dim),
            dtype=x.dtype,
        )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = gdn.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

    conv_out = nn.silu(gdn.conv1d(conv_input))
    q, k, v = [
        part.reshape(batch, seq, heads, dim)
        for part, heads, dim in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    out, state = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        gdn.A_log,
        gdn.dt_bias,
        state,
        None,
        use_kernel=not gdn.training,
    )
    if cache is not None:
        cache[1] = state
        cache.advance(seq)

    out = gdn.norm(out, z)
    return gdn.out_proj(out.reshape(batch, seq, -1))


def max_abs_diff(left: mx.array, right: mx.array) -> float:
    diff = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(diff)
    return float(diff.item())


def run_one(
    gdn: Any,
    fused_qkvz: dict[str, Any],
    fused_ba: dict[str, Any],
    seq: int,
    use_cache: bool,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    mx.random.seed(seed + seq + (100_000 if use_cache else 0))
    x = mx.random.normal((1, seq, gdn.hidden_size), dtype=mx.bfloat16)
    mx.eval(x)

    def reference() -> mx.array:
        return gdn(x, None, make_cache(use_cache))

    def fused_shape() -> mx.array:
        return ironmlx_shape_gdn(gdn, x, fused_qkvz, fused_ba, use_cache)

    ref_out = reference()
    fused_out = fused_shape()
    mx.eval(ref_out, fused_out)
    ref_bench = bench("mlx_lm_reference_gdn_split_proj", reference, warmup, runs)
    fused_bench = bench("ironmlx_shape_gdn_fused_proj", fused_shape, warmup, runs)

    return {
        "seq": seq,
        "use_cache": use_cache,
        "max_abs_diff_reference_vs_fused": max_abs_diff(ref_out, fused_out),
        "reference": ref_bench,
        "ironmlx_shape_fused": fused_bench,
        "fused_over_reference_p50": fused_bench["p50_ms"] / ref_bench["p50_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq", type=int, action="append", default=[])
    parser.add_argument("--cache-mode", choices=["both", "cache", "no-cache"], default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seqs = args.seq or [521]
    cache_modes = {
        "both": [False, True],
        "cache": [True],
        "no-cache": [False],
    }[args.cache_mode]

    model, _ = load(args.model_dir, lazy=False)
    mx.synchronize()
    gdn = find_linear_attn(model, args.layer)
    fused_qkvz = build_fused_quantized(gdn.in_proj_qkv, gdn.in_proj_z)
    fused_ba = build_fused_quantized(gdn.in_proj_b, gdn.in_proj_a)

    results = []
    for seq in seqs:
        for use_cache in cache_modes:
            results.append(
                run_one(
                    gdn=gdn,
                    fused_qkvz=fused_qkvz,
                    fused_ba=fused_ba,
                    seq=seq,
                    use_cache=use_cache,
                    warmup=args.warmup,
                    runs=args.runs,
                    seed=args.seed,
                )
            )

    output = {
        "meta": {
            "model_dir": os.path.abspath(args.model_dir),
            "layer": args.layer,
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "mlx_lm_qwen3_5": inspect.getfile(qwen3_5),
            "mlx_lm_gated_delta": inspect.getfile(gated_delta),
            "hidden": gdn.hidden_size,
            "conv_dim": gdn.conv_dim,
            "num_k_heads": gdn.num_k_heads,
            "num_v_heads": gdn.num_v_heads,
            "head_k_dim": gdn.head_k_dim,
            "head_v_dim": gdn.head_v_dim,
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
