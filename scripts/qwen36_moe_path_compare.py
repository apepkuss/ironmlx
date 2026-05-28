#!/usr/bin/env python3
"""Compare Qwen3.6 MoE routed-MLP execution shapes under MLX.

This is diagnostic instrumentation for the Qwen3.6 performance work. It loads
the checkpoint through mlx-lm, then benchmarks:

  1. mlx-lm reference Qwen3NextSparseMoeBlock.
  2. ironmlx-shaped routed MLP with split gate/up gather_qmm.
  3. ironmlx-shaped routed MLP with fused gate/up gather_qmm.

Run with the pinned benchmark venv:

  uv run --project scripts/bench-venvs/mlx-lm \
    python scripts/qwen36_moe_path_compare.py --model-dir "$MODEL_DIR" \
    --seq 521 --seq 1 --out /tmp/qwen36_moe_path_compare.json
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
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from mlx_lm.models import qwen3_5_moe, qwen3_next, switch_layers


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


def find_sparse_mlp(model: Any, layer: int) -> Any:
    candidates = [
        lambda: model.language_model.model.layers[layer].mlp,
        lambda: model.model.layers[layer].mlp,
        lambda: model.layers[layer].mlp,
    ]
    errors = []
    for get in candidates:
        try:
            mlp = get()
            if hasattr(mlp, "switch_mlp") and hasattr(mlp, "shared_expert"):
                return mlp
            errors.append(f"{type(mlp).__name__} is not a sparse MoE block")
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError("could not locate Qwen3NextSparseMoeBlock: " + "; ".join(errors))


def build_fused_gate_up(switch_mlp: Any) -> dict[str, Any]:
    gate = switch_mlp.gate_proj
    up = switch_mlp.up_proj
    gate_biases = gate.get("biases")
    up_biases = up.get("biases")
    if (gate_biases is None) != (up_biases is None):
        raise RuntimeError("gate/up quantized biases presence mismatch")

    fused = {
        "weight": mx.concatenate([gate["weight"], up["weight"]], axis=1),
        "scales": mx.concatenate([gate["scales"], up["scales"]], axis=1),
        "biases": None,
        "intermediate": gate.output_dims,
        "group_size": gate.group_size,
        "bits": gate.bits,
        "mode": gate.mode,
    }
    to_eval = [fused["weight"], fused["scales"]]
    if gate_biases is not None:
        fused["biases"] = mx.concatenate([gate_biases, up_biases], axis=1)
        to_eval.append(fused["biases"])
    mx.eval(*to_eval)
    return fused


def router_scores_and_indices(mlp: Any, flat_x: mx.array, top_k: int) -> tuple[mx.array, mx.array]:
    gates = mlp.gate(flat_x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[:, -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mlp.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return scores, inds.astype(mx.uint32)


def ironmlx_shape_mlp(
    mlp: Any,
    x: mx.array,
    fused: dict[str, Any],
    top_k: int,
    sorted_threshold: int,
    use_fused_gate_up: bool,
) -> mx.array:
    batch, seq, hidden = x.shape
    bs = batch * seq
    bs_k = bs * top_k
    flat_x = x.reshape(bs, hidden)
    scores, inds = router_scores_and_indices(mlp, flat_x, top_k)
    switch = mlp.switch_mlp
    gate = switch.gate_proj
    up = switch.up_proj
    down = switch.down_proj
    use_sorted = bs_k >= sorted_threshold

    if use_sorted:
        flat_topk = inds.reshape(bs_k)
        sort_perm = mx.argsort(flat_topk)
        sorted_topk = flat_topk[sort_perm]
        sorted_x = flat_x[sort_perm // top_k]
        sorted_x = mx.expand_dims(sorted_x, -2)

        if use_fused_gate_up:
            gate_up = mx.gather_qmm(
                sorted_x,
                fused["weight"],
                fused["scales"],
                fused["biases"],
                rhs_indices=sorted_topk,
                transpose=True,
                group_size=fused["group_size"],
                bits=fused["bits"],
                mode=fused["mode"],
                sorted_indices=True,
            )
            intermediate = fused["intermediate"]
            gate_out = gate_up[..., :intermediate]
            up_out = gate_up[..., intermediate:]
        else:
            gate_out = gate(sorted_x, sorted_topk, sorted_indices=True)
            up_out = up(sorted_x, sorted_topk, sorted_indices=True)

        act = swiglu(gate_out, up_out)
        down_raw = down(act, sorted_topk, sorted_indices=True)
        inv_perm = mx.argsort(sort_perm)
        down_out = down_raw.reshape(bs_k, hidden)[inv_perm].reshape(bs, top_k, hidden)
    else:
        x_in = mx.expand_dims(flat_x, (-2, -3))
        if use_fused_gate_up:
            gate_up = mx.gather_qmm(
                x_in,
                fused["weight"],
                fused["scales"],
                fused["biases"],
                rhs_indices=inds,
                transpose=True,
                group_size=fused["group_size"],
                bits=fused["bits"],
                mode=fused["mode"],
                sorted_indices=False,
            )
            intermediate = fused["intermediate"]
            gate_out = gate_up[..., :intermediate]
            up_out = gate_up[..., intermediate:]
        else:
            gate_out = gate(x_in, inds, sorted_indices=False)
            up_out = up(x_in, inds, sorted_indices=False)

        act = swiglu(gate_out, up_out)
        down_raw = down(act, inds, sorted_indices=False)
        down_out = down_raw.squeeze(-2)

    routed = (down_out * scores[..., None]).sum(axis=-2)
    shared = mlp.shared_expert(flat_x)
    shared = mx.sigmoid(mlp.shared_expert_gate(flat_x)) * shared
    return (routed + shared).reshape(batch, seq, hidden)


def max_abs_diff(left: mx.array, right: mx.array) -> float:
    diff = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(diff)
    return float(diff.item())


def run_one(
    mlp: Any,
    fused: dict[str, Any],
    seq: int,
    top_k: int,
    sorted_threshold: int,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    mx.random.seed(seed + seq)
    hidden = mlp.switch_mlp.gate_proj.input_dims
    x = mx.random.normal((1, seq, hidden), dtype=mx.bfloat16)
    mx.eval(x)

    def reference() -> mx.array:
        return mlp(x)

    def split_shape() -> mx.array:
        return ironmlx_shape_mlp(mlp, x, fused, top_k, sorted_threshold, False)

    def fused_shape() -> mx.array:
        return ironmlx_shape_mlp(mlp, x, fused, top_k, sorted_threshold, True)

    ref_out = reference()
    split_out = split_shape()
    fused_out = fused_shape()
    mx.eval(ref_out, split_out, fused_out)

    reference_bench = bench("mlx_lm_reference_sparse_moe", reference, warmup, runs)
    split_bench = bench("ironmlx_shape_split_gate_up", split_shape, warmup, runs)
    fused_bench = bench("ironmlx_shape_fused_gate_up", fused_shape, warmup, runs)

    return {
        "seq": seq,
        "routes": seq * top_k,
        "top_k": top_k,
        "sorted_threshold": sorted_threshold,
        "ironmlx_shape_uses_sorted": (seq * top_k) >= sorted_threshold,
        "reference_switchglu_uses_sorted": (seq * top_k) >= 64,
        "max_abs_diff_reference_vs_split": max_abs_diff(ref_out, split_out),
        "max_abs_diff_reference_vs_fused": max_abs_diff(ref_out, fused_out),
        "reference": reference_bench,
        "ironmlx_shape_split": split_bench,
        "ironmlx_shape_fused": fused_bench,
        "split_over_reference_p50": split_bench["p50_ms"] / reference_bench["p50_ms"],
        "fused_over_reference_p50": fused_bench["p50_ms"] / reference_bench["p50_ms"],
        "fused_over_split_p50": fused_bench["p50_ms"] / split_bench["p50_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq", type=int, action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sorted-threshold", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seqs = args.seq or [521]
    model, _ = load(args.model_dir, lazy=False)
    mx.synchronize()
    mlp = find_sparse_mlp(model, args.layer)
    fused = build_fused_gate_up(mlp.switch_mlp)

    results = [
        run_one(
            mlp=mlp,
            fused=fused,
            seq=seq,
            top_k=args.top_k,
            sorted_threshold=args.sorted_threshold,
            warmup=args.warmup,
            runs=args.runs,
            seed=args.seed,
        )
        for seq in seqs
    ]
    output = {
        "meta": {
            "model_dir": os.path.abspath(args.model_dir),
            "layer": args.layer,
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "mlx_lm_qwen3_5_moe": inspect.getfile(qwen3_5_moe),
            "mlx_lm_qwen3_next": inspect.getfile(qwen3_next),
            "mlx_lm_switch_layers": inspect.getfile(switch_layers),
            "hidden": mlp.switch_mlp.gate_proj.input_dims,
            "moe_intermediate": mlp.switch_mlp.gate_proj.output_dims,
            "num_experts": mlp.switch_mlp.gate_proj.num_experts,
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
