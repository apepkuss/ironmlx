#!/usr/bin/env python3
"""Generate small Flash-Next component fixtures from pinned MLX-VLM functions.

Uses a separately built MLX 0.32.5 Python extension; never loads model weights.
The caller supplies the checked-out source and downloaded immutable references.
"""
import argparse
import ast
import hashlib
import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

MLX_COMMIT = "a8f66b5901791a38ff527f249e1f1a84f46c03b6"
VLM_COMMIT = "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
VLM_BLOB = "9a9bee24fef808655bcb5826d9877fa0e4cb8e0e"
CAPTURE_PATCH_SHA256 = "91708e18c67f1c0e8f6c19cfbe7cc9f59ff257c05d61bb30b407ae4c36354169"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--capture-patch", type=Path, required=True)
    parser.add_argument("--mlx-source", type=Path, required=True)
    parser.add_argument("--mlx-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert mx.__version__ == "0.32.5", mx.__version__
    assert subprocess.check_output(["git", "-C", str(args.mlx_source), "rev-parse", "HEAD"], text=True).strip() == MLX_COMMIT
    assert not subprocess.check_output(["git", "-C", str(args.mlx_source), "status", "--porcelain"], text=True).strip()
    cache = {}
    for line in (args.mlx_build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("//", "#")):
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    assert Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() == args.mlx_source.resolve()
    assert Path(cache["MLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY"]).resolve() == Path(mx.__file__).resolve().parent
    source = args.source_file.read_bytes()
    assert hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest() == VLM_BLOB
    assert hashlib.sha256(args.capture_patch.read_bytes()).hexdigest() == CAPTURE_PATCH_SHA256
    functions = {"_qwen4_rms_norm", "_qwen4_hyper_pointwise", "_qwen4_inject",
                 "_qwen4_normalize_gated_delta_qk", "_splitmix64", "_build_layer_multipliers",
                 "_is_prime", "_find_nth_prime_after"}
    constants = {"_MASK64", "_SPLITMIX_GAMMA", "_SPLITMIX_M1", "_SPLITMIX_M2", "_PRIME_1"}
    nodes = [node for node in ast.parse(source).body if
             (isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpRMSNormGated") or
             (isinstance(node, ast.FunctionDef) and node.name in functions) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    namespace = {"mx": mx, "nn": nn, "math": math, "lru_cache": lru_cache}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source_file), "exec"), namespace)
    cases = []
    for dtype_name, dtype in [("float32", mx.float32), ("bfloat16", mx.bfloat16)]:
        x = mx.array([(i % 13 - 6) / 8 for i in range(32)], dtype=dtype).reshape(1, 2, 16)
        weight = mx.array([(i % 7 - 3) / 16 for i in range(16)], dtype=dtype)
        norm = namespace["_qwen4_rms_norm"](4, 1e-6)(x, weight)
        activate, mix, gate = namespace["_qwen4_hyper_pointwise"](4, 4)
        low = mx.array([-2, -1, 0, 1, 2, 3], dtype=dtype).reshape(1, 2, 3)
        projected_mix = x * 0.5
        injection = mx.array([i / 4 - 1 for i in range(8)], dtype=dtype).reshape(1, 2, 4)
        gate_output = gate(injection)
        mixed = mix(projected_mix, norm)
        residual = namespace["_qwen4_inject"](mixed, x, gate_output)
        q = mx.array([0, 0, 0, 0, 0.0001, 0.0002, 0.0003, 0.0004,
                      1, -2, 3, -4, 4, 3, 2, 1], dtype=dtype).reshape(1, 2, 2, 4)
        k = q * 0.5
        q_out, k_out = namespace["_qwen4_normalize_gated_delta_qk"](q, k)
        gated_norm = namespace["Qwen4ExpRMSNormGated"](4, 1e-6, "sigmoid")
        gated_norm.weight = mx.array([0.5, 0.75, 1.0, 1.25], dtype=dtype)
        arrays = dict(input=x, weight=weight, norm_output=norm, low_projection=low,
                      activated_low=activate(low), mix_projection=projected_mix,
                      mixed_output=mixed, injection_projection=injection,
                      injection_output=gate_output, residual_output=residual,
                      q=q, k=k, q_output=q_out, k_output=k_out,
                      gdn_norm_weight=gated_norm.weight, gdn_norm_output=gated_norm(q, k),
                      capture_mean=x.reshape(1, 2, 4, 4).mean(axis=-2))
        mx.eval(*arrays.values())
        cases.append({"dtype": dtype_name, **{name: value.reshape(-1).tolist() for name, value in arrays.items()}})
    multipliers = namespace["_build_layer_multipliers"](248320, 3, 0, 1234).tolist()
    sizes = [namespace["_find_nth_prime_after"](19999999, head + 1) for head in range(16)]
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    eos = 248044  # text_config.eos_token_id, not the generation stop-set's first ID
    history, tokens, rows = [eos, eos], [100, 200, eos, 300, 400], []
    for token in tokens:
        shifted = [token, history[-1], history[-2] if history[-1] != eos else eos]
        row = []
        for ngram in [2, 3]:
            mixed = shifted[0] * multipliers[0]
            for position in range(1, ngram):
                mixed ^= shifted[position] * multipliers[position]
            for head in range((ngram - 2) * 8, (ngram - 1) * 8):
                row.append(mixed % sizes[head] + offsets[head])
        rows.append(row)
        history = (history + [token])[-2:]
    output = {
        "mlx_commit": MLX_COMMIT, "mlx_version": mx.__version__, "mlx_vlm_commit": VLM_COMMIT,
        "mlx_vlm_blob": VLM_BLOB, "capture_patch_sha256": CAPTURE_PATCH_SHA256,
        "mlx_module_sha256": hashlib.sha256(Path(mx.__file__).read_bytes()).hexdigest(),
        "component_cases": cases,
        "ple": dict(vocab_size=248320, ngram_size=3, heads_per_ngram=8, seed=1234,
                    ple_layer_index=0, eos_token_id=eos, multipliers=multipliers,
                    head_vocab_sizes=sizes, padded_rows=math.ceil(total/128)*128, tokens=tokens, rows=rows),
        "dflash2": dict(capture="completed_layer_hc_mean", hc_count=4,
                       zero_based_layers=[5, 15, 25, 35, 45], hidden_size=2560,
                       target_rope_theta=10000000, draft_rope_theta=10000),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Generated Flash-Next component fixtures: {args.output}")


if __name__ == "__main__":
    main()
