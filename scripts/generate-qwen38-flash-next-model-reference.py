#!/usr/bin/env python3
"""Generate a fixed-trajectory full text oracle with the pinned MLX-VLM.

The approximately 30 GiB PLE table stays in its original SSD ranges. Ordinary
text forwards are used, without MTP, speculative verification or batch profiles.
Run this process separately from the native full-weight qualification process.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer
import mlx_vlm.models.qwen4_exp.language as reference
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.ple_storage import build_quantized_ple_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    assert 1 <= args.steps <= 256
    source_root = Path(reference.__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
    assert not subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip()
    assert mx.__version__ == "0.32.5"
    mx.set_cache_limit(512*1024*1024)
    mx.set_wired_limit(96 << 30)
    start = time.monotonic()
    snapshot = args.snapshot.resolve()
    raw = (snapshot / "config.json").read_bytes()
    root = json.loads(raw)
    cfg = TextConfig.from_dict(root["text_config"])
    manifest = args.output.with_suffix(".ple-manifest.json").resolve()
    build_quantized_ple_manifest(snapshot, manifest, cache_rows=4096)
    cfg.ple_storage = dict(manifest=str(manifest), cache_rows=4096)
    model = reference.LanguageModel(cfg)
    model.eval()
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for filename in sorted(set(weight_map.values())):
        shard = mx.load(str(snapshot / filename))
        for name, value in shard.items():
            if name.startswith("language_model.") and ".ngram_embedding." not in name and "mtp." not in name:
                weights[name.removeprefix("language_model.")] = value
        del shard
    quantization = root.get("quantization_config") or root["quantization"]
    def predicate(path, module):
        if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
            return False
        params = quantization.get(f"language_model.{path}", quantization)
        return {key: params[key] for key in ["group_size", "bits", "mode"]}
    nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                mode=quantization["mode"], class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    del weights
    gc.collect()
    mx.eval(model.parameters())
    print(f"Loaded native reference in {time.monotonic()-start:.2f}s", flush=True)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    cache = model.make_cache()
    arrays = {}
    layer_ids = {id(layer): index for index, layer in enumerate(model.model.layers)}
    original_layer_call = reference.Qwen4ExpDecoderLayer.__call__
    capture_enabled = [True]
    def traced_layer(layer, *inputs, **kwargs):
        result = original_layer_call(layer, *inputs, **kwargs)
        if capture_enabled[0]:
            arrays[f"prefill_layer_{layer_ids[id(layer)]}"] = result.reshape(
                *result.shape[:-1], cfg.hc_count, cfg.hidden_size).mean(axis=-2)
        return result
    reference.Qwen4ExpDecoderLayer.__call__ = traced_layer
    generated = []
    current = tokens
    for step in range(args.steps):
        hidden = model.model(mx.array([current], dtype=mx.int32), cache=cache)
        logits = model.lm_head(hidden)
        mx.eval(logits)
        capture_enabled[0] = False
        arrays[f"logits_{step}"] = logits[:, -1:]
        token = int(mx.argmax(logits[0, -1]).item())
        generated.append(token)
        current = [token]
        print(f"step={step} token={token}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), arrays)
    metadata = dict(reference_commit=commit, mlx_commit="a8f66b5901791a38ff527f249e1f1a84f46c03b6",
        mlx_version=mx.__version__, snapshot=str(snapshot), config_sha256=hashlib.sha256(raw).hexdigest(),
        index_sha256=hashlib.sha256((snapshot/"model.safetensors.index.json").read_bytes()).hexdigest(),
        prompt_tokens=tokens, generated_tokens=generated, generated_text=tokenizer.decode(generated),
        elapsed_seconds=time.monotonic()-start, active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory(),training=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
