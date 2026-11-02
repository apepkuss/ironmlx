#!/usr/bin/env python3
"""Generate real-weight HC/GDN layer oracles from the pinned MLX-VLM source.

Uses bounded checkpoint ranges, not a whole-model load. Artifacts are local
qualification data and must be paired with the exact snapshot they record.
"""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import (
    Qwen4ExpGatedDeltaNet, Qwen4ExpGatedResidual, Qwen4ExpAttention, QSAKVCache,
    Qwen4ExpPLELayer)
from mlx_vlm.models.qwen4_exp.ple_storage import build_quantized_ple_manifest
import mlx_vlm.models.qwen4_exp.language as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(reference.__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
    assert not subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip()
    assert mx.__version__ == "0.32.5"
    snapshot = args.snapshot.resolve()
    config_raw = (snapshot / "config.json").read_bytes()
    cfg = TextConfig.from_dict(json.loads(config_raw)["text_config"])
    index_raw = (snapshot / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index_raw)["weight_map"]
    headers = {}
    values = []
    prefixes = ["language_model.model.layers.0.linear_attn.",
                "language_model.model.layers.0.attn_hyper_connection.",
                "language_model.model.layers.3.self_attn.",
                "language_model.model.layers.1.ple."]
    total = 0
    # Extract only selected tensor ranges into one tiny decoder envelope.
    for name in sorted(weight_map):
        if not any(name.startswith(p) for p in prefixes):
            continue
        if ".ngram_embedding." in name:
            continue
        filename = weight_map[name]
        if filename not in headers:
            with (snapshot / filename).open("rb") as f:
                length = struct.unpack("<Q", f.read(8))[0]
                headers[filename] = (length + 8, json.loads(f.read(length)))
        start, header = headers[filename]
        info = header[name]
        begin, end = info["data_offsets"]
        total += end - begin
        assert total < 128 * 1024 * 1024
        with (snapshot / filename).open("rb") as f:
            f.seek(start + begin)
            data = f.read(end - begin)
        assert len(data) == end - begin
        values.append((name, info, data))
    envelope_header, payload = {}, bytearray()
    for name, info, data in values:
        envelope_header[name] = dict(dtype=info["dtype"], shape=info["shape"],
                                     data_offsets=[len(payload), len(payload)+len(data)])
        payload.extend(data)
    header = json.dumps(envelope_header).encode()
    header += b" " * (-len(header) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_path = args.output.with_suffix(".selected.safetensors")
    selected_path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    weights = mx.load(str(selected_path))

    def load(module, prefix):
        nn.quantize(module, group_size=32, bits=4)
        module.load_weights([(k[len(prefix):], v) for k, v in weights.items() if k.startswith(prefix)], strict=True)
        module.eval()
        mx.eval(module.parameters())
        return module

    hc = load(Qwen4ExpGatedResidual(cfg), prefixes[1])
    gdn = load(Qwen4ExpGatedDeltaNet(cfg), prefixes[0])
    length = 6
    hyper = (mx.arange(length*cfg.hidden_size*cfg.hc_count, dtype=mx.float32) % 31 - 15) / 16
    hyper = hyper.reshape(1, length, -1).astype(mx.bfloat16)
    mixed, residual, injection = hc(hyper)
    cache = ArraysCache(2)
    output = gdn(mixed, cache=cache)
    combined = residual + (output[..., None, :] * injection[..., None]).reshape(residual.shape)
    arrays = dict(hyper=hyper, mixed=mixed, injection=injection, output=output,
                  combined=combined, conv_state=cache[0], recurrent_state=cache[1])
    mx.eval(*arrays.values())
    attention = load(Qwen4ExpAttention(cfg), prefixes[2])
    attention_length = 2053  # crosses the real 2048-token QSA budget
    attention_input = (mx.arange(attention_length*cfg.hidden_size, dtype=mx.float32) % 37 - 18) / 16
    attention_input = attention_input.reshape(1, attention_length, -1).astype(mx.bfloat16)
    attention_cache = QSAKVCache()
    attention_output = attention(attention_input, mask="causal", cache=attention_cache)
    arrays.update(attention_input=attention_input, attention_output=attention_output,
                  attention_keys=attention_cache.keys[:,:,:attention_length],
                  attention_values=attention_cache.values[:,:,:attention_length],
                  attention_raw_index_keys=attention_cache.index_keys,
                  attention_block_keys=attention_cache.index_block_keys)
    mx.eval(*arrays.values())
    manifest = args.output.with_suffix(".ple-manifest.json").resolve()
    build_quantized_ple_manifest(snapshot, manifest, cache_rows=128)
    cfg.ple_storage = dict(manifest=str(manifest), cache_rows=128)
    ple = load(Qwen4ExpPLELayer(cfg, 1, 0), prefixes[3])
    ple_tokens = mx.array([[100, 200, cfg.eos_token_id, 300, 400, 500]], dtype=mx.int32)
    ple_cache = ArraysCache(4)
    ple_output = ple(hyper, ple_tokens, ple_cache, None)
    arrays.update(ple_tokens=ple_tokens, ple_output=ple_output,
                  ple_conv_state=ple_cache[2], ple_history=ple_cache[3])
    mx.eval(*arrays.values())
    mx.save_safetensors(str(args.output), arrays)
    metadata = dict(mlx_commit="a8f66b5901791a38ff527f249e1f1a84f46c03b6", mlx_version=mx.__version__,
                    reference_commit=commit, config_sha256=hashlib.sha256(config_raw).hexdigest(),
                    index_sha256=hashlib.sha256(index_raw).hexdigest(), snapshot=str(snapshot),
                    selected_payload_sha256=hashlib.sha256(payload).hexdigest(), selected_bytes=total,
                    sequence_length=length, training=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
