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
    parser.add_argument(
        "--prompt-file", type=Path, help="Exact raw templated prompt, UTF-8"
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Pinned local image; saves patches, positions and tower output",
    )
    parser.add_argument(
        "--image-text-file", type=Path, help="User text following the image placeholder"
    )
    parser.add_argument(
        "--vision-trace",
        action="store_true",
        help="Save intermediate primary vision activations",
    )
    args = parser.parse_args()
    assert 1 <= args.steps <= 256
    source_root = Path(reference.__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
    assert not subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip()
    assert mx.__version__ == "0.32.5"
    mx.set_cache_limit(512 * 1024 * 1024)
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
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {}
    vision_weights = {}
    for filename in sorted(set(weight_map.values())):
        shard = mx.load(str(snapshot / filename))
        for name, value in shard.items():
            if (
                name.startswith("language_model.")
                and ".ngram_embedding." not in name
                and "mtp." not in name
            ):
                weights[name.removeprefix("language_model.")] = value
            elif args.image and name.startswith("vision_tower."):
                vision_weights[name.removeprefix("vision_tower.")] = value
        del shard
    quantization = root.get("quantization_config") or root["quantization"]

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
            return False
        params = quantization.get(f"language_model.{path}", quantization)
        return {key: params[key] for key in ["group_size", "bits", "mode"]}

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization["mode"],
        class_predicate=predicate,
    )
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    gc.collect()
    mx.eval(model.parameters())
    print(f"Loaded native reference in {time.monotonic() - start:.2f}s", flush=True)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    prompt = (
        args.prompt_file.read_text()
        if args.prompt_file
        else "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    )
    arrays = {}
    image_embeddings = None
    position_ids = None
    rope_delta = 0
    if args.image:
        from PIL import Image
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
        from mlx_vlm.models.qwen4_exp.vision import VisionModel
        from mlx_vlm.models.qwen4_exp.config import VisionConfig, ModelConfig

        processor_config = json.loads(
            (snapshot / "preprocessor_config.json").read_text()
        )
        bounds = processor_config["size"]
        processor = Qwen3VLImageProcessor(
            min_pixels=bounds["shortest_edge"],
            max_pixels=bounds["longest_edge"],
            image_mean=processor_config["image_mean"],
            image_std=processor_config["image_std"],
        )
        processed = processor.preprocess([Image.open(args.image).convert("RGB")])
        pixels = mx.array(processed["pixel_values"], dtype=mx.bfloat16)
        grid = mx.array(processed["image_grid_thw"], dtype=mx.int32)
        tower = VisionModel(VisionConfig.from_dict(root["vision_config"]))
        tower.eval()
        tower.load_weights(list(vision_weights.items()), strict=True)
        del vision_weights
        if args.vision_trace:
            from mlx_vlm.models.qwen3_vl import vision as vision_reference

            original_patch = vision_reference.PatchEmbed.__call__

            def patch_call(module, *a, **kw):
                value = original_patch(module, *a, **kw)
                arrays["vision_patch"] = value
                return value

            vision_reference.PatchEmbed.__call__ = patch_call
            original_pos = type(tower).fast_pos_embed_interpolate

            def pos_call(module, *a, **kw):
                value = original_pos(module, *a, **kw)
                arrays["vision_position"] = value
                return value

            type(tower).fast_pos_embed_interpolate = pos_call
            original_rot = type(tower).rot_pos_emb

            def rot_call(module, *a, **kw):
                value = original_rot(module, *a, **kw)
                arrays["vision_rotary"] = value
                return value

            type(tower).rot_pos_emb = rot_call
            block_ids = {id(block): i for i, block in enumerate(tower.blocks)}
            original_block = vision_reference.Qwen3VLMoEVisionBlock.__call__

            def block_call(module, *a, **kw):
                value = original_block(module, *a, **kw)
                arrays[f"vision_block_{block_ids[id(module)]}"] = value
                return value

            vision_reference.Qwen3VLMoEVisionBlock.__call__ = block_call
        image_embeddings, _ = tower(pixels, grid)
        mx.eval(image_embeddings)
        arrays.update(
            pixel_values=pixels, image_grid_thw=grid, vision_embeds=image_embeddings
        )
        count = int(mx.prod(grid).item()) // 4
        image_text = (
            args.image_text_file.read_text()
            if args.image_text_file
            else "Describe the image briefly."
        )
        prompt = (
            "<|im_start|>user\n<|vision_start|>"
            + "<|image_pad|>" * count
            + "<|vision_end|>"
            + image_text
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    if args.image:
        model.config = ModelConfig.from_dict(root)
        position_ids, delta = model.get_rope_index(
            mx.array([tokens], dtype=mx.int32), image_grid_thw=grid
        )
        rope_delta = int(delta.item())
        arrays.update(
            input_ids=mx.array([tokens], dtype=mx.int32), position_ids=position_ids
        )
        embeds = model.model.embed_tokens(mx.array([tokens], dtype=mx.int32))
        indices = [
            i for i, token in enumerate(tokens) if token == root["image_token_id"]
        ]
        embeds[:, mx.array(indices)] = image_embeddings[None]
        arrays["input_embeds"] = embeds
    cache = model.make_cache()
    layer_ids = {id(layer): index for index, layer in enumerate(model.model.layers)}
    original_layer_call = reference.Qwen4ExpDecoderLayer.__call__
    capture_enabled = [True]

    def traced_layer(layer, *inputs, **kwargs):
        result = original_layer_call(layer, *inputs, **kwargs)
        if capture_enabled[0]:
            arrays[f"prefill_layer_{layer_ids[id(layer)]}"] = result.reshape(
                *result.shape[:-1], cfg.hc_count, cfg.hidden_size
            ).mean(axis=-2)
        return result

    reference.Qwen4ExpDecoderLayer.__call__ = traced_layer
    generated = []
    current = tokens
    for step in range(args.steps):
        pos = (
            position_ids
            if step == 0
            else (
                mx.arange(len(tokens) + step - 1, len(tokens) + step, dtype=mx.int32)
                + rope_delta
            )[None, None, :]
        )
        if step > 0:
            pos = mx.broadcast_to(pos, (3, 1, 1))
        hidden = model.model(
            mx.array([current], dtype=mx.int32),
            cache=cache,
            inputs_embeds=arrays.get("input_embeds")
            if args.image and step == 0
            else None,
            position_ids=pos if args.image else None,
        )
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
    metadata = dict(
        reference_commit=commit,
        mlx_commit="a8f66b5901791a38ff527f249e1f1a84f46c03b6",
        mlx_version=mx.__version__,
        snapshot=str(snapshot),
        config_sha256=hashlib.sha256(raw).hexdigest(),
        index_sha256=hashlib.sha256(
            (snapshot / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        prompt_tokens=tokens,
        generated_tokens=generated,
        generated_text=tokenizer.decode(generated),
        elapsed_seconds=time.monotonic() - start,
        active_bytes=mx.get_active_memory(),
        peak_bytes=mx.get_peak_memory(),
        training=False,
        image=str(args.image) if args.image else None,
        image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest()
        if args.image
        else None,
        rope_delta=rope_delta,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
