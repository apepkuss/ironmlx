#!/usr/bin/env python3
"""Generate a pinned vision-only oracle using the checkpoint's processor bounds."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mlx.core as mx
from PIL import Image
from mlx_vlm.models.qwen4_exp.config import VisionConfig
from mlx_vlm.models.qwen4_exp.vision import VisionModel
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
import mlx_vlm.models.qwen4_exp.vision as reference

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--snapshot", type=Path, required=True)
p.add_argument("--image", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--trace", action="store_true", help="Save intermediate vision activations"
)
a = p.parse_args()
source = Path(reference.__file__).resolve().parents[3]
sha = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
assert sha == "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
assert not subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain"], text=True
).strip()
assert mx.__version__ == "0.32.5"
root = json.loads((a.snapshot / "config.json").read_text())
processor = json.loads((a.snapshot / "preprocessor_config.json").read_text())
bounds = processor["size"]
processed = Qwen3VLImageProcessor(
    min_pixels=bounds["shortest_edge"],
    max_pixels=bounds["longest_edge"],
    image_mean=processor["image_mean"],
    image_std=processor["image_std"],
).preprocess([Image.open(a.image).convert("RGB")])
pixels = mx.array(processed["pixel_values"], dtype=mx.bfloat16)
grid = mx.array(processed["image_grid_thw"], dtype=mx.int32)
weights = {}
index = json.loads((a.snapshot / "model.safetensors.index.json").read_text())[
    "weight_map"
]
for file in sorted(
    {file for name, file in index.items() if name.startswith("vision_tower.")}
):
    for name, tensor in mx.load(str(a.snapshot / file)).items():
        if name.startswith("vision_tower."):
            weights[name.removeprefix("vision_tower.")] = tensor
model = VisionModel(VisionConfig.from_dict(root["vision_config"]))
model.eval()
model.load_weights(list(weights.items()), strict=True)
arrays = {}
if a.trace:
    from mlx_vlm.models.qwen3_vl import vision as vr

    patch_call = vr.PatchEmbed.__call__

    def patch(module, *args, **kwargs):
        value = patch_call(module, *args, **kwargs)
        arrays["vision_patch"] = value
        return value

    vr.PatchEmbed.__call__ = patch
    pos_call = type(model).fast_pos_embed_interpolate

    def pos(module, *args, **kwargs):
        value = pos_call(module, *args, **kwargs)
        arrays["vision_position"] = value
        return value

    type(model).fast_pos_embed_interpolate = pos
    rot_call = type(model).rot_pos_emb

    def rot(module, *args, **kwargs):
        value = rot_call(module, *args, **kwargs)
        arrays["vision_rotary"] = value
        return value

    type(model).rot_pos_emb = rot
    block_call = vr.Qwen3VLMoEVisionBlock.__call__
    block_ids = {id(block): i for i, block in enumerate(model.blocks)}

    def block(module, *args, **kwargs):
        value = block_call(module, *args, **kwargs)
        arrays[f"vision_block_{block_ids[id(module)]}"] = value
        return value

    vr.Qwen3VLMoEVisionBlock.__call__ = block
output, _ = model(pixels, grid)
mx.eval(output)
a.output.parent.mkdir(parents=True, exist_ok=True)
arrays.update(pixel_values=pixels, image_grid_thw=grid, vision_embeds=output)
mx.save_safetensors(str(a.output), arrays)
metadata = dict(
    reference_commit=sha,
    mlx_commit="a8f66b5901791a38ff527f249e1f1a84f46c03b6",
    image=str(a.image.resolve()),
    image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
    config_sha256=hashlib.sha256((a.snapshot / "config.json").read_bytes()).hexdigest(),
    grid=grid.tolist(),
    output_shape=output.shape,
    training=False,
)
a.output.with_suffix(".json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(metadata, ensure_ascii=False), flush=True)
