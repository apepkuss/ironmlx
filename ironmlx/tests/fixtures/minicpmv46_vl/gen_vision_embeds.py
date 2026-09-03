"""Generate MiniCPM-V-4.6-4bit vision-embeds reference via mlx-vlm.

The Rust parity test (`tests/minicpmv46_vision_parity.rs`) isolates the
VISION STACK from the LLaVA-UHD preprocessing/slicing pipeline (that is P2/P3).
So this script:

  1. Preprocesses ONE fixture image with slicing DISABLED (`slice_mode=False`),
     yielding exactly one packed pixel tensor + one tgt-size row.
  2. Captures the pixel tensor EXACTLY as `Model.get_vision_embedding`'s vision
     tower consumes it: the per-image `cur_pixels` is CHW-packed
     `[3, 14, n*14]`; `get_vision_embedding` transposes it `(1,2,0)` → HWC
     `[14, n*14, 3]` then `expand_dims(0)` → `[1, 14, n*14, 3]` before calling
     `vision_tower.embeddings`. We dump that post-transpose tensor so the Rust
     `compute_vision_embeds` (which expects `[1, 14, n*14, 3]`) consumes the
     identical bytes.
  3. Dumps the merged reference embeds `[N, 1024]` from
     `model.get_vision_embedding([[cur_pixels]], [[tgt]])[0]`.

Run from the editable mlx-vlm checkout (see memory reference_iron_rivals_baselines):

    cd /Users/xin/workspace/iron-rivals/mlx-vlm
    MINICPMV46_MODEL=<snapshot-dir> \
      uv run --with-editable . python \
      /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46_vl/gen_vision_embeds.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm import load

OUT_DIR = Path(__file__).parent
PATCH = 14

# Reuse the existing COCO fixture image (P6 Qwen3.5-VL shares it).
IMAGE_PATH = (
    Path(__file__).resolve().parents[1] / "qwen35_vl" / "coco_sample.jpg"
)

model_path = os.environ.get("MINICPMV46_MODEL")
if not model_path:
    raise SystemExit(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit checkpoint dir"
    )

print(f"mlx version: {mx.__version__}")
model, processor = load(model_path)
image_processor = processor.image_processor

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"image: {IMAGE_PATH.name} size(WxH)={image.size}")

# Single-slice: disable LLaVA-UHD slicing → exactly one packed pixel tensor.
# `preprocess` reads `self.slice_mode` (no per-call kwarg), so set it directly.
image_processor.slice_mode = False
out = image_processor.preprocess([[image]])

pixel_values = out["pixel_values"]  # list[batch] -> list[image] -> np.ndarray
tgt_sizes = out["tgt_sizes"]  # list[batch] -> np.ndarray [num_images, 2]

assert len(pixel_values) == 1, f"expected 1 batch, got {len(pixel_values)}"
assert len(pixel_values[0]) == 1, (
    f"slicing must be disabled — expected 1 pixel tensor, got {len(pixel_values[0])}"
)

# Pre-transpose CHW-packed tensor `[3, 14, n*14]` — the array `get_vision_embedding`
# itself transposes/expands.
cur_pixels_chw = np.asarray(pixel_values[0][0], dtype=np.float32)
tgt = np.asarray(tgt_sizes[0][0], dtype=np.int32)  # [grid_h, grid_w]
grid_h, grid_w = int(tgt[0]), int(tgt[1])

assert cur_pixels_chw.ndim == 3 and cur_pixels_chw.shape[0] == 3, (
    f"expected CHW-packed [3, 14, n*14], got {cur_pixels_chw.shape}"
)

# Reproduce the model's CHW->HWC transpose + expand_dims so the saved tensor is
# what the embeddings layer (and the Rust port) actually consumes.
cur_pixels_hwc = np.transpose(cur_pixels_chw, (1, 2, 0))  # [14, n*14, 3]
input_pixel_values = np.expand_dims(cur_pixels_hwc, 0)  # [1, 14, n*14, 3]

n = input_pixel_values.shape[2] // PATCH
assert input_pixel_values.shape[1] == PATCH, (
    f"packed height must be patch={PATCH}, got {input_pixel_values.shape[1]}"
)
assert n == grid_h * grid_w, (
    f"packed patch count n={n} must equal grid_h*grid_w={grid_h * grid_w}"
)

mx.save(
    str(OUT_DIR / "input_pixel_values.npy"),
    mx.array(input_pixel_values, dtype=mx.float32),
)
mx.save(
    str(OUT_DIR / "input_grid.npy"),
    mx.array(np.array([grid_h, grid_w], dtype=np.int32)),
)

# Reference embeds: feed the PRE-transpose CHW tensor + tgt with the nesting
# `get_vision_embedding` expects (pixel_values[batch][image], tgt_sizes[batch]).
emb = model.get_vision_embedding(
    [[cur_pixels_chw]],
    [np.array([[grid_h, grid_w]], dtype=np.int32)],
)[0]
emb = emb.astype(mx.float32)
mx.eval(emb)
mx.save(str(OUT_DIR / "expected_vision_embeds.npy"), emb)

print(
    f"grid=({grid_h},{grid_w}) n={n} "
    f"input_pixel_values.shape={tuple(input_pixel_values.shape)} "
    f"emb.shape={tuple(emb.shape)}"
)
print(f"saved fixtures to {OUT_DIR}")
