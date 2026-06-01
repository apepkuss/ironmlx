"""Generate MiniCPM-V-4.6-4bit LLaVA-UHD MULTI-SLICE preprocess reference.

P3 Task 2 parity. Drives mlx-vlm's `MiniCPMVImageProcessor.preprocess` with
slicing ENABLED (`slice_mode=True`, `max_slice_nums=9`) on a high-res image that
ACTUALLY slices, then dumps — per slice, in mlx-vlm's emitted order (source
overview first, then refine-image patches row-major) — the packed pixel tensor
and grid, plus the total slice count, so the Rust `preprocess_sliced` parity
test (`tests/minicpmv46_multislice_preprocess_parity.rs`) can compare 1:1.

We capture each slice's pixel tensor EXACTLY as `Model.get_vision_embedding`'s
vision tower consumes it: mlx-vlm's per-slice `cur_pixels` is CHW-packed
`[3, 14, n*14]`; `get_vision_embedding` transposes it `(1,2,0)` → HWC
`[14, n*14, 3]` then `expand_dims(0)` → `[1, 14, n*14, 3]`. We dump that
post-transpose tensor, matching the layout `image_processor::slice_to_array`
emits (and `gen_vision_embeds.py`'s single-slice fixture).

Fixture image: `p6_qwen35_vl/coco_sample.jpg` (640×480). ratio = 640*480/448² =
1.5306 → ceil → multiple=2 → best_grid (gx=2, gy=1) → 3 slices (1 source + 2
refine patches). Source grid (gh,gw)=(28,36); each refine patch (gh,gw)=(40,28).

Outputs (all gitignored via the fixture dir `.gitignore` `input_*` / `expected_*`):
  - `multislice_count.npy`        — int32 [1], total slice count (== 1 + gx*gy)
  - `multislice_grids.npy`        — int32 [count, 2], per-slice (grid_h, grid_w)
  - `multislice_pixels_{i}.npy`   — f32 [1, 14, n_i*14, 3], slice i's pixel tensor

Run from the editable mlx-vlm checkout (see memory reference_iron_rivals_baselines):

    cd /Users/xin/workspace/iron-rivals/mlx-vlm
    MINICPMV46_MODEL=<snapshot-dir> \
      uv run --with-editable . python \
      /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46_vl/gen_multislice_preprocess.py
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
MAX_SLICE_NUMS = 9

IMAGE_PATH = Path(__file__).resolve().parents[1] / "p6_qwen35_vl" / "coco_sample.jpg"

model_path = os.environ.get("MINICPMV46_MODEL")
if not model_path:
    raise SystemExit(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit checkpoint dir"
    )

print(f"mlx version: {mx.__version__}")
_model, processor = load(model_path)
image_processor = processor.image_processor

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"image: {IMAGE_PATH.name} size(WxH)={image.size}")

# Multi-slice: enable LLaVA-UHD slicing with the checkpoint default cap.
image_processor.slice_mode = True
image_processor.max_slice_nums = MAX_SLICE_NUMS
out = image_processor.preprocess([[image]])

pixel_values = out["pixel_values"][0]  # list[image-slice] of CHW-packed arrays
tgt_sizes = np.asarray(out["tgt_sizes"][0], dtype=np.int32)  # [count, 2] (gh, gw)
best_grid = out["grids"][0][0]  # (gx, gy) for the (single) image
count = len(pixel_values)

print(f"best_grid (gx,gy)={tuple(best_grid)}  slice_count={count}")
assert count > 1, (
    f"fixture image must actually slice (multiple>1); got count={count} "
    f"(grid={best_grid}) — pick a higher-res image"
)
assert count == tgt_sizes.shape[0], (
    f"slice count {count} != tgt_sizes rows {tgt_sizes.shape[0]}"
)
# count == 1 + gx*gy (source overview + refine patches).
gx, gy = int(best_grid[0]), int(best_grid[1])
assert count == 1 + gx * gy, f"count {count} != 1 + gx*gy = {1 + gx * gy}"

grids = np.zeros((count, 2), dtype=np.int32)
for i, cur_pixels_chw in enumerate(pixel_values):
    cur_pixels_chw = np.asarray(cur_pixels_chw, dtype=np.float32)
    assert cur_pixels_chw.ndim == 3 and cur_pixels_chw.shape[0] == 3, (
        f"slice {i}: expected CHW-packed [3, 14, n*14], got {cur_pixels_chw.shape}"
    )
    grid_h, grid_w = int(tgt_sizes[i, 0]), int(tgt_sizes[i, 1])
    grids[i] = (grid_h, grid_w)

    # Reproduce the model's CHW->HWC transpose + expand_dims so the saved tensor
    # is what the embeddings layer (and the Rust port) actually consumes.
    cur_pixels_hwc = np.transpose(cur_pixels_chw, (1, 2, 0))  # [14, n*14, 3]
    input_pixel_values = np.expand_dims(cur_pixels_hwc, 0)  # [1, 14, n*14, 3]

    n = input_pixel_values.shape[2] // PATCH
    assert input_pixel_values.shape[1] == PATCH, (
        f"slice {i}: packed height must be patch={PATCH}, "
        f"got {input_pixel_values.shape[1]}"
    )
    assert n == grid_h * grid_w, (
        f"slice {i}: packed patch count n={n} != grid_h*grid_w={grid_h * grid_w}"
    )

    mx.save(
        str(OUT_DIR / f"multislice_pixels_{i}.npy"),
        mx.array(input_pixel_values, dtype=mx.float32),
    )
    print(
        f"  slice {i}: grid=({grid_h},{grid_w}) n={n} "
        f"shape={tuple(input_pixel_values.shape)}"
    )

mx.save(str(OUT_DIR / "multislice_grids.npy"), mx.array(grids))
mx.save(
    str(OUT_DIR / "multislice_count.npy"),
    mx.array(np.array([count], dtype=np.int32)),
)

print(f"saved {count} slices + grids + count to {OUT_DIR}")
