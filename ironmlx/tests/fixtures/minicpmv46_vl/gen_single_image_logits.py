"""Generate MiniCPM-V-4.6-4bit single-image VL reference logits via mlx-vlm.

This is the P2a ACCEPTANCE-GATE fixture generator. It drives the FULL mlx-vlm
model (vision tower + cross-modal scatter + Qwen3.5 text backbone) on a fixed
single-image prompt with LLaVA-UHD slicing DISABLED (`slice_mode=False`), so the
image expands to exactly ONE `<image>...image_token*N...</image>` placeholder
block (no `<slice>` sub-tiles). It dumps:

  * `expected_input_ids_img.npy` — the EXACT int32 token ids the model consumes,
    with the `<image>` marker expanded to `[im_start] + [248056]*N + [im_end]`
    by the real processor pipeline (`_encode_with_multimodal_placeholders`).
  * `input_pixel_values.npy` — the single-slice `[1, 14, n*14, 3]` HWC pixel
    tensor (captured identically to `gen_vision_embeds.py`: post CHW→HWC
    transpose + `expand_dims(0)`, the layout the Rust `compute_vision_embeds`
    consumes).
  * `input_grid.npy` — `[gh, gw]` int32.
  * `expected_single_image_logits.npy` — f32 `[vocab]` last-token logits from the
    FULL model forward `model(input_ids, pixel_values=..., tgt_sizes=...,
    image_bound=...)` (vision INCLUDED, not language_model-only).

The Rust parity test (`tests/minicpmv46_single_image_parity.rs`) feeds the saved
ids + pixel tensor through `MiniCpmV46Model::forward_vl_chunk`, which (a) embeds
the ids, (b) scatters the SigLIP vision embeds into the `image_token_id` (248056)
positions via `replace_image_tokens`, (c) runs the text transformer, and (d)
projects the last-token logits. ironmlx scatters at positions where
`input_ids == 248056`; mlx-vlm scatters at the `image_bound` span `[im_start+1,
im_end)`. These two position sets are IDENTICAL by construction here (the only
248056 tokens are exactly the fill tokens between im_start/im_end), so the parity
is well-defined.

Run from the editable mlx-vlm checkout (see memory reference_iron_rivals_baselines):

    cd /Users/xin/workspace/iron-rivals/mlx-vlm
    MINICPMV46_MODEL=<snapshot-dir> \
      uv run --with-editable . python \
      /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_logits.py
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
IMAGE_TOKEN_ID = 248056

# Reuse the existing COCO fixture image (P1 vision-embeds fixture uses the same).
IMAGE_PATH = (
    Path(__file__).resolve().parents[1] / "p6_qwen35_vl" / "coco_sample.jpg"
)

# A simple raw prompt with a single `<image>` marker. We do NOT apply a chat
# template — feeding the raw ids isolates LM/vision-forward correctness from any
# chat-template / tokenizer-special-token discrepancy (same philosophy as the
# text-only `gen_logits.py`). The processor expands `<image>` in-place.
PROMPT = "<image>Describe this image."

model_path = os.environ.get("MINICPMV46_MODEL")
if not model_path:
    raise SystemExit(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit checkpoint dir"
    )

print(f"mlx version: {mx.__version__}")
model, processor = load(model_path)

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"image: {IMAGE_PATH.name} size(WxH)={image.size}")

# Drive the FULL processor with slicing DISABLED so the image expands to exactly
# one `<image>...</image>` placeholder block (single packed pixel tensor). The
# processor returns padded input_ids, the per-image pixel_values/tgt_sizes, and
# the image_bound spans the full model's `__call__` uses to scatter vision feats.
inputs = processor(
    text=PROMPT,
    images=[image],
    slice_mode=False,
    use_image_id=False,  # no <image_id>N</image_id> wrapper — keeps ids minimal
    padding=False,
    add_special_tokens=False,
)

input_ids = np.asarray(inputs["input_ids"], dtype=np.int32)  # [1, S]
assert input_ids.ndim == 2 and input_ids.shape[0] == 1, (
    f"expected single un-padded sequence [1, S], got {input_ids.shape}"
)
ids_flat = input_ids[0]

# Pixel values: list[batch] -> list[image] -> np.ndarray (CHW-packed [3,14,n*14]).
pixel_values = inputs["pixel_values"]
tgt_sizes = inputs["tgt_sizes"]
assert len(pixel_values) == 1, f"expected 1 batch, got {len(pixel_values)}"
assert len(pixel_values[0]) == 1, (
    f"slice_mode off must yield exactly 1 pixel tensor, got {len(pixel_values[0])}"
)

cur_pixels_chw = np.asarray(pixel_values[0][0], dtype=np.float32)  # [3, 14, n*14]
tgt = np.asarray(tgt_sizes[0][0], dtype=np.int32)  # [grid_h, grid_w]
grid_h, grid_w = int(tgt[0]), int(tgt[1])

assert cur_pixels_chw.ndim == 3 and cur_pixels_chw.shape[0] == 3, (
    f"expected CHW-packed [3, 14, n*14], got {cur_pixels_chw.shape}"
)

# Reproduce the model's CHW->HWC transpose + expand_dims so the saved tensor is
# what the embeddings layer (and the Rust port) actually consume.
cur_pixels_hwc = np.transpose(cur_pixels_chw, (1, 2, 0))  # [14, n*14, 3]
input_pixel_values = np.expand_dims(cur_pixels_hwc, 0)  # [1, 14, n*14, 3]

n = input_pixel_values.shape[2] // PATCH
assert input_pixel_values.shape[1] == PATCH, (
    f"packed height must be patch={PATCH}, got {input_pixel_values.shape[1]}"
)
assert n == grid_h * grid_w, (
    f"packed patch count n={n} must equal grid_h*grid_w={grid_h * grid_w}"
)

# N = image-token count actually present in input_ids. The processor builds the
# placeholder block as [im_start] + [image_token_id]*token_count + [im_end] with
# token_count = (grid_h*grid_w)//16 for downsample_mode "16x".
img_tok_count = int(np.count_nonzero(ids_flat == IMAGE_TOKEN_ID))
N = img_tok_count
expected_N = (grid_h // 4) * (grid_w // 4)
assert N == expected_N, (
    f"image-token count N={N} must equal (gh//4)*(gw//4)={expected_N} "
    f"(grid={(grid_h, grid_w)})"
)

# Vision-embed rows the full model produces (sanity: must equal N).
ve = model.get_vision_embedding(
    [[cur_pixels_chw]],
    [np.array([[grid_h, grid_w]], dtype=np.int32)],
)[0]
assert int(ve.shape[0]) == N, (
    f"vision-embed rows {ve.shape[0]} must equal image-token count N={N}"
)

# FULL model forward: vision INCLUDED. The model scatters vision feats into the
# image_bound span, then runs the Qwen3.5 text backbone. Returns
# LanguageModelOutput with `.logits` of shape [1, S, vocab].
out = model(
    input_ids=mx.array(input_ids),
    pixel_values=[[cur_pixels_chw]],
    tgt_sizes=[np.array([[grid_h, grid_w]], dtype=np.int32)],
    image_bound=inputs["image_bound"],
)
logits_all = out.logits if hasattr(out, "logits") else out
last = logits_all[0, -1, :].astype(mx.float32)
mx.eval(last)
vocab = int(last.shape[0])

# Dump fixtures.
mx.save(
    str(OUT_DIR / "expected_input_ids_img.npy"),
    mx.array(ids_flat, dtype=mx.int32),
)
mx.save(
    str(OUT_DIR / "input_pixel_values.npy"),
    mx.array(input_pixel_values, dtype=mx.float32),
)
mx.save(
    str(OUT_DIR / "input_grid.npy"),
    mx.array(np.array([grid_h, grid_w], dtype=np.int32)),
)
mx.save(str(OUT_DIR / "expected_single_image_logits.npy"), last)

argmax = int(mx.argmax(last).item())
print(
    f"grid=({grid_h},{grid_w}) n={n} N(image_tokens)={N} S(seq_len)={len(ids_flat)} "
    f"vocab={vocab} argmax={argmax}"
)
print(f"saved 4 fixtures to {OUT_DIR}")
