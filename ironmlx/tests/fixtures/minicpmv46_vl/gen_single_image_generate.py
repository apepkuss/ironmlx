"""Generate MiniCPM-V-4.6-4bit single-image GREEDY generation reference via mlx-vlm.

This is the P2b Task-4 ACCEPTANCE-GATE fixture generator. It is a thin extension
of the P2a logits generator (`gen_single_image_logits.py`): it drives the EXACT
same single-image processor pipeline (LLaVA-UHD slicing DISABLED via
`slice_mode=False` → one `<image>...image_token*N...</image>` placeholder block,
no `<slice>` sub-tiles), regenerates the three shared inputs fixtures so the Rust
side feeds byte-identical ids/pixels/grid, and ADDITIONALLY runs mlx-vlm's real
autoregressive generation loop (`mlx_vlm.generate.generate_step`) with GREEDY
decoding (temperature 0 → argmax) to capture the first K generated token ids.

The Rust e2e parity test (`tests/minicpmv46_single_image_generate_e2e.rs`) drives
ironmlx's `GenerationStream` over the SAME single-image input (same ids + pixel
tensor + grid) and asserts the first K greedy tokens match. P2a already proved the
model-level `forward_vl_chunk` last-token logits are bit-aligned; this fixture +
test close the loop on the FULL CLI/stream path (sequential VL prefill positions →
chunk loop → greedy sampler → sequential decode positions), a different code path
than the direct `forward_vl_chunk` call P2a exercised.

Fixtures written (all int32/f32, gitignored .npy — NOT committed):
  * `expected_input_ids_img.npy`  — int32 [S] ids the model consumes (placeholder
    already expanded to [im_start] + [248056]*N + [im_end]). Regenerated here so
    it stays in lockstep with the pixel/grid tensors below.
  * `input_pixel_values.npy`      — f32 [1, 14, n*14, 3] HWC pixels.
  * `input_grid.npy`              — int32 [gh, gw].
  * `expected_gen_tokens.npy`     — int32 [K] first K GREEDY generated token ids.

Run from the editable mlx-vlm checkout (see memory reference_iron_rivals_baselines):

    cd /Users/xin/workspace/iron-rivals/mlx-vlm
    MINICPMV46_MODEL=<snapshot-dir> \
      uv run --with-editable . python \
      /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46_vl/gen_single_image_generate.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm import load
from mlx_vlm.generate import generate_step

OUT_DIR = Path(__file__).parent
PATCH = 14
IMAGE_TOKEN_ID = 248056
# Number of greedy tokens to capture + assert in the Rust e2e test. Small (the
# acceptance criterion is the first token; K>1 confirms the decode-step position
# advance + cache continuity stay aligned, not just prefill).
K = 5

# Reuse the existing COCO fixture image (P1/P2a fixtures use the same).
IMAGE_PATH = (
    Path(__file__).resolve().parents[1] / "qwen35_vl" / "coco_sample.jpg"
)

# Same raw prompt as the P2a logits generator: a single `<image>` marker, NO
# chat template, so the saved ids are exactly what the model consumed and the
# Rust side can feed them verbatim (isolates LM/vision forward + generation-loop
# wiring from any chat-template / special-token discrepancy).
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
# one `<image>...</image>` placeholder block (single packed pixel tensor).
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

# N = image-token count actually present in input_ids.
img_tok_count = int(np.count_nonzero(ids_flat == IMAGE_TOKEN_ID))
N = img_tok_count
expected_N = (grid_h // 4) * (grid_w // 4)
assert N == expected_N, (
    f"image-token count N={N} must equal (gh//4)*(gw//4)={expected_N} "
    f"(grid={(grid_h, grid_w)})"
)

# --- GREEDY generation via mlx-vlm's real autoregressive loop --------------
# `generate_step` is the actual generation generator mlx-vlm's `generate`/
# `stream_generate` drive: it builds inputs_embeds (vision INCLUDED, scattered
# at the image_bound span), runs the language_model prefill, then the decode
# loop. temperature=0 → argmax (deterministic greedy), no penalties. We pass the
# processor's `tgt_sizes` + `image_bound` through as kwargs (consumed by
# MiniCPM_V.get_input_embeddings). prefill_step_size=None → single-shot prefill
# (no chunking), matching the Rust test's prefill_chunk_size=0.
gen_kwargs = {
    "tgt_sizes": [np.array([[grid_h, grid_w]], dtype=np.int32)],
    "image_bound": inputs["image_bound"],
}

gen_tokens: list[int] = []
for tok, _logprobs in generate_step(
    mx.array(input_ids),
    model,
    [[cur_pixels_chw]],
    mask=None,
    max_tokens=K,
    temperature=0.0,
    prefill_step_size=None,
    **gen_kwargs,
):
    gen_tokens.append(int(tok))
    if len(gen_tokens) >= K:
        break

assert len(gen_tokens) == K, f"expected {K} greedy tokens, got {len(gen_tokens)}"
gen_arr = np.array(gen_tokens, dtype=np.int32)

# Dump fixtures (shared inputs regenerated in lockstep + the new gen tokens).
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
mx.save(str(OUT_DIR / "expected_gen_tokens.npy"), mx.array(gen_arr))

decoded = processor.tokenizer.decode(gen_tokens)
print(
    f"grid=({grid_h},{grid_w}) n={n} N(image_tokens)={N} S(seq_len)={len(ids_flat)} "
    f"K={K} gen_tokens={gen_tokens}"
)
print(f"decoded greedy text: {decoded!r}")
print(f"saved 4 fixtures (3 inputs + expected_gen_tokens) to {OUT_DIR}")
