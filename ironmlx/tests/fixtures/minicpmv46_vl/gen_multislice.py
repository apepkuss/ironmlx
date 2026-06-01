"""Generate MiniCPM-V-4.6-4bit MULTI-SLICE + MULTI-IMAGE VL reference logits via mlx-vlm.

This is the P3 Task 5 FULL-VLM ACCEPTANCE-GATE fixture generator. It drives the
FULL mlx-vlm processor + model (LLaVA-UHD slicing ENABLED, `slice_mode=True`,
`max_slice_nums=9`, `use_image_id=False`) on two scenarios and dumps, for each,
the EXACT ids the model consumes, every per-slice pixel tensor (in mlx-vlm's
emitted image-major order: image0 source + image0 patches, image1 source +
image1 patches), the per-slice grids, and the full-model last-token logits.

The two scenarios:
  1. SINGLE image that ACTUALLY slices (`coco_sample.jpg`, 640×480 → grid (2,1)
     → 3 slices: 1 source + 2 refine patches). This exercises the slice ordering
     + scatter for a single image whose prompt has 1 `<image>` block + `<slice>`
     blocks.
  2. TWO images (`coco_sample.jpg` + `multi_image/image_0.jpg`). Both slice; the
     prompt has TWO `<image>` markers, each replaced in-place by that image's
     full `<image>...</image>` + `<slice>...` placeholder block. Vision rows are
     concatenated image-major (mlx-vlm `get_vision_embedding` concatenates per
     image then the scatter walks `image_bound` spans in text order, which is
     image-major), so the Rust side must feed all slices image-major.

mlx-vlm's per-slice `pixel_values[i]` is a FLAT list of every slice across every
image (image-major). We re-derive the model's CHW→HWC transpose + `expand_dims(0)`
so each saved tensor is `[1, 14, n*14, 3]` HWC — exactly what the Rust
`compute_vision_embeds` consumes (matching `gen_multislice_preprocess.py` /
`gen_single_image_logits.py`).

Fixtures written (all int32/f32, gitignored .npy — NOT committed):
  Single-image-sliced:
    * `expected_input_ids_sliced.npy`   — int32 [S] ids the model consumes
    * `multislice_count.npy`            — int32 [1] total slice count (== 1 + gx*gy)
    * `multislice_grids.npy`            — int32 [count, 2] per-slice (grid_h, grid_w)
    * `multislice_pixels_{i}.npy`       — f32 [1, 14, n*14, 3] slice i pixel tensor
    * `expected_sliced_logits.npy`      — f32 [vocab] full-model last logits
  Two-image:
    * `expected_input_ids_2img.npy`     — int32 [S] ids the model consumes
    * `multislice_2img_count.npy`       — int32 [1] total slice count (both images)
    * `multislice_2img_grids.npy`       — int32 [count, 2] per-slice (grid_h, grid_w)
    * `multislice_2img_pixels_{i}.npy`  — f32 [1, 14, n*14, 3] slice i pixel tensor
    * `expected_2img_logits.npy`        — f32 [vocab] full-model last logits

Run from the editable mlx-vlm checkout (see memory reference_iron_rivals_baselines):

    cd /Users/xin/workspace/iron-rivals/mlx-vlm
    MINICPMV46_MODEL=<snapshot-dir> \
      uv run --with-editable . python \
      /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46_vl/gen_multislice.py
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
MAX_SLICE_NUMS = 9

FIX = Path(__file__).resolve().parents[1]
COCO_PATH = FIX / "p6_qwen35_vl" / "coco_sample.jpg"
IMG0_PATH = FIX / "p6_qwen35_vl" / "multi_image" / "image_0.jpg"

model_path = os.environ.get("MINICPMV46_MODEL")
if not model_path:
    raise SystemExit(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit checkpoint dir"
    )

print(f"mlx version: {mx.__version__}")
model, processor = load(model_path)


def chw_to_hwc_expand(cur_pixels_chw: np.ndarray) -> np.ndarray:
    """Reproduce the model's CHW→HWC transpose + expand_dims(0).

    The saved tensor `[1, 14, n*14, 3]` is exactly what the embeddings layer
    (and the Rust port `compute_vision_embeds`) consume.
    """
    cur_pixels_chw = np.asarray(cur_pixels_chw, dtype=np.float32)
    assert cur_pixels_chw.ndim == 3 and cur_pixels_chw.shape[0] == 3, (
        f"expected CHW-packed [3, 14, n*14], got {cur_pixels_chw.shape}"
    )
    cur_pixels_hwc = np.transpose(cur_pixels_chw, (1, 2, 0))  # [14, n*14, 3]
    return np.expand_dims(cur_pixels_hwc, 0)  # [1, 14, n*14, 3]


def run_scenario(tag: str, prompt: str, image_paths: list[Path]) -> None:
    """Drive the full processor + model for `image_paths` (image-major) and dump
    ids / per-slice pixels / per-slice grids / last-token logits under `tag`."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    for p, im in zip(image_paths, images):
        print(f"[{tag}] image: {p.name} size(WxH)={im.size}")

    # Full processor: slicing ENABLED; no <image_id> wrapper; no chat template.
    inputs = processor(
        text=prompt,
        images=images,
        slice_mode=True,
        max_slice_nums=MAX_SLICE_NUMS,
        use_image_id=False,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = np.asarray(inputs["input_ids"], dtype=np.int32)  # [1, S]
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1, (
        f"[{tag}] expected single un-padded sequence [1, S], got {input_ids.shape}"
    )
    ids_flat = input_ids[0]

    # pixel_values[0] is a FLAT list of every slice across every image (image-major):
    #   image0 source + image0 patches, image1 source + image1 patches, ...
    sample_pixels = list(inputs["pixel_values"][0])
    sample_tgt = np.asarray(inputs["tgt_sizes"][0], dtype=np.int32)  # [count, 2]
    grids = inputs["grids"][0]  # list[image] of (gx, gy)
    count = len(sample_pixels)
    assert count == sample_tgt.shape[0], (
        f"[{tag}] slice count {count} != tgt_sizes rows {sample_tgt.shape[0]}"
    )

    # Per-image slice counts: 1 + gx*gy per sliced image (source + patches).
    expected_count = 0
    for gx, gy in grids:
        expected_count += 1 + int(gx) * int(gy)
    assert count == expected_count, (
        f"[{tag}] flat slice count {count} != sum(1 + gx*gy) = {expected_count} "
        f"(grids={grids})"
    )
    print(f"[{tag}] grids(gx,gy)={[tuple(g) for g in grids]} flat_slice_count={count}")

    # Save per-slice pixel tensors (image-major, post CHW→HWC transpose) + grids.
    out_grids = np.zeros((count, 2), dtype=np.int32)
    for i, cur_pixels_chw in enumerate(sample_pixels):
        grid_h, grid_w = int(sample_tgt[i, 0]), int(sample_tgt[i, 1])
        out_grids[i] = (grid_h, grid_w)
        ipv = chw_to_hwc_expand(cur_pixels_chw)
        n = ipv.shape[2] // PATCH
        assert ipv.shape[1] == PATCH, (
            f"[{tag}] slice {i}: packed height must be {PATCH}, got {ipv.shape[1]}"
        )
        assert n == grid_h * grid_w, (
            f"[{tag}] slice {i}: packed n={n} != grid_h*grid_w={grid_h * grid_w}"
        )
        mx.save(
            str(OUT_DIR / f"{tag}_pixels_{i}.npy"),
            mx.array(ipv, dtype=mx.float32),
        )
        print(f"  [{tag}] slice {i}: grid=({grid_h},{grid_w}) n={n} shape={tuple(ipv.shape)}")

    # Image-token (248056) count in ids must equal total vision-embed rows so the
    # scatter is well-defined. Rows = sum over slices of (gh//4)*(gw//4).
    img_tok_count = int(np.count_nonzero(ids_flat == IMAGE_TOKEN_ID))
    expected_rows = int(sum((int(gh) // 4) * (int(gw) // 4) for gh, gw in out_grids))
    assert img_tok_count == expected_rows, (
        f"[{tag}] image-token count {img_tok_count} != sum (gh//4)*(gw//4) "
        f"= {expected_rows}"
    )

    # Vision-embed rows the full model produces (sanity: must equal img_tok_count).
    ve = model.get_vision_embedding([sample_pixels], [sample_tgt])[0]
    assert int(ve.shape[0]) == img_tok_count, (
        f"[{tag}] vision-embed rows {ve.shape[0]} != image-token count {img_tok_count}"
    )

    # FULL model forward: vision INCLUDED. Scatters vision feats into the
    # image_bound spans, then runs the Qwen3.5 text backbone.
    out = model(
        input_ids=mx.array(input_ids),
        pixel_values=[sample_pixels],
        tgt_sizes=[sample_tgt],
        image_bound=inputs["image_bound"],
    )
    logits_all = out.logits if hasattr(out, "logits") else out
    last = logits_all[0, -1, :].astype(mx.float32)
    mx.eval(last)
    vocab = int(last.shape[0])

    # Dump ids / count / grids / logits.
    mx.save(str(OUT_DIR / f"expected_input_ids_{tag}.npy"), mx.array(ids_flat, dtype=mx.int32))
    mx.save(str(OUT_DIR / f"{tag}_count.npy"), mx.array(np.array([count], dtype=np.int32)))
    mx.save(str(OUT_DIR / f"{tag}_grids.npy"), mx.array(out_grids))
    mx.save(str(OUT_DIR / f"expected_{tag}_logits.npy"), last)

    argmax = int(mx.argmax(last).item())
    print(
        f"[{tag}] count={count} N(image_tokens)={img_tok_count} S(seq_len)={len(ids_flat)} "
        f"vocab={vocab} argmax={argmax}"
    )


# --- Scenario 1: single image that slices ---------------------------------------
# `sliced`: ids → expected_input_ids_sliced.npy, pixels → sliced_pixels_{i}.npy,
# grids → sliced_grids.npy, count → sliced_count.npy, logits → expected_sliced_logits.npy.
run_scenario("sliced", "<image>Describe this image.", [COCO_PATH])

# --- Scenario 2: two images (image-major) ---------------------------------------
# `2img`: ids → expected_input_ids_2img.npy, pixels → 2img_pixels_{i}.npy,
# grids → 2img_grids.npy, count → 2img_count.npy, logits → expected_2img_logits.npy.
run_scenario(
    "2img",
    "<image><image>Describe these images.",
    [COCO_PATH, IMG0_PATH],
)

print(f"saved all multi-slice + multi-image fixtures to {OUT_DIR}")
