#!/usr/bin/env python3
"""Generate reference fixtures for P6 Qwen3.5-VL logits-match e2e test.

Runs end-to-end through mlx-vlm 0.5.0:
  1. Loads Qwen3.5-VL model + processor
  2. Preprocesses coco_sample.jpg with the official processor
  3. Runs a single forward pass (NO token generation)
  4. Saves input_ids, pixel_values, image_grid_thw, last-position logits, first_token

Usage:
    QWEN35_MODEL=<snapshot-dir> ~/.venvs/mlxvlm-ref/bin/python gen_fixture.py

Outputs (written next to this script, NOT committed — see .gitignore):
    expected_input_ids.npy          int32  (1, S)
    expected_pixel_values.npy       float32  (N_patches, C*patch_h*patch_w)
    expected_image_grid_thw.npy     int32  (1, 3)  — (T, H, W) grid
    expected_last_logits.npy        float32  (1, vocab)
    expected_first_token.txt        "<token_id>\\n"

The script exits non-zero on any error so CI can detect regressions.
"""

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models import cache
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from PIL import Image

SCRIPT_DIR = Path(__file__).parent.resolve()
IMG_PATH = SCRIPT_DIR / "coco_sample.jpg"
PROMPT_TEXT = "What is in this image?"


def main() -> None:
    model_dir = os.environ.get("QWEN35_MODEL", "").strip()
    if not model_dir:
        print(
            "ERROR: QWEN35_MODEL env var is not set.\n"
            "Example:\n"
            "  QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit"
            "/snapshots/<sha>/ python gen_fixture.py",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = str(Path(model_dir).expanduser().resolve())
    if not Path(model_dir).is_dir():
        print(f"ERROR: QWEN35_MODEL path does not exist: {model_dir}", file=sys.stderr)
        sys.exit(1)

    if not IMG_PATH.exists():
        print(f"ERROR: coco_sample.jpg not found at {IMG_PATH}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load model + processor
    # ------------------------------------------------------------------
    print(f"[gen_fixture] Loading model from {model_dir} …")
    model, processor = load(model_dir)
    print(f"[gen_fixture] Model type : {type(model).__name__}")
    print(f"[gen_fixture] Processor  : {type(processor).__name__}")

    with open(Path(model_dir) / "config.json") as f:
        config = json.load(f)

    # ------------------------------------------------------------------
    # 2. Build multimodal prompt + preprocess
    # ------------------------------------------------------------------
    image = Image.open(IMG_PATH).convert("RGB")
    formatted_prompt = apply_chat_template(
        processor, config, PROMPT_TEXT, num_images=1
    )
    print(f"[gen_fixture] Formatted prompt (repr): {repr(formatted_prompt[:120])} …")

    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=formatted_prompt,
    )

    input_ids = inputs["input_ids"]          # (1, S) int64
    pixel_values = inputs["pixel_values"]    # (N_patches, D) float32/bfloat16
    image_grid_thw = inputs["image_grid_thw"]  # (1, 3) int64
    attention_mask = inputs.get("attention_mask")

    mx.eval(input_ids, pixel_values, image_grid_thw)
    if attention_mask is not None:
        mx.eval(attention_mask)

    print(f"[gen_fixture] input_ids        : shape={input_ids.shape}, dtype={input_ids.dtype}")
    print(f"[gen_fixture] pixel_values     : shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"[gen_fixture] image_grid_thw   : {np.array(image_grid_thw.astype(mx.int32))}")

    # ------------------------------------------------------------------
    # 3. Single-step forward — get last-position logits
    # ------------------------------------------------------------------
    print("[gen_fixture] Computing vision + text embeddings …")
    embedding_output = model.get_input_embeddings(
        input_ids,
        pixel_values,
        image_grid_thw=image_grid_thw,
        mask=attention_mask,
    )
    inputs_embeds = embedding_output.inputs_embeds
    print(f"[gen_fixture] inputs_embeds    : shape={inputs_embeds.shape}")

    prompt_cache = cache.make_prompt_cache(model.language_model)

    print("[gen_fixture] Running single language-model forward pass …")
    outputs = model.language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
    )

    # logits: (1, S, vocab)  — take last position
    logits = outputs.logits          # lazy mx.array
    last_logits = logits[:, -1, :]   # (1, vocab)

    # Materialise before numpy conversion (bfloat16 needs cast to float32 first)
    mx.eval(last_logits)
    last_logits_f32 = np.array(last_logits.astype(mx.float32))  # (1, vocab)

    print(f"[gen_fixture] last_logits      : shape={last_logits_f32.shape}, "
          f"min={last_logits_f32.min():.4f}, max={last_logits_f32.max():.4f}")

    first_token_id = int(last_logits_f32.argmax())
    print(f"[gen_fixture] first_token_id   : {first_token_id}")

    # ------------------------------------------------------------------
    # 4. Sanity check pixel_values range
    # ------------------------------------------------------------------
    pv_np = np.array(pixel_values.astype(mx.float32))
    print(f"[gen_fixture] pixel_values range: min={pv_np.min():.4f}, max={pv_np.max():.4f} "
          f"(expected roughly [-2, 2] after channel-wise normalisation)")

    # ------------------------------------------------------------------
    # 5. Save fixtures
    # ------------------------------------------------------------------
    out_input_ids      = SCRIPT_DIR / "expected_input_ids.npy"
    out_pixel_values   = SCRIPT_DIR / "expected_pixel_values.npy"
    out_grid_thw       = SCRIPT_DIR / "expected_image_grid_thw.npy"
    out_last_logits    = SCRIPT_DIR / "expected_last_logits.npy"
    out_first_token    = SCRIPT_DIR / "expected_first_token.txt"

    np.save(out_input_ids,    np.array(input_ids.astype(mx.int32)))
    np.save(out_pixel_values, pv_np)
    np.save(out_grid_thw,     np.array(image_grid_thw.astype(mx.int32)))
    np.save(out_last_logits,  last_logits_f32)
    out_first_token.write_text(f"{first_token_id}\n")

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print()
    print("[gen_fixture] Saved files:")
    for p in [out_input_ids, out_pixel_values, out_grid_thw, out_last_logits, out_first_token]:
        size = p.stat().st_size
        print(f"  {p.name:<40} {size:>10} bytes")

    print()
    print(f"[gen_fixture] first_token_id = {first_token_id}")
    print("[gen_fixture] Done.")


if __name__ == "__main__":
    main()
