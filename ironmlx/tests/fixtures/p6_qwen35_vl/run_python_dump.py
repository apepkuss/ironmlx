#!/usr/bin/env python
"""P6.1 mlx-vlm-side vision dump driver.

Reads an image, runs mlx-vlm's preprocess to produce pixel_values, saves
pixel_values as ``00_pixel_values.safetensors``, then runs the vision tower
once with MLXVLM_VISION_DUMP_DIR set so hooks in vision.py write tensors
``01..29`` into the same dir.

Requires the mlx-vlm fork at /Volumes/Dev/mlx-vlm with the dump hooks
(see Task 5 of the P6.1 plan).

Usage:
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \\
    ~/.venvs/mlxvlm-ref/bin/python run_python_dump.py \\
        --image coco_sample.jpg \\
        --out-dir /tmp/p6_diff/python
"""
import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

# Image to use when apply_chat_template is needed
_PROMPT_TEXT = "Describe this image."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mlx-vlm vision tower and dump intermediate tensors."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to input image.")
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Directory to write .safetensors dumps."
    )
    parser.add_argument(
        "--prompt",
        default=_PROMPT_TEXT,
        help="Text prompt used to drive prepare_inputs (not relevant to dump content).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = os.environ.get("QWEN35_MODEL", "").strip()
    if not model_dir:
        print("ERROR: QWEN35_MODEL env var is required.", file=sys.stderr)
        return 1
    model_dir = str(Path(model_dir).expanduser().resolve())
    if not Path(model_dir).is_dir():
        print(f"ERROR: QWEN35_MODEL path does not exist: {model_dir}", file=sys.stderr)
        return 1
    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 1. Load model + processor
    # ------------------------------------------------------------------
    print(f"[run_python_dump] Loading model from {model_dir} ...")
    model, processor = load(model_dir)
    print(f"[run_python_dump] model type  : {type(model).__name__}")
    print(f"[run_python_dump] processor   : {type(processor).__name__}")

    # Load config dict for apply_chat_template (same pattern as gen_fixture.py)
    import json
    with open(Path(model_dir) / "config.json") as f:
        config_json = json.load(f)

    # ------------------------------------------------------------------
    # 2. Preprocess image -> pixel_values + grid_thw
    #    (mirrors gen_fixture.py exactly)
    # ------------------------------------------------------------------
    from PIL import Image as PILImage
    image = PILImage.open(args.image).convert("RGB")

    formatted_prompt = apply_chat_template(
        processor, config_json, args.prompt, num_images=1
    )

    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=formatted_prompt,
    )

    pixel_values = inputs["pixel_values"]       # (N_patches, D) bfloat16 or float32
    grid_thw = inputs["image_grid_thw"]         # (1, 3)  int64

    mx.eval(pixel_values, grid_thw)
    print(f"[run_python_dump] pixel_values : shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"[run_python_dump] grid_thw     : {grid_thw.tolist()}")

    # ------------------------------------------------------------------
    # 3. Save pixel_values as 00 dump file (ironmlx consumes this as input)
    #    Cast to bfloat16 to match what the vision tower receives after the
    #    dtype cast in Model.get_input_embeddings.
    # ------------------------------------------------------------------
    pv_bf16 = pixel_values.astype(mx.bfloat16)
    mx.eval(pv_bf16)
    mx.save_safetensors(
        str(args.out_dir / "00_pixel_values.safetensors"),
        {"tensor": pv_bf16},
    )
    print(f"[run_python_dump] saved 00_pixel_values.safetensors  shape={pv_bf16.shape}")

    # ------------------------------------------------------------------
    # 4. Enable dump hooks, then run the vision tower
    #    (vision.py _maybe_dump checks MLXVLM_VISION_DUMP_DIR at call time)
    # ------------------------------------------------------------------
    os.environ["MLXVLM_VISION_DUMP_DIR"] = str(args.out_dir)

    # vision_tower is VisionModel; __call__(hidden_states, grid_thw) returns
    # (hidden_states, deepstack_feature_lists) per qwen3_vl/vision.py line 447
    embeds, _deepstack = model.vision_tower(pv_bf16, grid_thw)
    mx.eval(embeds)
    print(f"[run_python_dump] forward complete; embeds shape={embeds.shape}")

    # ------------------------------------------------------------------
    # 5. Verify all 30 dump files are present
    # ------------------------------------------------------------------
    expected = (
        ["00_pixel_values", "01_patch_embed_out", "02_pos_embed_contrib",
         "03_after_pos_embed", "04_rotary_freqs"]
        + [f"{5 + i:02d}_block_{i:02d}_out" for i in range(24)]
        + ["29_merger_out"]
    )
    missing = [
        n for n in expected
        if not (args.out_dir / f"{n}.safetensors").exists()
    ]
    if missing:
        print(f"ERROR: missing dump files ({len(missing)}): {missing}", file=sys.stderr)
        return 2

    print(f"[run_python_dump] all 30 dump files present in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
