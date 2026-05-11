#!/usr/bin/env python
"""P6.6 mlx-vlm-side multi-image dump driver.

Loads model + processor, builds a 1-chat-request prompt that includes
TWO image_url parts, runs preprocess to get per-image pixel_values +
image_grid_thw, runs the vision tower once to capture the concatenated
vision_embeds, then runs the full LM forward to capture last-position
logits. All outputs are saved as safetensors / .npy under --out-dir.

Optionally, when MLXVLM_VISION_DUMP_DIR is set externally, the existing
P6.1+P6.3b vision hooks (30 + 96 = 126 intermediate tensors) fire and
write into that dir for op-level Gate 2 diff.

Usage:
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \\
    ~/.venvs/mlxvlm-ref/bin/python run_p6_6_dump.py \\
        --image-0 /path/to/image_0.jpg \\
        --image-1 /path/to/image_1.jpg \\
        --out-dir /tmp/p6_diff_multi/python

API notes (mlx-vlm 0.5.0):
  - prepare_inputs takes flat images=[img0, img1] (PIL objects) and a
    formatted prompt with 2x <|image_pad|> tokens (via apply_chat_template
    num_images=2). The [[path0, path1]] nested-list form is NOT used.
  - model(input_ids, pixel_values=pv, image_grid_thw=grid_thw) — image_grid_thw
    is accepted via **kwargs in model.__call__ -> get_input_embeddings.
  - model.vision_tower(pv_bf16, grid_thw) returns (hidden_states, deepstack).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mlx-vlm on 2 images and dump fixture artifacts."
    )
    parser.add_argument("--image-0", required=True, type=Path, help="Path to first image.")
    parser.add_argument("--image-1", required=True, type=Path, help="Path to second image.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for output artifacts.")
    parser.add_argument(
        "--prompt",
        default="Describe both images in detail. Mention key objects you see in each.",
        help="Text prompt (2 image placeholders will be prepended by apply_chat_template).",
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
    for p in (args.image_0, args.image_1):
        if not p.exists():
            print(f"ERROR: image not found: {p}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # 1. Load model + processor
    # ------------------------------------------------------------------
    print(f"[run_p6_6_dump] Loading model from {model_dir} ...")
    model, processor = load(model_dir)
    print(f"[run_p6_6_dump] model type  : {type(model).__name__}")
    print(f"[run_p6_6_dump] processor   : {type(processor).__name__}")

    with open(Path(model_dir) / "config.json") as f:
        config_json = json.load(f)

    # ------------------------------------------------------------------
    # 2. Build multi-image prompt + preprocess
    #    apply_chat_template num_images=2 inserts 2x <|image_pad|> tokens.
    #    prepare_inputs takes flat [img0, img1] PIL list.
    # ------------------------------------------------------------------
    img0 = PILImage.open(args.image_0).convert("RGB")
    img1 = PILImage.open(args.image_1).convert("RGB")

    formatted_prompt = apply_chat_template(
        processor, config_json, args.prompt, num_images=2
    )
    print(f"[run_p6_6_dump] Formatted prompt (repr): {repr(formatted_prompt[:120])} ...")

    inputs = prepare_inputs(
        processor,
        images=[img0, img1],
        prompts=formatted_prompt,
    )

    input_ids = inputs["input_ids"]          # (1, S) int64
    pixel_values = inputs["pixel_values"]    # (N_total_patches, 1536) bfloat16 or float32
    grid_thw = inputs["image_grid_thw"]      # (2, 3) int64
    attention_mask = inputs.get("attention_mask")

    mx.eval(input_ids, pixel_values, grid_thw)
    if attention_mask is not None:
        mx.eval(attention_mask)

    grid_np = np.array(grid_thw.astype(mx.int32))
    print(f"[run_p6_6_dump] input_ids shape  : {input_ids.shape}")
    print(f"[run_p6_6_dump] pixel_values     : shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"[run_p6_6_dump] image_grid_thw   : {grid_np.tolist()}")

    # Sanity: must have 2 rows in grid_thw
    assert grid_thw.shape[0] == 2, \
        f"expected 2 images, got grid_thw shape={grid_thw.shape}"

    # ------------------------------------------------------------------
    # 3. Save input_ids + image_grid_thw
    # ------------------------------------------------------------------
    np.save(args.out_dir / "expected_input_ids.npy",
            np.array(input_ids.astype(mx.int32)))
    np.save(args.out_dir / "expected_image_grid_thw.npy", grid_np)
    print(f"[run_p6_6_dump] saved expected_input_ids.npy  shape={input_ids.shape}")
    print(f"[run_p6_6_dump] saved expected_image_grid_thw.npy  shape={grid_np.shape}")

    # ------------------------------------------------------------------
    # 4. Save concatenated pixel_values (bfloat16)
    # ------------------------------------------------------------------
    pv_bf16 = pixel_values.astype(mx.bfloat16)
    mx.eval(pv_bf16)
    mx.save_safetensors(
        str(args.out_dir / "expected_pixel_values.safetensors"),
        {"tensor": pv_bf16},
    )
    print(f"[run_p6_6_dump] saved expected_pixel_values.safetensors  shape={pv_bf16.shape}")

    # ------------------------------------------------------------------
    # 5. Save per-image pixel_values slices for Gate 1 (preprocess diff)
    #    Split by N_i = grid_h_i * grid_w_i (pre-merger patch count per image)
    # ------------------------------------------------------------------
    n0 = int(grid_np[0, 1] * grid_np[0, 2])
    n1 = int(grid_np[1, 1] * grid_np[1, 2])
    n_total = pixel_values.shape[0]
    assert n0 + n1 == n_total, \
        f"split mismatch: n0={n0}, n1={n1}, total={n_total}"

    pv_0 = pv_bf16[:n0]
    pv_1 = pv_bf16[n0:]
    mx.save_safetensors(
        str(args.out_dir / "image_0_pv.safetensors"),
        {"tensor": pv_0},
    )
    mx.save_safetensors(
        str(args.out_dir / "image_1_pv.safetensors"),
        {"tensor": pv_1},
    )
    print(f"[run_p6_6_dump] saved image_0_pv.safetensors  shape={pv_0.shape}")
    print(f"[run_p6_6_dump] saved image_1_pv.safetensors  shape={pv_1.shape}")

    # ------------------------------------------------------------------
    # 6. Run vision tower (P6.1+P6.3b dump hooks fire if MLXVLM_VISION_DUMP_DIR set)
    #    vision_tower(hidden_states, grid_thw) -> (embeds, deepstack_features)
    # ------------------------------------------------------------------
    embeds, _deepstack = model.vision_tower(pv_bf16, grid_thw)
    mx.eval(embeds)
    mx.save_safetensors(
        str(args.out_dir / "vision_embeds.safetensors"),
        {"tensor": embeds.astype(mx.bfloat16)},
    )
    print(f"[run_p6_6_dump] vision_embeds    : shape={embeds.shape}")

    # ------------------------------------------------------------------
    # 7. Full LM forward to get last-position logits + first token
    #    model(input_ids, pixel_values=pv, **{image_grid_thw=grid_thw})
    #    image_grid_thw is passed via **kwargs -> get_input_embeddings
    # ------------------------------------------------------------------
    output = model(
        input_ids,
        pixel_values=pv_bf16,
        image_grid_thw=grid_thw,
    )
    # output is LanguageModelOutput; .logits is (1, S, vocab); take last position
    logits = output.logits if hasattr(output, "logits") else output
    last = logits[:, -1, :]
    mx.eval(last)
    last_f32 = np.array(last.astype(mx.float32))
    np.save(args.out_dir / "expected_last_logits.npy", last_f32)
    print(f"[run_p6_6_dump] last_logits      : shape={last_f32.shape}, "
          f"min={last_f32.min():.4f}, max={last_f32.max():.4f}")

    first_token = int(last_f32.argmax())
    (args.out_dir / "expected_first_token.txt").write_text(str(first_token) + "\n")
    print(f"[run_p6_6_dump] first_token      : {first_token}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print()
    print(f"[run_p6_6_dump] Artifacts in {args.out_dir}:")
    artifact_names = [
        "expected_input_ids.npy",
        "expected_image_grid_thw.npy",
        "expected_pixel_values.safetensors",
        "image_0_pv.safetensors",
        "image_1_pv.safetensors",
        "vision_embeds.safetensors",
        "expected_last_logits.npy",
        "expected_first_token.txt",
    ]
    for name in artifact_names:
        p = args.out_dir / name
        size = p.stat().st_size if p.exists() else -1
        status = f"{size:>12} bytes" if size >= 0 else "   MISSING"
        print(f"  {name:<45} {status}")

    missing = [n for n in artifact_names if not (args.out_dir / n).exists()]
    if missing:
        print(f"\nERROR: {len(missing)} missing artifact(s): {missing}", file=sys.stderr)
        return 2

    print(f"\n[run_p6_6_dump] Done. All 8 artifacts written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
