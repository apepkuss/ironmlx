#!/usr/bin/env python
"""P6.6 mlx-vlm-side multi-image dump driver.

Loads model + processor, builds a 1-chat-request prompt that includes
N image_url parts (N >= 1), runs preprocess to get per-image pixel_values +
image_grid_thw, runs the vision tower once to capture the concatenated
vision_embeds, then runs the full LM forward to capture last-position
logits. All outputs are saved as safetensors / .npy under --out-dir.

Optionally, when MLXVLM_VISION_DUMP_DIR is set externally, the existing
P6.1+P6.3b vision hooks (30 + 96 = 126 intermediate tensors) fire and
write into that dir for op-level Gate 2 diff.

Usage:
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \\
    ~/.venvs/mlxvlm-ref/bin/python run_p6_6_dump.py \\
        --images /path/to/image_0.jpg /path/to/image_1.jpg [...] \\
        --out-dir /tmp/p6_diff_multi/python

For N=2 backward compatibility, --image-0 / --image-1 are still accepted.

API notes (mlx-vlm 0.5.0):
  - prepare_inputs takes flat images=[img0, img1, ...] (PIL objects) and a
    formatted prompt with N x <|image_pad|> tokens (via apply_chat_template
    num_images=N).
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
        description="Run mlx-vlm on N images and dump fixture artifacts."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=None,
        help="Paths to N images (N >= 1).",
    )
    parser.add_argument("--image-0", type=Path, default=None, help="Back-compat: first image (N=2).")
    parser.add_argument("--image-1", type=Path, default=None, help="Back-compat: second image (N=2).")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for output artifacts.")
    parser.add_argument(
        "--prompt",
        default="Describe both images in detail. Mention key objects you see in each.",
        help="Text prompt (N image placeholders will be prepended by apply_chat_template).",
    )
    args = parser.parse_args()

    image_paths: list[Path]
    if args.images:
        image_paths = list(args.images)
    elif args.image_0 is not None and args.image_1 is not None:
        image_paths = [args.image_0, args.image_1]
    else:
        print("ERROR: pass --images img0 [img1 ...] (or back-compat --image-0/--image-1).",
              file=sys.stderr)
        return 1

    n_images = len(image_paths)
    if n_images < 1:
        print("ERROR: need at least 1 image.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = os.environ.get("QWEN35_MODEL", "").strip()
    if not model_dir:
        print("ERROR: QWEN35_MODEL env var is required.", file=sys.stderr)
        return 1
    model_dir = str(Path(model_dir).expanduser().resolve())
    if not Path(model_dir).is_dir():
        print(f"ERROR: QWEN35_MODEL path does not exist: {model_dir}", file=sys.stderr)
        return 1
    for p in image_paths:
        if not p.exists():
            print(f"ERROR: image not found: {p}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # 1. Load model + processor
    # ------------------------------------------------------------------
    print(f"[run_p6_6_dump] N images       : {n_images}")
    print(f"[run_p6_6_dump] Loading model from {model_dir} ...")
    model, processor = load(model_dir)
    print(f"[run_p6_6_dump] model type  : {type(model).__name__}")
    print(f"[run_p6_6_dump] processor   : {type(processor).__name__}")

    with open(Path(model_dir) / "config.json") as f:
        config_json = json.load(f)

    # ------------------------------------------------------------------
    # 2. Build multi-image prompt + preprocess
    #    apply_chat_template num_images=N inserts N x <|image_pad|> tokens.
    # ------------------------------------------------------------------
    pil_images = [PILImage.open(p).convert("RGB") for p in image_paths]

    formatted_prompt = apply_chat_template(
        processor, config_json, args.prompt, num_images=n_images
    )
    print(f"[run_p6_6_dump] Formatted prompt (repr): {repr(formatted_prompt[:120])} ...")

    inputs = prepare_inputs(
        processor,
        images=pil_images,
        prompts=formatted_prompt,
    )

    input_ids = inputs["input_ids"]          # (1, S) int64
    pixel_values = inputs["pixel_values"]    # (N_total_patches, 1536) bfloat16 or float32
    grid_thw = inputs["image_grid_thw"]      # (N, 3) int64
    attention_mask = inputs.get("attention_mask")

    mx.eval(input_ids, pixel_values, grid_thw)
    if attention_mask is not None:
        mx.eval(attention_mask)

    grid_np = np.array(grid_thw.astype(mx.int32))
    print(f"[run_p6_6_dump] input_ids shape  : {input_ids.shape}")
    print(f"[run_p6_6_dump] pixel_values     : shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"[run_p6_6_dump] image_grid_thw   : {grid_np.tolist()}")

    assert grid_thw.shape[0] == n_images, \
        f"expected {n_images} images, got grid_thw shape={grid_thw.shape}"

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
    per_image_n = [int(grid_np[i, 1] * grid_np[i, 2]) for i in range(n_images)]
    n_total = pixel_values.shape[0]
    assert sum(per_image_n) == n_total, \
        f"split mismatch: sum(per_image_n)={sum(per_image_n)}, total={n_total}"

    offset = 0
    for i, n_i in enumerate(per_image_n):
        pv_i = pv_bf16[offset:offset + n_i]
        out_path = args.out_dir / f"image_{i}_pv.safetensors"
        mx.save_safetensors(str(out_path), {"tensor": pv_i})
        print(f"[run_p6_6_dump] saved image_{i}_pv.safetensors  shape={pv_i.shape}")
        offset += n_i

    # ------------------------------------------------------------------
    # 6. Run vision tower (P6.1+P6.3b dump hooks fire if MLXVLM_VISION_DUMP_DIR set)
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
    # ------------------------------------------------------------------
    output = model(
        input_ids,
        pixel_values=pv_bf16,
        image_grid_thw=grid_thw,
    )
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
        *(f"image_{i}_pv.safetensors" for i in range(n_images)),
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

    print(f"\n[run_p6_6_dump] Done. All {len(artifact_names)} artifacts written (N={n_images}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
