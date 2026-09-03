"""Generate Gemma4 unified processor/logits fixtures via mlx-vlm.

The fixture targets mlx-community/gemma-4-12B-it-4bit, whose config uses
model_type=gemma4_unified. It captures:

  text_input_ids.npy
  text_expected_layer0_stage_last_hiddens.npy
  text_expected_layer_last_hiddens.npy
  text_expected_last_hidden.npy
  text_expected_last_raw_logits.npy
  text_expected_last_logits.npy
  text_expected_first_token.txt
  vision_input_ids.npy
  vision_pixel_values.npy
  vision_image_position_ids.npy
  vision_grid_thw.npy
  vision_expected_layer_last_hiddens.npy
  vision_expected_last_hidden.npy
  vision_expected_last_raw_logits.npy
  vision_expected_last_logits.npy
  vision_expected_first_token.txt

Run from an editable mlx-vlm checkout:

  GEMMA4_UNIFIED_MODEL=/path/to/snapshot \
    uv run --with-editable . python gen_fixture.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm import load

OUT_DIR = Path(__file__).parent
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "qwen35_vl" / "coco_sample.jpg"
IMAGE_TOKEN_ID = 258880
POOLING_KERNEL_SIZE = 3
TEXT_PROMPT = "What is 2+2?"
VISION_SUFFIX = "Describe this image briefly."


def gemma4_placeholder(token_count: int) -> str:
    return "<|image>" + ("<|image|>" * token_count) + "<image|>"


def last_logits(output):
    logits = output.logits if hasattr(output, "logits") else output
    if len(logits.shape) == 3:
        last = logits[:, -1, :]
    else:
        last = logits
    last = last.astype(mx.float32)
    mx.eval(last)
    return last


def save_last_hidden_and_raw_logits(prefix: str, model, hidden):
    last_hidden = hidden[:, -1, :].astype(mx.float32)
    raw_logits = model.language_model.model.embed_tokens.as_linear(last_hidden).astype(mx.float32)
    mx.eval(last_hidden, raw_logits)
    mx.save(str(OUT_DIR / f"{prefix}_expected_last_hidden.npy"), last_hidden)
    mx.save(str(OUT_DIR / f"{prefix}_expected_last_raw_logits.npy"), raw_logits)
    return last_hidden, raw_logits


def save_layer_last_hiddens(prefix: str, model, input_ids=None, **kwargs):
    capture = []
    layer_count = len(model.language_model.model.layers)
    output = model.language_model.model(
        input_ids,
        capture_layer_ids=list(range(layer_count)),
        hidden_sink=capture,
        **kwargs,
    )
    del output
    stacked = mx.stack([h[:, -1, :].astype(mx.float32).reshape(-1) for h in capture], axis=0)
    mx.eval(stacked)
    mx.save(str(OUT_DIR / f"{prefix}_expected_layer_last_hiddens.npy"), stacked)
    return stacked


def save_text_layer0_stage_last_hiddens(model, input_ids):
    text_model = model.language_model.model
    layer = text_model.layers[0]
    h = text_model.embed_tokens(input_ids) * text_model.embed_scale
    masks = text_model._make_masks(h, [None] * len(text_model.layers))
    stages = [h[:, -1, :].astype(mx.float32).reshape(-1)]

    residual = h
    h_norm = layer.input_layernorm(h)
    stages.append(h_norm[:, -1, :].astype(mx.float32).reshape(-1))
    attn, shared_kv, offset = layer.self_attn(h_norm, masks[0], None)
    del shared_kv, offset
    stages.append(attn[:, -1, :].astype(mx.float32).reshape(-1))
    attn_norm = layer.post_attention_layernorm(attn)
    stages.append(attn_norm[:, -1, :].astype(mx.float32).reshape(-1))
    h = residual + attn_norm
    stages.append(h[:, -1, :].astype(mx.float32).reshape(-1))

    residual = h
    ffn_norm = layer.pre_feedforward_layernorm(h)
    stages.append(ffn_norm[:, -1, :].astype(mx.float32).reshape(-1))
    ffn = layer.mlp(ffn_norm)
    stages.append(ffn[:, -1, :].astype(mx.float32).reshape(-1))
    ffn_normed = layer.post_feedforward_layernorm(ffn)
    stages.append(ffn_normed[:, -1, :].astype(mx.float32).reshape(-1))
    h = residual + ffn_normed
    stages.append(h[:, -1, :].astype(mx.float32).reshape(-1))
    h = h * layer.layer_scalar
    stages.append(h[:, -1, :].astype(mx.float32).reshape(-1))

    stacked = mx.stack(stages, axis=0)
    mx.eval(stacked)
    mx.save(str(OUT_DIR / "text_expected_layer0_stage_last_hiddens.npy"), stacked)
    return stacked


def save_logits(prefix: str, logits):
    mx.save(str(OUT_DIR / f"{prefix}_expected_last_logits.npy"), logits)
    flat = logits.reshape(-1)
    token = int(mx.argmax(flat).item())
    (OUT_DIR / f"{prefix}_expected_first_token.txt").write_text(f"{token}\n")
    return token


def grid_from_positions(image_position_ids: np.ndarray) -> np.ndarray:
    grids = []
    for pos in image_position_ids:
        valid = ~np.all(pos == -1, axis=-1)
        if not np.any(valid):
            raise ValueError("image_position_ids contains no valid image positions")
        valid_pos = pos[valid]
        max_x = int(valid_pos[:, 0].max()) + 1
        max_y = int(valid_pos[:, 1].max()) + 1
        grids.append([1, max_y * POOLING_KERNEL_SIZE, max_x * POOLING_KERNEL_SIZE])
    return np.asarray(grids, dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMMA4_UNIFIED_MODEL"),
        help="Path to mlx-community/gemma-4-12B-it-4bit snapshot",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Image fixture path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("set GEMMA4_UNIFIED_MODEL or pass --model")

    print(f"loading model: {args.model}")
    model, processor = load(args.model)
    tokenizer = processor.tokenizer

    text_ids = tokenizer.encode(TEXT_PROMPT, add_special_tokens=False)
    text_input_ids = mx.array([text_ids], dtype=mx.int32)
    mx.save(str(OUT_DIR / "text_input_ids.npy"), mx.array(text_ids, dtype=mx.int32))
    save_text_layer0_stage_last_hiddens(model, text_input_ids)
    save_layer_last_hiddens("text", model, text_input_ids)
    text_hidden = model.language_model.model(text_input_ids)
    save_last_hidden_and_raw_logits("text", model, text_hidden)
    text_last = last_logits(model.language_model.logits_from_hidden(text_hidden))
    text_first = save_logits("text", text_last)
    print(
        f"text: seq={len(text_ids)} vocab={int(text_last.shape[-1])} "
        f"first_token={text_first}"
    )

    image = Image.open(args.image).convert("RGB")
    image_out, num_soft_tokens = processor.image_processor.preprocess([image])
    pixel_values = np.asarray(image_out["pixel_values"], dtype=np.float32)
    image_position_ids = np.asarray(image_out["image_position_ids"], dtype=np.int32)
    num_soft_tokens = [int(x) for x in num_soft_tokens]
    grid_thw = grid_from_positions(image_position_ids)

    prompt = gemma4_placeholder(num_soft_tokens[0]) + VISION_SUFFIX
    vision_ids = tokenizer.encode(prompt, add_special_tokens=False)
    image_token_count = sum(1 for token_id in vision_ids if token_id == IMAGE_TOKEN_ID)
    if image_token_count != sum(num_soft_tokens):
        raise ValueError(
            f"image token count {image_token_count} != soft tokens {sum(num_soft_tokens)}"
        )

    vision_input_ids = mx.array([vision_ids], dtype=mx.int32)
    pixel_values_mx = mx.array(pixel_values, dtype=mx.float32)
    image_position_ids_mx = mx.array(image_position_ids, dtype=mx.int32)
    input_embeddings_features = model.get_input_embeddings(
        input_ids=vision_input_ids,
        pixel_values=pixel_values_mx,
        image_position_ids=image_position_ids_mx,
    )
    vision_hidden = model.language_model.model(
        None,
        inputs_embeds=input_embeddings_features.inputs_embeds,
        per_layer_inputs=input_embeddings_features.per_layer_inputs,
    )
    save_layer_last_hiddens(
        "vision",
        model,
        None,
        inputs_embeds=input_embeddings_features.inputs_embeds,
        per_layer_inputs=input_embeddings_features.per_layer_inputs,
    )
    save_last_hidden_and_raw_logits("vision", model, vision_hidden)
    vision_last = last_logits(model.language_model.logits_from_hidden(vision_hidden))
    vision_first = save_logits("vision", vision_last)

    mx.save(str(OUT_DIR / "vision_input_ids.npy"), mx.array(vision_ids, dtype=mx.int32))
    mx.save(str(OUT_DIR / "vision_pixel_values.npy"), pixel_values_mx)
    mx.save(str(OUT_DIR / "vision_image_position_ids.npy"), image_position_ids_mx)
    mx.save(str(OUT_DIR / "vision_grid_thw.npy"), mx.array(grid_thw, dtype=mx.int32))

    print(
        "vision: "
        f"image={args.image.name} pixels={pixel_values.shape} "
        f"positions={image_position_ids.shape} grids={grid_thw.tolist()} "
        f"seq={len(vision_ids)} image_tokens={image_token_count} "
        f"vocab={int(vision_last.shape[-1])} first_token={vision_first}"
    )
    print(f"saved fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
