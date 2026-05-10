# P6 Qwen3.5-VL Fixtures

This directory contains reference fixtures for the P6 VL integration test suite.

## File Status

| File | Committed | Description |
|---|---|---|
| `coco_sample.jpg` | YES | COCO val2017 sample image used as the multimodal input |
| `coco_sample_normalized.bin` | YES | Raw normalized pixel tensor (Task 5 — smart-resize path) |
| `smart_resize_golden.txt` | YES | Expected smart-resize output dimensions (Task 3) |
| `p6_vit_attn_ref.safetensors` | YES | Vision-transformer attention reference tensors (Task 9) |
| `p6_pos_ids_ref.safetensors` | YES | Position-IDs reference tensors (Task 14) |
| `gen_fixture.py` | YES | Script that generates the `expected_*` files below |
| `expected_input_ids.npy` | **NO** | Tokenized prompt — shape (1, S) int32 |
| `expected_pixel_values.npy` | **NO** | Preprocessed image patches — float32 |
| `expected_image_grid_thw.npy` | **NO** | Per-image (T, H, W) grid — shape (1, 3) int32 |
| `expected_last_logits.npy` | **NO** | Last-position logits from full forward pass — (1, vocab) float32 |
| `expected_first_token.txt` | **NO** | Argmax token ID of `expected_last_logits` |

The `expected_*` files are **not committed** (listed in `.gitignore`) because they are
large binary artifacts that must be regenerated against the exact mlx-vlm 0.5.0 release
being tested. They are consumed by the Task 21 logits-match e2e test.

## How to Regenerate

Prerequisites:
- mlx-vlm 0.5.0 installed in `~/.venvs/mlxvlm-ref/`
  (editable install from `/Volumes/Dev/mlx-vlm/`)
- Model snapshot available locally — set `QWEN35_MODEL` to its path

```bash
QWEN35_MODEL=~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
~/.venvs/mlxvlm-ref/bin/python ironmlx/tests/fixtures/p6_qwen35_vl/gen_fixture.py
```

Run from the repo root **or** from this directory (the script uses `__file__`-relative
paths internally).

Expected output (approximate):

```
[gen_fixture] Loading model from …
[gen_fixture] Model type : Model
[gen_fixture] Processor  : Qwen3VLProcessor
[gen_fixture] Formatted prompt (repr): '<|im_start|>user\n<|vision_start|>…
[gen_fixture] input_ids        : shape=(1, 318), dtype=mlx.core.int64
[gen_fixture] pixel_values     : shape=(1200, 1536), dtype=mlx.core.float32
[gen_fixture] image_grid_thw   : [[1 30 40]]
[gen_fixture] Computing vision + text embeddings …
[gen_fixture] inputs_embeds    : shape=(1, 318, 2560)
[gen_fixture] Running single language-model forward pass …
[gen_fixture] last_logits      : shape=(1, 248320), min=…, max=…
[gen_fixture] first_token_id   : <int>
[gen_fixture] pixel_values range: min=… max=… (expected roughly [-2, 2] …)

[gen_fixture] Saved files:
  expected_input_ids.npy                     <bytes>
  expected_pixel_values.npy                  <bytes>
  expected_image_grid_thw.npy                <bytes>
  expected_last_logits.npy                   <bytes>
  expected_first_token.txt                   <bytes>

[gen_fixture] first_token_id = <int>
[gen_fixture] Done.
```

## mlx-vlm API Notes

- `mlx_vlm.load(model_dir)` returns `(model, processor)`.
  For Qwen3.5-VL, model type is `qwen3_5` and `model_type` in `config.json` is `"qwen3_5"`.
- `apply_chat_template(processor, config, prompt, num_images=1)` returns a formatted
  string with `<|vision_start|><|image_pad|><|vision_end|>` tokens embedded.
- `prepare_inputs(processor, images=[img], prompts=formatted)` returns a dict with keys:
  `input_ids`, `pixel_values`, `image_grid_thw`, `attention_mask`.
- The single-step forward uses `model.get_input_embeddings()` followed by
  `model.language_model(input_ids, inputs_embeds=..., cache=...)`.
  This avoids the full generation loop and yields deterministic logits.
- `pixel_values` is returned as `bfloat16` by the model's vision encoder; the fixture
  saves it as `float32` for portability. Values are roughly in `[-2, 2]` (normalised).
- `logits` are also `bfloat16` internally; saved as `float32`.
