"""Generate Qwen3.5-4B-MLX-4bit reference logits via mlx-lm."""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

EXPECTED_MLX_VERSION = "0.31.1"
if mx.__version__ != EXPECTED_MLX_VERSION:
    raise SystemExit(
        f"mlx version mismatch: got {mx.__version__}, expected {EXPECTED_MLX_VERSION}"
    )

OUT_DIR = Path(__file__).parent
PROMPT = "What is 2+2?"

model_path = os.environ.get("QWEN35_MODEL")
if not model_path:
    raise SystemExit(
        "QWEN35_MODEL env var must point to the Qwen3.5-4B-MLX-4bit checkpoint dir"
    )

model, tokenizer = load(model_path)

# Tokenize the prompt with no chat template (raw prompt — must match Rust side).
ids = tokenizer.encode(PROMPT, add_special_tokens=False)
print(f"prompt token count: {len(ids)}")
input_ids = mx.array([ids], dtype=mx.int32)
mx.save(str(OUT_DIR / "expected_input_ids.npy"), mx.array(ids, dtype=mx.int32))

# Forward — full prompt, no cache (matches Rust prefill semantics).
logits = model(input_ids)        # [1, S, vocab]
last = logits[0, -1, :]          # [vocab]
last_fp32 = last.astype(mx.float32)
mx.eval(last_fp32)
mx.save(str(OUT_DIR / "expected_last_logits.npy"), last_fp32)

print(f"saved expected_last_logits.npy with shape {last_fp32.shape} dtype {last_fp32.dtype}")
