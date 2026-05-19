#!/usr/bin/env python3
"""Dump first-step logits from mlx-vlm Qwen3.5-MoE forward pass for 5 prompts.

Output: reports/p5d-argmax/mlxvlm_logits_p<N>.npy (N=0..4)
Run via:
  cd /Users/xin/workspace/iron-rivals/mlx-vlm
  uv run --with-editable . python /Users/xin/workspace/ironmlx-backend/scripts/p5d_mlxvlm_dump_logits.py
"""
import os
import sys
import numpy as np

# Ensure we use the editable mlx-vlm install from iron-rivals repo
MLXVLM_PATH = '/Users/xin/workspace/iron-rivals/mlx-vlm'
if MLXVLM_PATH not in sys.path:
    sys.path.insert(0, MLXVLM_PATH)

from mlx_vlm import load
import mlx.core as mx

PROMPTS = [
    "Once upon a time, in a small village,",
    "The quick brown fox jumps over",
    "def fibonacci(n):\n    if n < 2:",
    "List three reasons why exercise is important:",
    "Translate to French: Good morning.",
]

MODEL_DIR = os.environ.get(
    'IRONMLX_MOE_MODEL_DIR',
    os.path.expanduser(
        '~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit'
        '/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec'
    )
)
OUT_DIR = '/Users/xin/workspace/ironmlx-backend/ironmlx/reports/p5d-argmax'
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[T4-mlxvlm] loading {MODEL_DIR}")
model, processor = load(MODEL_DIR)
print(f"[T4-mlxvlm] loaded")

# Get tokenizer from processor
tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

# Access the language model (Qwen3_5MoE VL wrapper exposes .language_model)
lm = model.language_model if hasattr(model, 'language_model') else model

for idx, prompt in enumerate(PROMPTS):
    print(f"[T4-mlxvlm] prompt {idx}: {prompt[:60]!r}")
    # Tokenize matching ironmlx: no special tokens, raw prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])  # shape [1, S]

    # Forward pass — position_ids computed internally by mlx-vlm
    # Returns LanguageModelOutput with .logits of shape [B, S, vocab]
    out = lm(input_ids)
    logits_all = out.logits  # [1, S, vocab]

    # Take last-position logits → [vocab]
    last_logits = logits_all[0, -1, :]

    # Convert to fp32 + numpy
    fp32_logits = last_logits.astype(mx.float32)
    mx.eval(fp32_logits)
    np_logits = np.array(fp32_logits)

    path = os.path.join(OUT_DIR, f'mlxvlm_logits_p{idx}.npy')
    np.save(path, np_logits)
    argmax = int(np.argmax(np_logits))
    print(f"  saved {path} shape={np_logits.shape} dtype={np_logits.dtype} argmax={argmax}")

print(f"[T4-mlxvlm] done — 5 .npy files in {OUT_DIR}")
