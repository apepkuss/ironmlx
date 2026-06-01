"""Generate MiniCPM-V-4.6-4bit text-only reference logits via mlx-vlm.

MiniCPM-V-4.6's language backbone is Qwen3.5-text verbatim (mlx-vlm:
`class LanguageModel(Qwen35LanguageModel)`). This dumps the last-position
logits of a pure-text forward through `model.language_model`, plus the exact
input token ids, so the Rust integration test
(`tests/minicpmv46_text_logits_match.rs`) can feed identical ids and isolate
LM-forward correctness from tokenizer parity.

Run from the editable mlx-vlm checkout (see memory reference_iron_rivals_baselines):

    cd /Users/xin/workspace/iron-rivals/mlx-vlm
    MINICPMV46_MODEL=<snapshot-dir> \
      uv run --with-editable . python \
      /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46/gen_logits.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

OUT_DIR = Path(__file__).parent

# Diverse text-only prompts (math / code / English prose / Chinese) to stress
# the hybrid gated-delta + gated-full attention backbone across token regimes.
PROMPTS = [
    "What is 2+2?",
    "def fibonacci(n):\n    if n < 2:",
    "The capital of France is",
    "请用一句话解释什么是机器学习。",
]

model_path = os.environ.get("MINICPMV46_MODEL")
if not model_path:
    raise SystemExit(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit checkpoint dir"
    )

print(f"mlx version: {mx.__version__}")
model, processor = load(model_path)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

for idx, prompt in enumerate(PROMPTS):
    # Raw prompt, no chat template, no special tokens — must match the Rust side.
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    mx.save(str(OUT_DIR / f"expected_input_ids_p{idx}.npy"), mx.array(ids, dtype=mx.int32))

    # Text-only forward through the Qwen3.5-text language backbone. Position ids
    # are computed internally (sequential for text-only). Returns
    # LanguageModelOutput with `.logits` of shape [1, S, vocab].
    out = model.language_model(mx.array([ids]))
    logits_all = out.logits if hasattr(out, "logits") else out
    last = logits_all[0, -1, :].astype(mx.float32)
    mx.eval(last)
    mx.save(str(OUT_DIR / f"expected_last_logits_p{idx}.npy"), last)

    argmax = int(mx.argmax(last).item())
    print(
        f"p{idx}: tokens={len(ids)} argmax={argmax} prompt={prompt[:40]!r}"
    )

print(f"saved {len(PROMPTS)} prompt fixtures to {OUT_DIR}")
