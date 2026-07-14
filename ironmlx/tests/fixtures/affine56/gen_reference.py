"""Generate pinned affine 4/5/6-bit reference logits with mlx-lm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

PROMPT = "What is 2+2?"
GREEDY_STEPS = 4
TOP_K = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--architecture", choices=("gemma4", "qwen3_5"), required=True)
    parser.add_argument("--bits", type=int, choices=(4, 5, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load(str(args.model))
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    generated: list[int] = []
    first_logits = None
    running_ids = list(ids)
    for _ in range(GREEDY_STEPS):
        input_ids = mx.array([running_ids], dtype=mx.int32)
        logits = model(input_ids)[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        if first_logits is None:
            first_logits = logits
        token_id = int(mx.argmax(logits).item())
        generated.append(token_id)
        running_ids.append(token_id)

    assert first_logits is not None
    values = first_logits.tolist()
    ranked = sorted(range(len(values)), key=values.__getitem__, reverse=True)[:TOP_K]
    logits_path = args.output.with_suffix(".npy")
    mx.save(str(logits_path), first_logits)
    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "architecture": args.architecture,
        "quantization": {"mode": "affine", "bits": args.bits, "group_size": 64},
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "input_ids": ids,
        "vocab_size": len(values),
        "next_token_id": ranked[0],
        "greedy_token_ids": generated,
        "logits_file": logits_path.name,
        "top_logits": [
            {"token_id": token_id, "logit": values[token_id]} for token_id in ranked
        ],
        "reference": {
            "mlx": mx.__version__,
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} and {logits_path}: "
        f"next_token_id={ranked[0]}, greedy={generated}, vocab={len(values)}"
    )


if __name__ == "__main__":
    main()
