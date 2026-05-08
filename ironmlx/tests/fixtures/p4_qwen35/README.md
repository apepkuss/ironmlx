# P4 Qwen3.5 logits-alignment fixture

Verifies `Qwen35Model::forward_on` matches mlx-lm's `model(input_ids)` last-position
logits on a real 4-bit checkpoint with two correctness gates:

- **top-1 greedy argmax MUST match exactly** (the meaningful inference correctness check)
- **max_abs_diff < 0.5** (loose structural sanity for 4-bit BF16 noise)

## Prerequisites

- `mlx-community/Qwen3.5-4B-MLX-4bit` downloaded locally (HF cache or anywhere
  with the standard `config.json` + `model.safetensors` + `tokenizer.json`
  layout).
- mlx-lm (Python) available in a Python env with MLX 0.31.1.

## Generate fixture

```text
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
  python ironmlx/tests/fixtures/p4_qwen35/gen_logits.py
```

Outputs in this directory (NOT committed — large bf16 logits ~ 500KB):
- `expected_input_ids.npy` — `[S]` int32, the tokenized prompt
- `expected_last_logits.npy` — `[vocab_size]` fp32, logits at the last prompt position

## Run the Rust test

```text
MLX_DIR=$HOME/.local/mlx \
  QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
  cargo test --release --ignored -p ironmlx -- p4_qwen35_logits_match -- --test-threads=1
```

If the test fails, investigate in this order:

1. **argmax mismatch** — per-layer hidden-state divergence (binary search by layer_idx).
2. **max_abs_diff > 0.5** — structural bug suspected: wrong layer count, missing residual,
   wrong norm position, or Loader sanitize not stripping `language_model.` prefix.
3. mlx-lm version: `mx.__version__` must be `0.31.1`. Different versions can change
   internal numerics in `mx.fast.scaled_dot_product_attention` and `mx.fast.rms_norm`.
4. Loader sanitize: confirm conv1d.weight shape became `[out, k, in]` after sanitize
   on this checkpoint. If `mlx-community/Qwen3.5-4B-MLX-4bit` ships pre-sanitized
   conv1d (last-dim==1), sanitize is a no-op; the test is unaffected.

The fixture pin: prompt is `"What is 2+2?"`; greedy sample; `max_tokens=1` (only
the first sample); no chat-template applied (raw prompt only).

## Why max_abs_diff threshold = 0.5

The 4-bit Qwen3.5 checkpoint uses BF16 compute over 32 decoder layers.
Per-element quantization + BF16 rounding accumulate ~17 ULPs of noise
across the layer chain (BF16 ULP at logit magnitude ~2 is 0.015625;
17 × 0.015625 ≈ 0.27). 0.5 is twice the observed noise floor — it
catches structural bugs (wrong layer count, missing residual, wrong
norm position) without false-positiving on legitimate quant noise.

The PRIMARY correctness check is the top-1 argmax equality. That's
what governs actual inference output. The max_abs_diff is a structural
sanity check, not a precision metric.
