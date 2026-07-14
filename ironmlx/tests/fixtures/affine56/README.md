# Affine 4-bit, 5-bit, and 6-bit real-model reference fixtures

These fixtures pin full first-step logits and four greedy decode tokens for
Gemma4 E2B-it and Qwen3.5-2B affine 4-bit/5-bit/6-bit checkpoints. The 4-bit
fixtures provide the existing implementation baseline. Generate them with
the same independent Python stack used by the MXFP release gate:

```text
uv run --python 3.13 --isolated \
  --with /Users/xin/workspace/iron-rivals/mlx \
  --with mlx-lm==0.31.3 --with transformers==5.7.0 \
  python gen_reference.py ...
```

The local MLX checkout must be at
`938006e4aee7d9e6c3ac9af3b6f343835a5438e2`, matching the production ironmlx
runtime. The recorded package version is
`0.32.1.dev20260710+938006e4a`.

Every JSON file records the exact Hub revision, quantization contract, prompt
hash, package versions, tokenized prompt, and greedy sequence. The adjacent
NPY file contains the complete float32 first-step logits vector.
