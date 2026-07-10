# Qwen3.5 MXFP reference fixtures

The JSON fixtures contain the top 64 final-position logits for the raw prompt
`What is 2+2?`. They are generated independently with pinned Python packages:

```text
uv run --python 3.13 --isolated \
  --with mlx-lm==0.31.3 --with mlx==0.32.0 --with transformers==5.7.0 \
  python gen_reference.py ...
```

Checkpoint identity is recorded in every fixture. The production test also
checks the local snapshot directory name and checkpoint quantization metadata
before comparing logits.
