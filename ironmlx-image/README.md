# ironmlx-image

`ironmlx-image` owns native MLX image-generation model implementations. It
contains checkpoint preflight, Diffusers-style component loading, denoising
schedulers, image transformers, VAEs and model-side generation pipelines.

The crate does not own HTTP protocol DTOs, Base64 response envelopes, global
request scheduling, model lifecycle, downloads or UI. Those remain in
`ironmlx`, `ironmlx-runtime` and `ironmlx-app` respectively.

Qwen Image 2.1 uses Qwen3-VL text conditioning, so this crate currently depends
on `ironmlx-lm` for tokenizer and text-model neural-network components. The
dependency is one-way: `ironmlx-lm` does not depend on `ironmlx-image`.

Run model-side tests with:

```sh
cargo test -p ironmlx-image
```

The ignored real-model acceptance test requires the complete checkpoint:

```sh
IRONMLX_QWEN_IMAGE_MODEL_DIR=/path/to/Qwen-Image-2.1-MLX-4bit \
  cargo test -p ironmlx-image --test qwen_image_21_real_model -- \
  --ignored --nocapture --test-threads=1
```
