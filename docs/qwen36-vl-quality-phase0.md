# Qwen3.6 MoE/VL Quality Phase 0

## Scope

Model: `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46`

Reference runtime: OMLX with `mlx==0.31.2`, `mlx-lm==0.31.3`, `mlx-vlm==0.5.0`.

ironmlx branch: `ironmlx-qwen36-vl-quality`.

## Findings

- The checkpoint is a Qwen3.6 MoE/VL model, not text-only. It exposes Qwen3.5-compatible hybrid MoE language layers plus a vision tower.
- Current product entry points dispatch this checkpoint by execution architecture: `model_type = "qwen3_5_moe"` maps to the shared `Qwen35MoeModel` path. The `Qwen36MoeModel` facade remains for checkpoint-identity validation and Qwen3.6-specific regression hooks, not for a distinct numeric graph.
- Text-only Qwen3.6 MoE paths pass the first-token regression across no-cache, full-cache, split-cache, and batched-prefill paths when linked against the verified static MLX install.
- Single-image VL parity requires image placeholders before the user text. The CLI already follows this layout; the OpenAI-compatible server now normalizes image content parts to the same Qwen-native layout.
- The CLI generation entry now passes `enable_thinking=false` by default when applying chat templates. This matches the server quality harness and prevents Qwen3.6 from returning thinking-process text for ordinary answer-style CLI prompts. Callers can opt in with `--enable-thinking`.
- Vision embedding parity was isolated before the language-core check. Injected Python reference vision embeddings matched the merged `inputs_embeds` path, so the remaining single-image first-token failure was not caused by image preprocessing, vision tower output, or image-token replacement.
- Layer 0 divergence first appeared inside GatedDeltaNet input projections. Loading GatedDeltaNet projections independently (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) mirrors the mlx-vlm/mlx-lm implementation and avoids shape-dependent fused projection behavior.
- The current local MLX static install at `/Users/xin/.local/mlx` is built from `/Users/xin/workspace/iron-rivals/mlx` commit `2165dc08d7b33258260aa849d39f087d50e62962`.
- The standalone P3b3 GatedDeltaNet fixture also isolates the newer local MLX issue to BF16 projection math: against `/Users/xin/.local/mlx`, the first projection already diverges from the independent Python reference; against the verified static install below, the same Rust GatedDeltaNet fixture passes.
- The same runtime sensitivity is visible in the Qwen3.6 text exact-token regression: the verified MLX install returns the documented direct-answer token, while `/Users/xin/.local/mlx` returns a different first token for the no-cache/full-cache prefill paths.
- Re-linking the same ironmlx code against `/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344` makes the Qwen3.6 single-image first-token regression pass. This confirms the remaining discrepancy is an MLX quantized-matmul runtime regression or behavior change in the newer local MLX build, not an ironmlx architecture error.

## Verification Commands

Text regression:

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
QWEN36_MOE_MODEL=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
cargo test -p ironmlx --features vision-dump --release --lib \
  qwen3_6_moe::model::tests::qwen36_moe_text_forward_paths_first_token_real_checkpoint \
  -- --ignored --nocapture
```

Observed result for the verified MLX install: `1 passed`.

Single-image VL regression with the verified MLX static install:

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
QWEN36_MOE_MODEL=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
cargo test -p ironmlx --features vision-dump --release --lib \
  qwen3_6_moe::model::tests::qwen36_moe_single_image_forward_first_token_real_checkpoint \
  -- --ignored --nocapture
```

Observed result for the verified MLX install: `1 passed`.

GatedDeltaNet standalone fixture with the verified MLX static install:

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
cargo test -p ironmlx --test p3b3_gated_delta_net -- --nocapture
```

Observed result for the verified MLX install: `1 passed`.

Multi-image model API smoke with the verified MLX static install:

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
QWEN36_MOE_MODEL=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
cargo test -p ironmlx --features vision-dump --release --lib \
  qwen3_6_moe::model::tests::qwen36_moe_multi_image_generation_smoke_real_checkpoint \
  -- --ignored --nocapture
```

Observed result for the verified MLX install: `1 passed`.

Full lib regression on the current local MLX install:

```bash
MLX_DIR=$HOME/.local/mlx RUST_TEST_THREADS=1 \
cargo test -p ironmlx --lib -- --nocapture
```

Observed result: `337 passed`, `15 ignored`.

Note: the default parallel lib test run hit a SIGSEGV once under the current local MLX install. The same tests pass serially, which points to MLX/global-runtime parallel test instability rather than a stable Rust test failure.

CLI text and multi-image smoke with the verified MLX static install:

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
/tmp/ironmlx-target-oldmlx/release/ironmlx generate \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --prompt 'Write one concise sentence about reproducible benchmarks.' \
  --max-tokens 32 \
  --temperature 0
```

Observed output starts directly with the answer: `Reproducible benchmarks ensure...`.

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
/tmp/ironmlx-target-oldmlx/release/ironmlx generate \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --prompt 'You are given two images. In one sentence per image, describe image 1 and image 2.' \
  --image ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg \
  --image ironmlx/tests/fixtures/p6_qwen35_vl/multi_image/image_1.jpg \
  --max-tokens 64 \
  --temperature 0
```

Observed output describes the kitchen scene for image 1 and the sidewalk/construction-wall scene for image 2.

Serve API quality harness with the verified MLX static install:

```bash
CARGO_TARGET_DIR=/tmp/ironmlx-target-oldmlx \
MLX_DIR=/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344 \
/tmp/ironmlx-target-oldmlx/release/ironmlx serve \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --host 127.0.0.1 \
  --port 18165 \
  --b-max 2 \
  --prefill-chunk-size 2048 \
  --max-cache-cap 32768
```

```bash
uv run python tools/qwen36_vl_quality.py \
  --target ironmlx=qwen3_6_moe=http://127.0.0.1:18165/v1 \
  --out-dir /tmp/ironmlx-qwen36-vl-quality/serve_final \
  --max-tokens 64 \
  --timeout-sec 900
```

Observed result: `3` records, `0` failures. The generated report covers `text_baseline`, `single_image_cats`, and `multi_image_kitchen_street`.

## Runtime Policy

Until the upstream MLX quantized-matmul regression at commit `2165dc08d7b33258260aa849d39f087d50e62962` is fixed or bisected, Qwen3.6 MoE/VL production validation should use the verified static MLX install captured above. The ironmlx model implementation should not add dequantization, tiny output chunking, or other slow compatibility paths to mask the runtime regression.
