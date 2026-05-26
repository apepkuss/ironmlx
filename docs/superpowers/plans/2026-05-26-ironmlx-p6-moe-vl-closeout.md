# P6 MoE-VL Closeout

| Field | Value |
| --- | --- |
| Date | 2026-05-26 |
| Worktree | `/Users/xin/workspace/ironmlx-backend-moe-vl` |
| Branch | `ironmlx-p6-moe-vl` |
| Checkpoint | `~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec` |

## Scope

This closeout covers Qwen3.5 MoE VL enablement after the shared vision module
extraction. The goal was to prove that the existing OpenAI-compatible HTTP
pipeline can serve `qwen3_5_moe` checkpoints with image input, not to complete
numerical parity against mlx-vlm.

## Implementation Summary

- Shared `VisionTower` moved to `crate::models::vision`.
- `Qwen35MoeConfig` parses top-level `vision_config`.
- `Qwen35MoeModel` owns an optional `VisionTower` and loads it from
  multimodal checkpoints.
- MoE implements the scheduler-facing VL methods:
  `compute_vision_embeds`, `forward_vl_chunk`, and `batched_prefill_vl`.
- `ironmlx serve` now supports both `qwen3_5` and `qwen3_5_moe` VL checkpoints
  through the same OpenAI chat completions path.

## Smoke Evidence

All smoke commands used:

```bash
MLX_DIR=$HOME/.local/mlx
MODEL=$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
```

Observed results:

| Area | Result |
| --- | --- |
| checkpoint discovery | `mlx-community/Qwen3.5-35B-A3B-4bit` is `image-text-to-text`; config has `vision_config`; index has `vision_tower.*` |
| real checkpoint load | `loads_qwen35_moe_vision_tower_from_real_checkpoint ... ok` |
| single-image unary OpenAI request | HTTP 200, `finish=stop`, non-empty image description |
| single-image SSE request | HTTP 200, 22 content chunks, non-empty image description |
| GS chunked unary | `--prefill-chunk-size 256`, HTTP 200, non-empty output |
| GS chunked SSE | `--prefill-chunk-size 256`, HTTP 200, 25 content chunks |
| two-image unary request | HTTP 200, non-empty two-image description |
| `--b-max 2` mixed concurrency | concurrent `VL + VL + text-only`, all HTTP 200 |
| 2-image semantic script | `p6_6_semantic_check.py --n-images 2`: PASS |
| 3-image semantic script | `p6_6_semantic_check.py --n-images 3`: PASS |

Semantic script reports were written during validation to:

- `/tmp/ironmlx_moe_vl_p6_6_semantic_report.md`
- `/tmp/ironmlx_moe_vl_p6_6_n3_semantic_report.md`

The final repeatable smoke command also passed after the review-fix update
that removed global process killing from the semantic script:

```bash
SKIP_BUILD=1 BASE_PORT=18600 OUT_DIR=/tmp/ironmlx-p6-moe-vl-smoke-after-review-fix \
  ./scripts/p6_moe_vl_smoke.sh
```

Summary output: `/tmp/ironmlx-p6-moe-vl-smoke-after-review-fix/summary.jsonl`.

## Rust Verification

Commands and observed outcomes:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib models::qwen3_5_moe
```

Result: 15 passed, 0 failed, 1 ignored.

```bash
QWEN35_MOE_VL_MODEL=$MODEL MLX_DIR=$HOME/.local/mlx \
  cargo test -p ironmlx --lib \
  models::qwen3_5_moe::model::tests::loads_qwen35_moe_vision_tower_from_real_checkpoint \
  -- --ignored --exact --nocapture
```

Result: 1 passed.

```bash
cargo fmt
cargo +nightly fmt --all -- --check
```

Result: both passed.

```bash
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Result: both fail without `MLX_DIR` because `mlx-sys/build.rs` requires MLX
headers/libs. With `MLX_DIR=$HOME/.local/mlx`, both commands passed:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Additional checks:

```bash
git diff --check
```

Result: passed.

## Non-Blocking Failures Observed

These failures were observed while broadening validation. They are recorded here
so they do not get confused with MoE VL regressions.

| Command | Failure | Classification |
| --- | --- | --- |
| `MLX_DIR=$HOME/.local/mlx cargo test --workspace` | `mlx` crate test `p2c_io` link failure due missing `gguf_*` symbols | dependency/environment link issue |
| `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx` | `tests/p3b1_mrope.rs` has 3 fixture shape failures (`expected [1,8,32]`, got `[1,8,64]`) | documented existing HEAD baseline issue in `docs/p5h+2-c-close-out.md` |
| `item3_semantic_check.py` | not run because `/tmp/p6vl_test_imgs` fixture directory is absent | missing external fixture |

## Repro Script

Use `scripts/p6_moe_vl_smoke.sh` to rerun the MoE VL smoke matrix. The script
builds the release binary by default, starts/stops local servers, runs unary,
SSE, GS chunked, mixed concurrency, and 2/3-image semantic checks.

## Follow-Ups

1. Decide whether to move `qwen3_5::image_processor` into `models::vision`.
2. Add mlx-vlm or HF reference numerical comparison for MoE VL first-token/logit parity.
3. Restore `item3_semantic_check.py` coverage once `/tmp/p6vl_test_imgs` is available.
4. Investigate the workspace `gguf_*` link issue and the existing `p3b1_mrope` fixture mismatch separately from MoE VL.
