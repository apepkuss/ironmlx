# MTP Acceptance Checklist

This checklist covers usage-layer validation for Qwen3.5/Qwen3.6 MTP support.
Real-checkpoint tests are ignored by default because they require local model
snapshots and an MLX runtime.

## Environment Variables

| Variable | Checkpoint |
| --- | --- |
| `MLX_DIR` | Local MLX C++ runtime directory, usually `$HOME/.local/mlx`. |
| `QWEN35_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` snapshot. |
| `QWEN35_MTP_MODEL` | `mlx-community/Qwen3.5-4B-MTP-4bit` snapshot. |
| `QWEN36_DENSE_MODEL` | `mlx-community/Qwen3.6-27B-4bit` snapshot. |
| `QWEN36_DENSE_MTP_MODEL` | `mlx-community/Qwen3.6-27B-MTP-4bit` snapshot. |
| `QWEN36_MOE_MODEL` | `mlx-community/Qwen3.6-35B-A3B-4bit` snapshot. |
| `QWEN36_MOE_MTP_MODEL` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` snapshot. |

## Fast Validation

```sh
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx \
  cli::generate::tests::mtp_support_policy_allows_qwen_text_and_vl_and_rejects_other_architectures

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode -- --nocapture

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib health_collector_mtp -- --nocapture

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --test cli_generate_mtp_e2e -- --list
```

## CLI Real-Checkpoint Smoke Tests

Text-only model-path coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen35_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_DENSE_MODEL=/path/to/Qwen3.6-27B-4bit/snapshots/<sha> \
QWEN36_DENSE_MTP_MODEL=/path/to/Qwen3.6-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen36_dense_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<sha> \
QWEN36_MOE_MTP_MODEL=/path/to/Qwen3.6-35B-A3B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen36_moe_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

VL model-path coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen35_vl_generate_with_mtp_accepts_image_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<sha> \
QWEN36_MOE_MTP_MODEL=/path/to/Qwen3.6-35B-A3B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen36_moe_vl_generate_with_mtp_accepts_image_request \
  -- --ignored --test-threads=1 --nocapture
```

## Server Real-Checkpoint Smoke Tests

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test vl_mtp_paged_prefix_e2e \
  qwen35_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<sha> \
QWEN36_MOE_MTP_MODEL=/path/to/Qwen3.6-35B-A3B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test vl_mtp_paged_prefix_e2e \
  qwen36_moe_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss \
  -- --ignored --test-threads=1 --nocapture
```

## `/healthz` Acceptance

When the server starts with `--mtp-model-dir`, `/healthz.mtp.enabled` must be
`true` and `draft_tokens` must match the configured or model-aware default draft
depth.

For greedy eligible requests:

- `prefill_count` increases when the actor calls the scheduler MTP prefill path.
- `step_count` increases when the actor calls the scheduler MTP decode path.
- `drafted_tokens` and `accepted_draft_tokens` reflect the latest cumulative
  scheduler MTP stats, with `accepted_draft_tokens <= drafted_tokens`.

For non-greedy or otherwise ineligible requests:

- `fallback_prefill_count` increases.
- `prefill_count` and `step_count` do not increase because that request uses the
  ordinary scheduler path.
