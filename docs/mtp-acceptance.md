# MTP Acceptance Checklist

This checklist covers usage-layer validation for Qwen3.5/Qwen3.6/Qwen3.8 MTP
and Gemma4 assistant-drafter support. Real-checkpoint tests are ignored by
default because they require local model snapshots and an MLX runtime.

MTP capability acceptance requires output-equivalent greedy generation plus
non-zero draft activity. It does not imply a speedup for every checkpoint,
prompt, context length, or draft depth; performance claims require a balanced
fixed-condition comparison against the same base model without MTP.

## Environment Variables

| Variable | Checkpoint |
| --- | --- |
| `MLX_DIR` | Local MLX C++ runtime directory, usually `$HOME/.local/mlx`. |
| `QWEN35_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` snapshot. |
| `QWEN35_MTP_MODEL` | `mlx-community/Qwen3.5-4B-MTP-4bit` snapshot. |
| `QWEN36_DENSE_MODEL` | `mlx-community/Qwen3.6-27B-4bit` snapshot. |
| `QWEN36_DENSE_MTP_MODEL` | `mlx-community/Qwen3.6-27B-MTP-4bit` snapshot. |
| `QWEN38_DENSE_MODEL` | `mlx-community/Qwen3.8-27B-4bit` snapshot. |
| `QWEN38_DENSE_MTP_MODEL` | `mlx-community/Qwen3.8-27B-MTP-4bit` snapshot. |
| `QWEN36_MOE_MODEL` | `mlx-community/Qwen3.6-35B-A3B-4bit` snapshot. |
| `QWEN36_MOE_MTP_MODEL` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` snapshot. |
| `GEMMA4_LONG_CONTEXT_MODEL` | Matching `gemma4` or `gemma4_unified` base checkpoint. |
| `GEMMA4_LONG_CONTEXT_DRAFTER` | Matching Gemma4 assistant checkpoint. |

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
QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen38_dense_text_generate_with_mtp_accepts_request \
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

Strict Qwen exact-verify matrix (the shared Qwen3.5 execution test also accepts
Qwen3.8 checkpoints):

```sh
MLX_DIR=$HOME/.local/mlx \
PROMPT_LOOKUP_VERIFY_QWEN35_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
PROMPT_LOOKUP_VERIFY_REQUIRE_ZERO_DIFF=1 \
PROMPT_LOOKUP_VERIFY_BATCHES=1,2,4,8 \
PROMPT_LOOKUP_VERIFY_PREFIX_LENS=1024,1025,4096,4097,8192,32768,65536 \
PROMPT_LOOKUP_VERIFY_WIDTHS=2,3,4,5,6,8 \
PROMPT_LOOKUP_VERIFY_MAX_WIDTH=8 \
cargo test --release -p ironmlx --test prompt_lookup_verify_qualification \
  qwen35_dense_qgt1_matches_sequential_verify \
  -- --ignored --test-threads=1 --nocapture
```

Qwen3.8-27B long-context exact-path coverage (default 8K; set
`MTP_LONG_CONTEXT_TOKENS=32768` or `65536` for the extended matrix):

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen38_dense_mtp_long_context_remains_on_exact_path \
  -- --ignored --test-threads=1 --nocapture
```

The request must increase `prefill_count`, `step_count`, `drafted_tokens`, and
`accepted_draft_tokens` without increasing `fallback_prefill_count`. MTP has no
separate 1024/4096 context cap; the model context limit, `--max-cache-cap`, and
memory budget still apply.

Gemma4 and Gemma4 Unified long-context assistant-drafter parity (defaults to
8K, 32K, and 64K contexts with 64 generated tokens):

```sh
MLX_DIR=$HOME/.local/mlx \
GEMMA4_LONG_CONTEXT_MODEL=/path/to/gemma4-base/snapshots/<sha> \
GEMMA4_LONG_CONTEXT_DRAFTER=/path/to/gemma4-assistant/snapshots/<sha> \
cargo test --release -p ironmlx --test gemma4_long_context_parity \
  gemma4_drafter_long_context_tokens_match_ordinary_q1_exactly \
  -- --ignored --test-threads=1 --nocapture
```

The test requires exact token equality with ordinary greedy Q1 generation and
non-zero verify windows and drafted tokens at every context length. Performance
must be measured separately against the same base checkpoint without a drafter;
passing parity does not imply a decode or end-to-end speedup.

Gemma4 PromptLookup exact verify has no separate 1024-token context cap. Verify
the production qualification at boundary and long-context lengths:

```sh
MLX_DIR=$HOME/.local/mlx \
PROMPT_LOOKUP_VERIFY_GEMMA4_MODEL=/path/to/gemma4-base/snapshots/<sha> \
PROMPT_LOOKUP_VERIFY_BATCHES=1,2,4,8 \
PROMPT_LOOKUP_VERIFY_PREFIX_LENS=1024,1025,8192,32768,65536 \
PROMPT_LOOKUP_VERIFY_WIDTHS=2,3,4,5 \
cargo test --release -p ironmlx --test prompt_lookup_verify_qualification \
  gemma4_qgt1_matches_sequential_verify \
  -- --ignored --test-threads=1 --nocapture
```

Affine4 Gemma4 checkpoints use sequential Q1 PromptLookup verification with
TurboQuant KV because K3V4/K4V4 Q>1 is not token exact. This does not affect the
separate assistant-drafter K3V4 path below.

For assistant-drafter K3V4, long-context Q>1 verify uses the stable attention
path regardless of reserved scheduler capacity. This test covers both one and
two active requests under `b_max=4`, requires exact Q1 token parity, and checks
that the second draft position is attempted:

```sh
MLX_DIR=$HOME/.local/mlx \
GEMMA4_LONG_CONTEXT_MODEL=/path/to/gemma4-base/snapshots/<sha> \
GEMMA4_LONG_CONTEXT_DRAFTER=/path/to/gemma4-assistant/snapshots/<sha> \
GEMMA4_K3V4_CONTEXT_TOKENS=8192 \
GEMMA4_K3V4_ACTIVE_REQUESTS=1,2 \
cargo test --release -p ironmlx --test gemma4_long_context_parity \
  gemma4_k3v4_long_context_scheduler_uses_multi_token_verify_exactly \
  -- --ignored --test-threads=1 --nocapture
```

Use `GEMMA4_K3V4_ACTIVE_REQUESTS=1` when validating 64K on hardware where the
B2 ordinary-Q1 prefill exceeds the Metal single-buffer or memory budget.

Qwen3.6-27B MTP Active KV swap-out/swap-in coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_DENSE_MODEL=/path/to/Qwen3.6-27B-4bit/snapshots/<sha> \
QWEN36_DENSE_MTP_MODEL=/path/to/Qwen3.6-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen36_dense_mtp_active_kv_offload_restores_speculative_side_cache \
  -- --ignored --test-threads=1 --nocapture
```

Qwen3.8-27B MTP Active KV swap-out/swap-in coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen38_dense_mtp_active_kv_offload_restores_speculative_side_cache \
  -- --ignored --test-threads=1 --nocapture
```

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
