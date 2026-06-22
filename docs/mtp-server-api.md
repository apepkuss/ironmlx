# Qwen MTP Support Matrix

## Scope

MTP is exposed in two product entry points:

- `ironmlx generate --mtp-model-dir ...` enables Qwen MTP for greedy CLI
  generation.
- `ironmlx serve --mtp-model-dir ...` enables Qwen MTP as a server-startup
  feature. OpenAI and Anthropic request bodies do not accept per-request MTP
  parameters.

## Support Matrix

| Model family | Text CLI | VL CLI | OpenAI server | Anthropic server | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 dense + matching MTP head | Supported | Supported | Supported | Supported | Uses the Qwen dense main model and Qwen MTP head. |
| Qwen3.5 MoE + matching MTP head | Supported | Supported | Supported | Supported | Uses the Qwen MoE main model and Qwen MoE MTP head. |
| Qwen3.6 dense + matching MTP head | Supported | Supported | Supported | Supported | Checkpoints still dispatch through the Qwen3.5 dense execution architecture; omitted draft depth defaults to the Phase 4 policy. |
| Qwen3.6 MoE + matching MTP head | Supported | Supported | Supported | Supported | `serve` preserves the Qwen3.6 MoE facade before entering the shared MoE execution kernel. |
| Non-Qwen architectures | Not supported | Not supported | Not supported | Not supported | `--mtp-model-dir` is rejected at startup/CLI validation. |

## Current constraints

- MTP is supported only for Qwen dense/MoE main models with a matching Qwen MTP
  head.
- `ironmlx generate --mtp-model-dir` supports Qwen text and Qwen VL greedy
  requests. VL requests use the scheduler-backed MTP path so the draft head sees
  the text-backbone hidden state produced after vision token replacement.
- `ironmlx serve --mtp-model-dir` supports `--b-max N` for `N >= 1`.
- Server MTP runs only for scheduler-eligible greedy requests. Qwen VL requests
  are eligible after the vision prefill path has produced the text-backbone
  state; non-greedy sampling and other non-eligible requests fall back to the
  regular scheduler path while keeping the request successful.
- `--paged-prefix-cache-dir` can be combined with `--mtp-model-dir`; repeated
  eligible text or Qwen VL prompts restore both the main paged prefix cache and
  the MTP draft cache state. Passing `--paged-prefix-cache-dir` without a value
  uses `~/.ironmlx/cache/paged_prefix_cache`.
- `--mtp-draft-tokens` is a startup-level setting. If omitted, the Phase 4
  model-aware default policy chooses the draft depth.

## `/healthz` MTP fields

`GET /healthz` always returns an `mtp` object:

```json
{
  "mtp": {
    "enabled": true,
    "draft_tokens": 2,
    "prefill_count": 7,
    "step_count": 42
  }
}
```

Field meanings:

- `enabled`: `true` when the server was started with an MTP head.
- `draft_tokens`: configured startup draft-token budget, or `null` when MTP is
  disabled.
- `prefill_count`: number of scheduler MTP prefill calls observed by the server.
- `step_count`: number of scheduler MTP decode-step calls observed by the server.

When MTP is disabled, the shape remains stable:

```json
{
  "mtp": {
    "enabled": false,
    "draft_tokens": null,
    "prefill_count": 0,
    "step_count": 0
  }
}
```

## Non-goals

The current MTP support does not add dynamic per-request `mtp_model_dir` or
`mtp_draft_tokens`. The request API remains compatible with the existing OpenAI
and Anthropic payloads, while server observability can confirm whether the
startup-level MTP path is active.

## Validation Targets

Core and unit coverage:

```sh
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx \
  cli::generate::tests::mtp_support_policy_allows_qwen_text_and_vl_and_rejects_other_architectures

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx \
  cli::serve::scheduler_profile_tests::serve_qwen_moe_dispatch_preserves_qwen36_checkpoint_identity

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx \
  core::server::openai::tests::chat_completions_routes_streaming_and_unary_scheduler_requests

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx \
  core::server::anthropic::tests::messages_routes_streaming_and_unary_scheduler_requests
```

Real-checkpoint VL + MTP + paged-prefix validation remains ignored by default:

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
