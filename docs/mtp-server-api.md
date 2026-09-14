# Qwen MTP Support Matrix

[简体中文](zh-CN/mtp-server-api.md)

For CLI/API integrators configuring Qwen MTP. Start with a compatible main model and matching MTP head; the server flag enables a path, not a guarantee that every request uses it.

## Scope

MTP is exposed in two product entry points:

- `ironmlx generate --mtp-model-dir ...` enables Qwen MTP for greedy CLI
  generation.
- `ironmlx serve --mtp-model-dir ...` enables Qwen MTP as a server-startup
  feature. OpenAI and Anthropic request bodies do not accept per-request MTP
  parameters.

## Support Matrix

| Main model | Required auxiliary model |
| --- | --- |
| Qwen3.5 / 3.6 / 3.8 Dense | Matching Qwen MTP head |
| Qwen3.5 / 3.6 MoE | Matching Qwen MoE MTP head |

Compatible combinations support text/VL CLI and OpenAI/Anthropic serving, subject to the request constraints below.
This page describes the Qwen MTP path. The same `--mtp-model-dir` CLI option also accepts a compatible Gemma4 assistant drafter, which uses a separate execution path and sampling rules. Do not apply the Qwen-only constraints below to Gemma4; see [Supported models](supported-models.md).

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
- `--mtp-draft-tokens` is a startup-level setting. If omitted, the model-aware default policy chooses the draft depth.

## `/healthz` MTP fields

`GET /healthz` always returns an `mtp` object:

```json
{
  "mtp": {
    "enabled": true,
    "draft_tokens": 2,
    "prefill_count": 7,
    "step_count": 42,
    "fallback_prefill_count": 1,
    "drafted_tokens": 84,
    "accepted_draft_tokens": 63
  }
}
```

Field meanings:

- `enabled`: `true` when the server was started with an MTP head.
- `draft_tokens`: configured startup draft-token budget, or `null` when MTP is
  disabled.
- `prefill_count`: number of scheduler MTP prefill calls observed by the server.
- `step_count`: number of scheduler MTP decode-step calls observed by the server.
- `fallback_prefill_count`: number of MTP-enabled prefill calls that used the
  ordinary scheduler path because the active batch was not MTP-eligible.
- `drafted_tokens`: latest cumulative number of draft tokens proposed by the
  scheduler MTP path.
- `accepted_draft_tokens`: latest cumulative number of proposed draft tokens
  accepted by the main-model verification path.

When MTP is disabled, the shape remains stable:

```json
{
  "mtp": {
    "enabled": false,
    "draft_tokens": null,
    "prefill_count": 0,
    "step_count": 0,
    "fallback_prefill_count": 0,
    "drafted_tokens": 0,
    "accepted_draft_tokens": 0
  }
}
```

## Non-goals

The current MTP support does not add dynamic per-request `mtp_model_dir` or
`mtp_draft_tokens`. The request API remains compatible with the existing OpenAI
and Anthropic payloads, while server observability can confirm whether the
startup-level MTP path is active.
