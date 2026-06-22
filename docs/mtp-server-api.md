# MTP Server API

## Scope

Phase 5 exposes MTP as a server-startup feature. Operators enable it with
`ironmlx serve --mtp-model-dir ...`; OpenAI and Anthropic request bodies do not
accept per-request MTP parameters.

## Current constraints

- MTP is supported only for Qwen dense/MoE main models with a matching Qwen MTP
  head.
- `ironmlx serve --mtp-model-dir` supports `--b-max N` for `N >= 1`.
- MTP runs only for scheduler-eligible greedy requests. Qwen VL requests are
  eligible after the vision prefill path has produced the text-backbone state;
  non-greedy sampling, unsupported architectures, and other non-eligible
  requests fall back to the regular scheduler path.
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

Phase 5 does not add dynamic per-request `mtp_model_dir` or
`mtp_draft_tokens`. The request API remains compatible with the existing OpenAI
and Anthropic payloads, while server observability can confirm whether the
startup-level MTP path is active.
