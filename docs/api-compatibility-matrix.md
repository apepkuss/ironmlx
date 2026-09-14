# IronMLX 0.1.0 API compatibility matrix

[简体中文](zh-CN/api-compatibility-matrix.md)

For client integrators: this document describes the 0.1.0 public protocol scope. Use documentation at the release tag; candidate Bundle metadata identifies its exact source commit. A moving branch is not a release baseline.

See the [API quick start](api.md), [protocol reference](api-reference.md) for nested fields, tools, reasoning and schemas, and [Supported models](supported-models.md) for model conditions.

## Shared rules

All three protocols support synchronous text, SSE, supported Structured Outputs and native-template-dependent client tools.
Responses uses typed events, Chat ends with `[DONE]`, and Messages uses its native event lifecycle.
Responses reasoning items and Messages thinking blocks require exact templates; Chat does not expose the same typed reasoning shape. Messages can combine thinking with final JSON constraints.

Images accept supported base64 shapes only, not remote URLs. The service does not execute external tools, provide hosted tools or store conversations.
The tables list accepted fields and conditions. Unlisted fields and shapes outside these conditions return 400 rather than being silently ignored.

## Chat Completions

| Fields | Accepted scope and limits |
| --- | --- |
| `model` | Optional; resolved by the current service/default model |
| `messages` | Text, supported content parts and assistant/tool history |
| `tools / tool_choice / parallel_tool_calls` | Function tools; auto/none/required/named function; false limits a turn to one call and requires tools |
| `response_format` | text, json_object or supported json_schema |
| `stream / stream_options` | Synchronous/SSE; stream_options only accepts include_usage |
| `max_tokens` | Output budget within model context capacity |
| `temperature / top_p` | Finite values in [0,2] / (0,1] |
| `reasoning_effort` | Qwen3.8 native low/medium/xhigh tiers; requires the matching template |
| `seed / ignore_eos / chat_template_kwargs` | IronMLX extensions: request seed, controlled-length generation and supported template kwargs |
| `functions / function_call / top_k / repetition_penalty` | Rejected; use tools instead of legacy function fields |

## Responses

| Fields | Accepted scope and limits |
| --- | --- |
| `model / instructions / input` | Stateless text or supported typed history |
| `tools` | Supported function and namespace subsets; bounded schemas |
| `tool_choice / parallel_tool_calls` | auto/none/required/named function; cannot force a namespace subfunction |
| `text` | json_object or supported json_schema format |
| `stream / stream_options` | Typed SSE; stream_options must be an object |
| `max_output_tokens` | Output budget within context capacity |
| `temperature / top_p` | Finite values in [0,2] / (0,1] |
| `reasoning` | Local effort/summary semantics; missing effort is none; plaintext output |
| `store / background` | Only false or omitted; true is rejected |
| `previous_response_id / conversation` | Rejected; no server-side history storage |
| `include` | Only reasoning.encrypted_content request shape; no encrypted content is generated |
| `prompt_cache_key / client_metadata / metadata` | Structure and length validation only, without hosted-platform semantics |
| `service_tier / truncation` | Tier: auto/default only; truncation: disabled only |
| `top_k / repetition_penalty` | Rejected |

## Anthropic Messages

| Fields | Accepted scope and limits |
| --- | --- |
| `model / messages / system` | Text, base64 images, tool history and signed thinking; system accepts text or text blocks |
| `tools / tool_choice` | Client functions; auto/any/named tool/none with supported parallel controls |
| `output_config.format` | Supported JSON Schema format only |
| `output_config.effort / thinking` | Requires supported thinking templates; disabled/enabled/adaptive and strictly validated budgets/display; not a calibrated Claude budget |
| `max_tokens / stream` | Output budget and synchronous/SSE choice |
| `temperature / top_p / top_k` | Finite [0,1] / (0,1]; top_k is a positive integer |
| `repetition_penalty / output_format` | Rejected; use output_config.format |
| `display:omitted / redacted_thinking` | Rejected; no encrypted hidden-thinking channel |

## Errors and retries

Chat/Responses use OpenAI error envelopes; Messages uses an Anthropic envelope with matching body request_id and request-id header. error.code is an IronMLX machine-readable extension.

| HTTP | Condition | Example code |
| ---: | --- | --- |
| 400 | Malformed JSON or unknown fields | `invalid_json` |
| 400 | Invalid sampling or constraints | `invalid_request / invalid_sampling_parameters` |
| 400 | Unsupported Schema, tools or thinking | `invalid_response_format / invalid_tools / invalid_request` |
| 413 | Body exceeds 32 MiB | `request_body_too_large` |
| 413 | Input plus output exceeds context | `request_token_capacity_exceeded` |
| 503 | Queue, engine, memory or storage backpressure | `scheduler_queue_full / engine_unavailable / memory_budget_exceeded` |
| 500 | Unexpected generation failure | `generation_error` |

Retryable 503 responses return JSON and `Retry-After: 5`; Messages uses `overloaded_error` for overload and `request_too_large` for 413. See the [protocol reference](api-reference.md) for more codes and cancellation boundaries.

## Runtime modes and capability boundaries

Ordinary serving, DFlash2, Gemma4 drafter, DiffusionGemma, EnginePool and App daemon share request validation, protocol errors, SSE headers and streaming cancellation/resource-release contracts, not identical model capabilities.
Fixed services select the model at startup; EnginePool/App daemon resolve the request model or default. App DFlash2 retains read-only discovery, without dynamic model management.
See [DFlash2](dflash2-server-api.md) for sampling, concurrency and incompatible combinations.

## SDK compatibility checks

| SDK | Pinned version | Coverage |
| --- | --- | --- |
| OpenAI Python | `2.48.0` | Chat / Responses, SSE, tools, Structured Outputs, reasoning, 400/413/503 |
| Anthropic Python | `0.121.0` | Messages, SSE, tools, Structured Outputs + thinking, 400/413/503 |

Pinned SDKs access a fixture server over real loopback HTTP/SSE to check client parsing; Rust tests separately cover production request/response contracts. Neither loads a model or establishes response quality, tool selection accuracy or performance.
Run from the repository root:

```bash
python3 -m venv /tmp/ironmlx-api-contract-sdk
/tmp/ironmlx-api-contract-sdk/bin/python -m pip install -r scripts/api-contract-sdk/requirements.txt
/tmp/ironmlx-api-contract-sdk/bin/python scripts/api-contract-sdk/contract.py --fixture
cargo test --locked --all-features -p ironmlx --lib core::server::
```

## Maintenance requirements

New fields must update strict parsing, contract tests, SDK checks and both translations together. Changes to error status, envelope, code or Retry-After require matching checks. Mark extensions explicitly rather than claiming upstream-standard behavior. Record model revision, template, quantization, sampling, response mode and results separately for real-model claims.
