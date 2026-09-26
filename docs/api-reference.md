# HTTP API protocol reference

[简体中文](zh-CN/api-reference.md) · [Quick start](api.md)

Jump to [Responses](#openai-responses), [Chat Completions](#openai-chat-completions), [Messages](#anthropic-messages), [Images](#images) or [Log-level management](#log-level-management-loopback-only).

For client integrators. The App defaults to `http://127.0.0.1:9068`; direct CLI serving defaults to port 8080. Use the actual configured endpoint.

## Single backend instance

One `ironmlx serve` backend is allowed per macOS user, regardless of arguments or port. Before MLX, metallib or model initialization, the process takes an exclusive nonblocking lock on `~/.ironmlx/run/backend.lock` until exit. Normal exit, crashes and SIGKILL release the lock; the file itself can remain and is not a liveness indicator.
A second instance exits with `ironmlx_instance_already_running`. The App stops its automatic recovery loop and asks the user to exit the existing instance.

## Health and model discovery

`GET /health` indicates HTTP responsiveness. `GET /healthz` returns version, model, scheduler, cache and memory state.
App daemon and EnginePool expose an OpenAI-compatible `GET /v1/models` list. Entries include `id`, `object:"model"`, `created`, `owned_by`, load policy and runtime state. Registered models can be discovered even before loading.

When reliably known, each causal model entry also includes these IronMLX extension fields:

- `context_window`: effective total token capacity, the smaller of the model's context capacity and the deployment's `max_cache_cap` (App **MAX CONTEXT TOKENS**).
- `max_output_tokens`: output-token ceiling, including reasoning, answer text and tool calls. Causal serving currently has no separate output-only hard cap, so this equals `context_window`. For each request, the output budget must still fit within `context_window - input_tokens`, including chat-template and multimodal input tokens. This is **not** the App's **MAX OUTPUT TOKENS** setting: that setting supplies a default Responses budget only when the request omits it.

For example, a model with a 262144-token context deployed with `max_cache_cap=65536` advertises `context_window:65536` and `max_output_tokens:65536`, even if its default output budget is 256. Clients must subtract input usage before choosing an output budget; the advertised ceiling does not guarantee that amount for a nonempty prompt or guarantee a complete answer. Admission and memory checks still apply.

Loaded models use their actual admission capacity. Unloaded causal models use explicit checkpoint metadata and the registered scheduler configuration without loading weights. Missing, invalid or unsupported capacity metadata is omitted, not returned as zero or inferred from generation defaults. Audio and DiffusionGemma currently omit both fields. App DFlash2 discovery reports the loaded target's effective capacity, never the draft model's capacity. These are optional extensions; clients should retain fallbacks when absent.

`memory.free_ram_bytes` is raw OS free memory for observation. `available_ram_bytes` includes reclaimable memory using the governor's accounting. `process_governor.pressure_level` determines memory health; `degraded_reasons` identifies queue, cache, pressure, telemetry or backpressure causes.

### App decision runtime metrics

The App-daemon-only `GET /admin/api/models/loaded` response includes
`decision_metrics` for a loaded model whose `runtime_kind` is `decision`. Other
runtime kinds omit this object.

| Field | Type | Meaning |
| --- | --- | --- |
| `window_seconds` | integer | Recent performance window in seconds; currently `60`. |
| `completed_requests` | integer | Successful requests since the model instance was loaded. |
| `failed_requests` | integer | Failed requests since the model instance was loaded. |
| `recent_completed_requests` | integer | Successful requests represented in the current recent window. |
| `latency_ms_p50` | number or `null` | Median end-to-end latency in the recent window, in milliseconds. |
| `input_tokens_per_second` | number or `null` | Median per-request input-token rate in the recent window. |
| `questions_per_second` | number or `null` | Median per-request question rate in the recent window. |
| `last_request_unix_ms` | integer or `null` | Completion time of the latest successful or failed request, as Unix epoch milliseconds. |

Recent performance fields are `null` when the window contains no successful
samples. Counts are cumulative only for the current loaded model instance and
reset when the model is unloaded and loaded again or when the backend restarts.
The Dashboard combines active and queued request counts with
`last_request_unix_ms` to distinguish **Processing**, **Just completed** and
**Idle** without requiring faster polling. See the [Laya guide](laya-systemone-api.md#status-and-decision-metrics)
for the user-facing presentation.

## Errors

Chat and Responses use an OpenAI error envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_json"
  }
}
```

Messages uses an Anthropic envelope and a matching `request-id` header:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "...",
    "code": "invalid_json"
  },
  "request_id": "req_..."
}
```

`error.code` is an IronMLX machine-readable extension. Classify by HTTP status and protocol error type first.

| HTTP | Conditions / codes | Retry |
| ---: | --- | --- |
| 400 | Invalid JSON, fields or constraints | Correct the request |
| 413 | `request_body_too_large`, `request_token_capacity_exceeded` | Reduce body size or token budget; context errors include capacity details |
| 503 | `scheduler_queue_full`, `scheduler_unavailable`, `scheduler_reply_lost`, `engine_unavailable`, `diffusion_lane_overloaded` | `Retry-After: 5` |
| 503 | `memory_budget_exceeded`, `memory_pressure`, `prefill_peak_unsafe`, `vision_prefill_peak_unsafe`, `cold_materialization_unsafe`, `prefix_store_backpressure` | `Retry-After: 5` |
| 500 | `generation_error` or internal failure | Inspect diagnostics |

Retryable 503 errors return JSON; Messages uses `overloaded_error` for overload. Messages 413 errors use `request_too_large`.

### Runtime topology

Ordinary causal serving, DFlash2, Gemma4 drafter, DiffusionGemma, EnginePool and App daemon share strict extraction, model-independent validation, protocol error rendering and SSE headers. This does not imply identical model capabilities.
Fixed services select a model at startup; EnginePool/App daemon resolve the request model or their default. `/v1/models` is available in EnginePool, App daemon and App DFlash2 discovery. `/admin/api/models/*` is App-daemon-only.

### SSE disconnection and cancellation

Once SSE starts, dropping the HTTP response publishes a cancellation signal. Encoders stop consuming generation events and do not fabricate a terminal event after observing disconnect.

| Path | Cancellation boundary | Released state |
| --- | --- | --- |
| Scheduler, including MTP/drafter | Next safe scheduling boundary after the current forward | Request, slot, KV cache and budget |
| DFlash2 | Next safe event boundary after target/draft forward | Per-request caches, slot and budget |
| Direct GenerationStream | Next token boundary after forward | Generation state and reservation |
| DiffusionGemma | Next event boundary after the current diffusion step | Lane and request state |

Cancellation does not interrupt an in-flight Metal operation. Resource release can therefore include the remainder of that operation. Version 0.1 does not promise cancellation of underlying non-streaming generation when its client disconnects.

## DFlash2 serving

`ironmlx serve --dflash2-model-dir ...` starts a fixed Qwen3.8 target/draft actor for text Chat, Responses and Messages, synchronously or with SSE. It supports greedy/exact sampling and `--max-sequences N` for positive N.
The App switches execution paths through a controlled restart. It preserves target model discovery but does not expose dynamic `/admin/api/models/*` in this mode. See [DFlash2](dflash2-server-api.md).

## OpenAI Responses

`POST /v1/responses` accepts complete typed history on every request. Use `store:false`; the service does not persist responses or conversations or execute tools. See the [quick start](api.md) for a minimal request.
SSE uses `response.created`, typed output items and text/function-argument deltas, ending with `response.completed`, `response.incomplete` or `response.failed`, without Chat's `[DONE]`.

### Reasoning

```json
{
  "model": "your-model-id",
  "input": "Analyze carefully, then answer briefly.",
  "reasoning": {
    "effort": "high",
    "summary": "none"
  },
  "store": false
}
```

Exact native templates expose reasoning as a separate item, with `response.reasoning_text.delta` and `.done` events rather than mixing it into `output_text`.
For Qwen3.8, `minimal`/`low` map to `low`, `medium` to `medium`, and `high`/`xhigh`/`max` to `xhigh`; `none` disables reasoning. Other supported templates treat non-`none` effort as an enable switch. These are not calibrated reasoning-token budgets.
Omitted/null `reasoning` or missing effort means `none`, echoed as the effective setting. Clients must request a non-`none` effort explicitly.

For causal Qwen3.5/3.6/3.8 and Gemma4/Unified with exact supported templates,
enabled reasoning automatically reserves `min(floor(max_output_tokens / 4), 1024)`
tokens for the answer or tool call, plus native framing/UTF-8 boundary space.
The remainder is the reasoning budget. If reasoning is still open at that
boundary, decoding constrains the next tokens to finish the model's native
closing marker (`</think>` or `<channel|>`) and then continues the response.
The marker enters the actual model context and counts toward the original
total output limit. Natural early closure and Gemma's direct-answer path are
preserved. The same policy applies to Chat Completions and Messages using their
`max_tokens` total; no client changes are required.

The policy composes with JSON/tool constraints and request-local speculative
fork/rollback. Disabled reasoning, unrecognized templates, other reasoning
dialects, DiffusionGemma, and budgets too small to fit framing plus a positive
reasoning/answer allocation retain their existing behavior. This first policy
is automatic, with no new App setting or request field. It never enlarges an
explicit client budget or retries a request. Reservation does not guarantee a
complete or correct answer; if the total limit is reached, existing incomplete
status and item-level truncation reporting still apply. Completing a native
reasoning marker indicates channel closure, not a guarantee of reasoning quality.
The model can still emit analysis-style prose in the answer channel after a forced
transition, particularly with a very small total budget; this is not treated as
proof that the task was solved. Native channel delimiters cannot reopen reasoning
or be repeated in the answer under this policy.

History accepts plaintext `reasoning_text` in reasoning items. IronMLX does not generate hosted `encrypted_content`; encrypted-only reasoning history is rejected with 400. `summary:"auto"` can request automatic capability selection but does not cause generation or truncation into an independent reasoning summary.

For replay compatibility, an orphan plaintext reasoning item without a following assistant body or function call is skipped, while other history and normally paired reasoning are retained. This also applies to old items without `status`; it does not prove that the old turn hit an output limit. Opaque encrypted history remains invalid even when orphaned (an empty `reasoning_text` is not decryptable content).

Reasoning items carry `status:"in_progress"` when added, and `completed` or `incomplete` in `response.output_item.done`, the final response, and non-streaming output. A length limit marks the item incomplete only if the native reasoning channel is still open. A consumed reasoning closing marker or subsequent answer/tool call leaves that reasoning completed even if the whole response is incomplete. Clients should retain the full item from `response.output_item.done` for replay. This compatibility handling prevents history-validation failures; it neither restores the interrupted answer nor guarantees answer completeness.

### Function tools and history

```json
{
  "model": "your-model-id",
  "input": "What is the weather in Tokyo?",
  "store": false,
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          }
        },
        "required": [
          "city"
        ],
        "additionalProperties": false
      },
      "strict": true
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

The client executes `function_call` items and appends both the original call and a `function_call_output` with the same `call_id` to complete history. Text, supported image message items and call/output items work in synchronous and SSE modes.
[Hermes Agent](hermes-agent.md), [oh-my-pi](oh-my-pi.md), and [DeepSeek Harness](dsh.md) provide configuration guides.

### Structured Outputs

JSON mode uses `{"text":{"format":{"type":"json_object"}}}`. Schema mode uses:

```json
{
  "text": {
    "format": {
      "type": "json_schema",
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          }
        },
        "required": [
          "city"
        ],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

The resulting JSON remains text in a message's `output_text`; clients parse it. Grammar constraints apply before sampling and the completed JSON is validated again.
Supported Schema types are `object`, `array`, `string`, `number`, `integer`, `boolean`, `null` and nullable type arrays. Keywords include `properties`, `required`, `items`, `enum`, `const`, `anyOf`, `minItems`, `maxItems`, `minLength`, `maxLength`, `minimum`, `maximum`, `exclusiveMinimum` and `exclusiveMaximum`. Maximum depth is 8.
Non-strict tools also support nested dynamic objects with `additionalProperties:true` or a supported value Schema. Top-level tool parameter objects must remain closed. `strict:true` requires recursive `additionalProperties:false` and every property in `required`. Unsupported keywords fail before generation, without weakening constraints.

With tools and an output format together, `none` permits only final JSON, `required` or a named function permits only calls, and `auto` permits calls or Schema-conforming final JSON. The combined constraint applies across supported tool dialects and ordinary, scheduled, speculative and DiffusionGemma generation.

### Stateless limits

- `store` is omitted or false; true is rejected. `previous_response_id`, conversations, background execution and response retrieve/delete/cancel APIs are unsupported.
- Client functions and the supported `type:"namespace"` function groups are accepted. Namespaces compile to a bounded dispatcher and restore public namespace/function/argument shapes in output and replay. Named namespace subfunctions cannot be forced through `tool_choice`; use `auto` or `required`.
- Dynamic parameters outside the Schema subset are allowed only for non-strict namespace subfunctions using a JSON argument envelope; strict mode never downgrades.
- Hosted Web/File Search, MCP, Code Interpreter and custom/freeform tools are unsupported.
- No persisted reasoning, independent summary/refusal items, OpenAI file IDs, audio input/output, image output or image function results are provided.
- Public sampling is finite `temperature` in `[0,2]` and `top_p` in `(0,1]`. `top_k`, `repetition_penalty` and unknown fields are rejected.

## OpenAI Chat Completions

`POST /v1/chat/completions` uses `model`, `messages` and `max_tokens`. Use `stream:true` for SSE and `stream_options.include_usage:true` for a final usage chunk.
Top-level fields, messages, content parts, image payloads and stream options are strict. Public sampling is finite `temperature` in `[0,2]` and `top_p` in `(0,1]`; `top_k` and `repetition_penalty` are not public fields.

Qwen3.8 supports top-level `reasoning_effort` values `low`, `medium` and `xhigh` (default). `chat_template_kwargs.enable_thinking:false` disables thinking; `preserve_thinking:false` omits historical `reasoning_content`. Use Responses or Messages for independent reasoning items/blocks.

### Structured Outputs

JSON mode uses `{"response_format":{"type":"json_object"}}`. Schema mode nests the definition under `response_format.json_schema`, unlike Responses:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          }
        },
        "required": [
          "city"
        ],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

The Schema subset and tool/output choice rules match Responses. Synchronous, SSE, Scheduler, MTP/drafter and DiffusionGemma share constraints. `finish_reason:"length"` can leave JSON incomplete; treat it as truncated output.

### Function tools

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": [
            "city"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

The client appends the original assistant call message and each `role:"tool"` result with its `tool_call_id`. `function.arguments` is a JSON string. SSE arguments can span multiple deltas identified by stable `index` and `id`; the finish reason is `tool_calls`.

- `tool_choice` accepts `auto`, `none`, `required` or `{"type":"function","function":{"name":"..."}}`.
- `parallel_tool_calls` defaults to true; false limits the turn to one call. Llama 3.1/3.2 native custom tools always allow only one call and exclude its built-in-tool dialect.
- Only function tools and supported native Qwen, Gemma/Unified, DiffusionGemma, GLM, Llama and MiniCPM templates are accepted. Unknown templates fail before generation.
- Legacy `functions` and `function_call` are unsupported. Strict tool schemas follow the rules above.
- Results must refer to preceding unfinished calls; orphaned, duplicate or missing IDs are rejected.

MiniCPM-V 4.6 and MiniCPM5 use distinct XML dialects. MiniCPM5 encodes string parameters containing `<`, `&` or newlines using CDATA. Gemma internally projects dynamic objects into deterministic key/value entries and restores the original objects on output; public Schema and argument shapes stay unchanged.

## Anthropic Messages

`POST /v1/messages` uses `model`, `messages` and `max_tokens`. Requests and nested content blocks use strict fields. Public sampling is finite `temperature` in `[0,1]`, `top_p` in `(0,1]` and positive integer `top_k`. `repetition_penalty` and unknown fields are rejected.

### Structured Outputs

```json
{
  "output_config": {
    "format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "notes": {
            "type": "string"
          }
        },
        "required": [
          "name"
        ],
        "additionalProperties": false
      }
    }
  }
}
```

JSON appears in ordinary text blocks. The Schema subset is shared with Responses, and unsupported fields fail before generation. `auto` permits a call or final JSON, `none` permits final JSON, and `any`/named `tool` permit calls only.
`stop_reason:"max_tokens"` can leave JSON incomplete. Structured Outputs do not allow last-assistant-message prefill. Only `output_config.format` is supported, not obsolete top-level `output_format`.

Thinking can be combined with Structured Outputs: the thinking section is unconstrained, final text uses the JSON grammar, and tool calls use their parameter grammar. The same constraints apply again after replaying tool results, across synchronous/SSE and supported execution paths.

### Extended and adaptive thinking

```json
{
  "thinking": {
    "type": "adaptive",
    "display": "summarized"
  },
  "output_config": {
    "effort": "high"
  },
  "max_tokens": 4096
}
```

Supported native templates accept `disabled`, `enabled` or `adaptive`. Enabled mode requires `budget_tokens >= 1024` and less than `max_tokens`. Adaptive effort accepts `low`, `medium`, `high`, `xhigh`, `max`; Qwen3.8 maps to its low/medium/xhigh tiers and other current templates use a boolean switch. These values are validated but are not a calibrated Claude token budget or quality tier. `max_tokens` remains the hard total limit.

Reasoning appears in a thinking block before text/tool-use blocks. SSE emits `thinking_delta`, `signature_delta` and `content_block_stop`; `usage.output_tokens_details.thinking_tokens` counts native reasoning tokens. History accepts one thinking block before visible assistant content and validates its locally generated integrity signature. Modified blocks return 400.
`display:"omitted"`, redacted thinking and multiple/interleaved thinking blocks are unsupported. Claude-hosted signatures cannot be replayed as local IronMLX history.

### Client tools

```json
{
  "tools": [
    {
      "name": "get_weather",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          }
        },
        "required": [
          "city"
        ],
        "additionalProperties": false
      },
      "strict": true
    }
  ],
  "tool_choice": {
    "type": "auto"
  },
  "max_tokens": 128
}
```

Responses use `tool_use` blocks with `id`, `name`, `input` and `stop_reason:"tool_use"`. Clients execute tools and replay the original assistant block followed immediately by a user message containing matching `tool_result` blocks:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Weather in Tokyo?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_123",
          "name": "get_weather",
          "input": {
            "city": "Tokyo"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_123",
          "content": "Sunny, 26 C",
          "is_error": false
        }
      ]
    }
  ]
}
```

- `tool_choice` accepts `auto`, `any`, named `tool` and `none`; the first three support `disable_parallel_tool_use`. Any requires at least one call; a named tool requires that tool.
- Assistant messages can contain text and multiple calls. User messages can contain multiple results followed by text or images. IDs must be nonempty, unique, correctly ordered and fully paired.
- SSE follows `message_start`, `content_block_*`, `message_delta`, `message_stop`. Tool arguments arrive in `input_json_delta.partial_json`; concatenate before parsing.
- Schema and template limits apply before generation. Only client-defined function tools are supported, not hosted/server tools, MCP, computer use, Web Search or Code Execution.

## Compatibility and validation

See the [API compatibility matrix](api-compatibility-matrix.md) for per-field support and SDK checks. Protocol validation is separate from model quality validation.

## Images

Chat accepts JPEG/PNG/WebP data URLs, Responses accepts its documented image-item shapes, and Messages accepts base64 image sources. Remote HTTP/HTTPS URLs are never fetched. See [Security boundaries](security-boundary.md) for body, count, byte and pixel limits.

## LAN

Use `https://<selected-ip>:<port>` with `Authorization: Bearer <API-Key>` on all LAN routes, including health. Trust the App-exported CA rather than disabling TLS verification.

## Log-level management (loopback only)

`GET /admin/api/log-level` returns `level`, `revision`, `process_id`. A custom CLI `RUST_LOG` filter is represented by null level until replaced; the App requires a canonical initial level.

```json
{
  "level": "DEBUG",
  "expected_revision": 0,
  "expected_process_id": 12345
}
```

Send this to `POST /admin/api/log-level`. Success returns the new snapshot. Stale process/revision returns 409; invalid JSON/levels are rejected and unavailable runtime control returns 503. The route is absent from LAN, even with a valid key.

### App log-setting application

The App reads the backend snapshot, updates the filter with process/revision preconditions, saves only `log_level`, then updates its own filter. While stopped, it saves the setting for the next launch without an HTTP call; startup, recovery, shutdown or another settings transaction blocks changes.
After a lost response or save failure it queries actual state and restores the previous level only while preconditions still match. A changed process/revision or unconfirmed rollback is not displayed as success.
Legacy TRACE/WARN map to ALL/WARNING. Third-party dependency logs are capped at WARNING, or ERROR when selected. Helper launches receive `IRONMLX_LOG_LEVEL`; independent CLI use can retain `RUST_LOG`.
