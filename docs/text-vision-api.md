# Text and vision API

[简体中文](zh-CN/text-vision-api.md) · [API reference](api-reference.md) · [Service and management API](service-api.md)

Reference for OpenAI Responses, Chat Completions and Anthropic Messages requests, responses, streaming events, tools and structured outputs. Vision means image input and understanding; image generation and editing have a separate [Image generation API](image-generation-api.md). Service addresses, authentication, health checks, model discovery and management endpoints are defined in [Service and management API](service-api.md).

## Reading conventions

“—” means omitted. Sampling values use the loaded model’s defaults when omitted; they are not fixed upstream-service defaults. Only Responses applies a configured model output budget before falling back to 256. Explicit budgets are never enlarged. JSON feature examples marked as fragments must be merged into a request containing its required fields.

## API compatibility and limits

IronMLX provides compatible OpenAI Responses, Chat Completions and Anthropic Messages interfaces. Supported fields and limits are defined below. IronMLX does not execute client tools or provide hosted tools, conversation storage or background response jobs.

Inference requests with unknown fields or unsupported field shapes return 400; unsupported capabilities are not silently ignored.

Fixed CLI services select their model at startup. App and EnginePool resolve the request `model` or an unambiguous default. Support for vision, thinking and tools depends on the model and its native template; see [Supported models](supported-models.md). [DFlash2 configuration](dflash2-server-api.md) describes its text-only serving and sampling constraints.

## Endpoint index

| Endpoint | Purpose / availability |
| --- | --- |
| [`POST /v1/responses`](#openai-responses) | Text/vision models supporting Responses |
| [`POST /v1/chat/completions`](#openai-chat-completions) | Text/vision models supporting Chat |
| [`POST /v1/messages`](#anthropic-messages) | Text/vision models supporting Messages |

## OpenAI Responses

`POST /v1/responses` · `application/json`

### Responses request fields

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | string | No | Service default | Model ID from /v1/models; required when no unambiguous default exists. |
| `input` | string / array | Yes | — | Text or typed message, reasoning, function_call and function_call_output history. |
| `instructions` | string | No | — | System/developer instructions for this request. |
| `tools` | array | No | `[]` | Client function tools or supported namespace groups. |
| `tool_choice` | string / object | No | `auto` | auto, none, required, or a named function; namespace subfunctions cannot be forced. |
| `parallel_tool_calls` | boolean | No | `true` | false restricts a turn to one call; requires tools when supplied. |
| `text` | object | No | `{"format":{"type":"text"}}` | text.format selects text, json_object or json_schema. |
| `stream` | boolean | No | `false` | true returns typed SSE events. |
| `stream_options` | object | No | — | Must be an object; does not enable Chat-style usage chunks. |
| `max_output_tokens` | integer | No | Model default or 256 | Positive total output budget, including reasoning and tools; must fit context. |
| `temperature` | number | No | Model sampling default | Finite value in [0,2]. |
| `top_p` | number | No | Model sampling default | Finite value in (0,1]. |
| `reasoning` | object | No | `effort: none` | effort: none/minimal/low/medium/high/xhigh/max; summary: none/auto. |
| `store` | boolean | No | `false` | Only false is supported; no response storage. |
| `background` | boolean | No | `false` | Only false is supported; no background jobs. |
| `include` | array of strings | No | `[]` | Only reasoning.encrypted_content is accepted; no encrypted content is generated. |
| `prompt_cache_key` | string | No | — | 1–256 bytes; validated only, without hosted-platform cache semantics. |
| `client_metadata` | object | No | — | Validated as an object; no hosted-platform semantics. |
| `metadata` | object | No | — | Validated as an object; not persistent conversation metadata. |
| `service_tier` | string | No | `default` | auto/default for foreground; flex for background priority in ordinary causal and DFlash2 serving. See [Request priority](#request-priority). |
| `truncation` | string | No | `disabled` | Only disabled; input is not automatically truncated. |

Responses uses the configured model output budget when max_output_tokens is omitted, then falls back to 256.

### Request example

```bash
curl http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "input": "Hello", "store": false, "max_output_tokens": 128, "stream": false}'
```

### Response

Success returns HTTP 200 JSON. Example IDs, timestamps and token counts are illustrative.

| Field | Type | Meaning |
| --- | --- | --- |
| `id / object / created_at` | string / string / integer | Response ID, response object type and Unix timestamp. |
| `status` | string | completed or incomplete for successful final responses; in_progress during streaming. |
| `output` | array | Typed message, reasoning or function_call items; inspect type before parsing. |
| `output[].content` | array, conditional | Message output_text or reasoning reasoning_text blocks. |
| `output[].arguments / call_id / name` | string, conditional | Function-call JSON arguments string, call correlation ID and public name; namespace is optional. |
| `usage` | object or null | input_tokens, output_tokens, total_tokens and token-detail objects; final usage is populated. |
| `incomplete_details` | object or null | reason:max_output_tokens when the output limit is reached. |
| `error` | object or null | null on success; streaming failure is described by response.failed. |
| `model` | string | Effective model ID. |
| `store / background / previous_response_id` | boolean / boolean / null | false / false / null: no server-side response state. |
| `reasoning / text / tools / tool_choice` | object / object / array / string or object | Effective reasoning, format and tool configuration. |
| `max_output_tokens / parallel_tool_calls` | integer / boolean | Effective output budget and parallel-call setting. |
| `temperature / top_p` | number or null | Explicit requested values, or null when omitted. |
| `instructions` | string or null | Requested instructions. |
| `service_tier / truncation` | string | Effective tier: default or flex; truncation: disabled. |

```json
{
  "id": "resp_example",
  "object": "response",
  "created_at": 0,
  "status": "completed",
  "background": false,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": 128,
  "model": "your-model-id",
  "output": [
    {
      "type": "message",
      "id": "msg_example",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "annotations": [],
          "logprobs": [],
          "text": "Hello!"
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": "none",
    "summary": null
  },
  "service_tier": "default",
  "store": false,
  "temperature": null,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": null,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 8,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 2,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 10
  }
}
```

### Streaming

Set `stream:true`. Events have `event:` names and JSON `data` containing `type` and increasing `sequence_number`. The lifecycle is `response.created` → item/content events and deltas → `response.completed`, `response.incomplete` or `response.failed`. There is no `[DONE]` sentinel. The final success event contains the response object and usage.

Text-event excerpt (other lifecycle events omitted):

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"content_index":0,"delta":"Hello!","item_id":"msg_example","logprobs":[],"output_index":0}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":4,"content_index":0,"item_id":"msg_example","logprobs":[],"output_index":0,"text":"Hello!"}
```

Tool argument deltas use `response.function_call_arguments.delta` / `.done`; concatenate `delta` before parsing JSON. Reasoning uses `response.reasoning_text.delta` / `.done`. Retain completed items from `response.output_item.done` for history replay.

### Reasoning

`reasoning.effort` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; missing/null reasoning or missing effort means `none`. Qwen3.8 maps minimal/low to low, medium to medium and high/xhigh/max to xhigh; other supported templates treat non-none effort as an enable switch. These are not calibrated thinking-token budgets. `summary` accepts none/auto but does not produce an independent summary.

Enabled native reasoning is a separate `reasoning` item with plaintext `reasoning_text`. Encrypted-only history returns 400. An orphan plaintext reasoning item without a following assistant message/call is skipped during replay. Preserve complete normally paired items and their status.

Reasoning items begin with `in_progress` and finish with `completed` or `incomplete`; only a still-open reasoning channel hitting the limit marks that item incomplete. Later answer truncation can make the response incomplete while reasoning remains completed. See [Output budgets](#output-budgets) for the shared budget policy.

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

The client executes `function_call` items and appends both the original call and a `function_call_output` with the same `call_id` to complete history. Retain the corresponding tool definitions on subsequent requests. Text, supported image message items and call/output items work in synchronous and SSE modes.

### Structured Outputs

JSON mode uses `{"text":{"format":{"type":"json_object"}}}`. Schema mode uses:

Request-field fragment; merge into a complete request with its required fields:

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

With tools and an output format together, `none` permits only final JSON, `required` or a named function permits only calls, and `auto` permits calls or Schema-conforming final JSON. The combined constraint applies across supported tool dialects and ordinary, scheduled, speculative and DiffusionGemma generation.

The JSON Schema subset is defined in [Common constraints](#json-schema).

### Limits

- `store` is omitted or false; true is rejected. `previous_response_id`, conversations, background execution and response retrieve/delete/cancel APIs are unsupported.
- Client functions and the supported `type:"namespace"` function groups are accepted. Namespaces compile to a bounded dispatcher and restore public namespace/function/argument shapes in output and replay. Named namespace subfunctions cannot be forced through `tool_choice`; use `auto` or `required`.
- Dynamic parameters outside the Schema subset are allowed only for non-strict namespace subfunctions using a JSON argument envelope; strict mode never downgrades.
- Hosted Web/File Search, MCP, Code Interpreter and custom/freeform tools are unsupported.
- No persisted reasoning, independent summary/refusal items, OpenAI file IDs, audio input/output, image output or image function results are provided.
- Public sampling is finite `temperature` in `[0,2]` and `top_p` in `(0,1]`. `top_k`, `repetition_penalty` and unknown fields are rejected.

## OpenAI Chat Completions

`POST /v1/chat/completions` · `application/json`

### Chat request fields

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | string | No | Service default | Model ID from /v1/models; required when no unambiguous default exists. |
| `messages` | array | Yes | — | Message history, including supported content blocks and tool results. |
| `tools` | array | No | — | Only type:function; see function tools below. |
| `tool_choice` | string / object | No | auto with tools | auto/none/required or a named function object. |
| `parallel_tool_calls` | boolean | No | `true` | false limits a turn to one call; supplying it requires tools. |
| `response_format` | object | No | `{"type":"text"}` | text, json_object or json_schema with nested json_schema. |
| `stream` | boolean | No | `false` | true returns Chat completion chunks over SSE. |
| `stream_options` | object | No | — | Only include_usage:boolean; defaults to false. |
| `max_tokens` | integer | No | `256` | Total output budget including reasoning and tool calls; context limit applies. |
| `service_tier` | string | No | `default` | auto/default for foreground; flex for background priority in ordinary causal and DFlash2 serving. See [Request priority](#request-priority). |
| `temperature` | number | No | Model sampling default | Finite value in [0,2]. |
| `top_p` | number | No | Model sampling default | Finite value in (0,1]. |
| `seed` | integer | No | — | IronMLX extension: unsigned request seed. |
| `ignore_eos` | boolean | No | `false` | IronMLX extension: continue to max_tokens despite EOS, for controlled measurements. |
| `reasoning_effort` | string | No | Native template default | Qwen3.8 low/medium/xhigh; requires matching template. |
| `chat_template_kwargs` | object | No | — | IronMLX extension: supported template parameters such as enable_thinking. |

### Request example

```bash
curl http://127.0.0.1:9068/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

### Response

Success returns HTTP 200 JSON. Example IDs, timestamps and token counts are illustrative.

| Field | Type | Meaning |
| --- | --- | --- |
| `id / object / created / model` | string / string / integer / string | Response ID, chat.completion type, Unix timestamp and model ID. |
| `choices` | array | Completion choices, each with index, message and finish_reason. |
| `choices[].message` | object | role:assistant, content:string or null, and optional tool_calls. |
| `choices[].message.tool_calls` | array, optional | Each call has id, type:function and function.name/arguments; arguments is a JSON string. |
| `choices[].finish_reason` | string | stop, length or tool_calls; length can truncate JSON or text. |
| `usage` | object | prompt_tokens, completion_tokens and total_tokens. |

```json
{
  "id": "chatcmpl_example",
  "object": "chat.completion",
  "created": 0,
  "model": "your-model-id",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 2,
    "total_tokens": 10
  }
}
```

### Streaming

Set `stream:true`. Each `data:` frame is a `chat.completion.chunk`; `choices[].delta` supplies the role, text or tool-call fragments, followed by a finish reason and `data: [DONE]`. A text stream looks like:

```text
data: {"id":"chatcmpl_example","object":"chat.completion.chunk","created":0,"model":"your-model-id","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl_example","object":"chat.completion.chunk","created":0,"model":"your-model-id","choices":[{"index":0,"delta":{"content":"Hello!"}}]}

data: {"id":"chatcmpl_example","object":"chat.completion.chunk","created":0,"model":"your-model-id","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

With `stream_options.include_usage:true`, an additional final usage chunk has `choices:[]` and `usage` before `[DONE]`. Tool argument strings can span multiple deltas; correlate calls by index/id and concatenate before JSON parsing.

### Structured Outputs

JSON mode uses `{"response_format":{"type":"json_object"}}`. Schema mode nests the definition under `response_format.json_schema`, unlike Responses:

Request-field fragment; merge into a complete request with its required fields:

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

Request-field fragment; merge into a complete request with its required fields:

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

Tool-call history requires the matching nonempty `tools` array. The client appends the original assistant call message and each `role:"tool"` result with its `tool_call_id`. `function.arguments` is a JSON string. SSE arguments can span multiple deltas identified by stable `index` and `id`; the finish reason is `tool_calls`.

- `tool_choice` accepts `auto`, `none`, `required` or `{"type":"function","function":{"name":"..."}}`.
- `parallel_tool_calls` defaults to true; false limits the turn to one call. Llama 3.1/3.2 native custom tools always allow only one call and exclude its built-in-tool dialect.
- Only function tools and supported native Qwen, Gemma/Unified, DiffusionGemma, GLM, Llama and MiniCPM templates are accepted. Unknown templates fail before generation.
- Legacy `functions` and `function_call` are unsupported. Strict tool schemas follow the rules above.
- Results must refer to preceding unfinished calls; orphaned, duplicate or missing IDs are rejected.

### Limits

Only function tools and supported native templates are accepted. Legacy functions/function_call, top_k and repetition_penalty are rejected. Qwen3.8 reasoning_effort accepts low/medium/xhigh (native default xhigh); chat_template_kwargs.enable_thinking:false disables thinking and preserve_thinking:false omits historical reasoning_content. Use Responses/Messages for independent reasoning items/blocks.

## Anthropic Messages

`POST /v1/messages` · `application/json`

### Messages request fields

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | string | No | Service default | Model ID from /v1/models; required when no unambiguous default exists. |
| `messages` | array | Yes | — | user/assistant history with text, base64 images, tools and local thinking blocks. |
| `system` | string / array | No | — | System text or supported text blocks. |
| `tools` | array | No | — | Client tools with name and input_schema. |
| `tool_choice` | object | No | auto with tools | type:auto/any/tool/none; supported choices accept disable_parallel_tool_use. |
| `output_config` | object | No | — | format uses type:json_schema and schema; effort requires enabled/adaptive thinking. |
| `thinking` | object | No | Disabled | type:disabled/enabled/adaptive; enabled requires budget_tokens >=1024 and <max_tokens. |
| `max_tokens` | integer | No | `256` | Total output budget including thinking and tool calls; context limit applies. |
| `stream` | boolean | No | `false` | true returns native Messages SSE events. |
| `temperature` | number | No | Model sampling default | Finite value in [0,1]. |
| `top_p` | number | No | Model sampling default | Finite value in (0,1]. |
| `top_k` | integer | No | Model sampling default | Positive integer. |

### Request example

```bash
curl http://127.0.0.1:9068/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

### Response

Success returns HTTP 200 JSON. Example IDs, timestamps and token counts are illustrative.

| Field | Type | Meaning |
| --- | --- | --- |
| `id / type / role / model` | string | Message ID, message type, assistant role and model ID. |
| `content` | array | Typed text, thinking or tool_use blocks; preserve their order. |
| `content[].thinking / signature` | string, conditional | Local thinking text and integrity signature; replay both unchanged. |
| `content[].id / name / input` | string / string / object, conditional | tool_use ID, tool name and parsed JSON arguments. |
| `stop_reason` | string or null | end_turn, max_tokens or tool_use; null while streaming is unfinished. |
| `stop_sequence` | null | No supported stop-sequence field. |
| `usage` | object | input_tokens, output_tokens and optional output_tokens_details.thinking_tokens. |

```json
{
  "id": "msg_example",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello!"
    }
  ],
  "model": "your-model-id",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 8,
    "output_tokens": 2
  }
}
```

### Streaming

Set `stream:true`. Native event order is message_start → content_block_start/delta/stop groups → message_delta → message_stop. Content indices identify blocks; tool arguments arrive as input_json_delta.partial_json. Thinking blocks use thinking_delta and signature_delta. A complete plain-text lifecycle example:

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_example","type":"message","role":"assistant","content":[],"model":"your-model-id","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":8,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
```

### Structured Outputs

Request-field fragment; merge into a complete request with its required fields:

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

Request-field fragment; merge into a complete request with its required fields:

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

Request-field fragment; merge into a complete request with its required fields:

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

Request-history fragment; retain the tool definitions in the complete request:

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

### Limits

No legacy output_format, repetition_penalty, hosted/server tools, MCP, computer use, Web Search or Code Execution. Thinking history must use the locally generated signature; Claude-hosted signatures and redacted/omitted/interleaved thinking cannot be replayed.

## Common constraints

### Request priority

Chat Completions and Responses accept `service_tier:auto/default` for foreground
priority. `flex` runs at background priority and may be preempted by foreground
work in ordinary causal and DFlash2 serving. Other tier values return 400.
Flex selects scheduling priority; it does not create a background response job.
Responses reports the effective tier as `default` or `flex`.

### Image inputs

Image inputs accept base64 JPEG, PNG or WebP only; remote HTTP/HTTPS URLs are not fetched and OpenAI file IDs are unsupported. Each protocol uses a different nested shape:

| Protocol | Location | Image content block |
| --- | --- | --- |
| Chat | `messages[].content[]` | `{"type":"image_url","image_url":{"url":"data:image/png;base64,<Base64>"}}` |
| Responses | `input[].content[]` | `{"type":"input_image","image_url":"data:image/png;base64,<Base64>"}` |
| Messages | `messages[].content[]` | `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<Base64>"}}` |

Replace <Base64> with actual image data and include the block alongside text in the message content array. A vision-capable model is required; see [Security boundary](security-boundary.md) for count, byte and pixel limits.

### JSON Schema

Supported types: object, array, string, number, integer, boolean, null and nullable type arrays. Supported keywords: properties, required, items, enum, const, anyOf, minItems/maxItems, minLength/maxLength, minimum/maximum and exclusiveMinimum/exclusiveMaximum. Maximum nesting depth is 8.

strict:true requires additionalProperties:false and every property in required at each object level. Top-level tool parameter objects must be closed. Non-strict tools can use nested additionalProperties:true or a supported value schema. Unsupported keywords return 400 before generation; strict constraints do not silently downgrade.

### Output budgets

Input (including templates and images) plus requested output must fit the effective context capacity; excess returns 413 request_token_capacity_exceeded. max_output_tokens/max_tokens include reasoning, answer text and tool calls.

For causal Qwen3.5/3.6/3.8 and Gemma4/Unified with exact supported reasoning templates, enabled reasoning reserves min(floor(total_output_budget/4),1024) tokens for the answer/tool call plus native framing space. Remaining tokens are available for reasoning. The same policy applies across the three protocols. It does not enlarge the total budget or retry a request. Disabled/unsupported reasoning, DiffusionGemma and budgets too small for the framing retain their existing behavior.

A completed thinking channel does not guarantee a complete answer. Check Responses status/incomplete_details, Chat finish_reason:length or Messages stop_reason:max_tokens before parsing constrained JSON as complete.

### Streaming and cancellation

Streaming responses use text/event-stream. Parse SSE frames across network-chunk boundaries and assemble deltas before parsing tool JSON. Errors before streaming use the ordinary HTTP envelope; a failure after HTTP 200 is reported in protocol stream events.

Client disconnect cancels streaming generation at the next safe execution boundary and releases request resources; it does not interrupt an in-flight Metal operation. No terminal event is fabricated after disconnect. Cancellation of already running non-streaming generation on client disconnect is not guaranteed.

## Errors

The request-body limit is 32 MiB; exceeding it returns 413 `request_body_too_large`.

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
| 400 | `invalid_json`, `invalid_request`, `invalid_sampling_parameters`, `invalid_response_format`, `invalid_tools`, or other field/constraint errors | Correct the request |
| 401 | `auth_invalid` | Check the LAN Bearer key |
| 404 | `model_not_found` | Select an enabled registered model |
| 413 | `request_body_too_large`, `request_token_capacity_exceeded` | Reduce body size or token budget; context errors include capacity details |
| 503 | `scheduler_queue_full`, `scheduler_unavailable`, `scheduler_reply_lost`, `engine_unavailable`, `diffusion_lane_overloaded` | `Retry-After: 5` |
| 503 | `memory_budget_exceeded`, `memory_pressure`, `prefill_peak_unsafe`, `vision_prefill_peak_unsafe`, `cold_materialization_unsafe`, `prefix_store_backpressure` | `Retry-After: 5` |
| 500 | `generation_error` or internal failure | Inspect diagnostics |

Retryable 503 errors return JSON; Messages uses `overloaded_error` for overload. Messages 413 errors use `request_too_large`.

These envelopes apply to inference and LAN authentication for the three text and vision protocols on this page. Speech, image generation and System One define errors in their own references. Local management success/error formats are documented in [Service and management API](service-api.md#management-endpoints).

## Related references

- [Service and management API](service-api.md): shared access conventions, health checks, model discovery and management endpoints.
- [Speech synthesis API](audio-speech-api.md): WAV/PCM, voice management and reference recordings.
- [Image generation API](image-generation-api.md): text-to-image and single-image conditional editing; no masked inpainting, image variations or Anthropic image-generation API.
- [System One API](laya-systemone-api.md): typed decision requests and model discovery.
- [SDK compatibility checks](building-from-source.md#sdk-compatibility-checks): protocol and client SDK validation.
- Agent configuration: [Hermes Agent](hermes-agent.md), [oh-my-pi](oh-my-pi.md), [DeepSeek Harness](dsh.md).
