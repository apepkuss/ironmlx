# HTTP API

The App serves the API at `http://127.0.0.1:9068` by default. A direct CLI
server uses port 8080 unless overridden; always use the endpoint shown by the
Dashboard or the actual startup arguments.

## Runtime and health

Only one `ironmlx serve` backend may run for a macOS user. A second instance exits
with `ironmlx_instance_already_running`. The lock is acquired before MLX,
metallib, or model initialization and is released by the operating system on
normal exit, crash, or SIGKILL.

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` means only that the HTTP process responds. `/healthz` returns a JSON
snapshot containing product version, models, scheduler, cache, memory, and
degraded-state details. `GET /v1/models` is an OpenAI-compatible list of
registered models, including models that are registered but not currently loaded.

## Error contract

Chat Completions and Responses use an OpenAI-style error envelope:

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

Anthropic Messages uses an Anthropic-style envelope and a matching `request-id`
header. `error.code` is a stable IronMLX machine-readable extension. Clients
should classify by HTTP status and `error.type`, then use `error.code` for the
specific cause.

| HTTP status | Stable codes | Meaning |
| ---: | --- | --- |
| 400 | `invalid_json` and field/constraint codes | Invalid JSON, fields, sampling, or output constraints |
| 413 | `request_body_too_large` | Request body exceeds 32 MiB |
| 413 | `request_token_capacity_exceeded` | Input plus output budget exceeds model context |
| 503 | `scheduler_queue_full`, `scheduler_unavailable`, `scheduler_reply_lost` | Scheduler temporarily unavailable |
| 503 | `memory_budget_exceeded`, `memory_pressure`, `prefill_peak_unsafe`, `vision_prefill_peak_unsafe` | Memory governor or storage backpressure |
| 503 | `engine_unavailable`, `diffusion_lane_overloaded` | Model engine temporarily unavailable |
| 500 | `generation_error` and internal codes | Unexpected server error |

Retryable 503 responses include `Retry-After: 5`. Messages maps temporary
overload to `overloaded_error`; 413 uses `request_too_large` in the Anthropic
envelope.

## OpenAI Chat Completions

Use `POST /v1/chat/completions` with an OpenAI-compatible `model` and `messages`.
Set `stream: true` for SSE. `temperature` and `top_p` are public sampling
fields. Tool calls are emitted as structured function calls; the client, not
IronMLX, executes the function and returns the result in the next request.

## OpenAI Responses

`POST /v1/responses` is the recommended interface for local Agent clients.
Responses is stateless: send the complete typed item history on every request;
IronMLX does not persist responses or conversations. Streaming uses typed SSE
events, including reasoning and function-call items when the selected model's
native template supports them.

## Anthropic Messages

Use `POST /v1/messages`. Non-streaming errors use the Anthropic envelope; the
streaming form emits Anthropic SSE events. `output_config.effort` converges to
the supported low/medium/xhigh reasoning tiers. Anthropic Messages additionally
accepts public `top_k`; `repetition_penalty` is not a public protocol field.

## Images and limits

Image input is accepted only as controlled base64 data URLs for JPEG, PNG, or
WebP. IronMLX never fetches HTTP/HTTPS image URLs. Requests are bounded to a
32 MiB body, 8 images, 10 MiB decoded bytes per image, 24 MiB decoded bytes in
total, 8192-pixel width/height, and the documented text/pixel/decoder budgets.

## Tools and reasoning

Tool support means structured function-call generation and history replay for
OpenAI or Anthropic protocols. IronMLX does not execute tools. Reasoning is
enabled only when the model type and native chat template match the supported
contract; it is not inferred from similar marker text. There is no independent
reasoning summary, refusal, audio-output, or image-output channel for ordinary
text models.

## Model and administrative routes

`GET /v1/models` is exposed by EnginePool and App-daemon topologies. Model
management routes under `/admin/api/models/*` are App-daemon-only. App model
search/download uses the upstream provider and user credentials; model rights
remain the user's responsibility (see [Model rights boundary](model-license-boundary.md)).

Qwen3.8 DFlash2 runs in a separate actor and supports text Chat/Responses/
Messages, greedy or exact sampling, and bounded request concurrency. It cannot
be mixed with MTP, Prompt Lookup, KV quantization, paged/SSD prefix cache, or
active-KV offload. See the [DFlash2 server API](dflash2-server-api.md) and the
[supported model matrix](supported-models.md).

## Cancellation and streaming

After an SSE response starts, a client disconnect publishes a protocol-neutral
cancellation signal. The encoder stops consuming generation events and does not
emit a synthetic terminal event after the disconnect. Cancellation takes effect
at the next safe scheduler/token/block boundary after the current Metal forward;
it does not forcibly interrupt a device operation.

Additional endpoint examples, compatibility details, request schemas, and
model-specific fields are also available in the [Simplified Chinese API reference](zh-CN/api.md).
