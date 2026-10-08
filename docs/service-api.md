# Service and management API

[简体中文](zh-CN/service-api.md) · [API reference](api-reference.md)

Reference for shared access conventions, health checks, model discovery and the management endpoints listed below in the App, EnginePool and regular CLI services. Inference request and response contracts belong to their dedicated references, listed in the [API reference](api-reference.md).

## Service address, authentication and conventions

The App defaults to `http://127.0.0.1:9068`; independent CLI serving defaults to port 8080. Use the actual configured endpoint. Loopback access needs no API key; SDKs requiring a nonempty key may use `local`.

LAN uses `https://<selected-ip>:<port>` and `Authorization: Bearer <API-Key>` on every route, including health checks. Trust the App-exported CA. JSON requests use `Content-Type: application/json`; image editing uses multipart data. See [Security boundary](security-boundary.md) for configuration and resource limits.

Standalone `ironmlx serve-systemone` uses its own port and authentication rules: `/v1/*` requires an API key even on loopback, while `/healthz` needs no credentials. See [System One API](laya-systemone-api.md#optional-standalone-cli). Missing or invalid LAN credentials return 401; error formats are defined by the corresponding endpoint.

## Service and management endpoints

| Endpoint | Purpose |
| --- | --- |
| [`GET /health`](#health-and-model-discovery) | HTTP responsiveness |
| [`GET /healthz`](#health-and-model-discovery) | Runtime health |
| [`GET /v1/models`](#health-and-model-discovery) | Model discovery |
| [`POST /v1/models/{model_id}/load`](engine-pool.md#http-apis) | Load a model |
| [`POST /v1/models/{model_id}/unload`](engine-pool.md#http-apis) | Unload a model |
| [`GET /admin/api/models/loaded`](#app-decision-runtime-metrics) | Loaded models and metrics |
| [`GET /admin/api/log-level`](#log-level-management-loopback-only) | Read the log level |
| [`POST /admin/api/log-level`](#log-level-management-loopback-only) | Update the log level |

## Health and model discovery

### GET /health

No request body. Success is HTTP 200 with plain text `ok`; this confirms HTTP responsiveness only.

### GET /healthz

No request body. Success is HTTP 200 JSON. Common keys are `status`, `degraded_reasons`, `version` and `mode`; model, scheduler, memory and pool details vary by mode. Treat unknown observational fields as optional. Inference errors still determine whether a specific request can run.

`memory.free_ram_bytes` is raw OS free memory for observation. `available_ram_bytes` includes reclaimable memory using the governor's accounting. `process_governor.pressure_level` determines memory health; `degraded_reasons` identifies queue, cache, pressure, telemetry or backpressure causes.

### GET /v1/models

No request body. Success is HTTP 200 JSON with `object:"list"` and `data`, an array of model objects. App and EnginePool also expose a TypeSafe `models` array for decision discovery; see [System One](laya-systemone-api.md). Registered models can be listed before loading.

| Field | Type | Meaning |
| --- | --- | --- |
| `data[].id` | string | Public model ID used in requests. |
| `data[].object` | string | model |
| `data[].created` | integer | Unix timestamp; pool entries currently use 0. |
| `data[].owned_by` | string | ironmlx |
| `data[].load_policy` | string, optional | Pool preload/lazy/disabled policy. |
| `data[].state` | string, optional | Pool runtime state, such as unloaded/loading/loaded/draining/failed/missing/disabled. |
| `data[].context_window` | integer, optional | Effective total context capacity. |
| `data[].max_output_tokens` | integer, optional | Output ceiling before subtracting input tokens. |

When reliably known, each causal model entry also includes these IronMLX extension fields:

- `context_window`: effective total token capacity, the smaller of the model's context capacity and the deployment's `max_cache_cap` (App **MAX CONTEXT TOKENS**).
- `max_output_tokens`: output-token ceiling, including reasoning, answer text and tool calls. Causal serving currently has no separate output-only hard cap, so this equals `context_window`. For each request, the output budget must still fit within `context_window - input_tokens`, including chat-template and multimodal input tokens. This is **not** the App's **MAX OUTPUT TOKENS** setting: that setting supplies a default Responses budget only when the request omits it.

For example, a model with a 262144-token context deployed with `max_cache_cap=65536` advertises `context_window:65536` and `max_output_tokens:65536`, even if its default output budget is 256. Clients must subtract input usage before choosing an output budget; the advertised ceiling does not guarantee that amount for a nonempty prompt or guarantee a complete answer. Admission and memory checks still apply.

Loaded models use their actual admission capacity. Unloaded causal models use explicit checkpoint metadata and the registered scheduler configuration without loading weights. Missing, invalid or unsupported capacity metadata is omitted, not returned as zero or inferred from generation defaults. Audio and DiffusionGemma currently omit both fields. App DFlash2 discovery reports the loaded target's effective capacity, never the draft model's capacity. These are optional extensions; clients should retain fallbacks when absent.

## Management endpoints

### App decision runtime metrics

`GET /admin/api/models/loaded` · App only. HTTP 200 returns an array of loaded-model objects. The fields below belong to decision_metrics on an entry with runtime_kind:decision; other runtime kinds omit this object.

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

### App embedding runtime metrics

Loaded entries with `runtime_kind:embedding` expose `embedding_metrics` through
`GET /admin/api/models/loaded` and the App's `/healthz` model entries. They report
vector request counts, recent latency, input-token throughput and vector throughput.
Field meanings, measurement windows and reset behavior are defined in the
[Embedding API](text-embeddings.md#runtime-status).

### Log-level management (loopback only)

`GET /admin/api/log-level` returns HTTP 200 JSON with level, revision and process_id. level is null when the CLI uses a custom RUST_LOG filter. Example:

```json
{
  "level": "INFO",
  "revision": 0,
  "process_id": 12345
}
```

`POST /admin/api/log-level` accepts the following JSON fields:

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `level` | string | Yes | — | ALL/DEBUG/INFO/WARNING/ERROR; TRACE/WARN are legacy aliases. |
| `expected_revision` | integer | Yes | — | Unsigned revision from the latest GET snapshot. |
| `expected_process_id` | integer | Yes | — | Process ID from the latest GET snapshot. |

```json
{
  "level": "DEBUG",
  "expected_revision": 0,
  "expected_process_id": 12345
}
```

Use the actual values returned by GET. Success is HTTP 200 with the updated snapshot (revision increments by one). Stale process/revision returns 409 with plain text log_control_changed. Unavailable control returns 503 with plain text log_control_unavailable. Invalid JSON/field types use the JSON extractor’s 4xx rejection; they are not inference error envelopes.

Both routes exist only on the loopback listener, even with a valid LAN key. App setting application and rollback are documented in [Build from source](building-from-source.md#app-log-setting-application).

## Related references

- [Developer guide](developer-guide.md#api-integration): choose an API and make the first request.
- [Text and vision API](text-vision-api.md): Responses, Chat Completions and Messages.
- [Embedding API](text-embeddings.md): text, image, audio and combined vectors.
- [Speech synthesis API](audio-speech-api.md): audio output and voice management.
- [Image generation API](image-generation-api.md): text-to-image and single-image conditional editing.
- [System One API](laya-systemone-api.md): choice, score and true/false probability.
