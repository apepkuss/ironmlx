# Service and management API

[简体中文](zh-CN/service-api.md) · [API reference](api-reference.md)

Reference for shared access conventions, health checks, model discovery and the management endpoints listed below in the App, EnginePool and regular CLI services. Inference request and response contracts belong to their dedicated references, listed in the [API reference](api-reference.md).

## Service address, authentication and conventions

The App defaults to `http://127.0.0.1:9068`; independent CLI serving defaults to port 8080. Use the actual configured endpoint. Loopback access needs no API key; SDKs requiring a nonempty key may use `local`.

LAN uses `https://<selected-ip>:<port>` and `Authorization: Bearer <API-Key>` on every route, including health checks. Trust the App-exported CA. JSON requests use `Content-Type: application/json`; image editing uses multipart data. See [Security boundary](security-boundary.md) for configuration and resource limits.

Standalone `ironmlx serve-systemone` uses its own port and authentication rules: `/v1/*`, including `/v1/systemone` and `/v1/decisions`, requires an API key even on loopback, while `/healthz` needs no credentials. See [System One API](laya-systemone-api.md#optional-standalone-cli). Missing or invalid LAN credentials return 401; error formats are defined by the corresponding endpoint.

## Service and management endpoints

| Endpoint | Purpose |
| --- | --- |
| [`GET /health`](#health-and-model-discovery) | HTTP responsiveness |
| [`GET /healthz`](#health-and-model-discovery) | Runtime health |
| [`GET /v1/models`](#health-and-model-discovery) | Model discovery |
| [`POST /v1/models/{model_id}/load`](engine-pool.md#http-apis) | Load a model |
| [`POST /v1/models/{model_id}/unload`](engine-pool.md#http-apis) | Unload a model |
| [`GET /admin/api/models/loaded`](#app-decision-runtime-metrics) | Loaded models and metrics |
| [`POST /admin/api/models/load`](#model-loads-app) | Load a model in the App |
| [`GET /admin/api/models/loads`](#model-loads-app) | Load progress |
| [`POST /admin/api/models/loads/{load_id}/cancel`](#model-loads-app) | Cancel a load |
| [`GET /admin/api/log-level`](#log-level-management-loopback-only) | Read the log level |
| [`POST /admin/api/log-level`](#log-level-management-loopback-only) | Update the log level |

## Health and model discovery

### GET /health

No request body. Success is HTTP 200 with plain text `ok`; this confirms HTTP responsiveness only.

### GET /healthz

No request body. Success is HTTP 200 JSON. Common keys are `status`, `degraded_reasons`, `version` and `mode`; model, scheduler, memory and pool details vary by mode. Treat unknown observational fields as optional. Inference errors still determine whether a specific request can run.

A DFlash2 server (`ironmlx serve` with `--dflash2-model-dir`, as the App runs a DFlash2 model) accepts connections while it loads its model at startup. Until the model is ready, `/healthz` returns HTTP 503 with `{"status":"starting"}`, `GET /admin/api/models/loads` reports the load, and every other route returns HTTP 503 with code `backend_starting` and `Retry-After: 5`. Treat any non-2xx `/healthz` as not ready.

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
| `data[].max_output_tokens` | integer, optional | Suggested client output budget; not an output-only hard limit. |

When reliably known, each causal model entry also includes these IronMLX extension fields:

- `context_window`: effective total token capacity, the smaller of the model's context capacity and the deployment's `max_cache_cap` (App **MAX CONTEXT TOKENS**).
- `max_output_tokens`: suggested client output budget, including reasoning, answer text and tool calls. It uses the model's configured **MAX OUTPUT TOKENS** when present, otherwise 4096, and is capped at half the effective context window (at least one token). It is not an independent output-only hard limit. Explicit request budgets may exceed this suggestion if they fit the remaining context.

For example, a model deployed with `context_window:65536` and no configured output budget advertises `max_output_tokens:4096`; configuring **MAX OUTPUT TOKENS** to 8192 advertises 8192. Chat Completions and Responses use the configured budget when requests omit it, falling back to 256 when no model default exists. Clients must still account for input usage; admission and memory checks apply.

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

### Loaded model scheduler configuration

Entries of `GET /admin/api/models/loaded` include `scheduler_profile`, the scheduler configuration the model runs with: `profile` (a dedicated scheduler profile), `default` (the default configuration) or `default_settings_mismatch` (the default configuration, because this Mac's profiles for the model were made for other runtime settings). Models that do not use the text-generation scheduler, such as audio, image-generation, decision and DiffusionGemma models, omit the field. See [Scheduler performance calibration](scheduler-profile-v5.md#app-display).

### Model loads (App)

These routes are App only. Like the other `/admin/api/models` routes, and unlike [log-level management](#log-level-management-loopback-only), they are also served on the LAN listener: no key on loopback, `Authorization: Bearer <API-Key>` on LAN. A DFlash2 server also answers `GET /admin/api/models/loads`, including during its startup load, but has no load or cancel route.

`POST /admin/api/models/load` loads a model; its model fields are shown in the [audio resource example](audio-speech-api.md#register-local-resources). Two optional fields control load tracking:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `load_id` | string | Generated UUID | ID for this load: 1–128 letters, digits, `-`, `_` or `.`. An invalid ID returns 400 `invalid_load_id`; an ID still listed by `GET /admin/api/models/loads` returns 409 `load_id_in_use`. |
| `async` | boolean | `false` | Return HTTP 202 with `{"load_id", "state"}` as soon as the load is registered, instead of waiting for it to end. |

Without `async`, the response arrives when the load ends: HTTP 200 with `success`, `status` (such as `loaded` or `reloaded`), `loaded_models` and `load_id`. A request for a model that is already loading with the same configuration joins that load, and the response carries that load's `load_id`, which can differ from the one sent; a different configuration returns 409 `load_in_progress` with the running load's `load_id`. With `async`, a request that needs no load, or whose load ends immediately, gets the same response as without `async`. Failures return `success:false`, `code` and `error` with the classified status: 400 for MTP and model-capability codes, 409 for `load_cancelled` and `load_in_progress`, and 503 for memory and capacity codes. The codes are listed under [inference errors](text-vision-api.md#errors).

`GET /admin/api/models/loads` returns `{"loads": [...]}`: loads in progress and loads that ended within the last 30 seconds.

| Field | Meaning |
| --- | --- |
| `load_id`, `model` | Load ID and model ID |
| `variant`, `kind`, `trigger` | `plain`, `mtp` or `dflash2`; `load` or `reload`; `admin`, `openai_control`, `request`, `deferred_reload` or `startup` |
| `state` | `queued`, `loading`, `cancelling`, `loaded`, `failed` or `cancelled` |
| `stage` | `queued`, `preparing_memory`, `reading_weights`, `building_model`, `reading_draft_weights`, `starting_engine` or `warming_up` |
| `completed_stages` | Finished stages as `{stage, elapsed_ms}` |
| `bytes_done`, `bytes_total` | Weight bytes read and expected; `bytes_total` starts from a directory estimate and becomes exact as each loader reports its size |
| `started_unix_ms`, `finished_unix_ms` | Unix milliseconds; `finished_unix_ms` is `null` while the load runs |
| `code`, `message` | `null` while the load runs and after it loaded. After a cancel, `code` is `load_cancelled` and `message` stays `null`. After a failure, `code` is its classified code (`load_failed` when it has no known class) and `message` its message |
| `warning_code`, `warning` | Set on a loaded entry with a caveat, such as `scheduler_profile_hardware_mismatch` or `scheduler_profile_unusable`; otherwise `null` |

`POST /admin/api/models/loads/{load_id}/cancel` takes no body. It returns HTTP 202 with `{"load_id", "state":"cancelling"}`, 404 `{"code":"load_not_found"}` for an unknown ID, 409 `{"code":"load_finished"}` when the load already ended, or 409 `{"code":"load_finishing"}` once the load has reached `starting_engine` and can no longer be cancelled. A cancelled load ends in state `cancelled`. A cancelled new load leaves the model unloaded, and requests waiting for it receive 409 `load_cancelled`; a cancelled reload keeps the version that was loaded before.

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
