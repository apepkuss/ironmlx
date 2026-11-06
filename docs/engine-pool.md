# CLI multi-model serving (EnginePool)

`ironmlx serve --model-manifest` starts a runtime model pool that routes
OpenAI and Anthropic requests by the request `model` field.

[简体中文](zh-CN/engine-pool.md)

For CLI users who need model routing within one server. This manifest belongs to `ironmlx serve`; the App manages its own models through Dashboard. Do not start a second backend while the App backend is running.

## Manifest

```json
{
  "default_model": "qwen-main",
  "max_loaded_models": 2,
  "models": [
    {
      "id": "qwen-main",
      "path": "/models/qwen-main",
      "load_policy": "preload",
      "scheduler_profile": "/profiles/qwen-main.json"
    },
    {
      "id": "qwen-mtp",
      "path": "/models/qwen-mtp",
      "load_policy": "lazy",
      "mtp_model_dir": "/models/qwen-mtp-head",
      "mtp_draft_tokens": 3
    },
    {
      "id": "disabled-exp",
      "path": "/models/disabled-exp",
      "load_policy": "disabled"
    }
  ]
}
```

Fields:

- `default_model`: optional model id used when a request omits `model`. If
  omitted and exactly one model is enabled, that model is the implicit default.
- `max_loaded_models`: optional cap on loaded engines. It must be at least `1`
  and at least the number of `preload` models.
- `models[].id`: stable external model id. It is also used as the paged prefix
  cache namespace.
- `models[].path`: local model directory.
- `models[].load_policy`: `preload`, `lazy`, or `disabled`. Omitted means
  `lazy`.
- `models[].scheduler_profile`: optional per-model scheduler profile. It
  overrides the global `--scheduler-profile` for that model.
- `models[].mtp_model_dir` and `models[].mtp_draft_tokens`: optional per-model
  MTP settings. Global MTP flags are rejected when `--model-manifest` is used.

Startup validation:

- Enabled models must point to local directories with a supported `config.json`
  `model_type`. Unsupported enabled models fail startup instead of failing on
  first lazy request.
- Supported `model_type` values are `qwen3_5`, `qwen3_5_moe`, `gemma4`, `gemma4_unified`,
  `glm4_moe_lite`, `llama`, `minicpmv4_6`, and `diffusion_gemma`.
- `disabled` models remain non-routable manifest entries and are not loaded.

## Runtime Behavior

- `preload` models load at startup. A preload failure fails startup.
- `lazy` models load on first request or through the control API.
- `disabled` models are listed in the manifest but are not routable.
- A failed lazy load leaves the model in `failed` state. Normal inference
  requests fail fast until the model is explicitly loaded again through the
  control API.
- When `max_loaded_models` is reached, EnginePool evicts the least recently used
  loaded `lazy` model that is not currently in use. `preload` models are pinned.

## Memory accounting and execution failures

The governor measures the greater of physical footprint and MLX active memory.
A loader that has evaluated a component records its storage as allocated;
first execution still follows Cold → Warming → Warm, but reserves only storage
not already allocated. Deferred vision loads retain their full first-use
reservation, and prefill/KV/workspace admission remains enabled.

`reserved_bytes` includes pending loads. Its `model_load_reserved_bytes` subset
continues to limit admission, while observed pressure excludes that subset to
avoid counting the same progressively allocated weights twice. Actual measured
usage still controls pressure and reclamation.
After cache reclaim, idle-model eviction uses a fresh authoritative sample.
Once observed usage (including non-load reservations) reaches the existing
recovery watermark, eviction stops even while the pressure grade cools down.
Admission hysteresis, recovery thresholds and degraded-telemetry protection
remain in effect.

An admitted execution failure is a terminal error, not a token. Ordinary,
MTP and DFlash2 paths preserve the complete cause message and scheduler
classification through prefill, decode and failed request cleanup. Responses
and Anthropic SSE use the same error classification as their non-streaming
counterparts. No success terminal event is emitted after a failure.
A priority-tail failure terminates active and queued requests before resetting
scheduler state. DFlash2 cache/prefill construction failures are execution
errors even when the admission reply has not yet been delivered; request
validation retains its own classification.
Chat Completions returns its retryable HTTP 503 code and `Retry-After`; an open
SSE stream reports the error without a successful finish or `[DONE]`.

## HTTP APIs

- `GET /v1/models`: lists enabled models and runtime state.
- `GET /healthz`: returns pool status, loaded count, per-model state, and loaded
  engine health where available.
- `POST /v1/models/:model_id/load`: explicitly loads a lazy model or retries a
  failed model.
- `POST /v1/models/:model_id/unload`: unloads a non-preload model if it is not
  currently in use.

Runtime states:

- `unloaded`: model is enabled but not loaded.
- `loading`: model load is in progress.
- `loaded`: model is loaded and routable.
- `failed`: last load attempt failed; use the load control API to retry.
- `disabled`: model exists in the manifest but cannot be served.

## Example

```bash
ironmlx serve \
  --model-manifest /path/to/models.json \
  --host 127.0.0.1 \
  --port 8080 \
  --paged-prefix-cache-dir ~/.ironmlx/cache/paged_prefix_cache
```

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s -X POST http://127.0.0.1:8080/v1/models/qwen-mtp/load
curl -s -X POST http://127.0.0.1:8080/v1/models/qwen-mtp/unload
```
