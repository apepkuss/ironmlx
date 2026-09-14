# DFlash2 server and CLI

[简体中文](zh-CN/dflash2-server-api.md)

For advanced App users and CLI/API integrators configuring DFlash2. Check the model combination first, then use the startup example. Execution details below support tuning and troubleshooting.

## Scope

DFlash2 has three entry points:

- `ironmlx generate --dflash2-model-dir ...` for local text generation.
- `ironmlx serve --dflash2-model-dir ...` for a fixed target/draft HTTP server.
- The App model settings, selecting a compatible draft and restarting into a separate actor.

Recorded validation covers:

| Role | Checkpoint |
| --- | --- |
| Target | `mlx-community/Qwen3.8-27B-4bit` or `mlx-community/Qwen3.8-27B-8bit` |
| Draft | `z-lab/Qwen3.8-27B-DFlash2` |

Text only: no image/video requests, and the draft cannot be loaded as a normal base model. Targets must use affine 4-bit or 8-bit; other quantized or unquantized targets are not marked compatible by the App.

## Startup example

```bash
ironmlx serve \
  --model /path/to/Qwen3.8-27B-8bit \
  --model-id mlx-community/Qwen3.8-27B-8bit \
  --dflash2-model-dir /path/to/Qwen3.8-27B-DFlash2 \
  --dflash2-block-size 4 \
  --dflash2-draft-bits 4 \
  --max-sequences 8 \
  --dflash2-tensor-batch-max-width 4 \
  --admission-queue-max 2 \
  --port 8080
```

| Option | Accepted values / behavior |
| --- | --- |
| `--dflash2-block-size` | 2–8 |
| `--dflash2-draft-bits` | 0, 4 or 8; 0 preserves BF16 draft weights |
| `--max-sequences` | Positive active-request limit |
| `--model-id` | Stable public ID; defaults to the model path if omitted |
| `--dflash2-tensor-batch-max-width` | Positive tensor-group width limit; default 4; 1 disables cross-request tensor batching, not request concurrency |

Actual group width is the minimum of max sequences, tensor width limit and the number of ready requests with compatible execution shapes. The width limit does not increase active slots or replace max sequences.

## App configuration

The scanner identifies `DFlash2DraftModel` as an auxiliary artifact. A draft appears in the selector only if it satisfies backend constraints and matches the target's hidden/intermediate size, vocabulary, context length, layer count, RMS epsilon and RoPE theta. Incomplete or incompatible artifacts are rejected.

Dashboard exposes the DFlash2 switch, compatible draft, block size, draft precision and Tensor Batch limit. An empty limit uses the backend default of 4. The effective width is also bounded by Max Sequences.
Enabling DFlash2 retains one default target and restarts the backend into the fixed target/draft actor. Disabling or changing its settings also requires a controlled restart. Validation, startup or recovery failure restores the prior configuration and model parameters, then restarts the previous path.

The App retains `GET /v1/models` with the stable target ID but no dynamic model-management API. Dashboard and menu bar recover target/draft state from health and saved settings without exposing the draft as an ordinary model.
Dashboard shows target/draft, block size, precision, TPS, acceptance rate, windows, rollbacks, residual corrections and peak memory. Health and diagnostics also include tensor width limit, observed maximum width, windows, group count and divergence splits.

## Execution and concurrency

The separate actor does not use ordinary Scheduler, MTP or Prompt Lookup execution. Each request owns target/draft caches, a sampler and PRNG state.
With one max sequence, one request advances; larger limits permit that many active requests. Ready requests with identical execution keys form bounded MLX tensor groups. Excess requests form subsequent groups and advance in rotation. Different constraints, sampling shapes or cache states remain separate; differing accepted lengths split caches back into requests, which can regroup later. Benefits depend on hardware, workload and acceptance rate.

When slots fill, requests wait in the admission queue. A full queue returns HTTP 503, `scheduler_queue_full`, and `Retry-After: 5`. Streaming disconnect releases caches and slots at the next safe boundary after the current forward.

## Sampling

Both greedy and sampled requests use DFlash2 verification. GreedyVerify preserves byte-for-byte equality with ordinary Q=1 decoding. SampledVerify uses exact speculative sampling: probabilistic acceptance, rejection residuals, bonus tokens and per-request reproducible PRNG state.

Public Chat/Responses sampling accepts `temperature` and `top_p`; Messages additionally accepts `top_k`. Omitted fields use checkpoint defaults; the recorded Qwen3.8 setup defaults to `top_k=20`.
A fixed seed promises reproducibility only within the same IronMLX/MLX versions, checkpoint, settings and execution shape, not across versions.

## HTTP protocols

| Endpoint | Synchronous / SSE | Termination |
| --- | --- | --- |
| `/v1/chat/completions` | Both | Chat chunks and `[DONE]` |
| `/v1/responses` | Both | Responses typed lifecycle |
| `/v1/messages` | Both | Anthropic Messages lifecycle |

See the [API reference](api-reference.md) and [compatibility matrix](api-compatibility-matrix.md) for strict fields, errors and disconnect behavior.

## Incompatible combinations

The path rejects MTP, Prompt Lookup, KV quantization, paged/persistent prefix cache, Active KV offload, Scheduler profiles and Scheduler autotune reports. It does not silently switch to another speculative path. Its own in-memory prefix cache is separate from these excluded caches.

## Health metrics

`GET /healthz` exposes a `dflash2` object with configuration and accumulated metrics. These numbers illustrate field shapes, not performance guarantees:

```json
{
  "dflash2": {
    "enabled": true,
    "block_size": 4,
    "draft_quantization_bits": 4,
    "requests": 3,
    "windows": 96,
    "drafted_tokens": 384,
    "accepted_draft_tokens": 256,
    "rollback_count": 31,
    "sampled_requests": 1,
    "exact_sampling_windows": 32,
    "exact_acceptance_draws": 128,
    "exact_residual_corrections": 9,
    "exact_bonus_samples": 18,
    "latest_generation_tps": 54.9,
    "latest_acceptance_rate": 0.68,
    "peak_memory_bytes": 21474836480
  }
}
```

`scheduler.b_max`, `b_active` and `b_queued` represent actor capacity, active requests and queued requests. Interpret performance metrics with the hardware, prompt, acceptance rate and sampling settings.
