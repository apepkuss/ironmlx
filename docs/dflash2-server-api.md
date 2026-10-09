# DFlash2 configuration and usage

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
| Target (MoE) | `mlx-community/Qwen3.6-35B-A3B-4bit`, `-5bit`, `-6bit` or `-8bit` |
| Draft (MoE) | `incoai/Qwen3.6-35B-A3B-DFlash2` (BF16) |

Text only: no image/video requests, and the draft cannot be loaded as a normal base model. Qwen3.8 targets must use affine 4-bit or 8-bit; Qwen3.6 35B A3B targets have their own limits, described in [Qwen3.6 35B A3B (MoE)](#qwen36-35b-a3b-moe). Other quantized or unquantized targets are not marked compatible by the App.

## Startup example

```bash
ironmlx serve \
  --model /path/to/Qwen3.8-27B-8bit \
  --model-id mlx-community/Qwen3.8-27B-8bit \
  --dflash2-model-dir /path/to/Qwen3.8-27B-DFlash2 \
  --dflash2-draft-bits 4 \
  --max-sequences 8 \
  --dflash2-tensor-batch-max-width 4 \
  --admission-queue-max 2 \
  --port 8080
```

| Option | Accepted values / behavior |
| --- | --- |
| `--dflash2-block-size` | Optional 2–16 override. When omitted, IronMLX uses the checkpoint width capped at the qualified Q8 default. Widths above 8 are explicit opt-in and require a compatible checkpoint. |
| `--dflash2-draft-bits` | 0, 4 or 8; 0 preserves BF16 draft weights |
| `--dflash2-tree-max-nodes` | 0–15; 0 is the stable linear default, while a positive value opts affine-4 B1 requests into the bounded best-first tree |
| `--dflash2-position-keyed-sampling` | Explicitly opts sampled requests into the versioned device-side position-keyed sampler; changes same-seed output |
| `--max-sequences` | Positive active-request limit |
| `--model-id` | Stable public ID; defaults to the model path if omitted |
| `--dflash2-tensor-batch-max-width` | Positive tensor-group width limit; default 4; 1 disables cross-request tensor batching, not request concurrency |

Actual group width is the minimum of max sequences, tensor width limit and the number of ready requests with compatible execution shapes. The width limit does not increase active slots or replace max sequences.

The recorded `z-lab/Qwen3.8-27B-DFlash2` checkpoint declares block size 8, so automatic resolution selects Q8 and it cannot exercise the Q16 proposal lane. Q16 is fail-closed and explicit: the target's affine-4 B1 verify capability and the draft checkpoint must both support the requested width. No checkpoint is widened implicitly.

## App configuration

The scanner identifies `DFlash2DraftModel` as an auxiliary artifact. A draft appears in the selector only if it satisfies backend constraints and matches the target's hidden size, vocabulary, context length, layer count, RMS epsilon and RoPE theta. FFN widths are not compared because the draft has its own FFN, and the number of target taps is independent of the draft's depth. Incomplete or incompatible artifacts are rejected.

Dashboard exposes the DFlash2 switch, compatible draft, block size, draft precision and Tensor Batch limit. An empty block size uses the selected checkpoint width capped at Q8. An empty tensor batch limit uses the backend default of 4. The effective tensor batch width is also bounded by Max Sequences.
Enabling DFlash2 retains one default target and restarts the backend into the fixed target/draft actor. Disabling or changing its settings also requires a controlled restart. Validation, startup or recovery failure restores the prior configuration and model parameters, then restarts the previous path.

The App retains `GET /v1/models` with the stable target ID but no dynamic model-management API. Dashboard and menu bar recover target/draft state from health and saved settings without exposing the draft as an ordinary model.
Dashboard shows target/draft, block size, precision, TPS, acceptance rate, windows, rollbacks, residual corrections and peak memory. Health and diagnostics also include tensor width limit, observed maximum width, windows, group count and divergence splits.

## Qwen3.6 35B A3B (MoE)

The MoE combination runs through its own target implementation and qualification, independent of Qwen3.8. It does not inherit Qwen3.8 profiles, tree defaults, draft-precision options or batching.

| Item | Limit |
| --- | --- |
| Target recipe | Affine, group size 64, 4/5/6/8 bits, with exactly the 8-bit `mlp.gate` and `shared_expert_gate` overrides the mlx-community checkpoints ship. Other recipes (including OptiQ) are rejected at load time. |
| Draft precision | BF16 only (`--dflash2-draft-bits 0`, the default for this target). 4 and 8 are rejected; the App offers only BF16. |
| Verify width | Up to 8 for every bit width. When `--dflash2-block-size` is omitted the checkpoint width is capped at 8; an explicit value above 8 is an error. |
| Execution | B1 only. `--max-sequences` still bounds how many requests are active, but every target forward carries one request: active requests take turns window by window, and cross-request tensor batching is disabled. |
| Tree | `--dflash2-tree-max-nodes` 1–15 enables the flat tree for greedy, unconstrained requests. Sampled requests and requests with a constraint (including the reasoning budget attached to thinking-enabled requests) run linear windows instead; `tree_fallback_linear_windows` counts them and the server logs the reason once. |
| Sampling | Exact speculative sampling is supported. `--dflash2-position-keyed-sampling` is rejected for this target. |

Every greedy result, linear or tree, is token-identical to ordinary decoding of the same target (except with the affine4 projection variable below). Linear windows are the default because the tree measured slower than linear windows at every bit width on M5 Max.

The M5 profile (`--m5-dflash2-profile`) uses its own MoE table, which is currently empty: no MoE setting showed an end-to-end gain, so the profile does not change MoE execution. `status=active` only means the profile is installed for this GPU; what it turns on is the startup log's `effective_settings` (`none` for MoE by default) and the non-null values in `healthz.dflash2.m5_profile.settings`. `healthz.dflash2.m5_profile.target` reports `qwen36-moe`, `status` is `target_not_qualified` for a target outside the qualified recipes, and `settings` lists two experimental opt-in variables:

| Variable | Effect |
| --- | --- |
| `IRONMLX_EXPERIMENTAL_M5_MOE_GROUPED_QMV=1` | Shares each routed expert's weights across verify rows (affine6/8, 8+ rows). Bit-identical to ordinary decoding. |
| `IRONMLX_EXPERIMENTAL_M5_MOE_AFFINE4_PROJECTIONS=1` | M5 tensor-unit projections for the affine4 target. Deterministic, but output differs from ordinary MLX decoding; the execution fingerprint reports `projections=m5-affine4-v1`. Needs a real Apple GPU of generation 17 or newer: elsewhere the variable is ignored with a warning, the generic projections run, and `settings` reports it as null. |

## Execution and concurrency

The separate actor does not use ordinary Scheduler, MTP or Prompt Lookup execution. Each request owns target/draft caches, a sampler and PRNG state.
With one max sequence, one request advances; larger limits permit that many active requests. Ready requests with identical execution keys form bounded MLX tensor groups. Excess requests form subsequent groups and advance in rotation. Different constraints, sampling shapes or cache states remain separate; differing accepted lengths split caches back into requests, which can regroup later. Benefits depend on hardware, workload and acceptance rate.

The optional tree builds a best-first candidate lattice capped at 15 nodes, batches at most eight root-to-leaf paths into one target verify forward, and commits only the accepted row's transactional cache state. It is limited to affine-4, unconstrained B1 execution. Enabling it sets the effective cross-request tensor width to one because its batch lanes are reserved for tree paths. `tree_windows` and `tree_drafted_nodes` expose actual use rather than merely configured eligibility.

When slots fill, requests wait in the admission queue. A full queue returns HTTP 503, `scheduler_queue_full`, and `Retry-After: 5`. Streaming disconnect releases caches and slots at the next safe boundary after the current forward.

## Foreground and background priority

Chat Completions and Responses requests may set `service_tier: "flex"` for
low-priority title generation, summaries, warmups, and similar work. Omitted
tiers plus `auto` and `default` are foreground. A foreground arrival causes an
active flex request to pause after its current decode/verify round. The actor
keeps the target/draft cache, sampler, PRNG, accepted-token history, and tensor
state in memory; it does not replay the prompt or restart the request. Paused
work resumes FIFO when no foreground request is active or queued.

DFlash2 serving routes every request through its priority-aware actor, so this
contract applies to every DFlash2 Chat Completions and Responses request. This
is an IronMLX-local scheduling interpretation of the OpenAI-compatible field,
not an implementation of hosted billing, SLA, project-tier, or capacity-pool
semantics. IronMLX accepts `auto`, `default`, and `flex`; other tier values are
rejected.

Priority changes latency scheduling, not memory admission. A paused request
continues to own its charged cache memory, so the memory governor may still
reject new work when the real resident set cannot fit it. `background: true`
in the Responses API remains unsupported because that field requests a stored
asynchronous job, not scheduling priority.

`healthz.scheduler.background_paused` is the live paused count;
`background_preemptions` and `background_resumes` are cumulative counters.

## Sampling

Both greedy and sampled requests use DFlash2 verification. GreedyVerify preserves byte-for-byte equality with ordinary Q=1 decoding. SampledVerify uses exact speculative sampling: probabilistic acceptance, rejection residuals, bonus tokens and per-request reproducible PRNG state.

Stateful exact sampling remains the default. `--dflash2-position-keyed-sampling` switches positive-temperature requests to `PositionKeyedV1`, where every draw is a device-side function of seed, absolute output position and token ID after the configured penalties and top-k/top-p/min-p filters. This makes a serial linear window and a drafted/tree window choose the same token at the same position, independent of batch shape. The mode intentionally does not preserve the default sampler's same-seed token sequence.

Public Chat/Responses sampling accepts `temperature` and `top_p`; Messages additionally accepts `top_k`. Omitted fields use checkpoint defaults; the recorded Qwen3.8 setup defaults to `top_k=20`.
A fixed seed promises reproducibility only within the same IronMLX/MLX versions, checkpoint, settings and execution shape, not across versions.

## HTTP protocols

| Endpoint | Synchronous / SSE | Termination |
| --- | --- | --- |
| `/v1/chat/completions` | Both | Chat chunks and `[DONE]` |
| `/v1/responses` | Both | Responses typed lifecycle |
| `/v1/messages` | Both | Anthropic Messages lifecycle |

See the [Text and vision API](text-vision-api.md) for strict fields, errors and disconnect behavior.

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
    "tree_max_nodes": 15,
    "position_keyed_sampling": true,
    "requests": 3,
    "windows": 96,
    "drafted_tokens": 384,
    "accepted_draft_tokens": 256,
    "rollback_count": 31,
    "tree_windows": 40,
    "tree_drafted_nodes": 512,
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

`scheduler.b_max`, `b_active`, `b_queued`, and `background_paused` represent actor capacity, active requests, queued requests, and in-memory paused flex requests. Interpret performance metrics with the hardware, prompt, acceptance rate and sampling settings.
