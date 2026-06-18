# DiffusionGemma Enhancements Evaluation

> Evaluation worktree: `/Users/xin/workspace/ironmlx-backend-diffusiongemma-enhancements-eval`
>
> Base commit: `95ac881 feat: support DiffusionGemma server serial lane`

## Summary

DiffusionGemma should continue to run through an independent block-diffusion lane.
The current causal scheduler is built around sequential token decode and the
`core::model::Model` trait, while DiffusionGemma uses prompt encoding plus
parallel canvas denoising and a model-specific cache. For the next increment,
the highest-value path is to harden the serial lane and add streaming of
committed block/token text. Full block-diffusion batching should be treated as a
separate scheduler design, not as an adapter into the existing causal scheduler.

OMLX is useful as a behavioral reference, but it should not be copied. Its
DiffusionGemma support also routes to a serial diffusion lane, uses a lock,
streams only committed diffusion outputs, and keeps diffusion out of the normal
autoregressive VLM scheduler.

## Current ironmlx State

- `ironmlx/src/core/server/diffusion_gemma.rs` is intentionally separate from
  the causal `AppState<M>` / `SchedulerActor` path. Its module header documents
  this boundary and says requests are completed as non-streaming responses.
- `DiffusionGemmaAppState` owns `Arc<Mutex<DiffusionGemmaModel>>`, a tokenizer,
  generation config, model id, and vision input config. This makes model access
  process-local and serial.
- OpenAI and Anthropic handlers return `unsupported_feature` for `stream: true`.
- `generate_completion` runs inside `tokio::task::spawn_blocking`, takes the
  model with `blocking_lock`, calls `generate_text` or `generate_image_text`,
  then collects all events into one response.
- `ironmlx/src/models/diffusion_gemma/generation.rs` returns
  `Vec<DiffusionGemmaGenerateEvent>`. Events are emitted after a canvas has been
  denoised and committed, not during every denoising step.
- The normal `core::model::Model` trait expects causal APIs:
  `make_cache(batch, cap, dtype)`, `forward_on`, `batched_prefill`,
  `forward_text_hidden`, and `LayerCache`. DiffusionGemma has its own
  `DiffusionGemmaCache`, `encode_tokens_on`, `encode_inputs_on`, and
  `decode_logits_on` flow.

## OMLX Reference Findings

- `omlx/model_discovery.py` treats `diffusion_gemma` as an mlx-vlm native VLM
  model even when `vision_config` is absent.
- `omlx/engine/vlm.py` detects block-diffusion models, logs that it is using a
  serial diffusion lane, disables the normal vision feature cache for diffusion,
  and does not create the normal async engine scheduler for this path.
- Its diffusion lane owns an async lock plus cancellation events. Streaming is
  implemented by iterating `mlx_vlm.generate.diffusion.stream_diffusion_generate`
  and yielding committed outputs at diffusion block boundaries or final flush.
- Tests assert that DiffusionGemma streaming uses the diffusion lane instead of
  the AR VLM path, and that unary chat collects streamed block outputs.

The relevant lesson is architectural, not source-level: keep DiffusionGemma out
of the autoregressive scheduler until there is a dedicated block-diffusion
batching design.

## Enhancement Candidates

### 1. Serial Lane Hardening

Recommended as the first enhancement.

Add explicit request admission around the current serial lane:

- expose active and queued counts in health or diagnostics;
- bound queued work instead of allowing unbounded `spawn_blocking` tasks to wait
  on the model mutex;
- return a clear overload status when the lane is saturated;
- propagate request cancellation so disconnected streaming clients can stop
  waiting for further events;
- keep non-streaming OpenAI and Anthropic behavior unchanged.

This improves production behavior without changing the model algorithm.

### 2. Streaming Committed Outputs

Recommended together with or immediately after lane hardening.

The generator already has a natural stream boundary: committed
`DiffusionGemmaGenerateEvent` values after each canvas is denoised. The server
can expose this as OpenAI and Anthropic SSE without claiming causal per-token
decode semantics.

Required design rules:

- stream only committed text, not draft denoising states;
- preserve the current unary API by keeping the vector-returning helper as a
  wrapper around the streaming primitive;
- use the existing OpenAI and Anthropic SSE framing style, but implement it in
  the DiffusionGemma server module because the causal scheduler event type does
  not match;
- document that first output may arrive after one full diffusion block, so
  latency differs from autoregressive streaming.

### 3. Dedicated Block-Diffusion Batching Scheduler

Feasible but not the next increment.

A real concurrent scheduler for DiffusionGemma would need a new scheduler family
that batches by diffusion work, not by causal decode step. It would need to
solve:

- prompt/image tensor padding and per-row prompt cache setup;
- aligning canvas length and denoising step schedules across requests;
- per-row stop handling while other rows continue denoising;
- fairness between long and short generations;
- queue admission, cancellation, and error routing;
- performance validation against serial execution on the target Apple Silicon
  hardware.

This should be designed as a new block-diffusion scheduler, not as an
implementation of the existing causal `Model` trait.

### 4. Multi-Replica Concurrency

Not recommended for `diffusiongemma-26B-A4B-it-4bit`.

Loading multiple model replicas would be simple conceptually, but memory and
Metal pressure are likely to dominate. It also avoids the real scheduling
problem rather than solving it.

## Recommended Next Step

Implement a small DiffusionGemma lane abstraction in this worktree:

1. Keep one model instance and one serial execution slot.
2. Add bounded admission and observable active/queued counters.
3. Refactor generation to support an event sink or iterator while retaining the
   current `Vec<DiffusionGemmaGenerateEvent>` API.
4. Add OpenAI and Anthropic SSE support for committed text events.
5. Add cancellation checks at safe boundaries, at least between committed
   canvas blocks.

After that lands, evaluate a dedicated block-diffusion batching scheduler with
real measurements. The existing causal scheduler should remain unchanged for
DiffusionGemma.

## Verification Needed For Implementation

When implementation starts, verification should include:

- unit tests that the vector API still collects the same events;
- unit tests for SSE framing on OpenAI and Anthropic DiffusionGemma routes;
- a concurrency test showing excess requests are queued or rejected according to
  the configured bound;
- a cancellation test showing the event stream stops and releases lane state;
- `cargo fmt`;
- `cargo +nightly fmt --all -- --check`;
- `cargo +nightly clippy --all-features --workspace -- -D warnings`;
- `cargo build --release`;
- real-model CLI/server smoke with
  `~/.ironmlx/models/mlx-community/diffusiongemma-26B-A4B-it-4bit` when local
  hardware budget allows it.
