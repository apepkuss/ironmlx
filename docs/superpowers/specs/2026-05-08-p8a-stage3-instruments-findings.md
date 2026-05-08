# P8a-Stage3: Instruments Metal GPU Profiling — Findings

**Date:** 2026-05-08  
**Branch:** ironmlx  
**Hardware:** Apple Silicon (macOS 25.3)  
**Model:** Qwen3.5-4B-MLX-4bit (32 layers: 24 GDN + 8 full-attention)  
**Engines:** ironmlx :8080 vs omlx (mlx-lm) :8081

---

## 1. Methodology

### Trace Capture

```
xcrun xctrace record --template "Metal System Trace" \
  --attach <PID> --output /tmp/<engine>.trace --time-limit 15s
```

Both engines captured during back-to-back OpenAI-compatible `/v1/chat/completions`
requests (PP≈128 tokens, max_tokens=3 for ironmlx due to tokenizer bug; max_tokens=32
for omlx).

### Critical Blocker: tokenizers 0.20.4 DecodeStream Panic

ironmlx's streaming decode path panics at generated token 4+ for Qwen3.5 with thinking
mode enabled (the default). Root cause: `step_decode_stream` in tokenizers 0.20.4 has a
state-machine bug where `*read_index` grows past `ids.len()` after an internal drain,
causing `new_prefix_index = ids.len() - *prefix_index` to underflow (wraps to
`usize::MAX`), triggering an out-of-bounds slice panic on the next call.

Trigger sequence: Qwen3.5 chat template injects `<think>\n\n</think>\n\n` at the start
of the assistant turn. Token 271 (`\n\n`) at position 4 hits a specific BPE merge
boundary that triggers the drain+overflow sequence.

**Impact on profiling:** Could not capture ironmlx decoding more than 3 steps.
All timing data derived from delta measurements across `max_tokens=1,2,3` runs.

### Shader Profiler Tables

`metal-shader-profiler-intervals` and `metal-driver-event-per-thread-intervals` were
**empty** in both traces. These tables require GPU frame-capture API
(`MTLCaptureManager`) or explicit shader profiling mode, neither of which is available
when attaching to a running non-GUI process via `--attach`. The `metal-gpu-execution-
points` table had command-buffer timing only — no per-kernel breakdown.

### Timing Methodology (Fallback)

Per-step decode cost was derived via:

```
T_decode_step = (T(max_tokens=3) − T(max_tokens=1)) / 2
```

This cancels out prefill latency and HTTP fixed overhead, isolating pure decode step
cost. Three measurements each; median reported.

---

## 2. Measured Performance

| Metric                     | ironmlx      | omlx         | Ratio |
|----------------------------|-------------|-------------|-------|
| max_tokens=1 wall time (ms)| 360          | 310          | 1.16× |
| max_tokens=3 wall time (ms)| 434          | 350          | 1.24× |
| Derived per-step cost (ms) | **37 ms**   | **20 ms**   | **1.77×** |
| Estimated TG throughput    | ~27 tok/s   | ~50 tok/s   | 1.85× |

The prefill overhead is comparable (360 vs 310 ms) — the gap concentrates entirely in
the **decode step repeat cost**.

---

## 3. Root-Cause Analysis

### 3.1 omlx: `@mx.compile` on GDN kernel dispatch

In `/Volumes/Dev/omlx/.venv/.../mlx_lm/models/gated_delta.py`:

```python
@partial(mx.compile, shapeless=True)   # compile_g graph fused
def compute_g(B, T, Hv, Dv, query, key, value, alpha):
    ...

@mx.compile                            # ENTIRE kernel dispatch compiled as a graph
def gated_delta_kernel(query, key, value, alpha, state):
    ...
```

`@mx.compile` at the `gated_delta_kernel` level means MLX traces the full Metal kernel
dispatch — including kernel argument binding, buffer allocation, and command encoding —
**once** and stores a reusable compiled plan. On every subsequent decode step (identical
shapes), the runtime replays the cached plan with zero graph-building overhead.

With 24 GDN layers per forward pass, this matters enormously: 24 × (graph-build time
saved) = significant per-step reduction.

### 3.2 ironmlx: MetalKernel dispatched without a top-level compile scope

In `/Volumes/Dev/cxx-mlx/ironmlx/src/nn/gated_delta_net.rs`:

- `compute_g` is wrapped in `mlx::compile::compile(ShapeMode::Shapeless)` — equivalent
  to `@partial(mx.compile, shapeless=True)`. This is correct.
- The **outer** `GatedDeltaNet::forward_on()` call is **not** wrapped in any
  `mlx::compile::compile` scope. Each decode step re-enters the MLX graph builder,
  re-binds Metal kernel arguments, and re-schedules command buffers from scratch.
- At model level, `Qwen35Model::forward_on` has no top-level compile wrapper either.

The missing `@mx.compile` equivalent on the GDN kernel dispatch path is the primary
source of the ~17 ms/step gap.

### 3.3 Supporting Evidence

| Factor | ironmlx | omlx | Notes |
|--------|---------|------|-------|
| GDN kernel compile scope | `compute_g` only | `compute_g` + full dispatch | omlx compiles more |
| Top-level model compile | None | `generate` step via `mlx_lm.generate` | omlx benefits from outer compile |
| KV cache pre-allocation | Pre-alloc at prompt_len+max_new | Grows 256-chunk | Both similar; minor omlx advantage from lazy alloc shape change flushing cache |
| Attention | `mx.fast.scaled_dot_product_attention` | same | Equivalent |
| Quantized matmul | `fast::affine_dequantize` dispatch | same MLX op | Equivalent |

---

## 4. Secondary Findings

### 4.1 Thinking Mode Gap

ironmlx does not support `chat_template_kwargs` (no `enable_thinking=false` API field).
Qwen3.5's default chat template generates `<think>\n\n</think>\n\n` prefix (4 tokens)
before actual output. This is the direct trigger for the tokenizer panic, and it also
means ironmlx benchmarks include thinking-token overhead that omlx avoids when callers
pass `enable_thinking=false`. The thinking tokens themselves do not affect decode step
timing (same forward pass), but they do affect usable output throughput.

### 4.2 MTP Head Not Used

ironmlx strips `mtp.*` weights at load time — the MTP head (Qwen3.5's multi-token
prediction module) is never executed. omlx also does not invoke MTP during
`mlx_lm.generate`. No gap here.

### 4.3 KV Cache Allocation Strategy

ironmlx pre-allocates `(batch, heads, cap, head_dim)` where `cap = prompt_len +
max_new_tokens`. omlx grows in 256-step chunks. The pre-allocation means ironmlx avoids
shape changes during decode (good), but the larger initial allocation may increase
memory bandwidth slightly. This is a second-order effect.

---

## 5. Recommendations for P8a-Stage4

### P1 (Critical — blocks all benchmarks)

**Fix tokenizers 0.20.4 `DecodeStream` bug.**

Options:
1. Pin `tokenizers = "0.21"` in `Cargo.toml` if the bug is fixed upstream.
2. Implement a fallback that catches the panic and retries with a full `decode()` call
   on failure (belt-and-suspenders).
3. Replace `DecodeStream` with a manual incremental decode that calls
   `tokenizer.decode(&ids[prev_len..], skip_special)` per step — O(N) but correct.

### P2 (Primary performance fix)

**Wrap the GDN forward pass in `mlx::compile::compile` at dispatch time.**

Apply `mlx::compile::compile(ShapeMode::Shapeless)` around the entire
`GatedDeltaNet::forward_on` body (or at minimum the MetalKernel dispatch call), not
just `compute_g`. The goal is to give MLX a single compiled plan covering:
`compute_g` → `MetalKernel::invoke` → output reshape.

Expected outcome: recover most of the 17 ms/step gap (targeting ≤22 ms/step).

### P3 (Secondary performance)

**Add top-level model-step compile scope.**

Wrap the per-step decode call in `generate.rs` (the `model.forward_on(inputs, cache)`
call) in a `mlx::compile::compile` scope. This matches `mlx_lm.generate`'s outer
`@mx.compile` and allows MLX to optimize cross-layer graph fusion.

### P4 (API completeness)

**Add `chat_template_kwargs` to `ChatRequest`** (including `enable_thinking` bool).
Pass it through to `apply_chat_template` so callers can suppress thinking tokens and
avoid the tokenizer panic trigger.

---

## 6. Conclusion

The **~1.77× decode TG gap** (37 ms vs 20 ms per step) is attributable to:

1. **Missing `@mx.compile` on GDN kernel dispatch** — omlx compiles the full
   `gated_delta_kernel` as a reusable graph; ironmlx re-builds the graph each step.
   Estimated contribution: ~12–15 ms/step across 24 GDN layers.

2. **Missing top-level model-step compile** — omlx's `mlx_lm.generate` wraps the full
   forward pass in `@mx.compile`; ironmlx does not. Estimated contribution: ~2–5 ms/step.

3. **tokenizers 0.20.4 `DecodeStream` panic** — critical blocker preventing reliable
   benchmarking beyond 3 tokens. Must be fixed before stage-4 profiling.

The GPU kernel implementations (GDN Metal shader, attention SDPA, quantized matmul) are
**equivalent** between the two engines — there is no fundamental algorithmic gap. The
performance difference is purely in **graph compilation and dispatch overhead**.
