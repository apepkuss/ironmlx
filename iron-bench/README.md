# iron-bench

Head-to-head HTTP benchmark harness for OpenAI-compatible LLM endpoints.

## What it measures

Per (target, prompt_len, max_tokens) cell, across N timed runs (after W warmup):

- **TTFT (ms)** — time from request send to first non-empty content token
- **TG (tok/s)** — decode tokens per second (completion_tokens / generation_duration)
- **TPOT (ms/tok)** — time per output token (excluding prefill)
- **PP (tok/s)** — prefill tokens per second (prompt_tokens / TTFT_seconds)
- **E2E (s)** — total wall-clock time

Reports median + p95 across runs. Output formats: Markdown (default), CSV (pandas-friendly,
one row per run), JSON (nested with raw runs preserved).

## Engine-neutral

iron-bench has no dependency on the `ironmlx` / `mlx` / `mlx-sys` crates. It drives
**any OpenAI-compatible `/v1/chat/completions` endpoint** — ironmlx, omlx, mlx-lm-server,
vllm-mlx, llama.cpp, third-party cloud providers — at the same external boundary users hit.

## Methodology highlights

- **Synthetic controlled-length prompts** — uses your model's `tokenizer.json` to round-trip
  a string to exactly N tokens (±2 BPE drift); per-run nonce in the prefix prevents prefix-cache
  hits across runs (omlx defaults to enable a tiered prefix cache; without nonce, the second
  run's prefill is ~0ms, invalidating PP measurement).
- **Greedy sampler** — `temperature=0, top_p=1` for both/all targets (deterministic, no
  sampler-algorithm bias).
- **stream_options.include_usage=true** — preferred for authoritative `prompt_tokens` and
  `completion_tokens` from the server; falls back to local SSE chunk count.
- **chat_template_kwargs.enable_thinking=false** — Qwen3+ chat template gates "thinking
  mode" via this kwarg. When enabled, omlx buffers the entire `<think>...</think>` block
  into a single SSE event, which collapses gen_duration to ~0 and inflates TG tok/s into
  the tens of thousands. Always disabled so both engines stream token-by-token under the
  same protocol.
- **Warmup excluded** — first N=1 run materializes MLX compile graphs / KV caches; not counted.

## Usage

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 5 --warmup 1 \
  --format markdown   # or csv, json
```

`--target name=URL` can be repeated for any number of endpoints. `--prompt-len` is
comma-separated; iron-bench iterates `prompt_len × target` cells.

## Sample Markdown output

```
# iron-bench results

- Targets: ironmlx=http://localhost:8080, omlx=http://localhost:8081
- Sampler: temperature=0, top_p=1 (greedy)
- Runs: 5 measured (after 1 warmup), median + p95

## TTFT (ms)
| target  | PP=128 TG=128 | PP=512 TG=128 | PP=2048 TG=128 |
|---|---|---|---|
| ironmlx | 45.2 (p95 47.1) | 152.4 (p95 156.0) | 521.8 (p95 530.4) |
| omlx    | 42.1 (p95 43.5) | 148.7 (p95 151.2) | 510.3 (p95 518.0) |

## Decode TG (tok/s)
... (same shape)
```

## CSV consumption

The `--format csv` output is pandas-friendly:

```python
import pandas as pd
df = pd.read_csv("results.csv")
df.groupby(["target", "pp_target"])["tg_tps"].median()
```

## Limitations

- **Single-request only**. Multi-request concurrency comes in v2 once ironmlx P8b ships
  the batched scheduler.
- **HTTP overhead** (~0.1-0.5ms loopback) is included in TTFT/E2E. Both targets bear it
  equally so it cancels in head-to-head comparison.
- **No GPU memory monitoring** — the HTTP layer is opaque to the engine's memory profile.
- **OpenAI endpoint only** in v1. Anthropic `/v1/messages` is symmetric work but deferred.

## Measured numbers — Qwen3.5-4B-MLX-4bit, M-series Apple Silicon

Single-request, greedy (`temperature=0`, `top_p=1`), `max_tokens=128`, `runs=3`, `warmup=1`,
ironmlx as built from current `ironmlx` branch (P8a applied), omlx 0.3.8 from
`/Volumes/Dev/omlx`.

| Target  | Decode TG (tok/s) median | TTFT PP=128 (ms) | TTFT PP=2048 (ms) | Prefill PP=2048 (tok/s) |
|---------|--------------------------|------------------|-------------------|-------------------------|
| ironmlx | 28.9 – 32.0              | 697              | 8530              | 240                     |
| omlx    | 53.2 – 54.9              | 604              | 7075              | 291                     |

**Decode TG gap**: omlx is ~1.7-1.9× faster across all PP cells. P8a's async-eval pipeline
+ incremental detokenizer landed cleanly (P4 fixture PASS, byte-identical token sequence
to mlx-lm reference) but only delivered ~5-9% TG improvement. The remaining gap is in the
GPU forward pass itself (kernel-level), not orchestration; addressing it requires kernel
profiling and is out of scope for this benchmark harness.

**TTFT / Prefill**: ironmlx is ~14-21% slower across PP — closer to parity than decode but
still a kernel-level gap. Prefill scales sub-linearly on both engines (GPU saturation
helps), so the relative gap shrinks as PP grows.

### Post-stage2 numbers (kernel fuse: SwiGLU + GDN proj concat + conv1d silu)

After P8a-stage2 (RmsNormGated SwiGLU compile-fuse + GatedDeltaNet 4→2 input
projection concat + conv1d-output silu compile-fuse), the same protocol re-run
yields essentially the same numbers as post-P8a:

| Target  | Decode TG (tok/s) median | TTFT PP=128 (ms) | TTFT PP=2048 (ms) | Prefill PP=2048 (tok/s) |
|---------|--------------------------|------------------|-------------------|-------------------------|
| ironmlx | 29.3 – 32.3              | 692              | 8474              | 242                     |
| omlx    | 53.3 – 55.0              | 609              | 7046              | 292                     |

**Stage2 acceptance MISSED**: target was decode TG ≥ 40 tok/s; achieved ~31 tok/s
(≈+1% over P8a). The three structural fuses landed cleanly (P4 fixture passes
byte-identical + a new `p4_model_forward_from_blocking_thread` regression test
confirms thread-correctness), but `mlx::compile` shapeless mode plus GDN
projection concat did NOT deliver the predicted ~6-8ms/step savings.

**Diagnostic conclusion**: Metal kernel dispatch overhead is NOT the dominant
factor in ironmlx's decode time. The actual bottleneck appears to be the GPU
compute time of the kernels themselves — i.e. the matmul / attention / SSM
kernels execute meaningfully slower on ironmlx's call paths than on
mlx-lm's, despite using the same MLX C++ primitives at the bottom layer.

Possible remaining root causes worth investigating in P8a-stage3:

1. **Per-step shape forcing recompile** — if `mlx::compile`'s shapeless cache
   is keyed on dtype/shape and our decode passes vary per layer, we may be
   recompiling every step rather than re-running a cached graph.
2. **Attention kernel selection** — `mlx::fast::scaled_dot_product_attention`
   may pick a different (slower) algorithm for ironmlx's tensor layout vs
   mlx-lm's. P8a-stage1 ruled out the args, but the chosen kernel may differ.
3. **Per-head reshape overhead** — the GDN forward reshapes q/k/v to
   per-head layout each step. mlx-lm does this too but maybe the Rust
   bindings introduce a stride / contiguity hit.
4. **bfloat16 vs float32 promotion edges** — mixed-precision behaviour at
   ironmlx call sites may differ from mlx-lm's.

These are out of scope for the iron-bench harness itself; they require
profiling under Metal frame capture / Instruments Time Profiler.
