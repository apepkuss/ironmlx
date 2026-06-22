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
- **Prefix-cache probe is explicit** — `--prefix-cache-probe` disables per-run
  nonce variation within a cell so server-side prefix caches can be measured.
  Default benchmarks still vary prompts to avoid accidental cache hits.

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

## Prefix-cache probe

Use `--prefix-cache-probe` only when the server is intentionally configured with
a prefix cache, such as ironmlx `--paged-prefix-cache-dir`. The flag reuses the
same synthetic prompt within each cell.

Sequential cold/write plus warm-hit probe:

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 2048 \
  --max-tokens 16 \
  --runs 3 --warmup 0 \
  --prefix-cache-probe \
  --format csv
```

In CSV/JSON output, run 0 is marked `cold_or_miss_candidate`; later measured
runs are marked `warm_hit_candidate`. With `--warmup > 0`, all measured runs
are marked `warm_hit_candidate`.

B>1 shared-prefix probe:

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 2048 \
  --max-tokens 16 \
  --concurrent 2 --duration 30 --warmup-duration 5 \
  --prefix-cache-probe \
  --format json
```

In concurrent mode, all workers reuse the same synthetic prompt within each
cell. This is useful for measuring shared-prefix TTFT and cache-path contention.

### Profile: qwen3.5-moe (Qwen3.5-35B-A3B-4bit MoE)

Model: `mlx-community/Qwen3.5-35B-A3B-4bit`
Local path hint: `~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<sha>/`

```sh
SNAP=$(ls -d $HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/*/ | head -1)
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model qwen3.5-moe \
  --model-dir "$SNAP" \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 50 --warmup 5 \
  --format markdown
```

Key differences from the dense 4B profile:

- `--runs 50 --warmup 5` (more steady-state samples to average over MoE expert routing variance)
- `--prompt-len 128,512,2048` — same prefill sweep, but MoE PP will be meaningfully slower at 2048

## Sample Markdown output

```text
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

## Integration smoke test

The `concurrent_smoke` test (`iron-bench/tests/concurrent_smoke.rs`) launches an
in-process axum mock SSE server and invokes the iron-bench binary with `--concurrent 2
--duration 1`. It self-skips gracefully when the tokenizer fixture is absent.

Stage the fixture once before running:

```sh
SNAP=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
cp "${SNAP}tokenizer.json" iron-bench/tests/fixtures/tokenizer.json
cargo test -p iron-bench --release --test concurrent_smoke
```

The fixture is gitignored (`iron-bench/tests/fixtures/.gitignore`) because the
Qwen tokenizer weighs ~19 MB.

## Concurrency modes

iron-bench supports two modes:

### v1 sequential (default)

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 5 --warmup 1
```

One request at a time per (target, prompt_len) cell. Reports median + p95 over the
`--runs` timed iterations. Good for **single-request latency** comparison.

### v2 concurrent (multi-worker)

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --concurrent 4 --duration 30 --warmup-duration 5
```

`N` concurrent workers per cell run for `--duration` seconds (after `--warmup-duration`
discarded warmup). Reports **p50/p95/p99 TTFT + ITL + aggregate tokens/s + per-worker
breakdown**. Good for **multi-request throughput** comparison.

Server requirements for v2:

- **ironmlx**: needs B1-p2.3c-3 (continuous batching, mid-batch admit). Set `b_max ≥ N`
  to avoid scheduler-full errors during the cell.
- **omlx**, **mlx-lm-server**: native multi-request support, no extra flags needed.
- **vllm-mlx**, **llama.cpp**: configure server-side `--max-num-seqs ≥ N`.

## Limitations

- **Closed-loop only.** Each worker awaits its response before firing the next.
  Open-loop (Poisson arrival rate) ships in **v3** when fairness metrics become
  meaningful (ironmlx 3d admission queue).
- **No fairness metrics** (Jain's index, per-tenant quotas). Deferred to v3.
- **No distributed load generation.** v2 runs from one machine. For higher load,
  scale `--concurrent` up (constrained by OS fd limits — `ulimit -n 65536` for N > 256).
- **HTTP overhead** (~0.1-0.5ms loopback) is included in TTFT/E2E. Both targets bear it
  equally so it cancels in head-to-head comparison.
- **No GPU memory monitoring** — the HTTP layer is opaque to the engine's memory profile.
- **OpenAI endpoint only**. Anthropic `/v1/messages` is symmetric work but deferred.

## Measured numbers — Qwen3.5-4B-MLX-4bit, M-series Apple Silicon

Single-request, greedy (`temperature=0`, `top_p=1`), `max_tokens=128`, `runs=3`, `warmup=1`,
ironmlx as built from current `ironmlx` branch (P8a applied), omlx 0.3.8 from
`/Volumes/Dev/omlx`.

| Target  | Decode TG (tok/s) median | TTFT PP=128 (ms) | TTFT PP=2048 (ms) | Prefill PP=2048 (tok/s) |
|---------|--------------------------|------------------|-------------------|-------------------------|
| ironmlx | 28.9 – 32.0              | 697              | 8530              | 240                     |
| omlx    | 53.2 – 54.9              | 604              | 7075              | 291                     |

**Decode TG gap**: omlx is ~1.7-1.9× faster across all PP cells. P8a's async-eval pipeline + incremental detokenizer landed cleanly (P4 fixture PASS, byte-identical token sequence to mlx-lm reference) but only delivered ~5-9% TG improvement. The remaining gap is in the GPU forward pass itself (kernel-level), not orchestration; addressing it requires kernel profiling and is out of scope for this benchmark harness.

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
