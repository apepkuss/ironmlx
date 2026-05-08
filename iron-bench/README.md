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
