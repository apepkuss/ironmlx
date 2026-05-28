# Qwen3.6 Performance Phase 7 NAX End-to-End Baseline

**Goal:** Re-run Qwen3.6 core, CLI, and HTTP black-box performance after rebuilding
the canonical MLX runtime with Metal NAX kernels enabled.

**Artifact root:** `/tmp/ironmlx-qwen36-perf-phase7-nax-e2e-20260528-223209`

## Runtime

- ironmlx branch: `ironmlx-qwen36-perf`
- MLX source: `/Users/xin/workspace/iron-rivals/mlx` at `2165dc08`
- MLX install: `/Users/xin/.local/mlx`
- `libmlx.a` sha256:
  `f2d5ade9b80d867ca484ca02abddd8729948be500580fd3e4e652f278d29d171`
- `mlx.metallib` sha256:
  `494272bebe94679ec5d3971ba6b98750bd745aaca17b70c6d8cbf7a2e7b2c3d3`
- `target/release/ironmlx` was rebuilt after the MLX install, so the static link
  uses the NAX-enabled runtime.
- omlx environment uses Python MLX `0.31.2`.

## Core API

Prompt file:
`/tmp/ironmlx-qwen36-perf-phase3-latest/captures/mlx_lm_direct_prompt.txt`

The prompt encoded to 524 tokens and each run generated 16 tokens.

| Core path | TTFT p50 | E2E p50 | Decode p50 | Decode tok/s p50 |
| --- | ---: | ---: | ---: | ---: |
| `GenerationStream::new_text_only` | 252.44 ms | 366.34 ms | 114.13 ms | 131.43 |
| `Scheduler::prefill_admitted` | 248.20 ms | 370.32 ms | 121.94 ms | 123.02 |

Previous Phase 3 scheduler TTFT was about 321.7 ms. The NAX runtime therefore
closes most of the prior in-process model-core gap.

Benchmark note: scheduler core bench needs explicit `--effective-cap-max 1024`
for this 524+16 token prompt. The default micro-benchmark cap is exactly
`prompt_len + max_tokens`, which trips the scheduler's 85% runtime admission soft
limit even though production `serve --max-cache-cap` normally has headroom.

## CLI

Command shape:

```bash
MLX_DIR=/Users/xin/.local/mlx ./target/release/ironmlx generate \
  --model "$MODEL" \
  --prompt 'Continue the numeric sequence. Output only comma-separated integers: 1, 2, 3,' \
  --max-tokens 16 \
  --temperature 0 \
  --prefill-chunk-size 2048
```

Result:

- wall time: 2.22 s, including process startup and model load
- output completed without error

This is a product-entry smoke, not a steady-state throughput metric.

## HTTP Black-Box Results

Both servers ran one at a time against:
`mlx-community/Qwen3.6-35B-A3B-4bit`.

ironmlx:

```bash
MLX_DIR=/Users/xin/.local/mlx ./target/release/ironmlx serve \
  --model "$MODEL" \
  --port 18140 \
  --host 127.0.0.1 \
  --b-max 4 \
  --prefill-chunk-size 2048 \
  --max-cache-cap 640
```

omlx:

```bash
/Users/xin/workspace/iron-rivals/omlx/.venv/bin/omlx serve \
  --model-dir "$OUT/omlx-model-root" \
  --host 127.0.0.1 \
  --port 18141 \
  --max-concurrent-requests 4 \
  --no-cache \
  --base-path "$OUT/omlx/base"
```

Fixed-prompt primary comparison, `pp512 tg16`:

| Cell | ironmlx TTFT p50 | omlx TTFT p50 | TTFT ratio | ironmlx E2E p50 | omlx E2E p50 | E2E ratio | ironmlx tok/s | omlx tok/s | tok/s ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c=1` | 292.68 ms | 221.46 ms | 1.322 | 510.01 ms | 350.94 ms | 1.453 | 31.44 | 45.28 | 0.694 |
| `c=2` | 427.12 ms | 249.90 ms | 1.709 | 651.16 ms | 624.69 ms | 1.042 | 49.17 | 51.37 | 0.957 |
| `c=4` | 702.28 ms | 727.88 ms | 0.965 | 926.29 ms | 1157.37 ms | 0.800 | 69.14 | 55.21 | 1.252 |

Validity:

- fixed prompt cells are all valid: every request ended with `finish_reason=length`
  and produced 16 streamed content chunks.
- sequential `tg16` synthetic prompts are background only: ironmlx stopped early
  once, and omlx stopped at 10 tokens for all five rows.
- the fixed-prompt harness was adjusted for this artifact to use streamed content
  chunks as the fallback completion-token evidence when an endpoint does not
  return usage fields.

## Interpretation

The NAX-enabled MLX rebuild materially changes the performance picture:

1. The large single-request model-core gap from Phase 3 is mostly gone.
2. c=1 HTTP latency is still materially slower than omlx, so further optimization
   is still justified.
3. c=2 is close on E2E and throughput, despite worse TTFT p50.
4. c=4 now favors ironmlx on both E2E latency and aggregate generated throughput.

This shifts the next optimization target away from broad scheduler admission
work. The remaining high-value work is narrower:

- explain the c=1 HTTP overhead between core scheduler p50 (~248 ms TTFT) and
  fixed-prompt HTTP p50 (~293 ms TTFT);
- compare ironmlx and omlx single-request model call shapes after NAX, especially
  output projection, GatedDeltaNet, and MoE routed MLP materialization;
- inspect c=2 TTFT distribution, because E2E/throughput are near parity while
  TTFT ratio remains poor.

## Next Tasks

- Add a small reusable fixed-prompt HTTP benchmark harness or promote the artifact
  script into a maintained tool before future comparisons.
- Run a focused c=1 attribution pass with NAX-enabled ironmlx release binaries.
- Revisit scheduler admission/prefill batching only after the c=1 overhead is
  explained; c=4 already demonstrates that current batching can outperform omlx
  under this workload.
