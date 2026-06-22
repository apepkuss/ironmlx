# Qwen3.6 Performance Phase 8 c=1 Attribution

**Goal:** explain the remaining Qwen3.6 MoE single-request latency after the
NAX-enabled MLX rebuild, and separate HTTP/scheduler overhead from model-runtime
materialization cost.

**Artifact root:** `/tmp/ironmlx-qwen36-perf-phase8-c1-attribution-20260528-230034`

## Runtime

- ironmlx branch: `ironmlx-qwen36-perf`
- base commit: `b5d37e6af21e0ecc6084f72915a618dcd53b7439`
- model:
  `/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46`
- MLX source: `/Users/xin/workspace/iron-rivals/mlx` at `2165dc08`
- MLX install: `/Users/xin/.local/mlx`
- `libmlx.a` sha256:
  `f2d5ade9b80d867ca484ca02abddd8729948be500580fd3e4e652f278d29d171`
- `mlx.metallib` sha256:
  `494272bebe94679ec5d3971ba6b98750bd745aaca17b70c6d8cbf7a2e7b2c3d3`

## Commands

Profiling build:

```bash
MLX_DIR=/Users/xin/.local/mlx \
```

Profiling server:

```bash
RUST_LOG=ironmlx=info,warn \
MLX_DIR=/Users/xin/.local/mlx \
./target/release/ironmlx serve \
  --model "$MODEL" \
  --port 18142 \
  --host 127.0.0.1 \
  --b-max 1 \
  --prefill-chunk-size 2048 \
  --max-cache-cap 640
```

The first profiling attempt inherited `RUST_LOG=warn`, which correctly served
therefore uses explicit `RUST_LOG=ironmlx=info,warn`.

After profiling, the local binary was restored to the normal release build:

```bash
MLX_DIR=/Users/xin/.local/mlx cargo build --release -p ironmlx --bins
```

## HTTP Black-Box Results

Fixed prompt: `pp512 tg16`, text-only OpenAI streaming endpoint.

| Target | c | b_max | Valid | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ironmlx release | 1 | 1 | 42/42 | 241.93 ms | 243.31 ms | 357.81 ms | 359.27 ms | 44.71 tok/s |
| ironmlx release | 1 | 4 | 35/35 | 249.61 ms | 250.65 ms | 434.08 ms | 437.04 ms | 36.82 tok/s |
| omlx | 1 | 1 | 45/45 | 212.60 ms | 214.91 ms | 336.35 ms | 348.62 ms | 47.20 tok/s |

Notes:

  per request. Use it for shape attribution, not as the production latency
  number.
- The normal release `b_max=1` run is the production single-request reference.
- `b_max=4` still helps the high-concurrency Phase 7 result, but it hurts
  single-request decode/E2E in this workload.


Input:

- server log:
- joined bench CSV:

Aggregator result:

- join rate: 7/7 request ids, 100%
- coverage median: 0.9804, above the 0.95 gate
- root span median:
  `server_request_recv_to_first_content_sse_write = 254.88 ms`
- client TTFT median for the same 7 requests: `256.79 ms`
- median client-minus-root gap: about `1.9 ms`

Key median spans:

| Span | Inclusive | Exclusive | Exclusive Share |
| --- | ---: | ---: | ---: |
| `http_parse_render_tokenize` | 0.725 ms | 0.278 ms | 0.11% |
| `tokenizer_encode` | 0.460 ms | 0.460 ms | 0.17% |
| `scheduler_admission` | 0.087 ms | 0.087 ms | 0.03% |
| `model_prefill_forward` | 6.934 ms | 0.211 ms | 0.08% |
| `first_token_sampling_materialize_and_sample` | 247.020 ms | 247.020 ms | 96.92% |
| `detok_format_first_content_chunk` | 0.009 ms | 0.009 ms | 0.00% |

Interpretation:

1. HTTP parsing, chat-template rendering, tokenization, scheduler admission, and
   first-content SSE formatting are not the c=1 bottleneck.
2. `model_prefill_forward` is mostly lazy graph construction. The real MLX work
   materializes when first-token sampling forces evaluation; the span name does
   not mean sampling alone costs 247 ms.
3. The remaining production TTFT gap to omlx is now about 29 ms with `b_max=1`,
   not the larger Phase 7 `b_max=4` c=1 gap.
4. The larger `b_max=4` E2E gap is mostly a decode-shape issue: the same single
   request generates at 36.82 tok/s with `b_max=4` versus 44.71 tok/s with
   `b_max=1`.

## Conclusions

The single-request problem is now much narrower than the earlier Phase 7
headline suggested.

- Operationally, latency-sensitive single-request serving should use
  `--b-max 1`.
- Product-wise, a fixed `b_max` is the wrong long-term shape: `b_max=1` is best
  for c=1 latency, while `b_max=4` won the Phase 7 c=4 throughput comparison.
- Architecturally, the next production-grade optimization should make active
  batch shape follow the real admitted rows, so a server configured for high
  concurrency does not pay padded MoE/decode work when only one request is
  active.
- The residual c=1 TTFT delta against omlx requires white-box comparison of the
  first-token materialized MLX graph, especially Qwen3.6 GatedDeltaNet,
  routed/shared MoE MLP, output projection, and sampling eval boundaries.

## Next Tasks

1. Inspect scheduler tensor-shape construction and decode loop to identify why
   `b_max=4` single-request E2E falls from 44.71 tok/s to 36.82 tok/s.
2. Design an adaptive active-row execution path: keep configured capacity for
   admission, but execute prefill/decode with current active row count when safe.
3. Run an omlx white-box design pass focused on first-token graph shape, not a
   Python-to-Rust translation.
4. Promote the temporary fixed-prompt harness into a maintained perf tool with
   source of truth.
