# TurboQuant KV Validation

This note records the validation workflow for TurboQuant-backed KV cache reads.
The current matrix covers four generation configurations:

| CLI value | KV bits |
| --- | --- |
| `none` | full precision KV cache |
| `turbo3` | `K3V3` |
| `turbo4` | `K4V4` |
| `k3v4` | `K3V4` |

## Reproduce

Run the full validation script from the repository root:

```bash
MLX_DIR=$HOME/.local/mlx scripts/turboquant_kv_validate.sh
```

Run the long-context pressure matrix with:

```bash
MLX_DIR=$HOME/.local/mlx scripts/turboquant_kv_long_context_validate.sh
```

Useful overrides:

| Environment variable | Default |
| --- | --- |
| `MODEL` | latest snapshot under `~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots` |
| `PROMPT_FILE` | generated technical-summary prompt |
| `KV_QUANTS` | `none,turbo3,turbo4,k3v4` |
| `MAX_TOKENS` | `16` logits replay tokens |
| `BENCH_MAX_TOKENS` | `32` generation benchmark tokens |
| `RUNS` | `3` measured benchmark runs |
| `WARMUP_RUNS` | `1` warmup benchmark run |
| `OUT_DIR` | `docs/benchmarks/turboquant-kv/<timestamp>` |

The script builds and runs:

```bash
cargo build --release \
  -p ironmlx \
  --bin ironmlx-turboquant-kv-validate \
  --bin ironmlx-core-bench
```

It then writes:

| Artifact | Content |
| --- | --- |
| `logits.json` | baseline-token replay metrics for logits drift |
| `core-bench/<kv>.json` | in-process generation benchmark records |
| `core-bench/<kv>.time.txt` | `/usr/bin/time -l` memory and timing output |
| `summary.md` | compact report generated from the raw artifacts |

`ironmlx-core-bench` can also be used directly with `--kv-quant none`,
`--kv-quant turbo3`, `--kv-quant turbo4`, or `--kv-quant k3v4`.

The long-context script defaults to `none,turbo4,k3v4`, 32 replay tokens, 32
generation tokens, and generated prompts around 4k, 8k, 16k, and 32k tokens.
It also reports macOS `peak memory footprint`, which is more representative of
MLX memory pressure than RSS for these runs.

## 2026-06-09 Result

Raw artifacts are stored under:

```text
docs/benchmarks/turboquant-kv/2026-06-09-233012/
```

Model:

```text
/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
```

Logits replay, 36 prompt tokens and 16 replay tokens:

| kv | exact-prefix tokens | first mismatch step | argmax matches | avg max abs | avg mean abs | avg rms | min cosine | avg top5 overlap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 16 | none | 16/16 | 0.000000 | 0.000000 | 0.000000 | 1.000000000 | 5.00 |
| `turbo3` | 6 | 6 | 15/16 | 1.633057 | 0.222057 | 0.279114 | 0.988286500 | 4.56 |
| `turbo4` | 16 | none | 16/16 | 0.905365 | 0.128348 | 0.161345 | 0.996525000 | 4.81 |
| `k3v4` | 16 | none | 16/16 | 1.162445 | 0.154378 | 0.195189 | 0.993040400 | 4.50 |

Core generation, 32 generated tokens, 3 measured runs:

| kv | generated tokens | ttft p50 ms | e2e p50 ms | tps p50 | max RSS GiB | exact-prefix vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 32 | 31.05 | 234.20 | 152.56 | 2.454 | 32 |
| `turbo3` | 32 | 31.58 | 242.66 | 146.89 | 2.455 | 2 |
| `turbo4` | 32 | 31.74 | 243.35 | 146.44 | 2.452 | 32 |
| `k3v4` | 32 | 31.60 | 243.51 | 146.29 | 2.455 | 2 |

## Short Prompt Recommendation

Use `none` as the default unless the caller explicitly enables TurboQuant KV.
For quality-sensitive TurboQuant KV usage, prefer `turbo4` in the current
mainline because it matched the baseline token sequence in both replay and
independent generation for this validation prompt.

`k3v4` remains a useful candidate for longer-context or memory-pressure
validation. It matched the baseline argmax sequence during the 16-token logits
replay, but the independent 32-token generation diverged after the first two
tokens in this short prompt. The generated text remained coherent, so this
result should be treated as a stability signal rather than a quality rejection.

`turbo3` has the largest drift in this matrix and should be limited to explicit
experiments where the additional compression is worth the quality risk.

The RSS measurements in this short validation are not representative of
long-context KV cache savings because model weights dominate process memory.
Use longer prompts and larger decode windows before making a memory-efficiency
decision from RSS alone.

## 2026-06-10 Long Context Result

Raw artifacts are stored under:

```text
docs/benchmarks/turboquant-kv/2026-06-10-long-context/
```

The matrix used the same Qwen3.5 4B MLX 4-bit model as the short validation.
Prompt sizes were measured by the ironmlx tokenizer at runtime:

| context | prompt tokens |
| --- | ---: |
| `ctx-4k` | 4,735 |
| `ctx-8k` | 9,399 |
| `ctx-16k` | 18,727 |
| `ctx-32k` | 37,383 |

Logits replay, 32 replay tokens:

| context | kv | exact-prefix tokens | first mismatch step | argmax matches | min cosine |
| --- | --- | ---: | --- | ---: | ---: |
| `ctx-4k` | `turbo4` | 32 | none | 32/32 | 0.987734560 |
| `ctx-4k` | `k3v4` | 32 | none | 32/32 | 0.968668340 |
| `ctx-8k` | `turbo4` | 24 | 24 | 31/32 | 0.974690900 |
| `ctx-8k` | `k3v4` | 32 | none | 32/32 | 0.945408640 |
| `ctx-16k` | `turbo4` | 32 | none | 32/32 | 0.971678900 |
| `ctx-16k` | `k3v4` | 30 | 30 | 31/32 | 0.967671800 |
| `ctx-32k` | `turbo4` | 32 | none | 32/32 | 0.985120700 |
| `ctx-32k` | `k3v4` | 30 | 30 | 31/32 | 0.967162550 |

Core generation, 32 generated tokens:

| context | kv | e2e p50 ms | tps p50 | peak footprint GiB | exact-prefix vs none |
| --- | --- | ---: | ---: | ---: | ---: |
| `ctx-4k` | `none` | 1,263.01 | 144.81 | 5.754 | 32 |
| `ctx-4k` | `turbo4` | 1,487.77 | 73.27 | 6.613 | 32 |
| `ctx-4k` | `k3v4` | 1,479.19 | 74.77 | 6.604 | 24 |
| `ctx-8k` | `none` | 2,376.48 | 137.31 | 7.357 | 32 |
| `ctx-8k` | `turbo4` | 2,898.28 | 49.67 | 9.193 | 32 |
| `ctx-8k` | `k3v4` | 2,977.71 | 48.65 | 9.168 | 30 |
| `ctx-16k` | `none` | 5,496.15 | 125.01 | 11.378 | 32 |
| `ctx-16k` | `turbo4` | 6,528.13 | 30.23 | 14.565 | 32 |
| `ctx-16k` | `k3v4` | 6,436.61 | 31.19 | 14.477 | 30 |
| `ctx-32k` | `none` | 14,447.83 | 101.02 | 29.197 | 32 |
| `ctx-32k` | `turbo4` | 16,814.57 | 16.20 | 37.518 | 32 |
| `ctx-32k` | `k3v4` | 16,813.10 | 17.26 | 36.942 | 30 |

All three configurations generated the correct checksum prefix through the
answer fields at every context length. `turbo4` matched the full 32-token core
generation sequence in all contexts. `k3v4` also produced the correct checksum,
but diverged from the baseline in the subsequent reasoning text.

This long-context run did not show memory savings from TurboQuant KV in the
shadow-storage implementation. `TurboQuantKVCache` stored packed K/V beside the
dense `KVCache`, and attention reads materialized dense K/V from the shadow.
That explains the higher peak footprint and lower generation throughput for
`turbo4`/`k3v4` under pressure.

Updated recommendation: keep `none` as the default. Among TurboQuant modes,
`turbo4` is still the best quality/stability candidate. Do not use TurboQuant KV
as a memory-saving default until the dense shadow cache and per-step
materialization overhead are removed or replaced.

## 2026-06-10 Packed KV Result

Raw artifacts are stored under:

```text
docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/
```

This run removes the dense shadow cache from the long-lived `KVCache`: K/V
history is stored only in TurboQuant packed arrays, and attention reads
materialize the logical prefix for the current dense SDPA path.

Logits replay, 32 replay tokens:

| context | kv | exact-prefix tokens | first mismatch step | argmax matches | min cosine |
| --- | --- | ---: | --- | ---: | ---: |
| `ctx-4k` | `turbo4` | 24 | 24 | 31/32 | 0.984687270 |
| `ctx-4k` | `k3v4` | 32 | none | 32/32 | 0.967960830 |
| `ctx-8k` | `turbo4` | 32 | none | 32/32 | 0.985789300 |
| `ctx-8k` | `k3v4` | 32 | none | 32/32 | 0.961803140 |
| `ctx-16k` | `turbo4` | 32 | none | 32/32 | 0.936246200 |
| `ctx-16k` | `k3v4` | 30 | 30 | 31/32 | 0.963063200 |
| `ctx-32k` | `turbo4` | 32 | none | 32/32 | 0.982761600 |
| `ctx-32k` | `k3v4` | 30 | 30 | 31/32 | 0.973182500 |

Core generation, 32 generated tokens:

| context | kv | e2e p50 ms | tps p50 | peak footprint GiB | exact-prefix vs none |
| --- | --- | ---: | ---: | ---: | ---: |
| `ctx-4k` | `none` | 1,258.48 | 145.66 | 5.758 | 32 |
| `ctx-4k` | `turbo4` | 1,322.25 | 114.94 | 6.075 | 32 |
| `ctx-4k` | `k3v4` | 1,322.04 | 114.89 | 6.069 | 24 |
| `ctx-8k` | `none` | 2,397.55 | 138.14 | 7.356 | 32 |
| `ctx-8k` | `turbo4` | 2,564.88 | 95.13 | 8.022 | 32 |
| `ctx-8k` | `k3v4` | 2,612.18 | 95.98 | 8.012 | 32 |
| `ctx-16k` | `none` | 5,445.32 | 127.39 | 11.377 | 32 |
| `ctx-16k` | `turbo4` | 5,757.16 | 71.56 | 12.440 | 32 |
| `ctx-16k` | `k3v4` | 5,718.69 | 70.87 | 12.422 | 32 |
| `ctx-32k` | `none` | 13,738.60 | 105.72 | 29.196 | 32 |
| `ctx-32k` | `turbo4` | 14,323.92 | 47.19 | 32.166 | 32 |
| `ctx-32k` | `k3v4` | 14,403.96 | 47.25 | 32.071 | 30 |

Compared with the shadow-storage run, packed KV reduces TurboQuant peak
footprint and end-to-end latency under long-context pressure. At `ctx-32k`,
`turbo4` peak footprint drops from 37.518 GiB to 32.166 GiB, and `k3v4` drops
from 36.942 GiB to 32.071 GiB. The remaining gap versus `none` is expected
because the current SDPA path still materializes a dense attention prefix in
addition to the packed cache.

Packed KV is therefore a useful intermediate step, but it is not yet the
preferred default for memory savings. The next optimization candidate is a
packed-attention path that can consume TurboQuant cache blocks without
materializing the full dense prefix per step.
