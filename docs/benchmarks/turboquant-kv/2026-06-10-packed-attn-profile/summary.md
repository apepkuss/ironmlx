# TurboQuant Packed Decode Attention Profiling

- branch: `codex/turboquant-packed-attn-profile`
- base: `126e00e` (`codex/turboquant-packed-attn-opt`)
- model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- prompt: `ctx-32k.txt` (`37383` prompt tokens)
- kv quant: `k3v4`
- max tokens: `32` for default-path regression, `8` for stage profiling

## Conclusion

This phase did not keep a new decode-kernel optimization. The useful result is a low-overhead profiling switch for solution 2:

- default path remains equivalent to `codex/turboquant-packed-attn-opt`
- profiling is enabled only with `IRONMLX_TURBOQUANT_ATTN_PROFILE=1`
- profiled path emits JSONL stage events to stderr
- the next high-value optimization targets are still `weighted_v_chunk` and `qk`

The first profiler implementation put stage probes directly in the default dispatch path. Even with profiling disabled, that changed the lazy graph shape enough to reduce `k3v4` decode from about `22.30` TPS to `17.15` TPS. The final implementation keeps the original dispatch body for the normal path and calls a separate `*_profiled` dispatch only when the env var is set.

## Default Path Regression Check

| run | decode ms | TPS | peak footprint GiB | generated tail |
| --- | ---: | ---: | ---: | --- |
| current gated profiler | 1390.19 | 22.30 | 30.290 | `validate long-context` |
| optimized solution 2 rerun | 1389.91 | 22.30 | 30.294 | `validate long-context` |
| historical optimized solution 2 | 1389.84 | 22.30 | 30.294 | `validate the KV` |
| inline profiler attempt | 1807.48 | 17.15 | 30.290 | `validate the KV` |

The final `current gated profiler` run matches the optimized solution 2 rerun within measurement noise. The last two generated tokens are a known sensitive boundary in this prompt; the stable prefix is unchanged through token 30.

## Stage Profile

Command shape:

```bash
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 \
MLX_DIR=$HOME/.local/mlx \
target/release/ironmlx-core-bench \
  --mode gs-text \
  --kv-quant k3v4 \
  --max-tokens 8 \
  --prefill-chunk-size 2048 \
  --b-max 1
```

The profiler forces `eval` after each stage, so absolute TPS from profiled runs is not comparable to the default path. It is intended for locating relative stage cost.

| stage | events | total ms | mean us | p50 us | p95 us | max us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `weighted_v_chunk` | 56 | 152.533 | 2723.80 | 2714 | 2810 | 2901 |
| `qk` | 56 | 138.695 | 2476.70 | 2464 | 2573 | 2650 |
| `q_rotate` | 56 | 60.081 | 1072.88 | 968 | 1003 | 8105 |
| `weighted_v_reduce` | 56 | 10.909 | 194.80 | 192 | 203 | 309 |
| `softmax` | 56 | 10.871 | 194.12 | 190 | 217 | 272 |

## Rejected Experiments

| experiment | decode ms | TPS | result |
| --- | ---: | ---: | --- |
| `V_CHUNK_SIZE=128` | 2028.10 | 15.29 | slower than 256 |
| `V_CHUNK_SIZE=512` | 2034.90 | 15.23 | slower than 256 |
| `vdim4` weighted-V kernel | 2285.67 | 13.56 | slower; not retained |

These results suggest that simply changing V chunk size or computing four V dimensions per threadgroup is not the right path. The next optimization should focus on the QK kernel and a more deliberate weighted-V reduction/fusion strategy.

## Artifacts

- default regression: `ctx-32k/profile/k3v4-32tok-final-profile-gated.json`
- optimized branch rerun: `ctx-32k/profile/k3v4-32tok-opt-rerun.json`
- profiler sample: `ctx-32k/profile/k3v4-8tok-profile-gated.stderr.txt`
- rejected experiments: `ctx-32k/profile/k3v4-32tok-vchunk128.json`, `ctx-32k/profile/k3v4-32tok-vchunk512.json`, `ctx-32k/profile/k3v4-32tok-vdim4.json`
