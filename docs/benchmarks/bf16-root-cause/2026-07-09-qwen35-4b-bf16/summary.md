# Qwen3.5-4B-MLX-bf16 decode root-cause summary

Date: 2026-07-09

Model:
`/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-bf16/snapshots/475632ded9a95863da4e4b235ab9ccbc5d3cc6bf`

Prompt: fixed agent-style long prompt, 11266 local tokens.

## Baseline finding

The bf16 decode/high-concurrency gap is real and is not explained by HTTP
measurement noise.

| model | c=1 decode TPOT median | c=8 ITL p50 |
| --- | ---: | ---: |
| Qwen3.5-4B-MLX-4bit | 7.510 ms | 7.610 ms |
| Qwen3.5-4B-MLX-8bit | 11.445 ms | 11.666 ms |
| Qwen3.5-4B-MLX-bf16 | 19.655 ms | 19.555 ms |

Relative to affine 8-bit, bf16 is +71.7% at c=1 TPOT and +67.6% at c=8 ITL.

Source:
`docs/benchmarks/bf16-decode-high-concurrency/2026-07-09-223455-strict-decode/`

## Root cause

Layer-level decode profiling shows bf16 is split almost evenly between
attention and MLP paths:

| stage | events | total | mean |
| --- | ---: | ---: | ---: |
| decode_attention_path | 96 | 56.046 ms | 583.8 us |
| decode_mlp_path | 96 | 54.685 ms | 569.6 us |

Within attention, the linear-attention/GDN layers are materially heavier than
full-attention layers:

| attention kind | events | total | mean |
| --- | ---: | ---: | ---: |
| linear attention | 72 | 46.618 ms | 647.5 us |
| full attention | 24 | 9.428 ms | 392.8 us |

This makes the bottleneck a combination of:

- bf16 dense matmul volume in decode;
- Qwen3.5 linear-attention/GDN layer cost;
- MLP path cost at similar magnitude to attention.

Source:
`core_profile_gs_4tok.json` and `core_profile_gs_4tok.stderr.log`

## Rejected optimization candidate

### GDN `b+a` projection fusion

I tested a GDN input-projection fusion candidate that fused the small fp/bf16
`in_proj_b + in_proj_a` projections. It was rejected for production.

Production-like single-layer GDN microbench:

| variant | seq=1 p50 | seq=8 p50 | seq=128 p50 |
| --- | ---: | ---: | ---: |
| no fusion | 0.952 ms | 1.567 ms | 1.061 ms |
| fused b+a | 0.938 ms | 1.389 ms | 1.111 ms |

The microbench looked promising at `seq=8` (-11.4% p50), but real HTTP decode
did not validate it:

| run | c=1 TPOT median | c=8 ITL p50 |
| --- | ---: | ---: |
| baseline bf16 strict decode | 19.655 ms | 19.555 ms |
| fused b+a HTTP check | 22.376 ms | 59.971 ms |

The c=8 aggregate throughput is not directly comparable because the strict
baseline was a 10s run and the validation run was a 30s run. The per-token ITL
is directly comparable enough to reject the change: the fused path regressed
decode ITL by roughly 3x under concurrent HTTP load.

Source:

- `gdn_layer0_bf16_no_fusion_prod.json`
- `gdn_layer0_bf16_fused_small_prod.json`
- `http_seq_tg512_c1_fused_small.json`
- `http_conc_tg512_c8_fused_small.json`

### Dense MLP `gate+up` projection fusion

I also tested the standard dense SwiGLU optimization of fusing `gate_proj` and
`up_proj` into one fp/bf16 matmul followed by a split. This is a common
launch-count reduction, but it was not valid for this serving path.

| run | c=1 TPOT median | c=8 ITL p50 |
| --- | ---: | ---: |
| baseline bf16 strict decode | 19.655 ms | 19.555 ms |
| fused MLP gate+up HTTP check | 22.806 ms | 57.810 ms |

The fused MLP path regressed c=1 TPOT by +16.0% and c=8 ITL by roughly 3x, so
it was removed from production code.

Source:

- `http_seq_tg512_c1_mlp_fused.json`
- `http_conc_tg512_c8_mlp_fused.json`

### Tied embedding output transpose cache

I tested caching a materialized `[dim, vocab]` dense bf16 embedding transpose
for the tied lm_head output projection. The layer-shape microbench suggested
this might improve decode matmul layout, but HTTP A/B did not justify a
production change.

Same-machine HTTP A/B, Qwen3.5-4B-MLX-bf16, PP=11266, TG=512:

| run | c=1 TPOT median | c=1 TTFT | c=8 ITL p50 | c=8 TTFT p50 | c=8 E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline transpose view | 22.477 ms | 2766.506 ms | 72.192 ms | 21704.946 ms | 77.080 s |
| cached materialized transpose | 22.221 ms | 2864.881 ms | 63.557 ms | 37843.024 ms | 86.771 s |

The cached path improved steady-state c=8 ITL, but it materially regressed
TTFT and E2E latency under concurrent long-prompt load. Since the requested
target is production-grade HTTP behavior, this candidate was rejected and
removed from production code.

Source:

- `http_seq_tg512_c1_embedding_output_ab_baseline.json`
- `http_seq_tg512_c1_embedding_output_cache.json`
- `http_conc_tg512_c8_embedding_output_ab_baseline.json`
- `http_conc_tg512_c8_embedding_output_cache.json`

## Code decision

No GDN fusion path, no dense MLP gate/up fusion path, and no tied embedding
output transpose cache path are enabled in production code. The attempted
optimization changes were removed after HTTP validation failed.

The remaining useful output of this pass is diagnostic:

- `ironmlx-gdn-bench` can run against dense Qwen3.5 bf16 checkpoints;
- bf16 decode root cause is narrowed to GDN/linear-attention plus MLP dense path;
- simple MLX-level projection fusion is not a safe production optimization for
  this path;
- tied lm_head output layout alone is not the first production optimization to
  land for this path.

## Next optimization direction

The next production-grade path should target the actual dense bf16 hotspots
rather than metadata-level fusion:

- GDN recurrence and projection layout/kernel path for decode shapes;
- dense MLP decode matmul path;
- tied embedding/lm_head decode cost if it remains visible after layer path
  improvements;
- TensorOps/custom kernel investigation for GDN if MLX default composition
  cannot expose the required layout or launch behavior.
