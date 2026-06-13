# TurboQuant Q-Rotate Dispatch Elimination Design

Date: 2026-06-12

Branch: `codex/turboquant-qrotate-dispatch-elim-design`

Base: `6254ce5 docs: record turboquant qrotate sweep`

## Context

The current TurboQuant K3V4 decode path already stores K/V history in packed
TurboQuant form and runs packed attention without materializing dense K/V. The
remaining measured bottleneck is the standalone query rotation stage:

- 32k K3V4 baseline profile: `decode_time_ms.mean = 1523.012`
- `q_rotate` total: `461.693 ms`
- `q_rotate` mean: `916.1 us`
- observed shape: `B=1`, `Hq=16`, `Hkv=4`, `D=256`, `seq_len=37446`

The previous simdgroup-per-query kernel sweep did not produce an end-to-end
win. The best candidate reduced only the local `q_rotate` total by about 0.8%
and left generation TPS flat or worse. That makes the standalone dispatch and
pipeline boundary, rather than the Hadamard arithmetic alone, the next target.

## Current Decode Data Flow

For Qwen3.5 gated full-attention decode, `GatedAttention::forward_on` currently
performs:

1. `q_proj`, `k_proj`, `v_proj`
2. reshape `q_proj` output to `[B, S, Hq, D * 2]`
3. split into query half and gate half
4. apply `q_norm` to the query half
5. transpose query to `[B, Hq, S, D]`
6. apply MRoPE to Q and K
7. write post-MRoPE K/V into the TurboQuant cache
8. run packed TurboQuant decode attention
9. inside packed attention, launch a separate `q_rotate` kernel:
   `q_rot = WHT(signs * query) / sqrt(D)`
10. run QK, softmax, weighted V, and inverse V rotation

K is quantized after MRoPE with the same TurboQuant key rotation. Query rotation
therefore must be applied to the post-MRoPE, post-q_norm query vector to preserve
dot products in the rotated K basis.

## Rejected Approach: Fold Q-Rotate Into `q_proj` Weights

Do not implement q projection weight folding for the current Qwen3.5 path.

Exact folding is blocked by three properties of the model path:

1. `q_norm` sits between `q_proj` and MRoPE. RMSNorm computes a vector-dependent
   scale and then applies a learned per-channel gamma. The TurboQuant Hadamard
   rotation mixes channels, so it does not commute with the learned diagonal
   gamma unless gamma is a scalar, which is not guaranteed.
2. MRoPE is position-dependent and only rotates `rot_dim = head_dim *
   partial_rotary_factor`. For the Qwen3.5 4B shape this is `64` of `256`
   channels. TurboQuant's random Hadamard rotation spans all `256` channels, so
   it mixes the rotary and non-rotary tail. A fixed q_proj weight transform
   cannot represent `R_tq * MRoPE(position) * q_norm(q_proj(x))`.
3. `q_proj` output contains both query and gate halves, with the query/gate split
   depending on the per-head layout. Any weight rewrite would have to transform
   only the query half while leaving the gate half bit-identical. Quantized MLX
   weights add another barrier: rewriting rows would require dequantize,
   transform, and re-quantize at load time, introducing quality and load-time
   risks.

This path would either be approximate or would require replacing q_norm/MRoPE
semantics. That is not acceptable for a quality-sensitive TurboQuant path.

## Rejected Approach: Fuse Q-Rotate Into QK Directly

Do not move the Hadamard query rotation into the existing QK score kernel as a
local per-score operation.

The current QK kernel maps simdgroups to individual `(batch, q_head, position)`
scores. Query rotation is per `(batch, q_head)` and should be reused across all
positions. Computing it inside each score or small position tile would repeat
the WHT across thousands of positions and likely lose far more than the
standalone dispatch currently costs.

A larger QK kernel redesign that computes one query rotation and many positions
per threadgroup is possible in theory, but it is a broader attention kernel
architecture change. It should not be the first dispatch-elimination attempt.

## Recommended Approach: Decode-Only MRoPE Plus Q-Rotate Fusion

Implement a decode-only fused path that produces a TurboQuant-ready query during
MRoPE application:

- keep the existing dense/prefill MRoPE path unchanged;
- only activate when TurboQuant packed decode attention is known to be usable:
  `seq == 1`, TurboQuant cache enabled, supported mask shape, `head_dim` power of
  two, and `value_head_dim == head_dim`;
- for Q heads, compute MRoPE and then apply the TurboQuant key rotation in the
  same dispatch:
  `q_tq = WHT(signs * MRoPE(q_norm(q_proj(x)), position)) / sqrt(D)`;
- for K heads, produce the normal post-MRoPE K that the existing TurboQuant cache
  writer will quantize;
- call a packed decode attention entry point that accepts pre-rotated queries and
  skips the standalone `q_rotate` stage;
- preserve the existing path as the fallback for dense attention, prefill,
  unsupported masks, or non-TurboQuant caches.

This is the smallest exact design that removes the `q_rotate` dispatch boundary.
It does not change the mathematical basis used by K/V storage and does not
require rewriting checkpoint weights.

## Required Interface Changes

The production implementation should introduce narrow, explicit APIs instead of
overloading the current functions with ambiguous query state:

- `KVCache` should expose an early shape-only predicate for TurboQuant decode
  eligibility before MRoPE is applied.
- `TurboQuantKVCache` should expose read-only accessors for key signs and bit
  metadata needed by the fused MRoPE query path.
- `Mrope` should expose a decode-only fused method that returns
  `(queries_turbo_rotated, k_rope)`.
- `mlx::fast::turboquant_sdpa_decode_parallel_on` should get a sibling entry
  point that accepts already-rotated queries and starts at the QK stage.
- profile JSONL should distinguish the new fused stage, for example
  `mrope_qrotate`, and the old `q_rotate` stage should disappear on the fused
  path.

## Correctness Gates

The implementation must prove exact path equivalence before benchmark work:

1. Add a small synthetic test where current decode and fused-MRoPE decode produce
   the same packed TurboQuant attention output within the existing tolerance.
2. Reuse the dense-materialized TurboQuant reference test for K3V4.
3. Run the real-model TurboQuant logits drift validator against the previous
   K3V4 quality matrix.
4. Verify fallback behavior for non-decode, dense cache, unsupported mask, and
   TurboQuant-disabled requests.

## Performance Gates

Use the existing 32k K3V4 long-context prompt and the same profile mode:

```bash
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx \
  target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text --kv-quant k3v4 --max-tokens 64 --runs 1 --warmup-runs 1 \
  --prefill-chunk-size 2048 --b-max 1
```

The first implementation should be retained only if all of the following hold:

- `q_rotate` is absent from the fused profile path;
- new fused MRoPE/query-rotate cost is lower than old `mrope_apply + q_rotate`;
- end-to-end decode time improves by at least 10% on the 32k/K3V4 profile;
- no measurable TTFT regression for prefill-heavy runs;
- K3V4 quality validation remains within the previously accepted drift bounds.

If the gain is under 5%, revert the implementation and keep only the benchmark
record. If the gain is between 5% and 10%, keep it only if repeated runs show low
variance and no scheduler-level regression.

## Implementation Order

1. Add tests and an internal API for "already TurboQuant-rotated query" packed
   decode.
2. Add the decode-only fused MRoPE plus q-rotate kernel.
3. Thread the early TurboQuant eligibility decision through `GatedAttention`.
4. Run synthetic equivalence tests.
5. Run real-model K3V4 quality validation.
6. Run 32k long-context profile and compare against
   `2026-06-12-qrotate-fusion-opt`.

## Decision

Proceed with the recommended decode-only MRoPE plus q-rotate fusion only after a
separate implementation approval. Do not pursue q_proj weight folding for the
current Qwen3.5 TurboQuant path.
