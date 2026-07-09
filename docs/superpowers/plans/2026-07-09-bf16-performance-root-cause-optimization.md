# bf16 decode performance root cause and optimization

## Goal

Make Qwen3.5 dense bf16 decode performance explainable and improve production
behavior where the fix is supported by direct measurements. The target scenario
is agent-style long prompt serving, with emphasis on decode and high
concurrency after long prefill.

## Evidence already established

- Strict HTTP decode matrix shows `Qwen3.5-4B-MLX-bf16` is materially slower
  than 4-bit and 8-bit on fixed 512-token output:
  - bf16 c1 TPOT: `19.655 ms`
  - 8-bit c1 TPOT: `11.445 ms`
  - 4-bit c1 TPOT: `7.510 ms`
- Dense bf16 Linear layout A/B on real Qwen3.5 weights showed forced
  row-major pretranspose is not a safe production-wide change:
  - `gate_proj` decode improves about 13-14%.
  - `down_proj` decode c1 regresses about 39%, despite c8 improving.

## Working hypotheses

1. The issue is primarily per-token dense bf16 arithmetic volume and MLX matmul
   dispatch shape, not checkpoint loading correctness or output length variance.
2. Current fp `Linear` uses rank-preserving matmul on `[B, S, K]`; decode uses
   `[B, 1, K]`. MLX may choose a less efficient batched matmul path than a
   flattened `[B*S, K] @ [K, N]` path for some projection shapes.
3. Tied embedding output projection may be a meaningful fraction of decode
   time because Qwen3.5 vocab is large.
4. Full-attention cache width is already protected by the 256 minimum cap, so a
   new bf16 slowdown should be proven before touching KV cache policy.

## Plan

1. Add focused diagnostics that time full decode, layer blocks, MLP/attention,
   fp Linear shape variants, and lm_head on real `Qwen3.5-4B-MLX-bf16`.
2. Compare baseline fp Linear behavior against a flatten-then-reshape variant
   without changing production behavior first.
3. If flattening is consistently beneficial and numerically equivalent, add a
   scoped production change for fp `Linear` only, with unit coverage for rank-3
   shape preservation and output equivalence.
4. Rerun direct real-model decode benchmarks for bf16 and compare against the
   pre-change strict HTTP baseline.
5. Run the required Rust verification gate:
   - `cargo fmt`
   - `cargo +nightly fmt --all -- --check`
   - `cargo +nightly clippy --all-features --workspace -- -D warnings`
   - `cargo build --release`

## Non-goals

- Do not migrate custom quant kernels to TensorOps in this task.
- Do not optimize OptiQ in this task.
- Do not add compatibility branches for old checkpoint layouts.
