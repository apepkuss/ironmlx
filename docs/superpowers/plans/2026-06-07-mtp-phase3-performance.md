# MTP Phase 3 Performance Plan

## Goal

Improve the scheduler MTP path after Phase 2.5 by applying the low-risk performance ideas that follow from the llama.cpp parity study:

- keep the draft MTP cache after full-accept windows instead of restoring and replaying the same accepted prefix;
- use a greedy vectorized sampling fast path for MTP draft/verify logits;
- reduce high-depth damage with an adaptive draft-token budget;
- preserve Phase 2.5 rollback correctness and cache commit semantics.

## Implementation

1. Add explicit MTP draft results that carry both drafted tokens and the pre-draft cache snapshot.
2. Restore the MTP cache only when a speculative window mismatches.
3. On full accept, keep the advanced draft MTP cache and commit only the missing tail token.
4. Track cache reuse and adaptive-budget counters in `MtpSpeculativeStats`.
5. Add scheduler tests for full-accept cache reuse and first-token-mismatch budget reduction.
6. Extend `ironmlx-core-bench` JSON stats so benchmark artifacts expose the new counters.

## Benchmark

The benchmark artifacts are under:

`docs/benchmarks/mtp-phase3-performance/2026-06-07-141108`

The fixed-prompt greedy benchmark shows:

- Qwen3.5-4B: best MTP config is `mtp_d1`, 1.136x over Phase 3 baseline.
- Qwen3.6-27B: best MTP config is `mtp_d2`, 1.591x over Phase 3 baseline.
- Qwen3.6-35B-A3B: best MTP config is `mtp_d2`, 1.098x over Phase 3 baseline.

The most useful Phase 3 gain is Qwen3.6-27B `mtp_d2`, which improves from 45.684 tok/s in Phase 2.5 to 50.245 tok/s in Phase 3.
