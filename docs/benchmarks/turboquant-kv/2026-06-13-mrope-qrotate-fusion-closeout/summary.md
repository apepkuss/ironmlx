# TurboQuant MRoPE / Q-Rotate Fusion Closeout

Date: 2026-06-13

Worktree: `/Users/xin/workspace/ironmlx-backend-turboquant-mrope-qrotate-fusion`

Branch: `codex/turboquant-mrope-qrotate-fusion`

## Final Retained State

- TurboQuant KV is supported through the existing CLI/API plumbing for K3V3, K3V4, and K4V4.
- The retained long-context decode path uses pre-rotated packed attention.
- The retained packed attention kernel constants are:
  - `QK_SIMDGROUPS_PER_THREADGROUP = 4`
  - `V_CHUNK_DIMS_PER_THREADGROUP = 16`
  - `PARALLEL_DECODE_V_CHUNK_SIZE = 256`
- The branch keeps benchmark evidence for rejected candidates, but the rejected experimental kernels were reverted from implementation.

## Retained Optimization

| Work | Commit | Decision | Main 32K K3V4 Result |
| --- | --- | --- | --- |
| Packed attention V chunk tuning | `d5e7e87` | Retain `V_CHUNK_DIMS_PER_THREADGROUP = 16` | p50 generation TPS `59.6725 -> 65.0343`, `+8.99%`; decode p50 `1055.762 ms -> 968.719 ms`, `-8.24%` |

Evidence:

- `docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/summary.md`

## Rejected Spikes

| Work | Commit | Decision | Main 32K K3V4 Result |
| --- | --- | --- | --- |
| QK simdgroup density `4 -> 8` | `72d5373` | Reject; keep `QK_SIMDGROUPS_PER_THREADGROUP = 4` | p50 generation TPS `65.7666 -> 65.1662`, `-0.91%` |
| Fused QK/softmax/V chunk kernel | `6a4745f` | Reject and remove fused candidate | p50 generation TPS `65.7666 -> 21.5486`, `-67.23%` |
| Score-driven V decode path | `dfed886` | Reject and remove score-driven V candidate | p50 generation TPS `65.7666 -> 65.7642`, `-0.004%` |

Evidence:

- `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/summary.md`
- `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/summary.md`
- `docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/summary.md`

## Interpretation

The successful optimization is the V chunk dimensional packing change. The rejected candidates reduced some profiled micro-stage timings, but did not improve the no-profile generation gate. The fused-kernel attempt was especially poor because it collapsed too much parallelism into a chunk-local kernel and lost the retained path's parallel QK and V-dimension work split.

## Recommended Next Work

Before starting a new kernel spike, run a clean final benchmark from the retained branch state:

- 32K K3V4, `--max-tokens 64 --runs 3 --warmup-runs 1`
- Optional confirmation for K4V4 and K3V3 if cross-bit coverage is needed

If the retained result is stable, the next optimization target should be selected from fresh decode-layer attribution rather than another attention-only fusion attempt.
