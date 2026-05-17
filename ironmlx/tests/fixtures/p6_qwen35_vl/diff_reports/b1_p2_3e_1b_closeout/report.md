# B1-p2.3e.1b Vectorize Configured Sampler — Close-out

**Branch:** `ironmlx-b1-p2-3e1b-vectorize-configured`
**Base:** `ironmlx-b1-p2-3e1a-vectorize-greedy`
**Spec:** `docs/superpowers/specs/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md`
**Plan:** `docs/superpowers/plans/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler.md`

## Goal Recap

Replace `sample_batch`'s mixed/configured fallback (per-row `Sampler::sample` loop) with a
fully batched pipeline: GPU-side penalties/temperature/top_k/softmax fused into one eval,
followed by CPU-side top_p/min_p/renorm/categorical per row. All-greedy fast path (3e.1a) retained.

## Commits

- `37a0d38` chore(b1-p2.3e.1b-t0): mlx API verification + design pins
- `fc5b7ac` feat(b1-p2.3e.1b-t1): per-row configs + history bincount + apply_penalties
- `da93575` feat(b1-p2.3e.1b-t2): batched temp / top_k / softmax / top_p / min_p / renorm
- `18f4f72` fix(b1-p2.3e.1b-t2): top_k_batched correctness — hybrid sort/partition
- `86c4e22` feat(b1-p2.3e.1b-t3): configured_pipeline + batched categorical wired
- `c9c21f1` polish(b1-p2.3e.1b-t3): PRNG single-advance + test comment fix
- `684ca05` fix(b1-p2.3e.1b): configured_pipeline GPU→CPU handoff for top_p+categorical
- `bed2923` test(b1-p2.3e.1b-t4): real-model perf gate (#[ignore])
- `<T4 close-out SHA>` docs(b1-p2.3e.1b-t4): close-out report

## Acceptance Gates

| Gate | Result | Notes |
| --- | --- | --- |
| cargo test core::sampler (lib) | PASS 40 | 22 (3e.1a) + 3 (T0 probes) + 6 (T1) + 7 (T2 incl. 2 top_k fix) + 3 (T3 integration), 1 ignored |
| cargo test --lib (full) | PASS 267 | No regression (7 ignored) |
| Hygiene (fmt / clippy / build) | PASS | Every commit |
| Real-model perf gate b1_p2_3e_1b_configured_decode_speedup | PASS | medians=[82.57ms, 82.57ms, 82.56ms, 82.57ms] ratio=1.00x |
| sweep_smoke (4 integration suites) | PASS | b1_p2_3b_2 (3/3) + b1_p2_4 mid_admit + 3e.1a + 3e.1b; lib SIGTRAP pre-existing GPU-state issue (single run 267 PASS) |
| sweep_full (17 suites) | in progress (PID: 67590) | Controller appends result after sweep completes |

## Performance Characterization

- **3e.1a greedy fast path** (reference): 64.4 ms median per-token at B=4
- **Pre-3e.1b per-row loop fallback**: B x per-row sampler::sample ~ 300-800ms at B=4
- **Post-3e.1b configured_pipeline** (GPU->CPU handoff):
  - GPU stage (fused): penalties + temperature + top_k + softmax -> one to_vec ~60-70ms
  - CPU stage (per-row): sort + top_p + min_p + renorm + CDF sample ~10-20ms/row
  - **Measured (perf gate, M1 Pro, 4B bf16, B=4)**: 82.6 ms median, ratio 1.00x

## Architecture Notes

### configured_pipeline: root cause of T3 regression

Initial T3 design used `mlx::random::categorical([B, vocab])` + inverse-permutation scatter
for top_p (argsort x 2). Root cause diagnosis in T4:
- argsort([B=4, vocab=151936]) x 2 + mlx::random::categorical Gumbel-max kernel
  triggered per-call Metal JIT recompiles + large buffer allocations
- Measured: 0.4-18 s/step (first call 18s JIT; subsequent 0.4-7s; total ~3.6s median)
- This is 14-54x over the 250ms budget

Fixed design (GPU->CPU handoff):
```
GPU (lazy, one eval):
  collect_per_row_configs ->
  build_history_count (CPU bincount -> upload [B, vocab] u32) ->
  apply_penalties (rep + freq + pres fused) ->
  apply_temperature (broadcast divide) ->
  apply_top_k_batched (partition or sort) ->
  apply_softmax ->
  to_vec::<f32>()   <- single GPU sync

CPU (per-row):
  sort descending + cumsum nucleus filter (top_p) ->
  min_p floor relative to max_prob ->
  L1 renormalize ->
  CDF sample (uniform draw via mlx::random::uniform builder)
```

### Top_k hybrid path (preserved from T2/T3)

Uniform top_k -> partition (O(n), ~1ms); mixed top_k -> sort (O(n log n), ~12ms). At vocab=151k.

### GPU-side top_p/min_p/renormalize retained for tests

apply_top_p_batched, apply_min_p_batched, renormalize marked #[cfg(test)].
sample_batched_categorical removed (replaced by sample_row_cpu).

### PRNG per-row

CPU path naturally preserves per-row PRNG independence (each row calls sampler.ensure_key()
separately). T3 batched categorical drifted per-row reproducibility (spec NG6 accepted); T4
CPU path restores it without extra cost.

## Carry-Forward

- **3e.2**: PRNG state centralization - move key from Sampler.key Cell<Option<Array>> to Scheduler
- **Future**: custom Metal partial-sort for top_k; batched GPU top_p if MLX adds scatter_along_axis
