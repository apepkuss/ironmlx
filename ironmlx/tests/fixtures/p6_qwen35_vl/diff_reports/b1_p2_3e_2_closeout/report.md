# B1-p2.3e.2 PRNG Key Centralization — Close-out

**Branch:** `ironmlx-b1-p2-3e2-prng-centralization`
**Base:** `ironmlx-b1-p2-3e1b-vectorize-configured` HEAD `d146d60`
**Spec:** `docs/superpowers/specs/2026-05-17-b1-p2-3e-2-prng-key-batching-design.md` (`d4c0fc3`)
**Plan:** `docs/superpowers/plans/2026-05-18-b1-p2-3e-2-prng-key-centralization.md` (`e3a152a`)

## Goal Recap

Remove `Sampler.key: Cell<Option<Array>>` field. Centralize per-row PRNG state in
`Scheduler.prng_state: Array` shape `[b_max, 2]` u32. Sampler becomes pure config POD
with auto-derived `Clone + Copy + Send + Sync`. `sample_row_cpu` / `configured_pipeline` /
`sample_batch` / `Sampler::sample` signatures gain `prng_state: &mut Array` parameter.

## Commits

- `45a49ad` chore(b1-p2.3e.2-t0): mlx slice_update probe + bench (274 ms/call -> batch-end stack)
- `2bc80de` refactor(b1-p2.3e.2-t1): Sampler struct shrink -- remove Cell, derive Copy
- `161aefd` feat(b1-p2.3e.2-t2): Scheduler.prng_state + batch-end stack plumbing
- `fdb88f5` chore(b1-p2.3e.2-t3): verify + polish -- bounds checks + stale comments
- `<T4 close-out SHA>` docs(b1-p2.3e.2-t4): close-out report

## Acceptance Gates

| Gate | Result | Notes |
| --- | --- | --- |
| Sampler: Send + Sync + Clone + Copy | PASS static_assertions | T1 const block |
| cargo test --lib | PASS 268 | post-T3; 8 ignored |
| Hygiene (fmt / clippy / build) | PASS all green | every commit |
| 3e.1b perf gate (reused) | PASS (isolated) | medians=[81.73ms x4] max=81.73ms ratio=1.00x; 272ms in sweep_smoke env = GPU degradation (see note) |
| sweep_smoke (4 suites) | 3/4 PASS | b1_p2_3b_2 + b1_p2_4::mid_admit + 3e.1a PASS; 3e.1b perf gate FAIL@272ms (env degradation after 554s b1_p2_4 run -- not regression) |
| sweep_full (17 suites) | in progress | started 11:29 JST, PID 57478 -- Appendix pending |

## Performance Characterization

- 3e.1b baseline perf gate: 82.57ms median per-token at B=4
- 3e.2 perf gate (isolated, before sweep_full started): 81.73ms median, ratio=1.00x
- sweep_smoke 4th suite: 272ms -- environmental degradation after b1_p2_4_batched_vl 554s run
  (same pattern diagnosed in 3e.1b: continuous Metal/GPU load degrades perf gate by 3-4x)
- T0 slice_update bench: 274 ms/call -> use batch-end stack alternative; 1 GPU op
  (mlx::ops::shape::stack) vs B x slice_update saves ~1.1s/step at B=4

## Architecture Notes

### Sampler struct (post-3e.2)
Pure config POD. No interior mutability. Auto-derived #[derive(Debug, Clone, Copy)].
Static assertions in const block verify Send + Sync + Clone + Copy. PRNG state
moved to Scheduler (batched) or GenerationStream (single request).

### Scheduler.prng_state
Shape [b_max, 2] u32 zero-init in Scheduler::new. Row i populated on admit via
init_row_prng(i, sampler.seed) which calls mlx::random::key(seed) and writes via
to_vec + host-replace + try_from pattern (NOT slice_update, which T0 measured at
274 ms/call). Eviction does not clear; next admit overwrites (R2 mitigation).

### configured_pipeline batch-end stack pattern
1. GPU stage: penalties + temp + top_k + softmax fused -> 1 to_vec sync
2. CPU stage start: read prng_state.to_vec() ONCE (~80us for [B, 2] u32 host transfer)
3. Per-row: build [2] u32 Array from host slice -> sample_row_cpu(.., &mut row_key) ->
   collect updated row_key into Vec<Array>
4. CPU stage end: mlx::ops::shape::stack(&refs, 0) -> reassign *prng_state. 1 GPU op
   replaces B x 274ms = 1.1s/step slice_update overhead.

### sample_row_cpu signature
(probs: &[f32], top_p: f32, min_p: f32, prng_state_row: &mut Array) -> Result<u32>.
No &Sampler. PRNG advance: mlx::random::split(prng_state_row) -> (next, sample_key);
write *prng_state_row = next; uniform draw uses sample_key.

### admit_mid_finalize (B=1)
Slice prng_state[row_idx] to host, build single-row Array, call Sampler::sample
with &mut row_key, write back via write_row_prng helper (same pattern).

### GenerationStream (generate.rs)
Single-request non-batched path. Holds own prng_state: Array ([2] u32) field, init
from request.sampler.seed in constructor. Two call sites at generate.rs:1118 +
generate.rs:1302 pass &mut self.prng_state to Sampler::sample.

### sample_async_greedy retained
Plan §1.2.1 claimed "0 callers -- delete", but T1 implementer found 2 production callers
in generate.rs:1088, 1231. Correctly retained since greedy argmax path needs no PRNG state.

## Carry-Forward

- Qwen3.5 MoE -- main path forward; sampler vectorization series complete
- (Optional, Boss-approved 3e.3 prep) T5: put_along_axis exposure +
  apply_top_p_batched refactor -- prepares 80% of future 3e.3 GPU re-enable
- (Optional) GPU sample re-enable in 3e.3: requires categorical Metal kernel JIT
  investigation + put_along_axis in apply_top_p_batched (T5 prep)
- (Optional) Sweep_full hygiene -- cooldown / shard for parallel runs (3e.1b carry-forward)
