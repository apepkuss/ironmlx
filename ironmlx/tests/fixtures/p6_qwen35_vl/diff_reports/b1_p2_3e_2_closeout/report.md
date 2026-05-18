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
- `3fe23f1` docs(b1-p2.3e.2-t4): close-out report (initial)
- `9cf2625` feat(b1-p2.3e.2-t5): expose put_along_axis + refactor apply_top_p_batched
- `<this commit>` docs(b1-p2.3e.2-t4): close-out addendum -- sweep_full + isolation + GPU resource verify

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

## T5 -- put_along_axis exposure (Boss-approved Option C, commit 9cf2625)

T5 expanded 3e.2 scope (Boss Option C) to prep 80% of future 3e.3 GPU re-enable:

- mlx-sys FFI bridge: cxx wrap for mlx::core::put_along_axis (ops.h:1089)
  - shim/include/cxx_mlx_shim/array.h + shim/src/array.cc mirror take_along_axis pattern
  - src/bridge/array.rs cxx bridge declaration
- mlx safe Rust wrapper (mlx/src/ops/indexing.rs):
  - pub fn put_along_axis(a, indices, values, axis) -> Result<Array>
  - pub fn put_along_axis_on(...) stream variant
  - Array::put_along_axis + Array::put_along_axis_on methods (mirror take_along_axis)
  - 2 unit tests PASS: identity round-trip + inverse of take
- ironmlx refactor (apply_top_p_batched #[cfg(test)] block):
  - Old: argsort(sort_idx) -> take_along_axis 2-step inverse-permutation workaround (3e.1b T0 era)
  - New: put_along_axis(&zeros, &sort_idx_desc, &sorted_masked, -1) -- 1 native op, semantics clearer
  - Existing apply_top_p_batched_keeps_nucleus_first_crossing_retained test still PASS
- Production CPU path UNCHANGED: configured_pipeline 仍 sample_row_cpu after GPU softmax sync. T5 does NOT re-enable GPU production path.

## Sweep_full Result + Environment Diagnosis

sweep_full.sh started 2026-05-18 11:29 JST (PID 57478), completed 12:06 JST in
**37m 27s** -- significantly faster than 3e.1b's 4h 29min and close to 3e.1a baseline 65min.
This sweep ran on a fresher system state (~12h continuous load vs 3e.1b's 18+h).

**Result: 15/16 PASS in single run** (16 base suites + 1 b1_p2_3c+ extension = 17 total). Only FAIL:

- b1_p2_3c_plus_chunked_admit_mid::chunked_admit_mid_stall_delta -- 665s timeout-style
  failure (test's timing-sensitive stall delta assertion exceeded under accumulated load).

Isolation re-run on idle system immediately after sweep completed:

| Suite | sweep_full | Isolation | Speedup |
| --- | --- | --- | --- |
| b1_p2_3c_plus_chunked_admit_mid | FAIL 665s (stall_delta assert) | **PASS 38.5s** | 17x |

**Effective 17/17 PASS** across 2-stage validation. **Zero 3e.2 code regression.**

Compare: 3e.1b sweep had 3 failures (batched_vl hung 76min + p4_http_smoke timeout +
chunked_admit_mid FAIL), all confirmed-environmental via isolation. 3e.2 has just 1
failure -- system is recovering toward 3e.1a baseline. Pattern consistent with 3e.1b
carry-forward observation (sweep cumulative load degrades timing-sensitive integration tests).

## GPU / Metal Resource Release Verification (Boss request 2026-05-18)

Post-sweep_full + isolation re-run, system inspected for proper GPU resource release:

| Check | Result | Notes |
| --- | --- | --- |
| ironmlx test processes alive | **0** | pgrep -fl cargo / target/release/deps/b1_p2 / ... empty |
| Zombie processes | **0** | ps -o stat \| grep Z empty |
| Free memory pages | 899570 (14.4 GB free of 32 GB) | Slight gain vs pre-sweep 14.2 GB |
| Top memory consumers | All GUI apps (Notion / Chrome / VSCode) | Nothing from ironmlx |
| Allocation latency probe (probe_slice_update_per_row_round_trip) | **1.86s** total (mostly compile) | Metal kernel cache healthy |
| Kernel cache health (apply_temperature_scales_per_row) | **0.04s** | Fast cache hit -- no JIT recompile |
| Swap usage | 4.5 GB used / 6.0 GB total | Normal Mac state under 12h+ load |

**Conclusion: GPU/Metal resources cleanly released.** All sweep_full test binaries
exited cleanly via OS process termination (Apple Silicon UMA memory automatically
reclaimed). Metal kernel cache state remains healthy post-sweep -- subsequent unit
tests show normal sub-second execution. No leak indicators.

## Carry-Forward

- Qwen3.5 MoE -- main path forward; sampler vectorization series complete (3e.1a -> 3e.1b -> 3e.2)
- (Optional, future 3e.3) GPU sample re-enable -- T5 已 prepared put_along_axis path. Remaining work:
  1. Re-bench GPU apply_top_p_batched (with put_along_axis) isolated -- quantify Metal kernel cache cost
  2. Trace mlx::random::categorical([B, vocab]) Metal kernel JIT behavior at vocab=151k
  3. If both yield significant wins (<80ms median) -> re-enable GPU production path in configured_pipeline
- (Optional) Sweep_full hygiene -- cooldown / shard for parallel runs (recurring across 3e.1b + 3e.2)
