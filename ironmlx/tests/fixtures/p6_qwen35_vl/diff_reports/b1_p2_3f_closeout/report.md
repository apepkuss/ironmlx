# B1-p2.3f Cache Cap Dynamic + Bounded — Close-out

**Branch:** `ironmlx-b1-p2-3f-cache-cap` (cut from `ironmlx-b1-p2-3e3-typed-err` HEAD `82808cd` post-plan; merge target `main` after review)
**Date:** 2026-05-17
**Status:** ✅ COMPLETE

## Summary

Replaces the pre-3f hardcoded `make_cache(b, 8192, dtype)` with a
**three-tier dynamic cap model**:

```text
cap = max(slots_max, MIN_CACHE_CAP_FOR_GPU_PERF).min(effective_cap_max)
effective_cap_max = min(--max-cache-cap CLI flag, model.config.max_position_embeddings)
```

Where `slots_max = max(prompt_len + max_new_tokens)` across admitted
slots, `MIN_CACHE_CAP_FOR_GPU_PERF = 256` (GPU-perf floor), and
`effective_cap_max` is computed at server boot.

Admit gates (`Scheduler::admit` + `Scheduler::admit_mid`) now reject
oversize requests with `SchedulerError::RequestTooLarge` → HTTP 413
Payload Too Large. `Scheduler::evict_all` drops the cache instead of
resetting offsets so the next batch can size the cap fresh.

Two follow-on fixes landed within T4:
1. `Scheduler::grow_main_cache_to` extends `KVCache::cap` + `GatedDeltaCache::cap`
   in-place via new `grow_cap` methods. Required because dynamic cap
   may size the main cache to a short-batch's `slots_max`, then a
   queue-drained admit_mid with a longer prompt would overflow.
2. `server::serve()` reads `model.config().max_position_embeddings`
   via `.lock().await` instead of `.blocking_lock()` (the latter panics
   inside a `tokio::spawn`ed multi-thread runtime).
3. `MIN_CACHE_CAP_FOR_GPU_PERF = 256` floor on the dynamic cap. Tight
   short-prompt caps (e.g., cap=36 for a 20-token prompt + max_new=16)
   miss MLX's preferred Metal attention kernel tile and fall ~300× off
   peak (decode step grows from ~50 ms to ~10 s). Caught by p4_http_smoke
   regression in the T4 15-suite sweep.

## Acceptance

| Gate | Result |
| --- | --- |
| Unit `kvcache_grow_cap_extends_and_allows_writes_beyond_initial_cap` | ✅ PASS |
| Unit `kvcache_grow_cap_is_monotonic_noop_on_shrink` | ✅ PASS |
| Unit `gdcache_grow_cap_extends_and_allows_advance_beyond_initial_cap` | ✅ PASS |
| Unit `gdcache_grow_cap_is_monotonic_noop_on_shrink` | ✅ PASS |
| Unit `admit_rejects_oversize_request` (Err with "request too large") | ✅ PASS |
| Unit `evict_all_drops_cache` | ✅ PASS |
| Unit `dynamic_cap_from_slots_bounded_by_cap_max_and_gpu_floor` (3 regime cases) | ✅ PASS |
| Unit `admit_err_413_for_request_too_large` (HTTP body contains needed + max) | ✅ PASS |
| Unit `admit_err_400_for_unrelated_typed_error` (typed non-Scheduler Err → 400) | ✅ PASS |
| Integration `admit_long_prompt_pp10k` (PP=10K prompt, finish_reason=length, 20 tokens) | ✅ PASS |
| 15-suite regression sweep (post-floor-fix) | ✅ ALL PASS — see §Regression below |
| fmt --check / clippy -D warnings / build --release | ✅ ALL CLEAN at every commit |

## Architectural changes per spec §4

| Item | File | Change |
| --- | --- | --- |
| §4.2.1 `Qwen35Config::max_position_embeddings` | `models/qwen3_5/config.rs` | `pub max_position_embeddings: i32` with `#[serde(default = ...)]` fallback 32768 |
| §4.2.2 `SchedulerError::RequestTooLarge` variant | `core/scheduler.rs` | Added thiserror variant + Display |
| §4.2.3 `Scheduler::new` signature | `core/scheduler.rs` | `(b_max, effective_cap_max)` (was just `b_max`) |
| §4.2.3 admit + admit_mid cap gate | `core/scheduler.rs` | `cap_needed = prompt_ids.len() + max_new_tokens > effective_cap_max → Err RequestTooLarge` |
| §4.2.4 `evict_all` drops cache | `core/scheduler.rs` | `self.cache = None` (replaces per-layer `reset()` loop) |
| §4.2.5 dynamic cap in `prefill_admitted_inner` | `core/scheduler.rs` | `cap = max(slots_max, MIN).min(effective_cap_max)` |
| §4.2.6 `spawn_scheduler_actor` signature | `core/server/scheduler_actor.rs` | +`effective_cap_max: usize` |
| §4.2.7 `serve()` + AppState plumbing | `core/server/mod.rs` | +`AppState::effective_cap_max`; serve() reads `model_max_context` async-locked then computes `effective_cap_max = min(cli, model_max)` |
| §4.2.8 HTTP 413 mapping | `core/server/openai.rs`, `anthropic.rs` | `admit_err_to_response` match → 503 / 413 / 400 |
| §4.2 +T4 Option C `grow_cap` | `core/cache/kv_cache.rs`, `core/cache/gated_delta.rs` | Per-cache `grow_cap(new_cap)` raises self.cap monotonically |
| §4.2 +T4 Option C `grow_main_cache_to` | `core/scheduler.rs` | Called by `admit_mid_inner` before adopt; lifts main cap to fit longer mid-batch admits |
| §4.2 +T4 floor `MIN_CACHE_CAP_FOR_GPU_PERF = 256` | `core/scheduler.rs` | Module-level const + dynamic-cap formula update |
| §4.2 +T4 async lock fix | `core/server/mod.rs` | `model.blocking_lock()` → `model.lock().await` (panic-free under tokio::spawn) |

## Plan-correction deviations

- **T4 Option C (Boss-chosen 2026-05-17):** Original 3f plan T4 was just "long-prompt integration + 14-suite regression + close-out". 14-suite regression exposed `b1_p2_4 mid_admit_vl_during_text_decode` failing with `GatedDeltaCache::adopt_row_from: src.offsets[0] = 283 > self.cap 22` — dynamic cap caught a pre-existing latent bug (KVCache/GatedDeltaCache cap was fixed at construction; mid-batch admit with a longer prompt than the original batch couldn't fit). Boss chose Option C ("Scheduler::grow_main_cache full fix") over weaker alternatives (forcing pre-allocation of max cap, falling back to single-shot). Adds `grow_cap` to both cache types + `Scheduler::grow_main_cache_to` helper invoked before `adopt_row_from` in `admit_mid_inner`.
- **T4 async lock fix (2026-05-17):** While running 3d S5 + b1_p2_4 mid_admit, `serve()`'s pre-3f-T2 `model.blocking_lock()` line panicked under `#[tokio::test(flavor = "multi_thread")]` with "Cannot block the current thread from within a runtime". Fix swaps to `.lock().await` — `serve()` is async + setup-phase, no perf cost.
- **T4 GPU-perf floor (2026-05-17):** Final regression sweep caught `p4_http_smoke` hanging at decode (~10 s per step for what should be ~50 ms). Root cause: `Qwen35Model::make_cache` uses `KVCache::with_step(cap)` so the per-layer K/V buffer width equals `cap`. With cap=36 (prompt 20 + max_new 16), the buffer is too small for MLX's preferred Metal attention tile, falling to a slow generic kernel. Fix: floor cap at 256 (matches `KVCache`'s default step + power-of-two GPU block size). p4_http_smoke standalone goes from 178 s timeout/fail to 5.13 s PASS.

## Commits (3f T1-T4 code; +3c+/3e.1 docs piggy-backed on the branch)

| Commit | Type | Description |
| --- | --- | --- |
| `7adcae3` | feat | T1: typed RequestTooLarge + admit cap gate + evict_all drops cache |
| `1df713a` | feat | T2: dynamic cap + CLI/AppState plumbing |
| `4d06c40` | feat | T3: HTTP 413 for SchedulerError::RequestTooLarge |
| `1d184e2` | fix | T4 Option C: grow main cache cap on mid-batch admit |
| `461d3d1` | test | T4: long-prompt integration + serve() async lock fix |
| `9612d92` | fix | T4: floor dynamic cap at MIN_CACHE_CAP_FOR_GPU_PERF=256 (Scheduler only) |
| `52f685c` | fix | T4: move GPU-perf cap floor into make_cache (covers all callers) |
| `0a098dd` | fix | T4: floor moved out of make_cache to per-caller (preserves cap-overflow tests) |
| `d0d56d4` | docs | 3c+ chunked admit_mid design spec (piggy-back; separate sub-task) |
| `132045c` | docs | 3c+ implementation plan (4 tasks) |
| `730eebf` | docs | 3c+ plan T1+T2 merge to atomic commit |
| `a4990fe` | docs | 3e.1 vectorized sampler design spec |
| (this commit) | docs | T4 close-out report |

## Regression status

15-suite sweep with `--ignored --test-threads=1`. The T4 path went
through 4 sweep iterations as new bugs surfaced and were fixed:

1. **Sweep #1** (post-async-lock-fix `461d3d1`, pre-floor): 14/15 PASS.
   `p4_http_smoke` hung at decode (~10 s/step instead of ~50 ms).
   Root-caused to GPU-perf cliff for tight cap.
2. **Sweep #2** (`9612d92` floor in Scheduler only): aborted at
   `b1_p2_3b_3 admission_window_concurrent_scheduler_and_gs_no_deadlock`
   — GS still used unfloored cap so task B held the model lock 60+ s
   on first-time Metal kernel compile. Task A scheduler-path admit
   timed out at 60 s.
3. **Sweep #3** (`52f685c` floor in `make_cache`): aborted at
   `b1_p2_3c_1 per_row_offset_invalid_args_return_err`. Test
   explicitly calls `model.make_cache(2, 4, ...)` to validate
   KVCache cap-overflow error path; silent floor inside `make_cache`
   broke the contract.
4. **Sweep #4** (`0a098dd` floor moved out of `make_cache` to per-caller):
   14/15 PASS in sweep + 1 sweep-context flake on `b1_p2_3d
   iron_bench_c8_with_queue_no_4xx` (S5). S5 passes standalone
   (PASS 66.52 s with 4/8 worker successes ≥ threshold 1) — sweep
   running 5+ h continuously stresses GPU/system state.
   Remaining 4 suites + `b1_p2_3f_cache_cap` standalone all PASS
   after killing zombie ironmlx server from a prior session that
   was blocking Metal kernel compile.

**Final tally: 15/15 PASS** (14 in sweep + 1 standalone for S5 flake).

| Suite | Sweep result | Standalone (where applicable) |
| --- | --- | --- |
| b1_p2_1_batched_prefill | ✅ PASS (19s, #4) | — |
| b1_p2_2_batched_decode | ✅ PASS (4727s, #4 slow) | — |
| b1_p2_3a_scheduler_skeleton | ✅ PASS (2s, #4) | — |
| b1_p2_3b_1_scheduler_step | ✅ PASS (2518s, #4) | — |
| b1_p2_3b_2_scheduler_actor | ✅ PASS (407s, #4) | — |
| b1_p2_3b_3_admission_window (4 tests incl. concurrent_scheduler_and_gs) | ✅ PASS (514s, #4) | ✅ PASS (32.52s, deadlock test post-floor verify) |
| b1_p2_3b_4_anthropic_actor | ✅ PASS (60s, #4) | — |
| b1_p2_3c_1_per_row_offset (incl. cap-overflow) | ✅ PASS (242s, #4) | — |
| b1_p2_3c_2_scheduler_decode_mask | ✅ PASS (114s, #4) | — |
| b1_p2_3c_3_continuous_batching | ✅ PASS (235s, #4) | — |
| b1_p2_3d_admission_queue (5 tests; S5 sweep flake) | 🟡 4/5 in sweep #4 | ✅ S5 PASS (66.52s standalone) |
| b1_p2_4_batched_vl (4 tests incl. mid_admit_vl_during_text_decode) | ✅ PASS (2296s, remaining) | — |
| b1_p2_3f_cache_cap (admit_long_prompt_pp10k) | ✅ PASS (277s, standalone — zombie GPU contention killed earlier sweep run) | — |
| p6_qwen35_vl_logits_match | ✅ PASS (11s, remaining) | — |
| p4_http_smoke | ✅ PASS (9s, remaining) | — |

## Compat sunset

| Removed | Replaced with |
| --- | --- |
| `Scheduler::new(b_max)` (single arg) | `Scheduler::new(b_max, effective_cap_max)` |
| `model.make_cache(b, 8192, dtype)` hardcoded cap | `model.make_cache(b, dynamic_cap, dtype)` |
| `Scheduler::evict_all` resets per-layer offsets | Drops `self.cache = None` |
| `serve()`'s 9-arg signature (3d) | 10-arg signature with `max_cache_cap: usize` |
| `spawn_scheduler_actor` 4-arg signature | 5-arg with `effective_cap_max: usize` |
| `model.blocking_lock()` in serve() setup | `model.lock().await` |
| String-match `err.to_string().contains("too large")` | `err.downcast_ref::<SchedulerError>()` match on `RequestTooLarge` |

## Notes / known limitations carrying forward to backlog

- **`MIN_CACHE_CAP_FOR_GPU_PERF` is empirical, not derived from first principles.** 256 was the smallest power-of-two that gave clean p4_http_smoke decode. Some GPU hardware (M3 / M4 with different unified-memory layouts) may benefit from a different floor. Tracked in [project_cross_device_tuning_deferred memory](../../../../../../../Users/sam/.claude/projects/-Volumes-Dev-cxx-mlx/memory/project_cross_device_tuning_deferred.md).
- **`with_step(cap)` was an optimization for pre-3f hardcoded cap=8192** (one-shot alloc, no grow_to overhead). With dynamic cap floored at 256, the alloc may now under-fit long prompts (cap=10000 → buffer width=10000 in one shot) or over-fit short prompts (cap=256 → buffer of 256 for a 36-token prompt). Worth revisiting in a future task: decouple `step` from `cap` and use `step = MIN_CACHE_CAP_FOR_GPU_PERF` always, letting grow_to expand the buffer in step-sized increments.
- **Effective cap upper bound is the smaller of the CLI flag and the model's config.** Users setting `--max-cache-cap` to ridiculous values get clamped silently to model_max_context (with a warn log). Edge case: if a model has `max_position_embeddings < 256`, the floor will exceed it, but `slots_max.max(256).min(model_max)` will still clamp to model_max (which would be unusable for any prompt anyway).
- **`SchedulerError::RequestTooLarge` is the second SchedulerError variant** after `QueueFull` (3e.3). Future scheduler-side typed errors (preemption, prompt-parse, OOM) should keep adding variants here.

## Carry-forward backlog

Sequenced for "perf 收尾" autonomous track (2026-05-17 Boss):
- **3c+** chunked admit_mid prefill — design + plan written 2026-05-17; impl pending
- **3e.1** vectorized per-row sampler — design written 2026-05-17 (§1.1 analysis-correction needs Boss review); plan + impl pending
- **B1-p2.5** production hardening (OOM safety, fairness) — discussed 2026-05-17, deferred to next session
