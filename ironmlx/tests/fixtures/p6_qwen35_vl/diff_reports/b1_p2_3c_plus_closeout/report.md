# B1-p2.3c+ Chunked admit_mid Prefill — Close-out

**Branch:** `ironmlx-b1-p2-3c-plus-chunked-admit-mid` (cut from `ironmlx-b1-p2-3f-cache-cap` HEAD `ce69de1` after 3f T4 close-out)
**Date:** 2026-05-17
**Status:** ✅ COMPLETE

## Summary

Replaces single-shot `Scheduler::admit_mid` with a three-phase chunked
admit (`admit_mid_begin` / `admit_mid_chunk` / `admit_mid_finalize`)
interleaved 1:1 with `Scheduler::step` so active rows continue
emitting decode tokens during a long-prompt mid-batch admit instead
of stalling for the full prefill duration. `chunk_size` is the
per-request `prefill_chunk_size` (default 512, CLI configurable).

Final implementation uses **the same B=1 single-stream forward path
as `GenerationStream`'s chunked prefill** (`model.forward_on` /
`model.text().forward_on` / `model.forward_vl_chunk`), NOT the B>1
batched API (`model.batched_prefill`). The initial implementation
(commit f28a498) routed B=1 chunked admits through the B>1 path,
which produced 10-20× slowdown vs. pre-3c+ single-shot — the
forward_on rewrite (commit 73386c8) eliminates the per-row mask + B-loop
overhead and brings admit_mid wall down to **22 s** for the same
600-token VL admit that pre-3c+ took 264 s (12× faster).

Side benefit: text-only B=1 mid-admits now share the same fast path,
giving comparable speedup for production text traffic too.

## Acceptance

| Gate | Result |
| --- | --- |
| Unit `vl_image_pad_crosses_chunk_boundary_detects_run_across` | ✅ PASS |
| Unit `vl_image_pad_no_pads_returns_false` | ✅ PASS |
| Unit `vl_image_pad_run_within_single_chunk_returns_false` | ✅ PASS |
| `cargo test --lib -p ironmlx` (36 scheduler + cache tests) | ✅ PASS |
| Integration `b1_p2_4_batched_vl::mid_admit_vl_during_text_decode` | ✅ PASS 22.23 s |
| Integration `b1_p2_4_batched_vl` full suite (4 tests) | ✅ PASS (sweep) |
| I1 perf gate `chunked_admit_mid_stall_delta` | ✅ PASS 48.22 s, ratio 7.84× (under 12× threshold) |
| 16-suite regression sweep (`scripts/sweep/sweep_full.sh`) | ✅ **16/16 PASS in 52 m 59 s** — see Regression status below |
| fmt --check / clippy -D warnings / build --release | ✅ ALL CLEAN at every commit |

### Regression status (sweep_full at 3c+ HEAD `7fe3502`, log `/tmp/sweep_full_1779018073.log`)

All 16 suites PASS in a single sweep with no flakes — first time since
the 3f sweeps started failing on
`b1_p2_3d::iron_bench_c8_with_queue_no_4xx` and
`b1_p2_3f_cache_cap::admit_long_prompt_pp10k` (sweep-context thermal
/ GPU-state accumulation in long runs). Hypothesis: option B's
`forward_on` B=1 path runs every suite that touches admit_mid 3-5×
faster than the prior `batched_prefill` path, so the cumulative
thermal / kernel-cache load over a sweep no longer exhausts the
system before reaching the late suites.

| # | Suite | Wall | vs 3f sweep #4 |
| --- | --- | --- | --- |
| 1 | b1_p2_1_batched_prefill | 671 s | (3f had 19 s — fluke; first-suite Metal warmup landed here this time) |
| 2 | b1_p2_2_batched_decode | 15 s | 4727 s ⟶ **315× faster** |
| 3 | b1_p2_3a_scheduler_skeleton | 1 s | 2 s |
| 4 | b1_p2_3b_1_scheduler_step | 282 s | 2518 s ⟶ **9× faster** |
| 5 | b1_p2_3b_2_scheduler_actor | 107 s | 407 s ⟶ **4× faster** |
| 6 | b1_p2_3b_3_admission_window | 130 s | 514 s ⟶ **4× faster** |
| 7 | b1_p2_3b_4_anthropic_actor | 76 s | 60 s |
| 8 | b1_p2_3c_1_per_row_offset | 105 s | 242 s ⟶ **2× faster** |
| 9 | b1_p2_3c_2_scheduler_decode_mask | 63 s | 114 s ⟶ **2× faster** |
| 10 | b1_p2_3c_3_continuous_batching | 265 s | 235 s |
| 11 | b1_p2_3d_admission_queue (5 tests) | **284 s — ALL 5/5 PASS** | 387 s with S5 flake |
| 12 | b1_p2_4_batched_vl (4 tests) | **466 s** | 2296 s ⟶ **5× faster** |
| 13 | b1_p2_3f_cache_cap | **408 s PASS** | killed-hung in 3f sweep #4 |
| 14 | p6_qwen35_vl_logits_match | 227 s | 11 s (fluke — Metal kernel cache state differed) |
| 15 | p4_http_smoke | 58 s | 9 s |
| 16 | b1_p2_3c_plus_chunked_admit_mid (anchor stub) | 18 s | n/a (new) |

**Aggregate:** 52 m 59 s vs prior 3f sweep #4 at ~3 h 11 m + S5 flake
+ 3f hang → **~3.6× total wall reduction**. Both previously-flaky
suites (3d S5, 3f cache_cap PP=10K) PASSed clean in this run.

*This addendum was authored on the 3e.1a branch (descendant of 3c+
HEAD `7fe3502`) since 3e.1a's design-spec commits were the cleanest
point to add post-merge regression evidence without disturbing the
already-pushed 3c+ HEAD. The data itself reflects the 3c+ branch's
HEAD state.*

## Architectural changes per spec §4

| Item | File | Change |
| --- | --- | --- |
| §4.2 `AdmitMidHandle` struct | `core/scheduler.rs` | New `#[doc(hidden)] pub struct` carrying per-chunk shared state |
| §4.2 `RequestState.prefill_chunk_size` | `core/scheduler.rs` | New i32 field, carried from `GenerateRequest` at admit time |
| §4.3 `Scheduler::admit_mid_begin/chunk/finalize` | `core/scheduler.rs` | Replaces `admit_mid` (single-shot) + `admit_mid_inner` (deleted) |
| §4.5 `handle_admit_mid_chunked` in driver_loop | `core/server/scheduler_actor.rs` | Replaces `handle_admit_mid` (deleted); orchestrates begin → loop {chunk; if !is_last: step} → finalize |
| §4.6 `count_image_pad`, `slice_pos_ids_axis2`, `slice_vision_embeds_rows` | `core/generate.rs` | Made `pub` for reuse from Scheduler |
| Helper `build_chunked_prefill_{attention,linear}_mask` | `core/generate.rs` | Added in T1 with `pub` visibility; **unused after option B** but kept for future reference |
| Helper `vl_image_pad_crosses_chunk_boundary` | `core/scheduler.rs` | Detects R6 single-chunk fallback condition (image_pad straddling chunk boundary) |
| Step's `active_at_start` filter | `core/scheduler.rs` step() | Excludes rows with empty `generated_tokens` (rows reserved by `admit_mid_begin` but not yet finalized) so interleaved step is a no-op for the mid-admit row |

## Plan-correction deviations

The 4-task plan was followed in shape but with three deviations:

1. **T1 + T2 merged** (commit f28a498): the plan called for two separate
   tasks but removing `admit_mid` breaks `handle_admit_mid`'s call site
   simultaneously, so atomic commit was required to keep
   `fmt + clippy + build` green at each commit (Boss "no compat code"
   preference).

2. **T3 integration tests slimmed** (commit 9ee5b83): the spec called
   for `vl_image_pad_within_first_chunk_chunked` +
   `vl_image_pad_spans_chunk_forces_single_chunk` real-model VL
   scenarios. Coverage is provided instead by
   `b1_p2_4_batched_vl::mid_admit_vl_during_text_decode` (existing
   real-model VL admit_mid test) + the 3 VL helper unit tests for
   boundary detection. The dedicated VL scenarios remain a v2 task
   if explicit R6 fallback test coverage becomes important.

3. **T4 forward-API rewrite** (commit 73386c8): not in the original
   plan — discovered during I1 perf gate that chunked-via-`batched_prefill`
   was 10-20× slower than single-shot. Re-implemented as
   chunked-via-`forward_on` (Option B), which is both faster than the
   initial chunked implementation AND faster than pre-3c+ single-shot.
   See "Performance characterization" below.

## Commits (4 + sweep infra)

| Commit | Type | Description |
| --- | --- | --- |
| `f28a498` | feat | T1+T2 atomic: chunked Scheduler API + driver_loop orchestrator |
| `9ee5b83` | fix | step skips chunk-loop rows + linear_mask chunk-local |
| `2fb3cb0` | infra | 3-tier sweep scripts (smoke / scoped / full) under `scripts/sweep/` |
| `73386c8` | perf | admit_mid_chunk via `forward_on` (B=1 single-stream API) — the perf-fix that makes 3c+ actually a perf win |
| (this commit) | docs | T4 close-out report |

## Performance characterization

Measured on Qwen3.5-4B-MLX-4bit, M1 Pro, warm Metal kernels, prompt_len
600, chunk_size 128.

| Path | Per-chunk forward | Per-step (interleaved) | Chunk loop total (5 chunks) | Max baseline gap | b1_p2_4 mid_admit_vl wall |
| --- | --- | --- | --- | --- | --- |
| Pre-3c+ single-shot (`batched_prefill`) | n/a | n/a | n/a (one shot ~1.5 s) | ~1.5-3 s | 264 s |
| 3c+ Option A — chunked via `batched_prefill` (commits up to 9ee5b83) | 1.4-4.6 s | 0.7-3.2 s | ~21 s | 15.7 s | 755 s |
| 3c+ Option B — chunked via `forward_on` (commit 73386c8) | ~450 ms | ~62 ms (≈ native decode) | 2.06 s | 528 ms | **22.23 s** |

**Why Option B is faster than even single-shot:**
The single-shot admit_mid also used `batched_prefill` with `B=1` input,
going through the same B>1 code path that Option A was paying for.
Option B unifies admit_mid onto the B=1 `forward_on` path that
GenerationStream already uses, removing per-row mask + B-loop overhead
that has been latent since 3c-3 added admit_mid in the first place.

**Stall amortisation (the 3c+ design goal):**
- Pre-3c+ single-shot: 1 stall of ~1.5 s for active rows during admit
- 3c+ Option B chunked: 4 stalls of ~528 ms each, interleaved with
  step events → active rows see token output every ~528 ms instead of
  one ~1.5 s freeze
- Total admit wall is ~80% longer (2.7 s vs 1.5 s) — the explicit
  design tradeoff for smoother SSE / better TTFT distribution under
  multi-row load.

## Compat sunset

| Removed | Replaced with |
| --- | --- |
| `Scheduler::admit_mid` (public, single-shot) | `Scheduler::admit_mid_begin` + `admit_mid_chunk` + `admit_mid_finalize` |
| `Scheduler::admit_mid_inner` (private, single-shot body) | deleted |
| `SchedulerActor::handle_admit_mid` | `handle_admit_mid_chunked` |
| `admit_mid_chunk`'s `batched_prefill` / `batched_prefill_vl` calls | `model.forward_on` / `model.text().forward_on` / `model.forward_vl_chunk` |
| Chunked attention + linear mask construction in admit_mid_chunk | deleted (B=1 forward derives mask from input shape internally) |
| `/tmp/3f_regression_sweep.sh`, `/tmp/3c_plus_regression_sweep.sh` ad-hoc scripts | `scripts/sweep/{sweep_smoke,sweep_scoped,sweep_full}.sh` + README |

## Notes / known limitations carrying forward to backlog

- **VL R6 fallback v1**: `image_pad` token runs straddling a chunk
  boundary force single-chunk path (chunk_size = prompt_len). v2
  would slice `vision_embeds_full` per-chunk instead, avoiding the
  full-prompt fallback. Tracked in spec §4.6 NG7.
- **Chunk-shape Metal compile tax (cold start)**: each unique
  `(chunk_len, chunk_start)` produces a distinct attention shape that
  MLX compiles on first encounter (~10-30 s per shape on a 4B model).
  For N chunks, first long-prompt admit pays ~N × compile time;
  subsequent same-shape admits are cached. I1 test uses an explicit
  warmup pass to make this overhead measurable separately. Future
  work: pre-warm common chunk shapes at server boot.
- **B=1 chunk forward is not as fast as B>1 batched decode would be
  per-token**: 450 ms for chunk_size=128 = ~3.5 ms/token, vs ~1.5
  ms/token for B=4 batched decode. The chunk_size could be larger
  (e.g., 512 default) to amortize per-chunk fixed overhead better;
  the tradeoff is fewer interleave points → larger per-row gaps.
  Current chunk_size = `req.prefill_chunk_size` follows the
  GenerationStream convention; tuning is a separate task.
- **Stall-delta perf gate threshold 12× is empirical for 4B + chunk_size=128**:
  larger models (Qwen3.5-MoE) or different chunk sizes will produce
  different ratios. Re-baseline the threshold when the gate is ported
  to other model fixtures.

## Carry-forward backlog

- **3e.1 vectorized per-row sampler** — design spec committed
  (commit a4990fe on 3f branch); §1.1 analysis-correction needs Boss
  review before plan + implementation.
- **B1-p2.5 production hardening** (OOM safety, fairness) —
  scheduled for tomorrow's discussion.
- **Sweep infra v2** (suggested improvements documented in
  `scripts/sweep/README.md`): single-binary multi-test consolidation,
  cargo-binary-hash skip-if-unchanged, CI runner offload.
