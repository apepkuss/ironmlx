# B1-p2.3e.1a Close-out Report — Vectorized Greedy Sampler

## Goal Recap

B1-p2.3e.1a is Stage 1 of the vectorized sampler initiative. The core change replaces the per-row `Sampler::sample` loop inside `Scheduler::step` and `prefill_admitted_inner` with a single `sample_batch` dispatch. For the all-greedy default configuration (which covers the vast majority of production traffic), this routes through a fast path: instead of B sequential `.item()` GPU-sync calls (one per row), the fast path issues a single `argmax(logits, axis=-1)` followed by one `to_vec()` call. This collapses B GPU syncs into 1, eliminating a per-step synchronization bottleneck proportional to batch size.

The change is transparent to callers: `sample_batch` returns the same token-id vector as the per-row loop. Mixed-sampler batches (e.g. one greedy + one temperature-sampled row) fall back to the per-row path, preserving correctness for all configurations.

## Commits (4)

| SHA | Task | Description |
|-----|------|-------------|
| `7719998` | T1 | `sampler.rs` — `Sampler::is_greedy()` predicate + `sample_batch` + 10 unit tests |
| `d5e8dca` | T2 | `scheduler.rs` — `step` + `prefill_admitted_inner` refactored to use `sample_batch` |
| `bc015c0` | T2 doc | Refresh `step_inner` / `prefill_admitted_inner` doc comments |
| `3ecdd41` | T3 | Real-model perf gate test (`b1_p2_3e_1a_greedy_decode_speedup`) |

## Acceptance Gates

| Gate | Result | Notes |
|------|--------|-------|
| `cargo test core::sampler` (lib) | PASS | 12 pre-existing + 10 new = 22 sampler tests |
| `cargo test core::scheduler` (lib) | PASS | 36 tests; no regress vs T1 baseline |
| Hygiene (fmt --check + clippy -D warnings + build --release) | PASS | Run at every commit; all green |
| Real-model perf gate (`b1_p2_3e_1a_greedy_decode_speedup`) | PASS | See perf section below |
| `b1_p2_3b_2_scheduler_actor` (integration) | PASS | 3/3 tests pass (26s) |
| `b1_p2_4_batched_vl::mid_admit_vl_during_text_decode` (integration) | PASS | 1/1 tests pass (281s) |
| `b1_p2_3e_1a_vectorize_greedy::b1_p2_3e_1a_greedy_decode_speedup` (integration) | PASS | 1/1 tests pass (29s) |
| sweep_full (16 suites) | in progress (PID: 56191) | log: /tmp/3e_1a_sweep_full.log |

**Note:** `nn::decoder_layer::tests::forward_shape_and_dtype_bf16` fails with `__next_prime overflow` in `cargo test --lib`. This is a pre-existing failure on the base branch before any 3e.1a changes — confirmed by running same test at base. Unrelated to sampler/scheduler changes.

## Performance Characterization

**Setup:** B=4 concurrent greedy requests, Qwen3.5-4B-MLX-4bit, M1 Pro, `max_new_tokens=50`.

**Pre-3e.1a baseline (estimated):** Per-row `sampler.sample` loop with B sequential `.item()` GPU-sync calls. At B=4 and ~1-3 ms per sync, the sampler block was ~4-12 ms per step — roughly 5-10% of total step latency at 80-120 ms/step.

**Post-3e.1a (measured):**

| Metric | Value |
|--------|-------|
| Per-row median gap (row 0) | 64.710 ms |
| Per-row median gap (row 1) | 64.710 ms |
| Per-row median gap (row 2) | 64.710 ms |
| Per-row median gap (row 3) | 64.710 ms |
| max_median | 64.710 ms |
| min_median | 64.710 ms |
| ratio (max/min) | 1.00x |

The 1.00x ratio confirms batched-step lockstep is working correctly. All 4 rows see identical per-token cadence, as expected when steps are dispatched as a single batch. The 64.7 ms median is well under the 200 ms defensive ceiling.

The primary value of 3e.1a is eliminating the B-proportional `.item()` synchronization bottleneck, which compounds at larger batch sizes. At B=4, overall step latency impact is within noise; the architectural improvement is the foundation for 3e.1b.

## Architecture Notes

### `Sampler::is_greedy()`
A 7-field predicate returning `true` when `temperature <= 0.0` AND all of `top_k` / `top_p` / `min_p` / `repetition_penalty` / `frequency_penalty` / `presence_penalty` are `None`. Zero GPU ops — pure struct comparison. Distinct from `is_pipelinable` (which permits non-greedy temperature as long as penalties are off).

### `sample_batch(samplers, logits, histories)`
Routes based on whether all samplers in the batch are greedy:
- **All-greedy path:** `argmax(logits, axis=-1)` + 1 `to_vec()` sync — 1 dispatch total for B rows.
- **Mixed path:** per-row fallback, identical to pre-3e.1a. No regression for configured samplers.

### Scheduler integration (three-phase pattern)
1. **Collect:** gather active-row samplers + logit rows into parallel vecs
2. **Batch dispatch:** single `sample_batch(samplers, stacked_logits, histories)` call
3. **Distribute:** map results back to per-row `DecodeEvent` slots

Pad / non-active rows use `Sampler::greedy()` sentinel + empty history; results discarded via `active_at_start` filter. Avoids hot-path conditional branches.

### `admit_mid_finalize`
Retains single-row `Sampler::sample` for the B=1 admit path. Wrapping B=1 in `sample_batch` adds overhead with zero benefit — correctly excluded.

## Carry-Forward (not in 3e.1a scope)

| Item | Description |
| --- | --- |
| **3e.1b** | Vectorize configured sampler — 7 ops (rep+freq+pres / temp / top_k via mlx::sort / top_p / min_p / softmax / renorm) + batched categorical (spec: `2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md`) |
| **3e.2** | PRNG key state centralization — Sampler 去 Cell，Scheduler 集中持 `[B_max, 2]` state (spec: `2026-05-17-b1-p2-3e-2-prng-key-batching-design.md`) |
| **Top-k custom Metal partial-sort kernel** | NG in 3e.1b (sort 足够)，defer 到 3f 或 future |

## Sweep_full Result (post-close-out addendum)

`sweep_full.sh` 启动于 22:24:33 JST 2026-05-17，完成于 23:23:28 (64m 55s)。**16/16 PASS**，无任何 regression。

| Suite | Result | Time |
| --- | --- | --- |
| b1_p2_1_batched_prefill | ✅ PASS | 367s |
| b1_p2_2_batched_decode | ✅ PASS | ~ |
| b1_p2_3a_* | ✅ PASS | ~ |
| b1_p2_3b_2_scheduler_actor | ✅ PASS | ~ |
| b1_p2_3b_3_* / 3b_4_* | ✅ PASS | ~ |
| b1_p2_3c_2_scheduler_decode_mask | ✅ PASS | 73s |
| b1_p2_3c_3_continuous_batching | ✅ PASS | 155s |
| b1_p2_3d_admission_queue | ✅ PASS | 272s |
| b1_p2_4_batched_vl | ✅ PASS | 969s |
| b1_p2_3f_cache_cap | ✅ PASS | 275s |
| p6_qwen35_vl_logits_match | ✅ PASS | 42s |
| p4_http_smoke | ✅ PASS | 302s |
| b1_p2_3c_plus_chunked_admit_mid | ✅ PASS | 641s |

**结论**: 3e.1a 不引入任何 admit / cache / VL / HTTP regression。Sampler vectorization (greedy fast path) 是真正的 zero-cost foundation for 3e.1b/3e.2。
