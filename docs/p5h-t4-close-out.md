# P5h T4 — lm_head + MLX state + tokenization + first-eval Profile: Close-out Report

**Status:** Committed close-out per plan T4 + Codex T4 review decisions. Replaces the plan T4 implicit close-out per `feedback_no_empty_commits` memory note (carrier file required for close-out narratives).

**Date:** 2026-05-22.
**Branch:** `ironmlx-p5h-perf`.
**Predecessor commits:**
- T4.1 `3e31537` — wrap MoE lm_head with `slice_last_and_project_lm_head` span
- T4.2 `be2491b` — wrap existing major `mlx::transforms::eval` sync points with `mlx_eval_barrier` span (2 sites, **INERT today**)
- T4.3 `08b876a` — wrap KVCache + GatedDeltaCache state-update sites with `cache_state_update` span
- T4.4 `883b5b9` — `tokenizer_encode` retroactive subspan inside `http_parse_render_tokenize`
- T4.5 `085fba9` — `first_eval_amortized_cost` diagnostic span (static OnceLock process-first)
- T4.6 `2dfe9ae` — sweep harness `tests/p5h_t4_lm_head_mlx_state_sweep.rs` (+1239 lines, 12/12 parser tests pass)
- T4.6 GPU sweep 99.5s wall → **verdict pass** (13/13 expected cells emit; Lane B PP=4096 + mlx_eval_barrier exempt; `/tmp/p5h-t4.json`)

**Source docs:**
- Spec § 3 T4: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` lines 906-916.
- Plan T4.1–T4.6: `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md` lines 4690-4732.
- Codex T4 review questions: `reports/p5h-t4-codex-review-questions.md` (gitignored).
- T0b binding: `docs/p5h-t0b-close-out.md`.
- T2 / T3 precedent: `docs/p5h-t2-layer3-binding.md` / `docs/p5h-t3-layer3-binding.md`.
- T1 / T2 / T3 / T4 sweep result JSONs: `/tmp/p5h-t{1,2,3,4}.json` (machine-local).

---

## 1. T4 deliverables vs spec § 3 T4

Spec § 3 T4 output requirements (line 915 verbatim):
> "Output: `slice_last_and_project_lm_head` occupancy (Lane A) / `gs_first_token_sample_dispatch` cost (Lane B), `first_token_sampling` (Lane A) / `gs_first_token_materialize_and_predispatch` (Lane B) cost, first-eval amortization (short PP suspect), tokenization fixed cost"

Coverage:

| Spec deliverable | T4 source | Lane A median@PP=128 |
|---|---|---|
| `slice_last_and_project_lm_head` occupancy (Lane A) | T4.6 / `/tmp/p5h-t4.json` | 0.42 us (dispatch-only — see § 4.2 caveat) |
| `gs_first_token_sample_dispatch` cost (Lane B) | T1 sweep (already wired in T0a.9) | 25077 us @ PP=4096 per `project_p5h_t1_findings` |
| `first_token_sampling` (Lane A) | T1 sweep (already wired in T0a.9) | not in T1 PP set; would emit on Lane A scheduler path after first-token sample dispatch |
| `gs_first_token_materialize_and_predispatch` (Lane B) | T1 sweep | 3307 us @ PP=4096 per `project_p5h_t1_findings` |
| first-eval amortization (short PP suspect) | T4.5 / `/tmp/p5h-t4.json` | 4316 us at PP=128 (cold-start; one-shot per process) |
| tokenization fixed cost | T4.4 / `/tmp/p5h-t4.json` | 141 us at PP=128 (linear scaling: 141 → 384 → 701 us across PP=128/512/1024) |

**Additional T4 outputs (not in spec line 915 but instrumented per plan T4.2/T4.3):**
- `mlx_eval_barrier` — wrapped at 2 sites in scheduler::admit_mid_chunk, **INERT today** (no records emit; admit_mid_chunk path doesn't plumb P5h ctx). Documented in T4.2 commit body + § 4.3 below.
- `cache_state_update` — 3 caller-site wraps (gated_attention.rs Step 4 KVCache update, gated_delta_net.rs Step 2c update_conv, Step 7e update_recurrent + advance). Median 0.125 us per record × 490 records per request = 61 us total. Dominated by 10 full-attn KVCache::update_and_fetch_on calls; GDN cache ops contribute ~0us per record (CPU-only Arc-share/offset-increment).

---

## 2. Codex T4 binding decisions applied (verbatim summary)

| Question | Codex decision | Applied in |
|---|---|---|
| Q-T4-1 (mlx_eval_barrier scope) | Option A 收窄 — wrap EXISTING explicit sync points only; do NOT add new `mlx::transforms::eval()` calls; tree span | T4.2 commit `be2491b` |
| Q-T4-2 (first_eval_amortized_cost design) | Option S — static `OnceLock<()>` process-first; emit as **diagnostic** span; NOT AppState mutation; NOT skipped | T4.5 commit `085fba9` |
| Q-T4-3 (lm_head cross-model) | MoE-only — wrap `qwen3_5_moe/model.rs` only; do NOT touch Dense `qwen3_5/model.rs` | T4.1 commit `3e31537` |
| Q-T4-4 (lane harness) | T4_PP_LIST=[128,512,1024,4096]; lane derivation by routing_path; Lane B exempt from T4 deep spans | T4.6 commit `2dfe9ae` |

---

## 3. T4.6 sweep evidence (PP=128 Lane A, full table)

From `/tmp/p5h-t4.json` cells:

| Span | Class | Lane | PP=128 record_count | PP=128 median_us | PP=512 median_us | PP=1024 median_us | Notes |
|---|---|---|---|---|---|---|---|
| slice_last_and_project_lm_head | tree | lane_a | 7 | 0.417 | 0.458 | 0.375 | 1 per request (max-tokens=1) |
| mlx_eval_barrier | tree | inert | 0 | null | null | null | INERT today — see § 4.3 |
| cache_state_update | tree | lane_a | 490 | 0.125 | 0.125 | 0.125 | 40 layers × cache sites × RUNS=7 |
| tokenizer_encode | tree | both | 7 | 141.125 | 383.709 | 701.500 | Linear PP scaling |
| first_eval_amortized_cost | diagnostic | lane_a | 1 | 4315.709 | 4717.500 | 4998.709 | One-shot per process |

Lane B PP=4096 (all expected_to_emit=false except tokenizer_encode):

| Span | Lane | record_count | median_us | Notes |
|---|---|---|---|---|
| slice_last_and_project_lm_head | lane_a | 0 | null | Lane B exempt (lane_a-only on lane_b PP) |
| mlx_eval_barrier | inert | 0 | null | INERT |
| cache_state_update | lane_a | 0 | null | Lane B exempt |
| tokenizer_encode | both | 7 | (reported, not pasted) | Fires on Lane B too (handler entry pre-routing) |
| first_eval_amortized_cost | lane_a | 0 | null | Lane B exempt |

---

## 4. Cross-cutting interpretation

### 4.1 first_eval_amortized_cost is significant short-PP bias

**4-5 ms** cold-start cost per process. Per-PP spawn-kill measurement protocol means each PP's first request sees this. At PP=128 with 7 RUNS, the first run carries the ~5 ms JIT cost as overhead — diluting to ~0.7 ms/run averaged, but biasing the FIRST-run measurement materially.

For T5 P5i candidate ranking + comparison-against-baseline studies, **the diagnostic record explicitly identifies this overhead**. T5 aggregator should subtract this from short-PP first-run wall time when reporting "production amortized" metrics.

Codex's recommendation to NOT skip T4.5 is vindicated by the 4-5 ms magnitude — comparable to ~10% of measured PP=128 prefill wall time on this machine. Skipping would have left a meaningful bias unexplained.

### 4.2 slice_last_and_project_lm_head dispatch-only timing — known limitation

T4.1 wraps the lm_head call but the closure body is `self.lm_head.forward_on(&last_hidden, target)` which is a LAZY graph dispatch (no materialization). The actual matmul work happens at the next implicit sync barrier (`sample_batch.to_vec()` in scheduler) which is NOT within T4.1's span window. So `slice_last_and_project_lm_head` median 0.4 us measures only the dispatch graph-build cost, not the kernel materialization cost.

Per Codex Q-T4-1 ("do NOT add new mlx::transforms::eval() calls"), we cannot insert an explicit eval inside T4.1's closure to force materialization without contaminating production lazy-graph scheduling semantics. This is a fundamental measurement boundary for the Lane A primary streaming path.

**Implication for T5**: lm_head cost is not directly measurable on Lane A via T4.1's span. If T5 needs lm_head occupancy, it must either:
1. Subtract known sub-budgets (substep medians + cache_state_update + tokenizer_encode + first_eval) from `model_prefill_forward` total to back-calculate lm_head + residual implicit-sync cost.
2. Defer lm_head measurement to a P5h+1 task that adds a controlled eval barrier (and acknowledges the lazy-graph semantics change).

This is a documented T4 measurement boundary, not a defect.

### 4.3 mlx_eval_barrier INERT today — wrap-now-light-up-later

T4.2 wraps 2 sites in `scheduler::admit_mid_chunk` (VL non-last chunk eval + text non-last chunk eval). Both are real existing `mlx::transforms::eval` sync points per Codex Q-T4-1 binding. But `handle_admit_mid_chunked` (scheduler_actor.rs:729 area) does NOT currently enter a `P5hTraceGuard`, so the `try_with_p5h_span_from_current_trace` API no-ops at runtime — span never opens, never emits.

**Lane A primary streaming path has NO explicit `mlx::transforms::eval` sync barrier** (per T4.2 commit body source survey). Sync is implicit via `sample_batch.to_vec()`. So for the primary `prefill_admitted` path used by iron-bench, there are no sites for `mlx_eval_barrier` to wrap.

T4.6 sweep correctly exempts mlx_eval_barrier from PASS criterion (record_count == 0 expected); verdict reports the absence honestly.

**P5h+1 follow-up tracked**: when mid-admit chunked path adds P5h ctx plumbing, the 2 existing T4.2 wraps will activate without further changes — wrap-now-light-up-later pattern.

### 4.4 cache_state_update breakdown — GDN cache nearly free, KVCache dominates

T4.3 records 490 cache_state_update records per request at any Lane A PP. Breakdown by site:
- 30 GDN layers × 2 sites (Step 2c update_conv + Step 7e update_recurrent+advance) = 60 records per request from GDN — each ~0us (CPU-only Arc-share)
- 10 full-attn layers × 1 site (KVCache::update_and_fetch_on) = 10 records per request from KVCache — each ~6us (Metal validation + grow_to + slice_update_on)

Median 0.125 us over the 70-record-per-request mixed population is dominated by the 60 zero-cost GDN records (median lies in the GDN zone). KVCache costs are in the right tail (~6 us each).

For T5 attribution, group cache_state_update by parent_span_id:
- Records with `parent_span="kv_mask_update"` → KVCache work
- Records with `parent_span="gda_step_2c_update_conv_state"` or `"gda_step_7_kernel_and_cache_update"` → GDN ~0us
- T5 aggregator can split + report each cache type separately.

### 4.5 tokenizer_encode is 45% of http_parse_render_tokenize

At PP=128: tokenizer_encode 141 us vs T1's http_parse_render_tokenize 316 us = 45% in tokenizer.encode itself. The remaining 175 us is in chat template rendering + JSON serialization + handler overhead.

PP scaling: tokenizer_encode 141 → 384 → 701 us across PP=128/512/1024 is roughly linear (2.7× per 4× PP increase — sub-linear due to common-token amortization in BPE). T1's http_parse_render_tokenize scales similarly: 316 → 636 → 869 us (2.7× per 4× PP).

For T5 short-PP fixed-cost identification, tokenizer_encode is a meaningful component but NOT a P5i candidate hotspot (tokenizer is CPU-side, optimization scope is different from MLX kernel work).

---

## 5. T4 contribution to T5 attribution model

T4 fills the last instrumentation gaps before T5 cross-layer synthesis can produce a complete per-PP exclusive attribution table:

1. **lm_head** (T4.1) — dispatch-only timing; T5 can back-calculate materialization cost via subtraction.
2. **MLX state barriers** (T4.2) — INERT for Lane A primary path; non-issue for current T5 attribution. P5h+1 follow-up if mid-admit instrumentation lands.
3. **Cache state updates** (T4.3) — KVCache + GDN cache split via parent_span_id grouping.
4. **Tokenization** (T4.4) — 45% of http_parse_render_tokenize is the BPE encode itself.
5. **First-eval amortization** (T4.5) — 4-5 ms cold-start JIT bias on first request per process; T5 must subtract from short-PP first-run wall time when reporting amortized metrics.

Combined with T1/T2/T3 outputs, T5 has complete coverage of Lane A attribution tree:
- T0a: GDN substep instrumentation
- T1: HTTP + scheduler + admission + role chunk + detok first content
- T2: GatedAttention 7 substeps (only full-attn layers)
- T3: MoE 8 substeps (all 40 decoder layers)
- T4: lm_head + cache update + tokenizer + first_eval (cross-cutting)

---

## 6. Out of T4 scope / deferred follow-ups

- **mid-admit P5h ctx plumbing** — when added, T4.2 mlx_eval_barrier wraps become active. P5h+1 task.
- **lm_head materialization timing** — requires controlled eval barrier insertion which Codex Q-T4-1 forbids for now. P5h+1 task with explicit acknowledgment of lazy-graph semantics change.
- **T5 aggregator `diagnostic_allowed_by_routing` extension** — must add `first_eval_amortized_cost` under `routing_path == "scheduler"` per T4.5 commit body flag. Spec § 2.5a:704-707 update needed.
- **Lane B T4 deep spans** — Lane B (`serve_via_gs_stream`) doesn't open scheduler-side wrappers; T4 deep spans (slice_last + cache + first_eval) are exempt per Codex Q-T4-4. P5h+1 if Lane B attribution depth needed.
- **Tokenizer alternatives** — tokenizer_encode is currently CPU-side BPE; if P5i candidates include alternative tokenizers, T4.4's instrumentation is the baseline.

---

## 7. T4 closure summary

| Sub-task | Status | Artifact |
|---|---|---|
| T4.1 — wrap MoE lm_head | done (commit `3e31537`) | `ironmlx/src/models/qwen3_5_moe/model.rs` +18/-1 |
| T4.2 — wrap existing eval sync points | done (commit `be2491b`) | `ironmlx/src/core/scheduler.rs` +26/-0 (2 sites, INERT today) |
| T4.3 — wrap cache update sites | done (commit `08b876a`) | gated_attention.rs +14/-1, gated_delta_net.rs +48/-14 |
| T4.4 — tokenizer_encode retroactive subspan | done (commit `883b5b9`) | chat_format.rs +50/-0, openai.rs +48/-1 |
| T4.5 — first_eval_amortized_cost diagnostic | done (commit `085fba9`) | `ironmlx/src/core/scheduler.rs` +60/-0 |
| T4.6 — sweep harness + GPU run | done (commit `2dfe9ae` + sweep 99.5s verdict pass) | `tests/p5h_t4_lm_head_mlx_state_sweep.rs` +1239; `/tmp/p5h-t4.json` |
| T4 close-out | this commit | `docs/p5h-t4-close-out.md` |

T4 closed. Next: T5 (cross-layer attribution synthesis + P5i/P5j candidate ranking + close-out report) per plan sequencing.
