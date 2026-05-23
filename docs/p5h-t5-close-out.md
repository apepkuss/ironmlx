# P5h T5 — Cross-Layer Attribution Synthesis + P5i/P5j Candidate Ranking: Close-out

**Status:** P5h ship state — **MEASURE-ONLY** close-out per plan T5.6 ("Ship state: nothing optimized — measure-only"). Replaces plan T5.7 `--allow-empty` template per `feedback_no_empty_commits` Boss preference.

**Date:** 2026-05-23.
**Branch:** `ironmlx-p5h-perf`.
**T5 commit chain:** `43ad953` (plan T5.1 desc fix) → `e328ee5` (T5 capture harness) → `c0e9644` (aggregator T5.1) → `f53b075` (roi_ranking T5.2) → `1b53dfa` (Codex T5-R 4-bug fixup) → this commit (T5.5 spec § 7.2.1 lock + T5.6 memory + T5.7 close-out doc).

**Sources:**
- Spec § 7.2.1 (T5 close-out verdict, conservative lock per Codex Q-T5-5): `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` lines ~1046+.
- Plan T5.1-T5.7: `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md` lines 4738-4898.
- Full T5 attribution detail (gitignored): `reports/p5h-attribution.md`.
- Codex T5 scope decisions: `reports/p5h-t5-codex-review-questions.md` (gitignored).
- Codex T5 results review: `reports/p5h-t5-results-codex-review.md` (gitignored).
- Memory: `project_p5h_findings.md` (overall P5h ship state).

---

## 1. T5 verdict per PP

| PP | observed lane | dominant span | share | verdict |
|---|---|---|---|---|
| 128 | A (`scheduler`) | `first_token_sampling` | 96.85% | `data_insufficient` |
| 512 | A | `first_token_sampling` | 98.55% | `data_insufficient` |
| 2048 | **B (`gs_chunked`) ⚠️ spec § 1.2 says A** | `gs_chunk_N` | 97.56% | `data_insufficient` |
| 4096 | B | `gs_chunk_N` | 98.60% | `data_insufficient` |
| 8192 | B | `gs_chunk_N` | 99.01% | `data_insufficient` |
| 16384 | B | `gs_chunk_N` | 99.36% | `data_insufficient` |

All 6 PPs verdict `data_insufficient` per 4-tier `FeasibilityVerdict` (per Codex Q-T5-4). The honest answer for actionable P5i/P5j candidate ranking given current instrumentation.

---

## 2. T5 spec § 7.2 gate state

Per spec § 7.2.1 conservative lock (Codex Q-T5-5):

| Gate # | Criterion | Status |
|---|---|---|
| 1 | Exclusive attribution coverage ≥ 95% per PP | ✓ LOCKED PASS (median 0.977-1.000 across 6 PPs) |
| 2 | Protocol-consistent data — dual-lane | ✓ LOCKED PASS (Lane A + Lane B as specified; PP=2048 spec/runtime mismatch documented) |
| 3 | UMA hardening | ✓ LOCKED PASS |
| 4 | Phase D root cause (T0b) | ✓ LOCKED PASS |
| **5** | P5i + P5j candidate ranking with ROI | **⚠️ DEFERRED to P5h+1** (infrastructure delivered; ranking blocked by wrapper-dominance) |
| **6** | Target feasibility assessment | **⚠️ DEFERRED to P5h+1** (honest verdict = `data_insufficient` per PP) |
| 7 | Reusable infra delivered | ✓ LOCKED PASS |
| 8 | Validation gates pass per task | ✓ LOCKED PASS (T0a/T0b/T1/T2/T3/T4 all close-out + T5 functional) |
| 9 | T0a HARD GATE | ✓ LOCKED PASS (pre-existing `ccbeff9`) |

**7 of 9 gates LOCKED PASS; gates #5 + #6 DEFERRED.**

---

## 3. Why DEFERRED — two structural instrumentation gaps

### Gap #1 — Lane A `first_token_sampling` lazy materialization wrapper

`first_token_sampling` span at `core/scheduler.rs:1059` brackets the sampler. MLX lazy graph means `model_prefill_forward` returns immediately with lazy nodes; actual GPU compute is forced by `sample_batch::argmax(...).to_vec()` at `core/scheduler.rs:1117` + `core/sampler.rs:279`, billing to the sampling span.

T2 GatedAttention + T3 MoE + T4 lm_head/cache substep instrumentation IS correct (records emit; `parent_span_id` correct), but `exclusive_us` is near-zero because compute hasn't materialized yet at substep close time. The "deferred" cost shows up as 96-99% of `first_token_sampling`.

**P5h+1 follow-up #1**: Lane A `first_token_sampling` lazy materialization boundary attribution. Options: insert explicit `mlx::transforms::eval(&[&logits])` before sampler (changes lazy semantics for measurement-only mode); OR split span into `first_token_sampling_eval_barrier` + `first_token_sampling_sampler_invoke`.

### Gap #2 — Lane B `gs_chunk_N` deep substep deferred

Per spec § 3 T0a, Lane B deep substep attribution explicitly suppressed/deferred to P5h+1. `gs_chunk_N` wraps per-chunk forward in `serve_via_gs_stream` but `attention_path` / `mlp_path` wrappers are NOT opened inside its body (only the Lane A `prefill_admitted` path opens them per `decoder_layer.rs` T0a.11 step 1).

**P5h+1 follow-up #2**: open `attention_path` + `mlp_path` wrappers inside `gs_chunk_N` body (mirror Lane A pattern). Validate by re-running T5 capture; expected `gs_chunk_N` share drops to ~30-50%.

### Gap #3 — Spec § 1.2 / runtime lane partition reconciliation (low priority)

PP=2048 spec says Lane A but runtime routes Lane B due to chat-template overhead = 12 tokens (per `project_p5h_t1_findings.md`). Either fix iron-bench `--prompt-len 2036` or update spec § 1.2. `roi_ranking.py` already emits stderr WARN on mismatch (Codex fixup `1b53dfa`).

---

## 4. Deliverables — what T5 produced

### Code / infrastructure (committed)
- `ironmlx/tests/p5h_t5_attribution_capture.rs` (raw-capture sweep harness, ~5min wall)
- `tools/p5h_aggregator/aggregator.py` extended (`compute_exclusive` + `synthesize_residual_leaves` + `coverage_pct` gate + diagnostic columns)
- `tools/p5h_aggregator/roi_ranking.py` (Candidate dataclass + top-3 + P5i/P5j + 4-tier verdict + Codex T5-R 4-bug fixup)
- 76 pytest tests pass (17 schema_validator + 17 aggregator + 27 roi_ranking + 15 from fixup)
- Spec § 7.2.1 T5 close-out verdict subsection (conservative lock per Codex Q-T5-5)

### Data (machine-local, gitignored)
- `/tmp/p5h-t5-server.log` (15525 raw `[p5h-profile]` records)
- `/tmp/p5h-t5-bench.csv` (42 measurement rows + request_id column for 100% join)
- `/tmp/p5h-t5-attribution.csv` + `.summary.csv` (per-PP aggregated)
- `/tmp/p5h-t5-ranking.csv` (164 candidates ranked)
- `/tmp/p5h-t5-verdict.json` (4-tier verdict per PP)

### Docs / memory
- `reports/p5h-attribution.md` (full T5 attribution + caveats + P5h+1 follow-up; gitignored detail)
- `docs/p5h-t5-close-out.md` (this doc; per-task close-out)
- `project_p5h_findings.md` memory (overall P5h ship state)
- `MEMORY.md` index updated

---

## 5. P5h+1 prerequisite list (consolidated)

**Critical (blocks P5i/P5j ranking)**:
1. Lane A `first_token_sampling` lazy materialization boundary attribution.
2. Lane B `gs_chunk_N` deep substep instrumentation (open `attention_path` + `mlp_path` wrappers inside).

**Low priority**:
3. Spec § 1.2 PP=2048 lane partition reconciliation (or iron-bench `--prompt-len 2036` workaround).

**Pre-existing (preserved from individual T-task close-outs)**:
- T0b: H4 same-mode control; H4 mechanism narrowing; H1 within-cycle metric refinement.
- T2: op-level ablation for `q_gate_k_v_proj` + `o_proj` IF future T5 finds dominant op-level GatedAttention hotspot — **NOT triggered by current T5** (`data_insufficient`).
- T3: op-level ablation for `shared_expert` + `router_logits_softmax_topk` IF future T5 finds dominant op-level MoE hotspot — **NOT triggered**.
- T4: mid-admit P5h ctx plumbing for T4.2 `mlx_eval_barrier` activation; lm_head materialization timing.
- Emit cost reduction: T0a 95% gate via buffered/binary emit path.

---

## 6. Take-away for Boss

T5 close-out is **measure-only**. Schema infrastructure validated end-to-end (76 pytest pass + 6/6 PP coverage gate ≥ 0.95 + 100% join + diagnostic spans + REQUIRED/ALLOWED split). The attribution numbers reveal wrapper-dominance instrumentation gaps that two prior tasks (T0a Lane B deferral + T4.2 mlx_eval_barrier INERT) already pointed to but didn't quantify. T5 now quantifies: both wrappers occupy ~97-99% of root_inclusive, making per-substep P5i/P5j ROI ranking impossible until P5h+1 closes gaps #1 + #2.

**P5i and P5j optimization phases should NOT dispatch on current T5 verdict** (PP=128 verdict yes is reporting artifact when read on the pre-Codex-fixup output; PP=512+ data_insufficient is the honest characterization after fixup). Recommended sequence: **P5h+1 closes gaps #1 + #2 → re-run T5 sweep with corrected infrastructure → re-rank P5i/P5j with substep-level ROI**.

Full attribution detail: `reports/p5h-attribution.md` (gitignored working tree).
