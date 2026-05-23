# P5h+1 — Attribution Gap Closure: Close-out

**Status:** Close Gate 4/4 PASS. P5h+1 ship state — actionable P5i/P5j candidate ranking unblocked.
**Date:** 2026-05-23
**Branch:** `ironmlx-p5h-perf`
**Implementation commits:** T1 `d57fbfb` + T2 `dfac330` + T3 `9f2c06c` + T1.5 `b14285a` + T4 `aa20283` + T5 (this commit)

**Sources:**
- Spec: `docs/superpowers/specs/2026-05-23-ironmlx-p5h+1-attribution-gap-closure-design.md` (commit `c5556fb`)
- Plan: `docs/superpowers/plans/2026-05-23-ironmlx-p5h+1-attribution-gap-closure.md` (commit `7c801fd`)
- Ranking snapshot (concise): `docs/p5h+1-ranking-snapshot.md` (T4 commit `aa20283`)
- Ranking snapshot (full detail, gitignored): `reports/p5h+1-ranking-snapshot.md`
- Codex review docs (gitignored): `reports/p5h+1-codex-review-questions.md` + `reports/p5h+1-close-gate-codex-review.md`
- Predecessor P5h close-out: `docs/p5h-t5-close-out.md`
- Memory: `project_p5h_findings.md` (P5h overall) + this close-out extends it.

---

## 1. Close Gate 4/4 PASS

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Lane A wrapper share ≤ 50% | ✅ PASS | `first_token_sampling_materialize_and_sample` not in top-5 at any Lane A PP |
| 2 | Lane B wrapper share ≤ 50% | ✅ PASS | `gs_chunk_N` not in top-5 at any Lane B PP |
| 3 | coverage_pct ≥ 0.95 per PP | ✅ PASS | 0.9796 / 0.9848 / 0.9907 / 0.9922 / 0.9935 / 0.9944 per PP=128/512/2048/4096/8192/16384 |
| 4 | Verdict ≠ data_insufficient ≥ 3 PPs | ✅ PASS | All 6 PPs (1× yes_with_scope_gate + 5× no_under_measured_cap) |

**P5h+1 spec § 2 Close Gate fully satisfied.**

## 2. Lane A `first_token_sampling` decomposition (T1)

Per Codex Q-P5h+1-2 hybrid approach.

**Phase 1 — span split** (`d57fbfb`): replaced `first_token_sampling` with sibling pair under the root span:
- `first_token_sampling_prepare` — logits reshape + per-row sampler/history construction (graph-construction work)
- `first_token_sampling_materialize_and_sample` — `sample_batch(...)` including `.to_vec()` materialization (lazy-graph forcing site)

Implementation in `ironmlx/src/core/scheduler.rs` uses explicit `open_p5h_span` + `close_p5h_span` (per Codex `[Verified Current Code Facts]` — no `with_p5h_span` helper exists). Close-on-error discipline verified: every Err branch closes its open span before `return Err(...)`; no `?` while a manual span is open.

**Phase 2 — measurement-only probe flag** (`d57fbfb`): `--p5h-measurement-eval-probes` CLI flag (gated by `p5h-profile` feature; default OFF). When ON, per-substep closures inside T2 GatedAttention 7 + T3 MoE 8 + GDN 11 + lm_head + cache_state_update force `mlx::transforms::eval` on returned Array(s) before span close so each substep accrues true incremental materialization cost.

Production parity verified: smoke test `p5_qwen35_moe_smoke` pp_tps within ±2% feature-off vs feature-on flag-OFF.

## 3. Lane B `gs_chunk_N` deep substep (T2)

Per Codex Q-P5h+1-3 mirror-Lane-A pattern.

**`core/p5h.rs::LANE_B_ALLOWED_TRY_SPAN_NAMES` extension** (`dfac330`): 3 → 37 names. Without this, T0a.11 wrapper opens and T1's substep probes silently no-op on Lane B because the try-helper rejects deep span names.

**`chunk_idx` schema field** (`dfac330`): `Option<u32>` added to `SpanFields` between `layer_idx` and `seq`. New thread-local `P5H_CURRENT_CHUNK_STACK` + `P5hChunkContextGuard` RAII (Drop pops + asserts ordering; safe under early `?` returns). Emission format inserts `chunk_idx` between `layer_idx` and `span_id`; explicit field wins, inherits from stack top otherwise. Spans outside chunk context emit `chunk_idx=null`.

**`GenerationStream::new` chunk loop** opens `gs_chunk_N` with `SpanFields { chunk_idx: Some(chunk_idx), ... }`; RAII guard entered at top of closure; `chunk_idx` increments after successful chunk before `pos += n`.

**Python schema validator**: `LANE_B_REQUIRED_TREE` extended with full decoder hierarchy + T2/T3/T4 chunk-descendants. `validate_chunk_ancestry` structural rule (every span under `gs_chunk_N` ancestor has matching non-null `chunk_idx`). Cross-validation pytest reads `p5h.rs` source via regex to assert Rust allow-list ⊆ Python ALLOWED.

## 4. T1.5 hotfix — `gda_step_7_kernel_dispatch_and_materialize`

Per Codex Q-P5h+1-Gate-1 Option B-lite (chosen over Option A accept-partial or Option C lower-threshold).

Pre-T1.5 sweep: coverage 0.923-0.937 (gate FAIL). Root cause per Codex investigation: `unattributed_gda_step_7_kernel_and_cache_update` accounted for 88-98% of all unattributed budget (4.4-6.8% per PP) because parent span owned actual kernel dispatch work in its own "self time" with only `cache_state_update` as a leaf child; aggregator synthesized the kernel self-time as a residual.

**Fix** (`b14285a`): inserted `gda_step_7_kernel_dispatch_and_materialize` child span inside parent's closure body. New sub-span wraps kernel select/build + state_in/t_arr construction + dispatch + take_at(0)×2 (Y + new_state per `ArrayVec::take_at` erase-and-shift contract) + probe eval (`eval(&[&y, &new_state])` — both outputs).

Parent becomes thin glue wrapper (2 children: new sub-span + existing `cache_state_update`). Post-fix sweep: coverage 0.9796-0.9944 (gate PASS); residual mass migrated to `gda_step_7_kernel_dispatch_and_materialize` ranking 5th at PP=512/2048 with 6.2-6.8% share (correctly attributed kernel-bound candidate).

Schema updates: added to LANE_B_ALLOWED_TRY_SPAN_NAMES (38 total), LANE_B_REQUIRED_TREE, and roi_ranking.py KERNEL_BOUND_SPANS. Parent removed from KERNEL_BOUND_SPANS (glue-only wrapper, no longer kernel-rewrite candidate). Lock-in test `test_is_kernel_bound_excludes_step_7_wrapper_after_t1_5_split` prevents regression.

## 5. P5i / P5j candidate ranking (post-P5h+1)

🔧 = `scope_gate_trigger=true` (kernel-bound; Boss Scope gate approval required before kernel rewrite per spec).

**Cross-PP stable candidates**:

1. 🔧 **gather_qmm_gate_up** — 20-25% all PPs (MoE quantized gate+up matmul). Top P5i + P5j candidate.
2. 🔧 **gather_qmm_down** — 10-12% all PPs (MoE quantized down matmul). Same kernel family.
3. **gda_step_1a_in_proj_qkvz** — 10-18% all PPs (GDN qkvz projection; op-level — no Scope gate).
4. 🔧 **gda_step_8_norm_proj** — 5-8% all PPs (GDN final norm+projection).
5. 🔧 **fused_sdpa** — 6-16% (long-PP O(S²) growth; top-2 at PP=16384).
6. 🔧 **gda_step_7_kernel_dispatch_and_materialize** — 5-7% (GDN kernel dispatch; new T1.5 sub-span).

Combined `gather_qmm_{gate_up + down}` share = 32-35% across all PPs — dominant kernel family.

**Per-PP feasibility verdict** (4-tier per Codex Q-T5-4):

| PP | target | op_only | with_kernel | verdict |
|---|---|---|---|---|
| 128 | +24% | 22.97% | 60.91% | yes_with_scope_gate |
| 512 | +74% | 22.22% | 61.10% | no_under_measured_cap |
| 2048 | +110% | 21.46% | 61.31% | no_under_measured_cap |
| 4096 | +115% | 20.43% | 61.66% | no_under_measured_cap |
| 8192 | +124% | 19.16% | 62.21% | no_under_measured_cap |
| 16384 | +126% | 18.05% | 62.66% | no_under_measured_cap |

**Honest interpretation**:
- PP=128 reachable per current candidate set with kernel rewrites.
- PP=512+ requires additional optimization sources beyond the current pool (gap +13% to +63% short). Options: higher realistic gain assumptions for first-pass quant kernel work; or new optimization discoveries; or partial-target outcome with explicit Boss approval.

Per spec § 6.5 denominator discipline: probe-mode root inclusive is 17-90% larger than production root (forced-eval overhead). Substep ROI shares are accurate; target feasibility uses production_root_us baseline as denominator. See `docs/p5h+1-ranking-snapshot.md` § "Per-PP table" for both columns.

## 6. Spec § 7.2.1 update (T5)

This commit updates spec § 7.2.1 gates #5 + #6 status:

- Gate #5 (P5i + P5j candidate ranking with ROI): **DEFERRED → LOCKED PASS** with actionable cross-PP candidate list above.
- Gate #6 (Target feasibility assessment): **DEFERRED → LOCKED PASS** with per-PP 4-tier verdict.

P5h overall: now **9/9 gates LOCKED PASS**.

## 7. Optional spec § 1.2 PP=2048 reconciliation

Per Codex Q-P5h+1-5 (both: spec edit + iron-bench boundary control noted).

Spec § 1.2 documents that PP=2048 is nominal P5j target; actual lane partition derived from runtime `routing_path`. Chat-template overhead (Qwen3 ChatML = 12 tokens per `project_p5h_t1_findings.md`) means `iron-bench --prompt-len 2048` routes through `gs_chunked` (Lane B) at default `prefill_chunk_size=2048`. An `iron-bench --prompt-len 2036` invocation is available as boundary-control sweep targeting exact PP=2048 Lane A behavior but does NOT replace PP=2048 as the primary measurement point.

`roi_ranking.py::observed_lane_for_pp` emits stderr WARN on mismatch (Codex T5-R fixup `1b53dfa`).

## 8. P5h+2 follow-up — consolidated

**Carried from P5h+1 reviews**:
- **`validate_chunk_ancestry` cycle vulnerability** (T2 code review Important #1) — 1-line `visited` set fix; matches pre-existing `under_decoder_layer` pattern; backport recommended per `[feedback_performance_stability_priority]`.
- **`P5hChunkContextGuard.active: bool` dead field** (T2 code review Minor #2) — always true; `if self.active` branch unreachable; remove or add `.disarm()` method.
- **`roi_ranking.py::LANE_A_WRAPPER_SPAN` references old `first_token_sampling`** (T1 code review note) — out-of-T1-scope; T4 verification used new `_materialize_and_sample` literal independently so no functional impact; consistency cleanup.
- **GA `kv_mask_update` outer probe duplicate-eval on cache=Some branch** (T1 code review Minor) — defensible no-op; documenting in comment if future Lane B sweeps surface concerns.

**Carried from P5h memory** (`project_p5h_findings.md`):
- **Emit cost reduction** — T0a 95% gate via buffered/binary emit per `project_p5h_emit_cost_followup`. P5h+1 didn't address; P5h+1 coverage gate at 0.95 PASSES, suggesting the 0.95 vs 0.50 T0a Lane-A median gap remains separate concern.
- **T0b H4 same-mode control** (Phase A × 2) — data confirmation; no downstream gate.
- **T2/T3 op-level ablation** — data-dependent; P5h+1 ranking surfaces `gda_step_1a_in_proj_qkvz` as top op-level candidate (10-18% across PPs). This now triggers the T2.4 / T3.4 binding doc op-level ablation condition (`docs/p5h-t2-layer3-binding.md` / `docs/p5h-t3-layer3-binding.md`). P5h+2 task candidate.
- **T4.2 mid-admit P5h ctx plumbing** — low priority.
- **Spec § 1.2 PP=2048 lane partition** — addressed via § 7 above (note added).

**New from P5h+1 ranking observations**:
- **gather_qmm kernel family is dominant cross-PP hotspot** (32-35% combined gate+down share at all PPs). P5i/P5j primary optimization target.
- **`fused_sdpa` long-PP O(S²) growth** — at PP=16384 becomes top-2 candidate. Long-PP attention strategy candidate for P5j.
- **PP=512+ target gap +13% to +63% short** even with all kernel candidates. P5i/P5j optimization phases need to consider whether higher realistic gain assumptions are warranted or new candidate discovery is needed beyond the current measured set.

## 9. P5i / P5j dispatch readiness

Per Codex Q-P5h+1-Gate-4 + spec § 8 sequential decision: P5h+1 Close Gate now PASSES, so P5i + P5j optimization phases may dispatch on the post-P5h+1 ranking.

**Recommended sequence**:
1. P5i starts with PP=128 (verdict yes_with_scope_gate; reachable per current candidates). Top candidate: `gather_qmm_gate_up` kernel rewrite (Scope gate approval needed first).
2. P5j starts with PP=2048+ but expects gap-short verdict; P5j scope decision may include extending the candidate pool or relaxing the +110-128% target.

Probe-mode pp_tps must NOT be used as production pp_tps evidence per spec § 6.5. P5i/P5j optimization measurement should run with `--p5h-measurement-eval-probes` OFF for production comparison (use prior P5h T5 baseline or re-capture with flag-OFF as needed).

## 10. Reusable infrastructure delivered

P5h+1 extends P5h reusable infra:
- **`--p5h-measurement-eval-probes` CLI flag** + `MEASUREMENT_EVAL_PROBES_ACTIVE` global atomic + `is_measurement_eval_probes_active()` getter (always available; cfg-branch to compile-time false in production)
- **`P5H_CURRENT_CHUNK_STACK` thread-local + `P5hChunkContextGuard` RAII** for chunk_idx propagation (pattern reusable for any future chunked-context attribution)
- **`chunk_idx` schema field** in `SpanFields` + `[p5h-profile]` emission format + Python `Span` dataclass + `P5H_LOG_RE` regex
- **`validate_chunk_ancestry` structural rule** in schema_validator (pattern reusable for any future ancestor-equality invariants)
- **Rust ↔ Python allow-list cross-validation test** (reads p5h.rs source via regex; pattern reusable for any future schema drift detection)
- **Per-substep eval probe pattern** — `if is_measurement_eval_probes_active() { mlx::transforms::eval(&[...])?; }` inside substep closures (canonical pattern for future ROI-candidate substeps)
- **gda_step_7 kernel-vs-glue split pattern** — separating kernel dispatch from state mutation into distinct spans (template for future parent-with-residual-self-time refactors)

## Take-away for Boss

P5h+1 closes the attribution gap that P5h T5 measure-only identified. Wrapper-dominance is gone (top-1 candidate dropped from 96-99% wrappers to 20-25% `gather_qmm_gate_up` kernel). Six cross-PP stable kernel + op candidates surfaced; P5i can dispatch on PP=128 reachable verdict; P5j has clear ranking but faces +13% to +63% target gap that needs additional optimization sources.

Spec § 7.2 P5h overall = **9/9 gates LOCKED PASS** (was 7/9 + 2 DEFERRED after P5h T5).
