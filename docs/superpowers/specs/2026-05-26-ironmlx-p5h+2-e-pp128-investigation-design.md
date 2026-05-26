# P5h+2.e — PP=128 ironmlx-specific within-CI Residual Investigation: Design Spec

**Status:** Active/in progress. Small-PP acceptance threshold reconciliation is integrated, but final close-out must wait for the active P5h+2.e run to finish.

**Date:** 2026-05-26.

**Branch:** `ironmlx-p5h+2-e-pp128-investigation`.

**Predecessor docs:**
- P5h+2.d close-out (binding parent): `docs/p5h+2-d-close-out.md` § 6 (P5h+2.e direction)
- Codex round-1 of P5h+2.d Stage 1: `reports/p5h+2-d-stage1-codex-review.md`
- P5h+2.e brainstorm consultation (gitignored): `reports/p5h+2-e-brainstorm-codex-questions.md`
- Phase 0 close-out: `docs/p5i-c-phase-0-close-out.md` § 1 #4 (STILL FAIL/DEFERRED)
- Phase 1 γ-lite spec (downstream): `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 2.3 (will be backfilled per § 11)

---

## § 0 Goal + scope

Investigate the PP=128 ironmlx-specific within-CI 4-5% residual confirmed IRONMLX-SPECIFIC by P5h+2.d δ (omlx PP=128 cd=0s within-CI 0.82% vs ironmlx 4.21% — 5× gap under identical hardware/model/protocol). Per Codex round-1 priority: H1.c preheat topology mismatch > H_small_batch + routing variance > H2 MLX state-decay.

**This phase is standard investigation + fix (NOT γ-lite).** It involves:
- iron-bench / harness env-var protocol change
- production sweep + acceptance verification
- conditional MoE expert occupancy instrumentation if T2 is approved

**Primary success path**: PP=128 envelope within the `small-PP acceptance threshold` (2.5%) and PP=512 within the standard acceptance threshold (2.0%) via measurement protocol stabilization (NOT ironmlx runtime bug fix per Codex Q9 wording binding). Phase 0 § 7 #4 backfill PASS requires actual P5h+2.e envelope numbers as evidence. If the gate does not pass, close-out keeps Phase 0 § 7 #4 FAIL/DEFERRED and escalates per § 3.2.

## § 1 Predecessor evidence + hypothesis matrix

### § 1.1 Confirmed mechanism (P5h+2.d)

- **Mechanism gate strong_yes**: cooldown {0s→120s} reduces trailing/fast-start residual 64-78% for both PPs
- **PP=512 Acceptance PASS** at cd=120s (envelope 0.91%)
- **PP=128 Acceptance FAIL** at cd=120s (envelope 4.71%; within-CI 4.71%, between-half 1.05% — within-sweep stuck regardless of cooldown)
- δ verified ironmlx-specific: omlx PP=128 cd=0s within-CI 0.82% (5× tighter than ironmlx under identical conditions)

### § 1.2 Hypothesis priority (Codex round-1 binding A1)

| Code | Hypothesis | Test approach (this spec) |
|---|---|---|
| **H1.c** | Preheat topology mismatch — preheat hardcoded `--prompt-len 512`; PP=128 measured loop runs against state warmed for PP=512 | T1: equal-budget same-shape preheat protocol per § 2 |
| H_small_batch | Per-run nonce-generated prompts at small batch cause variable MoE expert dispatch + variable kernel tile path | T2 (conditional): pin nonce + opt-in expert-occupancy summary stats |
| H2 | MLX allocator / JIT cache state-decay during long sweep, more sensitive at PP=128 | Deferred: too expensive in-phase; separate successor mini-phase if T1+T2 both reject |

## § 2 Equal-budget same-shape preheat protocol (Codex round-1 critical revision)

### § 2.1 Protocol design (Codex round-1 Q3+Q4 binding)

Total preheat run budget kept at **1100** (matches existing baseline). Budget split 550+550 across two phases:

| Measured PP | Phase 1 preheat | Phase 2 preheat (last = measured-shape) |
|---|---|---|
| PP=128 | 550 runs PP=512 | 550 runs PP=128 |
| PP=512 | 550 runs PP=512 | 550 runs PP=512 (no shape change) |

**Why equal budget**: isolates "shape effect" from "more total preheat time" effect. If we just appended `1100 PP=128` after `1100 PP=512` (total 2200 runs), a PASS could be attributed to "more preheat time" rather than "shape match". The 550+550 split holds total preheat work constant.

### § 2.2 Env var design (Codex round-1 Q3 binding)

New env var `P5I_C_PREHEAT_PP_LIST` with `{pp}` token substitution:
- Default `"512"` (existing baseline behavior; backward-compatible)
- P5h+2.e T1 set `"512,{pp}"` — harness substitutes `{pp}` with the measured PP per cell
- `P5I_C_PREHEAT_RUNS` env var: existing default `1100`; T1 sets `550` (half budget)

Harness reads env var, cycles through PP list in `monolithic_preheat`, runs `P5I_C_PREHEAT_RUNS` per shape. iron-bench already supports comma-separated `--prompt-len` natively.

Implementation detail: `monolithic_preheat` must receive the measured PP so `{pp}` substitution is cell-local. The harness must reject empty lists, non-positive PPs, or unresolved `{pp}` tokens before starting a long sweep.

### § 2.3 Backward compatibility

When `P5I_C_PREHEAT_PP_LIST` is unset OR == "512": harness behavior is equivalent to the P5h+2.d close-out baseline (single-shape 1100-run PP=512 preheat). T1 changes do NOT affect any prior Phase 0 / P5h+2.b/c/d data.

## § 3 Two-gate framework

### § 3.1 T1 acceptance gate

T1 PASS requires BOTH:

| # | Criterion | Method |
|---|---|---|
| A1 | PP=128 envelope ≤ small-PP acceptance threshold (2.5%) under new equal-budget same-shape preheat protocol, ≥3 fresh-spawn repeats, cd=120s | `tools/p5i_c_pp_tps_envelope.py` per spec § 8 binding; JSON must emit `target_policy=small_pp_acceptance_threshold` |
| A2 | PP=512 envelope ≤ standard acceptance threshold (2.0%) under same protocol (re-verify no regression vs P5h+2.d PASS at 0.91%) | same tool; JSON must emit `target_policy=standard_acceptance_threshold` |

T1/T2 acceptance uses the same all-runs envelope behavior as P5h+2.d by default. If the implementation plan decides to enable Rule B trimming, it MUST be implemented as a tested tool option before any sweep starts, applied uniformly to all cells, and recorded in both envelope JSON and cell metadata. Manual trimming is forbidden.

**Gate relaxation boundary** (Codex round-1 Q8 + Q2 binding, updated by 2026-05-26 threshold reconciliation):
- No generic ±3% accept
- The only accepted threshold expansion is PP=128's named `small-PP acceptance threshold` at 2.5%, emitted by the envelope tool as `target_policy=small_pp_acceptance_threshold`
- No post-hoc Rule C-style trim to make data fit
- No new exclusion rules added after seeing data

### § 3.2 Predeclared expansion rules (Codex round-1 Q2 binding)

T1 outcomes determine T2 trigger:

| T1 outcome | Definition | Action |
|---|---|---|
| **Strong PASS** | A1 + A2 both PASS under their per-PP acceptance target | STOP. Close P5h+2.e Strong PASS. Backfill Phase 0 § 7 #4 PASS with P5h+2.e envelope numbers as evidence. |
| **Weak** | No PP > 3%, and at least one PP exceeds its per-PP target but remains ≤ 3% | NO Phase 0 backfill. Document. Boss + Codex decide whether to expand to T2 OR close P5h+2.e weak-evidence. |
| **FAIL** | A1 OR A2 > 3% | NO Phase 0 backfill. T2 becomes the recommended next step, but T2 execution still requires Boss approval because it adds new diagnostic code + extra GPU budget. |

T2 outcomes (if triggered) similarly: PASS → close-out; FAIL → H2 escalation in a separate successor mini-phase, NOT this spec.

## § 4 T1 — equal-budget preheat sweep (always runs)

### § 4.1 Files touched

- Modify: `ironmlx/tests/p5i_c_phase_0_capture.rs` — `monolithic_preheat` reads `P5I_C_PREHEAT_PP_LIST` env var; substitutes `{pp}` token per measured PP; loops over PP list with `P5I_C_PREHEAT_RUNS` runs per shape
- Modify: `tools/p5h_2b_protocol_experiment.py` — add `--preheat-pp-list` CLI arg, pass through to env

`meta.json` MUST record `preheat_pp_list_effective`, `preheat_runs_per_shape`, and `preheat_total_runs_effective` for each cell so Phase 0 backfill evidence can be audited without re-reading driver logs.

### § 4.2 Sweep cells

`{PP=128, PP=512} × 3 fresh-spawn repeats = 6 cells` at cd=120s (P5h+2.d-resolved BEST cooldown). Same protocol otherwise: `same_spawn_per_pp` lifecycle, `quiet_acceptance` logging, RUNS=15, `--inter-run-cooldown-secs=120`, equal-budget same-shape preheat per § 2.

### § 4.3 Wall estimate

Per cell: 4-min preheat (550 PP=512) + 4-min preheat (550 PP=128 or 512) + 30-min measured (15 × 4s + 14 × 120s) ≈ 38 min. 6 cells = ~3.8 hr. Plus driver overhead ~30 min. **T1 GPU wall ~4 hr.**

## § 5 T2 — H_small_batch (conditional on T1 weak/FAIL, Boss approval required)

### § 5.1 nonce pinning (Codex round-1 Q5 binding)

New iron-bench CLI flag `--nonce-seed N` (production-grade; default = current time-based behavior; set = fixed seed). Semantics when set: one fixed base seed is used for the whole sequential cell, and measured-run nonce = `N ^ (run_idx << 8)`. Per-run xor variation STILL applies, so each run within a sweep gets distinct prompts derived from the fixed base seed. This gives reproducible nonce SEQUENCES across repeats while still varying prompts within a sweep.

**EXPLICITLY NOT for acceptance** (Codex round-1 Q5 binding): `--nonce-fixed` flag for fully-identical prompts every run is diagnostic-only if added at all; may trigger prefix/KV cache hits that contaminate PP measurement. NOT used in T2 acceptance sweep.

### § 5.2 MoE expert occupancy summary stats (Codex round-1 Q6 binding)

**NOT** stored in p5h `SpanFields` (existing schema not designed for 256-bucket histograms).

Instead: opt-in diagnostic logging in `sparse_moe.rs routing_sort_pack` substep, gated by new env var `IRONMLX_EXPERT_OCCUPANCY_LOG=1`. Per-cell summary stats emitted to the captured server log with a distinct diagnostic prefix:
- `nonempty_experts` (count of experts that received >0 tokens)
- `max_bucket` (most-loaded expert's token count)
- `p95_bucket` (95th percentile expert load)
- `entropy` (Shannon entropy of expert distribution)
- `top_expert_hash` (hash of top-5 expert IDs for cross-run dispatch consistency check)

Aggregator parses these diagnostic lines from each cell's `server.log` and emits per-cell occupancy summary JSON. **NOT used for acceptance gate**; diagnostic-only to characterize whether expert dispatch variance correlates with pp_tps within-CI variance.

Occupancy capture MUST NOT run in the same cells used for pp_tps acceptance because materializing or summarizing routing indices can perturb the MoE hot path. If T2 runs, split it into:
- **T2.A acceptance sweep**: `--nonce-seed FIXED_SEED`, no occupancy logging, same 6-cell envelope gate as T1.
- **T2.B diagnostic occupancy capture**: short opt-in capture with `IRONMLX_EXPERT_OCCUPANCY_LOG=1`, not used for envelope PASS/FAIL.

### § 5.3 T2 sweep design (if triggered)

T2.A uses the same 6-cell shape as T1 + `--nonce-seed FIXED_SEED`, with occupancy logging disabled. Wall ~4 hr (same as T1). T2.B is diagnostic-only and should use the smallest repeat/run count that can expose occupancy variability without turning into a second acceptance matrix; target ≤30 minutes and stay inside the T2 budget unless Boss explicitly extends it. Total T1+T2 budget = 8 hr GPU unless Boss explicitly extends it.

## § 6 Predeclared exclusion rules (Codex round-1 binding; matches P5h+2.d § 7)

| Rule | Status | Application |
|---|---|---|
| A (RUNS bump) | n/a | RUNS=15 fixed |
| **B** (drop first 1-2 cold-start runs) | OFF by default; allowed only as a predeclared, tested tool option | for envelope NUMBER trim only; pattern analyzer uses RAW; no manual trimming |
| ~~C (conditional drop last N)~~ | **REMOVED** | inherited from P5h+2.d; cannot exclude what we're studying |
| **D** (Rule D scan) | KEPT | inherits from T0 P5h+2.d work; any server.log ERROR → cell FAILS; non-allow-listed WARN → review |
| ~~E (post-hoc trim)~~ | **REMOVED** | Codex round-1 explicit prohibition: NO post-hoc rules to make PP=128 pass |

## § 7 Tasks + branch + budget + single commit

### § 7.1 Task split (Codex round-1 Q14 binding — T0 = preheat protocol; nonce/occupancy = T2 conditional)

| Task | Subject | Trigger |
|---|---|---|
| T0 | Harness `P5I_C_PREHEAT_PP_LIST` env var + `{pp}` substitution + driver pass-through + 1 unit test (preheat invocation captures multi-shape args) | always |
| T1 | Equal-budget same-shape preheat sweep (6 cells) + Acceptance gate analysis | always |
| T2 | (conditional on T1 weak/FAIL + Boss approval) iron-bench `--nonce-seed N` flag + T2.A 6-cell acceptance sweep without occupancy logging + T2.B short `IRONMLX_EXPERT_OCCUPANCY_LOG` diagnostic + analysis | gated |
| T3 | Close-out single commit attaching all infra + tests + docs + Phase 0 backfill | always |

= 4 tasks (3 always + 1 conditional); within `[feedback-task-breakdown-bounded]` ≤ 7 cap.

### § 7.2 Branch (Codex round-1 Q11 binding)

New branch `ironmlx-p5h+2-e-pp128-investigation` off `110a181`. Boss pushes current `ironmlx-p5h+2-d-thermal-investigation` branch BEFORE forking (4 commits ahead of origin).

### § 7.3 Budget (Codex round-1 Q13 binding — T1 cap with T2 needing Boss approval)

| Bucket | Cap |
|---|---|
| GPU wall (T1 = ~4 hr; T2 conditional = +~4 hr) | **8 hr CAP for T1 path including buffer**; T2 extension needs separate Boss approval |
| Docs/analysis wall (spec / brainstorm / Codex iteration / close-out) | 3 hr |
| **Total** | **11 hr** (T1 path) / **15 hr** (T1+T2 path with Boss approval) |

### § 7.4 Single-commit policy

Per P5h+2.b/c/d precedent: T3 produces ONE commit attaching all WIP. T0-T2 produce WIP only.

## § 8 Phase 0 backfill mechanics (Codex round-1 Q10 binding)

Per Codex round-1: NOT just narrative. Must include actual envelope numbers as evidence. Backfill action set:

| Outcome | Backfill action |
|---|---|
| T1 Strong PASS | (a) `docs/p5i-c-phase-0-close-out.md` § 1 #4 PASS with P5h+2.e envelope numbers `PP=128 X% / PP=512 Y%` as evidence + (b) `docs/p5i-c-phase-0-ranking-snapshot.md` preamble updated with P5h+2.e closure pointer + actual numbers |
| T1 Weak/FAIL followed by T2 PASS | same as Strong PASS but with caveat narrative around T2-method PASS and the fixed nonce-sequence protocol |
| FAIL on both | Additive failed-attempt note; criterion #4 STILL FAIL/DEFERRED; escalate per § 3.2 |

## § 9 Phase 1 spec § 2.3 protocol coupling (Codex round-1 Q18 binding)

If P5h+2.e PASSES, Phase 1 spec must adopt the P5h+2.e-resolved protocol. T3 close-out commit MUST include update to:
- `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 2.3 measurement protocol binding — reference P5h+2.e-resolved protocol, including equal-budget same-shape preheat and, if T2 is the passing path, the fixed nonce-sequence setting.

## § 10 Close-out narrative language (Codex round-1 Q9 + Q17 binding)

If H1.c PASSES, close-out language MUST be:
- **"measurement protocol stabilization"** (NOT "ironmlx runtime bug fix")
- **"ironmlx PP=128 shape-warmup sensitivity"** documented as known characteristic, working around via same-shape preheat
- Deferred question "why ironmlx is preheat-shape-sensitive while omlx is not": noted as open + tagged for Phase 1 or separate ironmlx-internals phase if relevant; NOT blocking P5h+2 chain closure

If T2 PASSES after T1 weak/FAIL, close-out language MUST additionally identify it as **"fixed prompt-sequence measurement stabilization"** and document that acceptance depends on the reproducible nonce sequence. Do NOT describe that result as a runtime performance fix.

## § 11 Out-of-scope (this phase)

- H2 MLX state-decay (deferred to separate mini-phase if T1+T2 both reject)
- omlx cd=120s half-speed anomaly (P5h+2.d δ observation; benchmark-protocol artifact for omlx; NOT actionable for ironmlx; documented in P5h+2.d close-out § 3.2)
- Deep root-cause "why ironmlx needs same-shape preheat" — deferred per § 10
- Cross-device tuning (per `[project-cross-device-tuning-deferred]`)
- Phase 1 implementation (`gather_qmm_gate_up` optimization) — REMAINS BLOCKED until P5h+2.e closes PASS and Phase 1 spec § 6 G1-G4 satisfied

## § 12 Risks

| Code | Risk | Mitigation |
|---|---|---|
| R1 | **H1.c rejected** — PP=128 within-CI 4-5% persists even with equal-budget same-shape preheat | T2 H_small_batch trigger (predeclared per § 3.2) |
| R2 | **PP=512 regression** — same-shape preheat changes break PP=512 PASS | § 3.1 A2 binding; T1 sweep re-verifies PP=512 ≤ standard acceptance threshold (2.0%) |
| R3 | **PASS but root cause unknown** — H1.c PASSES but we don't understand WHY ironmlx needs same-shape preheat | Codex round-1 Q17 binding: protocol-fix PASS is acceptable for Phase 0 backfill; deeper investigation deferred per § 10 |
| R4 | **Pre-existing `[tracing]`/KVCache WARN allow-list mismatch** (inherited from P5h+2.d) | Rule D scan output reviewed per-cell; non-allow-listed WARN marked for human triage; do NOT auto-drop |
| R5 | **8 hr GPU cap under-estimated** (M5 Max + 35B MoE actually slower than assumed) | T1 single-PP-pair narrower than P5h+2.d 18-cell; safer estimate. If T1 alone overruns, Boss approval needed before T2 |
| R6 | **Spec-protocol-coupling risk to Phase 1**: if Phase 1 spec § 2.3 update is brittle (later phases adopt different protocol), Phase 1 impl gets stuck on protocol mismatch | § 9 backfill is documentation-only; if protocol later changes, just update doc |
| R7 | **Diagnostic occupancy perturbs hot path** if enabled during acceptance sweeps | § 5.2 separates T2.A acceptance from T2.B diagnostic occupancy capture |

## § 13 References

- Predecessor close-out: `docs/p5h+2-d-close-out.md` § 6 (P5h+2.e direction binding)
- Codex round-1 of P5h+2.d Stage 1 (binding): `reports/p5h+2-d-stage1-codex-review.md`
- Codex round-1 of P5h+2.e brainstorm (binding): `reports/p5h+2-e-brainstorm-codex-questions.md`
- Phase 0 close-out (backfill target): `docs/p5i-c-phase-0-close-out.md` § 1 #4
- Phase 0 ranking snapshot (backfill target): `docs/p5i-c-phase-0-ranking-snapshot.md` preamble
- Phase 1 γ-lite spec (downstream protocol coupling target): `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 2.3
- Current implementation surface:
  - `ironmlx/tests/p5i_c_phase_0_capture.rs:324 fn monolithic_preheat` (T0 target)
  - `tools/p5h_2b_protocol_experiment.py` (T0 driver pass-through target)
  - `iron-bench/src/runner.rs:271 fn nonce_seed` (T2 conditional target)
  - `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs routing_sort_pack` substep (T2 conditional instrumentation target)
- Memory: `[project-p5h-2d-findings]`, `[project-p5h-2c-findings]`, `[project-p5i-c-phase-0-findings]`, `[project-p5h-t3-findings]`
- Stage 1 + δ raw data preserved on host: `/tmp/p5h+2-d-stage1-*/` + `/tmp/p5h+2-d-omlx-*/` (until reboot)
