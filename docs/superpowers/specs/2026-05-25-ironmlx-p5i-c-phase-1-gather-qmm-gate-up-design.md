# P5i.c Phase 1 γ-lite — `gather_qmm_gate_up` Optimization Design Exploration

**Status:** Design exploration ONLY (γ-lite). DO NOT commit until Boss approves per `[feedback-review-spec-before-commit]`. Implementation, benchmarking, kernel modification, and acceptance verification are ALL OUT-OF-SCOPE until § 6 G1-G4 are satisfied.

**Date:** 2026-05-25.
**Branch (proposed):** `ironmlx-p5i-c-phase-1-brainstorm` off the Boss-approved P5h+2.d design/plan base. If forked before P5h+2.d implementation closes, keep this branch docs-only so P5h+2.d measurement WIP stays isolated.
**Predecessor docs:**
- P5h+2.d spec § 11 (Phase 1 parallel boundaries): `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md`
- P5i.c Phase 0 ranking snapshot: `docs/p5i-c-phase-0-ranking-snapshot.md`
- P5i.c Phase 0 close-out: `docs/p5i-c-phase-0-close-out.md`
- Codex round-1 brainstorm review (gitignored): `reports/p5i-c-phase-1-brainstorm-codex-questions.md`

---

## § 0 Goal + scope binding

Design exploration of `gather_qmm_gate_up` optimization candidate (Phase 0 R1 default rule trigger; cross-PP tier-1 stable at PP=128 23.38% / PP=512 22.84% share). Output of this phase = **docs-only design artifacts**, with this spec as the primary tracked artifact. NO implementation code. NO kernel modifications. NO performance benchmarks. NO writing-plans handoff.

**This phase explicitly does NOT commit to:**
- Implementation timing (gated on stricter conditions, § 6)
- Project-level +10% target (gated on Phase 1 + Phase 2+ combined, § 2)
- Any acceptance criteria depending on Phase 0 envelope status currently FAIL/DEFERRED

## § 1 Phase 0 evidence + Amdahl analysis

### § 1.1 Hot-path share (from Phase 0 ranking)

| Tier | Candidate | PP=128 share | PP=512 share | Phase 1 relationship |
|---|---|---|---|---|
| 1 | `gather_qmm_gate_up` | 23.38% (±0.06%) | 22.84% (±0.35%) | **THIS PHASE** |
| 2 | `gather_qmm_down` | 12.34% (±0.08%) | 11.69% (±0.11%) | Phase 2 (interface-reserved per § 7) |
| 9-11 | `swiglu_activation` | 3.21% | 2.19% | Phase 2+ (not interface-reserved) |

### § 1.2 Amdahl ceiling (Codex round-1 binding)

`gate_up` ~23% share establishes a hard upper bound on Phase-1-alone end-to-end gain vs the current ironmlx baseline. This table is **NOT** a project-level vs-omlx gate; the project gate is `ironmlx >= 1.10 * omlx` and must wait for a stable P5h+2.d-backed comparator envelope.

Throughput gain is computed as `1 / (1 - share * reduction) - 1`.

| `gate_up` substep reduction | Theoretical e2e pp_tps gain vs ironmlx baseline |
|---|---|
| 30% | ~+7% |
| 43% | ~+11% |
| 50% | ~+13% |
| 100% | ~+30% (theoretical upper, eliminate candidate entirely) |

**Implication**: Phase 1 alone can plausibly produce a meaningful own-baseline gain, but it cannot by itself discharge the project-level `ironmlx >= 1.10 * omlx` target. Even a ≥43% `gate_up` reduction only means ~+10% vs ironmlx's own baseline; whether that beats omlx depends on the stable vs-omlx delta after P5h+2.d. Phase 2+ (`down` + others) remains the realistic path to project-level closure.

## § 2 Two-layer acceptance framework (Codex round-1 Q6 binding)

### § 2.1 Phase-1-local gate (this phase's success criteria)

Both must hold when eventually measured:

| # | Criterion | Method |
|---|---|---|
| L1 | `gather_qmm_gate_up` substep wall-time (P5h-style production-mode instrumentation) reduced by ≥30% vs the implementation-start baseline HEAD | per-substep median across ≥3 fresh-spawn repeats, within envelope precision |
| L2 | End-to-end pp_tps improvement ≥5% vs the same implementation-start baseline (statistically significant under the accepted per-PP envelope target) | `tools/p5i_c_pp_tps_envelope.py` per PP × 3 repeats, requires envelope PASS first |

### § 2.2 Project-level gate (NOT discharged by Phase 1)

Project target `ironmlx >= 1.10 * omlx` is a multi-phase aggregate, evaluated AFTER Phase 1 + Phase 2 (`down`) + any other phase combined. Phase 1 alone CANNOT discharge this gate; project gate evaluation is deferred.

### § 2.3 Measurement protocol binding

All Phase 1 measurements MUST use the accepted production protocol lineage: Phase 0 production-mode capture harness (`P5I_C_MODE=production`), `same_spawn_per_pp` lifecycle, `quiet_acceptance` logging, and the final P5h+2.e-resolved protocol once P5h+2.e closes. Until P5h+2.e backfills Phase 0 § 7 #4 PASS, Phase 1 implementation and measurement remain blocked. Mismatched protocol (e.g., bench-kernel isolated micro-benchmark) MUST NOT be used as primary evidence — only as secondary diagnostic.

## § 3 Hypothesis set — why `gate_up` is 23% (Codex round-1 Q18 binding)

P5h T3 layer-3 MoE diagnostic measured `gather_qmm_gate_up` at 1.25us per layer (PP=128 narrow layer-3 window). Phase 0 production-mode shows 23% share. The gap is real, not measurement error. Multi-source mechanism, NOT single cause:

| Code | Hypothesis | Implication for Phase 1 |
|---|---|---|
| W1 | Phase 0 attribution covers ALL decoder layers steady-state, not just layer-3 | Optimization affects all 40 MoE layers; per-layer gain amplified |
| W2 | P5i.a T2 fused gate+up weight → larger single span; cost concentrates in single substep, not split | Existing fused path is the right surface to optimize on |
| W3 | P5h T3 narrow-window diagnostic insufficient as baseline; gather/setup/memory/shape costs not isolated there | Baseline reference MUST be Phase 0 production-mode, NOT T3 layer-3 |
| W4 | Wrapper gap fix (P5h+1) exposed previously-hidden gather/setup/memory cost in the span | Real cost surface includes routing, indexing, memory-shuffle — NOT just matmul math |

**Key implication for Phase 1 design**: pure tile-selection improvements may saturate quickly if the dominant cost is NOT matmul math (W4). Custom Metal kernel design MUST consider routing + memory-shuffle + slice + eval-boundary overhead, not just compute tile geometry.

## § 4 Technical direction — staged α → β (Codex round-1 Q2 binding, rephrased)

### § 4.1 Stage α — diagnostic + low-risk verification

**Purpose**: cost decomposition + low-risk early signal. NOT a "likely sufficient" optimization path on its own (Codex binding).

Investigations (in eventual implementation; Phase 1 design only describes scope):
- Device-aware tile selection sweep for `mlx::quantization::gather_quantized_matmul_on` on M5 Max (M_threshold + tile geometry; per `[feedback-device-aware-tile]` 4-dim lookup `(device=M5Max, quant=Q4, shape=routing-gathered, phase=prefill)`)
- Cost breakdown via Metal capture (sort_perm scatter, expand_dims, gather index lookup, slice gate/up, eval barriers between MLX ops)
- Identify whether the 23% is dominated by compute-tile geometry OR by routing/memory/shape overhead

Stage α deliverables: diagnostic report, NOT performance acceptance. Expected gain: bounded (5-15% per [project-p5g-findings] op-level matmul saturation lesson).

### § 4.2 Stage β — custom Metal gather kernel (real value path)

**Purpose**: address the dominant cost identified in Stage α with integrated kernel design.

Design integrates:
- Sorted-routing input shape (sort_perm + expert-bucketed token reordering)
- Expert-id index lookup (replacing MLX-API `gather_quantized_matmul_on` per-call routing)
- Fused gate + up output (matches P5i.a T2 fused weight; output `(gate_out, up_out)` returned together without intermediate materialization)
- Optional: absorbing the `expand_dims` + downstream `slice` boundaries to remove MLX op-boundary overhead

Starting point evaluation (per Codex round-1 Q4 binding): adapt llama.cpp's `ggml_metal_mul_mat_id_q4_k_f32` (gather variant that already handles routing dimension) as REFERENCE OBSERVATION only — design independently per `[feedback-design-philosophy]` + `[feedback-no-spec-from-competitors]`.

P8a stage 9 `self_qmm` Q4_K_M kernel rewrite (NOT gather variant) is reference for kernel structure quality (+35% mlx baseline at bench-kernel level; 1.32× must gate at PP=2048) but NOT for routing handling.

Stage β deliverables after Stage α completes: finalized custom Metal kernel spec + integration plan. During γ-lite, this document only locks Stage β constraints and decision criteria; concrete kernel parameters, tile shapes, threadgroup geometry, and implementation tasks are deliberately deferred until α produces a measured cost decomposition.

### § 4.3 Staging order discipline

α MUST complete + provide cost decomposition BEFORE β kernel design specifics. If α reveals that compute-tile geometry is dominant (W4 false), β can scope down to tile-only optimization with less custom-Metal risk. If α confirms routing/memory dominance (W4 true), β must address that holistically. γ-lite may document β constraints, but MUST NOT pre-choose tile sizes, threadgroup geometry, or integration tasks before α.

## § 5 Correctness oracle requirements (Codex round-1 Q13 binding addition)

Eventual Phase 1 implementation testing MUST cover (spec scope for plan-stage):

- **Profile + production path equivalence**: both `#[cfg(feature = "p5h-profile")]` span path and default production path in `SparseMoeBlock::forward_on`. The profile path currently uses rank-4 sorted tensors; the default production path uses the P5i.a rank-3 sorted simplification.
- **Sorted vs default routing equivalence**: both routing branches (`BS*k >= SORTED_ROUTING_MIN_BS_K` and `< SORTED_ROUTING_MIN_BS_K`) because PP=128/512 and future batch shapes may exercise different paths.
- **Gate/up slice equivalence**: optimized kernel's `(gate_out, up_out)` numerically equal current `gather_quantized_matmul_on(fused) + slice` output within MLX eval precision tolerance
- **Top-k order invariance**: final routed output equivalent when the selected expert set is the same but top-k order differs; intermediate gate/up comparisons must canonicalize `(token, expert, score)` tuples before comparing
- **Weighted reduce equivalence**: post-MoE weighted_reduce output matches pre-optimization Qwen3.5-35B-A3B-4bit reference

Oracle MUST run against the actual MoE 35B model on a regression-test prompt suite (TBD in Phase 1 implementation plan, NOT this spec).

### § 5.1 Shape matrix the implementation plan must preserve

| Build/path | `x` shape | `rhs_indices` shape | `gate_out/up_out` shape |
|---|---|---|---|
| profile sorted | `[BS*k, 1, 1, H]` | `[BS*k, 1]` | `[BS*k, 1, 1, I]` |
| profile default | `[BS, 1, 1, H]` | `[BS, k]` | `[BS, k, 1, I]` |
| production sorted | `[BS*k, 1, H]` | `[BS*k]` | `[BS*k, 1, I]` |
| production default | `[BS, 1, 1, H]` | `[BS, k]` | `[BS, k, 1, I]` |

## § 6 Implementation gating (Codex round-1 binding — STRICTER)

Phase 1 implementation MAY start ONLY when ALL hold:

| # | Condition |
|---|---|
| G1 | P5h+2.e close-out doc EXPLICITLY backfills Phase 0 § 7 #4 PASS and allows Phase 1 implementation to proceed to Boss approval |
| G2 | Boss explicitly approves Phase 1 implementation kick-off |
| G3 | Phase 1 design spec (this doc) committed + Boss-approved |
| G4 | New branch `ironmlx-p5i-c-phase-1` forked from a P5h+2.e Acceptance-passed HEAD |

**NOT acceptable** as sole conditions:
- ~~P5h+2.d Stage 1 Mechanism gate = strong/weak yes alone~~ (per Codex binding; Mechanism gate is intermediate signal, not unblock authority)
- ~~P5h+2.d Acceptance gate intermediate progress~~ (must be superseded by CLOSED P5h+2.e + Phase 0 backfilled, not in-progress)
- ~~P5h+2.e in-progress envelope numbers without close-out~~ (must wait for the active P5h+2.e run to finish and publish final evidence)

During γ-lite (now until G1-G4): NO benchmarks, NO kernel changes, NO production-runtime code touched.

## § 7 Sister-extension interface (Codex round-1 Q1(d) binding)

Phase 1 design DEFINES — but does not implement — an extension surface that allows Phase 2 to add `gather_qmm_down` optimization with minimal re-architecture:

- Kernel API contract: parameterize on `(weight_layout, output_shape_constraint)` rather than hardcoding gate_up specifics
- Bench-kernel target shape: scaffold supports both `gate_up_fused` and `down` weight shapes
- Test oracle infrastructure: parameterizable over substep (gate_up_fused / down)

This is a DESIGN constraint, NOT implementation work. Parameterized weight handling keeps the Phase 2 re-architecture cost bounded; it is not assumed to be free.

## § 8 Risks (Codex round-1 Q13/Q14 binding addition)

| Code | Risk | Mitigation in spec |
|---|---|---|
| R1 | **Amdahl ceiling**: gate_up alone cannot discharge `ironmlx >= 1.10 * omlx`; ≥43% reduction only means ~+10% vs ironmlx's own baseline | § 1.2 + § 2.2 — project gate explicitly deferred; Phase 1 ≠ project +10% |
| R2 | **Op-level matmul saturation lesson** (P5g): pure tile tweaks may saturate fast | § 4.3 — α is diagnostic only; β is the real path |
| R3 | **Sorted/default dual-path risk**: PP=128 vs PP=512 may take different `SORTED_ROUTING_MIN_BS_K` branches; optimization may help one but not the other | § 5 oracle covers both; § 2.1 acceptance requires both PP=128 + PP=512 |
| R4 | **Fused weight lazy build noise**: `fused_gate_up(target)` lazy on first forward — must be excluded from measurement | Phase 1 measurement protocol must include explicit preheat/materialization before measured cells; Rule B first-run trim is a secondary guard, not the primary mitigation |
| R5 | **MLX op boundary / slice / eval materialization** may be dominant cost (not matmul math); custom Metal alone may not solve | § 3 W4 + § 4.2 — Stage β explicitly designs for routing/memory/shape overhead, not just compute tile |
| R6 | **Protocol mismatch**: measuring via bench-kernel isolated harness ≠ Phase 0 production-mode | § 2.3 — production-mode protocol binding mandatory; bench-kernel diagnostic only |
| R7 | **P5h+2.d Acceptance gate FAIL**: blocks Phase 1 implementation indefinitely | § 6 G1 stays unblocked condition; Phase 1 spec remains in design state if P5h+2.d FAIL → re-brainstorm needed |
| R8 | **Correctness regression on 35B-A3B-4bit**: hard to detect in Metal-level optimizations; numerical drift may pass unit oracles but fail downstream generation quality | § 5 oracle must run actual generation regression on MoE 35B reference prompts |

## § 9 Out-of-scope (this phase)

Defer to later phases:
- `gather_qmm_down` (tier-2 ~12% share) — Phase 2 (interface-reserved per § 7)
- `swiglu_activation` fusion — Phase 2+ (NOT interface-reserved; significant additional design)
- `routing_sort_pack` co-design with gather — Phase 3+ (architectural change)
- Cross-device tile tuning beyond M5 Max — separate phase per `[project-cross-device-tuning-deferred]`
- Attention family (`gda_step_1a_in_proj_qkvz` tier-3 10-15% share) — separate phase
- Qwen3.5 Dense (4B) optimization — separate, not MoE blocked
- Project-level +10% pp_tps achievement — multi-phase aggregate, deferred

## § 10 γ-lite output boundary (Codex round-1 Q7 binding)

This spec is the primary tracked output of Phase 1 γ-lite.

**Explicitly NOT produced in γ-lite:**
- Implementation plan (`docs/superpowers/plans/...`) — written ONLY after § 6 G1-G4 satisfied
- Code changes — none in this phase
- Performance benchmarks — none in this phase
- Acceptance verification — none in this phase

**Produced:**
- This design spec
- Codex consultation doc (gitignored): `reports/p5i-c-phase-1-brainstorm-codex-questions.md`
- Memory entry: new `[project-p5i-c-phase-1-findings]` summarizing γ-lite outcome + readiness state

## § 11 References

- Spec source chain: this doc + `reports/p5i-c-phase-1-brainstorm-codex-questions.md`
- Predecessor specs / plans:
  - `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md` § 11 (binding parent)
  - `docs/p5i-c-phase-0-ranking-snapshot.md` (Phase 0 R1 candidate evidence)
  - `docs/p5i-c-phase-0-close-out.md` § 1 #4 (Acceptance gate FAIL/DEFERRED awaiting P5h+2.d)
- Current implementation surface: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` profile path `gather_qmm_gate_up` span plus default production path; both use P5i.a T2 fused gate+up weight, but sorted-shape rank differs by build path.
- Reference observations (NOT specifications per `[feedback-no-spec-from-competitors]`):
  - llama.cpp `ggml_metal_mul_mat_id_q4_k_f32` (gather Q4_K kernel structure)
  - mlx `gather_quantized_matmul` (current implementation; ironmlx calls via `mlx::quantization::gather_quantized_matmul_on`)
  - P8a stage 9 `self_qmm` Q4_K_M kernel (Metal kernel structure quality reference; NOT routing handling)
- Memory: `[project-p5i-c-phase-0-findings]`, `[project-p5h-t3-findings]`, `[project-p8a-stage9-findings]`, `[feedback-device-aware-tile]`, `[project-cross-device-tuning-deferred]`, `[project-p5g-findings]`, `[feedback-design-philosophy]`, `[feedback-no-spec-from-competitors]`
- Codex round-1 brainstorm review: `reports/p5i-c-phase-1-brainstorm-codex-questions.md`
