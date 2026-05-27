# P5i.c Phase 1 γ-lite — `gather_qmm_gate_up` Optimization Design Exploration

**Status (2026-05-27):** Stage β design converged per Codex round-2 review and this in-place review. G1/G4 are satisfied (§ 6); G2/G3 remain the Boss review/commit gates before implementation execution. Stage α T0/T1 infra shipped via separate Stage α prep commit `a9c2beb` (per Codex round-3 attribution-clarity binding — supersedes earlier fold-into-Stage-β plan; see § 4.1 + § 10). Stage α T2-T4 sweep is skipped per first-principles (`[feedback-first-principles-no-redundant-sweep]`).

**Historical original status (2026-05-25, superseded by this update):** Design exploration ONLY (γ-lite). Implementation, benchmarking, kernel modification, and acceptance verification were out of scope until the stricter gates in § 6 were satisfied.

**Date:** 2026-05-25.
**Branch:** `ironmlx-p5i-c-phase-1` off P5h+2.e Acceptance-passed HEAD (`8ff074d` lineage).
**Predecessor docs:**
- P5h+2.d spec § 11 (Phase 1 parallel boundaries): `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md`
- P5i.c Phase 0 ranking snapshot: `docs/p5i-c-phase-0-ranking-snapshot.md`
- P5i.c Phase 0 close-out: `docs/p5i-c-phase-0-close-out.md`
- Codex round-1 brainstorm review (gitignored): `reports/p5i-c-phase-1-brainstorm-codex-questions.md`

---

## § 0 Goal + scope binding

Design and implementation boundary for `gather_qmm_gate_up` optimization candidate (Phase 0 R1 default rule trigger; cross-PP tier-1 stable at PP=128 23.38% / PP=512 22.84% share). The original γ-lite output was docs-only. This update records the Stage β design lock and the constraints that the implementation plan MUST follow after Boss approval.

**This phase explicitly does NOT commit to:**
- Project-level +10% target (gated on Phase 1 + Phase 2+ combined, § 2)
- Any project-level acceptance criteria depending on the stable vs-omlx delta, which remains separate from the Phase 1 local gate
- Starting implementation execution before Boss approves this updated spec and the subsequent implementation plan (§ 6)

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
| L1 | `gather_qmm_gate_up` substep wall-time reduced by ≥30% vs the implementation-start baseline HEAD, excluding `routing_sort_pack` | same-cohort P5h production-mode diagnostic at PP=128 and PP=512; use `default_profile` or `buffered_profile` because `quiet_acceptance` suppresses `[p5h-profile]` substep lines; compare baseline and candidate under the same logging mode |
| L2 | End-to-end pp_tps improvement ≥5% vs the same implementation-start baseline (statistically significant under the accepted per-PP envelope target) | `quiet_acceptance` production protocol with `tools/p5i_c_pp_tps_envelope.py` per PP × 3 repeats; requires envelope PASS first |

### § 2.2 Project-level gate (NOT discharged by Phase 1)

Project target `ironmlx >= 1.10 * omlx` is a multi-phase aggregate, evaluated AFTER Phase 1 + Phase 2 (`down`) + any other phase combined. Phase 1 alone CANNOT discharge this gate; project gate evaluation is deferred.

### § 2.3 Measurement protocol binding

All Phase 1 end-to-end acceptance measurements MUST use the accepted production protocol lineage: Phase 0 production-mode capture harness (`P5I_C_MODE=production`), `same_spawn_per_pp` lifecycle, `quiet_acceptance` logging, plus the P5h+2.e-resolved protocol from `docs/p5h+2-e-close-out.md`: cooldown `120s`, equal-budget same-shape preheat (`P5I_C_PREHEAT_PP_LIST="512,{pp}"`, `P5I_C_PREHEAT_RUNS=550`), and `tools/p5i_c_pp_tps_envelope.py` per-PP acceptance targets (`small_pp_acceptance_threshold` for PP=128; `standard_acceptance_threshold` otherwise).

Diagnostic measurements have narrower authority:
- Bench-kernel measurements select tile/threadgroup candidates and discharge EG-1 only; they do not prove L2.
- P5h substep diagnostics for L1 may use `default_profile` or `buffered_profile` because `quiet_acceptance` intentionally suppresses info-level `[p5h-profile]` decomposition lines. They MUST compare baseline and candidate under the same logging mode and MUST NOT be used as e2e acceptance evidence.
- Production e2e sweeps MUST NOT be used for tile search. Tile search stays bench-kernel-only and bounded by § 4.2.3.

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

### § 4.1 Stage α — instrumentation infra ship (sweep SKIPPED per first-principles)

**Status (2026-05-27)**: T0 sub-span instrumentation + T1 aggregator extension shipped via **Stage α prep commit** (separate from Stage β close-out per Codex round-3 attribution-clarity binding; see § 10). T2 12-cell diagnostic sweep, T3 cost decomposition analysis, T4 standalone close-out: **SKIPPED** per Boss + controller first-principles agreement on 2026-05-27. Codex round-2 Q6: agreed SKIP.

**Rationale for SKIP** (recorded in `[feedback-first-principles-no-redundant-sweep]`):

| Substep | MLX op semantics | Physically-bounded share |
|---|---|---|
| `gate_up_input_shape_prep` | `expand_dims_on` = view-only O(1) shape descriptor; no data motion | < 0.5% |
| `gate_up_gather_qmm_call` | `gather_quantized_matmul_on` = 4-bit MoE GEMM dominant compute | > 95% |
| `gate_up_slice_outputs` | 2× `slice_on` = view / lightweight strided indexing | < 5% |

The distribution is determined by MLX op semantics; a 12-cell sweep at any granularity can only confirm what physics already implies. Stage β direction (replace `gather_quantized_matmul_on` with a custom Metal kernel) is invariant under any sweep outcome.

**Shipped infra (Stage α prep commit)**:
- `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`: 3 child sub-spans wrapping `gather_qmm_gate_up`; runtime-gated by `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1` (OnceLock-cached); cfg-gated `p5h-profile`; default production path unaffected (env-default OFF; production build path doesn't compile the helpers)
- `ironmlx/src/core/p5h.rs` + `tools/p5h_aggregator/schema_validator.py`: 3 child span names registered in Rust + Python Lane-B allow-list (lockstep test verifies parity)
- `tools/p5h_aggregator/multi_repeat.py`: `attribute_child_spans()` + `load_spans_for_child_attribution()` for span_id/parent_span_id tree-identity attribution

**Future use**: any time Stage β v1 ships and we need to verify `expand_dims_on` + `slice_on` boundary overhead is < 3% (Q1 v2 trigger), the infra is in place — flip env var to `1` and re-run aggregator. No re-instrumentation needed.

### § 4.2 Stage β — custom Metal gather kernel (Codex round-2 locked design)

**Status (2026-05-27)**: Design is locked subject to Boss acceptance of this updated spec. Implementation planning MAY start after this spec review closes; implementation execution still requires the Stage β implementation plan review gate (§ 6).

**Purpose**: replace `mlx::quantization::gather_quantized_matmul_on` for the gate_up call sites in `SparseMoeBlock::forward_on` (sorted + default branches) with a custom Metal kernel that produces fused `(gate, up)` output, eliminating the dominant cost identified by op semantics in § 4.1.

#### § 4.2.1 Scope (v1 — Codex Q1 binding: v1 ≡ Option A)

**IN SCOPE for v1**:
- Custom Metal kernel for 4-bit affine `group_size=64` quantized matmul with gather (rhs_indices) routing
- Kernel output shape `[..., 2I]` (fused gate + up channels concatenated along last dim, matching P5i.a T2 fused weight)
- Rust caller performs `slice_on` (gate_out, up_out) on kernel output — **slice stays in Rust**, kernel does not return two buffers
- Kernel consumes existing sorted-branch (`sorted_x`, `sorted_topk_2d`) or default-branch (`expand_dims`-shaped `x`, `inds_u32`) inputs — kernel does **not** generate `sort_perm` (Codex Q4.1 binding: routing_sort_pack stays out)
- Kernel API parameterized on `(weight_layout, output_shape_constraint)` to scaffold Phase 2 `gather_qmm_down` interface (per § 7); Phase 2 kernel is **not** implemented in v1 (Codex R4 binding)

**OUT OF SCOPE for v1** (deferred to v2 conditional; see § 4.2.5):
- Absorbing `expand_dims_on` (input prep) into kernel
- Absorbing `slice_on` (output split into separate gate/up buffers) into kernel

**Hardware target**: M5 Max + 128 GB UMA only initial; cross-device tile tuning deferred per `[project-cross-device-tuning-deferred]`.

#### § 4.2.2 Kernel structure starting point (Codex Q4.2 binding)

- **Fork** ironmlx P8a stage 9 `self_qmm` Q4_K_M Metal kernel (block 256 / super-block 8; +35% mlx bench-kernel baseline; 1.32× must gate at PP=2048 per `[project-p8a-stage9-findings]`) as kernel structure quality reference
- **Add** routing dimension handling (gather via `rhs_indices`); independent design — llama.cpp `ggml_metal_mul_mat_id_q4_k_f32` is observation only per `[feedback-no-spec-from-competitors]`
- **Bench-kernel harness MUST add a routing variant** (Codex Q4.2 binding); non-routing P8a stage 9 self_qmm only proves tile structure quality, not gather kernel correctness/perf

#### § 4.2.3 Tile / threadgroup geometry (Codex Q5.3 binding)

- 4-dim lookup `(device=M5Max, quant=Q4_affine_gs64, shape=routing-gathered, phase=prefill)` per `[feedback-device-aware-tile]`
- Tile selection performed at **bench-kernel level**, small candidate set, ≤ 1-2 hr GPU budget; **NOT** in production e2e sweep
- Decode-phase tile selection deferred (per § 9)

#### § 4.2.4 Model parameter handling (Codex Q4.3 binding)

Kernel + plan + tests MUST read model dimensions from `text_config` / tensor shapes at runtime — **DO NOT hardcode** `hidden_size`, `moe_intermediate_size`, `num_experts`, `num_experts_per_tok`, `head_dim`, `num_hidden_layers`. Current Qwen3.5-35B-A3B-4bit verified values (for sanity-check only):

| Field | Value |
|---|---|
| `hidden_size` (H) | 2048 |
| `moe_intermediate_size` (I) | 512 |
| `num_experts` (E) | 256 |
| `num_experts_per_tok` (k) | 8 |
| `head_dim` | 256 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 |
| `num_hidden_layers` | 40 |
| `quantization` | `{bits: 4, group_size: 64, mode: affine}` |
| `shared_expert_intermediate_size` | 512 |

#### § 4.2.5 v2 conditional trigger (Codex Q1 binding)

Stage β v2 (absorb `expand_dims` + `slice`) MAY be considered **only if** post-v1 measurement (via Stage α T0/T1 infra) shows `expand_dims_on` + `slice_on` boundary overhead **> 3%** of `gate_up_gather_qmm_call` substep. v2 is a separate design pass; v1 close-out commits the kernel, wiring, oracle, acceptance evidence, and close-out doc. Stage α infra already shipped in prep commit `a9c2beb`; v1 close-out does not fold it again.

#### § 4.2.6 Acceptance gates (Codex Q3 binding)

Stage β plan MUST embed mandatory early-stop gates before irreversible or expensive work:

| Gate | Condition |
|---|---|
| EG-1 | bench-kernel routing variant achieves kernel-only ≥ 30% reduction vs MLX `gather_quantized_matmul_on` baseline (same shape) |
| EG-2a | Kernel-level correctness oracle (§ 5 shape matrix + shape-forced sorted/default branches + gate/up slice equivalence + top-k invariance) stable before production wiring |
| EG-2b | 35B-A3B-4bit 5-prompt generation regression stable after production wiring smoke and before any L1/L2 acceptance measurement |

If EG-1 fails: kernel design re-iterate (T0-T2); production wiring BLOCKED. If EG-2a fails: kernel implementation fixed; do not wire. If EG-2b fails: L1/L2 acceptance measurement and close-out BLOCKED; fix kernel or wiring first. Only after EG-1 + EG-2a + production wiring smoke + EG-2b pass may the plan run L1/L2 acceptance.

L1 + L2 acceptance per § 2.1 unchanged. Codex R1 binding: L1 same-cohort median comparison MUST use `gather_qmm_gate_up` (or its renamed equivalent) substep with **NO** `routing_sort_pack` contamination.

### § 4.3 Staging order discipline (Stage α SKIPPED; Stage β unblocked)

The earlier requirement to complete Stage α before specifying Stage β is **superseded** by first-principles SKIP (§ 4.1 rationale). Stage β design unblocked based on:
1. MLX op semantics determining cost distribution
2. P5h+1 op-level findings (`[project-p5h-findings]`: MoE substep gate_up GEMM saturated at op-level)
3. P8a stage 9 self_qmm precedent for Q4_K_M kernel structure

γ-lite output boundary (§ 10) updated accordingly: Stage β design is now the intended output of Phase 1 design phase; implementation plan and execution proceed only through § 6 gates.

Codex R3 binding: fused weight (`fused_gate_up`) lazy-build MUST be materialized before any measurement cell — covered by current measurement protocol § 2.3 preheat (550 runs equal-budget same-shape) + Rule B first-run trim as secondary guard. Codex R2 binding: oracle MUST shape-force sorted vs default branches (not rely on PP=128/512 naturally hitting `SORTED_ROUTING_MIN_BS_K` boundary).

## § 5 Correctness oracle requirements (Codex round-1 Q13 binding addition)

Eventual Phase 1 implementation testing MUST cover (spec scope for plan-stage):

- **Profile + production path equivalence**: both `#[cfg(feature = "p5h-profile")]` span path and default production path in `SparseMoeBlock::forward_on`. The profile path currently uses rank-4 sorted tensors; the default production path uses the P5i.a rank-3 sorted simplification.
- **Sorted vs default routing equivalence**: both routing branches (`BS*k >= SORTED_ROUTING_MIN_BS_K` and `< SORTED_ROUTING_MIN_BS_K`) — Codex round-2 R2 binding: oracle **MUST shape-force** both branches (e.g., explicit `BS*k` boundary forcing in oracle inputs), not rely on PP=128/512 naturally hitting the boundary.
- **Gate/up slice equivalence**: optimized kernel's `(gate_out, up_out)` numerically equal current `gather_quantized_matmul_on(fused) + slice` output within MLX eval precision tolerance
- **Top-k order invariance**: final routed output equivalent when the selected expert set is the same but top-k order differs; intermediate gate/up comparisons must canonicalize `(token, expert, score)` tuples before comparing
- **Weighted reduce equivalence**: post-MoE weighted_reduce output matches pre-optimization Qwen3.5-35B-A3B-4bit reference

Oracle MUST run against the actual MoE 35B model on a regression-test prompt suite — Codex round-2 Q2 binding scope: 5-10 prompts covering raw text + chat-template, short + medium-length; e2e acceptance only PP=128 + PP=512 (PP=2048 smoke pass not required as gate). Exact prompt list materialized in Stage β implementation plan, NOT this spec.

### § 5.1 Shape matrix the implementation plan must preserve

| Build/path | `x` shape | `rhs_indices` shape | `gate_out/up_out` shape |
|---|---|---|---|
| profile sorted | `[BS*k, 1, 1, H]` | `[BS*k, 1]` | `[BS*k, 1, 1, I]` |
| profile default | `[BS, 1, 1, H]` | `[BS, k]` | `[BS, k, 1, I]` |
| production sorted | `[BS*k, 1, H]` | `[BS*k]` | `[BS*k, 1, I]` |
| production default | `[BS, 1, 1, H]` | `[BS, k]` | `[BS, k, 1, I]` |

## § 6 Implementation execution gating (Codex round-1 binding — STRICTER)

Phase 1 implementation execution (kernel code, production wiring, performance benchmark execution) MAY start ONLY when ALL hold:

| # | Condition |
|---|---|
| G1 | P5h+2.e close-out doc (`docs/p5h+2-e-close-out.md`) EXPLICITLY backfills Phase 0 § 7 #4 PASS and allows Phase 1 implementation to proceed to Boss approval. **Satisfied by commit `9a35ae17`.** |
| G2 | Boss explicitly approves this updated Stage β design and the subsequent implementation plan. **Pending current review + plan review.** |
| G3 | Phase 1 design spec (this doc) committed + Boss-approved. **In-place updated 2026-05-27 with Codex round-2 bindings; pending Boss review + commit.** |
| G4 | New branch `ironmlx-p5i-c-phase-1` forked from a P5h+2.e Acceptance-passed HEAD. **Satisfied** (branch forked off `8ff074d`; Stage α T0/T1 working-tree state). |

**NOT acceptable** as sole conditions:
- ~~P5h+2.d Stage 1 Mechanism gate = strong/weak yes alone~~ (per Codex binding; Mechanism gate is intermediate signal, not unblock authority)
- ~~P5h+2.d Acceptance gate intermediate progress~~ (must be superseded by CLOSED P5h+2.e + Phase 0 backfilled, not in-progress)
- ~~P5h+2.e in-progress envelope numbers without close-out~~ (satisfied only after `docs/p5h+2-e-close-out.md` exists with final evidence)

Until G1-G4 all hold: no Stage β kernel implementation, no production-runtime wiring, and no performance benchmark execution. Drafting and reviewing the implementation plan is not implementation execution.

### § 6.5 Codex round-2 bindings (audit trail)

Source: `reports/p5i-c-phase-1-stage-beta-design-questions.md` (gitignored review report).

| Binding | Subject | Effect in this spec |
|---|---|---|
| Q1 | Stage β v1 ≡ Option A (no absorb expand_dims/slice); v2 conditional on > 3% boundary residual | § 4.2.1 IN/OUT scope; § 4.2.5 v2 trigger |
| Q2 | Shape-forced sorted/default oracle + 5-10 prompt 35B regression + acceptance only PP=128+512 (PP=2048 smoke not gate) | § 5 oracle requirements (R2 reinforced); planned in Stage β plan |
| Q3 | Single plan 6-7 task + mandatory early-stop gates (bench-kernel ≥30%; kernel oracle before wiring; 35B regression before L1/L2) | § 4.2.6 EG-1 + EG-2a + EG-2b |
| Q4.1 | NOT integrate `routing_sort_pack` / sort_perm generation in v1 kernel | § 4.2.1 IN scope kernel consumes existing inputs only |
| Q4.2 | bench-kernel MUST add routing variant; non-routing self_qmm only proves tile structure | § 4.2.2 starting point + plan T0 |
| Q4.3 | NO hardcode H/I/E/k/head_dim; read from `text_config` / tensor shape | § 4.2.4 model parameter handling table |
| Q5.1 | NO dispatch-time MLX fallback | § 9 explicit OOS |
| Q5.2 | NO alt quantization scheme evaluation | § 9 explicit OOS |
| Q5.3 | NO production e2e sweep for tile search (bench-kernel only ≤ 1-2 hr GPU) | § 4.2.3 tile geometry |
| Q6 | Stage α SKIP agreed | § 4.1 SKIP rationale |
| R1 | L1 same-cohort median; NO routing_sort_pack contamination | § 4.2.6 EG-1 cohort definition |
| R2 | Shape-forced sorted/default branch oracle | § 5 second bullet updated |
| R3 | Fused weight lazy build pre-materialize before measurement | § 4.3 protocol note |
| R4 | Phase 2 scaffold only interface; no `gather_qmm_down` impl in v1 | § 4.2.1 IN/OUT scope |

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
| R2 | **Op-level matmul saturation lesson** (P5g): pure tile tweaks may saturate fast | § 4.2 — Stage β integrates gather routing into a custom kernel and uses EG-1 before production wiring |
| R3 | **Sorted/default dual-path risk**: PP=128 vs PP=512 may take different `SORTED_ROUTING_MIN_BS_K` branches; optimization may help one but not the other | § 5 oracle covers both; § 2.1 acceptance requires both PP=128 + PP=512 |
| R4 | **Fused weight lazy build noise**: `fused_gate_up(target)` lazy on first forward — must be excluded from measurement | Phase 1 measurement protocol must include explicit preheat/materialization before measured cells; Rule B first-run trim is a secondary guard, not the primary mitigation |
| R5 | **MLX op boundary / slice / eval materialization** may be dominant cost (not matmul math); custom Metal alone may not solve | § 3 W4 + § 4.2 — Stage β explicitly designs for routing/memory/shape overhead, not just compute tile |
| R6 | **Protocol mismatch**: bench-kernel, substep profile, and production e2e answer different questions | § 2.3 taxonomy binds each evidence type: bench-kernel only discharges EG-1/tile diagnostics; L1 uses same-cohort P5h diagnostic; L2 uses `quiet_acceptance` |
| R7 | **Implementation gate drift**: design text may be read as authorization to start kernel work before Boss review / plan review / commit gates close | § 6 keeps G2/G3 pending and explicitly blocks Stage β kernel implementation until all gates hold |
| R8 | **Correctness regression on 35B-A3B-4bit**: hard to detect in Metal-level optimizations; numerical drift may pass unit oracles but fail downstream generation quality | § 5 oracle must run actual generation regression on MoE 35B reference prompts |
| R9 | **Unsupported shape or quant silently falling back to MLX**: fallback would hide dispatch mistakes and corrupt attribution | § 9 forbids dispatch-time MLX fallback; unsupported shapes MUST fail before enabling the new kernel |

## § 9 Out-of-scope (this phase)

Defer to later phases:
- `gather_qmm_down` (tier-2 ~12% share) — Phase 2 (interface-reserved per § 7; v1 kernel impl explicitly **not** in Stage β per Codex R4)
- `swiglu_activation` fusion — Phase 2+ (NOT interface-reserved; significant additional design)
- `routing_sort_pack` co-design with gather — Phase 3+ (architectural change; Codex Q4.1 binding: Stage β kernel consumes existing sort_perm output, does NOT generate it)
- Cross-device tile tuning beyond M5 Max — separate phase per `[project-cross-device-tuning-deferred]`
- Decode-phase tile selection (`phase=decode` in 4-dim lookup) — separate phase
- Attention family (`gda_step_1a_in_proj_qkvz` tier-3 10-15% share) — separate phase
- Qwen3.5 Dense (4B) optimization — separate, not MoE blocked
- Project-level +10% pp_tps achievement — multi-phase aggregate, deferred

**Explicitly OUT for Stage β v1 per Codex round-2 bindings**:
- Absorbing `expand_dims_on` (input prep) into kernel (Q1; defer to v2 conditional § 4.2.5)
- Absorbing `slice_on` (output split) into kernel (Q1; defer to v2 conditional § 4.2.5)
- Dispatch-time MLX `gather_quantized_matmul_on` fallback (Q5.1; fall-back is anti-pattern; new kernel MUST be unconditional or not enabled; unsupported shape/quant combinations block enabling the new path instead of falling back at dispatch)
- Alternative quantization schemes — Q4_0 / Q4_K_S / Q5_K_M / non-affine (Q5.2; v1 locked on current model's `Q4_affine_gs64`)
- Production e2e sweep as tile search method (Q5.3; tile selection bench-kernel only ≤ 1-2 hr GPU)

## § 10 γ-lite output boundary (Codex round-1 Q7 binding) + Stage β design lock (2026-05-27)

**γ-lite exit status (2026-05-27)**: G1/G4 satisfied per § 6; Stage β design is converged but still awaits Boss approval and commit closure through G2/G3. Phase moves out of γ-lite only after those gates close.

This spec is the primary tracked design artifact across Phase 1.

**Produced in γ-lite (closed)**:
- This design spec (initial brainstorm → Codex round-1 bindings)
- Codex round-1 consultation doc (gitignored): `reports/p5i-c-phase-1-brainstorm-codex-questions.md`
- Memory entry: `[project-p5i-c-phase-1-findings]` (γ-lite outcome + readiness)

**Produced in Stage β design lock (2026-05-27, this in-place update)**:
- § 4 rewrite with Codex round-2 bindings
- § 6.5 Codex round-2 audit trail
- § 9 expanded OOS items
- Codex round-2 consultation doc (gitignored): `reports/p5i-c-phase-1-stage-beta-design-questions.md`

**To produce next after this design review closes**:
- Stage β implementation plan (`docs/superpowers/plans/2026-05-27-ironmlx-p5i-c-phase-1-stage-beta-gather-kernel.md`) — 6-7 tasks per Codex Q3 binding + early-stop gates EG-1/EG-2a/EG-2b per § 4.2.6; drafting the plan is allowed before full G2 closes, but execution waits for Boss plan approval.

**Two-commit pattern (Codex round-3 attribution-clarity binding, supersedes earlier fold plan)**:
1. **Stage α prep commit** (`a9c2beb`, 2026-05-27): T0/T1 child-span instrumentation infra (6 files) + this spec § 4.1 / § 10 alignment edit. Standalone shippable; not gated on Stage β.
2. **Stage β close-out commit** (post-G1-G4 + plan + impl + acceptance): custom Metal gather kernel (`ironmlx-bench-kernel` + `ironmlx/src/...`) + correctness oracle harness + L1/L2 acceptance evidence + close-out doc. Single commit per `[feedback-no-empty-commits]` (kernel + wiring + acceptance + close-out doc are coherent perf deliverable).

Rationale: T0/T1 instrumentation infra + Stage β kernel wiring both touch `sparse_moe.rs` + `p5h.rs`. Folding them into one commit creates same-file two-layer modification merge → attribution mixed (T0 instrumentation vs Stage β wiring indistinguishable in hunks) + bisect-hostile (cannot localize regression to instrumentation vs kernel). Codex round-3 binding: attribution clarity outranks single-commit form.

## § 11 References

- Spec source chain: this doc + `reports/p5i-c-phase-1-brainstorm-codex-questions.md` (round-1) + `reports/p5i-c-phase-1-stage-beta-design-questions.md` (round-2)
- Predecessor specs / plans:
  - `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md` § 11 (binding parent)
  - `docs/p5i-c-phase-0-ranking-snapshot.md` (Phase 0 R1 candidate evidence)
  - `docs/p5i-c-phase-0-close-out.md` § 1 #4 (Acceptance gate backfilled PASS by P5h+2.e)
- Current implementation surface: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` profile path `gather_qmm_gate_up` span plus default production path; both use P5i.a T2 fused gate+up weight, but sorted-shape rank differs by build path.
- Reference observations (NOT specifications per `[feedback-no-spec-from-competitors]`):
  - llama.cpp `ggml_metal_mul_mat_id_q4_k_f32` (gather Q4_K kernel structure)
  - mlx `gather_quantized_matmul` (current implementation; ironmlx calls via `mlx::quantization::gather_quantized_matmul_on`)
  - P8a stage 9 `self_qmm` Q4_K_M kernel (Metal kernel structure quality reference; NOT routing handling)
- Memory: `[project-p5i-c-phase-0-findings]`, `[project-p5h-t3-findings]`, `[project-p8a-stage9-findings]`, `[project-p5h-findings]`, `[project-p5h-2e-findings]`, `[feedback-device-aware-tile]`, `[project-cross-device-tuning-deferred]`, `[project-p5g-findings]`, `[feedback-design-philosophy]`, `[feedback-no-spec-from-competitors]`, `[feedback-first-principles-no-redundant-sweep]`, `[feedback-performance-stability-priority]`, `[reference-current-machine]`
- Codex round-1 brainstorm review: `reports/p5i-c-phase-1-brainstorm-codex-questions.md`
