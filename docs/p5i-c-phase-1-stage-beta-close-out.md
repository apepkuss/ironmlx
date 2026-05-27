# P5i.c Phase 1 Stage β — Close-Out: Negative Finding on `gather_qmm_gate_up`

**Status (2026-05-27)**: Stage β closed as **negative finding**. `gather_qmm_gate_up` custom Metal kernel did not enter the EG-1 hard-stop ladder proceed band after three bounded redesign passes (first-attempt naive scalar / round-4 SG-MMA + cooperative loads single tile / round-5 sorted-only aligned with MLX `gather_qmm_rhs`). Phase 1 implementation pivots to a new candidate (`gda_step_1a_in_proj_qkvz`) per Codex round-6 Q1 binding; Stage β scope is retired.

**Predecessor commits**: `a9c2beb` (Stage α prep: T0/T1 child-span infra) + `023e174` (Stage β design lock round-2) + `5adf6db` (plan round-3) + `026dbd4` (round-4 EG-1 ladder) + `1648105` (round-5 sorted-only narrowing).
**Branch**: `ironmlx-p5i-c-phase-1`.
**Codex review rounds**: 6 (round-1 brainstorm + round-2 design lock + round-3 plan + round-4 EG-1 ladder + round-5 Option E sorted-only + round-6 pivot decision).

## § 1 Final verdict — BLOCKED per EG-1 hard-stop ladder (spec § 4.2.6 + § 4.2.7)

| Pass | Approach | sorted ratio (best) | default ratio | EG-1 verdict |
|---|---|---|---|---|
| First-attempt | Naive scalar (256 threads × 8 outputs/thread; no SG-MMA; no vec_loads) | 5.16× SLOWER | 3.25× SLOWER | BLOCKED |
| Round-4 redesign | SG-MMA + cooperative loads; single tile (32, 64, 32) self_qmm-shape | 2.35× SLOWER | 4.84× SLOWER (REGRESSION vs naive scalar) | BLOCKED; architectural insight: BM=32 incompatible with unsorted default |
| **Round-5 redesign (sorted-only)** | **MLX `gather_qmm_rhs` non-NAX tile (16, 32, 32); WM=1, WN=2; 64 threads/TG; threadgroup memory cache; per-TG expert-id boundary scan + sub-matmul slice store; default = per-cell scalar correctness-only** | **PP=128: 1.81× / PP=512: 1.0053×** | 3.97× (informational; not a gate) | **BLOCKED at worst sorted ratio 1.81×** |

Bit-identical numerical correctness preserved through all 3 passes (`max_abs_diff = 0.000000` vs MLX baseline; ~1e-2 SG-MMA register addressing bug warning threshold never tripped).

EG-1 hard-stop ladder bands (Codex round-4 + round-5 binding):
- ≤ 0.70 PASS / 0.70-0.85 PASS_WITH_DIAGNOSTIC / 0.85-1.0 HALT / > 1.0 BLOCKED

Round-5 worst sorted ratio 1.81 → BLOCKED. Tile sweep is NOT acceptable rescue per Codex round-4 Sup-2; per Codex round-5 Q1 + Q5 bounded-iteration cap is exhausted.

## § 2 Architectural root cause (per implementer analysis + Codex round-5 Q2 fact-check + round-6 Q1 binding)

**Observation 1 — Batch-size scaling shows ironmlx SG-MMA inner-loop K-axis approaches MLX efficiency at sufficient batch**:
- ironmlx PP=128 → PP=512: 1.81× → 1.0053× = **2.35× ratio improvement** with batch size
- MLX PP=128 → PP=512: 956.917 → 4055.375 us = **4.24× MLX scaling factor**
- PP=512 sorted ratio 1.0053 is essentially parity with MLX

**Observation 2 — 1.81× gap at PP=128 is dominated by small-M per-threadgroup fixed overhead**, not by ineffective inner-loop computation: kernel launch + threadgroup setup + barriers + smem staging amortize poorly at small M.

**Observation 3 — MLX advantage source is `mlx::steel` C++ template infrastructure** inaccessible via Stage β's `mlx::MetalKernel` Rust API:
- MLX `affine_gather_qmm_rhs` non-NAX (per `mlx/backend/metal/kernels/quantized.h:2245`) uses `mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, ...>` + `BlockLoader` + `QuantizedBlockLoader`
- These provide BK_padded shared-memory layout (bank-conflict prevention) + `vec<T>` ReadVector cooperative loads + direct shmem writes without register staging + template-specialized prefetch/pipeline overlap
- Stage β scope bindings forbid the necessary infrastructure access paths:
  - Codex round-2 binding: NO mlx-sys touch
  - Codex round-3 binding: no build.rs shader compilation hook
  - `mlx::steel` C++ headers not exposed through `mlx` Rust binding
- Implementer's independent raw Metal kernel (`simdgroup_half8x8` + `simdgroup_load` + `simdgroup_multiply_accumulate`) lacks these optimizations and pays the per-TG overhead difference at small-M

**Observation 4 — Default-routing path is structurally incompatible with BM=32 grouped SG-MMA** (round-4 finding; round-5 confirmed):
- Default `rhs_indices` produces ~32 unique experts per BM=32 tile → 87.5% SG-MMA 8×8 fragment waste
- MLX itself uses TWO distinct kernels: `affine_gather_qmm_rhs` for sorted; `gather_qmv_fast`/`gather_qmv` for default M=1 right_sorted=false
- Round-5 scope-narrowed default path to MLX baseline + correctness-only oracle (spec § 4.2.7)

## § 3 What was shipped vs not shipped

### Shipped (committed, retained)

- **Stage α T0/T1 child-span instrumentation infra** (commit `a9c2beb`) — runtime-gated by `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1`; cfg-gated `p5h-profile`; default production path unaffected. Future-reusable for any sub-span investigation (e.g., op-boundary measurement per Codex round-6 Q3 Pivot 2 evaluation). NOT affected by this close-out.
- **Stage β design spec** (commit `023e174` + amendments through round-5 in `1648105`) — full Codex round-1-5 audit trail (§ 6 + § 6.5 + § 6.6 + § 6.7) preserved in git history; this close-out adds § 6.8 round-6 audit.
- **Stage β implementation plan** (commit `5adf6db` + round-4 amendment `026dbd4` + round-5 amendment `1648105`) — full T0-T6 + Blocked Path structure preserved in git history; this close-out marks Stage β BLOCKED at T1 round-5.

### NOT shipped (experimental kernel code; archived only)

Per Codex round-6 Q5 binding: experimental kernel code is **not committed** to ironmlx codebase.

Working tree experimental code (5 files = `ironmlx/src/nn/gather_qmm/{mod, kernel, lookup}.rs + metal/qmm_gather.metal.in + ironmlx-bench-kernel/src/gather_qmm.rs` + 2 modified T0 carryover = `ironmlx/src/nn/mod.rs + ironmlx-bench-kernel/src/main.rs`) discarded from working tree.

**Archived to gitignored patch** for future re-attempt context (does NOT pollute git history):
- `reports/p5i-c-stage-beta-experimental-code.patch` (1959 lines; 81 KB) — full T0 scaffold + T1 first-attempt naive scalar + T1 round-4 SG-MMA + T1 round-5 MLX `gather_qmm_rhs`-aligned redesign with bench harness

If Phase 1 ever returns to gate_up (e.g., via mlx-sys extension Pivot 2 per Codex round-6 Q3 conditional), the patch + this close-out + round-1-6 audit trail provide complete re-start context.

### NOT executed (per Blocked Path B.1)

T2 sorted tile sweep / T3 EG-2a oracle / T4 production wiring / T4 EG-2b 35B regression / T5 L1+L2 acceptance — all SKIPPED per round-5 + round-6 Blocked Path binding. No long-wall measurements; no production-runtime changes.

## § 4 Codex round-6 pivot decision (Q1-Q6 bindings)

| Q | Codex binding | Disposition |
|---|---|---|
| Q1 | **Pivot 1**: close `gather_qmm_gate_up` Stage β as negative finding; start new candidate; Pivot 3 is softer framing — round-6 selects explicit close-out to avoid wall-time drag | This close-out doc + spec/plan retired status |
| Q2 | Skip R2 (`gather_qmm_down` same gather_qmm family + likely same mlx::steel gap → BLOCKED); next candidate = **`gda_step_1a_in_proj_qkvz`** (Phase 0 highest non-gather: PP=128 10.19% / PP=512 14.94%); **feasibility gate first** (Amdahl: needs PP=128 ~49% / PP=512 ~34% reduction for L2 ≥5% — aggressive; P5g had op-level saturation lesson — both warrant lightweight feasibility check before full implementation) | Phase B (new stage launch — separate from this close-out) |
| Q3 | If Pivot 2 (mlx-sys + fused-output) ever revisited: **measure op-boundary FIRST** via Stage α infra (no mlx-sys touch until empirical evidence justifies); want L2 ≥5% requires op-boundary ≥22% of gate_up; if expand/slice only 3-5% → e2e upper bound 0.7-1.2% (not worth mlx-sys); only if measured op-boundary 15-20% of gate_up should mlx-sys be reconsidered | Recorded; not pursued now |
| Q4 | Explicitly NOT pivot to `gather_qmm_down` (Phase 2 R2 candidate) — same family, same MLX infra gap likely repeats BLOCKED; round-5 Sup-1 + round-6 Q4 reinforce | Phase B candidate selection excludes gather_qmm family |
| Q5 | Do NOT commit experimental kernel code by default; archive diff to gitignored patch for future re-attempt context only | Patch archived `reports/p5i-c-stage-beta-experimental-code.patch`; working tree discarded |
| Q6 | Clean close + new stage (not rewrite old Stage β for new target); committed close-out doc; spec/plan mark Stage β blocked/retired; new candidate opens new stage/spec/plan; **first step = feasibility gate** | This close-out doc + spec § 6.8 + plan banner amendment; Phase B starts separately |
| Missed dimension | Next-round first principle: **compute reachable e2e upper bound BEFORE deciding to implement**; any candidate without clear path to PP=128/512 dual-point ≥5% e2e should NOT enter full implementation | Phase B feasibility gate MUST include Amdahl + e2e ceiling analysis as primary acceptance criterion |

## § 5 Phase 1 G1-G4 status (spec § 6)

| Gate | Status |
|---|---|
| G1 | Satisfied (P5h+2.e backfill `9a35ae17`) |
| G2 | Satisfied (Boss approved Stage β design + plan + round-1-6 amendments) |
| G3 | Satisfied (spec committed through `1648105`; this close-out adds round-6 final state) |
| G4 | Satisfied (branch `ironmlx-p5i-c-phase-1` forked off `8ff074d`) |

Phase 1 **infrastructure gates** (G1-G4) all satisfied. Phase 1 **substep candidate** = `gather_qmm_gate_up` produced negative finding (this close-out). Phase 1 **acceptance L1/L2** never measured (T5 skipped per BLOCKED). Phase 1 → Phase B (`gda_step_1a_in_proj_qkvz` feasibility gate) pivot per Codex round-6 Q1+Q2 binding.

## § 6 Memory + reference

- Negative-finding full evidence (implementer-authored, 165 lines): `reports/p5i-c-stage-beta-gate-up-negative-finding.md` (gitignored)
- Pivot decision audit (controller round-6 questions; Codex responses): `reports/p5i-c-phase-1-stage-beta-pivot-questions.md` (gitignored)
- Experimental code archive: `reports/p5i-c-stage-beta-experimental-code.patch` (gitignored)
- Stage α infra: commit `a9c2beb` (`ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` 3 child-span wrappers + `ironmlx/src/core/p5h.rs` + `tools/p5h_aggregator/{multi_repeat.py, schema_validator.py, tests/}`)
- Spec round-1-6: `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` (this close-out adds § 6.8)
- Plan round-3-5: `docs/superpowers/plans/2026-05-27-ironmlx-p5i-c-phase-1-stage-beta-gather-kernel.md` (this close-out adds banner)

## § 7 Process lessons (memory candidates)

1. **Codex bounded-iteration cap binding is engineering-honest**: round-5 Q1 + Q5 explicit cap prevented round-6 wall-time drag. Without it, controller likely would have proposed Layer 2 / Layer 3 unilateral redesign rounds chasing diminishing returns. The cap saved engineering capacity.

2. **MLX-saturated kernel finding is a Phase-1 invariant**: MLX `gather_quantized_matmul_on` uses `mlx::steel` template infra that beats independent raw-Metal kernels at small-M batch sizes. Future Phase 1 candidates should pre-screen "does this substep call into MLX kernels that are already heavily-tuned via mlx::steel?" If yes, default expectation = beating MLX is hard; pivot to op-boundary saving OR pre-screen feasibility before implementation.

3. **First-principles e2e ceiling computation as gating criterion**: Codex round-6 missed-dimension binding ("compute reachable e2e upper bound BEFORE deciding to implement") is universal — applies to all future Phase 1+ candidate selection. Should be added to Phase 1 spec template OR to a `[feedback-first-principles-feasibility-gate]` memory entry.

4. **Spec/plan iterative tightening through Codex rounds is valuable**: round-1 brainstorm → round-2 design lock → round-3 plan corrections → round-4 EG-1 ladder → round-5 Option E scope narrowing → round-6 pivot decision. Each round added concrete actionable bindings. Pattern is reusable for future complex optimization phases.

5. **Negative findings should be committed-doc shipped**: prior P5 phases occasionally let negative findings live only in gitignored reports/. Codex round-6 Q6 binding makes this committed doc explicit — future readers see why gate_up was NOT pursued without spelunking gitignored reports. Pattern worth keeping.

These lessons are candidates for memory write-back during Phase B start.
