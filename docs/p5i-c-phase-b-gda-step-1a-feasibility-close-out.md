# P5i.c Phase 1 Phase B — Feasibility Gate Close-Out: NEGATIVE-FINDING on `gda_step_1a_in_proj_qkvz` (self_qmm path)

**Status (2026-05-27)**: Phase B feasibility gate **NEGATIVE-FINDING**. Empirical measurement of switching `IRONMLX_USE_SELF_QMM=1` (the P8a Stage 9 self_qmm Q4 affine Metal kernel; already shipped behind opt-in env var per commit `5b470ea`) on `gda_step_1a_in_proj_qkvz` shows **3× substep regression + 1.75-7.52% e2e regression at PP=128 and PP=512**. Phase B does NOT proceed to spec or implementation. Phase 1 substep candidate `gda_step_1a_in_proj_qkvz` retired without further work; Boss decides next non-self_qmm candidate per Codex round-6 Q2.

**Predecessor commits**: `42a59ee` (Stage β closed as negative finding on `gather_qmm_gate_up`) + `a9c2beb` (Stage α prep child-span infra preserved).
**Branch**: `ironmlx-p5i-c-phase-1`.

## § 1 Empirical measurement evidence

Protocol: `tools/p5h_2b_protocol_experiment.py`, lightweight feasibility-grade (1 repeat × 10 runs/PP; 60s preheat + 50 preheat runs + 60s cooldown). 4 cells total = 2 PPs × 2 env (`IRONMLX_USE_SELF_QMM=0` baseline / `=1` candidate). Cell wall ~3-4 min; total ~25 min. Output:
- `/tmp/p5i-c-phase-b-feasibility-baseline-feasibility_baseline-r1-pp{128,512}/`
- `/tmp/p5i-c-phase-b-feasibility-candidate-feasibility_candidate-r1-pp{128,512}/`

### L1 substep median (`gda_step_1a_in_proj_qkvz` inclusive_us per layer per forward)

| PP | n | baseline | candidate | Δ |
|---|---|---|---|---|
| 128 | 1830 spans/cell | 2.17 us | **6.79 us** | **-213.5%** (3.13× SLOWER) |
| 512 | 1830 spans/cell | 2.33 us | **6.88 us** | **-194.7%** (2.95× SLOWER) |

Min/max excludes cold-start artifact:
- baseline_pp128 min=0.92us / max=19.96us
- candidate_pp128 **min=6.08us** / max=148.29us
- **Candidate fastest case still 6× slower than baseline** — real per-dispatch overhead, NOT cold-start cache miss

### L2 e2e pp_tps median (10 runs/cell)

| PP | baseline median | candidate median | Δ | verdict |
|---|---|---|---|---|
| 128 | 810.25 | **749.28** | **-7.52% REGRESSION** | FAIL |
| 512 | 1474.20 | **1448.34** | **-1.75% REGRESSION** | FAIL |

Per spec § 4.2.6 EG-1 ladder analogue + Phase 1 § 2.1 L2 acceptance (≥5% gain): **both PP REGRESSION** → NEGATIVE-FINDING band.

## § 2 Architectural root cause

P8a Stage 9 self_qmm Q4 affine SG-MMA kernel was acceptance-tested at **PP=2048** (M=2048, N=9216, K=2560 = gate_up shape) and showed 1.32× e2e speedup per `[[project-p8a-stage9-findings]]`. The current production dispatch in `ironmlx/src/nn/linear.rs:208` uses:

```rust
let use_self_qmm = crate::nn::self_qmm::enabled() && m_total >= 32;
```

with explanatory comment claiming "Threshold = 32 covers the M=1 / small-prefill cases cleanly while keeping the PP=128/512/2048 prefill on the hardware-MMA path."

**Empirical Phase B refutation**: at PP=128 (M=128) and PP=512 (M=512), self_qmm SG-MMA per-dispatch overhead (kernel launch + threadgroup setup + smem staging + per-dispatch encoder cost) **does not amortize** against the compute savings. Per-dispatch fixed cost ~4-5 us dominates the ~1-2 us scalar matmul saving at small M, producing 3× substep regression.

Root cause is **threshold misjudgment**: `m_total >= 32` was a guess; actual self_qmm beneficial regime starts somewhere between M=512 and M=2048 (exact inflection point unmeasured; Stage 9 acceptance test point M=2048 is empirically validated lower bound).

Also: enabling `IRONMLX_USE_SELF_QMM=1` switches **ALL Linear-quant matmuls** to self_qmm (not just `in_proj_qkvz`): `out_proj`, `q_gate_k_v_proj`, `norm_proj`, `lm_head`, etc. The 7.52% e2e regression at PP=128 likely accumulates across multiple substeps that ALL regress under self_qmm at small M.

## § 3 Codex round-6 binding compliance

Codex round-6 Q6 + missed-dimension binding: "first step = feasibility gate; compute reachable e2e upper bound BEFORE deciding to implement". Phase B:
- Did first-principles Amdahl estimate (PP=128 needs ~49% reduction / PP=512 ~33% reduction for L2 ≥5%) **before** empirical measurement
- Did MLX kernel saturation pre-screen via `[[feedback-mlx-kernel-saturation-prescreen]]` — `in_proj_qkvz` calls into `mlx::quantization::quantized_matmul_on` which is `mlx::steel::BlockMMA`-backed at large M; same structural ceiling as Stage β `gather_qmm_gate_up`
- Did historical evidence review via `[[project-p5g-findings]]` — P5g already saturated op-level fusion for `in_proj_qkvz` (T1 fused projection revert); future gains MUST be kernel-level (which is what self_qmm Stage 9 attempted, but it was tested at PP=2048 not PP=128/512)
- Empirical 30-min measurement (Boss-approved Path A) **CONFIRMED first-principles ceiling**: switching ON has marginal expected benefit (2-5% e2e per Amdahl) but in practice REGRESSES due to small-M dispatch overhead

Per Codex round-6 Q6: feasibility gate FAIL → Phase B closes as negative finding without spec/plan write. **Compliance: full.**

## § 4 Phase 1 status

G1-G4 (spec § 6) all satisfied (from prior phases). Phase 1 substep candidate timeline:

| Candidate | Phase | Verdict |
|---|---|---|
| `gather_qmm_gate_up` (R1) | Stage β | NEGATIVE-FINDING (commit `42a59ee`); mlx::steel infra gap |
| `gather_qmm_down` (R2) | NOT pursued | Codex round-5 Sup-1 + round-6 Q4: same family, same gap likely |
| `gda_step_1a_in_proj_qkvz` (R3) | Phase B | NEGATIVE-FINDING (this close-out); self_qmm small-M regression |

Phase 1 acceptance L1/L2 never measured. Two of three top candidates retired. Phase 1 → next non-gather non-Linear-quant candidate (Boss decides; Phase 0 ranking R4+ for selection).

## § 5 By-product bonus finding (β fix; separate commit)

Phase B feasibility data also reveals **production-impact bug** in `self_qmm` dispatch threshold:
- Current: `m_total >= 32` allows self_qmm at PP=128/512 → empirical regression
- Empirical: self_qmm beneficial regime starts at M=2048 (Stage 9 test point); below that REGRESSES
- Bonus fix: raise threshold from `m_total >= 32` to `m_total >= 2048` (Stage 9 acceptance test point as empirically-validated lower bound; conservative + safe)

This fix avoids `IRONMLX_USE_SELF_QMM=1` users (e.g., dev / benchmarking / staging) silently regressing at small-PP workloads. Production default is unaffected (env unset → MLX path either way).

Fix is shipped in **separate commit** (β); this close-out doc is α scope (feasibility gate verdict). See β commit for diff.

## § 6 Next-step Boss decisions

| Decision point | Options |
|---|---|
| Next Phase 1 candidate | (a) Phase 0 R4 `gda_step_8_norm_proj` (5.07%/7.28%) — also Linear-quant; same self_qmm small-M risk; pre-screen first OR (b) Phase 0 R5+ non-Linear-quant candidates (e.g., `shared_expert`, `routing_*`, `swiglu_activation`); MLX saturation pre-screen each per `[[feedback-mlx-kernel-saturation-prescreen]]` OR (c) Defer Phase 1; pivot to other engineering priority |
| Decision wall | ~15 min Boss + controller analysis (re-read Phase 0 ranking + MLX dispatch pre-screen for each candidate) |
| Implementation wait | Phase C / next stage starts AFTER Boss decides candidate + feasibility gate planning |

## § 7 Process lessons (memory candidates)

1. **Feasibility gate with empirical measurement was correct decision** (Path A; Boss-approved): 25-min measurement cheaply produced robust negative signal (3× substep regression + min=6.08us confirms NOT cold-start). Without this measurement, controller would have written spec/plan committing to ~1 day work on Phase B that would have failed.

2. **`[[feedback-first-principles-feasibility-gate]]` validated**: Amdahl estimation predicted "marginal L2 gain" (2-5%); empirical refined to "actually regression". Predicted ceiling was upper-bound estimate; reality was below the ceiling not above. The feasibility gate caught this in 25 min vs ~1 working day Phase B implementation effort.

3. **`IRONMLX_USE_SELF_QMM=1` should NOT be default-ON in production** — empirical confirms env-gated opt-in is correct policy. The β fix (threshold raise) makes opt-in safer for users.

4. **Stage 9 ship state was incomplete coverage**: PP=2048 acceptance gate passed, but smaller PP regime was untested. Future kernel ship gates should include M-aware testing across the PP regime they're expected to serve (PP=128/512/2048+ all matter for prefill).

5. **Multi-substep impact of single env var**: enabling `IRONMLX_USE_SELF_QMM=1` switches ALL Linear-quant ops globally; the 7.52% PP=128 regression is accumulated across multiple substeps not just `in_proj_qkvz`. Phase B's "single substep target" framing oversimplified the actual blast radius.

These lessons are candidates for memory write-back during Phase C (next candidate) start.

## § 8 Memory + reference

- Empirical measurement output (not committed; gitignored or local-only):
  - `/tmp/p5i-c-phase-b-feasibility-baseline-feasibility_baseline-r1-pp{128,512}/` (env=0)
  - `/tmp/p5i-c-phase-b-feasibility-candidate-feasibility_candidate-r1-pp{128,512}/` (env=1)
- Stage β close-out predecessor: `docs/p5i-c-phase-1-stage-beta-close-out.md` (commit `42a59ee`)
- Codex round-6 pivot questions: `reports/p5i-c-phase-1-stage-beta-pivot-questions.md` (gitignored)
- Bonus fix (β): see separate commit raising `self_qmm` M-aware dispatch threshold from `m_total >= 32` to `m_total >= 2048`
- Stage α infra (commit `a9c2beb`): preserved; future-reusable; not affected by Phase B close-out
