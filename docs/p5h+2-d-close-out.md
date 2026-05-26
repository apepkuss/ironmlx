# P5h+2.d — Thermal / Residual-Variance Investigation: Close-out (Mechanism-only)

**Status:** **Mechanism-only** per spec § 1.3 outcome matrix (Mechanism gate `strong_yes` + Acceptance gate PP=512 PASS / PP=128 FAIL). Phase 0 § 7 #4 production envelope **NOT** backfilled; criterion #4 STILL FAIL/DEFERRED. PP=128-specific follow-up required (P5h+2.e candidate; Codex round-1 binding).
**Date:** 2026-05-25.
**Branch:** `ironmlx-p5h+2-d-thermal-investigation` HEAD `<T5 commit SHA>`.
**Predecessor close-outs:** `docs/p5h+2-b-close-out.md` § 9-10, `docs/p5h+2-c-close-out.md`, `docs/p5i-c-phase-0-close-out.md` § 1 #4.

---

## § 1 Stage 1 — ironmlx cooldown matrix (T2)

### § 1.1 Sweep health

- 18/18 cells completed cleanly. PID 36645 exited 0. Sweep wall: ~6.3 hr (started 2026-05-25 14:24, finished ~20:43).
- **scheduler ERROR count across all 18 cells = 0** (P5h+2.c fix CONFIRMED solid at production scale on MoE Qwen3.5-35B-A3B-4bit — the real test, not just T2 smoke).
- All per-cell `server_log_scan.json` (Rule D) written; non-allow-listed WARN inspected, none introduced ERROR-level issues.

### § 1.2 Mechanism gate verdict (spec § 2.4)

```json
{
  "verdict": "strong_yes",
  "best_cooldown_per_pp": {"128": "120s", "512": "120s"},
  "reason": "both PPs >=50% reduction AND BEST residual <=10%",
  "details": {
    "128": {"baseline_residual_pct": 10.48, "best_residual_pct": 3.73, "reduction_pct": 64.43, "passes_50pct_and_residual_le_10pct": true},
    "512": {"baseline_residual_pct": 23.80, "best_residual_pct": 5.24, "reduction_pct": 77.97, "passes_50pct_and_residual_le_10pct": true}
  }
}
```

Trailing_slowdown / fast_start_drop residual metric reduces ≥50% for both PPs; BEST residual ≤10% for both. BEST cooldown = 120s for both PPs.

### § 1.3 Acceptance gate data (pp_tps envelope ≤ ±2%) — divergence from Mechanism gate

| PP | cd=0s envelope | cd=60s envelope | cd=120s envelope | within_max @ cd=120s | between_half @ cd=120s |
|---|---|---|---|---|---|
| **128** | 4.91% FAIL | 4.82% FAIL | **4.71% FAIL** | 4.71% | 1.05% |
| **512** | 10.71% FAIL | 1.25% PASS | **0.91% PASS** | 0.91% | 0.12% |

PP=512 cd=120s dual PASS — cooldown fully resolves both within-sweep and cross-spawn variance. PP=128 cd=120s: cross-spawn becomes consistent (4.91% → 1.05%) BUT within-sweep CI is **stuck at 4.71%** regardless of cooldown level.

### § 1.4 Raw within-sweep evidence (PP=128 cd=120s; smoking gun)

| Repeat | First 3 pp_tps | Last 3 pp_tps | All-15 (rounded) |
|---|---|---|---|
| r1 | 985.7, 829.0, 829.4 | 828.7, 861.5, 874.9 | 985.7 / 829-843 plateau (±50 jitter) |
| r2 | 983.1, 821.9, 844.0 | 812.6, 791.3, 830.2 | 983.1 / 791-851 plateau |
| r3 | 988.9, 820.2, 840.8 | 749.9, 805.2, 755.2 | 988.9 / plateau + 3 downward outliers (746/744/750 cluster runs 10-12) |

Pattern: **first run ≈ 985 (cold-start spike), runs 1-14 settle to noisy plateau ~820 ±50 (~±6% jitter)**. This is a **fast-start-drop pattern that cooldown did NOT eliminate**. PP=512 cd=120s contrasts cleanly (±1% plateau jitter after the spike).

## § 2 Stage 2 — sudo powermetrics overlay (T3): SKIPPED per Codex round-1

Per Codex round-1 review (`reports/p5h+2-d-stage1-codex-review.md`): Mechanism gate `strong_yes` was the primary unblocking signal Stage 2 was designed to provide; once achieved, powermetrics evidence becomes post-hoc H1.a/b/c sub-hypothesis classification. The new puzzle (PP=128 within-CI 4.7% residual) is NOT thermal at the sweep-boundary scale (cooldown didn't fix it); Stage 2 cannot diagnose it. ROI drop → skipped.

Honest narrative bound per Codex round-1 Q7: result is **consistent with H1 thermal/fan family, not causally proven without Stage 2**. H1.a (thermal-soak) / H1.b (fan-hysteresis) / H1.c (preheat-topology-mismatch) sub-hypothesis remains uncategorized.

## § 3 T4 deviation — δ omlx control PP=128-only (Boss + Codex round-1 approved)

Spec § 6 originally specified 12 omlx cells `{BEST, WORST=0s} × {PP=128, PP=512} × 3 repeats`. This phase ran a **diagnostic deviation** with Boss + Codex round-1 approval:

- **6 cells** `{BEST=120s, WORST=0s} × PP=128 ONLY × 3 repeats`
- PP=512 omlx control skipped — PP=512 ironmlx already at envelope PASS at cd=120s; cross-comparator adds no actionable information
- T3 sudo powermetrics skipped — see § 2

**This is explicitly NOT the spec-defined § 6 T4** (which would include PP=512 + would feed into Acceptance gate § 8.2 A3). It is a focused diagnostic to triage PP=128 within-CI 4.7% residual as system-level (omlx also shows it) vs ironmlx-specific (only ironmlx shows it).

### § 3.1 δ results

Sweep wall: ~1:50 (started 2026-05-25 21:49, finished 23:39). 6/6 cells complete.

| Tag | Cooldown | omlx PP=128 medians (3 repeats) | within-CI max | between-half | envelope | iron-bench verdict |
|---|---|---|---|---|---|---|
| **WORST** | 0s | 1039 / 1053 / 1063 | **0.82%** | 1.11% | **1.11%** | **PASS** |
| **BEST** | 120s | 505 / 502 / 481 | 2.64% | 2.46% | **2.64%** | FAIL |

**Raw within-sweep evidence:**
- omlx WORST cd=0s r1: `1036, 1031, 1039, 1031, 1051, 1057, 1045, 1047, 1046, 1037, 1044, 1037, 1044, 1026, 1011` — tight ~1040 plateau, no spike.
- omlx BEST cd=120s r1: `1070, 526, 518, 505, 510, 495, 535, 502, 499, 504, 465, 549, 512, 498, 503` — first run = 1070 spike, then DROPS to ~500 plateau (HALF the speed of WORST median).

Diagnostic field `fast_start_drop_pct` per repeat in BEST cells: 112.6%, 108.9%, 121.7% — meaning the spike is more than 2× the plateau median.

### § 3.2 δ interpretation: **ironmlx-specific** (Codex § 6.3 classification)

**Comparing within-CI residuals (the actual question we asked):**

| Impl | cd=0s within-CI | cd=120s within-CI |
|---|---|---|
| ironmlx PP=128 | 4.21% (cd=0s envelope 4.91% — same number; envelope = max(within, between)) | 4.71% |
| omlx PP=128 | **0.82%** | 2.64% |

**Two distinct findings:**

1. **ironmlx PP=128 has an intrinsic within-sweep noise that omlx does NOT have.** Both impls run on the same hardware (M5 Max), same model weights (Qwen3.5-35B-A3B-4bit), same prompt-synthesis protocol (iron-bench with run-varying nonce), same per-cell preheat (1100 runs PP=512), and same measured loop (15 runs PP=128). The ONLY difference is the server implementation. ironmlx PP=128 cd=0s shows within-CI 4.21% while omlx PP=128 cd=0s shows within-CI 0.82% — a **5× gap**. This is **NOT system-level thermal/protocol** noise (omlx would show it too).

2. **omlx BEST cd=120s drops to half speed (1050 → 500)** with 110% fast-start spike — this is an **omlx-specific cooldown sensitivity** (likely KV cache / engine state eviction during 120s sleep windows). Production LLM serving rarely encounters 120s gaps between requests; this is a benchmark protocol artifact for omlx, not a server bug. Relevant for benchmark design choices, NOT for the ironmlx PP=128 residual question.

**Verdict**: PP=128 within-CI 4-5% residual is **ironmlx-specific** (Codex § 6.3 classification 2). P5h+2.e investigation should target the ironmlx PP=128 code path, NOT benchmark protocol or hardware. Codex round-1 hypothesis priority (H1.c preheat topology > H_small_batch + routing variance > H2 MLX state-decay) is the correct direction — confirmed by δ.

**Out-of-scope for this close-out:** the omlx cd=120s half-speed observation is an interesting finding about omlx (KV cache state eviction under long inter-run gaps) but is NOT actionable for P5h+2.e / Phase 0 / Phase 1; logged here for future reference only.

## § 4 Phase 0 § 7 #4 backfill (Codex round-1 Q5 binding exact wording)

`docs/p5i-c-phase-0-close-out.md` § 1 #4 row appended with:

> **2026-05-25 P5h+2.d update**: Mechanism gate PASS; Acceptance PP=512 PASS / PP=128 FAIL; Phase 0 production envelope **NOT** backfilled; PP=128-specific follow-up required (P5h+2.e candidate).

Phase 0 criterion #4 status remains **STILL FAIL/DEFERRED**.

## § 5 Phase 1 implementation gating (Codex round-1 Q6 binding — STRICT language)

**Phase 1 implementation REMAINS BLOCKED.**

- `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 6 G1-G4 conditions NOT satisfied (G1: P5h+2.d close-out does NOT EXPLICITLY allow implementation — this close-out IS the source doc, and it explicitly states implementation remains blocked).
- Phase 1 brainstorm / design / spec work continues (γ-lite output already committed at `241d476`).
- Phase 1 implementation, performance verification, and any claim of `ironmlx >= 1.10 * omlx` are **all BLOCKED**.
- **PP=512 PASS does NOT substitute for PP=128 FAIL**. The Acceptance gate requires both PPs; passing one is not the same as passing the gate.

## § 6 P5h+2.e direction (next-phase candidate; NOT pre-declared per spec § 8.3)

PP=128-specific small-batch / preheat-topology investigation. Per Codex round-1 Q3 priority:

| Rank | Hypothesis | Test |
|---|---|---|
| 1 | **H1.c preheat topology mismatch** — preheat hardcoded `--prompt-len 512` (matches Stage 1 monolithic preheat); no PP=128 same-shape warmup. Measured loop runs PP=128 against a steady-state warmed for PP=512 | Add PP=128 same-shape preheat (e.g., 200-run PP=128 after the 1100-run PP=512 preheat) |
| 2 | **H_small_batch + prompt/routing variance** — `iron-bench` synthesizes per-run prompts with nonce-based variation; at PP=128 with `max-tokens=1`, each run's MoE expert dispatch + tile-path varies | Pin prompt nonce across runs; measure routing/expert occupancy stats; check post-trim plateau jitter |
| 3 | **H2 MLX state-decay** — allocator / JIT cache state degrades during long sweep; PP=128 path possibly more sensitive | Fresh-spawn-per-run control experiment |

Future observation items per Codex round-1 Q8 (binding for any P5h+2.e):
- PP=128 same-shape preheat protocol
- Fresh-spawn-per-run control
- Fixed / reproducible prompt nonce
- Routing / expert occupancy statistics
- Post-Rule-B-trim plateau jitter metric

**Do NOT add new post-hoc exclusion rules to make PP=128 pass the gate** (Codex round-1 Q8 binding).

## § 7 Reusable infrastructure shipped (regardless of outcome per spec § 9.4)

| Code | Path | Tests |
|---|---|---|
| `iron-bench --inter-run-cooldown-secs` production CLI flag (sequential v1 mode only; rejects v2; preserves byte-identity when default 0) | `iron-bench/src/main.rs` + `iron-bench/src/runner.rs` | `iron-bench/tests/inter_run_cooldown_secs.rs` (2 tests; concurrent-rejection + timing assertion) |
| Capture harness env var pass-through | `ironmlx/tests/p5i_c_phase_0_capture.rs` `P5I_C_INTER_RUN_COOLDOWN_SECS` | preheat byte-identity verified |
| `tools/p5h_2b_protocol_experiment.py` — Rule D `scan_server_log` (replaces narrower `check_no_scheduler_errors`) with level-field anchored matcher + ALLOWLISTED_WARN_SUBSTRINGS, per-cell `server_log_scan.json`, malformed-input KeyError guards | `tools/p5h_2b_protocol_experiment.py` | `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py` (5 pytests) |
| `tools/p5i_c_pp_tps_envelope.py` — per-repeat diagnostic fields `trailing_slowdown_pct`, `fast_start_drop_pct`, `first/last_3_runs_median_pp_tps`; flexible `--expected-runs` CLI override; pure (NO gate logic) | (modify) | `tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py` (4 new tests; 9 total) |
| `tools/p5h_2d_thermal_experiment.py` — Stage 1 sweep orchestrator + Mechanism gate analyzer (strong_yes / weak_yes / no; baseline-already-clean edge case; `_pick_best_cooldown` excludes 0s baseline; tie-breaker prefers shorter cooldown) | (new) | `tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py` (5 tests) |

T3 sidecar code + plist parser extension to `tools/p5h_2b_thermal_overlay.py`: **NOT shipped** (T3 skipped per Codex round-1; see § 2).

## § 8 Wall summary

| Bucket | Cap (spec § 9.3) | Actual |
|---|---|---|
| GPU wall (Stage 1 sweep + δ T4 omlx control) | 12 hr | Stage 1 6.3 hr + δ 1.83 hr = **8.13 hr** |
| Docs / analysis wall (spec / plan / Codex iteration / brainstorm / close-out) | 4 hr | ~5 hr (Codex round-1 brainstorm + Stage 1 nuanced verdict review + close-out + Phase 1 γ-lite parallel) |
| **Total** | **16 hr** | **~13.1 hr** (within cap) |

## § 9 References

- Spec: `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-d-thermal-investigation-design.md`
- Plan: `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-d-thermal-investigation.md`
- Phase 0 close-out (backfilled): `docs/p5i-c-phase-0-close-out.md` § 1 #4
- Predecessor: `docs/p5h+2-b-close-out.md` § 9-10 (re-attempt FAIL/DEFERRED + P5h+2.d phase design constraints), `docs/p5h+2-c-close-out.md` (scheduler ERROR fix)
- Phase 1 γ-lite spec (parallel work): `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md`
- Stage 1 raw data: `/tmp/p5h+2-d-stage1-r{1,2,3}-pp{128,512}-cd{0s,60s,120s}/{bench.csv,server.log,meta.json,server_log_scan.json}` + `/tmp/p5h+2-d-stage1-cd{0s,60s,120s}-pp{128,512}-envelope.json` + `/tmp/p5h+2-d-stage1-mechanism-gate.json`
- δ raw data: `/tmp/p5h+2-d-omlx-{best,worst}-r{1,2,3}-pp128/{bench.csv,server.log,preheat.csv}` (this T5 commit)
- Codex review chain (all gitignored per `[feedback-no-reports-commit]`):
  - `reports/p5h+2-d-brainstorm-codex-questions.md` (Codex round-5 P5h+2.d brainstorm)
  - `reports/p5h+2-d-stage1-codex-review.md` (Codex round-1 nuanced-verdict review; binding for δ decision + § 4 / § 5 / § 6 / § 7 language)
  - `reports/p5h+2-b-rerun-codex-review.md` (Codex round-4 Option α historical context)
- Memory: `[project-p5h-2d-findings]` (new entry written by this close-out), `[project-p5h-2c-findings]`, `[project-p5h-2b-findings]`, `[project-p5i-c-phase-0-findings]`, `[project-p5h-t3-findings]`
