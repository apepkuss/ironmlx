# P5i.a — gather_qmm Feasibility + Short-PP Baseline Design (2026-05-23)

**Status:** Spec — ready for plan writing.
**Branch:** new branch `ironmlx-p5i-a-gather-qmm-feasibility` (fork from `ironmlx-p5h-perf` HEAD `449bb9d`).
**Predecessor:** P5h+1 closed 2026-05-23 with Close Gate 4/4 PASS; spec § 7.2 = 9/9 LOCKED PASS; P5i/P5j dispatch unblocked per spec § 8.

**Source docs:**
- P5h+1 close-out: `docs/p5h+1-close-out.md`
- P5h+1 ranking snapshot (concise): `docs/p5h+1-ranking-snapshot.md`
- P5h+1 ranking snapshot (full, gitignored): `reports/p5h+1-ranking-snapshot.md`
- P5h spec gates: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` (§ 1.2 gap table, § 7.2 ship state, § 8 sequential decision)
- Codex P5i scope review: `reports/p5i-codex-review-questions.md` (gitignored)
- Boss memory: `[feedback_design_philosophy]`, `[feedback_no_spec_from_competitors]`, `[feedback_performance_stability_priority]`, `[feedback_iron_bench_priority]`, `[feedback_omlx_cli_default]`, `[feedback_task_breakdown_bounded]`, `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[project_p5h_findings]`, `[project_p5g_findings]`, `[project_p8a_stage9_findings]`, `[feedback_device_aware_tile]`, `[project_cross_device_tuning_deferred]`

---

## § 1 Goal + phase nature

P5i.a is an **exploration/convergence phase**, not a Metal kernel rewrite. It exists to:
1. Establish ironmlx vs omlx external pp_tps gap at PP=128/512 via controlled iron-bench (the P5h+1 ranking used ironmlx flag-OFF root as denominator — NOT a direct ironmlx vs omlx comparison).
2. Land repeatable low-risk wrapper/dispatch/fusion optimizations that can move PP=128 toward the +24% target; if PP=128 still misses after all in-scope candidates, quantify the residual gap and the required next phase explicitly.
3. Quantify PP=512 -13% gap remaining after low-risk work, and produce a **立项 / 否决** verdict for P5i.b (multi-sprint self-quant gather Metal kernel rewrite).
4. Optionally land Level (d) op-level work on `gda_step_1a_in_proj_qkvz` as backup gain source.

**Out of scope**: writing a Metal kernel for self-quant gather (that work belongs to a future P5i.b spec if T3 in this plan recommends 立项). No production behavior changes other than narrow optimization edits + measurement-only probe flag inherited from P5h+1 (`--p5h-measurement-eval-probes`, default OFF preserves production lazy graph).

---

## § 2 Background — P5h+1 ranking input

Post-P5h+1 sweep (commit `aa20283`) verdicts at P5i target PP set {128, 512}:

| PP | verdict | op_only | with_kernel | target gain |
|---|---|---|---|---|
| 128 | yes_with_scope_gate | 22.97% | 60.91% | +24% (reachable per current candidates with kernel rewrites) |
| 512 | no_under_measured_cap | 22.22% | 61.10% | +74% (gap -13% short even with kernel) |

Top 5 candidates at PP=128 (post-P5h+1 probe-mode shares):

| Rank | Span | max_gain | Kind |
|---|---|---|---|
| 1 | gather_qmm_gate_up | 25.02% | 🔧 KERNEL |
| 2 | gather_qmm_down | 12.08% | 🔧 KERNEL |
| 3 | gda_step_1a_in_proj_qkvz | 9.82% | op-level |
| 4 | gda_step_8_norm_proj | 4.99% | 🔧 KERNEL |
| 5 | shared_expert | 4.71% | op-level |

Combined `gather_qmm_{gate_up + down}` = 37% at PP=128, 35% at PP=512. **Dominant kernel family**; both call `mlx::quantization::gather_qmm` (upstream MLX `gather_qmm_rhs` Metal kernel; `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs:357 + 481`).

**Key insight from Codex P5i review**: ironmlx, mlx-lm, and omlx all call the same upstream MLX `gather_qmm` implementation before any Scope-gated Level (a) rewrite. ironmlx's 8-substep MoE unpacking is for measurement only. Before Level (a), ironmlx vs omlx pp_tps differences come from call shape, routing/sorting choices, launch count, scheduler/dispatcher overhead, cache update, and ironmlx-specific input shaping like `expand_dims` / `flat_x` reshape / rhs_indices rank promotion at `sparse_moe.rs:217-331`. A later Level (a) kernel rewrite may change the kernel implementation itself, but that work is outside P5i.a.

P5h+1 measurement gave us substep-level shares; P5i.a must convert those into actual ironmlx wall-time improvements vs omlx baseline. Both numbers matter — substep share tells where the time goes; omlx delta tells whether we beat the competition.

---

## § 3 Close Gate (success condition)

P5i is a multi-phase program. P5i.a is the first phase:

### 3.1 Program target (P5i overall)

**T-Target-B**: PP=128 AND PP=512 ≥ omlx+10% on Qwen3.5-35B-A3B-4bit at default settings (per spec § 1.2 gain table: PP=128 needs +24%, PP=512 needs +74% above measured omlx baseline to satisfy "ironmlx > omlx +10%").

### 3.2 P5i.a first-phase gate

P5i.a is a **feasibility gate**, not the final P5i program gate. It closes IFF all four deliverables are complete:

1. **External baseline established** — T0 produces iron-bench raw CSVs `/tmp/p5i-a-baseline-ironmlx.csv` + `/tmp/p5i-a-baseline-omlx.csv` and aggregation `/tmp/p5i-a-baseline-summary.json` with ironmlx flag-OFF vs omlx CLI default at PP=128 + PP=512, same model + same prompt policy + same warmup/cooldown + same metric set (pp_tps median across RUNS=7). Documented in `docs/p5i-a-baseline.md` (committed).

2. **PP=128 outcome resolved** — post P5i.a optimizations (Level b+c+d combined), ironmlx pp_tps at PP=128 is rerun with the T0 protocol and classified as:
   - **Full PASS**: ironmlx pp_tps at PP=128 ≥ omlx+10%; or
   - **Feasibility PASS**: PP=128 still misses, but the residual gap is quantified, every in-scope Level (b)/(c)/(d) candidate with plausible ≥1% pp_tps gain has either landed or has a documented negative result, and the remaining path is explicitly tied to P5i.b or another follow-up.

3. **PP=512 gap quantified** — post P5i.a optimizations, ironmlx pp_tps at PP=512 measured against same omlx baseline. The actual gap (positive or negative vs +10% target) is documented + cited as input for P5i.b 立项 decision.

4. **P5i.b 立项 / 否决 verdict** — T3 produces written design + ROI estimate for self-quant gather Metal kernel. T5 close-out cites the verdict: 立项 (recommend P5i.b spec) OR 否决 (PP=512 gap-closing strategy goes elsewhere, e.g., P5i.c new candidate discovery or partial-target acceptance).

Close-out status uses this vocabulary:
- **Full PASS**: all four deliverables complete and PP=128 reaches omlx+10%.
- **Feasibility PASS**: all four deliverables complete, PP=128 and/or PP=512 still miss the program target, and the residual gaps + next Scope gate decision are explicit.
- **Blocked**: baseline is invalid/missing, a required verdict is missing, or an in-scope candidate with plausible ≥1% pp_tps gain remains untested.

P5i.b does NOT auto-dispatch from any P5i.a status; it requires T3 立项 plus explicit Boss Scope gate approval.

---

## § 4 Tasks (6, per `[feedback_task_breakdown_bounded]` ≤7)

### § 4.1 T0 — Controlled iron-bench baseline (omlx vs ironmlx)

Per Codex Q4 baseline-first; per `[feedback_omlx_cli_default]` omlx is the default baseline reference; per `[feedback_iron_bench_priority]` use iron-bench harness.

**Approach**:
- Spawn ironmlx serve once for the ironmlx sweep (flag-OFF; production path; `cargo run --release -p ironmlx -- serve`) → iron-bench `--prompt-len 128/512 --runs 7 --warmup 1`
- Stop ironmlx, then spawn omlx CLI once for the omlx sweep (`/Users/xin/workspace/iron-rivals/omlx` source-CLI per memory; NOT pip `mlx_lm` package per `[feedback_omlx_cli_default]`) → same iron-bench protocol
- Same model dir `~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<sha>`
- Same prompt template, same RUNS=7 measured requests, same warmup=1 unmeasured request per PP, same cooldown between PPs
- Serial execution (per `[feedback_serial_perf_experiments]` — don't run ironmlx + omlx servers concurrently to avoid GPU memory contention)
- 5-min thermal preheat at each backend sweep entry after model load and before the first warmup request (per P5h T0b H1 thermal binding)
- Fixed PP order for both backends: 128, then 512. If a run aborts, rerun the whole backend sweep rather than mixing partial old/new data.
- Save raw output to `/tmp/p5i-a-baseline-ironmlx.csv` + `/tmp/p5i-a-baseline-omlx.csv`; save aggregation to `/tmp/p5i-a-baseline-summary.json`
- Per-PP analysis: ironmlx pp_tps median, omlx pp_tps median, delta_pct, +10%-target threshold
- Committed summary: `docs/p5i-a-baseline.md` (per-PP table; baseline state at HEAD `449bb9d`)

**Acceptance**: T0 closes when both CSV files + summary JSON are written, `docs/p5i-a-baseline.md` is committed, and per-PP delta_pct is reported. Approximate wall: 2 backend sweeps × (model load + 5min preheat + 2 PPs × (1 warmup + 7 measured runs) + cooldown).

### § 4.2 T1 — sparse_moe.rs shape/dispatch 检视 + Level (b) wrapper optimization

Per Codex Q2 Level (b) — wrapper/layout/dispatch improvements without writing new Metal kernel.

**Approach**:
- Read `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs:217-331` (MoE substep input shaping). Identify candidates:
  - Unnecessary `expand_dims` calls (line 221 area; gather_qmm input shaping)
  - Redundant `flat_x` reshape (line 276 area)
  - Rhs_indices rank promotion (line 309-331 area; rank-4 promotion for MLX `gather_qmm_rhs` fast path)
- For each candidate: hypothesize "can be removed/simplified without losing gather_qmm_rhs fast-path qualification". Verify by reading MLX C++ `mlx/backend/metal/quantized.cpp:1484` to understand fast-path entry conditions.
- Implement low-risk simplification per candidate (one src commit per simplification with parity bench).
- Per change: smoke test `p5_qwen35_moe_smoke` pp_tps within ±2% (no regression on argmax sentinel correctness).
- Per change: iron-bench sweep at PP=128/512 (same T0 protocol; reuse T0 omlx baseline for delta).

**Acceptance**: T1 closes when every listed candidate has either:
- landed with argmax parity, smoke pp_tps no regression beyond noise, and repeatable ≥1% pp_tps improvement at PP=128 or PP=512; or
- been rejected with a written negative finding explaining the fast-path/correctness/perf reason.

No simplification is allowed to land only to satisfy task closure. If none yields ≥1% repeatable improvement, T1 still closes as a negative-result task and contributes evidence to the P5i.a Feasibility PASS gate.

Expected wall: 1-2 days reading + experimentation. Likely 1-3 commits per simplification.

### § 4.3 T2 — gate+up fusion 可行性验证 (Level c)

Per Codex Q2 Level (c) — fuse `gate_proj` + `up_proj` gather_qmm calls into 1 call (currently 2 calls at `sparse_moe.rs:357 area`).

**Approach**:
- Check whether `mlx::quantization::gather_qmm` API supports combined gate_up weight (single matmul, slice post-projection).
- Check weight storage: per mlx-lm `qwen3_5_moe.py:43-49` sanitize step splits `gate_up_proj` into separate `gate_proj.weight` + `up_proj.weight` — implying upstream weights ARE combined but stored split. ironmlx weight loader follows similar convention (verify).
- PoC: stack gate_proj.weight + up_proj.weight along intermediate dim at load time; stack matching scales and affine biases the same way; preserve `bits`, `group_size`, `mode`, expert axis, and packed-weight layout exactly; issue one `gather_qmm` call; slice output for SwiGLU input.
- Correctness gate before perf acceptance:
  - sorted branch and default branch both compare fused output against the existing two-call path;
  - `gate_out` and `up_out` shapes match the current contracts (`[BS*k, 1, 1, I]` sorted, `[BS, k, 1, I]` default);
  - max_abs / max_rel error stays within the existing MoE smoke sentinel tolerance for 4-bit affine paths;
  - if either `gate_biases` or `up_biases` is absent, the fused path must prove equivalent handling or be rejected.
- Bench: launch overhead saving expected (2× kernel launches → 1×). At PP=128 (small batch) launch overhead is larger fraction; at PP=512 less so.
- If PoC passes correctness and yields repeatable ≥1% pp_tps improvement at PP=128 or PP=512 with no regression beyond noise on the other PP → land. If not → document negative + skip.

**Acceptance**: T2 closes with PoC bench result + 立项/否决 decision for fusion. If 立项: 1 src commit (weight loader + sparse_moe.rs fused call). If 否决: documented in T5 close-out.

Expected wall: 1-3 days PoC + bench. 0-1 commits.

### § 4.4 T3 — Self-quant gather Metal kernel 立项设计 (Level a feasibility ONLY)

Per Codex Q2 Level (a) feasibility-only — bench plan + minimal prototype design + ROI estimate. **NO Metal kernel implementation in this phase** (that's P5i.b if approved).

**Approach**:
- Read existing `ironmlx/src/nn/self_qmm/` infrastructure (415 LoC + `qmm_t.metal.in` template). Document the dense Linear self-quant pattern and summarize the existing +35% over MLX baseline evidence from `[project_p8a_stage9_findings]`; reproduce that microbench only if the prior evidence is stale, missing, or ambiguous.
- Design analogous `gather_qmm.metal.in` template structure for MoE gather pattern:
  - expert_indices indirection
  - per-expert weight gather (or per-token-expert combined gather)
  - scatter result back to output buffer
  - per-tile thread group layout (mirror self_qmm tile design adapted for gather)
- Estimate ROI:
  - if gain analogous to self_qmm +35% → gather_qmm wall reduction ~35% × 35% root share ≈ 12% root_inclusive at PP=128/512
  - if gain only 10-15% due to gather indirection overhead → ~3-5% root_inclusive
  - explicit confidence interval (per `[feedback_design_rigor]`)
- Cost estimate: 2-4 weeks Metal kernel impl + correctness validation + sweep + integration. Per `[feedback_task_breakdown_bounded]` would need P5i.b decomposition into sub-phases.
- Produce design memo `docs/p5i-a-gather-kernel-feasibility.md` (committed):
  - Self-quant gather kernel pseudocode + dispatch contract
  - Bench plan (microbench harness; correctness oracle vs MLX upstream)
  - ROI estimate with confidence range
  - 立项 / 否决 recommendation with rationale
  - If 立项: P5i.b spec outline + estimated task count + Scope gate hook (Boss approval required per spec § 5)

**Acceptance**: T3 closes with committed design memo + explicit recommendation. NO Metal source code written in P5i.a.

Expected wall: 2-3 days design + reading + memo writing. 1 commit (design memo).

### § 4.5 T4 — Level (d) backup: gda_step_1a_in_proj_qkvz op-level tuning

Per Codex Q2 Level (d) backup — low-risk, no Scope gate. Optional but explicitly considered after T1/T2 because `gda_step_1a_in_proj_qkvz` is the top op-level candidate from P5h+1.

**Approach**:
- Read `ironmlx/src/nn/gated_delta_net.rs` step_1a (in_proj_qkvz dense Linear; already uses self_qmm path).
- Hypotheses:
  - Tile param tuning for M5 Max (per `[project_cross_device_tuning_deferred]` self_qmm originally tuned on M1 Pro; M5 Max may have different optimal tile)
  - Fusion with adjacent steps (1a + 1b combined; or 1a output + step 3 reshape fused)
- Trigger: execute T4 if PP=128 is still below omlx+10% after T1/T2, OR PP=512 still misses the +10% target by >5%, OR T3 recommends 否决/延期 for Level (a). Skip only if T1/T2 already make both target PPs pass and T3 still produces its verdict.
- PoC: bench tile sweep on M5 Max; identify best params; integrate via lookup table extension (per `[feedback_device_aware_tile]` ironmlx Metal kernel should be device-aware).
- Bench: iron-bench pp_tps delta vs T0 baseline.
- Land if ≥1% improvement at PP=128 OR PP=512; otherwise document + skip.

**Acceptance**: T4 closes with tile sweep data + landed change (if applicable) + bench result. M5 Max only; M1/M2/M3/M4 deferred per `[project_cross_device_tuning_deferred]`.

Expected wall: 1-2 days. 0-1 commits.

### § 4.6 T5 — Close-out

Per `[feedback_no_empty_commits]` — close-out doc commit, not empty.

**Approach**: write `docs/p5i-a-close-out.md` (committed) with:
- Status (Full PASS / Feasibility PASS / Blocked) + date + branch + commit chain.
- § 1 Close Gate 4-condition result (cite measurements).
- § 2 Per-PP final state: post-P5i.a ironmlx pp_tps vs omlx (delta_pct). PP=128 +X% verdict; PP=512 -Y% gap remaining.
- § 3 What landed (T1 simplifications; T2 fusion if 立项; T4 tile tuning if 立项).
- § 4 Self-quant gather kernel verdict (T3 outcome — 立项 → recommend P5i.b spec; OR 否决 → document why + alternative for PP=512).
- § 5 P5i+ follow-up:
  - If T3 立项: P5i.b spec writing as next phase
  - If T3 否决: P5i.c new candidate discovery (e.g., scheduler overhead investigation, KV cache layout, etc.) — out of P5i.a scope
  - P5h+2 items carried (validate_chunk_ancestry cycle, etc.) — unchanged
- § 6 Memory update — extend `project_p5h_findings.md` with P5i.a closure section + create `project_p5i_a_findings.md` if scope warrants.
- § 7 References.

**Acceptance**: T5 closes with committed docs + memory updated. Per spec § 7.2 update IF P5i.a delivers PP=128 +10% achievement (otherwise spec § 7.2 stays unchanged from P5h+1 post state).

Expected wall: 1 day write-up. 1 commit.

---

## § 5 Scope gate — Level (a) Metal kernel Boss approval (per Codex Q7)

Per Codex P5i review Q7: any Level (a) self-quant gather Metal kernel rewrite work (writing new `.metal.in` template + `dispatch_gather_qmm_t` function + integrating into `ironmlx/src/nn/self_qmm/`) requires **explicit Boss approval BEFORE work begins**.

Within P5i.a: Level (a) work is restricted to T3 design memo + ROI estimate. Implementing the Metal kernel is **out of scope** for P5i.a. P5i.b (if 立项) is a separate spec that must request the Scope gate trigger explicitly.

---

## § 6 Out of scope (deferred)

Per Codex Q1 + Q6:

- **Self-quant gather Metal kernel implementation** → P5i.b (if T3 立项)
- **Long-PP optimization (PP=2048+)** → P5j (spec § 1.2 P5J_TARGET_PP_SET = {2048, 4096, 8192, 16384})
- **Cross-device tile tuning** (M1/M2/M3/M4) — preserve device-aware lookup structure but only validate M5 Max in P5i.a (per `[project_cross_device_tuning_deferred]` + `[reference_current_machine]`)
- **P5h+2 follow-ups** carried unchanged: validate_chunk_ancestry cycle vulnerability + P5hChunkContextGuard.active dead field + roi_ranking.py::LANE_A_WRAPPER_SPAN stale literal + GA kv_mask_update duplicate-eval + emit cost reduction + T0b H4 same-mode control + T4.2 mid-admit ctx plumbing + spec § 1.2 PP=2048 partition (addressed in P5h+1 § 7.2.1.5)
- **PP=128/512 algorithmic exploration** (expert grouping, k_top consolidation, etc.) — outside Level (a)/(b)/(c)/(d) Codex inventory; defer until basic Levels show ceiling

---

## § 7 Validation gates (P5i.a)

Inherits P5h spec § 4 gates + adds:

- **§ 7.1 Production parity** — every T1/T2/T4 src change must pass `p5_qwen35_moe_smoke` argmax sentinel + pp_tps within ±2% on feature-off build (no regression on production lazy-graph behavior).
- **§ 7.2 Cumulative pp_tps progression** — after each landed change, cumulative ironmlx pp_tps at PP=128/512 vs T0 baseline must show no statistically meaningful regression beyond the ±2% smoke/bench noise band. Literal monotonic increase is not required; neutral changes are acceptable only when they are necessary for a larger accepted optimization and do not regress either target PP beyond noise.
- **§ 7.3 omlx baseline integrity** — T0 baseline data is the canonical denominator. If model/snapshot version changes, T0 must be re-run; do NOT compare against stale baseline.
- **§ 7.4 Serial execution per `[feedback_serial_perf_experiments]`** — ironmlx + omlx servers never run concurrently during bench; one process at a time.
- **§ 7.5 5-min preheat per P5h T0b H1 binding** — every bench sweep entry includes 5-min thermal preheat.
- **§ 7.6 iron-bench `--prompt-len` exact** — T0/T1/T2/T4 benches all use `--prompt-len 128` and `--prompt-len 512` (NOT chat-template-adjusted; per P5h+1 § 7.2.1.5 PP=2048 reconciliation — short PPs unaffected by 12-token ChatML overhead at this prompt range).
- **§ 7.7 Rust hygiene** — `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, `cargo build --release` clean for every commit.

---

## § 8 Sequencing — P5i.b / P5j gated by P5i.a outcome

- **P5i.b dispatches IFF** T3 立项 verdict + Boss Scope gate approval.
- **P5j** can dispatch in parallel with P5i.b (long-PP Lane B work; different candidate set per spec § 1.2). P5j scope decision is separate.
- **P5i.a** must close as Full PASS or Feasibility PASS before P5i.b spec begins. Blocked status requires resolving the blocker first.

---

## § 9 References

- P5h+1 close-out: `docs/p5h+1-close-out.md`
- P5h+1 ranking snapshot: `docs/p5h+1-ranking-snapshot.md` + `reports/p5h+1-ranking-snapshot.md`
- P5h spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` (§ 1.2 gap table, § 7.2 ship state, § 8 sequential decision, § 7.2.1.5 PP=2048 partition)
- P5h+1 plan + spec: `docs/superpowers/{plans,specs}/2026-05-23-ironmlx-p5h+1-attribution-gap-closure-*.md`
- Codex P5i scope review (gitignored): `reports/p5i-codex-review-questions.md`
- ironmlx self_qmm precedent: `ironmlx/src/nn/self_qmm/` (415 LoC + qmm_t.metal.in)
- ironmlx gather_qmm call sites: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs:357 + 481`
- MLX upstream gather_qmm: `mlx/src/quantization.rs:324` (wraps `mlx_sys::quantization::ffi::gather_qmm` → C++ Metal `gather_qmm_rhs` at `mlx/backend/metal/quantized.cpp:1484`)
- mlx-lm baseline: `mlx_lm/models/qwen3_next.py:320` (`SwitchGLU`)
- omlx baseline: `/Users/xin/workspace/iron-rivals/omlx` source-CLI
- iron-bench harness: `/Users/xin/workspace/iron-rivals/iron-bench`
- Memory: `[project_p5h_findings]`, `[project_p5g_findings]`, `[project_p8a_stage9_findings]`, `[project_cross_device_tuning_deferred]`, `[reference_current_machine]`, `[reference_iron_rivals_baselines]`, `[feedback_design_philosophy]`, `[feedback_no_spec_from_competitors]`, `[feedback_performance_stability_priority]`, `[feedback_iron_bench_priority]`, `[feedback_omlx_cli_default]`, `[feedback_task_breakdown_bounded]`, `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[feedback_no_empty_commits]`, `[feedback_review_spec_before_commit]`, `[feedback_device_aware_tile]`, `[feedback_self_review_before_handoff]`
