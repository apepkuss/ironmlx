# P5h+2.a — PP=512 Measurement Protocol Fix Design (2026-05-23)

**Status:** Spec — ready for plan writing.
**Branch:** new branch `ironmlx-p5h+2-a-pp512-measurement` (fork from `ironmlx-p5i-a-gather-qmm-feasibility` HEAD `6e3b40e`).
**Predecessor:** P5i.a closed Feasibility PASS 2026-05-23; Codex round-3 directive specifies "P5h+2.a PP=512 measurement protocol fix; 目标 ±2% 或重定义可接受噪声带"; P5i.a T4 found 20-25% per-run variance + 7-run median ±5-10% standard error → current protocol cannot reliably detect ±2% effects at PP=512.

**Source docs:**
- P5i.a close-out: `docs/p5i-a-close-out.md` (commit `00178f0`)
- P5i.a results + Codex round-2 reviews: `reports/p5i-a-results-codex-review.md` (gitignored; § 7 records Codex decisions)
- P5i.a bench log (gitignored): `reports/p5i-a-bench-log.md` (T4 noise floor measurements)
- P5h spec § 7.2 (current ±2% noise band invariant): `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`
- Boss memory: `[feedback_design_rigor]`, `[feedback_iron_bench_priority]`, `[feedback_omlx_cli_default]`, `[feedback_task_breakdown_bounded]`, `[feedback_performance_stability_priority]`, `[feedback_serial_perf_experiments]`, `[feedback_no_empty_commits]`, `[feedback_review_spec_before_commit]`, `[project_p5h_findings]`

---

## § 1 Goal + phase nature

P5h+2.a is a **measurement protocol fix** phase, prerequisite for the PP=512 arm of P5i.c. It exists to:
1. Empirically characterize the actual standard error of the existing PP=512 measurement protocol (7-run median, 5-min preheat).
2. Iterate `(RUNS, cooldown, preheat)` configs to find one that supports `±2%` decision-making at PP=512 (Approach A).
3. Validate candidate protocols with **independent spawn+sweep repeats**, not only within-sweep bootstrap, so the selected band captures between-sweep drift.
4. If Approach A insufficient, do lightweight thermal/request-state/allocator investigation to identify or bound the variance root cause (Approach B).
5. If both A and B fail, redefine the spec § 7.2 noise band for PP=512 to an empirically-achievable level + document the root cause or quantified ambiguity + extend aggregator to emit per-PP 95% CI (Approach C fallback).

**Out of scope**: no ironmlx src changes outside aggregator + test files. No iron-bench tool enhancement (Approach C from brainstorming — new metrics framework — deferred to P5h+3). No other PPs (only PP=512). No P5i.c work (separate parallel phase).

---

## § 2 Background — P5i.a T4 noise finding

P5i.a T4 (Outcome C, no commit; documented in `reports/p5i-a-bench-log.md` T4 section) measured PP=512 under the canonical 5-min preheat protocol and found:

- **Within-sweep per-run variance**: 20-25% (e.g. confirm sweep PP=512 individual runs spanned 1331-1471 pp_tps over 7 measurements; second-order thermal bump on top of preheat saturation)
- **7-run median standard error**: ±5-10% (empirical from comparing first vs confirm sweep medians)
- **Practical consequence**: spec § 7.2 ±2% noise band is unachievable at PP=512 with current protocol; any single sweep can shift by ±5-10% from between-sweep drift / thermal state / scheduler state / allocator state. A single-sweep bootstrap alone is therefore insufficient because it cannot see cross-spawn non-stationarity.

This noise floor matters for downstream decisions:
- P5i.a T2 fusion's PP=512 perf signal was inconclusive (75s-preheat sweep reported +3.55%; canonical 5-min-preheat sweep reported -8.0%; both fall within ±5-10% protocol noise per Codex round-2 keep-vs-revert analysis)
- P5i.c's PP=512 arm cannot reliably land/reject ±1-5% optimizations without protocol fix
- Any P5h+2.b self_qmm lookup arity work driven by PP=512 ROI signal needs trustworthy SE

Codex round-3 directive (`reports/p5i-a-results-codex-review.md` § 7):
> "PP=512 不应继续用当前 7-run 协议判断 ±2% 级变化，应该先做 measurement protocol fix，至少把标准误压到能支持决策。"

---

## § 3 Close Gate (success condition)

P5h+2.a closes IFF ONE of:

### 3.1 Outcome (a) — ±2% achievable
New PP=512 protocol achieves **95% CI half-width ≤ ±2%** on median pp_tps for the target comparison mode, validated by:
- Within-sweep bootstrap CI on each candidate sweep; AND
- At least **3 independent spawn+sweep repeats** for the selected protocol, with between-sweep median spread also inside the same ±2% envelope; AND
- For ironmlx-vs-omlx target comparisons, both ironmlx and omlx PP=512 sweeps use the selected protocol and expose CI. If only ironmlx repeats are collected, Outcome (a) unblocks ironmlx pre/post regression decisions only, not external omlx-target claims.

Spec § 7.2 noise band stays at ±2% (unchanged); only the protocol parameters (RUNS / cooldown / preheat / repeat count) are updated. The new parameters are committed to `docs/p5h+2-a-pp512-protocol.md` + integrated into any aggregator validation that depends on per-PP RUNS expectations.

### 3.2 Outcome (b) — band redefined
±2% NOT achievable within the 7-task budget (T1 Approach A failed + T2 Approach B either identified a root cause that is not protocol-fixable in scope OR bounded the ambiguity enough to choose an honest decision band). Spec § 7.2 noise band amended for PP=512 to the empirically-achievable level (e.g. ±5% or ±10%), with:
- Documented root cause or quantified ambiguity from Approach B investigation (thermal drift / request-state drift / allocator non-determinism / scheduler state / etc.)
- Spec amendment in a new § 7.2.x subsection scoped to PP=512
- Aggregator emits per-PP 95% CI in `summary.json` output for every measured backend available (ironmlx and omlx when both are present), so downstream decisions see explicit confidence intervals, not just point estimates
- Future P5i.c PP=512 land conditions inherit redefined band

### 3.3 Status vocabulary (mirror P5i.a § 3.2 conventions)

- **Full PASS**: Outcome (a) achieved
- **Feasibility PASS**: Outcome (b) achieved + root cause documented or ambiguity quantified + future-fix path named (e.g. "thermal drift requires hardware cooldown beyond current iron-bench protocol; consider longer between-sweep intervals or different metric in P5h+3")
- **Blocked**: A + B + C all fail (e.g. bootstrap/repeat analysis incomplete, ambiguity not bounded enough to set a band, or aggregator can't emit CI). Should not happen given Approach C fallback exists.

---

## § 4 Tasks (6, per `[feedback_task_breakdown_bounded]` ≤7)

### § 4.1 T0 — Phase 0 characterization (Approach A)

**Goal**: empirically measure within-sweep CI per RUNS configuration from a single big sweep, and record enough drift signals to decide which candidates deserve independent repeat validation.

**Approach**:
- Single ironmlx spawn cycle at PP=512: 5-min preheat + 30× iron-bench measured runs
- Capture raw per-run `pp_tps` + per-run `ttft_ms` into `/tmp/p5h-2a-t0-pp512-30runs.csv`
- Create `tools/p5h_2a_se_analysis.py`: bootstrap-resample script that takes RUNS-30 raw data, computes 95% CI of median for RUNS subset sizes ∈ {7, 15, 21, 30}; output JSON per-RUNS SE
- Script must also emit per-run drift diagnostics (`run_idx` vs `pp_tps`, `run_idx` vs `ttft_ms`, robust slope estimate) so obvious non-stationarity is visible before selecting candidates
- Save to `/tmp/p5h-2a-t0-se-analysis.json`

**Acceptance**: T0 closes when RUNS-30 CSV + SE analysis JSON written + per-RUNS within-sweep CI + drift diagnostics reported. T0 is characterization only; it is NOT allowed to prove Outcome (a) by itself.

**Expected wall**: 1 spawn cycle (~5min preheat + ~30× ~5s per measured run + kill ≈ 10min wall) + analysis (~5min)

### § 4.2 T1 — Phase 1A protocol candidates (Approach A)

**Goal**: try 2-3 alternate `(RUNS, cooldown, preheat)` configs; pick best per cost vs SE.

**Approach**:
- Based on T0 RUNS sweep SE data, identify which configurations are most promising
- Try 2-3 alternate configs (e.g. RUNS=21 + same cooldown; RUNS=15 + 2× between-sweep cooldown; RUNS=7 + 10min preheat); each config starts with one ironmlx spawn cycle + fresh bench
- Run T0's SE bootstrap script on each config's data
- Promote any candidate whose within-sweep CI is near target (≤±2.5%) into **independent repeat validation**: run at least 3 independent spawn+sweep repeats for that candidate and compute between-sweep median spread
- **Selection rule**: pick config with **lowest (wall_time × combined CI)** that meets ±2% target in both within-sweep bootstrap and between-sweep repeat validation
- If NO config hits ±2% target → T2 Approach B triggers

**Acceptance**: T1 closes when 2-3 configs benched + SE analysis per config + repeat validation for any candidate near target + selection verdict ("config X meets ±2%" OR "no config hits target; T2 triggers")

**Expected wall**: 2-3 initial config sweeps plus 3 repeat sweeps for any near-target candidate. Budget ~2-4hr GPU if one candidate reaches repeat validation; less if all candidates clearly miss in the first sweep.

### § 4.3 T2 — Phase 1B fallback (CONDITIONAL — only if T1 doesn't hit ±2%)

**Goal**: identify or bound the variance root cause via lightweight thermal/request-state/allocator/scheduler investigation.

**Approach**:
- powermetrics polling during T0's RUNS=30 sweep (re-spawn with capture):
  - Attempt non-interactive telemetry first: `sudo -n powermetrics --sample-interval 1000 --samplers gpu_power,thermal -i 1000` running concurrent with the sweep
  - Save to `/tmp/p5h-2a-t2-powermetrics.log` if available
  - If `sudo -n` fails, telemetry is **non-blocking**: document "powermetrics unavailable in this execution context" and fall back to TTFT/pp_tps drift analysis
- Per-run TTFT drift analysis: linear regression of `ttft_ms` vs `run_idx` from T0 data; if slope significant → thermal drift across sweep is variance source
- Request-state determinism control: verify prompt, prompt_len, max_tokens, warmup, cache state, server fresh-start policy, request ordering, and any batching/chunking knobs are fixed across repeats. If a suspected state variable differs, re-bench with that variable pinned and compare CI.
- Allocator/scheduler/dispatch non-determinism: optional — check MLX allocator/cache/log envs only if they can be toggled without changing production semantics; document but do not make them bench-mandatory
- Identify dominant root cause OR quantify that multiple sources remain ambiguous, then propose either a targeted protocol fix or an empirically honest band redefinition

**Acceptance**: T2 closes with documented root cause (thermal / request-state / allocator / scheduler / unknown) OR quantified ambiguity, plus proposed targeted fix OR rejection note (e.g. "thermal drift requires cooldown beyond current scope; recommend Approach C")

**Expected wall**: ~1-2 spawn cycles + powermetrics analysis = ~1hr GPU + 30min analysis

### § 4.4 T3 — Spec new PP=512 protocol (Approach A landed) OR band redefinition (Approach C fallback)

**Goal**: spec the chosen protocol or redefined band in `docs/`.

**Path A** (T1 hit ±2%):
- Create `docs/p5h+2-a-pp512-protocol.md` committed:
  - Selected (RUNS, cooldown, preheat) values
  - Required independent repeat count (minimum 3) and whether the protocol is validated for ironmlx-only pre/post decisions or ironmlx-vs-omlx external target decisions
  - Empirical SE achieved
  - Rationale for selection (lowest wall × final_CI meeting target)
  - Reproducibility command (iron-bench invocation)
- Update `tools/p5i_a_baseline_aggregate.py` (and any downstream aggregator) `EXPECTED_RUNS_PER_PP` to support per-PP / per-target override if RUNS differs between PP=128 (still 7) and PP=512 (new value), and to emit CI fields when repeat/sweep data is available
- Update any spec § 7.x sub-references that cite RUNS=7

**Path B** (T1+T2 both failed; Approach C fallback):
- Create `docs/p5h+2-a-pp512-protocol.md` documenting:
  - Empirical max-achievable SE at PP=512 (e.g. ±5%)
  - Root cause or quantified ambiguity from T2 (cite powermetrics if available / TTFT drift / request-state determinism result)
  - Why ±2% is unachievable in scope (specific quantitative reason)
  - Recommended new spec § 7.2 noise band for PP=512 (e.g. ±5%)
- Edit `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2: add subsection `§ 7.2.2 PP=512 noise band amendment (post-P5h+2.a)` documenting the redefined band + cross-reference to root cause
- Update `tools/p5i_a_baseline_aggregate.py` to emit per-PP 95% CI in `summary.json` output (computed via bootstrap on actual sweep data; new field e.g. `ironmlx_pp_tps_ci95_half_width_pct`)
- Any downstream consumer of summary.json (P5i.c PP=512 land conditions) inherits the new band + uses CI for decisions

**Acceptance**: T3 closes with `docs/p5h+2-a-pp512-protocol.md` committed + aggregator updates landed (Path A: per-PP/per-target RUNS override + CI emission when data is available; Path B: per-PP CI emission + amended band)

**Expected wall**: 1-2 day write-up + small Python edits

### § 4.5 T4 — Validate new protocol

**Goal**: re-measure T0 baseline (or equivalent) under new protocol to verify achieved noise band.

**Approach**:
- Fresh ironmlx spawn + preheat + measurement sweep at PP=512 using the new (RUNS, cooldown, preheat) values
- For Path A: run at least 3 independent ironmlx spawn+sweep repeats and confirm combined within-sweep + between-sweep CI ≤ ±2%
- If the protocol will be used for ironmlx-vs-omlx target decisions, also run omlx PP=512 with the same selected protocol and emit omlx CI; otherwise explicitly mark external-target decisions as still blocked
- For Path B: confirm empirical SE matches T0+T1 prediction (within reason; not a tight gate since the band is redefined)
- Save validation data to `/tmp/p5h-2a-t4-validate.csv` + run aggregator with new CI emission

**Acceptance**: T4 closes when validation repeat sweeps run + SE/CI computed + matches T3 commit's stated band and comparison scope (ironmlx-only vs ironmlx-vs-omlx) is explicit

**Expected wall**: 3+ spawn cycles for ironmlx-only validation; add omlx repeats if external target decisions are in scope + analysis (~10min)

### § 4.6 T5 — Close-out

**Goal**: P5h+2.a close-out doc + memory + commit per `[feedback_no_empty_commits]`.

**Approach**: write `docs/p5h+2-a-close-out.md` (committed) with:
- Status (Full PASS Path A / Feasibility PASS Path B / Blocked) + date + branch + commit chain
- § 1 Close Gate result (cite per-RUNS SE measurements; Path A: chosen config + comparison scope; Path B: redefined band + root cause or quantified ambiguity)
- § 2 What landed (T3 docs/protocol committed; aggregator updates)
- § 3 P5i.c PP=512 arm unblocked status — explicitly state whether the new protocol supports ironmlx-only pre/post decisions, ironmlx-vs-omlx target decisions, or both
- § 4 Follow-up items (e.g. P5h+3 iron-bench tool enhancement for new metrics; Approach C from brainstorming originally)
- § 5 Memory update — extend `project_p5h_findings.md` with P5h+2.a closure section
- § 6 References

**Acceptance**: T5 closes with committed close-out doc + memory updated

**Expected wall**: 1 day write-up

---

## § 5 Approach A details

### 5.1 Bootstrap-resample methodology

For each candidate RUNS value N ∈ {7, 15, 21, 30} (T0) and each alternate config (T1):

1. From the RUNS-30 raw `pp_tps` distribution (T0), draw 1000 bootstrap samples of size N (with replacement)
2. For each bootstrap sample, compute median pp_tps
3. From the 1000 medians, compute 95% confidence interval (e.g. `[percentile_2.5, percentile_97.5]`)
4. SE = half-width of CI / point-estimate median = relative half-width as percentage
5. Compare to ±2% target as a **screening metric only**
6. For candidate protocols, combine within-sweep bootstrap with independent repeat medians:
   - compute each repeat's median
   - compute repeat-median range and/or bootstrap over repeat medians
   - final CI half-width is the conservative max of within-sweep CI and between-sweep CI
   - Outcome (a) requires this final CI ≤ ±2%

### 5.2 Selection rule

Cost function: `wall_time(config) × final_CI(config)`

- `wall_time(config)` = repeat_count × (preheat seconds + RUNS × per-run seconds + cooldown seconds)
- `final_CI(config)` = max(within-sweep bootstrap CI, between-sweep repeat CI), as fraction of median

Pick config with **lowest wall_time × final_CI** that meets ±2% target. If multiple meet target, prefer lower wall_time. If none meet target → trigger T2 Approach B.

### 5.3 Reference: per-PP RUNS asymmetry rationale

PP=128 has historically been stable at RUNS=7 per T0/T1/T2 P5i.a measurements (no analogous variance issue reported). PP=512 may need higher RUNS specifically because of its larger per-run variance (20-25% vs typical ±2% at small PP). Per-PP override in aggregator is the natural mechanism (instead of forcing RUNS=21 globally and wasting wall on PP=128).

---

## § 6 Approach B fallback details (T2 only if T1 fails)

### 6.1 Per-run TTFT drift analysis

- Read T0's per-run `ttft_ms` (or equivalent timing field) for the RUNS-30 sweep
- Compute linear regression of `ttft_ms` vs `run_idx`
- If slope is statistically significant (e.g. p < 0.05) AND magnitude is meaningful (≥10ms drift across 30 runs) → thermal drift is a likely variance source
- Document slope + p-value + drift magnitude

### 6.2 powermetrics correlation

Run during a fresh spawn-preheat-sweep cycle when non-interactive sudo is available:
```bash
sudo -n powermetrics --sample-interval 1000 --samplers gpu_power,thermal -i 1000 > /tmp/p5h-2a-t2-powermetrics.log &
PMETRICS_PID=$!
# ... do the iron-bench sweep ...
kill $PMETRICS_PID
```

If `sudo -n` fails, do not prompt or block the run. Record telemetry as unavailable and continue with TTFT/pp_tps drift analysis. If telemetry is available, correlate GPU thermal trace (temperature over time) with per-run pp_tps (run start time → pp_tps mapping). If thermal saturates partway through sweep and pp_tps drops in correlation → thermal drift confirmed.

### 6.3 Request-state determinism control

MoE router top-k selection during prefill is deterministic for fixed model weights and fixed input; it is not controlled by sampler PRNG seed. Do NOT chase a "routing seed" knob.

Instead verify and, if needed, pin request/protocol state that can alter timing or batching:
- identical prompt text / prompt_len / chat-template policy / max_tokens / warmup count
- fresh server state before each independent repeat, or explicitly documented cache state
- no concurrent clients; one backend active at a time
- fixed iron-bench request ordering and same PP=512-only sweep shape
- identical environment variables that affect MLX allocator/cache/compile behavior

If pinning a state variable materially reduces CI, document that variable as the variance source and make it part of the protocol. If no variable explains the spread, mark the root cause ambiguous and proceed to Path C with the empirically observed band.

### 6.4 Output verdict

T2 produces a markdown note in `reports/p5h+2-a-bench-log.md` (gitignored) summarizing:
- TTFT drift magnitude + significance
- powermetrics correlation summary
- Request-state determinism checks and any pinned-variable effect
- Identified root cause or quantified ambiguity ("multiple sources contribute; no single bounded fix in P5h+2.a")
- Recommended path: Path A (still try a config based on root cause) or Path C (redefine band)

---

## § 7 Approach C fallback details (T3 Path B; if A + B both fail)

### 7.1 Spec § 7.2 amendment

Add `§ 7.2.2` to the P5h spec:

> "**PP=512 noise band amendment (post-P5h+2.a)**: per P5h+2.a `docs/p5h+2-a-close-out.md`, the ±2% noise band of § 7.2 is amended for PP=512 to ±<X>% based on empirical SE of the canonical RUNS-<N> protocol on M5 Max. Root cause or bounded ambiguity: <thermal drift / request-state drift / allocator non-determinism / scheduler state / multiple sources>. P5h+2.a Approach B documented the path forward — see `docs/p5h+2-a-pp512-protocol.md` for full details + reproducibility."

Where `<X>` is the empirically-achievable SE rounded up (e.g. ±5% or ±10%), and `<N>` is the chosen RUNS count.

### 7.2 Aggregator CI emission

Update `tools/p5i_a_baseline_aggregate.py`:

- Add bootstrap CI computation (importable from new `tools/p5h_2a_se_analysis.py`)
- For each PP in summary JSON, add new fields:
  - `ironmlx_pp_tps_ci95_low`
  - `ironmlx_pp_tps_ci95_high`
  - `ironmlx_pp_tps_ci95_half_width_pct` (= (high - low) / 2 / median × 100)
  - Same for omlx when omlx is measured under the selected protocol
- Add new pytest test exercising CI computation

### 7.3 Downstream consumer guidance

Any downstream code that imports summary.json (P5i.c PP=512 land conditions; future P5h+2.b ROI gate) should:
- Use CI half-width for noise-bound comparison: a delta is meaningful only if `|delta_pct| > CI_half_width(ironmlx) + CI_half_width(omlx)` (combined noise envelope) when both backends are measured
- If only ironmlx CI is available, restrict conclusions to ironmlx pre/post regression or improvement decisions; do not make ironmlx-vs-omlx target claims
- Document in their spec which noise envelope they apply

---

## § 8 Out of scope (deferred)

- **Other PPs measurement protocol** (only PP=512 in scope; PP=128 stays at RUNS=7 unless evidence emerges)
- **Iron-bench tool enhancement** for new metrics framework (trimmed mean / IQR / per-iter throughput) — deferred to P5h+3
- **P5i.c work** (separate parallel phase; P5i.c.0 brainstorm next)
- **P5h+2.b self_qmm lookup arity extension** — conditional on P5i.c re-rank
- **P5h+2.c low-risk cleanup batch** — independent; deferred
- **T1-only canonical sweep** for historical attribution — optional, non-blocking
- **Spec § 1.2 doc-fix** for +24/+74 unit clarification — small P5h+2 doc task, separate from P5h+2.a
- **P5i.a T2 keep-vs-revert** — already decided KEEP per Codex round-2

---

## § 9 Validation gates (P5h+2.a)

- **§ 9.1 No production ironmlx src changes** — only `tools/p5i_a_baseline_aggregate.py` + new `tools/p5h_2a_se_analysis.py` + new `tests/test_p5h_2a_se_analysis.py` + docs. No `ironmlx/src/...` files touched.
- **§ 9.2 Statistical rigor** — bootstrap methodology + independent repeat validation documented in spec § 5 + reproducible script committed; results not just point estimates but with CI and explicit comparison scope
- **§ 9.3 Reproducibility** — protocol parameters in docs/ committed; aggregator validation enforces per-PP/per-target RUNS expectation and emits per-PP CI when data is available
- **§ 9.4 Production parity not impacted** — P5h+2.a is measurement-only work; flag-OFF production ironmlx pp_tps must not regress (verify smoke `p5_qwen35_moe_smoke` PASS unchanged after any aggregator/test changes)
- **§ 9.5 Serial GPU per `[feedback_serial_perf_experiments]`** — all sweeps run serially; never concurrent with P5i.c sweeps on same machine
- **§ 9.6 Python hygiene** — `uv run --with ruff ruff check` + `ruff format --check` on all new/modified Python files; pytest pass on new tests
- **§ 9.7 Rust hygiene** — N/A (no Rust changes expected); if any Rust files need touching despite scope, follow AGENTS.md Rust verification cycle

---

## § 10 Branch + sequencing — parallel with P5i.c Phase 0

- **New branch** `ironmlx-p5h+2-a-pp512-measurement` (fork from `ironmlx-p5i-a-gather-qmm-feasibility` HEAD `6e3b40e`)
- **Parallel with P5i.c Phase 0** (separate branch; spec/plan/dispatch independent; GPU sweeps serialize via `[feedback_serial_perf_experiments]`)
- **P5i.c PP=512 arm WAITS** for P5h+2.a close (only P5i.c PP=128 arm runs in parallel during P5h+2.a)
- **Estimated wall**: 3-5 days total
  - T0 (~15min GPU) + T1 (~2-4hr GPU with repeat validation if a candidate is near target) + T2 conditional (~1hr GPU, powermetrics non-blocking) + T3 (1-2 day docs+aggregator edit) + T4 (3+ repeat sweeps, plus omlx repeats if external target decisions are in scope) + T5 (~1 day docs)
- **Expected commits**: 1-2 (T3 protocol doc + aggregator updates; T5 close-out)

---

## § 11 References

- P5i.a close-out: `docs/p5i-a-close-out.md`
- P5i.a baseline: `docs/p5i-a-baseline.md` (T0 canonical measurement)
- P5i.a bench log (gitignored): `reports/p5i-a-bench-log.md` (T4 noise floor measurements)
- P5i.a results review (gitignored): `reports/p5i-a-results-codex-review.md` (§ 7 Codex round-2 directives)
- P5h spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` (§ 7.2 current ±2% noise band invariant)
- Aggregator: `tools/p5i_a_baseline_aggregate.py` (P5i.a Codex round-1 fixup `6e3b40e` — strict validation + run_idx set check)
- New tool (P5h+2.a T0): `tools/p5h_2a_se_analysis.py` (bootstrap-resample SE script)
- iron-bench: `iron-bench/` (workspace member; per `[feedback_iron_bench_priority]`)
- Memory: `[project_p5h_findings]`, `[feedback_design_rigor]`, `[feedback_iron_bench_priority]`, `[feedback_omlx_cli_default]`, `[feedback_task_breakdown_bounded]`, `[feedback_performance_stability_priority]`, `[feedback_serial_perf_experiments]`, `[feedback_no_empty_commits]`, `[feedback_review_spec_before_commit]`
