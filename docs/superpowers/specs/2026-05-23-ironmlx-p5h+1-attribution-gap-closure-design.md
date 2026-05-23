# P5h+1 — Attribution Gap Closure Design (2026-05-23)

**Status:** Spec — ready for plan writing.
**Branch:** new branch `ironmlx-p5h+1-attribution` (forked from `ironmlx-p5h-perf` HEAD `c4c4642`).
**Predecessor:** P5h closed measure-only 2026-05-23; 7/9 spec § 7.2 gates LOCKED PASS; gates #5 + #6 DEFERRED.

**Source docs:**
- P5h spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` (§ 7.2.1 T5 close-out verdict)
- P5h T5 close-out: `docs/p5h-t5-close-out.md`
- P5h T5 attribution detail (gitignored): `reports/p5h-attribution.md`
- P5h overall ship state memory: `project_p5h_findings.md`
- Codex P5h+1 review: `reports/p5h+1-codex-review-questions.md` (gitignored)

---

## § 1 Goal

P5h+1 is an **attribution gap closure** phase, **not** an optimization phase. It exists solely to unblock P5i (short-PP optimization +24-74% per spec § 1.2) and P5j (long-PP optimization +110-128%) candidate ranking + target feasibility verdict.

**Out of scope**: no optimization candidates implemented; no kernel rewrites; no production behavior changes. P5h+1 may add a `p5h-profile`-only measurement flag that deliberately changes MLX materialization timing for attribution runs; that mode is never used as production pp_tps evidence.

---

## § 2 Close Gate (P5h+1 success condition)

P5h+1 closes IFF all four:

1. **Lane A wrapper dominance resolved in measurement mode** — PP=128/512 T5 re-run with `--p5h-measurement-eval-probes` no longer returns `data_insufficient` due to `first_token_sampling_materialize_and_sample`; the old `first_token_sampling` wrapper must no longer emit after T1. The hard gate is: the largest Lane A lazy-materialization wrapper span share is ≤ `WRAPPER_DOMINANCE_THRESHOLD` (currently 50% in `roi_ranking.py`); ≤ 30% remains a desired target, not the close gate.
2. **Lane B wrapper dominance resolved in measurement mode** — PP=2048/4096/8192/16384 T5 re-run with `--p5h-measurement-eval-probes` no longer returns `data_insufficient` due to `gs_chunk_N`. Lane B `decoder_layer_N` + `attention_path` + `mlp_path` wrappers emit; T2 + T3 + T4 substeps chain under them per request × chunk × decoder layer. The hard gate is: `gs_chunk_N` share is ≤ `WRAPPER_DOMINANCE_THRESHOLD` for every P5j PP.
3. **Existing gates preserved** — coverage_pct ≥ 95% per PP, request_id join 100%, schema validator (REQUIRED + ALLOWED) all PASS unchanged.
4. **Actionable ranking surfaced** — re-run T5 ROI ranking produces at least one non-wrapper, non-`unattributed_*` top-3 candidate for every target PP: 128, 512, 2048, 4096, 8192, 16384. Per-PP verdict is either `yes`, `yes_with_scope_gate`, or `no_under_measured_cap` (NOT `data_insufficient`) for all target PPs. The verdict must label its timing source as measurement-probe attribution and must cite the flag-OFF / previous-T5 production root as the denominator for target feasibility; measurement-mode pp_tps is not production evidence.

If any of (1)-(4) fails after this spec's tasks execute, P5h+1 stays open — continue attribution work; do NOT dispatch P5i/P5j.

---

## § 3 Tasks (5)

Per `[feedback_task_breakdown_bounded]` (single plan ≤ 5-7 tasks):

| # | Task | Type | Approx commits |
|---|---|---|---|
| T1 | Lane A lazy-boundary split + measurement-only span-boundary eval probes | src + harness sanity | 2-3 |
| T2 | Lane B `gs_chunk_N` deep substep instrumentation + allow-list update + inherited `chunk_idx` context | src + schema + aggregator | 3-4 |
| T3 | Re-run T5 capture sweep + aggregator on updated instrumentation | GPU sweep + Python re-run | 0 commits (data refresh; existing harness) |
| T4 | Re-rank P5i/P5j + verify Close Gate (§ 2) | analysis + verdict | 1 commit (verdict snapshot) |
| T5 | P5h+1 close-out: docs/p5h+1-close-out.md + memory + spec § 7.2.1 update + optional spec § 1.2 PP=2048 reconciliation | docs | 1 commit |

Total expected: ~7-9 commits.

---

## § 4 Task detail

### § 4.1 T1 — Lane A lazy-boundary split + measurement-only eval probes

**Site**: `ironmlx/src/core/scheduler.rs:1059` (span open) + `:1117` (sample_batch call). Sampler trace via `ironmlx/src/core/sampler.rs:279` (`.to_vec()` host sync).

**Phase 1 (zero production impact)**: replace `first_token_sampling` with two sibling spans under the root:
- `first_token_sampling_prepare` — wraps reshape + per-row sampler/history preparation. This is CPU/graph-construction work and should stay small.
- `first_token_sampling_materialize_and_sample` — wraps `sample_batch(...)`, including any remaining lazy materialization plus sampler work. With measurement probes OFF, this span is expected to remain the diagnostic lazy wrapper. With measurement probes ON, most prefill materialization should have moved into probed T2/T3/T4 spans, leaving mostly sampler/argmax/to_vec cost.

Do **not** keep `first_token_sampling` as an emitted wrapper after T1. Keeping both old and new names would require compatibility handling in schema + ranking; this phase intentionally uses the new schema only.

**Phase 2 (measurement-only opt-in)**: add `--p5h-measurement-eval-probes` to `ironmlx serve`, available only under `--features p5h-profile`. When ON, selected T2/T3/T4 span bodies force materialization of their output arrays before the span closes. This is **not** a single eval before the sampler: a sampler-side eval would only move cost from `first_token_sampling` into `model_prefill_forward` residual / eval-barrier time, and would not retroactively assign GPU wait to already-closed substep spans.

The probe rule is:
- For an existing P5h substep that returns one or more `mlx::Array` values and is an ROI candidate, call `mlx::transforms::eval(&[...])` on the returned array(s) inside that span body when the flag is ON.
- For CPU-only spans or spans whose returned value cannot be materialized as an MLX array, keep current timing and document them as construction/CPU spans.
- The T5 ROI ranking produced by P5h+1 uses measurement-probe data for attribution. Production pp_tps and behavior evidence still comes from the flag OFF path. Feasibility reporting must separate these two sources: probe data ranks candidates; flag-OFF production root time is the denominator for any target-gain discussion.

**Rationale for this design (per Codex Q-P5h+1-2 + review round 1)**:
- Span split alone documents the lazy boundary but cannot make already-closed T2/T3/T4 spans accrue GPU wait.
- A single sampler-side eval changes where materialization happens but still does not produce substep-level ROI.
- Span-boundary eval probes intentionally serialize selected lazy work in measurement mode so each probe span owns the incremental materialization cost needed for ranking.
- The flag preserves production/default semantics and makes the measurement-mode scheduling change explicit in reports.

**Acceptance criteria (T1 close)**:
- Both subspans emit on every Lane A request.
- With the flag OFF, no new `mlx::transforms::eval(...)` executes on the scheduler path; smoke outputs match and pp_tps remains within ±2% on the existing smoke benchmark.
- With the flag ON, PP=128/512 no longer fail ROI ranking due to `first_token_sampling_materialize_and_sample` wrapper dominance.
- T5 close-out records both flag-OFF parity data and flag-ON attribution data; only flag-ON data is used for P5i/P5j candidate ranking.

### § 4.2 T2 — Lane B `gs_chunk_N` deep substep instrumentation + `chunk_idx` schema

**Sites**: `ironmlx/src/core/server/openai.rs::serve_via_gs_stream` (Lane B entry), `ironmlx/src/core/generate.rs::GenerationStream::new` chunk loop, and the existing decoder-layer / GDN / MoE / lm_head substep sites reached from `model.forward_on` / `model.forward_text_hidden`.

**Wrapper opening**: mirror Lane A T0a.11 step 1 pattern at decoder-layer granularity. Lane B must emit `decoder_layer_N`, `input_norm`, `attention_path`, `residual_overhead`, `post_attention_norm`, `mlp_path`, and the existing T2/T3/T4 substeps reached under those wrappers. Do NOT add a coarse child directly under `gs_chunk_N` as the only deep signal; per-decoder-layer attribution is required so Lane A and Lane B rankings are symmetric.

**Central allow-list update**: `core/p5h.rs::try_with_p5h_span_from_current_trace` currently suppresses all Lane B deep spans except the top-level allow-list. P5h+1 must update that allow-list to include every Lane B span name opened through `try_with_p5h_span_from_current_trace`; otherwise the deep call sites will continue to no-op even though the design says they emit. This Rust try-helper allow-list is a subset of `schema_validator.py` `LANE_B_ALLOWED_TREE` because some top-level spans are opened through explicit APIs rather than the try helper.

**Measurement eval probes**: Lane B shares the same lazy attribution problem as Lane A. Intermediate chunks currently materialize at `eval(hidden)` after `forward_text_hidden` returns; the last chunk can materialize at first-token sampling. Therefore T2 must enable the same `--p5h-measurement-eval-probes` span-boundary probes for Lane B substeps. Merely opening `attention_path` / `mlp_path` wrappers is insufficient to close `gs_chunk_N` dominance.

**`chunk_idx` schema field + propagation**:
- Add `chunk_idx: Option<u32>` to `SpanFields` and to the emission format (`tracing::info!("[p5h-profile] ... chunk_idx={} ...")`).
- Add a P5h thread-local chunk context stack, e.g. `P5H_CURRENT_CHUNK_STACK: RefCell<Vec<u32>>`.
- When opening `gs_chunk_N`, `GenerationStream::new` supplies the 0-indexed chunk number via `SpanFields { chunk_idx: Some(chunk_idx), ... }`; the P5h wrapper pushes that value for the duration of the chunk body and pops it on exit.
- For nested spans inside `gs_chunk_N`, emission uses `fields.chunk_idx.or(current_chunk_stack_top)`. This avoids manually threading `chunk_idx` through every decoder/GDN/MoE call site.
- Spans outside chunked context emit `chunk_idx=null` (Lane A, Lane B root, `gs_kv_cache_alloc`, role/content SSE spans, diagnostics).

**Schema validator extension**:
- Update `tools/p5h_aggregator/schema_validator.py` `LANE_B_REQUIRED_TREE` to include `decoder_layer_N`, `input_norm`, `attention_path`, `residual_overhead`, `post_attention_norm`, `mlp_path`, plus the T2/T3 substeps and T4 `cache_state_update` / `slice_last_and_project_lm_head` names that must emit under Lane B chunks. `tokenizer_encode` remains a pre-routing child of `http_parse_render_tokenize`; keep its Lane B REQUIRED/ALLOWED status aligned with the existing T4 rule rather than treating it as a `gs_chunk_N` descendant.
- Add `chunk_idx` to `Span` dataclass + `parse_line` regex.
- Update `LANE_B_ALLOWED_TREE` correspondingly.
- Enforce `chunk_idx != null` for spans transitively under `gs_chunk_N`; enforce `chunk_idx=null` for spans not under `gs_chunk_N`.

**Aggregator extension**:
- `compute_exclusive` already handles parent_span_id-based tree — no algorithmic change needed.
- Attribution CSV must include `chunk_idx` so raw analysis can trace repeated substeps back to request × chunk × layer.
- ROI ranking continues to use per-request sum-by-name (Codex T5-R Fix A); it ignores `chunk_idx` for default top-3 so repeated chunks are still summed correctly.
- Per-chunk diagnostic breakdown is optional and deferred unless P5h+1 data shows chunk-level variance changes the decision.

**Acceptance criteria (T2 close)**:
- Lane B `[p5h-profile]` records include `chunk_idx={int}` for `gs_chunk_N` and every descendant span inside that chunk.
- T2 + T3 substep span_names emit on Lane B requests (mirror Lane A coverage).
- Every Lane B deep span listed in `schema_validator.py` and opened through `try_with_p5h_span_from_current_trace` is present in the Rust try-helper allow-list.
- schema_validator accepts new spans without rejection.
- Lane B re-run T5 coverage_pct stays ≥ 95% per PP.

### § 4.3 T3 — Re-run T5 capture sweep

No new attribution logic beyond T1 + T2. Re-run `ironmlx/tests/p5h_t5_attribution_capture.rs` (commit `e328ee5`) against the updated instrumentation:

```bash
IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
cargo test -p ironmlx --release --features p5h-profile \
  --test p5h_t5_attribution_capture -- --ignored --test-threads=1 --nocapture
```

The T5 measurement capture used for ranking MUST spawn `ironmlx serve` with `--p5h-measurement-eval-probes`. The flag is a server CLI arg, not a `cargo test` arg. Either:
- Add `--p5h-measurement-eval-probes` to the existing `t5_attribution_capture` harness's `spawn_server_to_log` invocation for the measurement phase.
- OR add a second ignored test `t5_attribution_capture_measurement_mode` that uses the flag and writes distinct `/tmp/p5h+1-t5-*` outputs.

The flag-OFF parity smoke is a separate validation command; it is not the ranking input. T4 must nevertheless carry a production denominator: either reuse the prior P5h T5 flag-OFF root medians if T1/T2 parity holds within ±2%, or run a short flag-OFF root-baseline capture on the same PP set if parity is inconclusive.

Run aggregator + roi_ranking on the new `/tmp/p5h-t5-*` outputs (overwrites prior T5 data).

**No commit for T3** unless harness modification is needed (then folded into T1 task commits).

### § 4.4 T4 — Re-rank P5i/P5j + Close Gate verification

**Process**:
1. Read new `/tmp/p5h-t5-verdict.json` after T3 sweep + aggregator + roi_ranking.
2. Verify Close Gate (§ 2) — all four conditions met?
3. Snapshot detailed verdict + top candidates into `reports/p5h+1-ranking-snapshot.md` (gitignored detail).
4. Commit a concise summary in `docs/p5h+1-ranking-snapshot.md` (or fold the same summary into T5 close-out if T4 and T5 are executed together). The summary must separate `probe_attribution_root_us`, `production_root_us`, and the verdict timing-source caveat. No empty commits.

**If Close Gate fails**: do NOT close P5h+1; iterate on T1 / T2 instrumentation until gate passes.

### § 4.5 T5 — Close-out

Create `docs/p5h+1-close-out.md` (committed) summarizing:
- Goals + Close Gate satisfied (cite measurements)
- Lane A `first_token_sampling` replacement + measurement probe details
- Lane B `gs_chunk_N` deep substep details + `chunk_idx` schema field
- New P5i/P5j top candidates + verdict
- Spec § 7.2.1 update: gates #5 + #6 status change from DEFERRED to LOCKED PASS (or to a more specific blocked state if Gate fails)
- P5h+2 follow-up list (the items punted from this scope: emit cost reduction, T0b H4 control, T2/T3 op-level ablation if triggered, T4.2 mid-admit, etc.)

Update memory: extend `project_p5h_findings.md` with P5h+1 closure note (do NOT create new `project_p5h+1_findings.md` unless follow-up complexity warrants).

**Optional within T5 commit**: spec § 1.2 PP=2048 reconciliation (Codex Q-P5h+1-5 both, low priority):
- Spec § 1.2 docstring note: "PP=2048 is nominal P5j target; actual lane partition derived from runtime `routing_path` per `roi_ranking.py:observed_lane_for_pp`; iron-bench `--prompt-len 2036` available as boundary-control sweep but NOT replacing PP=2048 as primary measurement point."
- iron-bench: no code change needed (already supports any `--prompt-len`).

Commit: `docs(p5h+1): close-out — attribution gaps closed; P5i/P5j ranking unblocked`.

---

## § 5 Schema changes

### 5.1 `[p5h-profile]` emission format

Current (per `core/p5h.rs:313-332`, 13 fields):
```
[p5h-profile] request_id={} routing_path={} prompt_tokens={} seq={} layer_idx={} span_id={} parent_span_id={} span_name={} parent_span={} start_ns={} end_ns={} mode={} span_kind={}
```

After P5h+1:
```
[p5h-profile] request_id={} routing_path={} prompt_tokens={} seq={} layer_idx={} chunk_idx={} span_id={} parent_span_id={} span_name={} parent_span={} start_ns={} end_ns={} mode={} span_kind={}
```

`chunk_idx` slots after `layer_idx`. Default emitted as `null` for spans outside chunked context (Lane A + Lane B outer spans + diagnostic spans).

### 5.2 Span name additions

**Lane A new spans**:
- `first_token_sampling_prepare`
- `first_token_sampling_materialize_and_sample`

**Lane B new spans** (existing names, but newly emitted on Lane B):
- `decoder_layer_N`, `input_norm`, `attention_path`, `residual_overhead`, `post_attention_norm`, `mlp_path` (wrappers/substeps; previously Lane A only)
- All T2 GatedAttention 7 substeps + T3 MoE 8 substeps + T4 `cache_state_update` / `slice_last_and_project_lm_head` substeps reached inside `gs_chunk_N` (cascade via `P5H_CURRENT_SPAN_STACK`). `tokenizer_encode` is not a chunk descendant; it remains under `http_parse_render_tokenize`.

**Update `schema_validator.py`**:
- `LANE_A_REQUIRED_TREE`: REPLACE `first_token_sampling` with `first_token_sampling_prepare` + `first_token_sampling_materialize_and_sample` (T1 fully replaces the old wrapper; no grace period — the old span no longer emits after T1 lands).
- `LANE_B_REQUIRED_TREE`: add `decoder_layer_N`, `input_norm`, `attention_path`, `residual_overhead`, `post_attention_norm`, `mlp_path`, plus the T2/T3 substeps and T4 chunk-descendant names. Preserve the existing T4 treatment of `tokenizer_encode` as pre-routing HTTP work.
- `LANE_B_ALLOWED_TREE`: superset of REQUIRED.
- `Span` dataclass: add `chunk_idx: int | None = None`.
- `parse_line` regex: add `chunk_idx=(?P<chunk_idx>null|\d+)` group.
- Add structural rule: every tree span under a `gs_chunk_N` ancestor has non-null `chunk_idx`, and that value equals the ancestor `gs_chunk_N.chunk_idx`.
- Add closed-set rule: every Rust Lane B try-helper allowed span is contained in Python `LANE_B_ALLOWED_TREE`, and every Python Lane B allowed deep span that is opened via the try helper is contained in the Rust allow-list.

**Update `aggregator.py` output**:
- Attribution CSV header becomes `pp, request_id, routing_path, chunk_idx, span_name, span_kind, parent_span_id, span_id, inclusive_us, exclusive_us`.
- Summary CSV may remain unchanged unless a per-chunk diagnostic mode is later added.
- `roi_ranking.py` must tolerate and preserve the extra input column but continues to rank by per-request sum of same-name spans.

---

## § 6 Validation gates (P5h+1)

Inherits P5h spec § 4 gates + adds:
- **§ 6.1** Rust hygiene for any Rust code change: `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, `cargo build --release`.
- **§ 6.2** Production/default parity: flag OFF path performs no new forced eval. Run `cargo build --release -p ironmlx` plus the existing smoke benchmark (`p5_qwen35_moe_smoke` or successor) and require output match + pp_tps within ±2%.
- **§ 6.3** Measurement mode capture: T5 capture harness spawns `ironmlx serve --p5h-measurement-eval-probes` under `--features p5h-profile`; resulting data is labeled measurement-probe data and used only for attribution ranking.
- **§ 6.4** Measurement mode cost documentation: forced probes may degrade pp_tps by design. Document magnitude in P5h+1 close-out; do not use the degraded pp_tps as production performance evidence.
- **§ 6.5** Feasibility denominator discipline: `roi_ranking.py` / close-out must not interpret measurement-probe pp_tps as production pp_tps. Candidate ordering may use probe shares; target feasibility must cite flag-OFF production root medians separately.
- **§ 6.6** `chunk_idx` schema validation: every Lane B chunked-context span MUST emit `chunk_idx ≥ 0`; every Lane A or non-chunk Lane B span MUST emit `chunk_idx=null`. schema_validator enforces ancestor equality.
- **§ 6.7** Lane B coverage: still ≥ 95% per PP (spec § 7.1 unchanged); new substeps must not break coverage gate.
- **§ 6.8** Aggregator regression: 76 existing pytest pass + new pytest for chunk_idx parsing, chunk ancestor validation, Lane B allow-list expansion, attribution CSV extra column, and wrapper detection renamed from `first_token_sampling` to `first_token_sampling_materialize_and_sample`.

---

## § 7 Out of scope (deferred to P5h+2 or later)

Per Codex Q-P5h+1-1 Option A — explicitly NOT in P5h+1:
- Emit cost reduction (T0a 95% gate via buffered/binary emit) — independent infrastructure work; doesn't gate P5i/P5j dispatch.
- T0b H4 same-mode control (Phase A × 2) — data-confirmation only; no downstream gate.
- T2/T3 op-level ablation — data-dependent; trigger condition determined by P5h+1 re-run T5 ranking outcome.
- T4.2 mid-admit P5h ctx plumbing — low priority, doesn't affect Lane A primary path measurement.
- Any optimization candidate implementation (P5i / P5j scope).

These items remain in `project_p5h_findings.md` "P5h+1 follow-up" section; T5 close-out doc references them. P5h+1 close-out re-references them as "deferred to P5h+2" if and when triggered.

---

## § 8 Sequencing — P5i / P5j wait for P5h+1

Per Codex Q-P5h+1-4 sequential decision:
- P5h+1 closes first; P5i / P5j cannot dispatch until P5h+1 Close Gate satisfied.
- Reason: T5 measure-only verdict `data_insufficient` means current candidate ranking is structurally unreliable. Parallel P5i would re-use P5g-era hotspot intuition (in_proj_qkvz Linear 4-bit quant matmul per `project_p5g_findings.md`) — viable as a research spike but NOT as official P5i implementation basis.
- After P5h+1 close, P5i + P5j dispatch on re-ranked actionable candidates.

---

## § 9 References

- P5h spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`
- P5h plan: `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md`
- P5h close-out docs: `docs/p5h-t0b-close-out.md`, `docs/p5h-t2-layer3-binding.md`, `docs/p5h-t3-layer3-binding.md`, `docs/p5h-t4-close-out.md`, `docs/p5h-t5-close-out.md`
- P5h Codex reviews (all gitignored): `reports/p5h-t0b-codex-review-questions.md`, `reports/p5h-t1-codex-review-questions.md`, `reports/p5h-t2-codex-review-questions.md`, `reports/p5h-t4-codex-review-questions.md`, `reports/p5h-t5-codex-review-questions.md`, `reports/p5h-t5-results-codex-review.md`, `reports/p5h+1-codex-review-questions.md`
- Memory: `[project_p5h_findings]`, `[project_p5h_t0b_findings]`, `[project_p5h_t1_findings]`, `[project_p5h_t2_findings]`, `[project_p5h_t3_findings]`, `[project_p5h_t4_findings]`, `[project_p5g_findings]`, `[project_p5h_emit_cost_followup]`
- Reusable Rust: `ironmlx/src/core/p5h.rs` + `ironmlx/tests/p5h_common/mod.rs`
- Reusable Python: `tools/p5h_aggregator/{aggregator,schema_validator,roi_ranking}.py`
- Codex P5h+1 review (gitignored): `reports/p5h+1-codex-review-questions.md`
- Boss memory: `[feedback_task_breakdown_bounded]`, `[feedback_no_empty_commits]`, `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[feedback_iron_bench_priority]`
