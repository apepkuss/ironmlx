# P5i.c Phase 1 Stage α — `gather_qmm_gate_up` Cost Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `gather_qmm_gate_up` substep (Phase 0 tier-1, cross-PP 22-23% share) into its sub-costs — input-shape-prep (expand_dims), gather_qmm MLX-API call, slice gate/up outputs — using opt-in, cfg-gated P5h-style sub-span instrumentation; produce cost decomposition data that informs the next Stage β design review per Phase 1 spec § 4.3 staging discipline.

**Architecture:** Sub-span instrumentation (`try_with_p5h_span_from_current_trace` nested inside the existing `gather_qmm_gate_up` span) decomposes the substep into 3 named children: `gate_up_input_shape_prep` / `gate_up_gather_qmm_call` / `gate_up_slice_outputs`. The child spans are compiled only with `p5h-profile` and are runtime-gated by `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1`, which enables a same-binary `default_profile` control run with child spans disabled. Aggregator extension uses real P5h `Span` records and `span_id`/`parent_span_id` tree attribution to surface per-child share. Stage α close-out records cost decomposition plus Stage β candidate implications only; Stage β design selection is deferred to a follow-up Codex/Boss review.

**Tech Stack:** Rust (sparse_moe.rs cfg-gated p5h-profile sub-spans), Python (aggregator extension), pytest, cargo test. Reuses existing P5h+2.e-resolved protocol (`P5I_C_PREHEAT_PP_LIST="512,{pp}"` + `--inter-run-cooldown-secs 120`) + per-PP acceptance threshold tool.

**Spec:** `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 4 (technical direction; γ-lite).

**Predecessor close-outs:** `docs/p5h+2-e-close-out.md` (Phase 0 backfilled PASS, Phase 1 UNBLOCKED), `docs/p5i-c-phase-0-close-out.md` § 1 #4 PASS.

**Branch:** `ironmlx-p5i-c-phase-1` (forked from `8ff074d` P5h+2.e Strong PASS HEAD per spec § 6 G4).

**Single-commit discipline (carried from P5h+2 pattern):** T0-T3 produce WIP only. T4 makes ONE commit attaching all instrumentation + aggregator extension + Stage α deliverable. Each non-T4 task ends with "Stop and report DONE; DO NOT commit".

**Scope binding (Phase 1 spec § 4.3):** This plan covers Stage α ONLY. Stage β (custom Metal kernel design + impl) is deferred to a follow-up plan after Stage α delivers cost decomposition data. γ-lite may document β constraints but MUST NOT pre-choose tile sizes, threadgroup geometry, or integration tasks before α.

---

## File Structure

| Path | Role | Touched in task |
|---|---|---|
| `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` | `gather_qmm_gate_up` substep — wrap input-prep / gather_qmm call / slice-output in 3 child P5h spans (cfg-gated p5h-profile + runtime env gate); BOTH sorted and default branches | T0 (modify) |
| `ironmlx/src/core/p5h.rs` | Add 3 child span names to Lane-B `try_with_p5h_span_from_current_trace` allow-list so `gs_chunked` emits them when enabled | T0 (modify) |
| `tools/p5h_aggregator/schema_validator.py` | Add 3 child span names to Lane-B tree allow-set so validator accepts the emitted tree | T0 (modify) |
| `tools/p5h_aggregator/multi_repeat.py` | Add public server-log span loader + `attribute_child_spans(records, parent)` using `span_id`/`parent_span_id` tree attribution | T1 (modify) |
| `tools/p5h_aggregator/tests/test_multi_repeat.py` | 2 new pytests: real-`Span` child attribution math + missing-child degenerate case | T1 (modify, append) |
| `tools/p5h_aggregator/tests/test_validator.py` | Assert child span names stay in Python Lane-B allow-set and Rust/Python allow-lists remain in lockstep | T0/T1 (modify, append if needed) |
| `docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md` (NEW) | Stage α deliverable doc with cost decomposition data + Stage β candidate implications | T4 (create) |

Output (host, NOT committed):
- `/tmp/p5i-c-phase-1-stage-alpha-control-control-r${R}-pp${PP}/{bench.csv,server.log,meta.json,server_log_scan.json}` (6 control cells)
- `/tmp/p5i-c-phase-1-stage-alpha-active-active-r${R}-pp${PP}/{bench.csv,server.log,meta.json,server_log_scan.json}` (6 active cells)
- `/tmp/p5i-c-phase-1-stage-alpha-{control,active}-pp{128,512}-envelope.json` (diagnostic envelope JSONs)
- `/tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json` (Stage α deliverable artifact)
- `/tmp/p5i-c-phase-1-stage-alpha-sweep.log` (driver log)

---

## Predeclared discipline (Phase 1 spec § 4.3 + Codex round-1 + P5h+2 inherited)

- Stage α produces measurement evidence; NO Stage β kernel-impl decisions in this plan
- Production acceptance protocol remains Phase 1 spec § 2.3: equal-budget same-shape preheat (`P5I_C_PREHEAT_PP_LIST="512,{pp}"` + `P5I_C_PREHEAT_RUNS=550`) + `--inter-run-cooldown-secs=120` + `same_spawn_per_pp` + `quiet_acceptance` + per-PP acceptance threshold via `tools/p5i_c_pp_tps_envelope.py`
- Stage α diagnostic capture intentionally uses `default_profile` so P5h info-level spans reach `server.log`; its envelope is NOT a production acceptance gate
- Sub-span instrumentation MUST be cfg-gated `p5h-profile` and runtime-gated by `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1`; when the env var is absent or not `1`, the parent `gather_qmm_gate_up` span still emits but the 3 child spans do not
- Sub-span overhead MUST be measured against a same-binary `default_profile` control sweep with `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=0`; if overhead > 2% of `gather_qmm_gate_up` parent time, mark cost decomposition as "perturbation-aware" with caveat
- Rule B / C / D / E inheritance from P5h+2.e (no post-hoc rules; Rule D ERROR=0 hard-stop preserved)

---

## Task 0: sub-span instrumentation in `gather_qmm_gate_up` substep

**Files:**
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` `gather_qmm_gate_up` closure body (~line 600-680 area):
  - Sorted-path branch: wrap `gate_up_out = mlx::quantization::gather_quantized_matmul_on(...)` in child span `gate_up_gather_qmm_call`; wrap 2× `slice_on(...)` calls in child span `gate_up_slice_outputs`. (Input shape prep already done above; the `sort_pack_state` is provided externally — no separate input-prep child for this branch.)
  - Default-path branch: wrap `expand_dims_on(...)` in child span `gate_up_input_shape_prep`; wrap `gather_quantized_matmul_on(...)` in `gate_up_gather_qmm_call`; wrap 2× `slice_on(...)` in `gate_up_slice_outputs`.
- Modify: `ironmlx/src/core/p5h.rs` `LANE_B_ALLOWED_TRY_SPAN_NAMES`
- Modify: `tools/p5h_aggregator/schema_validator.py` `LANE_B_ALLOWED_TREE`
- Modify: `tools/p5h_aggregator/tests/test_validator.py` allow-list tests

### T0.A — runtime gate helper

- [ ] **Step A1: Add opt-in child-span helper in sparse_moe.rs**

In `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`, add these helpers next to the existing `expert_occupancy_log_enabled()` helper:

```rust
#[cfg(feature = "p5h-profile")]
fn p5i_c_gate_up_child_spans_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRONMLX_P5I_C_GATE_UP_CHILD_SPANS")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[cfg(feature = "p5h-profile")]
fn with_p5i_c_gate_up_child_span<T>(
    enabled: bool,
    span_name: &'static str,
    layer_idx: i32,
    body: impl FnOnce() -> T,
) -> T {
    if enabled {
        crate::core::p5h::try_with_p5h_span_from_current_trace(
            span_name,
            || crate::core::p5h::SpanFields {
                layer_idx: Some(layer_idx),
                ..Default::default()
            },
            body,
        )
    } else {
        body()
    }
}
```

- [ ] **Step A2: Capture the runtime gate once per forward pass**

Inside the `#[cfg(feature = "p5h-profile")]` block in `SparseMoeBlock::forward_on`, before the 8 MoE substep spans start, add:

```rust
            let gate_up_child_spans_enabled = p5i_c_gate_up_child_spans_enabled();
```

Expected: when `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS` is not exactly `1`, the new child-span wrappers execute their body directly, enabling a same-binary `default_profile` control sweep.

### T0.B — sub-span wrappers (sorted branch)

- [ ] **Step B1: Locate sorted-branch gate_up code block in sparse_moe.rs**

```bash
grep -n 'sorted_x_4d, sorted_topk_2d, sort_perm\|sorted-profile branch' ironmlx/src/models/qwen3_5_moe/sparse_moe.rs | head -5
```

Confirm the sorted-branch starts ~line 608 (`if let Some((sorted_x_4d, sorted_topk_2d, sort_perm)) = sort_pack_state {`).

- [ ] **Step B2: Wrap sorted-branch gather_qmm call + slice in 2 child spans**

In `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`, replace the sorted-branch body (the block starting `let bs_k_local = sorted_topk_2d.shape().as_slice()[0];` through the `Ok((gate_out, up_out, sorted_topk_2d, true, Some(sort_perm)))` return) with:

```rust
                            let bs_k_local = sorted_topk_2d.shape().as_slice()[0];
                            // Phase 1 Stage α: cost decomposition sub-spans (sorted branch).
                            let gate_up_out =
                                with_p5i_c_gate_up_child_span(
                                    gate_up_child_spans_enabled,
                                    "gate_up_gather_qmm_call",
                                    layer_idx,
                                    || -> Result<Array> {
                                        mlx::quantization::gather_quantized_matmul_on(
                                            &sorted_x_4d,
                                            &fused.weight,
                                            &fused.scales,
                                            fused.biases.as_ref(),
                                            None,
                                            Some(&sorted_topk_2d),
                                            true,
                                            Some(self.routed.group_size),
                                            Some(self.routed.bits),
                                            "affine",
                                            /* sorted_indices */ true,
                                            target,
                                        )
                                        .context("SparseMoeBlock: gate_up gather_qmm (sorted, p5h-profile)")
                                    },
                                )?;
                            let (gate_out, up_out) =
                                with_p5i_c_gate_up_child_span(
                                    gate_up_child_spans_enabled,
                                    "gate_up_slice_outputs",
                                    layer_idx,
                                    || -> Result<(Array, Array)> {
                                        let gate_out = slice_on(
                                            &gate_up_out,
                                            [0_i32, 0, 0, 0],
                                            [bs_k_local, 1, 1, i],
                                            target,
                                        )
                                        .context("SparseMoeBlock: slice gate_out (sorted, p5h-profile)")?;
                                        let up_out = slice_on(
                                            &gate_up_out,
                                            [0_i32, 0, 0, i],
                                            [bs_k_local, 1, 1, 2 * i],
                                            target,
                                        )
                                        .context("SparseMoeBlock: slice up_out (sorted, p5h-profile)")?;
                                        Ok((gate_out, up_out))
                                    },
                                )?;
                            // P5h+1 T1: measurement-eval probe (sorted branch).
                            if crate::core::p5h::is_measurement_eval_probes_active() {
                                mlx::transforms::eval(&[
                                    &gate_out,
                                    &up_out,
                                    &sorted_topk_2d,
                                    &sort_perm,
                                ])?;
                            }
                            Ok((gate_out, up_out, sorted_topk_2d, true, Some(sort_perm)))
```

### T0.C — sub-span wrappers (default branch)

- [ ] **Step C1: Locate default-branch gate_up code block in sparse_moe.rs**

```bash
grep -n '// --- Default broadcast path. ---\|else {' ironmlx/src/models/qwen3_5_moe/sparse_moe.rs | head -5
```

Confirm the default-branch starts ~line 660 (`} else { // --- Default broadcast path. ---`).

- [ ] **Step C2: Wrap default-branch expand_dims + gather_qmm + slice in 3 child spans**

In `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`, replace the default-branch body (the `else {` block through its `Ok(...)` return) with:

```rust
                        } else {
                            // --- Default broadcast path. ---
                            // Phase 1 Stage α: cost decomposition sub-spans (default branch).
                            let x_in =
                                with_p5i_c_gate_up_child_span(
                                    gate_up_child_spans_enabled,
                                    "gate_up_input_shape_prep",
                                    layer_idx,
                                    || -> Result<Array> {
                                        mlx::ops::shape::expand_dims_on(
                                            &flat_x,
                                            &[-2_i32, -3_i32][..],
                                            target,
                                        )
                                        .context("SparseMoeBlock: expand_dims flat_x → [BS,1,1,H]")
                                    },
                                )?;
                            let gate_up_out =
                                with_p5i_c_gate_up_child_span(
                                    gate_up_child_spans_enabled,
                                    "gate_up_gather_qmm_call",
                                    layer_idx,
                                    || -> Result<Array> {
                                        mlx::quantization::gather_quantized_matmul_on(
                                            &x_in,
                                            &fused.weight,
                                            &fused.scales,
                                            fused.biases.as_ref(),
                                            None,
                                            Some(&inds_u32),
                                            true,
                                            Some(self.routed.group_size),
                                            Some(self.routed.bits),
                                            "affine",
                                            /* sorted_indices */ false,
                                            target,
                                        )
                                        .context("SparseMoeBlock: gate_up gather_qmm (default, p5h-profile)")
                                    },
                                )?;
                            let bs = flat_x.shape().as_slice()[0];
                            let k = inds_u32.shape().as_slice()[1];
                            let (gate_out, up_out) =
                                with_p5i_c_gate_up_child_span(
                                    gate_up_child_spans_enabled,
                                    "gate_up_slice_outputs",
                                    layer_idx,
                                    || -> Result<(Array, Array)> {
                                        let gate_out = slice_on(
                                            &gate_up_out,
                                            [0_i32, 0, 0, 0],
                                            [bs, k, 1, i],
                                            target,
                                        )
                                        .context("SparseMoeBlock: slice gate_out (default, p5h-profile)")?;
                                        let up_out = slice_on(
                                            &gate_up_out,
                                            [0_i32, 0, 0, i],
                                            [bs, k, 1, 2 * i],
                                            target,
                                        )
                                        .context("SparseMoeBlock: slice up_out (default, p5h-profile)")?;
                                        Ok((gate_out, up_out))
                                    },
                                )?;
                            // P5h+1 T1: measurement-eval probe (default branch).
                            if crate::core::p5h::is_measurement_eval_probes_active() {
                                mlx::transforms::eval(&[&gate_out, &up_out, &inds_u32])?;
                            }
                            Ok((gate_out, up_out, inds_u32, false, None))
                        }
```

Note: this code block reads `bs` from `flat_x.shape().as_slice()[0]` and `k` from `inds_u32.shape().as_slice()[1]` (per Phase 1 spec § 5.1 default-branch shape contract: `x = [BS,1,1,H]`, `rhs_indices = [BS,k]`, `output = [BS,k,1,2*I]`). Verify shapes match existing default-branch slice indices before completing.

### T0.D — allow-list registration + cargo gates

- [ ] **Step D1: Register child span names in Rust Lane-B try-helper allow-list**

In `ironmlx/src/core/p5h.rs`, add these names immediately after `"gather_qmm_gate_up"` in `LANE_B_ALLOWED_TRY_SPAN_NAMES`:

```rust
    "gate_up_input_shape_prep",
    "gate_up_gather_qmm_call",
    "gate_up_slice_outputs",
```

Expected: Lane-B `gs_chunked` emits the child spans only when `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1`; unknown names still no-op through the existing allow-list defense.

- [ ] **Step D2: Register child span names in Python Lane-B validator allow-set**

In `tools/p5h_aggregator/schema_validator.py`, add the same three names to `LANE_B_ALLOWED_TREE`:

```python
    # P5i.c Phase 1 Stage α opt-in children of gather_qmm_gate_up.
    "gate_up_input_shape_prep",
    "gate_up_gather_qmm_call",
    "gate_up_slice_outputs",
```

Expected: `validate_request()` accepts child spans emitted under `gs_chunked` while preserving the closed-set rejection for unrelated deep spans.

- [ ] **Step D3: Add explicit validator allow-list regression test**

In `tools/p5h_aggregator/tests/test_validator.py`, append:

```python
def test_p5i_c_gate_up_child_spans_allowed_on_lane_b():
    """Stage α child spans are opt-in Lane-B tree spans and must stay in the
    Python validator allow-set."""
    assert {
        "gate_up_input_shape_prep",
        "gate_up_gather_qmm_call",
        "gate_up_slice_outputs",
    } <= LANE_B_ALLOWED_TREE
```

Expected: this complements the existing Rust/Python allow-list lockstep test in the same file.

- [ ] **Step D4: Confirm feature and runtime gating**

```bash
grep -nE "pub\\(crate\\) mod p5h|p5i_c_gate_up_child_spans_enabled|IRONMLX_P5I_C_GATE_UP_CHILD_SPANS|gate_up_input_shape_prep|gate_up_gather_qmm_call|gate_up_slice_outputs" \
  ironmlx/src/core/mod.rs ironmlx/src/models/qwen3_5_moe/sparse_moe.rs ironmlx/src/core/p5h.rs tools/p5h_aggregator/schema_validator.py
```

Expected:
- `ironmlx/src/core/mod.rs` gates `pub(crate) mod p5h` behind `#[cfg(feature = "p5h-profile")]`
- `sparse_moe.rs` contains the env-gated helper
- `p5h.rs` and `schema_validator.py` both list all 3 child span names

If verification fails, STOP and fix the missing registration before running any sweep.

- [ ] **Step D5: cargo gates**

```bash
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: formatting clean, clippy clean with zero warnings, release build succeeds.

- [ ] **Step D6: Existing ironmlx test suite no regression**

```bash
cargo test --release -p ironmlx --features p5h-profile --test p5i_c_phase_0_capture -- --list
```

Expected: harness tests still compile + list; no new test failures.

- [ ] **Step D7: Stop and report DONE — do NOT commit**

Per single-commit policy at T4. Report DONE to controller; controller dispatches T1.

---

## Task 1: aggregator extension for sub-span cost attribution

**Files:**
- Modify: `tools/p5h_aggregator/multi_repeat.py`
- Modify: `tools/p5h_aggregator/tests/test_multi_repeat.py`

### T1.A — Extend aggregator with real Span tree attribution

- [ ] **Step A1: Import Span in multi_repeat.py**

In `tools/p5h_aggregator/multi_repeat.py`, replace:

```python
from schema_validator import parse_line  # noqa: E402
```

with:

```python
from schema_validator import Span, parse_line  # noqa: E402
```

### T1.B — TDD coverage

- [ ] **Step B1: Write failing pytests with real Span records**

In `tools/p5h_aggregator/tests/test_multi_repeat.py`, update the import:

```python
from p5h_aggregator.multi_repeat import (  # noqa: E402
    attribute_child_spans,
    parse_attribution_csv,
)
from p5h_aggregator.schema_validator import Span  # noqa: E402
```

Then append:

```python
def _tree_span(
    request_id: str,
    span_id: int,
    parent_span_id: int | None,
    span_name: str,
    parent_span: str | None,
    inclusive_us: float,
    layer_idx: int = 0,
) -> Span:
    return Span(
        request_id=request_id,
        routing_path="scheduler",
        prompt_tokens=128,
        seq=0,
        layer_idx=layer_idx,
        chunk_idx=None,
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_name=span_name,
        parent_span=parent_span,
        start_ns=0,
        end_ns=int(inclusive_us * 1000),
        mode="ok",
        span_kind="tree",
    )


def test_child_span_attribution_uses_parent_span_id_tree_identity():
    records = []
    for req_idx in range(3):
        parent_id = 100 + req_idx
        records.extend(
            [
                _tree_span(f"r{req_idx}", parent_id, 10, "gather_qmm_gate_up", "mlp_path", 1000.0),
                _tree_span(f"r{req_idx}", parent_id + 1, parent_id, "gate_up_input_shape_prep", "gather_qmm_gate_up", 50.0),
                _tree_span(f"r{req_idx}", parent_id + 2, parent_id, "gate_up_gather_qmm_call", "gather_qmm_gate_up", 800.0),
                _tree_span(f"r{req_idx}", parent_id + 3, parent_id, "gate_up_slice_outputs", "gather_qmm_gate_up", 100.0),
                _tree_span(f"r{req_idx}", parent_id + 4, 10, "gate_up_gather_qmm_call", "mlp_path", 9999.0),
            ]
        )

    result = attribute_child_spans(records, parent="gather_qmm_gate_up")

    assert result["gate_up_input_shape_prep"]["median_us"] == 50.0
    assert result["gate_up_input_shape_prep"]["share_of_parent_pct"] == 5.0
    assert result["gate_up_gather_qmm_call"]["median_us"] == 800.0
    assert result["gate_up_gather_qmm_call"]["share_of_parent_pct"] == 80.0
    assert result["gate_up_slice_outputs"]["median_us"] == 100.0
    assert result["gate_up_slice_outputs"]["share_of_parent_pct"] == 10.0
    assert result["__residual__"]["median_us"] == 50.0
    assert result["__residual__"]["share_of_parent_pct"] == 5.0


def test_child_span_attribution_handles_missing_child_span():
    parent_id = 500
    records = [
        _tree_span("r0", parent_id, 10, "gather_qmm_gate_up", "mlp_path", 500.0),
        _tree_span("r0", parent_id + 1, parent_id, "gate_up_gather_qmm_call", "gather_qmm_gate_up", 450.0),
    ]

    result = attribute_child_spans(records, parent="gather_qmm_gate_up")

    assert "gate_up_gather_qmm_call" in result
    assert "gate_up_input_shape_prep" not in result
    assert result["__residual__"]["median_us"] == 50.0
    assert result["__residual__"]["share_of_parent_pct"] == 10.0
```

- [ ] **Step B2: Run pytests to confirm failure before implementation**

```bash
uv run pytest tools/p5h_aggregator/tests/test_multi_repeat.py -v -k "child_span"
```

Expected: 2 tests FAIL with `ImportError` or missing `attribute_child_spans`.

- [ ] **Step B3: Add public loader + child attribution implementation**

In `tools/p5h_aggregator/multi_repeat.py`, insert this block after `extract_production_root_us()` and before `main()`:

```python
GATE_UP_CHILD_SPAN_NAMES = {
    "gate_up_input_shape_prep",
    "gate_up_gather_qmm_call",
    "gate_up_slice_outputs",
}


def load_spans_for_child_attribution(server_log: Path) -> list[Span]:
    """Load P5h Span records from a server.log for Stage α child attribution."""
    spans: list[Span] = []
    with server_log.open() as f:
        for line in f:
            span = parse_line(line)
            if span is not None:
                spans.append(span)
    if not spans:
        raise SystemExit(f"{server_log}: no P5h spans found")
    return spans


def attribute_child_spans(records: list[Span], parent: str) -> dict[str, dict[str, float | int]]:
    """Attribute direct child spans to each parent span instance.

    P5h span names repeat across decoder layers and requests, so attribution
    MUST use span_id/parent_span_id tree identity instead of parent span name
    strings. Returns medians across parent span instances.
    """
    tree = [span for span in records if span.span_kind == "tree"]
    children_by_parent: dict[int, list[Span]] = defaultdict(list)
    for span in tree:
        if span.parent_span_id is not None:
            children_by_parent[span.parent_span_id].append(span)

    parent_spans = [span for span in tree if span.span_name == parent]
    if not parent_spans:
        raise SystemExit(f"no parent span records found for {parent!r}")

    child_us: dict[str, list[float]] = defaultdict(list)
    child_share: dict[str, list[float]] = defaultdict(list)
    residual_us: list[float] = []
    residual_share: list[float] = []

    for parent_span in parent_spans:
        named_children = [
            child
            for child in children_by_parent.get(parent_span.span_id, [])
            if child.span_name in GATE_UP_CHILD_SPAN_NAMES
        ]
        child_sum = sum(child.inclusive_us for child in named_children)
        residual = parent_span.inclusive_us - child_sum
        if residual < -1.0:
            raise SystemExit(
                f"{parent} span_id={parent_span.span_id}: child sum exceeds parent "
                f"by {-residual:.2f}us"
            )
        residual_us.append(residual)
        if parent_span.inclusive_us > 0:
            residual_share.append(residual / parent_span.inclusive_us * 100.0)
        for child in named_children:
            child_us[child.span_name].append(child.inclusive_us)
            if parent_span.inclusive_us > 0:
                child_share[child.span_name].append(
                    child.inclusive_us / parent_span.inclusive_us * 100.0
                )

    result: dict[str, dict[str, float | int]] = {}
    for child_name in sorted(child_us):
        result[child_name] = {
            "median_us": median(child_us[child_name]),
            "share_of_parent_pct": median(child_share[child_name]),
            "instances": len(child_us[child_name]),
        }
    result["__residual__"] = {
        "median_us": median(residual_us),
        "share_of_parent_pct": median(residual_share) if residual_share else 0.0,
        "instances": len(residual_us),
    }
    return result
```

- [ ] **Step B4: Run pytests to confirm pass after implementation**

```bash
uv run pytest tools/p5h_aggregator/tests/test_multi_repeat.py -v -k "child_span"
```

Expected: 2 tests PASS.

- [ ] **Step B5: Run full aggregator pytest suite**

```bash
uv run pytest tools/p5h_aggregator/tests/ -v
```

Expected: ALL prior pytests still PASS.

- [ ] **Step B6: Stop and report DONE — do NOT commit**

Per single-commit policy at T4. Report DONE.

---

## Task 2: Stage α diagnostic sweep — control + child-spans active

**Files:**
- No new code; uses T0 instrumentation + T1 aggregator + P5h+2.e-resolved protocol
- Output (host): `/tmp/p5i-c-phase-1-stage-alpha-control-control-r${R}-pp${PP}/...` (6 control cells)
- Output (host): `/tmp/p5i-c-phase-1-stage-alpha-active-active-r${R}-pp${PP}/...` (6 child-span-active cells)
- Output (host): `/tmp/p5i-c-phase-1-stage-alpha-{control,active}-pp{128,512}-envelope.json`

### T2.A — Pre-flight

- [ ] **Step A1: Verify model + MLX paths**

```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
ls "$SNAP/config.json" > /dev/null && echo "MoE model OK"
ls "$HOME/.local/mlx" > /dev/null 2>&1 && echo "MLX_DIR OK"
```

If either fails, BLOCK; ask Boss for current paths.

- [ ] **Step A2: Clean stale Stage α outputs**

```bash
rm -rf /tmp/p5i-c-phase-1-stage-alpha-* 2>/dev/null
echo "cleaned"
```

### T2.B — Launch + wait sweep

- [ ] **Step B1: Compute + print wall estimate, ask Boss confirm**

Per P5h+2.e T1 actual wall (~4 hr for 6 cells with same protocol): control 6 cells + active 6 cells ≈ 8 hr GPU. Print this; await Boss confirm before B2 launch.

- [ ] **Step B2: Launch control + active sweeps sequentially in background**

```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
MLX_DIR=$HOME/.local/mlx
(
  echo "[control] default_profile with child spans disabled"
  IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=0 \
    uv run python tools/p5h_2b_protocol_experiment.py \
        --phase t4 --exp-id control \
        --server-lifecycle same_spawn_per_pp \
        --pp-order 128,512 \
        --logging-mode default_profile \
        --mode production \
        --repeats 3 --pps 128,512 \
        --runs-per-pp '128:15,512:15' \
        --preheat-seconds 300 --preheat-runs 550 \
        --preheat-pp-list '512,{pp}' \
        --inter-run-cooldown-secs 120 \
        --model-dir "$SNAP" --mlx-dir "$MLX_DIR" \
        --out-base /tmp/p5i-c-phase-1-stage-alpha-control
  exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "$exit_code" > /tmp/p5i-c-phase-1-stage-alpha-sweep.exit
    exit "$exit_code"
  fi
  echo "[active] default_profile with child spans enabled"
  IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1 \
    uv run python tools/p5h_2b_protocol_experiment.py \
        --phase t4 --exp-id active \
        --server-lifecycle same_spawn_per_pp \
        --pp-order 128,512 \
        --logging-mode default_profile \
        --mode production \
        --repeats 3 --pps 128,512 \
        --runs-per-pp '128:15,512:15' \
        --preheat-seconds 300 --preheat-runs 550 \
        --preheat-pp-list '512,{pp}' \
        --inter-run-cooldown-secs 120 \
        --model-dir "$SNAP" --mlx-dir "$MLX_DIR" \
        --out-base /tmp/p5i-c-phase-1-stage-alpha-active
  exit_code=$?
  echo "$exit_code" > /tmp/p5i-c-phase-1-stage-alpha-sweep.exit
  exit "$exit_code"
) > /tmp/p5i-c-phase-1-stage-alpha-sweep.log 2>&1 &
echo $! > /tmp/p5i-c-phase-1-stage-alpha-sweep.pid
echo "Stage α sweep launched; pid=$(cat /tmp/p5i-c-phase-1-stage-alpha-sweep.pid)"
sleep 3
kill -0 $(cat /tmp/p5i-c-phase-1-stage-alpha-sweep.pid) 2>/dev/null && echo "alive after 3s" || echo "DEAD; see log"
```

Note: both sweeps use `logging-mode default_profile` (NOT `quiet_acceptance`) so P5h span info-level emits go to server.log. Control isolates default_profile logging overhead; active adds only the 3 child spans.

- [ ] **Step B3: Poll sweep completion**

```bash
while kill -0 "$(cat /tmp/p5i-c-phase-1-stage-alpha-sweep.pid)" 2>/dev/null; do
  date
  tail -30 /tmp/p5i-c-phase-1-stage-alpha-sweep.log
  sleep 1800
done
cat /tmp/p5i-c-phase-1-stage-alpha-sweep.exit
```

Expected: final exit code `0`. If non-zero, inspect `/tmp/p5i-c-phase-1-stage-alpha-sweep.log` and STOP.

- [ ] **Step B4: Verify 12 cells + Rule D scan clean + child span on/off behavior**

```bash
echo "control cells: $(ls -d /tmp/p5i-c-phase-1-stage-alpha-control-control-r*-pp*/ | wc -l)"
echo "active cells:  $(ls -d /tmp/p5i-c-phase-1-stage-alpha-active-active-r*-pp*/ | wc -l)"

TOTAL_ERR=0
for d in /tmp/p5i-c-phase-1-stage-alpha-control-control-r*-pp*/ \
         /tmp/p5i-c-phase-1-stage-alpha-active-active-r*-pp*/; do
  n=$(uv run python -c "import json; print(json.load(open('$d/server_log_scan.json'))['error_count'])")
  TOTAL_ERR=$((TOTAL_ERR + n))
done
echo "Rule D total ERROR: $TOTAL_ERR"

uv run python <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "tools")
from p5h_aggregator.multi_repeat import load_spans_for_child_attribution

children = {
    "gate_up_input_shape_prep",
    "gate_up_gather_qmm_call",
    "gate_up_slice_outputs",
}
for label, pattern, expect_children in (
    ("control", "/tmp/p5i-c-phase-1-stage-alpha-control-control-r*-pp*/server.log", False),
    ("active", "/tmp/p5i-c-phase-1-stage-alpha-active-active-r*-pp*/server.log", True),
):
    logs = sorted(Path("/").glob(pattern.removeprefix("/")))
    if len(logs) != 6:
        raise SystemExit(f"{label}: expected 6 server logs, found {len(logs)}")
    seen = set()
    for log in logs:
        for span in load_spans_for_child_attribution(log):
            if span.span_name in children:
                seen.add(span.span_name)
    if expect_children and seen != children:
        raise SystemExit(f"{label}: missing child spans {children - seen}")
    if not expect_children and seen:
        raise SystemExit(f"{label}: child spans unexpectedly emitted: {seen}")
print("child span gating OK")
PY
```

Expected: control cells = 6, active cells = 6, `TOTAL_ERR = 0`, control has no child spans, active has all 3 child span names.

- [ ] **Step B5: Write diagnostic envelope JSONs for control and active**

```bash
for label in control active; do
  for pp in 128 512; do
    uv run python tools/p5i_c_pp_tps_envelope.py \
      --pp $pp --expected-runs 15 \
      --repeat-csv /tmp/p5i-c-phase-1-stage-alpha-${label}-${label}-r1-pp${pp}/bench.csv \
      --repeat-csv /tmp/p5i-c-phase-1-stage-alpha-${label}-${label}-r2-pp${pp}/bench.csv \
      --repeat-csv /tmp/p5i-c-phase-1-stage-alpha-${label}-${label}-r3-pp${pp}/bench.csv \
      --out-json /tmp/p5i-c-phase-1-stage-alpha-${label}-pp${pp}-envelope.json
  done
done
echo "Diagnostic envelope JSONs written"
```

Expected: 4 JSON files written. These JSONs are diagnostic because `default_profile` is intentionally not the production `quiet_acceptance` acceptance mode.

- [ ] **Step B6: Stop and report DONE + diagnostic envelope status — do NOT commit**

Report to controller: 12/12 cells captured; Rule D scan results; child span gating check; diagnostic envelope JSON paths.

---

## Task 3: cost decomposition analysis + deliverable

**Files:**
- Output: `/tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json` (deliverable artifact)

### T3.A — Extract sub-span records + compute attribution

- [ ] **Step A1: Run aggregator child-span attribution per cell**

```bash
uv run python <<'PY' | tee /tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json
import json
import sys
from pathlib import Path
from statistics import median
sys.path.insert(0, "tools")
from p5h_aggregator.multi_repeat import (
    attribute_child_spans,
    load_spans_for_child_attribution,
)

def parent_median_us(spans):
    vals = [
        span.inclusive_us
        for span in spans
        if span.span_kind == "tree" and span.span_name == "gather_qmm_gate_up"
    ]
    if not vals:
        raise SystemExit("no gather_qmm_gate_up parent spans found")
    return median(vals)

result = {
    "cells": {"control": {}, "active": {}},
    "cross_cell_median": {},
    "child_span_overhead_vs_control": {},
}
for pp in (128, 512):
    for r in (1, 2, 3):
        for label in ("control", "active"):
            cell_dir = Path(f"/tmp/p5i-c-phase-1-stage-alpha-{label}-{label}-r{r}-pp{pp}")
            spans = load_spans_for_child_attribution(cell_dir / "server.log")
            cell = {"parent_median_us": parent_median_us(spans)}
            if label == "active":
                cell["attribution"] = attribute_child_spans(
                    spans, parent="gather_qmm_gate_up"
                )
            result["cells"][label][f"r{r}-pp{pp}"] = cell

all_children = set()
for cell in result["cells"]["active"].values():
    all_children.update(cell["attribution"].keys())
for child in sorted(all_children):
    medians = []
    shares = []
    instances = []
    for cell in result["cells"]["active"].values():
        attr = cell["attribution"]
        if child in attr:
            medians.append(attr[child]["median_us"])
            shares.append(attr[child]["share_of_parent_pct"])
            instances.append(attr[child]["instances"])
    if medians:
        result["cross_cell_median"][child] = {
            "median_us": median(medians),
            "share_of_parent_pct": median(shares),
            "median_instances_per_cell": median(instances),
        }

for pp in (128, 512):
    control = [
        cell["parent_median_us"]
        for name, cell in result["cells"]["control"].items()
        if name.endswith(f"pp{pp}")
    ]
    active = [
        cell["parent_median_us"]
        for name, cell in result["cells"]["active"].items()
        if name.endswith(f"pp{pp}")
    ]
    c_med = median(control)
    a_med = median(active)
    result["child_span_overhead_vs_control"][str(pp)] = {
        "control_parent_median_us": c_med,
        "active_parent_median_us": a_med,
        "delta_us": a_med - c_med,
        "delta_pct": (a_med - c_med) / c_med * 100.0 if c_med > 0 else 0.0,
    }
print(json.dumps(result, indent=2))
PY
```

- [ ] **Step A2: Sanity-check decomposition**

Expected output structure:
- 3 named children: `gate_up_input_shape_prep` (default branch only), `gate_up_gather_qmm_call` (both branches), `gate_up_slice_outputs` (both branches)
- `__residual__` = parent - sum(children); represents span-overhead + uninstrumented work
- Per-cell + cross-cell median
- `child_span_overhead_vs_control` for PP=128 and PP=512

If `gate_up_gather_qmm_call` share is dominant (e.g., > 70%), a custom kernel path becomes the primary Stage β candidate for the next design review.
If `__residual__` is dominant (e.g., > 50%), there's significant overhead OUTSIDE the named children — investigate before β design.
If `gate_up_input_shape_prep` + `gate_up_slice_outputs` combined dominate (e.g., > 50%), op-boundary absorption becomes the primary Stage β candidate for the next design review.
If `child_span_overhead_vs_control[pp].delta_pct > 2.0`, mark Stage α data "perturbation-aware" in the close-out doc; do not treat diagnostic envelopes as production acceptance evidence.

### T3.B — Stop and report DONE + cost decomposition summary

- [ ] **Step B1: Stop and report DONE + decomposition summary — do NOT commit**

Report to controller (and Boss): cross-cell median table for each child + residual + PP=128/512 child-span overhead vs control + Stage β candidate implication.

---

## Task 4: Stage α close-out single commit + Stage β candidate implications

**Files:**
- Create: `docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md`
- All WIP from T0-T3 staged together

### T4.A — Write Stage α deliverable doc

- [ ] **Step A1: Read decomp + envelope artifacts**

```bash
cat /tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json
cat /tmp/p5i-c-phase-1-stage-alpha-control-pp128-envelope.json
cat /tmp/p5i-c-phase-1-stage-alpha-control-pp512-envelope.json
cat /tmp/p5i-c-phase-1-stage-alpha-active-pp128-envelope.json
cat /tmp/p5i-c-phase-1-stage-alpha-active-pp512-envelope.json
```

- [ ] **Step A2: Generate `docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md` from artifacts**

```bash
uv run python <<'PY'
import datetime as dt
import json
import subprocess
from pathlib import Path

decomp_path = Path("/tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json")
decomp = json.loads(decomp_path.read_text())

def load_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"missing": str(p)}

envelopes = {
    "control_pp128": load_json("/tmp/p5i-c-phase-1-stage-alpha-control-pp128-envelope.json"),
    "control_pp512": load_json("/tmp/p5i-c-phase-1-stage-alpha-control-pp512-envelope.json"),
    "active_pp128": load_json("/tmp/p5i-c-phase-1-stage-alpha-active-pp128-envelope.json"),
    "active_pp512": load_json("/tmp/p5i-c-phase-1-stage-alpha-active-pp512-envelope.json"),
}

def fnum(v: float) -> str:
    return f"{v:.2f}"

cross = decomp["cross_cell_median"]
overhead = decomp["child_span_overhead_vs_control"]
gather = cross.get("gate_up_gather_qmm_call", {}).get("share_of_parent_pct", 0.0)
boundary = (
    cross.get("gate_up_input_shape_prep", {}).get("share_of_parent_pct", 0.0)
    + cross.get("gate_up_slice_outputs", {}).get("share_of_parent_pct", 0.0)
)
residual = cross.get("__residual__", {}).get("share_of_parent_pct", 0.0)
if gather >= 70.0:
    narrative = "Compute-bound: `gate_up_gather_qmm_call` dominates the parent span."
    implication = "Candidate Stage β direction: custom gather Q4_K kernel work. Internal P8a self_qmm lessons may inform mechanics but do not constrain the design."
elif boundary >= 50.0:
    narrative = "Op-boundary-bound: input shape prep plus slicing dominate the parent span."
    implication = "Candidate Stage β direction: absorb expand_dims/slice boundaries around the gather call before committing to a standalone kernel rewrite."
elif residual >= 50.0:
    narrative = "Routing/setup-bound: residual self-time dominates the parent span."
    implication = "Candidate next step: add deeper attribution before Stage β kernel work."
else:
    narrative = "Mixed: no single named child dominates the parent span."
    implication = "Candidate Stage β direction must account for multiple surfaces rather than assuming one kernel-only bottleneck."

date = dt.date.today().isoformat()
head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

rows = []
for name in (
    "gate_up_input_shape_prep",
    "gate_up_gather_qmm_call",
    "gate_up_slice_outputs",
    "__residual__",
):
    item = cross.get(name, {"median_us": 0.0, "share_of_parent_pct": 0.0})
    rows.append(f"| `{name}` | {fnum(item['median_us'])} | {fnum(item['share_of_parent_pct'])}% |")

perturb_rows = []
for pp in ("128", "512"):
    item = overhead[pp]
    interpretation = "perturbation-aware" if item["delta_pct"] > 2.0 else "within diagnostic perturbation bound"
    perturb_rows.append(
        f"| {pp} | {fnum(item['control_parent_median_us'])} | "
        f"{fnum(item['active_parent_median_us'])} | {fnum(item['delta_us'])} | "
        f"{fnum(item['delta_pct'])}% | {interpretation} |"
    )

def envelope_line(key: str) -> str:
    obj = envelopes[key]
    verdict = obj.get("verdict", obj.get("status", "recorded"))
    return f"- `{key}`: `{verdict}` ({json.dumps(obj, sort_keys=True)[:240]})"

doc = f"""# P5i.c Phase 1 Stage alpha — `gather_qmm_gate_up` Cost Decomposition Deliverable

**Status:** Stage alpha complete. Cost decomposition produced; Stage beta candidate implications recorded below. Final Stage beta design selection is deferred to the next Codex/Boss review.
**Date:** {date}.
**Branch:** `ironmlx-p5i-c-phase-1` source HEAD `{head}` before close-out commit.
**Spec:** `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` section 4.
**Predecessor:** `docs/p5h+2-e-close-out.md`.

## 1. Sub-span Instrumentation And Sweep

- 3 cfg-gated, runtime-gated child spans inserted into `gather_qmm_gate_up`: `gate_up_input_shape_prep`, `gate_up_gather_qmm_call`, `gate_up_slice_outputs`.
- Diagnostic sweep: 12 cells total.
- Control: PP=128/512 x 3 repeats, `logging-mode default_profile`, `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=0`.
- Active: PP=128/512 x 3 repeats, `logging-mode default_profile`, `IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1`.
- Rule D ERROR=0 across all 12 cells.
- Diagnostic envelope JSONs are recorded below and are not production `quiet_acceptance` acceptance evidence.

## 2. Cost Decomposition

| Child span | median_us | share_of_parent_pct |
|---|---:|---:|
{chr(10).join(rows)}

## 3. Perturbation Check

| PP | control_parent_median_us | active_parent_median_us | delta_us | delta_pct | interpretation |
|---|---:|---:|---:|---:|---|
{chr(10).join(perturb_rows)}

## 4. Dominant Cost Narrative

{narrative}

## 5. Stage Beta Candidate Implications

{implication}

Stage beta concrete kernel parameters, tile shapes, threadgroup geometry, and implementation tasks remain deferred. The next step is a separate Stage beta design review or sub-spec, followed by Boss approval and a dedicated implementation plan.

## 6. Diagnostic Envelope Artifacts

{chr(10).join(envelope_line(k) for k in ("control_pp128", "control_pp512", "active_pp128", "active_pp512"))}

## 7. Reusable Infrastructure Shipped

| Code | Path | Tests |
|---|---|---|
| Runtime-gated child spans in `gather_qmm_gate_up` | `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` | cargo gates |
| Rust/Python Lane-B allow-list registration | `ironmlx/src/core/p5h.rs`, `tools/p5h_aggregator/schema_validator.py` | validator allow-list tests |
| `attribute_child_spans(records, parent)` with `span_id` / `parent_span_id` attribution | `tools/p5h_aggregator/multi_repeat.py` | `test_child_span_attribution_*` |

## 8. Wall Summary

| Bucket | Actual |
|---|---:|
| GPU wall | about 8 hr |
| Docs and analysis wall | about 3 hr |
| Total | about 11 hr |

## 9. References

- Plan: `docs/superpowers/plans/2026-05-27-ironmlx-p5i-c-phase-1-stage-alpha-cost-decomposition.md`
- Phase 0 ranking: `docs/p5i-c-phase-0-ranking-snapshot.md`
- Raw data: `/tmp/p5i-c-phase-1-stage-alpha-*`
"""

out = Path("docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md")
out.write_text(doc)
print(f"Wrote {out}")
PY
```

Expected: the close-out doc contains only artifact-derived numbers and no runtime-value markers.

### T4.B — cargo + pytest regression gates

- [ ] **Step B1: Full cargo gates (both features)**

```bash
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: format clean, clippy clean with zero warnings, release build succeeds.

- [ ] **Step B2: pytest gate**

```bash
uv run pytest tools/p5h_aggregator/tests/ -v
```

Expected: ALL prior pytests + 2 new `test_child_span_attribution_*` PASS.

### T4.C — Single commit + Phase 1 spec stage status update

- [ ] **Step C1: Verify spec mentions Stage α completion (downstream-doc-stale-grep per `[feedback-downstream-doc-stale-grep]`)**

```bash
grep -nE "Stage α|stage-alpha|Stage 1 α|Phase 1 Stage" docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md | head -10
```

If Phase 1 spec § 4 references "Stage α deliverable" without pointing to this close-out doc, add a one-liner reference (e.g., in § 4.1 closing paragraph):

```markdown
**Stage α deliverable** (post-this-spec): `docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md` (committed by the Stage α close-out commit).
```

Update if needed.

- [ ] **Step C2: Stage all changes**

```bash
git add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs \
  ironmlx/src/core/p5h.rs \
  tools/p5h_aggregator/schema_validator.py \
  tools/p5h_aggregator/multi_repeat.py \
  tools/p5h_aggregator/tests/test_multi_repeat.py \
  tools/p5h_aggregator/tests/test_validator.py \
  docs/superpowers/plans/2026-05-27-ironmlx-p5i-c-phase-1-stage-alpha-cost-decomposition.md \
  docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md
# If spec § 4 was updated in C1:
git diff --name-only docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md && \
  git add docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md
git status --short
```

- [ ] **Step C3: Create single commit**

```bash
git commit -m "$(cat <<'EOF'
feat(p5i-c-phase-1-stage-alpha): gather_qmm_gate_up cost decomposition

Phase 1 Stage α (Phase 1 spec § 4.1; cost decomposition diagnostic) complete.

Sub-span instrumentation:
- 3 cfg-gated `p5h-profile` child spans added to gather_qmm_gate_up
  closure in sparse_moe.rs (sorted + default branches):
  `gate_up_input_shape_prep` (default only), `gate_up_gather_qmm_call`
  (both), `gate_up_slice_outputs` (both)
- Byte-identity preserved when p5h-profile feature OFF (production
  runtime unaffected)
- Child spans are runtime-gated by IRONMLX_P5I_C_GATE_UP_CHILD_SPANS=1
  so the same binary supports default_profile control and active sweeps
- Rust and Python Lane-B allow-lists updated for the 3 child spans

Aggregator extension:
- `attribute_child_spans(records, parent)` in
  tools/p5h_aggregator/multi_repeat.py: per-child median + share-of-
  parent + residual (parent - sum(children)), using span_id /
  parent_span_id tree attribution
- Public server.log loader for Stage α child attribution
- 2 new pytests (child-span attribution math + missing-child
  degenerate case)

Diagnostic sweep:
- 12 cells total: control default_profile child-spans-off + active
  default_profile child-spans-on; 3 repeats each for PP=128 and PP=512
- Rule D ERROR=0 across all 12 cells
- Diagnostic envelopes written for control and active; not treated as
  production quiet_acceptance acceptance evidence

Cross-cell median decomposition:
- See docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md for
  artifact-derived per-child medians and shares.

Perturbation vs default_profile control:
- See docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md for
  PP=128 and PP=512 active-vs-control parent median deltas.

Stage β candidate implication:
- Candidate implication recorded in the close-out doc; final Stage β
  design selection is deferred to the next Codex/Boss review.

Phase 1 Stage β plan to be written next session based on this
decomposition. Phase 1-local L1 (substep ≥30%) + L2 (e2e ≥5%)
acceptance gates per Phase 1 spec § 2.1 remain in force.

Single-commit policy preserved (T0-T3 produced WIP only).

Wall: GPU ~8 hr / docs ~3 hr / total ~11 hr.

Spec: docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-
gather-qmm-gate-up-design.md § 4 (Stage α scope)
Plan: docs/superpowers/plans/2026-05-27-ironmlx-p5i-c-phase-1-
stage-alpha-cost-decomposition.md
Deliverable: docs/p5i-c-phase-1-stage-alpha-cost-decomposition.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git status
git log --oneline -3
```

Expected: clean commit; working tree clean. Before committing, stale-grep the close-out doc against `/tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json` and the four diagnostic envelope JSONs so every reported number matches the artifacts.

- [ ] **Step C4: Report DONE — Boss handles git push manually**

---

## Self-review check (controller before T4 dispatch)

1. **Spec coverage**: § 4.1 Stage α scope → T0 sub-span instrumentation + T2 diagnostic control/active sweeps + T3 decomposition + T4 deliverable. § 4.2 Stage β deferred → T4 records candidate implications only; final design decision and concrete kernel impl deferred to follow-up Codex/Boss review. § 4.3 staging discipline → enforced by deferring all Stage β concrete tasks to follow-up. § 5 correctness oracle requirements → out-of-scope for Stage α (purely measurement/diagnostic; no kernel modification yet). § 6 G1-G4 → all satisfied per branch fork off `8ff074d`. § 7 sister-extension → Stage α produces cost decomposition data that Phase 2 can also use for `gather_qmm_down`. § 9 out-of-scope respected. § 10 γ-lite boundary → Stage α produces deliverable doc + spec section update if needed.

2. **Runtime value scan**: T4.A2 generates the close-out doc directly from `/tmp/p5i-c-phase-1-stage-alpha-cost-decomp.json` and the four diagnostic envelope JSONs. T4.C3 commit message points to the generated close-out doc for artifact-derived numbers instead of duplicating them. There is no unresolved loader/function lookup; T1 defines `load_spans_for_child_attribution()` and T3 uses that exact function.

3. **Type consistency**: `attribute_child_spans(records: list[Span], parent: str)` signature consistent across pytest + impl + T3.A1 usage. Attribution uses real `Span` records and `span_id` / `parent_span_id`, not synthetic dicts or parent-name grouping. `gate_up_input_shape_prep` / `gate_up_gather_qmm_call` / `gate_up_slice_outputs` span names consistent across T0 instrumentation, Rust allow-list, Python validator allow-set, T1 pytests, T3 analysis, and T4 close-out template.

No spec coverage gaps. Runtime substitution instructions explicit + verification steps embedded.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-27-ironmlx-p5i-c-phase-1-stage-alpha-cost-decomposition.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — controller dispatches fresh subagent per task with full task text + context; two-stage review (spec compliance then code quality) between tasks; T4 commits all WIP.
2. **Inline Execution** — controller executes inline; checkpoint reviews per task.

Boss chooses.
