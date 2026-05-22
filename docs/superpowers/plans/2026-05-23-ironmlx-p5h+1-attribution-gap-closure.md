# P5h+1 Attribution Gap Closure Implementation Plan

> Required execution skill for implementation agents: use
> `superpowers:executing-plans` or `superpowers:subagent-driven-development`.
> Work task-by-task, run the verification listed for each task, and do not
> mark a task complete while its verification is still failing.

## Goal

Close the two structural attribution gaps that blocked P5h T5 candidate
ranking:

- Lane A: `first_token_sampling` dominates 96-99% of root time.
- Lane B: `gs_chunk_N` dominates 97-99% of root time.

After P5h+1, the T5 ranking must produce actionable non-wrapper P5i/P5j
candidates for every PP point. Probe-mode attribution may add overhead, but
production behavior must remain unchanged when `--p5h-measurement-eval-probes`
is off.

## Non-goals

- Do not change ROI ranking math.
- Do not tune kernels in this phase.
- Do not add parser compatibility for the old `[p5h-profile]` schema. Update
  emitters, validator, aggregator, and tests together.
- Do not use probe-mode wall time as the production denominator. Production
  feasibility uses a flag-off production root baseline.

## Verified Current Code Facts

These facts were checked against the current branch and should be preserved
during implementation:

- `ironmlx/src/core/p5h.rs::SpanFields` currently has:
  `layer_idx: Option<i32>`, `seq: Option<u32>`, `mode: Option<&'static str>`.
  Keep `seq` as `Option<u32>`.
- There is no explicit-context `with_p5h_span` helper. Use
  `open_p5h_span`, `open_p5h_span_at`, and `close_p5h_span` for scheduler
  sibling spans.
- The Lane B try-helper allow-list to edit is
  `LANE_B_ALLOWED_TRY_SPAN_NAMES`.
- The attribution CSV writer to edit is
  `tools/p5h_aggregator/aggregator.py::write_attribution_csv`.
- Current attribution CSV already has `routing_path`, `span_kind`,
  `parent_span_id`, `span_id`, `inclusive_us`, and `exclusive_us`. P5h+1 only
  adds `chunk_idx`.
- Current GDN span names are:
  `gda_step_1a_in_proj_qkvz`, `gda_step_1b_in_proj_ba`,
  `gda_step_2a_prepend_conv_state`, `gda_step_2b_conv1d_silu`,
  `gda_step_2c_update_conv_state`, `gda_step_3_split_reshape_per_head`,
  `gda_step_4_qk_rmsnorm`, `gda_step_5_compute_g`,
  `gda_step_6_sigmoid_beta`, `gda_step_7_kernel_and_cache_update`,
  `gda_step_8_norm_proj`.

## Files

Rust:

- Modify `ironmlx/src/core/p5h.rs`
- Modify `ironmlx/src/cli/serve.rs`
- Modify `ironmlx/src/core/server/mod.rs`
- Modify `ironmlx/src/core/scheduler.rs`
- Modify `ironmlx/src/core/generate.rs`
- Modify `ironmlx/src/nn/gated_attention.rs`
- Modify `ironmlx/src/nn/gated_delta_net.rs`
- Modify `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`
- Modify `ironmlx/src/models/qwen3_5_moe/model.rs`
- Modify `ironmlx/tests/p5h_t5_attribution_capture.rs`

Python:

- Modify `tools/p5h_aggregator/schema_validator.py`
- Modify `tools/p5h_aggregator/aggregator.py`
- Modify `tools/p5h_aggregator/tests/test_validator.py`
- Modify `tools/p5h_aggregator/tests/test_aggregator.py`

Docs:

- Create `reports/p5h+1-ranking-snapshot.md`
- Create `docs/p5h+1-ranking-snapshot.md`
- Create `docs/p5h+1-close-out.md`
- Modify `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`
- Optionally append the P5h+1 result to the local memory file if the active
  agent is using one; do not put external memory paths in repo commits.

## Cross-Cutting Invariants

- `--p5h-measurement-eval-probes` is a measurement-only flag and defaults off.
- The CLI flag is exposed only when the `p5h-profile` feature is enabled.
- Feature-off builds still compile and run without a public probe flag.
- Feature-on plus flag-off preserves production lazy-graph behavior.
- Probe mode forces `mlx::transforms::eval` inside selected span bodies so
  substep spans accrue the work that would otherwise be charged downstream.
- Any span opened before a fallible operation must be closed before returning
  the error. Do not use `?` while a manual span is open unless a guard owns
  the close path.
- `chunk_idx` is zero-based and only non-null inside Lane B chunk context.
- Every span under a `gs_chunk_N` ancestor must inherit the same `chunk_idx`
  as that ancestor.

## Task 1: Lane A Split And Measurement Probe Flag

### Step 1.1 Add Measurement Probe Global

Edit `ironmlx/src/core/p5h.rs`.

- Extend the existing atomic import to include `AtomicBool`.
- Add a feature-gated global:

```rust
#[cfg(feature = "p5h-profile")]
static MEASUREMENT_EVAL_PROBES_ACTIVE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "p5h-profile")]
pub fn set_measurement_eval_probes_active(active: bool) {
    MEASUREMENT_EVAL_PROBES_ACTIVE.store(active, Ordering::Relaxed);
}

#[inline]
pub fn is_measurement_eval_probes_active() -> bool {
    #[cfg(feature = "p5h-profile")]
    {
        MEASUREMENT_EVAL_PROBES_ACTIVE.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "p5h-profile"))]
    {
        false
    }
}
```

Verification:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5h-profile -p ironmlx
```

### Step 1.2 Add The CLI Flag

Edit `ironmlx/src/cli/serve.rs`.

- Add the flag to `ServeArgs` behind the `p5h-profile` feature:

```rust
#[cfg(feature = "p5h-profile")]
/// Force selected P5h+1 span bodies to materialize MLX work before span close.
#[arg(long, default_value_t = false)]
pub p5h_measurement_eval_probes: bool,
```

- At the `server::serve(...)` call site, derive the argument with cfg blocks:

```rust
#[cfg(feature = "p5h-profile")]
let p5h_measurement_eval_probes = args.p5h_measurement_eval_probes;
#[cfg(not(feature = "p5h-profile"))]
let p5h_measurement_eval_probes = false;
```

Edit `ironmlx/src/core/server/mod.rs`.

- Add `p5h_measurement_eval_probes: bool` to `serve(...)`.
- Search all `server::serve(` call sites and pass the new boolean explicitly.
- Near startup, before constructing shared state, set the global:

```rust
#[cfg(feature = "p5h-profile")]
crate::core::p5h::set_measurement_eval_probes_active(p5h_measurement_eval_probes);
#[cfg(not(feature = "p5h-profile"))]
let _ = p5h_measurement_eval_probes;
```

Verification:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5h-profile -p ironmlx
```

### Step 1.3 Split `first_token_sampling`

Edit `ironmlx/src/core/scheduler.rs`.

Replace the current single explicit `first_token_sampling` span with two
sibling explicit spans under the same root parent:

- `first_token_sampling_prepare`: covers logits reshape, sentinel sampler
  setup, per-row sampler refs, and per-row history construction.
- `first_token_sampling_materialize_and_sample`: covers `history_refs`
  construction and `sample_batch(...)`, including the `.to_vec()`-driven
  materialization inside sampling.

Use the current explicit API. Do not introduce or call a nonexistent
`with_p5h_span` helper.

Implementation discipline:

- Open `first_token_sampling_prepare` before `let logits_shape = logits.shape()`.
- If reshape fails, close the prepare span before returning the error.
- Close `first_token_sampling_prepare` after `row_histories` is populated and
  before `history_refs` is built.
- Open `first_token_sampling_materialize_and_sample` immediately before
  `history_refs` construction.
- Capture the `sample_batch(...)` result into a local `Result`.
- Close `first_token_sampling_materialize_and_sample`.
- Only then apply `?` to the captured result.

The resulting structure inside the existing `tokens_result` closure should
follow this shape:

```rust
#[cfg(feature = "p5h-profile")]
let mut prepare_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
    crate::core::p5h::open_p5h_span(ctx, Some(root_span), "first_token_sampling_prepare")
});

let logits_shape = logits.shape();
let vocab = logits_shape.as_slice()[2];
let logits_bv_result = logits.reshape(&[b as i32, vocab][..]).map_err(|e| {
    anyhow!("prefill_admitted: reshape logits [B,1,vocab]->[B,vocab] failed: {e:?}")
});

let logits_bv = match logits_bv_result {
    Ok(value) => value,
    Err(err) => {
        #[cfg(feature = "p5h-profile")]
        if let (Some((ctx, _)), Some(span)) = (p5h_trace.as_ref(), prepare_span.take()) {
            crate::core::p5h::close_p5h_span(
                ctx,
                span,
                crate::core::p5h::monotonic_ns_public(),
                crate::core::p5h::SpanFields::default(),
            );
        }
        return Err(err);
    }
};

// Build row_samplers and row_histories here.

#[cfg(feature = "p5h-profile")]
if let (Some((ctx, _)), Some(span)) = (p5h_trace.as_ref(), prepare_span.take()) {
    crate::core::p5h::close_p5h_span(
        ctx,
        span,
        crate::core::p5h::monotonic_ns_public(),
        crate::core::p5h::SpanFields::default(),
    );
}

#[cfg(feature = "p5h-profile")]
let materialize_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
    crate::core::p5h::open_p5h_span(
        ctx,
        Some(root_span),
        "first_token_sampling_materialize_and_sample",
    )
});

let history_refs: Vec<&[u32]> = row_histories.iter().map(|h| h.as_slice()).collect();
let sample_result = crate::core::sampler::sample_batch(
    &row_samplers,
    &logits_bv,
    &history_refs,
    &mut self.prng_state,
)
.map_err(|e| anyhow!("prefill_admitted: sample_batch failed: {e:?}"));

#[cfg(feature = "p5h-profile")]
if let (Some((ctx, _)), Some(span)) = (p5h_trace.as_ref(), materialize_span) {
    crate::core::p5h::close_p5h_span(
        ctx,
        span,
        crate::core::p5h::monotonic_ns_public(),
        crate::core::p5h::SpanFields::default(),
    );
}

let tokens = sample_result?;
```

The important part is the close-on-error discipline: reshape errors close the
prepare span before returning, and sampling errors close the materialize span
before returning.

### Step 1.4 Add Per-Substep Eval Probes

In each existing `try_with_p5h_span_from_current_trace` closure below, call
`mlx::transforms::eval` on returned `Array` values when
`is_measurement_eval_probes_active()` is true.

Pattern:

```rust
if crate::core::p5h::is_measurement_eval_probes_active() {
    mlx::transforms::eval(&[&array_to_charge_to_this_span])?;
}
```

For tuple-returning substeps, evaluate every returned `Array` in the tuple.
For closures that return `Result<()>`, do not add a dummy eval.

Required sites:

- `ironmlx/src/nn/gated_attention.rs`:
  `q_gate_k_v_proj`, `q_split_norm_reshape`, `mrope_apply`,
  `kv_mask_update`, `fused_sdpa`, `gate_sigmoid_mul`, `o_proj`,
  and the existing `cache_state_update` child.
- `ironmlx/src/nn/gated_delta_net.rs`:
  `gda_step_1a_in_proj_qkvz`, `gda_step_1b_in_proj_ba`,
  `gda_step_2a_prepend_conv_state`, `gda_step_2b_conv1d_silu`,
  `gda_step_2c_update_conv_state`, `gda_step_3_split_reshape_per_head`,
  `gda_step_4_qk_rmsnorm`, `gda_step_5_compute_g`,
  `gda_step_6_sigmoid_beta`, `gda_step_7_kernel_and_cache_update`,
  `gda_step_8_norm_proj`, plus the `cache_state_update` children under
  Step 2c and Step 7.
- `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`:
  `router_logits_softmax_topk`, `routing_sort_pack`,
  `gather_qmm_gate_up`, `swiglu_activation`, `gather_qmm_down`,
  `routing_unsort_weighted_reduce`, `shared_expert`, `moe_output_sum`.
- `ironmlx/src/models/qwen3_5_moe/model.rs`:
  `slice_last_and_project_lm_head`.

Skip `tokenizer_encode`; it is CPU work and already complete by the time the
retroactive span opens.

### Step 1.5 Update Lane A Validator Requirements

Edit `tools/p5h_aggregator/schema_validator.py`.

- In `LANE_A_REQUIRED_TREE`, replace `first_token_sampling` with both:
  `first_token_sampling_prepare` and
  `first_token_sampling_materialize_and_sample`.
- Do not add `chunk_idx` in Task 1. That belongs to Task 2 where the schema
  emit format changes.

Edit `tools/p5h_aggregator/tests/test_validator.py`.

Add tests asserting:

- `first_token_sampling` is no longer required for Lane A.
- `first_token_sampling_prepare` is required.
- `first_token_sampling_materialize_and_sample` is required.

### Step 1.6 Verify Task 1

Run:

```bash
uv run python -m pytest tools/p5h_aggregator/tests/test_validator.py -v
uv run ruff check tools/p5h_aggregator/schema_validator.py tools/p5h_aggregator/tests/test_validator.py
uv run ruff format --check tools/p5h_aggregator/schema_validator.py tools/p5h_aggregator/tests/test_validator.py
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5h-profile -p ironmlx
```

Recommended smoke tests if model files are available:

```bash
MLX_DIR=$HOME/.local/mlx IRONMLX_MOE_MODEL_DIR=$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
MLX_DIR=$HOME/.local/mlx IRONMLX_MOE_MODEL_DIR=$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

Commit after all required verification passes.

## Task 2: Lane B Deep Attribution And `chunk_idx`

### Step 2.1 Add `chunk_idx` To Rust Span Fields And Emission

Edit `ironmlx/src/core/p5h.rs`.

- Add `chunk_idx: Option<u32>` to `SpanFields`. Preserve `seq` as
  `Option<u32>`.

```rust
#[derive(Default, Debug)]
pub struct SpanFields {
    pub layer_idx: Option<i32>,
    pub chunk_idx: Option<u32>,
    pub seq: Option<u32>,
    pub mode: Option<&'static str>,
}
```

- Add a Lane B chunk context stack and an RAII guard. Do not expose manual
  push/pop as the call-site API because early `?` returns would leak stack
  state.

```rust
thread_local! {
    pub(crate) static P5H_CURRENT_CHUNK_STACK: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

pub(crate) struct P5hChunkContextGuard {
    chunk_idx: u32,
    active: bool,
}

pub(crate) fn enter_chunk_context(chunk_idx: u32) -> P5hChunkContextGuard {
    P5H_CURRENT_CHUNK_STACK.with(|s| s.borrow_mut().push(chunk_idx));
    P5hChunkContextGuard {
        chunk_idx,
        active: true,
    }
}

fn current_chunk_idx() -> Option<u32> {
    P5H_CURRENT_CHUNK_STACK.with(|s| s.borrow().last().copied())
}

impl Drop for P5hChunkContextGuard {
    fn drop(&mut self) {
        if self.active {
            P5H_CURRENT_CHUNK_STACK.with(|s| {
                let popped = s.borrow_mut().pop();
                assert_eq!(
                    popped,
                    Some(self.chunk_idx),
                    "P5hChunkContextGuard dropped out of order"
                );
            });
        }
    }
}
```

- In `emit_log_line_with_end_ns`, add `chunk_idx` between `layer_idx` and
  `span_id`. Use explicit field value first, then inherited chunk context:

```rust
let chunk_idx = fields.chunk_idx.or_else(current_chunk_idx);
```

The emitted field order must become:

```text
request_id routing_path prompt_tokens seq layer_idx chunk_idx span_id parent_span_id span_name parent_span start_ns end_ns mode span_kind
```

### Step 2.2 Add `chunk_idx` To Python Schema

Edit `tools/p5h_aggregator/schema_validator.py`.

- Add `chunk_idx=(?P<chunk_idx>null|\d+)` to `P5H_LOG_RE` between
  `layer_idx` and `span_id`.
- Add `chunk_idx: int | None` to the `Span` dataclass between `layer_idx`
  and `span_id`.
- Parse `"null"` to `None`; parse digits to `int`.

Edit `tools/p5h_aggregator/tests/test_validator.py`.

Add tests for parsing:

- A Lane A line with `chunk_idx=null`.
- A Lane B line with `chunk_idx=2`.

### Step 2.3 Extend Lane B Try-Helper Allow-List

Edit `ironmlx/src/core/p5h.rs`.

Extend the existing `LANE_B_ALLOWED_TRY_SPAN_NAMES` constant. Keep a single
Lane B try-helper allow-list.

Required names:

```rust
const LANE_B_ALLOWED_TRY_SPAN_NAMES: &[&str] = &[
    "gs_kv_cache_alloc",
    "gs_chunk_N",
    "gs_first_token_sample_dispatch",
    "decoder_layer_N",
    "input_norm",
    "attention_path",
    "residual_overhead",
    "post_attention_norm",
    "mlp_path",
    "q_gate_k_v_proj",
    "q_split_norm_reshape",
    "mrope_apply",
    "kv_mask_update",
    "fused_sdpa",
    "gate_sigmoid_mul",
    "o_proj",
    "router_logits_softmax_topk",
    "routing_sort_pack",
    "gather_qmm_gate_up",
    "swiglu_activation",
    "gather_qmm_down",
    "routing_unsort_weighted_reduce",
    "shared_expert",
    "moe_output_sum",
    "cache_state_update",
    "slice_last_and_project_lm_head",
    "gda_step_1a_in_proj_qkvz",
    "gda_step_1b_in_proj_ba",
    "gda_step_2a_prepend_conv_state",
    "gda_step_2b_conv1d_silu",
    "gda_step_2c_update_conv_state",
    "gda_step_3_split_reshape_per_head",
    "gda_step_4_qk_rmsnorm",
    "gda_step_5_compute_g",
    "gda_step_6_sigmoid_beta",
    "gda_step_7_kernel_and_cache_update",
    "gda_step_8_norm_proj",
];
```

Update the surrounding comments: Lane B is no longer top-level-only in P5h+1;
it emits the listed decoder and substep spans under `gs_chunk_N`.

### Step 2.4 Propagate `chunk_idx` From `gs_chunk_N`

Edit `ironmlx/src/core/generate.rs`.

In `GenerationStream::new`, the feature-gated chunk loop already opens
`gs_chunk_N` through `try_with_p5h_span_from_current_trace`.

- Add `let mut chunk_idx: u32 = 0;` before the loop.
- In the `SpanFields` for `gs_chunk_N`, set:

```rust
crate::core::p5h::SpanFields {
    seq: Some(chunk_size as u32),
    chunk_idx: Some(chunk_idx),
    ..Default::default()
}
```

- At the top of the `gs_chunk_N` closure, enter the RAII chunk context:

```rust
let _chunk_guard = crate::core::p5h::enter_chunk_context(chunk_idx);
```

- Increment `chunk_idx` exactly once after a successful chunk iteration and
  before `pos += n` moves to the next chunk. If the chunk returns an error,
  the RAII guard must still drop before the error propagates.

Do not add chunk context to `GenerationStream::new_text_only` unless the T5
OpenAI Lane B capture actually uses it. The OpenAI server path currently uses
`GenerationStream::new`.

### Step 2.5 Extend Lane B Validator Requirements

Edit `tools/p5h_aggregator/schema_validator.py`.

- Add the same deep span names from `LANE_B_ALLOWED_TRY_SPAN_NAMES` to
  `LANE_B_REQUIRED_TREE`, except `tokenizer_encode`, which remains allowed
  but not required.
- Keep:

```python
LANE_B_ALLOWED_TREE = LANE_B_REQUIRED_TREE | {"tokenizer_encode"}
```

- Add a structural check in per-request validation:

```python
def validate_chunk_ancestry(spans: list[Span]) -> list[str]:
    by_id = {s.span_id: s for s in spans}
    failures: list[str] = []

    for span in spans:
        if span.span_name == "gs_chunk_N" and span.chunk_idx is None:
            failures.append(f"gs_chunk_N span_id={span.span_id} has null chunk_idx")
            continue

        ancestor = by_id.get(span.parent_span_id) if span.parent_span_id is not None else None
        chunk_ancestor: Span | None = None
        while ancestor is not None:
            if ancestor.span_name == "gs_chunk_N":
                chunk_ancestor = ancestor
                break
            ancestor = (
                by_id.get(ancestor.parent_span_id)
                if ancestor.parent_span_id is not None
                else None
            )

        if chunk_ancestor is None:
            if span.span_name != "gs_chunk_N" and span.chunk_idx is not None:
                failures.append(
                    f"span_id={span.span_id} ({span.span_name}) is outside gs_chunk_N "
                    f"but has chunk_idx={span.chunk_idx}"
                )
            continue

        if chunk_ancestor.chunk_idx is None:
            failures.append(
                f"gs_chunk_N ancestor span_id={chunk_ancestor.span_id} has null chunk_idx"
            )
        elif span.chunk_idx != chunk_ancestor.chunk_idx:
            failures.append(
                f"span_id={span.span_id} ({span.span_name}) has chunk_idx={span.chunk_idx} "
                f"but gs_chunk_N ancestor span_id={chunk_ancestor.span_id} has chunk_idx={chunk_ancestor.chunk_idx}"
            )

    return failures
```

Hook every returned failure into the existing `ValidationReport`.

Tests to add:

- `LANE_B_REQUIRED_TREE` includes decoder wrappers, attention substeps, MoE
  substeps, GDN substeps, `cache_state_update`, and
  `slice_last_and_project_lm_head`.
- `validate_chunk_ancestry` passes for matching chunk ids.
- `validate_chunk_ancestry` fails for a descendant with mismatched chunk id.
- `validate_chunk_ancestry` fails when `gs_chunk_N` has null `chunk_idx`.
- `validate_chunk_ancestry` fails when a span outside `gs_chunk_N` has a
  non-null `chunk_idx`.
- A test reads `ironmlx/src/core/p5h.rs`, extracts
  `LANE_B_ALLOWED_TRY_SPAN_NAMES`, and asserts the extracted names are a
  subset of Python `LANE_B_ALLOWED_TREE`.

### Step 2.6 Add `chunk_idx` To Attribution CSV

Edit `tools/p5h_aggregator/aggregator.py`.

- Update `ResidualLeaf` to carry `chunk_idx: int | None`.
- In `synthesize_residual_leaves`, set residual `chunk_idx` from the parent
  span's `chunk_idx`.
- Update `write_attribution_csv` header by inserting `chunk_idx` after
  `routing_path`.
- For tree rows, write `s.chunk_idx` or empty string.
- For synthesized rows, write `r.chunk_idx` or empty string.
- For diagnostic rows, write `d.chunk_idx` or empty string.
- Do not remove or reorder existing columns other than inserting `chunk_idx`.

Final CSV order:

```text
pp,request_id,routing_path,chunk_idx,span_name,span_kind,parent_span_id,span_id,inclusive_us,exclusive_us
```

Edit `tools/p5h_aggregator/tests/test_aggregator.py`.

Add a test that verifies:

- The header contains `chunk_idx` after `routing_path`.
- Tree and diagnostic rows preserve their parsed chunk id.
- Synthesized residual rows inherit the parent chunk id.

### Step 2.7 Verify Task 2

Run:

```bash
uv run python -m pytest tools/p5h_aggregator/tests/ -v
uv run ruff check tools/p5h_aggregator/
uv run ruff format --check tools/p5h_aggregator/
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5h-profile -p ironmlx --tests
```

Recommended smoke tests if model files are available:

```bash
MLX_DIR=$HOME/.local/mlx IRONMLX_MOE_MODEL_DIR=$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec cargo test -p ironmlx --release --features p5h-profile --test p5_qwen35_moe_smoke -- --ignored --test-threads=1
```

Commit after all required verification passes.

## Task 3: Re-Run T5 Capture With Probe Mode

### Step 3.1 Update Capture Harness

Edit `ironmlx/tests/p5h_t5_attribution_capture.rs`.

Add `--p5h-measurement-eval-probes` to the server launch arguments used by
the T5 attribution capture test.

Verification:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5h-profile -p ironmlx --tests
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
```

### Step 3.2 Run The Sweep

Run on the GPU-capable local machine:

```bash
lsof -i :18099 || echo "PORT_FREE"
rm -f /tmp/p5h-t5-server.log /tmp/p5h-t5-bench.csv /tmp/p5h-t5-attribution.csv /tmp/p5h-t5-attribution.summary.csv /tmp/p5h-t5-ranking.csv /tmp/p5h-t5-verdict.json /tmp/p5h-t5.log

IRONMLX_MOE_MODEL_DIR=$HOME/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec \
MLX_DIR=$HOME/.local/mlx \
cargo test -p ironmlx --release --features p5h-profile \
  --test p5h_t5_attribution_capture -- --ignored --test-threads=1 --nocapture > /tmp/p5h-t5.log 2>&1
echo "exit=$?" >> /tmp/p5h-t5.log
```

Expected:

- `/tmp/p5h-t5.log` ends with `exit=0`.
- `/tmp/p5h-t5-server.log` contains deep Lane B spans under `gs_chunk_N`.
- `/tmp/p5h-t5-bench.csv` has one header plus the expected T5 data rows.

### Step 3.3 Aggregate And Rank

Run:

```bash
uv run python -m tools.p5h_aggregator.aggregator \
    --server-log /tmp/p5h-t5-server.log \
    --bench-csv /tmp/p5h-t5-bench.csv \
    --out /tmp/p5h-t5-attribution.csv \
    --summary-out /tmp/p5h-t5-attribution.summary.csv

uv run python -m tools.p5h_aggregator.roi_ranking \
    --attribution-csv /tmp/p5h-t5-attribution.csv \
    --summary-csv /tmp/p5h-t5-attribution.summary.csv \
    --out-ranking /tmp/p5h-t5-ranking.csv \
    --out-verdict /tmp/p5h-t5-verdict.json
```

Expected:

- Schema validation passes for every non-aborted request.
- Coverage for every PP is at least 0.95.
- `chunk_idx` is present in attribution CSV.
- Lane B rows under `gs_chunk_N` have non-empty `chunk_idx`.
- Verdict JSON has no `data_insufficient` PP result.

## Task 4: Close Gate And Ranking Snapshot

### Step 4.1 Verify Close Gate

Read `/tmp/p5h-t5-verdict.json`, `/tmp/p5h-t5-ranking.csv`, and
`/tmp/p5h-t5-attribution.summary.csv`.

The close gate passes only if all of the following are true:

- Every PP verdict is one of `yes`, `yes_with_scope_gate`, or
  `no_under_measured_cap`.
- No PP has `data_insufficient`.
- The top actionable candidate for every PP is not
  `first_token_sampling_materialize_and_sample`.
- The top actionable candidate for every Lane B PP is not `gs_chunk_N`.
- The top actionable candidate does not start with `unattributed_`.
- Coverage for every PP is at least 0.95.
- Probe-mode root time and production root time are stored as separate values.

Helpful wrapper-top1 check:

```bash
python3 - <<'PY'
import csv

wrappers = {"first_token_sampling_materialize_and_sample", "gs_chunk_N"}
with open("/tmp/p5h-t5-attribution.summary.csv", newline="") as f:
    for row in csv.DictReader(f):
        pp = int(row["pp"])
        top1 = row["top1_span_name"]
        share = float(row["top1_share"])
        if top1 in wrappers:
            status = "PASS" if share <= 0.5 else "FAIL"
            print(f"PP={pp} top1={top1} share={share:.4f} {status}")
        else:
            print(f"PP={pp} top1={top1} share={share:.4f} PASS")
PY
```

If the close gate fails, stop and fix Tasks 1 or 2. Do not write a successful
close-out.

### Step 4.2 Write Ranking Snapshots

Create `reports/p5h+1-ranking-snapshot.md` with full detail:

- Date and branch.
- Implementation commits used for the sweep.
- Sweep command and wall time.
- Per-PP observed lane.
- Per-PP `probe_attribution_root_us`.
- Per-PP `production_root_us`.
- Per-PP top three non-wrapper candidates.
- Per-PP verdict and explanation.
- Coverage per PP.
- Scope-gate trigger count.
- Any blocked or partial state if the close gate failed.

Create `docs/p5h+1-ranking-snapshot.md` as the committed concise summary:

- One paragraph stating whether P5h+1 close gate passed.
- A per-PP table populated with actual data from the generated CSV/JSON files.
- A note that probe-mode attribution data was generated with
  `--p5h-measurement-eval-probes`.
- A note that production target feasibility uses the production root baseline,
  not probe-mode root time.
- A link to `reports/p5h+1-ranking-snapshot.md` for local full detail.

Do not commit empty candidate names, fake numbers, or instructions to fill
values later.

Commit after the snapshot is complete.

## Task 5: Close-Out And Spec Update

### Step 5.1 Write Close-Out

Create `docs/p5h+1-close-out.md` with:

- Status, date, branch, and implementation commits.
- Close Gate result with per-PP evidence.
- Lane A changes: sampler split and per-substep eval probes.
- Lane B changes: deep allow-list, `gs_chunk_N` chunk context, and `chunk_idx`
  schema.
- P5i and P5j candidate summary from `docs/p5h+1-ranking-snapshot.md`.
- Target feasibility verdicts using production root time as denominator.
- P5h+2 follow-up list.
- References to the design and this implementation plan.

### Step 5.2 Update P5h Spec Gates

Edit
`docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`.

In section 7.2.1:

- Update Gate 5 from deferred to the actual P5h+1 result.
- Update Gate 6 from deferred to the actual P5h+1 result.
- Use real candidate names and verdicts from
  `docs/p5h+1-ranking-snapshot.md`.
- If Close Gate failed, write the specific blocked state instead of claiming
  locked pass.

Optional reconciliation note in section 1.2:

- Document that PP=2048 is the nominal P5j point, while runtime lane is derived
  from observed `routing_path`.
- Document the ChatML token overhead explanation for why
  `iron-bench --prompt-len 2048` can route through `gs_chunked`.
- Document that a `--prompt-len 2036` boundary-control sweep is optional and
  does not replace PP=2048 as the primary measurement point.

### Step 5.3 Verify Task 5

Search the generated docs for unfinished markers, fake value stubs, date
stubs, and instructions to fill values later.

Expected: no unfinished markers remain.

Commit after the docs are complete.

## Final Verification

Run before declaring implementation complete:

```bash
uv run python -m pytest tools/p5h_aggregator/tests/ -v
uv run ruff check tools/p5h_aggregator/
uv run ruff format --check tools/p5h_aggregator/
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
MLX_DIR=$HOME/.local/mlx cargo build --release --features p5h-profile -p ironmlx --tests
```

If the model is available locally, also run the smoke tests and the T5 capture
sweep from Task 3.

## Plan Self-Review Checklist

The implementation agent must keep these checks true while editing:

- No use of nonexistent `with_p5h_span`.
- No introduction of a second Lane B try-helper allow-list constant.
- GDN names match the verified current code facts at the top of this plan.
- `seq` remains `Option<u32>`.
- No parser compatibility path for old logs.
- No manual chunk push/pop at call sites; use `P5hChunkContextGuard`.
- No unfinished value stubs in committed docs.
- `write_attribution_csv` remains the attribution writer being edited.
- `chunk_idx` is inserted after `routing_path` in attribution CSV.
- Rust verification uses the repository-required commands:
  `cargo fmt`,
  `cargo +nightly fmt --all -- --check`,
  `cargo +nightly clippy --all-features --workspace -- -D warnings`,
  and `cargo build --release`.
