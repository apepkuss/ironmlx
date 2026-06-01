# Scheduler Autotune 诊断推荐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an opt-in scheduler/autotune startup report that diagnoses current scheduler parameters and recommends next tuning actions without changing runtime behavior.

**Architecture:** Put all recommendation logic in a pure `core::scheduler_autotune` module, feed it from `server::serve` after `ModelMeta` and `effective_cap_max` are known, and expose it via a boolean `ironmlx serve --scheduler-autotune-report` flag. The default path remains unchanged.

**Tech Stack:** Rust, clap, tracing, existing `ModelMeta` / memory budget helpers.

---

## File Structure

- Create `ironmlx/src/core/scheduler_autotune.rs` — pure report input, derived budget stats, model prompt-limit sampling, recommendation generation, text rendering, unit tests.
- Modify `ironmlx/src/core/mod.rs` — export the module.
- Modify `ironmlx/src/cli/serve.rs` — add `--scheduler-autotune-report` and pass it to `server::serve`.
- Modify `ironmlx/src/core/server/mod.rs` — accept the flag and print the report after resolving `ModelMeta`.
- Modify `ironmlx/tests/p4_http_smoke.rs` and `ironmlx/tests/b1_p2_3d_admission_queue.rs` — update `server::serve` call sites with the new boolean.
- Create `ironmlx/tests/scheduler_autotune_report.rs` — integration tests for the public diagnostic API.
- Keep `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — Chinese design research.

### Task 1: Public Diagnostic API

**Files:**
- Create: `ironmlx/tests/scheduler_autotune_report.rs`
- Create: `ironmlx/src/core/scheduler_autotune.rs`
- Modify: `ironmlx/src/core/mod.rs`

- [x] **Step 1: Write failing tests**

Create `ironmlx/tests/scheduler_autotune_report.rs` with tests that import `ironmlx::core::scheduler_autotune` before that module exists:

```rust
use ironmlx::core::memory_budget::ModelMeta;
use ironmlx::core::scheduler_autotune::{
    build_scheduler_autotune_report, prompt_batch_limits_for_model, PromptBatchLimit,
    SchedulerAutotuneInput,
};
use ironmlx::core::Model;
use ironmlx::nn::LayerCache;
use mlx::{Array, Dtype, StreamOrDevice};

struct BatchLimitedModel;

impl Model for BatchLimitedModel {
    fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> ironmlx::Result<Vec<LayerCache>> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn forward_on(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn batched_prefill(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn forward_text_hidden(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn fresh_prefill_batch_limit(prompt_len: usize, b_max: usize) -> usize {
        if prompt_len >= 1024 {
            b_max.min(2)
        } else {
            b_max
        }
    }

    fn model_meta(&self) -> ModelMeta {
        sample_meta()
    }

    fn num_hidden_layers(&self) -> usize {
        0
    }
}

fn sample_meta() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 28,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_size: 4096,
        head_dim: None,
        weight_bytes: 3 * 1024 * 1024 * 1024,
        max_position_embeddings: 32768,
        spatial_merge_size: 2,
    }
}

fn sample_input(total_ram_bytes: usize) -> SchedulerAutotuneInput {
    SchedulerAutotuneInput {
        model_name: "test-model".to_string(),
        meta: sample_meta(),
        prefill_chunk_size: 2048,
        b_max: 4,
        admission_deadline_ms: 5,
        admission_queue_max: 32,
        requested_max_cache_cap: 32768,
        effective_cap_max: 32768,
        total_ram_bytes,
    }
}

#[test]
fn report_is_diagnose_only_and_never_applies_parameters() {
    let report = build_scheduler_autotune_report(
        sample_input(64 * 1024 * 1024 * 1024),
        vec![PromptBatchLimit {
            prompt_len: 2048,
            limit: 2,
        }],
    );

    assert!(report.diagnose_only);
    let text = report.render_text();
    assert!(text.contains("diagnose-only"));
    assert!(text.contains("no runtime parameters changed"));
}

#[test]
fn report_warns_when_reserved_kv_exceeds_available_budget() {
    let report = build_scheduler_autotune_report(
        sample_input(8 * 1024 * 1024 * 1024),
        vec![PromptBatchLimit {
            prompt_len: 2048,
            limit: 2,
        }],
    );

    assert!(report
        .recommendations
        .iter()
        .any(|item| item.code == "memory_budget_overrun"));
}

#[test]
fn prompt_batch_limits_sample_model_trait_policy() {
    let samples = prompt_batch_limits_for_model::<BatchLimitedModel>(4);

    assert_eq!(
        samples,
        vec![
            PromptBatchLimit {
                prompt_len: 512,
                limit: 4,
            },
            PromptBatchLimit {
                prompt_len: 1024,
                limit: 2,
            },
            PromptBatchLimit {
                prompt_len: 2048,
                limit: 2,
            },
            PromptBatchLimit {
                prompt_len: 8192,
                limit: 2,
            },
        ]
    );
}
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_report -- --nocapture
```

Expected: compile failure because `ironmlx::core::scheduler_autotune` does not exist.

- [x] **Step 3: Implement minimal module**

Create `scheduler_autotune.rs` with public structs used by the tests, deterministic budget derivation from `SchedulerAutotuneInput`, `prompt_batch_limits_for_model::<M>()`, recommendation generation, and `render_text()`.

- [x] **Step 4: Run tests to verify GREEN**

Run the same test command. Expected: all three tests pass.

### Task 2: Serve CLI Wiring

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`
- Modify: `ironmlx/src/core/server/mod.rs`
- Modify: `ironmlx/tests/p4_http_smoke.rs`
- Modify: `ironmlx/tests/b1_p2_3d_admission_queue.rs`

- [x] **Step 1: Write failing compile integration**

Add `scheduler_autotune_report: bool` to `server::serve` signature and update no call sites yet. Run `cargo check -p ironmlx` to verify missing arguments are caught.

- [x] **Step 2: Implement CLI flag**

Add to `ServeArgs`:

```rust
    /// Print scheduler/autotune diagnostics and recommendations at startup.
    /// Diagnose-only: this does not change any runtime parameter.
    #[arg(long, default_value_t = false)]
    pub scheduler_autotune_report: bool,
```

Pass it from `serve_with_model` into `server::serve`.

- [x] **Step 3: Implement server report emission**

After resolving `meta`, `model_max_context`, and `effective_cap_max`, call:

```rust
if scheduler_autotune_report {
    let report = crate::core::scheduler_autotune::build_scheduler_autotune_report(
        crate::core::scheduler_autotune::SchedulerAutotuneInput { ... },
        crate::core::scheduler_autotune::prompt_batch_limits_for_model::<M>(b_max),
    );
    tracing::info!(target: "ironmlx::scheduler_autotune", "\n{}", report.render_text());
}
```

- [x] **Step 4: Update call sites**

Pass `false` in existing tests that call `server::serve` directly.

- [x] **Step 5: Run verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_report -- --nocapture
cargo test --release -p ironmlx --lib scheduler_autotune -- --nocapture
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all commands pass.

### Task 3: Documentation Review

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md`
- Modify: `docs/superpowers/plans/2026-06-01-scheduler-autotune.md`

- [x] **Step 1: Check mermaid syntax**

Confirm the single `flowchart TD` diagram uses quoted labels and no invalid punctuation.

- [x] **Step 2: Check scope language**

Confirm docs state this is diagnose-only and future stages are separate.

- [x] **Step 3: Final git review**

Run:

```bash
git diff --stat
git diff --check
git status --short
```

Expected: only scheduler/autotune files changed; no whitespace errors.
