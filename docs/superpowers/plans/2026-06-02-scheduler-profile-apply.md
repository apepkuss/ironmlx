# Scheduler Profile Apply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users export a selected scheduler/autotune profile and apply it directly at `ironmlx serve` startup while preserving manual CLI overrides.

**Architecture:** Add a minimal runtime profile schema in `core::scheduler_autotune` that contains only metadata and the selected `SchedulerAutotuneProfileConfig`. Extend `scheduler-autotune select` to write that profile, and extend `serve` to resolve effective scheduler parameters from defaults, optional profile, and explicit CLI overrides.

**Tech Stack:** Rust, serde, serde_json, clap, existing scheduler/autotune selector structs.

---

## File Structure

- Modify `ironmlx/src/core/scheduler_autotune.rs` — add `SchedulerAutotuneRuntimeProfile` and a builder from `SchedulerAutotuneProfileSelection`.
- Modify `ironmlx/tests/scheduler_autotune_profile.rs` — add TDD coverage for runtime profile export and no-selected error.
- Modify `ironmlx/src/cli/scheduler_autotune.rs` — add `--write-profile` to `select` and legacy select mode.
- Modify `ironmlx/src/cli/mod.rs` — add parser coverage for `--write-profile` and `serve --scheduler-profile`.
- Modify `ironmlx/src/cli/serve.rs` — add `--scheduler-profile`, resolve profile/default/CLI override config, and pass effective values to server startup.
- Modify `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — document export/apply workflow.

### Task 1: Runtime Profile Schema

- [x] **Step 1: Write failing core tests**

Add tests to `ironmlx/tests/scheduler_autotune_profile.rs`:

```rust
use ironmlx::core::scheduler_autotune::build_scheduler_autotune_runtime_profile;
```

Test names:

- `runtime_profile_uses_selected_config_and_metadata`
- `runtime_profile_requires_selected_candidate`

Expected behavior:

- A selection with a selected candidate produces `schema_version=1`, same `model_name`, same `hardware_label`, and the selected config.
- A selection with no valid candidate returns an error containing `selected`.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_profile runtime_profile -- --nocapture
```

Expected: compile failure until the profile API exists.

- [x] **Step 3: Implement runtime profile schema**

Add:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerAutotuneRuntimeProfile {
    pub schema_version: u32,
    pub model_name: String,
    pub hardware_label: String,
    pub config: SchedulerAutotuneProfileConfig,
}

pub fn build_scheduler_autotune_runtime_profile(
    selection: &SchedulerAutotuneProfileSelection,
) -> Result<SchedulerAutotuneRuntimeProfile>
```

Return an error if `selection.selected` is `None`.

- [x] **Step 4: Verify GREEN**

Run the same `runtime_profile` test command. Expected: both profile tests pass.

### Task 2: Selector Profile Export

- [x] **Step 1: Write failing CLI parser test**

Add a parser test in `ironmlx/src/cli/mod.rs` for:

```bash
ironmlx scheduler-autotune select \
  --input calibration.json \
  --format json \
  --write-profile scheduler-profile.json
```

Assert that `write_profile` parses as `scheduler-profile.json`.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::tests::scheduler_autotune_select_subcommand_parses_write_profile -- --nocapture
```

Expected: compile failure until `write_profile` exists.

- [x] **Step 3: Implement `--write-profile`**

Add `write_profile: Option<PathBuf>` to `SchedulerAutotuneSelectArgs` and legacy `SchedulerAutotuneArgs`. In `run_select`, after selecting, call `build_scheduler_autotune_runtime_profile` and write pretty JSON plus trailing newline when `write_profile` is present.

- [x] **Step 4: Verify GREEN**

Run the same parser test and the existing legacy scheduler-autotune parser test.

### Task 3: Serve Profile Apply

- [x] **Step 1: Write failing serve parser and resolver tests**

Add parser coverage in `ironmlx/src/cli/mod.rs` for:

```bash
ironmlx serve --model /tmp/model --scheduler-profile scheduler-profile.json
```

Add unit tests in `ironmlx/src/cli/serve.rs`:

- `scheduler_profile_supplies_missing_scheduler_values`
- `cli_scheduler_values_override_profile_values`

The resolver must use profile values when CLI options are absent and CLI values when present.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::tests::serve_subcommand_parses_scheduler_profile -- --nocapture
cargo test --release -p ironmlx cli::serve::scheduler_profile_tests::scheduler_profile -- --nocapture
```

Expected: compile failure until `--scheduler-profile` and resolver exist.

- [x] **Step 3: Implement serve profile apply**

Change scheduler-related `ServeArgs` fields from required defaults to `Option`:

- `prefill_chunk_size: Option<usize>`
- `b_max: Option<usize>`
- `admission_deadline_ms: Option<u64>`
- `admission_queue_max: Option<usize>`
- `max_cache_cap: Option<usize>`

Add default constants matching existing behavior: `2048`, `1`, `5`, `32`, `32768`.

Add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SchedulerServeConfig {
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
}
```

Resolve config with priority:

1. CLI explicit value.
2. Runtime profile config.
3. Existing default.

Read profile JSON from `--scheduler-profile`, validate `schema_version == 1`, log profile metadata, and pass resolved values to `server::serve`.

- [x] **Step 4: Verify GREEN**

Run the same serve parser/resolver tests. Expected: tests pass.

### Task 4: Docs and Verification

- [x] **Step 1: Update Chinese design doc**

Document:

```bash
cargo run --release -p ironmlx -- \
  scheduler-autotune select \
  --input calibration.json \
  --format text \
  --write-profile scheduler-profile.json
```

Then:

```bash
cargo run --release -p ironmlx -- serve \
  --model /path/to/model \
  --scheduler-profile scheduler-profile.json
```

Explain that explicit CLI scheduler parameters override profile values.

- [x] **Step 2: Run required Rust verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
cargo test --release -p ironmlx cli::tests::scheduler_autotune_select_subcommand_parses_write_profile -- --nocapture
cargo test --release -p ironmlx cli::tests::serve_subcommand_parses_scheduler_profile -- --nocapture
cargo test --release -p ironmlx cli::serve::scheduler_profile_tests::scheduler_profile -- --nocapture
git diff --check
```

- [x] **Step 3: Commit**

```bash
git add ironmlx/src/core/scheduler_autotune.rs ironmlx/tests/scheduler_autotune_profile.rs ironmlx/src/cli/scheduler_autotune.rs ironmlx/src/cli/serve.rs ironmlx/src/cli/mod.rs docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md docs/superpowers/plans/2026-06-02-scheduler-profile-apply.md
git commit -m "feat: apply scheduler autotune profiles"
```
