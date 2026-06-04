# Scheduler Autotune Calibration Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline calibration JSON merge/validation path so multiple `iron-bench --format autotune-json` candidate files can be safely combined before profile selection.

**Architecture:** Keep the merge logic in `core::scheduler_autotune` as pure functions. Extend `ironmlx scheduler-autotune` with a `merge` subcommand while preserving the existing `scheduler-autotune --input ... --format ...` select path. The merge command reads multiple candidate JSON files, validates metadata/scenario consistency, and prints a single calibration JSON.

**Tech Stack:** Rust, serde, serde_json, clap, existing `SchedulerAutotuneCalibrationInput`.

---

## File Structure

- Modify `ironmlx/src/core/scheduler_autotune.rs` — add pure merge/validation structs and function.
- Create `ironmlx/tests/scheduler_autotune_merge.rs` — TDD coverage for metadata mismatch, incomplete coverage, and successful merge.
- Modify `ironmlx/src/cli/scheduler_autotune.rs` — add `merge` subcommand and dispatch.
- Modify `ironmlx/src/cli/mod.rs` — add parser tests for `scheduler-autotune merge`.
- Modify `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — replace jq-only workflow with the built-in merge command.

### Task 1: Pure Merge API

- [x] **Step 1: Write failing merge tests**

Create `ironmlx/tests/scheduler_autotune_merge.rs` importing `merge_scheduler_autotune_calibrations`, `SchedulerAutotuneCalibrationInput`, `SchedulerAutotuneMeasurement`, `SchedulerAutotuneObjective`, and `SchedulerAutotuneProfileConfig`.

Tests:

- successful merge preserves model/hardware/objective and concatenates measurements;
- metadata mismatch returns an error mentioning `model_name`;
- missing scenario coverage returns an error mentioning `scenario coverage`.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_merge -- --nocapture
```

Expected: compile failure until the merge API exists.

- [x] **Step 3: Implement pure merge**

Add:

- `SchedulerAutotuneMergeOptions { require_complete_coverage: bool }`;
- `merge_scheduler_autotune_calibrations(inputs, options) -> anyhow::Result<SchedulerAutotuneCalibrationInput>`.

Validation:

- at least one input;
- all inputs must have `schema_version == 1`;
- all inputs must share `model_name`, `hardware_label`, and normalized objective;
- each input must contain measurements;
- when `require_complete_coverage` is true, every distinct config must cover the same `(prompt_len, max_new_tokens, concurrency)` scenario set.

- [x] **Step 4: Verify GREEN**

Run the same merge test command. Expected: merge tests pass.

### Task 2: CLI Merge Subcommand

- [x] **Step 1: Write failing CLI parser test**

Add a parser test for:

```bash
ironmlx scheduler-autotune merge \
  --input candidate-a.json \
  --input candidate-b.json \
  --output calibration.json
```

The test should assert that the `Merge` action parses with two inputs and an output path.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::tests::scheduler_autotune_merge_subcommand_parses_inputs_and_output -- --nocapture
```

Expected: compile failure until the merge action exists.

- [x] **Step 3: Implement CLI dispatch**

Refactor `SchedulerAutotuneArgs` to support:

- legacy select mode: `ironmlx scheduler-autotune --input calibration.json --format text`;
- explicit select mode: `ironmlx scheduler-autotune select --input calibration.json --format json`;
- merge mode: `ironmlx scheduler-autotune merge --input a.json --input b.json --output calibration.json`.

Merge mode writes pretty JSON to `--output` when supplied, otherwise stdout.

- [x] **Step 4: Verify GREEN**

Run the same CLI parser test command. Expected: parser test passes.

### Task 3: Docs, Verification, Commit

- [x] **Step 1: Update Chinese design doc**

Document:

```bash
cargo run --release -p ironmlx -- \
  scheduler-autotune merge \
  --input candidate-b1.json \
  --input candidate-b2.json \
  --output calibration.json
```

Then:

```bash
cargo run --release -p ironmlx -- \
  scheduler-autotune select \
  --input calibration.json \
  --format text
```

- [x] **Step 2: Run required Rust verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --release -p ironmlx --test scheduler_autotune_merge -- --nocapture
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
cargo test --release -p ironmlx cli::tests::scheduler_autotune_merge_subcommand_parses_inputs_and_output -- --nocapture
git diff --check
```

- [x] **Step 3: Commit**

```bash
git add ironmlx/src/core/scheduler_autotune.rs ironmlx/tests/scheduler_autotune_merge.rs ironmlx/src/cli/scheduler_autotune.rs ironmlx/src/cli/mod.rs docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md docs/superpowers/plans/2026-06-02-scheduler-autotune-merge.md
git commit -m "feat: merge scheduler autotune calibrations"
```
