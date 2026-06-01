# Scheduler Autotune 离线 Profile 选择器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a diagnose-only offline profile selector that consumes scheduler calibration measurements and recommends one scheduler config without changing runtime behavior.

**Architecture:** Extend `core::scheduler_autotune` with pure profile-selection structs and scoring logic, then add a thin `ironmlx scheduler-autotune` CLI wrapper that reads JSON and renders text or JSON output. Keep benchmark execution and runtime parameter application out of scope.

**Tech Stack:** Rust, serde, clap, existing `core::scheduler_autotune`.

---

## File Structure

- Modify `ironmlx/src/core/scheduler_autotune.rs` — add calibration input schema, candidate filtering, scoring, selection output, and text rendering.
- Create `ironmlx/tests/scheduler_autotune_profile.rs` — TDD coverage for selection, rejection, and agent coverage warnings.
- Modify `ironmlx/src/cli/mod.rs` — add `scheduler-autotune` subcommand and parser test.
- Create `ironmlx/src/cli/scheduler_autotune.rs` — read calibration JSON, call pure selector, render text or JSON.
- Modify `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — document stage 2 behavior, schema, scoring, and limits.

### Task 1: Pure Profile Selection API

- [x] **Step 1: Write failing integration tests**

Create `ironmlx/tests/scheduler_autotune_profile.rs` with tests importing `select_scheduler_autotune_profile`, `SchedulerAutotuneCalibrationInput`, `SchedulerAutotuneMeasurement`, `SchedulerAutotuneObjective`, and `SchedulerAutotuneProfileConfig` before those APIs exist.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
```

Expected: compile failure for unresolved profile-selection imports.

- [x] **Step 3: Implement minimal pure selector**

Add public calibration structs, reject memory-unsafe/cached-token candidates, require complete scenario coverage, normalize TTFT/ITL/E2E/throughput by scenario, and select the lowest weighted score.

- [x] **Step 4: Verify GREEN**

Run the same test command. Expected: 4 tests pass.

### Task 2: CLI Post-Processing Entry

- [x] **Step 1: Write parser test**

Add a `cli::tests::scheduler_autotune_subcommand_parses_input_and_json_format` parser test that expects:

```bash
ironmlx scheduler-autotune --input calibration.json --format json
```

to parse into the new subcommand.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p ironmlx cli::tests::scheduler_autotune_subcommand_parses_input_and_json_format -- --nocapture
```

Expected: compile failure until the CLI module exists.

- [x] **Step 3: Implement CLI module**

Create `ironmlx/src/cli/scheduler_autotune.rs` with `--input` and `--format text|json`; deserialize `SchedulerAutotuneCalibrationInput`, call `select_scheduler_autotune_profile`, and print output.

- [x] **Step 4: Verify GREEN**

Run the same parser test command. Expected: parser test passes.

### Task 3: Documentation And Final Gates

- [x] **Step 1: Update Chinese design doc**

Document stage 2 status, input schema, scoring strategy, warnings, and remaining gaps.

- [x] **Step 2: Run required Rust verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --release -p ironmlx --test scheduler_autotune_report -- --nocapture
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
cargo test --release -p ironmlx cli::tests::scheduler_autotune_subcommand_parses_input_and_json_format -- --nocapture
git diff --check
```

- [x] **Step 3: Commit**

```bash
git add ironmlx/src/core/scheduler_autotune.rs ironmlx/tests/scheduler_autotune_profile.rs ironmlx/src/cli/mod.rs ironmlx/src/cli/scheduler_autotune.rs docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md docs/superpowers/plans/2026-06-01-scheduler-autotune-profile.md
git commit -m "feat: add scheduler autotune profile selector"
```
