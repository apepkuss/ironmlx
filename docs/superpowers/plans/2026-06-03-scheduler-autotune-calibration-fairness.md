# Scheduler Autotune Calibration Fairness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce deterministic candidate-order bias in `ironmlx scheduler-autotune calibrate` and record the actual run order for diagnosis.

**Architecture:** Build an explicit candidate benchmark plan before execution. Run by concurrency level and mirror candidate order across concurrency levels so the same candidate is not always earliest or latest. Write a run-order manifest before benchmarks start.

**Tech Stack:** Rust, `clap`, `serde_json`, existing `ironmlx` CLI tests.

---

### Task 1: Candidate Run Plan

**Files:**
- Modify: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`

- [x] **Step 1: Write failing test**

Test `candidate_benchmark_plan_mirrors_candidate_order_across_concurrency_levels` asserts candidate order `(0,1,2)` for C1 and `(2,1,0)` for C2.

- [x] **Step 2: Run test to verify it fails**

Run:

```sh
MLX_DIR=/Users/xin/.local/mlx cargo test -p ironmlx cli::scheduler_autotune_calibrate::tests::candidate_benchmark_plan_mirrors_candidate_order_across_concurrency_levels -- --nocapture
```

Expected: FAIL because `build_candidate_benchmark_plan` does not exist.

- [x] **Step 3: Implement minimal plan builder**

Add `CandidateBenchmarkJob` and `build_candidate_benchmark_plan`.

- [x] **Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 2: Execute Plan And Manifest

**Files:**
- Modify: `ironmlx/src/cli/scheduler_autotune_calibrate.rs`

- [x] **Step 1: Write failing manifest test**

Test run-order manifest JSON includes strategy, ordinal, candidate index, concurrency, config, and artifact paths.

- [x] **Step 2: Run test to verify it fails**

Run targeted manifest test and expect FAIL before implementation.

- [x] **Step 3: Execute jobs from the plan**

Replace candidate-major loop with plan-order execution. Use per-candidate/per-concurrency serve logs.

- [x] **Step 4: Write run-order manifest**

Write `run-order.json` before executing jobs.

- [x] **Step 5: Run targeted tests**

Run scheduler autotune calibrate unit tests and expect PASS.

### Task 3: Verification

**Files:**
- Modify: `reports/scheduler-autotune-glm47-current-tg128-2026-06-03/README.md`

- [x] **Step 1: Update report**

Document that the next code change addresses candidate-order bias with mirrored concurrency-major scheduling and run-order manifest.

- [x] **Step 2: Full Rust checks**

Run:

```sh
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/xin/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=/Users/xin/.local/mlx cargo build --release
```

- [x] **Step 3: Commit**

Commit implementation and tests.
