# Gemma4 Drafter Active KV Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in heavy regression runner for Gemma4 target + Gemma4 assistant drafter under App daemon mode with paged prefix cache and Active KV offload.

**Architecture:** Keep `iron-bench` as the HTTP timing client and add a Python orchestration script for ironmlx-specific startup, dynamic model loading, health assertions, and report generation. The runner writes deterministic artifacts under `docs/benchmarks/gemma4-drafter-active-kv-regression/<timestamp>/` and supports `--dry-run` for machines without local checkpoints.

**Tech Stack:** Python standard library, existing release `ironmlx` and `iron-bench` binaries, App daemon HTTP admin API, `/healthz`, Rust verification gates.

## Global Constraints

- Worktree: `/Users/xin/workspace/ironmlx-backend-gemma4-drafter-active-kv-regression`
- Branch: `test/gemma4-drafter-active-kv-regression`
- Heavy real-model runs are opt-in and must not be wired into default CI.
- New runner must not change scheduler, model, or App runtime behavior.
- Default Gemma4 12B MAX TOKENS / logical cap remains `262144`.
- Rust changes, if any, require `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.

---

### Task 1: Add Failing Tests For Run Plan And Commands

**Files:**
- Create: `scripts/test_gemma4_drafter_active_kv_regression.py`
- Create: `scripts/gemma4_drafter_active_kv_regression.py`

**Interfaces:**
- Produces: `RegressionConfig`, `RegressionVariant`, `build_default_variants()`, `build_run_plan(config)`, `build_serve_command(config, variant, port, variant_dir)`, `build_load_payload(config, variant)`, and `build_bench_command(config, port, prompt_len, concurrent)`.

- [x] **Step 1: Write tests for default variants**

Create tests that import `scripts/gemma4_drafter_active_kv_regression.py` and assert default variants are `e4b_b2`, `e4b_b4`, and `12b_b2`, with `max_cache_cap=262144`, `mtp_draft_tokens=2`, and Active KV expected.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: import or attribute failure because the runner is not implemented yet.

- [x] **Step 3: Implement minimal dataclasses and command builders**

Implement enough script code for the tests to import the module, build variants, and construct commands without starting a server.

- [x] **Step 4: Run tests to verify GREEN**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: tests pass.

### Task 2: Add Health Assertion And Summary Tests

**Files:**
- Modify: `scripts/test_gemma4_drafter_active_kv_regression.py`
- Modify: `scripts/gemma4_drafter_active_kv_regression.py`

**Interfaces:**
- Produces: `assert_health_delta(variant, before, after)`, `summarize_bench_payload(...)`, `write_summary_files(out_root, rows)`, and `render_markdown(rows)`.

- [x] **Step 1: Write failing tests for health invariants**

Add tests where `before.scheduler.memory_budget_exceeded_count=1` and `after=1` passes, but `after=2` fails. Add tests that reject `active_kv_offload.degraded=true`, `swap_error_count>0`, missing MTP, wrong draft tokens, and full-resident KV budget policy.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: failures for missing assertion helpers.

- [x] **Step 3: Implement assertion and report helpers**

Implement health extraction with explicit error messages, median helpers, row status fields, and JSON/CSV/Markdown summary writers.

- [x] **Step 4: Run tests to verify GREEN**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: tests pass.

### Task 3: Add Runner Lifecycle And CLI

**Files:**
- Modify: `scripts/test_gemma4_drafter_active_kv_regression.py`
- Modify: `scripts/gemma4_drafter_active_kv_regression.py`
- Create: `docs/benchmarks/gemma4-drafter-active-kv-regression/README.md`

**Interfaces:**
- Produces: `run_regression(config)`, `parse_args(argv)`, `main(argv)`, `write_run_commands(out_root, plan)`, and documented environment variables.

- [x] **Step 1: Write failing tests for dry-run artifacts**

Add a test that runs `main(["--dry-run", "--out-root", tmp])` and asserts `run_commands.sh`, `metadata.json`, `summary.json`, `summary.csv`, and `summary.md` are written without requiring real model directories.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: failures for missing CLI/dry-run behavior.

- [x] **Step 3: Implement lifecycle**

Implement build option, process startup/termination, `/health` readiness, `/admin/api/models/load`, `iron-bench` execution, `/healthz` capture, error rows, and `--allow-failures`.

- [x] **Step 4: Document usage**

Add README with example:

```bash
python3 scripts/gemma4_drafter_active_kv_regression.py \
  --build \
  --variant 12b_b2 \
  --out-root docs/benchmarks/gemma4-drafter-active-kv-regression/manual-12b
```

- [x] **Step 5: Run tests to verify GREEN**

Run:

```bash
python3 scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: tests pass.

### Task 4: Verify And Commit

**Files:**
- Modify: all files from Tasks 1-3.

- [x] **Step 1: Run Python tests**

Run:

```bash
python3 -m unittest scripts/test_gemma4_drafter_active_kv_regression.py
```

Expected: all tests pass.

- [x] **Step 2: Run existing iron-bench regression**

Run:

```bash
cargo test -p iron-bench --release
```

Expected: all tests pass.

- [x] **Step 3: Run required Rust gates**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all pass.

- [x] **Step 4: Review diff and commit**

Run:

```bash
git diff --check
git diff --stat
git status --short
git add docs/superpowers/specs/2026-07-03-gemma4-drafter-active-kv-regression-design.md \
  docs/superpowers/plans/2026-07-03-gemma4-drafter-active-kv-regression.md \
  docs/benchmarks/gemma4-drafter-active-kv-regression/README.md \
  scripts/gemma4_drafter_active_kv_regression.py \
  scripts/test_gemma4_drafter_active_kv_regression.py
git commit -m "test(gemma4): add drafter active kv regression harness"
```

Expected: commit succeeds.
