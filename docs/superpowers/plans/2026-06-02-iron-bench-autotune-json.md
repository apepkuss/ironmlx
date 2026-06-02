# iron-bench Autotune JSON 导出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `iron-bench` 能把一次 benchmark 结果导出为 `ironmlx scheduler-autotune` 可直接读取的 calibration JSON。

**Architecture:** 保持 `iron-bench` engine-neutral，不依赖 `ironmlx` crate。由 CLI 显式传入本次被测的 scheduler config，report 层把顺序/并发统计映射为 calibration measurements，profile 选择仍由 `ironmlx scheduler-autotune` 完成。

**Tech Stack:** Rust, clap, serde_json, existing `iron-bench` report/runner structs.

---

## File Structure

- Modify `iron-bench/src/report.rs` — add autotune export option/config structs and sequential/concurrent JSON renderers.
- Modify `iron-bench/src/main.rs` — add `autotune-json` output format and config flags, validate target/config constraints, dispatch renderer.
- Modify `docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md` — document the bench-to-autotune bridge.

### Task 1: Autotune JSON Renderer

- [x] **Step 1: Write failing report tests**

Add report unit tests that call `render_autotune_json_sequential` and `render_autotune_json_concurrent` before those APIs exist. Tests must assert:

- top-level `schema_version`, `model_name`, `hardware_label`;
- one measurement per benchmark cell;
- config fields match the explicit scheduler config;
- sequential mode exports `concurrency=1`;
- concurrent mode exports the actual worker count;
- cached-token warnings propagate.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p iron-bench autotune_json -- --nocapture
```

Expected: compile failure for missing autotune renderer/config APIs.

- [x] **Step 3: Implement renderer**

Add:

- `AutotuneProfileConfig`;
- `AutotuneExportOptions`;
- `render_autotune_json_sequential`;
- `render_autotune_json_concurrent`.

Sequential mapping:

- `prompt_len = pp_target`;
- `max_new_tokens = tg_target`;
- `concurrency = 1`;
- `ttft_ms_p95 = CellStats.ttft_ms_p95`;
- `itl_ms_p95 = CellStats.tpot_ms_p95`;
- `e2e_s_p95 = CellStats.e2e_s_p95`;
- `tokens_per_sec = CellStats.tg_tps_median`.

Concurrent mapping:

- `prompt_len = pp_target`;
- `max_new_tokens = tg_target`;
- `concurrency = ConcurrentCellStats.concurrent`;
- `ttft_ms_p95 = ConcurrentCellStats.ttft_ms_p95`;
- `itl_ms_p95 = ConcurrentCellStats.itl_ms_p95`;
- `e2e_s_p95 = ConcurrentCellStats.e2e_s_p95`;
- `tokens_per_sec = ConcurrentCellStats.agg_tokens_per_sec`.

- [x] **Step 4: Verify GREEN**

Run the same test command. Expected: autotune report tests pass.

### Task 2: CLI Wiring

- [x] **Step 1: Write failing CLI parser test**

Add a parser test for:

```bash
iron-bench \
  --target ironmlx=http://localhost:8080 \
  --model-dir /tmp/model \
  --format autotune-json \
  --autotune-b-max 2 \
  --autotune-prefill-chunk-size 1024 \
  --autotune-admission-deadline-ms 5 \
  --autotune-admission-queue-max 32 \
  --autotune-max-cache-cap 32768
```

The test should assert that `OutputFormat::AutotuneJson` parses and all config fields are present. `hardware_label` is generated automatically when the optional `--autotune-hardware-label` override is omitted.

- [x] **Step 2: Verify RED**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo test --release -p iron-bench autotune_cli -- --nocapture
```

Expected: compile failure until the new enum variant and CLI fields exist.

- [x] **Step 3: Implement CLI validation and dispatch**

Add `autotune-json` to `OutputFormat`, add explicit `--autotune-*` flags, and validate:

- `--format autotune-json` requires exactly one target;
- `--format autotune-json` requires all config fields and generates `hardware_label` when no override is provided;
- non-autotune formats keep their existing output unchanged.

- [x] **Step 4: Verify GREEN**

Run the same CLI test command. Expected: parser test passes.

### Task 3: Docs, Verification, Commit

- [x] **Step 1: Update Chinese design doc**

Document how to run one candidate, save JSON, combine multiple candidates, and feed the result into:

```bash
cargo run --release -p ironmlx -- scheduler-autotune --input calibration.json --format text
```

- [x] **Step 2: Run required Rust verification**

Run:

```bash
source ~/.local/mlx/mlx-env.sh
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --release -p iron-bench autotune -- --nocapture
cargo test --release -p ironmlx --test scheduler_autotune_profile -- --nocapture
git diff --check
```

- [x] **Step 3: Commit**

```bash
git add iron-bench/src/report.rs iron-bench/src/main.rs docs/superpowers/specs/2026-06-01-scheduler-autotune-research.md docs/superpowers/plans/2026-06-02-iron-bench-autotune-json.md
git commit -m "feat: export iron-bench autotune calibration json"
```
