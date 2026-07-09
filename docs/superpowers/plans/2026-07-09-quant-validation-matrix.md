# Quant Validation Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible production-readiness validation matrix for the new bf16, affine 8-bit, and OptiQ quantization support.

**Architecture:** Keep the runtime implementation unchanged. Add an orchestration script that starts `ironmlx serve` for one model at a time, waits for `/health`, runs `iron-bench` sequential and concurrent HTTP benchmarks, runs direct multi-turn OpenAI-compatible requests, captures logs and machine-readable outputs, then writes a Markdown summary.

**Tech Stack:** Rust release binaries (`ironmlx`, `iron-bench`), Python 3 standard library only, existing Hugging Face cache snapshots under `~/.ironmlx/models`.

## Global Constraints

- Replies to Boss must be in Chinese and start with "Boss".
- Do not add compatibility code unless Boss explicitly requests it.
- Keep validation tooling separate from runtime quantization logic.
- If Rust code is edited, run `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.
- Use HTTP external user perspective for E2E measurements.

---

### Task 1: Add Matrix Orchestrator

**Files:**
- Create: `scripts/quant_validation_matrix.py`

**Interfaces:**
- Consumes: local model snapshot paths, `target/release/ironmlx`, `target/release/iron-bench`.
- Produces: `docs/benchmarks/quant-validation/<timestamp>/manifest.json`, per-case JSON/CSV/Markdown benchmark outputs, per-model `server.log`, and `summary.md`.

- [ ] **Step 1: Create the script**

Create `scripts/quant_validation_matrix.py` with these behaviors:

```text
1. Parse CLI flags:
   --out-root
   --port-base
   --sequential-prompt-lens
   --long-prompt-lens
   --concurrent-prompt-lens
   --concurrent
   --duration
   --warmup-duration
   --max-tokens
   --stability-runs
   --skip-build
   --model label=path

2. Build release binaries unless --skip-build is passed:
   cargo build --release -p ironmlx -p iron-bench

3. For each model:
   start target/release/ironmlx serve --model PATH --host 127.0.0.1 --port PORT --max-sequences max(concurrent, 1) --max-cache-cap 32768 --prefill-chunk-size 2048
   wait for http://127.0.0.1:PORT/health
   run iron-bench sequential HTTP E2E
   run iron-bench long-context HTTP E2E
   run iron-bench concurrent HTTP E2E for each requested concurrency level
   run direct multi-turn chat/completions requests
   run sequential stability loop
   stop server

4. Write manifest.json and summary.md.
```

- [ ] **Step 2: Validate script syntax**

Run:

```bash
python3 -m py_compile scripts/quant_validation_matrix.py
```

Expected: exit 0.

- [ ] **Step 3: Smoke argument parsing**

Run:

```bash
python3 scripts/quant_validation_matrix.py --help
```

Expected: help text exits 0 and lists `--model`.

### Task 2: Run Real Model Matrix

**Files:**
- Create: `docs/benchmarks/quant-validation/<timestamp>/...`

**Interfaces:**
- Consumes:
  - `MiniCPM5-1B-8bit=/Users/xin/.ironmlx/models/models--mlx-community--MiniCPM5-1B-8bit/snapshots/3d164befdfbe496a9e280704c42bcb34ebde9443`
  - `Qwen3.5-4B-MLX-8bit=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-8bit/snapshots/5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d`
  - `gemma-4-e2b-it-bf16=/Users/xin/.ironmlx/models/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/fb0b166bbb9a0eb4b37915bfc515a197c9122f39`
- Produces: pass/fail and latency/throughput matrix.

- [ ] **Step 1: Run the matrix**

Run:

```bash
MLX_DIR=/Users/xin/.local/mlx python3 scripts/quant_validation_matrix.py \
  --model MiniCPM5-1B-8bit=/Users/xin/.ironmlx/models/models--mlx-community--MiniCPM5-1B-8bit/snapshots/3d164befdfbe496a9e280704c42bcb34ebde9443 \
  --model Qwen3.5-4B-MLX-8bit=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-8bit/snapshots/5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d \
  --model gemma-4-e2b-it-bf16=/Users/xin/.ironmlx/models/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/fb0b166bbb9a0eb4b37915bfc515a197c9122f39
```

Expected: exit 0, each model has `ok=true` in the manifest.

- [ ] **Step 2: Inspect summary**

Open:

```bash
docs/benchmarks/quant-validation/<timestamp>/summary.md
```

Expected: summary contains rows for sequential, long-context, concurrent, multi-turn, and stability.

### Task 3: Final Quality Gate

**Files:**
- Verify only.

**Interfaces:**
- Consumes: branch source tree and generated benchmark artifacts.
- Produces: final evidence for Boss.

- [ ] **Step 1: Format and lint**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/xin/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=/Users/xin/.local/mlx cargo build --release
```

Expected: all exit 0.

- [ ] **Step 2: Diff check**

Run:

```bash
git diff --check
```

Expected: exit 0.

### Task 4: Extend Admission Matrix to 8K/32K and Higher Concurrency

**Files:**
- Modify: `scripts/quant_validation_matrix.py`
- Modify: `iron-bench/src/report.rs`
- Create: `scripts/test_quant_validation_matrix.py`
- Create: `docs/benchmarks/quant-validation/<timestamp>/...`

**Interfaces:**
- Consumes: the same four real model snapshots used by the base quant validation run.
- Produces: a single manifest and summary that include 8K/32K long-context rows, multiple concurrent worker levels, and non-null concurrent `e2e_s_p95` values.

- [ ] **Step 1: Write the failing Python test**

Run:

```bash
python3 scripts/test_quant_validation_matrix.py
```

Expected before implementation: FAIL because `--concurrent 4,8` is rejected.

- [ ] **Step 2: Write the failing Rust report test**

Run:

```bash
cargo test -p iron-bench report::tests::render_json_concurrent_exports_e2e_tail_latency --quiet
```

Expected before implementation: FAIL because concurrent JSON cells do not export `e2e_s_p95`.

- [ ] **Step 3: Implement the matrix/report changes**

Required behavior:

```text
--concurrent accepts comma-separated worker levels, for example 4,8.
--concurrent-prompt-lens accepts comma-separated prompt lengths for concurrent cells.
ironmlx serve starts with --max-sequences equal to the maximum requested concurrency.
Each concurrency level writes a separate concurrent_c<N>.json artifact.
manifest.json records each concurrent_c<N> check independently.
summary.csv and summary.md include real concurrent e2e_s_p95 values from iron-bench JSON.
```

- [ ] **Step 4: Run the extended matrix**

Run:

```bash
MLX_DIR=/Users/xin/.local/mlx python3 scripts/quant_validation_matrix.py \
  --long-prompt-lens 8192,32768 \
  --concurrent-prompt-lens 8192,32768 \
  --concurrent 4,8 \
  --duration 30 \
  --warmup-duration 5 \
  --request-timeout 1800 \
  --startup-timeout 900 \
  --serve-max-cache-cap 65536 \
  --model MiniCPM5-1B-8bit=/Users/xin/.ironmlx/models/models--mlx-community--MiniCPM5-1B-8bit/snapshots/3d164befdfbe496a9e280704c42bcb34ebde9443 \
  --model Qwen3.5-4B-MLX-8bit=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-8bit/snapshots/5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d \
  --model gemma-4-e2b-it-bf16=/Users/xin/.ironmlx/models/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/fb0b166bbb9a0eb4b37915bfc515a197c9122f39 \
  --model gemma-4-e4b-it-OptiQ-4bit=/Users/xin/.ironmlx/models/models--mlx-community--gemma-4-e4b-it-OptiQ-4bit/snapshots/6ffaa01fb83dcd8cb1d743a41fb45b4a1430503b
```

Expected: exit 0, each model has `ok=true`, and summary rows exist for PP 8192 and 32768 plus concurrency 4 and 8.
