# TurboQuant QK Kernel Opt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune the TurboQuant packed decode QK kernel after the retained weighted-V chunk optimization.

**Architecture:** Keep the current packed decode attention pipeline intact and change only the QK kernel's score packing density per Metal threadgroup. Use 32K K3V4 no-profile generation throughput as the retention gate, with attention profile output used to explain the result.

**Tech Stack:** Rust, MLX custom Metal kernels, `ironmlx-core-bench`, TurboQuant packed K/V cache.

---

## Context

Current retained baseline:

- Branch: `codex/turboquant-mrope-qrotate-fusion`
- Last optimization: `V_CHUNK_DIMS_PER_THREADGROUP = 16`
- Latest 32K K3V4 no-profile p50 generation throughput: `65.034 TPS`
- Latest profile means:
  - `qk`: `745.313 us`
  - `softmax`: `186.988 us`
  - `weighted_v_chunk`: `687.779 us`
  - `weighted_v_reduce`: `191.229 us`

The next target is QK because it is now the largest remaining packed-attention stage.

## Files

- Modify: `mlx/src/fast/turboquant.rs`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/core-bench/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/*.stderr.txt`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/summary.md`

## Task 1: Baseline control and profile

- [ ] **Step 1: Build the profile benchmark binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: exit `0`.

- [ ] **Step 2: Run baseline no-profile control**

Run:

```bash
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 3 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/core-bench/baseline-qksg4-k3v4-3x64.json
```

Expected: JSON summary has `valid_runs = 3`.

- [ ] **Step 3: Run baseline attention profile**

Run:

```bash
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 16 \
  --runs 1 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/baseline-qksg4-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/baseline-qksg4-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/baseline-qksg4-k3v4-profile-1x16.stderr.txt
```

Expected: profile stderr contains `turboquant_attn_stage` events for `qk`, `softmax`, `weighted_v_chunk`, and `weighted_v_reduce`.

## Task 2: Try QK simdgroup packing variant

- [ ] **Step 1: Add a failing guard test for the first candidate**

In `mlx/src/fast/turboquant.rs`, add a unit test that expects `QK_SIMDGROUPS_PER_THREADGROUP == 8`.

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx qk_simdgroup_tuning_constant_matches_candidate -- --nocapture --test-threads=1
```

Expected: FAIL with `left: 4 right: 8`.

- [ ] **Step 2: Change the QK simdgroup constant to 8**

In `mlx/src/fast/turboquant.rs`, change:

```rust
const QK_SIMDGROUPS_PER_THREADGROUP: i32 = 4;
```

to:

```rust
const QK_SIMDGROUPS_PER_THREADGROUP: i32 = 8;
```

- [ ] **Step 3: Verify the guard test passes**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx qk_simdgroup_tuning_constant_matches_candidate -- --nocapture --test-threads=1
```

Expected: PASS.

- [ ] **Step 4: Rebuild release profile benchmark**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: exit `0`.

- [ ] **Step 5: Run candidate no-profile control**

Run:

```bash
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 3 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/core-bench/qksg8-k3v4-3x64.json
```

Expected: JSON summary has `valid_runs = 3`.

- [ ] **Step 6: Run candidate attention profile**

Run:

```bash
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 16 \
  --runs 1 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/qksg8-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/qksg8-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/qksg8-k3v4-profile-1x16.stderr.txt
```

Expected: profile stderr contains QK events.

## Task 3: Decision and verification

- [ ] **Step 1: Compare no-profile p50 TPS**

Keep `QK_SIMDGROUPS_PER_THREADGROUP = 8` only if:

```text
candidate_generation_tps_p50 >= baseline_generation_tps_p50 * 1.03
```

and the attention profile does not show a QK regression that contradicts the no-profile gain.

- [ ] **Step 2: Revert rejected variant if needed**

If the candidate misses the gate, revert `QK_SIMDGROUPS_PER_THREADGROUP` to `4` and remove the candidate guard test with `apply_patch`.

- [ ] **Step 3: Remove zero-byte stdout files**

Run:

```bash
find docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt -type f -size 0 -delete
```

Expected: no zero-byte files remain in the experiment directory.

- [ ] **Step 4: Write summary**

Create `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/profile/summary.md` with setup, control table, profile table, and decision.

- [ ] **Step 5: Run required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
git diff --check
```

Expected: all commands exit `0`. Existing upstream MLX C++ header warnings are acceptable only if the commands exit `0`.

- [ ] **Step 6: Commit**

If retained:

```bash
git add mlx/src/fast/turboquant.rs docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt
git commit -m "perf: tune turboquant qk decode kernel"
```

If rejected:

```bash
git add docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt
git commit -m "perf: evaluate turboquant qk decode kernel"
```
