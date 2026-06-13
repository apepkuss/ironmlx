# TurboQuant Packed Attention Kernel Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve or reject the next packed TurboQuant attention kernel optimization with 32K `k3v4` evidence.

**Architecture:** Treat packed attention as a measured kernel experiment, not a speculative rewrite. First capture the current 32K `k3v4` profile/control baseline, then test one focused weighted-V kernel variant at a time, keep only a variant that improves no-profile decode throughput and preserves existing correctness checks.

**Tech Stack:** Rust, MLX custom Metal kernels, TurboQuant packed KV cache, `ironmlx-core-bench`, Cargo tests/checks.

---

### Task 1: Baseline Checkpoint

**Files:**
- Create: `docs/superpowers/plans/2026-06-13-turboquant-packed-attention-kernel-opt.md`

- [ ] **Step 1: Save this implementation plan**

Run:

```bash
git add docs/superpowers/plans/2026-06-13-turboquant-packed-attention-kernel-opt.md
git commit -m "docs: plan turboquant packed attention kernel opt"
```

Expected: a docs-only commit on `codex/turboquant-mrope-qrotate-fusion`.

### Task 2: Current Baseline Profile

**Files:**
- Create directory: `docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/`
- Create directory: `docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/core-bench/`

- [ ] **Step 1: Build the profile bench binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: release profile bench build succeeds.

- [ ] **Step 2: Run current no-profile 32K `k3v4` control**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/core-bench/baseline-k3v4-3x64.json
```

Expected: all runs valid; this is the true performance baseline.

- [ ] **Step 3: Run current 32K `k3v4` profile sample**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/baseline-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/baseline-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/baseline-k3v4-profile-1x16.stderr.txt
```

Expected: profile JSON and stderr trace exist; `weighted_v_chunk` and `qk` remain the largest packed-attention stages.

### Task 3: TDD Guard for Weighted-V Kernel Tuning

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`

- [ ] **Step 1: Add a failing unit test**

Add this test to the existing `#[cfg(test)] mod tests` in `mlx/src/fast/turboquant.rs`:

```rust
#[test]
fn weighted_v_dim_group_tuning_constant_matches_retained_variant() {
    assert_eq!(V_CHUNK_DIMS_PER_THREADGROUP, 16);
}
```

- [ ] **Step 2: Run RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx weighted_v_dim_group_tuning_constant_matches_retained_variant -- --nocapture --test-threads=1
```

Expected: FAIL while the current retained constant is still `8`.

### Task 4: Weighted-V Dim-Group Variant

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`

- [ ] **Step 1: Change the weighted-V dim-group constant**

Change:

```rust
const V_CHUNK_DIMS_PER_THREADGROUP: i32 = 8;
```

to:

```rust
const V_CHUNK_DIMS_PER_THREADGROUP: i32 = 16;
```

- [ ] **Step 2: Run GREEN for the tuning guard**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx weighted_v_dim_group_tuning_constant_matches_retained_variant -- --nocapture --test-threads=1
```

Expected: PASS.

- [ ] **Step 3: Run packed attention correctness regression**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx kvcache_turboquant_decode_attention_uses_packed_path -- --nocapture --test-threads=1
```

Expected: PASS; packed decode output remains close to dense reference and does not allocate dense K/V.

### Task 5: Variant Benchmark and Decision

**Files:**
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/summary.md`
- Create benchmark JSON/stderr artifacts under `docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/`

- [ ] **Step 1: Rebuild profile bench after the kernel change**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: build succeeds.

- [ ] **Step 2: Run variant no-profile 32K `k3v4` control**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/core-bench/vchunk-vdim16-k3v4-3x64.json
```

Expected: all runs valid; compare decode p50 and generation p50 TPS against baseline.

- [ ] **Step 3: Run variant profile sample**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/vchunk-vdim16-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/vchunk-vdim16-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt/ctx-32k/profile/vchunk-vdim16-k3v4-profile-1x16.stderr.txt
```

Expected: profile trace shows whether `weighted_v_chunk` improves enough without pushing cost into `weighted_v_reduce` or `qk`.

- [ ] **Step 4: Keep or revert the variant**

Decision rule:

- Keep `V_CHUNK_DIMS_PER_THREADGROUP = 16` only if no-profile generation p50 TPS improves by at least 3% over baseline and correctness tests pass.
- Revert the constant and remove the tuning test if the variant is flat or slower.

- [ ] **Step 5: Write benchmark summary**

Create `summary.md` with setup, baseline table, variant table, profile table, and the keep/reject decision.

### Task 6: Required Rust Verification and Commit

**Files:**
- Stage only files changed by this plan.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
```

Expected: both commands succeed.

- [ ] **Step 2: Clippy**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
```

Expected: command exits 0.

- [ ] **Step 3: Release build**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: command exits 0.

- [ ] **Step 4: Diff whitespace check**

Run:

```bash
git diff --check
```

Expected: command exits 0.

- [ ] **Step 5: Commit retained result**

Run:

```bash
git add mlx/src/fast/turboquant.rs docs/benchmarks/turboquant-kv/2026-06-13-packed-attn-kernel-opt
git commit -m "perf: tune turboquant weighted v chunk kernel"
```

Expected: final implementation and benchmark artifacts are committed. If the variant is rejected, commit only the benchmark summary/artifacts with message `perf: evaluate turboquant weighted v chunk dim group`.
