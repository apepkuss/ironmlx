# TurboQuant Packed Attention QK Block Opt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a TurboQuant packed-attention QK kernel variant that computes multiple adjacent sequence positions per simdgroup to reuse the rotated query vector.

**Architecture:** Keep the existing packed decode attention pipeline intact. Change only the QK score kernel and its dispatch geometry, then retain the change only if 32K K3V4 no-profile throughput improves without breaking packed-attention correctness.

**Tech Stack:** Rust, MLX custom Metal kernels, TurboQuant packed KV cache, `ironmlx-core-bench`, Cargo tests/checks.

---

## Context

Current retained packed attention state on `codex/scheduler-autotune-v2`:

- `QK_SIMDGROUPS_PER_THREADGROUP = 4`.
- `V_CHUNK_DIMS_PER_THREADGROUP = 16`.
- Previous rejected QK candidate: `QK_SIMDGROUPS_PER_THREADGROUP = 8`.
- Previous qksg8 result: profiled QK mean improved, but no-profile p50 TPS regressed, so the next QK experiment must change the kernel work shape rather than only the number of simdgroups per threadgroup.

Root-cause hypothesis:

- The current QK kernel maps one simdgroup to one `(batch, q_head, pos)` score.
- For a fixed `(batch, q_head)`, each score repeatedly reads the same `q_rot[q_head, dim]` values from global memory.
- A block-position kernel can compute multiple adjacent positions per simdgroup, load each query lane value once, and accumulate several scores.

## Files

- Modify: `mlx/src/fast/turboquant.rs`
- Create: `docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/core-bench/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/*.stderr.txt`
- Create: `docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/summary.md`

## Task 1: Baseline and Correctness Control

- [ ] **Step 1: Commit this plan**

```bash
git add docs/superpowers/plans/2026-06-14-turboquant-packed-attn-qk-block-opt.md
git commit -m "docs: plan turboquant packed qk block opt"
```

- [ ] **Step 2: Run packed-attention correctness baseline**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_sdpa_decode_parallel_pre_rotated_matches_regular_parallel -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx forward_decode_with_turboquant_cache_uses_packed_attention_path -- --nocapture --test-threads=1
```

Expected: both commands pass.

- [ ] **Step 3: Build the release benchmark binary**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: exit `0`.

- [ ] **Step 4: Run current no-profile 32K K3V4 control**

```bash
mkdir -p docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/core-bench
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 3 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/core-bench/baseline-k3v4-3x64.json
```

Expected: JSON summary has `valid_runs = 3`.

- [ ] **Step 5: Run current attention profile**

```bash
mkdir -p docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 16 \
  --runs 1 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/baseline-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/baseline-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/baseline-k3v4-profile-1x16.stderr.txt
```

Expected: profile stderr contains `qk`, `softmax`, `weighted_v_chunk`, and `weighted_v_reduce`.

## Task 2: QK Position-Block Candidate

- [ ] **Step 1: Add a failing tuning guard**

Add this test to `mlx/src/fast/turboquant.rs`:

```rust
#[test]
fn qk_positions_per_simdgroup_tuning_constant_matches_candidate() {
    assert_eq!(QK_POSITIONS_PER_SIMDGROUP, 2);
}
```

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx qk_positions_per_simdgroup_tuning_constant_matches_candidate -- --nocapture --test-threads=1
```

Expected: RED failure because `QK_POSITIONS_PER_SIMDGROUP` does not exist yet.

- [ ] **Step 2: Implement the block-position QK kernel**

In `mlx/src/fast/turboquant.rs`:

- Add `const QK_POSITIONS_PER_SIMDGROUP: i32 = 2;`.
- Change QK dispatch from total-score tiles to `(batch * q_heads * ceil(seq_len / QK_POSITIONS_PER_SIMDGROUP))` block tiles.
- Change `TURBOQUANT_QK_DECODE_SOURCE` so each simdgroup computes two adjacent `pos` scores for the same `(batch, q_head)`, reusing each lane's `q_rot` value while accumulating `acc[QK_POSITIONS_PER_SIMDGROUP]`.
- Preserve mask handling and output score layout `[B, Hq, 1, S]`.

- [ ] **Step 3: Run GREEN for the tuning guard**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx qk_positions_per_simdgroup_tuning_constant_matches_candidate -- --nocapture --test-threads=1
```

Expected: PASS.

- [ ] **Step 4: Run correctness tests**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_sdpa_decode_parallel_pre_rotated_matches_regular_parallel -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx forward_decode_with_turboquant_cache_uses_packed_attention_path -- --nocapture --test-threads=1
```

Expected: both commands pass.

## Task 3: Benchmark and Decision

- [ ] **Step 1: Rebuild release benchmark**

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: exit `0`.

- [ ] **Step 2: Run qkpos2 no-profile control**

```bash
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 3 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/core-bench/qkpos2-k3v4-3x64.json
```

Expected: JSON summary has `valid_runs = 3`.

- [ ] **Step 3: Run qkpos2 attention profile**

```bash
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 16 \
  --runs 1 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/qkpos2-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/qkpos2-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/qkpos2-k3v4-profile-1x16.stderr.txt
```

Expected: profile stderr contains QK events.

- [ ] **Step 4: Keep or reject qkpos2**

Keep the code only if:

```text
candidate_generation_tps_p50 >= baseline_generation_tps_p50 * 1.03
```

and correctness tests pass. Otherwise revert the code and keep only the benchmark evidence.

- [ ] **Step 5: Write summary and clean artifacts**

Create `docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt/ctx-32k/profile/summary.md` with control results, profile results, and decision.

Run:

```bash
find docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt -type f -name '*.stdout' -size 0 -delete
git diff --check
```

Expected: no whitespace errors.

## Task 4: Required Rust Verification and Commit

- [ ] **Step 1: Run required Rust checks**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: all commands exit `0`.

- [ ] **Step 2: Commit result**

If retained:

```bash
git add mlx/src/fast/turboquant.rs docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt
git commit -m "perf: optimize turboquant qk block decode"
```

If rejected:

```bash
git add docs/benchmarks/turboquant-kv/2026-06-14-packed-attn-qk-block-opt
git commit -m "perf: evaluate turboquant qk block decode"
```

- [ ] **Step 3: Push branch**

```bash
git push -u origin codex/turboquant-packed-attn-qk-opt
```

Expected: branch is available on remote. Do not create a PR.
