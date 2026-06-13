# TurboQuant Fused Decode Attention Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a restricted fused TurboQuant long-context decode attention path for the current K3V4 32K target.

**Architecture:** Keep the existing packed parallel path as the fallback. Add a candidate path only for `K3V4`, `head_dim=128`, `seq_len>=256`, and no attention mask: one Metal kernel computes per-chunk QK, local softmax stats, and V numerators; a second Metal kernel reduces chunks with global softmax normalization and the existing inverse WHT/sign recovery.

**Tech Stack:** Rust, MLX custom Metal kernels, TurboQuant packed KV cache, `ironmlx-core-bench`.

---

## Files

- Modify: `mlx/src/fast/turboquant.rs`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/core-bench/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/*.stderr.txt`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/summary.md`

## Task 1: Guard the restricted candidate selection

- [ ] **Step 1: Add a failing helper test**

In `mlx/src/fast/turboquant.rs`, add a unit test that calls a private helper named `should_use_turboquant_fused_parallel_decode` and expects:

```rust
assert!(should_use_turboquant_fused_parallel_decode(3, 4, 128, 256, false));
assert!(!should_use_turboquant_fused_parallel_decode(4, 4, 128, 256, false));
assert!(!should_use_turboquant_fused_parallel_decode(3, 4, 64, 256, false));
assert!(!should_use_turboquant_fused_parallel_decode(3, 4, 128, 255, false));
assert!(!should_use_turboquant_fused_parallel_decode(3, 4, 128, 256, true));
```

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx fused_parallel_decode_candidate_gate_matches_target_scope -- --nocapture --test-threads=1
```

Expected: FAIL because the helper does not exist yet.

- [ ] **Step 2: Implement the helper**

Add the helper with exact target gating:

```rust
fn should_use_turboquant_fused_parallel_decode(
    k_bits: u8,
    v_bits: u8,
    head_dim: i32,
    seq_len: i32,
    has_mask: bool,
) -> bool {
    k_bits == 3
        && v_bits == 4
        && head_dim == 128
        && seq_len >= PARALLEL_DECODE_V_CHUNK_SIZE
        && !has_mask
}
```

Run the same test again. Expected: PASS.

## Task 2: Add the fused candidate kernels

- [ ] **Step 1: Add failing parity coverage**

Add a unit test in `mlx/src/fast/turboquant.rs` that quantizes deterministic `K3V4` tensors with `head_dim=128`, calls the existing explicit parallel decode function, calls the new fused rotated dispatch through the gated path, and compares every output element with tolerance `2.0e-3`.

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_fused_parallel_decode_matches_existing_parallel_k3v4_head128 -- --nocapture --test-threads=1
```

Expected: FAIL until the fused kernels and dispatch are implemented.

- [ ] **Step 2: Add chunk and reduce kernels**

Add `TURBOQUANT_FUSED_PARALLEL_CHUNK_SOURCE` and `TURBOQUANT_FUSED_PARALLEL_REDUCE_SOURCE`.

The chunk kernel must write:

- `chunk_max`: `[B, Hq, chunks]`
- `chunk_denom`: `[B, Hq, chunks]`
- `v_partial`: `[B, Hq, chunks, D]`

The reduce kernel must combine chunk stats with:

```text
global_max = max(chunk_max)
global_denom = sum(chunk_denom * exp(chunk_max - global_max))
output_dim = sum(v_partial_dim * exp(chunk_max - global_max)) / global_denom
```

Then it must apply the existing inverse WHT and `v_signs` recovery.

- [ ] **Step 3: Add cached kernel builders**

Add `cached_turboquant_fused_parallel_chunk_kernel` and `cached_turboquant_fused_parallel_reduce_kernel`.

- [ ] **Step 4: Wire the gated path**

In both profiled and unprofiled rotated parallel dispatch, route supported inputs through the fused candidate. Profile stage names:

- `fused_qk_softmax_v_chunk`
- `fused_v_reduce`

Unsupported inputs must continue through the existing QK, softmax, weighted-V chunk, weighted-V reduce path.

- [ ] **Step 5: Verify parity**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_fused_parallel_decode_matches_existing_parallel_k3v4_head128 -- --nocapture --test-threads=1
```

Expected: PASS.

## Task 3: Benchmark and retention decision

- [ ] **Step 1: Build benchmark binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --features p5h-profile --bin ironmlx-core-bench
```

Expected: exit `0`.

- [ ] **Step 2: Run 32K K3V4 no-profile candidate**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/core-bench/fused-k3v4-3x64.json
```

Expected: JSON summary has `valid_runs = 3`.

- [ ] **Step 3: Run 32K K3V4 profile candidate**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/fused-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/fused-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/fused-k3v4-profile-1x16.stderr.txt
```

Expected: profile stderr contains `fused_qk_softmax_v_chunk` and `fused_v_reduce`.

- [ ] **Step 4: Decide retention**

Use the latest committed qksg4 baseline from `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/core-bench/baseline-qksg4-k3v4-3x64.json`.

Keep fused code only if:

```text
candidate_generation_tps_p50 >= baseline_generation_tps_p50 * 1.03
```

If the candidate misses the gate, revert all code changes and keep only benchmark evidence.

- [ ] **Step 5: Write summary**

Create `docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike/ctx-32k/profile/summary.md` with setup, baseline comparison, profile table, and final decision.

## Task 4: Required verification and commit

- [ ] **Step 1: Run Rust verification**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx kvcache_turboquant_decode_attention_uses_packed_path -- --nocapture --test-threads=1
git diff --check
```

Expected: all commands exit `0`. Existing upstream MLX C++ header warnings are acceptable only if commands exit `0`.

- [ ] **Step 2: Commit**

If fused code is retained:

```bash
git add mlx/src/fast/turboquant.rs docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike
git commit -m "perf: add turboquant fused decode attention spike"
```

If fused code is rejected:

```bash
git add docs/benchmarks/turboquant-kv/2026-06-13-fused-decode-attention-spike
git commit -m "perf: evaluate turboquant fused decode attention"
```
