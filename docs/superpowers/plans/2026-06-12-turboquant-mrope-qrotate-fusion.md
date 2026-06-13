# TurboQuant MRoPE Q-Rotate Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the standalone TurboQuant decode `q_rotate` dispatch on the long-context packed decode path by fusing query TurboQuant rotation into decode-time MRoPE.

**Architecture:** Add a decode-only `Mrope` fused kernel that emits `(q_turbo_rotated, k_rope)` for `S=1`, then add a TurboQuant packed decode entry point that accepts pre-rotated queries and starts at QK. `GatedAttention` uses the fused path only when the existing TurboQuant decode predicate plus the parallel decode threshold are both satisfied; all other paths keep the current behavior.

**Tech Stack:** Rust workspace, MLX custom Metal kernels, TurboQuant packed K/V cache, existing `mlx/tests/turboquant_fast.rs` and `ironmlx/src/nn/*` unit tests.

---

### Task 1: Pre-Rotated TurboQuant Decode API

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`
- Test: `mlx/tests/turboquant_fast.rs`

- [ ] **Step 1: Write failing API equivalence test**

Add a test named `turboquant_sdpa_decode_parallel_pre_rotated_matches_regular_parallel` that:
- builds `[B=1,Hq=2,S=513,D=64]` packed K/V as the existing parallel test does;
- computes `q_rot` in CPU test code with the same `signs * WHT / sqrt(D)` transform;
- calls `mlx::fast::turboquant_sdpa_decode_parallel_pre_rotated(...)`;
- compares against `mlx::fast::turboquant_sdpa_decode_parallel(...)`.

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test turboquant_fast turboquant_sdpa_decode_parallel_pre_rotated_matches_regular_parallel -- --nocapture --test-threads=1
```

Expected before implementation: compile failure because `turboquant_sdpa_decode_parallel_pre_rotated` does not exist.

- [ ] **Step 2: Implement minimal pre-rotated API**

In `mlx/src/fast/turboquant.rs`:
- expose `pub const TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD: i32 = 128`;
- add `turboquant_sdpa_decode_parallel_pre_rotated` and `_on`;
- validate `q_rot` as `[B,Hq,D]`;
- reuse existing packed K/V, codebook, mask, and output validation;
- factor the QK/softmax/V code so the pre-rotated path skips only the query-rotate dispatch.

- [ ] **Step 3: Verify green**

Run the same test command and confirm the test passes.

### Task 2: Fused MRoPE Query Turbo Rotation

**Files:**
- Modify: `ironmlx/src/nn/mrope.rs`

- [ ] **Step 1: Write failing fused-MRoPE test**

Add a test named `apply_decode_query_turbo_rotation_matches_apply_plus_wht` that:
- uses `Mrope::new(8, 10000.0, 1.0, &[2, 1, 1], true)`;
- builds `q: [1,2,1,8]`, `k: [1,1,1,8]`, non-trivial `cos/sin`;
- calls the new `apply_decode_query_turbo_rotation(&q, &k, &cos, &sin, &signs)`;
- compares `k_rot` with `apply(&q,&k,...)`'s K output;
- computes expected `q_turbo` by applying existing `apply` first, then CPU WHT with signs.

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::mrope::tests::apply_decode_query_turbo_rotation_matches_apply_plus_wht -- --nocapture --test-threads=1
```

Expected before implementation: compile failure because `apply_decode_query_turbo_rotation` does not exist.

- [ ] **Step 2: Implement decode fused kernel**

In `Mrope`:
- add a new `OnceLock<MetalKernel>` field for the fused decode kernel;
- implement `apply_decode_query_turbo_rotation`;
- reject non-interleaved configs, non-rank-4 inputs, non-`S=1`, wrong `cos/sin` shape or dtype, and wrong `signs` shape;
- use one threadgroup per Q or K vector with `HEAD_DIM` threads;
- Q groups apply MRoPE then TurboQuant sign/WHT rotation into `[B,Hq,D]`;
- K groups apply only MRoPE into `[B,Hkv,1,D]`.

- [ ] **Step 3: Verify green**

Run the same `cargo test -p ironmlx ...apply_decode...` command and confirm the test passes.

### Task 3: Thread Fused Path Through KVCache and GatedAttention

**Files:**
- Modify: `ironmlx/src/core/cache/kv_cache.rs`
- Modify: `ironmlx/src/core/cache/turboquant_kv.rs`
- Modify: `ironmlx/src/nn/gated_attention.rs`

- [ ] **Step 1: Write failing integration test**

Extend `forward_decode_with_turboquant_cache_uses_packed_attention_path` or add a sibling test that sets a new test-only/profile env flag and proves the decode path can use pre-rotated queries. The test should verify shape, dtype, cache offsets, and that fallback still works for the prefix call.

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::gated_attention::tests::forward_decode_with_turboquant_cache_uses_fused_query_rotation_path -- --nocapture --test-threads=1
```

Expected before implementation: compile failure or assertion failure because no fused path marker exists.

- [ ] **Step 2: Add cache eligibility/accessors**

Add narrow accessors:
- `TurboQuantKVCache::key_signs(&self) -> &Array`
- `TurboQuantKVCache::can_use_pre_rotated_decode(seq_after_update: i32) -> bool`
- `KVCache::turboquant_fused_query_decode_signs(...) -> Option<&Array>`
- `KVCache::try_update_and_attend_decode_pre_rotated_on(...)`

The predicate must include the existing TurboQuant decode checks and require
`seq_after_update >= mlx::fast::TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD`.

- [ ] **Step 3: Wire GatedAttention**

In both `p5h-profile` and non-profile branches:
- after q/k/v reshape and q/k norm, ask the cache for fused decode signs;
- if available, call `mrope.apply_decode_query_turbo_rotation`;
- route the pre-rotated query into `try_update_and_attend_decode_pre_rotated_on`;
- fall back to the existing `mrope.apply` plus `try_update_and_attend_decode_on` path when unavailable.

- [ ] **Step 4: Verify green**

Run the new `gated_attention` test and the existing packed path test.

### Task 4: Full Verification and Benchmark

**Files:**
- Create: `docs/benchmarks/turboquant-kv/2026-06-12-mrope-qrotate-fusion/summary.md`
- Add benchmark JSON/profile artifacts under that directory.

- [ ] **Step 1: Rust checks**

Run:

```bash
cargo fmt
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

- [ ] **Step 2: Targeted correctness**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test turboquant_fast -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::mrope::tests::apply_decode_query_turbo_rotation_matches_apply_plus_wht -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::gated_attention::tests::forward_decode_with_turboquant_cache_uses_fused_query_rotation_path -- --nocapture --test-threads=1
```

- [ ] **Step 3: 32k/K3V4 profile**

Build `ironmlx-core-bench`, run the same 32k/K3V4 profile as the prior
`2026-06-12-qrotate-fusion-opt` baseline, and save stdout/stderr/JSON under the
new benchmark directory.

- [ ] **Step 4: Retain or revert**

Retain implementation only if the profile has no standalone `q_rotate` stage on
the fused path, correctness passes, and decode time shows a meaningful win. If
the fused MRoPE dispatch erases most of the expected gain, revert production
code and keep the benchmark record.
