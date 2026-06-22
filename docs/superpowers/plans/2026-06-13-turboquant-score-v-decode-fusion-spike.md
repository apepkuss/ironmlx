# TurboQuant Score V Decode Fusion Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate whether keeping the retained parallel QK kernel while replacing `softmax -> weighted_v_chunk -> weighted_v_reduce` with a score-driven chunked softmax/V path improves 32k K3V4 decode throughput.

**Architecture:** Keep `TURBOQUANT_QK_DECODE_SOURCE` unchanged. Add an experimental long-sequence path that consumes QK scores directly: chunk stats computes per-chunk max/denom, score-weighted V chunk accumulates unnormalized V partials from scores without materializing the full weights tensor, and softmax reduce combines chunk partials with global softmax normalization before inverse WHT/sign recovery. Guard the path to K3V4 `head_dim=256`, `seq_len >= 256`; retain it only if the no-profile 32k benchmark beats the retained baseline by at least 3%.

**Tech Stack:** Rust, MLX custom Metal kernels, Cargo tests, `ironmlx-core-bench`, TurboQuant K3V4 packed KV cache.

---

### Task 1: TDD Scope Gate

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`

- [ ] **Step 1: Write the failing gate test**

Add a test near the existing TurboQuant tests:

```rust
#[test]
fn score_v_decode_candidate_gate_matches_target_scope() {
    assert!(should_use_turboquant_score_v_decode(3, 4, 256, 256));
    assert!(!should_use_turboquant_score_v_decode(4, 4, 256, 256));
    assert!(!should_use_turboquant_score_v_decode(3, 3, 256, 256));
    assert!(!should_use_turboquant_score_v_decode(3, 4, 128, 256));
    assert!(!should_use_turboquant_score_v_decode(3, 4, 256, 255));
}
```

- [ ] **Step 2: Verify red**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx score_v_decode_candidate_gate_matches_target_scope -- --nocapture --test-threads=1
```

Expected: FAIL because `should_use_turboquant_score_v_decode` is not defined.

- [ ] **Step 3: Implement minimal gate helper**

Add near the decode constants:

```rust
fn should_use_turboquant_score_v_decode(k_bits: u8, v_bits: u8, head_dim: i32, seq_len: i32) -> bool {
    k_bits == 3 && v_bits == 4 && head_dim == 256 && seq_len >= PARALLEL_DECODE_V_CHUNK_SIZE
}
```

- [ ] **Step 4: Verify green**

Run the same test. Expected: PASS.

### Task 2: TDD Numeric Parity Harness

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`

- [ ] **Step 1: Write failing parity test**

Add a deterministic test that quantizes K/V, computes the expected output with the existing pre-rotated path using a zero mask to bypass the candidate gate, and calls the new private candidate dispatch directly:

```rust
#[test]
fn turboquant_score_v_decode_matches_existing_parallel_k3v4_head256() {
    let batch = 1_i32;
    let q_heads = 2_i32;
    let kv_heads = 1_i32;
    let seq_len = 256_i32;
    let head_dim = 256_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let k_values_per_word = 10_i32;
    let v_values_per_word = 8_i32;
    let k_packed_dim = (head_dim + k_values_per_word - 1) / k_values_per_word;
    let v_packed_dim = (head_dim + v_values_per_word - 1) / v_values_per_word;

    let k_data: Vec<f32> = (0..(batch * kv_heads * seq_len * head_dim))
        .map(|i| ((i as f32) * 0.017).sin() * 0.8)
        .collect();
    let v_data: Vec<f32> = (0..(batch * kv_heads * seq_len * head_dim))
        .map(|i| ((i as f32) * 0.023).cos() * 0.9)
        .collect();
    let q_rot_data: Vec<f32> = (0..(batch * q_heads * head_dim))
        .map(|i| ((i as f32) * 0.031).sin() * 0.7)
        .collect();
    let k: Array = (k_data.as_slice(), &[batch, kv_heads, seq_len, head_dim][..]).try_into().unwrap();
    let v: Array = (v_data.as_slice(), &[batch, kv_heads, seq_len, head_dim][..]).try_into().unwrap();
    let q_rot: Array = (q_rot_data.as_slice(), &[batch, q_heads, head_dim][..]).try_into().unwrap();

    let k_sign_values = turboquant::wht::generate_signs(head_dim as usize, 0x5455_5242_4f51_5541_u64);
    let v_sign_values = turboquant::wht::generate_signs(head_dim as usize, 0x5455_5242_4f51_5541_u64);
    let k_signs: Array = (k_sign_values.as_slice(), &[head_dim][..]).try_into().unwrap();
    let v_signs: Array = (v_sign_values.as_slice(), &[head_dim][..]).try_into().unwrap();
    let k_codebook_values = turboquant::codebook::Codebook::new(k_bits, head_dim as usize);
    let v_codebook_values = turboquant::codebook::Codebook::new(v_bits, head_dim as usize);
    let k_codebook: Array = (k_codebook_values.centroids.as_slice(), &[k_codebook_values.centroids.len() as i32][..]).try_into().unwrap();
    let v_codebook: Array = (v_codebook_values.centroids.as_slice(), &[v_codebook_values.centroids.len() as i32][..]).try_into().unwrap();

    let (k_packed, k_norms) = turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) = turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let zero_mask_data = vec![0.0_f32; (batch * q_heads * seq_len) as usize];
    let zero_mask: Array = (zero_mask_data.as_slice(), &[batch, q_heads, 1_i32, seq_len][..]).try_into().unwrap();
    let scale = (head_dim as f32).sqrt().recip();

    let expected = turboquant_sdpa_decode_parallel_pre_rotated(
        &q_rot, &k_packed, &k_norms, &v_packed, &v_norms, &k_codebook, &v_signs, &v_codebook,
        scale, k_bits, v_bits, Some(&zero_mask), Dtype::Float32,
    )
    .expect("existing masked parallel path");

    let actual = turboquant_sdpa_decode_parallel_dispatch_rotated_score_v(
        &q_rot, &k_packed, &k_norms, &v_packed, &v_norms, &k_codebook, &v_signs, &v_codebook,
        scale, k_bits, v_bits, Dtype::Float32, (), batch, q_heads, kv_heads, seq_len, head_dim,
        q_heads / kv_heads, k_values_per_word, v_values_per_word, k_packed_dim, v_packed_dim,
    )
    .expect("score-v path");

    assert_eq!(actual.shape().as_slice(), expected.shape().as_slice());
    let actual = actual.to_vec::<f32>().unwrap();
    let expected = expected.to_vec::<f32>().unwrap();
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((actual - expected).abs() < 2.0e-3, "idx={idx} actual={actual} expected={expected}");
    }
}
```

- [ ] **Step 2: Verify red**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_score_v_decode_matches_existing_parallel_k3v4_head256 -- --nocapture --test-threads=1
```

Expected: FAIL because `turboquant_sdpa_decode_parallel_dispatch_rotated_score_v` is not defined.

### Task 3: Implement Score-Driven Chunked Softmax/V Candidate

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`

- [ ] **Step 1: Add Metal sources**

Add three sources after `TURBOQUANT_WEIGHTED_V_REDUCE_SOURCE`:

- `TURBOQUANT_SCORE_CHUNK_STATS_SOURCE`: input `scores`; outputs `chunk_max`, `chunk_denom`; one threadgroup per `(batch, q_head, chunk)` with `V_CHUNK_SIZE=256`.
- `TURBOQUANT_SCORE_WEIGHTED_V_CHUNK_SOURCE`: inputs `scores`, `chunk_max`, `v_packed`, `v_norms`, `v_codebook`; output `v_partial`; same V dim-group layout as retained weighted V chunk, but uses `exp(score - chunk_max)` instead of reading materialized weights.
- `TURBOQUANT_SCORE_WEIGHTED_V_REDUCE_SOURCE`: inputs `chunk_max`, `chunk_denom`, `v_partial`, `v_signs`; output `out`; computes global softmax scale across chunks and then inverse WHT/sign recovery.

- [ ] **Step 2: Add cached kernel builders**

Add builders:

```rust
fn cached_turboquant_score_chunk_stats_kernel() -> Result<&'static MetalKernel>
fn cached_turboquant_score_weighted_v_chunk_kernel() -> Result<&'static MetalKernel>
fn cached_turboquant_score_weighted_v_reduce_kernel() -> Result<&'static MetalKernel>
```

- [ ] **Step 3: Add private dispatch**

Implement:

```rust
fn turboquant_sdpa_decode_parallel_dispatch_rotated_score_v(...) -> Result<Array>
```

The function should run QK exactly as the retained dispatch does, then run chunk stats, score-weighted V chunk, and score-weighted V reduce.

- [ ] **Step 4: Verify parity green**

Run the parity test from Task 2. Expected: PASS.

### Task 4: Add Routing and Profile Stages

**Files:**
- Modify: `mlx/src/fast/turboquant.rs`

- [ ] **Step 1: Route unprofiled long K3V4 decode**

At the start of `turboquant_sdpa_decode_parallel_dispatch_rotated`, if `should_use_turboquant_score_v_decode(k_bits, v_bits, head_dim, seq_len)` is true, return the score-V candidate.

- [ ] **Step 2: Add profiled dispatch**

Add a profiled variant or profiling branch with stage names:

- `qk`
- `score_chunk_stats`
- `score_weighted_v_chunk`
- `score_weighted_v_reduce`

- [ ] **Step 3: Run targeted tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx score_v_decode_candidate_gate_matches_target_scope -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p mlx turboquant_score_v_decode_matches_existing_parallel_k3v4_head256 -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx kvcache_turboquant_decode_attention_uses_packed_path -- --nocapture --test-threads=1
```

Expected: all PASS.

### Task 5: Benchmark Gate and Documentation

**Files:**
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/core-bench/score-v-k3v4-3x64.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/score-v-k3v4-profile-1x16.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/score-v-k3v4-profile-1x16.stderr.txt`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/summary.md`

- [ ] **Step 1: Build benchmark binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --bin ironmlx-core-bench
```

- [ ] **Step 2: Run no-profile benchmark**

Run:

```bash
mkdir -p docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/core-bench docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 3 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/core-bench/score-v-k3v4-3x64.json
```

Retention threshold: compare against `docs/benchmarks/turboquant-kv/2026-06-13-qk-kernel-opt/ctx-32k/core-bench/baseline-qksg4-k3v4-3x64.json`; retain code only if candidate p50 generation TPS is at least `67.7396`.

- [ ] **Step 3: Run profile benchmark**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/score-v-k3v4-profile-1x16.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/score-v-k3v4-profile-1x16.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike/ctx-32k/profile/score-v-k3v4-profile-1x16.stderr.txt
```

- [ ] **Step 4: Retain or revert**

If the candidate misses the retention threshold, remove all candidate code and tests from `mlx/src/fast/turboquant.rs`, keep benchmark evidence, and document the rejection. If it passes, keep code and tests.

### Task 6: Final Verification and Commit

**Files:**
- Modify or create according to retention decision.

- [ ] **Step 1: Run required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
git diff --check
```

- [ ] **Step 2: Commit final result**

If retained:

```bash
git add mlx/src/fast/turboquant.rs docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike
git commit -m "perf: add turboquant score-v decode fusion"
```

If rejected:

```bash
git add docs/benchmarks/turboquant-kv/2026-06-13-score-v-decode-fusion-spike
git commit -m "perf: evaluate turboquant score-v decode fusion"
```
