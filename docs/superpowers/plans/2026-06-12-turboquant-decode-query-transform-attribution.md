# TurboQuant Decode Query Transform Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the profiled cost of TurboQuant decode query transform into upstream input materialization and the fused MRoPE + WHT kernel itself.

**Architecture:** Keep production execution unchanged unless a measured no-profile improvement appears. Under `IRONMLX_TURBOQUANT_ATTN_PROFILE`, materialize and report the inputs to `Mrope::apply_decode_query_turbo_rotation` before dispatching the fused kernel, then report the fused kernel output materialization separately.

**Tech Stack:** Rust, MLX custom Metal kernels, `ironmlx/src/nn/mrope.rs`, JSON-line stderr profile events, existing 32k/K3V4 benchmark harness.

---

### Task 1: Profile Event Shape

**Files:**
- Modify: `ironmlx/src/nn/mrope.rs`

- [ ] **Step 1: Write the failing formatter test**

Add a unit test beside `format_decode_query_turbo_profile_line_is_stable_json`:

```rust
#[test]
fn format_decode_query_turbo_profile_line_supports_input_stage() {
    let line = format_decode_query_turbo_profile_line(DecodeQueryTurboProfileEvent {
        stage: "decode_query_turbo_inputs",
        elapsed_us: 77,
        batch: 1,
        q_heads: 16,
        kv_heads: 4,
        head_dim: 256,
        rot_dim: 64,
    });

    assert_eq!(
        line,
        "{\"event\":\"turboquant_mrope_stage\",\"stage\":\"decode_query_turbo_inputs\",\"elapsed_us\":77,\"batch\":1,\"q_heads\":16,\"kv_heads\":4,\"head_dim\":256,\"rot_dim\":64}"
    );
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx format_decode_query_turbo_profile_line_supports_input_stage -- --nocapture --test-threads=1
```

Expected: compilation fails because `DecodeQueryTurboProfileEvent` has no `stage` field.

- [ ] **Step 3: Add a `stage` field to the profile event**

Change the private event struct and formatter:

```rust
#[derive(Clone, Copy)]
struct DecodeQueryTurboProfileEvent {
    stage: &'static str,
    elapsed_us: u128,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rot_dim: i32,
}

fn format_decode_query_turbo_profile_line(event: DecodeQueryTurboProfileEvent) -> String {
    format!(
        "{{\"event\":\"turboquant_mrope_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"batch\":{},\"q_heads\":{},\"kv_heads\":{},\"head_dim\":{},\"rot_dim\":{}}}",
        event.stage,
        event.elapsed_us,
        event.batch,
        event.q_heads,
        event.kv_heads,
        event.head_dim,
        event.rot_dim
    )
}
```

- [ ] **Step 4: Update the existing formatter test**

Set `stage: "decode_query_turbo_rotation"` in the existing event literal so the original JSON contract remains stable.

- [ ] **Step 5: Run formatter tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx format_decode_query_turbo_profile_line -- --nocapture --test-threads=1
```

Expected: both formatter tests pass.

### Task 2: Split Input and Kernel Attribution

**Files:**
- Modify: `ironmlx/src/nn/mrope.rs`

- [ ] **Step 1: Add input materialization before fused dispatch**

Inside `Mrope::apply_decode_query_turbo_rotation`, after validation and before retrieving/dispatching `decode_query_turbo_kernel`, add:

```rust
let profile_enabled = std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_some();
if profile_enabled {
    let start = Instant::now();
    mlx::transforms::eval(&[q, k, cos, sin, query_signs]).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "{}",
        format_decode_query_turbo_profile_line(DecodeQueryTurboProfileEvent {
            stage: "decode_query_turbo_inputs",
            elapsed_us: start.elapsed().as_micros(),
            batch: b,
            q_heads: hq,
            kv_heads: hkv,
            head_dim: self.head_dim,
            rot_dim: self.rot_dim,
        })
    );
}
```

- [ ] **Step 2: Measure the fused kernel output separately**

Replace the existing `profile_start` construction with:

```rust
let profile_start = profile_enabled.then(Instant::now);
```

When emitting the output event, include:

```rust
stage: "decode_query_turbo_rotation",
```

- [ ] **Step 3: Run focused correctness tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::mrope::tests::apply_decode_query_turbo_rotation_matches_apply_plus_wht -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx core::cache::kv_cache::tests::kvcache_turboquant_pre_rotated_decode_matches_regular_decode -- --nocapture --test-threads=1
```

Expected: both pass.

### Task 3: K3V4 32k Profile

**Files:**
- Create: `docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution/summary.md`
- Create: `docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution/ctx-32k/profile/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution/ctx-32k/profile/*.stderr.txt`

- [ ] **Step 1: Build the profile benchmark binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --bin ironmlx-core-bench
```

Expected: exit 0.

- [ ] **Step 2: Run 32k/K3V4 attribution profile**

Run:

```bash
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 1 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution/ctx-32k/profile/decode-query-transform-attribution-profile-1x64.json \
  > docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution/ctx-32k/profile/decode-query-transform-attribution-profile-1x64.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution/ctx-32k/profile/decode-query-transform-attribution-profile-1x64.stderr.txt
```

Expected: JSON output has `valid_runs=1`; stderr contains both `decode_query_turbo_inputs` and `decode_query_turbo_rotation`.

- [ ] **Step 3: Summarize stage metrics**

Run a small Python parser over the stderr JSON lines and record `count`, `p50`, `p95`, and `mean` for:

- `turboquant_mrope_stage:decode_query_turbo_inputs`
- `turboquant_mrope_stage:decode_query_turbo_rotation`
- `turboquant_attn_stage:qk`
- `turboquant_attn_stage:softmax`
- `turboquant_attn_stage:weighted_v_chunk`
- `turboquant_attn_stage:weighted_v_reduce`

- [ ] **Step 4: Decide whether to optimize production code**

If `decode_query_turbo_rotation` remains near the previous broad `929us`, inspect the fused Metal kernel for WHT/barrier optimizations before changing code. If the kernel event drops sharply and `decode_query_turbo_inputs` dominates, do not change the kernel; document that the bottleneck belongs to upstream lazy Q/K materialization.

### Task 4: Verification and Commit

**Files:**
- Modify/create files from Tasks 1-3.

- [ ] **Step 1: Run required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: all exit 0.

- [ ] **Step 2: Commit**

Run:

```bash
git add ironmlx/src/nn/mrope.rs docs/benchmarks/turboquant-kv/2026-06-12-decode-query-transform-attribution
git commit -m "perf: split turboquant decode query transform profile"
```
