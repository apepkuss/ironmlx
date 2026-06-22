# TurboQuant Decode Attention Input Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the current `decode_query_turbo_inputs` cost into upstream decode attention input materialization, QKV projection, Q/K norm/reshape, and the residual MRoPE input cost.

**Architecture:** Keep production execution unchanged unless `IRONMLX_TURBOQUANT_ATTN_PROFILE` is set. Add a small JSON formatter in `gated_attention.rs`, then place profile-only `mlx::transforms::eval` probes at decode-only boundaries before the existing fused MRoPE+TurboQuant query rotation call.

**Tech Stack:** Rust, MLX lazy arrays, existing `IRONMLX_TURBOQUANT_ATTN_PROFILE` stderr JSON profile stream, existing `ironmlx-core-bench` K3V4 long-context harness.

---

### Task 1: Plan Commit

**Files:**
- Create: `docs/superpowers/plans/2026-06-13-turboquant-decode-attention-input-attribution.md`

- [ ] **Step 1: Save this plan**

Run:

```bash
git add docs/superpowers/plans/2026-06-13-turboquant-decode-attention-input-attribution.md
git commit -m "docs: plan turboquant decode attention input attribution"
```

Expected: a docs-only commit.

### Task 2: Formatter TDD

**Files:**
- Modify: `ironmlx/src/nn/gated_attention.rs`

- [ ] **Step 1: Add the failing formatter test**

Add a unit test near the existing gated-attention tests:

```rust
#[test]
fn format_decode_attention_turbo_profile_line_is_stable_json() {
    let line = format_decode_attention_turbo_profile_line(DecodeAttentionTurboProfileEvent {
        stage: "decode_qkv_proj",
        elapsed_us: 42,
        layer_idx: 3,
        batch: 1,
        seq: 1,
        q_heads: 16,
        kv_heads: 4,
        head_dim: 256,
    });

    assert_eq!(
        line,
        "{\"event\":\"turboquant_gated_attention_stage\",\"stage\":\"decode_qkv_proj\",\"elapsed_us\":42,\"layer_idx\":3,\"batch\":1,\"seq\":1,\"q_heads\":16,\"kv_heads\":4,\"head_dim\":256}"
    );
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx format_decode_attention_turbo_profile_line_is_stable_json -- --nocapture --test-threads=1
```

Expected before implementation: compile failure because `DecodeAttentionTurboProfileEvent` and `format_decode_attention_turbo_profile_line` do not exist.

- [ ] **Step 3: Add formatter implementation**

Add near the top of `gated_attention.rs`:

```rust
#[derive(Clone, Copy)]
struct DecodeAttentionTurboProfileEvent {
    stage: &'static str,
    elapsed_us: u128,
    layer_idx: i32,
    batch: i32,
    seq: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
}

fn format_decode_attention_turbo_profile_line(event: DecodeAttentionTurboProfileEvent) -> String {
    format!(
        "{{\"event\":\"turboquant_gated_attention_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"layer_idx\":{},\"batch\":{},\"seq\":{},\"q_heads\":{},\"kv_heads\":{},\"head_dim\":{}}}",
        event.stage,
        event.elapsed_us,
        event.layer_idx,
        event.batch,
        event.seq,
        event.q_heads,
        event.kv_heads,
        event.head_dim,
    )
}
```

- [ ] **Step 4: Verify GREEN**

Run the same formatter test. Expected: one test passes.

### Task 3: Profile-Only Decode Boundary Probes

**Files:**
- Modify: `ironmlx/src/nn/gated_attention.rs`

- [ ] **Step 1: Add helper function**

Add a helper that returns immediately when profiling is disabled or `seq != 1`:

```rust
fn profile_decode_attention_turbo_stage(
    stage: &'static str,
    arrays: &[&Array],
    layer_idx: i32,
    batch: i32,
    seq: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
) -> Result<()> {
    if seq != 1 || std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_none() {
        return Ok(());
    }
    let start = std::time::Instant::now();
    mlx::transforms::eval(arrays).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "{}",
        format_decode_attention_turbo_profile_line(DecodeAttentionTurboProfileEvent {
            stage,
            elapsed_us: start.elapsed().as_micros(),
            layer_idx,
            batch,
            seq,
            q_heads,
            kv_heads,
            head_dim,
        })
    );
    Ok(())
}
```

- [ ] **Step 2: Add probes to both cfg paths**


In the current decode attention path:

- after reading `batch/seq/h_q/h_kv/d`, call `profile_decode_attention_turbo_stage("decode_attention_input", &[x], ...)`
- after `q_proj/k_proj/v_proj`, call `profile_decode_attention_turbo_stage("decode_qkv_proj", &[&q_full, &k, &v], ...)`
- after `queries/k/v/gate_flat` are produced, call `profile_decode_attention_turbo_stage("decode_q_split_norm_reshape", &[&queries, &k, &v, &gate_flat], ...)`

- [ ] **Step 3: Verify functionality**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::gated_attention::tests::forward_decode_with_turboquant_cache_uses_fused_query_rotation_path -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx nn::mrope::tests::apply_decode_query_turbo_rotation_matches_apply_plus_wht -- --nocapture --test-threads=1
```

Expected: both tests pass.

### Task 4: Attribution Benchmark

**Files:**
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/summary.md`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/ctx-32k/profile/*.json`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/ctx-32k/profile/*.stderr.txt`

- [ ] **Step 1: Build profile binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --bin ironmlx-core-bench
```

Expected: build succeeds.

- [ ] **Step 2: Run profile attribution**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/ctx-32k/profile/decode-attention-input-attribution-profile-1x64.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/ctx-32k/profile/decode-attention-input-attribution-profile-1x64.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/ctx-32k/profile/decode-attention-input-attribution-profile-1x64.stderr.txt
```

Expected: JSON output has `valid_runs=1`; stderr contains `decode_attention_input`, `decode_qkv_proj`, `decode_q_split_norm_reshape`, `decode_query_turbo_inputs`, and `decode_query_turbo_rotation`.

- [ ] **Step 3: Run no-profile control**

Run the same benchmark without `IRONMLX_TURBOQUANT_ATTN_PROFILE`, with `--runs 3 --warmup-runs 1`, and save to `decode-attention-input-attribution-noprofile-3x64.json`.

Expected: `valid_runs=3`.

- [ ] **Step 4: Summarize**

Write a summary with p50/p95/mean for:

- `turboquant_gated_attention_stage:decode_attention_input`
- `turboquant_gated_attention_stage:decode_qkv_proj`
- `turboquant_gated_attention_stage:decode_q_split_norm_reshape`
- `turboquant_mrope_stage:decode_query_turbo_inputs`
- `turboquant_mrope_stage:decode_query_turbo_rotation`
- `turboquant_attn_stage:qk`
- `turboquant_attn_stage:weighted_v_chunk`

The decision rule is: optimize the largest production-relevant partition next. If `decode_attention_input` dominates, the next target is previous-layer output scheduling; if `decode_qkv_proj` dominates, inspect projection fusion/parallelism; if `decode_q_split_norm_reshape` dominates, inspect q/k norm and reshape materialization.

### Task 5: Verification and Commit

**Files:**
- Modify: `ironmlx/src/nn/gated_attention.rs`
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution/**`

- [ ] **Step 1: Required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: all commands exit 0. External `mlx-sys` C++ warnings may appear.

- [ ] **Step 2: Commit**

Run:

```bash
git add ironmlx/src/nn/gated_attention.rs docs/benchmarks/turboquant-kv/2026-06-13-decode-attention-input-attribution
git commit -m "perf: attribute turboquant decode attention inputs"
```

Expected: implementation and benchmark artifacts are committed, and `git status --short` is empty.
