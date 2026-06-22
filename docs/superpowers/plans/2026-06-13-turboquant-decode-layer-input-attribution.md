# TurboQuant Decode Layer Input Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute the newly exposed `decode_attention_input` cost to the dense Qwen3.5 decoder-layer stages that produce the full-attention input.

**Architecture:** Add env-gated decode-only eval probes in `ironmlx/src/nn/decoder_layer.rs`, mirroring the existing TurboQuant attention probes. Preserve the public `DecoderLayer::forward_on` API by delegating to a new internal layer-index-aware implementation, and let `Qwen35TextModel` pass real layer indices so profile events can be grouped by full/linear layer.

**Tech Stack:** Rust, MLX lazy eval, `IRONMLX_TURBOQUANT_ATTN_PROFILE`, `ironmlx-core-bench`, Cargo tests/checks.

---

### Task 1: Plan Checkpoint

**Files:**
- Create: `docs/superpowers/plans/2026-06-13-turboquant-decode-layer-input-attribution.md`

- [ ] **Step 1: Save this implementation plan**

Run:

```bash
git add docs/superpowers/plans/2026-06-13-turboquant-decode-layer-input-attribution.md
git commit -m "docs: plan turboquant decode layer input attribution"
```

Expected: a docs-only commit on `codex/turboquant-mrope-qrotate-fusion`.

### Task 2: Formatter RED Test

**Files:**
- Modify: `ironmlx/src/nn/decoder_layer.rs`

- [ ] **Step 1: Add a failing unit test**

Add this test to the existing `#[cfg(test)] mod tests` in `ironmlx/src/nn/decoder_layer.rs`:

```rust
#[test]
fn format_decode_layer_turbo_profile_line_is_stable_json() {
    let line = format_decode_layer_turbo_profile_line(DecodeLayerTurboProfileEvent {
        stage: "decode_input_norm",
        elapsed_us: 42,
        layer_idx: 7,
        attn_kind: "full",
        batch: 1,
        seq: 1,
        hidden_size: 2560,
    });

    assert_eq!(
        line,
        "{\"event\":\"turboquant_decoder_layer_stage\",\"stage\":\"decode_input_norm\",\"elapsed_us\":42,\"layer_idx\":7,\"attn_kind\":\"full\",\"batch\":1,\"seq\":1,\"hidden_size\":2560}"
    );
}
```

- [ ] **Step 2: Run RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx format_decode_layer_turbo_profile_line_is_stable_json -- --nocapture --test-threads=1
```

Expected: FAIL because `DecodeLayerTurboProfileEvent` and `format_decode_layer_turbo_profile_line` do not exist.

### Task 3: Decoder Layer Profile Helpers

**Files:**
- Modify: `ironmlx/src/nn/decoder_layer.rs`

- [ ] **Step 1: Implement the profile event, shape, formatter, and probe helper**

Add near the top of `decoder_layer.rs`, after imports:

```rust
#[derive(Clone, Copy)]
struct DecodeLayerTurboProfileEvent {
    stage: &'static str,
    elapsed_us: u128,
    layer_idx: i32,
    attn_kind: &'static str,
    batch: i32,
    seq: i32,
    hidden_size: i32,
}

#[derive(Clone, Copy)]
struct DecodeLayerTurboProfileShape {
    layer_idx: i32,
    attn_kind: &'static str,
    batch: i32,
    seq: i32,
    hidden_size: i32,
}

fn format_decode_layer_turbo_profile_line(event: DecodeLayerTurboProfileEvent) -> String {
    format!(
        "{{\"event\":\"turboquant_decoder_layer_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"layer_idx\":{},\"attn_kind\":\"{}\",\"batch\":{},\"seq\":{},\"hidden_size\":{}}}",
        event.stage,
        event.elapsed_us,
        event.layer_idx,
        event.attn_kind,
        event.batch,
        event.seq,
        event.hidden_size,
    )
}

fn profile_decode_layer_turbo_stage(
    stage: &'static str,
    arrays: &[&Array],
    shape: DecodeLayerTurboProfileShape,
) -> Result<()> {
    if shape.seq != 1 || std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_none() {
        return Ok(());
    }

    let start = std::time::Instant::now();
    mlx::transforms::eval(arrays).map_err(|e| anyhow!("{e}"))?;
    eprintln!(
        "{}",
        format_decode_layer_turbo_profile_line(DecodeLayerTurboProfileEvent {
            stage,
            elapsed_us: start.elapsed().as_micros(),
            layer_idx: shape.layer_idx,
            attn_kind: shape.attn_kind,
            batch: shape.batch,
            seq: shape.seq,
            hidden_size: shape.hidden_size,
        })
    );
    Ok(())
}
```

- [ ] **Step 2: Run GREEN for formatter**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx format_decode_layer_turbo_profile_line_is_stable_json -- --nocapture --test-threads=1
```

Expected: PASS.

### Task 4: Decode Stage Probes and Layer Indices

**Files:**
- Modify: `ironmlx/src/nn/decoder_layer.rs`
- Modify: `ironmlx/src/models/qwen3_5/text_model.rs`

- [ ] **Step 1: Preserve public API and add layer-index-aware implementation**

In `DecoderLayer::forward_on`, keep the current signature and delegate to a new private method:

```rust
self.forward_on_with_layer_idx(
    x,
    mrope,
    cos,
    sin,
    full_attn_mask,
    linear_attn_mask,
    per_row_lens,
    cache,
    target,
    -1,
)
```

Move the current body into `fn forward_on_with_layer_idx(..., layer_idx: i32) -> Result<Array>`.
Because `Qwen35TextModel` lives outside `crate::nn::decoder_layer`, the helper must be
`pub(crate) fn forward_on_with_layer_idx(...) -> Result<Array>`.

- [ ] **Step 2: Add decode-only probes inside `forward_on_with_layer_idx`**

After `preflight_x`, compute:

```rust
let dims = x.shape();
let profile_shape = DecodeLayerTurboProfileShape {
    layer_idx,
    attn_kind: match self.kind() {
        AttnKind::Full => "full",
        AttnKind::Linear => "linear",
    },
    batch: dims[0],
    seq: dims[1],
    hidden_size: dims[2],
};
profile_decode_layer_turbo_stage("decode_layer_input", &[x], profile_shape)?;
```

Then add probes after each stage:

```rust
profile_decode_layer_turbo_stage("decode_input_norm", &[&normed_in], profile_shape)?;
profile_decode_layer_turbo_stage("decode_attention_path", &[&attn], profile_shape)?;
profile_decode_layer_turbo_stage("decode_attention_residual", &[&h], profile_shape)?;
profile_decode_layer_turbo_stage("decode_post_attention_norm", &[&normed_post], profile_shape)?;
profile_decode_layer_turbo_stage("decode_mlp_path", &[&mlp_out], profile_shape)?;
profile_decode_layer_turbo_stage("decode_layer_output", &[&out], profile_shape)?;
```

Return `out` after the final probe.

- [ ] **Step 3: Pass real layer indices from Qwen35TextModel**

In `ironmlx/src/models/qwen3_5/text_model.rs`, change both layer loops in `forward_post_embedding_on` to enumerate layers and call `forward_on_with_layer_idx(..., i as i32)`.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx format_decode_layer_turbo_profile_line_is_stable_json -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx forward_decode_with_turboquant_cache_uses_packed_attention_path -- --nocapture --test-threads=1
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx forward_shape_and_dtype_bf16 -- --nocapture --test-threads=1
```

Expected: all targeted tests pass.

### Task 5: Long-context Attribution Benchmark

**Files:**
- Create directory: `docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/`

- [ ] **Step 1: Build profile bench binary**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --bin ironmlx-core-bench
```

Expected: release profile build succeeds.

- [ ] **Step 2: Run profile benchmark**

Run:

```bash
mkdir -p docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile
IRONMLX_TURBOQUANT_ATTN_PROFILE=1 MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file docs/benchmarks/turboquant-kv/2026-06-10-packed-kv-long-context/prompts/ctx-32k.txt \
  --mode gs-text \
  --max-tokens 64 \
  --runs 1 \
  --warmup-runs 1 \
  --kv-quant k3v4 \
  --out docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/decode-layer-input-attribution-profile-1x64.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/decode-layer-input-attribution-profile-1x64.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/decode-layer-input-attribution-profile-1x64.stderr.txt
```

Expected: command exits 0 and stderr contains `turboquant_decoder_layer_stage` JSON lines.

- [ ] **Step 3: Run no-profile control**

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
  --out docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/decode-layer-input-attribution-noprofile-3x64.json \
  > docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/decode-layer-input-attribution-noprofile-3x64.stdout \
  2> docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/decode-layer-input-attribution-noprofile-3x64.stderr.txt
```

Expected: 3 valid runs.

### Task 6: Summary, Verification, Commit

**Files:**
- Create: `docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution/ctx-32k/profile/summary.md`

- [ ] **Step 1: Parse profile events**

Run a JSON parser over profile stderr and compute count, mean, p50, p95, p99, min, max for:

```text
turboquant_decoder_layer_stage
turboquant_gated_attention_stage
turboquant_mrope_stage
turboquant_attn_stage
```

- [ ] **Step 2: Write summary**

Document setup, profile table, no-profile control table, and the next root-cause target.

- [ ] **Step 3: Run required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Commit**

Run:

```bash
git add ironmlx/src/nn/decoder_layer.rs ironmlx/src/models/qwen3_5/text_model.rs docs/benchmarks/turboquant-kv/2026-06-13-decode-layer-input-attribution
git commit -m "perf: attribute turboquant decode layer inputs"
```

Expected: worktree clean after commit.
