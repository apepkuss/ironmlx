# MTP Phase 4 Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared MTP draft-token policy so omitted `--mtp-draft-tokens` values use Phase 3 benchmark-backed defaults.

**Architecture:** Put the policy in `ironmlx/src/core/speculative.rs`, because all MTP entry points already depend on that module. Change CLI argument fields from `usize` to `Option<usize>` so code can distinguish explicit values from omitted values. Resolve the final draft depth after the model loader has raw config metadata.

**Tech Stack:** Rust, clap, serde_json, existing ironmlx core/model modules.

---

### Task 1: Add Shared MTP Policy Helper

**Files:**
- Modify: `ironmlx/src/core/speculative.rs`

- [ ] **Step 1: Write failing policy tests**

Add tests in `ironmlx/src/core/speculative.rs` under a local `#[cfg(test)]` module:

```rust
#[test]
fn mtp_policy_defaults_qwen35_dense_4b_to_d1() {
    let raw = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 2560,
            "num_hidden_layers": 32
        }
    });

    assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
    assert_eq!(
        resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Omitted),
        1
    );
}

#[test]
fn mtp_policy_defaults_qwen36_dense_27b_to_d2() {
    let raw = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64
        }
    });

    assert_eq!(default_mtp_draft_tokens_for_config(&raw), 2);
}

#[test]
fn mtp_policy_defaults_qwen36_moe_35b_a3b_to_d2() {
    let raw = serde_json::json!({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8
        }
    });

    assert_eq!(default_mtp_draft_tokens_for_config(&raw), 2);
}

#[test]
fn mtp_policy_preserves_explicit_value() {
    let raw = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64
        }
    });

    assert_eq!(
        resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Explicit(1)),
        1
    );
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx mtp_policy_ -- --nocapture
```

Expected: fail because `default_mtp_draft_tokens_for_config`, `resolve_mtp_draft_tokens`, and `MtpDraftTokensArg` are not defined.

- [ ] **Step 3: Implement policy helper**

Add to `ironmlx/src/core/speculative.rs` near `MtpSpeculativeConfig`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDraftTokensArg {
    Explicit(usize),
    Omitted,
}

pub fn resolve_mtp_draft_tokens(
    raw_config: &serde_json::Value,
    arg: MtpDraftTokensArg,
) -> usize {
    match arg {
        MtpDraftTokensArg::Explicit(value) => value,
        MtpDraftTokensArg::Omitted => default_mtp_draft_tokens_for_config(raw_config),
    }
}

pub fn default_mtp_draft_tokens_for_config(raw_config: &serde_json::Value) -> usize {
    let model_type = raw_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let text = raw_config
        .get("text_config")
        .and_then(serde_json::Value::as_object);
    let hidden_size = text
        .and_then(|v| v.get("hidden_size"))
        .and_then(serde_json::Value::as_i64);
    let layers = text
        .and_then(|v| v.get("num_hidden_layers"))
        .and_then(serde_json::Value::as_i64);
    let experts = text
        .and_then(|v| v.get("num_experts"))
        .and_then(serde_json::Value::as_i64);
    let experts_per_tok = text
        .and_then(|v| v.get("num_experts_per_tok"))
        .and_then(serde_json::Value::as_i64);

    match (model_type, hidden_size, layers, experts, experts_per_tok) {
        ("qwen3_5", Some(5120), Some(64), None, None) => 2,
        ("qwen3_5_moe", Some(2048), Some(40), Some(256), Some(8)) => 2,
        _ => 1,
    }
}
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx mtp_policy_ -- --nocapture
```

Expected: all policy tests pass.

### Task 2: Wire Policy Through CLI And Bench Args

**Files:**
- Modify: `ironmlx/src/cli/generate.rs`
- Modify: `ironmlx/src/cli/serve.rs`
- Modify: `ironmlx/src/bin/ironmlx-core-bench.rs`

- [ ] **Step 1: Write failing parse and override tests**

Update existing tests so omitted MTP draft tokens parse as `None`, and explicit values parse as `Some(value)`.

Expected assertions:

```rust
assert_eq!(default_cli.args.mtp_draft_tokens, None);
assert_eq!(enabled_cli.args.mtp_draft_tokens, Some(6));
```

Add a serve resolver test that passes Qwen3.6 dense raw config and omitted draft tokens, then expects `draft_tokens == 2`.

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx mtp_draft_tokens serve_mtp_config_accepts_qwen36_default -- --nocapture
```

Expected: fail because argument fields are still `usize` and serve resolver has no raw-config policy input.

- [ ] **Step 3: Update argument fields and resolve final draft tokens**

Change `mtp_draft_tokens` fields to:

```rust
#[arg(long = "mtp-draft-tokens")]
pub mtp_draft_tokens: Option<usize>,
```

Use:

```rust
let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
    loader.config_raw_value(),
    args.mtp_draft_tokens
        .map(crate::core::speculative::MtpDraftTokensArg::Explicit)
        .unwrap_or(crate::core::speculative::MtpDraftTokensArg::Omitted),
);
```

For explicit validation paths, only call `MtpSpeculativeConfig::new(value, sampler)` when the option is `Some(value)`. After loading the model, construct the final config with the resolved value.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx mtp_draft_tokens serve_mtp_config_accepts_qwen36_default -- --nocapture
```

Expected: tests pass.

### Task 3: Full Verification And Benchmark

**Files:**
- Add: `docs/benchmarks/mtp-phase4-policy/<timestamp>/summary.md`
- Add: `docs/benchmarks/mtp-phase4-policy/<timestamp>/summary.csv`
- Add: raw benchmark JSON files under the same directory

- [ ] **Step 1: Run required Rust verification**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: all commands exit 0.

- [ ] **Step 2: Run policy benchmark**

Run the same fixed-prompt matrix as Phase 3, but for MTP configs omit `--mtp-draft-tokens` to verify policy defaults select `d1` for Qwen3.5-4B and `d2` for both Qwen3.6 models.

- [ ] **Step 3: Save benchmark report**

Generate `summary.csv` and `summary.md` comparing Phase 4 default-policy TPS to Phase 3 best-config TPS.

- [ ] **Step 4: Final verification**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx mtp_ -- --nocapture
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
git diff --check
```

Expected: all commands exit 0.
