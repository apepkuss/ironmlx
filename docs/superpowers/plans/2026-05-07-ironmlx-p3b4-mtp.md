# ironmlx P3b4 — Multi-Token Prediction (MTP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MTP model-layer for Qwen3.5 / Qwen3-Next — `nn::DecoderLayer` (full-attention only, reused by future P4 main text model) + `nn::Mtp` head + `core::cache::MtpCache`. Numerically-equivalent to vllm-mlx's `_MTPModule.mtp_forward` minus tied lm_head (caller's responsibility).

**Architecture:** Three new ironmlx components (`nn::DecoderLayer`, `core::cache::MtpCache`, `nn::Mtp`) wired through existing P1/P2/P3b1/P3b2 building blocks (`Linear`, `RmsNorm`, `Mlp`, `KVCache`, `Mrope`, `GatedAttention`). DecoderLayer's full-attention path covers MTP's only need (`layer_idx = fa_idx`); the linear-attention SSM branch is a future additive change in P4. Mtp returns post-norm hidden state (caller projects to logits via tied `Embedding::as_linear`).

**Tech Stack:** Rust 2021, ironmlx (`anyhow::Result`), cxx-mlx (`mlx::ops::shape::concatenate_on`), P1 nn (Linear/Mlp/RmsNorm/Embedding), P2 KVCache, P3b1 Mrope, P3b2 GatedAttention. Python fixture via `mlx.core` + numpy. **Spec:** [`docs/superpowers/specs/2026-05-07-ironmlx-p3b4-mtp-design.md`](../specs/2026-05-07-ironmlx-p3b4-mtp-design.md).

---

## Conventions Recap

- **TDD per step**: failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before each commit** (`.claude/CLAUDE.md`):
  ```
  cargo fmt
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  cargo build --release
  ```
- **`MLX_DIR=$HOME/.local/mlx`** required for tests that exercise MLX FFI / GPU.
- **MLX source location**: `/Volumes/Dev/mlx`. mlx-lm source: `/Volumes/Dev/mlx-lm`. vllm-mlx source: `/Volumes/Dev/vllm-mlx`.
- **ironmlx error type**: `anyhow::{Error, Result}` re-exported as `crate::{Error, Result}`. Use `anyhow::anyhow!(...)`.
- **ASCII commit messages.**
- **Test seam pattern**: production constructors are `from_loader`; test-only constructors are `pub + #[doc(hidden)] fn from_components(...)` so integration tests in `ironmlx/tests/` (compiled as external crates) can use them.
- **`pre-flight validation` pattern** (P3b3 stability hardening precedent): each public `forward_on` validates rank + shape + cache invariants before dispatching MLX ops; explicit bounds beat trusting callers.

---

## File Structure (after P3b4)

```
ironmlx/
├── src/
│   ├── core/
│   │   └── cache/
│   │       ├── mtp_cache.rs                         # NEW — MtpCache (Vec<KVCache> wrapper)
│   │       └── mod.rs                               # MODIFIED — pub mod mtp_cache + re-export
│   └── nn/
│       ├── decoder_layer.rs                         # NEW — DecoderLayer (full-attn) + Config
│       ├── mlp.rs                                   # MODIFIED — add `from_components` test seam
│       ├── mtp.rs                                   # NEW — Mtp + MtpConfig
│       └── mod.rs                                   # MODIFIED — pub mod decoder_layer + mtp + re-exports
└── tests/
    ├── fixtures/
    │   └── p3b4_mtp/                                # NEW
    │       ├── README.md
    │       ├── gen_fixture.py
    │       ├── input_hidden.npy                     # [1, 4, 32] bf16
    │       ├── input_next_embeds.npy                # [1, 4, 32] bf16
    │       ├── input_position_ids.npy               # [3, 1, 4] i32
    │       ├── input_inv_freq.npy                   # [4] fp32
    │       ├── pre_fc_norm_hidden_weight.npy        # [32] fp32
    │       ├── pre_fc_norm_embedding_weight.npy     # [32] fp32
    │       ├── fc_weight.npy                        # [32, 64] bf16
    │       ├── layer0_input_layernorm_weight.npy    # [32] fp32
    │       ├── layer0_q_proj_weight.npy             # [64, 32] bf16
    │       ├── layer0_k_proj_weight.npy             # [16, 32] bf16
    │       ├── layer0_v_proj_weight.npy             # [16, 32] bf16
    │       ├── layer0_o_proj_weight.npy             # [32, 32] bf16
    │       ├── layer0_q_norm_weight.npy             # [8] fp32
    │       ├── layer0_k_norm_weight.npy             # [8] fp32
    │       ├── layer0_post_attention_layernorm_weight.npy   # [32] fp32
    │       ├── layer0_mlp_gate_proj_weight.npy      # [64, 32] bf16
    │       ├── layer0_mlp_up_proj_weight.npy        # [64, 32] bf16
    │       ├── layer0_mlp_down_proj_weight.npy      # [32, 64] bf16
    │       ├── norm_weight.npy                      # [32] fp32
    │       ├── expected_cos.npy                     # [1, 4, 4] fp32
    │       ├── expected_sin.npy                     # [1, 4, 4] fp32
    │       └── expected_mtp_out.npy                 # [1, 4, 32] (output dtype = fp32; see Task 4 § Note on dtype)
    └── p3b4_mtp.rs                                  # NEW
```

---

## Task 1: `nn::DecoderLayer` (full-attention only) + `Mlp::from_components` test seam

**Files:**
- Modify: `ironmlx/src/nn/mlp.rs` (add `from_components` test seam)
- Create: `ironmlx/src/nn/decoder_layer.rs`
- Modify: `ironmlx/src/nn/mod.rs` (`pub mod decoder_layer` + re-export `DecoderLayer`, `DecoderLayerConfig`)

### Goal

Create `DecoderLayer` — one Qwen3-Next-style decoder block in full-attention configuration: `input_layernorm → GatedAttention → +residual → post_attention_layernorm → Mlp → +residual`. Mirrors mlx-lm `Qwen3NextDecoderLayer.__call__` (`is_linear=False` branch), and is reused unchanged by both `nn::Mtp` (this phase) and the future P4 main text model (with a future enum field gating linear-attn variant).

### Steps

- [ ] **Step 1.1: Add `Mlp::from_components` test seam**

Edit [`ironmlx/src/nn/mlp.rs`](../../../ironmlx/src/nn/mlp.rs). After `from_loader`, insert:

```rust
    /// Test/composition seam: build an `Mlp` from pre-built sub-projections.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(gate: Linear, up: Linear, down: Linear) -> Self {
        Self { gate, up, down }
    }
```

- [ ] **Step 1.2: Run check, verify it compiles**

```
cargo +nightly fmt --all -- --check && cargo build --release -p ironmlx
```

Expected: clean build. Skip running tests (no behavior change).

- [ ] **Step 1.3: Write failing construction test for DecoderLayer**

Create `ironmlx/src/nn/decoder_layer.rs`:

```rust
//! Single Qwen3.5 / Qwen3-Next decoder block (full-attention path only).
//!
//! Mirrors mlx-lm `Qwen3NextDecoderLayer.__call__` (`is_linear=False` branch):
//!
//! ```text
//! r   = self_attn(input_layernorm(x), mask, cache)
//! h   = x + r
//! out = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! Reused by both [`crate::nn::Mtp`] and (in P4) the main Qwen3.5 text model.
//! The linear-attention SSM branch (Qwen3-Next's `is_linear=True`) will be
//! folded in additively — most likely as an `enum` field — when P4 lands.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{GatedAttention, GatedAttentionConfig, Mlp, Mrope, RmsNorm};
use crate::Result;

/// Configuration for [`DecoderLayer`]. Mirrors the subset of Qwen3-Next
/// `ModelArgs` that drives a single full-attention decoder block.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
}

/// One full-attention decoder block.
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: GatedAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    cfg: DecoderLayerConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    use crate::nn::Linear;

    fn rand_w(shape: &[i32], dtype: Dtype) -> Array {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0123).sin()).collect();
        let arr: Array = (data.as_slice(), shape).try_into().unwrap();
        mlx::ops::cast::astype(&arr, dtype).unwrap()
    }

    fn ones_w(dim: i32) -> Array {
        mlx::ops::constructors::ones((dim,), Dtype::Float32).unwrap()
    }

    fn small_cfg() -> DecoderLayerConfig {
        DecoderLayerConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        }
    }

    fn build_decoder_layer(cfg: DecoderLayerConfig) -> DecoderLayer {
        // Random small weights — only structural / shape behavior is validated here.
        let q_w = rand_w(&[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size], Dtype::Bfloat16);
        let k_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Bfloat16);
        let v_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Bfloat16);
        let o_w = rand_w(&[cfg.hidden_size, cfg.num_heads * cfg.head_dim], Dtype::Bfloat16);

        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );

        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let down_w = rand_w(&[cfg.hidden_size, cfg.intermediate_size], Dtype::Bfloat16);

        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w, None),
        );

        DecoderLayer::from_components(
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            mlp,
            cfg,
        )
    }

    #[test]
    fn from_components_carries_config() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg);
        let kept = layer.config();
        assert_eq!(kept.hidden_size, cfg.hidden_size);
        assert_eq!(kept.intermediate_size, cfg.intermediate_size);
        assert_eq!(kept.num_heads, cfg.num_heads);
        assert_eq!(kept.num_kv_heads, cfg.num_kv_heads);
        assert_eq!(kept.head_dim, cfg.head_dim);
    }
}
```

- [ ] **Step 1.4: Run, verify it fails to compile**

```
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --tests
```

Expected: errors like `cannot find struct DecoderLayer in this scope`, `from_components` not found, etc.

- [ ] **Step 1.5: Implement `DecoderLayer::from_components` and `config`**

Append to `ironmlx/src/nn/decoder_layer.rs`:

```rust
impl DecoderLayer {
    /// Test/composition seam: build a `DecoderLayer` from pre-built sub-modules.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(
        input_layernorm: RmsNorm,
        self_attn: GatedAttention,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        cfg: DecoderLayerConfig,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            cfg,
        }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &DecoderLayerConfig {
        &self.cfg
    }
}
```

- [ ] **Step 1.6: Wire `DecoderLayer` into `nn::mod.rs`**

Edit [`ironmlx/src/nn/mod.rs`](../../../ironmlx/src/nn/mod.rs). Add `pub mod decoder_layer;` (alphabetical: between `conv` and `embedding`) and add to the `pub use ...` block: `pub use decoder_layer::{DecoderLayer, DecoderLayerConfig};`.

- [ ] **Step 1.7: Run unit test, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer::tests::from_components_carries_config
```

Expected: `1 passed`.

- [ ] **Step 1.8: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 1.9: Commit**

```bash
git add ironmlx/src/nn/decoder_layer.rs ironmlx/src/nn/mlp.rs ironmlx/src/nn/mod.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p3b4): DecoderLayer full-attn scaffold + Mlp::from_components

Adds the bare struct + DecoderLayerConfig + from_components test seam
+ config accessor. forward / forward_on land in the next steps. Also
exposes Mlp::from_components so DecoderLayer (and downstream Mtp) can
be assembled in tests without writing safetensors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 1.10: Write failing forward shape + dtype test (fp32)**

Add to the `tests` module in `ironmlx/src/nn/decoder_layer.rs`:

```rust
    fn build_inputs_fp32(cfg: DecoderLayerConfig) -> (Array, Mrope, Array, Array) {
        // Synthesize fp32 inputs to exercise forward shape/dtype path.
        let b = 1_i32;
        let s = 4_i32;
        let n_streams = 3_i32;

        // x: [B, S, H] fp32 random.
        let x = rand_w(&[b, s, cfg.hidden_size], Dtype::Float32);

        // Mrope with full rotary (partial=1.0) over head_dim=8 → rot_dim=8 → half=4 → sections=[2,1,1].
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();

        // Build position_ids = broadcast of arange(s) across n_streams + batch.
        let pos1d = mlx::ops::constructors::arange(0.0, s as f64, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, s)).unwrap();
        let position_ids =
            mlx::ops::shape::broadcast_to_on(&pos1d, &[n_streams, b, s], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();
        (x, mrope, cos, sin)
    }

    #[test]
    fn forward_shape_and_dtype_fp32() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg);
        let (x, mrope, cos, sin) = build_inputs_fp32(cfg);
        let out = layer.forward(&x, &mrope, &cos, &sin, None, None).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, 4, cfg.hidden_size]);
        // RmsNorm with fp32 weight + bf16 attn weight → fp32 promotes; final residual
        // sums fp32 + fp32 → fp32. Dtype is fp32 even though attn weights are bf16.
        assert_eq!(out.dtype(), Dtype::Float32);
    }
```

- [ ] **Step 1.11: Run, verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer::tests::forward_shape_and_dtype_fp32
```

Expected: compile error (`forward` not defined on DecoderLayer).

- [ ] **Step 1.12: Implement `DecoderLayer::forward` + `forward_on`**

Append to `impl DecoderLayer` in `ironmlx/src/nn/decoder_layer.rs`:

```rust
    /// Default-stream forward pass. See [`forward_on`](Self::forward_on).
    pub fn forward(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        self.forward_on(x, mrope, cos, sin, mask, cache, ())
    }

    /// Stream-targeted forward. `x: [B, S, hidden_size]` → `[B, S, hidden_size]`.
    ///
    /// Computes (mlx-lm `Qwen3NextDecoderLayer.__call__` is_linear=False):
    ///
    /// ```text
    /// r   = self_attn(input_layernorm(x), mask, cache)
    /// h   = x + r
    /// out = h + mlp(post_attention_layernorm(h))
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Pre-flight validation (production-grade stability — explicit bounds > trust caller).
        if x.ndim() != 3 {
            return Err(anyhow!(
                "DecoderLayer::forward_on: x must be rank-3 [B, S, hidden_size], got rank {}",
                x.ndim()
            ));
        }
        let dims = x.shape();
        let dims = dims.as_slice();
        if dims[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "DecoderLayer::forward_on: x last-axis = {} but cfg.hidden_size = {}",
                dims[2],
                self.cfg.hidden_size
            ));
        }

        // Block 1: input_layernorm + self_attn + residual
        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn = self
            .self_attn
            .forward_on(&normed_in, mrope, cos, sin, mask, cache, target)?;
        let h = (x + &attn)?;

        // Block 2: post_attention_layernorm + mlp + residual
        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let mlp_out = self.mlp.forward_on(&normed_post, target)?;
        &h + &mlp_out
    }
```

- [ ] **Step 1.13: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer::tests::forward_shape_and_dtype_fp32
```

Expected: `1 passed`.

- [ ] **Step 1.14: Write failing bf16 dtype-preservation test**

Add to the `tests` module:

```rust
    #[test]
    fn forward_shape_and_dtype_bf16() {
        // bf16 input (with bf16 norm weights) → bf16 output preserved.
        let cfg = small_cfg();

        // bf16 attn + mlp weights matching small_cfg.
        let q_w = rand_w(&[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size], Dtype::Bfloat16);
        let k_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Bfloat16);
        let v_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Bfloat16);
        let o_w = rand_w(&[cfg.hidden_size, cfg.num_heads * cfg.head_dim], Dtype::Bfloat16);
        // bf16 norm weights to keep dtype contained at bf16 throughout.
        let qn = rand_w(&[cfg.head_dim], Dtype::Bfloat16);
        let kn = rand_w(&[cfg.head_dim], Dtype::Bfloat16);
        let pre_norm_w = rand_w(&[cfg.hidden_size], Dtype::Bfloat16);
        let post_norm_w = rand_w(&[cfg.hidden_size], Dtype::Bfloat16);
        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let down_w = rand_w(&[cfg.hidden_size, cfg.intermediate_size], Dtype::Bfloat16);

        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(qn, cfg.rms_norm_eps),
            RmsNorm::new(kn, cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w, None),
        );
        let layer = DecoderLayer::from_components(
            RmsNorm::new(pre_norm_w, cfg.rms_norm_eps),
            attn,
            RmsNorm::new(post_norm_w, cfg.rms_norm_eps),
            mlp,
            cfg,
        );

        let x = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = layer.forward(&x, &mrope, &cos, &sin, None, None).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, 4, cfg.hidden_size]);
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        // Sanity: outputs are finite.
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32).unwrap().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
```

- [ ] **Step 1.15: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer::tests::forward_shape_and_dtype_bf16
```

Expected: `1 passed`.

- [ ] **Step 1.16: Write failing residual-paths test**

Add to the `tests` module:

```rust
    #[test]
    fn forward_residual_paths_zero_blocks_yield_input() {
        // Zero out attn (o_proj=0) AND mlp (down_proj=0); the two residual chains
        // independently reduce DecoderLayer to identity:  out = x + 0 + 0 = x.
        let cfg = small_cfg();

        // Build attention with o_proj weight = 0 → attn output is exactly 0.
        let q_w = rand_w(&[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size], Dtype::Float32);
        let k_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Float32);
        let v_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Float32);
        let o_w_zero = mlx::ops::constructors::zeros(
            (cfg.hidden_size, cfg.num_heads * cfg.head_dim),
            Dtype::Float32,
        )
        .unwrap();
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w_zero, None),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );

        // Mlp with down_proj=0 → mlp output is exactly 0.
        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Float32);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Float32);
        let down_w_zero = mlx::ops::constructors::zeros(
            (cfg.hidden_size, cfg.intermediate_size),
            Dtype::Float32,
        )
        .unwrap();
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w_zero, None),
        );

        let layer = DecoderLayer::from_components(
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            mlp,
            cfg,
        );

        let x = rand_w(&[1, 4, cfg.hidden_size], Dtype::Float32);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = layer.forward(&x, &mrope, &cos, &sin, None, None).unwrap();

        let xv: Vec<f32> = x.to_vec().unwrap();
        let ov: Vec<f32> = out.to_vec().unwrap();
        for (xi, oi) in xv.iter().zip(ov.iter()) {
            assert!(
                (xi - oi).abs() < 1e-5,
                "residual path broken: x={xi}, out={oi}"
            );
        }
    }
```

- [ ] **Step 1.17: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer::tests::forward_residual_paths_zero_blocks_yield_input
```

Expected: `1 passed`.

- [ ] **Step 1.18: Write `DecoderLayer::from_loader` (production constructor)**

Append to `impl DecoderLayer`:

```rust
    /// Production constructor: load all sub-modules from a project [`Loader`].
    ///
    /// Reads (under prefix):
    ///
    /// - `{prefix}.input_layernorm.weight`            `[hidden_size]`
    /// - `{prefix}.self_attn.q_proj.weight`           `[num_heads * head_dim * 2, hidden_size]`
    /// - `{prefix}.self_attn.k_proj.weight`           `[num_kv_heads * head_dim, hidden_size]`
    /// - `{prefix}.self_attn.v_proj.weight`           `[num_kv_heads * head_dim, hidden_size]`
    /// - `{prefix}.self_attn.o_proj.weight`           `[hidden_size, num_heads * head_dim]`
    /// - `{prefix}.self_attn.q_norm.weight`           `[head_dim]`
    /// - `{prefix}.self_attn.k_norm.weight`           `[head_dim]`
    /// - `{prefix}.post_attention_layernorm.weight`   `[hidden_size]`
    /// - `{prefix}.mlp.gate_proj.weight`              `[intermediate_size, hidden_size]`
    /// - `{prefix}.mlp.up_proj.weight`                `[intermediate_size, hidden_size]`
    /// - `{prefix}.mlp.down_proj.weight`              `[hidden_size, intermediate_size]`
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: DecoderLayerConfig) -> Result<Self> {
        let input_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.input_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let self_attn = GatedAttention::from_loader(
            loader,
            &format!("{prefix}.self_attn"),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        )?;
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let mlp = Mlp::from_loader(loader, &format!("{prefix}.mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            cfg,
        })
    }
```

> Construction-time dim sanity checks are intentionally omitted — `Linear`'s matmul will surface any prefix / cfg mismatch at first `forward_on` with a clear shape error. This matches the precedent set by `GatedAttention::from_loader`.

- [ ] **Step 1.19: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 1.20: Run all DecoderLayer tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer
```

Expected: 4 passed (`from_components_carries_config`, `forward_shape_and_dtype_fp32`, `forward_shape_and_dtype_bf16`, `forward_residual_paths_zero_blocks_yield_input`).

- [ ] **Step 1.21: Commit**

```bash
git add ironmlx/src/nn/decoder_layer.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p3b4): DecoderLayer forward + from_loader (full-attn path)

Implements the full-attention DecoderLayer body —
input_layernorm + GatedAttention + residual + post_attention_layernorm
+ Mlp + residual — with rank/hidden_size pre-flight validation. Adds 3
behavior tests: fp32 shape/dtype, bf16 dtype preservation, and
residual-paths-yield-identity (zero attn + zero mlp). Also adds the
production from_loader path with documented weight key contract.

Reused unchanged by Mtp (P3b4) and the future Qwen3.5 main text model
(P4) for full-attention layers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `core::cache::MtpCache`

**Files:**
- Create: `ironmlx/src/core/cache/mtp_cache.rs`
- Modify: `ironmlx/src/core/cache/mod.rs` (`pub mod mtp_cache` + re-export)

### Goal

Add `MtpCache` — thin wrapper over `Vec<KVCache>` so an MTP head's `N` layers each own their own KV cache. Mirrors the cap-bounded construction pattern of P2 `KVCache::new` and P3b3 `GatedDeltaCache::new_with_cap`.

### Steps

- [ ] **Step 2.1: Write the failing construction test**

Create `ironmlx/src/core/cache/mtp_cache.rs`:

```rust
//! KV caches for an MTP head's layers — one [`KVCache`] per layer.
//!
//! Mirrors the cap-bounded design of P2 [`crate::core::cache::KVCache`]: capacity
//! is fixed at construction; per-layer `KVCache::update_and_fetch_on` enforces
//! `offset ≤ cap` independently. `num_layers` is locked at construction and
//! validated by the consumer ([`crate::nn::Mtp::forward_on`]) at forward time.

use anyhow::anyhow;
use mlx::Dtype;

use crate::core::cache::KVCache;
use crate::Result;

/// KV caches for the layers of an MTP head.
pub struct MtpCache {
    layers: Vec<KVCache>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(num_layers: usize) -> MtpCache {
        // Match P2 KVCache::new signature: (batch, n_kv_heads, head_dim, v_head_dim, dtype, cap)
        MtpCache::new_with_cap(num_layers, 1, 2, 8, 8, Dtype::Bfloat16, 16).expect("new_with_cap")
    }

    #[test]
    fn mtp_cache_new_with_cap_layers_and_zero_offset() {
        let cache = make_cache(3);
        assert_eq!(cache.num_layers(), 3);
        // All layer offsets start at 0; the wrapper exposes layer 0's offset by invariant.
        assert_eq!(cache.offset(), 0);
    }
}
```

- [ ] **Step 2.2: Run, verify it fails to compile**

```
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --tests
```

Expected: `cannot find function new_with_cap` / `num_layers` / `offset` on `MtpCache`.

- [ ] **Step 2.3: Implement `MtpCache::new_with_cap`, `num_layers`, `offset`**

Append to `impl MtpCache` in `ironmlx/src/core/cache/mtp_cache.rs`:

```rust
impl MtpCache {
    /// Construct caches for `num_layers` layers, each a fresh [`KVCache`] with
    /// the same `cap`, `n_kv_heads`, `head_dim`, `v_head_dim`, and `dtype`.
    ///
    /// `num_layers` must be `> 0`. `cap` is the hard maximum sequence length
    /// (forwarded to each [`KVCache::new`] as its `cap` argument).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cap(
        num_layers: usize,
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        dtype: Dtype,
        cap: i32,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(anyhow!("MtpCache::new_with_cap: num_layers must be > 0"));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KVCache::new(batch, n_kv_heads, head_dim, v_head_dim, dtype, cap));
        }
        Ok(Self { layers })
    }

    /// Number of cached layers (fixed at construction).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Immutable view of one layer's cache.
    pub fn layer(&self, idx: usize) -> &KVCache {
        &self.layers[idx]
    }

    /// Mutable view of one layer's cache (used by the consumer's per-layer forward path).
    pub fn layer_mut(&mut self, idx: usize) -> &mut KVCache {
        &mut self.layers[idx]
    }

    /// Reset every contained [`KVCache`] back to `offset = 0`. Buffers are retained for reuse.
    pub fn reset(&mut self) {
        for c in &mut self.layers {
            c.reset();
        }
    }

    /// Returns the offset of layer 0; all layers share the same offset by invariant.
    pub fn offset(&self) -> i32 {
        self.layers.first().map(|c| c.offset()).unwrap_or(0)
    }
}
```

- [ ] **Step 2.4: Wire into `core::cache::mod.rs`**

Edit [`ironmlx/src/core/cache/mod.rs`](../../../ironmlx/src/core/cache/mod.rs):

```rust
//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod gated_delta;
pub mod kv_cache;
pub mod mtp_cache;

pub use gated_delta::GatedDeltaCache;
pub use kv_cache::KVCache;
pub use mtp_cache::MtpCache;
```

- [ ] **Step 2.5: Run, verify the construction test passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::mtp_cache::tests::mtp_cache_new_with_cap_layers_and_zero_offset
```

Expected: `1 passed`.

- [ ] **Step 2.6: Write failing reset test**

Add to the `tests` module of `mtp_cache.rs`:

```rust
    #[test]
    fn mtp_cache_reset_resets_all_layer_offsets() {
        let mut cache = make_cache(2);
        // Drive layer 0 forward by one update to advance its offset.
        let k0: mlx::Array =
            mlx::ops::constructors::zeros((1, 2, 4, 8), Dtype::Bfloat16).unwrap();
        let v0: mlx::Array =
            mlx::ops::constructors::zeros((1, 2, 4, 8), Dtype::Bfloat16).unwrap();
        cache.layer_mut(0).update_and_fetch(&k0, &v0).unwrap();
        // Drive layer 1 forward similarly.
        cache.layer_mut(1).update_and_fetch(&k0, &v0).unwrap();
        assert_eq!(cache.layer(0).offset(), 4);
        assert_eq!(cache.layer(1).offset(), 4);

        cache.reset();
        assert_eq!(cache.layer(0).offset(), 0);
        assert_eq!(cache.layer(1).offset(), 0);
        assert_eq!(cache.offset(), 0);
    }
```

- [ ] **Step 2.7: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::mtp_cache::tests::mtp_cache_reset_resets_all_layer_offsets
```

Expected: `1 passed`.

- [ ] **Step 2.8: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 2.9: Commit**

```bash
git add ironmlx/src/core/cache/mtp_cache.rs ironmlx/src/core/cache/mod.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p3b4): MtpCache wrapper over per-layer KVCache

Cap-bounded construction (mirrors P2 KVCache + P3b3 GatedDeltaCache),
per-layer accessors, group reset. num_layers is fixed at construction;
the consumer (Mtp::forward_on) validates it against the model's layer
count at forward time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `nn::Mtp` (Multi-Token Prediction head)

**Files:**
- Create: `ironmlx/src/nn/mtp.rs`
- Modify: `ironmlx/src/nn/mod.rs` (`pub mod mtp` + re-export)

### Goal

Add `Mtp` — the MTP head: `pre_fc_norm_hidden`/`pre_fc_norm_embedding` → concat (along last axis, order `[e, h]`) → `fc Linear 2H→H` → `N × DecoderLayer` → `mtp.norm`. Returns the post-norm hidden state; the caller projects to logits via tied `Embedding::as_linear`.

### Steps

- [ ] **Step 3.1: Write failing construction + num_layers test**

Create `ironmlx/src/nn/mtp.rs`:

```rust
//! MTP (Multi-Token Prediction) head — the speculative-decoding draft head.
//!
//! Mirrors vllm-mlx's `_MTPModule` (`/Volumes/Dev/vllm-mlx/vllm_mlx/patches/qwen3_5_mtp.py:204-216`):
//!
//! ```text
//! e = pre_fc_norm_embedding(next_token_embeds)
//! h = pre_fc_norm_hidden(hidden_states)
//! x = fc(concat([e, h], axis=-1))           # 2H -> H, no bias
//! for layer in layers:                      # N DecoderLayers, fa-only
//!     x = layer(x, mask=causal, cache=mtp_cache[i])
//! x = norm(x)
//! return x  # caller does tied lm_head: embed_tokens.as_linear(x)
//! ```
//!
//! Caller responsibilities (kept out of this module to preserve isolation):
//! - Embed `next_token_ids` to `next_token_embeds` via the main model's `Embedding`.
//! - Project the returned post-norm hidden state to logits via `Embedding::as_linear`
//!   (when `tie_word_embeddings = true`) or a separate `lm_head` Linear.
//! - Run the speculative-decoding loop (draft / verify / accept / KV rollback).
//!   That layer lands in P8c, not here.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::MtpCache;
use crate::core::Loader;
use crate::nn::{DecoderLayer, DecoderLayerConfig, Linear, Mrope, RmsNorm};
use crate::Result;

/// Configuration for [`Mtp`].
#[derive(Debug, Clone, Copy)]
pub struct MtpConfig {
    pub hidden_size: i32,
    /// Number of MTP DecoderLayers. Qwen3.5 checkpoints ship with `1`.
    pub num_mtp_layers: i32,
    /// Per-layer config (forwarded verbatim to each `DecoderLayer::from_loader`).
    pub layer: DecoderLayerConfig,
}

/// Multi-Token Prediction head.
pub struct Mtp {
    pre_fc_norm_hidden: RmsNorm,
    pre_fc_norm_embedding: RmsNorm,
    fc: Linear, // [2H] -> [H], no bias
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    cfg: MtpConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    use crate::nn::{GatedAttention, GatedAttentionConfig, Mlp};

    fn rand_w(shape: &[i32], dtype: Dtype) -> Array {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0173).sin()).collect();
        let arr: Array = (data.as_slice(), shape).try_into().unwrap();
        mlx::ops::cast::astype(&arr, dtype).unwrap()
    }

    fn ones_w(dim: i32) -> Array {
        mlx::ops::constructors::ones((dim,), Dtype::Float32).unwrap()
    }

    fn small_layer_cfg() -> DecoderLayerConfig {
        DecoderLayerConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        }
    }

    fn build_decoder_layer(cfg: DecoderLayerConfig) -> DecoderLayer {
        let q_w = rand_w(&[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size], Dtype::Bfloat16);
        let k_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Bfloat16);
        let v_w = rand_w(&[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size], Dtype::Bfloat16);
        let o_w = rand_w(&[cfg.hidden_size, cfg.num_heads * cfg.head_dim], Dtype::Bfloat16);
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );
        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let down_w = rand_w(&[cfg.hidden_size, cfg.intermediate_size], Dtype::Bfloat16);
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w, None),
        );
        DecoderLayer::from_components(
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            mlp,
            cfg,
        )
    }

    fn build_mtp(num_layers: i32) -> Mtp {
        let layer_cfg = small_layer_cfg();
        let cfg = MtpConfig {
            hidden_size: layer_cfg.hidden_size,
            num_mtp_layers: num_layers,
            layer: layer_cfg,
        };
        let h = cfg.hidden_size;
        let layers = (0..num_layers).map(|_| build_decoder_layer(layer_cfg)).collect();

        Mtp::from_components(
            RmsNorm::new(ones_w(h), layer_cfg.rms_norm_eps),
            RmsNorm::new(ones_w(h), layer_cfg.rms_norm_eps),
            Linear::new_fp(rand_w(&[h, 2 * h], Dtype::Bfloat16), None),
            layers,
            RmsNorm::new(ones_w(h), layer_cfg.rms_norm_eps),
            cfg,
        )
    }

    #[test]
    fn mtp_construction_components() {
        let mtp = build_mtp(1);
        assert_eq!(mtp.num_layers(), 1);
        assert_eq!(mtp.config().num_mtp_layers, 1);
        assert_eq!(mtp.config().hidden_size, 32);
    }
}
```

- [ ] **Step 3.2: Run, verify it fails to compile**

```
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx --tests
```

Expected: errors on missing `Mtp::from_components`, `num_layers`, `config`.

- [ ] **Step 3.3: Implement `Mtp::from_components`, `config`, `num_layers`**

Append to `ironmlx/src/nn/mtp.rs`:

```rust
impl Mtp {
    /// Test/composition seam: build an `Mtp` from pre-built sub-modules.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(
        pre_fc_norm_hidden: RmsNorm,
        pre_fc_norm_embedding: RmsNorm,
        fc: Linear,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        cfg: MtpConfig,
    ) -> Self {
        Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
            cfg,
        }
    }

    /// Read-only view of the head config.
    pub fn config(&self) -> &MtpConfig {
        &self.cfg
    }

    /// Number of DecoderLayers in this MTP head.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
```

- [ ] **Step 3.4: Wire into `nn::mod.rs`**

Edit [`ironmlx/src/nn/mod.rs`](../../../ironmlx/src/nn/mod.rs). Add `pub mod mtp;` (alphabetical: between `mrope` and `norm`) and append to the `pub use` block: `pub use mtp::{Mtp, MtpConfig};`.

- [ ] **Step 3.5: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp::tests::mtp_construction_components
```

Expected: `1 passed`.

- [ ] **Step 3.6: Write failing forward shape + dtype test**

Add to the `tests` module of `mtp.rs`:

```rust
    #[test]
    fn forward_shape_and_dtype() {
        let mtp = build_mtp(1);
        let cfg = small_layer_cfg();

        // Inputs: bf16 hidden + bf16 next-token embeddings.
        let hidden = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let next_embeds = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = mtp
            .forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None)
            .unwrap();

        assert_eq!(out.shape().as_slice(), &[1, 4, cfg.hidden_size]);
        // Precise dtype is not asserted (RmsNorm with fp32 weights promotes path);
        // the integration test in Task 4 verifies bit-exact-modulo-tol against Python.
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32).unwrap().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
```

- [ ] **Step 3.7: Run, verify it fails**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp::tests::forward_shape_and_dtype
```

Expected: `forward` not defined.

- [ ] **Step 3.8: Implement `Mtp::forward` + `forward_on` + `validate_inputs`**

Append to `impl Mtp` in `ironmlx/src/nn/mtp.rs`:

```rust
    /// Default-stream forward. See [`forward_on`](Self::forward_on).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
    ) -> Result<Array> {
        self.forward_on(
            hidden_states,
            next_token_embeds,
            mrope,
            cos,
            sin,
            mask,
            mtp_cache,
            (),
        )
    }

    /// Stream-targeted forward.
    ///
    /// Inputs:
    /// - `hidden_states`: post-norm hidden state from the main model, `[B, S, hidden_size]`.
    ///   Caller MUST pass `inner.norm(...)`-applied hidden state, matching mlx-lm's
    ///   `qwen3_5_mtp.py:366` (`return out, normed`).
    /// - `next_token_embeds`: caller-pre-computed embedding of the next-token ids,
    ///   `[B, S, hidden_size]` (typically `embed_tokens(next_token_ids)`).
    /// - `cos`/`sin`: precomputed by [`Mrope::cos_sin`].
    /// - `mask`: forwarded to each [`DecoderLayer`] (currently always-causal in
    ///   [`crate::nn::GatedAttention`]).
    /// - `mtp_cache`: optional KV caches for the `N` MTP layers; if `Some`, must
    ///   satisfy `mtp_cache.num_layers() == self.num_layers()`.
    ///
    /// Output: `[B, S, hidden_size]` — the post-`mtp.norm` hidden state. Caller
    /// projects to logits via tied `Embedding::as_linear` to obtain `[B, S, vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        mut mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Pre-flight validation (production-grade stability).
        self.validate_inputs(hidden_states, next_token_embeds, mtp_cache.as_deref())?;

        // Step 1: pre-FC norms.
        let h = self.pre_fc_norm_hidden.forward_on(hidden_states, target)?;
        let e = self
            .pre_fc_norm_embedding
            .forward_on(next_token_embeds, target)?;

        // Step 2: concat([e, h], axis=-1)  →  [B, S, 2H]
        // Order is [e, h] — not [h, e] — to match mlx-lm `qwen3_5_mtp.py:380`.
        let concat = mlx::ops::shape::concatenate_on(&[&e, &h], -1, target)?;

        // Step 3: fc 2H -> H (no bias).
        let mut x = self.fc.forward_on(&concat, target)?;

        // Step 4: feed through N DecoderLayers, each with its own KV cache slot.
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = mtp_cache.as_deref_mut().map(|mc| mc.layer_mut(i));
            x = layer.forward_on(&x, mrope, cos, sin, mask, layer_cache, target)?;
        }

        // Step 5: final norm.
        self.norm.forward_on(&x, target)
    }

    /// Pre-flight validation of input shapes and cache layout. Returns Err on first mismatch.
    fn validate_inputs(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mtp_cache: Option<&MtpCache>,
    ) -> Result<()> {
        if hidden_states.ndim() != 3 || next_token_embeds.ndim() != 3 {
            return Err(anyhow!(
                "Mtp::forward_on: hidden_states and next_token_embeds must be rank-3, \
                 got ranks {}/{}",
                hidden_states.ndim(),
                next_token_embeds.ndim(),
            ));
        }
        let hs = hidden_states.shape();
        let es = next_token_embeds.shape();
        let hs = hs.as_slice();
        let es = es.as_slice();
        if hs != es {
            return Err(anyhow!(
                "Mtp::forward_on: hidden_states {:?} and next_token_embeds {:?} \
                 must have identical shape",
                hs,
                es,
            ));
        }
        if hs[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "Mtp::forward_on: last-axis must equal cfg.hidden_size {}, got {}",
                self.cfg.hidden_size,
                hs[2],
            ));
        }
        if let Some(c) = mtp_cache {
            if c.num_layers() != self.layers.len() {
                return Err(anyhow!(
                    "Mtp::forward_on: mtp_cache.num_layers() = {} but Mtp has {} layers",
                    c.num_layers(),
                    self.layers.len(),
                ));
            }
        }
        Ok(())
    }
```

- [ ] **Step 3.9: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp::tests::forward_shape_and_dtype
```

Expected: `1 passed`.

- [ ] **Step 3.10: Write failing shape-mismatch validation test**

Add to the `tests` module:

```rust
    #[test]
    fn forward_validates_shape_mismatch() {
        let mtp = build_mtp(1);
        let cfg = small_layer_cfg();

        let hidden = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        // Mismatched S between hidden (S=4) and next_embeds (S=3) → Err.
        let next_embeds = rand_w(&[1, 3, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let r = mtp.forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None);
        let err = r.expect_err("mismatched shapes must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("identical shape"),
            "expected shape-mismatch message, got: {msg}"
        );
    }
```

- [ ] **Step 3.11: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp::tests::forward_validates_shape_mismatch
```

Expected: `1 passed`.

- [ ] **Step 3.12: Write failing cache-mismatch validation test**

Add to the `tests` module:

```rust
    #[test]
    fn forward_validates_cache_layers_mismatch() {
        let mtp = build_mtp(1); // 1 MTP layer
        let cfg = small_layer_cfg();

        let hidden = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let next_embeds = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        // Cache with 2 layers, but Mtp has 1 layer → Err.
        let mut wrong_cache =
            MtpCache::new_with_cap(2, 1, cfg.num_kv_heads, cfg.head_dim, cfg.head_dim, Dtype::Bfloat16, 16)
                .unwrap();
        let r = mtp.forward(
            &hidden,
            &next_embeds,
            &mrope,
            &cos,
            &sin,
            None,
            Some(&mut wrong_cache),
        );
        let err = r.expect_err("cache num_layers mismatch must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("num_layers")
                && msg.contains('2')
                && msg.contains('1'),
            "expected cache-num_layers-mismatch message, got: {msg}"
        );
    }
```

- [ ] **Step 3.13: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp::tests::forward_validates_cache_layers_mismatch
```

Expected: `1 passed`.

- [ ] **Step 3.14: Write failing concat-layout test (pins `[e, h]` order)**

Add to the `tests` module of `mtp.rs`. This test pins `concat([e, h])` (not `[h, e]`) — the most subtle correctness invariant in `Mtp::forward`. Strategy: build an identity-on-zero-residual `DecoderLayer` (zero `o_proj` + zero `down_proj` ⇒ `out = x + 0 + 0 = x`) and an asymmetric `fc` weight so the two orderings produce vectors with **different directions** (not just different magnitudes); `mtp.norm` then preserves the directional signature.

Concretely: with `fc[i, i] = 1` (selects `e`-channel `i` with coef 1) and `fc[i, H+i] = 3` (selects `h`-channel `i` with coef 3), the post-fc value is `1·e_norm + 3·h_norm`. With `e` one-hot at channel 0 and `h` one-hot at channel 1, the final output's ratio `y[1]/y[0]` is exactly 3.0 under `[e, h]` ordering and 1/3 under the swap. Detailed derivation in the test comment.

```rust
    #[test]
    fn forward_concat_layout_e_then_h() {
        // Pin concat order [e, h] (NOT [h, e]) — matches mlx-lm qwen3_5_mtp.py:380.
        //
        // Build pieces:
        //   fc weight (row-major [out=H, in=2H]):
        //     row i has W[i, i] = 1 (e-half coef) and W[i, H+i] = 3 (h-half coef), zeros elsewhere.
        //     ⇒ fc(concat([e, h])) = 1·e + 3·h.
        //     ⇒ fc(concat([h, e])) = 1·h + 3·e  (swap e/h).
        //   identity-DecoderLayer: zero o_proj + zero down_proj ⇒ out = x + 0 + 0 = x.
        //   pre_fc_norm and mtp.norm: RmsNorm with weight=ones (direction-preserving).
        //
        // Inputs:
        //   e = one-hot at channel 0 (broadcast over [B=1, S=4, H])  → pre_fc_norm(e)[k] = sqrt(H)·δ_{k,0}
        //   h = one-hot at channel 1                                 → pre_fc_norm(h)[k] = sqrt(H)·δ_{k,1}
        //
        // Under [e, h] ordering:
        //   fc out:    y[0] = 1·sqrt(H),  y[1] = 3·sqrt(H),  rest 0.
        //   identity:  unchanged.
        //   mtp.norm:  RMS(y) = sqrt((H + 9H)/H) = sqrt(10);  y'[0] = sqrt(H)/sqrt(10), y'[1] = 3·sqrt(H)/sqrt(10).
        //   ratio y'[1]/y'[0] = 3.0  ←— this PINS the order.
        // Under accidental [h, e] swap:
        //   ratio would be 1/3.
        let layer_cfg = small_layer_cfg();
        let cfg = MtpConfig {
            hidden_size: layer_cfg.hidden_size,
            num_mtp_layers: 1,
            layer: layer_cfg,
        };
        let h_dim = cfg.hidden_size as usize;

        // identity-on-zero-residual DecoderLayer.
        let q_w = rand_w(&[layer_cfg.num_heads * layer_cfg.head_dim * 2, layer_cfg.hidden_size], Dtype::Float32);
        let k_w = rand_w(&[layer_cfg.num_kv_heads * layer_cfg.head_dim, layer_cfg.hidden_size], Dtype::Float32);
        let v_w = rand_w(&[layer_cfg.num_kv_heads * layer_cfg.head_dim, layer_cfg.hidden_size], Dtype::Float32);
        let o_w_zero = mlx::ops::constructors::zeros(
            (layer_cfg.hidden_size, layer_cfg.num_heads * layer_cfg.head_dim),
            Dtype::Float32,
        ).unwrap();
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w_zero, None),
            RmsNorm::new(ones_w(layer_cfg.head_dim), layer_cfg.rms_norm_eps),
            RmsNorm::new(ones_w(layer_cfg.head_dim), layer_cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: layer_cfg.num_heads,
                num_kv_heads: layer_cfg.num_kv_heads,
                head_dim: layer_cfg.head_dim,
                rms_norm_eps: layer_cfg.rms_norm_eps,
                attention_bias: layer_cfg.attention_bias,
            },
        );
        let gate_w = rand_w(&[layer_cfg.intermediate_size, layer_cfg.hidden_size], Dtype::Float32);
        let up_w = rand_w(&[layer_cfg.intermediate_size, layer_cfg.hidden_size], Dtype::Float32);
        let down_w_zero = mlx::ops::constructors::zeros(
            (layer_cfg.hidden_size, layer_cfg.intermediate_size),
            Dtype::Float32,
        ).unwrap();
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w_zero, None),
        );
        let identity_layer = DecoderLayer::from_components(
            RmsNorm::new(ones_w(layer_cfg.hidden_size), layer_cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(layer_cfg.hidden_size), layer_cfg.rms_norm_eps),
            mlp,
            layer_cfg,
        );

        // Asymmetric fc weight: e-half coef = 1, h-half coef = 3.
        let mut fc_data = vec![0.0_f32; h_dim * 2 * h_dim];
        for i in 0..h_dim {
            fc_data[i * (2 * h_dim) + i] = 1.0;
            fc_data[i * (2 * h_dim) + (h_dim + i)] = 3.0;
        }
        let fc_w: Array =
            (fc_data.as_slice(), &[cfg.hidden_size, 2 * cfg.hidden_size][..]).try_into().unwrap();

        let mtp = Mtp::from_components(
            RmsNorm::new(ones_w(cfg.hidden_size), 1e-6),
            RmsNorm::new(ones_w(cfg.hidden_size), 1e-6),
            Linear::new_fp(fc_w, None),
            vec![identity_layer],
            RmsNorm::new(ones_w(cfg.hidden_size), 1e-6),
            cfg,
        );

        // Inputs: e one-hot at channel 0; h one-hot at channel 1; broadcast over [B=1, S=4, H].
        let mut e_data = vec![0.0_f32; 4 * h_dim];
        let mut h_data = vec![0.0_f32; 4 * h_dim];
        for s in 0..4 {
            e_data[s * h_dim] = 1.0;
            h_data[s * h_dim + 1] = 1.0;
        }
        let next_embeds: Array =
            (e_data.as_slice(), &[1, 4, cfg.hidden_size][..]).try_into().unwrap();
        let hidden: Array =
            (h_data.as_slice(), &[1, 4, cfg.hidden_size][..]).try_into().unwrap();

        let mrope = Mrope::new(layer_cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = mtp
            .forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None)
            .unwrap();

        let v: Vec<f32> = out.to_vec().unwrap();
        // First row (b=0, s=0), channels 0 and 1.
        let c0 = v[0];
        let c1 = v[1];
        assert!(c0 > 0.0 && c1 > 0.0, "expected positive c0/c1, got c0={c0}, c1={c1}");
        let ratio = c1 / c0;
        assert!(
            (ratio - 3.0_f32).abs() < 1e-3,
            "concat order broken: expected c1/c0 ≈ 3.0 ([e, h] order), got {ratio}",
        );
    }
```

- [ ] **Step 3.15: Run, verify the concat-layout test passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp::tests::forward_concat_layout_e_then_h
```

Expected: `1 passed`. The assertion `(ratio - 3.0).abs() < 1e-3` confirms `[e, h]` order; an accidental `[h, e]` swap would yield ratio ≈ 1/3 and fail loudly.

- [ ] **Step 3.16: Implement `Mtp::from_loader` (production constructor)**

Append to `impl Mtp` in `ironmlx/src/nn/mtp.rs`:

```rust
    /// Production constructor: load all components from a project [`Loader`].
    ///
    /// Reads (under prefix `mtp.`):
    ///
    /// - `{prefix}.pre_fc_norm_hidden.weight`           `[hidden_size]`
    /// - `{prefix}.pre_fc_norm_embedding.weight`        `[hidden_size]`
    /// - `{prefix}.fc.weight`                           `[hidden_size, 2 * hidden_size]` (no bias)
    /// - `{prefix}.layers.{0..N-1}.{...}`               (per [`DecoderLayer::from_loader`])
    /// - `{prefix}.norm.weight`                         `[hidden_size]`
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: MtpConfig) -> Result<Self> {
        let pre_fc_norm_hidden = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.pre_fc_norm_hidden"),
            cfg.layer.rms_norm_eps,
        )?;
        let pre_fc_norm_embedding = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.pre_fc_norm_embedding"),
            cfg.layer.rms_norm_eps,
        )?;
        let fc = Linear::from_loader(loader, &format!("{prefix}.fc"))?;
        let norm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.norm"),
            cfg.layer.rms_norm_eps,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_mtp_layers as usize);
        for i in 0..cfg.num_mtp_layers {
            layers.push(DecoderLayer::from_loader(
                loader,
                &format!("{prefix}.layers.{i}"),
                cfg.layer,
            )?);
        }

        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
            cfg,
        })
    }
```

> No construction-time `fc` dim sanity check — the matmul on `concat([e, h])` will surface any cfg / weight mismatch at first `forward_on` with a clear shape error (matches `GatedAttention::from_loader` precedent).

- [ ] **Step 3.17: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 3.18: Run all Mtp tests**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp
```

Expected: 5 passed (`mtp_construction_components`, `forward_shape_and_dtype`, `forward_validates_shape_mismatch`, `forward_validates_cache_layers_mismatch`, `forward_concat_layout_e_then_h`).

- [ ] **Step 3.19: Commit**

```bash
git add ironmlx/src/nn/mtp.rs ironmlx/src/nn/mod.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p3b4): Mtp head — pre_fc_norms + fc + N×DecoderLayer + norm

Implements the MTP draft head: pre_fc_norm_hidden / pre_fc_norm_embedding
project to [B,S,H], concat along last axis (order [e, h] — pinned by a
ratio test that distinguishes from the [h, e] swap), fc 2H -> H, run
through N DecoderLayers each with optional MtpCache slot, then mtp.norm.
Returns the post-norm hidden state; tied lm_head is the caller's job.

Pre-flight validation rejects rank mismatch, shape mismatch between
hidden_states and next_token_embeds, hidden_size mismatch, and
cache.num_layers != model.num_layers. from_loader covers the
production weight-key contract documented in the spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Python fixture + numerical-correctness integration test

**Files:**
- Create: `ironmlx/tests/fixtures/p3b4_mtp/README.md`
- Create: `ironmlx/tests/fixtures/p3b4_mtp/gen_fixture.py`
- Create: 22 `.npy` fixture files (regenerated by running `gen_fixture.py`)
- Create: `ironmlx/tests/p3b4_mtp.rs`

### Goal

End-to-end correctness: for a fixed small-scale config (B=1, S=4, H=32, Hq=4, Hkv=2, D=8, intermediate_size=64, num_mtp_layers=1), generate inputs + weights + the expected `Mtp::forward` output via an independent Python re-implementation built only from `mx.core` ops. The Rust integration test loads them and asserts numerical equivalence.

### Steps

- [ ] **Step 4.1: Write the README**

Create `ironmlx/tests/fixtures/p3b4_mtp/README.md`:

```markdown
# P3b4 MTP fixture

Tiny synthetic fixture for verifying numerical correctness of `nn::Mtp::forward`
against an independent Python reference built only from `mlx.core` primitives
(no `mlx-lm` patch dependency, no quantization, no MoE).

## Config

- B=1, S=4, hidden_size=32
- Hq=4, Hkv=2, head_dim=8, intermediate_size=64
- num_mtp_layers=1, rms_norm_eps=1e-6
- partial_rotary_factor=1.0 (rot_dim=8, half=4)
- mrope sections=[2, 1, 1]
- attention_bias=false

## Files

| File | Shape | dtype | Notes |
|---|---|---|---|
| `input_hidden.npy` | `[1, 4, 32]` | bf16 | post-norm hidden state from main model (synthetic) |
| `input_next_embeds.npy` | `[1, 4, 32]` | bf16 | embedding of (synthetic) next-token ids |
| `input_position_ids.npy` | `[3, 1, 4]` | i32 | mrope 3-stream position ids |
| `input_inv_freq.npy` | `[4]` | fp32 | precomputed by Mrope::new |
| `pre_fc_norm_hidden_weight.npy` | `[32]` | fp32 | RmsNorm weight |
| `pre_fc_norm_embedding_weight.npy` | `[32]` | fp32 | RmsNorm weight |
| `fc_weight.npy` | `[32, 64]` | bf16 | Linear 2H -> H, no bias |
| `layer0_input_layernorm_weight.npy` | `[32]` | fp32 | DecoderLayer.input_layernorm |
| `layer0_q_proj_weight.npy` | `[64, 32]` | bf16 | Hq*D*2 (queries + gate) |
| `layer0_k_proj_weight.npy` | `[16, 32]` | bf16 | Hkv*D |
| `layer0_v_proj_weight.npy` | `[16, 32]` | bf16 | Hkv*D |
| `layer0_o_proj_weight.npy` | `[32, 32]` | bf16 | hidden_size <- Hq*D |
| `layer0_q_norm_weight.npy` | `[8]` | fp32 | per-head dim |
| `layer0_k_norm_weight.npy` | `[8]` | fp32 | per-head dim |
| `layer0_post_attention_layernorm_weight.npy` | `[32]` | fp32 | DecoderLayer.post_attention_layernorm |
| `layer0_mlp_gate_proj_weight.npy` | `[64, 32]` | bf16 | SwiGLU gate |
| `layer0_mlp_up_proj_weight.npy` | `[64, 32]` | bf16 | SwiGLU up |
| `layer0_mlp_down_proj_weight.npy` | `[32, 64]` | bf16 | SwiGLU down |
| `norm_weight.npy` | `[32]` | fp32 | mtp.norm |
| `expected_cos.npy` | `[1, 4, 4]` | fp32 | Mrope::cos_sin output |
| `expected_sin.npy` | `[1, 4, 4]` | fp32 | Mrope::cos_sin output |
| `expected_mtp_out.npy` | `[1, 4, 32]` | fp32 | post-`mtp.norm` hidden state |

`expected_mtp_out` ends up at fp32 because all RmsNorm weights in this fixture are
fp32. Mixed-precision matmul with bf16 attn / mlp weights upgrades intermediates
to fp32 by the time they hit a fp32 RmsNorm — the final `mtp.norm` outputs fp32.

## Regenerate

```text
cd ironmlx/tests/fixtures/p3b4_mtp && python gen_fixture.py
```

Pinned MLX version: `0.31.1` (script will refuse to run on a different version).
```

- [ ] **Step 4.2: Write the Python fixture generator**

Create `ironmlx/tests/fixtures/p3b4_mtp/gen_fixture.py`:

```python
"""Generate P3b4 MTP fixtures.

Independent re-implementation of vllm-mlx's `_MTPModule.mtp_forward` algorithm
(`/Volumes/Dev/vllm-mlx/vllm_mlx/patches/qwen3_5_mtp.py:369-391`) using `mlx.core`
primitives only. Outputs `.npy` files alongside this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

EXPECTED_MLX_VERSION = "0.31.1"
_mlx_version = mx.__version__
if _mlx_version != EXPECTED_MLX_VERSION:
    raise SystemExit(
        f"mlx version mismatch: got {_mlx_version}, expected "
        f"{EXPECTED_MLX_VERSION}. Bump and regenerate the .npy fixtures."
    )

OUT_DIR = Path(__file__).parent

# ---- Small synthetic config (matches MtpConfig in p3b4_mtp.rs) ----
B = 1
S = 4
HIDDEN = 32
HQ = 4
HKV = 2
D = 8
INTERMEDIATE = 64
NUM_MTP_LAYERS = 1
RMS_EPS = 1e-6
PARTIAL = 1.0
ROT_DIM = int(D * PARTIAL) & ~1   # 8
HALF = ROT_DIM // 2               # 4
SECTIONS = [2, 1, 1]              # sum = HALF
THETA = 1e7


def _build_inv_freq() -> mx.array:
    idx = mx.arange(0, HALF, dtype=mx.float32)
    return mx.exp(-(idx * (2.0 / ROT_DIM)) * float(np.log(THETA)))


def _build_position_ids() -> mx.array:
    one = mx.arange(0, S, dtype=mx.int32).reshape((1, 1, S))
    return mx.broadcast_to(one, (3, B, S))


def _ref_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    pos_f = position_ids.astype(mx.float32)
    pos_unsq = pos_f[..., None]
    inv_unsq = inv_freq.reshape((1, 1, 1, -1))
    freqs = pos_unsq * inv_unsq
    cos_per = mx.cos(freqs)
    sin_per = mx.sin(freqs)
    offsets = [0]
    for n in SECTIONS:
        offsets.append(offsets[-1] + n)
    cos_segs = []
    sin_segs = []
    for s, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        cos_segs.append(cos_per[s, :, :, lo:hi])
        sin_segs.append(sin_per[s, :, :, lo:hi])
    return mx.concatenate(cos_segs, axis=-1), mx.concatenate(sin_segs, axis=-1)


def _ref_apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    rot = x[..., :ROT_DIM]
    tail = x[..., ROT_DIM:]
    even = rot[..., 0::2]
    odd = rot[..., 1::2]
    c = cos[:, None, :, :]
    s = sin[:, None, :, :]
    rot_even = (even.astype(mx.float32) * c - odd.astype(mx.float32) * s).astype(x.dtype)
    rot_odd = (even.astype(mx.float32) * s + odd.astype(mx.float32) * c).astype(x.dtype)
    out_rot = mx.stack([rot_even, rot_odd], axis=-1).reshape(x.shape[:-1] + (ROT_DIM,))
    return mx.concatenate([out_rot, tail], axis=-1)


def _ref_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _ref_gated_attention(
    x: mx.array,
    q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    q_norm_w: mx.array, k_norm_w: mx.array,
    cos: mx.array, sin: mx.array,
) -> mx.array:
    """Independent re-impl of Qwen3NextAttention (causal, no cache, no mask)."""
    q_full = x @ q_w.T
    k = x @ k_w.T
    v = x @ v_w.T
    q_per_head = q_full.reshape((B, S, HQ, D * 2))
    queries, gate = mx.split(q_per_head, 2, axis=-1)
    gate_flat = gate.reshape((B, S, HQ * D))
    queries = _ref_rms_norm(queries, q_norm_w, RMS_EPS)
    queries = queries.transpose(0, 2, 1, 3)
    k = k.reshape((B, S, HKV, D))
    k = _ref_rms_norm(k, k_norm_w, RMS_EPS)
    k = k.transpose(0, 2, 1, 3)
    v = v.reshape((B, S, HKV, D)).transpose(0, 2, 1, 3)
    queries = _ref_apply_rope(queries, cos, sin)
    k = _ref_apply_rope(k, cos, sin)
    scale = D ** -0.5
    sdpa_out = mx.fast.scaled_dot_product_attention(queries, k, v, scale=scale, mask="causal")
    sdpa_flat = sdpa_out.transpose(0, 2, 1, 3).reshape((B, S, HQ * D))
    gated = sdpa_flat * mx.sigmoid(gate_flat)
    return gated @ o_w.T


def _ref_mlp(
    x: mx.array,
    gate_w: mx.array, up_w: mx.array, down_w: mx.array,
) -> mx.array:
    """SwiGLU: down( silu(gate(x)) * up(x) )."""
    g = x @ gate_w.T
    u = x @ up_w.T
    g_sig = mx.sigmoid(g)
    activated = g * g_sig * u
    return activated @ down_w.T


def _ref_decoder_layer(
    x: mx.array,
    in_ln_w: mx.array,
    q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    q_norm_w: mx.array, k_norm_w: mx.array,
    post_ln_w: mx.array,
    mlp_gate_w: mx.array, mlp_up_w: mx.array, mlp_down_w: mx.array,
    cos: mx.array, sin: mx.array,
) -> mx.array:
    """Mirrors ironmlx::nn::DecoderLayer::forward (full-attn path, no cache, no mask)."""
    normed_in = _ref_rms_norm(x, in_ln_w, RMS_EPS)
    attn = _ref_gated_attention(
        normed_in, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, cos, sin,
    )
    h = x + attn
    normed_post = _ref_rms_norm(h, post_ln_w, RMS_EPS)
    mlp_out = _ref_mlp(normed_post, mlp_gate_w, mlp_up_w, mlp_down_w)
    return h + mlp_out


def _ref_mtp(
    hidden: mx.array,
    next_embeds: mx.array,
    pre_fc_norm_hidden_w: mx.array,
    pre_fc_norm_embedding_w: mx.array,
    fc_w: mx.array,
    # layer 0 weights
    in_ln_w: mx.array,
    q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    q_norm_w: mx.array, k_norm_w: mx.array,
    post_ln_w: mx.array,
    mlp_gate_w: mx.array, mlp_up_w: mx.array, mlp_down_w: mx.array,
    norm_w: mx.array,
    cos: mx.array, sin: mx.array,
) -> mx.array:
    """Mirrors ironmlx::nn::Mtp::forward."""
    h = _ref_rms_norm(hidden, pre_fc_norm_hidden_w, RMS_EPS)
    e = _ref_rms_norm(next_embeds, pre_fc_norm_embedding_w, RMS_EPS)
    concat = mx.concatenate([e, h], axis=-1)  # [B, S, 2H]; ORDER [e, h] is pinned.
    x = concat @ fc_w.T                       # [B, S, H]
    x = _ref_decoder_layer(
        x, in_ln_w,
        q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
        post_ln_w,
        mlp_gate_w, mlp_up_w, mlp_down_w,
        cos, sin,
    )
    return _ref_rms_norm(x, norm_w, RMS_EPS)


def main() -> None:
    np.random.seed(46)

    inv_freq = _build_inv_freq()
    position_ids = _build_position_ids()
    cos, sin = _ref_cos_sin(position_ids, inv_freq)

    def randn(shape, dtype=mx.bfloat16, scale=0.1):
        a = np.random.randn(*shape).astype(np.float32) * scale
        return mx.array(a).astype(dtype)

    # Inputs.
    hidden = randn((B, S, HIDDEN), dtype=mx.bfloat16)
    next_embeds = randn((B, S, HIDDEN), dtype=mx.bfloat16)

    # Mtp top-level weights.
    pre_fc_norm_hidden_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    pre_fc_norm_embedding_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    fc_w = randn((HIDDEN, 2 * HIDDEN), dtype=mx.bfloat16)

    # Layer 0 weights.
    in_ln_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    q_w = randn((HQ * D * 2, HIDDEN), dtype=mx.bfloat16)
    k_w = randn((HKV * D, HIDDEN), dtype=mx.bfloat16)
    v_w = randn((HKV * D, HIDDEN), dtype=mx.bfloat16)
    o_w = randn((HIDDEN, HQ * D), dtype=mx.bfloat16)
    q_norm_w = randn((D,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    k_norm_w = randn((D,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    post_ln_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    mlp_gate_w = randn((INTERMEDIATE, HIDDEN), dtype=mx.bfloat16)
    mlp_up_w = randn((INTERMEDIATE, HIDDEN), dtype=mx.bfloat16)
    mlp_down_w = randn((HIDDEN, INTERMEDIATE), dtype=mx.bfloat16)
    norm_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])

    out = _ref_mtp(
        hidden, next_embeds,
        pre_fc_norm_hidden_w, pre_fc_norm_embedding_w, fc_w,
        in_ln_w,
        q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
        post_ln_w,
        mlp_gate_w, mlp_up_w, mlp_down_w,
        norm_w,
        cos, sin,
    )

    mx.eval(
        cos, sin, hidden, next_embeds,
        pre_fc_norm_hidden_w, pre_fc_norm_embedding_w, fc_w,
        in_ln_w, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
        post_ln_w, mlp_gate_w, mlp_up_w, mlp_down_w, norm_w,
        out,
    )

    def save(name: str, arr) -> None:
        path = OUT_DIR / f"{name}.npy"
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_hidden", hidden)
    save("input_next_embeds", next_embeds)
    save("input_position_ids", position_ids)
    save("input_inv_freq", inv_freq)
    save("pre_fc_norm_hidden_weight", pre_fc_norm_hidden_w)
    save("pre_fc_norm_embedding_weight", pre_fc_norm_embedding_w)
    save("fc_weight", fc_w)
    save("layer0_input_layernorm_weight", in_ln_w)
    save("layer0_q_proj_weight", q_w)
    save("layer0_k_proj_weight", k_w)
    save("layer0_v_proj_weight", v_w)
    save("layer0_o_proj_weight", o_w)
    save("layer0_q_norm_weight", q_norm_w)
    save("layer0_k_norm_weight", k_norm_w)
    save("layer0_post_attention_layernorm_weight", post_ln_w)
    save("layer0_mlp_gate_proj_weight", mlp_gate_w)
    save("layer0_mlp_up_proj_weight", mlp_up_w)
    save("layer0_mlp_down_proj_weight", mlp_down_w)
    save("norm_weight", norm_w)
    save("expected_cos", cos)
    save("expected_sin", sin)
    save("expected_mtp_out", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.3: Generate the fixtures**

```
cd ironmlx/tests/fixtures/p3b4_mtp && python gen_fixture.py
```

Expected: 23 `.npy` files written; output shapes match the README table.

- [ ] **Step 4.4: Write the failing integration test**

Create `ironmlx/tests/p3b4_mtp.rs`:

```rust
//! P3b4 Mtp numerical-correctness integration test.
//!
//! Loads `.npy` fixtures from `tests/fixtures/p3b4_mtp/` (generated by
//! `gen_fixture.py` against an independent Python reference of vllm-mlx's
//! `_MTPModule.mtp_forward`) and verifies that `nn::Mtp::forward` produces
//! numerically equivalent output.
//!
//! Tolerance: `atol = 1e-3` against the fp32-promoted reference, in line
//! with bf16 rounding limits in the matmul stages.
//!
//! Regenerate fixtures via:
//!
//! ```text
//! cd ironmlx/tests/fixtures/p3b4_mtp && python gen_fixture.py
//! ```

use mlx::{Array, Dtype};

use ironmlx::nn::{
    DecoderLayer, DecoderLayerConfig, GatedAttention, GatedAttentionConfig, Linear, Mlp, Mrope,
    Mtp, MtpConfig, RmsNorm,
};

const FIXTURE_DIR: &str = "tests/fixtures/p3b4_mtp";

const HIDDEN: i32 = 32;
const HQ: i32 = 4;
const HKV: i32 = 2;
const D: i32 = 8;
const INTERMEDIATE: i32 = 64;
const RMS_EPS: f32 = 1e-6;

fn load(name: &str) -> Array {
    let path = format!("{FIXTURE_DIR}/{name}.npy");
    mlx::io::load_npy(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).unwrap();
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).unwrap();
    let av: Vec<f32> = a32.to_vec().unwrap();
    let bv: Vec<f32> = b32.to_vec().unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn mtp_forward_matches_python_fixture() {
    // Load top-level Mtp weights.
    let pre_fc_h_w = load("pre_fc_norm_hidden_weight");
    let pre_fc_e_w = load("pre_fc_norm_embedding_weight");
    let fc_w = load("fc_weight");
    let mtp_norm_w = load("norm_weight");

    // Load layer 0 weights.
    let in_ln_w = load("layer0_input_layernorm_weight");
    let q_w = load("layer0_q_proj_weight");
    let k_w = load("layer0_k_proj_weight");
    let v_w = load("layer0_v_proj_weight");
    let o_w = load("layer0_o_proj_weight");
    let q_norm_w = load("layer0_q_norm_weight");
    let k_norm_w = load("layer0_k_norm_weight");
    let post_ln_w = load("layer0_post_attention_layernorm_weight");
    let gate_w = load("layer0_mlp_gate_proj_weight");
    let up_w = load("layer0_mlp_up_proj_weight");
    let down_w = load("layer0_mlp_down_proj_weight");

    let layer_cfg = DecoderLayerConfig {
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        num_heads: HQ,
        num_kv_heads: HKV,
        head_dim: D,
        rms_norm_eps: RMS_EPS,
        attention_bias: false,
    };

    let attn = GatedAttention::from_components(
        Linear::new_fp(q_w, None),
        Linear::new_fp(k_w, None),
        Linear::new_fp(v_w, None),
        Linear::new_fp(o_w, None),
        RmsNorm::new(q_norm_w, RMS_EPS),
        RmsNorm::new(k_norm_w, RMS_EPS),
        GatedAttentionConfig {
            num_heads: HQ,
            num_kv_heads: HKV,
            head_dim: D,
            rms_norm_eps: RMS_EPS,
            attention_bias: false,
        },
    );

    let mlp = Mlp::from_components(
        Linear::new_fp(gate_w, None),
        Linear::new_fp(up_w, None),
        Linear::new_fp(down_w, None),
    );

    let layer0 = DecoderLayer::from_components(
        RmsNorm::new(in_ln_w, RMS_EPS),
        attn,
        RmsNorm::new(post_ln_w, RMS_EPS),
        mlp,
        layer_cfg,
    );

    let cfg = MtpConfig {
        hidden_size: HIDDEN,
        num_mtp_layers: 1,
        layer: layer_cfg,
    };

    let mtp = Mtp::from_components(
        RmsNorm::new(pre_fc_h_w, RMS_EPS),
        RmsNorm::new(pre_fc_e_w, RMS_EPS),
        Linear::new_fp(fc_w, None),
        vec![layer0],
        RmsNorm::new(mtp_norm_w, RMS_EPS),
        cfg,
    );

    // Inputs + precomputed mrope tables.
    let hidden = load("input_hidden");
    let next_embeds = load("input_next_embeds");
    let cos = load("expected_cos");
    let sin = load("expected_sin");
    let expected = load("expected_mtp_out");

    // Mrope is reconstructed (the Rust path lazy-builds inv_freq + position_ids from
    // config) — but at forward time only the (cos, sin) tables are consumed, and we
    // load them from the fixture to bit-match the Python reference.
    let mrope = Mrope::new(D, 1e7, 1.0, &[2, 1, 1], true).unwrap();

    let out = mtp
        .forward(&hidden, &next_embeds, &mrope, &cos, &sin, None, None)
        .expect("forward");

    assert_eq!(out.shape().as_slice(), expected.shape().as_slice());
    assert_eq!(out.shape().as_slice(), &[1, 4, HIDDEN]);
    // expected dtype is fp32 (all RmsNorm weights in this fixture are fp32 → final
    // mtp.norm output is fp32). If a future fixture changes that, update accordingly.
    assert_eq!(out.dtype(), Dtype::Float32);
    assert_eq!(expected.dtype(), Dtype::Float32);

    let err = max_abs_diff(&out, &expected);
    assert!(err < 1e-3, "Mtp output max abs diff = {err} > 1e-3");
}
```

- [ ] **Step 4.5: Run, verify it passes**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b4_mtp
```

Expected: `1 passed`.

> If the test fails with `max abs diff > 1e-3`, the most likely culprits are
> (in order): (a) `concat` order swap (`[h, e]` vs `[e, h]`); (b) `fc.weight`
> shape mismatch (Rust expects `[H, 2H]` row-major, Python writes
> `[H, 2H]` via `mx.save`, both bf16); (c) RmsNorm eps drift between
> Rust call sites (every site uses `RMS_EPS = 1e-6`). Re-check those before
> tightening or loosening the tolerance.

- [ ] **Step 4.6: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 4.7: Run the full ironmlx P3b4 test set**

```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::mtp_cache
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b4_mtp
```

Expected: 4 + 5 + 2 + 1 = 12 tests passing across the four targets.

- [ ] **Step 4.8: Commit**

```bash
git add ironmlx/tests/fixtures/p3b4_mtp/ ironmlx/tests/p3b4_mtp.rs
git commit -m "$(cat <<'EOF'
test(ironmlx-p3b4): MTP integration test + Python reference fixture

Adds tests/fixtures/p3b4_mtp/ with a self-contained gen_fixture.py
re-implementing _MTPModule.mtp_forward using only mlx.core ops (no
mlx-lm patch dependency), plus the Rust integration test that loads
the 23 .npy artifacts and asserts max-abs-diff < 1e-3 against the
Python reference. Configuration matches MtpConfig: B=1 S=4 H=32
Hq=4 Hkv=2 D=8 intermediate=64 num_mtp_layers=1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Acceptance

After Tasks 1-4 are complete and committed, verify the spec's § 8 acceptance criteria are met:

- [ ] **Acceptance gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release && \
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::decoder_layer && \
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::mtp && \
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::mtp_cache && \
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p3b4_mtp
```

Expected: clean build + 12 tests passing.

- [ ] **Spec invariants confirmed**

  1. P3b2 `nn::GatedAttention` source unchanged: `git diff main..HEAD -- ironmlx/src/nn/gated_attention.rs` is empty.
  2. P1 `nn::Linear`/`nn::RmsNorm` source unchanged: `git diff main..HEAD -- ironmlx/src/nn/linear.rs ironmlx/src/nn/norm.rs` is empty.
  3. P2 `core::cache::KVCache` source unchanged: `git diff main..HEAD -- ironmlx/src/core/cache/kv_cache.rs` is empty.
  4. Only `nn::Mlp` gains a non-trivial change (the `from_components` test seam).
  5. Public API now exposes `nn::DecoderLayer`, `nn::DecoderLayerConfig`, `nn::Mtp`, `nn::MtpConfig`, `core::cache::MtpCache`.

- [ ] **Optional (smoke): assert public API surface**

```
cargo doc --release -p ironmlx --no-deps 2>&1 | rg -i 'warning|error' || echo "docs clean"
```

Expected: no rustdoc warnings on the new public types (their docstrings are linked via ../specs/...).

---

## Self-Review Notes (delete before merging)

These were checked while writing the plan; recording them so a future reader doesn't have to re-derive:

- **Spec coverage** — every spec §3 sub-section maps to a task: §3.1 DecoderLayerConfig (Task 1), §3.2 DecoderLayer struct + from_loader (Task 1), §3.3 forward / forward_on (Task 1), §3.4 MtpCache (Task 2), §3.5 MtpConfig (Task 3), §3.6 Mtp struct + from_loader (Task 3), §3.7 forward / forward_on / validate_inputs (Task 3). §4 testing maps to in-file unit tests (Tasks 1-3) plus the fixture-driven integration test (Task 4). §6 task split is preserved.
- **Spec deviations** — two construction-time dim sanity checks mentioned in spec § 3.2 (`mlp.hidden_size()`) and § 3.6 (`fc` `in_features` / `out_features`) are intentionally dropped, with rationale inline in Tasks 1.18 and 3.16: matmul will surface mismatches at first forward, matching the precedent set by the shipped `GatedAttention::from_loader`. No `Linear::in_features()` / `out_features()` accessor is added, keeping the change set minimal.
- **Type consistency** — `DecoderLayerConfig` fields match between `decoder_layer.rs` and the consumer `mtp.rs` (`MtpConfig::layer: DecoderLayerConfig`). `MtpCache::new_with_cap` takes the full `KVCache::new` parameter list (`batch, n_kv_heads, head_dim, v_head_dim, dtype, cap`), not the spec's lossy 3-arg version — this was forced by checking `kv_cache.rs:41`. Plan tests reflect that signature.
- **Concat order** — pinned to `[e, h]` in three places: spec § 1.1 + § 3.7, plan Task 3 Step 3.8 implementation, plan Task 3 Step 3.14 test (`c1/c0 ≈ 3.0`), and plan Task 4 Python ref (`mx.concatenate([e, h], axis=-1)`).
