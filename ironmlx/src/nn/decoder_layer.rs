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
        // `&Array + &Array` is panic-on-shape-mismatch; shape is guaranteed by pre-flight above.
        let h = x + &attn;

        // Block 2: post_attention_layernorm + mlp + residual
        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let mlp_out = self.mlp.forward_on(&normed_post, target)?;
        Ok(&h + &mlp_out)
    }

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

        // No construction-time dim checks — shape errors surface at first forward_on
        // (see GatedAttention::from_loader for the same pattern).
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            cfg,
        })
    }
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
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let o_w = rand_w(
            &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
            Dtype::Bfloat16,
        );

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

    #[test]
    fn forward_shape_and_dtype_bf16() {
        // bf16 input (with bf16 norm weights) → bf16 output preserved.
        let cfg = small_cfg();

        // bf16 attn + mlp weights matching small_cfg.
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let o_w = rand_w(
            &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
            Dtype::Bfloat16,
        );
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
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_residual_paths_zero_blocks_yield_input() {
        // Zero out attn (o_proj=0) AND mlp (down_proj=0); the two residual chains
        // independently reduce DecoderLayer to identity:  out = x + 0 + 0 = x.
        let cfg = small_cfg();

        // Build attention with o_proj weight = 0 → attn output is exactly 0.
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Float32,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Float32,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Float32,
        );
        let o_w_zero = Array::zeros(
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
        let down_w_zero =
            Array::zeros((cfg.hidden_size, cfg.intermediate_size), Dtype::Float32).unwrap();
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
}
