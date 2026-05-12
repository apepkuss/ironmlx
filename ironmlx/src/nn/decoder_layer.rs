//! Single Qwen3.5 / Qwen3-Next decoder block.
//!
//! Mirrors mlx-lm `Qwen3NextDecoderLayer.__call__`:
//!
//! ```text
//! r   = self_attn_or_linear_attn(input_layernorm(x), mask, cache)
//! h   = x + r
//! out = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! The attention path is selected at construction time per `AttnKind`. Full-
//! attention layers consume `KVCache`; linear-attention SSM layers consume
//! `GatedDeltaCache`. Both are wrapped uniformly via [`LayerCache`].

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::Loader;
use crate::nn::{
    GatedAttention, GatedAttentionConfig, GatedDeltaNet, GatedDeltaNetConfig, Mlp, Mrope, RmsNorm,
};
use crate::Result;

/// Which attention path a [`DecoderLayer`] uses. Selected per layer index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Standard gated full attention (P3b2). Consumes [`KVCache`].
    Full,
    /// Gated delta-net linear attention SSM (P3b3). Consumes [`GatedDeltaCache`].
    Linear,
}

/// Configuration for [`DecoderLayer`]. Mirrors the subset of Qwen3.5
/// `TextModelArgs` that drives a single decoder block.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    /// Linear-attn parameters (only consulted when `AttnKind::Linear`).
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
}

/// Attention path variant — owns either a full-attention or a linear-attention block.
///
/// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can construct it
/// via [`DecoderLayer::from_components_full`] / [`DecoderLayer::from_components_linear`].
#[doc(hidden)]
pub enum AttnPath {
    Full(GatedAttention),
    Linear(GatedDeltaNet),
}

/// Per-layer cache, paired with [`AttnPath`].
#[doc(hidden)]
pub enum LayerCache {
    Full(KVCache),
    Linear(GatedDeltaCache),
}

/// One decoder block. Full or linear attention selected at construction.
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    attn: AttnPath,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    cfg: DecoderLayerConfig,
}

impl DecoderLayer {
    /// Test/composition seam — full-attention variant. Equivalent to P3b4's
    /// `from_components` (renamed for symmetry with the linear-attn variant).
    #[doc(hidden)]
    pub fn from_components_full(
        input_layernorm: RmsNorm,
        self_attn: GatedAttention,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        cfg: DecoderLayerConfig,
    ) -> Self {
        Self {
            input_layernorm,
            attn: AttnPath::Full(self_attn),
            post_attention_layernorm,
            mlp,
            cfg,
        }
    }

    /// Test/composition seam — linear-attention SSM variant.
    #[doc(hidden)]
    pub fn from_components_linear(
        input_layernorm: RmsNorm,
        linear_attn: GatedDeltaNet,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        cfg: DecoderLayerConfig,
    ) -> Self {
        Self {
            input_layernorm,
            attn: AttnPath::Linear(linear_attn),
            post_attention_layernorm,
            mlp,
            cfg,
        }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &DecoderLayerConfig {
        &self.cfg
    }

    /// Which path this layer uses (introspection helper for the test/cache layer).
    pub fn kind(&self) -> AttnKind {
        match &self.attn {
            AttnPath::Full(_) => AttnKind::Full,
            AttnPath::Linear(_) => AttnKind::Linear,
        }
    }

    /// Pre-flight: enforce rank-3 input + last-axis matches `cfg.hidden_size`.
    /// `caller` is embedded in the diagnostic so callers (forward_on,
    /// forward_on_full_kv) surface in the error string.
    #[inline]
    fn preflight_x(&self, x: &Array, caller: &str) -> Result<()> {
        if x.ndim() != 3 {
            return Err(anyhow!(
                "{caller}: x must be rank-3 [B, S, hidden_size], got rank {}",
                x.ndim()
            ));
        }
        let dims_owned = x.shape();
        let dims = dims_owned.as_slice();
        if dims[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "{caller}: x last-axis = {} but cfg.hidden_size = {}",
                dims[2],
                self.cfg.hidden_size
            ));
        }
        Ok(())
    }

    /// Default-stream forward pass. The single `mask` parameter is interpreted
    /// per layer kind: the full-attention path treats it as the SDPA-style
    /// `[B, 1, T_q, T_kv]` additive mask, the linear-attention path treats it
    /// as the `[B, T]` boolean per-token validity mask. For hybrid models that
    /// need to pass different masks to the two paths, call
    /// [`Self::forward_on`] directly.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
    ) -> Result<Array> {
        // Convenience: forward `mask` to whichever path applies. The other
        // gets `None`. Callers that need to populate both should use
        // `forward_on` directly.
        let (full_mask, linear_mask) = match self.kind() {
            AttnKind::Full => (mask, None),
            AttnKind::Linear => (None, mask),
        };
        self.forward_on(x, mrope, cos, sin, full_mask, linear_mask, cache, ())
    }

    /// Stream-targeted forward.
    ///
    /// `x: [B, S, hidden_size]` → `[B, S, hidden_size]`. Cache type must match
    /// `self.kind()`; mismatch returns `Err`. Linear-attn ignores `mrope`/`cos`/`sin`
    /// (passed through for signature uniformity with the Full path).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        full_attn_mask: Option<&Array>,
        linear_attn_mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Pre-flight (existing P3b4 invariants).
        self.preflight_x(x, "DecoderLayer::forward_on")?;

        // Block 1: input_layernorm + attn dispatch + residual
        //
        // Hybrid routing: full-attention layers consume `full_attn_mask`
        // (`[B, 1, T_q, T_kv]` additive bf16 for SDPA); linear-attention
        // layers consume `linear_attn_mask` (`[B, T]` boolean per-token
        // validity for the `gated_delta_step` kernel). The two have
        // incompatible shapes and dtypes — they cannot be unified.
        let normed_in = self.input_layernorm.forward_on(x, target)?;
        // Full attention also consumes `linear_attn_mask` (when Some) as its
        // K/V-validity mask, zeroing pad-position K/V cells before the cache
        // write. The `[B, T]` boolean shape and "real-vs-pad per token"
        // semantics are identical to what linear attention uses; reusing it
        // avoids defining a third mask. See `attention::forward_on` for
        // details.
        let attn = match (&self.attn, cache) {
            (AttnPath::Full(a), Some(LayerCache::Full(kv))) => a.forward_on(
                &normed_in,
                mrope,
                cos,
                sin,
                full_attn_mask,
                linear_attn_mask,
                Some(kv),
                target,
            )?,
            (AttnPath::Full(a), None) => a.forward_on(
                &normed_in,
                mrope,
                cos,
                sin,
                full_attn_mask,
                linear_attn_mask,
                None,
                target,
            )?,
            (AttnPath::Linear(a), Some(LayerCache::Linear(gdc))) => {
                a.forward_on(&normed_in, linear_attn_mask, Some(gdc), target)?
            }
            (AttnPath::Linear(a), None) => {
                a.forward_on(&normed_in, linear_attn_mask, None, target)?
            }
            (AttnPath::Full(_), Some(LayerCache::Linear(_))) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on: Full attn layer received Linear cache (kind mismatch)"
                ));
            }
            (AttnPath::Linear(_), Some(LayerCache::Full(_))) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on: Linear attn layer received Full cache (kind mismatch)"
                ));
            }
        };
        let h = x + &attn;

        // Block 2: post_norm + mlp + residual
        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let mlp_out = self.mlp.forward_on(&normed_post, target)?;
        Ok(&h + &mlp_out)
    }

    /// Production constructor. `kind` selects which attention path to load
    /// (Full → reads `{prefix}.self_attn.*`; Linear → reads `{prefix}.linear_attn.*`).
    ///
    /// No construction-time dim sanity checks — Linear's matmul surfaces shape errors
    /// at first forward_on (matches GatedAttention::from_loader precedent).
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: DecoderLayerConfig,
        kind: AttnKind,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.input_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let attn = match kind {
            AttnKind::Full => {
                let ga = GatedAttention::from_loader(
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
                AttnPath::Full(ga)
            }
            AttnKind::Linear => {
                let gdn = GatedDeltaNet::from_loader(
                    loader,
                    &format!("{prefix}.linear_attn"),
                    GatedDeltaNetConfig {
                        hidden_size: cfg.hidden_size,
                        num_v_heads: cfg.linear_num_value_heads,
                        num_k_heads: cfg.linear_num_key_heads,
                        head_k_dim: cfg.linear_key_head_dim,
                        head_v_dim: cfg.linear_value_head_dim,
                        conv_kernel_size: cfg.linear_conv_kernel_dim,
                        rms_norm_eps: cfg.rms_norm_eps,
                    },
                )?;
                AttnPath::Linear(gdn)
            }
        };
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let mlp = Mlp::from_loader(loader, &format!("{prefix}.mlp"))?;
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
            cfg,
        })
    }
}

impl DecoderLayer {
    /// Package-private helper for [`crate::nn::Mtp`]: same as [`forward_on`](Self::forward_on)
    /// but accepts `Option<&mut KVCache>` directly, avoiding a wrapper allocation.
    ///
    /// Returns `Err` if called on a `Linear` layer (MTP layers are always Full).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_on_full_kv(
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

        self.preflight_x(x, "DecoderLayer::forward_on_full_kv")?;

        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn_out = match &self.attn {
            AttnPath::Full(a) => {
                a.forward_on(&normed_in, mrope, cos, sin, mask, None, cache, target)?
            }
            AttnPath::Linear(_) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on_full_kv: called on Linear layer (MTP requires Full)"
                ));
            }
        };
        let h = x + &attn_out;

        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let mlp_out = self.mlp.forward_on(&normed_post, target)?;
        Ok(&h + &mlp_out)
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
            linear_num_value_heads: 0,
            linear_num_key_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
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

        DecoderLayer::from_components_full(
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
        let layer = DecoderLayer::from_components_full(
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

        let layer = DecoderLayer::from_components_full(
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

    #[test]
    fn from_components_full_carries_kind_and_config() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg); // existing helper builds Full variant
        assert_eq!(layer.kind(), AttnKind::Full);
        assert_eq!(layer.config().hidden_size, cfg.hidden_size);
    }

    #[test]
    fn from_components_linear_carries_kind() {
        // GatedDeltaNet::from_components requires P3b3 internals (Conv1d,
        // RmsNormGated, Linear etc.) that are heavy to wire up here. Keep this
        // test symbolic — verify the AttnPath::Linear and LayerCache::Linear
        // discriminators compile. Concrete construction is exercised in T4.
        let _ = AttnPath::Linear;
        let _ = LayerCache::Linear;
    }

    #[test]
    fn full_layer_with_linear_cache_errors() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg);
        let mut bad_cache = LayerCache::Linear(
            GatedDeltaCache::new_with_cap(
                /* batch */ 1,
                /* kernel_size */ 4,
                /* conv_dim */ 16,
                /* num_v_heads */ 4,
                /* head_v_dim */ 8,
                /* head_k_dim */ 8,
                mlx::Dtype::Bfloat16,
                /* cap */ 16,
            )
            .expect("GatedDeltaCache::new_with_cap"),
        );
        let (x, mrope, cos, sin) = build_inputs_fp32(cfg);
        let r = layer.forward(&x, &mrope, &cos, &sin, None, Some(&mut bad_cache));
        let err = r.expect_err("Full layer + Linear cache must Err");
        let msg = format!("{err}");
        assert!(
            msg.contains("kind mismatch") && msg.contains("Linear cache"),
            "expected kind-mismatch message, got: {msg}"
        );
    }

    #[test]
    fn linear_cache_full_arm_compiles() {
        // The Linear-layer + Full-cache mismatch arm in forward_on requires
        // a real GatedDeltaNet to construct (heavy P3b3 internals). The
        // dispatch arm itself is exercised in T4 (Qwen35Model assembly tests);
        // here we only confirm the LayerCache::Full discriminator compiles.
        let _ = LayerCache::Full;
    }
}
