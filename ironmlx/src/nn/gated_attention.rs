//! Gated full attention block — Qwen3.5 / Qwen3-Next canonical attention.
//!
//! Mirrors mlx-lm's `Qwen3NextAttention` (`/Volumes/Dev/mlx-lm/mlx_lm/models/qwen3_next.py`).
//! `qwen3_5.py` imports it directly: `from .qwen3_next import Qwen3NextAttention as Attention`.
//!
//! Differs from P1 [`crate::nn::Attention`] (standard) in exactly two places:
//! 1. `q_proj` produces `num_heads * head_dim * 2` outputs; the second half is the gate.
//! 2. After SDPA + reshape, the result is element-wise multiplied by `sigmoid(gate)` before
//!    `o_proj`.
//!
//! See P3b2 spec § 2 for the data flow.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Linear, Mrope, RmsNorm};
use crate::Result;

/// Configuration for [`GatedAttention`].
///
/// Notably differs from [`crate::nn::AttentionConfig`] by:
/// - `attention_bias` field (Qwen3.5: false; carried from model config)
/// - No `has_qk_norm` field — Qwen3.5 always has q/k_norm
#[derive(Debug, Clone, Copy)]
pub struct GatedAttentionConfig {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
}

/// Qwen3.5 / Qwen3-Next gated full attention block.
#[allow(dead_code)]
pub struct GatedAttention {
    q_proj: Linear, // out = num_heads * head_dim * 2
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    cfg: GatedAttentionConfig,
    scale: f32,
}

impl GatedAttention {
    /// Production constructor: load from a project [`Loader`].
    ///
    /// Reads `{prefix}.{q,k,v,o}_proj.{weight,bias?,scales?,biases?}` and
    /// `{prefix}.{q,k}_norm.weight`. `bias` presence is auto-detected per
    /// [`Linear::from_loader`].
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedAttentionConfig) -> Result<Self> {
        let q_proj = Linear::from_loader(loader, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::from_loader(loader, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::from_loader(loader, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::from_loader(loader, &format!("{prefix}.o_proj"))?;
        let q_norm = RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?;

        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cfg,
            scale,
        })
    }

    /// Test/composition seam: build a `GatedAttention` from pre-built nn building blocks.
    ///
    /// Used by unit tests and the integration fixture path to avoid synthesizing a real
    /// `model_dir/safetensors` for tiny test cases. Production code uses [`from_loader`].
    #[allow(dead_code)]
    pub(crate) fn from_components(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        cfg: GatedAttentionConfig,
    ) -> Self {
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cfg,
            scale,
        }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &GatedAttentionConfig {
        &self.cfg
    }

    /// Forward pass — see [`forward_on`](Self::forward_on) for stream-targeted variant.
    ///
    /// **Stub at T1**: returns `Err`. Real implementation lands in T2.
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

    /// Stream-targeted forward — currently stubbed (T2 fills body).
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
        let _ = (x, mrope, cos, sin, mask, cache, target);
        Err(anyhow!(
            "GatedAttention::forward not implemented at T1 — body lands in T2"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::constructors;
    use mlx::{Array, Dtype};

    /// Build a small synthetic GatedAttention for unit tests.
    /// B=1, S=4, Hq=4, Hkv=2, D=8, hidden=32; partial=1.0 → rot_dim=8.
    fn small_gated_attention() -> GatedAttention {
        // q_proj: [Hq*D*2=64, hidden=32]
        let q_w = Array::zeros((64_i32, 32), Dtype::Float32).unwrap();
        let k_w = Array::zeros((16_i32, 32), Dtype::Float32).unwrap();
        let v_w = Array::zeros((16_i32, 32), Dtype::Float32).unwrap();
        let o_w = Array::zeros((32_i32, 32), Dtype::Float32).unwrap();
        let q_n = constructors::ones((8_i32,), Dtype::Float32).unwrap();
        let k_n = constructors::ones((8_i32,), Dtype::Float32).unwrap();

        let cfg = GatedAttentionConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
        };

        GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(q_n, cfg.rms_norm_eps),
            RmsNorm::new(k_n, cfg.rms_norm_eps),
            cfg,
        )
    }

    #[test]
    fn from_components_carries_config() {
        let attn = small_gated_attention();
        let cfg = attn.config();
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.head_dim, 8);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(!cfg.attention_bias);
    }

    #[test]
    fn from_components_computes_scale() {
        let attn = small_gated_attention();
        // scale = 1 / sqrt(head_dim=8)
        let expected = 1.0 / 8.0_f32.sqrt();
        assert!((attn.scale - expected).abs() < 1e-6);
    }

    #[test]
    fn forward_returns_err_at_t1() {
        let attn = small_gated_attention();
        let x = Array::zeros((1_i32, 4, 32), Dtype::Bfloat16).unwrap();
        let mrope = Mrope::new(8, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let cos = Array::zeros((1_i32, 4, 4), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 4, 4), Dtype::Float32).unwrap();

        let r = attn.forward(&x, &mrope, &cos, &sin, None, None);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("not implemented at T1"), "msg: {msg}");
    }
}
