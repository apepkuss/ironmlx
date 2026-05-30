//! GLM-4.7-Flash (`glm4_moe_lite`) decoder layer.
//!
//! Mirrors `glm4_moe_lite.py:293-317` (`Glm4MoeLiteDecoderLayer`):
//! `input_layernorm → MlaAttention → residual → post_attention_layernorm →
//! FFN → residual`. The FFN is the dense [`Mlp`] for layer 0 (`first_k_dense_replace`)
//! and the noaux_tc [`Glm4MoeBlock`] for layers `>= first_k_dense_replace`.
//!
//! p5h profiling spans (see `qwen3_5_moe/decoder_layer.rs`) are intentionally
//! NOT wired here: this is the first correct GLM integration and the plain
//! residual path keeps the hot path free of profiling-only branching. A future
//! task can add `#[cfg(feature = "p5h-profile")]` spans mirroring the Qwen MoE
//! layer if GLM attribution is needed.

use anyhow::Result;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Mlp, RmsNorm};

use super::config::Glm4MoeLiteConfig;
use super::mla_attention::MlaAttention;
use super::mla_cache::MlaLatentCache;
use super::moe::Glm4MoeBlock;

/// Feed-forward sub-block: dense SwiGLU MLP (layer 0) or the MoE router block.
///
/// `Glm4MoeBlock` is large (stacked expert weights), so it is boxed to keep the
/// enum (and the `Vec<Glm4DecoderLayer>` element stride) small.
enum Ffn {
    Dense(Mlp),
    Moe(Box<Glm4MoeBlock>),
}

/// One GLM-4.7-Flash transformer decoder layer.
pub struct Glm4DecoderLayer {
    input_layernorm: RmsNorm,
    attn: MlaAttention,
    post_attention_layernorm: RmsNorm,
    ffn: Ffn,
}

impl Glm4DecoderLayer {
    /// Load all submodules at `model.layers.{layer_idx}.{name}`.
    pub fn from_loader(loader: &Loader, layer_idx: i32, cfg: &Glm4MoeLiteConfig) -> Result<Self> {
        let p = format!("model.layers.{layer_idx}");
        let input_layernorm =
            RmsNorm::from_loader(loader, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?;
        let attn = MlaAttention::from_loader(loader, &format!("{p}.self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{p}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let ffn = if cfg.is_moe_layer(layer_idx) {
            Ffn::Moe(Box::new(Glm4MoeBlock::from_loader(
                loader,
                &format!("{p}.mlp"),
                cfg,
            )?))
        } else {
            Ffn::Dense(Mlp::from_loader(loader, &format!("{p}.mlp"))?)
        };
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            ffn,
        })
    }

    /// Pre-norm transformer block: `h = x + attn(norm(x)); out = h + ffn(norm(h))`.
    ///
    /// `offset` is the per-row `[B]` i32 RoPE start position; `cache` is this
    /// layer's latent MLA cache; `per_row_lens` is the number of new tokens
    /// written this step; `mask` is the engine's additive float attention mask
    /// (`None` for decode, or for prefill where the model derives the causal
    /// mask itself).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        offset: &Array,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        let target = target.into();
        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn = self
            .attn
            .forward_on(&normed_in, offset, cache, per_row_lens, mask, target)?;
        let h = mlx::ops::binary::add_on(x, &attn, target)?;

        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let ffn_out = match &self.ffn {
            Ffn::Dense(m) => m.forward_on(&normed_post, target)?,
            Ffn::Moe(b) => b.forward_on(&normed_post, target, layer_idx)?,
        };
        Ok(mlx::ops::binary::add_on(&h, &ffn_out, target)?)
    }
}
