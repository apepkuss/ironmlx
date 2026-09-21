//! Standard Llama decoder layer.
//!
//! Pre-norm transformer block:
//! `h = x + attn(input_layernorm(x)); out = h + mlp(post_attention_layernorm(h))`.

use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Mlp, RmsNorm};
use crate::Result;

use super::attention::LlamaAttention;
use super::config::LlamaConfig;

/// One standard Llama transformer decoder layer.
pub struct LlamaDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: LlamaAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl LlamaDecoderLayer {
    /// Load all submodules at `model.layers.{layer_idx}.{name}`.
    pub fn from_loader(loader: &Loader, layer_idx: i32, cfg: &LlamaConfig) -> Result<Self> {
        let p = format!("model.layers.{layer_idx}");
        let input_layernorm =
            RmsNorm::from_loader(loader, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?;
        let self_attn = LlamaAttention::from_loader(
            loader,
            &format!("{p}.self_attn"),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.effective_head_dim(),
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
        )?;
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{p}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let mlp = Mlp::from_loader(loader, &format!("{p}.mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Pre-norm block forward. See [`LlamaAttention::forward_on`] for the
    /// `offset` / `per_row_lens` / `mask` semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        offset: &Array,
        offset_values: &[i32],
        cache: &mut KVCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        exact_batched_verify: bool,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn = self.self_attn.forward_on(
            &normed_in,
            offset,
            offset_values,
            per_row_lens,
            mask,
            cache,
            exact_batched_verify,
            target,
        )?;
        let h = mlx::ops::binary::add_on(x, &attn, target)?;

        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let ffn_out = self.mlp.forward_on(&normed_post, target)?;
        Ok(mlx::ops::binary::add_on(&h, &ffn_out, target)?)
    }
}
