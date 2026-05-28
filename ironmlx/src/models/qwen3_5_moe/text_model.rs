//! Qwen3.5 MoE text model — embed + N×DecoderLayerMoe + final RmsNorm.
//!
//! Owns the per-instance Mrope so cos/sin tables are computed once per forward
//! and shared across all layers.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Embedding, LayerCache, Mrope, RmsNorm};
use crate::Result;

use super::config::Qwen35MoeConfig;
use super::decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};

pub struct Qwen35MoeTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayerMoe>,
    norm: RmsNorm,
    mrope: Mrope,
    cfg: Qwen35MoeConfig,
}

impl Qwen35MoeTextModel {
    pub fn from_loader(loader: &Loader, cfg: Qwen35MoeConfig) -> Result<Self> {
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")?;

        let head_dim = cfg.effective_head_dim();
        if cfg.rope_parameters.mrope_section.is_empty() {
            return Err(anyhow!(
                "Qwen35MoeTextModel::from_loader: rope_parameters.mrope_section must be non-empty"
            ));
        }
        let mrope = Mrope::new(
            head_dim,
            cfg.rope_parameters.rope_theta,
            cfg.rope_parameters.partial_rotary_factor,
            &cfg.rope_parameters.mrope_section,
            /* interleaved = */ true,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            let layer_cfg = DecoderLayerMoeConfig {
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
                linear_num_value_heads: cfg.linear_num_value_heads,
                linear_num_key_heads: cfg.linear_num_key_heads,
                linear_key_head_dim: cfg.linear_key_head_dim,
                linear_value_head_dim: cfg.linear_value_head_dim,
                linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
                num_experts: cfg.num_experts,
                num_experts_per_tok: cfg.num_experts_per_tok,
                norm_topk_prob: cfg.norm_topk_prob,
            };
            let kind = cfg.layer_kind(i);
            layers.push(DecoderLayerMoe::from_loader(
                loader,
                &format!("model.layers.{i}"),
                layer_cfg,
                kind,
            )?);
        }
        let norm = RmsNorm::from_loader(loader, "model.norm", cfg.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            mrope,
            cfg,
        })
    }

    /// Test seam — accept pre-built building blocks.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn from_components(
        embed_tokens: Embedding,
        layers: Vec<DecoderLayerMoe>,
        norm: RmsNorm,
        mrope: Mrope,
        cfg: Qwen35MoeConfig,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            norm,
            mrope,
            cfg,
        }
    }

    pub fn config(&self) -> &Qwen35MoeConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Embed token ids to hidden states `[B, S, hidden_size]`.
    pub fn embed_on(&self, input_ids: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.forward_on(input_ids, target)
    }

    /// Transformer + final-norm forward on pre-embedded hidden `[B, S, H]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_post_embedding_on(
        &self,
        hidden: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        attention_mask: Option<&Array>,
        linear_attention_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if let Some(c) = cache.as_deref() {
            if c.len() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35MoeTextModel::forward_post_embedding_on: cache.len()={} != num_layers={}",
                    c.len(),
                    self.layers.len()
                ));
            }
        }
        let (cos, sin) = self.mrope.cos_sin(position_ids)?;
        let mut x = hidden.clone();
        match cache {
            Some(c) => {
                for (i, (layer, cell)) in self.layers.iter().zip(c.iter_mut()).enumerate() {
                    x = layer.forward_on(
                        &x,
                        &self.mrope,
                        &cos,
                        &sin,
                        attention_mask,
                        linear_attention_mask,
                        per_row_lens,
                        Some(cell),
                        target,
                        i as i32,
                    )?;
                }
            }
            None => {
                for (i, layer) in self.layers.iter().enumerate() {
                    x = layer.forward_on(
                        &x,
                        &self.mrope,
                        &cos,
                        &sin,
                        attention_mask,
                        linear_attention_mask,
                        per_row_lens,
                        None,
                        target,
                        i as i32,
                    )?;
                }
            }
        }
        self.norm.forward_on(&x, target)
    }

    /// Full forward: embed → layers → final norm.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if input_ids.ndim() != 2 {
            return Err(anyhow!(
                "Qwen35MoeTextModel::forward_on: input_ids must be rank-2 [B, S], got rank {}",
                input_ids.ndim()
            ));
        }
        if let Some(c) = cache.as_deref() {
            if c.len() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35MoeTextModel::forward_on: cache.len()={} != num_layers={}",
                    c.len(),
                    self.layers.len()
                ));
            }
        }
        let hidden = self.embed_on(input_ids, target)?;
        self.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            decode_mask,
            None,
            per_row_lens,
            target,
        )
    }
}
