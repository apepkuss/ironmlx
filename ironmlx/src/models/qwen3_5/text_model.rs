//! Qwen3.5 text model — embed + N×DecoderLayer + final RmsNorm.
//!
//! Owns the per-instance Mrope so cos/sin tables are computed once per forward
//! and shared across all layers. Caller drives token-id input + per-layer
//! caches. Logit projection (tied or via lm_head) lives in [`super::Qwen35Model`].

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{DecoderLayer, DecoderLayerConfig, Embedding, LayerCache, Mrope, RmsNorm};
use crate::Result;

use super::config::Qwen35Config;

/// Qwen3.5 text-only core: embed_tokens + N×DecoderLayer + final RmsNorm + per-instance Mrope.
pub struct Qwen35TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    mrope: Mrope,
    cfg: Qwen35Config,
}

impl Qwen35TextModel {
    /// Production constructor. Reads `model.embed_tokens`, `model.layers.{i}.*`,
    /// `model.norm`. Constructs `Mrope` from `cfg.rope_parameters` + effective
    /// head_dim. Per-layer kind picked by `cfg.layer_kind(i)`.
    pub fn from_loader(loader: &Loader, cfg: Qwen35Config) -> Result<Self> {
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")?;

        let head_dim = cfg.effective_head_dim();
        if cfg.rope_parameters.mrope_section.is_empty() {
            return Err(anyhow!(
                "Qwen35TextModel::from_loader: rope_parameters.mrope_section must be non-empty"
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
            let layer_cfg = DecoderLayerConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: cfg.intermediate_size,
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
            };
            let kind = cfg.layer_kind(i);
            layers.push(DecoderLayer::from_loader(
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
    pub fn from_components(
        embed_tokens: Embedding,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        mrope: Mrope,
        cfg: Qwen35Config,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            norm,
            mrope,
            cfg,
        }
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Embed token ids to hidden states `[B, S, hidden_size]`.
    ///
    /// Thin wrapper around `embed_tokens.forward_on` exposed for the VL path so
    /// that `Qwen35Model::forward_vl` can embed first, inject vision embeddings,
    /// then continue through the transformer layers.
    pub fn embed_on(&self, input_ids: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.forward_on(input_ids, target)
    }

    /// Transformer + final-norm forward on a pre-embedded hidden state `[B, S, hidden_size]`.
    ///
    /// Runs `cos/sin → N×DecoderLayer → RmsNorm`, returns post-norm hidden states.
    /// The caller is responsible for validating `hidden` shape and cache length.
    pub fn forward_post_embedding_on(
        &self,
        hidden: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if let Some(c) = cache.as_deref() {
            if c.len() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35TextModel::forward_post_embedding_on: cache.len()={} != num_layers={}",
                    c.len(),
                    self.layers.len()
                ));
            }
        }
        let (cos, sin) = self.mrope.cos_sin(position_ids)?;
        let mut x = hidden.clone();
        match cache {
            Some(c) => {
                for (layer, cell) in self.layers.iter().zip(c.iter_mut()) {
                    x = layer.forward_on(&x, &self.mrope, &cos, &sin, None, Some(cell), target)?;
                }
            }
            None => {
                for layer in &self.layers {
                    x = layer.forward_on(&x, &self.mrope, &cos, &sin, None, None, target)?;
                }
            }
        }
        self.norm.forward_on(&x, target)
    }

    /// Forward through embed → 32 × DecoderLayer → final RmsNorm.
    ///
    /// `input_ids: [B, S] uint32` — token ids.
    /// `position_ids: [3, B, S] int32` — three streams per Mrope contract.
    /// `cache: Some(slice)` — `slice.len() == self.num_layers()`; per-layer kind
    /// must match the layer's `AttnPath` (mismatch returns Err from DecoderLayer).
    /// Returns hidden states `[B, S, hidden_size]` (post-final-norm).
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if input_ids.ndim() != 2 {
            return Err(anyhow!(
                "Qwen35TextModel::forward_on: input_ids must be rank-2 [B, S], got rank {}",
                input_ids.ndim()
            ));
        }
        if let Some(c) = cache.as_deref() {
            if c.len() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35TextModel::forward_on: cache.len()={} != num_layers={}",
                    c.len(),
                    self.layers.len()
                ));
            }
        }
        let hidden = self.embed_on(input_ids, target)?;
        self.forward_post_embedding_on(&hidden, position_ids, cache, target)
    }

    /// Project hidden state to vocab logits via the (tied) `embed_tokens` matrix.
    pub fn as_output_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.as_output_on(hidden, target)
    }
}
