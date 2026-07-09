//! Qwen3.5 MoE text model — embed + N×DecoderLayerMoe + final RmsNorm.
//!
//! Owns the per-instance Mrope so cos/sin tables are computed once per forward
//! and shared across all layers.

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

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

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct QwenMoeLayerDiff {
    pub layer_idx: usize,
    pub kind: crate::nn::AttnKind,
    pub row0_max_abs_diff: f32,
    pub row1_max_abs_diff: f32,
    pub row0_row1_max_abs_diff: f32,
}

#[cfg(test)]
fn max_abs_diff(a: &Array, b: &Array) -> Result<f32> {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32)?;
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32)?;
    let av: Vec<f32> = a32.to_vec()?;
    let bv: Vec<f32> = b32.to_vec()?;
    if av.len() != bv.len() {
        return Err(anyhow!(
            "max_abs_diff: len mismatch {} != {}",
            av.len(),
            bv.len()
        ));
    }
    Ok(av
        .iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max))
}

#[cfg(test)]
fn slice_last_hidden(hidden: &Array, row: i32, pos: i32, target: StreamOrDevice) -> Result<Array> {
    let dims = hidden.shape();
    let dims = dims.as_slice();
    if dims.len() != 3 {
        return Err(anyhow!(
            "slice_last_hidden: hidden must be [B,S,H], got rank {}",
            dims.len()
        ));
    }
    Ok(mlx::ops::indexing::slice_strided_on(
        hidden,
        &[row, pos, 0_i32][..],
        &[row + 1, pos + 1, dims[2]][..],
        &[1_i32, 1, 1][..],
        target,
    )?)
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

    pub fn hidden_dtype(&self) -> Dtype {
        self.embed_tokens.output_dtype()
    }

    pub fn mrope(&self) -> &Mrope {
        &self.mrope
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

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_b1_b2_same_prompt_layer_diffs_on(
        &self,
        input_b1: &Array,
        position_b1: &Array,
        cache_b1: &mut [LayerCache],
        input_b2: &Array,
        position_b2: &Array,
        cache_b2: &mut [LayerCache],
        attention_mask_b2: &Array,
        linear_attention_mask_b2: &Array,
        per_row_lens_b2: &[i32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<Vec<QwenMoeLayerDiff>> {
        let target = target.into();
        if per_row_lens_b2.len() != 2 || per_row_lens_b2[0] != per_row_lens_b2[1] {
            return Err(anyhow!(
                "debug_b1_b2_same_prompt_layer_diffs_on requires two equal-length rows"
            ));
        }
        let last_pos = per_row_lens_b2[0] - 1;
        let mut x1 = self.embed_on(input_b1, target)?;
        let mut x2 = self.embed_on(input_b2, target)?;
        let (cos1, sin1) = self.mrope.cos_sin(position_b1)?;
        let (cos2, sin2) = self.mrope.cos_sin(position_b2)?;
        let mut out = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            x1 = layer.forward_on(
                &x1,
                &self.mrope,
                &cos1,
                &sin1,
                None,
                None,
                None,
                Some(&mut cache_b1[i]),
                target,
                i as i32,
            )?;
            x2 = layer.forward_on(
                &x2,
                &self.mrope,
                &cos2,
                &sin2,
                Some(attention_mask_b2),
                Some(linear_attention_mask_b2),
                Some(per_row_lens_b2),
                Some(&mut cache_b2[i]),
                target,
                i as i32,
            )?;
            let last_b1 = slice_last_hidden(&x1, 0, last_pos, target)?;
            let last_b2_row0 = slice_last_hidden(&x2, 0, last_pos, target)?;
            let last_b2_row1 = slice_last_hidden(&x2, 1, last_pos, target)?;
            out.push(QwenMoeLayerDiff {
                layer_idx: i,
                kind: layer.kind(),
                row0_max_abs_diff: max_abs_diff(&last_b1, &last_b2_row0)?,
                row1_max_abs_diff: max_abs_diff(&last_b1, &last_b2_row1)?,
                row0_row1_max_abs_diff: max_abs_diff(&last_b2_row0, &last_b2_row1)?,
            });
        }
        Ok(out)
    }
}
