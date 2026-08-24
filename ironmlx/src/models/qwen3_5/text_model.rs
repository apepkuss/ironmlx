//! Qwen3.5 text model — embed + N×DecoderLayer + final RmsNorm.
//!
//! Owns the per-instance Mrope so cos/sin tables are computed once per forward
//! and shared across all layers. Caller drives token-id input + per-layer
//! caches. Logit projection (tied or via lm_head) lives in [`super::Qwen35Model`].

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{
    AttnKind, DecoderLayer, DecoderLayerConfig, Embedding, LayerCache, Mrope, RmsNorm,
};
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

    /// DFlash2-only constructor. It materializes fused input projections one
    /// layer at a time and immediately releases the loader-owned source rows,
    /// keeping load-time peak memory bounded to one layer of duplicate weights.
    pub(crate) fn from_loader_dflash2(loader: &mut Loader, cfg: Qwen35Config) -> Result<Self> {
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")?;

        let head_dim = cfg.effective_head_dim();
        if cfg.rope_parameters.mrope_section.is_empty() {
            return Err(anyhow!(
                "Qwen35TextModel::from_loader_dflash2: rope_parameters.mrope_section must be non-empty"
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
            let prefix = format!("model.layers.{i}");
            layers.push(DecoderLayer::from_loader_dflash2(
                loader, &prefix, layer_cfg, kind,
            )?);

            let mut released = vec![
                format!("{prefix}.mlp.gate_proj"),
                format!("{prefix}.mlp.up_proj"),
            ];
            match kind {
                AttnKind::Full => released.extend([
                    format!("{prefix}.self_attn.q_proj"),
                    format!("{prefix}.self_attn.k_proj"),
                    format!("{prefix}.self_attn.v_proj"),
                ]),
                AttnKind::Linear => released.extend([
                    format!("{prefix}.linear_attn.in_proj_qkv"),
                    format!("{prefix}.linear_attn.in_proj_z"),
                    format!("{prefix}.linear_attn.in_proj_b"),
                    format!("{prefix}.linear_attn.in_proj_a"),
                ]),
            }
            loader.release_projection_prefixes(&released);
            mlx::clear_cache();
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

    pub fn hidden_dtype(&self) -> Dtype {
        self.embed_tokens.output_dtype()
    }

    pub fn mrope(&self) -> &Mrope {
        &self.mrope
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
                for (i, (layer, cell)) in self.layers.iter().zip(c.iter_mut()).enumerate() {
                    x = layer.forward_on_with_layer_idx(
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
                    x = layer.forward_on_with_layer_idx(
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

    pub(crate) fn restore_dflash2_speculative_prefix_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[crate::nn::LayerCacheSnapshot],
        accepted_len: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        if cache.len() != self.layers.len() || snapshots.len() != self.layers.len() {
            return Err(anyhow!(
                "Qwen35 DFlash2 restore requires {} cache layers, got cache={} snapshots={}",
                self.layers.len(),
                cache.len(),
                snapshots.len()
            ));
        }
        let target = target.into();
        for ((layer, cache), snapshot) in self.layers.iter().zip(cache).zip(snapshots) {
            layer.restore_speculative_prefix_on(cache, snapshot, accepted_len, target)?;
        }
        Ok(())
    }

    pub(crate) fn restore_dflash2_speculative_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[crate::nn::LayerCacheSnapshot],
        accepted_lens: &[usize],
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        if cache.len() != self.layers.len() || snapshots.len() != self.layers.len() {
            return Err(anyhow!(
                "Qwen35 DFlash2 per-row restore requires {} cache layers, got cache={} snapshots={}",
                self.layers.len(),
                cache.len(),
                snapshots.len()
            ));
        }
        let target = target.into();
        for ((layer, cache), snapshot) in self.layers.iter().zip(cache).zip(snapshots) {
            layer.restore_speculative_prefix_rows_on(cache, snapshot, accepted_lens, target)?;
        }
        Ok(())
    }

    /// DFlash2-only target forward that captures the post-layer hidden states
    /// selected by the draft checkpoint.
    ///
    /// This is a separate execution entry point so the existing text/MTP
    /// forwards retain their signatures and graphs unchanged. Captured tensors
    /// are concatenated on the feature axis in `target_layer_ids` order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_dflash2_taps_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target_layer_ids: &[usize],
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target = target.into();
        if input_ids.ndim() != 2 || input_ids.shape().as_slice()[0] <= 0 {
            return Err(anyhow!(
                "Qwen35 DFlash2 target forward requires input_ids [B,S] with B>0, got {:?}",
                input_ids.shape().as_slice()
            ));
        }
        if target_layer_ids.is_empty() {
            return Err(anyhow!(
                "Qwen35 DFlash2 target forward requires at least one target layer"
            ));
        }
        let mut previous = None;
        for &layer in target_layer_ids {
            if layer >= self.layers.len() {
                return Err(anyhow!(
                    "Qwen35 DFlash2 target layer {layer} is outside {} layers",
                    self.layers.len()
                ));
            }
            if previous.is_some_and(|prior| layer <= prior) {
                return Err(anyhow!(
                    "Qwen35 DFlash2 target layers must be strictly increasing"
                ));
            }
            previous = Some(layer);
        }
        if let Some(cache) = cache.as_deref() {
            if cache.len() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35 DFlash2 target cache has {} layers, expected {}",
                    cache.len(),
                    self.layers.len()
                ));
            }
        }

        let hidden = self.embed_on(input_ids, target)?;
        let (cos, sin) = self.mrope.cos_sin(position_ids)?;
        let mut x = hidden;
        let mut captured = Vec::with_capacity(target_layer_ids.len());
        let mut next_capture = 0_usize;
        match cache {
            Some(cache) => {
                for (index, (layer, layer_cache)) in
                    self.layers.iter().zip(cache.iter_mut()).enumerate()
                {
                    x = layer.forward_on_with_layer_idx(
                        &x,
                        &self.mrope,
                        &cos,
                        &sin,
                        None,
                        None,
                        None,
                        Some(layer_cache),
                        target,
                        index as i32,
                    )?;
                    if target_layer_ids.get(next_capture) == Some(&index) {
                        captured.push(x.clone());
                        next_capture += 1;
                    }
                }
            }
            None => {
                for (index, layer) in self.layers.iter().enumerate() {
                    x = layer.forward_on_with_layer_idx(
                        &x,
                        &self.mrope,
                        &cos,
                        &sin,
                        None,
                        None,
                        None,
                        None,
                        target,
                        index as i32,
                    )?;
                    if target_layer_ids.get(next_capture) == Some(&index) {
                        captured.push(x.clone());
                        next_capture += 1;
                    }
                }
            }
        }
        if captured.len() != target_layer_ids.len() {
            return Err(anyhow!(
                "Qwen35 DFlash2 captured {} layers, expected {}",
                captured.len(),
                target_layer_ids.len()
            ));
        }
        let refs = captured.iter().collect::<Vec<_>>();
        let context_hidden = mlx::ops::shape::concatenate_on(&refs, -1, target)?;
        Ok((self.norm.forward_on(&x, target)?, context_hidden))
    }

    /// Forward through embed → 32 × DecoderLayer → final RmsNorm.
    ///
    /// `input_ids: [B, S] uint32` — token ids.
    /// `position_ids: [3, B, S] int32` — three streams per Mrope contract.
    /// `cache: Some(slice)` — `slice.len() == self.num_layers()`; per-layer kind
    /// must match the layer's `AttnPath` (mismatch returns Err from DecoderLayer).
    /// Returns hidden states `[B, S, hidden_size]` (post-final-norm).
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

    /// Project hidden state to vocab logits via the (tied) `embed_tokens` matrix.
    pub fn as_output_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.as_output_on(hidden, target)
    }
}
