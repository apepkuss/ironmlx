//! Decoder layer for Qwen3.5 MoE — same hybrid attention as dense
//! but FFN is SparseMoeBlock instead of nn::Mlp. See spec §3.5.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{
    AttnKind, AttnPath, GatedAttention, GatedAttentionConfig, GatedDeltaNet, GatedDeltaNetConfig,
    LayerCache, Mrope, RmsNorm,
};
use crate::Result;

use super::sparse_moe::SparseMoeBlock;

/// Config for `DecoderLayerMoe`. Mirrors `nn::DecoderLayerConfig` plus MoE
/// fields needed by `SparseMoeBlock`.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerMoeConfig {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
}

pub struct DecoderLayerMoe {
    input_layernorm: RmsNorm,
    attn: AttnPath,
    post_attention_layernorm: RmsNorm,
    ffn: SparseMoeBlock,
    cfg: DecoderLayerMoeConfig,
}

impl DecoderLayerMoe {
    /// Production constructor. Reads `{prefix}.input_layernorm`,
    /// `{prefix}.{self_attn,linear_attn}.*`, `{prefix}.post_attention_layernorm`,
    /// and `{prefix}.mlp.*` (handled by SparseMoeBlock).
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: DecoderLayerMoeConfig,
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
        let ffn =
            SparseMoeBlock::from_loader(loader, &format!("{prefix}.mlp"), cfg.num_experts_per_tok)?;
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            ffn,
            cfg,
        })
    }

    pub fn config(&self) -> &DecoderLayerMoeConfig {
        &self.cfg
    }

    pub fn kind(&self) -> AttnKind {
        match &self.attn {
            AttnPath::Full(_) => AttnKind::Full,
            AttnPath::Linear(_) => AttnKind::Linear,
        }
    }

    /// Stream-targeted forward pass.
    /// Mirrors `nn::DecoderLayer::forward_on` exactly except FFN is SparseMoeBlock.
    ///
    /// `layer_idx` — index of this decoder block in the model stack. Threaded
    /// unconditionally (default build + p5h-profile build) into all attention
    /// and FFN callees so they can construct P5h `SpanFields { layer_idx }`
    /// without their callers needing to know the index. Non-decoder callers
    /// (CLI / standalone tests) pass `-1` (spec § 2.5a).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        full_attn_mask: Option<&Array>,
        linear_attn_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut LayerCache>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        let target = target.into();

        if x.ndim() != 3 {
            return Err(anyhow!(
                "DecoderLayerMoe::forward_on: x must be rank-3 [B, S, hidden_size], got rank {}",
                x.ndim()
            ));
        }

        #[cfg(feature = "p5h-profile")]
        {
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "decoder_layer_N",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    // (1) input_norm
                    let normed_in = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "input_norm",
                        || crate::core::p5h::SpanFields {
                            layer_idx: Some(layer_idx),
                            ..Default::default()
                        },
                        || self.input_layernorm.forward_on(x, target),
                    )?;

                    // (2) attention_path WRAPPER (GDN substeps emit inside the
                    // GatedDeltaNet body; full-attn substeps to be filled by T2).
                    let attn = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "attention_path",
                        || crate::core::p5h::SpanFields {
                            layer_idx: Some(layer_idx),
                            ..Default::default()
                        },
                        || -> Result<Array> {
                            match (&self.attn, cache) {
                                (AttnPath::Full(a), Some(LayerCache::Full(kv))) => a.forward_on(
                                    &normed_in,
                                    mrope,
                                    cos,
                                    sin,
                                    full_attn_mask,
                                    linear_attn_mask,
                                    per_row_lens,
                                    Some(kv),
                                    target,
                                    layer_idx,
                                ),
                                (AttnPath::Full(a), None) => a.forward_on(
                                    &normed_in,
                                    mrope,
                                    cos,
                                    sin,
                                    full_attn_mask,
                                    linear_attn_mask,
                                    per_row_lens,
                                    None,
                                    target,
                                    layer_idx,
                                ),
                                (AttnPath::Linear(a), Some(LayerCache::Linear(gdc))) => a
                                    .forward_on(
                                        &normed_in,
                                        linear_attn_mask,
                                        per_row_lens,
                                        Some(gdc),
                                        target,
                                        layer_idx,
                                    ),
                                (AttnPath::Linear(a), None) => a.forward_on(
                                    &normed_in,
                                    linear_attn_mask,
                                    per_row_lens,
                                    None,
                                    target,
                                    layer_idx,
                                ),
                                (AttnPath::Full(_), Some(LayerCache::Linear(_))) => Err(anyhow!(
                                    "DecoderLayerMoe::forward_on: Full attn layer received Linear cache (kind mismatch)"
                                )),
                                (AttnPath::Linear(_), Some(LayerCache::Full(_))) => Err(anyhow!(
                                    "DecoderLayerMoe::forward_on: Linear attn layer received Full cache (kind mismatch)"
                                )),
                            }
                        },
                    )?;

                    // (3) residual_overhead — residual add 1 (x + attn).
                    let h = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "residual_overhead",
                        || crate::core::p5h::SpanFields {
                            layer_idx: Some(layer_idx),
                            ..Default::default()
                        },
                        || -> Result<Array> { Ok(x + &attn) },
                    )?;

                    // (4) post_attention_norm
                    let normed_post = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "post_attention_norm",
                        || crate::core::p5h::SpanFields {
                            layer_idx: Some(layer_idx),
                            ..Default::default()
                        },
                        || self.post_attention_layernorm.forward_on(&h, target),
                    )?;

                    // (5) mlp_path WRAPPER (8 MoE substeps to be filled by T3).
                    let ffn_out = crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "mlp_path",
                        || crate::core::p5h::SpanFields {
                            layer_idx: Some(layer_idx),
                            ..Default::default()
                        },
                        || self.ffn.forward_on(&normed_post, target, layer_idx),
                    )?;

                    // (6) residual_overhead — residual add 2 (h + ffn_out). Same
                    // span name as (3); distinct span_id under the same
                    // decoder_layer_N parent.
                    crate::core::p5h::try_with_p5h_span_from_current_trace(
                        "residual_overhead",
                        || crate::core::p5h::SpanFields {
                            layer_idx: Some(layer_idx),
                            ..Default::default()
                        },
                        || -> Result<Array> { Ok(&h + &ffn_out) },
                    )
                },
            )
        }

        #[cfg(not(feature = "p5h-profile"))]
        {
            // Block 1: input_layernorm + attn dispatch + residual
            let normed_in = self.input_layernorm.forward_on(x, target)?;
            let attn = match (&self.attn, cache) {
                (AttnPath::Full(a), Some(LayerCache::Full(kv))) => a.forward_on(
                    &normed_in,
                    mrope,
                    cos,
                    sin,
                    full_attn_mask,
                    linear_attn_mask,
                    per_row_lens,
                    Some(kv),
                    target,
                    layer_idx,
                )?,
                (AttnPath::Full(a), None) => a.forward_on(
                    &normed_in,
                    mrope,
                    cos,
                    sin,
                    full_attn_mask,
                    linear_attn_mask,
                    per_row_lens,
                    None,
                    target,
                    layer_idx,
                )?,
                (AttnPath::Linear(a), Some(LayerCache::Linear(gdc))) => a.forward_on(
                    &normed_in,
                    linear_attn_mask,
                    per_row_lens,
                    Some(gdc),
                    target,
                    layer_idx,
                )?,
                (AttnPath::Linear(a), None) => a.forward_on(
                    &normed_in,
                    linear_attn_mask,
                    per_row_lens,
                    None,
                    target,
                    layer_idx,
                )?,
                (AttnPath::Full(_), Some(LayerCache::Linear(_))) => {
                    return Err(anyhow!(
                        "DecoderLayerMoe::forward_on: Full attn layer received Linear cache (kind mismatch)"
                    ));
                }
                (AttnPath::Linear(_), Some(LayerCache::Full(_))) => {
                    return Err(anyhow!(
                        "DecoderLayerMoe::forward_on: Linear attn layer received Full cache (kind mismatch)"
                    ));
                }
            };
            let h = x + &attn;

            // Block 2: post_norm + SparseMoeBlock + residual
            let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
            let ffn_out = self.ffn.forward_on(&normed_post, target, layer_idx)?;
            Ok(&h + &ffn_out)
        }
    }
}
