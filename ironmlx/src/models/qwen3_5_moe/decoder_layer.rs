//! Decoder layer for Qwen3.5 MoE — same hybrid attention as dense
//! but FFN is SparseMoeBlock instead of nn::Mlp. See spec §3.5.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::{cache::KVCache, Loader};
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
    pub norm_topk_prob: bool,
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
                AttnPath::Full(Box::new(ga))
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
                AttnPath::Linear(Box::new(gdn))
            }
        };
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let ffn = SparseMoeBlock::from_loader(
            loader,
            &format!("{prefix}.mlp"),
            cfg.num_experts_per_tok,
            cfg.norm_topk_prob,
        )?;
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
    /// into attention and FFN callees for layer-aware diagnostics. Non-decoder
    /// callers (CLI / standalone tests) pass `-1`.
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
                (_, Some(LayerCache::Mla(_))) => {
                    return Err(anyhow!(
                        "DecoderLayerMoe::forward_on: received Mla cache (kind mismatch)"
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

    /// Helper for [`super::mtp::Qwen35MoeMtp`]: MTP layers are always full attention,
    /// so accept a raw KV cache slot and reject Linear-attention layers explicitly.
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

        if x.ndim() != 3 {
            return Err(anyhow!(
                "DecoderLayerMoe::forward_on_full_kv: x must be rank-3 [B, S, hidden_size], got rank {}",
                x.ndim()
            ));
        }
        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        if dims[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "DecoderLayerMoe::forward_on_full_kv: x last-axis = {} but cfg.hidden_size = {}",
                dims[2],
                self.cfg.hidden_size
            ));
        }

        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn_out = match &self.attn {
            AttnPath::Full(a) => a.forward_on(
                &normed_in, mrope, cos, sin, mask, None, None, cache, target, -1,
            )?,
            AttnPath::Linear(_) => {
                return Err(anyhow!(
                    "DecoderLayerMoe::forward_on_full_kv: called on Linear layer (MTP requires Full)"
                ));
            }
        };
        let h = x + &attn_out;

        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let ffn_out = self.ffn.forward_on(&normed_post, target, -1)?;
        Ok(&h + &ffn_out)
    }
}
