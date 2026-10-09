use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Linear, Mlp, RmsNorm};
use crate::Result;

use super::attention::{DFlash2Attention, DFlash2KvCache};
use super::config::DFlash2Config;
use super::conv::DFlash2GroupedConv;
use super::load_linear;

pub(super) struct DFlash2DecoderLayer {
    input_layernorm: RmsNorm,
    attention: DFlash2Attention,
    attention_conv: DFlash2GroupedConv,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    mlp_conv: DFlash2GroupedConv,
}

impl DFlash2DecoderLayer {
    pub(super) fn projections(&self) -> Vec<&Linear> {
        let mut out = self.attention.projections();
        out.push(self.attention_conv.projection());
        out.extend(self.mlp.activation_projections());
        out.push(self.mlp_conv.projection());
        out
    }

    pub(super) fn from_loader(
        loader: &Loader,
        index: i32,
        cfg: &DFlash2Config,
        draft_bits: Option<i32>,
    ) -> Result<Self> {
        let prefix = format!("layers.{index}");
        Ok(Self {
            input_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.input_layernorm"),
                cfg.rms_norm_eps,
            )?,
            attention: DFlash2Attention::from_loader(
                loader,
                &format!("{prefix}.self_attn"),
                cfg,
                draft_bits,
            )?,
            attention_conv: DFlash2GroupedConv::from_loader(
                loader,
                &format!("{prefix}.attention_conv"),
                cfg.hidden_size,
                cfg.dflash_config.conv_kernel_size,
                cfg.dflash_config.conv_group_size,
                draft_bits,
            )?,
            post_attention_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
            mlp: Mlp::from_components_dflash2(
                load_linear(loader, &format!("{prefix}.mlp.gate_proj"), draft_bits)?,
                load_linear(loader, &format!("{prefix}.mlp.up_proj"), draft_bits)?,
                load_linear(loader, &format!("{prefix}.mlp.down_proj"), draft_bits)?,
            )?,
            mlp_conv: DFlash2GroupedConv::from_loader(
                loader,
                &format!("{prefix}.mlp_conv"),
                cfg.hidden_size,
                cfg.dflash_config.conv_kernel_size,
                cfg.dflash_config.conv_group_size,
                draft_bits,
            )?,
        })
    }

    pub(super) fn forward_on(
        &self,
        hidden: &Array,
        context: &Array,
        mask: &Array,
        cache: &mut DFlash2KvCache,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let residual = hidden;
        let normed = self.input_layernorm.forward_on(hidden, target)?;
        let (normed, kernel) = self.attention_conv.prepare_on(&normed, target)?;
        let attention = self
            .attention
            .forward_on(&normed, context, mask, cache, target)?;
        let hidden = self
            .attention_conv
            .finish_with_residual_on(&attention, &kernel, residual, target)?;

        let residual = &hidden;
        let normed = self.post_attention_layernorm.forward_on(&hidden, target)?;
        let (normed, kernel) = self.mlp_conv.prepare_on(&normed, target)?;
        let mlp = self.mlp.forward_on(&normed, target)?;
        self.mlp_conv
            .finish_with_residual_on(&mlp, &kernel, residual, target)
    }

    /// [`Self::forward_on`] for rows at different draft-cache positions; only
    /// the attention core is per row.
    pub(super) fn forward_ragged_on(
        &self,
        hidden: &Array,
        contexts: &[Array],
        masks: &[Array],
        caches: &mut [&mut DFlash2KvCache],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let residual = hidden;
        let normed = self.input_layernorm.forward_on(hidden, target)?;
        let (normed, kernel) = self.attention_conv.prepare_on(&normed, target)?;
        let attention = self
            .attention
            .forward_ragged_on(&normed, contexts, masks, caches, target)?;
        let hidden = self
            .attention_conv
            .finish_with_residual_on(&attention, &kernel, residual, target)?;

        let residual = &hidden;
        let normed = self.post_attention_layernorm.forward_on(&hidden, target)?;
        let (normed, kernel) = self.mlp_conv.prepare_on(&normed, target)?;
        let mlp = self.mlp.forward_on(&normed, target)?;
        self.mlp_conv
            .finish_with_residual_on(&mlp, &kernel, residual, target)
    }
}
