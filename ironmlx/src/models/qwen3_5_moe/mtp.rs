//! Qwen3.5 MoE MTP head.
//!
//! Dense Qwen MTP can reuse `nn::Mtp`; the A3B checkpoint cannot, because its
//! MTP decoder layer contains `SparseMoeBlock` weights under `layers.N.mlp.*`.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::MtpCache;
use crate::core::Loader;
use crate::nn::{AttnKind, Linear, Mrope, RmsNorm};
use crate::Result;

use super::decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};

#[derive(Debug, Clone, Copy)]
pub struct Qwen35MoeMtpConfig {
    pub hidden_size: i32,
    pub num_mtp_layers: i32,
    pub layer: DecoderLayerMoeConfig,
}

pub struct Qwen35MoeMtp {
    pre_fc_norm_hidden: RmsNorm,
    pre_fc_norm_embedding: RmsNorm,
    fc: Linear,
    layers: Vec<DecoderLayerMoe>,
    norm: RmsNorm,
    cfg: Qwen35MoeMtpConfig,
}

impl Qwen35MoeMtp {
    #[doc(hidden)]
    pub fn from_components(
        pre_fc_norm_hidden: RmsNorm,
        pre_fc_norm_embedding: RmsNorm,
        fc: Linear,
        layers: Vec<DecoderLayerMoe>,
        norm: RmsNorm,
        cfg: Qwen35MoeMtpConfig,
    ) -> Self {
        Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
            cfg,
        }
    }

    pub fn config(&self) -> &Qwen35MoeMtpConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
    ) -> Result<Array> {
        self.forward_on(
            hidden_states,
            next_token_embeds,
            mrope,
            cos,
            sin,
            mask,
            mtp_cache,
            (),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        mut mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        self.validate_inputs(hidden_states, next_token_embeds, mtp_cache.as_deref())?;

        let h = self.pre_fc_norm_hidden.forward_on(hidden_states, target)?;
        let e = self
            .pre_fc_norm_embedding
            .forward_on(next_token_embeds, target)?;
        let concat = mlx::ops::shape::concatenate_on(&[&e, &h], -1, target)?;
        let mut x = self.fc.forward_on(&concat, target)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = mtp_cache.as_deref_mut().map(|mc| mc.layer_mut(i));
            x = layer.forward_on_full_kv(&x, mrope, cos, sin, mask, layer_cache, target)?;
        }

        self.norm.forward_on(&x, target)
    }

    fn validate_inputs(
        &self,
        hidden_states: &Array,
        next_token_embeds: &Array,
        mtp_cache: Option<&MtpCache>,
    ) -> Result<()> {
        if hidden_states.ndim() != 3 || next_token_embeds.ndim() != 3 {
            return Err(anyhow!(
                "Qwen35MoeMtp::forward_on: hidden_states and next_token_embeds must be rank-3, \
                 got ranks {}/{}",
                hidden_states.ndim(),
                next_token_embeds.ndim(),
            ));
        }
        let hs = hidden_states.shape();
        let es = next_token_embeds.shape();
        let hs = hs.as_slice();
        let es = es.as_slice();
        if hs != es {
            return Err(anyhow!(
                "Qwen35MoeMtp::forward_on: hidden_states {:?} and next_token_embeds {:?} \
                 must have identical shape",
                hs,
                es,
            ));
        }
        if hs[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "Qwen35MoeMtp::forward_on: last-axis must equal cfg.hidden_size {}, got {}",
                self.cfg.hidden_size,
                hs[2],
            ));
        }
        if let Some(c) = mtp_cache {
            if c.num_layers() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35MoeMtp::forward_on: mtp_cache.num_layers() = {} but head has {} layers",
                    c.num_layers(),
                    self.layers.len(),
                ));
            }
        }
        Ok(())
    }

    pub fn from_loader(loader: &Loader, prefix: &str, cfg: Qwen35MoeMtpConfig) -> Result<Self> {
        let key = |leaf: &str| -> String {
            if prefix.is_empty() {
                leaf.to_owned()
            } else {
                format!("{prefix}.{leaf}")
            }
        };

        let pre_fc_norm_hidden =
            RmsNorm::from_loader(loader, &key("pre_fc_norm_hidden"), cfg.layer.rms_norm_eps)?;
        let pre_fc_norm_embedding = RmsNorm::from_loader(
            loader,
            &key("pre_fc_norm_embedding"),
            cfg.layer.rms_norm_eps,
        )?;
        let fc = Linear::from_loader(loader, &key("fc"))?;

        let expected_in = (cfg.hidden_size * 2) as usize;
        let expected_out = cfg.hidden_size as usize;
        if fc.in_features() != expected_in || fc.out_features() != expected_out {
            return Err(anyhow!(
                "Qwen35MoeMtp.fc weight shape mismatch under prefix '{}': \
                 expected [in={expected_in}, out={expected_out}], got [in={}, out={}]",
                key("fc"),
                fc.in_features(),
                fc.out_features(),
            ));
        }

        let norm = RmsNorm::from_loader(loader, &key("norm"), cfg.layer.rms_norm_eps)?;

        let mut layers = Vec::with_capacity(cfg.num_mtp_layers as usize);
        for i in 0..cfg.num_mtp_layers {
            layers.push(DecoderLayerMoe::from_loader(
                loader,
                &key(&format!("layers.{i}")),
                cfg.layer,
                AttnKind::Full,
            )?);
        }

        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
            cfg,
        })
    }
}
