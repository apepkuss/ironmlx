use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};
use std::time::Instant;

use crate::core::Loader;
use crate::nn::{LayerCache, Linear, RmsNorm};
use crate::Result;

use super::attention::{Gemma4Attention, SharedKv};
use super::config::{Gemma4LayerKind, Gemma4TextConfig};
use super::mlp::Gemma4GeGluMlp;
use super::ops::gelu_approx_mul_on;
use super::profile;
use super::rope::RopeOffsets;

pub struct Gemma4DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Gemma4Attention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: Gemma4GeGluMlp,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    layer_scalar: Array,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
}

impl Gemma4DecoderLayer {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let layer_kind = cfg.layer_kind(layer_idx);
        let mlp_intermediate =
            if cfg.use_double_wide_mlp && layer_idx >= cfg.first_kv_shared_layer_idx() {
                cfg.intermediate_size * 2
            } else {
                cfg.intermediate_size
            };
        let has_per_layer_input = cfg.hidden_size_per_layer_input > 0;
        Ok(Self {
            input_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.input_layernorm"),
                cfg.rms_norm_eps,
            )?,
            self_attn: Gemma4Attention::from_loader(
                loader,
                &format!("{prefix}.self_attn"),
                cfg,
                layer_idx,
            )?,
            post_attention_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
            pre_feedforward_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.pre_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            mlp: Gemma4GeGluMlp::from_loader(
                loader,
                &format!("{prefix}.mlp"),
                mlp_intermediate,
                layer_idx,
                layer_kind,
            )?,
            post_feedforward_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            per_layer_input_gate: if has_per_layer_input {
                Some(Linear::from_loader(
                    loader,
                    &format!("{prefix}.per_layer_input_gate"),
                )?)
            } else {
                None
            },
            per_layer_projection: if has_per_layer_input {
                Some(Linear::from_loader(
                    loader,
                    &format!("{prefix}.per_layer_projection"),
                )?)
            } else {
                None
            },
            post_per_layer_input_norm: if has_per_layer_input {
                Some(RmsNorm::from_loader(
                    loader,
                    &format!("{prefix}.post_per_layer_input_norm"),
                    cfg.rms_norm_eps,
                )?)
            } else {
                None
            },
            layer_scalar: loader.tensor(&format!("{prefix}.layer_scalar"))?.clone(),
            layer_idx,
            layer_kind,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        per_layer_input: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        offsets: &RopeOffsets,
        shared_kv: Option<&SharedKv>,
        cache: Option<&mut LayerCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, SharedKv)> {
        let target = target.into();
        let profile = profile::vl_layer_enabled();
        let total_t0 = Instant::now();
        let residual = x.clone();
        let t0 = Instant::now();
        let h = self.input_layernorm.forward_on(x, target)?;
        profile::eval_layer(
            "gemma4_text_layer_input_norm",
            self.layer_idx,
            self.layer_kind,
            &[&h],
            t0,
            profile,
        )?;
        let kv_cache = match cache {
            Some(LayerCache::Full(kv)) => Some(kv),
            Some(LayerCache::Linear(_)) => {
                return Err(anyhow!(
                    "Gemma4DecoderLayer: expected Full KV cache, got Linear"
                ));
            }
            None => None,
        };
        let t0 = Instant::now();
        let (attn, kv) = self.self_attn.forward_on(
            &h,
            mask,
            per_row_lens,
            offsets,
            shared_kv,
            kv_cache,
            target,
        )?;
        profile::eval_layer(
            "gemma4_text_layer_attention",
            self.layer_idx,
            self.layer_kind,
            &[&attn],
            t0,
            profile,
        )?;
        let t0 = Instant::now();
        let attn = self.post_attention_layernorm.forward_on(&attn, target)?;
        let mut h = &residual + &attn;
        profile::eval_layer(
            "gemma4_text_layer_attention_residual",
            self.layer_idx,
            self.layer_kind,
            &[&h],
            t0,
            profile,
        )?;

        let residual = h.clone();
        let t0 = Instant::now();
        let ffn = self.pre_feedforward_layernorm.forward_on(&h, target)?;
        let ffn = self.mlp.forward_on(&ffn, target)?;
        let ffn = self.post_feedforward_layernorm.forward_on(&ffn, target)?;
        h = &residual + &ffn;
        profile::eval_layer(
            "gemma4_text_layer_ffn",
            self.layer_idx,
            self.layer_kind,
            &[&h],
            t0,
            profile,
        )?;

        if let Some(side) = per_layer_input {
            let t0 = Instant::now();
            let gate = self
                .per_layer_input_gate
                .as_ref()
                .ok_or_else(|| anyhow!("Gemma4DecoderLayer: per-layer input gate missing"))?
                .forward_on(&h, target)?;
            let gate = gelu_approx_mul_on(&gate, side, target)?;
            let gate = self
                .per_layer_projection
                .as_ref()
                .ok_or_else(|| anyhow!("Gemma4DecoderLayer: per-layer projection missing"))?
                .forward_on(&gate, target)?;
            let gate = self
                .post_per_layer_input_norm
                .as_ref()
                .ok_or_else(|| anyhow!("Gemma4DecoderLayer: post per-layer input norm missing"))?
                .forward_on(&gate, target)?;
            h = &h + &gate;
            profile::eval_layer(
                "gemma4_text_layer_side_input",
                self.layer_idx,
                self.layer_kind,
                &[&h],
                t0,
                profile,
            )?;
        }

        h = &h * &self.layer_scalar;
        profile::eval_layer(
            "gemma4_text_layer_decoder_total",
            self.layer_idx,
            self.layer_kind,
            &[&h],
            total_t0,
            profile,
        )?;
        Ok((h, kv))
    }
}
