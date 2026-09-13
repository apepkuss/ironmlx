use anyhow::{anyhow, Context};
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::config::{DiffusionGemmaLayerKind, DiffusionGemmaTextConfig};
use super::rope::{DiffusionGemmaRope, RopeOffsets};

#[derive(Clone)]
pub struct LayerKv {
    pub keys: Array,
    pub values: Array,
}

pub struct DiffusionGemmaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: DiffusionGemmaRope,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    layer_kind: DiffusionGemmaLayerKind,
    sliding_window: i32,
}

impl DiffusionGemmaAttention {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &DiffusionGemmaTextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let layer_kind = cfg.layer_kind(layer_idx);
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let n_kv_heads = cfg.kv_heads_for_layer(layer_idx);
        let v_proj = if layer_kind == DiffusionGemmaLayerKind::Sliding {
            Some(
                Linear::from_loader(loader, &format!("{prefix}.v_proj"))
                    .with_context(|| format!("loading DiffusionGemma v_proj `{prefix}`"))?,
            )
        } else {
            None
        };
        Ok(Self {
            q_proj: Linear::from_loader(loader, &format!("{prefix}.q_proj"))
                .with_context(|| format!("loading DiffusionGemma q_proj `{prefix}`"))?,
            k_proj: Linear::from_loader(loader, &format!("{prefix}.k_proj"))
                .with_context(|| format!("loading DiffusionGemma k_proj `{prefix}`"))?,
            v_proj,
            o_proj: Linear::from_loader(loader, &format!("{prefix}.o_proj"))
                .with_context(|| format!("loading DiffusionGemma o_proj `{prefix}`"))?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?,
            rope: DiffusionGemmaRope::new(head_dim, false, cfg.rope_params_for(layer_kind))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            layer_kind,
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn layer_kind(&self) -> DiffusionGemmaLayerKind {
        self.layer_kind
    }

    pub fn forward_encoder_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
        prior: Option<&LayerKv>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, LayerKv)> {
        let target = target.into();
        let (q, k_new, v_new) = self.project_qkv_on(x, offset, target)?;
        let (keys, values) = match prior {
            Some(kv) => (
                mlx::ops::shape::concatenate_on(&[&kv.keys, &k_new], 2, target)?,
                mlx::ops::shape::concatenate_on(&[&kv.values, &v_new], 2, target)?,
            ),
            None => (k_new, v_new),
        };
        let out = self.attend_on(&q, &keys, &values, mask, target)?;
        Ok((out, LayerKv { keys, values }))
    }

    pub fn forward_decoder_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        encoder_kv: Option<&LayerKv>,
        encoder_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let (q, k_canvas, v_canvas) = self.project_qkv_on(x, encoder_len, target)?;
        let (keys, values) = match encoder_kv {
            Some(kv) if self.layer_kind == DiffusionGemmaLayerKind::Sliding => {
                let window = (self.sliding_window - 1).max(0);
                let (enc_k, enc_v) = if window > 0 && kv.keys.shape_at(2) > window {
                    let s_k = kv.keys.shape();
                    let ks = s_k.as_slice();
                    let start = ks[2] - window;
                    (
                        mlx::ops::indexing::slice_strided_on(
                            &kv.keys,
                            [0_i32, 0, start, 0],
                            [ks[0], ks[1], ks[2], ks[3]],
                            [1_i32, 1, 1, 1],
                            target,
                        )?,
                        mlx::ops::indexing::slice_strided_on(
                            &kv.values,
                            [0_i32, 0, start, 0],
                            [ks[0], ks[1], ks[2], kv.values.shape_at(3)],
                            [1_i32, 1, 1, 1],
                            target,
                        )?,
                    )
                } else {
                    (kv.keys.clone(), kv.values.clone())
                };
                (
                    mlx::ops::shape::concatenate_on(&[&enc_k, &k_canvas], 2, target)?,
                    mlx::ops::shape::concatenate_on(&[&enc_v, &v_canvas], 2, target)?,
                )
            }
            Some(kv) => (
                mlx::ops::shape::concatenate_on(&[&kv.keys, &k_canvas], 2, target)?,
                mlx::ops::shape::concatenate_on(&[&kv.values, &v_canvas], 2, target)?,
            ),
            None => (k_canvas, v_canvas),
        };
        self.attend_on(&q, &keys, &values, mask, target)
    }

    fn project_qkv_on(
        &self,
        x: &Array,
        offset: i32,
        target: StreamOrDevice,
    ) -> Result<(Array, Array, Array)> {
        let shape = x.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "DiffusionGemmaAttention: x must be [B,S,H], got {:?}",
                dims
            ));
        }
        let (batch, seq) = (dims[0], dims[1]);
        let offsets = RopeOffsets::from_values(vec![offset; batch as usize])?;

        let q = self
            .q_proj
            .forward_on(x, target)?
            .reshape_on((batch, seq, self.n_heads, self.head_dim), target)?;
        let q = self.q_norm.forward_on(&q, target)?;
        let q = q.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let q = self.rope.apply_on(&q, &offsets, target)?;

        let raw_k = self
            .k_proj
            .forward_on(x, target)?
            .reshape_on((batch, seq, self.n_kv_heads, self.head_dim), target)?;
        let raw_v = match &self.v_proj {
            Some(v_proj) => v_proj
                .forward_on(x, target)?
                .reshape_on((batch, seq, self.n_kv_heads, self.head_dim), target)?,
            None => raw_k.clone(),
        };
        let k = self.k_norm.forward_on(&raw_k, target)?;
        let k = k.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let k = self.rope.apply_on(&k, &offsets, target)?;
        let v = rms_norm_no_scale_on(&raw_v, self.rms_norm_eps, target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        Ok((q, k, v))
    }

    fn attend_on(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        mask: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let out = mlx::fast::scaled_dot_product_attention_on(q, k, v, 1.0, "", mask, None, target)?;
        let shape = q.shape();
        let dims = shape.as_slice();
        let (batch, seq) = (dims[0], dims[2]);
        let out = out
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.n_heads * self.head_dim), target)?;
        self.o_proj.forward_on(&out, target)
    }
}

fn rms_norm_no_scale_on(x: &Array, eps: f32, target: impl Into<StreamOrDevice>) -> Result<Array> {
    Ok(mlx::fast::rms_norm_on(x, None, eps, target)?)
}
