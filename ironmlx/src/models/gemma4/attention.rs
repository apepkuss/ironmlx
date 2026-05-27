use anyhow::Context;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::config::{Gemma4LayerKind, Gemma4TextConfig};
use super::ops::rms_norm_no_scale_on;
use super::rope::{Gemma4Rope, RopeOffsets};

#[derive(Clone)]
pub struct SharedKv {
    pub keys: Array,
    pub values: Array,
}

pub struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Gemma4Rope,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    use_k_eq_v: bool,
}

impl Gemma4Attention {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let kind = cfg.layer_kind(layer_idx);
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let n_kv_heads = cfg.kv_heads_for_layer(layer_idx);
        let use_k_eq_v = kind == Gemma4LayerKind::Full && cfg.attention_k_eq_v;
        let rope = Gemma4Rope::new(head_dim, cfg.rope_traditional, cfg.rope_params_for(kind))?;
        Ok(Self {
            q_proj: Linear::from_loader(loader, &format!("{prefix}.q_proj"))
                .with_context(|| format!("loading Gemma4 q_proj `{prefix}`"))?,
            k_proj: Linear::from_loader(loader, &format!("{prefix}.k_proj"))
                .with_context(|| format!("loading Gemma4 k_proj `{prefix}`"))?,
            v_proj: if use_k_eq_v {
                None
            } else {
                Some(
                    Linear::from_loader(loader, &format!("{prefix}.v_proj"))
                        .with_context(|| format!("loading Gemma4 v_proj `{prefix}`"))?,
                )
            },
            o_proj: Linear::from_loader(loader, &format!("{prefix}.o_proj"))
                .with_context(|| format!("loading Gemma4 o_proj `{prefix}`"))?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?,
            rope,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            use_k_eq_v,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        offsets: &RopeOffsets,
        shared_kv: Option<&SharedKv>,
        cache: Option<&mut KVCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, SharedKv)> {
        let target = target.into();
        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        let (batch, seq) = (dims[0], dims[1]);

        let q = self
            .q_proj
            .forward_on(x, target)?
            .reshape_on((batch, seq, self.n_heads, self.head_dim), target)?;
        let q = self.q_norm.forward_on(&q, target)?;
        let q = q.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let q = self.rope.apply_on(&q, offsets, target)?;

        let kv = match shared_kv {
            Some(kv) => kv.clone(),
            None => {
                let raw_k = self
                    .k_proj
                    .forward_on(x, target)?
                    .reshape_on((batch, seq, self.n_kv_heads, self.head_dim), target)?;
                let raw_v = if self.use_k_eq_v {
                    raw_k.clone()
                } else {
                    self.v_proj
                        .as_ref()
                        .expect("v_proj exists when use_k_eq_v=false")
                        .forward_on(x, target)?
                        .reshape_on((batch, seq, self.n_kv_heads, self.head_dim), target)?
                };

                let k = self.k_norm.forward_on(&raw_k, target)?;
                let k = k.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
                let k = self.rope.apply_on(&k, offsets, target)?;
                let v = rms_norm_no_scale_on(&raw_v, self.rms_norm_eps, target)?
                    .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;

                let (keys, values) = match cache {
                    Some(c) => {
                        let lens_owned;
                        let lens = match per_row_lens {
                            Some(l) => l,
                            None => {
                                lens_owned = vec![seq; batch as usize];
                                &lens_owned
                            }
                        };
                        c.update_and_fetch_on(&k, &v, lens, target)?
                    }
                    None => (k, v),
                };
                SharedKv { keys, values }
            }
        };

        let out = match mask {
            Some(m) => mlx::fast::scaled_dot_product_attention_on(
                &q,
                &kv.keys,
                &kv.values,
                1.0,
                "",
                Some(m),
                None,
                target,
            )?,
            None => mlx::fast::scaled_dot_product_attention_on(
                &q, &kv.keys, &kv.values, 1.0, "causal", None, None, target,
            )?,
        };

        let out = out
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.n_heads * self.head_dim), target)?;
        Ok((self.o_proj.forward_on(&out, target)?, kv))
    }
}
