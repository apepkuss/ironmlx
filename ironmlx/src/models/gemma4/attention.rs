use anyhow::Context;
use mlx::{Array, StreamOrDevice};
use std::time::Instant;

use crate::core::cache::KVCache;
use crate::core::Loader;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::config::{Gemma4LayerKind, Gemma4TextConfig};
use super::ops::{rms_norm_default_rope_decode_on, rms_norm_no_scale_on, RmsNormDefaultRopeDecode};
use super::profile;
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
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
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
            layer_idx,
            layer_kind: kind,
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
        let profile = profile::vl_layer_enabled();
        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        let (batch, seq) = (dims[0], dims[1]);
        let single_row_decode =
            cache.as_ref().is_some() && batch == 1 && seq == 1 && per_row_lens.is_none();

        let t0 = Instant::now();
        let q = self
            .q_proj
            .forward_on(x, target)?
            .reshape_on((batch, seq, self.n_heads, self.head_dim), target)?;
        let q =
            match self.decode_default_rope_on(&q, &self.q_norm, self.n_heads, offsets, target)? {
                Some(q) => q,
                None => {
                    let q = self.q_norm.forward_on(&q, target)?;
                    let q = q.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
                    self.rope.apply_on(&q, offsets, target)?
                }
            };
        profile::eval_layer(
            "gemma4_attn_q_path",
            self.layer_idx,
            self.layer_kind,
            &[&q],
            t0,
            profile,
        )?;

        let t0 = Instant::now();
        let kv = match shared_kv {
            Some(kv) => {
                profile::log_layer(
                    "gemma4_attn_kv_reuse",
                    self.layer_idx,
                    self.layer_kind,
                    t0,
                    profile,
                );
                kv.clone()
            }
            None => {
                let t0 = Instant::now();
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
                profile::eval_layer(
                    "gemma4_attn_kv_project",
                    self.layer_idx,
                    self.layer_kind,
                    &[&raw_k, &raw_v],
                    t0,
                    profile,
                )?;

                let t0 = Instant::now();
                let k = match self.decode_default_rope_on(
                    &raw_k,
                    &self.k_norm,
                    self.n_kv_heads,
                    offsets,
                    target,
                )? {
                    Some(k) => k,
                    None => {
                        let k = self.k_norm.forward_on(&raw_k, target)?;
                        let k = k.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
                        self.rope.apply_on(&k, offsets, target)?
                    }
                };
                let v = rms_norm_no_scale_on(&raw_v, self.rms_norm_eps, target)?
                    .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
                profile::eval_layer(
                    "gemma4_attn_kv_norm_rope",
                    self.layer_idx,
                    self.layer_kind,
                    &[&k, &v],
                    t0,
                    profile,
                )?;

                let (keys, values) = match cache {
                    Some(c) => {
                        let t0 = Instant::now();
                        let lens_owned;
                        let lens = match per_row_lens {
                            Some(l) => l,
                            None => {
                                lens_owned = vec![seq; batch as usize];
                                &lens_owned
                            }
                        };
                        let (keys, values) = c.update_and_fetch_on(&k, &v, lens, target)?;
                        profile::eval_layer(
                            "gemma4_attn_cache_update_fetch",
                            self.layer_idx,
                            self.layer_kind,
                            &[&keys, &values],
                            t0,
                            profile,
                        )?;
                        (keys, values)
                    }
                    None => (k, v),
                };
                SharedKv { keys, values }
            }
        };
        profile::eval_layer(
            "gemma4_attn_kv_path",
            self.layer_idx,
            self.layer_kind,
            &[&kv.keys, &kv.values],
            t0,
            profile,
        )?;

        let t0 = Instant::now();
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
            None if single_row_decode => mlx::fast::scaled_dot_product_attention_on(
                &q, &kv.keys, &kv.values, 1.0, "", None, None, target,
            )?,
            None => mlx::fast::scaled_dot_product_attention_on(
                &q, &kv.keys, &kv.values, 1.0, "causal", None, None, target,
            )?,
        };
        profile::eval_layer(
            "gemma4_attn_sdpa",
            self.layer_idx,
            self.layer_kind,
            &[&out],
            t0,
            profile,
        )?;

        let t0 = Instant::now();
        let out = out
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
            .reshape_on((batch, seq, self.n_heads * self.head_dim), target)?;
        let out = self.o_proj.forward_on(&out, target)?;
        profile::eval_layer(
            "gemma4_attn_o_proj",
            self.layer_idx,
            self.layer_kind,
            &[&out],
            t0,
            profile,
        )?;
        Ok((out, kv))
    }

    fn decode_default_rope_on(
        &self,
        x: &Array,
        norm: &RmsNorm,
        heads: i32,
        offsets: &RopeOffsets,
        target: StreamOrDevice,
    ) -> Result<Option<Array>> {
        let Some((dims, base, traditional)) = self.rope.default_params() else {
            return Ok(None);
        };
        if dims != self.head_dim || offsets.scalar().is_none() {
            return Ok(None);
        }
        let params = RmsNormDefaultRopeDecode {
            weight: norm.weight(),
            offsets: offsets.values_array(),
            eps: norm.eps(),
            base,
            traditional,
            heads,
            head_dim: self.head_dim,
        };
        rms_norm_default_rope_decode_on(x, params, target)
    }
}
