use anyhow::{anyhow, Context};
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

struct ProjectedQkv {
    q: Array,
    raw_kv: Option<(Array, Array)>,
}

enum Gemma4AttentionProjection {
    FusedQkv {
        qkv: Linear,
    },
    SeparateQkv {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Option<Linear>,
    },
    QOnly {
        q_proj: Linear,
    },
}

pub struct Gemma4Attention {
    projection: Gemma4AttentionProjection,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Gemma4Rope,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
}

impl Gemma4AttentionProjection {
    fn from_loader(loader: &Loader, prefix: &str, owns_kv: bool, use_k_eq_v: bool) -> Result<Self> {
        if owns_kv && !use_k_eq_v {
            return Ok(Self::FusedQkv {
                qkv: load_fused_qkv(loader, prefix)
                    .with_context(|| format!("loading fused Gemma4 q/k/v `{prefix}`"))?,
            });
        }

        let q_proj = Linear::from_loader(loader, &format!("{prefix}.q_proj"))
            .with_context(|| format!("loading Gemma4 q_proj `{prefix}`"))?;
        if !owns_kv {
            return Ok(Self::QOnly { q_proj });
        }

        let k_proj = Linear::from_loader(loader, &format!("{prefix}.k_proj"))
            .with_context(|| format!("loading Gemma4 k_proj `{prefix}`"))?;
        let v_proj = if use_k_eq_v {
            None
        } else {
            Some(
                Linear::from_loader(loader, &format!("{prefix}.v_proj"))
                    .with_context(|| format!("loading Gemma4 v_proj `{prefix}`"))?,
            )
        };
        Ok(Self::SeparateQkv {
            q_proj,
            k_proj,
            v_proj,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_on(
        &self,
        x: &Array,
        need_kv: bool,
        batch: i32,
        seq: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        target: StreamOrDevice,
    ) -> Result<ProjectedQkv> {
        match self {
            Self::FusedQkv { qkv } => {
                if !need_kv {
                    return Err(anyhow!(
                        "Gemma4AttentionProjection: fused q/k/v projection cannot serve q-only shared-KV layer"
                    ));
                }
                let projected = qkv.forward_on(x, target)?;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv_heads * head_dim;
                let axis = projected.ndim() as i32 - 1;
                let parts = mlx::ops::shape::split_at_on(
                    &projected,
                    &[q_dim, q_dim + kv_dim],
                    axis,
                    target,
                )?;
                if parts.len() != 3 {
                    return Err(anyhow!(
                        "Gemma4AttentionProjection: fused q/k/v split expected 3 parts, got {}",
                        parts.len()
                    ));
                }
                let q = parts[0].reshape_on((batch, seq, n_heads, head_dim), target)?;
                let raw_k = parts[1].reshape_on((batch, seq, n_kv_heads, head_dim), target)?;
                let raw_v = parts[2].reshape_on((batch, seq, n_kv_heads, head_dim), target)?;
                Ok(ProjectedQkv {
                    q,
                    raw_kv: Some((raw_k, raw_v)),
                })
            }
            Self::SeparateQkv {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let q = q_proj
                    .forward_on(x, target)?
                    .reshape_on((batch, seq, n_heads, head_dim), target)?;
                let raw_kv = if need_kv {
                    let raw_k = k_proj
                        .forward_on(x, target)?
                        .reshape_on((batch, seq, n_kv_heads, head_dim), target)?;
                    let raw_v = match v_proj {
                        Some(v_proj) => v_proj
                            .forward_on(x, target)?
                            .reshape_on((batch, seq, n_kv_heads, head_dim), target)?,
                        None => raw_k.clone(),
                    };
                    Some((raw_k, raw_v))
                } else {
                    None
                };
                Ok(ProjectedQkv { q, raw_kv })
            }
            Self::QOnly { q_proj } => {
                if need_kv {
                    return Err(anyhow!(
                        "Gemma4AttentionProjection: q-only shared-KV projection cannot produce K/V"
                    ));
                }
                let q = q_proj
                    .forward_on(x, target)?
                    .reshape_on((batch, seq, n_heads, head_dim), target)?;
                Ok(ProjectedQkv { q, raw_kv: None })
            }
        }
    }
}

fn load_fused_qkv(loader: &Loader, prefix: &str) -> Result<Linear> {
    let q = format!("{prefix}.q_proj");
    let k = format!("{prefix}.k_proj");
    let v = format!("{prefix}.v_proj");

    let q_w = loader.tensor(&format!("{q}.weight"))?.clone();
    let k_w = loader.tensor(&format!("{k}.weight"))?.clone();
    let v_w = loader.tensor(&format!("{v}.weight"))?.clone();
    let weight = mlx::ops::shape::concatenate(&[&q_w, &k_w, &v_w], 0)?;

    let q_bias = loader.tensor_opt(&format!("{q}.bias")).cloned();
    let k_bias = loader.tensor_opt(&format!("{k}.bias")).cloned();
    let v_bias = loader.tensor_opt(&format!("{v}.bias")).cloned();
    let bias = match (q_bias, k_bias, v_bias) {
        (Some(qb), Some(kb), Some(vb)) => Some(mlx::ops::shape::concatenate(&[&qb, &kb, &vb], 0)?),
        (None, None, None) => None,
        _ => {
            return Err(anyhow!(
                "Gemma4Attention: q/k/v bias presence mismatch at `{prefix}`"
            ));
        }
    };

    let q_scales_key = format!("{q}.scales");
    let k_scales_key = format!("{k}.scales");
    let v_scales_key = format!("{v}.scales");
    let any_quant = loader.contains(&q_scales_key)
        || loader.contains(&k_scales_key)
        || loader.contains(&v_scales_key);
    if any_quant {
        if !loader.contains(&q_scales_key)
            || !loader.contains(&k_scales_key)
            || !loader.contains(&v_scales_key)
        {
            return Err(anyhow!(
                "Gemma4Attention: q/k/v quantization mismatch at `{prefix}`"
            ));
        }
        let qmeta = loader.quant_meta().ok_or_else(|| {
            anyhow!("Gemma4Attention: quantized q/k/v present but Loader has no quant meta")
        })?;
        let q_scales = loader.tensor(&q_scales_key)?.clone();
        let k_scales = loader.tensor(&k_scales_key)?.clone();
        let v_scales = loader.tensor(&v_scales_key)?.clone();
        let scales = mlx::ops::shape::concatenate(&[&q_scales, &k_scales, &v_scales], 0)?;

        let q_biases = loader.tensor_opt(&format!("{q}.biases")).cloned();
        let k_biases = loader.tensor_opt(&format!("{k}.biases")).cloned();
        let v_biases = loader.tensor_opt(&format!("{v}.biases")).cloned();
        let biases = match (q_biases, k_biases, v_biases) {
            (Some(qb), Some(kb), Some(vb)) => {
                Some(mlx::ops::shape::concatenate(&[&qb, &kb, &vb], 0)?)
            }
            (None, None, None) => None,
            _ => {
                return Err(anyhow!(
                    "Gemma4Attention: q/k/v quantized biases mismatch at `{prefix}`"
                ));
            }
        };
        {
            let mut to_eval: Vec<&Array> = vec![&weight, &scales];
            if let Some(b) = &biases {
                to_eval.push(b);
            }
            if let Some(b) = &bias {
                to_eval.push(b);
            }
            mlx::transforms::eval(&to_eval)
                .map_err(|e| anyhow!("{prefix}: eager eval of fused q/k/v tensors failed: {e}"))?;
        }
        Ok(Linear::new_quant(
            weight,
            scales,
            biases,
            bias,
            qmeta.group_size,
            qmeta.bits,
        ))
    } else {
        {
            let mut to_eval: Vec<&Array> = vec![&weight];
            if let Some(b) = &bias {
                to_eval.push(b);
            }
            mlx::transforms::eval(&to_eval)
                .map_err(|e| anyhow!("{prefix}: eager eval of fused q/k/v tensors failed: {e}"))?;
        }
        Ok(Linear::new_fp(weight, bias))
    }
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
        let owns_kv = cfg.previous_kv_layer(layer_idx) == layer_idx;
        let rope = Gemma4Rope::new(head_dim, cfg.rope_traditional, cfg.rope_params_for(kind))?;
        Ok(Self {
            projection: Gemma4AttentionProjection::from_loader(
                loader, prefix, owns_kv, use_k_eq_v,
            )?,
            o_proj: Linear::from_loader(loader, &format!("{prefix}.o_proj"))
                .with_context(|| format!("loading Gemma4 o_proj `{prefix}`"))?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?,
            rope,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
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
        let projected = self.projection.forward_on(
            x,
            shared_kv.is_none(),
            batch,
            seq,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            target,
        )?;
        let q = match self.decode_default_rope_on(
            &projected.q,
            &self.q_norm,
            self.n_heads,
            offsets,
            target,
        )? {
            Some(q) => q,
            None => {
                let q = self.q_norm.forward_on(&projected.q, target)?;
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
        let mut paged_decode_out: Option<Array> = None;
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
                let (raw_k, raw_v) = projected.raw_kv.ok_or_else(|| {
                    anyhow!(
                        "Gemma4Attention: missing raw K/V for non-shared layer {}",
                        self.layer_idx
                    )
                })?;
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
                        let maybe_paged_out = if c.paged().is_some() {
                            c.try_update_and_attend_decode_on(&q, &k, &v, lens, 1.0, mask, target)?
                        } else {
                            None
                        };
                        let (keys, values) = if maybe_paged_out.is_some() {
                            c.materialize_current_paged_prefix_on(target)?
                        } else {
                            c.update_and_fetch_for_attention_on(&k, &v, lens, target)?
                        };
                        paged_decode_out = maybe_paged_out;
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
        let out = if let Some(out) = paged_decode_out {
            out
        } else {
            match mask {
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
            }
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
