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
use super::quant_fusion::{fused_quant_compatibility, FusedQuantCompatibility};
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
    k_norm: Option<RmsNorm>,
    rope: Gemma4Rope,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
    sliding_window: i32,
}

impl Gemma4AttentionProjection {
    fn from_loader(loader: &Loader, prefix: &str, owns_kv: bool, use_k_eq_v: bool) -> Result<Self> {
        if owns_kv && !use_k_eq_v {
            let q = format!("{prefix}.q_proj");
            let k = format!("{prefix}.k_proj");
            let v = format!("{prefix}.v_proj");
            let compat = fused_quant_compatibility(
                loader,
                &[q.as_str(), k.as_str(), v.as_str()],
                "Gemma4Attention q/k/v",
            )?;
            if compat != FusedQuantCompatibility::MixedQuantized {
                return Ok(Self::FusedQkv {
                    qkv: load_fused_qkv(loader, prefix, compat)
                        .with_context(|| format!("loading fused Gemma4 q/k/v `{prefix}`"))?,
                });
            }
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

fn load_fused_qkv(
    loader: &Loader,
    prefix: &str,
    compat: FusedQuantCompatibility,
) -> Result<Linear> {
    if compat == FusedQuantCompatibility::MixedQuantized {
        return Err(anyhow!(
            "Gemma4Attention: mixed q/k/v quantization cannot be fused at `{prefix}`"
        ));
    }

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

    if let FusedQuantCompatibility::Quantized(qmeta) = compat {
        let q_scales_key = format!("{q}.scales");
        let k_scales_key = format!("{k}.scales");
        let v_scales_key = format!("{v}.scales");
        let q_scales = loader.tensor(&q_scales_key)?.clone();
        let k_scales = loader.tensor(&k_scales_key)?.clone();
        let v_scales = loader.tensor(&v_scales_key)?.clone();
        let scales = mlx::ops::shape::concatenate(&[&q_scales, &k_scales, &v_scales], 0)?;

        let q_biases = loader.tensor_opt(&format!("{q}.biases")).cloned();
        let k_biases = loader.tensor_opt(&format!("{k}.biases")).cloned();
        let v_biases = loader.tensor_opt(&format!("{v}.biases")).cloned();
        qmeta.validate_storage(&q, &q_w, &q_scales, q_biases.as_ref())?;
        qmeta.validate_storage(&k, &k_w, &k_scales, k_biases.as_ref())?;
        qmeta.validate_storage(&v, &v_w, &v_scales, v_biases.as_ref())?;
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
        Ok(Linear::new_quant_with_mode(
            weight,
            scales,
            biases,
            bias,
            qmeta.group_size,
            qmeta.bits,
            qmeta.mode,
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
        let owns_kv = cfg.previous_kv_layer(layer_idx) == layer_idx;
        Self::from_loader_with_owns_kv(loader, prefix, cfg, layer_idx, owns_kv)
    }

    pub fn from_loader_kv_shared_only(
        loader: &Loader,
        prefix: &str,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        Self::from_loader_with_owns_kv(loader, prefix, cfg, layer_idx, false)
    }

    fn from_loader_with_owns_kv(
        loader: &Loader,
        prefix: &str,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        owns_kv: bool,
    ) -> Result<Self> {
        let kind = cfg.layer_kind(layer_idx);
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let n_kv_heads = cfg.kv_heads_for_layer(layer_idx);
        let use_k_eq_v = kind == Gemma4LayerKind::Full && cfg.attention_k_eq_v;
        let rope = Gemma4Rope::new(head_dim, cfg.rope_traditional, cfg.rope_params_for(kind))?;
        Ok(Self {
            projection: Gemma4AttentionProjection::from_loader(
                loader, prefix, owns_kv, use_k_eq_v,
            )?,
            o_proj: Linear::from_loader(loader, &format!("{prefix}.o_proj"))
                .with_context(|| format!("loading Gemma4 o_proj `{prefix}`"))?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: if owns_kv {
                Some(RmsNorm::from_loader(
                    loader,
                    &format!("{prefix}.k_norm"),
                    cfg.rms_norm_eps,
                )?)
            } else {
                None
            },
            rope,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            layer_idx,
            layer_kind: kind,
            sliding_window: cfg.sliding_window,
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
        // The scheduler arms this only for K3V4 multi-row speculative verify.
        // MLPs remain batched; Q/K/V and O projections retain B=1 numerics.
        let _stable_qmm = crate::nn::batch_stable_qmm::context_is_armed()
            .then(crate::nn::batch_stable_qmm::linear_scope);
        let profile = profile::vl_layer_enabled();
        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        let (batch, seq) = (dims[0], dims[1]);
        let reuses_shared_kv = shared_kv.is_some();
        let stable_verify_attention = crate::nn::gemma4_verify_attention::is_armed() && seq > 1;
        let position_stable_full = crate::nn::position_stable_qmm::is_armed()
            && seq > 1
            && self.layer_kind == Gemma4LayerKind::Full;
        let segment_stable_verify = stable_verify_attention && batch == 1;
        let batch_stable_verify = stable_verify_attention && batch > 1;
        let query_isolated =
            segment_stable_verify && self.layer_kind == Gemma4LayerKind::Full && !reuses_shared_kv;
        let segment_stable_sliding =
            segment_stable_verify && self.layer_kind == Gemma4LayerKind::Sliding;
        let batch_stable_sliding =
            batch_stable_verify && self.layer_kind == Gemma4LayerKind::Sliding;
        let batch_stable_full = batch_stable_verify && self.layer_kind == Gemma4LayerKind::Full;
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
        let mut cached_attention_out: Option<Array> = None;
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
                let k_norm = self.k_norm.as_ref().ok_or_else(|| {
                    anyhow!(
                        "Gemma4Attention: k_norm missing for K/V-owning layer {}",
                        self.layer_idx
                    )
                })?;
                let k = match self.decode_default_rope_on(
                    &raw_k,
                    k_norm,
                    self.n_kv_heads,
                    offsets,
                    target,
                )? {
                    Some(k) => k,
                    None => {
                        let k = k_norm.forward_on(&raw_k, target)?;
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
                        let maybe_cached_out = if c.paged().is_some() {
                            c.try_update_and_attend_decode_on(&q, &k, &v, lens, 1.0, mask, target)?
                        } else if stable_verify_attention
                            && !query_isolated
                            && !segment_stable_sliding
                            && !batch_stable_verify
                        {
                            c.try_update_and_attend_multirow_on(
                                &q, &k, &v, lens, 1.0, mask, target,
                            )?
                        } else {
                            None
                        };
                        let (keys, values) = if maybe_cached_out.is_some() {
                            if c.paged().is_some() {
                                c.materialize_current_paged_prefix_on(target)?
                            } else {
                                let len = c.offsets().iter().copied().max().unwrap_or(0);
                                c.turboquant()
                                    .ok_or_else(|| {
                                        anyhow!(
                                            "Gemma4Attention: cached attention returned without paged or TurboQuant storage"
                                        )
                                    })?
                                    .materialize_prefix_on(len, c.dtype(), target)?
                            }
                        } else {
                            c.update_and_fetch_for_attention_on(&k, &v, lens, target)?
                        };
                        cached_attention_out = maybe_cached_out;
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
        let (kv, sliced_mask) = if segment_stable_sliding || batch_stable_sliding {
            (kv, None)
        } else {
            self.sliding_attention_view_on(&kv, mask, seq, target)?
        };
        let mask = sliced_mask.as_ref().or(mask);

        let t0 = Instant::now();
        let out = if let Some(out) = cached_attention_out {
            out
        } else if segment_stable_sliding {
            segment_stable_sliding_attention_on(
                &q,
                &kv,
                per_row_lens,
                offsets.values(),
                self.sliding_window,
                target,
            )?
        } else if batch_stable_sliding {
            row_isolated_causal_attention_on(
                &q,
                &kv,
                mask,
                per_row_lens,
                offsets.values(),
                Some(self.sliding_window),
                target,
            )?
        } else if position_stable_full {
            query_position_isolated_full_attention_on(
                &q,
                &kv,
                mask,
                per_row_lens,
                offsets.values(),
                target,
            )?
        } else if query_isolated {
            query_isolated_full_attention_on(&q, &kv, per_row_lens, offsets.values(), target)?
        } else if batch_stable_full {
            row_isolated_causal_attention_on(
                &q,
                &kv,
                mask,
                per_row_lens,
                offsets.values(),
                None,
                target,
            )?
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

    fn sliding_attention_view_on(
        &self,
        kv: &SharedKv,
        mask: Option<&Array>,
        query_len: i32,
        target: StreamOrDevice,
    ) -> Result<(SharedKv, Option<Array>)> {
        if self.layer_kind != Gemma4LayerKind::Sliding {
            return Ok((kv.clone(), None));
        }
        if kv.keys.shape().as_slice()[0] != 1 {
            return Ok((kv.clone(), None));
        }
        let kv_len = kv.keys.shape().as_slice()[2];
        let view_len = sliding_attention_view_len(kv_len, query_len, self.sliding_window);
        if view_len >= kv_len {
            let aligned_mask = align_attention_mask_tail_on(mask, kv_len, target)?;
            return Ok((kv.clone(), aligned_mask));
        }
        let sliced = slice_shared_kv_tail_on(kv, view_len, target)?;
        let sliced_mask = align_attention_mask_tail_on(mask, view_len, target)?;
        Ok((sliced, sliced_mask))
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

fn query_isolated_full_attention_on(
    queries: &Array,
    kv: &SharedKv,
    per_row_lens: Option<&[i32]>,
    offsets: &[i32],
    target: StreamOrDevice,
) -> Result<Array> {
    let q_shape = queries.shape();
    let q_dims = q_shape.as_slice();
    let (batch, heads, query_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let kv_len = kv.keys.shape().as_slice()[2];
    let lens_owned;
    let lens = match per_row_lens {
        Some(lens) => lens,
        None => {
            lens_owned = vec![query_len; batch as usize];
            &lens_owned
        }
    };
    if offsets.len() != batch as usize || lens.len() != batch as usize {
        return Err(anyhow!(
            "Gemma4 query-isolated attention expected {batch} offsets/lens, got {}/{}",
            offsets.len(),
            lens.len()
        ));
    }
    let mut batch_outputs = Vec::with_capacity(batch as usize);
    for row in 0..batch {
        let valid_queries = lens[row as usize];
        let mut row_outputs = Vec::with_capacity(query_len as usize);
        for query_idx in 0..query_len {
            let query = mlx::ops::indexing::slice_strided_on(
                queries,
                &[row, 0, query_idx, 0][..],
                &[row + 1, heads, query_idx + 1, head_dim][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )?;
            if query_idx >= valid_queries {
                row_outputs.push(&query * 0.0_f32);
                continue;
            }
            let key_end = offsets[row as usize] + query_idx + 1;
            if key_end <= 0 || key_end > kv_len {
                return Err(anyhow!(
                    "Gemma4 query-isolated attention key end {key_end} outside (0,{kv_len}] for row {row} query {query_idx}"
                ));
            }
            let keys = mlx::ops::indexing::slice_strided_on(
                &kv.keys,
                &[row, 0, 0, 0][..],
                &[row + 1, kv.keys.shape().as_slice()[1], key_end, head_dim][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )?;
            let values = mlx::ops::indexing::slice_strided_on(
                &kv.values,
                &[row, 0, 0, 0][..],
                &[
                    row + 1,
                    kv.values.shape().as_slice()[1],
                    key_end,
                    kv.values.shape().as_slice()[3],
                ][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )?;
            row_outputs.push(mlx::fast::scaled_dot_product_attention_on(
                &query, &keys, &values, 1.0, "", None, None, target,
            )?);
        }
        let row_refs = row_outputs.iter().collect::<Vec<_>>();
        batch_outputs.push(mlx::ops::shape::concatenate_on(&row_refs, 2, target)?);
    }
    let batch_refs = batch_outputs.iter().collect::<Vec<_>>();
    Ok(mlx::ops::shape::concatenate_on(&batch_refs, 0, target)?)
}

fn query_position_isolated_full_attention_on(
    queries: &Array,
    kv: &SharedKv,
    mask: Option<&Array>,
    per_row_lens: Option<&[i32]>,
    offsets: &[i32],
    target: StreamOrDevice,
) -> Result<Array> {
    let q_shape = queries.shape();
    let q_dims = q_shape.as_slice();
    if q_dims.len() != 4 {
        return Err(anyhow!(
            "Gemma4 exact full attention expected rank-4 queries, got {q_dims:?}"
        ));
    }
    let (batch, heads, query_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let lens_owned;
    let lens = match per_row_lens {
        Some(lens) => lens,
        None => {
            lens_owned = vec![query_len; batch as usize];
            &lens_owned
        }
    };
    if offsets.len() != batch as usize || lens.len() != batch as usize {
        return Err(anyhow!(
            "Gemma4 exact full attention expected {batch} offsets/lens, got {}/{}",
            offsets.len(),
            lens.len()
        ));
    }
    if query_len <= 1 {
        return Err(anyhow!(
            "Gemma4 exact full attention requires Q>1, got Q={query_len}"
        ));
    }
    for (row, &len) in lens.iter().enumerate() {
        if len < 0 || len > query_len {
            return Err(anyhow!(
                "Gemma4 exact full attention invalid row {row} length {len} for Q={query_len}"
            ));
        }
    }

    let key_shape = kv.keys.shape();
    let key_dims = key_shape.as_slice();
    let value_shape = kv.values.shape();
    let value_dims = value_shape.as_slice();
    if key_dims.len() != 4
        || value_dims.len() != 4
        || key_dims[0] != batch
        || value_dims[0] != batch
        || key_dims[2] != value_dims[2]
    {
        return Err(anyhow!(
            "Gemma4 exact full attention incompatible KV shapes: keys={key_dims:?}, values={value_dims:?}, batch={batch}"
        ));
    }
    let final_key_len = key_dims[2];
    let mask_shape = mask.map(Array::shape);
    if let Some(mask_shape) = mask_shape.as_ref() {
        let dims = mask_shape.as_slice();
        if dims.len() != 4
            || (!matches!(dims[0], 1) && dims[0] != batch)
            || dims[2] < query_len
            || dims[3] < final_key_len
        {
            return Err(anyhow!(
                "Gemma4 exact full attention mask {dims:?} cannot cover B={batch}, Q={query_len}, K={final_key_len}"
            ));
        }
    }

    let mut outputs = Vec::with_capacity(query_len as usize);
    for depth in 0..query_len {
        let key_end = offsets
            .iter()
            .zip(lens.iter())
            .map(|(&offset, &len)| offset + (depth + 1).min(len))
            .max()
            .unwrap_or(0);
        if key_end <= 0 || key_end > final_key_len {
            return Err(anyhow!(
                "Gemma4 exact full attention key end {key_end} outside (0,{final_key_len}] at depth {depth}"
            ));
        }
        let query = mlx::ops::indexing::slice_strided_on(
            queries,
            &[0_i32, 0, depth, 0][..],
            &[batch, heads, depth + 1, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let keys = mlx::ops::indexing::slice_strided_on(
            &kv.keys,
            &[0_i32, 0, 0, 0][..],
            &[batch, key_dims[1], key_end, key_dims[3]][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let values = mlx::ops::indexing::slice_strided_on(
            &kv.values,
            &[0_i32, 0, 0, 0][..],
            &[batch, value_dims[1], key_end, value_dims[3]][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let depth_mask = mask
            .map(|mask| {
                let dims = mask_shape
                    .as_ref()
                    .expect("mask dimensions exist when mask exists")
                    .as_slice();
                mlx::ops::indexing::slice_strided_on(
                    mask,
                    &[0_i32, 0, depth, 0][..],
                    &[dims[0], dims[1], depth + 1, key_end][..],
                    &[1_i32, 1, 1, 1][..],
                    target,
                )
                .map_err(anyhow::Error::from)
            })
            .transpose()?;
        outputs.push(mlx::fast::scaled_dot_product_attention_on(
            &query,
            &keys,
            &values,
            1.0,
            "",
            depth_mask.as_ref(),
            None,
            target,
        )?);
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    mlx::ops::shape::concatenate_on(&refs, 2, target).map_err(Into::into)
}

fn segment_stable_sliding_attention_on(
    queries: &Array,
    kv: &SharedKv,
    per_row_lens: Option<&[i32]>,
    offsets: &[i32],
    sliding_window: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let q_shape = queries.shape();
    let q_dims = q_shape.as_slice();
    let (batch, heads, query_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    if batch != 1 || sliding_window <= 0 {
        return Err(anyhow!(
            "Gemma4 segment-stable sliding attention requires B=1 and a positive window, got B={batch}, window={sliding_window}"
        ));
    }
    let valid_query_len = match per_row_lens {
        Some([len]) => *len,
        Some(lens) => {
            return Err(anyhow!(
                "Gemma4 segment-stable sliding attention expected one row length, got {}",
                lens.len()
            ));
        }
        None => query_len,
    };
    if valid_query_len <= 0 || valid_query_len > query_len || offsets.len() != 1 {
        return Err(anyhow!(
            "Gemma4 segment-stable sliding attention invalid row length {valid_query_len}, query length {query_len}, or offset count {}",
            offsets.len()
        ));
    }
    let kv_dims = kv.keys.shape();
    let kv_dims = kv_dims.as_slice();
    let kv_len = kv_dims[2];

    let mut query_rows = Vec::with_capacity(valid_query_len as usize);
    let mut key_windows = Vec::with_capacity(valid_query_len as usize);
    let mut value_windows = Vec::with_capacity(valid_query_len as usize);
    for query_idx in 0..valid_query_len {
        let (key_start, key_end) =
            stable_sliding_query_window(kv_len, offsets[0], query_idx, sliding_window)?;
        query_rows.push(mlx::ops::indexing::slice_strided_on(
            queries,
            &[0_i32, 0, query_idx, 0][..],
            &[1_i32, heads, query_idx + 1, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?);
        key_windows.push(mlx::ops::indexing::slice_strided_on(
            &kv.keys,
            &[0_i32, 0, key_start, 0][..],
            &[1_i32, kv_dims[1], key_end, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?);
        value_windows.push(mlx::ops::indexing::slice_strided_on(
            &kv.values,
            &[0_i32, 0, key_start, 0][..],
            &[
                1_i32,
                kv.values.shape().as_slice()[1],
                key_end,
                kv.values.shape().as_slice()[3],
            ][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?);
    }
    let query_refs = query_rows.iter().collect::<Vec<_>>();
    let key_refs = key_windows.iter().collect::<Vec<_>>();
    let value_refs = value_windows.iter().collect::<Vec<_>>();
    let packed_queries = mlx::ops::shape::concatenate_on(&query_refs, 0, target)?;
    let keys = mlx::ops::shape::concatenate_on(&key_refs, 0, target)?;
    let values = mlx::ops::shape::concatenate_on(&value_refs, 0, target)?;
    let out = mlx::fast::scaled_dot_product_attention_on(
        &packed_queries,
        &keys,
        &values,
        1.0,
        "",
        None,
        None,
        target,
    )?
    .transpose_axes_on(&[2_i32, 1, 0, 3][..], target)?;
    if valid_query_len == query_len {
        return Ok(out);
    }
    let padding = mlx::ops::indexing::slice_strided_on(
        queries,
        &[0_i32, 0, valid_query_len, 0][..],
        &[1_i32, heads, query_len, head_dim][..],
        &[1_i32, 1, 1, 1][..],
        target,
    )? * 0.0_f32;
    mlx::ops::shape::concatenate_on(&[&out, &padding], 2, target).map_err(Into::into)
}

fn row_isolated_causal_attention_on(
    queries: &Array,
    kv: &SharedKv,
    mask: Option<&Array>,
    per_row_lens: Option<&[i32]>,
    offsets: &[i32],
    sliding_window: Option<i32>,
    target: StreamOrDevice,
) -> Result<Array> {
    let q_shape = queries.shape();
    let q_dims = q_shape.as_slice();
    let (batch, heads, query_len, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let lens_owned;
    let lens = match per_row_lens {
        Some(lens) => lens,
        None => {
            lens_owned = vec![query_len; batch as usize];
            &lens_owned
        }
    };
    if offsets.len() != batch as usize || lens.len() != batch as usize {
        return Err(anyhow!(
            "Gemma4 row-isolated causal attention expected {batch} offsets/lens, got {}/{}",
            offsets.len(),
            lens.len()
        ));
    }
    if sliding_window.is_some_and(|window| window <= 0) {
        return Err(anyhow!(
            "Gemma4 row-isolated causal attention requires a positive sliding window"
        ));
    }
    let kv_dims = kv.keys.shape();
    let kv_dims = kv_dims.as_slice();
    let kv_len = kv_dims[2];
    let mut batch_outputs = Vec::with_capacity(batch as usize);
    for row in 0..batch {
        let valid_query_len = lens[row as usize];
        if valid_query_len < 0 || valid_query_len > query_len {
            return Err(anyhow!(
                "Gemma4 row-isolated causal attention invalid row {row} length {valid_query_len} for query length {query_len}"
            ));
        }
        if valid_query_len == 0 {
            batch_outputs.push(
                mlx::ops::indexing::slice_strided_on(
                    queries,
                    &[row, 0, 0, 0][..],
                    &[row + 1, heads, query_len, head_dim][..],
                    &[1_i32, 1, 1, 1][..],
                    target,
                )? * 0.0_f32,
            );
            continue;
        }
        let key_end = offsets[row as usize] + valid_query_len;
        let key_start = sliding_window.map_or(0, |window| {
            key_end - window.saturating_add(valid_query_len).saturating_sub(1)
        });
        if key_start < 0 || key_end > kv_len {
            return Err(anyhow!(
                "Gemma4 row-isolated causal window is invalid: row={row}, kv_len={kv_len}, offset={}, query_len={valid_query_len}, sliding_window={sliding_window:?}",
                offsets[row as usize]
            ));
        }
        let query = mlx::ops::indexing::slice_strided_on(
            queries,
            &[row, 0, 0, 0][..],
            &[row + 1, heads, valid_query_len, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let keys = mlx::ops::indexing::slice_strided_on(
            &kv.keys,
            &[row, 0, key_start, 0][..],
            &[row + 1, kv_dims[1], key_end, head_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let values = mlx::ops::indexing::slice_strided_on(
            &kv.values,
            &[row, 0, key_start, 0][..],
            &[
                row + 1,
                kv.values.shape().as_slice()[1],
                key_end,
                kv.values.shape().as_slice()[3],
            ][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let row_mask = mask
            .map(|mask| {
                let mask_shape = mask.shape();
                let mask_dims = mask_shape.as_slice();
                if mask_dims.len() != 4 {
                    return Err(anyhow!(
                        "Gemma4 row-isolated causal attention expected rank-4 mask, got {:?}",
                        mask_dims
                    ));
                }
                let mask_row = if mask_dims[0] == 1 { 0 } else { row };
                if mask_row >= mask_dims[0]
                    || valid_query_len > mask_dims[2]
                    || key_end > mask_dims[3]
                {
                    return Err(anyhow!(
                        "Gemma4 row-isolated causal mask cannot cover row={row}, query_len={valid_query_len}, key_end={key_end}, shape={mask_dims:?}"
                    ));
                }
                Ok(mlx::ops::indexing::slice_strided_on(
                    mask,
                    &[mask_row, 0, 0, key_start][..],
                    &[mask_row + 1, mask_dims[1], valid_query_len, key_end][..],
                    &[1_i32, 1, 1, 1][..],
                    target,
                )?)
            })
            .transpose()?;
        let mut out = mlx::fast::scaled_dot_product_attention_on(
            &query,
            &keys,
            &values,
            1.0,
            if row_mask.is_some() { "" } else { "causal" },
            row_mask.as_ref(),
            None,
            target,
        )?;
        if valid_query_len < query_len {
            let padding = mlx::ops::indexing::slice_strided_on(
                queries,
                &[row, 0, valid_query_len, 0][..],
                &[row + 1, heads, query_len, head_dim][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )? * 0.0_f32;
            out = mlx::ops::shape::concatenate_on(&[&out, &padding], 2, target)?;
        }
        batch_outputs.push(out);
    }
    let batch_refs = batch_outputs.iter().collect::<Vec<_>>();
    Ok(mlx::ops::shape::concatenate_on(&batch_refs, 0, target)?)
}

fn stable_sliding_query_window(
    kv_len: i32,
    offset: i32,
    query_idx: i32,
    sliding_window: i32,
) -> Result<(i32, i32)> {
    let key_end = offset + query_idx + 1;
    let key_start = key_end - sliding_window;
    if query_idx < 0 || sliding_window <= 0 || key_start < 0 || key_end > kv_len {
        return Err(anyhow!(
            "Gemma4 segment-stable sliding window is invalid: kv_len={kv_len}, offset={offset}, query_idx={query_idx}, window={sliding_window}"
        ));
    }
    Ok((key_start, key_end))
}

fn sliding_attention_view_len(kv_len: i32, query_len: i32, window: i32) -> i32 {
    if kv_len <= 0 {
        return 0;
    }
    if query_len <= 0 || window <= 0 {
        return kv_len;
    }
    let keep = window.saturating_add(query_len).saturating_sub(1);
    kv_len.min(keep.max(1))
}

fn slice_shared_kv_tail_on(
    kv: &SharedKv,
    keep_len: i32,
    target: StreamOrDevice,
) -> Result<SharedKv> {
    let keys_shape = kv.keys.shape();
    let keys_dims = keys_shape.as_slice();
    let values_shape = kv.values.shape();
    let values_dims = values_shape.as_slice();
    let kv_len = keys_dims[2];
    if keep_len >= kv_len {
        return Ok(kv.clone());
    }
    let start = kv_len - keep_len;
    let keys = mlx::ops::indexing::slice_strided_on(
        &kv.keys,
        &[0_i32, 0, start, 0][..],
        &[keys_dims[0], keys_dims[1], kv_len, keys_dims[3]][..],
        &[1_i32, 1, 1, 1][..],
        target,
    )?;
    let values = mlx::ops::indexing::slice_strided_on(
        &kv.values,
        &[0_i32, 0, start, 0][..],
        &[values_dims[0], values_dims[1], kv_len, values_dims[3]][..],
        &[1_i32, 1, 1, 1][..],
        target,
    )?;
    Ok(SharedKv { keys, values })
}

fn slice_attention_mask_tail_on(
    mask: &Array,
    keep_len: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = mask.shape();
    let dims = shape.as_slice();
    if dims.is_empty() {
        return Err(anyhow!("Gemma4 attention mask must have rank >= 1"));
    }
    let k_axis = dims.len() - 1;
    let k_len = dims[k_axis];
    if keep_len > k_len {
        return Err(anyhow!(
            "Gemma4 attention mask K length {k_len} shorter than requested keep_len {keep_len}"
        ));
    }
    if keep_len == k_len {
        return Ok(mask.clone());
    }
    let mut start = vec![0_i32; dims.len()];
    let stop = dims.to_vec();
    let strides = vec![1_i32; dims.len()];
    start[k_axis] = k_len - keep_len;
    Ok(mlx::ops::indexing::slice_strided_on(
        mask,
        start.as_slice(),
        stop.as_slice(),
        strides.as_slice(),
        target,
    )?)
}

fn align_attention_mask_tail_on(
    mask: Option<&Array>,
    keep_len: i32,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    match mask {
        Some(mask) => Ok(Some(slice_attention_mask_tail_on(mask, keep_len, target)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        align_attention_mask_tail_on, query_isolated_full_attention_on,
        row_isolated_causal_attention_on, segment_stable_sliding_attention_on,
        sliding_attention_view_len, stable_sliding_query_window, SharedKv,
    };
    use mlx::{Array, Dtype};
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
    fn query_isolated_full_attention_handles_ragged_rows() {
        let queries: Array = (
            &[
                1.0_f32, 0.0, 0.0, 1.0, // row 0
                1.0, 1.0, 9.0, 9.0, // row 1; second query is padding
            ][..],
            &[2_i32, 1, 2, 2][..],
        )
            .try_into()
            .unwrap();
        let keys: Array = (
            &[
                1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, // row 0
                1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, // row 1
            ][..],
            &[2_i32, 1, 4, 2][..],
        )
            .try_into()
            .unwrap();
        let values = keys.clone();

        let out = query_isolated_full_attention_on(
            &queries,
            &SharedKv { keys, values },
            Some(&[2, 1]),
            &[1, 2],
            ().into(),
        )
        .unwrap();
        assert_eq!(out.shape().as_slice(), &[2, 1, 2, 2]);
        let values: Vec<f32> = out.to_vec().unwrap();
        assert_eq!(&values[6..8], &[0.0, 0.0]);
    }

    #[test]
    fn sliding_attention_view_keeps_window_plus_query_minus_one() {
        assert_eq!(sliding_attention_view_len(20_400, 2_048, 1_024), 3_071);
        assert_eq!(sliding_attention_view_len(20_400, 1, 1_024), 1_024);
        assert_eq!(sliding_attention_view_len(512, 2_048, 1_024), 512);
    }

    #[test]
    fn stable_sliding_windows_preserve_the_same_absolute_causal_prefix() {
        assert_eq!(
            stable_sliding_query_window(513, 511, 1, 512).unwrap(),
            (1, 513)
        );
        assert_eq!(
            stable_sliding_query_window(514, 511, 0, 512).unwrap(),
            (0, 512)
        );
        assert!(stable_sliding_query_window(511, 511, 0, 512).is_err());
    }

    #[test]
    #[serial(mlx_metal)]
    fn stable_sliding_attention_is_invariant_to_verify_segment_boundaries() {
        let global_keys = (0..515)
            .flat_map(|idx| [idx as f32 / 515.0, (idx % 17) as f32 / 17.0])
            .collect::<Vec<_>>();
        let global_keys: Array = (global_keys.as_slice(), &[1_i32, 1, 515, 2][..])
            .try_into()
            .unwrap();
        let cap1_keys =
            mlx::ops::indexing::slice(&global_keys, [0_i32, 0, 0, 0], [1_i32, 1, 513, 2]).unwrap();
        let cap2_keys =
            mlx::ops::indexing::slice(&global_keys, [0_i32, 0, 1, 0], [1_i32, 1, 515, 2]).unwrap();
        let cap1_queries: Array = (&[0.1_f32, 0.2, 0.3, 0.4][..], &[1_i32, 1, 2, 2][..])
            .try_into()
            .unwrap();
        let cap2_queries: Array = (
            &[0.3_f32, 0.4, 0.5, 0.6, 0.7, 0.8][..],
            &[1_i32, 1, 3, 2][..],
        )
            .try_into()
            .unwrap();

        let cap1 = segment_stable_sliding_attention_on(
            &cap1_queries,
            &SharedKv {
                keys: cap1_keys.clone(),
                values: cap1_keys,
            },
            None,
            &[511],
            512,
            ().into(),
        )
        .unwrap();
        let cap2 = segment_stable_sliding_attention_on(
            &cap2_queries,
            &SharedKv {
                keys: cap2_keys.clone(),
                values: cap2_keys,
            },
            None,
            &[511],
            512,
            ().into(),
        )
        .unwrap();
        let cap1_values = cap1.to_vec::<f32>().unwrap();
        let cap2_values = cap2.to_vec::<f32>().unwrap();
        assert_eq!(&cap1_values[2..4], &cap2_values[0..2]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn stable_sliding_attention_preserves_rows_and_zero_length_padding() {
        let queries: Array = (
            &[0.1_f32, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4][..],
            &[2_i32, 1, 2, 2][..],
        )
            .try_into()
            .unwrap();
        let row_keys = (0..513)
            .flat_map(|idx| [idx as f32 / 513.0, (idx % 17) as f32 / 17.0])
            .collect::<Vec<_>>();
        let row_keys: Array = (row_keys.as_slice(), &[1_i32, 1, 513, 2][..])
            .try_into()
            .unwrap();
        let keys = mlx::ops::shape::concatenate_on(&[&row_keys, &row_keys], 0, ()).unwrap();
        let full = row_isolated_causal_attention_on(
            &queries,
            &SharedKv {
                keys: keys.clone(),
                values: keys.clone(),
            },
            None,
            Some(&[2, 2]),
            &[511, 511],
            Some(512),
            ().into(),
        )
        .unwrap();
        let full_values = full.to_vec::<f32>().unwrap();
        assert_eq!(&full_values[..4], &full_values[4..]);

        let padded = row_isolated_causal_attention_on(
            &queries,
            &SharedKv {
                keys: keys.clone(),
                values: keys,
            },
            None,
            Some(&[2, 0]),
            &[511, 511],
            Some(512),
            ().into(),
        )
        .unwrap();
        let padded_values = padded.to_vec::<f32>().unwrap();
        assert_eq!(&padded_values[..4], &full_values[..4]);
        assert_eq!(&padded_values[4..], &[0.0_f32; 4]);
    }

    #[test]
    fn sliding_attention_mask_aligns_to_already_sliced_kv() {
        let mask = Array::zeros((1_i32, 1_i32, 3_i32, 5_504_i32), Dtype::Float32).unwrap();
        let aligned = align_attention_mask_tail_on(Some(&mask), 514, ().into())
            .unwrap()
            .expect("aligned mask");

        assert_eq!(aligned.shape().as_slice(), &[1_i32, 1, 3, 514]);
    }

    #[test]
    fn sliding_attention_mask_rejects_shorter_than_kv() {
        let mask = Array::zeros((1_i32, 1_i32, 3_i32, 10_i32), Dtype::Float32).unwrap();
        let err = align_attention_mask_tail_on(Some(&mask), 11, ().into())
            .expect_err("short mask must fail");

        assert!(
            err.to_string().contains("shorter than requested"),
            "unexpected error: {err:#}"
        );
    }
}
