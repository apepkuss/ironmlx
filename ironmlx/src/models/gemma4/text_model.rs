use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};
use std::time::Instant;

use crate::core::Loader;
use crate::nn::{Embedding, LayerCache, Linear, RmsNorm};
use crate::Result;

use super::attention::SharedKv;
use super::config::{Gemma4LayerKind, Gemma4TextConfig};
use super::decoder_layer::Gemma4DecoderLayer;
use super::profile;
use super::rope::RopeOffsets;

#[derive(Clone, Default)]
pub struct Gemma4SharedKvStates {
    sliding: Option<SharedKv>,
    full: Option<SharedKv>,
}

pub struct Gemma4TextForwardOutput {
    pub hidden: Array,
    pub shared_kv: Gemma4SharedKvStates,
}

impl Gemma4SharedKvStates {
    pub fn insert(&mut self, kind: Gemma4LayerKind, kv: SharedKv) {
        match kind {
            Gemma4LayerKind::Sliding => self.sliding = Some(kv),
            Gemma4LayerKind::Full => self.full = Some(kv),
        }
    }

    pub fn get(&self, kind: Gemma4LayerKind) -> Option<&SharedKv> {
        match kind {
            Gemma4LayerKind::Sliding => self.sliding.as_ref(),
            Gemma4LayerKind::Full => self.full.as_ref(),
        }
    }

    pub fn require(&self, kind: Gemma4LayerKind) -> Result<&SharedKv> {
        self.get(kind)
            .ok_or_else(|| anyhow!("Gemma4 shared K/V missing {:?}", kind))
    }
}

pub struct Gemma4TextModel {
    embed_tokens: Embedding,
    embed_tokens_per_layer: Option<Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    layers: Vec<Gemma4DecoderLayer>,
    norm: RmsNorm,
    cfg: Gemma4TextConfig,
}

impl Gemma4TextModel {
    pub fn from_loader(loader: &Loader, cfg: Gemma4TextConfig) -> Result<Self> {
        Self::from_loader_impl(loader, cfg, false)
    }

    pub fn from_loader_external_shared_kv(loader: &Loader, cfg: Gemma4TextConfig) -> Result<Self> {
        Self::from_loader_impl(loader, cfg, true)
    }

    fn from_loader_impl(
        loader: &Loader,
        cfg: Gemma4TextConfig,
        kv_shared_only: bool,
    ) -> Result<Self> {
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")?;
        let has_per_layer = cfg.hidden_size_per_layer_input > 0;
        let embed_tokens_per_layer = if has_per_layer {
            Some(Embedding::from_loader(
                loader,
                "model.embed_tokens_per_layer",
            )?)
        } else {
            None
        };
        let per_layer_model_projection = if has_per_layer {
            Some(Linear::from_loader(
                loader,
                "model.per_layer_model_projection",
            )?)
        } else {
            None
        };
        let per_layer_projection_norm = if has_per_layer {
            Some(RmsNorm::from_loader(
                loader,
                "model.per_layer_projection_norm",
                cfg.rms_norm_eps,
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers as usize {
            layers.push(if kv_shared_only {
                Gemma4DecoderLayer::from_loader_kv_shared_only(
                    loader,
                    &format!("model.layers.{i}"),
                    &cfg,
                    i,
                )?
            } else {
                Gemma4DecoderLayer::from_loader(loader, &format!("model.layers.{i}"), &cfg, i)?
            });
        }
        let norm = RmsNorm::from_loader(loader, "model.norm", cfg.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            cfg,
        })
    }

    pub fn config(&self) -> &Gemma4TextConfig {
        &self.cfg
    }

    pub fn hidden_dtype(&self) -> Dtype {
        Dtype::Float32
    }

    pub fn embed_on(&self, input_ids: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let h = self.embed_tokens.forward_on(input_ids, target)?;
        Ok(&h * (self.cfg.hidden_size as f32).sqrt())
    }

    pub fn as_output_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.as_output_on(hidden, target)
    }

    pub(crate) fn dense_embedding_weight_on(
        &self,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.embed_tokens.dense_weight_on(target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if input_ids.ndim() != 2 {
            return Err(anyhow!(
                "Gemma4TextModel::forward_on: input_ids must be rank-2 [B,S], got rank {}",
                input_ids.ndim()
            ));
        }
        let hidden = self.embed_on(input_ids, target)?;
        self.forward_embeddings_on(&hidden, input_ids, per_row_lens, cache, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_embeddings_on(
        &self,
        hidden: &Array,
        per_layer_token_ids: &Array,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let profile = profile::vl_layer_enabled();
        let total_t0 = Instant::now();
        let t0 = Instant::now();
        let per_layer_inputs = self.per_layer_inputs_on(per_layer_token_ids, hidden, target)?;
        if let Some(pli) = per_layer_inputs.as_ref() {
            profile::eval("gemma4_text_per_layer_inputs", &[pli], t0, profile)?;
        } else {
            profile::log("gemma4_text_per_layer_inputs", t0, profile);
        }
        let out = self.forward_post_embedding_on(
            hidden,
            per_layer_inputs.as_ref(),
            per_row_lens,
            cache,
            target,
            None,
        )?;
        profile::eval(
            "gemma4_text_forward_embeddings_breakdown_total",
            &[&out],
            total_t0,
            profile,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_embeddings_with_shared_kv_on(
        &self,
        hidden: &Array,
        per_layer_token_ids: &Array,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Gemma4TextForwardOutput> {
        let target = target.into();
        let per_layer_inputs = self.per_layer_inputs_on(per_layer_token_ids, hidden, target)?;
        self.forward_post_embedding_with_shared_kv_on(
            hidden,
            per_layer_inputs.as_ref(),
            per_row_lens,
            cache,
            target,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_post_embedding_on(
        &self,
        hidden: &Array,
        per_layer_inputs: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
        layer_last_trace: Option<&mut Vec<Array>>,
    ) -> Result<Array> {
        Ok(self
            .forward_post_embedding_with_shared_kv_on(
                hidden,
                per_layer_inputs,
                per_row_lens,
                cache,
                target,
                layer_last_trace,
            )?
            .hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_post_embedding_with_shared_kv_on(
        &self,
        hidden: &Array,
        per_layer_inputs: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        mut cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
        mut layer_last_trace: Option<&mut Vec<Array>>,
    ) -> Result<Gemma4TextForwardOutput> {
        let dims_borrow = hidden.shape();
        let dims = dims_borrow.as_slice();
        let (batch, seq) = (dims[0], dims[1]);
        let first_cache_layer = self.cfg.first_kv_shared_layer_idx();
        if let Some(c) = cache.as_deref() {
            if c.len() != first_cache_layer {
                return Err(anyhow!(
                    "Gemma4TextModel: cache.len()={} != cache-bearing layers {}",
                    c.len(),
                    first_cache_layer
                ));
            }
        }

        let lens_owned;
        let lens = match per_row_lens {
            Some(l) => l,
            None => {
                lens_owned = vec![seq; batch as usize];
                &lens_owned
            }
        };
        if lens.len() != batch as usize {
            return Err(anyhow!(
                "Gemma4TextModel: per_row_lens.len()={} != batch={}",
                lens.len(),
                batch
            ));
        }
        let profile = profile::vl_layer_enabled();
        let t0 = Instant::now();
        let offsets = RopeOffsets::from_values(cache_offsets(cache.as_deref(), batch)?)?;
        profile::log("gemma4_text_cache_offsets", t0, profile);
        let explicit_masks = cache.is_some() || per_row_lens.is_some() || batch > 1;
        let single_row_decode = cache.is_some() && per_row_lens.is_none() && batch == 1 && seq == 1;
        let single_row_decode_len = offsets.values().first().copied().unwrap_or(0) + seq;
        let full_mask = if explicit_masks && !single_row_decode {
            let t0 = Instant::now();
            let mask =
                build_attention_mask(offsets.values(), lens, seq, None, Dtype::Bfloat16, target)?;
            profile::eval("gemma4_text_full_mask_build", &[&mask], t0, profile)?;
            Some(mask)
        } else {
            profile::log("gemma4_text_full_mask_build", Instant::now(), profile);
            None
        };
        let needs_sliding_decode_mask =
            single_row_decode && single_row_decode_len > self.cfg.sliding_window;
        let sliding_mask = if needs_sliding_decode_mask
            || (!single_row_decode && (explicit_masks || seq > self.cfg.sliding_window))
        {
            let t0 = Instant::now();
            let mask = build_attention_mask(
                offsets.values(),
                lens,
                seq,
                Some(self.cfg.sliding_window),
                Dtype::Bfloat16,
                target,
            )?;
            profile::eval("gemma4_text_sliding_mask_build", &[&mask], t0, profile)?;
            Some(mask)
        } else {
            profile::log("gemma4_text_sliding_mask_build", Instant::now(), profile);
            None
        };

        let mut x = hidden.clone();
        let mut intermediates: Vec<Option<SharedKv>> = vec![None; self.layers.len()];
        let mut shared_kv = Gemma4SharedKvStates::default();
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_kind = self.cfg.layer_kind(idx);
            let mask = match layer_kind {
                Gemma4LayerKind::Sliding => sliding_mask.as_ref(),
                Gemma4LayerKind::Full => full_mask.as_ref(),
            };
            let pli = match per_layer_inputs {
                Some(all) => {
                    let t0 = Instant::now();
                    let side = slice_per_layer_input(
                        all,
                        idx as i32,
                        self.cfg.hidden_size_per_layer_input,
                        target,
                    )?;
                    profile::eval_layer(
                        "gemma4_text_layer_side_slice",
                        idx,
                        layer_kind,
                        &[&side],
                        t0,
                        profile,
                    )?;
                    Some(side)
                }
                None => None,
            };
            let prev_idx = self.cfg.previous_kv_layer(idx);
            let shared = if prev_idx == idx {
                None
            } else {
                intermediates[prev_idx].as_ref()
            };
            let cache_cell = match cache.as_deref_mut() {
                Some(c) if idx < first_cache_layer => Some(&mut c[idx]),
                _ => None,
            };
            let t0 = Instant::now();
            let (next, kv) = layer.forward_on(
                &x,
                mask,
                pli.as_ref(),
                per_row_lens,
                &offsets,
                shared,
                cache_cell,
                target,
                None,
            )?;
            profile::eval_layer(
                "gemma4_text_layer_total",
                idx,
                layer_kind,
                &[&next],
                t0,
                profile,
            )?;
            x = next;
            if let Some(trace) = layer_last_trace.as_deref_mut() {
                trace.push(slice_last_token(&x, target)?);
            }
            shared_kv.insert(layer_kind, kv.clone());
            intermediates[idx] = Some(kv);
        }
        let t0 = Instant::now();
        let out = self.norm.forward_on(&x, target)?;
        profile::eval("gemma4_text_final_norm", &[&out], t0, profile)?;
        Ok(Gemma4TextForwardOutput {
            hidden: out,
            shared_kv,
        })
    }

    pub(crate) fn forward_external_shared_kv_on(
        &self,
        hidden: &Array,
        shared_kv: &Gemma4SharedKvStates,
        masks: &super::drafter::Gemma4DrafterMasks,
        position: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let offsets = RopeOffsets::from_values(vec![position])?;
        let mut x = hidden.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_kind = self.cfg.layer_kind(idx);
            let mask = masks.get(layer_kind);
            let kv = shared_kv.require(layer_kind)?;
            let (next, _) =
                layer.forward_on(&x, mask, None, None, &offsets, Some(kv), None, target, None)?;
            x = next;
        }
        self.norm.forward_on(&x, target)
    }

    #[doc(hidden)]
    pub fn forward_layer_last_trace_on(
        &self,
        input_ids: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Vec<Array>> {
        let target = target.into();
        if input_ids.ndim() != 2 {
            return Err(anyhow!(
                "Gemma4TextModel::forward_layer_last_trace_on: input_ids must be rank-2 [B,S], got rank {}",
                input_ids.ndim()
            ));
        }
        let hidden = self.embed_on(input_ids, target)?;
        let per_layer_inputs = self.per_layer_inputs_on(input_ids, &hidden, target)?;
        let mut trace = Vec::with_capacity(self.layers.len());
        let _ = self.forward_post_embedding_on(
            &hidden,
            per_layer_inputs.as_ref(),
            None,
            None,
            target,
            Some(&mut trace),
        )?;
        Ok(trace)
    }

    #[doc(hidden)]
    pub fn forward_text_layer0_stage_last_trace_on(
        &self,
        input_ids: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Vec<Array>> {
        let target = target.into();
        if input_ids.ndim() != 2 {
            return Err(anyhow!(
                "Gemma4TextModel::forward_text_layer0_stage_last_trace_on: input_ids must be rank-2 [B,S], got rank {}",
                input_ids.ndim()
            ));
        }
        let hidden = self.embed_on(input_ids, target)?;
        let mut trace = Vec::with_capacity(10);
        trace.push(slice_last_token(&hidden, target)?);
        let seq = hidden.shape().as_slice()[1];
        let lens = vec![seq; hidden.shape().as_slice()[0] as usize];
        let offsets = RopeOffsets::from_values(vec![0; hidden.shape().as_slice()[0] as usize])?;
        let mask = if seq > self.cfg.sliding_window {
            Some(build_attention_mask(
                offsets.values(),
                &lens,
                seq,
                Some(self.cfg.sliding_window),
                Dtype::Bfloat16,
                target,
            )?)
        } else {
            None
        };
        let per_layer_inputs = self.per_layer_inputs_on(input_ids, &hidden, target)?;
        let pli = match per_layer_inputs.as_ref() {
            Some(all) => Some(slice_per_layer_input(
                all,
                0,
                self.cfg.hidden_size_per_layer_input,
                target,
            )?),
            None => None,
        };
        let _ = self.layers[0].forward_on(
            &hidden,
            mask.as_ref(),
            pli.as_ref(),
            None,
            &offsets,
            None,
            None,
            target,
            Some(&mut trace),
        )?;
        Ok(trace)
    }

    fn per_layer_inputs_on(
        &self,
        input_ids: &Array,
        hidden: &Array,
        target: StreamOrDevice,
    ) -> Result<Option<Array>> {
        if self.cfg.hidden_size_per_layer_input <= 0 {
            return Ok(None);
        }
        let profile = profile::vl_layer_enabled();
        let total_t0 = Instant::now();
        let t0 = Instant::now();
        let token_inputs = self
            .embed_tokens_per_layer
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4TextModel: embed_tokens_per_layer missing"))?
            .forward_on(input_ids, target)?;
        let token_inputs = &token_inputs * (self.cfg.hidden_size_per_layer_input as f32).sqrt();
        profile::eval(
            "gemma4_text_per_layer_token_inputs",
            &[&token_inputs],
            t0,
            profile,
        )?;

        let dims_borrow = input_ids.shape();
        let dims = dims_borrow.as_slice();
        let (batch, seq) = (dims[0], dims[1]);
        let layers = self.cfg.num_hidden_layers;
        let pli = self.cfg.hidden_size_per_layer_input;
        let token_inputs = token_inputs.reshape_on((batch, seq, layers, pli), target)?;

        let t0 = Instant::now();
        let projected = self
            .per_layer_model_projection
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4TextModel: per_layer_model_projection missing"))?
            .forward_on(hidden, target)?;
        let projected = &projected * (self.cfg.hidden_size as f32).powf(-0.5);
        let projected = projected.reshape_on((batch, seq, layers, pli), target)?;
        profile::eval(
            "gemma4_text_per_layer_model_projection",
            &[&projected],
            t0,
            profile,
        )?;
        let t0 = Instant::now();
        let projected = self
            .per_layer_projection_norm
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4TextModel: per_layer_projection_norm missing"))?
            .forward_on(&projected, target)?;
        let out = (&projected + &token_inputs) * 2.0_f32.powf(-0.5);
        profile::eval(
            "gemma4_text_per_layer_projection_norm",
            &[&out],
            t0,
            profile,
        )?;
        profile::eval(
            "gemma4_text_per_layer_inputs_total",
            &[&out],
            total_t0,
            profile,
        )?;
        Ok(Some(out))
    }
}

fn cache_offsets(cache: Option<&[LayerCache]>, batch: i32) -> Result<Vec<i32>> {
    match cache {
        Some(cells) => {
            for cell in cells {
                if let LayerCache::Full(kv) = cell {
                    return Ok(kv.offsets().to_vec());
                }
            }
            Err(anyhow!("Gemma4TextModel: cache has no Full KV layer"))
        }
        None => Ok(vec![0; batch as usize]),
    }
}

fn slice_per_layer_input(
    per_layer_inputs: &Array,
    layer_idx: i32,
    hidden_size_per_layer_input: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = per_layer_inputs.shape();
    let s = shape.as_slice();
    Ok(mlx::ops::indexing::slice_strided_on(
        per_layer_inputs,
        &[0_i32, 0, layer_idx, 0][..],
        &[s[0], s[1], layer_idx + 1, hidden_size_per_layer_input][..],
        &[1_i32, 1, 1, 1][..],
        target,
    )?
    .reshape_on((s[0], s[1], hidden_size_per_layer_input), target)?)
}

fn slice_last_token(hidden: &Array, target: StreamOrDevice) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 {
        return Err(anyhow!(
            "Gemma4TextModel: expected hidden [B,S,H], got {dims:?}"
        ));
    }
    let (b, s, h) = (dims[0], dims[1], dims[2]);
    Ok(mlx::ops::indexing::slice_strided_on(
        hidden,
        &[0_i32, s - 1, 0][..],
        &[b, s, h][..],
        &[1_i32, 1, 1][..],
        target,
    )?
    .reshape_on((b, h), target)?)
}

fn build_attention_mask(
    offsets: &[i32],
    per_row_lens: &[i32],
    q_len: i32,
    window: Option<i32>,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Array> {
    if offsets.len() != per_row_lens.len() {
        return Err(anyhow!(
            "Gemma4 mask: offsets.len()={} != per_row_lens.len()={}",
            offsets.len(),
            per_row_lens.len()
        ));
    }
    if q_len <= 0 {
        return Err(anyhow!("Gemma4 mask: q_len must be > 0, got {q_len}"));
    }
    let b = offsets.len();
    let k_len = offsets
        .iter()
        .zip(per_row_lens.iter())
        .map(|(o, n)| o + n)
        .max()
        .unwrap_or(0);
    if k_len <= 0 {
        return Err(anyhow!("Gemma4 mask: computed k_len <= 0"));
    }
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; b * q_len as usize * k_len as usize];
    for row in 0..b {
        let offset = offsets[row];
        let real = per_row_lens[row].max(0);
        let row_real_k = offset + real;
        for q in 0..q_len {
            let base = (row * q_len as usize + q as usize) * k_len as usize;
            if q < real {
                let q_abs = offset + q;
                let min_k = window.map_or(0, |w| (q_abs - w + 1).max(0));
                let max_k = q_abs.min(row_real_k - 1);
                if max_k >= min_k {
                    for k in min_k..=max_k {
                        flat[base + k as usize] = 0.0;
                    }
                }
            } else {
                let self_k = (offset + q).clamp(0, k_len - 1);
                flat[base + self_k as usize] = 0.0;
            }
        }
    }
    let arr_f32: Array = (&flat[..], &[b as i32, 1_i32, q_len, k_len][..]).try_into()?;
    Ok(mlx::ops::cast::astype_on(&arr_f32, dtype, target)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_mask_window_edges() {
        let mask =
            build_attention_mask(&[0], &[514], 514, Some(512), Dtype::Float32, ().into()).unwrap();
        let v: Vec<f32> = mask.to_vec().unwrap();
        let q_len = 514usize;
        let k_len = 514usize;
        let at = |q: usize, k: usize| v[q * k_len + k];
        assert_eq!(q_len, 514);
        assert_eq!(at(0, 0), 0.0);
        assert!(at(512, 0).is_infinite() && at(512, 0).is_sign_negative());
        assert_eq!(at(512, 1), 0.0);
        assert_eq!(at(513, 2), 0.0);
    }
}
