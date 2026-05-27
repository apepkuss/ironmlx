use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Embedding, LayerCache, Linear, RmsNorm};
use crate::Result;

use super::attention::SharedKv;
use super::config::{Gemma4LayerKind, Gemma4TextConfig};
use super::decoder_layer::Gemma4DecoderLayer;
use super::rope::RopeOffsets;

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
            layers.push(Gemma4DecoderLayer::from_loader(
                loader,
                &format!("model.layers.{i}"),
                &cfg,
                i,
            )?);
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

    pub fn embed_on(&self, input_ids: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let h = self.embed_tokens.forward_on(input_ids, target)?;
        Ok(&h * (self.cfg.hidden_size as f32).sqrt())
    }

    pub fn as_output_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.as_output_on(hidden, target)
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
        let per_layer_inputs = self.per_layer_inputs_on(input_ids, &hidden, target)?;
        self.forward_post_embedding_on(
            &hidden,
            per_layer_inputs.as_ref(),
            per_row_lens,
            cache,
            target,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_post_embedding_on(
        &self,
        hidden: &Array,
        per_layer_inputs: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        mut cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
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
        let offsets = RopeOffsets::from_values(cache_offsets(cache.as_deref(), batch)?)?;
        let explicit_masks = cache.is_some() || per_row_lens.is_some() || batch > 1;
        let full_mask = if explicit_masks {
            Some(build_attention_mask(
                offsets.values(),
                lens,
                seq,
                None,
                Dtype::Bfloat16,
                target,
            )?)
        } else {
            None
        };
        let sliding_mask = if explicit_masks || seq > self.cfg.sliding_window {
            Some(build_attention_mask(
                offsets.values(),
                lens,
                seq,
                Some(self.cfg.sliding_window),
                Dtype::Bfloat16,
                target,
            )?)
        } else {
            None
        };

        let mut x = hidden.clone();
        let mut intermediates: Vec<Option<SharedKv>> = vec![None; self.layers.len()];
        for (idx, layer) in self.layers.iter().enumerate() {
            let mask = match self.cfg.layer_kind(idx) {
                Gemma4LayerKind::Sliding => sliding_mask.as_ref(),
                Gemma4LayerKind::Full => full_mask.as_ref(),
            };
            let pli = match per_layer_inputs {
                Some(all) => Some(slice_per_layer_input(
                    all,
                    idx as i32,
                    self.cfg.hidden_size_per_layer_input,
                    target,
                )?),
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
            let (next, kv) = layer.forward_on(
                &x,
                mask,
                pli.as_ref(),
                per_row_lens,
                &offsets,
                shared,
                cache_cell,
                target,
            )?;
            x = next;
            intermediates[idx] = Some(kv);
        }
        self.norm.forward_on(&x, target)
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
        let token_inputs = self
            .embed_tokens_per_layer
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4TextModel: embed_tokens_per_layer missing"))?
            .forward_on(input_ids, target)?;
        let token_inputs = &token_inputs * (self.cfg.hidden_size_per_layer_input as f32).sqrt();

        let dims_borrow = input_ids.shape();
        let dims = dims_borrow.as_slice();
        let (batch, seq) = (dims[0], dims[1]);
        let layers = self.cfg.num_hidden_layers;
        let pli = self.cfg.hidden_size_per_layer_input;
        let token_inputs = token_inputs.reshape_on((batch, seq, layers, pli), target)?;

        let projected = self
            .per_layer_model_projection
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4TextModel: per_layer_model_projection missing"))?
            .forward_on(hidden, target)?;
        let projected = &projected * (self.cfg.hidden_size as f32).powf(-0.5);
        let projected = projected.reshape_on((batch, seq, layers, pli), target)?;
        let projected = self
            .per_layer_projection_norm
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4TextModel: per_layer_projection_norm missing"))?
            .forward_on(&projected, target)?;
        Ok(Some((&projected + &token_inputs) * 2.0_f32.powf(-0.5)))
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
