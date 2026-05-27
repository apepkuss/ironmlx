use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};
use std::time::Instant;

use crate::core::cache::KVCache;
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::LayerCache;
use crate::Result;

use super::config::Gemma4Config;
use super::ops::logit_softcap_on;
use super::text_model::Gemma4TextModel;
use super::vision::{MultimodalEmbedder, VisionModel};

pub struct Gemma4Model {
    text: Gemma4TextModel,
    vision: Option<VisionModel>,
    embed_vision: Option<MultimodalEmbedder>,
    vision_config: Option<super::config::Gemma4VisionConfig>,
    vision_soft_tokens_per_image: i32,
    image_token_id: Option<i32>,
    audio_token_id: Option<i32>,
}

fn per_row_slice_last(
    hidden: &Array,
    last_positions: &[i32],
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let dims_borrow = hidden.shape();
    let dims = dims_borrow.as_slice();
    let (b, s, h) = (dims[0], dims[1], dims[2]);
    if last_positions.len() as i32 != b {
        return Err(anyhow!(
            "Gemma4 per_row_slice_last: last_positions.len()={} != batch={}",
            last_positions.len(),
            b
        ));
    }
    let mut rows = Vec::with_capacity(b as usize);
    for (i, &pos) in last_positions.iter().enumerate() {
        if pos < 0 || pos >= s {
            return Err(anyhow!(
                "Gemma4 per_row_slice_last: last_positions[{i}]={pos} out of [0,{s})"
            ));
        }
        rows.push(mlx::ops::indexing::slice_strided_on(
            hidden,
            &[i as i32, pos, 0][..],
            &[i as i32 + 1, pos + 1, h][..],
            &[1_i32, 1, 1][..],
            target,
        )?);
    }
    let refs: Vec<&Array> = rows.iter().collect();
    Ok(mlx::ops::shape::concatenate_on(&refs, 0, target)?)
}

fn vl_profile_enabled() -> bool {
    std::env::var_os("IRONMLX_GEMMA4_VL_PROFILE").is_some()
}

fn vl_profile_eval(label: &str, arrays: &[&Array], start: Instant, enabled: bool) -> Result<()> {
    if enabled {
        mlx::transforms::eval(arrays)?;
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

struct ExactUniqueImageRows {
    unique_indices: Vec<usize>,
    image_rows: Vec<(usize, usize)>,
}

fn exact_unique_image_rows(
    pixel_values: &[Array],
    group_indices: &[usize],
    profile: bool,
) -> Result<ExactUniqueImageRows> {
    if group_indices.len() <= 1 {
        let idx = group_indices
            .first()
            .copied()
            .ok_or_else(|| anyhow!("Gemma4 exact image dedup: empty shape group"))?;
        return Ok(ExactUniqueImageRows {
            unique_indices: vec![idx],
            image_rows: vec![(idx, 0)],
        });
    }

    let t0 = Instant::now();
    let mut unique_indices = Vec::with_capacity(group_indices.len());
    let mut unique_pixels: Vec<Vec<f32>> = Vec::with_capacity(group_indices.len());
    let mut image_rows = Vec::with_capacity(group_indices.len());

    for &idx in group_indices {
        let pv = &pixel_values[idx];
        if pv.dtype() != Dtype::Float32 {
            return Err(anyhow!(
                "Gemma4 exact image dedup expects Float32 pixel_values, image {idx} has {:?}",
                pv.dtype()
            ));
        }
        let pixels = pv.to_vec::<f32>()?;
        if let Some(row) = unique_pixels
            .iter()
            .position(|existing| existing == &pixels)
        {
            image_rows.push((idx, row));
        } else {
            let row = unique_indices.len();
            unique_indices.push(idx);
            unique_pixels.push(pixels);
            image_rows.push((idx, row));
        }
    }

    if profile {
        tracing::info!(
            "[gemma4-vl-profile] compute_vision_exact_dedup_ms={:.3} images={} unique={} duplicates={}",
            t0.elapsed().as_secs_f64() * 1000.0,
            group_indices.len(),
            unique_indices.len(),
            group_indices.len() - unique_indices.len()
        );
    }

    Ok(ExactUniqueImageRows {
        unique_indices,
        image_rows,
    })
}

impl Gemma4Model {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Gemma4Config::from_loader(loader).context("parsing Gemma4Config")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Gemma4Config) -> Result<Self> {
        let image_token_id = cfg.image_token_id;
        let audio_token_id = cfg.audio_token_id;
        let vision_config = cfg.vision_config.clone();
        let vision = if let Some(vc) = vision_config.as_ref() {
            if loader.contains("vision_tower.patch_embedder.input_proj.weight") {
                Some(VisionModel::from_loader(loader, vc.clone())?)
            } else {
                None
            }
        } else {
            None
        };
        let embed_vision = if vision.is_some() {
            Some(MultimodalEmbedder::from_loader(
                loader,
                "embed_vision",
                vision_config
                    .as_ref()
                    .map(|vc| vc.rms_norm_eps)
                    .unwrap_or(1e-6),
            )?)
        } else {
            None
        };
        let vision_soft_tokens_per_image = cfg.vision_soft_tokens_per_image;
        let text = Gemma4TextModel::from_loader(loader, cfg.text_config)?;
        Ok(Self {
            text,
            vision,
            embed_vision,
            vision_config,
            vision_soft_tokens_per_image,
            image_token_id,
            audio_token_id,
        })
    }

    pub fn text(&self) -> &Gemma4TextModel {
        &self.text
    }

    pub fn config(&self) -> &super::config::Gemma4TextConfig {
        self.text.config()
    }

    fn slice_last_and_project(
        &self,
        hidden: &Array,
        last_positions: Option<&[i32]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let dims_borrow = hidden.shape();
        let dims = dims_borrow.as_slice();
        let (b, s, h) = (dims[0], dims[1], dims[2]);
        let last_hidden = match last_positions {
            Some(positions) if s > 1 => per_row_slice_last(hidden, positions, target)?,
            _ if s > 1 => mlx::ops::indexing::slice_strided_on(
                hidden,
                &[0_i32, s - 1, 0][..],
                &[b, s, h][..],
                &[1_i32, 1, 1][..],
                target,
            )?,
            _ => hidden.clone(),
        };
        let logits = self.text.as_output_on(&last_hidden, target)?;
        match self.config().final_logit_softcapping {
            Some(softcap) => logit_softcap_on(&logits, softcap, target),
            None => Ok(logits),
        }
    }

    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        self.reject_multimodal_tokens(input_ids)?;
        let hidden = self.text.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )?;
        self.slice_last_and_project(&hidden, None, target)
    }

    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = self.config();
        let first_shared = cfg.first_kv_shared_layer_idx();
        let mut out = Vec::with_capacity(first_shared);
        for i in 0..first_shared {
            let head_dim = cfg.head_dim_for_layer(i);
            out.push(LayerCache::Full(
                KVCache::new(
                    batch,
                    cfg.kv_heads_for_layer(i),
                    head_dim,
                    head_dim,
                    dtype,
                    cap,
                )
                .with_step(cap),
            ));
        }
        Ok(out)
    }

    fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let inter = cfg.intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;
        let pli = cfg.hidden_size_per_layer_input as usize;
        let per_layer_vocab = cfg.vocab_size_per_layer_input() as usize;

        let embed = vocab * h / 2;
        let per_layer_embed = if pli > 0 {
            per_layer_vocab * l * pli / 2
        } else {
            0
        };
        let per_layer_projection = if pli > 0 { h * l * pli * 2 } else { 0 };
        let mut attn = 0usize;
        for i in 0..l {
            let hd = cfg.head_dim_for_layer(i) as usize;
            let n_heads = cfg.num_attention_heads as usize;
            let n_kv = cfg.kv_heads_for_layer(i) as usize;
            attn += (n_heads * hd * h + 2 * n_kv * hd * h + h * n_heads * hd) / 2;
        }
        let mlp = l * (2 * inter * h + h * inter) / 2;
        embed + per_layer_embed + per_layer_projection + attn + mlp
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = self.config();
        ModelMeta {
            // Memory budget uses this as KV-bearing layer count.
            num_hidden_layers: cfg.first_kv_shared_layer_idx() as i32,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            // Conservative for mixed 256/512 K/V layers.
            head_dim: Some(cfg.global_head_dim.unwrap_or(cfg.head_dim)),
            weight_bytes: self.approx_weight_bytes(),
            max_position_embeddings: cfg.max_position_embeddings,
            spatial_merge_size: self
                .vision_config
                .as_ref()
                .map(|vc| vc.pooling_kernel_size)
                .unwrap_or(3),
        }
    }

    fn reject_multimodal_tokens(&self, input_ids: &Array) -> Result<()> {
        if input_ids.ndim() == 2 && input_ids.shape_at(1) == 1 {
            return Ok(());
        }
        let ids_i32 = mlx::ops::astype(input_ids, Dtype::Int32)?;
        let ids = ids_i32.to_vec::<i32>()?;
        if let Some(image_id) = self.image_token_id {
            if ids.contains(&image_id) {
                return Err(anyhow!(
                    "Gemma4Model: image token {image_id} encountered, but Gemma4 vision/audio is out of scope for this Dense text-only task"
                ));
            }
        }
        if let Some(audio_id) = self.audio_token_id {
            if ids.contains(&audio_id) {
                return Err(anyhow!(
                    "Gemma4Model: audio token {audio_id} encountered, but Gemma4 vision/audio is out of scope for this Dense text-only task"
                ));
            }
        }
        Ok(())
    }

    fn zero_image_token_ids(&self, input_ids: &Array, image_token_id: i32) -> Result<Array> {
        let shape = input_ids.shape();
        let dims = shape.as_slice();
        if dims.len() != 2 {
            return Err(anyhow!(
                "Gemma4Model::zero_image_token_ids expects [B,S], got {dims:?}"
            ));
        }
        let ids_i32 = mlx::ops::astype(input_ids, Dtype::Int32)?;
        let mut ids = ids_i32.to_vec::<i32>()?;
        for id in &mut ids {
            if *id == image_token_id {
                *id = 0;
            }
        }
        Ok((ids.as_slice(), dims).try_into()?)
    }

    fn count_image_tokens(&self, input_ids: &Array, image_token_id: i32) -> Result<usize> {
        let ids_i32 = mlx::ops::astype(input_ids, Dtype::Int32)?;
        let ids = ids_i32.to_vec::<i32>()?;
        Ok(ids.iter().filter(|&&id| id == image_token_id).count())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_hidden_on(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let profile = vl_profile_enabled();
        let total_t0 = Instant::now();
        let t0 = Instant::now();
        let img_count = self.count_image_tokens(input_ids, image_token_id)?;
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] forward_vl_count_image_tokens_ms={:.3} image_tokens={}",
                t0.elapsed().as_secs_f64() * 1000.0,
                img_count
            );
        }
        if img_count > 0 && vision_embeds_slice.is_none() {
            return Err(anyhow!(
                "Gemma4Model::forward_vl_hidden: chunk has {img_count} image tokens but no vision embeddings"
            ));
        }
        let t0 = Instant::now();
        let mut hidden = self.text.embed_on(input_ids, target)?;
        vl_profile_eval("forward_vl_text_embed", &[&hidden], t0, profile)?;
        if let Some(ve) = vision_embeds_slice {
            let t0 = Instant::now();
            hidden =
                super::cross_modal::replace_image_tokens(&hidden, input_ids, ve, image_token_id)?;
            vl_profile_eval("forward_vl_replace_image_tokens", &[&hidden], t0, profile)?;
        }
        let t0 = Instant::now();
        let per_layer_ids = if img_count > 0 {
            self.zero_image_token_ids(input_ids, image_token_id)?
        } else {
            input_ids.clone()
        };
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] forward_vl_zero_image_token_ids_ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t0 = Instant::now();
        let hidden = self.text.forward_embeddings_on(
            &hidden,
            &per_layer_ids,
            per_row_lens,
            cache,
            target,
        )?;
        vl_profile_eval(
            "forward_vl_text_forward_embeddings",
            &[&hidden],
            t0,
            profile,
        )?;
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] forward_vl_hidden_total_ms={:.3}",
                total_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(hidden)
    }
}

impl Model for Gemma4Model {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        Gemma4Model::make_cache(self, batch, cap, dtype)
    }

    fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Gemma4Model::forward_on(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.reject_multimodal_tokens(input_ids)?;
        let hidden = self.text.forward_on(
            input_ids,
            position_ids,
            Some(per_row_lens),
            None,
            cache,
            target,
        )?;
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }

    fn forward_text_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.reject_multimodal_tokens(input_ids)?;
        self.text.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn model_meta(&self) -> ModelMeta {
        Gemma4Model::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }
}

impl crate::core::scheduler::DenseVlMethods for Gemma4Model {
    fn batched_prefill_vl(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&[Array]>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let target: StreamOrDevice = target;
        let profile = vl_profile_enabled();
        let total_t0 = Instant::now();
        let b = per_row_lens.len();
        if per_row_pixel_values.len() != b || per_row_grid_thw.len() != b {
            return Err(anyhow!(
                "Gemma4Model::batched_prefill_vl: per-row vision arg lengths must equal B={b}"
            ));
        }

        let t0 = Instant::now();
        let mut hidden = self.text.embed_on(input_ids, target)?;
        vl_profile_eval("vl_text_embed", &[&hidden], t0, profile)?;
        let mut all_vision = Vec::new();
        let t0 = Instant::now();
        for i in 0..b {
            match (per_row_pixel_values[i], per_row_grid_thw[i]) {
                (Some(pv), Some(grids)) if !grids.is_empty() => {
                    all_vision.push(self.compute_vision_embeds(pv, grids, target)?);
                }
                (Some(_), None) => {
                    return Err(anyhow!(
                        "Gemma4Model::batched_prefill_vl: row {i} has pixel_values but no grid_thw"
                    ));
                }
                _ => {}
            }
        }
        if !all_vision.is_empty() {
            let refs: Vec<&Array> = all_vision.iter().collect();
            vl_profile_eval("vl_compute_vision_embeds_all", &refs, t0, profile)?;
        }
        if !all_vision.is_empty() {
            let t0 = Instant::now();
            let vision_concat = if all_vision.len() == 1 {
                all_vision.pop().expect("len checked")
            } else {
                let refs: Vec<&Array> = all_vision.iter().collect();
                mlx::ops::shape::concatenate_on(&refs, 0, target)?
            };
            hidden = super::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                &vision_concat,
                image_token_id,
            )?;
            vl_profile_eval("vl_replace_image_tokens", &[&hidden], t0, profile)?;
        }
        let t0 = Instant::now();
        let per_layer_ids = self.zero_image_token_ids(input_ids, image_token_id)?;
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] vl_zero_image_token_ids_ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t0 = Instant::now();
        let hidden = self.text.forward_embeddings_on(
            &hidden,
            &per_layer_ids,
            Some(per_row_lens),
            cache,
            target,
        )?;
        vl_profile_eval("vl_text_forward_embeddings", &[&hidden], t0, profile)?;
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        let t0 = Instant::now();
        let logits = self.slice_last_and_project(&hidden, Some(&last_positions), target)?;
        vl_profile_eval("vl_slice_project", &[&logits], t0, profile)?;
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] vl_batched_prefill_total_ms={:.3}",
                total_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(logits)
    }

    fn compute_vision_embeds(
        &self,
        pixel_values: &[Array],
        grid_thw: &[(i32, i32, i32)],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let target: StreamOrDevice = target;
        let profile = vl_profile_enabled();
        let total_t0 = Instant::now();
        if pixel_values.is_empty() {
            return Err(anyhow!(
                "Gemma4Model::compute_vision_embeds: pixel_values cannot be empty"
            ));
        }
        if grid_thw.is_empty() {
            return Err(anyhow!(
                "Gemma4Model::compute_vision_embeds: grid_thw cannot be empty"
            ));
        }
        if pixel_values.len() != grid_thw.len() {
            return Err(anyhow!(
                "Gemma4Model::compute_vision_embeds: pixel_values.len()={} must equal grid_thw.len()={}",
                pixel_values.len(),
                grid_thw.len()
            ));
        }
        let vision = self.vision.as_ref().ok_or_else(|| {
            anyhow!("Gemma4Model has no vision_tower; use Loader::open_multimodal")
        })?;
        let embed_vision = self
            .embed_vision
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4Model has no embed_vision projection"))?;
        let pool = self
            .vision_config
            .as_ref()
            .map(|vc| vc.pooling_kernel_size)
            .unwrap_or(3);
        let mut processed = vec![false; pixel_values.len()];
        let mut per_image: Vec<Option<Array>> = (0..pixel_values.len()).map(|_| None).collect();
        for group_start in 0..pixel_values.len() {
            if processed[group_start] {
                continue;
            }
            let group_shape = pixel_values[group_start].shape();
            let group_dims = group_shape.as_slice();
            if group_dims.len() != 4 || group_dims[0] != 1 || group_dims[1] != 3 {
                return Err(anyhow!(
                    "Gemma4Model::compute_vision_embeds image {group_start} expected [1,3,H,W], got {group_dims:?}"
                ));
            }
            let mut group_indices = Vec::new();
            for idx in group_start..pixel_values.len() {
                if processed[idx] {
                    continue;
                }
                if pixel_values[idx].shape().as_slice() == group_dims {
                    group_indices.push(idx);
                    processed[idx] = true;
                }
            }
            if profile && group_indices.len() > 1 {
                tracing::info!(
                    "[gemma4-vl-profile] compute_vision_batch_group images={} shape={:?}",
                    group_indices.len(),
                    group_dims
                );
            }
            let exact_rows = exact_unique_image_rows(pixel_values, &group_indices, profile)?;
            if profile && exact_rows.unique_indices.len() != group_indices.len() {
                tracing::info!(
                    "[gemma4-vl-profile] compute_vision_exact_reuse images={} unique={}",
                    group_indices.len(),
                    exact_rows.unique_indices.len()
                );
            }
            let t0 = Instant::now();
            let batch = if exact_rows.unique_indices.len() == 1 {
                pixel_values[exact_rows.unique_indices[0]].clone()
            } else {
                let refs: Vec<&Array> = exact_rows
                    .unique_indices
                    .iter()
                    .map(|&idx| &pixel_values[idx])
                    .collect();
                mlx::ops::shape::concatenate_on(&refs, 0, target)?
            };
            vl_profile_eval("compute_vision_batch_concat", &[&batch], t0, profile)?;
            let t0 = Instant::now();
            let features = vision.forward_on(&batch, target)?;
            vl_profile_eval("compute_vision_tower", &[&features], t0, profile)?;
            let t0 = Instant::now();
            let projected = embed_vision.forward_on(&features, target)?;
            vl_profile_eval(
                "compute_embed_vision_projection",
                &[&projected],
                t0,
                profile,
            )?;
            let shape = projected.shape();
            let dims = shape.as_slice();
            let unique_len = i32::try_from(exact_rows.unique_indices.len())
                .context("Gemma4Model::compute_vision_embeds group size overflow")?;
            if dims.len() != 3 || dims[0] != unique_len {
                return Err(anyhow!(
                    "Gemma4Model::compute_vision_embeds expected batched projection [{unique_len},N,H], got {dims:?}"
                ));
            }
            let n = dims[1];
            let hidden = dims[2];
            let mut unique_outputs: Vec<Option<Array>> =
                (0..exact_rows.unique_indices.len()).map(|_| None).collect();
            for (unique_row, (&idx, output_slot)) in exact_rows
                .unique_indices
                .iter()
                .zip(unique_outputs.iter_mut())
                .enumerate()
            {
                let (t, gh, gw) = grid_thw[idx];
                let expected = t * (gh / pool) * (gw / pool);
                if expected != n {
                    return Err(anyhow!(
                        "Gemma4Model::compute_vision_embeds image {idx}: grid_thw implies {expected} soft tokens but vision produced {n}; max per image {}",
                        self.vision_soft_tokens_per_image
                    ));
                }
                let t0 = Instant::now();
                let row = i32::try_from(unique_row)
                    .context("Gemma4Model::compute_vision_embeds batch row overflow")?;
                let sliced = mlx::ops::indexing::slice_strided_on(
                    &projected,
                    &[row, 0, 0][..],
                    &[row + 1, n, hidden][..],
                    &[1_i32, 1, 1][..],
                    target,
                )?;
                let out = sliced.reshape_on((n, hidden), target)?;
                vl_profile_eval("compute_vision_reshape", &[&out], t0, profile)?;
                *output_slot = Some(out);
            }
            for (idx, unique_row) in exact_rows.image_rows {
                let out = unique_outputs
                    .get(unique_row)
                    .and_then(|item| item.as_ref())
                    .ok_or_else(|| {
                        anyhow!(
                            "Gemma4Model::compute_vision_embeds missing unique image row {unique_row}"
                        )
                    })?
                    .clone();
                let (t, gh, gw) = grid_thw[idx];
                let expected = t * (gh / pool) * (gw / pool);
                if expected != n {
                    return Err(anyhow!(
                        "Gemma4Model::compute_vision_embeds image {idx}: grid_thw implies {expected} soft tokens but reused vision row has {n}; max per image {}",
                        self.vision_soft_tokens_per_image
                    ));
                }
                per_image[idx] = Some(out);
            }
        }
        let mut per_image: Vec<Array> = per_image
            .into_iter()
            .enumerate()
            .map(|(idx, item)| {
                item.ok_or_else(|| {
                    anyhow!("Gemma4Model::compute_vision_embeds image {idx} was not processed")
                })
            })
            .collect::<Result<_>>()?;
        let out = if per_image.len() == 1 {
            per_image.pop().expect("len checked")
        } else {
            let t0 = Instant::now();
            let refs: Vec<&Array> = per_image.iter().collect();
            let merged = mlx::ops::shape::concatenate_on(&refs, 0, target)?;
            vl_profile_eval("compute_vision_concat", &[&merged], t0, profile)?;
            merged
        };
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] compute_vision_embeds_total_ms={:.3}",
                total_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(out)
    }

    fn forward_vl_chunk(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let profile = vl_profile_enabled();
        let total_t0 = Instant::now();
        let hidden = self.forward_vl_hidden_on(
            input_ids,
            _position_ids,
            per_row_lens,
            _decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )?;
        let t0 = Instant::now();
        let logits = self.slice_last_and_project(&hidden, None, target)?;
        vl_profile_eval("forward_vl_slice_project", &[&logits], t0, profile)?;
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] forward_vl_chunk_total_ms={:.3}",
                total_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(logits)
    }

    fn forward_vl_hidden(
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.forward_vl_hidden_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_unique_image_rows_reuses_identical_pixels() {
        let a: Array = (
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0][..],
            &[1_i32, 3, 1, 2][..],
        )
            .try_into()
            .unwrap();
        let b: Array = (
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0][..],
            &[1_i32, 3, 1, 2][..],
        )
            .try_into()
            .unwrap();
        let c: Array = (
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 6.0][..],
            &[1_i32, 3, 1, 2][..],
        )
            .try_into()
            .unwrap();
        let images = vec![a, b, c];

        let exact_rows = exact_unique_image_rows(&images, &[0, 1, 2], false).unwrap();

        assert_eq!(exact_rows.unique_indices, vec![0, 2]);
        assert_eq!(exact_rows.image_rows, vec![(0, 0), (1, 0), (2, 1)]);
    }

    #[test]
    fn exact_unique_image_rows_preserves_unique_order() {
        let a: Array = (&[1.0_f32; 6][..], &[1_i32, 3, 1, 2][..])
            .try_into()
            .unwrap();
        let b: Array = (&[2.0_f32; 6][..], &[1_i32, 3, 1, 2][..])
            .try_into()
            .unwrap();
        let images = vec![a, b];

        let exact_rows = exact_unique_image_rows(&images, &[0, 1], false).unwrap();

        assert_eq!(exact_rows.unique_indices, vec![0, 1]);
        assert_eq!(exact_rows.image_rows, vec![(0, 0), (1, 1)]);
    }
}
