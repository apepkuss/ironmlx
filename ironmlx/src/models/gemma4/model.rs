use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::LayerCache;
use crate::Result;

use super::config::Gemma4Config;
use super::ops::logit_softcap_on;
use super::text_model::Gemma4TextModel;

pub struct Gemma4Model {
    text: Gemma4TextModel,
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

impl Gemma4Model {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Gemma4Config::from_loader(loader).context("parsing Gemma4Config")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Gemma4Config) -> Result<Self> {
        let image_token_id = cfg.image_token_id;
        let audio_token_id = cfg.audio_token_id;
        let text = Gemma4TextModel::from_loader(loader, cfg.text_config)?;
        Ok(Self {
            text,
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
            spatial_merge_size: 2,
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
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _per_row_pixel_values: &[Option<&Array>],
        _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        _image_token_id: i32,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Gemma4Model::batched_prefill_vl: Gemma4 Dense support is text-only in this task"
        ))
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &Array,
        _grid_thw: &[(i32, i32, i32)],
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Gemma4Model::compute_vision_embeds: Gemma4 vision/audio is out of scope"
        ))
    }

    fn forward_vl_chunk(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _vision_embeds_slice: Option<&Array>,
        _image_token_id: i32,
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Gemma4Model::forward_vl_chunk: Gemma4 Dense support is text-only in this task"
        ))
    }

    fn forward_vl_hidden(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _vision_embeds_slice: Option<&Array>,
        _image_token_id: i32,
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Gemma4Model::forward_vl_hidden: Gemma4 Dense support is text-only in this task"
        ))
    }
}
