//! GLM-4.7-Flash (`glm4_moe_lite`) model.
//!
//! Task 1 SKELETON: parses + validates config and registers with the scheduler.
//! The forward pipeline (absorbed-MLA attention, noaux_tc MoE, decoder layers)
//! is implemented in Task 6 — until then the `Model` forward methods error.
//!
//! GLM is the first text-only model in this engine: the scheduler-facing
//! `DenseVlMethods` surface is implemented as a text-only stub that errors.

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::LayerCache;
use crate::Result;

use super::config::Glm4MoeLiteConfig;

pub struct Glm4MoeLiteModel {
    cfg: Glm4MoeLiteConfig,
}

impl Glm4MoeLiteModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Glm4MoeLiteConfig::from_loader(loader)?;
        Ok(Self { cfg })
    }

    pub fn config(&self) -> &Glm4MoeLiteConfig {
        &self.cfg
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = &self.cfg;
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: Some(cfg.v_head_dim),
            weight_bytes: 0,
            max_position_embeddings: cfg.max_position_embeddings,
            spatial_merge_size: 2,
        }
    }
}

impl Model for Glm4MoeLiteModel {
    fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
        Err(anyhow!(
            "Glm4MoeLiteModel::make_cache not yet implemented (Task 6)"
        ))
    }

    fn forward_on(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel::forward_on not yet implemented (Task 6)"
        ))
    }

    fn batched_prefill(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel::batched_prefill not yet implemented (Task 6)"
        ))
    }

    fn forward_text_hidden(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> Result<Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel::forward_text_hidden not yet implemented (Task 6)"
        ))
    }

    fn requires_position_ids(&self) -> bool {
        false
    }

    fn model_meta(&self) -> ModelMeta {
        Glm4MoeLiteModel::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.cfg.num_hidden_layers as usize
    }
}

impl crate::core::scheduler::DenseVlMethods for Glm4MoeLiteModel {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batched_prefill_vl(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _attention_mask: &mlx::Array,
        _linear_attention_mask: &mlx::Array,
        _per_row_lens: &[i32],
        _per_row_pixel_values: &[Option<&[mlx::Array]>],
        _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        _image_token_id: i32,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &[mlx::Array],
        _grid_thw: &[(i32, i32, i32)],
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_chunk(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _vision_embeds_slice: Option<&mlx::Array>,
        _image_token_id: i32,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_hidden(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _vision_embeds_slice: Option<&mlx::Array>,
        _image_token_id: i32,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }
}
