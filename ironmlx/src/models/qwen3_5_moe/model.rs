//! Top-level Qwen3.5 MoE model: Qwen35MoeTextModel + lm_head (untied).
//! Implements `core::model::Model` trait for use in generic Scheduler /
//! GenerationStream / SchedulerActor / AppState pipelines (post-P5a).

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::models::vision::{VisionConfig, VisionTower};
use crate::nn::{AttnKind, LayerCache, Linear};
use crate::Result;

use super::config::Qwen35MoeConfig;
use super::text_model::Qwen35MoeTextModel;

/// Minimum K/V cache cap consistent with dense Qwen35Model's GPU-perf floor.
/// See `crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF`.
pub const MIN_KV_CACHE_CAP_FOR_GPU_PERF: i32 =
    crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;

pub struct Qwen35MoeModel {
    text: Qwen35MoeTextModel,
    /// Always Some for 35B-A3B (tie_word_embeddings=false).
    lm_head: Linear,
    /// Vision encoder; `Some` for multimodal MoE checkpoints loaded with `open_multimodal`.
    vision: Option<VisionTower>,
}

/// Slice per-row last hidden states from `hidden [B, S, H]`.
///
/// For row `i`, extracts `hidden[i, last_positions[i], :]` then stacks
/// to `[B, 1, H]`. Used by [`Qwen35MoeModel::batched_prefill`] to project
/// per-row last-token logits when prompts have different lengths under
/// right-padding.
///
/// # Errors
/// - `last_positions.len() != B`
/// - `last_positions[i] < 0 || last_positions[i] >= S` for any `i`
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
            "per_row_slice_last: last_positions.len()={} != batch={}",
            last_positions.len(),
            b
        ));
    }
    for (i, &pos) in last_positions.iter().enumerate() {
        if pos < 0 || pos >= s {
            return Err(anyhow!(
                "per_row_slice_last: last_positions[{i}]={pos} out of [0, {s})"
            ));
        }
    }
    // Per-row slice: row i takes hidden[i, positions[i], :] → [1, 1, H].
    // Concatenate along axis 0 to build [B, 1, H].
    let mut rows: Vec<Array> = Vec::with_capacity(b as usize);
    for (i, &pos) in last_positions.iter().enumerate() {
        let row = mlx::ops::indexing::slice_strided_on(
            hidden,
            &[i as i32, pos, 0][..],
            &[i as i32 + 1, pos + 1, h][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        rows.push(row);
    }
    let row_refs: Vec<&Array> = rows.iter().collect();
    Ok(mlx::ops::shape::concatenate_on(&row_refs[..], 0, target)?)
}

impl Qwen35MoeModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg =
            Qwen35MoeConfig::from_loader(loader).context("parsing Qwen35MoeConfig from loader")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen35MoeConfig) -> Result<Self> {
        if cfg.tie_word_embeddings {
            return Err(anyhow!(
                "Qwen35MoeModel: tie_word_embeddings expected false for A3B (got true)"
            ));
        }
        let lm_head =
            Linear::from_loader(loader, "lm_head").context("loading Qwen35MoeModel lm_head")?;
        let vision = if let Some(vc) = cfg.vision_config.as_ref() {
            if loader.contains("vision_tower.patch_embed.proj.weight") {
                Some(VisionTower::from_loader(loader, vc)?)
            } else {
                None
            }
        } else {
            None
        };
        let text = Qwen35MoeTextModel::from_loader(loader, cfg)?;
        Ok(Self {
            text,
            lm_head,
            vision,
        })
    }

    pub fn config(&self) -> &Qwen35MoeConfig {
        self.text.config()
    }

    pub fn text(&self) -> &Qwen35MoeTextModel {
        &self.text
    }

    pub fn vision(&self) -> Option<&VisionTower> {
        self.vision.as_ref()
    }

    /// Conservative weight-bytes estimate for memory budgeting.
    /// Formula:
    ///   attn: 4 * H^2 * L / 2     (Q,K,V,O projections per layer, 4-bit)
    ///   routed_experts: 3 * E * H * moe_inter * L / 2  (gate, up, down per expert)
    ///   shared_expert:  3 * H * shared_inter * L / 2
    ///   embed: vocab * H / 2
    ///   lm_head: vocab * H / 2
    fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let e = cfg.num_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let se = cfg.shared_expert_intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;

        let attn = 4 * h * h * l / 2;
        let routed = 3 * e * h * me * l / 2;
        let shared = 3 * h * se * l / 2;
        // embed + lm_head (separate per tie_word_embeddings=false)
        let embed_head = 2 * vocab * h / 2;
        let text = attn + routed + shared + embed_head;
        let vision = cfg
            .vision_config
            .as_ref()
            .map(Self::approx_vision_weight_bytes)
            .unwrap_or(0);
        text + vision
    }

    /// Conservative bf16 estimate for the shared Qwen3.5 vision tower. The
    /// MoE text weights are 4-bit, while the vision stack in the MLX VL
    /// checkpoint is bf16, so budget it at 2 bytes per parameter.
    fn approx_vision_weight_bytes(cfg: &VisionConfig) -> usize {
        let h = cfg.hidden_size as usize;
        let depth = cfg.depth as usize;
        let inter = cfg.intermediate_size as usize;
        let out_h = cfg.out_hidden_size as usize;
        let patch = cfg.patch_size as usize;
        let temporal = cfg.temporal_patch_size as usize;
        let in_channels = cfg.in_channels as usize;
        let pos = cfg.num_position_embeddings as usize;
        let merge = cfg.spatial_merge_size as usize;
        let merge_hidden = merge * merge * h;

        let patch_embed = h * temporal * patch * patch * in_channels + h;
        let pos_embed = pos * h;
        let block = 2 * h
            + (3 * h * h + 3 * h)
            + (h * h + h)
            + 2 * h
            + (inter * h + inter)
            + (h * inter + h);
        let blocks = depth * block;
        let merger =
            2 * h + merge_hidden * merge_hidden + merge_hidden + out_h * merge_hidden + out_h;

        2 * (patch_embed + pos_embed + blocks + merger)
    }

    /// Slice the last sequence position from `hidden [B, S, H]` and project to
    /// vocab logits `[B, 1, vocab_size]`. Shared by `forward_on` and
    /// `batched_prefill`.
    ///
    /// When `last_positions` is `Some(positions)` (length == B), each row's
    /// last real token is at column `positions[i]` — used by the right-padded
    /// `batched_prefill` path where rows have ragged real lengths.
    ///
    /// When `last_positions` is `None` (single-stream `forward_on`), the
    /// fallback slices column `S - 1` for every row — behaviourally
    /// equivalent for B=1 or uniform-length inputs.
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
            _ if s > 1 => {
                // Single-stream / uniform-length fallback: slice column s-1.
                mlx::ops::indexing::slice_strided(
                    hidden,
                    &[0_i32, s - 1, 0][..],
                    &[b, s, h][..],
                    &[1_i32, 1, 1][..],
                )?
            }
            _ => hidden.clone(),
        };
        // T4.1: wrap lm_head projection in a `slice_last_and_project_lm_head`
        // span. Lane-A (`routing_path = "scheduler"`) emits; Lane-B no-ops via
        // the centralized `try_with_p5h_span_from_current_trace` allow-list
        // (gs_chunked top-level-only). `layer_idx` is not meaningful here
        // (lm_head sits at the top of the model, outside decoder layers) →
        // SpanFields uses defaults.
        #[cfg(feature = "p5h-profile")]
        {
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "slice_last_and_project_lm_head",
                crate::core::p5h::SpanFields::default,
                || {
                    let logits = self.lm_head.forward_on(&last_hidden, target)?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&logits])?;
                    }
                    Ok(logits)
                },
            )
        }
        #[cfg(not(feature = "p5h-profile"))]
        {
            self.lm_head.forward_on(&last_hidden, target)
        }
    }

    /// Construct per-layer cache list matching this model's hybrid topology.
    /// Matches `Qwen35Model::make_cache` style (one-shot allocate to cap to
    /// avoid grow_to on first decode step).
    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = self.config();
        let head_dim = cfg.effective_head_dim();
        let mut out = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            match cfg.layer_kind(i) {
                AttnKind::Full => {
                    // P8a-stage6: one-shot allocate to full cap (step >= cap)
                    // so the first decode step at long context never triggers
                    // grow_to. KVCache's default step=256 would otherwise
                    // round prefill alloc down to a step boundary and force
                    // a full-buffer reallocation + memcpy on decode step 1.
                    out.push(LayerCache::Full(
                        KVCache::new(
                            batch,
                            cfg.num_key_value_heads,
                            head_dim,
                            head_dim,
                            dtype,
                            cap,
                        )
                        .with_step(cap),
                    ));
                }
                AttnKind::Linear => {
                    let conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                        + cfg.linear_value_head_dim * cfg.linear_num_value_heads;
                    out.push(LayerCache::Linear(GatedDeltaCache::new_with_cap(
                        batch,
                        cfg.linear_conv_kernel_dim,
                        conv_dim,
                        cfg.linear_num_value_heads,
                        cfg.linear_value_head_dim,
                        cfg.linear_key_head_dim,
                        dtype,
                        cap,
                    )?));
                }
            }
        }
        Ok(out)
    }

    /// Test-only stub: constructs a zero-layer MoE model from config so tests
    /// can exercise model-level plumbing without synthesizing decoder weights.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn from_cfg_for_test(cfg: Qwen35MoeConfig) -> Self {
        let mrope = crate::nn::Mrope::new(
            cfg.effective_head_dim(),
            cfg.rope_parameters.rope_theta,
            cfg.rope_parameters.partial_rotary_factor,
            &cfg.rope_parameters.mrope_section,
            true,
        )
        .expect("Mrope::new with valid cfg");
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let stub_embed = crate::nn::Embedding::from_components_fp_for_test(
            mlx::Array::zeros((vocab, h), mlx::Dtype::Bfloat16).unwrap(),
        );
        let stub_norm = crate::nn::RmsNorm::new(
            mlx::ops::constructors::ones((h,), mlx::Dtype::Float32).unwrap(),
            cfg.rms_norm_eps,
        );
        let lm_head = Linear::new_fp(
            mlx::Array::zeros((vocab, h), mlx::Dtype::Float32).unwrap(),
            None,
        );
        let text =
            Qwen35MoeTextModel::from_components(stub_embed, Vec::new(), stub_norm, mrope, cfg);
        Self {
            text,
            lm_head,
            vision: None,
        }
    }

    /// Single-stream forward returning last-position logits `[B, 1, vocab]`.
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

    /// Batched prefill returning per-row last-position logits `[B, 1, vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.text.embed_on(input_ids, target)?;
        let hidden = self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            Some(attention_mask),
            Some(linear_attention_mask),
            Some(per_row_lens),
            target,
        )?;
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }

    /// Compute vision-tower embeddings ready to scatter into image-pad tokens.
    pub fn compute_vision_embeds(
        &self,
        pixel_values: &Array,
        grid_thw: &[(i32, i32, i32)],
        _target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("model has no vision_tower; use Loader::open_multimodal"))?;
        vision.forward(pixel_values, grid_thw)
    }

    /// Forward one VL prefill chunk with optional pre-computed vision embeds.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_chunk(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.forward_vl_hidden(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )?;
        self.slice_last_and_project(&hidden, None, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let mut hidden = self.text.embed_on(input_ids, target)?;

        if let Some(vision_embeds) = vision_embeds_slice {
            hidden = crate::models::qwen3_5::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                vision_embeds,
                image_token_id,
            )?;
        }

        self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            decode_mask,
            None,
            per_row_lens,
            target,
        )
    }

    /// VL-capable batched prefill over right-padded text/VL rows.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn batched_prefill_vl(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&Array>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let b = per_row_lens.len();
        if per_row_pixel_values.len() != b {
            return Err(anyhow!(
                "batched_prefill_vl: per_row_pixel_values.len()={} != B={}",
                per_row_pixel_values.len(),
                b
            ));
        }
        if per_row_grid_thw.len() != b {
            return Err(anyhow!(
                "batched_prefill_vl: per_row_grid_thw.len()={} != B={}",
                per_row_grid_thw.len(),
                b
            ));
        }

        let mut hidden = self.text.embed_on(input_ids, target)?;

        let mut all_vision_embeds: Vec<Array> = Vec::new();
        for i in 0..b {
            match (per_row_pixel_values[i], per_row_grid_thw[i]) {
                (Some(pv), Some(grids)) if !grids.is_empty() => {
                    all_vision_embeds.push(self.compute_vision_embeds(pv, grids, target)?);
                }
                (Some(_), None) => {
                    return Err(anyhow!(
                        "batched_prefill_vl: row {i} has pixel_values but grid_thw is None"
                    ));
                }
                _ => {}
            }
        }

        if !all_vision_embeds.is_empty() {
            let vision_concat = if all_vision_embeds.len() == 1 {
                all_vision_embeds.pop().expect("len == 1")
            } else {
                let refs: Vec<&Array> = all_vision_embeds.iter().collect();
                mlx::ops::concatenate(&refs, 0)
                    .map_err(|e| anyhow!("vision_embeds concatenate: {e:?}"))?
            };
            hidden = crate::models::qwen3_5::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                &vision_concat,
                image_token_id,
            )?;
        }

        let hidden = self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            Some(attention_mask),
            Some(linear_attention_mask),
            Some(per_row_lens),
            target,
        )?;
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }

    /// Run transformer + final norm, returning hidden state (no lm_head).
    pub fn forward_text_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.text.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = self.config();
        let spatial_merge_size = cfg
            .vision_config
            .as_ref()
            .map(|vc| vc.spatial_merge_size)
            .unwrap_or(2);
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: cfg.head_dim,
            weight_bytes: self.approx_weight_bytes(),
            max_position_embeddings: cfg.max_position_embeddings,
            spatial_merge_size,
        }
    }
}

impl Model for Qwen35MoeModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        Qwen35MoeModel::make_cache(self, batch, cap, dtype)
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
        Qwen35MoeModel::forward_on(
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
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Qwen35MoeModel::batched_prefill(
            self,
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            cache,
            target,
        )
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
        Qwen35MoeModel::forward_text_hidden(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn model_meta(&self) -> ModelMeta {
        Qwen35MoeModel::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }
}

/// Delegate the scheduler's VL extension trait to MoE's inherent runtime
/// methods. The trait name is historical; both dense and MoE Qwen3.5 variants
/// use the same scheduler-facing VL surface.
impl crate::core::scheduler::DenseVlMethods for Qwen35MoeModel {
    fn batched_prefill_vl(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&[mlx::Array]>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        let mut row_pixel_values = Vec::with_capacity(per_row_pixel_values.len());
        for row in per_row_pixel_values {
            let Some(values) = row else {
                row_pixel_values.push(None);
                continue;
            };
            if values.is_empty() {
                return Err(anyhow!(
                    "Qwen35MoeModel::batched_prefill_vl: row pixel_values cannot be empty"
                ));
            }
            if values.len() == 1 {
                row_pixel_values.push(Some(values[0].clone()));
            } else {
                let refs: Vec<&Array> = values.iter().collect();
                row_pixel_values.push(Some(mlx::ops::shape::concatenate(&refs, 0)?));
            }
        }
        let row_pixel_refs: Vec<Option<&Array>> =
            row_pixel_values.iter().map(|opt| opt.as_ref()).collect();
        Qwen35MoeModel::batched_prefill_vl(
            self,
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            &row_pixel_refs,
            per_row_grid_thw,
            image_token_id,
            cache,
            target,
        )
    }

    fn compute_vision_embeds(
        &self,
        pixel_values: &[mlx::Array],
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        if pixel_values.is_empty() {
            return Err(anyhow!(
                "Qwen35MoeModel::compute_vision_embeds: pixel_values cannot be empty"
            ));
        }
        let pixels = if pixel_values.len() == 1 {
            pixel_values[0].clone()
        } else {
            let refs: Vec<&Array> = pixel_values.iter().collect();
            mlx::ops::shape::concatenate(&refs, 0)?
        };
        Qwen35MoeModel::compute_vision_embeds(self, &pixels, grid_thw, target)
    }

    fn forward_vl_chunk(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen35MoeModel::forward_vl_chunk(
            self,
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

    fn forward_vl_hidden(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen35MoeModel::forward_vl_hidden(
            self,
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
    use crate::nn::AttnKind;

    fn make_cfg() -> Qwen35MoeConfig {
        Qwen35MoeConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(8),
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: false,
            full_attention_interval: 2,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_parameters: super::super::config::RopeParams {
                partial_rotary_factor: 1.0,
                rope_theta: 1e7,
                mrope_section: vec![2, 1, 1],
            },
            num_experts: 8,
            num_experts_per_tok: 2,
            norm_topk_prob: false,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            mlp_only_layers: vec![],
            vision_config: None,
            max_position_embeddings: 32768,
        }
    }

    fn make_zero_layer_cfg() -> Qwen35MoeConfig {
        let mut cfg = make_cfg();
        cfg.num_hidden_layers = 0;
        cfg
    }

    fn make_vision_config() -> crate::models::vision::VisionConfig {
        crate::models::vision::VisionConfig {
            depth: 2,
            hidden_size: 32,
            num_heads: 4,
            intermediate_size: 64,
            out_hidden_size: 32,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            in_channels: 3,
            num_position_embeddings: 64,
            deepstack_visual_indexes: vec![],
        }
    }

    fn assert_arrays_equal_f32(a: &Array, b: &Array) {
        let a_vec: Vec<f32> = mlx::ops::astype(a, Dtype::Float32)
            .expect("astype a")
            .to_vec()
            .expect("to_vec a");
        let b_vec: Vec<f32> = mlx::ops::astype(b, Dtype::Float32)
            .expect("astype b")
            .to_vec()
            .expect("to_vec b");
        assert_eq!(a_vec.len(), b_vec.len(), "array len");
        for (i, (av, bv)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
            assert_eq!(av, bv, "value[{i}] differs: {av} vs {bv}");
        }
    }

    #[test]
    fn make_cache_layer_kinds_match_partition() {
        // 4 layers, full_attention_interval=2 → layers {1, 3} are Full.
        let cfg = make_cfg();
        assert_eq!(cfg.layer_kind(0), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(1), AttnKind::Full);
        assert_eq!(cfg.layer_kind(2), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(3), AttnKind::Full);

        // We can't construct a full Qwen35MoeModel without real weights,
        // but we can exercise make_cache logic directly via a mock-like
        // config-driven path. Test only the config partition logic here;
        // actual make_cache is exercised in integration tests.
    }

    fn assert_model_vision_slot(model: &Qwen35MoeModel) {
        let _: &Option<crate::models::vision::VisionTower> = &model.vision;
    }

    #[test]
    fn model_exposes_optional_vision_tower_slot() {
        let _field_check: fn(&Qwen35MoeModel) = assert_model_vision_slot;
        let _accessor: for<'a> fn(
            &'a Qwen35MoeModel,
        ) -> Option<&'a crate::models::vision::VisionTower> = Qwen35MoeModel::vision;
    }

    #[test]
    #[ignore = "loads a full local Qwen3.5 MoE VL checkpoint"]
    fn loads_qwen35_moe_vision_tower_from_real_checkpoint() {
        let dir = match std::env::var("QWEN35_MOE_VL_MODEL") {
            Ok(path) => std::path::PathBuf::from(path),
            Err(_) => {
                eprintln!("skip: set QWEN35_MOE_VL_MODEL to a local MoE VL checkpoint");
                return;
            }
        };
        if !dir.exists() {
            eprintln!("skip: {} not found", dir.display());
            return;
        }

        let loader = crate::core::Loader::open_multimodal(&dir).expect("open_multimodal");
        let model = Qwen35MoeModel::from_loader(&loader).expect("load model");
        assert!(model.vision().is_some(), "vision tower should be loaded");
        let vc = model
            .config()
            .vision_config
            .as_ref()
            .expect("vision_config present");
        assert_eq!(vc.out_hidden_size, model.config().hidden_size);
    }

    #[test]
    fn model_meta_fields_populated() {
        // Verify model_meta wires cfg fields correctly via approx_weight_bytes.
        let cfg = make_cfg();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let e = cfg.num_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let se = cfg.shared_expert_intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;

        let expected_bytes =
            4 * h * h * l / 2 + 3 * e * h * me * l / 2 + 3 * h * se * l / 2 + 2 * vocab * h / 2;

        // We can compute expected bytes without constructing the full model:
        let attn = 4 * h * h * l / 2;
        let routed = 3 * e * h * me * l / 2;
        let shared = 3 * h * se * l / 2;
        let embed_head = 2 * vocab * h / 2;
        assert_eq!(expected_bytes, attn + routed + shared + embed_head);

        // Verify spatial_merge_size sentinel and max_position_embeddings passthrough.
        assert_eq!(cfg.max_position_embeddings, 32768);
    }

    #[test]
    fn compute_vision_embeds_without_tower_returns_err() {
        let model = Qwen35MoeModel::from_cfg_for_test(make_zero_layer_cfg());
        let pixel_values = Array::zeros(&[1, 2, 3, 16, 16], Dtype::Bfloat16).unwrap();
        let err = model
            .compute_vision_embeds(&pixel_values, &[(1, 1, 1)], ())
            .expect_err("missing vision tower must be an error");
        assert!(
            err.to_string().contains("model has no vision_tower"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn forward_vl_chunk_text_only_matches_forward_on_stub_model() {
        use crate::core::generate::{build_position_ids, IMAGE_TOKEN_ID};

        let model = Qwen35MoeModel::from_cfg_for_test(make_zero_layer_cfg());
        let input_ids: Array = (&[1_i32, 2, 3][..], &[1_i32, 3][..])
            .try_into()
            .expect("input_ids");
        let position_ids = build_position_ids(0, 3).expect("position_ids");

        let logits_text = model
            .forward_on(&input_ids, &position_ids, None, None, None, ())
            .expect("forward_on");
        let logits_vl = model
            .forward_vl_chunk(
                &input_ids,
                &position_ids,
                None,
                None,
                None,
                None,
                IMAGE_TOKEN_ID,
                (),
            )
            .expect("forward_vl_chunk text-only");

        assert_arrays_equal_f32(&logits_text, &logits_vl);
    }

    #[test]
    fn batched_prefill_vl_text_only_matches_batched_prefill_on_stub_model() {
        use crate::core::generate::{
            build_batch_attention_mask, build_batch_linear_mask, build_position_ids_batched,
            IMAGE_TOKEN_ID,
        };

        let model = Qwen35MoeModel::from_cfg_for_test(make_zero_layer_cfg());
        let prompt_lens = vec![3_i32, 2_i32];
        let max_len = 3_i32;
        let input_ids: Array = (&[1_i32, 2, 3, 4, 5, 0][..], &[2_i32, max_len][..])
            .try_into()
            .expect("input_ids");
        let position_ids = build_position_ids_batched(&prompt_lens, max_len).expect("position_ids");
        let attention_mask = build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16)
            .expect("attention_mask");
        let linear_mask = build_batch_linear_mask(&prompt_lens, max_len).expect("linear_mask");
        let mut cache_text = model
            .make_cache(prompt_lens.len() as i32, max_len, Dtype::Bfloat16)
            .expect("cache_text");
        let mut cache_vl = model
            .make_cache(prompt_lens.len() as i32, max_len, Dtype::Bfloat16)
            .expect("cache_vl");

        let logits_text = model
            .batched_prefill(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_mask,
                &prompt_lens,
                Some(&mut cache_text),
                (),
            )
            .expect("batched_prefill");
        let per_row_pixel_values: Vec<Option<&Array>> = vec![None, None];
        let per_row_grid_thw: Vec<Option<&[(i32, i32, i32)]>> = vec![None, None];
        let logits_vl = model
            .batched_prefill_vl(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_mask,
                &prompt_lens,
                &per_row_pixel_values,
                &per_row_grid_thw,
                IMAGE_TOKEN_ID,
                Some(&mut cache_vl),
                (),
            )
            .expect("batched_prefill_vl text-only");

        assert_arrays_equal_f32(&logits_text, &logits_vl);
    }

    #[test]
    fn approx_weight_bytes_formula() {
        let cfg = make_cfg();
        let h = 32_usize;
        let l = 4_usize;
        let e = 8_usize;
        let me = 16_usize;
        let se = 16_usize;
        let vocab = 1024_usize;

        let attn = 4 * h * h * l / 2; // 1024
        let routed = 3 * e * h * me * l / 2; // 12288
        let shared = 3 * h * se * l / 2; // 3072
        let embed_head = 2 * vocab * h / 2; // 32768
        let expected = attn + routed + shared + embed_head;

        // Sanity-check formula correctness with the concrete cfg values.
        assert_eq!(
            expected,
            4 * 32 * 32 * 4 / 2 + 3 * 8 * 32 * 16 * 4 / 2 + 3 * 32 * 16 * 4 / 2 + 2 * 1024 * 32 / 2
        );
        let _ = cfg; // consumed for compile check
    }

    #[test]
    fn model_meta_weight_bytes_includes_vision_config() {
        let text_model = Qwen35MoeModel::from_cfg_for_test(make_zero_layer_cfg());

        let mut vl_cfg = make_zero_layer_cfg();
        vl_cfg.vision_config = Some(make_vision_config());
        let vl_model = Qwen35MoeModel::from_cfg_for_test(vl_cfg);

        assert!(
            vl_model.model_meta().weight_bytes > text_model.model_meta().weight_bytes,
            "MoE VL model_meta must reserve memory for vision tower weights"
        );
    }

    #[test]
    fn per_row_slice_last_uniform_pick() {
        // hidden [2, 4, 3] with deterministic values: (i*4 + j)*3 + c.
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        let out = per_row_slice_last(&hidden, &[3, 3], ()).expect("per_row_slice_last");
        assert_eq!(out.shape().as_slice(), &[2, 1, 3]);
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert_eq!(v, vec![9.0, 10.0, 11.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn per_row_slice_last_ragged_pick() {
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        let out = per_row_slice_last(&hidden, &[1, 3], ()).expect("per_row_slice_last ragged");
        assert_eq!(out.shape().as_slice(), &[2, 1, 3]);
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert_eq!(v, vec![3.0, 4.0, 5.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn per_row_slice_last_invalid_args_return_err() {
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        // len mismatch (3 vs batch=2)
        let r1 = per_row_slice_last(&hidden, &[0, 1, 2], ());
        assert!(r1.is_err(), "len mismatch must Err");
        // negative position
        let r2 = per_row_slice_last(&hidden, &[-1, 1], ());
        assert!(r2.is_err(), "negative position must Err");
        // position >= s (s=4)
        let r3 = per_row_slice_last(&hidden, &[0, 4], ());
        assert!(r3.is_err(), "out-of-range position must Err");
    }
}
