//! MiniCPM-V-4.6 top-level model: text core + (optionally) vision encoder.
//!
//! Owns `Qwen35TextModel` directly (does NOT wrap `Qwen35Model`); the vision
//! tower (`MiniCpmV46Vision`) is loaded when the checkpoint was opened via
//! [`Loader::open_multimodal`] AND contains `vision_tower.*` keys.
//!
//! `lm_head` is `None` for MiniCPM-V-4.6 (uses `tie_word_embeddings = true`);
//! the field is kept for correctness if a future untied variant appears.
//!
//! `DenseVlMethods` is implemented in this module. The `model_from_loader`
//! facade in `mod.rs` now returns `MiniCpmV46Model` directly (P2a Task 3).

use anyhow::anyhow;
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::Loader;
use crate::nn::{AttnKind, LayerCache, Linear};
use crate::Result;

use super::config::{text_config_from_loader, MiniCpmV46VisionConfig};
use super::vision::MiniCpmV46Vision;
use crate::models::Qwen35TextModel;

/// Top-level MiniCPM-V-4.6 model: Qwen3.5 text core + optional vision tower.
pub struct MiniCpmV46Model {
    text: Qwen35TextModel,
    /// `Some` when `!tie_word_embeddings`. MiniCPM-V-4.6 ties (`= None`).
    lm_head: Option<Linear>,
    /// Vision encoder; `Some` when opened via `open_multimodal` AND
    /// `vision_tower.embeddings.patch_embedding.weight` is present.
    vision: Option<MiniCpmV46Vision>,
    /// Parsed model-level vision profile retained for mandatory transient
    /// peak estimation before scheduler graph construction.
    vision_config: Option<MiniCpmV46VisionConfig>,
    /// Tokenizer id for the per-patch image placeholder.
    image_token_id: i32,
}

// Copied from qwen3_5/model.rs — keep in sync; bug fixes here must be mirrored there (and vice versa).
/// Slice per-row last hidden states from `hidden [B, S, H]`.
///
/// For row `i`, extracts `hidden[i, last_positions[i], :]` then stacks
/// to `[B, 1, H]`. Used by [`MiniCpmV46Model::batched_prefill`] to project
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

impl MiniCpmV46Model {
    /// Production constructor.
    ///
    /// 1. Parses `text_config` → `Qwen35Config`.
    /// 2. Loads `Qwen35TextModel` (weights live under `model.*` after Loader
    ///    sanitize strips `language_model.` prefix).
    /// 3. Loads `lm_head` only when `!tie_word_embeddings`.
    /// 4. Loads `MiniCpmV46Vision` when `open_multimodal` AND vision weights
    ///    present (detected via `vision_tower.embeddings.patch_embedding.weight`).
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = text_config_from_loader(loader)?;
        let tie = cfg.tie_word_embeddings;

        let lm_head = if tie {
            None
        } else {
            Some(Linear::from_loader(loader, "lm_head")?)
        };

        let vcfg = MiniCpmV46VisionConfig::from_loader(loader).ok();

        let vision = if let Some(ref vc) = vcfg {
            if loader.contains("vision_tower.embeddings.patch_embedding.weight") {
                Some(MiniCpmV46Vision::from_loader(loader, vc)?)
            } else {
                None
            }
        } else {
            None
        };
        let image_token_id = vcfg
            .as_ref()
            .map(|v| v.image_token_id)
            .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID);

        let text = Qwen35TextModel::from_loader(loader, cfg)?;

        Ok(Self {
            text,
            lm_head,
            vision,
            vision_config: vcfg,
            image_token_id,
        })
    }

    pub fn config(&self) -> &crate::models::Qwen35Config {
        self.text.config()
    }

    /// Returns the image-placeholder token id stored in the model config.
    ///
    /// P2b callers (CLI `--image` flow) use this to avoid re-parsing the
    /// config and tokenizer separately. The field is populated from
    /// `MiniCpmV46VisionConfig::image_token_id` when present, otherwise
    /// falls back to [`crate::core::generate::IMAGE_TOKEN_ID`].
    pub fn image_token_id(&self) -> i32 {
        self.image_token_id
    }

    // Copied from qwen3_5/model.rs — keep in sync; bug fixes here must be mirrored there (and vice versa).
    /// Slice the last sequence position from `hidden [B, S, H]` and project to
    /// vocab logits `[B, 1, vocab_size]`.
    ///
    /// When `last_positions` is `Some(positions)` (length == B), each row's
    /// last real token is at column `positions[i]` — used by the right-padded
    /// `batched_prefill` path. When `None`, slices column `S - 1` uniformly.
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
        match &self.lm_head {
            Some(head) => head.forward_on(&last_hidden, target),
            None => self.text.as_output_on(&last_hidden, target),
        }
    }

    // Copied from qwen3_5/model.rs — keep in sync; bug fixes here must be mirrored there (and vice versa).
    /// Construct a per-layer cache list matching this model's hybrid topology.
    ///
    /// Mirrors `Qwen35Model::make_cache` exactly — same Full/Linear partition
    /// logic driven by `cfg.layer_kind(i)`, same GPU-perf note re: cap floor.
    /// Bug fixes must be mirrored in `qwen3_5/model.rs`.
    ///
    /// **GPU-perf note (B1-p2.3f T4):** the per-layer K/V buffer width
    /// equals the cap because `KVCache::with_step(cap)` is used for
    /// one-shot allocation (avoids grow_to + memcpy on first decode
    /// step at long context — P8a-stage6 optimization). Production
    /// callers (Scheduler main cache + admit_mid temp cache,
    /// GenerationStream cache) MUST pre-clamp their requested cap to
    /// at least `crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF`
    /// to avoid the MLX Metal kernel slow path (cap < ~256 → 100-300×
    /// decode-step slowdown on Apple Silicon — verified in T4 sweep
    /// regression against p4_http_smoke + b1_p2_3b_3 concurrent-gs test).
    ///
    /// `make_cache` does NOT apply the floor itself so unit tests that
    /// validate tight-cap overflow rejection (e.g.
    /// `b1_p2_3c_1_per_row_offset_invalid_args_return_err`) keep
    /// working unchanged.
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
}

impl crate::core::model::Model for MiniCpmV46Model {
    fn make_cache(
        &self,
        batch: i32,
        cap: i32,
        dtype: mlx::Dtype,
    ) -> crate::Result<Vec<crate::nn::LayerCache>> {
        MiniCpmV46Model::make_cache(self, batch, cap, dtype)
    }

    fn forward_on(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
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

    fn batched_prefill(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        // Embed: [B, S_max] → [B, S_max, hidden_size]
        let hidden = self.text.embed_on(input_ids, target)?;

        // Transformer + final norm with both attention masks.
        let hidden = self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            Some(attention_mask),
            Some(linear_attention_mask),
            Some(per_row_lens),
            target,
        )?;

        // Project last position per batch row to vocab logits.
        // Under right-padding, row i's last real token sits at column
        // per_row_lens[i] - 1.
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }

    fn forward_text_hidden(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        self.text.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
        let cfg = self.config();
        // MiniCPM-V-4.6 uses a 4× spatial downsample product: 2×2 VitMerger ×
        // 2×2 Merger = 16× total. The `spatial_merge_size` field is the square-root
        // equivalent used by CLI image-token-count helpers; we store 4 (= 2 × 2
        // per side) matching the merge_group constant in MiniCpmV46VisionConfig.
        // This is only consumed by P2b CLI image-token estimation; inference is
        // unaffected by this value.
        let spatial_merge_size = 4;
        crate::core::memory_budget::ModelMeta {
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

    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }

    fn vl_positions_sequential(&self) -> bool {
        true
    }
}

impl MiniCpmV46Model {
    /// Conservative weight-bytes estimate for memory budgeting.
    /// Mirrors `Qwen35Model::approx_weight_bytes`.
    fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        // 16 hidden² total (FF + attn) per layer, divide by 2 for 4-bit storage.
        let ff_attn = l * h * h * 16 / 2;
        let embed = (cfg.vocab_size as usize) * h / 2;
        ff_attn + embed
    }
}

// ---------------------------------------------------------------------------
// DenseVlMethods — SigLIP vision tower + reused cross-modal scatter
// ---------------------------------------------------------------------------

impl MiniCpmV46Model {
    /// Run SigLIP vision encoder on a list of images.
    ///
    /// Each `(pixel_values[i], (t, h, w))` → `vision.compute_vision_embeds(pixel_values[i], h, w)`
    /// → `[N_patches, 1024]`. When multiple images are provided, outputs are
    /// concatenated along axis 0 to produce a single `[N_total, 1024]` tensor.
    pub fn compute_vision_embeds(
        &self,
        pixel_values: &[Array],
        grid_thw: &[(i32, i32, i32)],
        target: impl Into<StreamOrDevice>,
    ) -> crate::Result<Array> {
        let target = target.into();
        if pixel_values.is_empty() {
            return Err(anyhow!(
                "MiniCpmV46Model::compute_vision_embeds: pixel_values cannot be empty"
            ));
        }
        if pixel_values.len() != grid_thw.len() {
            return Err(anyhow!(
                "compute_vision_embeds: pixel_values.len()={} must equal grid_thw.len()={}",
                pixel_values.len(),
                grid_thw.len()
            ));
        }
        let vision = self.vision.as_ref().ok_or_else(|| {
            anyhow!("MiniCpmV46Model has no vision tower; use Loader::open_multimodal")
        })?;
        if pixel_values.len() == 1 {
            let (_t, h, w) = grid_thw[0];
            vision.compute_vision_embeds(&pixel_values[0], h, w, target)
        } else {
            let mut embeds: Vec<Array> = Vec::with_capacity(pixel_values.len());
            for (pix, &(_t, h, w)) in pixel_values.iter().zip(grid_thw.iter()) {
                let ve = vision.compute_vision_embeds(pix, h, w, target)?;
                embeds.push(ve);
            }
            let refs: Vec<&Array> = embeds.iter().collect();
            concatenate_on(&refs, 0, target)
                .map_err(|e| anyhow!("compute_vision_embeds concatenate: {e:?}"))
        }
    }

    /// Forward a single VL prefill chunk to last-position logits.
    ///
    /// When `vision_embeds_slice` is `None`, the vision scatter step is skipped
    /// and the output is numerically identical to `forward_on`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_chunk(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> crate::Result<Array> {
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

    /// Forward one VL prefill chunk through embeddings + transformer + final
    /// norm, returning hidden states without projecting to logits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> crate::Result<Array> {
        let target = target.into();
        let mut hidden = self.text.embed_on(input_ids, target)?;
        if let Some(ve) = vision_embeds_slice {
            hidden = crate::models::qwen3_5::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                ve,
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

    /// VL-capable batched prefill over `[B, S_max]` right-padded mixed text/VL
    /// prompts. Each row independently carries per-row SigLIP pixel_values + grid_thw
    /// (vision row) or both `None` (text row).
    ///
    /// Mirrors Qwen35Model::batched_prefill_vl (SigLIP vision instead of NaViT) — keep in sync.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn batched_prefill_vl(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&[Array]>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> crate::Result<Array> {
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

        // Embed: [B, S_max] → [B, S_max, hidden]
        let mut hidden = self.text.embed_on(input_ids, target)?;

        // Per-row vision encoder calls (sequential — avoids GPU-memory contention).
        let mut all_vision_embeds: Vec<Array> = Vec::new();
        for i in 0..b {
            match (per_row_pixel_values[i], per_row_grid_thw[i]) {
                (Some(pv), Some(grids)) if !grids.is_empty() => {
                    let ve = self.compute_vision_embeds(pv, grids, target)?;
                    all_vision_embeds.push(ve);
                }
                (Some(_), None) => {
                    return Err(anyhow!(
                        "batched_prefill_vl: row {i} has pixel_values but grid_thw is None"
                    ));
                }
                _ => { /* text row or VL row with empty grids — skipped */ }
            }
        }

        // Scatter vision embeds into image_pad positions (only if any vision rows).
        if !all_vision_embeds.is_empty() {
            let vision_concat = if all_vision_embeds.len() == 1 {
                all_vision_embeds.pop().expect("len == 1")
            } else {
                let refs: Vec<&Array> = all_vision_embeds.iter().collect();
                concatenate_on(&refs, 0, target)
                    .map_err(|e| anyhow!("vision_embeds concatenate: {e:?}"))?
            };
            hidden = crate::models::qwen3_5::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                &vision_concat,
                image_token_id,
            )?;
        }

        // Transformer + final norm with both attention masks.
        let hidden = self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            Some(attention_mask),
            Some(linear_attention_mask),
            Some(per_row_lens),
            target,
        )?;

        // Per-row last-position slice + lm_head project.
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }
}

impl crate::core::scheduler::DenseVlMethods for MiniCpmV46Model {
    fn estimate_vision_prefill_peak_bytes(
        &self,
        pixel_values: &[mlx::Array],
        grid_thw: &[(i32, i32, i32)],
    ) -> crate::Result<usize> {
        anyhow::ensure!(
            pixel_values.len() == grid_thw.len(),
            "MiniCpmV46Model vision peak estimator requires pixel_values.len()={} to equal grid_thw.len()={}",
            pixel_values.len(),
            grid_thw.len()
        );
        anyhow::ensure!(
            self.vision.is_some(),
            "MiniCpmV46Model vision peak estimator requires a loaded vision tower"
        );
        let config = self.vision_config.as_ref().ok_or_else(|| {
            anyhow!("MiniCpmV46Model vision peak estimator requires vision_config")
        })?;
        let merge_h = usize::try_from(config.merge_group.0)
            .map_err(|_| anyhow!("MiniCPM vision merge height must be positive"))?;
        let merge_w = usize::try_from(config.merge_group.1)
            .map_err(|_| anyhow!("MiniCPM vision merge width must be positive"))?;
        crate::core::scheduler::estimate_transformer_vision_prefill_peak_bytes(
            pixel_values,
            grid_thw,
            crate::core::scheduler::VisionPrefillMemoryProfile {
                hidden_size: usize::try_from(config.hidden_size)
                    .map_err(|_| anyhow!("MiniCPM vision hidden_size must be positive"))?,
                intermediate_size: usize::try_from(config.intermediate_size)
                    .map_err(|_| anyhow!("MiniCPM vision intermediate_size must be positive"))?,
                num_attention_heads: usize::try_from(config.num_attention_heads)
                    .map_err(|_| anyhow!("MiniCPM vision num_attention_heads must be positive"))?,
                output_hidden_size: usize::try_from(self.config().hidden_size)
                    .map_err(|_| anyhow!("MiniCPM text hidden_size must be positive"))?,
                spatial_merge_area: merge_h.saturating_mul(merge_w),
                activation_bytes: Dtype::Bfloat16.byte_size(),
            },
        )
    }

    fn compute_vision_embeds(
        &self,
        pixel_values: &[mlx::Array],
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        MiniCpmV46Model::compute_vision_embeds(self, pixel_values, grid_thw, target)
    }

    #[allow(clippy::too_many_arguments)]
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
        MiniCpmV46Model::forward_vl_chunk(
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

    #[allow(clippy::too_many_arguments)]
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
        MiniCpmV46Model::forward_vl_hidden(
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

    #[allow(clippy::too_many_arguments)]
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
        MiniCpmV46Model::batched_prefill_vl(
            self,
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            per_row_pixel_values,
            per_row_grid_thw,
            image_token_id,
            cache,
            target,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model::Model;
    use crate::core::scheduler::DenseVlMethods;

    #[test]
    fn minicpmv46_model_implements_model_and_vl_traits() {
        fn assert_model<M: crate::core::model::Model + DenseVlMethods>() {}
        assert_model::<MiniCpmV46Model>();
    }

    /// Text-only equivalence: `forward_on` must produce byte-equal output to
    /// `forward_vl_chunk(..., vision_embeds_slice=None, ...)`.
    ///
    /// Mirrors `qwen3_6_moe::tests::text_only_vl_chunk_delegates_to_core_forward`.
    ///
    /// Run with:
    /// ```text
    /// MINICPMV46_MODEL=<path> cargo test --release -p ironmlx --lib \
    ///   minicpmv4_6::model::tests::text_only_vl_chunk_delegates_to_core_forward -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires MINICPMV46_MODEL env var pointing to a real 4-bit checkpoint"]
    fn text_only_vl_chunk_delegates_to_core_forward() {
        use crate::core::generate::{build_position_ids, IMAGE_TOKEN_ID};
        use crate::core::Loader;

        let model_dir = std::env::var("MINICPMV46_MODEL")
            .expect("MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit snapshot dir");
        let loader = Loader::open(std::path::Path::new(&model_dir)).expect("Loader::open");
        let model = MiniCpmV46Model::from_loader(&loader).expect("MiniCpmV46Model::from_loader");

        let input_ids: Array = (&[100_i32, 101, 102][..], &[1_i32, 3][..])
            .try_into()
            .expect("input_ids");
        let position_ids = build_position_ids(0, 3).expect("position_ids");

        // text-only path via Model::forward_on
        let logits_text = model
            .forward_on(
                &input_ids,
                &position_ids,
                None,
                None,
                None,
                mlx::StreamOrDevice::default(),
            )
            .expect("forward_on");

        // DenseVlMethods::forward_vl_chunk with vision_embeds_slice=None must
        // produce byte-equal output (vision scatter is fully skipped).
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

        let a: Vec<f32> = mlx::ops::astype(&logits_text, Dtype::Float32)
            .expect("astype text")
            .to_vec()
            .expect("to_vec text");
        let b: Vec<f32> = mlx::ops::astype(&logits_vl, Dtype::Float32)
            .expect("astype vl")
            .to_vec()
            .expect("to_vec vl");
        assert_eq!(
            a, b,
            "forward_on vs forward_vl_chunk(vision=None) must be byte-equal"
        );
    }
}
