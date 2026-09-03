//! Top-level Qwen3.5 model: text model + (tied or explicit) lm_head + heterogeneous cache.

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache, MtpCache};
use crate::core::Loader;
use crate::models::vision::VisionTower;
use crate::nn::{AttnKind, LayerCache, Linear, Mtp, MtpStepOutput};
use crate::Result;

use super::config::Qwen35Config;
use super::text_model::Qwen35TextModel;

/// Minimum K/V cache cap that `make_cache` allocates regardless of the
/// caller-requested cap. Empirically (B1-p2.3f T4), `KVCache::with_step(cap)`
/// with `cap < ~256` makes the bf16 attention forward run ~100-300× slower
/// on Apple Silicon (a 4B decode step grows from ~50 ms to ~10 s). The
/// most likely cause is MLX's Metal kernel tile picker missing its
/// preferred power-of-two tile for tight K/V buffer widths.
///
/// 256 is the natural floor: matches `KVCache`'s default `step`, aligns
/// with common GPU block sizes, and costs only a few MB across 32
/// 4B-bf16 layers. Larger caps (long prompts) pass through unchanged.
///
/// Applied silently inside `make_cache` so every caller (Scheduler main
/// cache, Scheduler admit_mid temp cache, GenerationStream cache, test
/// fixtures) is protected. The raise does NOT shrink user-set
/// `--max-cache-cap`: cap_max is enforced at admit time on the
/// **request size** (`prompt_len + max_new_tokens`); a request cleared
/// by the admit gate may still be backed by a slightly oversized K/V
/// buffer.
pub const MIN_KV_CACHE_CAP_FOR_GPU_PERF: i32 = 256;

/// Top-level Qwen3.5 dense model: hybrid 32-layer text core + tied/untied lm_head.
///
/// `vision` is present only when the model was loaded via [`Loader::open_multimodal`]
/// AND the config contains a `vision_config` block. Text-only inference is unaffected when
/// `vision` is `None`.
pub struct Qwen35Model {
    text: Qwen35TextModel,
    exact_batched_verify_profile: super::speculative::ExactBatchedVerifyProfile,
    /// Exact storage bytes retained by the sanitized production loader.
    /// Composition-only constructors fall back to the architecture estimate.
    loaded_weight_bytes: Option<usize>,
    /// `Some` when `!tie_word_embeddings`. `None` reuses `text.embed_tokens` for output projection.
    lm_head: Option<Linear>,
    /// Vision encoder; `Some` for VL models loaded with `open_multimodal`. `None` for text-only.
    vision: Option<VisionTower>,
}

/// Slice per-row last hidden states from `hidden [B, S, H]`.
///
/// For row `i`, extracts `hidden[i, last_positions[i], :]` then stacks
/// to `[B, 1, H]`. Used by [`Qwen35Model::batched_prefill`] to project
/// per-row last-token logits when prompts have different lengths under
/// right-padding.
///
/// # Errors
/// - `last_positions.len() != B`
/// - `last_positions[i] < 0 || last_positions[i] >= S` for any `i`
// Note: MiniCpmV46Model (minicpmv4_6/model.rs) copies this — keep the two in sync when fixing bugs here.
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

impl Qwen35Model {
    /// Production constructor. Calls [`Qwen35Config::from_loader`] then
    /// [`Qwen35TextModel::from_loader`]; loads `lm_head` only when not tied.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Qwen35Config::from_loader(loader)
            .context("parsing Qwen35Config from loader.config_raw_value")?;
        Self::from_loader_with_config_impl(loader, cfg)
    }

    pub(crate) fn from_loader_dflash2(loader: &mut Loader) -> Result<Self> {
        let loaded_weight_bytes = loader.loaded_tensor_bytes();
        let cfg = Qwen35Config::from_loader(loader)
            .context("parsing Qwen35Config from loader.config_raw_value")?;
        let exact_batched_verify_profile = super::speculative::dense_exact_batched_verify_profile(
            loader.quant_meta(),
            loader
                .config_raw_value()
                .pointer("/text_config/dtype")
                .and_then(serde_json::Value::as_str),
        );
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(Linear::from_loader(loader, "lm_head")?)
        };
        let vision = if let Some(vc) = cfg.vision_config.as_ref() {
            if loader.contains("vision_tower.patch_embed.proj.weight") {
                Some(VisionTower::from_loader(loader, vc)?)
            } else {
                None
            }
        } else {
            None
        };
        let text = Qwen35TextModel::from_loader_dflash2(loader, cfg)?;
        Ok(Self {
            text,
            exact_batched_verify_profile,
            loaded_weight_bytes: Some(loaded_weight_bytes),
            lm_head,
            vision,
        })
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen35Config) -> Result<Self> {
        Self::from_loader_with_config_impl(loader, cfg)
    }

    fn from_loader_with_config_impl(loader: &Loader, cfg: Qwen35Config) -> Result<Self> {
        let loaded_weight_bytes = loader.loaded_tensor_bytes();
        let exact_batched_verify_profile = super::speculative::dense_exact_batched_verify_profile(
            loader.quant_meta(),
            loader
                .config_raw_value()
                .pointer("/text_config/dtype")
                .and_then(serde_json::Value::as_str),
        );
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(Linear::from_loader(loader, "lm_head")?)
        };

        // Load VisionTower when vision_config is present in the model config AND the loader
        // actually has vision_tower.* tensor keys retained (i.e. opened via open_multimodal).
        // Detection strategy: use `loader.contains("vision_tower.patch_embed.proj.weight")` as a
        // lightweight sentinel rather than attempting VisionTower::from_loader and catching errors.
        // This avoids spurious error messages for text-only callers who use Loader::open (which
        // drops all vision_tower.* keys during sanitize).
        let vision = if let Some(vc) = cfg.vision_config.as_ref() {
            if loader.contains("vision_tower.patch_embed.proj.weight") {
                Some(VisionTower::from_loader(loader, vc)?)
            } else {
                None
            }
        } else {
            None
        };

        let text = Qwen35TextModel::from_loader(loader, cfg)?;
        Ok(Self {
            text,
            exact_batched_verify_profile,
            loaded_weight_bytes: Some(loaded_weight_bytes),
            lm_head,
            vision,
        })
    }

    /// Test seam.
    #[doc(hidden)]
    pub fn from_components(text: Qwen35TextModel, lm_head: Option<Linear>) -> Self {
        Self {
            text,
            exact_batched_verify_profile: super::speculative::ExactBatchedVerifyProfile::Disabled,
            loaded_weight_bytes: None,
            lm_head,
            vision: None,
        }
    }

    pub fn config(&self) -> &Qwen35Config {
        self.text.config()
    }

    pub(crate) fn vision_loaded(&self) -> bool {
        self.vision.is_some()
    }

    /// Extract memory-budget-relevant model attributes for Scheduler::new
    /// (B1-p2.5 G1).
    pub fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
        let cfg = self.config();
        let spatial_merge_size = cfg
            .vision_config
            .as_ref()
            .map(|vc| vc.spatial_merge_size)
            .unwrap_or(2);
        crate::core::memory_budget::ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: cfg.head_dim,
            weight_bytes: self
                .loaded_weight_bytes
                .unwrap_or_else(|| self.approx_weight_bytes()),
            max_position_embeddings: cfg.max_position_embeddings,
            spatial_merge_size,
        }
    }

    /// Composition-only fallback estimate for memory budgeting (B1-p2.5).
    /// Production models use the sanitized loader's exact tensor storage bytes.
    fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        // 16 hidden² total (FF + attn) per layer, divide by 2 for 4-bit storage.
        let ff_attn = l * h * h * 16 / 2;
        let embed = (cfg.vocab_size as usize) * h / 2;
        ff_attn + embed
    }

    pub fn text(&self) -> &Qwen35TextModel {
        &self.text
    }

    pub fn hidden_dtype(&self) -> Dtype {
        self.text.hidden_dtype()
    }

    pub(crate) fn supports_affine8_b4_mtp_exact_hot_path(
        &self,
        batch_width: usize,
        verify_width: usize,
    ) -> bool {
        self.exact_batched_verify_profile
            == super::speculative::ExactBatchedVerifyProfile::Affine8Dense
            && batch_width == 4
            && verify_width == 2
    }

    pub fn load_mtp_head(&self, loader: &Loader) -> Result<Mtp> {
        let mtp_text_cfg = Qwen35Config::from_mtp_loader(loader)
            .context("parsing Qwen35 MTP text_config from loader")?;
        self.config()
            .ensure_mtp_compatible(&mtp_text_cfg)
            .context("validating Qwen35 MTP compatibility")?;
        Mtp::from_loader(loader, "", mtp_text_cfg.mtp_config()?)
    }

    pub fn project_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden_shape = hidden.shape();
        let hidden_shape = hidden_shape.as_slice();
        let exact_batched_verify = crate::nn::verify_qmm::is_armed()
            && hidden_shape.len() == 3
            && crate::models::qwen3_5::speculative::exact_batched_verify_shape_qualified(
                self.exact_batched_verify_profile,
                hidden_shape[0] as usize,
                hidden_shape[1] as usize,
            );
        if exact_batched_verify {
            if hidden_shape[0] == 1 {
                return match &self.lm_head {
                    Some(head) => head.forward_positions_isolated_on(hidden, target),
                    None => {
                        let _product_stable = crate::nn::product_stable_qmm::scope();
                        self.text.as_output_on(hidden, target)
                    }
                };
            }
            return crate::models::qwen3_5::speculative::project_positions_isolated_on(
                hidden,
                target,
                |position_hidden, target| {
                    self.project_hidden_unisolated_on(position_hidden, target)
                },
            );
        }
        self.project_hidden_unisolated_on(hidden, target)
    }

    fn project_hidden_unisolated_on(
        &self,
        hidden: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        match &self.lm_head {
            Some(head) => head.forward_on(hidden, target),
            None => self.text.as_output_on(hidden, target),
        }
    }

    pub(crate) fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let shape = hidden.shape();
        if shape.as_slice().first() == Some(&4)
            && shape.as_slice().get(1) == Some(&2)
            && shape.as_slice().get(2) == Some(&self.config().hidden_size)
            && crate::nn::position_stable_qmm::exact_affine8_b4_q2_is_armed()
        {
            return self.project_hidden_unisolated_on(hidden, target);
        }
        if shape
            .as_slice()
            .get(1)
            .is_some_and(|&sequence| sequence > 1)
        {
            if shape.as_slice().first() == Some(&1) {
                return match &self.lm_head {
                    Some(head) => head.forward_positions_isolated_on(hidden, target),
                    None => {
                        let _product_stable = crate::nn::product_stable_qmm::scope();
                        self.text.as_output_on(hidden, target)
                    }
                };
            }
            return crate::models::qwen3_5::speculative::project_positions_isolated_on(
                hidden,
                target,
                |position_hidden, target| {
                    self.project_hidden_unisolated_on(position_hidden, target)
                },
            );
        }
        self.project_hidden_unisolated_on(hidden, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mtp_forward_hidden_on(
        &self,
        mtp: &Mtp,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        if mtp.config().hidden_size != self.config().hidden_size {
            return Err(anyhow!(
                "Qwen35Model::mtp_forward_hidden_on: mtp hidden_size {} != model hidden_size {}",
                mtp.config().hidden_size,
                self.config().hidden_size
            ));
        }
        let next_embeds = self.text.embed_on(next_token_ids, target)?;
        let (cos, sin) = self.text.mrope().cos_sin(position_ids)?;
        mtp.forward_on(
            hidden_states,
            &next_embeds,
            self.text.mrope(),
            &cos,
            &sin,
            mask,
            mtp_cache,
            target,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mtp_forward_on(
        &self,
        mtp: &Mtp,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        let target = target.into();
        let mtp_hidden = self.mtp_forward_hidden_on(
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )?;
        let _product_stable_projection =
            (mtp_hidden.shape().as_slice()[0] > 1).then(crate::nn::product_stable_qmm::scope);
        let logits = self.project_hidden_on(&mtp_hidden, target)?;
        Ok(MtpStepOutput {
            hidden_states: mtp_hidden,
            logits,
        })
    }

    /// Forward to last-position logits `[B, 1, vocab_size]`.
    ///
    /// Sampling only consumes the final position; computing the lm_head
    /// projection over the entire prefill sequence wastes ~`(S-1)/S` of the
    /// projection work (vocab=151936 in Qwen3.5 — the largest matmul in the
    /// graph). Slice the last hidden state before the projection so the
    /// per-forward lm_head cost is constant in `S`.
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
        let input_shape = input_ids.shape();
        let input_shape = input_shape.as_slice();
        let batch_stable_affine8_q1 = self.exact_batched_verify_profile
            == super::speculative::ExactBatchedVerifyProfile::Affine8Dense
            && input_shape.len() == 2
            && input_shape[0] > 1
            && input_shape[1] == 1;
        // MLX affine8 QMM can select a different accumulation morphology when
        // the leading product changes from B1 to B>1, even though every row is
        // independent. Preserve the B1 accumulation tree for ordinary
        // concurrent Q=1 decode while retaining one multi-row dispatch.
        let _product_stable_qmm =
            batch_stable_affine8_q1.then(crate::nn::product_stable_qmm::scope);
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

    /// Multimodal forward: routes `pixel_values` through the vision tower, replaces
    /// image-token positions in the text embeddings, then runs the full text backbone.
    ///
    /// When `pixel_values` is `None` the output is **numerically identical** to
    /// [`forward_on`] — the same embed → layers → norm → slice → project path.
    ///
    /// Run transformer + lm_head on pre-built `inputs_embeds [B, S, hidden]`.
    ///
    /// Bypasses embed_tokens and vision tower. Used in integration tests to
    /// isolate LM accuracy from vision tower accuracy.
    #[doc(hidden)]
    pub fn forward_from_embeds(
        &self,
        inputs_embeds: &Array,
        position_ids: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.text.forward_post_embedding_on(
            inputs_embeds,
            position_ids,
            None,
            None,
            None,
            None,
            target,
        )?;
        self.slice_last_and_project(&hidden, None, target)
    }

    /// Run only the vision tower; returns the post-merger embeddings
    /// `[N_total_patches / spatial_merge_size^2, hidden]` ready to be
    /// scattered into the LM embedding stream by
    /// [`cross_modal::replace_image_tokens`] (or its chunked equivalent).
    ///
    /// Split out from `forward_vl` so callers that drive multi-chunk
    /// prefill (see `core::generate::GenerationStream`) can run the
    /// vision tower once and reuse the embeddings across chunks.
    ///
    /// # Arguments
    /// - `pixel_values` — per-image `[N, T, C, H, W]` pre-processed patches.
    /// - `grid_thw`     — per-image `(temporal, height, width)`; must be
    ///   non-empty and sum to `N` along T·H·W.
    /// - `target`       — compute device / stream.
    pub fn compute_vision_embeds(
        &self,
        pixel_values: &[Array],
        grid_thw: &[(i32, i32, i32)],
        _target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        if pixel_values.is_empty() {
            return Err(anyhow!(
                "compute_vision_embeds: pixel_values cannot be empty"
            ));
        }
        if pixel_values.len() != grid_thw.len() {
            return Err(anyhow!(
                "compute_vision_embeds: pixel_values.len()={} must equal grid_thw.len()={}",
                pixel_values.len(),
                grid_thw.len()
            ));
        }
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("model has no vision_tower; use Loader::open_multimodal"))?;
        if pixel_values.len() == 1 {
            vision.forward(&pixel_values[0], grid_thw)
        } else {
            let refs: Vec<&Array> = pixel_values.iter().collect();
            let merged = mlx::ops::concatenate(&refs, 0)
                .map_err(|e| anyhow!("compute_vision_embeds pixel_values concatenate: {e:?}"))?;
            vision.forward(&merged, grid_thw)
        }
    }

    /// Forward a single chunk of a VL prefill. Expects the caller has
    /// pre-computed `vision_embeds_slice` for the `k_i` `<|image_pad|>`
    /// occurrences in this chunk's `input_ids`. Pass `None` if the chunk
    /// contains no image tokens (pure-text segment of a VL prompt).
    ///
    /// Compared to `forward_vl`, this method:
    /// - Does **not** run the vision tower.
    /// - Skips the scatter step entirely when
    ///   `vision_embeds_slice.is_none()`, falling back to the text-only
    ///   embedding path.
    ///
    /// # Invariants
    /// - When `vision_embeds_slice.is_some()`, its row count must equal
    ///   the number of `image_token_id` occurrences in `input_ids`.
    ///   `cross_modal::replace_image_tokens` enforces this.
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

    /// Forward one VL prefill chunk through embeddings + transformer + final
    /// norm, returning hidden states without projecting to logits.
    ///
    /// This is used by scheduler prefix prefill paths where the chunk only
    /// needs to populate KV / recurrent cache; the final token chunk performs
    /// the vocabulary projection.
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
        if let Some(ve) = vision_embeds_slice {
            hidden =
                super::cross_modal::replace_image_tokens(&hidden, input_ids, ve, image_token_id)?;
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

    /// # Arguments
    /// - `input_ids`      — `[B, S]` int32 token ids (B must be 1 for P6).
    /// - `position_ids`   — `[3, B, S]` int32 per Mrope contract.
    /// - `cache`          — optional per-layer cache slice.
    /// - `pixel_values`   — per-image pre-processed image patches `[N, T, C, H, W]`.
    /// - `grid_thw`       — per-image `(temporal, height, width)` grid sizes;
    ///   **required** when `pixel_values.is_some()`.
    /// - `image_token_id` — tokenizer id of the per-patch image placeholder
    ///   (e.g. `<|image_pad|>` = 248056 for Qwen3.5-VL).
    /// - `target`         — compute device / stream.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        pixel_values: Option<&[Array]>,
        grid_thw: Option<&[(i32, i32, i32)]>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        let vision_embeds = match (pixel_values, grid_thw) {
            (Some(pv), Some(g)) => Some(self.compute_vision_embeds(pv, g, target)?),
            (Some(_), None) => {
                return Err(anyhow!("grid_thw required when pixel_values is provided"));
            }
            (None, _) => None,
        };

        self.forward_vl_chunk(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds.as_ref(),
            image_token_id,
            target,
        )
    }

    /// Static batched prefill — runs one transformer forward across B prompts
    /// packed right-padded into `input_ids[B, S_max]`. Returns last-position
    /// logits `[B, 1, vocab]`.
    ///
    /// Phase 1 of B1-p2 (multi-request batched serving). Pure text — for VL
    /// B>1 see B1-p2.4. The caller is responsible for:
    ///   1. Right-padding each prompt to `S_max` with any pad-token id (real
    ///      tokens at columns `[0..L_i)`, pad at columns `[L_i..S_max)`). The
    ///      attention mask zeroes out pad positions regardless of which id is
    ///      used; choosing a real token id is fine.
    ///   2. Building `position_ids` via [`build_position_ids_batched`] so the
    ///      real region runs `0..L_i-1` at columns `[0..L_i)` and the pad
    ///      region is 0 at columns `[L_i..S_max)`.
    ///   3. Building `attention_mask` via [`build_batch_attention_mask`] —
    ///      the SDPA-style `[B, 1, T_q, T_kv]` additive mask consumed by the
    ///      full-attention layers.
    ///   4. Building `linear_attention_mask` via [`build_batch_linear_mask`]
    ///      — the `[B, T]` boolean per-token validity mask consumed by the
    ///      hybrid model's linear-attention layers (`GatedDeltaNet`).
    ///   5. Allocating `cache` with [`Self::make_cache`] using `batch = B`.
    ///
    /// The two masks have incompatible shapes and dtypes because the
    /// underlying attention paths are fundamentally different (SDPA with
    /// additive scores vs gated-delta-step kernel with per-token compute
    /// guards). They cannot be unified.
    ///
    /// Numerical contract: for batch row `i`, the last-position logits
    /// `out[i, :]` should match `forward_on(prompt_i)` to within
    /// `max_abs_diff < 1e-3`, and the greedy argmax must be bit-identical.
    /// The KV cache row `i` must match the state a per-stream `forward_on`
    /// would have written (verified by `tests/batched_prefill.rs`).
    ///
    /// [`build_position_ids_batched`]: crate::core::generate::build_position_ids_batched
    /// [`build_batch_attention_mask`]: crate::core::generate::build_batch_attention_mask
    /// [`build_batch_linear_mask`]: crate::core::generate::build_batch_linear_mask
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

        // Embed: [B, S_max] → [B, S_max, hidden_size]
        let hidden = self.text.embed_on(input_ids, target)?;

        // Transformer + final norm with both attention masks routed to
        // their respective attention paths inside DecoderLayer.
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
        // prompt_lens[i] - 1 — build that vector and let
        // slice_last_and_project per-row slice + concatenate.
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }

    /// VL-capable batched prefill. Single transformer forward over `[B, S_max]`
    /// right-padded mixed text/VL prompts. Each row independently carries
    /// `pixel_values + grid_thw` (vision row) or both `None` (text row).
    ///
    /// Vision encoder is run **per-row** (sequential) inside this function;
    /// the resulting `vision_embeds_i` are concatenated along axis 0 and
    /// scattered into `image_pad` positions across the whole batch via
    /// [`cross_modal::replace_image_tokens`]. Per-row concat ordering must
    /// match row-major scan of `input_ids` — guaranteed by iterating slots
    /// in row order both in the vision-embed collection loop and in the
    /// downstream scatter.
    ///
    /// Returns `[B, 1, vocab]` last-position logits (per-row, sliced via
    /// `slice_last_and_project` with `last_positions = per_row_lens - 1`).
    ///
    /// Numerical contract: for batch row `i`, `out[i, :]` matches
    /// `forward_vl(prompt_i_alone)` to within `max_abs_diff < 1e-3` and the
    /// greedy argmax is bit-identical. Verified by integration scenarios in
    /// the unit tests below (text-only equivalence vs `batched_prefill`,
    /// and B=1 equivalence vs `forward_vl`).
    ///
    /// Note: `MiniCpmV46Model::batched_prefill_vl` mirrors this (SigLIP vision
    /// instead of NaViT) — keep the two in sync when fixing bugs here.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn batched_prefill_vl(
        &self,
        input_ids: &Array,                         // [B, S_max] right-padded
        position_ids: &Array,                      // [3, B, S_max] MRoPE
        attention_mask: &Array,                    // [B, 1, S_max, S_max] additive bf16
        linear_attention_mask: &Array,             // [B, S_max] bool
        per_row_lens: &[i32],                      // real prompt lens
        per_row_pixel_values: &[Option<&[Array]>], // None for text rows
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

        // Embed: [B, S_max] → [B, S_max, hidden]
        let mut hidden = self.text.embed_on(input_ids, target)?;

        // Per-row vision encoder calls (sequential — see NG1).
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

        // Scatter vision embeds into image_pad positions (only if any).
        if !all_vision_embeds.is_empty() {
            let vision_concat = if all_vision_embeds.len() == 1 {
                all_vision_embeds.pop().expect("len == 1")
            } else {
                let refs: Vec<&Array> = all_vision_embeds.iter().collect();
                mlx::ops::concatenate(&refs, 0)
                    .map_err(|e| anyhow!("vision_embeds concatenate: {e:?}"))?
            };
            hidden = super::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                &vision_concat,
                image_token_id,
            )?;
        }

        // Transformer + final norm (same path as batched_prefill).
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

    /// Slice the last sequence position from `hidden [B, S, H]` and project to
    /// vocab logits `[B, 1, vocab_size]`. Shared by [`forward_on`] and [`forward_vl`].
    ///
    /// When `last_positions` is `Some(positions)` (length == B), each row's
    /// last real token is at column `positions[i]` — used by the right-padded
    /// `batched_prefill` path where rows have ragged real lengths. The
    /// function per-row slices `hidden[i, positions[i], :]` and concatenates
    /// along axis 0 to produce `[B, 1, H]`.
    ///
    /// When `last_positions` is `None` (single-stream `forward_on` and VL
    /// chunk callers), the fallback slices column `S - 1` for every row —
    /// behaviourally equivalent for B=1 or uniform-length inputs.
    // Note: MiniCpmV46Model (minicpmv4_6/model.rs) copies this — keep the two in sync when fixing bugs here.
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
        self.project_hidden_on(&last_hidden, target)
    }

    /// Construct a per-layer cache list matching this model's hybrid topology.
    ///
    /// **GPU-perf note (B1-p2.3f T4):** the per-layer K/V buffer width
    /// equals the cap because `KVCache::with_step(cap)` is used for
    /// one-shot allocation (avoids grow_to + memcpy on first decode
    /// step at long context — P8a-stage6 optimization). Production
    /// callers (Scheduler main cache + admit_mid temp cache,
    /// GenerationStream cache) MUST pre-clamp their requested cap to
    /// at least `MIN_KV_CACHE_CAP_FOR_GPU_PERF` to avoid the MLX
    /// Metal kernel slow path (cap < ~256 → 100-300× decode-step
    /// slowdown on Apple Silicon — verified in T4 sweep regression
    /// against p4_http_smoke + b1_p2_3b_3 concurrent-gs test).
    ///
    /// `make_cache` does NOT apply the floor itself so unit tests that
    /// validate tight-cap overflow rejection (e.g.
    /// `b1_p2_3c_1_per_row_offset_invalid_args_return_err`) keep
    /// working unchanged.
    // Note: MiniCpmV46Model (minicpmv4_6/model.rs) copies this — keep the two in sync when fixing bugs here.
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

    /// Test-only stub: constructs a Qwen35Model whose `text` field is unsuitable
    /// for forward (the layers vec is empty, embeddings are stubs) but whose
    /// `make_cache` is fully driven by `cfg`. Used only by tests in this
    /// module to verify cache-partition behavior without synthesizing weights.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn from_cfg_for_test(cfg: Qwen35Config) -> Self {
        let mrope = crate::nn::Mrope::new(
            cfg.effective_head_dim(),
            cfg.rope_parameters.rope_theta,
            cfg.rope_parameters.partial_rotary_factor,
            &cfg.rope_parameters.mrope_section,
            true,
        )
        .expect("Mrope::new with valid cfg");
        let h = cfg.hidden_size;
        let stub_embed = crate::nn::Embedding::from_components_fp_for_test(
            mlx::Array::zeros((cfg.vocab_size, h), mlx::Dtype::Bfloat16).unwrap(),
        );
        let stub_norm = crate::nn::RmsNorm::new(
            mlx::ops::constructors::ones((h,), mlx::Dtype::Float32).unwrap(),
            cfg.rms_norm_eps,
        );
        let text = Qwen35TextModel::from_components(stub_embed, Vec::new(), stub_norm, mrope, cfg);
        Self {
            text,
            exact_batched_verify_profile: super::speculative::ExactBatchedVerifyProfile::Disabled,
            loaded_weight_bytes: None,
            lm_head: None,
            vision: None,
        }
    }
}

impl crate::core::model::Model for Qwen35Model {
    fn make_cache(
        &self,
        batch: i32,
        cap: i32,
        dtype: mlx::Dtype,
    ) -> crate::Result<Vec<crate::nn::LayerCache>> {
        Qwen35Model::make_cache(self, batch, cap, dtype)
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
        Qwen35Model::forward_on(
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
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen35Model::batched_prefill(
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

    fn requires_split_batched_prefill_for_token_parity(&self) -> bool {
        true
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
        let input_shape = input_ids.shape();
        let input_shape = input_shape.as_slice();
        let exact_batched_verify = crate::nn::verify_qmm::is_armed()
            && input_shape.len() == 2
            && crate::models::qwen3_5::speculative::exact_batched_verify_shape_qualified(
                self.exact_batched_verify_profile,
                input_shape[0] as usize,
                input_shape[1] as usize,
            );
        let _position_stable_qmm = exact_batched_verify.then(crate::nn::position_stable_qmm::scope);
        self.text().forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn project_hidden_on(
        &self,
        hidden: &mlx::Array,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen35Model::project_hidden_on(self, hidden, target)
    }

    fn supports_exact_batched_speculative_verify(
        &self,
        batch_width: usize,
        context_tokens: usize,
        verify_width: usize,
    ) -> bool {
        super::speculative::exact_batched_verify_qualified(
            self.exact_batched_verify_profile,
            batch_width,
            context_tokens,
            verify_width,
        )
    }

    fn supports_sequential_prompt_lookup_verify(
        &self,
        _batch_width: usize,
        context_tokens: usize,
        _verify_width: usize,
    ) -> bool {
        super::speculative::sequential_prompt_lookup_verify_qualified(
            self.exact_batched_verify_profile,
            context_tokens,
        )
    }

    fn max_prompt_lookup_draft_tokens(&self, configured_max_draft_tokens: usize) -> usize {
        super::speculative::prompt_lookup_max_draft_tokens(
            self.exact_batched_verify_profile,
            configured_max_draft_tokens,
        )
    }

    fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
        Qwen35Model::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }
}

impl crate::models::dflash2::DFlash2Target for Qwen35Model {
    fn dflash2_target_cache_cost(&self) -> crate::models::dflash2::DFlash2TargetCacheCost {
        qwen35_dflash2_target_cache_cost(self.config())
    }

    fn dflash2_embed_on(
        &self,
        input_ids: &mlx::Array,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        self.text.embed_on(input_ids, target)
    }

    fn dflash2_forward_target_on(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target_layer_ids: &[usize],
        mode: crate::models::dflash2::DFlash2TargetForwardMode,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<crate::models::dflash2::DFlash2TargetOutput> {
        let input_shape = input_ids.shape();
        let input_dims = input_shape.as_slice();
        let batch_width = input_dims.first().copied().unwrap_or(0) as usize;
        let verify_width = input_dims.get(1).copied().unwrap_or(0) as usize;
        let is_verify = mode.is_verify();
        let _batch_stable_prefill =
            (mode == crate::models::dflash2::DFlash2TargetForwardMode::Prefill && batch_width > 1)
                .then(crate::nn::batch_stable_qmm::linear_scope);
        // Cache-bearing attention and GatedDelta prefill must retain the
        // scheduler B1 row morphology for affine4 and affine8. Their B=N state
        // is numerically close but not exact, and the difference is amplified
        // by subsequent tensor verification.
        let _batch_stable_prefill_state =
            (mode == crate::models::dflash2::DFlash2TargetForwardMode::Prefill && batch_width > 1)
                .then(crate::nn::batch_stable_qmm::context_scope);
        if is_verify
            && verify_width > 1
            && !crate::models::qwen3_5::speculative::dflash2_exact_batched_verify_shape_qualified(
                self.exact_batched_verify_profile,
                batch_width,
                verify_width,
            )
        {
            return Err(anyhow!(
                "Qwen35 DFlash2 exact verify is not qualified for shape {:?}",
                input_dims
            ));
        }
        // Exact DFlash2 verification must preserve the ordinary decode Q=1
        // accumulation tree. The speculative verify QMM candidate is
        // intentionally not armed here; the position- and product-stable
        // routes below provide that contract for the batched verify block.
        let _position_stable_linear = mode
            .requires_position_stability()
            .then(crate::nn::position_stable_linear::scope);
        let _position_stable_qmm = mode
            .requires_position_stability()
            .then(crate::nn::position_stable_qmm::scope);
        let _dflash2_bulk_attention = mode
            .requires_position_stability()
            .then(crate::nn::position_stable_qmm::dflash2_bulk_attention_scope);
        let affine8_wide_tiling = mode.is_verify()
            && verify_width == 3
            && self.exact_batched_verify_profile
                == super::speculative::ExactBatchedVerifyProfile::Affine8Dense;
        let _affine8_wide_qmm =
            affine8_wide_tiling.then(crate::nn::product_stable_qmm::affine8_wide_scope);
        let _product_stable_qmm = (mode.is_verify() && verify_width > 1 && !affine8_wide_tiling)
            .then(crate::nn::product_stable_qmm::scope);
        // Both verify modes keep the whole block batched. Product-stable QMM
        // retains the ordinary Q=1 affine accumulation tree; bulk attention
        // preserves its position-sensitive cache updates, and GatedDelta
        // captures enough replay state to restore an exact accepted prefix.
        // This avoids replaying all decoder layers or quantized projections
        // once per verify token without changing target logits.
        let (hidden, context_hidden) = self.text.forward_with_dflash2_taps_on(
            input_ids,
            position_ids,
            cache,
            target_layer_ids,
            target,
        )?;
        Ok(crate::models::dflash2::DFlash2TargetOutput {
            hidden,
            context_hidden,
        })
    }

    fn dflash2_project_hidden_on(
        &self,
        hidden: &mlx::Array,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 || dims[0] <= 0 {
            return Err(anyhow!(
                "Qwen35 DFlash2 target projection requires [B,S,H] with B>0, got {dims:?}"
            ));
        }
        if dims[1] == 1 {
            return self.project_hidden_unisolated_on(hidden, target);
        }
        // The product-stable affine kernel evaluates all verify positions in
        // one dispatch while retaining the exact Q=1 accumulation tree. Q3
        // affine8 uses its explicit four-vector tiling hint; every other
        // product-stable projection remains on the default tile.
        let affine8_wide_tiling = dims[1] == 3
            && self.exact_batched_verify_profile
                == super::speculative::ExactBatchedVerifyProfile::Affine8Dense;
        let _affine8_wide_qmm =
            affine8_wide_tiling.then(crate::nn::product_stable_qmm::affine8_wide_scope);
        let _product_stable = (!affine8_wide_tiling).then(crate::nn::product_stable_qmm::scope);
        self.project_hidden_unisolated_on(hidden, target)
    }

    fn dflash2_restore_target_prefix_on(
        &self,
        cache: &mut [crate::nn::LayerCache],
        snapshots: &[crate::nn::LayerCacheSnapshot],
        accepted_len: usize,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<()> {
        self.text
            .restore_dflash2_speculative_prefix_on(cache, snapshots, accepted_len, target)
    }

    fn dflash2_restore_target_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[crate::nn::LayerCacheSnapshot],
        accepted_lens: &[usize],
        target: StreamOrDevice,
    ) -> crate::Result<()> {
        self.text.restore_dflash2_speculative_prefix_rows_on(
            cache,
            snapshots,
            accepted_lens,
            target,
        )
    }
}

fn qwen35_dflash2_target_cache_cost(
    cfg: &Qwen35Config,
) -> crate::models::dflash2::DFlash2TargetCacheCost {
    let layer_count = usize::try_from(cfg.num_hidden_layers)
        .expect("validated Qwen3.5 layer count must be positive");
    let full_attention_interval = usize::try_from(cfg.full_attention_interval)
        .expect("validated Qwen3.5 full-attention interval must be positive");
    let full_attention_layers = (1..=layer_count)
        .filter(|layer| layer % full_attention_interval == 0)
        .count();
    let linear_attention_layers = layer_count.saturating_sub(full_attention_layers);
    let kv_heads = usize::try_from(cfg.num_key_value_heads)
        .expect("validated Qwen3.5 KV head count must be positive");
    let head_dim = usize::try_from(cfg.effective_head_dim())
        .expect("validated Qwen3.5 head dimension must be positive");
    let bytes_per_token = full_attention_layers
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(2);

    let value_heads = usize::try_from(cfg.linear_num_value_heads)
        .expect("validated Qwen3.5 linear value head count must be positive");
    let key_heads = usize::try_from(cfg.linear_num_key_heads)
        .expect("validated Qwen3.5 linear key head count must be positive");
    let key_dim = usize::try_from(cfg.linear_key_head_dim)
        .expect("validated Qwen3.5 linear key dimension must be positive");
    let value_dim = usize::try_from(cfg.linear_value_head_dim)
        .expect("validated Qwen3.5 linear value dimension must be positive");
    let kernel_dim = usize::try_from(cfg.linear_conv_kernel_dim)
        .expect("validated Qwen3.5 linear convolution width must be positive");
    let conv_dim = key_dim
        .saturating_mul(key_heads)
        .saturating_mul(2)
        .saturating_add(value_dim.saturating_mul(value_heads));
    let conv_state_bytes = kernel_dim
        .saturating_sub(1)
        .saturating_mul(conv_dim)
        .saturating_mul(2);
    let recurrent_state_bytes = value_heads
        .saturating_mul(value_dim)
        .saturating_mul(key_dim)
        .saturating_mul(4);
    let fixed_bytes_per_sequence = linear_attention_layers
        .saturating_mul(conv_state_bytes.saturating_add(recurrent_state_bytes));

    crate::models::dflash2::DFlash2TargetCacheCost {
        bytes_per_token,
        fixed_bytes_per_sequence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::AttnKind;
    use mlx::{Array, Dtype};
    use serial_test::serial;

    fn make_cfg() -> Qwen35Config {
        // 4 layers, full_attention_interval=2 → layers {1, 3} are Full.
        Qwen35Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(8),
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: true,
            full_attention_interval: 2,
            mtp_num_hidden_layers: 1,
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
            vision_config: None,
            max_position_embeddings: 32768,
        }
    }

    #[test]
    fn dflash2_cache_cost_separates_full_attention_growth_from_linear_state() {
        let cost = qwen35_dflash2_target_cache_cost(&make_cfg());

        assert_eq!(cost.bytes_per_token, 128);
        assert_eq!(cost.fixed_bytes_per_sequence, 2_816);
        assert_eq!(cost.request_bytes(10), 4_096);
    }

    #[test]
    fn make_cache_layer_kinds_match_partition() {
        let cfg = make_cfg();
        // Verify partition logic on the config alone first.
        assert_eq!(cfg.layer_kind(0), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(1), AttnKind::Full);
        assert_eq!(cfg.layer_kind(2), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(3), AttnKind::Full);

        let model = Qwen35Model::from_cfg_for_test(cfg);
        let cache = model
            .make_cache(/* batch */ 1, /* cap */ 16, Dtype::Bfloat16)
            .unwrap();
        assert_eq!(cache.len(), 4);
        assert!(
            matches!(cache[0], LayerCache::Linear(_)),
            "layer 0 should be Linear"
        );
        assert!(
            matches!(cache[1], LayerCache::Full(_)),
            "layer 1 should be Full"
        );
        assert!(
            matches!(cache[2], LayerCache::Linear(_)),
            "layer 2 should be Linear"
        );
        assert!(
            matches!(cache[3], LayerCache::Full(_)),
            "layer 3 should be Full"
        );
    }

    #[test]
    fn model_meta_uses_exact_loaded_tensor_bytes() {
        let mut model = Qwen35Model::from_cfg_for_test(make_cfg());
        let fallback = model.model_meta().weight_bytes;
        let affine8_storage_bytes = fallback * 2;

        model.loaded_weight_bytes = Some(affine8_storage_bytes);

        assert_eq!(model.model_meta().weight_bytes, affine8_storage_bytes);
    }

    #[test]
    #[serial(mlx_metal)]
    fn mtp_forward_projects_every_mtp_position() {
        use crate::core::generate::build_position_ids;
        use crate::nn::{DecoderLayerConfig, Linear, Mtp, MtpConfig, RmsNorm};

        let cfg = make_cfg();
        let model = Qwen35Model::from_cfg_for_test(cfg.clone());
        let layer_cfg = DecoderLayerConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.effective_head_dim(),
            rms_norm_eps: cfg.rms_norm_eps,
            attention_bias: cfg.attention_bias,
            linear_num_value_heads: cfg.linear_num_value_heads,
            linear_num_key_heads: cfg.linear_num_key_heads,
            linear_key_head_dim: cfg.linear_key_head_dim,
            linear_value_head_dim: cfg.linear_value_head_dim,
            linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
        };
        let mtp_cfg = MtpConfig {
            hidden_size: cfg.hidden_size,
            num_mtp_layers: 0,
            layer: layer_cfg,
        };
        let ones = mlx::ops::constructors::ones((cfg.hidden_size,), Dtype::Float32).unwrap();
        let fc_weight =
            mlx::Array::zeros((cfg.hidden_size, 2 * cfg.hidden_size), Dtype::Float32).unwrap();
        let mtp = Mtp::from_components(
            RmsNorm::new(ones.clone(), cfg.rms_norm_eps),
            RmsNorm::new(ones.clone(), cfg.rms_norm_eps),
            Linear::new_fp(fc_weight, None),
            Vec::new(),
            RmsNorm::new(ones, cfg.rms_norm_eps),
            mtp_cfg,
        );

        let hidden = mlx::Array::zeros((1, 2, cfg.hidden_size), Dtype::Float32).unwrap();
        let next_token_ids: mlx::Array = (&[1_i32, 2][..], &[1_i32, 2][..]).try_into().unwrap();
        let position_ids = build_position_ids(0, 2).expect("position ids");

        let out = model
            .mtp_forward_on(
                &mtp,
                &hidden,
                &next_token_ids,
                &position_ids,
                None,
                None,
                (),
            )
            .expect("mtp forward");

        assert_eq!(
            out.hidden_states.shape().as_slice(),
            &[1, 2, cfg.hidden_size]
        );
        assert_eq!(out.logits.shape().as_slice(), &[1, 2, cfg.vocab_size]);
    }

    fn qwen35_checkpoint(env_name: &str) -> Option<std::path::PathBuf> {
        let path = match std::env::var(env_name) {
            Ok(path) => std::path::PathBuf::from(path),
            Err(_) => {
                eprintln!("skip: set {env_name} to a local Qwen3.5 checkpoint");
                return None;
            }
        };
        if !path.exists() {
            eprintln!("skip: {} not found", path.display());
            return None;
        }
        Some(path)
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 target checkpoint"]
    #[serial(mlx_metal)]
    fn qwen38_dflash2_verify_matches_ordinary_q1_logits_exactly() {
        use crate::core::generate::build_position_ids;
        use crate::core::Model;
        use crate::models::dflash2::{DFlash2Target, DFlash2TargetForwardMode};

        let Some(model_dir) = qwen35_checkpoint("QWEN38_MODEL") else {
            return;
        };
        let mut loader = crate::core::Loader::open(&model_dir).expect("open Qwen3.8 model");
        let model = Qwen35Model::from_loader_dflash2(&mut loader)
            .expect("load DFlash2-optimized Qwen3.8 model");
        let cap = 32;
        let mut ordinary_cache = model
            .make_cache(1, cap, model.cache_dtype())
            .expect("ordinary cache");
        let mut ordinary_prefix_cache = model
            .make_cache(1, cap, model.cache_dtype())
            .expect("ordinary prefix cache");
        let mut dflash_cache = model
            .make_cache(1, cap, model.cache_dtype())
            .expect("DFlash2 cache");
        let mut sampled_dflash_cache = model
            .make_cache(1, cap, model.cache_dtype())
            .expect("sampled DFlash2 cache");
        let target_layers = [5_usize, 19, 33, 47, 61];

        let prefill: Array = (&[100_u32, 200, 300][..], &[1_i32, 3][..])
            .try_into()
            .expect("prefill ids");
        let prefill_positions = build_position_ids(0, 3).expect("prefill positions");
        let ordinary_first = model
            .forward_on(
                &prefill,
                &prefill_positions,
                None,
                None,
                Some(&mut ordinary_cache),
                (),
            )
            .expect("ordinary prefill");
        let ordinary_prefix_first = model
            .forward_on(
                &prefill,
                &prefill_positions,
                None,
                None,
                Some(&mut ordinary_prefix_cache),
                (),
            )
            .expect("ordinary prefix prefill");
        let dflash_prefill = model
            .dflash2_forward_target_on(
                &prefill,
                &prefill_positions,
                Some(&mut dflash_cache),
                &target_layers,
                DFlash2TargetForwardMode::Prefill,
                ().into(),
            )
            .expect("DFlash2 prefill");
        let sampled_dflash_prefill = model
            .dflash2_forward_target_on(
                &prefill,
                &prefill_positions,
                Some(&mut sampled_dflash_cache),
                &target_layers,
                DFlash2TargetForwardMode::Prefill,
                ().into(),
            )
            .expect("sampled DFlash2 prefill");
        let dflash_last_hidden = mlx::ops::indexing::slice_strided(
            &dflash_prefill.hidden,
            &[0_i32, 2, 0][..],
            &[1_i32, 3, model.config().hidden_size][..],
            &[1_i32, 1, 1][..],
        )
        .expect("slice DFlash2 last prefill hidden");
        let dflash_first = model
            .dflash2_project_hidden_on(&dflash_last_hidden, ().into())
            .expect("DFlash2 first logits");
        let sampled_dflash_last_hidden = mlx::ops::indexing::slice_strided(
            &sampled_dflash_prefill.hidden,
            &[0_i32, 2, 0][..],
            &[1_i32, 3, model.config().hidden_size][..],
            &[1_i32, 1, 1][..],
        )
        .expect("slice sampled DFlash2 last prefill hidden");
        let sampled_dflash_first = model
            .dflash2_project_hidden_on(&sampled_dflash_last_hidden, ().into())
            .expect("sampled DFlash2 first logits");
        mlx::transforms::eval(&[
            &ordinary_first,
            &ordinary_prefix_first,
            &dflash_first,
            &sampled_dflash_first,
        ])
        .expect("evaluate prefill logits");
        assert_array_exact(&ordinary_first, &ordinary_prefix_first, "ordinary prefill");
        assert_array_exact(&ordinary_first, &dflash_first, "DFlash2 prefill");
        assert_array_exact(
            &ordinary_first,
            &sampled_dflash_first,
            "sampled DFlash2 prefill",
        );

        let verify_ids = [400_u32, 500, 600, 700];
        let mut ordinary_logits = Vec::with_capacity(verify_ids.len());
        for (index, &token) in verify_ids.iter().enumerate() {
            let input: Array = (&[token][..], &[1_i32, 1][..])
                .try_into()
                .expect("ordinary verify input");
            let positions = build_position_ids(3 + index as i32, 1).expect("verify position");
            let logits = model
                .forward_on(
                    &input,
                    &positions,
                    None,
                    None,
                    Some(&mut ordinary_cache),
                    (),
                )
                .expect("ordinary Q=1 verify");
            mlx::transforms::eval(&[&logits]).expect("evaluate ordinary Q=1 logits");
            ordinary_logits.push(logits);
        }
        let ordinary_refs = ordinary_logits.iter().collect::<Vec<_>>();
        let ordinary_logits = mlx::ops::shape::concatenate(&ordinary_refs, 1)
            .expect("concatenate ordinary verify logits");

        let dflash_snapshots = dflash_cache
            .iter()
            .map(crate::nn::LayerCache::snapshot)
            .collect::<Vec<_>>();
        for layer in &mut dflash_cache {
            layer
                .begin_speculative_prefix_capture()
                .expect("begin DFlash2 prefix capture");
        }
        let verify: Array = (&verify_ids[..], &[1_i32, verify_ids.len() as i32][..])
            .try_into()
            .expect("DFlash2 verify ids");
        let verify_positions =
            build_position_ids(3, verify_ids.len() as i32).expect("DFlash2 verify positions");
        let dflash_verify = model
            .dflash2_forward_target_on(
                &verify,
                &verify_positions,
                Some(&mut dflash_cache),
                &target_layers,
                DFlash2TargetForwardMode::GreedyVerify,
                ().into(),
            )
            .expect("DFlash2 sequence-stable verify");
        let dflash_logits = model
            .dflash2_project_hidden_on(&dflash_verify.hidden, ().into())
            .expect("DFlash2 verify logits");
        mlx::transforms::eval(&[&ordinary_logits, &dflash_logits]).expect("evaluate verify logits");
        assert_array_exact(&ordinary_logits, &dflash_logits, "DFlash2 verify");

        let sampled_dflash_snapshots = sampled_dflash_cache
            .iter()
            .map(crate::nn::LayerCache::snapshot)
            .collect::<Vec<_>>();
        for layer in &mut sampled_dflash_cache {
            layer
                .begin_speculative_prefix_capture()
                .expect("begin sampled DFlash2 prefix capture");
        }
        let sampled_dflash_verify = model
            .dflash2_forward_target_on(
                &verify,
                &verify_positions,
                Some(&mut sampled_dflash_cache),
                &target_layers,
                DFlash2TargetForwardMode::SampledVerify,
                ().into(),
            )
            .expect("sampled DFlash2 verify");
        let sampled_dflash_logits = model
            .dflash2_project_hidden_on(&sampled_dflash_verify.hidden, ().into())
            .expect("sampled DFlash2 verify logits");
        mlx::transforms::eval(&[&ordinary_logits, &sampled_dflash_logits])
            .expect("evaluate sampled verify logits");
        assert_array_exact(
            &ordinary_logits,
            &sampled_dflash_logits,
            "sampled DFlash2 verify",
        );

        for (index, &token) in verify_ids[..2].iter().enumerate() {
            let input: Array = (&[token][..], &[1_i32, 1][..])
                .try_into()
                .expect("ordinary prefix verify input");
            let positions =
                build_position_ids(3 + index as i32, 1).expect("ordinary prefix verify position");
            let logits = model
                .forward_on(
                    &input,
                    &positions,
                    None,
                    None,
                    Some(&mut ordinary_prefix_cache),
                    (),
                )
                .expect("ordinary accepted-prefix verify");
            mlx::transforms::eval(&[&logits]).expect("evaluate accepted-prefix logits");
        }
        model
            .dflash2_restore_target_prefix_on(&mut dflash_cache, &dflash_snapshots, 2, ().into())
            .expect("restore accepted DFlash2 prefix");
        model
            .dflash2_restore_target_prefix_on(
                &mut sampled_dflash_cache,
                &sampled_dflash_snapshots,
                2,
                ().into(),
            )
            .expect("restore accepted sampled DFlash2 prefix");

        let correction: Array = (&[800_u32][..], &[1_i32, 1][..])
            .try_into()
            .expect("correction input");
        let correction_position = build_position_ids(5, 1).expect("correction position");
        let ordinary_correction = model
            .forward_on(
                &correction,
                &correction_position,
                None,
                None,
                Some(&mut ordinary_prefix_cache),
                (),
            )
            .expect("ordinary correction");
        let dflash_correction = model
            .dflash2_forward_target_on(
                &correction,
                &correction_position,
                Some(&mut dflash_cache),
                &target_layers,
                DFlash2TargetForwardMode::GreedyVerify,
                ().into(),
            )
            .expect("DFlash2 correction");
        let dflash_correction = model
            .dflash2_project_hidden_on(&dflash_correction.hidden, ().into())
            .expect("DFlash2 correction logits");
        let sampled_dflash_correction = model
            .dflash2_forward_target_on(
                &correction,
                &correction_position,
                Some(&mut sampled_dflash_cache),
                &target_layers,
                DFlash2TargetForwardMode::SampledVerify,
                ().into(),
            )
            .expect("sampled DFlash2 correction");
        let sampled_dflash_correction = model
            .dflash2_project_hidden_on(&sampled_dflash_correction.hidden, ().into())
            .expect("sampled DFlash2 correction logits");
        mlx::transforms::eval(&[
            &ordinary_correction,
            &dflash_correction,
            &sampled_dflash_correction,
        ])
        .expect("evaluate correction logits");
        assert_array_exact(
            &ordinary_correction,
            &dflash_correction,
            "DFlash2 restored prefix continuation",
        );
        assert_array_exact(
            &ordinary_correction,
            &sampled_dflash_correction,
            "sampled DFlash2 restored prefix continuation",
        );
    }

    fn repeat_rank3_row(row: &Array, batch: i32) -> Array {
        let rows = std::iter::repeat_n(row, batch as usize).collect::<Vec<_>>();
        mlx::ops::shape::concatenate(&rows, 0).expect("repeat rank-3 row")
    }

    fn slice_rank3_row(array: &Array, row: i32) -> Array {
        let dims = array.shape();
        let dims = dims.as_slice();
        assert_eq!(dims.len(), 3, "expected rank-3 array");
        mlx::ops::indexing::slice(
            array,
            &[row, 0_i32, 0_i32][..],
            &[row + 1, dims[1], dims[2]][..],
        )
        .expect("slice rank-3 row")
    }

    fn assert_array_exact(reference: &Array, candidate: &Array, label: &str) {
        let reference = mlx::ops::cast::astype(reference, Dtype::Float32)
            .expect("cast reference")
            .to_vec::<f32>()
            .expect("read reference");
        let candidate = mlx::ops::cast::astype(candidate, Dtype::Float32)
            .expect("cast candidate")
            .to_vec::<f32>()
            .expect("read candidate");
        assert_eq!(reference, candidate, "{label}");
    }

    #[test]
    #[ignore = "loads full local Qwen3.5 main and MTP checkpoints"]
    fn qwen35_product_stable_mtp_batched_matches_b1_rows_exactly() {
        let Some(model_dir) = qwen35_checkpoint("QWEN35_MODEL") else {
            return;
        };
        let Some(mtp_dir) = qwen35_checkpoint("QWEN35_MTP_MODEL") else {
            return;
        };
        let loader = crate::core::Loader::open(&model_dir).expect("open Qwen3.5 model");
        let model = Qwen35Model::from_loader(&loader).expect("load Qwen3.5 model");
        let mtp_loader = crate::core::Loader::open(&mtp_dir).expect("open Qwen3.5 MTP");
        let mtp = model.load_mtp_head(&mtp_loader).expect("load Qwen3.5 MTP");
        let key = mlx::random::key(20260729).expect("random key");
        let hidden_row = mlx::random::normal()
            .shape((1_i32, 1_i32, model.config().hidden_size))
            .dtype(model.hidden_dtype())
            .key(&key)
            .sample()
            .expect("sample hidden row");
        let token_row: Array = (&[100_u32][..], &[1_i32, 1_i32][..])
            .try_into()
            .expect("token row");
        let position_row = crate::core::generate::build_position_ids(0, 1).expect("position row");
        let reference = model
            .mtp_forward_on(
                &mtp,
                &hidden_row,
                &token_row,
                &position_row,
                None,
                None,
                mlx::StreamOrDevice::default(),
            )
            .expect("B1 MTP reference");

        for batch in [2_i32, 4, 8] {
            let hidden = repeat_rank3_row(&hidden_row, batch);
            let token_values = vec![100_u32; batch as usize];
            let tokens: Array = (&token_values[..], &[batch, 1_i32][..])
                .try_into()
                .expect("batched tokens");
            let positions =
                mlx::ops::shape::broadcast_to(&position_row, &[3_i32, batch, 1_i32][..])
                    .expect("batched positions");
            let candidate = model
                .mtp_forward_on(
                    &mtp,
                    &hidden,
                    &tokens,
                    &positions,
                    None,
                    None,
                    mlx::StreamOrDevice::default(),
                )
                .expect("batched product-stable MTP");
            mlx::transforms::eval(&[
                &reference.hidden_states,
                &reference.logits,
                &candidate.hidden_states,
                &candidate.logits,
            ])
            .expect("evaluate MTP outputs");

            for row in 0..batch {
                assert_array_exact(
                    &reference.hidden_states,
                    &slice_rank3_row(&candidate.hidden_states, row),
                    &format!("B{batch} row {row} hidden state diverged from B1"),
                );
                assert_array_exact(
                    &reference.logits,
                    &slice_rank3_row(&candidate.logits, row),
                    &format!("B{batch} row {row} logits diverged from B1"),
                );
            }
        }
    }

    /// Integration test: text-only `forward_vl` (pixel_values=None) must produce
    /// output numerically identical to `forward_on`.
    ///
    /// Run with:
    /// ```
    /// QWEN35_MODEL=<path> cargo test -p ironmlx --lib --release forward_vl_text_only_matches_forward_on -- --ignored
    /// ```
    #[test]
    #[ignore] // real-model heavy
    fn forward_vl_text_only_matches_forward_on() {
        use crate::core::generate::build_position_ids;
        use crate::core::Loader;

        let env = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL not set");
        let loader =
            Loader::open_multimodal(std::path::Path::new(&env)).expect("loader open_multimodal");
        let model = Qwen35Model::from_loader(&loader).expect("model");

        let input_ids: mlx::Array = (&[100_i32, 101, 102][..], &[1_i32, 3][..])
            .try_into()
            .expect("input_ids");
        let pos = build_position_ids(0, 3).expect("build_position_ids");

        // text-only path via forward_on
        let logits_a = model
            .forward_on(&input_ids, &pos, None, None, None, ())
            .expect("forward_on");

        // forward_vl with pixel_values=None must be numerically identical
        let logits_b = model
            .forward_vl(
                &input_ids,
                &pos,
                None,
                None,
                None,
                None,
                None,
                crate::core::generate::IMAGE_TOKEN_ID,
                (),
            )
            .expect("forward_vl text-only");

        // Compute max absolute difference
        let diff = mlx::ops::subtract(&logits_a, &logits_b).expect("subtract");
        let abs_diff = mlx::ops::abs(&diff).expect("abs");
        let max_diff_arr = mlx::ops::max(&abs_diff, mlx::ops::All, false).expect("max");
        let max_diff_f32: Vec<f32> = mlx::ops::astype(&max_diff_arr, mlx::Dtype::Float32)
            .expect("astype")
            .to_vec()
            .expect("to_vec");
        let max_diff = max_diff_f32[0];

        assert!(
            max_diff < 1e-5,
            "forward_vl text-only diverged from forward_on: max_diff={max_diff}"
        );
    }

    /// Integration test: text-only batch (all per_row_pv = None) →
    /// `batched_prefill_vl` must be byte-equal to `batched_prefill` because
    /// the vision path is fully skipped when no row carries pixel_values.
    ///
    /// Run with:
    /// ```
    /// IRONMLX_MODEL_DIR=<path> cargo test -p ironmlx --lib \
    ///   batched_prefill_vl_text_only_matches_batched_prefill -- --nocapture
    /// ```
    #[test]
    #[ignore] // real-model heavy: needs IRONMLX_MODEL_DIR
    fn batched_prefill_vl_text_only_matches_batched_prefill() {
        use crate::core::generate::{
            build_batch_attention_mask, build_batch_linear_mask, build_position_ids_batched,
            IMAGE_TOKEN_ID,
        };
        use crate::core::Loader;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR")
            .or_else(|_| {
                let glob = format!(
                    "{}/.ironmlx/models/huggingface/mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                    std::env::var("HOME").unwrap()
                );
                let entries = std::fs::read_dir(&glob).map_err(|e| e.to_string())?;
                let first = entries
                    .filter_map(|e| e.ok())
                    .next()
                    .ok_or_else(|| "no snapshot dir".to_string())?;
                Ok::<String, String>(first.path().to_string_lossy().into_owned())
            })
            .expect("model dir");
        let model_path = std::path::PathBuf::from(model_dir);

        let loader = Loader::open(&model_path).expect("Loader::open");
        let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");

        let prompt_lens = vec![5_i32, 4_i32];
        let max_len = 5_i32;
        let b = prompt_lens.len() as i32;

        // Build input_ids [B=2, max_len=5] right-padded
        let mut flat: Vec<i32> = vec![0; (b * max_len) as usize];
        flat[0..5].copy_from_slice(&[1_i32, 2, 3, 4, 5]);
        flat[5..9].copy_from_slice(&[10, 11, 12, 13]);
        // flat[9] stays 0 (pad)
        let input_ids: Array = (&flat[..], &[b, max_len][..]).try_into().unwrap();

        let position_ids = build_position_ids_batched(&prompt_lens, max_len).unwrap();
        let attention_mask =
            build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16).unwrap();
        let linear_mask = build_batch_linear_mask(&prompt_lens, max_len).unwrap();

        let mut cache_a = model.make_cache(b, max_len, Dtype::Bfloat16).unwrap();
        let mut cache_b = model.make_cache(b, max_len, Dtype::Bfloat16).unwrap();

        let logits_baseline = model
            .batched_prefill(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_mask,
                &prompt_lens,
                Some(&mut cache_a),
                (),
            )
            .unwrap();

        let per_row_pv: Vec<Option<&[Array]>> = vec![None, None];
        let per_row_grids: Vec<Option<&[(i32, i32, i32)]>> = vec![None, None];
        let logits_vl = model
            .batched_prefill_vl(
                &input_ids,
                &position_ids,
                &attention_mask,
                &linear_mask,
                &prompt_lens,
                &per_row_pv,
                &per_row_grids,
                IMAGE_TOKEN_ID,
                Some(&mut cache_b),
                (),
            )
            .unwrap();

        let a: Vec<f32> = mlx::ops::astype(&logits_baseline, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        let b_vec: Vec<f32> = mlx::ops::astype(&logits_vl, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();

        assert_eq!(a.len(), b_vec.len(), "logits length");
        for (i, (av, bv)) in a.iter().zip(b_vec.iter()).enumerate() {
            assert_eq!(av, bv, "logits[{i}] differ: {av} vs {bv}");
        }
    }

    /// Integration test: B=1 with a single image — `batched_prefill_vl` must
    /// produce logits numerically equivalent to `forward_vl` on the same
    /// single-stream input (both paths share vision encoder + scatter +
    /// transformer + last-position project). Allows small bf16-roundoff
    /// tolerance (max_abs < 1e-3) and requires bit-identical greedy argmax.
    ///
    /// Run with:
    /// ```
    /// IRONMLX_MODEL_DIR=<path> cargo test -p ironmlx --lib \
    ///   batched_prefill_vl_b1_matches_forward_vl -- --nocapture
    /// ```
    #[test]
    #[ignore] // real-model heavy: needs IRONMLX_MODEL_DIR
    fn batched_prefill_vl_b1_matches_forward_vl() {
        use crate::core::generate::{
            build_batch_attention_mask, build_batch_linear_mask, build_position_ids_vl,
            build_position_ids_vl_batched, IMAGE_TOKEN_ID,
        };
        use crate::core::Loader;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/huggingface/mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            let entries = std::fs::read_dir(&glob).expect("snapshots dir");
            entries
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let model_path = std::path::PathBuf::from(model_dir);

        let loader = Loader::open_multimodal(&model_path).expect("Loader::open_multimodal");
        let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");

        // Synthesize a real preprocessed pixel_values from image_0 fixture.
        let fixture_bytes = std::fs::read("tests/fixtures/qwen35_vl/multi_image/image_0.jpg")
            .expect("image_0 fixture");
        let (pixel_values, grid_h, grid_w) =
            crate::models::qwen3_5::image_processor::preprocess(&fixture_bytes)
                .expect("preprocess");
        let merge_size = 2_i32;
        let grids_real: Vec<(i32, i32, i32)> = vec![(1, grid_h, grid_w)];
        let n_pads = (grid_h * grid_w / (merge_size * merge_size)) as usize;
        // Compose prompt: [1, 2, 3, IMG×n_pads, 4, 5]
        let mut prompt_ids: Vec<i32> = vec![1, 2, 3];
        prompt_ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(n_pads));
        prompt_ids.extend([4_i32, 5]);
        let prompt_len = prompt_ids.len() as i32;

        // forward_vl baseline (B=1)
        let input_ids_b1: Array = (&prompt_ids[..], &[1_i32, prompt_len][..])
            .try_into()
            .unwrap();
        let position_ids_b1 =
            build_position_ids_vl(&prompt_ids, &grids_real, IMAGE_TOKEN_ID, merge_size).unwrap();
        let mut cache_a = model.make_cache(1, prompt_len, Dtype::Bfloat16).unwrap();
        let logits_a = model
            .forward_vl(
                &input_ids_b1,
                &position_ids_b1,
                None,
                None,
                Some(&mut cache_a),
                Some(std::slice::from_ref(&pixel_values)),
                Some(&grids_real),
                IMAGE_TOKEN_ID,
                (),
            )
            .unwrap();

        // batched_prefill_vl B=1
        let position_ids_batched = build_position_ids_vl_batched(
            &[&prompt_ids[..]],
            &[Some(&grids_real[..])],
            IMAGE_TOKEN_ID,
            merge_size,
            prompt_len,
        )
        .unwrap();
        let attention_mask =
            build_batch_attention_mask(&[prompt_len], prompt_len, Dtype::Bfloat16).unwrap();
        let linear_mask = build_batch_linear_mask(&[prompt_len], prompt_len).unwrap();
        let per_row_pv: Vec<Option<&[Array]>> = vec![Some(std::slice::from_ref(&pixel_values))];
        let per_row_grids: Vec<Option<&[(i32, i32, i32)]>> = vec![Some(&grids_real[..])];
        let mut cache_b = model.make_cache(1, prompt_len, Dtype::Bfloat16).unwrap();
        let logits_b = model
            .batched_prefill_vl(
                &input_ids_b1,
                &position_ids_batched,
                &attention_mask,
                &linear_mask,
                &[prompt_len],
                &per_row_pv,
                &per_row_grids,
                IMAGE_TOKEN_ID,
                Some(&mut cache_b),
                (),
            )
            .unwrap();

        let a: Vec<f32> = mlx::ops::astype(&logits_a, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        let b_vec: Vec<f32> = mlx::ops::astype(&logits_b, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();

        assert_eq!(a.len(), b_vec.len(), "logits length");
        // bf16 round-trip can introduce ULP-level diffs; require <1e-3 max-abs diff
        // and bit-identical greedy argmax.
        let mut max_abs = 0.0_f32;
        for (av, bv) in a.iter().zip(b_vec.iter()) {
            let d = (av - bv).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(max_abs < 1e-3, "max-abs logits diff = {max_abs} >= 1e-3");

        let argmax_a = a
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;
        let argmax_b = b_vec
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax_a, argmax_b, "greedy argmax mismatch");
    }
}

#[cfg(test)]
mod per_row_slice_tests {
    use super::*;

    #[test]
    fn per_row_slice_last_uniform_pick() {
        // hidden [2, 4, 3] with deterministic values: hidden[i, j, c] = (i*4 + j)*3 + c.
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        // Pick last positions [3, 3] (the same column = degenerate per-row case).
        let out = per_row_slice_last(&hidden, &[3, 3], ()).expect("per_row_slice_last");
        assert_eq!(out.shape().as_slice(), &[2, 1, 3]);
        // Row 0 last (j=3): values 9,10,11
        // Row 1 last (j=3): values 21,22,23
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert_eq!(v, vec![9.0, 10.0, 11.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn per_row_slice_last_ragged_pick() {
        // hidden [2, 4, 3] same as above.
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        // Row 0 last position = 1 (only 2 real tokens); row 1 last position = 3 (all 4).
        let out = per_row_slice_last(&hidden, &[1, 3], ()).expect("per_row_slice_last ragged");
        assert_eq!(out.shape().as_slice(), &[2, 1, 3]);
        // Row 0 j=1: values 3,4,5
        // Row 1 j=3: values 21,22,23
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
