//! Top-level Qwen3.5 model: text model + (tied or explicit) lm_head + heterogeneous cache.

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::Loader;
use crate::nn::{AttnKind, LayerCache, Linear};
use crate::Result;

use super::config::Qwen35Config;
use super::text_model::Qwen35TextModel;
use super::vision::VisionTower;

/// Top-level Qwen3.5 dense model: hybrid 32-layer text core + tied/untied lm_head.
///
/// `vision` is present only when the model was loaded via [`Loader::open_multimodal`]
/// AND the config contains a `vision_config` block. Text-only inference is unaffected when
/// `vision` is `None`.
pub struct Qwen35Model {
    text: Qwen35TextModel,
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
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen35Config) -> Result<Self> {
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
            lm_head,
            vision,
        })
    }

    /// Test seam.
    #[doc(hidden)]
    pub fn from_components(text: Qwen35TextModel, lm_head: Option<Linear>) -> Self {
        Self {
            text,
            lm_head,
            vision: None,
        }
    }

    pub fn config(&self) -> &Qwen35Config {
        self.text.config()
    }

    pub fn text(&self) -> &Qwen35TextModel {
        &self.text
    }

    /// Forward to last-position logits `[B, 1, vocab_size]`.
    ///
    /// Sampling only consumes the final position; computing the lm_head
    /// projection over the entire prefill sequence wastes ~`(S-1)/S` of the
    /// projection work (vocab=151936 in Qwen3.5 — the largest matmul in the
    /// graph). Slice the last hidden state before the projection so the
    /// per-forward lm_head cost is constant in `S`.
    #[allow(clippy::too_many_arguments)]
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
    /// - `pixel_values` — `[N, T, C, H, W]` pre-processed patches.
    /// - `grid_thw`     — per-image `(temporal, height, width)`; must be
    ///   non-empty and sum to `N` along T·H·W.
    /// - `target`       — compute device / stream.
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

        // Step 1: embed token ids → [B, S, hidden_size]
        let mut hidden = self.text.embed_on(input_ids, target)?;

        // Step 2: if a vision_embeds slice was provided, scatter it into
        // the image-pad positions of this chunk. The slice's row count
        // must match the chunk's image-pad count (enforced by callee).
        if let Some(ve) = vision_embeds_slice {
            hidden =
                super::cross_modal::replace_image_tokens(&hidden, input_ids, ve, image_token_id)?;
        }

        // Step 3: run transformer layers + final norm.
        let hidden = self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            decode_mask,
            None,
            per_row_lens,
            target,
        )?;

        // Step 4: slice last position and project to logits.
        // VL chunk path is single-stream B=1; no per-row last position needed.
        self.slice_last_and_project(&hidden, None, target)
    }

    /// # Arguments
    /// - `input_ids`      — `[B, S]` int32 token ids (B must be 1 for P6).
    /// - `position_ids`   — `[3, B, S]` int32 per Mrope contract.
    /// - `cache`          — optional per-layer cache slice.
    /// - `pixel_values`   — pre-processed image patches `[N, T, C, H, W]`.
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
        pixel_values: Option<&Array>,
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
    /// would have written (verified by `tests/b1_p2_1_batched_prefill.rs`).
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

    /// Construct a per-layer cache list matching this model's hybrid topology.
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
            lm_head: None,
            vision: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::AttnKind;
    use mlx::Dtype;

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
        }
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
