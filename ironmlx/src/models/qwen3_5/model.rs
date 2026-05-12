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
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self
            .text
            .forward_on(input_ids, position_ids, cache, target)?;
        self.slice_last_and_project(&hidden, target)
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
        let hidden =
            self.text
                .forward_post_embedding_on(inputs_embeds, position_ids, None, target)?;
        self.slice_last_and_project(&hidden, target)
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
        let hidden = self
            .text
            .forward_post_embedding_on(&hidden, position_ids, cache, target)?;

        // Step 4: slice last position and project to logits.
        self.slice_last_and_project(&hidden, target)
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
        cache: Option<&mut [LayerCache]>,
        pixel_values: Option<&Array>,
        grid_thw: Option<&[(i32, i32, i32)]>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Step 1: embed token ids → [B, S, hidden_size]
        let mut hidden = self.text.embed_on(input_ids, target)?;

        // Step 2: if pixel_values provided, route through vision tower and replace
        //         image-token positions in the embedded sequence.
        if let Some(pv) = pixel_values {
            let grids = grid_thw
                .ok_or_else(|| anyhow!("grid_thw required when pixel_values is provided"))?;
            let vision = self
                .vision
                .as_ref()
                .ok_or_else(|| anyhow!("model has no vision_tower; use Loader::open_multimodal"))?;
            let vision_embeds = vision.forward(pv, grids)?;
            hidden = super::cross_modal::replace_image_tokens(
                &hidden,
                input_ids,
                &vision_embeds,
                image_token_id,
            )?;
        }

        // Step 3: run transformer layers + final norm on the (possibly patched) hidden state.
        let hidden = self
            .text
            .forward_post_embedding_on(&hidden, position_ids, cache, target)?;

        // Step 4: slice last position and project to logits.
        self.slice_last_and_project(&hidden, target)
    }

    /// Slice the last sequence position from `hidden [B, S, H]` and project to
    /// vocab logits `[B, 1, vocab_size]`. Shared by [`forward_on`] and [`forward_vl`].
    fn slice_last_and_project(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        let dims_borrow = hidden.shape();
        let dims = dims_borrow.as_slice();
        let (b, s, h) = (dims[0], dims[1], dims[2]);
        let last_hidden = if s > 1 {
            mlx::ops::indexing::slice_strided(
                hidden,
                &[0_i32, s - 1, 0][..],
                &[b, s, h][..],
                &[1_i32, 1, 1][..],
            )?
        } else {
            hidden.clone()
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
            .forward_on(&input_ids, &pos, None, ())
            .expect("forward_on");

        // forward_vl with pixel_values=None must be numerically identical
        let logits_b = model
            .forward_vl(
                &input_ids,
                &pos,
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
