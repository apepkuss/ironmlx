//! MiniCPM-V-4.6 top-level model: text core + (optionally) vision encoder.
//!
//! Owns `Qwen35TextModel` directly (does NOT wrap `Qwen35Model`); the vision
//! tower (`MiniCpmV46Vision`) is loaded when the checkpoint was opened via
//! [`Loader::open_multimodal`] AND contains `vision_tower.*` keys.
//!
//! `lm_head` is `None` for MiniCPM-V-4.6 (uses `tie_word_embeddings = true`);
//! the field is kept for correctness if a future untied variant appears.
//!
//! `DenseVlMethods` (required by generate/serve/bench dispatch) is added in
//! P2a Task 3; until then `MiniCpmV46Model` is pub + re-exported but not yet
//! wired into `model_from_loader` dispatch.

use anyhow::anyhow;
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
    /// Wired by `DenseVlMethods` in P2a Task 3.
    #[allow(dead_code)]
    vision: Option<MiniCpmV46Vision>,
    /// Tokenizer id for the per-patch image placeholder.
    /// Wired by `DenseVlMethods` in P2a Task 3.
    #[allow(dead_code)]
    image_token_id: i32,
}

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

        let vision = if vcfg.is_some()
            && loader.contains("vision_tower.embeddings.patch_embedding.weight")
        {
            vcfg.as_ref()
                .map(|vc| MiniCpmV46Vision::from_loader(loader, vc))
                .transpose()?
        } else {
            None
        };

        let image_token_id = vcfg
            .map(|v| v.image_token_id)
            .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID);

        let text = Qwen35TextModel::from_loader(loader, cfg)?;

        Ok(Self {
            text,
            lm_head,
            vision,
            image_token_id,
        })
    }

    pub fn config(&self) -> &crate::models::Qwen35Config {
        self.text.config()
    }

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

    /// Construct a per-layer cache list matching this model's hybrid topology.
    ///
    /// Mirrors `Qwen35Model::make_cache` exactly — same Full/Linear partition
    /// logic driven by `cfg.layer_kind(i)`, same GPU-perf note re: cap floor.
    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = self.config();
        let head_dim = cfg.effective_head_dim();
        let mut out = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            match cfg.layer_kind(i) {
                AttnKind::Full => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minicpmv46_model_implements_model_trait() {
        fn assert_model<M: crate::core::model::Model>() {}
        assert_model::<MiniCpmV46Model>();
    }
}
