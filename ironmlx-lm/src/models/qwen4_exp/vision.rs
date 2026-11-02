//! Flash-Next uses the Qwen3-VL image tower and three-axis RoPE, while retaining
//! its own request-private QSA/PLE/GDN state and raw HC draft captures.
use super::{model::row_lengths, text_model::FlashTextModel};
use crate::{
    core::{cache::layer::LayerCache, vision::*},
    models::dflash2::DFlash2TargetForwardMode,
};
use anyhow::{ensure, Context, Result};
use mlx::{Array, StreamOrDevice};

impl FlashTextModel {
    pub(crate) fn image_inputs_on(
        &self,
        ids: &Array,
        embeds: Option<&Array>,
        image_id: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let text = self.embed_on(ids, target)?;
        match embeds {
            Some(vision) => crate::models::qwen3_5::cross_modal::replace_image_tokens(
                &text, ids, vision, image_id,
            ),
            None => Ok(text),
        }
    }
}

impl DenseVlMethods for FlashTextModel {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batched_prefill_vl(
        &self,
        ids: &Array,
        positions: &Array,
        _attention: &Array,
        linear: &Array,
        lengths: &[i32],
        pixels: &[Option<&[Array]>],
        grids: &[Option<&[(i32, i32, i32)]>],
        image_id: i32,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = ids.shape();
        let d = shape.as_slice();
        ensure!(
            d.len() == 2 && pixels.len() == d[0] as usize && grids.len() == pixels.len(),
            "Flash vision batch lengths disagree"
        );
        let mut rows = Vec::new();
        for row in 0..d[0] {
            let row_ids = ids.slice(&[row, 0][..], &[row + 1, d[1]][..])?;
            let embeds = match (pixels[row as usize], grids[row as usize]) {
                (Some(p), Some(g)) => Some(self.compute_vision_embeds(p, g, target)?),
                (None, None) => None,
                _ => anyhow::bail!("Flash image patches and grids must occur together"),
            };
            rows.push(self.image_inputs_on(&row_ids, embeds.as_ref(), image_id, target)?);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let embeds = mlx::ops::shape::concatenate_on(&refs, 0, target)?;
        let lengths = row_lengths(d[0] as usize, d[1], Some(lengths), Some(linear))?;
        let output = self.runtime_inputs_on(
            ids,
            Some(positions),
            Some(&embeds),
            Some(&lengths),
            None,
            cache,
            &[],
            DFlash2TargetForwardMode::Prefill,
            target,
        )?;
        self.last_logits_on(&output.hidden, &lengths, target)
    }
    fn estimate_vision_prefill_tensor_bytes(
        &self,
        pixels: &[Array],
        grids: &[(i32, i32, i32)],
    ) -> Result<usize> {
        ensure!(
            !pixels.is_empty() && pixels.len() == grids.len(),
            "Flash image patches and grids disagree"
        );
        ensure!(
            grids.iter().all(|&(t, _, _)| t == 1),
            "Flash video inputs are not qualified; image grids must have temporal size 1"
        );
        let c = self
            .vision_config
            .as_ref()
            .context("Flash checkpoint has no vision tower")?;
        let merge = usize::try_from(c.spatial_merge_size)?;
        estimate_transformer_vision_prefill_tensor_bytes(
            pixels,
            grids,
            VisionPrefillMemoryProfile {
                hidden_size: usize::try_from(c.hidden_size)?,
                intermediate_size: usize::try_from(c.intermediate_size)?,
                num_attention_heads: usize::try_from(c.num_heads)?,
                output_hidden_size: usize::try_from(c.out_hidden_size)?,
                spatial_merge_area: merge.checked_mul(merge).context("vision merge overflow")?,
                activation_bytes: 2,
                attention: VisionAttention::FusedSdpa {
                    head_dim: c.head_dim()?,
                },
            },
        )
    }
    fn compute_vision_embeds(
        &self,
        pixels: &[Array],
        grids: &[(i32, i32, i32)],
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.estimate_vision_prefill_tensor_bytes(pixels, grids)?;
        self.vision.load_deferred()?;
        let tower = self
            .vision
            .get()
            .context("Flash vision tower did not load")?;
        let refs: Vec<&Array> = pixels.iter().collect();
        let patches = mlx::ops::shape::concatenate_on(&refs, 0, target)?;
        let output = tower.forward(&patches, grids)?;
        output.eval()?;
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_vl_chunk(
        &self,
        ids: &Array,
        positions: &Array,
        lengths: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        embeds: Option<&Array>,
        image_id: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let d = ids.shape();
        let d = d.as_slice();
        ensure!(d.len() == 2, "Flash image token input must be [B,S]");
        let valid = row_lengths(d[0] as usize, d[1], lengths, mask)?;
        let hidden = self.forward_vl_hidden(
            ids, positions, lengths, mask, cache, embeds, image_id, target,
        )?;
        self.last_logits_on(&hidden, &valid, target)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_vl_hidden(
        &self,
        ids: &Array,
        positions: &Array,
        lengths: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        embeds: Option<&Array>,
        image_id: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let embeddings = self.image_inputs_on(ids, embeds, image_id, target)?;
        Ok(self
            .runtime_inputs_on(
                ids,
                Some(positions),
                Some(&embeddings),
                lengths,
                mask,
                cache,
                &[],
                DFlash2TargetForwardMode::Prefill,
                target,
            )?
            .hidden)
    }
}
