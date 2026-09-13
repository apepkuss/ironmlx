//! Model-side vision capabilities and tensor working-set estimates.
//! No scheduling, system-memory probing, admission policy or HTTP types belong here.

/// Extension trait for VL-capable models, intentionally NOT part of `core::Model`
/// (per P5 spec §3.1 — VL methods stay inherent / extension-trait-only).
///
/// Model-side operations for vision encoding and multimodal prefill. Execution
/// drivers consume this interface; implementations do not depend on a scheduler.
pub trait DenseVlMethods {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
        cache: Option<&mut [crate::core::cache::layer::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;

    /// Estimate the peak tensor working set for this exact vision payload.
    /// Includes payload, retained outputs and temporary tensors, but excludes
    /// runtime safety margins and allocator/Metal overhead. The execution driver
    /// applies its reservation policy once per row. Invalid/unsupported payloads
    /// must return an error rather than a zero estimate.
    fn estimate_vision_prefill_tensor_bytes(
        &self,
        pixel_values: &[mlx::Array],
        grid_thw: &[(i32, i32, i32)],
    ) -> crate::Result<usize>;

    fn compute_vision_embeds(
        &self,
        pixel_values: &[mlx::Array],
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_chunk(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::core::cache::layer::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_hidden(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::core::cache::layer::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array>;
}

/// Architecture parameters used by the common vision-prefill peak estimator.
/// Model implementations remain responsible for selecting the correct values
/// from their loaded vision configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VisionPrefillMemoryProfile {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub output_hidden_size: usize,
    pub spatial_merge_area: usize,
    pub activation_bytes: usize,
}

/// Conservative upper bound for a transformer-style vision tower. It covers
/// payload storage, QKV/attention/MLP temporaries, positional intermediates,
/// and merged output retained for cross-modal scatter. Runtime reservation
/// policy adds its safety margin separately. Saturating
/// arithmetic intentionally turns overflow into a fail-safe reservation
/// rejection instead of underestimating the peak.
pub(crate) fn estimate_transformer_vision_prefill_tensor_bytes(
    pixel_values: &[mlx::Array],
    grid_thw: &[(i32, i32, i32)],
    profile: VisionPrefillMemoryProfile,
) -> crate::Result<usize> {
    anyhow::ensure!(
        !pixel_values.is_empty(),
        "vision peak estimator requires non-empty pixel_values"
    );
    anyhow::ensure!(
        !grid_thw.is_empty(),
        "vision peak estimator requires non-empty grid_thw"
    );
    anyhow::ensure!(
        profile.hidden_size > 0
            && profile.intermediate_size > 0
            && profile.num_attention_heads > 0
            && profile.output_hidden_size > 0
            && profile.spatial_merge_area > 0
            && profile.activation_bytes > 0,
        "vision peak estimator received an invalid model profile"
    );

    let payload_bytes = pixel_values.iter().fold(0usize, |total, pixels| {
        total.saturating_add(pixels.size().saturating_mul(pixels.dtype().byte_size()))
    });
    let mut total_tokens = 0usize;
    let mut attention_scores = 0usize;
    for &(t, h, w) in grid_thw {
        anyhow::ensure!(
            t > 0 && h > 0 && w > 0,
            "vision peak estimator requires positive grid dimensions, got ({t}, {h}, {w})"
        );
        let tokens = usize::try_from(t)
            .unwrap_or(usize::MAX)
            .saturating_mul(usize::try_from(h).unwrap_or(usize::MAX))
            .saturating_mul(usize::try_from(w).unwrap_or(usize::MAX));
        total_tokens = total_tokens.saturating_add(tokens);
        attention_scores = attention_scores.saturating_add(
            profile
                .num_attention_heads
                .saturating_mul(tokens)
                .saturating_mul(tokens)
                .saturating_mul(std::mem::size_of::<f32>()),
        );
    }

    let hidden_activations = total_tokens
        .saturating_mul(profile.hidden_size)
        .saturating_mul(profile.activation_bytes);
    let qkv_peak = hidden_activations.saturating_mul(4);
    let attention_peak = qkv_peak.saturating_add(attention_scores);
    let mlp_peak = total_tokens
        .saturating_mul(profile.intermediate_size)
        .saturating_mul(profile.activation_bytes)
        .saturating_add(hidden_activations.saturating_mul(2));
    let positional_peak = hidden_activations.saturating_mul(4);
    let merged_tokens =
        total_tokens.saturating_add(profile.spatial_merge_area - 1) / profile.spatial_merge_area;
    let retained_output = merged_tokens
        .saturating_mul(profile.output_hidden_size)
        .saturating_mul(profile.activation_bytes);
    let merger_peak = hidden_activations
        .saturating_mul(profile.spatial_merge_area)
        .saturating_add(retained_output);
    let stage_peak = qkv_peak
        .max(attention_peak)
        .max(mlp_peak)
        .max(positional_peak)
        .max(merger_peak);
    let unscaled = payload_bytes
        .saturating_add(retained_output)
        .saturating_add(stage_peak);
    let estimated = unscaled;
    anyhow::ensure!(
        estimated > 0 && estimated != usize::MAX,
        "vision peak estimate overflowed or produced zero"
    );
    Ok(estimated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Array;

    #[test]
    fn transformer_vision_peak_estimator_is_conservative_and_deterministic() {
        let pixels: Array = (&[0.0_f32; 16][..], &[1_i32, 16][..])
            .try_into()
            .expect("pixel payload");
        let estimate = estimate_transformer_vision_prefill_tensor_bytes(
            &[pixels],
            &[(1, 2, 2)],
            VisionPrefillMemoryProfile {
                hidden_size: 8,
                intermediate_size: 16,
                num_attention_heads: 2,
                output_hidden_size: 8,
                spatial_merge_area: 4,
                activation_bytes: 2,
            },
        )
        .expect("valid estimate");

        assert_eq!(estimate, 464);
        let concatenated_pixels: Array = (&[0.0_f32; 32][..], &[2_i32, 16][..])
            .try_into()
            .expect("concatenated pixel payload");
        let multi_grid_estimate = estimate_transformer_vision_prefill_tensor_bytes(
            &[concatenated_pixels],
            &[(1, 2, 2), (1, 2, 2)],
            VisionPrefillMemoryProfile {
                hidden_size: 8,
                intermediate_size: 16,
                num_attention_heads: 2,
                output_hidden_size: 8,
                spatial_merge_area: 4,
                activation_bytes: 2,
            },
        )
        .expect("one concatenated tensor may describe multiple image grids");
        assert!(multi_grid_estimate > estimate);
        assert!(estimate_transformer_vision_prefill_tensor_bytes(
            &[],
            &[],
            VisionPrefillMemoryProfile {
                hidden_size: 8,
                intermediate_size: 16,
                num_attention_heads: 2,
                output_hidden_size: 8,
                spatial_merge_area: 4,
                activation_bytes: 2,
            },
        )
        .is_err());
    }
}
