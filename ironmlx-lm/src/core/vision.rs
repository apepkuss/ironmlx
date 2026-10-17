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
    pub attention: VisionAttention,
}

/// How a vision tower runs attention over an image's patches, which decides
/// whether its scores are ever materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VisionAttention {
    /// The `heads x tokens x tokens` score matrix exists in f32.
    MaterializedScores,
    /// The Qwen tower's attention: f32 rotary embedding of q and k
    /// (`apply_rotary_vision`), then unmasked
    /// `mlx::fast::scaled_dot_product_attention` per image with this head
    /// dim. MLX's full-attention kernel streams the scores, needing beside
    /// its output at most contiguous copies of q, k and v; images it does
    /// not cover (short sequences, other head dims) still count their
    /// scores.
    FusedSdpa { head_dim: usize },
}

impl VisionAttention {
    /// Head dims MLX's fused full-attention kernel takes (query length > 8,
    /// equal query/value head dims; `has_fused_kernel` in
    /// `scaled_dot_product_attention.cpp`).
    const FUSED_FULL_ATTENTION_HEAD_DIMS: [usize; 7] = [64, 72, 80, 96, 128, 192, 256];

    /// Bytes attention adds to the q/k/v activations for one image of
    /// `tokens` patches.
    fn image_bytes(self, profile: &VisionPrefillMemoryProfile, tokens: usize) -> usize {
        match self {
            Self::FusedSdpa { head_dim }
                if tokens > 8 && Self::FUSED_FULL_ATTENTION_HEAD_DIMS.contains(&head_dim) =>
            {
                // In hidden-size activations: the rotary of q and k, each
                // an f32 product pair and sum (2 x 3 x 2), then contiguous
                // q/k/v copies and the output (4). Measured against the
                // Qwen3.6 tower's footprint on MLX 73ad5df2, about 23 such
                // activations (with the 4 of the q/k/v stage) are live at
                // 65536 patches.
                tokens
                    .saturating_mul(profile.hidden_size)
                    .saturating_mul(profile.activation_bytes)
                    .saturating_mul(16)
            }
            _ => profile
                .num_attention_heads
                .saturating_mul(tokens)
                .saturating_mul(tokens)
                .saturating_mul(std::mem::size_of::<f32>()),
        }
    }
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
    let mut attention_bytes = 0usize;
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
        attention_bytes =
            attention_bytes.saturating_add(profile.attention.image_bytes(&profile, tokens));
    }

    let hidden_activations = total_tokens
        .saturating_mul(profile.hidden_size)
        .saturating_mul(profile.activation_bytes);
    let qkv_peak = hidden_activations.saturating_mul(4);
    let attention_peak = qkv_peak.saturating_add(attention_bytes);
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
                attention: VisionAttention::MaterializedScores,
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
                attention: VisionAttention::MaterializedScores,
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
                attention: VisionAttention::MaterializedScores,
            },
        )
        .is_err());
    }

    /// A fused-attention tower counts linear q/k/v copies instead of the
    /// quadratic score matrix, but only where MLX's fused kernel applies.
    #[test]
    fn fused_attention_replaces_the_score_matrix_where_mlx_fuses() {
        // A 128 x 128 grid: 16384 patches, as a 4.2M-pixel Qwen image.
        let pixels: Array = (&[0.0_f32; 16][..], &[1_i32, 16][..])
            .try_into()
            .expect("pixel payload");
        let profile = |attention| VisionPrefillMemoryProfile {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            output_hidden_size: 2048,
            spatial_merge_area: 4,
            activation_bytes: 2,
            attention,
        };
        let estimate = |grid, attention| {
            estimate_transformer_vision_prefill_tensor_bytes(
                std::slice::from_ref(&pixels),
                &[grid],
                profile(attention),
            )
            .expect("valid estimate")
        };
        let materialized = estimate((1, 128, 128), VisionAttention::MaterializedScores);
        let fused = estimate((1, 128, 128), VisionAttention::FusedSdpa { head_dim: 72 });
        // 16 heads x 16384^2 f32 scores dominate the materialized estimate.
        assert!(materialized > 16 * 16384 * 16384 * 4);
        assert!(fused < 2 << 30, "fused estimate {fused}");
        // A head dim MLX does not fuse, or a sequence short enough for the
        // vector kernel, keeps the score matrix.
        assert_eq!(
            estimate((1, 128, 128), VisionAttention::FusedSdpa { head_dim: 112 }),
            materialized
        );
        assert_eq!(
            estimate((1, 2, 4), VisionAttention::FusedSdpa { head_dim: 72 }),
            estimate((1, 2, 4), VisionAttention::MaterializedScores)
        );
    }
}
