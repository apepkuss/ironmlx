use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::{QuantMeta, QuantMode};
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactBatchedVerifyProfile {
    Disabled,
    Affine4,
    Affine5Dense,
    Affine5Moe,
    Affine6Dense,
    Affine6Moe,
    Affine8Dense,
    Affine8Moe,
}

pub(crate) fn dense_exact_batched_verify_profile(
    quant_meta: Option<QuantMeta>,
    checkpoint_dtype: Option<&str>,
) -> ExactBatchedVerifyProfile {
    if checkpoint_dtype != Some("bfloat16") {
        return ExactBatchedVerifyProfile::Disabled;
    }
    match quant_meta {
        Some(QuantMeta {
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        }) => ExactBatchedVerifyProfile::Affine4,
        Some(QuantMeta {
            group_size: 64,
            bits: 5,
            mode: QuantMode::Affine,
        }) => ExactBatchedVerifyProfile::Affine5Dense,
        Some(QuantMeta {
            group_size: 64,
            bits: 6,
            mode: QuantMode::Affine,
        }) => ExactBatchedVerifyProfile::Affine6Dense,
        Some(QuantMeta {
            group_size: 64,
            bits: 8,
            mode: QuantMode::Affine,
        }) => ExactBatchedVerifyProfile::Affine8Dense,
        None | Some(_) => ExactBatchedVerifyProfile::Disabled,
    }
}

pub(crate) fn moe_exact_batched_verify_profile(
    quant_meta: Option<QuantMeta>,
    checkpoint_dtype: Option<&str>,
) -> ExactBatchedVerifyProfile {
    match dense_exact_batched_verify_profile(quant_meta, checkpoint_dtype) {
        ExactBatchedVerifyProfile::Affine4 => ExactBatchedVerifyProfile::Affine4,
        ExactBatchedVerifyProfile::Affine5Dense => ExactBatchedVerifyProfile::Affine5Moe,
        ExactBatchedVerifyProfile::Affine6Dense => ExactBatchedVerifyProfile::Affine6Moe,
        ExactBatchedVerifyProfile::Affine8Dense => ExactBatchedVerifyProfile::Affine8Moe,
        ExactBatchedVerifyProfile::Disabled
        | ExactBatchedVerifyProfile::Affine5Moe
        | ExactBatchedVerifyProfile::Affine6Moe
        | ExactBatchedVerifyProfile::Affine8Moe => ExactBatchedVerifyProfile::Disabled,
    }
}

pub(crate) fn exact_batched_verify_qualified(
    profile: ExactBatchedVerifyProfile,
    batch_width: usize,
    context_tokens: usize,
    verify_width: usize,
) -> bool {
    let max_context_tokens = match profile {
        ExactBatchedVerifyProfile::Affine5Moe
        | ExactBatchedVerifyProfile::Affine6Moe
        | ExactBatchedVerifyProfile::Affine8Moe => 1_024,
        ExactBatchedVerifyProfile::Disabled
        | ExactBatchedVerifyProfile::Affine4
        | ExactBatchedVerifyProfile::Affine5Dense
        | ExactBatchedVerifyProfile::Affine6Dense
        | ExactBatchedVerifyProfile::Affine8Dense => 4_096,
    };
    context_tokens <= max_context_tokens
        && exact_batched_verify_shape_qualified(profile, batch_width, verify_width)
}

pub(crate) fn exact_batched_verify_shape_qualified(
    profile: ExactBatchedVerifyProfile,
    batch_width: usize,
    verify_width: usize,
) -> bool {
    match profile {
        ExactBatchedVerifyProfile::Disabled => false,
        ExactBatchedVerifyProfile::Affine4
        | ExactBatchedVerifyProfile::Affine5Moe
        | ExactBatchedVerifyProfile::Affine6Moe => {
            batch_width > 0 && batch_width <= 8 && verify_width > 1 && verify_width <= 5
        }
        ExactBatchedVerifyProfile::Affine5Dense
        | ExactBatchedVerifyProfile::Affine6Dense
        | ExactBatchedVerifyProfile::Affine8Dense => match batch_width {
            1 => verify_width > 1 && verify_width <= 5,
            2 => verify_width > 1 && verify_width <= 4,
            3 | 4 => verify_width == 2,
            _ => false,
        },
        ExactBatchedVerifyProfile::Affine8Moe => {
            batch_width > 0 && batch_width <= 8 && verify_width == 2
        }
    }
}

pub(crate) fn project_positions_isolated_on(
    hidden: &Array,
    target: StreamOrDevice,
    mut project: impl FnMut(&Array, StreamOrDevice) -> Result<Array>,
) -> Result<Array> {
    let shape = hidden.shape();
    let shape = shape.as_slice();
    let Some(&[batch, sequence, hidden_size]) = <&[i32; 3]>::try_from(shape).ok() else {
        return Err(anyhow!(
            "exact Qwen verify projection requires [B,Q,H], got {shape:?}"
        ));
    };

    let mut positions = Vec::with_capacity(sequence as usize);
    for position in 0..sequence {
        let position_hidden = mlx::ops::indexing::slice_strided_on(
            hidden,
            &[0_i32, position, 0][..],
            &[batch, position + 1, hidden_size][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        positions.push(project(&position_hidden, target)?);
    }
    let position_refs = positions.iter().collect::<Vec<_>>();
    mlx::ops::shape::concatenate_on(&position_refs, 1, target).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_batched_verify_qualification_is_shape_and_precision_scoped() {
        let affine4 = Some(QuantMeta {
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        });
        let affine8 = Some(QuantMeta {
            group_size: 64,
            bits: 8,
            mode: QuantMode::Affine,
        });
        let affine5 = Some(QuantMeta {
            group_size: 64,
            bits: 5,
            mode: QuantMode::Affine,
        });
        let affine6 = Some(QuantMeta {
            group_size: 64,
            bits: 6,
            mode: QuantMode::Affine,
        });
        assert_eq!(
            dense_exact_batched_verify_profile(None, Some("bfloat16")),
            ExactBatchedVerifyProfile::Disabled
        );
        assert_eq!(
            dense_exact_batched_verify_profile(affine4, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine4
        );
        assert_eq!(
            dense_exact_batched_verify_profile(affine8, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine8Dense
        );
        assert_eq!(
            dense_exact_batched_verify_profile(affine5, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine5Dense
        );
        assert_eq!(
            dense_exact_batched_verify_profile(affine6, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine6Dense
        );
        assert_eq!(
            moe_exact_batched_verify_profile(affine5, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine5Moe
        );
        assert_eq!(
            moe_exact_batched_verify_profile(affine6, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine6Moe
        );
        assert_eq!(
            moe_exact_batched_verify_profile(affine8, Some("bfloat16")),
            ExactBatchedVerifyProfile::Affine8Moe
        );
        assert_eq!(
            dense_exact_batched_verify_profile(affine4, Some("float16")),
            ExactBatchedVerifyProfile::Disabled
        );
        assert!(exact_batched_verify_qualified(
            ExactBatchedVerifyProfile::Affine4,
            8,
            4096,
            5
        ));
        assert!(!exact_batched_verify_qualified(
            ExactBatchedVerifyProfile::Disabled,
            8,
            4096,
            5
        ));
        assert!(!exact_batched_verify_qualified(
            ExactBatchedVerifyProfile::Affine4,
            8,
            4097,
            5
        ));
        for profile in [
            ExactBatchedVerifyProfile::Affine5Dense,
            ExactBatchedVerifyProfile::Affine6Dense,
        ] {
            assert!(exact_batched_verify_qualified(profile, 1, 4_096, 5));
            assert!(exact_batched_verify_qualified(profile, 2, 4_096, 4));
            assert!(exact_batched_verify_qualified(profile, 4, 4_096, 2));
            assert!(!exact_batched_verify_qualified(profile, 2, 4_096, 5));
            assert!(!exact_batched_verify_qualified(profile, 4, 4_096, 4));
            assert!(!exact_batched_verify_qualified(profile, 8, 4_096, 2));
            assert!(!exact_batched_verify_qualified(profile, 1, 4_097, 5));
        }
        for profile in [
            ExactBatchedVerifyProfile::Affine5Moe,
            ExactBatchedVerifyProfile::Affine6Moe,
        ] {
            assert!(exact_batched_verify_qualified(profile, 8, 1_024, 5));
            assert!(!exact_batched_verify_qualified(profile, 8, 1_025, 5));
        }

        let affine8 = ExactBatchedVerifyProfile::Affine8Dense;
        assert!(exact_batched_verify_shape_qualified(affine8, 1, 5));
        assert!(exact_batched_verify_shape_qualified(affine8, 2, 4));
        assert!(exact_batched_verify_shape_qualified(affine8, 3, 2));
        assert!(exact_batched_verify_shape_qualified(affine8, 4, 2));
        assert!(!exact_batched_verify_shape_qualified(affine8, 2, 5));
        assert!(!exact_batched_verify_shape_qualified(affine8, 4, 3));
        assert!(!exact_batched_verify_shape_qualified(affine8, 8, 2));
        assert!(!exact_batched_verify_shape_qualified(affine8, 1, 1));

        let affine8_moe = ExactBatchedVerifyProfile::Affine8Moe;
        assert!(exact_batched_verify_qualified(affine8_moe, 8, 1_024, 2));
        assert!(!exact_batched_verify_qualified(affine8_moe, 8, 1_025, 2));
        assert!(!exact_batched_verify_qualified(affine8_moe, 8, 1_024, 4));

        let affine4 = ExactBatchedVerifyProfile::Affine4;
        assert!(exact_batched_verify_shape_qualified(affine4, 8, 5));
        assert!(!exact_batched_verify_shape_qualified(affine4, 9, 5));
        assert!(!exact_batched_verify_shape_qualified(affine4, 8, 6));
    }
}
