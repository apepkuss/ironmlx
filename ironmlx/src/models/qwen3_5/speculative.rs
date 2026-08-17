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
    _context_tokens: usize,
    verify_width: usize,
) -> bool {
    // Context length does not change the Q>1 execution morphology. Exactness
    // is guaranteed by position-isolated backbone/MoE execution, sequence-stable
    // GatedDelta state transitions, and replaying the final projection as
    // contiguous Q=1 inputs.
    exact_batched_verify_shape_qualified(profile, batch_width, verify_width)
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
            let max_verify_width = if profile == ExactBatchedVerifyProfile::Affine4 {
                8
            } else {
                5
            };
            batch_width > 0
                && batch_width <= 8
                && verify_width > 1
                && verify_width <= max_verify_width
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

pub(crate) fn sequential_prompt_lookup_verify_qualified(
    profile: ExactBatchedVerifyProfile,
    _context_tokens: usize,
) -> bool {
    // Affine4 has a real mixed-source counterexample even with chained
    // transactional Q1 verify. Keep it on the position-isolated exact path at
    // every context length instead of introducing a context-dependent mode
    // switch.
    profile != ExactBatchedVerifyProfile::Affine4
}

pub(crate) fn prompt_lookup_max_draft_tokens(
    profile: ExactBatchedVerifyProfile,
    configured_max_draft_tokens: usize,
) -> usize {
    // Affine4 exact verification is qualified through Q8, which represents
    // the current token plus at most seven copied draft tokens.
    if profile == ExactBatchedVerifyProfile::Affine4 {
        configured_max_draft_tokens.min(7)
    } else {
        configured_max_draft_tokens
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
        )?
        .contiguous_on(false, target)?;
        positions.push(project(&position_hidden, target)?);
    }
    let position_refs = positions.iter().collect::<Vec<_>>();
    mlx::ops::shape::concatenate_on(&position_refs, 1, target).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use mlx::Dtype;
    use serial_test::serial;

    const LONG_CONTEXTS: [usize; 7] = [1_024, 1_025, 4_096, 4_097, 8_192, 32_768, 65_536];

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
        for context_tokens in LONG_CONTEXTS {
            assert!(exact_batched_verify_qualified(
                ExactBatchedVerifyProfile::Affine4,
                8,
                context_tokens,
                8
            ));
            assert!(!exact_batched_verify_qualified(
                ExactBatchedVerifyProfile::Affine4,
                8,
                context_tokens,
                9
            ));
        }
        assert!(!exact_batched_verify_qualified(
            ExactBatchedVerifyProfile::Disabled,
            8,
            4096,
            5
        ));
        assert!(!exact_batched_verify_qualified(
            ExactBatchedVerifyProfile::Affine4,
            9,
            65_536,
            5,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            ExactBatchedVerifyProfile::Affine4,
            1_024
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            ExactBatchedVerifyProfile::Affine4,
            1_025
        ));
        assert!(sequential_prompt_lookup_verify_qualified(
            ExactBatchedVerifyProfile::Affine5Dense,
            4_096
        ));
        assert_eq!(
            prompt_lookup_max_draft_tokens(ExactBatchedVerifyProfile::Affine4, 32),
            7
        );
        assert_eq!(
            prompt_lookup_max_draft_tokens(ExactBatchedVerifyProfile::Affine5Dense, 32),
            32
        );
        for profile in [
            ExactBatchedVerifyProfile::Affine5Dense,
            ExactBatchedVerifyProfile::Affine6Dense,
        ] {
            assert!(exact_batched_verify_qualified(profile, 1, 65_536, 5));
            assert!(exact_batched_verify_qualified(profile, 2, 65_536, 4));
            assert!(exact_batched_verify_qualified(profile, 4, 65_536, 2));
            assert!(!exact_batched_verify_qualified(profile, 2, 65_536, 5));
            assert!(!exact_batched_verify_qualified(profile, 4, 65_536, 4));
            assert!(!exact_batched_verify_qualified(profile, 8, 65_536, 2));
        }
        for profile in [
            ExactBatchedVerifyProfile::Affine5Moe,
            ExactBatchedVerifyProfile::Affine6Moe,
        ] {
            assert!(exact_batched_verify_qualified(profile, 8, 65_536, 5));
            assert!(!exact_batched_verify_qualified(profile, 8, 65_536, 6));
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
        assert!(exact_batched_verify_qualified(affine8_moe, 8, 65_536, 2));
        assert!(!exact_batched_verify_qualified(affine8_moe, 8, 65_536, 4));

        let affine4 = ExactBatchedVerifyProfile::Affine4;
        assert!(exact_batched_verify_shape_qualified(affine4, 8, 8));
        assert!(!exact_batched_verify_shape_qualified(affine4, 9, 8));
        assert!(!exact_batched_verify_shape_qualified(affine4, 8, 9));
    }

    #[test]
    #[serial(mlx_metal)]
    fn position_isolated_projection_matches_contiguous_q1_inputs_exactly() {
        const BATCH: i32 = 2;
        const SEQUENCE: i32 = 5;
        const INPUT_WIDTH: i32 = 64;
        const OUTPUT_WIDTH: i32 = 64;
        const BITS: i32 = 4;
        const GROUP_SIZE: i32 = 64;

        let packed_width = INPUT_WIDTH * BITS / 32;
        let weights = (0..OUTPUT_WIDTH * packed_width)
            .map(|index| (index as u32).wrapping_mul(2_654_435_761).rotate_left(7))
            .collect::<Vec<_>>();
        let scales = (0..OUTPUT_WIDTH)
            .map(|index| 0.004_f32 * (1 + index % 5) as f32)
            .collect::<Vec<_>>();
        let biases = (0..OUTPUT_WIDTH)
            .map(|index| -0.03_f32 + (index % 7) as f32 * 0.002)
            .collect::<Vec<_>>();
        let weights = Array::try_from((weights.as_slice(), &[OUTPUT_WIDTH, packed_width][..]))
            .expect("quantized weights");
        let scales = Array::try_from((scales.as_slice(), &[OUTPUT_WIDTH, 1][..]))
            .expect("quantized scales")
            .astype(Dtype::Bfloat16)
            .expect("bf16 scales");
        let biases = Array::try_from((biases.as_slice(), &[OUTPUT_WIDTH, 1][..]))
            .expect("quantized biases")
            .astype(Dtype::Bfloat16)
            .expect("bf16 biases");
        let linear = Linear::new_quant(weights, scales, Some(biases), None, GROUP_SIZE, BITS);

        let hidden_values = (0..BATCH * SEQUENCE * INPUT_WIDTH)
            .map(|index| ((index * 19 + 7) % 71) as f32 * 0.003 - 0.105)
            .collect::<Vec<_>>();
        let hidden = Array::try_from((
            hidden_values.as_slice(),
            &[BATCH, SEQUENCE, INPUT_WIDTH][..],
        ))
        .expect("hidden input")
        .astype(Dtype::Bfloat16)
        .expect("bf16 hidden input");

        let isolated = project_positions_isolated_on(&hidden, ().into(), |position, target| {
            linear.forward_on(position, target)
        })
        .expect("position-isolated projection");
        let mut reference = Vec::with_capacity(SEQUENCE as usize);
        for position in 0..SEQUENCE {
            let mut values = Vec::with_capacity((BATCH * INPUT_WIDTH) as usize);
            for batch in 0..BATCH {
                let start = ((batch * SEQUENCE + position) * INPUT_WIDTH) as usize;
                values.extend_from_slice(&hidden_values[start..start + INPUT_WIDTH as usize]);
            }
            let input = Array::try_from((values.as_slice(), &[BATCH, 1, INPUT_WIDTH][..]))
                .expect("contiguous Q1 input")
                .astype(Dtype::Bfloat16)
                .expect("bf16 Q1 input");
            reference.push(linear.forward(&input).expect("Q1 projection"));
        }
        let reference = mlx::ops::shape::concatenate(&reference.iter().collect::<Vec<_>>(), 1)
            .expect("concatenated Q1 projection");
        mlx::transforms::eval(&[&isolated, &reference]).expect("evaluate projections");

        let isolated = isolated
            .astype(Dtype::Float32)
            .expect("f32 isolated projection")
            .to_vec::<f32>()
            .expect("isolated projection values");
        let reference = reference
            .astype(Dtype::Float32)
            .expect("f32 Q1 projection")
            .to_vec::<f32>()
            .expect("Q1 projection values");
        assert_eq!(isolated, reference);
    }
}
