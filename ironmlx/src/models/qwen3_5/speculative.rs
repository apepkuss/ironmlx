use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::{QuantMeta, QuantMode};
use crate::Result;

pub(crate) fn exact_batched_verify_precision_qualified(
    quant_meta: Option<QuantMeta>,
    checkpoint_dtype: Option<&str>,
) -> bool {
    checkpoint_dtype == Some("bfloat16")
        && match quant_meta {
            Some(QuantMeta {
                group_size: 64,
                bits: 4,
                mode: QuantMode::Affine,
            }) => true,
            None | Some(_) => false,
        }
}

pub(crate) fn exact_batched_verify_qualified(
    precision_qualified: bool,
    batch_width: usize,
    context_tokens: usize,
    verify_width: usize,
) -> bool {
    const MAX_QUALIFIED_CONTEXT_TOKENS: usize = 4096;

    precision_qualified
        && exact_batched_verify_shape_qualified(batch_width, verify_width)
        && context_tokens <= MAX_QUALIFIED_CONTEXT_TOKENS
}

pub(crate) fn exact_batched_verify_shape_qualified(
    batch_width: usize,
    verify_width: usize,
) -> bool {
    const MAX_QUALIFIED_BATCH: usize = 8;
    const MAX_QUALIFIED_VERIFY_WIDTH: usize = 5;

    batch_width > 0
        && batch_width <= MAX_QUALIFIED_BATCH
        && verify_width > 1
        && verify_width <= MAX_QUALIFIED_VERIFY_WIDTH
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
        assert!(!exact_batched_verify_precision_qualified(
            None,
            Some("bfloat16")
        ));
        assert!(exact_batched_verify_precision_qualified(
            affine4,
            Some("bfloat16")
        ));
        assert!(!exact_batched_verify_precision_qualified(
            affine4,
            Some("float16")
        ));
        assert!(!exact_batched_verify_precision_qualified(
            Some(QuantMeta {
                group_size: 64,
                bits: 6,
                mode: QuantMode::Affine,
            }),
            Some("bfloat16"),
        ));
        assert!(exact_batched_verify_qualified(true, 8, 4096, 5));
        assert!(!exact_batched_verify_qualified(false, 8, 4096, 5));
        assert!(!exact_batched_verify_qualified(true, 0, 4096, 5));
        assert!(!exact_batched_verify_qualified(true, 9, 4096, 5));
        assert!(!exact_batched_verify_qualified(true, 8, 4097, 5));
        assert!(!exact_batched_verify_qualified(true, 8, 4096, 1));
        assert!(!exact_batched_verify_qualified(true, 8, 4096, 6));
        assert!(exact_batched_verify_shape_qualified(8, 5));
        assert!(!exact_batched_verify_shape_qualified(9, 5));
        assert!(!exact_batched_verify_shape_qualified(8, 6));
    }
}
