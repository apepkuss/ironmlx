use crate::core::{QuantMeta, QuantMode};

pub(crate) fn exact_batched_verify_precision_qualified(
    lm_head_quant_meta: Option<QuantMeta>,
    checkpoint_dtype: Option<&str>,
) -> bool {
    checkpoint_dtype == Some("bfloat16")
        && matches!(
            lm_head_quant_meta,
            Some(QuantMeta {
                group_size: 64,
                bits: 8,
                mode: QuantMode::Affine,
            })
        )
}

pub(crate) fn exact_batched_verify_qualified(
    precision_qualified: bool,
    batch_width: usize,
    context_tokens: usize,
    verify_width: usize,
) -> bool {
    const MAX_QUALIFIED_BATCH: usize = 8;
    const MAX_QUALIFIED_CONTEXT_TOKENS: usize = 1_024;
    const MAX_QUALIFIED_VERIFY_WIDTH: usize = 5;

    precision_qualified
        && batch_width > 0
        && batch_width <= MAX_QUALIFIED_BATCH
        && context_tokens <= MAX_QUALIFIED_CONTEXT_TOKENS
        && verify_width > 1
        && verify_width <= MAX_QUALIFIED_VERIFY_WIDTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_batched_verify_qualification_is_shape_and_precision_scoped() {
        let affine8 = Some(QuantMeta {
            group_size: 64,
            bits: 8,
            mode: QuantMode::Affine,
        });
        assert!(exact_batched_verify_precision_qualified(
            affine8,
            Some("bfloat16")
        ));
        assert!(!exact_batched_verify_precision_qualified(
            affine8,
            Some("float16")
        ));
        assert!(!exact_batched_verify_precision_qualified(
            None,
            Some("bfloat16")
        ));
        assert!(!exact_batched_verify_precision_qualified(
            Some(QuantMeta {
                group_size: 64,
                bits: 4,
                mode: QuantMode::Affine,
            }),
            Some("bfloat16")
        ));
        for bits in [5, 6] {
            assert!(!exact_batched_verify_precision_qualified(
                Some(QuantMeta {
                    group_size: 64,
                    bits,
                    mode: QuantMode::Affine,
                }),
                Some("bfloat16")
            ));
        }
        assert!(!exact_batched_verify_precision_qualified(
            Some(QuantMeta {
                group_size: 64,
                bits: 8,
                mode: QuantMode::OptiQ,
            }),
            Some("bfloat16")
        ));

        assert!(exact_batched_verify_qualified(true, 8, 1_024, 5));
        assert!(!exact_batched_verify_qualified(false, 8, 1_024, 5));
        assert!(!exact_batched_verify_qualified(true, 0, 1_024, 5));
        assert!(!exact_batched_verify_qualified(true, 9, 1_024, 5));
        assert!(!exact_batched_verify_qualified(true, 8, 1_025, 5));
        assert!(!exact_batched_verify_qualified(true, 8, 1_024, 1));
        assert!(!exact_batched_verify_qualified(true, 8, 1_024, 6));
    }
}
