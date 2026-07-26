use crate::core::{QuantMeta, QuantMode};

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
    const MAX_QUALIFIED_VERIFY_WIDTH: usize = 5;

    precision_qualified
        && batch_width == 1
        && context_tokens <= MAX_QUALIFIED_CONTEXT_TOKENS
        && verify_width > 1
        && verify_width <= MAX_QUALIFIED_VERIFY_WIDTH
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
        assert!(exact_batched_verify_qualified(true, 1, 4096, 5));
        assert!(!exact_batched_verify_qualified(false, 1, 4096, 5));
        assert!(!exact_batched_verify_qualified(true, 0, 4096, 5));
        assert!(!exact_batched_verify_qualified(true, 2, 4096, 5));
        assert!(!exact_batched_verify_qualified(true, 1, 4097, 5));
        assert!(!exact_batched_verify_qualified(true, 1, 4096, 1));
        assert!(!exact_batched_verify_qualified(true, 1, 4096, 6));
    }
}
