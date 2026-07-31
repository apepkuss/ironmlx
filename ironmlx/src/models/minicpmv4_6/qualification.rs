use mlx::Dtype;

use crate::core::{QuantMeta, QuantMode};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PromptLookupVerifyProfile {
    Disabled,
    Bfloat16,
    Affine4,
    Affine5,
    Affine6,
    Affine8,
}

pub(crate) fn prompt_lookup_verify_profile(
    quant_meta: Option<QuantMeta>,
    hidden_dtype: Dtype,
) -> PromptLookupVerifyProfile {
    if hidden_dtype != Dtype::Bfloat16 {
        return PromptLookupVerifyProfile::Disabled;
    }
    match quant_meta {
        None => PromptLookupVerifyProfile::Bfloat16,
        Some(QuantMeta {
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        }) => PromptLookupVerifyProfile::Affine4,
        Some(QuantMeta {
            group_size: 64,
            bits: 5,
            mode: QuantMode::Affine,
        }) => PromptLookupVerifyProfile::Affine5,
        Some(QuantMeta {
            group_size: 64,
            bits: 6,
            mode: QuantMode::Affine,
        }) => PromptLookupVerifyProfile::Affine6,
        Some(QuantMeta {
            group_size: 64,
            bits: 8,
            mode: QuantMode::Affine,
        }) => PromptLookupVerifyProfile::Affine8,
        Some(_) => PromptLookupVerifyProfile::Disabled,
    }
}

pub(crate) fn sequential_prompt_lookup_verify_qualified(
    profile: PromptLookupVerifyProfile,
    batch_width: usize,
    context_tokens: usize,
    verify_width: usize,
) -> bool {
    if !(1..=8).contains(&batch_width) || !(2..=5).contains(&verify_width) {
        return false;
    }
    let max_context_tokens = match profile {
        PromptLookupVerifyProfile::Disabled | PromptLookupVerifyProfile::Affine6 => return false,
        PromptLookupVerifyProfile::Affine4 => 1_024,
        PromptLookupVerifyProfile::Bfloat16
        | PromptLookupVerifyProfile::Affine5
        | PromptLookupVerifyProfile::Affine8 => 4_096,
    };
    context_tokens <= max_context_tokens
}

pub(crate) fn exact_batched_verify_qualified(
    profile: PromptLookupVerifyProfile,
    batch_width: usize,
    context_tokens: usize,
    verify_width: usize,
) -> bool {
    let max_context_tokens = match profile {
        PromptLookupVerifyProfile::Disabled | PromptLookupVerifyProfile::Affine6 => return false,
        PromptLookupVerifyProfile::Affine4 => 1_024,
        PromptLookupVerifyProfile::Bfloat16
        | PromptLookupVerifyProfile::Affine5
        | PromptLookupVerifyProfile::Affine8 => 4_096,
    };
    context_tokens <= max_context_tokens
        && exact_batched_verify_shape_qualified(profile, batch_width, verify_width)
}

pub(crate) fn exact_batched_verify_shape_qualified(
    profile: PromptLookupVerifyProfile,
    batch_width: usize,
    verify_width: usize,
) -> bool {
    profile != PromptLookupVerifyProfile::Disabled
        && profile != PromptLookupVerifyProfile::Affine6
        && (1..=8).contains(&batch_width)
        && (2..=5).contains(&verify_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affine(bits: i32) -> Option<QuantMeta> {
        Some(QuantMeta {
            group_size: 64,
            bits,
            mode: QuantMode::Affine,
        })
    }

    #[test]
    fn supported_bfloat16_and_affine_contracts_are_recognized_strictly() {
        assert_eq!(
            prompt_lookup_verify_profile(None, Dtype::Bfloat16),
            PromptLookupVerifyProfile::Bfloat16
        );
        assert_eq!(
            prompt_lookup_verify_profile(affine(4), Dtype::Bfloat16),
            PromptLookupVerifyProfile::Affine4
        );
        assert_eq!(
            prompt_lookup_verify_profile(affine(5), Dtype::Bfloat16),
            PromptLookupVerifyProfile::Affine5
        );
        assert_eq!(
            prompt_lookup_verify_profile(affine(6), Dtype::Bfloat16),
            PromptLookupVerifyProfile::Affine6
        );
        assert_eq!(
            prompt_lookup_verify_profile(affine(8), Dtype::Bfloat16),
            PromptLookupVerifyProfile::Affine8
        );
    }

    #[test]
    fn unknown_precision_contracts_fail_closed() {
        assert_eq!(
            prompt_lookup_verify_profile(None, Dtype::Float16),
            PromptLookupVerifyProfile::Disabled
        );
        assert_eq!(
            prompt_lookup_verify_profile(
                Some(QuantMeta {
                    group_size: 32,
                    bits: 4,
                    mode: QuantMode::Affine,
                }),
                Dtype::Bfloat16,
            ),
            PromptLookupVerifyProfile::Disabled
        );
        assert_eq!(
            prompt_lookup_verify_profile(
                Some(QuantMeta {
                    group_size: 64,
                    bits: 4,
                    mode: QuantMode::Mxfp4,
                }),
                Dtype::Bfloat16,
            ),
            PromptLookupVerifyProfile::Disabled
        );
    }

    #[test]
    fn qualification_is_shape_and_context_scoped() {
        for profile in [
            PromptLookupVerifyProfile::Bfloat16,
            PromptLookupVerifyProfile::Affine5,
            PromptLookupVerifyProfile::Affine8,
        ] {
            assert!(sequential_prompt_lookup_verify_qualified(
                profile, 1, 4_096, 2
            ));
            assert!(sequential_prompt_lookup_verify_qualified(
                profile, 8, 4_096, 5
            ));
            assert!(!sequential_prompt_lookup_verify_qualified(
                profile, 8, 4_097, 5
            ));
        }

        assert!(sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Affine4,
            8,
            1_024,
            5,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Affine4,
            8,
            1_025,
            5,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Disabled,
            1,
            128,
            2,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Affine6,
            1,
            128,
            2,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Bfloat16,
            0,
            128,
            2,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Bfloat16,
            9,
            128,
            2,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Bfloat16,
            1,
            128,
            1,
        ));
        assert!(!sequential_prompt_lookup_verify_qualified(
            PromptLookupVerifyProfile::Bfloat16,
            1,
            128,
            6,
        ));
    }

    #[test]
    fn exact_qualification_is_shape_context_and_precision_scoped() {
        for profile in [
            PromptLookupVerifyProfile::Bfloat16,
            PromptLookupVerifyProfile::Affine4,
            PromptLookupVerifyProfile::Affine5,
            PromptLookupVerifyProfile::Affine8,
        ] {
            assert!(exact_batched_verify_qualified(profile, 1, 128, 2));
            assert!(exact_batched_verify_qualified(profile, 8, 128, 5));
            assert!(!exact_batched_verify_qualified(profile, 0, 128, 2));
            assert!(!exact_batched_verify_qualified(profile, 9, 128, 2));
            assert!(!exact_batched_verify_qualified(profile, 1, 128, 1));
            assert!(!exact_batched_verify_qualified(profile, 1, 128, 6));
        }
        assert!(exact_batched_verify_qualified(
            PromptLookupVerifyProfile::Affine4,
            8,
            1_024,
            5,
        ));
        assert!(!exact_batched_verify_qualified(
            PromptLookupVerifyProfile::Affine4,
            8,
            1_025,
            5,
        ));
        assert!(!exact_batched_verify_qualified(
            PromptLookupVerifyProfile::Affine6,
            1,
            128,
            2,
        ));
    }
}
