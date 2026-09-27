//! Qwen3.8-27B DFlash2 lane execution profile.
//!
//! This module is the single qualification boundary for the model-family
//! pack.  It intentionally does not affect ordinary Q=1 decoding.  Wide
//! verification gets the row-invariant product-stable QMM, the bulk
//! position-stable attention/cache route, transactional GDN replay, prepared
//! fused projection storage, and bounded layer graph submission as one
//! versioned unit.

use crate::models::dflash2::{DFlash2LaneKernelPack, DFlash2TargetForwardMode};

use super::config::Qwen35Config;
use super::speculative::ExactBatchedVerifyProfile;

const QWEN38_27B_HIDDEN: i32 = 5_120;
const QWEN38_27B_INTERMEDIATE: i32 = 17_408;
const QWEN38_27B_LAYERS: i32 = 64;
const QWEN38_27B_FULL_ATTN_INTERVAL: i32 = 4;
const LAYER_SUBMIT_INTERVAL: usize = 4;

pub(crate) fn is_qwen38_27b(cfg: &Qwen35Config) -> bool {
    cfg.hidden_size == QWEN38_27B_HIDDEN
        && cfg.intermediate_size == QWEN38_27B_INTERMEDIATE
        && cfg.num_hidden_layers == QWEN38_27B_LAYERS
        && cfg.full_attention_interval == QWEN38_27B_FULL_ATTN_INTERVAL
}

pub(crate) fn kernel_pack(
    cfg: &Qwen35Config,
    profile: ExactBatchedVerifyProfile,
) -> Option<DFlash2LaneKernelPack> {
    if !is_qwen38_27b(cfg) {
        return None;
    }
    let (quant_bits, max_lanes) = match profile {
        ExactBatchedVerifyProfile::Affine4 => (4, 64),
        ExactBatchedVerifyProfile::Affine8Dense => (8, 16),
        _ => return None,
    };
    Some(DFlash2LaneKernelPack {
        family: "qwen3.8-27b-dflash2".to_owned(),
        revision: 1,
        quant_bits,
        quant_group_size: 64,
        max_lanes,
        // Quantized projection rows are physically coalesced once at load,
        // then split into shared views for Q1.  Wide QMM streams that single
        // packed allocation once for every lane without retaining a second
        // full checkpoint-sized copy.
        prepared_layout: "fused-output-tiles-v1".to_owned(),
        attention_layout: "bulk-position-stable-v1".to_owned(),
        state_layout: "logical-kv-gdn-prefix-v2".to_owned(),
        layer_submit_interval: LAYER_SUBMIT_INTERVAL,
    })
}

pub(crate) fn layer_submit_interval(
    cfg: &Qwen35Config,
    profile: ExactBatchedVerifyProfile,
    mode: DFlash2TargetForwardMode,
    verify_width: usize,
) -> Option<usize> {
    if !mode.is_verify() || verify_width <= 1 {
        return None;
    }
    kernel_pack(cfg, profile).map(|pack| pack.layer_submit_interval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::config::RopeParams;

    fn qwen38_cfg() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 5_120,
            intermediate_size: 17_408,
            num_hidden_layers: 64,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            head_dim: Some(128),
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: false,
            full_attention_interval: 4,
            mtp_num_hidden_layers: 0,
            linear_num_value_heads: 64,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_parameters: RopeParams::default(),
            vision_config: None,
            max_position_embeddings: 262_144,
        }
    }

    #[test]
    fn lane_pack_is_strictly_qwen38_and_precision_scoped() {
        let cfg = qwen38_cfg();
        let affine4 = kernel_pack(&cfg, ExactBatchedVerifyProfile::Affine4).unwrap();
        assert_eq!(affine4.quant_bits, 4);
        assert!(affine4.supports(8, 8));
        assert!(!affine4.supports(9, 8));

        let affine8 = kernel_pack(&cfg, ExactBatchedVerifyProfile::Affine8Dense).unwrap();
        assert!(affine8.supports(4, 4));
        assert!(!affine8.supports(4, 5));
        assert!(kernel_pack(&cfg, ExactBatchedVerifyProfile::Affine5Dense).is_none());

        let mut other = qwen38_cfg();
        other.hidden_size = 4_096;
        assert!(kernel_pack(&other, ExactBatchedVerifyProfile::Affine4).is_none());
    }

    #[test]
    fn layered_submission_is_verify_only() {
        let cfg = qwen38_cfg();
        assert_eq!(
            layer_submit_interval(
                &cfg,
                ExactBatchedVerifyProfile::Affine4,
                DFlash2TargetForwardMode::GreedyVerify,
                8,
            ),
            Some(4)
        );
        assert_eq!(
            layer_submit_interval(
                &cfg,
                ExactBatchedVerifyProfile::Affine4,
                DFlash2TargetForwardMode::Prefill,
                8,
            ),
            None
        );
        assert_eq!(
            layer_submit_interval(
                &cfg,
                ExactBatchedVerifyProfile::Affine4,
                DFlash2TargetForwardMode::OrdinaryDecode,
                1,
            ),
            None
        );
    }
}
