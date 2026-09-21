use anyhow::anyhow;

use crate::core::{Loader, QuantMeta};
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FusedQuantCompatibility {
    Dense,
    Quantized(QuantMeta),
    MixedQuantized,
}

pub(super) fn fused_quant_compatibility(
    loader: &Loader,
    prefixes: &[&str],
    context: &str,
) -> Result<FusedQuantCompatibility> {
    let mut metas = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let scales_key = format!("{prefix}.scales");
        let meta = if loader.contains(&scales_key) {
            Some(loader.quant_meta_for(prefix).ok_or_else(|| {
                anyhow!(
                    "{context}: `{scales_key}` present but Loader has no quantization meta for `{prefix}`"
                )
            })?)
        } else {
            None
        };
        metas.push((*prefix, meta));
    }
    classify_fused_quant_metas(&metas, context)
}

fn classify_fused_quant_metas(
    metas: &[(&str, Option<QuantMeta>)],
    context: &str,
) -> Result<FusedQuantCompatibility> {
    let mut first: Option<QuantMeta> = None;
    let mut any_quantized = false;
    let mut any_dense = false;

    for (prefix, maybe_meta) in metas {
        if let Some(meta) = maybe_meta {
            any_quantized = true;
            if let Some(first_meta) = first {
                if first_meta != *meta {
                    return Ok(FusedQuantCompatibility::MixedQuantized);
                }
            } else {
                first = Some(*meta);
            }
        } else {
            let _ = prefix;
            any_dense = true;
        }
    }

    match (any_quantized, any_dense, first) {
        (false, true, None) => Ok(FusedQuantCompatibility::Dense),
        (true, false, Some(meta)) => Ok(FusedQuantCompatibility::Quantized(meta)),
        (true, true, _) => Err(anyhow!(
            "{context}: projection group mixes dense and quantized tensors"
        )),
        _ => Err(anyhow!("{context}: empty projection group")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::QuantMode;

    fn meta(bits: i32, mode: QuantMode) -> QuantMeta {
        QuantMeta {
            group_size: if matches!(mode, QuantMode::Mxfp4 | QuantMode::Mxfp8) {
                32
            } else {
                64
            },
            bits,
            mode,
        }
    }

    #[test]
    fn mixed_optiq_bits_are_not_fused() {
        let q = "model.layers.19.self_attn.q_proj";
        let k = "model.layers.19.self_attn.k_proj";
        let v = "model.layers.19.self_attn.v_proj";
        let metas = [
            (q, Some(meta(4, QuantMode::OptiQ))),
            (k, Some(meta(8, QuantMode::OptiQ))),
            (v, Some(meta(8, QuantMode::OptiQ))),
        ];

        let compat = classify_fused_quant_metas(&metas, "qkv").unwrap();
        assert_eq!(compat, FusedQuantCompatibility::MixedQuantized);
    }

    #[test]
    fn matching_optiq_bits_can_be_fused() {
        let gate = "model.layers.2.mlp.gate_proj";
        let up = "model.layers.2.mlp.up_proj";
        let metas = [
            (gate, Some(meta(4, QuantMode::OptiQ))),
            (up, Some(meta(4, QuantMode::OptiQ))),
        ];

        let compat = classify_fused_quant_metas(&metas, "gate_up").unwrap();
        assert_eq!(
            compat,
            FusedQuantCompatibility::Quantized(meta(4, QuantMode::OptiQ))
        );
    }

    #[test]
    fn matching_affine_5bit_and_6bit_metadata_can_be_fused() {
        for bits in [5, 6] {
            let gate = "model.layers.2.mlp.gate_proj";
            let up = "model.layers.2.mlp.up_proj";
            let metas = [
                (gate, Some(meta(bits, QuantMode::Affine))),
                (up, Some(meta(bits, QuantMode::Affine))),
            ];

            let compat = classify_fused_quant_metas(&metas, "gate_up").unwrap();
            assert_eq!(
                compat,
                FusedQuantCompatibility::Quantized(meta(bits, QuantMode::Affine))
            );
        }
    }

    #[test]
    fn matching_mxfp_metadata_can_be_fused() {
        for (bits, mode) in [(4, QuantMode::Mxfp4), (8, QuantMode::Mxfp8)] {
            let q = "model.layers.1.self_attn.q_proj";
            let k = "model.layers.1.self_attn.k_proj";
            let v = "model.layers.1.self_attn.v_proj";
            let metas = [
                (q, Some(meta(bits, mode))),
                (k, Some(meta(bits, mode))),
                (v, Some(meta(bits, mode))),
            ];

            let compat = classify_fused_quant_metas(&metas, "qkv").unwrap();
            assert_eq!(compat, FusedQuantCompatibility::Quantized(meta(bits, mode)));
        }
    }

    #[test]
    fn affine_and_mxfp_are_never_fused_together() {
        let gate = "model.layers.2.mlp.gate_proj";
        let up = "model.layers.2.mlp.up_proj";
        let metas = [
            (gate, Some(meta(4, QuantMode::Affine))),
            (up, Some(meta(4, QuantMode::Mxfp4))),
        ];

        let compat = classify_fused_quant_metas(&metas, "gate_up").unwrap();
        assert_eq!(compat, FusedQuantCompatibility::MixedQuantized);
    }

    #[test]
    fn partial_quantized_projection_group_is_an_error() {
        let gate = "model.layers.2.mlp.gate_proj";
        let up = "model.layers.2.mlp.up_proj";
        let metas = [(gate, Some(meta(4, QuantMode::OptiQ))), (up, None)];

        let err = classify_fused_quant_metas(&metas, "gate_up")
            .expect_err("partial quantization should fail");
        assert!(err
            .to_string()
            .contains("mixes dense and quantized tensors"));
    }
}
