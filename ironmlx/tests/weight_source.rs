//! Layer construction must work without the model-specific Loader.

use std::collections::HashMap;

use ironmlx::core::weights::{QuantMeta, QuantMode, WeightMap, WeightSource};
use ironmlx::nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, RmsNorm, RmsNormGated};
use ironmlx::Result;
use mlx::Array;

#[derive(Default)]
struct MemoryWeights {
    tensors: HashMap<String, Array>,
    quant: HashMap<String, QuantMeta>,
}

impl WeightSource for MemoryWeights {
    fn tensor(&self, key: &str) -> Result<&Array> {
        self.tensor_opt(key)
            .ok_or_else(|| anyhow::anyhow!("memory source missing {key}"))
    }

    fn tensor_opt(&self, key: &str) -> Option<&Array> {
        self.tensors.get(key)
    }

    fn quant_meta_for(&self, prefix: &str) -> Option<QuantMeta> {
        self.quant.get(prefix).copied()
    }
}

fn array<T: mlx::Element>(data: &[T], shape: impl mlx::IntoShape) -> Array {
    Array::try_from((data, shape)).unwrap()
}

fn close(actual: &Array, expected: &[f32]) {
    let values = actual.to_vec::<f32>().unwrap();
    assert_eq!(values.len(), expected.len());
    for (a, b) in values.iter().zip(expected) {
        assert!((a - b).abs() <= 1e-5, "{a} != {b}");
    }
}

#[test]
fn fp_layers_accept_an_independent_weight_source() {
    let source = MemoryWeights {
        tensors: HashMap::from([
            (
                "projection.weight".into(),
                array(&[1.0_f32, 2.0, 3.0, 4.0], (2, 2)),
            ),
            ("projection.bias".into(), array(&[0.5_f32, -0.5], (2,))),
            ("norm.weight".into(), array(&[1.0_f32, 1.0], (2,))),
            ("conv.weight".into(), array(&[1.0_f32, 2.0], (1, 2, 1))),
        ]),
        ..Default::default()
    };
    let x = array(&[2.0_f32, 3.0], (1, 2));
    // Exercise both statically dispatched providers and an erased provider.
    let linear = Linear::from_loader(&source, "projection").unwrap();
    close(&linear.forward(&x).unwrap(), &[8.5, 17.5]);
    let erased: &dyn WeightSource = &source;
    let embedding = Embedding::from_loader(erased, "projection").unwrap();
    let ids = array(&[1_u32, 0], (2,));
    close(&embedding.forward(&ids).unwrap(), &[3.0, 4.0, 1.0, 2.0]);

    let rms = RmsNorm::from_loader(erased, "norm", 1e-5).unwrap();
    let scale = (6.5_f32 + 1e-5).sqrt();
    close(&rms.forward(&x).unwrap(), &[2.0 / scale, 3.0 / scale]);
    let gated = RmsNormGated::from_loader(erased, "norm", 1e-5).unwrap();
    close(
        &gated.forward(&x, None).unwrap(),
        &[2.0 / scale, 3.0 / scale],
    );
    let norm = LayerNorm::from_loader(erased, "norm", 1e-5).unwrap();
    let scale = (0.25_f32 + 1e-5).sqrt();
    close(&norm.forward(&x).unwrap(), &[-0.5 / scale, 0.5 / scale]);

    let conv = Conv1d::from_loader(
        erased,
        "conv",
        Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
    )
    .unwrap();
    let input = array(&[1.0_f32, 2.0, 3.0], (1, 3, 1));
    close(&conv.forward(&input).unwrap(), &[5.0, 8.0]);
}

#[test]
fn quantized_layers_use_metadata_for_each_prefix() {
    let mut source = MemoryWeights::default();
    let raw = array(
        &(0..128)
            .map(|i| (i as f32 * 0.13).sin())
            .collect::<Vec<_>>(),
        (2, 64),
    );
    for bits in [4, 8] {
        let prefix = format!("q{bits}");
        let q = mlx::quantization::quantize(&raw, Some(64), Some(bits), "affine", None).unwrap();
        for (suffix, tensor) in ["weight", "scales", "biases"].into_iter().zip(q) {
            source.tensors.insert(format!("{prefix}.{suffix}"), tensor);
        }
        source.quant.insert(
            prefix,
            QuantMeta {
                group_size: 64,
                bits,
                mode: QuantMode::Affine,
            },
        );
    }
    let input = array(&[0.25_f32; 64], (1, 64));
    for bits in [4, 8] {
        let prefix = format!("q{bits}");
        let weight = source.tensor(&format!("{prefix}.weight")).unwrap();
        let scales = source.tensor(&format!("{prefix}.scales")).unwrap();
        let biases = source.tensor(&format!("{prefix}.biases")).unwrap();
        let expected = mlx::quantization::quantized_matmul(
            &input,
            weight,
            scales,
            Some(biases),
            true,
            Some(64),
            Some(bits),
            "affine",
        )
        .unwrap();
        let linear = Linear::from_loader(&source, &prefix).unwrap();
        assert_eq!(linear.in_features(), 64);
        close(
            &linear.forward(&input).unwrap(),
            &expected.to_vec::<f32>().unwrap(),
        );
        let embedding = Embedding::from_loader(&source, &prefix).unwrap();
        let decoded = mlx::quantization::dequantize(
            weight,
            scales,
            Some(biases),
            Some(64),
            Some(bits),
            "affine",
            None,
            None,
        )
        .unwrap();
        let ids = array(&[1_u32], (1,));
        let expected = decoded.take(&ids, 0).unwrap();
        close(
            &embedding.forward(&ids).unwrap(),
            &expected.to_vec::<f32>().unwrap(),
        );
    }
}

#[test]
fn missing_tensor_preserves_provider_error_context() {
    let source = MemoryWeights::default();
    let error = Linear::from_loader(&source, "absent").err().unwrap();
    assert_eq!(error.to_string(), "memory source missing absent.weight");
}

#[test]
fn quantized_storage_is_validated_for_independent_providers() {
    let mut source = MemoryWeights {
        tensors: HashMap::from([
            ("q.weight".into(), array(&[0_u32; 8], (1, 8))),
            ("q.scales".into(), array(&[1.0_f32], (1, 1))),
        ]),
        ..Default::default()
    };
    assert!(Linear::from_loader(&source, "q")
        .err()
        .unwrap()
        .to_string()
        .contains("no quantization meta"));
    source.quant.insert(
        "q".into(),
        QuantMeta {
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        },
    );
    assert!(Linear::from_loader(&source, "q")
        .err()
        .unwrap()
        .to_string()
        .contains("requires quantization biases"));
    assert!(Embedding::from_loader(&source, "q")
        .err()
        .unwrap()
        .to_string()
        .contains("requires quantization biases"));
}

#[test]
fn owned_weight_map_preserves_layer_parameters_after_projection_release() {
    let global = QuantMeta {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Affine,
    };
    let override_meta = QuantMeta { bits: 8, ..global };
    let mut weights = WeightMap::new(
        HashMap::from([
            ("projection.weight".into(), array(&[1.0_f32, 2.0], (1, 2))),
            ("other.weight".into(), array(&[3.0_f32], (1, 1))),
        ]),
        Some(global),
        HashMap::from([("other".into(), override_meta)]),
    );
    assert_eq!(weights.quant_meta_for("projection"), Some(global));
    assert_eq!(weights.quant_meta_for("other"), Some(override_meta));
    let projection = Linear::from_loader(&weights, "projection").unwrap();
    weights.retain(|key, _| !key.starts_with("projection."));
    assert!(!weights.contains("projection.weight"));
    assert!(weights.contains("other.weight"));
    close(
        &projection.forward(&array(&[2.0_f32, 3.0], (1, 2))).unwrap(),
        &[8.0],
    );
}
