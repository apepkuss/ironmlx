//! Model-specific tied-output routing over a shared embedding table.
use crate::core::weights::{QuantMode, WeightSource};
use crate::Result;
use mlx::{Array, Dtype, StreamOrDevice};

pub struct Embedding {
    inner: ironmlx_core::nn::Embedding,
}
impl Embedding {
    pub fn from_loader(loader: &(impl WeightSource + ?Sized), prefix: &str) -> Result<Self> {
        Ok(Self {
            inner: ironmlx_core::nn::Embedding::from_loader(loader, prefix)?,
        })
    }
    pub fn output_dtype(&self) -> Dtype {
        self.inner.output_dtype()
    }
    pub fn forward(&self, tokens: &Array) -> Result<Array> {
        self.forward_on(tokens, ())
    }
    pub fn forward_on(&self, tokens: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.inner.forward_on(tokens, target)
    }
    pub fn as_output(&self, hidden: &Array) -> Result<Array> {
        self.as_output_on(hidden, ())
    }
    pub fn as_output_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        if let Some(parts) = self.inner.quantized_parts() {
            let product_stable = super::product_stable_qmm::is_armed()
                && hidden.ndim() == 3
                && hidden.shape().as_slice()[..2].iter().product::<i32>() > 1
                && matches!(parts.bits, 4 | 5 | 6 | 8)
                && parts.mode == QuantMode::Affine;
            if product_stable {
                return super::product_stable_qmm::forward_on(
                    hidden,
                    parts.weight,
                    parts.scales,
                    parts.biases,
                    true,
                    parts.group_size,
                    parts.bits,
                    parts.mode.mlx_backend_mode(),
                    target,
                );
            }
        }
        self.inner.as_output_on(hidden, target)
    }
    pub(crate) fn dense_weight_on(&self, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.inner.dense_weight_on(target)
    }
    #[cfg(test)]
    #[doc(hidden)]
    pub fn from_components_fp_for_test(weight: Array) -> Self {
        Self {
            inner: ironmlx_core::nn::Embedding::new_fp(weight),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::weights::{QuantMeta, WeightMap};
    use std::collections::HashMap;

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn shared_embedding_wrapper_preserves_product_stable_tied_output() {
        for bits in [4, 5, 6, 8] {
            let values: Vec<f32> = (0..32 * 64)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.017)
                .collect();
            let weight: Array = (values.as_slice(), (32, 64)).try_into().unwrap();
            let weight = weight.astype(Dtype::Bfloat16).unwrap();
            let quant =
                mlx::quantization::quantize(&weight, Some(64), Some(bits), "affine", None).unwrap();
            let source = WeightMap::new(
                HashMap::from([
                    ("embed.weight".into(), quant[0].clone()),
                    ("embed.scales".into(), quant[1].clone()),
                    ("embed.biases".into(), quant[2].clone()),
                ]),
                Some(QuantMeta {
                    group_size: 64,
                    bits,
                    mode: QuantMode::Affine,
                }),
                HashMap::new(),
            );
            let layer = Embedding::from_loader(&source, "embed").unwrap();
            let base = ironmlx_core::nn::Embedding::from_loader(&source, "embed").unwrap();
            for (batch, sequence) in [(1, 3), (2, 2)] {
                let input: Vec<f32> = (0..batch * sequence * 64)
                    .map(|i| ((i % 17) as f32 - 8.0) * 0.021)
                    .collect();
                let hidden: Array = (input.as_slice(), (batch, sequence, 64))
                    .try_into()
                    .unwrap();
                let hidden = hidden.astype(Dtype::Bfloat16).unwrap();
                let plain = layer.as_output(&hidden).unwrap();
                assert_eq!(
                    plain
                        .astype(Dtype::Float32)
                        .unwrap()
                        .to_vec::<f32>()
                        .unwrap(),
                    base.as_output(&hidden)
                        .unwrap()
                        .astype(Dtype::Float32)
                        .unwrap()
                        .to_vec::<f32>()
                        .unwrap()
                );
                let flat = hidden.reshape((batch * sequence, 1, 64)).unwrap();
                let mut reference = Vec::new();
                for index in 0..batch * sequence {
                    let row = flat.slice([index, 0, 0], [index + 1, 1, 64]).unwrap();
                    reference.extend(
                        base.as_output(&row)
                            .unwrap()
                            .astype(Dtype::Float32)
                            .unwrap()
                            .to_vec::<f32>()
                            .unwrap(),
                    );
                }
                let _scope = super::super::product_stable_qmm::scope();
                let actual = layer
                    .as_output(&hidden)
                    .unwrap()
                    .astype(Dtype::Float32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                assert_eq!(actual, reference, "bits={bits}, B={batch}, Q={sequence}");
            }
        }
    }
}
