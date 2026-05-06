//! `nn::Linear` — single struct with private enum dispatch over fp and
//! quantized backends.
//!
//! Construction goes through [`Linear::from_loader`], which probes
//! `{prefix}.scales` to choose the variant. Forward computes
//! `y = x @ W^T + bias` for fp weights, or calls
//! [`mlx::quantization::quantized_matmul`] (with `transpose=true`) for
//! quantized weights, then optionally adds the (non-quantized) `bias`.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::Result;

/// Linear projection layer. Handles both full-precision and quantized
/// weight checkpoints transparently.
pub struct Linear {
    inner: LinearImpl,
}

/// Internal backend variant. Private — callers use [`Linear`].
enum LinearImpl {
    Fp {
        /// `[out, in]` dense weight, dtype as stored in the checkpoint.
        weight: Array,
        /// Optional `[out]` bias.
        bias: Option<Array>,
    },
    Quant {
        /// Packed quantized weight (layout per `mlx::quantization`).
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group zero-points (affine quantization).
        biases: Option<Array>,
        /// Optional Linear bias term, applied after the quantized matmul.
        bias: Option<Array>,
        /// Group size from quantization metadata.
        group_size: i32,
        /// Bits per quantized weight (4 / 6 / 8).
        bits: i32,
    },
}

impl Linear {
    /// Build a `Linear` from `loader`, looking for tensors at
    /// `{prefix}.weight` (required), `{prefix}.bias` (optional),
    /// `{prefix}.scales` (signals quantized variant), and
    /// `{prefix}.biases` (optional zero-points for affine quant).
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let bias_key = format!("{prefix}.bias");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");

        let weight = loader.tensor(&weight_key)?.clone();
        let bias = loader.tensor_opt(&bias_key).cloned();

        if loader.contains(&scales_key) {
            let qmeta = loader.quant_meta().ok_or_else(|| {
                anyhow!(
                    "Linear `{prefix}`: `{scales_key}` present but Loader has no quantization meta"
                )
            })?;
            let scales = loader.tensor(&scales_key)?.clone();
            let biases = loader.tensor_opt(&biases_key).cloned();
            Ok(Linear {
                inner: LinearImpl::Quant {
                    weight,
                    scales,
                    biases,
                    bias,
                    group_size: qmeta.group_size,
                    bits: qmeta.bits,
                },
            })
        } else {
            Ok(Linear {
                inner: LinearImpl::Fp { weight, bias },
            })
        }
    }

    /// Forward pass: `y = x @ W^T (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        match &self.inner {
            LinearImpl::Fp { weight, bias } => {
                let wt = weight.transpose_on(target)?;
                let mut y = x.matmul_on(&wt, target)?;
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
            LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            } => {
                let mut y = mlx::quantization::quantized_matmul_on(
                    x,
                    weight,
                    scales,
                    biases.as_ref(),
                    /* transpose = */ true,
                    Some(*group_size),
                    Some(*bits),
                    "affine",
                    target,
                )?;
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use mlx::Array;

    fn fp_linear(weight: Array, bias: Option<Array>) -> Linear {
        Linear {
            inner: LinearImpl::Fp { weight, bias },
        }
    }

    #[test]
    fn fp_forward_matches_manual_matmul() {
        // weight [out=2, in=3] = [[1,2,3],[4,5,6]]
        // x [batch=1, in=3] = [1,1,1]
        // y = x @ W^T = [[1+2+3, 4+5+6]] = [[6, 15]]
        let weight =
            Array::try_from((&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).unwrap();
        let x = Array::try_from((&[1.0f32, 1.0, 1.0][..], &[1, 3][..])).unwrap();
        let layer = fp_linear(weight, None);

        let y = layer.forward(&x).expect("forward");
        let v = y.to_vec::<f32>().expect("to_vec");
        assert_eq!(v, vec![6.0, 15.0]);
        assert_eq!(y.shape().as_slice(), &[1, 2]);
    }

    #[test]
    fn fp_forward_with_bias() {
        // weight = identity [[1,0],[0,1]], x = [3, 4], bias = [10, 20]
        // y = [3*1+4*0, 3*0+4*1] + [10, 20] = [13, 24]
        let weight = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).unwrap();
        let bias = Array::try_from((&[10.0f32, 20.0][..], &[2][..])).unwrap();
        let x = Array::try_from((&[3.0f32, 4.0][..], &[1, 2][..])).unwrap();
        let layer = fp_linear(weight, Some(bias));

        let y = layer.forward(&x).expect("forward");
        let v = y.to_vec::<f32>().expect("to_vec");
        assert_abs_diff_eq!(v[0], 13.0, epsilon = 1e-6);
        assert_abs_diff_eq!(v[1], 24.0, epsilon = 1e-6);
    }

    #[test]
    fn fp_dtype_preserved() {
        let weight = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).unwrap();
        let x = Array::try_from((&[1.0f32, 2.0][..], &[1, 2][..])).unwrap();
        let layer = fp_linear(weight, None);

        let y = layer.forward(&x).expect("forward");
        assert_eq!(x.dtype(), mlx::Dtype::Float32);
        assert_eq!(y.dtype(), mlx::Dtype::Float32);
    }
}
