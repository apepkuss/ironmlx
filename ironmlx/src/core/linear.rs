//! Shared linear parameters and ordinary projection. Model optimization routing stays in nn::linear.

use super::weights::QuantMode;
use crate::Result;
use mlx::{Array, StreamOrDevice};

/// Borrowed quantized Linear internals for architecture-specific fused paths.
#[derive(Clone, Copy)]
pub(crate) struct QuantizedLinearParts<'a> {
    pub(crate) weight: &'a Array,
    pub(crate) scales: &'a Array,
    pub(crate) biases: Option<&'a Array>,
    pub(crate) bias: Option<&'a Array>,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    pub(crate) mode: QuantMode,
}

/// Owned parameters for ordinary linear projection, independent of model routing.
pub(crate) enum LinearParameters {
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
        /// Bits per quantized weight.
        bits: i32,
        /// Quantization scheme from loader metadata.
        mode: QuantMode,
    },
}

impl LinearParameters {
    /// Ordinary MLX projection with no model-specific verification scopes.
    pub(crate) fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        match self {
            Self::Fp { weight, bias } => {
                let wt = weight.transpose_on(target)?;
                let mut y = x.matmul_on(&wt, target)?;
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
            Self::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                mode,
            } => {
                let mut y = mlx::quantization::quantized_matmul_on(
                    x,
                    weight,
                    scales,
                    biases.as_ref(),
                    true,
                    Some(*group_size),
                    Some(*bits),
                    mode.mlx_backend_mode(),
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
