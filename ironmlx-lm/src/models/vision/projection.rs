use anyhow::Result;
use ironmlx_core::weights::WeightSource;
use mlx::{Array, StreamOrDevice};

use crate::nn::Linear;

/// Vision projection that preserves the established fused FP `addmm` path
/// while also accepting MLX affine-quantized weights.
pub(super) enum VisionProjection {
    FullPrecision { weight: Array, bias: Option<Array> },
    Quantized(Linear),
}

impl VisionProjection {
    pub(super) fn from_loader(loader: &(impl WeightSource + ?Sized), prefix: &str) -> Result<Self> {
        if loader.contains(&format!("{prefix}.scales")) {
            return Ok(Self::Quantized(Linear::from_loader(loader, prefix)?));
        }
        Ok(Self::FullPrecision {
            weight: loader.tensor(&format!("{prefix}.weight"))?.clone(),
            bias: loader.tensor_opt(&format!("{prefix}.bias")).cloned(),
        })
    }

    pub(super) fn new_fp(weight: Array, bias: Option<Array>) -> Self {
        Self::FullPrecision { weight, bias }
    }

    pub(super) fn forward_on(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        match self {
            Self::FullPrecision { weight, bias } => {
                let transposed = weight.transpose_on(target)?;
                match bias {
                    Some(bias) => Ok(mlx::ops::addmm_on(
                        bias,
                        input,
                        &transposed,
                        1.0,
                        1.0,
                        target,
                    )?),
                    None => Ok(input.matmul_on(&transposed, target)?),
                }
            }
            Self::Quantized(linear) => linear.forward_on(input, target),
        }
    }
}
