//! Normalization layers — thin wrappers over `mlx::fast::*` fused kernels.
//!
//! Both [`RmsNorm`] and [`LayerNorm`] delegate to single fused Metal kernels
//! (`mlx::core::fast::rms_norm` / `layer_norm`) — there is no Rust-side
//! composition. Norms always operate in floating point; quantization does
//! not apply.
//!
//! Each layer exposes a default `forward` (current default stream) and a
//! stream-targeted `forward_on` variant (P5.7 contract).

use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::Result;

/// Root-mean-square normalization with a learned per-feature scale.
///
/// Computes `y = x * weight / sqrt(mean(x^2) + eps)` along the last axis,
/// matching the LLaMA / Qwen3.5 convention.
pub struct RmsNorm {
    /// `[dim]` learned scale.
    weight: Array,
    /// Numerical-stability epsilon added inside the square root.
    eps: f32,
}

impl RmsNorm {
    /// Build an `RmsNorm` from `loader`, looking for `{prefix}.weight`.
    pub fn from_loader(loader: &Loader, prefix: &str, eps: f32) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        Ok(Self { weight, eps })
    }

    /// Use a pre-loaded weight directly. Useful when the caller already
    /// holds the parameter (e.g. `q_norm` / `k_norm` constructed inside
    /// attention from a tensor passed in).
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass on the current default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        Ok(mlx::fast::rms_norm_on(
            x,
            Some(&self.weight),
            self.eps,
            target,
        )?)
    }
}

/// Layer normalization with a learned per-feature scale and optional bias.
///
/// Computes `y = (x - mean(x)) / sqrt(var(x) + eps) * weight (+ bias)` along
/// the last axis.
pub struct LayerNorm {
    /// `[dim]` learned scale.
    weight: Array,
    /// Optional `[dim]` learned bias.
    bias: Option<Array>,
    /// Numerical-stability epsilon added inside the square root.
    eps: f32,
}

impl LayerNorm {
    /// Build a `LayerNorm` from `loader`, looking for `{prefix}.weight`
    /// (required) and `{prefix}.bias` (optional).
    pub fn from_loader(loader: &Loader, prefix: &str, eps: f32) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        let bias = loader.tensor_opt(&format!("{prefix}.bias")).cloned();
        Ok(Self { weight, bias, eps })
    }

    /// Forward pass on the current default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        Ok(mlx::fast::layer_norm_on(
            x,
            Some(&self.weight),
            self.bias.as_ref(),
            self.eps,
            target,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn rmsnorm_unit_weight_normalizes() {
        // weight = ones, so RMSNorm(x) = x / sqrt(mean(x^2) + eps).
        let weight: Array = (&[1.0_f32; 4][..], (4,)).try_into().unwrap();
        let norm = RmsNorm::new(weight, 1e-6);
        let x: Array = (&[2.0_f32, 2.0, 2.0, 2.0][..], (1, 4)).try_into().unwrap();
        let y = norm.forward(&x).unwrap();
        // mean(x^2) = 4, sqrt(4) = 2, so each element -> 2 * 1 / 2 = 1.
        let v: Vec<f32> = y.to_vec().unwrap();
        for val in v {
            assert!((val - 1.0).abs() < 1e-4, "got {val}");
        }
    }

    #[test]
    fn layernorm_runs_without_panic() {
        let weight: Array = (&[1.0_f32; 4][..], (4,)).try_into().unwrap();
        let bias: Array = (&[0.0_f32; 4][..], (4,)).try_into().unwrap();
        let norm = LayerNorm {
            weight,
            bias: Some(bias),
            eps: 1e-5,
        };
        let x = Array::zeros((1, 4), Dtype::Float32).unwrap();
        let _ = norm.forward(&x).unwrap();
    }
}
