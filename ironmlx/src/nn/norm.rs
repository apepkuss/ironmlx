//! Normalization layers — thin wrappers over `mlx::fast::*` fused kernels.
//!
//! Both [`RmsNorm`] and [`LayerNorm`] delegate to single fused Metal kernels
//! (`mlx::core::fast::rms_norm` / `layer_norm`) — there is no Rust-side
//! composition. Norms always operate in floating point; quantization does
//! not apply.
//!
//! Each layer exposes a default `forward` (current default stream) and a
//! stream-targeted `forward_on` variant (P5.7 contract).

use std::sync::OnceLock;

use anyhow::anyhow;
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::{Array, Dtype, StreamOrDevice};

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

    pub(crate) fn weight(&self) -> &Array {
        &self.weight
    }

    pub(crate) fn eps(&self) -> f32 {
        self.eps
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
    /// Build a `LayerNorm` from pre-loaded weight and optional bias.
    ///
    /// Useful when the caller already holds the parameters (e.g. constructed
    /// inside a ViT block from tensors passed in directly).
    pub fn new(weight: Array, bias: Option<Array>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Borrow the weight tensor (for eager-eval bookkeeping).
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// Borrow the optional bias tensor.
    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }

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
        let target = target.into();
        let input_dtype = x.dtype();

        let weight_cast;
        let weight = if self.weight.dtype() == input_dtype {
            &self.weight
        } else {
            weight_cast = mlx::ops::cast::astype_on(&self.weight, input_dtype, target)?;
            &weight_cast
        };

        let bias_cast;
        let bias = match self.bias.as_ref() {
            Some(b) if b.dtype() == input_dtype => Some(b),
            Some(b) => {
                bias_cast = mlx::ops::cast::astype_on(b, input_dtype, target)?;
                Some(&bias_cast)
            }
            None => None,
        };

        let out = mlx::fast::layer_norm_on(x, Some(weight), bias, self.eps, target)?;
        if out.dtype() == input_dtype {
            Ok(out)
        } else {
            Ok(mlx::ops::cast::astype_on(&out, input_dtype, target)?)
        }
    }
}

/// RMSNorm with optional sigmoid-style gate, matching mlx-lm's `Qwen3NextRMSNormGated`.
///
/// `forward(hidden, None)` → `cast(rms_norm(hidden, weight, eps), hidden.dtype())`.
/// `forward(hidden, Some(gate))` → `cast(silu(gate_fp32) * rms_norm_fp32, hidden.dtype())`,
/// matching the precise-SwiGLU pattern: fp32 intermediate, cast back to input dtype.
pub struct RmsNormGated {
    weight: Array,
    eps: f32,
    gated_mul: OnceLock<CompiledFn>,
}

impl RmsNormGated {
    /// Production constructor: load `{prefix}.weight`.
    pub fn from_loader(loader: &Loader, prefix: &str, eps: f32) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        Ok(Self {
            weight,
            eps,
            gated_mul: OnceLock::new(),
        })
    }

    /// Test/composition seam: build from in-memory weight + eps.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it
    /// — those tests are compiled as external crates. Hidden from rustdoc via
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new(weight: Array, eps: f32) -> Self {
        Self {
            weight,
            eps,
            gated_mul: OnceLock::new(),
        }
    }

    fn gated_mul(&self) -> &CompiledFn {
        self.gated_mul.get_or_init(|| {
            compile(
                |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
                    let hidden = inputs[0];
                    let gate = inputs[1];
                    let normed = inputs[2];
                    let hidden_dtype = hidden.dtype();

                    let gate_f32 = mlx::ops::cast::astype(gate, Dtype::Float32)?;
                    let gate_sig = gate_f32.sigmoid()?;
                    let gate_silu = &gate_f32 * &gate_sig;
                    let normed_f32 = mlx::ops::cast::astype(normed, Dtype::Float32)?;
                    let out_f32 = &gate_silu * &normed_f32;
                    let out = mlx::ops::cast::astype(&out_f32, hidden_dtype)?;
                    Ok(vec![out])
                },
                ShapeMode::Shapeless,
            )
            .expect("RmsNormGated gated_mul compile")
        })
    }

    /// Forward pass with default stream.
    pub fn forward(&self, hidden: &Array, gate: Option<&Array>) -> Result<Array> {
        self.forward_on(hidden, gate, ())
    }

    /// Stream-targeted forward.
    pub fn forward_on(
        &self,
        hidden: &Array,
        gate: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden_dtype = hidden.dtype();

        let normed = mlx::fast::rms_norm_on(hidden, Some(&self.weight), self.eps, target)?;

        match gate {
            Some(g) => {
                let mut outs = self
                    .gated_mul()
                    .invoke(&[hidden, g, &normed])
                    .map_err(|e| anyhow!("RmsNormGated gated_mul invoke failed: {e}"))?;
                outs.pop()
                    .ok_or_else(|| anyhow!("RmsNormGated gated_mul returned no outputs"))
            }
            None => Ok(mlx::ops::cast::astype(&normed, hidden_dtype)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
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

    #[test]
    #[serial(mlx_metal)]
    fn layernorm_preserves_input_dtype_with_bf16_parameters() {
        let weight = mlx::ops::cast::astype(
            &mlx::ops::constructors::ones((4_i32,), Dtype::Float32).unwrap(),
            Dtype::Bfloat16,
        )
        .unwrap();
        let bias = mlx::ops::cast::astype(
            &Array::zeros((4_i32,), Dtype::Float32).unwrap(),
            Dtype::Bfloat16,
        )
        .unwrap();
        let norm = LayerNorm::new(weight, Some(bias), 1e-6);
        let x: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1, 4)).try_into().unwrap();

        let y = norm.forward(&x).unwrap();

        assert_eq!(y.dtype(), Dtype::Float32);
    }

    #[test]
    #[serial(mlx_metal)]
    fn rms_norm_gated_none_path_shape_dtype() {
        // Verify shape/dtype/finiteness of the gate=None code path.
        // (Strict equivalence to a separate RmsNorm computation is not asserted —
        // the integration test in T7 covers that against the Python fixture.)
        let weight = mlx::ops::constructors::ones((4_i32,), Dtype::Float32).unwrap();
        let norm = RmsNormGated::new(weight, 1e-6);
        // input: [1, 4] fp32 with non-trivial values
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x: Array = (x_data.as_slice(), (1_i32, 4)).try_into().unwrap();

        let y = norm.forward(&x, None).expect("forward no gate");
        assert_eq!(y.shape().as_slice(), &[1, 4]);
        assert_eq!(y.dtype(), Dtype::Float32);
        // Check finiteness — exact RMSNorm value isn't asserted (relative shapes vs gate path matter).
        let v: Vec<f32> = y.to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[serial(mlx_metal)]
    fn rms_norm_gated_with_gate_matches_silu_rmsnorm() {
        // With gate=Some, dispatch should produce finite output (exact silu * rmsnorm
        // values aren't asserted at unit level — those go in the integration test).
        let weight = mlx::ops::constructors::ones((4_i32,), Dtype::Float32).unwrap();
        let norm = RmsNormGated::new(weight, 1e-6);
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x: Array = (x_data.as_slice(), (1_i32, 4)).try_into().unwrap();
        let g_data: Vec<f32> = vec![0.5, -0.5, 0.0, 1.0];
        let g: Array = (g_data.as_slice(), (1_i32, 4)).try_into().unwrap();

        let y = norm.forward(&x, Some(&g)).expect("forward with gate");
        assert_eq!(y.shape().as_slice(), &[1, 4]);
        assert_eq!(y.dtype(), Dtype::Float32);
        let v: Vec<f32> = y.to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));

        let rms = ((1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-6).sqrt();
        let expected: Vec<f32> = x_data
            .iter()
            .zip(g_data.iter())
            .map(|(&x_i, &g_i)| {
                let silu = g_i * (1.0 / (1.0 + (-g_i).exp()));
                silu * (x_i / rms)
            })
            .collect();
        for (idx, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "compiled gated path mismatch at {idx}: got {got}, want {want}"
            );
        }
    }
}
