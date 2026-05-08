//! Normalization layers — thin wrappers over `mlx::fast::*` fused kernels.
//!
//! Both [`RmsNorm`] and [`LayerNorm`] delegate to single fused Metal kernels
//! (`mlx::core::fast::rms_norm` / `layer_norm`) — there is no Rust-side
//! composition. Norms always operate in floating point; quantization does
//! not apply.
//!
//! Each layer exposes a default `forward` (current default stream) and a
//! stream-targeted `forward_on` variant (P5.7 contract).

use std::sync::{Mutex, OnceLock};

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

/// RMSNorm with optional sigmoid-style gate, matching mlx-lm's `Qwen3NextRMSNormGated`.
///
/// `forward(hidden, None)` → `cast(rms_norm(hidden, weight, eps), hidden.dtype())`.
/// `forward(hidden, Some(gate))` → `cast(silu(gate_fp32) * rms_norm_fp32, hidden.dtype())`,
/// matching the precise-SwiGLU pattern: fp32 intermediate, cast back to input dtype.
pub struct RmsNormGated {
    weight: Array,
    eps: f32,
}

impl RmsNormGated {
    /// Production constructor: load `{prefix}.weight`.
    pub fn from_loader(loader: &Loader, prefix: &str, eps: f32) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        Ok(Self { weight, eps })
    }

    /// Test/composition seam: build from in-memory weight + eps.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it
    /// — those tests are compiled as external crates. Hidden from rustdoc via
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
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
                // Precise SwiGLU via module-level mlx::compile cell — fuses
                // 6 elementwise ops (astype, sigmoid, mul, astype, mul) into
                // a single Metal dispatch. Output is fp32; cast back to
                // hidden_dtype outside the compiled graph (per-call data).
                let outs = swiglu_fused_invoke(&[g, &normed])?;
                let mul_f32 = outs
                    .into_iter()
                    .next()
                    .expect("swiglu_fused returns one output");
                Ok(mlx::ops::cast::astype(&mul_f32, hidden_dtype)?)
            }
            None => Ok(mlx::ops::cast::astype(&normed, hidden_dtype)?),
        }
    }
}

/// Module-level lazy-initialized SwiGLU graph for [`RmsNormGated`]'s gated
/// path. Mirrors mlx-lm's `@partial(mx.compile, shapeless=True)` decorator
/// pattern from `qwen3_next.py:58-62`. Single `OnceLock` shared across all
/// `RmsNormGated` instances — only one trace per process lifetime.
///
/// `CompiledFn` is `Send` but not `Sync` (MLX C++ object with internal state);
/// we wrap it in a `Mutex` so the static can be `Sync`. The lock is acquired
/// only at invoke time. Decode is single-threaded, so contention is absent.
///
/// Inputs (in order): `g` (gate, any dtype), `normed` (rms-normed hidden,
/// any dtype). Output: f32 Array equal to `silu(g_f32) * normed_f32` —
/// caller is responsible for casting back to the input dtype.
static SWIGLU_FUSED: OnceLock<Mutex<CompiledFn>> = OnceLock::new();

fn swiglu_fused_invoke(inputs: &[&Array]) -> crate::Result<Vec<Array>> {
    let cell = SWIGLU_FUSED.get_or_init(|| {
        let cfn = compile(
            |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
                let g = inputs[0];
                let normed = inputs[1];
                let g_f32 = mlx::ops::cast::astype(g, Dtype::Float32)?;
                let g_sig = g_f32.sigmoid()?;
                let g_silu = &g_f32 * &g_sig;
                let normed_f32 = mlx::ops::cast::astype(normed, Dtype::Float32)?;
                let mul_f32 = &g_silu * &normed_f32;
                Ok(vec![mul_f32])
            },
            ShapeMode::Shapeless,
        )
        .expect("compile swiglu_fused");
        Mutex::new(cfn)
    });
    Ok(cell.lock().expect("swiglu_fused mutex").invoke(inputs)?)
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

    #[test]
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
    fn rms_norm_gated_with_gate_finite() {
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
        // gate=0 channel (index 2): silu(0) = 0 * sigmoid(0) = 0, so y[2] = 0
        assert!(
            v[2].abs() < 1e-6,
            "gate=0 should yield zero output, got {}",
            v[2]
        );
    }

    #[test]
    fn swiglu_fused_matches_reference_path() {
        // Build small [4, 4] gate + normed Arrays, run through the
        // module-level swiglu_fused() compile cell and through a hand-rolled
        // reference (sigmoid → mul → mul). Assert close in fp32.
        let g_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let normed_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let shape = &[4_i32, 4][..];
        let g: Array = (g_data.as_slice(), shape).try_into().unwrap();
        let normed: Array = (normed_data.as_slice(), shape).try_into().unwrap();

        // Fused path
        let fused_outs = swiglu_fused_invoke(&[&g, &normed]).unwrap();
        let fused = fused_outs.into_iter().next().unwrap();
        let fused_vec: Vec<f32> = fused.to_vec().unwrap();

        // Reference unfused path: silu(g) * normed, all in fp32 (inputs already fp32).
        let g_sig = g.sigmoid().unwrap();
        let g_silu = &g * &g_sig;
        let ref_arr = &g_silu * &normed;
        let ref_vec: Vec<f32> = ref_arr.to_vec().unwrap();

        assert_eq!(fused_vec.len(), ref_vec.len());
        for (i, (a, b)) in fused_vec.iter().zip(ref_vec.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at index {i}: fused={a}, ref={b}",
            );
        }
    }
}
