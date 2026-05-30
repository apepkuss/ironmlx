//! SwiGLU MLP — `down( silu(gate(x)) * up(x) )`.

use std::sync::OnceLock;

use mlx::compile::CompiledFn;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::activations::{build_swiglu, invoke_swiglu};
use crate::nn::Linear;
use crate::Result;

/// SwiGLU feed-forward block, as used by Llama / Qwen / Mistral families.
///
/// Computes `down( silu(gate(x)) * up(x) )` where `silu(z) = z * sigmoid(z)`.
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    swiglu: OnceLock<CompiledFn>,
}

impl Mlp {
    /// Build an `Mlp` from `loader`, expecting the three sub-projections at
    /// `{prefix}.gate_proj`, `{prefix}.up_proj`, and `{prefix}.down_proj`.
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?,
            up: Linear::from_loader(loader, &format!("{prefix}.up_proj"))?,
            down: Linear::from_loader(loader, &format!("{prefix}.down_proj"))?,
            swiglu: OnceLock::new(),
        })
    }

    /// Test/composition seam: build an `Mlp` from pre-built sub-projections.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(gate: Linear, up: Linear, down: Linear) -> Self {
        Self {
            gate,
            up,
            down,
            swiglu: OnceLock::new(),
        }
    }

    fn swiglu(&self) -> &CompiledFn {
        self.swiglu.get_or_init(build_swiglu)
    }

    fn swiglu_on(&self, gate: &Array, up: &Array) -> Result<Array> {
        invoke_swiglu(self.swiglu(), gate, up)
    }

    /// Forward pass on the default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let g = self.gate.forward_on(x, target)?;
        let u = self.up.forward_on(x, target)?;
        let activated = self.swiglu_on(&g, &u)?;
        self.down.forward_on(&activated, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
    fn swiglu_forward_matches_manual_silu_mul() {
        let identity: Array = (&[1.0_f32, 0.0, 0.0, 1.0][..], (2, 2)).try_into().unwrap();
        let mlp = Mlp::from_components(
            Linear::new_fp(identity.clone(), None),
            Linear::new_fp(identity.clone(), None),
            Linear::new_fp(identity, None),
        );
        let x: Array = (&[0.5_f32, -1.0][..], (1, 2)).try_into().unwrap();

        let y = mlp.forward(&x).expect("forward");
        let got: Vec<f32> = y.to_vec().expect("to_vec");
        let want: Vec<f32> = x
            .to_vec::<f32>()
            .expect("input to_vec")
            .into_iter()
            .map(|v| v * (1.0 / (1.0 + (-v).exp())) * v)
            .collect();

        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want.iter()) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-5);
        }
    }
}
