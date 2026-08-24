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
    gate_up: GateUp,
    down: Linear,
    swiglu: OnceLock<CompiledFn>,
}

enum GateUp {
    Separate { gate: Linear, up: Linear },
    Fused(Box<FusedGateUp>),
}

struct FusedGateUp {
    projection: Linear,
    gate: Linear,
    up: Linear,
}

impl Mlp {
    /// Build an `Mlp` from `loader`, expecting the three sub-projections at
    /// `{prefix}.gate_proj`, `{prefix}.up_proj`, and `{prefix}.down_proj`.
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_up: GateUp::Separate {
                gate: Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?,
                up: Linear::from_loader(loader, &format!("{prefix}.up_proj"))?,
            },
            down: Linear::from_loader(loader, &format!("{prefix}.down_proj"))?,
            swiglu: OnceLock::new(),
        })
    }

    pub(crate) fn from_loader_dflash2(loader: &Loader, prefix: &str) -> Result<Self> {
        let gate = Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?;
        let up = Linear::from_loader(loader, &format!("{prefix}.up_proj"))?;
        if gate.out_features() != up.out_features() {
            anyhow::bail!(
                "DFlash2 fused MLP requires matching gate/up widths, got {} and {}",
                gate.out_features(),
                up.out_features()
            );
        }
        let projection =
            Linear::fuse_quantized_outputs(&[&gate, &up], "DFlash2 fused MLP gate/up")?;
        let mut separate = projection.split_quantized_outputs(
            &[gate.out_features(), up.out_features()],
            "DFlash2 split MLP gate/up",
        )?;
        Ok(Self {
            gate_up: GateUp::Fused(Box::new(FusedGateUp {
                projection,
                gate: separate.remove(0),
                up: separate.remove(0),
            })),
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
            gate_up: GateUp::Separate { gate, up },
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
        let (g, u) = match &self.gate_up {
            GateUp::Separate { gate, up } => {
                (gate.forward_on(x, target)?, up.forward_on(x, target)?)
            }
            GateUp::Fused(fused) => {
                let FusedGateUp {
                    projection,
                    gate,
                    up,
                } = fused.as_ref();
                if super::product_stable_qmm::is_armed() {
                    let output = projection.forward_on(x, target)?;
                    let mut parts = mlx::ops::shape::split_n_on(&output, 2, -1, target)?;
                    if parts.len() != 2 {
                        anyhow::bail!("DFlash2 fused MLP gate/up returned {} parts", parts.len());
                    }
                    (parts.remove(0), parts.remove(0))
                } else {
                    (gate.forward_on(x, target)?, up.forward_on(x, target)?)
                }
            }
        };
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
