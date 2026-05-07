//! SwiGLU MLP — `down( silu(gate(x)) * up(x) )`.

use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

/// SwiGLU feed-forward block, as used by Llama / Qwen / Mistral families.
///
/// Computes `down( silu(gate(x)) * up(x) )` where `silu(z) = z * sigmoid(z)`.
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    /// Build an `Mlp` from `loader`, expecting the three sub-projections at
    /// `{prefix}.gate_proj`, `{prefix}.up_proj`, and `{prefix}.down_proj`.
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?,
            up: Linear::from_loader(loader, &format!("{prefix}.up_proj"))?,
            down: Linear::from_loader(loader, &format!("{prefix}.down_proj"))?,
        })
    }

    /// Test/composition seam: build an `Mlp` from pre-built sub-projections.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(gate: Linear, up: Linear, down: Linear) -> Self {
        Self { gate, up, down }
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
        // silu(g) = g * sigmoid(g); `&Array * &Array` is panic-on-err here.
        let g_sig = g.sigmoid_on(target)?;
        let activated_lhs = &g * &g_sig;
        let activated = &activated_lhs * &u;
        self.down.forward_on(&activated, target)
    }
}
