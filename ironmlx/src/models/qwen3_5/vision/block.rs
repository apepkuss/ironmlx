//! Qwen3.5 ViT block — norm1 → attn (rotary) → norm2 → mlp.
//! See spec §4.3.

use anyhow::Result;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;

// sqrt(2/π) = 0.7978845608028654  (tanh GELU approximation constant)
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// GELU with tanh approximation (matches PyTorch `approximate="tanh"` / mlx-vlm `gelu_approx`).
///
/// Formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
///
/// There is no built-in `gelu_approx` in the mlx Rust bindings, so this is
/// hand-rolled from the exact polynomial formula used by mlx-vlm / PyTorch.
fn gelu_tanh(x: &Array, target: StreamOrDevice) -> Result<Array> {
    // x^3 via x * x * x  (avoid power() to stay on the fast path)
    let x2 = x * x;
    let x3 = &x2 * x;
    // inner = sqrt(2/π) * (x + 0.044715 * x^3)
    let inner = (&x3 * 0.044_715_f32 + x) * SQRT_2_OVER_PI;
    // tanh(inner)
    let t = inner.tanh_on(target)?;
    // 0.5 * x * (1 + t)
    let out = x * 0.5_f32 * (&t + 1.0_f32);
    Ok(out)
}

/// Two-layer MLP inside each ViT block.
///
/// Architecture: `linear_fc1` (d_model→4*d_model) → GELU-tanh → `linear_fc2` (4*d_model→d_model).
/// Both layers have bias terms.
pub struct VitMLP {
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
}

impl VitMLP {
    /// Construct from pre-loaded weight Arrays.
    ///
    /// `fc1_w` shape: `[ffn_dim, d_model]`, e.g. `[4096, 1024]`.
    /// `fc1_b` shape: `[ffn_dim]`.
    /// `fc2_w` shape: `[d_model, ffn_dim]`, e.g. `[1024, 4096]`.
    /// `fc2_b` shape: `[d_model]`.
    pub fn new(fc1_w: Array, fc1_b: Array, fc2_w: Array, fc2_b: Array) -> Self {
        Self {
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        }
    }

    /// Load from a safetensors checkpoint via `loader`.
    ///
    /// Expected tensor names:
    /// - `{prefix}.linear_fc1.weight` / `.bias`
    /// - `{prefix}.linear_fc2.weight` / `.bias`
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let fc1_w = loader
            .tensor(&format!("{prefix}.linear_fc1.weight"))?
            .clone();
        let fc1_b = loader.tensor(&format!("{prefix}.linear_fc1.bias"))?.clone();
        let fc2_w = loader
            .tensor(&format!("{prefix}.linear_fc2.weight"))?
            .clone();
        let fc2_b = loader.tensor(&format!("{prefix}.linear_fc2.bias"))?.clone();
        Ok(Self::new(fc1_w, fc1_b, fc2_w, fc2_b))
    }

    /// Forward pass on the default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward pass.
    ///
    /// Computes: `fc2(gelu_tanh(fc1(x)))` where each linear is `x @ W^T + b`.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();

        // fc1: [T, d_model] @ [d_model, ffn_dim] + bias  →  [T, ffn_dim]
        let wt1 = self.fc1_w.transpose_on(target)?;
        let h = x.matmul_on(&wt1, target)?;
        let h = &h + &self.fc1_b;

        // GELU tanh approx
        let h = gelu_tanh(&h, target)?;

        // fc2: [T, ffn_dim] @ [ffn_dim, d_model] + bias  →  [T, d_model]
        let wt2 = self.fc2_w.transpose_on(target)?;
        let out = h.matmul_on(&wt2, target)?;
        let out = &out + &self.fc2_b;

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn vit_mlp_output_shape() {
        let fc1_w = Array::zeros(&[4096, 1024], Dtype::Bfloat16).unwrap();
        let fc1_b = Array::zeros(&[4096], Dtype::Bfloat16).unwrap();
        let fc2_w = Array::zeros(&[1024, 4096], Dtype::Bfloat16).unwrap();
        let fc2_b = Array::zeros(&[1024], Dtype::Bfloat16).unwrap();
        let mlp = VitMLP::new(fc1_w, fc1_b, fc2_w, fc2_b);
        let x = Array::zeros(&[4, 1024], Dtype::Bfloat16).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape().as_slice(), &[4, 1024]);
    }

    #[test]
    fn gelu_tanh_zero_maps_to_zero() {
        // gelu_tanh(0) = 0.5 * 0 * (1 + tanh(0)) = 0
        let zero = Array::try_from((&[0.0_f32][..], &[][..])).unwrap();
        let out = gelu_tanh(&zero, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!(
            (v - 0.0_f32).abs() < 1e-6,
            "gelu_tanh(0) should be 0, got {v}"
        );
    }

    #[test]
    fn gelu_tanh_positive_passes_through() {
        // For large positive x, gelu_tanh(x) ≈ x.
        let x = Array::try_from((&[10.0_f32][..], &[][..])).unwrap();
        let out = gelu_tanh(&x, ().into()).unwrap();
        let v = out.item::<f32>().unwrap();
        assert!((v - 10.0_f32).abs() < 0.1, "gelu_tanh(10) ≈ 10, got {v}");
    }
}
