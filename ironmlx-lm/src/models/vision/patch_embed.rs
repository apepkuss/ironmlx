//! Patch embedding for Qwen3.5 vision encoder.

use anyhow::Result;
use mlx::{ops, Array, StreamOrDevice};

use ironmlx_core::weights::WeightSource;

pub struct PatchEmbed {
    weight_2d: Array,
    bias: Array,
}

impl PatchEmbed {
    /// Construct from already-loaded `weight_5d` shape
    /// `[hidden_size, kT, kH, kW, Cin]` (Qwen3.5-VL: `[1024, 2, 16, 16, 3]`)
    /// and `bias` shape `[hidden_size]`. Returns `Err` if `weight_5d` cannot
    /// be reshaped to `[hidden_size, kT*kH*kW*Cin]` (size mismatch).
    pub fn new(weight_5d: Array, bias: Array, hidden_size: i32) -> Result<Self> {
        let weight_2d = weight_5d.reshape(&[hidden_size, 2 * 16 * 16 * 3][..])?;
        Ok(Self { weight_2d, bias })
    }

    pub fn from_loader(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
        hidden_size: i32,
    ) -> Result<Self> {
        let source = loader.tensor(&format!("{prefix}.weight"))?;
        let shape = source.shape();
        let weight = match shape.as_slice() {
            // Diffusers OI(T)HW -> MLX-friendly O(T)HWI.
            [out, 3, 2, 16, 16] if *out == hidden_size => {
                source.transpose_axes(&[0_i32, 2, 3, 4, 1][..])?
            }
            [out, 2, 16, 16, 3] if *out == hidden_size => source.clone(),
            actual => anyhow::bail!(
                "vision patch embedding {prefix}.weight has unsupported shape {actual:?}"
            ),
        };
        let bias = loader.tensor(&format!("{prefix}.bias"))?.clone();
        let result = Self::new(weight, bias, hidden_size)?;
        mlx::transforms::eval(&[&result.weight_2d, &result.bias])?;
        Ok(result)
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        // x: [N, T=2, C=3, H=16, W=16]  →  [N, T, H, W, C]
        let x = ops::shape::transpose_axes(x, [0_i32, 1, 3, 4, 2])?;
        let n = x.shape().as_slice()[0];
        let x = x
            .reshape(&[n, 2 * 16 * 16 * 3][..])?
            .astype_on(self.weight_2d.dtype(), target)?;
        let wt = self.weight_2d.transpose_on(target)?;
        let out = x.matmul_on(&wt, target)?;
        Ok(&out + &self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Array;

    #[test]
    fn patch_embed_output_shape() {
        crate::models::vision::init_test_metallib();
        let weight = Array::zeros([1024, 2, 16, 16, 3], mlx::Dtype::Bfloat16).unwrap();
        let bias = Array::zeros([1024], mlx::Dtype::Bfloat16).unwrap();
        let pe = PatchEmbed::new(weight, bias, 1024).unwrap();
        let input = Array::zeros([4, 2, 3, 16, 16], mlx::Dtype::Bfloat16).unwrap();
        let out = pe.forward(&input).unwrap();
        assert_eq!(out.shape().as_slice(), &[4, 1024]);
    }
}
