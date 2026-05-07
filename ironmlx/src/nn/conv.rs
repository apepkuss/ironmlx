//! 1D convolution layer wrapping `mlx::ops::conv1d`.
//!
//! Weight layout (matching MLX C++): `[out_channels, kernel_size, in_channels / groups]`.
//! For depthwise: `groups = in_channels = out_channels`, so weight is
//! `[in_channels, kernel_size, 1]`.

use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::Result;

/// Configuration for [`Conv1d`].
#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub kernel_size: i32,
    pub stride: i32,
    pub padding: i32,
    pub dilation: i32,
    /// `groups = in_channels = out_channels` for depthwise conv.
    pub groups: i32,
}

/// 1D convolution layer.
pub struct Conv1d {
    weight: Array,
    bias: Option<Array>,
    cfg: Conv1dConfig,
}

impl Conv1d {
    /// Production constructor: load weight from `{prefix}.weight` and optional
    /// bias from `{prefix}.bias`.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: Conv1dConfig) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        let bias = loader.tensor_opt(&format!("{prefix}.bias")).cloned();
        Ok(Self { weight, bias, cfg })
    }

    /// Test/composition seam: build from in-memory weight and optional bias.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it
    /// — those tests are compiled as external crates. Hidden from rustdoc via
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new(weight: Array, bias: Option<Array>, cfg: Conv1dConfig) -> Self {
        Self { weight, bias, cfg }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &Conv1dConfig {
        &self.cfg
    }

    /// Forward pass with default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Stream-targeted forward.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let mut y = mlx::ops::conv1d_on(
            x,
            &self.weight,
            self.cfg.stride,
            self.cfg.padding,
            self.cfg.dilation,
            self.cfg.groups,
            target,
        )?;
        if let Some(b) = &self.bias {
            // Bias broadcasts over (N, L) on last axis (C_out).
            y = &y + b;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    fn small_depthwise_conv() -> Conv1d {
        let weight = Array::zeros((6_i32, 3, 1), Dtype::Float32).unwrap();
        let cfg = Conv1dConfig {
            in_channels: 6,
            out_channels: 6,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 6,
        };
        Conv1d::new(weight, None, cfg)
    }

    #[test]
    fn conv1d_construction_from_components() {
        let conv = small_depthwise_conv();
        assert_eq!(conv.config().out_channels, 6);
        assert_eq!(conv.config().groups, 6);
    }

    #[test]
    fn conv1d_forward_shape_depthwise() {
        let conv = small_depthwise_conv();
        // input: [N=1, L=4, C=6]; output: [1, 4-3+1=2, 6]
        let x = Array::zeros((1_i32, 4, 6), Dtype::Float32).unwrap();
        let y = conv.forward(&x).expect("forward");
        assert_eq!(y.shape().as_slice(), &[1, 2, 6]);
        assert_eq!(y.dtype(), Dtype::Float32);
    }
}
