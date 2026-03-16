use super::module::{Module, get_weight};
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;
use std::collections::HashMap;

/// Depthwise 1D causal convolution.
///
/// Expects input `[B, T, C]` and weight `[C, 1, K]` (depthwise: groups = C).
/// Output is `[B, T, C]` with causal (left) zero-padding so the output length
/// matches the input length.
pub struct Conv1d {
    pub weight: Array,
    pub bias: Option<Array>,
    pub kernel_size: usize,
    pub groups: usize,
}

impl Conv1d {
    pub fn new(weight: Array, bias: Option<Array>, kernel_size: usize, groups: usize) -> Self {
        Self {
            weight,
            bias,
            kernel_size,
            groups,
        }
    }

    /// Apply depthwise 1D causal convolution.
    ///
    /// Input shape: `[B, T, C]`
    /// Weight shape: `[C, 1, K]` (reshaped internally to `[1, 1, C]` per kernel tap)
    ///
    /// Implementation: causal zero-pad on the left by `K-1`, then for each kernel
    /// position `k`, slice `x[:, k:k+T, :]` and multiply by `weight[:, 0, k]`.
    /// Sum over all `K` taps to produce `[B, T, C]`.
    /// Apply depthwise 1D convolution without internal padding.
    ///
    /// Input shape: `[B, T_in, C]` where `T_in >= K`.
    /// Output shape: `[B, T_in - K + 1, C]`.
    ///
    /// For causal usage, the caller should prepend `K-1` zeros (or conv state)
    /// to the input before calling this method.
    pub fn forward_no_pad(&self, x: &Array, stream: &Stream) -> Result<Array> {
        self.forward_impl(x, false, stream)
    }

    /// Apply depthwise 1D causal convolution with automatic left-padding.
    ///
    /// Input shape: `[B, T, C]`.
    /// Output shape: `[B, T, C]` (same length, causal padding preserves length).
    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        self.forward_impl(x, true, stream)
    }

    fn forward_impl(&self, x: &Array, causal_pad: bool, stream: &Stream) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let t = shape[1];
        let c = shape[2];
        let k = self.kernel_size as i32;

        let (x_padded, out_t) = if causal_pad {
            // Causal left-padding: prepend (K-1) zeros along the time axis.
            let pad_len = k - 1;
            let pad = ops::zeros(&[b, pad_len, c], x.dtype(), stream)?;
            let cat = VectorArray::from_arrays(&[&pad, x]);
            (ops::concatenate(&cat, 1, stream)?, t)
        } else {
            // No padding: output length = T_in - K + 1
            (x.clone(), t - k + 1)
        };

        // weight shape: [C, 1, K] -> extract weight[:, 0, k] for each tap k,
        // reshape to [1, 1, C] for broadcasting with x slices [B, T_out, C].
        let mut out: Option<Array> = None;
        for ki in 0..k {
            // Slice x_padded[:, ki:ki+out_t, :]
            let x_slice = ops::slice(
                &x_padded,
                &[0, ki, 0],
                &[b, ki + out_t, c],
                &[1, 1, 1],
                stream,
            )?;

            // Extract weight[:, 0, ki] -> shape [C], then reshape to [1, 1, C]
            let w_tap = ops::slice(
                &self.weight,
                &[0, 0, ki],
                &[c, 1, ki + 1],
                &[1, 1, 1],
                stream,
            )?;
            let w_tap = ops::reshape(&w_tap, &[1, 1, c], stream)?;

            // x_slice * w_tap -> [B, T, C]
            let prod = ops::multiply(&x_slice, &w_tap, stream)?;

            out = Some(match out {
                Some(acc) => ops::add(&acc, &prod, stream)?,
                None => prod,
            });
        }

        let mut result = out.unwrap();

        if let Some(ref bias) = self.bias {
            // bias shape: [C] -> reshape to [1, 1, C] for broadcasting
            let b_reshaped = ops::reshape(bias, &[1, 1, c], stream)?;
            result = ops::add(&result, &b_reshaped, stream)?;
        }

        Ok(result)
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Array) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        self.forward_with_stream(x, &stream)
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        self.weight = get_weight(weights, prefix, "weight")?;
        let bias_key = if prefix.is_empty() {
            "bias".to_string()
        } else {
            format!("{}.bias", prefix)
        };
        self.bias = weights.get(&bias_key).cloned();
        Ok(())
    }
}
