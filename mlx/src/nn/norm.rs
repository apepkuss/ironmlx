use super::activations::silu;
use super::module::{Module, get_weight};
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::fast;
use crate::ops;
use crate::stream::Stream;
use std::collections::HashMap;

pub struct RMSNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        fast::rms_norm(x, Some(&self.weight), self.eps, stream)
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        self.forward_with_stream(x, &stream)
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        vec![("weight".to_string(), &self.weight)]
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        self.weight = get_weight(weights, prefix, "weight")?;
        Ok(())
    }
}

pub struct LayerNorm {
    pub weight: Option<Array>,
    pub bias: Option<Array>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Option<Array>, bias: Option<Array>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        fast::layer_norm(
            x,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
            stream,
        )
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        self.forward_with_stream(x, &stream)
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(("weight".to_string(), w));
        }
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        let w_key = if prefix.is_empty() {
            "weight".to_string()
        } else {
            format!("{}.weight", prefix)
        };
        let b_key = if prefix.is_empty() {
            "bias".to_string()
        } else {
            format!("{}.bias", prefix)
        };
        self.weight = weights.get(&w_key).cloned();
        self.bias = weights.get(&b_key).cloned();
        Ok(())
    }
}

/// Gated RMS normalization: rms_norm(x) * silu(z)
/// Used by Qwen3.5's GatedDeltaNet.
pub struct RMSNormGated {
    pub weight: Array,
    pub eps: f32,
}

impl RMSNormGated {
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward: rms_norm(x, weight, eps) * silu(z)
    pub fn forward_with_stream(&self, x: &Array, z: &Array, stream: &Stream) -> Result<Array> {
        let normed = fast::rms_norm(x, Some(&self.weight), self.eps, stream)?;
        let gate = silu(z, stream)?;
        ops::multiply(&normed, &gate, stream)
    }
}
