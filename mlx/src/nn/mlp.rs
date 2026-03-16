use super::activations::silu;
use super::linear::LinearLayer;
use super::module::Module;
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use std::collections::HashMap;

/// Standard Llama-style MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct MLP {
    pub gate_proj: LinearLayer,
    pub up_proj: LinearLayer,
    pub down_proj: LinearLayer,
}

impl MLP {
    pub fn new(gate_proj: LinearLayer, up_proj: LinearLayer, down_proj: LinearLayer) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let gate = self.gate_proj.forward_with_stream(x, stream)?;
        let gate = silu(&gate, stream)?;
        let up = self.up_proj.forward_with_stream(x, stream)?;
        let combined = ops::multiply(&gate, &up, stream)?;
        self.down_proj.forward_with_stream(&combined, stream)
    }
}

impl Module for MLP {
    fn forward(&self, x: &Array) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        self.forward_with_stream(x, &stream)
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        let mut params = Vec::new();
        for (name, arr) in self.gate_proj.parameters() {
            params.push((format!("gate_proj.{}", name), arr));
        }
        for (name, arr) in self.up_proj.parameters() {
            params.push((format!("up_proj.{}", name), arr));
        }
        for (name, arr) in self.down_proj.parameters() {
            params.push((format!("down_proj.{}", name), arr));
        }
        params
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Array>, _prefix: &str) -> Result<()> {
        // LinearLayer is constructed with weights via from_weights; runtime reload not supported
        unimplemented!("use LinearLayer::from_weights during model construction")
    }
}
