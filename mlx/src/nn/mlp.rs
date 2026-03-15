use super::activations::silu;
use super::linear::Linear;
use super::module::Module;
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use std::collections::HashMap;

/// Standard Llama-style MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct MLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl MLP {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
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

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        let p = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}.{}", prefix, name)
            }
        };
        self.gate_proj.load_weights(weights, &p("gate_proj"))?;
        self.up_proj.load_weights(weights, &p("up_proj"))?;
        self.down_proj.load_weights(weights, &p("down_proj"))?;
        Ok(())
    }
}
