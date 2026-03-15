use super::module::{get_weight, Module};
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use std::collections::HashMap;

pub struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}

impl Linear {
    pub fn new(weight: Array, bias: Option<Array>) -> Self {
        Self { weight, bias }
    }

    /// Forward: x @ weight^T + bias
    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let wt = ops::transpose(&self.weight, stream)?;
        let mut out = ops::matmul(x, &wt, stream)?;
        if let Some(ref b) = self.bias {
            out = ops::add(&out, b, stream)?;
        }
        Ok(out)
    }
}

impl Module for Linear {
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
        // bias is optional
        let bias_key = if prefix.is_empty() {
            "bias".to_string()
        } else {
            format!("{}.bias", prefix)
        };
        self.bias = weights.get(&bias_key).cloned();
        Ok(())
    }
}
