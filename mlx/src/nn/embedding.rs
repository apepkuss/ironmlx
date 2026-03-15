use super::module::{Module, get_weight};
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use std::collections::HashMap;

pub struct Embedding {
    pub weight: Array,
}

impl Embedding {
    pub fn new(weight: Array) -> Self {
        Self { weight }
    }

    pub fn forward_with_stream(&self, tokens: &Array, stream: &Stream) -> Result<Array> {
        ops::take_axis(&self.weight, tokens, 0, stream)
    }
}

impl Module for Embedding {
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
