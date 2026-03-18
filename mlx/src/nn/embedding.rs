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

/// Quantized embedding layer. Stores weights in packed format and dequantizes
/// before lookup.
pub struct QuantizedEmbedding {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl QuantizedEmbedding {
    pub fn new(weight: Array, scales: Array, biases: Array, group_size: i32, bits: i32) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode: "affine".to_string(),
        }
    }

    pub fn forward_with_stream(&self, tokens: &Array, stream: &Stream) -> Result<Array> {
        let full_weight = ops::dequantize(
            &self.weight,
            &self.scales,
            Some(&self.biases),
            Some(self.group_size),
            Some(self.bits),
            &self.mode,
            stream,
        )?;
        ops::take_axis(&full_weight, tokens, 0, stream)
    }
}

impl Module for QuantizedEmbedding {
    fn forward(&self, x: &Array) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        self.forward_with_stream(x, &stream)
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        vec![
            ("weight".to_string(), &self.weight),
            ("scales".to_string(), &self.scales),
            ("biases".to_string(), &self.biases),
        ]
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        self.weight = get_weight(weights, prefix, "weight")?;
        self.scales = get_weight(weights, prefix, "scales")?;
        self.biases = get_weight(weights, prefix, "biases")?;
        Ok(())
    }
}

/// Unified embedding layer that dispatches to either `Embedding` or `QuantizedEmbedding`.
pub enum EmbeddingLayer {
    Full(Embedding),
    Quantized(QuantizedEmbedding),
}

impl EmbeddingLayer {
    /// Create from weights map. Auto-detects quantized weights.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let scales_key = if prefix.is_empty() {
            "scales".to_string()
        } else {
            format!("{}.scales", prefix)
        };

        if weights.contains_key(&scales_key) {
            let weight = get_weight(weights, prefix, "weight")?;
            let scales = get_weight(weights, prefix, "scales")?;
            let biases = get_weight(weights, prefix, "biases")?;
            Ok(Self::Quantized(QuantizedEmbedding::new(
                weight, scales, biases, group_size, bits,
            )))
        } else {
            let weight = get_weight(weights, prefix, "weight")?;
            Ok(Self::Full(Embedding::new(weight)))
        }
    }

    pub fn forward_with_stream(&self, tokens: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Full(e) => e.forward_with_stream(tokens, stream),
            Self::Quantized(q) => q.forward_with_stream(tokens, stream),
        }
    }
}

impl Module for EmbeddingLayer {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Full(e) => e.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        match self {
            Self::Full(e) => e.parameters(),
            Self::Quantized(q) => q.parameters(),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        match self {
            Self::Full(e) => e.load_weights(weights, prefix),
            Self::Quantized(q) => q.load_weights(weights, prefix),
        }
    }
}
