use super::module::{Module, get_weight};
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use std::collections::HashMap;

/// Quantized linear layer. Stores weights in packed format with per-group
/// scales and biases. Forward pass uses `quantized_matmul` to avoid
/// full dequantization.
pub struct QuantizedLinear {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedLinear {
    pub fn new(weight: Array, scales: Array, biases: Array, group_size: i32, bits: i32) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            Some(self.group_size),
            Some(self.bits),
            stream,
        )
    }
}

impl Module for QuantizedLinear {
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

/// Unified linear layer that dispatches to either `Linear` or `QuantizedLinear`.
pub enum LinearLayer {
    Full(Linear),
    Quantized(QuantizedLinear),
}

impl LinearLayer {
    /// Create from weights map. Auto-detects quantized weights by checking
    /// for the presence of `{prefix}.scales`.
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
            Ok(Self::Quantized(QuantizedLinear::new(
                weight, scales, biases, group_size, bits,
            )))
        } else {
            let weight = get_weight(weights, prefix, "weight")?;
            let bias_key = if prefix.is_empty() {
                "bias".to_string()
            } else {
                format!("{}.bias", prefix)
            };
            let bias = weights.get(&bias_key).cloned();
            Ok(Self::Full(Linear::new(weight, bias)))
        }
    }

    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Full(l) => l.forward_with_stream(x, stream),
            Self::Quantized(q) => q.forward_with_stream(x, stream),
        }
    }
}

impl Module for LinearLayer {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Full(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    fn parameters(&self) -> Vec<(String, &Array)> {
        match self {
            Self::Full(l) => l.parameters(),
            Self::Quantized(q) => q.parameters(),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        match self {
            Self::Full(l) => l.load_weights(weights, prefix),
            Self::Quantized(q) => q.load_weights(weights, prefix),
        }
    }
}

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
