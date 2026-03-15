use crate::array::Array;
use crate::error::Result;
use std::collections::HashMap;

/// Trait for neural network modules.
pub trait Module {
    /// Forward pass.
    fn forward(&self, x: &Array) -> Result<Array>;

    /// Return all named parameters as (name, array) pairs.
    fn parameters(&self) -> Vec<(String, &Array)>;

    /// Load weights from a name->Array map, using the given prefix.
    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()>;
}

/// Helper: look up a weight by name, return error if missing.
pub fn get_weight(weights: &HashMap<String, Array>, prefix: &str, name: &str) -> Result<Array> {
    let key = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    };
    weights
        .get(&key)
        .cloned()
        .ok_or_else(|| crate::error::Error::Mlx(format!("missing weight: {}", key)))
}
