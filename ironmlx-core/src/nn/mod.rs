//! Shared neural-network layers without model-specific execution policies.

pub mod activations;
pub mod conv;
pub mod norm;

pub use conv::{Conv1d, Conv1dConfig};
pub use norm::{LayerNorm, RmsNorm, RmsNormGated};

pub mod linear;
pub use linear::Linear;

pub mod embedding;
pub use embedding::Embedding;
