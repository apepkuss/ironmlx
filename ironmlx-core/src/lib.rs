//! Shared model-independent weights and neural-network computation.

pub mod nn;
pub mod weights;

pub use anyhow::{Error, Result};

pub mod sampler;
