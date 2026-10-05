//! Shared model-independent weights and neural-network computation.

pub mod m5_profile;
pub mod nn;
pub mod weights;

pub use anyhow::{Error, Result};

pub mod sampler;
