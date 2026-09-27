//! Native image-generation models and model-side inference operations for MLX.
//! Runtime scheduling and HTTP transport are provided by their callers.

pub mod loader;
pub mod models;

pub use anyhow::{Error, Result};
pub use loader::{
    preflight_model_metadata, ComponentLoader, ImageModelMetadataPreflight,
    ImageQuantizationMetadataPreflight,
};
