//! LLM/VLM models and model-side inference operations for Apple Silicon.
//! Runtime scheduling and HTTP transport are provided by their callers.
pub mod core;
pub mod models;
pub mod nn;
pub use anyhow::{Error, Result};
pub use core::{ChatTemplate, KVCache, Loader, Message, Model, QuantMeta, Sampler, Tokenizer};
#[cfg(feature = "test-support")]
pub mod test_support;
