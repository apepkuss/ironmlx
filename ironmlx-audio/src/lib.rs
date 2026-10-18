//! Audio domain primitives, resource loading, and worker-local model interfaces.
//!
//! This crate depends on MLX and the shared core, independently of the HTTP
//! server, inference runtime, and language-model library. Model resources are
//! resolved local files; production code never downloads or runs Python.
mod api;
pub mod diarization;
mod error;
pub mod features;
pub mod indextts25;
pub mod io;
pub mod qwen3_asr;
pub mod resources;
pub mod signal;
pub mod vad;
pub use api::*;
pub use error::{AudioError, Result};

pub mod text;
