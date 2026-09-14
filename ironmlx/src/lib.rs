//! HTTP protocol adapters and command-line application for IronMLX.
//! Model computation and inference lifecycle live in `ironmlx-lm` and `ironmlx-runtime`.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("ironmlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

pub mod cli;
pub mod logging;
pub mod server;
pub use anyhow::{Error, Result};
