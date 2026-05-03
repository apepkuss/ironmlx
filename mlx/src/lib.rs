//! Safe Rust bindings to Apple MLX.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod dtype;
mod error;

pub use dtype::Dtype;
pub use error::{Error, Result};
