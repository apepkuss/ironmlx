//! Safe Rust bindings to Apple MLX.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod array;
mod dtype;
mod element;
mod error;

pub use array::Array;
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
