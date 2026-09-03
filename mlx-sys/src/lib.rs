//! Raw FFI bindings to MLX C++.
//!
//! This crate is the `-sys` half of the IronMLX MLX bindings. For a safe, idiomatic API,
//! depend on the `mlx` crate instead.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx-sys only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod bridge;

pub use bridge::array;
pub use bridge::compile;
pub use bridge::conv;
pub use bridge::fast;
pub use bridge::io;
pub use bridge::memory;
pub use bridge::metal;
pub use bridge::quantization;
pub use bridge::random;
pub use bridge::stream;
pub use bridge::transforms;
