//! Raw FFI bindings to MLX C++.
//!
//! This crate is the `-sys` half of `cxx-mlx`. For a safe, idiomatic API,
//! depend on the `mlx` crate instead.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx-sys only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod bridge;

pub use bridge::array;
