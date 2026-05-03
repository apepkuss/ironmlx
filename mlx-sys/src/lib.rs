//! Raw FFI bindings to MLX C++.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx-sys only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod bridge;

pub use bridge::array;
